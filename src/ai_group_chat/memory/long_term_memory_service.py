"""
长期记忆服务

职责：
1) 从原始消息增量抽取并归档（异步）
2) 按 user/group/agent 三层作用域检索
3) 进行阈值过滤、排序与 token 预算裁剪
4) 生成分区注入块，并记录审计日志
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from typing import Any
from uuid import uuid4

import tiktoken
from loguru import logger

from ..dao import long_term_memory_dao
from ..models import GroupChat, Message
from .memory_extractor import MemoryExtractor
from .memory_gateway import MemoryGateway


class LongTermMemoryService:
    """长期记忆业务服务"""

    ARCHIVE_BATCH_LIMIT = 300
    ARCHIVE_THRESHOLD = 10
    MAX_RETRIES = 3

    def __init__(self, repo):
        self.repo = repo
        self.dao = long_term_memory_dao
        self.extractor = MemoryExtractor()
        self.gateway = MemoryGateway()
        self._last_retrieval: dict[tuple[str, str], dict[str, Any]] = {}

        try:
            self.encoder = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self.encoder = None

    async def archive_incremental(
        self,
        group: GroupChat,
        user_id: str,
        *,
        force: bool = False,
        reason: str = "event",
    ) -> None:
        """从历史库按游标增量抽取并归档长期记忆"""
        if not group.memory_enabled or not group.archive_enabled:
            return

        checkpoint = self.dao.get_checkpoint(group.id, user_id)
        last_created_at = checkpoint.get("last_message_created_at") if checkpoint else None
        last_message_id = checkpoint.get("last_message_id") if checkpoint else ""

        messages = self.repo.get_messages_since_cursor(
            group_id=group.id,
            last_created_at=last_created_at,
            last_message_id=last_message_id,
            limit=self.ARCHIVE_BATCH_LIMIT,
        )
        raw_messages = [m for m in messages if not m.is_compressed]
        if not raw_messages:
            return

        if not force and len(raw_messages) < self.ARCHIVE_THRESHOLD:
            return

        request_id = str(uuid4())
        self.dao.add_audit_log(
            request_id=request_id,
            group_id=group.id,
            user_id=user_id,
            event_type="archive_start",
            detail=f"reason={reason}, count={len(raw_messages)}",
        )

        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                extracted = await self.extractor.extract(raw_messages)
                prepared = self._prepare_memories(group, user_id, raw_messages, extracted, reason=reason)
                memory_ids = await self.gateway.add_memories(prepared) if prepared else []

                last_msg = raw_messages[-1]
                self.dao.upsert_checkpoint(
                    group_id=group.id,
                    user_id=user_id,
                    last_message_id=last_msg.id,
                    last_message_created_at=last_msg.created_at,
                )

                self.dao.add_audit_log(
                    request_id=request_id,
                    group_id=group.id,
                    user_id=user_id,
                    event_type="archive_success",
                    memory_ids=memory_ids,
                    detail=f"prepared={len(prepared)}, saved={len(memory_ids)}",
                )
                return
            except Exception as e:
                if attempt < self.MAX_RETRIES:
                    await asyncio.sleep(2 ** attempt)
                    continue

                payload = {
                    "reason": reason,
                    "group_id": group.id,
                    "user_id": user_id,
                    "checkpoint": checkpoint,
                    "message_count": len(raw_messages),
                }
                self.dao.add_dead_letter(
                    group_id=group.id,
                    user_id=user_id,
                    error=str(e),
                    payload=payload,
                    retry_count=attempt,
                )
                self.dao.add_audit_log(
                    request_id=request_id,
                    group_id=group.id,
                    user_id=user_id,
                    event_type="archive_failed",
                    detail=str(e),
                )
                logger.error(f"长期记忆归档失败: {e}")

    async def build_injection_context(
        self,
        group: GroupChat,
        user_id: str,
        query: str,
        *,
        max_context_tokens: int = 128000,
        memory_types: set[str] | None = None,
        scopes: set[str] | None = None,
    ) -> str:
        """
        构建长期记忆注入块：
        [长期背景]
        1. ...
        """
        if not group.memory_enabled or not group.retrieve_enabled:
            return ""

        request_id = str(uuid4())
        persona_versions = self._build_persona_versions(group)
        candidates: list[dict] = []
        query_embedding = await self.gateway.build_query_embedding(query)

        if group.scope_user_global:
            candidates.extend(
                self.gateway.search_scope(
                    scope="user_global",
                    user_id=user_id,
                    min_confidence=group.memory_min_confidence,
                    query_embedding=query_embedding,
                    limit=12,
                )
            )

        if group.scope_group_local:
            candidates.extend(
                self.gateway.search_scope(
                    scope="group_local",
                    user_id=user_id,
                    group_id=group.id,
                    min_confidence=group.memory_min_confidence,
                    query_embedding=query_embedding,
                    limit=12,
                )
            )

        if group.scope_agent_local:
            for member_id, version in persona_versions.items():
                candidates.extend(
                    self.gateway.search_scope(
                        scope="agent_local",
                        user_id=user_id,
                        group_id=group.id,
                        member_id=member_id,
                        persona_version=version,
                        min_confidence=group.memory_min_confidence,
                        query_embedding=query_embedding,
                        limit=6,
                    )
                )

        candidates = self._filter_candidates(
            candidates,
            memory_types=memory_types,
            scopes=scopes,
        )

        if not candidates:
            self.dao.add_audit_log(
                request_id=request_id,
                group_id=group.id,
                user_id=user_id,
                event_type="retrieve_empty",
                detail="no candidates after filter",
            )
            return ""

        scored = self._score_and_filter(candidates, query, min_score=group.memory_score_threshold)
        selected = self._apply_budget(
            scored,
            max_context_tokens=max_context_tokens,
            ratio=group.memory_injection_ratio,
            top_n=group.memory_top_n,
        )

        if not selected:
            self.dao.add_audit_log(
                request_id=request_id,
                group_id=group.id,
                user_id=user_id,
                event_type="retrieve_empty",
                detail="filtered out",
            )
            return ""

        ids = [row["id"] for row in selected]
        self.dao.touch_used(ids)
        self.dao.add_audit_log(
            request_id=request_id,
            group_id=group.id,
            user_id=user_id,
            event_type="retrieve_hit",
            scope="mixed",
            memory_ids=ids,
            detail=f"selected={len(ids)}",
        )

        self._last_retrieval[(group.id, user_id)] = {
            "retrieved_at": datetime.now().isoformat(),
            "query": query[:120],
            "selected_count": len(selected),
            "selected_ids": ids,
            "budget_ratio": group.memory_injection_ratio,
            "memory_types_filter": sorted(memory_types) if memory_types else None,
            "scopes_filter": sorted(scopes) if scopes else None,
        }
        return self._format_injection_block(selected)

    def get_group_stats(self, group_id: str) -> dict:
        """获取长期记忆运行统计（用于前端展示）"""
        db_stats = self.dao.get_group_stats(group_id)
        latest = None
        for (g_id, _), value in self._last_retrieval.items():
            if g_id != group_id:
                continue
            if not latest or value.get("retrieved_at", "") > latest.get("retrieved_at", ""):
                latest = value
        return {
            **db_stats,
            "last_retrieval": latest,
        }

    def _prepare_memories(
        self,
        group: GroupChat,
        user_id: str,
        raw_messages: list[Message],
        extracted: list[dict[str, Any]],
        *,
        reason: str,
    ) -> list[dict[str, Any]]:
        if not extracted:
            return []

        member_name_to_id = {m.name: m.id for m in group.members}
        persona_versions = self._build_persona_versions(group)
        last_source = raw_messages[-1]
        prepared: list[dict[str, Any]] = []

        for item in extracted:
            scope = item.get("scope")
            content = (item.get("content") or "").strip()
            confidence = float(item.get("confidence", 0.8))
            if not scope or not content or confidence < 0.55:
                continue

            if scope == "user_global" and not group.scope_user_global:
                continue
            if scope == "group_local" and not group.scope_group_local:
                continue
            if scope == "agent_local" and not group.scope_agent_local:
                continue

            record: dict[str, Any] = {
                "scope": scope,
                "memory_type": item.get("memory_type", "discussion_asset"),
                "content": content,
                "confidence": confidence,
                "user_id": user_id,
                "group_id": group.id if scope in {"group_local", "agent_local"} else None,
                "member_id": None,
                "persona_version": None,
                "source_message_id": last_source.id,
                "source_created_at": last_source.created_at,
                "metadata": {
                    "reason": reason,
                    "sender_name": item.get("sender_name"),
                },
            }

            if scope == "agent_local":
                sender_name = item.get("sender_name")
                member_id = member_name_to_id.get(sender_name or "")
                if not member_id:
                    continue
                record["member_id"] = member_id
                record["persona_version"] = persona_versions.get(member_id)

            # 群聊结论默认 180 天过期，用户偏好不过期
            if scope == "group_local":
                record["expires_at"] = datetime.now() + timedelta(days=180)
            else:
                record["expires_at"] = None

            prepared.append(record)
        return prepared

    @staticmethod
    def _build_persona_versions(group: GroupChat) -> dict[str, str]:
        versions: dict[str, str] = {}
        for m in group.members:
            raw = f"{m.id}|{m.model_id}|{m.description or ''}|{m.task or ''}|{m.thinking}|{m.temperature}"
            versions[m.id] = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
        return versions

    def _score_and_filter(self, rows: list[dict], query: str, min_score: float) -> list[dict]:
        seen = set()
        scored: list[dict] = []
        for row in rows:
            memory_id = row.get("id")
            if not memory_id or memory_id in seen:
                continue
            seen.add(memory_id)

            confidence = float(row.get("confidence", 0.0) or 0.0)
            content = row.get("content", "")
            lexical = self._lexical_score(query, content)
            recency = self._recency_bonus(row.get("updated_at"))
            decay = float(row.get("decay_score", 1.0) or 1.0)
            vector_score = float(row.get("vector_score", 0.0) or 0.0)
            vector_score = max(0.0, min(1.0, vector_score))

            if vector_score > 0:
                score = 0.40 * vector_score + 0.25 * lexical + 0.20 * confidence + 0.10 * recency + 0.05 * decay
            else:
                score = 0.55 * lexical + 0.25 * confidence + 0.1 * recency + 0.1 * decay

            if score < min_score:
                continue

            row = dict(row)
            row["vector_score"] = round(vector_score, 4)
            row["retrieval_score"] = round(score, 4)
            scored.append(row)

        scored.sort(key=lambda x: x["retrieval_score"], reverse=True)
        return scored

    def _apply_budget(self, rows: list[dict], *, max_context_tokens: int, ratio: float, top_n: int) -> list[dict]:
        token_budget = max(128, int(max_context_tokens * ratio))
        selected: list[dict] = []
        used = 0

        for row in rows:
            if len(selected) >= top_n:
                break
            content = row.get("content", "")
            t = self._count_tokens(content)
            if used + t > token_budget:
                continue
            selected.append(row)
            used += t

        return selected

    @staticmethod
    def _filter_candidates(
        rows: list[dict],
        *,
        memory_types: set[str] | None = None,
        scopes: set[str] | None = None,
    ) -> list[dict]:
        """按 memory_type/scope 过滤候选。"""
        if not rows:
            return []

        type_filter = {str(x).strip().lower() for x in (memory_types or set()) if str(x).strip()}
        scope_filter = {str(x).strip() for x in (scopes or set()) if str(x).strip()}
        if not type_filter and not scope_filter:
            return rows

        filtered: list[dict] = []
        for row in rows:
            row_type = str(row.get("memory_type", "")).strip().lower()
            row_scope = str(row.get("scope", "")).strip()
            if type_filter and row_type not in type_filter:
                continue
            if scope_filter and row_scope not in scope_filter:
                continue
            filtered.append(row)
        return filtered

    @staticmethod
    def _lexical_score(query: str, content: str) -> float:
        query = (query or "").strip().lower()
        content = (content or "").strip().lower()
        if not query or not content:
            return 0.0

        seq_ratio = SequenceMatcher(None, query, content).ratio()

        q_tokens = set(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]+", query))
        c_tokens = set(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]+", content))
        overlap = len(q_tokens & c_tokens) / max(1, len(q_tokens))

        return min(1.0, 0.6 * seq_ratio + 0.4 * overlap)

    @staticmethod
    def _recency_bonus(updated_at) -> float:
        if not updated_at:
            return 0.0
        if isinstance(updated_at, str):
            try:
                updated_at = datetime.fromisoformat(updated_at)
            except Exception:
                return 0.0
        age_hours = max(0.0, (datetime.now() - updated_at).total_seconds() / 3600)
        if age_hours <= 24:
            return 1.0
        if age_hours <= 24 * 7:
            return 0.6
        if age_hours <= 24 * 30:
            return 0.3
        return 0.1

    def _count_tokens(self, text: str) -> int:
        if not text:
            return 0
        if self.encoder:
            return len(self.encoder.encode(text))
        return max(1, len(text) // 2)

    @staticmethod
    def _format_injection_block(rows: list[dict]) -> str:
        lines = ["[长期背景]"]
        for i, row in enumerate(rows, start=1):
            dt = row.get("updated_at")
            if isinstance(dt, str):
                date_text = dt[:10]
            elif isinstance(dt, datetime):
                date_text = dt.strftime("%Y-%m-%d")
            else:
                date_text = "-"
            scope = row.get("scope", "unknown")
            conf = float(row.get("confidence", 0.0) or 0.0)
            lines.append(f"{i}. [{scope}] [时间:{date_text}] [置信:{conf:.2f}] {row.get('content', '')}")
        lines.append("注意：如与用户本轮明确输入冲突，以本轮输入为准。")
        return "\n".join(lines)

"""
长期记忆网关

对上提供统一读写接口：
- 本地长期记忆表为主存（可审计、可控）
- 可选同步写入 Mem0（如果环境可用）
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
from typing import Any

from loguru import logger

from ..config import get_settings
from ..dao import long_term_memory_dao
from .embedding_service import EmbeddingService


def _normalize_text(text: str) -> str:
    return " ".join((text or "").strip().split()).lower()


class MemoryGateway:
    """长期记忆统一网关"""

    def __init__(self):
        self.dao = long_term_memory_dao
        self.settings = get_settings()
        self.mem0_client = self._init_mem0_client()
        self.embedding = EmbeddingService()
        self.vector_enabled = bool(self.embedding.enabled and self.dao.vector_available)
        if self.vector_enabled:
            logger.info("✅ MemoryGateway: 启用向量写入/检索")
        else:
            logger.warning("⚠️ MemoryGateway: 向量能力未启用，使用规则检索")

    def _init_mem0_client(self):
        """初始化可选 Mem0 客户端（失败不影响主流程）"""
        if not self.settings.mem0_enabled:
            return None
        try:
            from mem0 import AsyncMemory  # type: ignore

            # 这里仅做最小可用接入，详细参数由环境变量或 Mem0 默认配置管理
            return AsyncMemory()
        except Exception as e:
            logger.warning(f"Mem0 客户端初始化失败，将仅使用本地长期记忆: {e}")
            return None

    async def add_memories(self, memories: list[dict[str, Any]]) -> list[str]:
        """
        幂等写入长期记忆。
        先写本地，再尽力异步同步到 Mem0（不阻塞主流程）。
        """
        memory_ids: list[str] = []
        if not memories:
            return memory_ids

        embedding_literals: list[str | None] = [None] * len(memories)
        if self.vector_enabled:
            embedding_literals = await self._embed_memory_contents(memories)

        sync_tasks = []
        for idx, memory in enumerate(memories):
            content = (memory.get("content") or "").strip()
            if not content:
                continue

            fp_source = _normalize_text(content)
            fingerprint = hashlib.sha256(fp_source.encode("utf-8")).hexdigest()
            embedding_literal = embedding_literals[idx]

            record = {
                **memory,
                "fingerprint": fingerprint,
                "embedding": embedding_literal,
                "embedding_model": self.embedding.model if embedding_literal else None,
            }

            memory_id = self.dao.upsert_memory(record)
            memory_ids.append(memory_id)

            if self.mem0_client:
                sync_tasks.append(self._sync_to_mem0(record))

        if sync_tasks:
            await asyncio.gather(*sync_tasks, return_exceptions=True)
        return memory_ids

    async def _embed_memory_contents(self, memories: list[dict[str, Any]]) -> list[str | None]:
        """批量生成 memory embedding（失败自动降级）"""
        tasks = [
            self.embedding.embed((m.get("content") or "").strip())
            for m in memories
        ]
        vectors = await asyncio.gather(*tasks, return_exceptions=True)
        literals: list[str | None] = []
        for item in vectors:
            if isinstance(item, Exception):
                literals.append(None)
                continue
            literals.append(self.embedding.to_pgvector_literal(item))
        return literals

    async def _sync_to_mem0(self, record: dict[str, Any]) -> None:
        """
        尽力同步到 Mem0。
        不依赖固定 SDK 返回结构，避免版本差异导致主流程失败。
        """
        if not self.mem0_client:
            return

        try:
            add_fn = getattr(self.mem0_client, "add", None)
            if not callable(add_fn):
                return

            kwargs = {
                "messages": [{"role": "system", "content": record["content"]}],
                "user_id": record.get("user_id"),
                "metadata": {
                    "scope": record.get("scope"),
                    "group_id": record.get("group_id"),
                    "member_id": record.get("member_id"),
                    "persona_version": record.get("persona_version"),
                    "confidence": record.get("confidence", 0.8),
                    "memory_type": record.get("memory_type"),
                },
            }
            result = add_fn(**kwargs)
            if inspect.isawaitable(result):
                await result
        except Exception as e:
            logger.warning(f"Mem0 同步写入失败（已降级忽略）: {e}")

    def search_scope(
        self,
        *,
        scope: str,
        user_id: str,
        group_id: str | None = None,
        member_id: str | None = None,
        persona_version: str | None = None,
        min_confidence: float = 0.0,
        query_embedding: str | None = None,
        limit: int = 20,
    ) -> list[dict]:
        """按作用域检索候选（本地主存）"""
        return self.dao.list_candidates(
            scope=scope,
            user_id=user_id,
            group_id=group_id,
            member_id=member_id,
            persona_version=persona_version,
            min_confidence=min_confidence,
            query_embedding=query_embedding if self.vector_enabled else None,
            limit=limit,
        )

    async def build_query_embedding(self, query: str) -> str | None:
        """为检索 query 生成 embedding literal"""
        if not self.vector_enabled:
            return None
        vec = await self.embedding.embed(query)
        return self.embedding.to_pgvector_literal(vec)

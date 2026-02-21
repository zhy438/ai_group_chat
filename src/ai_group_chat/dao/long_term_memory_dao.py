"""
长期记忆 DAO

负责长期记忆、归档游标、审计与死信表的数据访问。
"""

import json
from typing import Optional
from uuid import uuid4

from loguru import logger

from .base import BaseDAO


class LongTermMemoryDAO(BaseDAO):
    """长期记忆数据访问对象"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vector_available = self._detect_vector_available()

    def _detect_vector_available(self) -> bool:
        """检测当前数据库是否可用 pgvector + embedding 列"""
        try:
            row = self.db.fetch_one(
                """
                SELECT EXISTS(
                    SELECT 1
                    FROM information_schema.columns
                    WHERE table_name = 'long_term_memories'
                      AND column_name = 'embedding'
                ) AS has_embedding_col
                """
            )
            has_col = bool((row or {}).get("has_embedding_col"))
            if has_col:
                logger.info("✅ LongTermMemoryDAO: pgvector 列可用")
            else:
                logger.warning("⚠️ LongTermMemoryDAO: embedding 列不可用，降级为非向量检索")
            return has_col
        except Exception as e:
            logger.warning(f"⚠️ LongTermMemoryDAO: 检测 pgvector 失败，降级非向量检索: {e}")
            return False

    # ===== 记忆主表 =====

    def _find_duplicate(
        self,
        scope: str,
        user_id: str,
        group_id: str | None,
        member_id: str | None,
        persona_version: str | None,
        fingerprint: str,
    ) -> Optional[dict]:
        sql = """
            SELECT * FROM long_term_memories
            WHERE scope = ?
              AND user_id = ?
              AND COALESCE(group_id, '') = COALESCE(?, '')
              AND COALESCE(member_id, '') = COALESCE(?, '')
              AND COALESCE(persona_version, '') = COALESCE(?, '')
              AND fingerprint = ?
            LIMIT 1
        """
        return self.db.fetch_one(sql, (scope, user_id, group_id, member_id, persona_version, fingerprint))

    def upsert_memory(self, record: dict) -> str:
        """按作用域 + fingerprint 幂等写入长期记忆"""
        duplicate = self._find_duplicate(
            scope=record["scope"],
            user_id=record["user_id"],
            group_id=record.get("group_id"),
            member_id=record.get("member_id"),
            persona_version=record.get("persona_version"),
            fingerprint=record["fingerprint"],
        )
        metadata = record.get("metadata")
        metadata_json = json.dumps(metadata, ensure_ascii=False) if metadata else None
        embedding_literal = record.get("embedding")
        embedding_model = record.get("embedding_model")

        if duplicate:
            if self.vector_available and embedding_literal:
                self.db.execute(
                    """
                    UPDATE long_term_memories
                    SET content = ?,
                        confidence = ?,
                        source_message_id = ?,
                        source_created_at = ?,
                        expires_at = ?,
                        metadata = ?,
                        embedding = ?::vector,
                        embedding_model = ?,
                        embedding_updated_at = CURRENT_TIMESTAMP,
                        is_active = TRUE,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                    """,
                    (
                        record["content"],
                        record.get("confidence", 0.8),
                        record.get("source_message_id"),
                        record.get("source_created_at"),
                        record.get("expires_at"),
                        metadata_json,
                        embedding_literal,
                        embedding_model,
                        duplicate["id"],
                    ),
                )
            else:
                self.db.execute(
                    """
                    UPDATE long_term_memories
                    SET content = ?,
                        confidence = ?,
                        source_message_id = ?,
                        source_created_at = ?,
                        expires_at = ?,
                        metadata = ?,
                        is_active = TRUE,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                    """,
                    (
                        record["content"],
                        record.get("confidence", 0.8),
                        record.get("source_message_id"),
                        record.get("source_created_at"),
                        record.get("expires_at"),
                        metadata_json,
                        duplicate["id"],
                    ),
                )
            return duplicate["id"]

        memory_id = str(uuid4())
        if self.vector_available and embedding_literal:
            self.db.execute(
                """
                INSERT INTO long_term_memories (
                    id, group_id, user_id, member_id, scope, memory_type, content,
                    confidence, fingerprint, persona_version, source_message_id,
                    source_created_at, expires_at, metadata, embedding, embedding_model, embedding_updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?::vector, ?, CURRENT_TIMESTAMP)
                """,
                (
                    memory_id,
                    record.get("group_id"),
                    record["user_id"],
                    record.get("member_id"),
                    record["scope"],
                    record.get("memory_type", "discussion_asset"),
                    record["content"],
                    record.get("confidence", 0.8),
                    record["fingerprint"],
                    record.get("persona_version"),
                    record.get("source_message_id"),
                    record.get("source_created_at"),
                    record.get("expires_at"),
                    metadata_json,
                    embedding_literal,
                    embedding_model,
                ),
            )
        else:
            self.db.execute(
                """
                INSERT INTO long_term_memories (
                    id, group_id, user_id, member_id, scope, memory_type, content,
                    confidence, fingerprint, persona_version, source_message_id,
                    source_created_at, expires_at, metadata
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    memory_id,
                    record.get("group_id"),
                    record["user_id"],
                    record.get("member_id"),
                    record["scope"],
                    record.get("memory_type", "discussion_asset"),
                    record["content"],
                    record.get("confidence", 0.8),
                    record["fingerprint"],
                    record.get("persona_version"),
                    record.get("source_message_id"),
                    record.get("source_created_at"),
                    record.get("expires_at"),
                    metadata_json,
                ),
            )
        return memory_id

    def list_candidates(
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
        """按作用域读取候选长期记忆（仅返回可用数据）"""
        if self.vector_available and query_embedding:
            sql = """
            SELECT *,
                   COALESCE(1 - (embedding <=> ?::vector), 0) AS vector_score
            FROM long_term_memories
            WHERE scope = ?
              AND user_id = ?
              AND confidence >= ?
              AND is_active = TRUE
              AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
            """
            params: list = [query_embedding, scope, user_id, min_confidence]
        else:
            sql = """
            SELECT *, 0::REAL AS vector_score
            FROM long_term_memories
            WHERE scope = ?
              AND user_id = ?
              AND confidence >= ?
              AND is_active = TRUE
              AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
            """
            params = [scope, user_id, min_confidence]

        if group_id is not None:
            sql += " AND COALESCE(group_id, '') = COALESCE(?, '')"
            params.append(group_id)
        if member_id is not None:
            sql += " AND COALESCE(member_id, '') = COALESCE(?, '')"
            params.append(member_id)
        if persona_version is not None:
            sql += " AND COALESCE(persona_version, '') = COALESCE(?, '')"
            params.append(persona_version)

        if self.vector_available and query_embedding:
            sql += " ORDER BY vector_score DESC, updated_at DESC LIMIT ?"
        else:
            sql += " ORDER BY updated_at DESC LIMIT ?"
        params.append(limit)
        return self.db.fetch_all(sql, tuple(params))

    def touch_used(self, memory_ids: list[str]) -> None:
        """更新命中记忆的使用时间和衰减分"""
        if not memory_ids:
            return
        sql = """
            UPDATE long_term_memories
            SET last_used_at = CURRENT_TIMESTAMP,
                decay_score = LEAST(COALESCE(decay_score, 1.0) + 0.05, 1.0)
            WHERE id = ANY(%s)
        """
        # 这里直接使用 execute 的原生 postgres 占位符能力
        self.db.execute(sql, (memory_ids,))

    # ===== 归档游标 =====

    def get_checkpoint(self, group_id: str, user_id: str) -> Optional[dict]:
        return self.db.fetch_one(
            """
            SELECT * FROM memory_checkpoints
            WHERE group_id = ? AND user_id = ?
            """,
            (group_id, user_id),
        )

    def upsert_checkpoint(self, group_id: str, user_id: str, last_message_id: str, last_message_created_at) -> None:
        self.db.execute(
            """
            INSERT INTO memory_checkpoints (group_id, user_id, last_message_id, last_message_created_at, updated_at)
            VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT (group_id, user_id)
            DO UPDATE SET
                last_message_id = EXCLUDED.last_message_id,
                last_message_created_at = EXCLUDED.last_message_created_at,
                updated_at = CURRENT_TIMESTAMP
            """,
            (group_id, user_id, last_message_id, last_message_created_at),
        )

    # ===== 可观测/审计 =====

    def add_dead_letter(self, group_id: str, user_id: str, error: str, payload: dict, retry_count: int) -> None:
        self.db.execute(
            """
            INSERT INTO memory_dead_letters (group_id, user_id, error, payload, retry_count)
            VALUES (?, ?, ?, ?, ?)
            """,
            (group_id, user_id, error, json.dumps(payload, ensure_ascii=False), retry_count),
        )

    def add_audit_log(
        self,
        *,
        request_id: str,
        group_id: str,
        user_id: str,
        event_type: str,
        scope: str | None = None,
        memory_ids: list[str] | None = None,
        detail: str | None = None,
    ) -> None:
        memory_ids_text = ",".join(memory_ids or [])
        self.db.execute(
            """
            INSERT INTO memory_audit_logs (request_id, group_id, user_id, event_type, scope, memory_ids, detail)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (request_id, group_id, user_id, event_type, scope, memory_ids_text, detail),
        )

    def get_group_stats(self, group_id: str) -> dict:
        scope_rows = self.db.fetch_all(
            """
            SELECT scope, COUNT(*) AS cnt
            FROM long_term_memories
            WHERE group_id = ?
              AND is_active = TRUE
            GROUP BY scope
            """,
            (group_id,),
        )
        dead_row = self.db.fetch_one(
            """
            SELECT COUNT(*) AS cnt
            FROM memory_dead_letters
            WHERE group_id = ?
            """,
            (group_id,),
        )
        scope_counts = {row["scope"]: int(row["cnt"]) for row in scope_rows}
        vec_row = self.db.fetch_one(
            """
            SELECT COUNT(*) AS cnt
            FROM long_term_memories
            WHERE group_id = ?
              AND is_active = TRUE
              AND embedding IS NOT NULL
            """,
            (group_id,),
        ) if self.vector_available else {"cnt": 0}
        return {
            "scope_counts": scope_counts,
            "dead_letter_count": int((dead_row or {}).get("cnt", 0)),
            "total_records": int(sum(scope_counts.values())),
            "vector_enabled": bool(self.vector_available),
            "embedded_records": int((vec_row or {}).get("cnt", 0)),
        }


long_term_memory_dao = LongTermMemoryDAO()

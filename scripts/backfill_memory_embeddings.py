#!/usr/bin/env python3
"""
长期记忆 embedding 回填脚本（pgvector）

用途：
1) 给历史 long_term_memories 记录补 embedding。
2) 支持按 group_id / user_id / scope 定向回填。
3) 支持 dry-run 预览，不写库。

示例：
  # 预览待回填数量
  .venv/bin/python scripts/backfill_memory_embeddings.py --dry-run

  # 全量回填（仅 embedding 为空的记录）
  .venv/bin/python scripts/backfill_memory_embeddings.py --batch-size 50

  # 只回填某个群
  .venv/bin/python scripts/backfill_memory_embeddings.py --group-id <GROUP_ID>

  # 强制重算（覆盖已有 embedding）
  .venv/bin/python scripts/backfill_memory_embeddings.py --force
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any

import psycopg2
from openai import AsyncOpenAI


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ai_group_chat.config import get_settings  # noqa: E402


def to_pgvector_literal(vector: list[float]) -> str:
    return "[" + ",".join(f"{float(x):.8f}" for x in vector) + "]"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill embeddings for long_term_memories")
    parser.add_argument("--batch-size", type=int, default=40, help="每批处理数量，默认 40")
    parser.add_argument("--max-rows", type=int, default=0, help="最多处理多少条，0 表示不限")
    parser.add_argument("--group-id", type=str, default="", help="仅处理某个 group_id")
    parser.add_argument("--user-id", type=str, default="", help="仅处理某个 user_id")
    parser.add_argument(
        "--scope",
        type=str,
        default="",
        choices=["", "user_global", "group_local", "agent_local"],
        help="仅处理指定 scope",
    )
    parser.add_argument("--model", type=str, default="", help="embedding 模型，默认取配置 mem_embedding_model")
    parser.add_argument("--dry-run", action="store_true", help="仅预览，不写库")
    parser.add_argument("--force", action="store_true", help="强制覆盖已有 embedding")
    return parser.parse_args()


class EmbeddingBackfiller:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.settings = get_settings()
        self.model = args.model or self.settings.mem_embedding_model
        self.client = AsyncOpenAI(
            base_url=self.settings.ai_api_base,
            api_key=self.settings.ai_api_key,
        )

    def get_conn(self):
        # settings.database_url 形如: postgresql://user:pwd@host:5432/db
        return psycopg2.connect(self.settings.database_url)

    def check_vector_ready(self, conn) -> None:
        cur = conn.cursor()
        cur.execute("SELECT extname FROM pg_extension WHERE extname='vector'")
        has_ext = bool(cur.fetchone())
        if not has_ext:
            raise RuntimeError("数据库未安装 pgvector 扩展（vector）")

        cur.execute(
            """
            SELECT EXISTS(
                SELECT 1
                FROM information_schema.columns
                WHERE table_name='long_term_memories'
                  AND column_name='embedding'
            )
            """
        )
        has_col = bool(cur.fetchone()[0])
        cur.close()
        if not has_col:
            raise RuntimeError("long_term_memories.embedding 列不存在，请先重启后端执行迁移")

    def build_where(self) -> tuple[str, list[Any]]:
        cond = ["is_active = TRUE", "content IS NOT NULL", "content <> ''"]
        params: list[Any] = []

        if not self.args.force:
            cond.append("embedding IS NULL")
        if self.args.group_id:
            cond.append("group_id = %s")
            params.append(self.args.group_id)
        if self.args.user_id:
            cond.append("user_id = %s")
            params.append(self.args.user_id)
        if self.args.scope:
            cond.append("scope = %s")
            params.append(self.args.scope)

        return " AND ".join(cond), params

    def count_candidates(self, conn) -> int:
        where_sql, params = self.build_where()
        cur = conn.cursor()
        cur.execute(f"SELECT COUNT(*) FROM long_term_memories WHERE {where_sql}", tuple(params))
        total = int(cur.fetchone()[0])
        cur.close()
        return total

    def fetch_batch(self, conn, limit: int) -> list[dict]:
        where_sql, params = self.build_where()
        cur = conn.cursor()
        cur.execute(
            f"""
            SELECT id, content
            FROM long_term_memories
            WHERE {where_sql}
            ORDER BY updated_at ASC, id ASC
            LIMIT %s
            """,
            tuple(params + [limit]),
        )
        rows = [{"id": r[0], "content": r[1]} for r in cur.fetchall()]
        cur.close()
        return rows

    async def embed_one(self, text: str, retries: int = 2) -> list[float] | None:
        payload = (text or "").strip()
        if not payload:
            return None

        last_error = None
        for _ in range(retries + 1):
            try:
                resp = await self.client.embeddings.create(
                    model=self.model,
                    input=[payload],
                )
                if not resp.data:
                    return None
                return list(resp.data[0].embedding)
            except Exception as e:
                last_error = e
                await asyncio.sleep(0.6)
        print(f"[WARN] embedding 失败，跳过该条: {last_error}")
        return None

    async def embed_batch(self, rows: list[dict]) -> list[tuple[str, str, str]]:
        tasks = [self.embed_one(row["content"]) for row in rows]
        vectors = await asyncio.gather(*tasks)
        updates: list[tuple[str, str, str]] = []
        for row, vec in zip(rows, vectors):
            if not vec:
                continue
            updates.append((to_pgvector_literal(vec), self.model, row["id"]))
        return updates

    def apply_updates(self, conn, updates: list[tuple[str, str, str]]) -> int:
        if not updates:
            return 0
        cur = conn.cursor()
        cur.executemany(
            """
            UPDATE long_term_memories
            SET embedding = %s::vector,
                embedding_model = %s,
                embedding_updated_at = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = %s
            """,
            updates,
        )
        conn.commit()
        affected = cur.rowcount
        cur.close()
        return affected

    async def run(self) -> None:
        if not self.settings.ai_api_key:
            raise RuntimeError("AI_API_KEY 未配置，无法生成 embedding")

        conn = self.get_conn()
        try:
            self.check_vector_ready(conn)
            total = self.count_candidates(conn)
            print(f"[INFO] 待回填记录数: {total}")
            if self.args.dry_run:
                print("[INFO] dry-run 模式，不执行写库")
                return
            if total <= 0:
                print("[INFO] 无需回填")
                return

            processed = 0
            updated = 0
            max_rows = self.args.max_rows if self.args.max_rows > 0 else 10**12

            while processed < max_rows:
                current_batch = min(self.args.batch_size, max_rows - processed)
                rows = self.fetch_batch(conn, current_batch)
                if not rows:
                    break

                updates = await self.embed_batch(rows)
                affected = self.apply_updates(conn, updates)

                processed += len(rows)
                updated += affected
                print(f"[INFO] batch done: fetched={len(rows)} updated={affected} total_updated={updated}")

            print(f"[DONE] 回填完成: processed={processed}, updated={updated}")
        finally:
            conn.close()


async def main():
    args = parse_args()
    runner = EmbeddingBackfiller(args)
    await runner.run()


if __name__ == "__main__":
    asyncio.run(main())

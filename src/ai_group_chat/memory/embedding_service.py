"""
向量嵌入服务

基于 OpenAI 兼容接口生成文本 embedding。
失败时返回空，调用方降级到非向量检索。
"""

from __future__ import annotations

from typing import Optional

from loguru import logger
from openai import AsyncOpenAI

from ..config import get_settings


class EmbeddingService:
    """Embedding 生成服务"""

    def __init__(self):
        self.settings = get_settings()
        self.enabled = bool(self.settings.mem_vector_enabled)
        self.model = self.settings.mem_embedding_model
        self.dimensions = int(self.settings.mem_embedding_dimensions)
        self.client = AsyncOpenAI(
            base_url=self.settings.ai_api_base,
            api_key=self.settings.ai_api_key,
        )

    async def embed(self, text: str) -> Optional[list[float]]:
        if not self.enabled:
            return None
        payload = (text or "").strip()
        if not payload:
            return None
        try:
            resp = await self.client.embeddings.create(
                model=self.model,
                input=[payload],
            )
            if not resp.data:
                return None
            vec = list(resp.data[0].embedding)
            if self.dimensions > 0 and len(vec) != self.dimensions:
                logger.warning(
                    f"embedding 维度不匹配: got={len(vec)} expected={self.dimensions}, 将按返回值继续"
                )
            return vec
        except Exception as e:
            logger.warning(f"embedding 生成失败，降级非向量检索: {e}")
            return None

    @staticmethod
    def to_pgvector_literal(vector: list[float] | None) -> str | None:
        """将 embedding 转成 pgvector 文本格式"""
        if not vector:
            return None
        return "[" + ",".join(f"{float(x):.8f}" for x in vector) + "]"

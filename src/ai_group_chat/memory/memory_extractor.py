"""
长期记忆抽取器

优先尝试 LLM 结构化抽取，失败时降级为规则抽取。
"""

from __future__ import annotations

import json
import re
from typing import Any

from loguru import logger

from ..llm import llm_client
from ..models import Message
from ..prompts import (
    MEMORY_EXTRACT_SYSTEM_PROMPT,
    build_memory_extract_user_prompt,
)


class MemoryExtractor:
    """长期记忆抽取器"""

    MAX_RETRIES = 2

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model

    async def extract(self, messages: list[Message]) -> list[dict[str, Any]]:
        """从增量原始消息中抽取候选长期记忆"""
        if not messages:
            return []

        text = self._build_conversation_text(messages[-80:])
        extracted = await self._extract_by_llm(text)
        if extracted:
            return self._normalize(extracted)

        logger.warning("长期记忆 LLM 抽取失败，使用规则降级")
        return self._fallback_extract(messages)

    async def _extract_by_llm(self, conversation_text: str) -> list[dict[str, Any]]:
        user_prompt = build_memory_extract_user_prompt(conversation_text)
        last_err = None
        for _ in range(self.MAX_RETRIES):
            try:
                content = await llm_client.chat(
                    model=self.model,
                    messages=[{"role": "user", "content": user_prompt}],
                    system_prompt=MEMORY_EXTRACT_SYSTEM_PROMPT,
                    temperature=0.1,
                    max_tokens=1200,
                )
                data = self._parse_json_array(content)
                if data:
                    return data
            except Exception as e:
                last_err = e
        if last_err:
            logger.warning(f"长期记忆 LLM 抽取异常: {last_err}")
        return []

    @staticmethod
    def _parse_json_array(raw: str) -> list[dict[str, Any]]:
        raw = raw.strip()
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if not match:
            return []
        try:
            arr = json.loads(match.group(0))
            if isinstance(arr, list):
                return [x for x in arr if isinstance(x, dict)]
        except Exception:
            return []
        return []

    @staticmethod
    def _build_conversation_text(messages: list[Message]) -> str:
        lines: list[str] = []
        for msg in messages:
            sender = msg.sender_name or ("用户" if msg.role == "user" else "AI")
            lines.append(f"[{sender}] {msg.content}")
        return "\n".join(lines)

    @staticmethod
    def _normalize(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for item in items:
            scope = str(item.get("scope", "")).strip()
            memory_type = str(item.get("memory_type", "")).strip() or "discussion_asset"
            content = str(item.get("content", "")).strip()
            if scope not in {"user_global", "group_local", "agent_local"} or not content:
                continue

            try:
                confidence = float(item.get("confidence", 0.8))
            except Exception:
                confidence = 0.8
            confidence = max(0.0, min(1.0, confidence))

            normalized.append(
                {
                    "scope": scope,
                    "memory_type": memory_type,
                    "content": content[:200],
                    "confidence": confidence,
                    "sender_name": str(item.get("sender_name", "")).strip() or None,
                }
            )
        return normalized

    @staticmethod
    def _fallback_extract(messages: list[Message]) -> list[dict[str, Any]]:
        """
        规则降级：
        - 用户偏好句 -> user_global
        - 总结/结论句 -> group_local
        """
        user_pref_keywords = ("偏好", "喜欢", "请用", "尽量", "习惯", "以后", "希望你")
        group_asset_keywords = ("结论", "最终", "建议", "方案", "总结", "达成一致")

        results: list[dict[str, Any]] = []
        for msg in messages[-60:]:
            content = (msg.content or "").strip()
            if not content:
                continue

            if msg.role == "user" and any(k in content for k in user_pref_keywords):
                results.append(
                    {
                        "scope": "user_global",
                        "memory_type": "user_profile",
                        "content": content[:160],
                        "confidence": 0.8,
                        "sender_name": msg.sender_name,
                    }
                )
                continue

            if msg.role == "assistant" and any(k in content for k in group_asset_keywords):
                results.append(
                    {
                        "scope": "group_local",
                        "memory_type": "discussion_asset",
                        "content": content[:160],
                        "confidence": 0.76,
                        "sender_name": msg.sender_name,
                    }
                )

        return results[:20]

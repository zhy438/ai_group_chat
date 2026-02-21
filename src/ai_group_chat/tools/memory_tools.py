"""长期记忆相关工具工厂。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from loguru import logger

from ..models import GroupChat

if TYPE_CHECKING:
    from ..memory.long_term_memory_service import LongTermMemoryService


ToolCallable = Callable[..., Any] | Callable[..., Awaitable[Any]]


def create_long_term_memory_search_tool(
    *,
    group: GroupChat,
    user_id: str,
    memory_service: "LongTermMemoryService",
    max_context_tokens: int = 128000,
) -> ToolCallable:
    """创建长期记忆检索工具（供群聊成员共享）。"""

    async def search_long_term_memory(query: str) -> str:
        """
        检索当前用户在本群聊可见作用域内的长期记忆。

        适用场景：
        - 需要回忆用户偏好、历史结论、既往约束
        - 当前问题与过去讨论有关联
        """
        cleaned = (query or "").strip()
        if not cleaned:
            return "检索失败：query 不能为空。"
        logger.info(
            f"🛠️ long_memory_tool invoked: group_id={group.id}, user_id={user_id}, query={cleaned[:80]}"
        )

        block = await memory_service.build_injection_context(
            group=group,
            user_id=user_id,
            query=cleaned,
            max_context_tokens=max_context_tokens,
        )
        if not block:
            logger.info(
                f"🛠️ long_memory_tool empty: group_id={group.id}, user_id={user_id}"
            )
            return "未检索到匹配的长期记忆，请基于当前对话继续推理。"
        logger.info(
            f"🛠️ long_memory_tool hit: group_id={group.id}, user_id={user_id}"
        )
        return block

    search_long_term_memory.__name__ = "search_long_term_memory"
    return search_long_term_memory

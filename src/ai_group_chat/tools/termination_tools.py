"""讨论终止工具工厂。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from loguru import logger

from ..models import GroupChat

ToolCallable = Callable[..., Any] | Callable[..., Awaitable[Any]]
TERMINATE_DISCUSSION_TOOL_NAME = "terminate_discussion"


def create_manager_terminate_tool(*, group: GroupChat, user_id: str) -> ToolCallable:
    """创建仅供管理员使用的讨论终止工具。"""

    async def terminate_discussion(reason: str = "当前话题已形成可执行结论") -> str:
        """
        终止当前讨论回合。

        适用时机：
        - 讨论目标已达成，继续讨论只会重复
        - 讨论明显偏题或进入无效寒暄
        """
        cleaned_reason = (reason or "").strip() or "当前话题已形成可执行结论"
        logger.info(
            f"🛑 terminate_tool invoked: group_id={group.id}, user_id={user_id}, reason={cleaned_reason[:120]}"
        )
        return f"已确认提前终止讨论：{cleaned_reason}"

    terminate_discussion.__name__ = TERMINATE_DISCUSSION_TOOL_NAME
    return terminate_discussion

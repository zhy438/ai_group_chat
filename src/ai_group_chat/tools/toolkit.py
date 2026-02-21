"""共享工具集构建器。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..models import GroupChat
from .memory_tools import create_long_term_memory_search_tool
from .termination_tools import create_manager_terminate_tool

if TYPE_CHECKING:
    from ..memory.long_term_memory_service import LongTermMemoryService


ToolCallable = Callable[..., Any] | Callable[..., Awaitable[Any]]


@dataclass(slots=True)
class GroupToolkitBundle:
    """群聊工具集拆分：成员共享工具 + 管理员专属工具。"""

    member_tools: list[ToolCallable]
    manager_tools: list[ToolCallable]


def build_group_toolkits(
    *,
    group: GroupChat,
    user_id: str,
    memory_service: "LongTermMemoryService",
    max_context_tokens: int,
) -> GroupToolkitBundle:
    """构建群聊工具集。"""
    member_tools: list[ToolCallable] = []
    manager_tools: list[ToolCallable] = [
        create_manager_terminate_tool(group=group, user_id=user_id),
    ]

    # 仅在记忆检索开启时为成员注入长期记忆工具。
    if group.memory_enabled and group.retrieve_enabled:
        member_tools.append(
            create_long_term_memory_search_tool(
                group=group,
                user_id=user_id,
                memory_service=memory_service,
                max_context_tokens=max_context_tokens,
            )
        )

    return GroupToolkitBundle(member_tools=member_tools, manager_tools=manager_tools)


def build_shared_toolkit(
    *,
    group: GroupChat,
    user_id: str,
    memory_service: "LongTermMemoryService",
    max_context_tokens: int,
) -> list[ToolCallable]:
    """兼容旧接口：仅返回成员共享工具。"""
    return build_group_toolkits(
        group=group,
        user_id=user_id,
        memory_service=memory_service,
        max_context_tokens=max_context_tokens,
    ).member_tools

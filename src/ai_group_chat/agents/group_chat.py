"""基于 AutoGen 的 AI 群聊实现 (新版 API)"""

import re
import asyncio
from collections import Counter
from collections.abc import Awaitable, Callable
from typing import Any, AsyncGenerator
from loguru import logger
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage, ToolCallExecutionEvent
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.conditions import ExternalTermination, FunctionCallTermination, MaxMessageTermination
from autogen_agentchat.base import TaskResult
from autogen_core.models import ModelInfo
from autogen_ext.models.openai import OpenAIChatCompletionClient

from ..models import AIMember, DiscussionMode
from ..config import get_settings
from ..prompts import (
    SELECTOR_PROMPT,
    DISCUSSION_SUMMARIZER_SYSTEM_PROMPT,
    build_manager_system_prompt,
    build_member_system_prompt,
)
from ..tools import TERMINATE_DISCUSSION_TOOL_NAME


# 默认管理员模型
DEFAULT_MANAGER_MODEL = "qwen-flash"
ToolCallable = Callable[..., Any] | Callable[..., Awaitable[Any]]
INTERNAL_STREAM_MESSAGE_TYPES = {
    "ToolCallRequestEvent",
    "ToolCallExecutionEvent",
    "ToolCallSummaryMessage",
    "ModelClientStreamingChunkEvent",
    "ThoughtEvent",
    "SelectSpeakerEvent",
    "SelectorEvent",
    "MemoryQueryEvent",
    "CodeGenerationEvent",
    "CodeExecutionEvent",
}
TOOL_TRACE_PREFIXES = (
    "[FunctionCall(",
    "[FunctionExecutionResult(",
    "FunctionCall(",
    "FunctionExecutionResult(",
)


def _sanitize_name(name: str) -> str:
    """将名称转换为 AutoGen 兼容格式"""
    name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    if not re.match(r'^[a-zA-Z_]', name):
        name = '_' + name
    return name


def _build_unique_name(base_name: str, used_names: set[str]) -> str:
    """构造不重复的 agent 名称，避免同模型多实例冲突"""
    base = _sanitize_name(base_name or "agent")
    candidate = base
    idx = 2
    while candidate in used_names:
        candidate = f"{base}_{idx}"
        idx += 1
    used_names.add(candidate)
    return candidate


def _safe_signature(source: Any, content: Any) -> tuple[str, str]:
    """构造可哈希的消息签名，避免 list/dict content 导致 set 查询报错。"""
    src = str(source or "")
    if isinstance(content, str):
        body = content
    else:
        body = repr(content)
    return src, body


def _is_user_visible_stream_message(message: Any) -> bool:
    """
    判断消息是否应推送到前端聊天区。
    过滤工具调用事件与非字符串内容，避免渲染 FunctionCall/Execution 原始对象。
    """
    msg_type = getattr(message, "type", type(message).__name__)
    if msg_type in INTERNAL_STREAM_MESSAGE_TYPES:
        return False
    content = getattr(message, "content", None)
    if not isinstance(content, str):
        return False
    text = content.strip()
    if not text:
        return False
    # 防止部分模型把函数调用对象串成字符串回传到聊天区
    if text.startswith(TOOL_TRACE_PREFIXES):
        return False
    return True


def _get_model_client(
    model_id: str,
    temperature: float = 0.7,
    thinking: bool = False,
) -> OpenAIChatCompletionClient:
    """获取模型客户端"""
    settings = get_settings()
    
    model_info = ModelInfo(
        vision=False,
        function_calling=True,
        json_output=True,
        family="unknown",
    )
    
    extra_kwargs = {}
    if thinking:
        extra_kwargs["extra_body"] = {"enable_thinking": True}
    
    return OpenAIChatCompletionClient(
        model=model_id,
        base_url=settings.ai_api_base,
        api_key=settings.ai_api_key,
        model_info=model_info,
        temperature=temperature,
        **extra_kwargs,
    )


def _build_system_prompt(
    member: AIMember,
    all_members: list[AIMember],
    mode: DiscussionMode,
    agent_name_map: dict[str, str],
    tool_names: list[str] | None = None,
    manager_name: str | None = None,
) -> str:
    """构建成员的系统提示词"""

    my_name = agent_name_map.get(member.id, _sanitize_name(member.name or member.model_id))
    other_members = [agent_name_map.get(m.id, _sanitize_name(m.name or m.model_id)) for m in all_members if m.id != member.id]
    members_str = "、".join(other_members) if other_members else "暂无其他成员"
    return build_member_system_prompt(
        my_name=my_name,
        members_str=members_str,
        persona=member.description or "",
        mode=mode,
        tool_names=tool_names,
        manager_name=manager_name,
    )


class AIGroupChat:
    """
    基于 AutoGen 的 AI 群聊
    
    使用原生 Team 组件 (SelectorGroupChat / RoundRobinGroupChat)
    """
    
    def __init__(
        self,
        members: list[AIMember],
        user_name: str = "用户",
        max_rounds: int = 2,
        mode: DiscussionMode = DiscussionMode.FREE,
        manager_model: str = DEFAULT_MANAGER_MODEL,
        manager_thinking: bool = False,
        manager_temperature: float = 0.7,
        history: list[TextMessage] | None = None,
        shared_tools: list[ToolCallable] | None = None,
        manager_tools: list[ToolCallable] | None = None,
        external_termination: ExternalTermination | None = None,
    ):
        # 计算最大消息数：历史消息数 + 本轮限制 (每轮每个成员发言一次 + 用户问题)
        # 若系统Agent可终止，额外预留少量发言配额。
        history = list(history or [])
        manager_slots = 2 if manager_tools else 0
        max_messages = len(history) + (max_rounds * len(members) + 1 + manager_slots)
        self.members = members
        self.user_name = user_name
        self.mode = mode
        self.history = history
        self.manager_model = manager_model
        self.manager_thinking = manager_thinking
        self.manager_temperature = manager_temperature
        self.agents: list[AssistantAgent] = []
        self.member_agents: list[AssistantAgent] = []
        self.shared_tools = list(shared_tools or [])
        self.manager_tools = list(manager_tools or [])
        self.system_agent_name: str | None = None
        self.last_stop_reason: str | None = None
        self.last_system_termination_reason: str | None = None
        self.external_termination = external_termination
        tool_names = [getattr(tool, "__name__", type(tool).__name__) for tool in self.shared_tools]
        manager_tool_names = [getattr(tool, "__name__", type(tool).__name__) for tool in self.manager_tools]
        
        logger.info(f"🔧 初始化群聊: {len(members)} 个成员, 模式: {mode}, 管理模型: {manager_model}")
        
        # 名称映射
        self.name_map = {}
        self.agent_name_map: dict[str, str] = {}
        used_names: set[str] = set()
        if self.manager_tools:
            self.system_agent_name = _build_unique_name("system_agent", used_names)
            self.name_map[self.system_agent_name] = "系统"

        # 第一阶段：为每个成员分配唯一 agent 名称
        for member in members:
            agent_name = _build_unique_name(member.name or member.model_id, used_names)
            self.agent_name_map[member.id] = agent_name
            self.name_map[agent_name] = member.name or member.model_id

        # 第二阶段：创建 Agents
        for member in members:
            agent_name = self.agent_name_map[member.id]
            
            logger.info(f"  👤 创建 Agent: {agent_name} (成员: {member.name}, 模型: {member.model_id})")
            
            agent = AssistantAgent(
                name=agent_name,
                system_message=_build_system_prompt(
                    member=member,
                    all_members=members,
                    mode=mode,
                    agent_name_map=self.agent_name_map,
                    tool_names=tool_names,
                    manager_name=self.system_agent_name,
                ),
                description=f"普通成员。人设：{member.description or '普通群友'}",
                model_client=_get_model_client(
                    member.model_id,
                    temperature=member.temperature,
                    thinking=member.thinking,
                ),
                tools=self.shared_tools or None,
                max_tool_iterations=3,
            )
            self.member_agents.append(agent)
            self.agents.append(agent)

        if self.manager_tools and self.system_agent_name:
            manager_agent = AssistantAgent(
                name=self.system_agent_name,
                description="【系统Agent】仅在需要终止讨论时被选择，并执行终止工具。",
                system_message=build_manager_system_prompt(
                    my_name=self.system_agent_name,
                    members_str="、".join(self.name_map[self.agent_name_map[m.id]] for m in members),
                    tool_name=TERMINATE_DISCUSSION_TOOL_NAME,
                ),
                model_client=_get_model_client(
                    manager_model,
                    temperature=manager_temperature,
                    thinking=manager_thinking,
                ),
                tools=self.manager_tools,
                max_tool_iterations=1,
            )
            self.agents.append(manager_agent)
        
        # 创建 Team
        termination = MaxMessageTermination(max_messages=max_messages)
        if self.external_termination:
            termination = termination | self.external_termination
        if TERMINATE_DISCUSSION_TOOL_NAME in manager_tool_names:
            termination = termination | FunctionCallTermination(TERMINATE_DISCUSSION_TOOL_NAME)
        
        self.team = SelectorGroupChat(
            participants=self.agents,
            model_client=_get_model_client(
                manager_model,
                temperature=manager_temperature,
                thinking=manager_thinking,
            ),
            termination_condition=termination,
            selector_prompt=SELECTOR_PROMPT,
        )

    def was_terminated_by_system(self) -> bool:
        """本轮是否由系统Agent终止工具触发结束。"""
        return bool(self.last_stop_reason and TERMINATE_DISCUSSION_TOOL_NAME in self.last_stop_reason)

    def was_terminated_externally(self) -> bool:
        """本轮是否由外部终止（手动停止/客户端断开）触发结束。"""
        return bool(self.last_stop_reason and "External termination requested" in self.last_stop_reason)

    @staticmethod
    def _extract_system_termination_reason(messages: list[Any]) -> str | None:
        """从框架消息中提取系统Agent终止工具执行结果。"""
        for message in messages:
            if not isinstance(message, ToolCallExecutionEvent):
                continue
            for execution in message.content:
                if execution.name == TERMINATE_DISCUSSION_TOOL_NAME and execution.content:
                    return execution.content.strip()
        return None
    

    
    async def stream_qa_discussion(self, question: str) -> AsyncGenerator[dict, None]:
        """
        [QA模式] 一问一答并发模式
        """
        logger.info(f"🚀 开始并发问答 (QA): {question}")
        
        # 构造用户消息
        user_msg = TextMessage(content=question, source="user")
        
        # 定义单个 Agent 的生成任务
        async def generate_reply(agent):
            # 获取 agent 对应的原始 name
            display_name = self.name_map.get(agent.name, agent.name)
            
            try:
                # 使用 agent.on_messages 直接生成回复
                # 将历史消息作为上下文传入
                messages = self.history + [user_msg]
                response = await agent.on_messages(
                    messages=messages,
                    cancellation_token=None,
                )
                
                content = response.chat_message.content
                logger.info(f"✅ {display_name} 回复完成")
                return {"sender": display_name, "content": content}
                
            except Exception as e:
                logger.error(f"❌ {display_name} 生成失败: {e}")
                return {"sender": display_name, "content": f"生成失败: {str(e)}"}

        # 并发执行所有任务
        tasks = [generate_reply(agent) for agent in self.member_agents]
        
        # 使用 as_completed 逐个 yield 完成的结果
        for coro in asyncio.as_completed(tasks):
            result = await coro
            yield result

    
    async def stream_discuss(
        self,
        question: str,
        max_rounds: int = 2,
    ) -> AsyncGenerator[dict, None]:
        """
        流式讨论，使用 Team 的原生流式方法
        """
        logger.info(f"🚀 开始讨论: {question}")
        self.last_stop_reason = None
        self.last_system_termination_reason = None
        
        msg_count = 0
        # 收集框架返回的原始消息对象
        framework_messages = []

        # 记录历史消息指纹计数，仅用于过滤“本次 run_stream 起始回放”的历史内容。
        # 注意不能用 set，否则当模型新回复与历史文本完全一致时会被误判为历史并永久跳过。
        history_signatures = Counter()
        if self.history:
            for h in self.history:
                history_signatures[_safe_signature(h.source, h.content)] += 1
        
        task = self.history + [TextMessage(content=question, source="user")] if self.history else question
        async for message in self.team.run_stream(task=task):
            # TaskResult 表示结束
            if isinstance(message, TaskResult):
                self.last_stop_reason = message.stop_reason
                if self.was_terminated_by_system():
                    self.last_system_termination_reason = self._extract_system_termination_reason(
                        list(message.messages)
                    )
                    logger.info(
                        f"🛑 讨论由系统Agent终止工具提前结束: reason={self.last_system_termination_reason or '-'}"
                    )
                logger.info(f"✅ 讨论结束，共 {msg_count} 条 AI 回复")
                # 输出最终的框架对话历史
                self._log_framework_history(message.messages, "最终")
                break
            
            # 收集消息
            framework_messages.append(message)
            
            # 检查是否有 source 属性
            if hasattr(message, 'source'):
                # 跳过用户消息
                if message.source == "user":
                    continue
                
                # 跳过历史消息回显（按计数扣减，只过滤回放次数，不误伤后续同文新消息）
                sig = _safe_signature(message.source, message.content)
                if history_signatures.get(sig, 0) > 0:
                    history_signatures[sig] -= 1
                    if history_signatures[sig] <= 0:
                        history_signatures.pop(sig, None)
                    logger.debug(f"🚫 跳过历史消息回显: {message.source}")
                    continue

                if not _is_user_visible_stream_message(message):
                    logger.debug(f"⏭️ 跳过内部事件: {getattr(message, 'type', type(message).__name__)}")
                    continue
                
                display_name = self.name_map.get(message.source, message.source)
                
                # 输出 selector 选择信息
                logger.info(f"🎯 Selector 选择发言: {display_name}")
                
                content = message.content
                msg_count += 1

                # 输出当前框架收集的对话历史
                self._log_framework_history(framework_messages, "当前")

                yield {"sender": display_name, "content": content}
    
    def _log_framework_history(self, messages: list, label: str = ""):
        """输出框架管理的对话历史"""
        logger.info("=" * 60)
        logger.info(f"📋 {label}对话历史 ({len(messages)} 条):")
        for i, msg in enumerate(messages):
            src = getattr(msg, 'source', 'unknown')
            # 转换为显示名
            display_src = self.name_map.get(src, src)
            content = getattr(msg, 'content', str(msg))
            msg_type = type(msg).__name__
            if isinstance(content, str):
                text = content[:120] + '...' if len(content) > 120 else content
            else:
                text = str(content)[:120]
            logger.info(f"  [{i+1}] ({msg_type}) {display_src}: {text}")
        logger.info("=" * 60)
    
    async def discuss(
        self,
        question: str,
        max_rounds: int = 2,
    ) -> list[dict]:
        """
        同步讨论，返回所有消息
        """
        messages = []
        async for msg in self.stream_discuss(question, max_rounds):
            messages.append(msg)
        return messages

    async def summarize(self, instruction: str = "请总结上述讨论") -> dict:
        """
        使用管理员模型对历史讨论进行总结
        """
        logger.info(f"📝 开始总结讨论: {instruction}")
        
        # 创建总结 Agent (复用 Manager Model)
        model_client = _get_model_client(
            self.manager_model, 
            thinking=self.manager_thinking, 
            temperature=self.manager_temperature
        )
        agent = AssistantAgent(
            name="DialogSummarizer",
            model_client=model_client,
            system_message=DISCUSSION_SUMMARIZER_SYSTEM_PROMPT,
        )
        
        # 构造输入
        # 历史已经在 self.history 中
        user_msg = TextMessage(content=instruction, source="user")
        messages = self.history + [user_msg]
        
        try:
            response = await agent.on_messages(messages, cancellation_token=None)
            content = response.chat_message.content
            return {"sender": "总结助手", "content": content}
        except Exception as e:
            logger.error(f"总结失败: {e}")
            return {"sender": "系统", "content": f"总结生成失败: {str(e)}"}

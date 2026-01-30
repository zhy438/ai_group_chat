"""基于 AutoGen 的 AI 群聊实现 (新版 API)"""

import re
import asyncio
from typing import AsyncGenerator
from loguru import logger
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.teams import SelectorGroupChat, RoundRobinGroupChat
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.base import TaskResult
from autogen_core.models import ModelInfo
from autogen_ext.models.openai import OpenAIChatCompletionClient

from ..models import AIMember, DiscussionMode
from ..config import get_settings
from autogen_agentchat.teams import SelectorGroupChat, RoundRobinGroupChat
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.base import TaskResult
from autogen_core.models import ModelInfo
from autogen_ext.models.openai import OpenAIChatCompletionClient

from ..models import AIMember, DiscussionMode
from ..config import get_settings


# 默认管理员模型
DEFAULT_MANAGER_MODEL = "gpt-4o-mini"


def _sanitize_name(name: str) -> str:
    """将名称转换为 AutoGen 兼容格式"""
    name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    if not re.match(r'^[a-zA-Z_]', name):
        name = '_' + name
    return name


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


def _build_system_prompt(member: AIMember, all_members: list[AIMember], mode: DiscussionMode) -> str:
    """构建成员的系统提示词"""
    
    # 使用 sanitized name 作为身份标识
    my_name = _sanitize_name(member.model_id)
    other_members = [_sanitize_name(m.model_id) for m in all_members if m.model_id != member.model_id]
    members_str = "、".join(other_members) if other_members else "暂无其他成员"
    
    base_prompt = f"""
你是一个ai智能助手，你的名字是"{my_name}"，你正在一个群聊里和其他ai助手聊天，目的是解决用户的问题

【群成员列表】
群里除了你之外还有：{members_str}
（如果要@某人，请使用上面的名字，不要编造不存在的名字）

【重要规则】
1. 你们的任务是解决用户的问题，一切的回答都是要以解决用户问题为目的
2. 可以用口语化表达，偶尔用表情符号
3. 如果不知道就说不知道，不要编造
4. 可以@其他群友的名字来回应ta的观点（不强制），但绝对不要@自己（你的名字是"{my_name}"）
5. 回复需要言简意赅，除非问题确实需要详细解答
6. 如果已经得出结论了，可以简单附和或点评，不要重复之前说过的话

【你的人设】
{member.description or '普通群友，性格随和'}

【绝对禁止】
绝对不要在回复中包含 @{my_name}！这是在@你自己，是错误的！
"""
    
    # 根据模式调整提示词
    if mode == DiscussionMode.QA:
        base_prompt += "\n【当前模式：一问一答 (QA)】\n请直接回答用户的问题，提供高质量、独立的见解。\n尽力减少与其他群成员的闲聊或互动，除非必须引用他人的观点。\n重点在于展示你独特的视角和知识。"
    else:
        base_prompt += "\n【当前模式：自由讨论】\n自然地参与讨论，积极与其他成员互动，可以补充、附和或提出不同看法。\n通过协作和交流来解决问题。"
    
    return base_prompt


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
        history: list[TextMessage] = [],
    ):
        # 计算最大消息数：历史消息数 + 本轮限制 (每轮每个成员发言一次 + 用户问题)
        # 注意：AutoGen 的 MaxMessageTermination 计算的是总消息数
        max_messages = len(history) + (max_rounds * len(members) + 1)
        self.members = members
        self.user_name = user_name
        self.mode = mode
        self.history = history
        self.manager_model = manager_model
        self.manager_thinking = manager_thinking
        self.manager_temperature = manager_temperature
        self.agents: list[AssistantAgent] = []
        
        logger.info(f"🔧 初始化群聊: {len(members)} 个成员, 模式: {mode}, 管理员: {manager_model}")
        
        # 名称映射
        self.name_map = {}

        # 创建 Agents
        for member in members:
            # 必须使用合法的 Python 标识符
            agent_name = _sanitize_name(member.model_id)
            self.name_map[agent_name] = member.model_id
            
            logger.info(f"  👤 创建 Agent: {agent_name} (原名: {member.model_id})")
            
            agent = AssistantAgent(
                name=agent_name,
                system_message=_build_system_prompt(member, members, mode),
                model_client=_get_model_client(
                    member.model_id,
                    temperature=member.temperature,
                    thinking=member.thinking,
                ),
            )
            self.agents.append(agent)
        
        # 创建 Team
        termination = MaxMessageTermination(max_messages=max_messages)
        
        # 自定义 selector 提示词（群管理员）
        selector_prompt = """你是一个群聊的主持人，负责决定下一个谁来发言。

当前群成员：{participants}

各成员简介：
{roles}

最近的对话历史：
{history}

【选择规则】
1. 优先让还没发言过或发言较少的成员发言
2. 如果有人被@了，优先让被@的人回复
3. 避免同一个人连续发言
4. 如果讨论已经收敛（大家意见一致），可以让新的角度的人发言

请只回复下一个发言者的名字，不要有其他内容。"""
        
        self.team = SelectorGroupChat(
            participants=self.agents,
            model_client=_get_model_client(
                manager_model,
                temperature=manager_temperature,
                thinking=manager_thinking,
            ),
            termination_condition=termination,
            selector_prompt=selector_prompt,
        )
    

    
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
        tasks = [generate_reply(agent) for agent in self.agents]
        
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
        
        msg_count = 0
        # 收集框架返回的原始消息对象
        framework_messages = []

        # 记录历史消息指纹以防回显
        history_signatures = set()
        if self.history:
            for h in self.history:
                history_signatures.add((h.source, h.content))
        
        task = self.history + [TextMessage(content=question, source="user")] if self.history else question
        async for message in self.team.run_stream(task=task):
            # TaskResult 表示结束
            if isinstance(message, TaskResult):
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
                
                # 跳过历史消息回显
                if self.history and (message.source, message.content) in history_signatures:
                    logger.debug(f"🚫 跳过历史消息回显: {message.source}")
                    continue
                
                display_name = self.name_map.get(message.source, message.source)
                
                # 输出 selector 选择信息
                logger.info(f"🎯 Selector 选择发言: {display_name}")
                
                if hasattr(message, 'content'):
                    content = message.content if isinstance(message.content, str) else str(message.content)
                    if content.strip():
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
            system_message="你是一个专业的讨论记录员和总结者。请仔细阅读提供的对话历史，提炼出核心议题、各方观点、达成的共识以及任何悬而未决的问题。最终得出一个清晰的结论。"
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

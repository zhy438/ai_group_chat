"""聊天服务 - 业务逻辑层"""

import re
from loguru import logger
from autogen_agentchat.messages import TextMessage

from ..models import (
    GroupChat, GroupChatCreate,
    AIMember, AIMemberCreate, AIMemberUpdate,
    Message, MessageRole,
    DiscussionRequest, DiscussionResponse, SummarizeRequest,
    DiscussionMode,
)
from ..agents import AIGroupChat
from .chat_repository import ChatRepository

# 容错导入预设数据
try:
    from .presets import PRESET_GROUPS
except (ImportError, ModuleNotFoundError):
    PRESET_GROUPS = []


class ChatService:
    """
    聊天服务
    
    业务逻辑层：负责编排业务流程、调用 Repository 进行数据存取。
    不包含任何 SQL 语句。
    """
    
    def __init__(self):
        self.repo = ChatRepository()
        self._load_presets()
    
    def _load_presets(self):
        """加载预设测试数据"""
        if not PRESET_GROUPS:
            return
        
        for preset in PRESET_GROUPS:
            if self.repo.get_group_by_name(preset["name"]):
                continue

            # 创建群聊 (Use repository)
            group = self.repo.create_group(
                name=preset["name"],
                manager_model=preset.get("manager_model", "gpt-4o-mini"),
                discussion_mode=DiscussionMode.FREE
            )
            
            # 添加成员
            for member_data in preset["members"]:
                self.repo.add_raw_member(
                    group_id=group.id,
                    name=member_data["name"],
                    model_id=member_data["model_id"],
                    description=member_data.get("description"),
                    thinking=member_data.get("thinking", False),
                    temperature=member_data.get("temperature", 0.7)
                )
            
            logger.info(f"📦 初始化预设群聊: {preset['name']} ({len(preset['members'])} 个成员)")

    # ============ 群聊管理 ============
    
    def create_group(self, data: GroupChatCreate) -> GroupChat:
        return self.repo.create_group(data.name, data.mode)
    
    def get_group(self, group_id: str) -> GroupChat | None:
        return self.repo.get_group(group_id)
    
    def list_groups(self) -> list[GroupChat]:
        return self.repo.list_groups()
    
    def delete_group(self, group_id: str) -> bool:
        return self.repo.delete_group(group_id)
    
    # ============ 成员管理 ============
    
    def add_member(self, group_id: str, data: AIMemberCreate) -> AIMember | None:
        if not self.repo.get_group(group_id):
            return None
        return self.repo.add_member(group_id, data)
    
    def update_member(self, group_id: str, member_id: str, data: AIMemberUpdate) -> AIMember | None:
        return self.repo.update_member(group_id, member_id, data)
    
    def set_manager_config(self, group_id: str, model_id: str, thinking: bool = None, temperature: float = None) -> bool:
        if not self.repo.get_group(group_id):
            return False
        return self.repo.update_manager_config(group_id, model_id, thinking, temperature)
    
    def remove_member(self, group_id: str, member_id: str) -> bool:
        return self.repo.remove_member(group_id, member_id)
    
    def update_member_task(self, group_id: str, member_id: str, task: str) -> bool:
        return self.repo.update_member_persona(group_id, member_id, task)
    
    # ============ 讨论功能 ============

    async def start_discussion(self, group_id: str, request: DiscussionRequest) -> DiscussionResponse:
        """启动群聊讨论"""
        group = self.get_group(group_id)
        if not group or not group.members:
            raise ValueError("群聊不存在或没有成员")
        
        mode = request.mode if request.mode else group.discussion_mode

        # 保存用户消息
        self.repo.save_message(group_id, MessageRole.USER, request.content, request.user_name, mode)

        if mode == DiscussionMode.QA:
             # QA 模式不需要很长的上下文
             history_msgs = []
        else:
            # FREE 模式需要上下文
            history_msgs = self._get_history_as_autogen_messages(group_id, limit=50, exclude_last=True)
        
        ai_group_chat = AIGroupChat(
            members=group.members,
            user_name=request.user_name,
            mode=mode,
            history=history_msgs,
        )
        
        # 运行讨论
        if mode == DiscussionMode.QA:
            messages_data = []
            async for msg in ai_group_chat.stream_qa_discussion(request.content):
                messages_data.append(msg)
        else:
             messages_data = await ai_group_chat.discuss(
                question=request.content,
                max_rounds=request.max_rounds,
            )
        
        # 保存结果
        result_messages = []
        for msg_data in messages_data:
            message = self.repo.save_message(
                group_id, 
                MessageRole.ASSISTANT, 
                msg_data["content"], 
                msg_data["sender"], 
                mode
            )
            result_messages.append(message)
        
        return DiscussionResponse(messages=result_messages, summary=None)
    
    async def stream_discussion(self, group_id: str, request: DiscussionRequest):
        """流式启动群聊讨论"""
        group = self.get_group(group_id)
        if not group or not group.members:
            raise ValueError("群聊不存在或没有成员")
        
        mode = request.mode if request.mode else group.discussion_mode

        # 保存用户消息
        self.repo.save_message(group_id, MessageRole.USER, request.content, request.user_name, mode)

        if mode == DiscussionMode.QA:
            history_msgs = []
        else:
            history_msgs = self._get_history_as_autogen_messages(group_id, limit=50, exclude_last=True)

        ai_group_chat = AIGroupChat(
            members=group.members,
            user_name=request.user_name,
            max_rounds=request.max_rounds,
            mode=mode,
            manager_model=group.manager_model,
            manager_thinking=group.manager_thinking,
            manager_temperature=group.manager_temperature,
            history=history_msgs,
        )

        if mode == DiscussionMode.QA:
            generator = ai_group_chat.stream_qa_discussion(request.content)
        else:
            generator = ai_group_chat.stream_discuss(request.content, request.max_rounds)
            
        async for msg_data in generator:
            message = self.repo.save_message(
                group_id, 
                MessageRole.ASSISTANT, 
                msg_data["content"], 
                msg_data["sender"], 
                mode
            )
            yield message
            
    async def summarize_discussion(self, group_id: str, request: SummarizeRequest):
        """对群聊进行总结"""
        group = self.get_group(group_id)
        if not group: return
        
        history_msgs = self._get_history_as_autogen_messages(group_id, limit=100)
        
        ai_group_chat = AIGroupChat(
            members=group.members,
            user_name="User",
            mode=DiscussionMode.FREE,
            manager_model=group.manager_model, 
            history=history_msgs
        )
        
        result = await ai_group_chat.summarize(request.instruction)
        
        message = self.repo.save_message(
            group_id,
            MessageRole.ASSISTANT,
            result["content"],
            result["sender"],
            DiscussionMode.FREE
        )
        yield message
    
    def _get_history_as_autogen_messages(self, group_id: str, limit: int = 50, exclude_last: bool = False) -> list[TextMessage]:
        """获取群聊历史并转换为 AutoGen 格式 (Helper)"""
        # 注意：这里调用 self.get_messages，它最终调用 repo.get_messages
        messages = self.get_messages(group_id, limit + 1 if exclude_last else limit)
        if exclude_last and messages:
            messages = messages[:-1]
            
        autogen_msgs = []
        for msg in messages:
            source = "user" if msg.role == MessageRole.USER else _sanitize_name(msg.sender_name)
            autogen_msgs.append(TextMessage(content=msg.content, source=source))
        return autogen_msgs
    
    def get_messages(self, group_id: str, limit: int = 50) -> list[Message]:
        return self.repo.get_messages(group_id, limit)


def _sanitize_name(name: str) -> str:
    """将名称转换为 AutoGen 兼容格式"""
    if not name:
        return "unknown"
    name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    if not re.match(r'^[a-zA-Z_]', name):
        name = '_' + name
    return name


# 全局服务实例
chat_service = ChatService()

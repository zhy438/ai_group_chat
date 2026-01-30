"""Pydantic 数据模型定义"""

from pydantic import BaseModel, Field
from typing import Optional, Literal
from datetime import datetime
from enum import Enum


# ============ 枚举类型 ============

class DiscussionMode(str, Enum):
    """讨论模式"""
    FREE = "free"           # 自由讨论
    QA = "qa"               # 一问一答


class MessageRole(str, Enum):
    """消息角色"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


# ============ AI成员模型 ============

class AIMember(BaseModel):
    """群聊中的AI成员"""
    id: str
    name: str                           # 显示名称
    model_id: str                       # 模型ID
    avatar: Optional[str] = None        # 头像URL
    description: Optional[str] = None   # 模型描述/人设
    task: Optional[str] = None          # 任务模式下分配的任务
    thinking: bool = False              # 是否开启深度思考
    temperature: float = 0.7            # 温度参数


class AIMemberCreate(BaseModel):
    """创建AI成员的请求"""
    name: str
    model_id: str
    avatar: Optional[str] = None
    description: Optional[str] = None
    thinking: bool = False
    temperature: float = 0.7


class AIMemberUpdate(BaseModel):
    """更新AI成员参数"""
    description: Optional[str] = None
    thinking: Optional[bool] = None
    temperature: Optional[float] = None


# ============ 群聊模型 ============

class GroupChat(BaseModel):
    """群聊"""
    id: str
    name: str
    members: list[AIMember] = []
    manager_model: str = "gpt-4o-mini"
    manager_thinking: bool = False
    manager_temperature: float = 0.7
    discussion_mode: DiscussionMode = DiscussionMode.FREE
    created_at: datetime = Field(default_factory=datetime.now)


class GroupChatCreate(BaseModel):
    """创建群聊的请求"""
    name: str
    discussion_mode: DiscussionMode = DiscussionMode.FREE


# ============ 消息模型 ============

class Message(BaseModel):
    """聊天消息"""
    id: str
    group_id: str
    role: MessageRole
    content: str
    sender_id: Optional[str] = None     # AI成员ID，用户消息为None
    sender_name: Optional[str] = None   # 发送者名称
    mode: Optional[DiscussionMode] = None # 消息所属的模式
    created_at: datetime = Field(default_factory=datetime.now)


class MessageCreate(BaseModel):
    """发送消息的请求"""
    content: str


# ============ 讨论请求模型 ============

class DiscussionRequest(BaseModel):
    """发起讨论的请求"""
    content: str                                        # 用户问题/话题
    user_name: str = "用户"                             # 用户昵称
    max_rounds: int = 3                                 # 最大讨论轮数
    mode: Optional[DiscussionMode] = None               # [可选] 本次讨论的模式，覆盖群组默认设置


class DiscussionResponse(BaseModel):
    """讨论响应"""
    messages: list[Message]             # 讨论产生的所有消息
    summary: Optional[str] = None       # 讨论总结


class SummarizeRequest(BaseModel):
    """总结请求"""
    instruction: Optional[str] = "请对以上讨论进行总结，得出最终结论。"


# ============ 模型能力定义 ============

class ModelCapability(BaseModel):
    """模型能力描述"""
    model_id: str
    name: str
    provider: str                       # 提供商：openai, anthropic, google, etc.
    supports_tools: bool = False        # 是否支持工具调用
    context_window: int = 8192          # 上下文窗口大小
    description: Optional[str] = None

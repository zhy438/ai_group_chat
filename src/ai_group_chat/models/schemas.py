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


class MessageType(str, Enum):
    """
    消息类型 - 用于上下文压缩时的分类
    
    不同类型的消息有不同的保留策略和权重
    """
    USER = "user"           # 用户消息：最重要，全部保留
    STATUS = "status"       # 关键状态：任务完成/失败等决策节点
    REASONING = "reasoning" # 推理过程：思考、方案比较等
    FAILURE = "failure"     # 失败记录：尝试失败的原因
    NORMAL = "normal"       # 普通消息：未分类的默认类型


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
    manager_model: str = "qwen-flash"
    manager_thinking: bool = False
    manager_temperature: float = 0.7
    discussion_mode: DiscussionMode = DiscussionMode.FREE
    compression_threshold: float = 0.8
    memory_enabled: bool = True
    archive_enabled: bool = True
    retrieve_enabled: bool = True
    scope_user_global: bool = True
    scope_group_local: bool = True
    scope_agent_local: bool = True
    memory_injection_ratio: float = 0.2
    memory_top_n: int = 5
    memory_min_confidence: float = 0.75
    memory_score_threshold: float = 0.35
    created_at: datetime = Field(default_factory=datetime.now)


class GroupChatCreate(BaseModel):
    """创建群聊的请求"""
    name: str


# ============ 消息模型 ============

class Message(BaseModel):
    """聊天消息"""
    id: str
    group_id: str
    role: MessageRole
    content: str
    sender_id: Optional[str] = None     # AI成员ID，用户消息为None
    user_id: Optional[str] = None       # 发起用户ID（用于长期记忆隔离）
    sender_name: Optional[str] = None   # 发送者名称
    mode: Optional[DiscussionMode] = None # 消息所属的模式
    created_at: datetime = Field(default_factory=datetime.now)
    
    # ====== 记忆管理相关字段 ======
    message_type: MessageType = MessageType.NORMAL  # 消息分类
    is_compressed: bool = False                      # 是否已被压缩
    original_content: Optional[str] = None           # 压缩前的原始内容
    value_score: Optional[float] = None              # 价值评分（用于排序）


class MessageCreate(BaseModel):
    """发送消息的请求"""
    content: str


# ============ 讨论请求模型 ============

class DiscussionRequest(BaseModel):
    """发起讨论的请求"""
    content: str                                        # 用户问题/话题
    user_name: str = "用户"                             # 用户昵称
    user_id: str = "default-user"                       # 用户稳定ID（长期记忆隔离键）
    max_rounds: int = 3                                 # 最大讨论轮数
    mode: Optional[DiscussionMode] = None               # [可选] 本次讨论的模式，覆盖群组默认设置


class DiscussionResponse(BaseModel):
    """讨论响应"""
    messages: list[Message]             # 讨论产生的所有消息
    summary: Optional[str] = None       # 讨论总结


class SummarizeRequest(BaseModel):
    """总结请求"""
    instruction: Optional[str] = "请对以上讨论进行总结，得出最终结论。"
    user_name: str = "用户"
    user_id: str = "default-user"


class MemorySettingsUpdate(BaseModel):
    """长期记忆配置更新"""
    memory_enabled: Optional[bool] = None
    archive_enabled: Optional[bool] = None
    retrieve_enabled: Optional[bool] = None
    scope_user_global: Optional[bool] = None
    scope_group_local: Optional[bool] = None
    scope_agent_local: Optional[bool] = None
    memory_injection_ratio: Optional[float] = Field(default=None, ge=0.05, le=0.5)
    memory_top_n: Optional[int] = Field(default=None, ge=1, le=10)
    memory_min_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    memory_score_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)


# ============ 模型能力定义 ============

class ModelCapability(BaseModel):
    """模型能力描述"""
    model_id: str
    name: str
    provider: str                       # 提供商：openai, anthropic, google, etc.
    supports_tools: bool = False        # 是否支持工具调用
    context_window: int = 8192          # 上下文窗口大小
    description: Optional[str] = None

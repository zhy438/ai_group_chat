"""API 路由定义"""

from pathlib import Path
import yaml
import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from ..models import (
    GroupChat, GroupChatCreate,
    AIMember, AIMemberCreate, AIMemberUpdate,
    Message,
    Message,
    DiscussionRequest, DiscussionResponse, SummarizeRequest,
    ModelCapability,
    ModelCapability,
)
from ..services import chat_service
from pydantic import BaseModel


# ============ 路由器 ============

router = APIRouter(prefix="/api/v1", tags=["AI群聊"])


# ============ 模型配置加载 ============

def load_models_config() -> list[ModelCapability]:
    """从 config/models.yaml 加载模型配置"""
    # routes.py -> api -> ai_group_chat -> src -> ai_group_chat (project root)
    project_root = Path(__file__).parent.parent.parent.parent
    config_path = project_root / "config" / "models.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"模型配置文件不存在: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    models = []
    for m in config.get("models", []):
        models.append(ModelCapability(
            model_id=m["model_id"],
            name=m["name"],
            provider=m.get("provider", "unknown"),
            supports_tools=m.get("supports_tools", False),
            context_window=m.get("context_window", 8192),
            description=m.get("description"),
        ))
    return models


# 缓存模型配置
_models_cache: list[ModelCapability] | None = None


def get_models() -> list[ModelCapability]:
    """获取模型列表（带缓存）"""
    global _models_cache
    if _models_cache is None:
        _models_cache = load_models_config()
    return _models_cache


# ============ 群聊管理 ============

@router.post("/groups", response_model=GroupChat)
async def create_group(data: GroupChatCreate):
    """创建新群聊"""
    return chat_service.create_group(data)


@router.get("/groups", response_model=list[GroupChat])
async def list_groups():
    """获取所有群聊列表"""
    return chat_service.list_groups()


@router.get("/groups/{group_id}", response_model=GroupChat)
async def get_group(group_id: str):
    """获取群聊详情"""
    group = chat_service.get_group(group_id)
    if not group:
        raise HTTPException(status_code=404, detail="群聊不存在")
    return group


@router.delete("/groups/{group_id}")
async def delete_group(group_id: str):
    """删除群聊"""
    if not chat_service.delete_group(group_id):
        raise HTTPException(status_code=404, detail="群聊不存在")
    return {"message": "删除成功"}


# ============ 成员管理 ============

@router.post("/groups/{group_id}/members", response_model=AIMember)
async def add_member(group_id: str, data: AIMemberCreate):
    """向群聊添加AI成员"""
    member = chat_service.add_member(group_id, data)
    if not member:
        raise HTTPException(status_code=404, detail="群聊不存在")
    return member


@router.delete("/groups/{group_id}/members/{member_id}")
async def remove_member(group_id: str, member_id: str):
    """从群聊移除AI成员"""
    if not chat_service.remove_member(group_id, member_id):
        raise HTTPException(status_code=404, detail="群聊或成员不存在")
    return {"message": "移除成功"}


@router.patch("/groups/{group_id}/members/{member_id}/task")
async def update_member_task(group_id: str, member_id: str, task: str):
    """更新成员的任务分配"""
    if not chat_service.update_member_task(group_id, member_id, task):
        raise HTTPException(status_code=404, detail="群聊或成员不存在")
    return {"message": "更新成功"}


@router.patch("/groups/{group_id}/members/{member_id}", response_model=AIMember)
async def update_member(group_id: str, member_id: str, data: AIMemberUpdate):
    """更新AI成员参数（thinking/temperature/description）"""
    member = chat_service.update_member(group_id, member_id, data)
    if not member:
        raise HTTPException(status_code=404, detail="群聊或成员不存在")
    return member


class ManagerConfigRequest(BaseModel):
    model_id: str
    thinking: bool = False
    temperature: float = 0.7


@router.put("/groups/{group_id}/manager")
async def set_manager_config(group_id: str, data: ManagerConfigRequest):
    """设置群聊管理员配置"""
    if not chat_service.set_manager_config(
        group_id, 
        data.model_id,
        data.thinking,
        data.temperature
    ):
        raise HTTPException(status_code=404, detail="群聊不存在")
    return {"message": "管理员配置设置成功"}


# ============ 讨论功能 ============

@router.post("/groups/{group_id}/discuss", response_model=DiscussionResponse)
async def start_discussion(group_id: str, request: DiscussionRequest):
    """
    启动群聊讨论（同步模式）
    
    - **content**: 用户问题或话题
    - **max_rounds**: 最大讨论轮数
    """
    try:
        return await chat_service.start_discussion(group_id, request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/groups/{group_id}/discuss/stream")
async def start_discussion_stream(group_id: str, request: DiscussionRequest):
    """
    启动群聊讨论（流式模式，实时推送每个成员的回复）
    
    返回 Server-Sent Events (SSE) 流，每条消息格式：
    data: {"type": "message", "sender_name": "小米", "content": "..."}
    
    最后会发送：
    data: {"type": "done"}
    """
    async def event_generator():
        try:
            async for message in chat_service.stream_discussion(group_id, request):
                event_data = json.dumps({
                    "type": "message",
                    "id": message.id,
                    "sender_name": message.sender_name,
                    "content": message.content,
                    "created_at": message.created_at.isoformat() if message.created_at else None,
                }, ensure_ascii=False)
                yield f"data: {event_data}\n\n"
            
            # 发送完成信号
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        except ValueError as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@router.post("/groups/{group_id}/summarize")
async def summarize_discussion(group_id: str, request: SummarizeRequest):
    """
    对当前讨论进行总结
    """
    async def event_generator():
        try:
            async for message in chat_service.summarize_discussion(group_id, request):
                event_data = json.dumps({
                    "type": "message",
                    "id": message.id,
                    "sender_name": message.sender_name,
                    "content": message.content,
                    "created_at": message.created_at.isoformat() if message.created_at else None,
                }, ensure_ascii=False)
                yield f"data: {event_data}\n\n"
            
            # 发送完成信号
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        except ValueError as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


# ============ 消息历史 ============

@router.get("/groups/{group_id}/messages", response_model=list[Message])
async def get_messages(group_id: str, limit: int = 50):
    """获取群聊消息历史"""
    return chat_service.get_messages(group_id, limit)


# ============ 模型能力 ============

@router.get("/models", response_model=list[ModelCapability])
async def list_available_models():
    """
    获取可用的AI模型列表
    
    从 config/models.yaml 配置文件加载
    """
    return get_models()


@router.post("/models/reload")
async def reload_models():
    """
    重新加载模型配置
    
    修改 config/models.yaml 后调用此接口刷新缓存
    """
    global _models_cache
    _models_cache = None
    models = get_models()
    return {"message": f"已重新加载 {len(models)} 个模型配置"}


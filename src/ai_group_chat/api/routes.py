"""API 路由定义"""

import asyncio
from pathlib import Path
import yaml
import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger

from ..models import (
    GroupChat, GroupChatCreate,
    AIMember, AIMemberCreate, AIMemberUpdate,
    Message,
    DiscussionRequest, DiscussionResponse, SummarizeRequest,
    MemorySettingsUpdate,
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


class CompressionConfig(BaseModel):
    threshold: float


@router.put("/groups/{group_id}/compression/threshold")
async def update_compression_threshold(group_id: str, config: CompressionConfig):
    """更新群聊压缩阈值"""
    if not await chat_service.update_compression_threshold(group_id, config.threshold):
        raise HTTPException(status_code=404, detail="群聊不存在")
    return {"message": "更新成功"}


@router.get("/groups/{group_id}/memory/settings")
async def get_memory_settings(group_id: str):
    """获取群聊长期记忆配置"""
    group = chat_service.get_group(group_id)
    if not group:
        raise HTTPException(status_code=404, detail="群聊不存在")
    return {
        "memory_enabled": group.memory_enabled,
        "archive_enabled": group.archive_enabled,
        "retrieve_enabled": group.retrieve_enabled,
        "scope_user_global": group.scope_user_global,
        "scope_group_local": group.scope_group_local,
        "scope_agent_local": group.scope_agent_local,
        "memory_injection_ratio": group.memory_injection_ratio,
        "memory_top_n": group.memory_top_n,
        "memory_min_confidence": group.memory_min_confidence,
        "memory_score_threshold": group.memory_score_threshold,
    }


@router.put("/groups/{group_id}/memory/settings")
async def update_memory_settings(group_id: str, data: MemorySettingsUpdate):
    """更新群聊长期记忆配置"""
    try:
        raw = data.model_dump()
    except AttributeError:
        raw = data.dict()
    payload = {k: v for k, v in raw.items() if v is not None}
    if not payload:
        raise HTTPException(status_code=400, detail="没有可更新的字段")
    if not chat_service.update_memory_settings(group_id, payload):
        raise HTTPException(status_code=404, detail="群聊不存在")
    return {"message": "更新成功"}


@router.get("/groups/{group_id}/memory/stats")
async def get_memory_stats(group_id: str):
    """获取群聊长期记忆统计"""
    try:
        return chat_service.get_memory_stats(group_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


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
                    "role": message.role,
                    "sender_name": message.sender_name,
                    "content": message.content,
                    "mode": message.mode,
                    "created_at": message.created_at.isoformat() if message.created_at else None,
                }, ensure_ascii=False)
                yield f"data: {event_data}\n\n"
                try:
                    stats_data = await chat_service.get_context_stats(group_id)
                    yield f"data: {json.dumps({'type': 'stats', 'stats': stats_data}, ensure_ascii=False)}\n\n"
                except Exception as stats_err:
                    logger.warning(f"推送实时上下文统计失败: {stats_err}")
            
            # 发送完成信号
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        except asyncio.CancelledError:
            chat_service.stop_discussion(group_id)
            logger.info(f"客户端已断开，已请求停止讨论: group_id={group_id}")
            raise
        except ValueError as e:
            logger.warning(f"流式讨论业务错误: group_id={group_id}, err={e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        except Exception as e:
            logger.exception(f"流式讨论服务异常: group_id={group_id}")
            yield f"data: {json.dumps({'type': 'error', 'message': f'讨论服务异常: {str(e)}'})}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@router.post("/groups/{group_id}/discuss/stop")
async def stop_discussion(group_id: str):
    """手动终止当前群聊正在进行的讨论。"""
    stopped = chat_service.stop_discussion(group_id)
    if stopped:
        return {"stopped": True, "message": "已请求终止讨论"}
    return {"stopped": False, "message": "当前没有运行中的讨论"}


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
                    "role": message.role,
                    "sender_name": message.sender_name,
                    "content": message.content,
                    "mode": message.mode,
                    "created_at": message.created_at.isoformat() if message.created_at else None,
                }, ensure_ascii=False)
                yield f"data: {event_data}\n\n"
                try:
                    stats_data = await chat_service.get_context_stats(group_id)
                    yield f"data: {json.dumps({'type': 'stats', 'stats': stats_data}, ensure_ascii=False)}\n\n"
                except Exception as stats_err:
                    logger.warning(f"推送实时上下文统计失败: {stats_err}")
            
            # 发送完成信号
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        except ValueError as e:
            logger.warning(f"流式总结业务错误: group_id={group_id}, err={e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        except Exception as e:
            logger.exception(f"流式总结服务异常: group_id={group_id}")
            yield f"data: {json.dumps({'type': 'error', 'message': f'总结服务异常: {str(e)}'})}\n\n"
    
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


# ============ 上下文管理（调试用） ============

@router.get("/groups/{group_id}/context/stats")
async def get_context_stats(group_id: str):
    """
    获取群聊的上下文统计信息
    
    用于调试和监控上下文压缩效果
    """
    if not chat_service.get_group(group_id):
        raise HTTPException(status_code=404, detail="群聊不存在")

    return await chat_service.get_context_stats(group_id)


@router.post("/groups/{group_id}/context/compress")
async def force_compress(group_id: str):
    """
    强制执行上下文压缩（忽略阈值）
    
    用于测试压缩效果
    """
    messages = chat_service.get_messages(group_id, limit=1000)
    
    if not messages:
        raise HTTPException(status_code=404, detail="没有消息可压缩")
    
    # 获取压缩前统计
    before_stats = chat_service.context_manager.get_stats(messages)
    
    # 强制执行压缩
    compressed = chat_service.context_manager.process(messages, force=True)
    
    # 获取压缩后统计
    after_stats = chat_service.context_manager.get_stats(compressed)
    
    return {
        "before": {
            "message_count": before_stats["message_count"],
            "tokens": before_stats["current_tokens"],
        },
        "after": {
            "message_count": after_stats["message_count"],
            "tokens": after_stats["current_tokens"],
        },
        "saved": {
            "messages": before_stats["message_count"] - after_stats["message_count"],
            "tokens": before_stats["current_tokens"] - after_stats["current_tokens"],
            "ratio": f"{(1 - after_stats['current_tokens'] / before_stats['current_tokens']) * 100:.1f}%"
        }
    }


@router.put("/groups/{group_id}/context/threshold")
async def set_compression_threshold(group_id: str, ratio: float = 0.8):
    """
    临时调整压缩触发阈值（用于测试）
    
    - ratio: 0.1 ~ 1.0，默认 0.8 表示 80%
    """
    if not 0.1 <= ratio <= 1.0:
        raise HTTPException(status_code=400, detail="ratio 必须在 0.1 到 1.0 之间")
    
    old_ratio = chat_service.context_manager.threshold_ratio
    chat_service.context_manager.threshold_ratio = ratio
    chat_service.context_manager.threshold_tokens = int(
        chat_service.context_manager.max_tokens * ratio
    )
    
    return {
        "message": f"阈值已从 {old_ratio*100:.0f}% 调整为 {ratio*100:.0f}%",
        "new_threshold_tokens": chat_service.context_manager.threshold_tokens
    }

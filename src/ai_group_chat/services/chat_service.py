"""聊天服务 - 业务逻辑层"""

import asyncio
import re
from collections import Counter
from pathlib import Path
import yaml
from loguru import logger
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.conditions import ExternalTermination

from ..models import (
    GroupChat, GroupChatCreate,
    AIMember, AIMemberCreate, AIMemberUpdate,
    Message, MessageRole, MessageType,
    DiscussionRequest, DiscussionResponse, SummarizeRequest,
    DiscussionMode,
)
from ..agents import AIGroupChat
from ..memory import ContextManager, LongTermMemoryService
from ..tools import GroupToolkitBundle, build_group_toolkits
from .chat_repository import ChatRepository

# 容错导入预设数据
try:
    from .presets import PRESET_GROUPS
except (ImportError, ModuleNotFoundError):
    PRESET_GROUPS = []


def _load_models_config() -> dict:
    """加载模型配置，返回 model_id -> context_window 映射"""
    project_root = Path(__file__).parent.parent.parent.parent
    config_path = project_root / "config" / "models.yaml"
    
    if not config_path.exists():
        return {}
    
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        return {
            m["model_id"]: m.get("context_window", 128000)
            for m in config.get("models", [])
        }
    except Exception as e:
        logger.warning(f"加载模型配置失败: {e}")
        return {}


# 全局模型配置缓存
_MODEL_CONTEXT_WINDOWS: dict = {}


class ChatService:
    """
    聊天服务
    
    业务逻辑层：负责编排业务流程、调用 Repository 进行数据存取。
    不包含任何 SQL 语句。
    """
    
    DEFAULT_CONTEXT_WINDOW = 128000  # 默认上下文窗口
    AUTO_INJECTION_MEMORY_TYPES = {"user_profile", "user_preference", "user_habit", "user_constraint"}
    AUTO_INJECTION_SCOPES = {"user_global"}
    
    def __init__(self):
        self.repo = ChatRepository()
        self.context_manager = ContextManager()  # 上下文管理器
        self.long_term_memory = LongTermMemoryService(self.repo)
        self._active_discussions: dict[str, ExternalTermination] = {}
        self._discussion_lock = asyncio.Lock()
        self._ensure_models_loaded()
        self._load_presets()
    
    def _ensure_models_loaded(self):
        """确保模型配置已加载"""
        global _MODEL_CONTEXT_WINDOWS
        if not _MODEL_CONTEXT_WINDOWS:
            _MODEL_CONTEXT_WINDOWS = _load_models_config()
            logger.info(f"📋 已加载 {len(_MODEL_CONTEXT_WINDOWS)} 个模型的上下文窗口配置")
    
    def get_min_context_window(self, group: GroupChat) -> int:
        """
        获取群聊中所有模型的最小上下文窗口
        
        Args:
            group: 群聊对象
            
        Returns:
            最小上下文窗口大小（tokens）
        """
        if not group.members:
            return self.DEFAULT_CONTEXT_WINDOW
        
        context_windows = []
        
        # 收集所有成员模型的上下文窗口
        for member in group.members:
            model_id = member.model_id
            window = _MODEL_CONTEXT_WINDOWS.get(model_id, self.DEFAULT_CONTEXT_WINDOW)
            context_windows.append(window)
        
        # 如果有 manager 模型，也要考虑
        if group.manager_model:
            manager_window = _MODEL_CONTEXT_WINDOWS.get(
                group.manager_model, self.DEFAULT_CONTEXT_WINDOW
            )
            context_windows.append(manager_window)
        
        min_window = min(context_windows) if context_windows else self.DEFAULT_CONTEXT_WINDOW
        logger.debug(f"📐 群聊 {group.name} 最小上下文窗口: {min_window} tokens")
        return min_window
    
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
        return self.repo.create_group(data.name)
    
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

    async def update_compression_threshold(self, group_id: str, threshold: float) -> bool:
        """更新群聊压缩阈值，并立即触发压缩检查"""
        if not self.repo.get_group(group_id):
            return False
            
        # 更新数据库配置
        updated = self.repo.update_group_compression_threshold(group_id, threshold)
        if not updated:
            return False
            
        # 立即触发一次压缩逻辑以应用新阈值
        try:
            await self._get_history_as_autogen_messages(group_id, limit=0)
        except Exception as e:
            logger.error(f"Error triggering immediate compression: {e}")
            # Do not fail request, user config is updated
            
        return True

    def update_memory_settings(self, group_id: str, settings: dict) -> bool:
        """更新群聊长期记忆配置"""
        if not self.repo.get_group(group_id):
            return False
        return self.repo.update_group_memory_settings(group_id, settings)

    def get_memory_stats(self, group_id: str) -> dict:
        """获取长期记忆统计"""
        if not self.repo.get_group(group_id):
            raise ValueError("群聊不存在")
        return self.long_term_memory.get_group_stats(group_id)

    def _build_toolkits(self, group: GroupChat, user_id: str) -> GroupToolkitBundle:
        """构建群聊工具集：成员共享工具 + 系统Agent专属工具。"""
        try:
            return build_group_toolkits(
                group=group,
                user_id=user_id,
                memory_service=self.long_term_memory,
                max_context_tokens=self.get_min_context_window(group),
            )
        except Exception as e:
            logger.warning(f"构建工具集失败，降级为无工具模式: {e}")
            return GroupToolkitBundle(member_tools=[], manager_tools=[])

    @staticmethod
    def _build_system_termination_notice(reason: str | None = None) -> str:
        """构造系统Agent提前终止提示。"""
        base = "系统已判断当前话题已完成，已主动终止本轮讨论。"
        cleaned = (reason or "").strip()
        cleaned = re.sub(r"^已确认提前终止讨论[:：]?", "", cleaned).strip()
        if cleaned:
            return f"{base}（终止原因：{cleaned}）"
        return base

    @staticmethod
    def _build_manual_termination_notice() -> str:
        """构造手动终止提示。"""
        return "已手动终止本轮讨论。"

    async def _register_active_discussion(self, group_id: str, termination: ExternalTermination) -> None:
        """注册当前群聊的活跃讨论；若已有运行中的讨论，先请求其停止。"""
        async with self._discussion_lock:
            previous = self._active_discussions.get(group_id)
            if previous and previous is not termination:
                previous.set()
            self._active_discussions[group_id] = termination

    async def _clear_active_discussion(self, group_id: str, termination: ExternalTermination) -> None:
        """清理活跃讨论注册（仅清理当前实例）。"""
        async with self._discussion_lock:
            current = self._active_discussions.get(group_id)
            if current is termination:
                self._active_discussions.pop(group_id, None)

    def stop_discussion(self, group_id: str) -> bool:
        """手动终止群聊中正在运行的讨论。"""
        termination = self._active_discussions.get(group_id)
        if not termination:
            return False
        termination.set()
        return True

    async def _build_auto_injection_memory_block(self, group: GroupChat, user_id: str, query: str) -> str:
        """自动注入仅包含用户偏好，不注入群聊结论/agent画像。"""
        return await self.long_term_memory.build_injection_context(
            group=group,
            user_id=user_id,
            query=query,
            max_context_tokens=self.get_min_context_window(group),
            memory_types=self.AUTO_INJECTION_MEMORY_TYPES,
            scopes=self.AUTO_INJECTION_SCOPES,
        )

    @staticmethod
    def _copy_model(obj, updates: dict):
        """兼容 Pydantic v1/v2 的模型拷贝"""
        try:
            return obj.model_copy(update=updates)
        except AttributeError:
            return obj.copy(update=updates)

    @staticmethod
    def _extract_invalid_model_id(err_msg: str) -> str | None:
        """从错误信息中提取无权限/无效模型ID"""
        if not err_msg:
            return None
        if "Incorrect model ID" not in err_msg and "do not have permission to use this model" not in err_msg:
            return None

        patterns = [
            r"use this model\s+([a-zA-Z0-9._-]+)",
            r"model\s+([a-zA-Z0-9._-]+)\s+\(tid:",
            r"'model':\s*'([a-zA-Z0-9._-]+)'",
        ]
        for pattern in patterns:
            match = re.search(pattern, err_msg)
            if match:
                return match.group(1)
        return None

    def _pick_fallback_model(self, bad_model_id: str, group: GroupChat) -> str | None:
        """选择一个备用模型用于自动降级重试"""
        candidates: list[str] = []
        candidates.extend(list(_MODEL_CONTEXT_WINDOWS.keys()))
        candidates.extend([m.model_id for m in group.members])
        if group.manager_model:
            candidates.append(group.manager_model)

        for model_id in candidates:
            if model_id and model_id != bad_model_id:
                return model_id
        return None

    def _replace_bad_model(self, group: GroupChat, bad_model_id: str, fallback_model_id: str) -> GroupChat | None:
        """构造替换了无效模型ID的临时群聊对象（仅本次请求使用）"""
        replaced = False
        patched_members = []
        for member in group.members:
            if member.model_id == bad_model_id:
                patched_members.append(self._copy_model(member, {"model_id": fallback_model_id}))
                replaced = True
            else:
                patched_members.append(member)

        manager_model = group.manager_model
        if manager_model == bad_model_id:
            manager_model = fallback_model_id
            replaced = True

        if not replaced:
            return None

        return self._copy_model(group, {"members": patched_members, "manager_model": manager_model})

    def _try_build_fallback_group(self, group: GroupChat, err_msg: str) -> tuple[GroupChat | None, str | None]:
        """命中模型权限错误时，尝试构造降级重试群聊"""
        bad_model_id = self._extract_invalid_model_id(err_msg)
        if not bad_model_id:
            return None, None

        fallback_model_id = self._pick_fallback_model(bad_model_id, group)
        if not fallback_model_id:
            return None, f"模型 `{bad_model_id}` 不可用，且没有可用备用模型。"

        patched_group = self._replace_bad_model(group, bad_model_id, fallback_model_id)
        if not patched_group:
            return None, f"模型 `{bad_model_id}` 不在当前群聊配置中，无法自动替换。"

        tip = f"检测到模型 `{bad_model_id}` 无权限，已自动回退到 `{fallback_model_id}` 重试本轮请求。"
        return patched_group, tip
    
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
        
        mode = request.mode if request.mode else DiscussionMode.FREE

        # 保存用户消息
        self.repo.save_message(
            group_id=group_id,
            role=MessageRole.USER,
            content=request.content,
            sender_name=request.user_name,
            mode=mode,
            sender_id=None,
            user_id=request.user_id,
        )
        runtime_group = group
        for attempt in range(2):
            member_id_map = {m.name: m.id for m in runtime_group.members}
            try:
                if mode == DiscussionMode.QA:
                    # QA 模式不需要很长的上下文
                    history_msgs = []
                else:
                    # FREE 模式需要上下文
                    history_msgs = await self._get_history_as_autogen_messages(group_id, limit=50, exclude_last=True)

                # 条件注入长期记忆（仅注入一次，不会每轮灌入）
                memory_block = await self._build_auto_injection_memory_block(
                    group=runtime_group,
                    user_id=request.user_id,
                    query=request.content,
                )
                if memory_block:
                    history_msgs = [TextMessage(content=memory_block, source="system")] + history_msgs
                toolkits = self._build_toolkits(runtime_group, request.user_id)

                ai_group_chat = AIGroupChat(
                    members=runtime_group.members,
                    user_name=request.user_name,
                    max_rounds=request.max_rounds,
                    mode=mode,
                    manager_model=runtime_group.manager_model,
                    manager_thinking=runtime_group.manager_thinking,
                    manager_temperature=runtime_group.manager_temperature,
                    history=history_msgs,
                    shared_tools=toolkits.member_tools,
                    manager_tools=toolkits.manager_tools,
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
                        group_id=group_id,
                        role=MessageRole.ASSISTANT,
                        content=msg_data["content"],
                        sender_name=msg_data["sender"],
                        mode=mode,
                        sender_id=member_id_map.get(msg_data["sender"]),
                        user_id=request.user_id,
                    )
                    result_messages.append(message)

                if ai_group_chat.was_terminated_by_system():
                    notice = self.repo.save_message(
                        group_id=group_id,
                        role=MessageRole.SYSTEM,
                        content=self._build_system_termination_notice(
                            ai_group_chat.last_system_termination_reason
                        ),
                        sender_name="系统",
                        mode=mode,
                        sender_id=None,
                        user_id=request.user_id,
                    )
                    result_messages.append(notice)

                self._schedule_memory_archive(group=group, user_id=request.user_id, reason="discussion_sync")
                return DiscussionResponse(messages=result_messages, summary=None)
            except Exception as e:
                err_msg = str(e)
                if attempt == 0:
                    fallback_group, tip = self._try_build_fallback_group(runtime_group, err_msg)
                    if fallback_group:
                        logger.warning(tip)
                        runtime_group = fallback_group
                        continue
                logger.error(f"讨论执行失败: {err_msg}")
                if "RateLimitError" in err_msg or "429" in err_msg:
                    raise ValueError("模型调用触发限流（429）：免费额度已用尽，请切换付费模型或稍后重试。")
                if self._extract_invalid_model_id(err_msg):
                    raise ValueError(
                        f"模型不可用或无权限：{err_msg}\n请在成员/管理员设置里切换到可用模型后重试。"
                    )
                raise ValueError(f"讨论执行失败: {err_msg}")

        raise ValueError("讨论执行失败：自动回退后仍未成功。")
    
    async def stream_discussion(self, group_id: str, request: DiscussionRequest):
        """流式启动群聊讨论"""
        group = self.get_group(group_id)
        if not group or not group.members:
            raise ValueError("群聊不存在或没有成员")
        
        mode = request.mode if request.mode else DiscussionMode.FREE

        # 保存用户消息
        self.repo.save_message(
            group_id=group_id,
            role=MessageRole.USER,
            content=request.content,
            sender_name=request.user_name,
            mode=mode,
            sender_id=None,
            user_id=request.user_id,
        )
        external_termination = ExternalTermination()
        await self._register_active_discussion(group_id, external_termination)
        runtime_group = group
        try:
            for attempt in range(2):
                member_id_map = {m.name: m.id for m in runtime_group.members}
                emitted_count = 0
                try:
                    # 获取历史消息作为上下文
                    # 注意: exclude_last=True 是为了避免重复包含刚刚保存的用户消息，
                    # 因为在 AutoGen 中，用户的提问通常作为 initiate_chat 的 message 参数传入
                    history_msgs = await self._get_history_as_autogen_messages(group_id, limit=50, exclude_last=True)
                    memory_block = await self._build_auto_injection_memory_block(
                        group=runtime_group,
                        user_id=request.user_id,
                        query=request.content,
                    )
                    if memory_block:
                        history_msgs = [TextMessage(content=memory_block, source="system")] + history_msgs
                    toolkits = self._build_toolkits(runtime_group, request.user_id)

                    ai_group_chat = AIGroupChat(
                        members=runtime_group.members,
                        user_name=request.user_name,
                        max_rounds=request.max_rounds,
                        mode=mode,
                        manager_model=runtime_group.manager_model,
                        manager_thinking=runtime_group.manager_thinking,
                        manager_temperature=runtime_group.manager_temperature,
                        history=history_msgs,
                        shared_tools=toolkits.member_tools,
                        manager_tools=toolkits.manager_tools,
                        external_termination=external_termination,
                    )

                    if mode == DiscussionMode.QA:
                        generator = ai_group_chat.stream_qa_discussion(request.content)
                    else:
                        generator = ai_group_chat.stream_discuss(request.content, request.max_rounds)

                    async for msg_data in generator:
                        message = self.repo.save_message(
                            group_id=group_id,
                            role=MessageRole.ASSISTANT,
                            content=msg_data["content"],
                            sender_name=msg_data["sender"],
                            mode=mode,
                            sender_id=member_id_map.get(msg_data["sender"]),
                            user_id=request.user_id,
                        )
                        emitted_count += 1
                        yield message

                    if ai_group_chat.was_terminated_by_system():
                        notice = self.repo.save_message(
                            group_id=group_id,
                            role=MessageRole.SYSTEM,
                            content=self._build_system_termination_notice(
                                ai_group_chat.last_system_termination_reason
                            ),
                            sender_name="系统",
                            mode=mode,
                            sender_id=None,
                            user_id=request.user_id,
                        )
                        emitted_count += 1
                        yield notice
                    elif ai_group_chat.was_terminated_externally():
                        notice = self.repo.save_message(
                            group_id=group_id,
                            role=MessageRole.SYSTEM,
                            content=self._build_manual_termination_notice(),
                            sender_name="系统",
                            mode=mode,
                            sender_id=None,
                            user_id=request.user_id,
                        )
                        emitted_count += 1
                        yield notice
                    self._schedule_memory_archive(group=group, user_id=request.user_id, reason="discussion_stream")
                    return
                except Exception as e:
                    err_msg = str(e)
                    bad_model_id = self._extract_invalid_model_id(err_msg)
                    if attempt == 0 and emitted_count == 0:
                        fallback_group, tip = self._try_build_fallback_group(runtime_group, err_msg)
                        if fallback_group:
                            logger.warning(tip)
                            runtime_group = fallback_group
                            continue
                    logger.error(f"讨论流式执行失败: {err_msg}")
                    if "RateLimitError" in err_msg or "429" in err_msg:
                        raise ValueError("模型调用触发限流（429）：免费额度已用尽，请切换付费模型或稍后重试。")
                    if bad_model_id:
                        # 流式模式可能已产生部分回复；此时不抛异常中断前端，而是给出系统提示并结束本轮
                        if emitted_count > 0:
                            tip = (
                                f"成员模型 `{bad_model_id}` 无权限，已提前结束本轮讨论。"
                                "请在成员/管理员设置里切换到可用模型后重试。"
                            )
                            logger.warning(tip)
                            notice = self.repo.save_message(
                                group_id=group_id,
                                role=MessageRole.SYSTEM,
                                content=tip,
                                sender_name="系统",
                                mode=mode,
                                sender_id=None,
                                user_id=request.user_id,
                            )
                            yield notice
                            self._schedule_memory_archive(group=group, user_id=request.user_id, reason="discussion_stream")
                            return
                        raise ValueError(
                            f"模型不可用或无权限：{err_msg}\n请在成员/管理员设置里切换到可用模型后重试。"
                        )
                    raise ValueError(f"讨论执行失败: {err_msg}")
        finally:
            await self._clear_active_discussion(group_id, external_termination)
            
    async def summarize_discussion(self, group_id: str, request: SummarizeRequest):
        """对群聊进行总结"""
        group = self.get_group(group_id)
        if not group: return
        runtime_group = group
        for attempt in range(2):
            member_id_map = {m.name: m.id for m in runtime_group.members}
            try:
                history_msgs = await self._get_history_as_autogen_messages(group_id, limit=100)
                memory_block = await self._build_auto_injection_memory_block(
                    group=runtime_group,
                    user_id=request.user_id,
                    query=request.instruction or "总结并提炼群聊结论",
                )
                if memory_block:
                    history_msgs = [TextMessage(content=memory_block, source="system")] + history_msgs
                toolkits = self._build_toolkits(runtime_group, request.user_id)

                ai_group_chat = AIGroupChat(
                    members=runtime_group.members,
                    user_name=request.user_name or "User",
                    mode=DiscussionMode.FREE,
                    manager_model=runtime_group.manager_model,
                    manager_thinking=runtime_group.manager_thinking,
                    manager_temperature=runtime_group.manager_temperature,
                    history=history_msgs,
                    shared_tools=toolkits.member_tools,
                )

                result = await ai_group_chat.summarize(request.instruction)
                message = self.repo.save_message(
                    group_id=group_id,
                    role=MessageRole.ASSISTANT,
                    content=result["content"],
                    sender_name=result["sender"],
                    mode=DiscussionMode.FREE,
                    sender_id=member_id_map.get(result["sender"]),
                    user_id=request.user_id,
                )
                self._schedule_memory_archive(group=group, user_id=request.user_id, reason="summarize")
                yield message
                return
            except Exception as e:
                err_msg = str(e)
                if attempt == 0:
                    fallback_group, tip = self._try_build_fallback_group(runtime_group, err_msg)
                    if fallback_group:
                        logger.warning(tip)
                        runtime_group = fallback_group
                        continue
                logger.error(f"总结执行失败: {err_msg}")
                if "RateLimitError" in err_msg or "429" in err_msg:
                    raise ValueError("总结触发限流（429）：免费额度已用尽，请切换付费模型或稍后重试。")
                if self._extract_invalid_model_id(err_msg):
                    raise ValueError(
                        f"模型不可用或无权限：{err_msg}\n请在成员/管理员设置里切换到可用模型后重试。"
                    )
                raise ValueError(f"总结执行失败: {err_msg}")

    def _schedule_memory_archive(self, group: GroupChat, user_id: str, reason: str) -> None:
        """后台异步归档长期记忆，不阻塞主链路"""
        async def runner():
            try:
                await self.long_term_memory.archive_incremental(
                    group=group,
                    user_id=user_id,
                    force=True,
                    reason=reason,
                )
            except Exception as e:
                logger.error(f"后台长期记忆归档失败: {e}")
        asyncio.create_task(runner())
    
    async def _get_history_as_autogen_messages(self, group_id: str, limit: int = 50, exclude_last: bool = False) -> list[TextMessage]:
        """
        获取群聊历史并转换为 AutoGen 格式（异步版本）
        
        包含上下文压缩逻辑：当 Token 超过阈值时自动压缩
        上下文窗口大小动态设置为群聊中模型的最小值
        压缩过程使用异步 LLM 调用，不阻塞主线程
        """
        # 获取群聊信息以确定最小上下文窗口
        group = self.get_group(group_id)
        if group:
            min_context_window = self.get_min_context_window(group)
            self.context_manager.set_max_tokens(min_context_window)
            
            # 动态应用压缩阈值配置
            self.context_manager.threshold_ratio = group.compression_threshold
            self.context_manager.threshold_tokens = int(self.context_manager.max_tokens * self.context_manager.threshold_ratio)
        
        # 1. 尝试加载最新的上下文快照
        snapshot = self.repo.get_latest_snapshot(group_id)
        
        final_messages = []
        messages_to_process = []
        last_processed_msg_id = None
        snapshot_loaded = False
        
        if snapshot:
            try:
                # 反序列化快照内容
                import json
                snapshot_data = json.loads(snapshot['context_content'])
                
                # 尝试使用 Pydantic 的解析方法，兼容 v1 和 v2
                try:
                    final_messages = [Message.model_validate(item) for item in snapshot_data]
                except AttributeError:
                    # Pydantic v1 fallback
                    final_messages = [Message.parse_obj(item) for item in snapshot_data]
                except Exception:
                    # Fallback manually
                    final_messages = [Message(**item) for item in snapshot_data]
                
                last_processed_msg_id = snapshot['last_message_id']
                logger.info(f"📸 加载上下文快照成功 (ID: {snapshot['id']}), Token: {snapshot['token_count']}")
                
                # 加载增量消息
                messages_to_process = self.repo.get_messages_after(group_id, last_processed_msg_id)
                logger.info(f"📥 增量加载了 {len(messages_to_process)} 条新消息")
                snapshot_loaded = True
                
            except Exception as e:
                logger.error(f"❌ 加载快照失败，回退到全量加载: {e}")
                final_messages = []
                snapshot_loaded = False
        
        if not snapshot_loaded:
            # 全量加载
            messages_to_process = self.repo.get_messages(group_id, limit=0)
            logger.info(f"📚 全量加载历史消息，总数: {len(messages_to_process)}")
        
        if exclude_last and messages_to_process:
            messages_to_process = messages_to_process[:-1]
        
        # 分批累加与压缩策略
        current_batch = []
        save_snapshot = False
        
        # 如果快照本身已经超限（例如窗口设置变小了），也需要压缩
        if self.context_manager.should_compress(final_messages):
             logger.info("⚠️ 快照内容超过当前阈值，重新压缩...")
             final_messages = await self.context_manager.process_async(final_messages)
             save_snapshot = True
        
        for msg in messages_to_process:
            current_batch.append(msg)
            last_processed_msg_id = msg.id
            
            check_context = final_messages + current_batch
            
            if self.context_manager.should_compress(check_context):
                logger.info(f"⚡️ 触发分批压缩循环，当前总数: {len(check_context)}")
                final_messages = await self.context_manager.process_async(check_context)
                current_batch = []
                save_snapshot = True
        
        # 处理剩余的 batch
        if current_batch:
            final_messages = final_messages + current_batch
            if self.context_manager.should_compress(final_messages):
                logger.info(f"⚡️ 触发最终压缩")
                final_messages = await self.context_manager.process_async(final_messages)
                save_snapshot = True
            elif snapshot_loaded and current_batch:
                # 即使没有触发压缩，但我们有新的增量消息，也可以选择更新快照
                # 为了下次更快的加载，这通常是好的
                save_snapshot = True

        # 保存新的快照
        if save_snapshot and last_processed_msg_id and final_messages:
             try:
                 token_count = self.context_manager.count_messages_tokens(final_messages)
                 self.repo.save_snapshot(group_id, last_processed_msg_id, final_messages, token_count)
                 logger.info(f"💾 上下文快照已更新 (Msg: {last_processed_msg_id})")
             except Exception as e:
                 logger.error(f"❌ 保存快照失败: {e}")
        
        # 转换为 AutoGen 格式
        autogen_msgs = []
        for msg in final_messages:
            source = "user" if msg.role == MessageRole.USER else _sanitize_name(msg.sender_name)
            autogen_msgs.append(TextMessage(content=msg.content, source=source))
        return autogen_msgs

    async def get_context_stats(self, group_id: str) -> dict:
        """获取群聊上下文统计（用于 API 拉取与 SSE 实时推送）"""
        group = self.get_group(group_id)
        if not group:
            raise ValueError("群聊不存在")

        min_context_window = self.get_min_context_window(group)
        self.context_manager.set_max_tokens(min_context_window)

        autogen_msgs = await self._get_history_as_autogen_messages(group_id, limit=0)
        current_tokens = sum(self.context_manager.count_tokens(m.content) for m in autogen_msgs)

        raw_messages = self.get_messages(group_id, limit=1000)
        type_counts = Counter(msg.message_type.value for msg in raw_messages)
        member_windows = {
            m.name: _MODEL_CONTEXT_WINDOWS.get(m.model_id, self.DEFAULT_CONTEXT_WINDOW)
            for m in group.members
        } if group.members else {}
        memory_stats = self.long_term_memory.get_group_stats(group_id)

        return {
            "current_tokens": current_tokens,
            "message_count": len(autogen_msgs),
            "type_distribution": dict(type_counts),
            "compression_config": {
                "max_tokens": self.context_manager.max_tokens,
                "threshold_ratio": group.compression_threshold,
                "threshold_tokens": int(self.context_manager.max_tokens * group.compression_threshold),
            },
            "dynamic_context_window": {
                "min_context_window": min_context_window,
                "member_windows": member_windows,
            },
            "memory_stats": memory_stats,
        }
    
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

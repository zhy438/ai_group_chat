"""
聊天数据仓库 (Repository)

组合使用 DAO 层，提供业务级别的数据访问接口
"""

from typing import Optional, List

from ..models import (
    GroupChat, AIMember, Message,
    AIMemberCreate, AIMemberUpdate, MessageRole, MessageType
)
from ..dao import group_dao, member_dao, message_dao, context_snapshot_dao


class ChatRepository:
    """
    聊天数据仓库 (Repository)
    
    作为 DAO 层的门面，组合多个 DAO 提供完整的业务数据操作。
    负责将 DAO 返回的原始数据转换为领域对象。
    """
    
    def __init__(self):
        self.group_dao = group_dao
        self.member_dao = member_dao
        self.message_dao = message_dao
        self.context_snapshot_dao = context_snapshot_dao

    # ============ Group Operations ============

    def get_group_by_name(self, name: str) -> Optional[GroupChat]:
        row = self.group_dao.get_by_name(name)
        return self._build_group(row) if row else None

    def get_group(self, group_id: str) -> Optional[GroupChat]:
        row = self.group_dao.get_by_id(group_id)
        return self._build_group(row) if row else None

    def list_groups(self) -> List[GroupChat]:
        rows = self.group_dao.list_all()
        return [self._build_group(row) for row in rows]

    def create_group(self, name: str, discussion_mode: str = 'free',
                     manager_model: str = "qwen-flash") -> GroupChat:
        group_id = self.group_dao.create(name, discussion_mode, manager_model)
        return self.get_group(group_id)

    def delete_group(self, group_id: str) -> bool:
        return self.group_dao.delete(group_id)

    def update_manager_config(self, group_id: str, model_id: str,
                              thinking: Optional[bool] = None,
                              temperature: Optional[float] = None) -> bool:
        return self.group_dao.update_manager_config(group_id, model_id, thinking, temperature)

    def _build_group(self, row: dict) -> GroupChat:
        """构建完整的 GroupChat 对象（包含成员）"""
        members = self._get_members_for_group(row['id'])
        return self.group_dao._row_to_group(row, members)

    def _get_members_for_group(self, group_id: str) -> List[AIMember]:
        """获取群聊的所有成员"""
        rows = self.member_dao.get_by_group(group_id)
        return [self.member_dao._row_to_member(row) for row in rows]

    # ============ Member Operations ============

    def add_member(self, group_id: str, data: AIMemberCreate) -> AIMember:
        member_id = self.member_dao.add(group_id, data)
        row = self.member_dao.get_by_id(member_id)
        return self.member_dao._row_to_member(row)

    def add_raw_member(self, group_id: str, name: str, model_id: str,
                       description: str, thinking: bool, temperature: float) -> None:
        """用于预设数据的底层添加"""
        self.member_dao.add_raw(group_id, name, model_id, description, thinking, temperature)

    def update_member(self, group_id: str, member_id: str, data: AIMemberUpdate) -> Optional[AIMember]:
        if self.member_dao.update(group_id, member_id, data):
            row = self.member_dao.get_by_id(member_id)
            return self.member_dao._row_to_member(row) if row else None
        return None

    def remove_member(self, group_id: str, member_id: str) -> bool:
        return self.member_dao.delete(group_id, member_id)

    def update_member_persona(self, group_id: str, member_id: str, persona: str) -> bool:
        return self.member_dao.update_persona(group_id, member_id, persona)

    # ============ Message Operations ============

    def save_message(self, group_id: str, role: MessageRole, content: str,
                     sender_name: str, mode: str,
                     sender_id: str = None,
                     user_id: str = "default-user",
                     message_type: MessageType = MessageType.NORMAL) -> Message:
        msg_id = self.message_dao.save(
            group_id=group_id,
            role=role,
            content=content,
            sender_name=sender_name,
            mode=mode,
            sender_id=sender_id,
            user_id=user_id,
            message_type=message_type,
        )
        row = self.message_dao.get_by_id(msg_id)
        return self.message_dao._row_to_message(row)

    def update_message_compression(self, message_id: str,
                                   is_compressed: bool,
                                   compressed_content: str,
                                   original_content: str) -> bool:
        return self.message_dao.update_compression(
            message_id, is_compressed, compressed_content, original_content
        )

    def get_messages(self, group_id: str, limit: int) -> List[Message]:
        rows = self.message_dao.get_by_group(group_id, limit)
        return [self.message_dao._row_to_message(row) for row in rows]

    def get_messages_after(self, group_id: str, last_message_id: str) -> List[Message]:
        """增量加载消息"""
        rows = self.message_dao.get_messages_after(group_id, last_message_id)
        return [self.message_dao._row_to_message(row) for row in rows]

    def get_messages_since_cursor(self, group_id: str, last_created_at, last_message_id: str, limit: int = 200) -> List[Message]:
        """按游标增量读取消息"""
        rows = self.message_dao.get_messages_since_cursor(group_id, last_created_at, last_message_id, limit)
        return [self.message_dao._row_to_message(row) for row in rows]

    # ============ Context Snapshots ============
    
    def get_latest_snapshot(self, group_id: str) -> Optional[dict]:
        return self.context_snapshot_dao.get_latest(group_id)

    def save_snapshot(self, group_id: str, last_message_id: str, context: List[Message], token_count: int):
        self.context_snapshot_dao.save(group_id, last_message_id, context, token_count)

    def update_group_compression_threshold(self, group_id: str, threshold: float) -> bool:
        return self.group_dao.update_compression_threshold(group_id, threshold)

    def update_group_memory_settings(self, group_id: str, settings: dict) -> bool:
        return self.group_dao.update_memory_settings(group_id, settings)

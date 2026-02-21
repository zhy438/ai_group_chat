"""
消息 DAO

负责消息相关的数据库操作
"""

from typing import Optional, List
from uuid import uuid4

from ..models import Message, MessageRole, MessageType
from .base import BaseDAO


class MessageDAO(BaseDAO):
    """
    消息数据访问对象
    
    负责 messages 表的 CRUD 操作
    """
    
    def _row_to_message(self, row: dict) -> Message:
        """将数据库行转换为 Message 对象"""
        created_at = self.parse_datetime(row['created_at'])
        
        # 处理 message_type 字段
        msg_type_str = row.get('message_type', 'normal')
        try:
            msg_type = MessageType(msg_type_str) if msg_type_str else MessageType.NORMAL
        except ValueError:
            msg_type = MessageType.NORMAL
        
        return Message(
            id=row['id'],
            group_id=row['group_id'],
            role=row['role'],
            content=row['content'],
            sender_id=row.get('sender_id'),
            user_id=row.get('user_id', 'default-user'),
            sender_name=row['sender_name'],
            mode=row['mode'],
            created_at=created_at,
            message_type=msg_type,
            is_compressed=bool(row.get('is_compressed', False)),
            original_content=row.get('original_content'),
            value_score=row.get('value_score'),
        )
    
    def get_by_id(self, message_id: str) -> Optional[dict]:
        """根据 ID 获取消息原始数据"""
        return self.db.fetch_one("SELECT * FROM messages WHERE id = ?", (message_id,))
    
    def get_by_group(self, group_id: str, limit: int = 50) -> List[dict]:
        """
        获取群聊的消息列表（按时间升序）
        
        如果 limit > 0: 获取最新的 N 条
        如果 limit <= 0: 获取所有消息
        """
        if limit > 0:
            sql = """
                SELECT * FROM (
                    SELECT * FROM messages 
                    WHERE group_id = ? 
                    ORDER BY created_at DESC 
                    LIMIT ?
                ) AS recent_msgs ORDER BY created_at ASC
            """
            return self.db.fetch_all(sql, (group_id, limit))
        else:
            sql = """
                SELECT * FROM messages 
                WHERE group_id = ? 
                ORDER BY created_at ASC
            """
            return self.db.fetch_all(sql, (group_id,))
            
    def get_messages_after(self, group_id: str, last_message_id: str) -> List[dict]:
        """
        获取指定消息之后的所有消息（增量加载）
        
        用于配合上下文快照，只加载上次快照之后的新消息。
        """
        # 1. 获取 last_message 的 created_at
        last_msg = self.get_by_id(last_message_id)
        if not last_msg:
            # 如果找不到参照消息（可能被物理删除了），则回退到全量加载
            return self.get_by_group(group_id, limit=0)
            
        last_created_at = last_msg['created_at']
        
        # 2. 查询之后的 msg
        sql = """
            SELECT * FROM messages 
            WHERE group_id = ? AND created_at > ?
            ORDER BY created_at ASC
        """
        return self.db.fetch_all(sql, (group_id, last_created_at))
    
    def save(self, group_id: str, role: MessageRole, content: str,
             sender_name: str, mode: str,
             sender_id: str = None,
             user_id: str = "default-user",
             message_type: MessageType = MessageType.NORMAL) -> str:
        """
        保存消息
        
        Returns:
            新消息的 ID
        """
        msg_id = str(uuid4())
        self.db.execute("""
            INSERT INTO messages (id, group_id, role, content, sender_id, user_id, sender_name, mode, message_type)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (msg_id, group_id, role, content, sender_id, user_id, sender_name, mode, message_type.value))
        return msg_id

    def get_messages_since_cursor(
        self,
        group_id: str,
        last_created_at,
        last_message_id: str = "",
        limit: int = 200,
    ) -> List[dict]:
        """按(created_at, id)游标增量获取消息"""
        if not last_created_at:
            sql = """
                SELECT * FROM messages
                WHERE group_id = ?
                ORDER BY created_at ASC, id ASC
                LIMIT ?
            """
            return self.db.fetch_all(sql, (group_id, limit))

        sql = """
            SELECT * FROM messages
            WHERE group_id = ?
              AND (
                    created_at > ?
                    OR (created_at = ? AND id > ?)
                  )
            ORDER BY created_at ASC, id ASC
            LIMIT ?
        """
        return self.db.fetch_all(sql, (group_id, last_created_at, last_created_at, last_message_id or "", limit))
    
    def update_compression(self, message_id: str,
                          is_compressed: bool,
                          compressed_content: str,
                          original_content: str = None) -> bool:
        """更新消息的压缩状态"""
        cursor = self.db.execute("""
            UPDATE messages 
            SET is_compressed = ?, content = ?, original_content = ?
            WHERE id = ?
        """, (is_compressed, compressed_content, original_content, message_id))
        return cursor.rowcount > 0
    
    def update_type(self, message_id: str, message_type: MessageType) -> bool:
        """更新消息类型"""
        cursor = self.db.execute(
            "UPDATE messages SET message_type = ? WHERE id = ?",
            (message_type.value, message_id)
        )
        return cursor.rowcount > 0
    
    def update_score(self, message_id: str, value_score: float) -> bool:
        """更新消息价值评分"""
        cursor = self.db.execute(
            "UPDATE messages SET value_score = ? WHERE id = ?",
            (value_score, message_id)
        )
        return cursor.rowcount > 0
    
    def delete_by_group(self, group_id: str) -> int:
        """删除群聊的所有消息"""
        cursor = self.db.execute(
            "DELETE FROM messages WHERE group_id = ?",
            (group_id,)
        )
        return cursor.rowcount


# 全局实例
message_dao = MessageDAO()

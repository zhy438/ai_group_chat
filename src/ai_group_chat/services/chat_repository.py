from datetime import datetime
from typing import Optional, List
from uuid import uuid4

from ..models import (
    GroupChat, AIMember, Message, DiscussionMode,
    AIMemberCreate, AIMemberUpdate, MessageRole
)
from .database import db

class ChatRepository:
    """
    聊天数据仓库 (Repository)
    
    负责将领域对象 (Domain Models) 与数据库表进行映射和交互。
    所有 SQL 语句应仅出现在此类中。
    """
    
    def __init__(self):
        self.db = db

    # ============ ORM Mappers (Row -> Object) ============

    def _row_to_group(self, row: dict) -> GroupChat:
        members_rows = self.db.fetch_all("SELECT * FROM members WHERE group_id = ?", (row['id'],))
        members = [self._row_to_member(m) for m in members_rows]
        
        # Postgres returns datetime objects, SQLite returns strings (Compat)
        created_at = row['created_at']
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        elif not created_at:
            created_at = datetime.now()
        
        return GroupChat(
            id=row['id'],
            name=row['name'],
            created_at=created_at,
            manager_model=row['manager_model'],
            manager_thinking=bool(row['manager_thinking']),
            manager_temperature=row['manager_temperature'],
            discussion_mode=row.get('discussion_mode', DiscussionMode.FREE), 
            members=members
        )

    def _row_to_member(self, row: dict) -> AIMember:
        return AIMember(
            id=row['id'],
            name=row['name'],
            model_id=row['model_id'],
            description=row['description'],
            persona=row['persona'],
            thinking=bool(row['thinking']),
            temperature=row['temperature']
        )
    
    def _row_to_message(self, row: dict) -> Message:
        created_at = row['created_at']
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        elif not created_at:
            created_at = datetime.now()
        
        return Message(
            id=row['id'],
            group_id=row['group_id'],
            role=row['role'],
            content=row['content'],
            sender_name=row['sender_name'],
            mode=row['mode'],
            created_at=created_at,
        )

    # ============ Group Operations ============

    def get_group_by_name(self, name: str) -> Optional[GroupChat]:
        row = self.db.fetch_one("SELECT * FROM groups WHERE name = ?", (name,))
        return self._row_to_group(row) if row else None

    def get_group(self, group_id: str) -> Optional[GroupChat]:
        row = self.db.fetch_one("SELECT * FROM groups WHERE id = ?", (group_id,))
        return self._row_to_group(row) if row else None

    def list_groups(self) -> List[GroupChat]:
        rows = self.db.fetch_all("SELECT * FROM groups ORDER BY created_at DESC")
        return [self._row_to_group(row) for row in rows]

    def create_group(self, name: str, discussion_mode: str = 'free', 
                     manager_model: str = "gpt-4o-mini") -> GroupChat:
        group_id = str(uuid4())
        self.db.execute("""
            INSERT INTO groups (id, name, discussion_mode, manager_model)
            VALUES (?, ?, ?, ?)
        """, (group_id, name, discussion_mode, manager_model))
        return self.get_group(group_id)

    def delete_group(self, group_id: str) -> bool:
        cursor = self.db.execute("DELETE FROM groups WHERE id = ?", (group_id,))
        return cursor.rowcount > 0

    def update_manager_config(self, group_id: str, model_id: str, 
                            thinking: Optional[bool] = None, 
                            temperature: Optional[float] = None) -> bool:
        update_fields = ["manager_model = ?"]
        params = [model_id]
        
        if thinking is not None:
            update_fields.append("manager_thinking = ?")
            params.append(bool(thinking))
        if temperature is not None:
            update_fields.append("manager_temperature = ?")
            params.append(temperature)
            
        params.append(group_id)
        cursor = self.db.execute(f"UPDATE groups SET {', '.join(update_fields)} WHERE id = ?", tuple(params))
        return cursor.rowcount > 0

    # ============ Member Operations ============

    def add_member(self, group_id: str, data: AIMemberCreate) -> AIMember:
        member_id = str(uuid4())
        self.db.execute("""
            INSERT INTO members (id, group_id, name, model_id, description, thinking, temperature)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            member_id, group_id, data.model_id, data.model_id, data.description,
            bool(data.thinking), data.temperature
        ))
        row = self.db.fetch_one("SELECT * FROM members WHERE id = ?", (member_id,))
        return self._row_to_member(row)

    def add_raw_member(self, group_id: str, name: str, model_id: str, 
                       description: str, thinking: bool, temperature: float) -> None:
        """用于预设数据的底层添加"""
        member_id = str(uuid4())
        self.db.execute("""
            INSERT INTO members (id, group_id, name, model_id, description, thinking, temperature)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            member_id, group_id, name, model_id, description, 
            bool(thinking), temperature
        ))

    def update_member(self, group_id: str, member_id: str, data: AIMemberUpdate) -> Optional[AIMember]:
        fields = []
        params = []
        if data.description is not None:
            fields.append("description = ?")
            params.append(data.description)
        if data.thinking is not None:
            fields.append("thinking = ?")
            params.append(bool(data.thinking))
        if data.temperature is not None:
            fields.append("temperature = ?")
            params.append(data.temperature)
            
        if not fields:
            return None
            
        params.append(member_id)
        params.append(group_id)
        
        sql = f"UPDATE members SET {', '.join(fields)} WHERE id = ? AND group_id = ?"
        self.db.execute(sql, tuple(params))
        
        row = self.db.fetch_one("SELECT * FROM members WHERE id = ?", (member_id,))
        return self._row_to_member(row) if row else None

    def remove_member(self, group_id: str, member_id: str) -> bool:
        cursor = self.db.execute("DELETE FROM members WHERE id = ? AND group_id = ?", (member_id, group_id))
        return cursor.rowcount > 0

    def update_member_persona(self, group_id: str, member_id: str, persona: str) -> bool:
        cursor = self.db.execute("UPDATE members SET persona = ? WHERE id = ? AND group_id = ?", (persona, member_id, group_id))
        return cursor.rowcount > 0

    # ============ Message Operations ============

    def save_message(self, group_id: str, role: MessageRole, content: str, 
                     sender_name: str, mode: str) -> Message:
        msg_id = str(uuid4())
        self.db.execute("""
            INSERT INTO messages (id, group_id, role, content, sender_name, mode)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (msg_id, group_id, role, content, sender_name, mode))
        
        # 为了性能，这里可以不回查数据库，直接返回构造的对象 (除了 created_at 需要数据库时间，但我们可以认为它是 now)
        # 或者为了严谨，查一次。这里选择严谨查一次。
        row = self.db.fetch_one("SELECT * FROM messages WHERE id = ?", (msg_id,))
        return self._row_to_message(row)

    def get_messages(self, group_id: str, limit: int) -> List[Message]:
        sql = """
            SELECT * FROM (
                SELECT * FROM messages 
                WHERE group_id = ? 
                ORDER BY created_at DESC 
                LIMIT ?
            ) AS recent_msgs ORDER BY created_at ASC
        """
        rows = self.db.fetch_all(sql, (group_id, limit))
        return [self._row_to_message(row) for row in rows]

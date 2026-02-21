"""
成员 DAO

负责 AI 成员相关的数据库操作
"""

from typing import Optional, List
from uuid import uuid4

from ..models import AIMember, AIMemberCreate, AIMemberUpdate
from .base import BaseDAO


class MemberDAO(BaseDAO):
    """
    成员数据访问对象
    
    负责 members 表的 CRUD 操作
    """
    
    def _row_to_member(self, row: dict) -> AIMember:
        """将数据库行转换为 AIMember 对象"""
        return AIMember(
            id=row['id'],
            name=row['name'],
            model_id=row['model_id'],
            description=row['description'],
            task=row.get('persona'),
            thinking=bool(row['thinking']),
            temperature=row['temperature']
        )
    
    def get_by_id(self, member_id: str) -> Optional[dict]:
        """根据 ID 获取成员原始数据"""
        return self.db.fetch_one("SELECT * FROM members WHERE id = ?", (member_id,))
    
    def get_by_group(self, group_id: str) -> List[dict]:
        """获取群聊的所有成员原始数据"""
        return self.db.fetch_all("SELECT * FROM members WHERE group_id = ?", (group_id,))
    
    def add(self, group_id: str, data: AIMemberCreate) -> str:
        """
        添加成员
        
        Returns:
            新成员的 ID
        """
        member_id = str(uuid4())
        self.db.execute("""
            INSERT INTO members (id, group_id, name, model_id, description, thinking, temperature)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            member_id, group_id, data.model_id, data.model_id, data.description,
            bool(data.thinking), data.temperature
        ))
        return member_id
    
    def add_raw(self, group_id: str, name: str, model_id: str,
                description: str, thinking: bool, temperature: float) -> str:
        """
        添加成员（原始参数版本，用于预设数据）
        
        Returns:
            新成员的 ID
        """
        member_id = str(uuid4())
        self.db.execute("""
            INSERT INTO members (id, group_id, name, model_id, description, thinking, temperature)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            member_id, group_id, name, model_id, description,
            bool(thinking), temperature
        ))
        return member_id
    
    def update(self, group_id: str, member_id: str, data: AIMemberUpdate) -> bool:
        """更新成员信息"""
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
            return False
        
        params.extend([member_id, group_id])
        sql = f"UPDATE members SET {', '.join(fields)} WHERE id = ? AND group_id = ?"
        cursor = self.db.execute(sql, tuple(params))
        return cursor.rowcount > 0
    
    def delete(self, group_id: str, member_id: str) -> bool:
        """删除成员"""
        cursor = self.db.execute(
            "DELETE FROM members WHERE id = ? AND group_id = ?",
            (member_id, group_id)
        )
        return cursor.rowcount > 0
    
    def update_persona(self, group_id: str, member_id: str, persona: str) -> bool:
        """更新成员人设"""
        cursor = self.db.execute(
            "UPDATE members SET persona = ? WHERE id = ? AND group_id = ?",
            (persona, member_id, group_id)
        )
        return cursor.rowcount > 0


# 全局实例
member_dao = MemberDAO()

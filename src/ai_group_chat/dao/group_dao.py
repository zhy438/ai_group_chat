"""
群聊 DAO

负责群聊相关的数据库操作
"""

from typing import Optional, List
from uuid import uuid4

from ..models import GroupChat, DiscussionMode
from .base import BaseDAO


class GroupDAO(BaseDAO):
    """
    群聊数据访问对象
    
    负责 groups 表的 CRUD 操作
    """
    
    def _row_to_group(self, row: dict, members: List = None) -> GroupChat:
        """将数据库行转换为 GroupChat 对象"""
        created_at = self.parse_datetime(row['created_at'])
        
        return GroupChat(
            id=row['id'],
            name=row['name'],
            created_at=created_at,
            manager_model=row['manager_model'],
            manager_thinking=bool(row['manager_thinking']),
            manager_temperature=row['manager_temperature'],
            discussion_mode=row.get('discussion_mode', DiscussionMode.FREE),
            compression_threshold=row.get('compression_threshold', 0.8),
            memory_enabled=bool(row.get('memory_enabled', True)),
            archive_enabled=bool(row.get('archive_enabled', True)),
            retrieve_enabled=bool(row.get('retrieve_enabled', True)),
            scope_user_global=bool(row.get('scope_user_global', True)),
            scope_group_local=bool(row.get('scope_group_local', True)),
            scope_agent_local=bool(row.get('scope_agent_local', True)),
            memory_injection_ratio=row.get('memory_injection_ratio', 0.2),
            memory_top_n=row.get('memory_top_n', 5),
            memory_min_confidence=row.get('memory_min_confidence', 0.75),
            memory_score_threshold=row.get('memory_score_threshold', 0.35),
            members=members or []
        )
    
    def get_by_id(self, group_id: str) -> Optional[dict]:
        """根据 ID 获取群聊原始数据"""
        return self.db.fetch_one("SELECT * FROM groups WHERE id = ?", (group_id,))
    
    def get_by_name(self, name: str) -> Optional[dict]:
        """根据名称获取群聊原始数据"""
        return self.db.fetch_one("SELECT * FROM groups WHERE name = ?", (name,))
    
    def list_all(self) -> List[dict]:
        """获取所有群聊的原始数据"""
        return self.db.fetch_all("SELECT * FROM groups ORDER BY created_at DESC")
    
    def create(self, name: str, discussion_mode: str = 'free',
               manager_model: str = "gpt-4o-mini") -> str:
        """
        创建群聊
        
        Returns:
            新创建的群聊 ID
        """
        group_id = str(uuid4())
        self.db.execute("""
            INSERT INTO groups (id, name, discussion_mode, manager_model)
            VALUES (?, ?, ?, ?)
        """, (group_id, name, discussion_mode, manager_model))
        return group_id
    
    def delete(self, group_id: str) -> bool:
        """删除群聊"""
        cursor = self.db.execute("DELETE FROM groups WHERE id = ?", (group_id,))
        return cursor.rowcount > 0
    
    def update_manager_config(self, group_id: str, model_id: str,
                              thinking: Optional[bool] = None,
                              temperature: Optional[float] = None) -> bool:
        """更新群聊管理员配置"""
        update_fields = ["manager_model = ?"]
        params = [model_id]
        
        if thinking is not None:
            update_fields.append("manager_thinking = ?")
            params.append(bool(thinking))
        if temperature is not None:
            update_fields.append("manager_temperature = ?")
            params.append(temperature)
        
        params.append(group_id)
        cursor = self.db.execute(
            f"UPDATE groups SET {', '.join(update_fields)} WHERE id = ?",
            tuple(params)
        )
        return cursor.rowcount > 0

    def update_compression_threshold(self, group_id: str, threshold: float) -> bool:
        """更新群聊压缩阈值"""
        cursor = self.db.execute(
            "UPDATE groups SET compression_threshold = ? WHERE id = ?",
            (threshold, group_id)
        )
        return cursor.rowcount > 0

    def update_memory_settings(self, group_id: str, settings: dict) -> bool:
        """更新群聊长期记忆配置"""
        if not settings:
            return False
        update_fields = []
        params = []
        for key, value in settings.items():
            update_fields.append(f"{key} = ?")
            params.append(value)
        params.append(group_id)
        cursor = self.db.execute(
            f"UPDATE groups SET {', '.join(update_fields)} WHERE id = ?",
            tuple(params)
        )
        return cursor.rowcount > 0



# 全局实例
group_dao = GroupDAO()

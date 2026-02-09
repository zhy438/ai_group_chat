"""
上下文快照 DAO

负责 group_context_snapshots 表的操作
"""

import json
from typing import Optional, List
from .base import BaseDAO
from ..models import Message

class ContextSnapshotDAO(BaseDAO):
    """
    上下文快照数据访问对象
    """
    
    def get_latest(self, group_id: str) -> Optional[dict]:
        """获取群聊最新的上下文快照"""
        sql = """
            SELECT * FROM group_context_snapshots
            WHERE group_id = ?
            ORDER BY created_at DESC
            LIMIT 1
        """
        return self.db.fetch_one(sql, (group_id,))

    def save(self, group_id: str, last_message_id: str, context_messages: List[Message], token_count: int) -> int:
        """保存上下文快照"""
        # 序列化 Messages 为 JSON 字符串
        # 尝试兼容 Pydantic v1/v2
        try:
            # Pydantic v2
            if hasattr(Message, 'model_dump'):
                data = [msg.model_dump(mode='json') for msg in context_messages]
            else:
                # Pydantic v1
                data = [msg.dict() for msg in context_messages]
                
            context_content = json.dumps(data, ensure_ascii=False, default=str)
            
        except Exception as e:
            # Fallback
            import logging
            logging.warning(f"Snapshot serialization warning: {e}")
            context_content = json.dumps([vars(msg) for msg in context_messages], ensure_ascii=False, default=str)
        
        sql = """
            INSERT INTO group_context_snapshots (group_id, last_message_id, context_content, token_count)
            VALUES (?, ?, ?, ?)
        """
        # 执行插入
        self.db.execute(sql, (group_id, last_message_id, context_content, token_count))
        return 0

context_snapshot_dao = ContextSnapshotDAO()

"""
数据访问对象层 (DAO)

将数据库操作与业务逻辑分离，提供统一的数据访问接口
"""

from .base import BaseDAO
from .database import Database, db
from .context_snapshot_dao import ContextSnapshotDAO, context_snapshot_dao
from .group_dao import GroupDAO, group_dao
from .member_dao import MemberDAO, member_dao
from .message_dao import MessageDAO, message_dao

__all__ = [
    # 数据库连接
    "Database",
    "db",
    # 基类
    "BaseDAO",
    # 群聊
    "GroupDAO",
    "group_dao",
    # 成员
    "MemberDAO", 
    "member_dao",
    # 消息
    "MessageDAO",
    "message_dao",
]

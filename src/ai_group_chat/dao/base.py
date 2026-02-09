"""
基础 DAO 类

提供数据库连接和通用的行转对象方法
"""

from datetime import datetime
from typing import Optional

from .database import db, Database


class BaseDAO:
    """
    基础数据访问对象
    
    所有 DAO 类的基类，提供数据库连接
    """
    
    def __init__(self, database: Database = None):
        self.db = database or db
    
    @staticmethod
    def parse_datetime(value) -> datetime:
        """解析日期时间字段（兼容 PostgreSQL 和 SQLite）"""
        if isinstance(value, str):
            return datetime.fromisoformat(value)
        elif value is None:
            return datetime.now()
        return value

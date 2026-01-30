"""配置管理模块"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    """应用配置"""
    
    # 应用配置
    app_name: str = "AI群聊"
    debug: bool = True
    
    # 数据库配置
    database_url: str = "postgresql://admin:password@localhost:5432/ai_chat_db"
    
    # AI模型API配置 (通过 aihubmix 或其他代理)
    ai_api_base: str = "https://aihubmix.com/v1"
    ai_api_key: str = ""
    
    # 默认模型配置
    default_model: str = "mimo-v2-flash-free"
    
    # LangGraph 配置
    max_discussion_rounds: int = 5  # 最大讨论轮数
    
    # LangSmith 追踪配置
    langchain_tracing_v2: bool = False
    langchain_endpoint: str = "https://api.smith.langchain.com"
    langchain_api_key: str = ""
    langchain_project: str = "ai-group-chat"
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # 忽略未定义的环境变量
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()


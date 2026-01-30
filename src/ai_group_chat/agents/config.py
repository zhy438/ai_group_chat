"""AutoGen LLM 配置"""

from ..config import get_settings


def get_llm_config(model_id: str) -> dict:
    """
    获取 AutoGen 的 LLM 配置
    
    Args:
        model_id: 模型ID（如 qwen-flash, deepseek-chat 等）
    
    Returns:
        AutoGen 格式的 LLM 配置字典
    """
    settings = get_settings()
    
    return {
        "config_list": [{
            "model": model_id,
            "base_url": settings.ai_api_base,
            "api_key": settings.ai_api_key,
        }],
        "temperature": 0.7,
        "timeout": 120,
    }

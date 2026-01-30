"""LLM 客户端封装 - 使用 OpenAI SDK 调用 aihubmix"""

from typing import Optional
from openai import AsyncOpenAI
import logging

from ..config import get_settings

logger = logging.getLogger(__name__)


class LLMClient:
    """
    LLM客户端 - 使用 OpenAI SDK 调用 aihubmix
    
    aihubmix 完全兼容 OpenAI API 格式
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.client = AsyncOpenAI(
            base_url=self.settings.ai_api_base,
            api_key=self.settings.ai_api_key,
        )
    
    async def chat(
        self,
        model: str,
        messages: list[dict],
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        """
        发送聊天请求
        
        Args:
            model: 模型ID，如 "mimo-v2-flash-free"
            messages: 消息列表 [{"role": "user", "content": "..."}]
            system_prompt: 系统提示词
            temperature: 温度参数
            max_tokens: 最大token数
            
        Returns:
            模型回复内容
        """
        # 构建完整消息列表
        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)
        
        logger.info(f"Calling model: {model}")
        
        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=full_messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            # 提取回复内容
            message = response.choices[0].message
            content = message.content or ""
            
            # 处理可能的 reasoning_content（某些模型的思考过程）
            reasoning = getattr(message, "reasoning_content", None)
            if not content and reasoning:
                content = reasoning
            
            return content or "[模型未返回内容]"
            
        except Exception as e:
            logger.error(f"API Error: {e}")
            raise


# 全局客户端实例
llm_client = LLMClient()





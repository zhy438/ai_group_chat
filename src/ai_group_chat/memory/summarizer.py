"""
智能摘要器

使用 LLM 对消息进行结构化摘要
"""

import asyncio
from typing import List, Optional
from loguru import logger

from ..models import Message
from ..llm.client import llm_client
from ..prompts import SUMMARIZE_SYSTEM_PROMPT, build_summarize_user_prompt


class Summarizer:
    """
    智能摘要器
    
    使用 LLM 对消息进行摘要
    """
    
    def __init__(self, model: str = "gpt-4o-mini"):
        """
        初始化摘要器
        
        Args:
            model: 用于摘要的模型ID
        """
        self.model = model
        self.client = llm_client
    MAX_RETRIES = 3  # 最大重试次数
    RETRY_DELAY = 1  # 重试间隔（秒）
    
    async def summarize(self, messages: List[Message]) -> Optional[str]:
        """
        对消息列表生成摘要
        
        包含重试逻辑：失败时最多重试3次
        
        Args:
            messages: 需要摘要的消息列表
            
        Returns:
            摘要文本，如果所有重试都失败则返回 None
        """
        if not messages:
            return None
        
        # 构建对话文本
        conversation_lines = []
        for msg in messages:
            sender = msg.sender_name or ("用户" if msg.role == "user" else "AI")
            conversation_lines.append(f"[{sender}]: {msg.content}")
        
        conversation_text = "\n".join(conversation_lines)
        user_prompt = build_summarize_user_prompt(conversation_text)
        
        # 带重试的 LLM 调用
        last_error = None
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                summary = await self.client.chat(
                    model=self.model,
                    messages=[{"role": "user", "content": user_prompt}],
                    system_prompt=SUMMARIZE_SYSTEM_PROMPT,
                    temperature=0.3,
                    max_tokens=500,
                )
                
                logger.info(f"✅ LLM 摘要生成成功（第 {attempt} 次尝试），原文 {len(conversation_text)} 字 → 摘要 {len(summary)} 字")
                return summary
                
            except Exception as e:
                last_error = e
                logger.warning(f"⚠️ LLM 摘要失败（第 {attempt}/{self.MAX_RETRIES} 次）: {e}")
                
                if attempt < self.MAX_RETRIES:
                    await asyncio.sleep(self.RETRY_DELAY)
        
        # 所有重试都失败了
        logger.error(f"❌ LLM 摘要彻底失败，已重试 {self.MAX_RETRIES} 次: {last_error}")
        return None  # 返回 None，让上层决定不压缩
    
    def summarize_sync(self, messages: List[Message]) -> Optional[str]:
        """
        同步版本的摘要方法（用于非异步环境）
        
        Returns:
            摘要文本，失败返回 None
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self.summarize(messages))
                    return future.result(timeout=60)  # 增加超时以容纳重试
            else:
                return loop.run_until_complete(self.summarize(messages))
        except Exception as e:
            logger.error(f"同步摘要失败: {e}")
            return None  # 失败返回 None


# 全局摘要器实例
summarizer = Summarizer()

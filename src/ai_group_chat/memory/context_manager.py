"""
上下文管理器

核心入口：检测 Token 阈值、协调调用分类器、评分器、压缩器
"""

import tiktoken
from typing import List, Optional
from loguru import logger

from ..models import Message
from .classifier import MessageClassifier
from .value_scorer import ValueScorer
from .compressor import ContextCompressor


class ContextManager:
    """
    上下文管理器
    
    负责：
    1. 计算当前上下文的 Token 数量
    2. 判断是否需要触发压缩
    3. 协调调用分类、评分、压缩流程
    """
    
    # 默认配置
    DEFAULT_MODEL = "gpt-4"  # 用于 token 计算的模型
    DEFAULT_MAX_TOKENS = 128000  # 默认最大 token 数
    DEFAULT_THRESHOLD_RATIO = 0.8  # 触发压缩的阈值（80%）
    
    def __init__(self,
                 model: str = DEFAULT_MODEL,
                 max_tokens: int = DEFAULT_MAX_TOKENS,
                 threshold_ratio: float = DEFAULT_THRESHOLD_RATIO):
        """
        初始化上下文管理器
        
        Args:
            model: 用于 token 计算的模型名称
            max_tokens: 模型的最大上下文长度
            threshold_ratio: 触发压缩的阈值比例
        """
        self.model = model
        self.max_tokens = max_tokens
        self.threshold_ratio = threshold_ratio
        self.threshold_tokens = int(max_tokens * threshold_ratio)
        
        # 初始化 tiktoken 编码器
        try:
            self.encoder = tiktoken.encoding_for_model(model)
        except KeyError:
            # 如果模型不支持，使用 cl100k_base（GPT-4 使用的编码）
            self.encoder = tiktoken.get_encoding("cl100k_base")
        
        # 初始化子组件
        self.classifier = MessageClassifier()
        self.scorer = ValueScorer()
        self.compressor = ContextCompressor()
    
    def set_max_tokens(self, max_tokens: int) -> None:
        """
        动态设置最大 token 数
        
        用于根据群聊中模型的最小上下文窗口调整
        
        Args:
            max_tokens: 新的最大 token 数
        """
        if max_tokens != self.max_tokens:
            old_max = self.max_tokens
            self.max_tokens = max_tokens
            self.threshold_tokens = int(max_tokens * self.threshold_ratio)
            logger.debug(f"📐 上下文窗口调整: {old_max} → {max_tokens} tokens")
    
    def count_tokens(self, text: str) -> int:
        """计算文本的 token 数量"""
        return len(self.encoder.encode(text))
    
    def count_messages_tokens(self, messages: List[Message]) -> int:
        """
        计算消息列表的总 token 数
        
        注意：这是一个估算值，实际 API 调用时还会有额外的格式化开销
        """
        total = 0
        for msg in messages:
            # 消息内容
            total += self.count_tokens(msg.content)
            # 发送者名称（约 4 tokens 的开销）
            if msg.sender_name:
                total += self.count_tokens(msg.sender_name) + 4
        
        # 添加一些额外的格式化开销估算
        total += len(messages) * 4  # 每条消息约 4 tokens 的格式开销
        
        return total
    
    def should_compress(self, messages: List[Message]) -> bool:
        """
        判断是否需要触发压缩
        
        Args:
            messages: 当前消息列表
            
        Returns:
            是否需要压缩
        """
        current_tokens = self.count_messages_tokens(messages)
        should = current_tokens >= self.threshold_tokens
        
        if should:
            logger.warning(
                f"⚠️ Token 超过阈值: {current_tokens}/{self.max_tokens} "
                f"({current_tokens/self.max_tokens*100:.1f}%) >= {self.threshold_ratio*100:.0f}%"
            )
        
        return should
    
    def process(self, messages: List[Message], 
                force: bool = False) -> List[Message]:
        """
        处理消息列表（同步版本）
        
        核心流程：
        1. 检查是否需要压缩
        2. 消息分类
        3. 价值评分
        4. 执行压缩
        
        Args:
            messages: 原始消息列表
            force: 是否强制执行压缩（忽略阈值检查）
            
        Returns:
            处理后的消息列表（可能被压缩）
        """
        if not messages:
            return messages
        
        # 1. 检查是否需要压缩
        if not force and not self.should_compress(messages):
            return messages
        
        logger.info(f"🔄 开始上下文优化流程，当前消息数: {len(messages)}")
        
        # 2. 消息分类
        self.classifier.update_message_types(messages)
        
        # 3. 价值评分
        self.scorer.score_messages(messages)
        
        # 4. 执行压缩
        compressed_messages = self.compressor.compress(messages)
        
        # 统计压缩效果
        original_tokens = self.count_messages_tokens(messages)
        compressed_tokens = self.count_messages_tokens(compressed_messages)
        saved_tokens = original_tokens - compressed_tokens
        saved_ratio = saved_tokens / original_tokens * 100 if original_tokens > 0 else 0
        
        logger.info(
            f"✨ 压缩完成: {original_tokens} → {compressed_tokens} tokens "
            f"(节省 {saved_tokens} tokens, {saved_ratio:.1f}%)"
        )
        
        return compressed_messages
    
    async def process_async(self, messages: List[Message], 
                            force: bool = False) -> List[Message]:
        """
        异步处理消息列表（不阻塞主线程）
        
        核心流程：
        1. 检查是否需要压缩
        2. 消息分类（异步 LLM）
        3. 价值评分
        4. 执行压缩（异步 LLM 摘要）
        
        Args:
            messages: 原始消息列表
            force: 是否强制执行压缩（忽略阈值检查）
            
        Returns:
            处理后的消息列表（可能被压缩）
        """
        if not messages:
            return messages
        
        # 1. 检查是否需要压缩
        if not force and not self.should_compress(messages):
            return messages
        
        logger.info(f"🔄 开始异步上下文优化流程，当前消息数: {len(messages)}")
        
        # 2. 消息分类（异步）
        await self.classifier.update_message_types_async(messages)
        
        # 3. 价值评分（CPU 操作，无需异步）
        self.scorer.score_messages(messages)
        
        # 4. 执行压缩（异步）
        compressed_messages = await self.compressor.compress_async(messages)
        
        # 统计压缩效果
        original_tokens = self.count_messages_tokens(messages)
        compressed_tokens = self.count_messages_tokens(compressed_messages)
        saved_tokens = original_tokens - compressed_tokens
        saved_ratio = saved_tokens / original_tokens * 100 if original_tokens > 0 else 0
        
        logger.info(
            f"✨ 异步压缩完成: {original_tokens} → {compressed_tokens} tokens "
            f"(节省 {saved_tokens} tokens, {saved_ratio:.1f}%)"
        )
        
        return compressed_messages
    
    def get_stats(self, messages: List[Message]) -> dict:
        """
        获取当前上下文的统计信息
        
        Args:
            messages: 消息列表
            
        Returns:
            统计信息字典
        """
        current_tokens = self.count_messages_tokens(messages)
        return {
            "message_count": len(messages),
            "current_tokens": current_tokens,
            "max_tokens": self.max_tokens,
            "threshold_tokens": self.threshold_tokens,
            "usage_ratio": current_tokens / self.max_tokens,
            "needs_compression": current_tokens >= self.threshold_tokens,
        }

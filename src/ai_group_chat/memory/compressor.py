"""
上下文压缩引擎

根据消息价值分数执行不同的压缩策略：
- 高分消息：全部保留
- 中分消息：结构化摘要
- 低分消息：直接丢弃
"""

from typing import List, Tuple, Optional
from loguru import logger

from ..models import Message, MessageType
from .value_scorer import ValueThresholds


class ContextCompressor:
    """
    上下文压缩器
    
    对已评分的消息列表执行压缩操作
    """
    
    def __init__(self, 
                 high_threshold: float = ValueThresholds.HIGH,
                 medium_threshold: float = ValueThresholds.MEDIUM,
                 summarizer = None):
        """
        初始化压缩器
        
        Args:
            high_threshold: 高分阈值（以上全部保留）
            medium_threshold: 中分阈值（以上做摘要，以下丢弃）
            summarizer: 摘要生成器（可选，用于中分消息的摘要）
        """
        self.high_threshold = high_threshold
        self.medium_threshold = medium_threshold
        self.summarizer = summarizer
    
    def triage_messages(self, messages: List[Message]) -> Tuple[List[Message], List[Message], List[Message]]:
        """
        根据价值分数将消息分流
        
        Args:
            messages: 已评分的消息列表
            
        Returns:
            (高分消息, 中分消息, 低分消息) 三个列表
        """
        high_value = []
        medium_value = []
        low_value = []
        
        for msg in messages:
            score = msg.value_score or 0
            
            if score >= self.high_threshold:
                high_value.append(msg)
            elif score >= self.medium_threshold:
                medium_value.append(msg)
            else:
                low_value.append(msg)
        
        logger.debug(f"📊 消息分流: 高分={len(high_value)}, 中分={len(medium_value)}, 低分={len(low_value)}")
        return high_value, medium_value, low_value
    
    def summarize_messages(self, messages: List[Message]) -> Optional[Message]:
        """
        对一组消息生成摘要
        
        使用 LLM 进行智能摘要，失败时降级到规则摘要
        
        Args:
            messages: 需要摘要的消息列表
            
        Returns:
            摘要消息，如果无法摘要则返回 None
        """
        if not messages:
            return None
        
        # 使用 LLM 智能摘要
        from .summarizer import summarizer
        
        try:
            summary_text = summarizer.summarize_sync(messages)
        except Exception as e:
            logger.error(f"摘要生成异常: {e}")
            summary_text = None
        
        # 如果摘要失败，返回 None（不压缩这些消息）
        if not summary_text:
            logger.warning(f"⚠️ 摘要生成失败，将保留原始 {len(messages)} 条中分消息")
            return None
        
        # 创建摘要消息（复用第一条消息的元数据）
        first_msg = messages[0]
        summary_message = Message(
            id=f"summary_{first_msg.id}",
            group_id=first_msg.group_id,
            role=first_msg.role,
            content=summary_text,
            sender_name="📋 历史摘要",
            mode=first_msg.mode,
            created_at=first_msg.created_at,
            message_type=MessageType.STATUS,
            is_compressed=True,
            original_content=None,
            value_score=ValueThresholds.HIGH,
        )
        
        return summary_message
    
    def compress(self, messages: List[Message], 
                 keep_recent: int = 5) -> List[Message]:
        """
        执行压缩（同步版本）
        
        策略：
        1. 最近 N 条消息无条件保留
        2. 高分消息全部保留
        3. 中分消息合并摘要
        4. 低分消息丢弃
        
        Args:
            messages: 已评分的消息列表（按时间顺序）
            keep_recent: 无条件保留的最近消息数量
            
        Returns:
            压缩后的消息列表
        """
        if len(messages) <= keep_recent:
            return messages
        
        # 分离最近的消息（无条件保留）
        recent_messages = messages[-keep_recent:]
        older_messages = messages[:-keep_recent]
        
        # 对较早的消息进行分流
        high_value, medium_value, low_value = self.triage_messages(older_messages)
        
        # 构建压缩后的消息列表
        compressed = []
        
        # 1. 添加高分消息
        compressed.extend(high_value)
        
        # 2. 对中分消息生成摘要（如果失败则保留原消息）
        if medium_value:
            summary = self.summarize_messages(medium_value)
            if summary:
                compressed.append(summary)
                logger.info(f"📝 已将 {len(medium_value)} 条中分消息压缩为摘要")
            else:
                # 摘要失败，保留原消息不压缩
                compressed.extend(medium_value)
                logger.info(f"📌 摘要失败，保留原始 {len(medium_value)} 条中分消息")
        
        # 3. 低分消息直接丢弃
        if low_value:
            logger.info(f"🗑️ 已丢弃 {len(low_value)} 条低分消息")
        
        # 4. 按时间排序（保持对话顺序）
        compressed.sort(key=lambda m: m.created_at)
        
        # 5. 添加最近的消息
        compressed.extend(recent_messages)
        
        logger.info(f"✅ 压缩完成: {len(messages)} → {len(compressed)} 条消息")
        return compressed
    
    async def summarize_messages_async(self, messages: List[Message]) -> Optional[Message]:
        """
        异步生成摘要
        
        Args:
            messages: 需要摘要的消息列表
            
        Returns:
            摘要消息，如果无法摘要则返回 None
        """
        if not messages:
            return None
        
        from .summarizer import summarizer
        
        try:
            summary_text = await summarizer.summarize(messages)
        except Exception as e:
            logger.error(f"摘要生成异常: {e}")
            summary_text = None
        
        if not summary_text:
            logger.warning(f"⚠️ 摘要生成失败，将保留原始 {len(messages)} 条中分消息")
            return None
        
        first_msg = messages[0]
        summary_message = Message(
            id=f"summary_{first_msg.id}",
            group_id=first_msg.group_id,
            role=first_msg.role,
            content=summary_text,
            sender_name="📋 历史摘要",
            mode=first_msg.mode,
            created_at=first_msg.created_at,
            message_type=MessageType.STATUS,
            is_compressed=True,
            original_content=None,
            value_score=ValueThresholds.HIGH,
        )
        
        return summary_message
    
    async def compress_async(self, messages: List[Message], 
                             keep_recent: int = 5) -> List[Message]:
        """
        异步执行压缩（不阻塞主线程）
        
        策略同 compress()，但使用异步 LLM 调用
        """
        if len(messages) <= keep_recent:
            return messages
        
        recent_messages = messages[-keep_recent:]
        older_messages = messages[:-keep_recent]
        
        high_value, medium_value, low_value = self.triage_messages(older_messages)
        
        compressed = []
        compressed.extend(high_value)
        
        # 异步生成摘要
        if medium_value:
            summary = await self.summarize_messages_async(medium_value)
            if summary:
                compressed.append(summary)
                logger.info(f"📝 已将 {len(medium_value)} 条中分消息压缩为摘要")
            else:
                compressed.extend(medium_value)
                logger.info(f"📌 摘要失败，保留原始 {len(medium_value)} 条中分消息")
        
        if low_value:
            logger.info(f"🗑️ 已丢弃 {len(low_value)} 条低分消息")
        
        compressed.sort(key=lambda m: m.created_at)
        compressed.extend(recent_messages)
        
        logger.info(f"✅ 压缩完成: {len(messages)} → {len(compressed)} 条消息")
        return compressed

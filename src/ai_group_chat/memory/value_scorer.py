"""
价值计算器

计算每条消息的保留价值：Value = ΣWeight × Time_Decay
"""

import math
from datetime import datetime, timedelta
from typing import List, Dict
from ..models import Message, MessageType


class ValueScorer:
    """
    消息价值评分器
    
    根据消息类型权重和时间衰减计算每条消息的价值分数
    """
    
    # 消息类型权重配置（可调整）
    DEFAULT_WEIGHTS: Dict[MessageType, float] = {
        MessageType.USER: 10.0,      # 用户消息权重最高
        MessageType.STATUS: 7.0,     # 关键状态次之
        MessageType.FAILURE: 5.0,    # 失败记录需要保留教训
        MessageType.REASONING: 2.0,  # 推理过程权重较低
        MessageType.NORMAL: 3.0,     # 普通消息中等
    }
    
    # 时间衰减配置
    DECAY_HALF_LIFE_HOURS = 24.0  # 半衰期：24小时后价值减半
    
    def __init__(self, weights: Dict[MessageType, float] = None):
        """
        初始化价值评分器
        
        Args:
            weights: 自定义权重配置，None 则使用默认权重
        """
        self.weights = weights or self.DEFAULT_WEIGHTS
    
    def calculate_time_decay(self, created_at: datetime, reference_time: datetime = None) -> float:
        """
        计算时间衰减因子
        
        使用指数衰减：decay = 0.5 ^ (hours_elapsed / half_life)
        
        Args:
            created_at: 消息创建时间
            reference_time: 参考时间（默认为当前时间）
            
        Returns:
            衰减因子，范围 (0, 1]，时间越久越接近 0
        """
        if reference_time is None:
            reference_time = datetime.now()
        
        # 计算经过的小时数
        time_diff = reference_time - created_at
        hours_elapsed = time_diff.total_seconds() / 3600
        
        # 确保不会出现负数（消息来自未来？）
        if hours_elapsed < 0:
            hours_elapsed = 0
        
        # 指数衰减公式
        decay = math.pow(0.5, hours_elapsed / self.DECAY_HALF_LIFE_HOURS)
        
        return decay
    
    def calculate_value(self, message: Message, reference_time: datetime = None) -> float:
        """
        计算单条消息的价值分数
        
        Value = Weight × Time_Decay
        
        Args:
            message: 消息对象
            reference_time: 参考时间
            
        Returns:
            价值分数
        """
        # 获取消息类型权重
        weight = self.weights.get(message.message_type, self.weights[MessageType.NORMAL])
        
        # 计算时间衰减
        time_decay = self.calculate_time_decay(message.created_at, reference_time)
        
        # 计算最终价值
        value = weight * time_decay
        
        return round(value, 4)
    
    def score_messages(self, messages: List[Message], reference_time: datetime = None) -> List[Message]:
        """
        为消息列表中的每条消息计算价值分数
        
        Args:
            messages: 消息列表
            reference_time: 参考时间
            
        Returns:
            更新了 value_score 字段的消息列表
        """
        for msg in messages:
            msg.value_score = self.calculate_value(msg, reference_time)
        return messages
    
    def sort_by_value(self, messages: List[Message], descending: bool = True) -> List[Message]:
        """
        按价值分数排序消息
        
        Args:
            messages: 消息列表
            descending: 是否降序（高价值在前）
            
        Returns:
            排序后的消息列表
        """
        return sorted(messages, key=lambda m: m.value_score or 0, reverse=descending)


# 价值分数阈值配置
class ValueThresholds:
    """价值分数阈值，用于决定压缩策略"""
    HIGH = 5.0      # 高分：全部保留
    MEDIUM = 2.0    # 中分：结构化摘要
    # 低于 MEDIUM 的消息：直接丢弃

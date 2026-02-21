"""
消息分类器

使用 LLM 对消息进行智能分类，支持批量处理和重试机制
"""

import asyncio
import json
import re
from typing import List, Optional
from loguru import logger

from ..models import Message, MessageRole, MessageType
from ..llm.client import llm_client
from ..prompts import CLASSIFY_SYSTEM_PROMPT, build_classify_user_prompt


class MessageClassifier:
    """
    消息分类器
    
    使用 LLM 进行智能分类，失败时降级到规则匹配
    """
    
    MAX_RETRIES = 3
    RETRY_DELAY = 1
    BATCH_SIZE = 20  # 每批处理的消息数量
    
    # 规则匹配的关键词（用于降级）
    STATUS_KEYWORDS = [
        "完成", "成功", "已经", "确定", "决定", "最终",
        "结论", "总结", "采用", "选择", "确认",
        "done", "completed", "success", "decided", "conclusion"
    ]
    
    REASONING_KEYWORDS = [
        "考虑", "分析", "比较", "权衡", "思考", "评估",
        "方案", "选项", "可能", "或者", "如果",
        "think", "consider", "analyze", "compare", "option", "maybe"
    ]
    
    FAILURE_KEYWORDS = [
        "失败", "错误", "问题", "无法", "不能", "报错",
        "异常", "bug", "error", "failed", "issue", "cannot"
    ]
    
    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self.client = llm_client
        
        # 编译正则表达式（用于降级）
        self._status_pattern = re.compile(
            '|'.join(self.STATUS_KEYWORDS), re.IGNORECASE
        )
        self._reasoning_pattern = re.compile(
            '|'.join(self.REASONING_KEYWORDS), re.IGNORECASE
        )
        self._failure_pattern = re.compile(
            '|'.join(self.FAILURE_KEYWORDS), re.IGNORECASE
        )
    
    async def classify_batch_async(self, messages: List[Message]) -> List[MessageType]:
        """
        使用 LLM 批量分类消息
        
        Args:
            messages: 消息列表
            
        Returns:
            消息类型列表（与输入顺序对应）
        """
        if not messages:
            return []
        
        # 构建消息描述
        msg_descriptions = []
        for i, msg in enumerate(messages):
            sender = msg.sender_name or ("用户" if msg.role == MessageRole.USER else "AI")
            msg_descriptions.append(f"[{i}] [{sender}]: {msg.content}")
        
        messages_text = "\n".join(msg_descriptions)
        user_prompt = build_classify_user_prompt(messages_text)
        
        # 带重试的 LLM 调用
        last_error = None
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                response = await self.client.chat(
                    model=self.model,
                    messages=[{"role": "user", "content": user_prompt}],
                    system_prompt=CLASSIFY_SYSTEM_PROMPT,
                    temperature=0.1,  # 低温度保证一致性
                    max_tokens=1000,
                )
                
                # 解析 JSON 响应
                types = self._parse_response(response, len(messages))
                
                if types:
                    logger.info(f"✅ LLM 分类成功（第 {attempt} 次尝试），分类了 {len(messages)} 条消息")
                    return types
                else:
                    raise ValueError("解析分类结果失败")
                    
            except Exception as e:
                last_error = e
                logger.warning(f"⚠️ LLM 分类失败（第 {attempt}/{self.MAX_RETRIES} 次）: {e}")
                
                if attempt < self.MAX_RETRIES:
                    await asyncio.sleep(self.RETRY_DELAY)
        
        # 所有重试都失败，降级到规则匹配
        logger.warning(f"⚠️ LLM 分类彻底失败，降级到规则匹配: {last_error}")
        return [self._classify_by_rules(msg) for msg in messages]
    
    def _parse_response(self, response: str, expected_count: int) -> Optional[List[MessageType]]:
        """解析 LLM 响应的 JSON"""
        try:
            # 尝试提取 JSON 数组
            # 有时 LLM 会在 JSON 前后加其他文字
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if not json_match:
                return None
            
            data = json.loads(json_match.group())
            
            if not isinstance(data, list):
                return None
            
            # 构建类型映射
            type_map = {}
            for item in data:
                if isinstance(item, dict) and "index" in item and "type" in item:
                    idx = item["index"]
                    type_str = item["type"].lower()
                    
                    # 映射到 MessageType
                    if type_str == "user":
                        type_map[idx] = MessageType.USER
                    elif type_str == "status":
                        type_map[idx] = MessageType.STATUS
                    elif type_str == "reasoning":
                        type_map[idx] = MessageType.REASONING
                    elif type_str == "failure":
                        type_map[idx] = MessageType.FAILURE
                    else:
                        type_map[idx] = MessageType.NORMAL
            
            # 按顺序构建结果列表
            result = []
            for i in range(expected_count):
                result.append(type_map.get(i, MessageType.NORMAL))
            
            return result
            
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.error(f"解析分类响应失败: {e}")
            return None
    
    def _classify_by_rules(self, message: Message) -> MessageType:
        """规则匹配分类（降级方案）"""
        if message.role == MessageRole.USER:
            return MessageType.USER
        
        content = message.content
        
        failure_matches = len(self._failure_pattern.findall(content))
        status_matches = len(self._status_pattern.findall(content))
        reasoning_matches = len(self._reasoning_pattern.findall(content))
        
        if failure_matches >= 2:
            return MessageType.FAILURE
        if status_matches >= 2:
            return MessageType.STATUS
        if reasoning_matches >= 3:
            return MessageType.REASONING
        
        return MessageType.NORMAL
    
    def classify(self, message: Message) -> MessageType:
        """
        同步分类单条消息（使用规则匹配，避免单条调用 LLM）
        """
        return self._classify_by_rules(message)
    
    def classify_batch(self, messages: List[Message]) -> List[MessageType]:
        """
        同步批量分类消息
        
        尝试使用 LLM，失败则降级到规则匹配
        """
        if not messages:
            return []
        
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self.classify_batch_async(messages))
                    return future.result(timeout=60)
            else:
                return loop.run_until_complete(self.classify_batch_async(messages))
        except Exception as e:
            logger.error(f"批量分类失败，降级到规则匹配: {e}")
            return [self._classify_by_rules(msg) for msg in messages]
    
    def update_message_types(self, messages: List[Message]) -> List[Message]:
        """
        更新消息列表中每条消息的 message_type 字段
        
        使用 LLM 批量分类（同步版本）
        """
        if not messages:
            return messages
        
        types = self.classify_batch(messages)
        
        for msg, msg_type in zip(messages, types):
            msg.message_type = msg_type
        
        # 统计分类结果
        type_counts = {}
        for t in types:
            type_counts[t.value] = type_counts.get(t.value, 0) + 1
        logger.info(f"📊 分类结果: {type_counts}")
        
        return messages
    
    async def update_message_types_async(self, messages: List[Message]) -> List[Message]:
        """
        异步更新消息列表中每条消息的 message_type 字段
        
        使用 LLM 批量分类（异步版本，不阻塞主线程）
        """
        if not messages:
            return messages
        
        types = await self.classify_batch_async(messages)
        
        for msg, msg_type in zip(messages, types):
            msg.message_type = msg_type
        
        # 统计分类结果
        type_counts = {}
        for t in types:
            type_counts[t.value] = type_counts.get(t.value, 0) + 1
        logger.info(f"📊 分类结果: {type_counts}")
        
        return messages

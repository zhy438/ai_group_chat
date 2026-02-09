#!/usr/bin/env python3
"""
上下文压缩测试脚本

用于验证压缩功能是否正常工作
"""

import requests
import time

BASE_URL = "http://localhost:8000/api/v1"


def get_groups():
    """获取群聊列表"""
    resp = requests.get(f"{BASE_URL}/groups")
    return resp.json()


def get_context_stats(group_id: str):
    """获取上下文统计"""
    resp = requests.get(f"{BASE_URL}/groups/{group_id}/context/stats")
    return resp.json()


def force_compress(group_id: str):
    """强制执行压缩"""
    resp = requests.post(f"{BASE_URL}/groups/{group_id}/context/compress")
    return resp.json()


def set_threshold(group_id: str, ratio: float):
    """设置压缩阈值"""
    resp = requests.put(f"{BASE_URL}/groups/{group_id}/context/threshold?ratio={ratio}")
    return resp.json()


def send_message(group_id: str, content: str):
    """发送消息触发对话"""
    resp = requests.post(
        f"{BASE_URL}/groups/{group_id}/chat/stream",
        json={"content": content, "user_name": "测试用户", "max_rounds": 2}
    )
    return resp.status_code == 200


def main():
    print("=" * 60)
    print("🧪 上下文压缩功能测试")
    print("=" * 60)
    
    # 1. 获取群聊
    groups = get_groups()
    if not groups:
        print("❌ 没有找到群聊，请先创建一个群聊")
        return
    
    group = groups[0]
    group_id = group["id"]
    print(f"\n📍 使用群聊: {group['name']} ({group_id})")
    
    # 2. 查看当前状态
    print("\n" + "-" * 40)
    print("📊 当前上下文状态:")
    stats = get_context_stats(group_id)
    print(f"   消息数量: {stats['message_count']}")
    print(f"   Token 数量: {stats['current_tokens']}")
    print(f"   最大 Token: {stats['max_tokens']}")
    print(f"   使用率: {stats['usage_ratio']*100:.1f}%")
    print(f"   触发阈值: {stats['compression_config']['threshold_ratio']*100:.0f}%")
    print(f"   消息类型分布: {stats.get('type_distribution', {})}")
    
    # 3. 强制执行压缩测试
    print("\n" + "-" * 40)
    print("🔄 强制执行压缩...")
    
    if stats['message_count'] < 3:
        print("   ⚠️ 消息太少，无法测试压缩效果")
        print("   💡 请先在群聊中进行一些对话，然后重新运行此脚本")
        return
    
    result = force_compress(group_id)
    print(f"   压缩前: {result['before']['message_count']} 条消息, {result['before']['tokens']} tokens")
    print(f"   压缩后: {result['after']['message_count']} 条消息, {result['after']['tokens']} tokens")
    print(f"   节省: {result['saved']['messages']} 条消息, {result['saved']['tokens']} tokens ({result['saved']['ratio']})")
    
    # 4. 测试阈值调整
    print("\n" + "-" * 40)
    print("⚙️ 测试阈值调整...")
    
    # 临时设置很低的阈值
    threshold_result = set_threshold(group_id, 0.1)
    print(f"   {threshold_result['message']}")
    
    # 恢复默认阈值
    threshold_result = set_threshold(group_id, 0.8)
    print(f"   恢复: {threshold_result['message']}")
    
    print("\n" + "=" * 60)
    print("✅ 测试完成！")
    print("=" * 60)
    
    print("""
💡 接下来你可以：

1. 通过 API 查看状态:
   curl http://localhost:8000/api/v1/groups/{group_id}/context/stats

2. 强制压缩:
   curl -X POST http://localhost:8000/api/v1/groups/{group_id}/context/compress

3. 调低阈值触发自动压缩:
   curl -X PUT "http://localhost:8000/api/v1/groups/{group_id}/context/threshold?ratio=0.1"

4. 查看后端日志观察压缩过程:
   tail -f backend.log | grep -E "(压缩|摘要|分类)"
""")


if __name__ == "__main__":
    main()

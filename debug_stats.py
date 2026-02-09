
import requests
import json

BASE_URL = "http://localhost:8000/api/v1"

def debug_context_stats():
    # 1. 获取群组
    try:
        resp = requests.get(f"{BASE_URL}/groups")
        groups = resp.json()
        if not groups:
            print("❌ 没有找到任何群组")
            return
            
        group_id = groups[0]['id']
        group_name = groups[0]['name']
        print(f"🔍 检查群组: {group_name} (ID: {group_id})")
        
        # 2. 获取 Context Stats
        stats_url = f"{BASE_URL}/groups/{group_id}/context/stats"
        print(f"👉 请求: {stats_url}")
        
        stats_resp = requests.get(stats_url)
        if stats_resp.status_code != 200:
            print(f"❌ 请求失败: {stats_resp.text}")
            return
            
        data = stats_resp.json()
        print("\n📊 统计数据:")
        print(json.dumps(data, indent=2, ensure_ascii=False))
        
        if data['current_tokens'] == 0:
            print("\n⚠️ 警告: current_tokens 为 0！")
        else:
            print(f"\n✅ current_tokens = {data['current_tokens']}")
            
    except Exception as e:
        print(f"❌ 发生错误: {e}")

if __name__ == "__main__":
    debug_context_stats()

"""预设测试数据 - 开发调试用"""

# 预设群聊配置
PRESET_GROUPS = [
    {
        "name": "AI测试群",
        "manager_model": "gpt-4o-mini",
        "members": [
            {
                "name": "mimo-v2-flash-free",
                "model_id": "mimo-v2-flash-free",
                "description": "活泼开朗的程序员，擅长Python和Web开发",
                "thinking": False,
                "temperature": 0.7,
            },
            {
                "name": "qwen-flash",
                "model_id": "qwen-flash",
                "description": "稳重的技术专家，知识面广，善于分析问题",
                "thinking": False,
                "temperature": 0.7,
            },
            {
                "name": "deepseek-chat",
                "model_id": "deepseek-chat",
                "description": "专注于深度思考，擅长复杂问题的推理和解答",
                "thinking": False,
                "temperature": 0.7,
            },
        ],
    },
]

# AI 群聊项目 🤖💬

> 让多个AI大模型在群聊中协作讨论，解决你的问题！

## 项目简介

你可以在一个群聊中添加多个不同的大模型作为成员，这些大模型旨在解决你提出的问题：

- 🗣️ **自由讨论**: 提出一个想法，让AI们自行讨论，最后给你一个结果
- 👑 **主导模式**: 让某个成员率先提出观点，让其他成员以它的方向为基准进行讨论
- 📋 **任务分配**: 给不同的模型分配不同的任务（gemini负责多模态理解、gpt负责总结等）
- 🏃 **抢答模式**: 一个模型率先提出方案，后续模型附和或质疑

## 技术栈

- **后端框架**: FastAPI
- **包管理**: uv
- **AI编排**: LangGraph
- **模型调用**: LiteLLM + aihubmix
- **数据库**: PostgreSQL (with pgvector)

## 项目结构

```
ai_group_chat/
├── docker-compose.yml          # Docker配置（PostgreSQL）
├── pyproject.toml              # 项目配置
├── .env.example                # 环境变量示例
└── src/
    └── ai_group_chat/
        ├── __init__.py
        ├── main.py             # FastAPI 应用入口
        ├── config.py           # 配置管理
        ├── api/
        │   ├── __init__.py
        │   └── routes.py       # API 路由定义
        ├── models/
        │   ├── __init__.py
        │   └── schemas.py      # Pydantic 数据模型
        ├── services/
        │   ├── __init__.py
        │   └── chat_service.py # 业务逻辑层
        ├── llm/
        │   ├── __init__.py
        │   └── client.py       # LiteLLM 客户端封装
        └── graph/
            ├── __init__.py
            ├── state.py        # LangGraph 状态定义
            ├── nodes.py        # LangGraph 节点定义
            └── builder.py      # LangGraph 图构建器
```

## 快速开始

### 1. 环境配置

```bash
# 复制环境变量配置
cp .env.example .env

# 编辑 .env 文件，填入你的 API Key
```

### 2. 启动数据库

```bash
docker-compose up -d
```

### 3. 安装依赖

```bash
uv sync
```

### 4. 启动服务

```bash
uv run uvicorn ai_group_chat.main:app --reload --port 8000
```

### 5. 访问 API 文档

打开浏览器访问: http://localhost:8000/docs

## API 接口

### 群聊管理
- `POST /api/v1/groups` - 创建群聊
- `GET /api/v1/groups` - 获取群聊列表
- `GET /api/v1/groups/{id}` - 获取群聊详情
- `DELETE /api/v1/groups/{id}` - 删除群聊

### 成员管理
- `POST /api/v1/groups/{id}/members` - 添加AI成员
- `DELETE /api/v1/groups/{id}/members/{mid}` - 移除AI成员
- `PATCH /api/v1/groups/{id}/members/{mid}/task` - 更新成员任务

### 讨论功能
- `POST /api/v1/groups/{id}/discuss` - 启动讨论
- `POST /api/v1/groups/{id}/discuss/stream` - 流式讨论 (SSE)
- `GET /api/v1/groups/{id}/messages` - 获取消息历史

### 模型能力
- `GET /api/v1/models` - 获取可用模型列表

## 讨论模式

| 模式 | 说明 |
|------|------|
| `free` | 自由讨论：所有模型自行讨论 |
| `leader` | 主导模式：指定模型主导方向 |
| `task` | 任务分配：给每个模型分配特定任务 |
| `race` | 抢答模式：先答为主，后续附和或质疑 |

## 后续规划

- [ ] 数据库持久化 (PostgreSQL + SQLAlchemy)
- [ ] 用户认证系统
- [ ] 向量检索 (pgvector)
- [ ] WebSocket 实时通信
- [ ] 前端界面 (Vue3)

## License

MIT

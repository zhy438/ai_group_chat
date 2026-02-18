# AI Group Chat

多模型协作群聊后端服务，支持流式讨论、并发一问一答、总结、上下文压缩与实时 token 状态推送。

## 功能特性
- 群聊管理：创建/删除群聊，管理成员与管理员模型参数
- 讨论模式：
  - `free`：多模型自由讨论
  - `qa`：一问一答（成员并发回答）
- 流式输出：SSE 实时返回消息
- 上下文管理：长对话压缩、快照、增量加载
- 实时状态：SSE 推送 `stats` 事件，前端可按消息更新 token 使用量

## 技术栈
- FastAPI
- AutoGen AgentChat
- PostgreSQL (Docker)
- uv / Python 3.11+

## 项目结构
```text
ai_group_chat/
├── config/models.yaml
├── docker-compose.yml
├── docs/
├── src/ai_group_chat/
│   ├── api/routes.py
│   ├── agents/group_chat.py
│   ├── dao/
│   ├── memory/
│   ├── models/schemas.py
│   └── services/chat_service.py
├── start.sh
└── pyproject.toml
```

## 快速开始
1. 安装依赖
```bash
uv sync
```

2. 配置环境变量
```bash
cp .env.example .env
```
编辑 `.env` 填写你的 `AI_API_KEY`。

3. 启动服务（推荐）
```bash
./start.sh
```
该脚本会自动：
- 启动 PostgreSQL 容器
- 等待数据库就绪
- 启动后端（8000）与调试前端（8001）

4. 访问
- API 文档：`http://localhost:8000/docs`
- 健康检查：`http://localhost:8000/health`

## 关键接口
- 群聊：
  - `POST /api/v1/groups`
  - `GET /api/v1/groups`
  - `GET /api/v1/groups/{group_id}`
  - `DELETE /api/v1/groups/{group_id}`
- 成员：
  - `POST /api/v1/groups/{group_id}/members`
  - `PATCH /api/v1/groups/{group_id}/members/{member_id}`
  - `DELETE /api/v1/groups/{group_id}/members/{member_id}`
- 讨论：
  - `POST /api/v1/groups/{group_id}/discuss/stream`（SSE）
  - `POST /api/v1/groups/{group_id}/summarize`（SSE）
- 上下文：
  - `GET /api/v1/groups/{group_id}/context/stats`
  - `PUT /api/v1/groups/{group_id}/compression/threshold`
- 模型：
  - `GET /api/v1/models`
  - `POST /api/v1/models/reload`

## 安全说明
- 请勿提交 `.env`、日志、数据库文件到 GitHub（已在 `.gitignore` 处理）。
- 如果历史上曾泄露过密钥，请立即在服务商后台轮换。

## License
MIT

#!/bin/bash

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}=== AI群聊启动脚本 ===${NC}"

# 清理旧进程
echo -e "${BLUE}🧹 清理旧进程 (端口 8000, 8001)...${NC}"
lsof -ti:8000 | xargs kill -9 2>/dev/null
lsof -ti:8001 | xargs kill -9 2>/dev/null

# 启动后端
echo -e "${GREEN}🚀 启动后端服务 (Port 8000)...${NC}"
# 检查是否安装了 uv
if ! command -v uv &> /dev/null; then
    echo -e "${RED}❌ 未检测到 uv 命令，请先安装 uv 或使用 pip 运行。${NC}"
    exit 1
fi

# 后台运行后端，日志输出到 backend.log
uv run uvicorn ai_group_chat.main:app --host 0.0.0.0 --port 8000 --reload > backend.log 2>&1 &
BACKEND_PID=$!

# 等待几秒确保后端启动
sleep 3

# 启动前端
echo -e "${GREEN}🎨 启动前端服务 (Port 8001)...${NC}"
cd test-ui
# 后台运行前端，日志输出到 frontend.log
python3 -m http.server 8001 > ../frontend.log 2>&1 &
FRONTEND_PID=$!
cd ..

echo -e "${GREEN}✅ 服务已启动！${NC}"
echo -e "后端 PID: ${BACKEND_PID}"
echo -e "前端 PID: ${FRONTEND_PID}"
echo -e "后端日志: tail -f backend.log"
echo -e "前端访问: ${BLUE}http://localhost:8001${NC}"

# 捕获退出信号以清理进程
cleanup() {
    echo -e "\n${RED}🛑 正在停止服务...${NC}"
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    
    # 强制清理端口占用（确保子进程被杀死）
    lsof -ti:8000 | xargs kill -9 2>/dev/null
    lsof -ti:8001 | xargs kill -9 2>/dev/null
    exit
}

trap cleanup INT

echo -e "${BLUE}按 Ctrl+C 停止服务${NC}"

# 保持脚本运行
wait

"""FastAPI 应用入口"""

# 🔑 必须在所有其他导入之前加载环境变量！
# 这样 LangSmith 追踪才能正确读取配置
from dotenv import load_dotenv
load_dotenv()

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .api import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时
    print("🚀 AI群聊后端启动中...")
    settings = get_settings()
    print(f"📝 Debug模式: {settings.debug}")
    print(f"🔗 数据库: {settings.database_url}")
    
    yield
    
    # 关闭时
    print("👋 AI群聊后端关闭")


def create_app() -> FastAPI:
    """创建 FastAPI 应用"""
    settings = get_settings()
    
    app = FastAPI(
        title="AI群聊 API",
        description="""
        ## AI群聊后端API
        
        让多个AI大模型在群聊中协作讨论，解决你的问题！
        
        ### 核心功能
        - 🗣️ **多模型协作**: 支持GPT、Claude、Gemini、DeepSeek等多种大模型
        - 💬 **群聊讨论**: AI成员可以自由讨论、互相补充、质疑
        - 🎯 **多种模式**: 自由讨论、主导模式、任务分配、抢答模式
        - 🖼️ **多模态支持**: 支持图片理解（需模型支持）
        - 📝 **自动总结**: 讨论结束后自动生成总结
        """,
        version="0.1.0",
        lifespan=lifespan,
    )
    
    # CORS 配置
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 生产环境应该限制
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # 注册路由
    app.include_router(router)
    
    # 健康检查
    @app.get("/health")
    async def health_check():
        return {"status": "healthy", "app": settings.app_name}
    
    return app


# 创建应用实例
app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("ai_group_chat.main:app", host="0.0.0.0", port=8000, reload=True)

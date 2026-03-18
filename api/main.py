"""
HI-Photonics API 服务

FastAPI 后端服务，提供工作流执行、模型训练、逆向设计等 API。
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from api.routers import workflow_router, models_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化
    print("=" * 60)
    print("HI-Photonics API Server Starting...")
    print("=" * 60)
    
    yield
    
    # 关闭时清理
    print("HI-Photonics API Server Shutting down...")


app = FastAPI(
    title="HI-Photonics API",
    description="""
    光子学逆向设计平台 API
    
    ## 功能模块
    
    * **工作流管理**: 创建、保存、执行可视化工作流
    * **模型训练**: TNN/MDN/CGAN/PINN/GNN/HiLab 模型训练
    * **逆向设计**: 基于目标性能生成设计参数
    * **仿真接口**: Meep FDTD / RCWA 仿真
    """,
    version="0.2.0",
    lifespan=lifespan
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 开发环境允许所有来源
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(workflow_router, prefix="/api")
app.include_router(models_router, prefix="/api")


@app.get("/")
async def root():
    """根路径"""
    return {
        "name": "HI-Photonics API",
        "version": "0.2.0",
        "docs": "/docs",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy"}


@app.get("/api/system/info")
async def system_info():
    """系统信息"""
    import torch
    
    return {
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

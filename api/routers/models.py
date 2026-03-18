"""
模型训练与推理 API 路由
"""

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import asyncio
import json

from api.models.schemas import (
    TrainingConfig, TrainingProgress, 
    InverseDesignRequest, InverseDesignResponse,
    ModelTypeEnum
)

router = APIRouter(prefix="/models", tags=["models"])


class ModelInfo(BaseModel):
    type: str
    name: str
    description: str
    icon: str
    params: List[Dict[str, Any]]


# 模型信息
MODEL_INFOS = {
    "tnn": ModelInfo(
        type="tnn",
        name="Tandem Network",
        description="级联正向-逆向网络",
        icon="Layers",
        params=[
            {"key": "hidden_dim", "label": "隐藏层维度", "type": "number", "default": 128},
            {"key": "num_layers", "label": "网络层数", "type": "number", "default": 4}
        ]
    ),
    "mdn": ModelInfo(
        type="mdn",
        name="Mixture Density Network",
        description="输出混合高斯分布，支持多解",
        icon="BubbleChart",
        params=[
            {"key": "num_components", "label": "高斯分量数", "type": "number", "default": 5},
            {"key": "hidden_dim", "label": "隐藏层维度", "type": "number", "default": 128}
        ]
    ),
    "cgan": ModelInfo(
        type="cgan",
        name="Conditional GAN",
        description="条件生成对抗网络",
        icon="Gesture",
        params=[
            {"key": "latent_dim", "label": "潜在维度", "type": "number", "default": 64},
            {"key": "noise_dim", "label": "噪声维度", "type": "number", "default": 32}
        ]
    ),
    "pinn": ModelInfo(
        type="pinn",
        name="Physics-Informed NN",
        description="融入物理约束的神经网络",
        icon="Science",
        params=[
            {"key": "physics_weight", "label": "物理损失权重", "type": "number", "default": 0.1},
            {"key": "hidden_dim", "label": "隐藏层维度", "type": "number", "default": 128}
        ]
    ),
    "gnn": ModelInfo(
        type="gnn",
        name="Graph Neural Network",
        description="处理图结构数据",
        icon="DeviceHub",
        params=[
            {"key": "hidden_dim", "label": "隐藏维度", "type": "number", "default": 64},
            {"key": "num_layers", "label": "GNN 层数", "type": "number", "default": 3}
        ]
    ),
    "hilab": ModelInfo(
        type="hilab",
        name="HiLab",
        description="VAE + 贝叶斯优化混合框架",
        icon="AutoAwesome",
        params=[
            {"key": "latent_dim", "label": "潜在维度", "type": "number", "default": 32},
            {"key": "acquisition", "label": "采集函数", "type": "select", "default": "ei"}
        ]
    )
}


@router.get("/")
async def list_models() -> List[ModelInfo]:
    """列出所有可用模型"""
    return list(MODEL_INFOS.values())


@router.get("/{model_type}")
async def get_model_info(model_type: str) -> ModelInfo:
    """获取模型详细信息"""
    if model_type not in MODEL_INFOS:
        raise HTTPException(status_code=404, detail="Model not found")
    return MODEL_INFOS[model_type]


# WebSocket 连接管理
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_progress(self, websocket: WebSocket, progress: TrainingProgress):
        await websocket.send_json(progress.dict())


manager = ConnectionManager()


@router.websocket("/train/{model_type}")
async def websocket_train(
    websocket: WebSocket,
    model_type: str,
    epochs: int = 100,
    batch_size: int = 32
):
    """WebSocket 训练端点，实时推送训练进度"""
    await manager.connect(websocket)
    
    try:
        # 模拟训练过程
        for epoch in range(1, epochs + 1):
            progress = TrainingProgress(
                epoch=epoch,
                total_epochs=epochs,
                loss=1.0 / epoch + 0.01 * (epoch % 10),
                val_loss=1.0 / epoch + 0.02 * (epoch % 10) if epoch % 5 == 0 else None,
                metrics={"accuracy": min(0.99, epoch / epochs)},
                status="running"
            )
            
            await manager.send_progress(websocket, progress)
            await asyncio.sleep(0.1)  # 模拟训练时间
        
        # 发送完成状态
        final_progress = TrainingProgress(
            epoch=epochs,
            total_epochs=epochs,
            loss=0.01,
            val_loss=0.015,
            metrics={"accuracy": 0.98},
            status="completed"
        )
        await manager.send_progress(websocket, final_progress)
        
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@router.post("/train")
async def train_model(config: TrainingConfig) -> Dict[str, Any]:
    """启动模型训练（非 WebSocket）"""
    import uuid
    
    training_id = str(uuid.uuid4())
    
    # 这里应该启动后台任务进行训练
    # 简化版：直接返回训练 ID
    
    return {
        "training_id": training_id,
        "model_type": config.model_type,
        "status": "started",
        "message": f"Training started with {config.epochs} epochs"
    }


@router.post("/inverse-design", response_model=InverseDesignResponse)
async def inverse_design(request: InverseDesignRequest) -> InverseDesignResponse:
    """执行逆向设计"""
    import numpy as np
    
    # 模拟逆向设计
    num_samples = request.num_samples
    design_shape = (100, 22)
    
    designs = []
    predicted_performances = []
    confidences = []
    
    for i in range(num_samples):
        # 生成模拟设计
        design = np.random.rand(*design_shape).astype(np.float32).tolist()
        designs.append(design)
        
        # 模拟预测性能（接近目标）
        noise = np.random.randn(3) * 0.05
        perf = [t + n for t, n in zip(request.target_performance, noise)]
        predicted_performances.append(perf)
        
        # 模拟置信度
        confidences.append(0.9 - 0.1 * np.random.rand())
    
    return InverseDesignResponse(
        designs=designs,
        predicted_performance=predicted_performances,
        confidence=confidences
    )


@router.get("/pretrained/list")
async def list_pretrained_models() -> List[Dict[str, Any]]:
    """列出预训练模型"""
    return [
        {
            "id": "hilab_grating_v1",
            "type": "hilab",
            "name": "HiLab 光栅耦合器 v1",
            "description": "在光栅耦合器数据集上预训练",
            "metrics": {"efficiency": 0.92, "bandwidth": 0.85}
        },
        {
            "id": "mdn_metagrating_v1",
            "type": "mdn",
            "name": "MDN 超光栅 v1",
            "description": "在超光栅数据集上预训练",
            "metrics": {"efficiency": 0.88}
        }
    ]

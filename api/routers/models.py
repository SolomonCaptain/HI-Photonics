"""
模型训练与推理 API 路由
"""

import sys
from pathlib import Path
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import asyncio
import json
import torch

# 添加项目根目录
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

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


class TrainingRequest(BaseModel):
    """训练请求"""
    model_type: str
    challenge_name: str = "grating_coupler"
    num_samples: int = 3000
    design_shape: List[int] = [100, 22]
    performance_dim: int = 3
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-3
    save_format: str = "safetensors"  # "torch" 或 "safetensors"
    model_name: Optional[str] = None


class TrainingResult(BaseModel):
    """训练结果"""
    training_id: str
    model_type: str
    status: str
    model_path: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class InverseDesignRequestAPI(BaseModel):
    """逆向设计 API 请求"""
    model_path: str
    target_performance: List[float]
    num_samples: int = 1
    model_type: str = "tnn"


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

# 训练任务存储
_training_tasks: Dict[str, TrainingResult] = {}


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


@router.post("/train-full", response_model=TrainingResult)
async def train_model_full(request: TrainingRequest, background_tasks: BackgroundTasks) -> TrainingResult:
    """
    完整的模型训练端点
    
    支持 Windows 平台优化和 safetensors 格式保存。
    """
    import uuid
    from datetime import datetime
    
    training_id = str(uuid.uuid4())
    
    # 创建训练结果
    result = TrainingResult(
        training_id=training_id,
        model_type=request.model_type,
        status="started"
    )
    _training_tasks[training_id] = result
    
    # 启动后台训练任务
    background_tasks.add_task(
        _run_training_task,
        training_id,
        request
    )
    
    return result


async def _run_training_task(training_id: str, request: TrainingRequest):
    """后台训练任务"""
    import platform
    import gc
    from datetime import datetime
    
    result = _training_tasks[training_id]
    
    try:
        # Windows 平台设置
        if platform.system() == 'Windows':
            torch.multiprocessing.set_sharing_strategy('file_system')
        
        # 导入必要的模块
        from data.loaders.pipeline import SyntheticDataset, create_dataloaders
        from models.inverse.tnn import TandemNetwork, TandemNetworkConfig
        
        # 创建数据集
        dataset = SyntheticDataset(
            num_samples=request.num_samples,
            design_shape=tuple(request.design_shape),
            performance_dim=request.performance_dim,
            noise_level=0.05
        )
        
        train_loader, val_loader, _ = create_dataloaders(
            dataset,
            batch_size=request.batch_size
        )
        
        # 创建模型
        config = TandemNetworkConfig(
            name=request.model_name or f"{request.model_type}_{training_id[:8]}",
            pretrain_forward=True,
            freeze_forward=True
        )
        model = TandemNetwork(config)
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        
        # 训练前向网络
        optimizer = torch.optim.AdamW(model.forward_net.parameters(), lr=request.learning_rate)
        
        result.status = "training_forward"
        best_val_loss = float('inf')
        
        for epoch in range(request.epochs):
            model.forward_net.train()
            train_loss = 0.0
            
            for batch in train_loader:
                design = batch['design'].to(device)
                performance = batch['performance'].to(device)
                
                optimizer.zero_grad()
                pred = model.forward_net(design)
                loss = torch.nn.functional.mse_loss(pred, performance)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # 验证
            model.forward_net.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    design = batch['design'].to(device)
                    performance = batch['performance'].to(device)
                    pred = model.forward_net(design)
                    val_loss += torch.nn.functional.mse_loss(pred, performance).item()
            
            val_loss /= len(val_loader)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            
            # 清理内存
            if platform.system() == 'Windows':
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        model.forward_trained = True
        
        # 冻结前向网络，训练逆向网络
        model.forward_net.freeze()
        optimizer = torch.optim.AdamW(model.inverse_net.parameters(), lr=request.learning_rate * 0.1)
        
        result.status = "training_inverse"
        
        for epoch in range(request.epochs):
            model.inverse_net.train()
            train_loss = 0.0
            
            for batch in train_loader:
                performance = batch['performance'].to(device)
                design_target = batch['design'].to(device)
                
                optimizer.zero_grad()
                design_pred = model.inverse_net(performance)
                performance_pred = model.forward_net(design_pred)
                loss = torch.nn.functional.mse_loss(performance_pred, performance)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            if platform.system() == 'Windows':
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        model.inverse_trained = True
        
        # 保存模型
        output_dir = Path("op_models/pretrained")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"{request.model_type}_{timestamp}"
        
        if request.save_format == "safetensors":
            # 保存为 safetensors 格式
            try:
                import safetensors.torch
                
                forward_path = output_dir / f"{model_name}_forward.safetensors"
                inverse_path = output_dir / f"{model_name}_inverse.safetensors"
                
                safetensors.torch.save_file(
                    model.forward_net.state_dict(),
                    str(forward_path)
                )
                safetensors.torch.save_file(
                    model.inverse_net.state_dict(),
                    str(inverse_path)
                )
                
                # 保存元数据
                metadata = {
                    'model_name': config.name,
                    'model_type': request.model_type,
                    'design_shape': request.design_shape,
                    'performance_dim': request.performance_dim,
                    'training_id': training_id,
                    'saved_at': datetime.now().isoformat(),
                    'platform': platform.system(),
                }
                
                metadata_path = output_dir / f"{model_name}_metadata.json"
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)
                
                model_path = str(forward_path.parent / model_name)
                
            except ImportError:
                # 回退到 PyTorch 格式
                model_path = str(output_dir / f"{model_name}.pth")
                model.save(model_path, format="torch")
        else:
            model_path = str(output_dir / f"{model_name}.pth")
            model.save(model_path, format="torch")
        
        # 更新结果
        result.status = "completed"
        result.model_path = model_path
        result.metrics = {
            'final_val_loss': best_val_loss,
            'epochs': request.epochs * 2,
            'parameters': model.count_parameters()
        }
        
    except Exception as e:
        result.status = "error"
        result.error = str(e)


@router.get("/train/{training_id}", response_model=TrainingResult)
async def get_training_status(training_id: str) -> TrainingResult:
    """获取训练状态"""
    if training_id not in _training_tasks:
        raise HTTPException(status_code=404, detail="Training task not found")
    return _training_tasks[training_id]


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


@router.post("/inverse-design-loaded")
async def inverse_design_with_model(request: InverseDesignRequestAPI) -> Dict[str, Any]:
    """
    使用已加载的模型执行逆向设计
    
    支持 safetensors 和 PyTorch 格式。
    """
    import numpy as np
    from pathlib import Path
    
    model_path = Path(request.model_path)
    
    if not model_path.exists():
        # 尝试添加前缀
        possible_paths = [
            Path("op_models/pretrained") / f"{request.model_path}_forward.safetensors",
            Path("op_models/pretrained") / f"{request.model_path}.pth",
            Path("op_models/custom") / f"{request.model_path}_forward.safetensors",
        ]
        for p in possible_paths:
            if p.exists():
                model_path = p
                break
        else:
            raise HTTPException(status_code=404, detail=f"Model not found: {request.model_path}")
    
    try:
        # 根据文件类型加载模型
        if model_path.suffix == '.safetensors':
            import safetensors.torch
            from models.inverse.tnn import TandemNetwork, TandemNetworkConfig
            
            # 加载元数据
            metadata_path = model_path.parent / model_path.name.replace("_forward.safetensors", "_metadata.json")
            if metadata_path.exists():
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                config = TandemNetworkConfig(name=metadata.get('model_name', 'loaded'))
            else:
                config = TandemNetworkConfig(name="loaded")
            
            model = TandemNetwork(config)
            
            # 加载权重
            forward_path = model_path
            inverse_path = model_path.parent / model_path.name.replace("_forward.safetensors", "_inverse.safetensors")
            
            if forward_path.exists() and inverse_path.exists():
                forward_state = safetensors.torch.load_file(str(forward_path))
                inverse_state = safetensors.torch.load_file(str(inverse_path))
                
                model.forward_net.load_state_dict(forward_state)
                model.inverse_net.load_state_dict(inverse_state)
                model.forward_trained = True
                model.inverse_trained = True
            else:
                raise HTTPException(status_code=404, detail="Model weights incomplete")
        else:
            # 加载 PyTorch 格式
            from models.inverse.tnn import TandemNetwork, TandemNetworkConfig
            config = TandemNetworkConfig(name="loaded")
            model = TandemNetwork(config)
            model.load(model_path)
        
        # 执行逆向设计
        model.eval()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        
        target = torch.tensor([request.target_performance], dtype=torch.float32, device=device)
        
        with torch.no_grad():
            design = model.inverse_net(target)
            pred_performance = model.forward_net(design)
        
        return {
            'design': design[0].cpu().numpy().tolist(),
            'predicted_performance': pred_performance[0].cpu().numpy().tolist(),
            'target_performance': request.target_performance,
            'model_path': str(model_path)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inverse design failed: {str(e)}")


@router.get("/pretrained/list")
async def list_pretrained_models() -> List[Dict[str, Any]]:
    """列出预训练模型"""
    import glob
    
    models = []
    
    # 扫描 op_models 目录
    pretrained_dir = Path("op_models/pretrained")
    custom_dir = Path("op_models/custom")
    
    for directory in [pretrained_dir, custom_dir]:
        if not directory.exists():
            continue
        
        # 查找 safetensors 文件
        for sf_file in directory.glob("*_forward.safetensors"):
            model_name = sf_file.name.replace("_forward.safetensors", "")
            metadata_file = sf_file.parent / f"{model_name}_metadata.json"
            
            model_info = {
                "id": model_name,
                "path": str(sf_file.parent / model_name),
                "format": "safetensors",
                "directory": str(directory)
            }
            
            if metadata_file.exists():
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                model_info.update({
                    "model_type": metadata.get("model_type", "unknown"),
                    "name": metadata.get("model_name", model_name),
                    "saved_at": metadata.get("saved_at"),
                    "platform": metadata.get("platform"),
                    "design_shape": metadata.get("design_shape"),
                    "performance_dim": metadata.get("performance_dim")
                })
            
            models.append(model_info)
        
        # 查找 PyTorch 文件
        for pt_file in directory.glob("*.pth"):
            model_name = pt_file.stem
            if model_name.endswith("_forward") or model_name.endswith("_inverse"):
                continue
            
            models.append({
                "id": model_name,
                "path": str(pt_file),
                "format": "torch",
                "directory": str(directory)
            })
    
    return models

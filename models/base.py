"""
神经网络模型基类

为所有深度学习模型提供统一的接口和功能。
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List, Union, Literal
from pathlib import Path
from datetime import datetime
import json
import torch
import torch.nn as nn
from dataclasses import dataclass

# safetensors 支持
try:
    import safetensors.torch
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False


@dataclass
class ModelConfig:
    """模型配置基类"""
    name: str = "base_model"
    device: str = "auto"  # 'auto', 'cuda', 'cpu'
    
    def get_device(self) -> torch.device:
        """获取计算设备"""
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)


class BaseModel(nn.Module, ABC):
    """
    所有神经网络模型的基类
    
    提供统一的接口：
    - 前向传播
    - 损失计算
    - 评估指标
    - 模型保存/加载
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__()
        self.config = config or ModelConfig()
        self.device = self.config.get_device()
        self.to(self.device)
        
        # 训练状态跟踪
        self.training_history: Dict[str, List[float]] = {
            'train_loss': [],
            'val_loss': [],
            'metrics': []
        }
    
    @abstractmethod
    def forward(self, *args, **kwargs) -> torch.Tensor:
        """
        前向传播
        
        Returns:
            模型输出
        """
        pass
    
    def compute_loss(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        计算损失
        
        Args:
            output: 模型输出
            target: 目标值
            
        Returns:
            损失值
        """
        # 默认使用 MSE 损失
        return nn.functional.mse_loss(output, target)
    
    def compute_metrics(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        **kwargs
    ) -> Dict[str, float]:
        """
        计算评估指标
        
        Args:
            output: 模型输出
            target: 目标值
            
        Returns:
            指标字典
        """
        with torch.no_grad():
            mse = nn.functional.mse_loss(output, target).item()
            mae = nn.functional.l1_loss(output, target).item()
            
            # 计算 R²
            ss_res = ((target - output) ** 2).sum()
            ss_tot = ((target - target.mean()) ** 2).sum()
            r2 = 1 - ss_res / (ss_tot + 1e-8)
            
            return {
                'mse': mse,
                'mae': mae,
                'r2': r2.item()
            }
    
    def save(
        self, 
        path: Union[str, Path], 
        format: Literal["torch", "safetensors"] = "torch",
        include_optimizer: bool = False, 
        optimizer: Optional[torch.optim.Optimizer] = None
    ) -> None:
        """
        保存模型
        
        Args:
            path: 保存路径
            format: 保存格式，'torch' 或 'safetensors'
            include_optimizer: 是否保存优化器状态
            optimizer: 优化器实例
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # 准备模型数据
        state_dict = self.state_dict()
        config_dict = self.config.__dict__ if hasattr(self.config, '__dict__') else {}
        
        if format == "safetensors":
            if not SAFETENSORS_AVAILABLE:
                raise ImportError("safetensors 库未安装，请运行: pip install safetensors")
            
            # 确保路径后缀正确
            if path.suffix not in ['.safetensors', '.sft']:
                path = path.with_suffix('.safetensors')
            
            # 保存模型权重为 safetensors 格式
            safetensors.torch.save_file(state_dict, str(path))
            
            # 保存元数据为 JSON
            metadata = {
                'model_name': self.config.name,
                'config': config_dict,
                'training_history': self.training_history,
                'saved_at': datetime.now().isoformat(),
                'pytorch_version': torch.__version__,
            }
            
            if include_optimizer and optimizer is not None:
                # 注意：优化器状态不适合 safetensors，单独保存
                optimizer_path = path.with_suffix('.optimizer.pt')
                torch.save(optimizer.state_dict(), optimizer_path)
                metadata['optimizer_saved'] = str(optimizer_path)
            
            metadata_path = path.with_suffix('.json')
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
        else:
            # 默认 PyTorch 格式
            checkpoint = {
                'model_state_dict': state_dict,
                'config': config_dict,
                'training_history': self.training_history
            }
            
            if include_optimizer and optimizer is not None:
                checkpoint['optimizer_state_dict'] = optimizer.state_dict()
            
            torch.save(checkpoint, path)
    
    def load(
        self, 
        path: Union[str, Path], 
        format: Literal["torch", "safetensors", "auto"] = "auto",
        optimizer: Optional[torch.optim.Optimizer] = None,
        strict: bool = True
    ) -> Dict[str, Any]:
        """
        加载模型
        
        Args:
            path: 模型路径
            format: 加载格式，'torch', 'safetensors' 或 'auto'（自动检测）
            optimizer: 优化器实例（用于恢复训练）
            strict: 是否严格匹配参数
            
        Returns:
            加载的元数据字典
        """
        path = Path(path)
        metadata = {}
        
        # 自动检测格式
        if format == "auto":
            if path.suffix in ['.safetensors', '.sft']:
                format = "safetensors"
            else:
                format = "torch"
        
        if format == "safetensors":
            if not SAFETENSORS_AVAILABLE:
                raise ImportError("safetensors 库未安装，请运行: pip install safetensors")
            
            # 加载 safetensors 权重
            state_dict = safetensors.torch.load_file(str(path))
            self.load_state_dict(state_dict, strict=strict)
            
            # 加载元数据
            metadata_path = path.with_suffix('.json')
            if metadata_path.exists():
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                # 恢复训练历史
                if 'training_history' in metadata:
                    self.training_history = metadata['training_history']
                
                # 加载优化器状态
                if optimizer is not None and 'optimizer_saved' in metadata:
                    optimizer_path = Path(metadata['optimizer_saved'])
                    if optimizer_path.exists():
                        optimizer.load_state_dict(torch.load(optimizer_path, map_location=self.device))
        else:
            # 加载 PyTorch 格式
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            
            self.load_state_dict(checkpoint['model_state_dict'], strict=strict)
            
            if 'training_history' in checkpoint:
                self.training_history = checkpoint['training_history']
                metadata['training_history'] = checkpoint['training_history']
            
            if optimizer is not None and 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if 'config' in checkpoint:
                metadata['config'] = checkpoint['config']
        
        return metadata
    
    def count_parameters(self) -> int:
        """计算可训练参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def freeze(self) -> None:
        """冻结所有参数"""
        for param in self.parameters():
            param.requires_grad = False
    
    def unfreeze(self) -> None:
        """解冻所有参数"""
        for param in self.parameters():
            param.requires_grad = True
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            'name': self.config.name,
            'device': str(self.device),
            'parameters': self.count_parameters(),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }


class SurrogateModel(BaseModel):
    """
    代理模型基类
    
    用于替代昂贵的仿真计算，快速预测性能。
    输入：设计参数
    输出：性能指标
    """
    
    @abstractmethod
    def forward(self, design: torch.Tensor) -> torch.Tensor:
        """
        从设计参数预测性能
        
        Args:
            design: 设计参数张量 [B, ...] 或 [H, W]
            
        Returns:
            性能指标 [B, performance_dim]
        """
        pass


class InverseModel(BaseModel):
    """
    逆向设计模型基类
    
    从性能目标生成设计参数。
    输入：性能指标
    输出：设计参数
    """
    
    @abstractmethod
    def forward(self, performance: torch.Tensor) -> torch.Tensor:
        """
        从性能指标生成设计
        
        Args:
            performance: 性能指标 [B, performance_dim]
            
        Returns:
            设计参数 [B, ...]
        """
        pass


class GenerativeModel(BaseModel):
    """
    生成模型基类
    
    支持条件生成和多样性控制。
    """
    
    @abstractmethod
    def forward(self, condition: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        条件生成
        
        Args:
            condition: 条件信息（如性能目标）
            noise: 随机噪声（可选）
            
        Returns:
            生成的数据
        """
        pass
    
    def sample(self, condition: torch.Tensor, num_samples: int = 1) -> torch.Tensor:
        """
        从条件采样多个结果
        
        Args:
            condition: 条件信息
            num_samples: 采样数量
            
        Returns:
            生成的样本列表
        """
        samples = []
        for _ in range(num_samples):
            sample = self.forward(condition)
            samples.append(sample)
        return torch.stack(samples)

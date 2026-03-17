"""
神经网络模型基类

为所有深度学习模型提供统一的接口和功能。
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List, Union
from pathlib import Path
import torch
import torch.nn as nn
from dataclasses import dataclass


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
    
    def save(self, path: Union[str, Path], include_optimizer: bool = False, 
             optimizer: Optional[torch.optim.Optimizer] = None) -> None:
        """
        保存模型
        
        Args:
            path: 保存路径
            include_optimizer: 是否保存优化器状态
            optimizer: 优化器实例
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': self.config.__dict__ if hasattr(self.config, '__dict__') else {},
            'training_history': self.training_history
        }
        
        if include_optimizer and optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        
        torch.save(checkpoint, path)
    
    def load(self, path: Union[str, Path], optimizer: Optional[torch.optim.Optimizer] = None,
             strict: bool = True) -> None:
        """
        加载模型
        
        Args:
            path: 模型路径
            optimizer: 优化器实例（用于恢复训练）
            strict: 是否严格匹配参数
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.load_state_dict(checkpoint['model_state_dict'], strict=strict)
        
        if 'training_history' in checkpoint:
            self.training_history = checkpoint['training_history']
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
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

"""
设计挑战基类模块

定义光子学逆向设计问题的标准接口。
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, List, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
import torch
import numpy as np
from pathlib import Path

from interfaces.simulators.base import (
    SimulatorInterface, SimulationConfig, DesignRegion,
    SimulationResult
)


@dataclass
class DesignSpec:
    """设计规格"""
    # 尺寸约束
    design_size: Tuple[float, ...]  # 微米
    resolution: int = 50  # 像素/微米
    
    # 材料约束
    min_eps: float = 1.0
    max_eps: float = 12.0
    
    # 制造约束
    min_feature_size: float = 0.1  # 微米
    min_gap_size: float = 0.1  # 微米
    
    # 优化约束
    max_volume_fraction: float = 1.0
    symmetry: Optional[str] = None  # 'horizontal', 'vertical', 'quad', None
    
    # 波长
    wavelengths: List[float] = field(default_factory=lambda: [1.55])
    
    def get_grid_shape(self) -> Tuple[int, ...]:
        """获取网格形状"""
        return tuple(int(s * self.resolution) for s in self.design_size)


@dataclass
class PerformanceTarget:
    """性能目标"""
    # 目标指标
    metrics: Dict[str, float] = field(default_factory=dict)
    
    # 权重
    weights: Dict[str, float] = field(default_factory=dict)
    
    # 约束（必须满足）
    constraints: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    
    def get_target(self, name: str) -> Optional[float]:
        """获取目标值"""
        return self.metrics.get(name)
    
    def get_weight(self, name: str) -> float:
        """获取权重"""
        return self.weights.get(name, 1.0)
    
    def check_constraints(self, metrics: Dict[str, float]) -> Tuple[bool, Dict[str, float]]:
        """检查约束是否满足"""
        violations = {}
        satisfied = True
        
        for name, (min_val, max_val) in self.constraints.items():
            value = metrics.get(name, 0.0)
            if value < min_val:
                violations[name] = min_val - value
                satisfied = False
            elif value > max_val:
                violations[name] = value - max_val
                satisfied = False
        
        return satisfied, violations


class DesignChallenge(ABC):
    """
    设计挑战基类
    
    定义一个完整的逆向设计问题。
    """
    
    def __init__(
        self,
        name: str,
        spec: DesignSpec,
        target: PerformanceTarget,
        device: torch.device = None
    ):
        """
        初始化设计挑战
        
        Args:
            name: 挑战名称
            spec: 设计规格
            target: 性能目标
            device: 计算设备
        """
        self.name = name
        self.spec = spec
        self.target = target
        self.device = device or torch.device('cpu')
        
        # 仿真器
        self.simulator: Optional[SimulatorInterface] = None
        
        # 当前设计
        self.current_design: Optional[torch.Tensor] = None
        
        # 历史记录
        self.history: List[Dict[str, Any]] = []
    
    @abstractmethod
    def setup_simulator(self) -> SimulatorInterface:
        """
        设置仿真器
        
        Returns:
            配置好的仿真器实例
        """
        pass
    
    @abstractmethod
    def compute_objective(
        self,
        result: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        计算目标函数值
        
        Args:
            result: 仿真结果
            
        Returns:
            标量目标值（越小越好）
        """
        pass
    
    @abstractmethod
    def get_initial_design(self) -> torch.Tensor:
        """
        获取初始设计
        
        Returns:
            初始设计参数
        """
        pass
    
    def evaluate(
        self,
        design_params: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        评估设计
        
        Args:
            design_params: 设计参数
            
        Returns:
            (目标值, 信息字典)
        """
        if self.simulator is None:
            self.simulator = self.setup_simulator()
        
        # 运行仿真
        result = self.simulator.run(design_params, **kwargs)
        
        # 计算目标
        objective = self.compute_objective(result)
        
        # 计算性能指标
        metrics = self._compute_metrics(result)
        
        # 记录历史
        info = {
            'objective': objective.item(),
            'metrics': metrics,
            'design_params': design_params.detach().clone()
        }
        self.history.append(info)
        
        return objective, info
    
    def _compute_metrics(
        self,
        result: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """计算性能指标"""
        metrics = {}
        
        # 透射率
        if 'transmission' in result:
            metrics['transmission'] = result['transmission'].mean().item()
        
        # 反射率
        if 'reflection' in result:
            metrics['reflection'] = result['reflection'].mean().item()
        
        # 体积分数
        if self.current_design is not None:
            metrics['volume_fraction'] = self.current_design.mean().item()
        
        return metrics
    
    def check_constraints(
        self,
        design_params: torch.Tensor
    ) -> Tuple[bool, Dict[str, float]]:
        """检查设计约束"""
        violations = {}
        
        # 检查值范围
        if design_params.min() < 0 or design_params.max() > 1:
            violations['value_range'] = max(
                abs(design_params.min().item()),
                abs(design_params.max().item() - 1)
            )
        
        # 检查对称性
        if self.spec.symmetry == 'horizontal':
            asymmetry = (design_params - torch.flip(design_params, [-1])).abs().mean().item()
            if asymmetry > 0.01:
                violations['symmetry'] = asymmetry
        
        # 检查体积分数
        volume = design_params.mean().item()
        if volume > self.spec.max_volume_fraction:
            violations['volume'] = volume - self.spec.max_volume_fraction
        
        return len(violations) == 0, violations
    
    def get_design_region(self) -> DesignRegion:
        """获取设计区域"""
        return DesignRegion(
            name='design',
            center=(0, 0, 0),
            size=tuple(s for s in self.spec.design_size if s > 0),
            min_permittivity=self.spec.min_eps,
            max_permittivity=self.spec.max_eps
        )
    
    def visualize(
        self,
        design_params: Optional[torch.Tensor] = None,
        save_path: Optional[str] = None
    ):
        """可视化设计"""
        import matplotlib.pyplot as plt
        
        if design_params is None:
            design_params = self.current_design
        
        if design_params is None:
            raise ValueError("没有可用的设计参数")
        
        design_np = design_params.detach().cpu().numpy()
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 设计参数
        ax = axes[0]
        im = ax.imshow(design_np.T, cmap='viridis', origin='lower', vmin=0, vmax=1)
        ax.set_title(f'{self.name} - Design')
        ax.set_xlabel('x (pixels)')
        ax.set_ylabel('y (pixels)')
        plt.colorbar(im, ax=ax, label='Density')
        
        # 性能历史
        ax = axes[1]
        if self.history:
            objectives = [h['objective'] for h in self.history]
            ax.plot(objectives, 'b-', linewidth=2)
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Objective')
            ax.set_title('Optimization History')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No history', ha='center', va='center', transform=ax.transAxes)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def save(self, path: str):
        """保存挑战状态"""
        torch.save({
            'name': self.name,
            'spec': self.spec,
            'target': self.target,
            'current_design': self.current_design,
            'history': self.history
        }, path)
    
    def load(self, path: str):
        """加载挑战状态"""
        data = torch.load(path)
        self.name = data['name']
        self.spec = data['spec']
        self.target = data['target']
        self.current_design = data['current_design']
        self.history = data['history']


class ChallengeFactory:
    """挑战工厂"""
    
    _challenges: Dict[str, type] = {}
    
    @classmethod
    def register(cls, name: str, challenge_class: type):
        """注册挑战"""
        cls._challenges[name] = challenge_class
    
    @classmethod
    def create(
        cls,
        name: str,
        **kwargs
    ) -> DesignChallenge:
        """创建挑战实例"""
        if name not in cls._challenges:
            raise ValueError(f"未知的挑战类型: {name}，可用: {list(cls._challenges.keys())}")
        return cls._challenges[name](**kwargs)
    
    @classmethod
    def list_available(cls) -> List[str]:
        """列出可用的挑战"""
        return list(cls._challenges.keys())


def register_challenge(name: str):
    """挑战注册装饰器"""
    def decorator(cls):
        ChallengeFactory.register(name, cls)
        return cls
    return decorator

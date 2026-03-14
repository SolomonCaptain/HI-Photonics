from abc import ABC, abstractmethod
from typing import Any, Dict
import torch

class SimulatorInterface(ABC):
    """仿真器接口，提供前向计算和梯度计算"""
    @abstractmethod
    def run(self, design_params: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """
        运行仿真，返回包含所有感兴趣物理量的字典（如场分布、投射率）
        design_params: 设计参数张量，应与仿真器所需格式一致。
        """
        pass

    @abstractmethod
    def compute_gradient(self, design_params: torch.Tensor, objective_grad: Dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        """
        根据目标函数对输出物理量的梯度，计算设计参数的梯度（伴随法）
        objective_grad: 目标函数对各输出物理量的梯度字典
        返回设计参数的梯度
        """
        pass
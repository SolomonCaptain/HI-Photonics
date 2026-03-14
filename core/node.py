from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import torch
from .utils.typing import TensorLike, Params

class Node(ABC):
    """所有计算节点的基类"""
    def __init__(self, name: str, params: Optional[Params] = None):
        self.name = name
        self.params = params or {}
        self._inputs: List['Node'] = [] # 输入节点列表
        self._outputs: List['Node'] = [] # 输出节点列表
        self._cached_output: Optional[TensorLike] = None # 缓存输出

    def add_input(self, node: 'Node'):
        """添加输入节点，自动构建计算图连接"""
        self._inputs.append(node)
        node._outputs.append(self) # 反向连接

    @abstractmethod
    def forward(self, **kwargs) -> TensorLike:
        """
        执行节点的前向计算
        参数 kwargs 可用于传递额外的运行时信息（如仿真精度）
        返回计算结果（torch.Tensor 或字典）
        """
        pass

    def backward(self, grad_output: torch.Tensor):
        """
        反向传播梯度
        默认实现调用 torch.autograd.backward
        若节点内部有需要手动处理的梯度，可重写此方法
        """
        if isinstance(self._cached_output, torch.Tensor):
            torch.autograd.backward(self._cached_output, grad_output)
        # 如果输出是字典，可能需要更复杂的处理，暂略

    def clear_cache(self):
        """清理前向缓存，用于重新计算"""
        self._cached_output = None
import torch
from ..node import Node
from ..utils.typing import TensorLike

class ParameterizationNode(Node):
    """设计参数节点，通常作为图的根节点"""
    def __init__(self, name: str, initial_value: torch.Tensor, requires_grad: bool = True):
        super().__init__(name)
        self.value = torch.nn.Parameter(initial_value, requires_grad=requires_grad)

    def forward(self, **kwargs) -> TensorLike:
        self._cached_output = self.value
        return self.value

    def set_value(self, new_value: torch.Tensor):
        """更新参数值（用于外部优化器）"""
        self.value.data = new_value
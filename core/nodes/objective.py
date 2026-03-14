import torch
from ..node import Node
from ..utils.typing import TensorLike

class ObjectiveNode(Node):
    """目标函数节点，计算标量损失值。"""
    def __init__(self, name: str, input_node: Node, loss_fn):
        """
        loss_fn: 可调用对象，接收输入节点的输出（TensorLike）并返回一个标量张量。
        """
        super().__init__(name)
        self.add_input(input_node)
        self.loss_fn = loss_fn

    def forward(self, **kwargs) -> torch.Tensor:
        input_val = self._inputs[0].forward(**kwargs)
        loss = self.loss_fn(input_val)
        self._cached_output = loss
        return loss
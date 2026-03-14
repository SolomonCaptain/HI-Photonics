from typing import List, Set
import torch
from .utils.typing import TensorLike
from .node import Node

class Graph:
    """管理计算图的执行顺序和缓存。"""
    def __init__(self, output_nodes: List[Node]):
        self.output_nodes = output_nodes
        self._sorted_nodes = None

    def _topological_sort(self) -> List[Node]:
        """对图进行拓扑排序，确保节点按依赖顺序执行。"""
        visited: Set[Node] = set()
        stack: List[Node] = []

        def dfs(node: Node):
            if node in visited:
                return
            visited.add(node)
            for inp in node._inputs:
                dfs(inp)
            stack.append(node)

        for out in self.output_nodes:
            dfs(out)
        return stack

    def forward(self, clear_cache: bool = True, **kwargs) -> List[TensorLike]:
        """
        执行整个计算图的前向传播。
        clear_cache: 是否清除之前的缓存，强制重新计算。
        """
        if clear_cache:
            for node in self._topological_sort():
                node.clear_cache()
        nodes = self._topological_sort()
        outputs = []
        for node in nodes:
            # 如果节点有缓存且不强制重新计算，则直接使用缓存
            if node._cached_output is not None:
                out = node._cached_output
            else:
                out = node.forward(**kwargs)
                node._cached_output = out
            if node in self.output_nodes:
                outputs.append(out)
        return outputs

    def backward(self, grad_outputs: List[torch.Tensor]):
        """反向传播梯度。grad_outputs 应与 output_nodes 一一对应。"""
        # 简单实现：假设图是线性的，从输出节点反向遍历
        # 更通用的做法是利用 PyTorch 的自动微分，但需确保所有节点输出为 Tensor 且可微
        # 这里我们手动调用每个节点的 backward 方法
        for node, grad in zip(reversed(self._topological_sort()), reversed(grad_outputs)):
            node.backward(grad)
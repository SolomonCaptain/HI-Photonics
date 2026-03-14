import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('.')  # 确保可以导入项目模块

from core.node import Node
from core.nodes.parameterization import ParameterizationNode
from core.nodes.simulation import SimulationNode
from core.nodes.objective import ObjectiveNode
from core.graph import Graph

# ========== 自定义 TMM 仿真器（不使用外部接口，直接作为节点） ==========
class TMMLayer(Node):
    """传输矩阵层：计算单层膜的传输矩阵。"""
    def __init__(self, name: str, n: float, thickness_node: Node, wavelength_node: Node):
        super().__init__(name)
        self.n = n
        self.add_input(thickness_node)
        self.add_input(wavelength_node)

    def forward(self, **kwargs) -> torch.Tensor:
        d = self._inputs[0].forward(**kwargs)  # 厚度
        lam = self._inputs[1].forward(**kwargs)  # 波长
        k0 = 2 * torch.pi / lam
        phase = k0 * self.n * d
        # 传输矩阵
        M = torch.stack([
            torch.cos(phase), 1j * torch.sin(phase) / self.n,
            1j * self.n * torch.sin(phase), torch.cos(phase)
        ]).reshape(2, 2)
        return M

class TMMStack(Node):
    """多层膜堆叠：将多个层的传输矩阵相乘。"""
    def __init__(self, name: str, layer_nodes: list):
        super().__init__(name)
        for ln in layer_nodes:
            self.add_input(ln)

    def forward(self, **kwargs) -> torch.Tensor:
        M_total = torch.eye(2, dtype=torch.complex64)
        for inp in self._inputs:
            M = inp.forward(**kwargs)
            M_total = M_total @ M
        return M_total

class TMTransmission(Node):
    """计算透射率（从堆叠的总传输矩阵）。"""
    def __init__(self, name: str, stack_node: Node):
        super().__init__(name)
        self.add_input(stack_node)

    def forward(self, **kwargs) -> torch.Tensor:
        M = self._inputs[0].forward(**kwargs)
        # 透射系数 t = 1 / M[0,0] （假设从空气入射到衬底）
        t = 1.0 / M[0, 0]
        T = torch.abs(t)**2
        return T

def test_1d_multilayer():
    # 1. 定义设计变量：两层膜的厚度
    d1 = ParameterizationNode('d1', torch.tensor([0.1], requires_grad=True))
    d2 = ParameterizationNode('d2', torch.tensor([0.2], requires_grad=True))
    # 固定波长
    lam = ParameterizationNode('lambda', torch.tensor([0.55]), requires_grad=False)

    # 2. 构建仿真图
    layer1 = TMMLayer('layer1', n=1.5, thickness_node=d1, wavelength_node=lam)
    layer2 = TMMLayer('layer2', n=2.0, thickness_node=d2, wavelength_node=lam)
    stack = TMMStack('stack', [layer1, layer2])
    T_node = TMTransmission('T', stack)

    # 3. 定义目标函数：最大化透射率，即最小化 -T
    def loss_fn(T):
        return -T  # 注意 T 是标量张量
    objective = ObjectiveNode('objective', T_node, loss_fn)

    # 4. 构建计算图并执行前向
    graph = Graph([objective])
    outputs = graph.forward()
    print(f"Initial loss: {outputs[0].item()}")

    # 5. 反向传播计算梯度
    graph.backward([torch.tensor(1.0)])  # 对 loss 的梯度为 1
    print(f"Gradient d1: {d1.value.grad.item()}, d2: {d2.value.grad.item()}")

    # 6. 简单的梯度下降优化
    optimizer = torch.optim.SGD([d1.value, d2.value], lr=0.01)
    for i in range(100):
        optimizer.zero_grad()
        loss = graph.forward(clear_cache=True)[0]
        graph.backward([torch.ones_like(loss)])
        optimizer.step()
        if i % 20 == 0:
            print(f"Step {i}, loss: {loss.item()}")

    print(f"Final d1: {d1.value.item()}, d2: {d2.value.item()}")

if __name__ == "__main__":
    test_1d_multilayer()
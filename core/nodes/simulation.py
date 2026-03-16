from typing import Optional, Dict
import torch
from ..node import Node
from interfaces.simulators.base import SimulatorInterface

class SimulationNode(Node):
    """封装仿真器调用的节点"""
    def __init__(self, name: str, simulator: SimulatorInterface, design_node: Node):
        super().__init__(name)
        self.simulator = simulator
        self.add_input(design_node)  # 设计参数节点作为输入
        self._last_design: Optional[torch.Tensor] = None
        self._last_output: Optional[Dict[str, torch.Tensor]] = None

    def forward(self, **kwargs) -> Dict[str, torch.Tensor]:
        # 获取输入设计参数
        design = self._inputs[0].forward(**kwargs)
        if isinstance(design, torch.Tensor):
            # 如果设计有更新，重新运行仿真
            if self._last_design is None or not torch.equal(design, self._last_design):
                self._last_output = self.simulator.run(design.detach().cpu().numpy(), **kwargs)
                self._last_design = design.detach().clone()
                # 将仿真结果转换为 torch 张量（需要梯度？通常仿真结果不直接求导）
                # 我们可以将输出包装为 requires_grad=False 的张量
                self._cached_output = {k: torch.tensor(v, dtype=torch.float32) for k, v in self._last_output.items()}
            return self._cached_output
        else:
            raise TypeError("SimulationNode 期望从设计节点得到一个张量输入。")

    def backward(self, grad_output: torch.Tensor):
        """
        当目标函数对仿真结果有梯度时，需要通过伴随法反传。
        grad_output 通常是一个字典，与 _cached_output 结构相同。
        """
        if isinstance(grad_output, dict):
            # 调用仿真器的伴随计算
            design_grad = self.simulator.compute_gradient(self._last_design, grad_output)
            # 将梯度传递给输入节点（设计参数）
            if self._inputs[0]._cached_output is not None:
                # 手动设置输入节点的梯度（因为输入是 ParameterizationNode，其 value 是 Parameter）
                if isinstance(self._inputs[0].value, torch.nn.Parameter):
                    if self._inputs[0].value.grad is None:
                        self._inputs[0].value.grad = torch.zeros_like(self._inputs[0].value)
                    self._inputs[0].value.grad.add_(torch.tensor(design_grad))
        else:
            raise TypeError("grad_output for SimulationNode 的输出必须是一个字典。")
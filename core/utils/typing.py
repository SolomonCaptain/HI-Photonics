from typing import Dict, Union, Any
import torch

# 计算图中节点之间传递的数据格式：可以是一个张量，或一个包含多个张量的字典
TensorLike = Union[torch.Tensor, Dict[str, torch.Tensor]]
Params = Dict[str, Any] # 节点的超参数
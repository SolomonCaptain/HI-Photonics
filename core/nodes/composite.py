"""
组合节点模块

提供节点组合和管道操作，支持复杂的数据流处理。
在光子学逆向设计中，组合节点用于:
1. 构建处理管道（滤波 -> 投影 -> 约束）
2. 多输入合并
3. 条件执行
4. 缓存和优化
"""

import torch
from typing import Optional, Union, Callable, List, Dict, Any, Tuple
from ..node import Node
from ..utils.typing import TensorLike


class CompositeNode(Node):
    """
    组合节点基类
    
    将多个子节点组合成一个节点，支持复杂的数据流。
    """
    
    def __init__(
        self,
        name: str,
        input_nodes: List[Node],
        combine_fn: Optional[Callable] = None,
        device: Optional[torch.device] = None
    ):
        """
        初始化组合节点
        
        Args:
            name: 节点名称
            input_nodes: 输入节点列表
            combine_fn: 组合函数，接收输入列表返回输出
            device: 计算设备
        """
        super().__init__(name)
        for node in input_nodes:
            self.add_input(node)
        self.combine_fn = combine_fn
        self.device = device or torch.device('cpu')
    
    def forward(self, **kwargs) -> TensorLike:
        """执行组合操作"""
        inputs = [node.forward(**kwargs) for node in self._inputs]
        
        if self.combine_fn is not None:
            result = self.combine_fn(inputs)
        else:
            # 默认行为：返回输入列表
            result = inputs
        
        self._cached_output = result
        return result


class PipelineNode(Node):
    """
    管道节点
    
    将多个节点按顺序连接，形成处理管道。
    """
    
    def __init__(
        self,
        name: str,
        stages: List[Node],
        device: Optional[torch.device] = None
    ):
        """
        Args:
            name: 节点名称
            stages: 处理阶段列表，按顺序执行
            device: 计算设备
        """
        super().__init__(name)
        self.stages = stages
        self.device = device or torch.device('cpu')
        
        # 构建连接：每个阶段的输出作为下一阶段的输入
        if len(stages) > 0:
            self.add_input(stages[0])  # 第一个阶段作为输入
    
    def forward(self, **kwargs) -> TensorLike:
        """按顺序执行管道各阶段"""
        if not self.stages:
            raise ValueError("PipelineNode 至少需要一个阶段")
        
        # 执行第一个阶段
        result = self.stages[0].forward(**kwargs)
        
        # 顺序执行后续阶段
        for stage in self.stages[1:]:
            # 将前一阶段的结果传递给下一阶段
            stage._cached_output = None  # 清除缓存强制重新计算
            result = stage.forward(**kwargs)
        
        self._cached_output = result
        return result
    
    def add_stage(self, stage: Node) -> 'PipelineNode':
        """添加处理阶段"""
        self.stages.append(stage)
        return self
    
    def get_stage_output(self, stage_index: int) -> Optional[TensorLike]:
        """获取指定阶段的输出"""
        if 0 <= stage_index < len(self.stages):
            return self.stages[stage_index]._cached_output
        return None


class MergeNode(Node):
    """
    合并节点
    
    将多个输入节点合并为一个输出。
    """
    
    def __init__(
        self,
        name: str,
        input_nodes: List[Node],
        merge_type: str = 'concat',  # 'concat', 'stack', 'sum', 'mean', 'max', 'min', 'custom'
        dim: int = 0,
        merge_fn: Optional[Callable] = None,
        device: Optional[torch.device] = None
    ):
        """
        Args:
            name: 节点名称
            input_nodes: 输入节点列表
            merge_type: 合并类型
            dim: 合并维度（用于 concat 和 stack）
            merge_fn: 自定义合并函数
            device: 计算设备
        """
        super().__init__(name)
        for node in input_nodes:
            self.add_input(node)
        self.merge_type = merge_type
        self.dim = dim
        self.merge_fn = merge_fn
        self.device = device or torch.device('cpu')
    
    def forward(self, **kwargs) -> TensorLike:
        """执行合并操作"""
        inputs = [node.forward(**kwargs) for node in self._inputs]
        
        # 检查输入类型
        for inp in inputs:
            if isinstance(inp, dict):
                raise TypeError("MergeNode 期望张量输入，不支持字典输入")
        
        tensors = [inp.to(self.device) for inp in inputs]
        
        if self.merge_type == 'concat':
            result = torch.cat(tensors, dim=self.dim)
        elif self.merge_type == 'stack':
            result = torch.stack(tensors, dim=self.dim)
        elif self.merge_type == 'sum':
            result = sum(tensors)
        elif self.merge_type == 'mean':
            result = sum(tensors) / len(tensors)
        elif self.merge_type == 'max':
            result = torch.stack(tensors, dim=0).max(dim=0)[0]
        elif self.merge_type == 'min':
            result = torch.stack(tensors, dim=0).min(dim=0)[0]
        elif self.merge_type == 'custom' and self.merge_fn is not None:
            result = self.merge_fn(tensors)
        else:
            raise ValueError(f"不支持的合并类型: {self.merge_type}")
        
        self._cached_output = result
        return result


class SplitNode(Node):
    """
    分裂节点
    
    将输入张量分裂为多个输出。
    """
    
    def __init__(
        self,
        name: str,
        input_node: Node,
        split_type: str = 'chunk',  # 'chunk', 'split', 'custom'
        num_chunks: int = 2,
        split_sizes: Optional[List[int]] = None,
        dim: int = 0,
        device: Optional[torch.device] = None
    ):
        """
        Args:
            name: 节点名称
            input_node: 输入节点
            split_type: 分裂类型
            num_chunks: 分裂块数（用于 chunk）
            split_sizes: 各块大小列表（用于 split）
            dim: 分裂维度
            device: 计算设备
        """
        super().__init__(name)
        self.add_input(input_node)
        self.split_type = split_type
        self.num_chunks = num_chunks
        self.split_sizes = split_sizes
        self.dim = dim
        self.device = device or torch.device('cpu')
    
    def forward(self, **kwargs) -> List[torch.Tensor]:
        """执行分裂操作"""
        input_val = self._inputs[0].forward(**kwargs)
        
        if isinstance(input_val, dict):
            raise TypeError("SplitNode 期望张量输入")
        
        x = input_val.to(self.device)
        
        if self.split_type == 'chunk':
            result = list(torch.chunk(x, self.num_chunks, dim=self.dim))
        elif self.split_type == 'split':
            if self.split_sizes is None:
                raise ValueError("split 类型需要指定 split_sizes")
            result = list(torch.split(x, self.split_sizes, dim=self.dim))
        else:
            raise ValueError(f"不支持的分裂类型: {self.split_type}")
        
        self._cached_output = result
        return result


class SelectorNode(Node):
    """
    选择器节点
    
    根据条件选择输入。
    """
    
    def __init__(
        self,
        name: str,
        input_nodes: List[Node],
        condition: Union[int, Callable, Node],
        device: Optional[torch.device] = None
    ):
        """
        Args:
            name: 节点名称
            input_nodes: 输入节点列表
            condition: 选择条件
                - int: 固定索引
                - Callable: 函数，返回索引
                - Node: 节点，输出为索引
            device: 计算设备
        """
        super().__init__(name)
        for node in input_nodes:
            self.add_input(node)
        self.condition = condition
        self.device = device or torch.device('cpu')
        
        # 如果条件是节点，添加为输入
        if isinstance(condition, Node) and condition not in input_nodes:
            self.add_input(condition)
    
    def forward(self, **kwargs) -> TensorLike:
        """执行选择操作"""
        inputs = [node.forward(**kwargs) for node in self._inputs]
        
        # 确定选择索引
        if isinstance(self.condition, int):
            idx = self.condition
        elif isinstance(self.condition, Node):
            # 条件节点是最后一个输入
            idx = int(inputs[-1].item())
            inputs = inputs[:-1]  # 移除条件输入
        elif callable(self.condition):
            idx = self.condition(inputs)
        else:
            raise TypeError(f"不支持的条件类型: {type(self.condition)}")
        
        if not 0 <= idx < len(inputs):
            raise IndexError(f"选择索引 {idx} 超出范围 [0, {len(inputs)})")
        
        result = inputs[idx]
        if isinstance(result, torch.Tensor):
            result = result.to(self.device)
        
        self._cached_output = result
        return result


class LambdaNode(Node):
    """
    Lambda 节点
    
    使用自定义函数处理输入。
    """
    
    def __init__(
        self,
        name: str,
        input_node: Node,
        fn: Callable,
        device: Optional[torch.device] = None
    ):
        """
        Args:
            name: 节点名称
            input_node: 输入节点
            fn: 处理函数
            device: 计算设备
        """
        super().__init__(name)
        self.add_input(input_node)
        self.fn = fn
        self.device = device or torch.device('cpu')
    
    def forward(self, **kwargs) -> TensorLike:
        """执行自定义函数"""
        input_val = self._inputs[0].forward(**kwargs)
        
        if isinstance(input_val, torch.Tensor):
            input_val = input_val.to(self.device)
        
        result = self.fn(input_val)
        
        if isinstance(result, torch.Tensor):
            result = result.to(self.device)
        
        self._cached_output = result
        return result


class CachedNode(Node):
    """
    缓存节点
    
    缓存输入节点的输出，避免重复计算。
    """
    
    def __init__(
        self,
        name: str,
        input_node: Node,
        cache_key_fn: Optional[Callable] = None,
        device: Optional[torch.device] = None
    ):
        """
        Args:
            name: 节点名称
            input_node: 输入节点
            cache_key_fn: 缓存键生成函数
            device: 计算设备
        """
        super().__init__(name)
        self.add_input(input_node)
        self.cache_key_fn = cache_key_fn
        self.device = device or torch.device('cpu')
        self._cache: Dict[Any, TensorLike] = {}
    
    def forward(self, **kwargs) -> TensorLike:
        """执行缓存查找或计算"""
        # 生成缓存键
        if self.cache_key_fn is not None:
            cache_key = self.cache_key_fn(kwargs)
        else:
            # 默认使用 kwargs 的字符串表示
            cache_key = str(sorted(kwargs.items()))
        
        # 检查缓存
        if cache_key in self._cache:
            self._cached_output = self._cache[cache_key]
            return self._cached_output
        
        # 计算并缓存
        result = self._inputs[0].forward(**kwargs)
        if isinstance(result, torch.Tensor):
            result = result.to(self.device)
        
        self._cache[cache_key] = result
        self._cached_output = result
        return result
    
    def clear_cache(self):
        """清除缓存"""
        super().clear_cache()
        self._cache.clear()
    
    def get_cache_size(self) -> int:
        """获取缓存大小"""
        return len(self._cache)


class ResidualNode(Node):
    """
    残差节点
    
    实现残差连接: output = input + transform(input)
    """
    
    def __init__(
        self,
        name: str,
        input_node: Node,
        transform_node: Node,
        scale: float = 1.0,
        device: Optional[torch.device] = None
    ):
        """
        Args:
            name: 节点名称
            input_node: 输入节点
            transform_node: 变换节点
            scale: 残差缩放因子
            device: 计算设备
        """
        super().__init__(name)
        self.add_input(input_node)
        self.add_input(transform_node)
        self.scale = scale
        self.device = device or torch.device('cpu')
    
    def forward(self, **kwargs) -> TensorLike:
        """执行残差连接"""
        input_val = self._inputs[0].forward(**kwargs)
        transform_val = self._inputs[1].forward(**kwargs)
        
        if isinstance(input_val, dict) or isinstance(transform_val, dict):
            raise TypeError("ResidualNode 期望张量输入")
        
        x = input_val.to(self.device)
        t = transform_val.to(self.device)
        
        result = x + self.scale * t
        self._cached_output = result
        return result


class BranchNode(Node):
    """
    分支节点
    
    将输入同时传递给多个下游节点。
    """
    
    def __init__(
        self,
        name: str,
        input_node: Node,
        branch_nodes: List[Node],
        device: Optional[torch.device] = None
    ):
        """
        Args:
            name: 节点名称
            input_node: 输入节点
            branch_nodes: 分支节点列表
            device: 计算设备
        """
        super().__init__(name)
        self.add_input(input_node)
        self.branch_nodes = branch_nodes
        self.device = device or torch.device('cpu')
    
    def forward(self, **kwargs) -> Dict[str, TensorLike]:
        """执行分支操作"""
        input_val = self._inputs[0].forward(**kwargs)
        
        results = {}
        for i, branch in enumerate(self.branch_nodes):
            # 将输入缓存到分支节点
            branch._cached_output = input_val
            results[f"branch_{i}"] = branch.forward(**kwargs)
        
        self._cached_output = results
        return results


class WeightedSumNode(Node):
    """
    加权求和节点
    
    对多个输入进行加权求和。
    """
    
    def __init__(
        self,
        name: str,
        input_nodes: List[Node],
        weights: Optional[List[float]] = None,
        learnable: bool = False,
        device: Optional[torch.device] = None
    ):
        """
        Args:
            name: 节点名称
            input_nodes: 输入节点列表
            weights: 权重列表
            learnable: 权重是否可学习
            device: 计算设备
        """
        super().__init__(name)
        for node in input_nodes:
            self.add_input(node)
        self.device = device or torch.device('cpu')
        
        if weights is None:
            weights = [1.0 / len(input_nodes)] * len(input_nodes)
        
        if learnable:
            self.weights = torch.nn.Parameter(
                torch.tensor(weights, device=self.device)
            )
        else:
            self.weights = torch.tensor(weights, device=self.device)
    
    def forward(self, **kwargs) -> TensorLike:
        """执行加权求和"""
        inputs = [node.forward(**kwargs) for node in self._inputs]
        
        # 归一化权重
        weights = torch.softmax(self.weights, dim=0)
        
        result = None
        for inp, w in zip(inputs, weights):
            if isinstance(inp, dict):
                raise TypeError("WeightedSumNode 期望张量输入")
            x = inp.to(self.device) * w
            if result is None:
                result = x
            else:
                result = result + x
        
        self._cached_output = result
        return result


class DesignPipelineNode(Node):
    """
    设计管道节点
    
    专门为光子学逆向设计设计的管道节点，
    集成了滤波、投影和约束处理。
    """
    
    def __init__(
        self,
        name: str,
        input_node: Node,
        filter_type: Optional[str] = 'gaussian',
        filter_radius: int = 2,
        projection_type: str = 'sigmoid',
        threshold: float = 0.5,
        sharpness: float = 10.0,
        constraints: Optional[List[Node]] = None,
        device: Optional[torch.device] = None
    ):
        """
        Args:
            name: 节点名称
            input_node: 输入节点（设计参数）
            filter_type: 滤波类型 (None 表示不滤波)
            filter_radius: 滤波半径
            projection_type: 投影类型
            threshold: 投影阈值
            sharpness: 投影锐度
            constraints: 约束节点列表
            device: 计算设备
        """
        super().__init__(name)
        self.add_input(input_node)
        self.filter_type = filter_type
        self.filter_radius = filter_radius
        self.projection_type = projection_type
        self.threshold = threshold
        self.sharpness = sharpness
        self.constraints = constraints or []
        self.device = device or torch.device('cpu')
        
        # 延迟导入避免循环依赖
        from .filter import FilterNode
        from .projection import ProjectionNode
        
        # 构建处理管道
        self._filter_node = None
        self._projection_node = None
        
        if filter_type is not None:
            self._filter_node = FilterNode(
                f"{name}_filter",
                input_node,
                filter_type=filter_type,
                kernel_size=2 * filter_radius + 1,
                device=device
            )
            filter_output = self._filter_node
        else:
            filter_output = input_node
        
        self._projection_node = ProjectionNode(
            f"{name}_projection",
            filter_output if isinstance(filter_output, Node) else input_node,
            projection_type=projection_type,
            threshold=threshold,
            sharpness=sharpness,
            device=device
        )
    
    def forward(self, **kwargs) -> TensorLike:
        """执行设计管道"""
        # 获取原始输入
        input_val = self._inputs[0].forward(**kwargs)
        
        if isinstance(input_val, dict):
            raise TypeError("DesignPipelineNode 期望张量输入")
        
        x = input_val.to(self.device)
        
        # 滤波
        if self._filter_node is not None:
            x = self._filter_node.forward(**kwargs)
        
        # 投影
        x = self._projection_node.forward(**kwargs)
        
        self._cached_output = x
        return x
    
    def get_constraint_values(self, **kwargs) -> Dict[str, torch.Tensor]:
        """获取所有约束的值"""
        design = self.forward(**kwargs)
        constraint_values = {}
        
        for i, constraint in enumerate(self.constraints):
            # 将设计缓存到约束节点
            constraint._cached_output = None
            value = constraint.forward(**kwargs)
            constraint_values[f"constraint_{i}"] = value
        
        return constraint_values

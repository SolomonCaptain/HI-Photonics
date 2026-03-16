"""
投影节点模块

提供各种投影操作，将连续设计变量映射到特定范围或离散值。
在光子学逆向设计中，投影用于:
1. 二值化设计变量（0 或 1）
2. 将设计变量限制在 [0, 1] 范围
3. 实现密度惩罚
4. 拓扑优化中的 Heaviside 投影
"""

import torch
import torch.nn.functional as F
from typing import Optional, Union, Callable
from ..node import Node
from ..utils.typing import TensorLike


class ProjectionNode(Node):
    """
    投影节点基类
    
    将输入张量投影到特定范围或离散值。
    """
    
    def __init__(
        self,
        name: str,
        input_node: Node,
        projection_type: str = 'sigmoid',
        threshold: float = 0.5,
        sharpness: float = 10.0,
        min_val: float = 0.0,
        max_val: float = 1.0,
        device: Optional[torch.device] = None
    ):
        """
        初始化投影节点
        
        Args:
            name: 节点名称
            input_node: 输入节点
            projection_type: 投影类型
                - 'sigmoid': Sigmoid 投影（软阈值）
                - 'heaviside': Heaviside 阶跃函数（硬阈值）
                - 'tanh': Tanh 投影
                - 'relu': ReLU 投影（截断）
                - 'clamp': 直接截断
                - 'softmax': Softmax 投影（多材料）
                - 'ramp': 斜坡投影
            threshold: 阈值参数
            sharpness: 投影锐度（用于软投影）
            min_val: 最小值
            max_val: 最大值
            device: 计算设备
        """
        super().__init__(name)
        self.add_input(input_node)
        self.projection_type = projection_type
        self.threshold = threshold
        self.sharpness = sharpness
        self.min_val = min_val
        self.max_val = max_val
        self.device = device or torch.device('cpu')
    
    def forward(self, **kwargs) -> TensorLike:
        """执行投影操作"""
        input_val = self._inputs[0].forward(**kwargs)
        
        if isinstance(input_val, dict):
            raise TypeError("ProjectionNode 期望张量输入")
        
        x = input_val.to(self.device)
        
        if self.projection_type == 'sigmoid':
            # Sigmoid 投影: σ(β(x - η))
            # β: 锐度, η: 阈值
            result = torch.sigmoid(self.sharpness * (x - self.threshold))
            
        elif self.projection_type == 'heaviside':
            # Heaviside 阶跃函数（使用直通估计器近似）
            result = (x >= self.threshold).float()
            # 使用直通估计器允许梯度通过
            result = result + torch.sigmoid(self.sharpness * (x - self.threshold)) - \
                     torch.sigmoid(self.sharpness * (x - self.threshold)).detach()
            
        elif self.projection_type == 'tanh':
            # Tanh 投影: (tanh(β(x - η)) + 1) / 2
            result = (torch.tanh(self.sharpness * (x - self.threshold)) + 1) / 2
            
        elif self.projection_type == 'relu':
            # ReLU 投影: min(max(x, min_val), max_val)
            result = torch.clamp(x, self.min_val, self.max_val)
            
        elif self.projection_type == 'clamp':
            # 直接截断
            result = torch.clamp(x, self.min_val, self.max_val)
            
        elif self.projection_type == 'ramp':
            # 斜坡投影: 在阈值附近线性过渡
            half_width = 1.0 / self.sharpness
            result = torch.clamp(
                (x - self.threshold + half_width) / (2 * half_width),
                self.min_val,
                self.max_val
            )
            
        elif self.projection_type == 'softmax':
            # Softmax 投影（用于多材料设计）
            # 假设最后一维是材料维度
            result = F.softmax(self.sharpness * x, dim=-1)
            
        else:
            raise ValueError(f"不支持的投影类型: {self.projection_type}")
        
        self._cached_output = result
        return result


class SigmoidProjectionNode(ProjectionNode):
    """Sigmoid 投影节点 - 便捷类"""
    
    def __init__(
        self,
        name: str,
        input_node: Node,
        threshold: float = 0.5,
        sharpness: float = 10.0,
        **kwargs
    ):
        super().__init__(
            name, input_node,
            projection_type='sigmoid',
            threshold=threshold,
            sharpness=sharpness,
            **kwargs
        )


class HeavisideProjectionNode(ProjectionNode):
    """Heaviside 阶跃投影节点 - 便捷类"""
    
    def __init__(
        self,
        name: str,
        input_node: Node,
        threshold: float = 0.5,
        sharpness: float = 10.0,
        **kwargs
    ):
        super().__init__(
            name, input_node,
            projection_type='heaviside',
            threshold=threshold,
            sharpness=sharpness,
            **kwargs
        )


class TanhProjectionNode(ProjectionNode):
    """Tanh 投影节点 - 便捷类"""
    
    def __init__(
        self,
        name: str,
        input_node: Node,
        threshold: float = 0.5,
        sharpness: float = 10.0,
        **kwargs
    ):
        super().__init__(
            name, input_node,
            projection_type='tanh',
            threshold=threshold,
            sharpness=sharpness,
            **kwargs
        )


class SoftmaxProjectionNode(ProjectionNode):
    """Softmax 投影节点 - 用于多材料设计"""
    
    def __init__(
        self,
        name: str,
        input_node: Node,
        sharpness: float = 1.0,
        **kwargs
    ):
        super().__init__(
            name, input_node,
            projection_type='softmax',
            sharpness=sharpness,
            **kwargs
        )


class ThresholdProjectionNode(Node):
    """
    阈值投影节点
    
    使用可学习的阈值进行投影，支持连续优化。
    """
    
    def __init__(
        self,
        name: str,
        input_node: Node,
        initial_threshold: float = 0.5,
        learnable: bool = False,
        sharpness: float = 10.0,
        device: Optional[torch.device] = None
    ):
        """
        Args:
            name: 节点名称
            input_node: 输入节点
            initial_threshold: 初始阈值
            learnable: 阈值是否可学习
            sharpness: 投影锐度
            device: 计算设备
        """
        super().__init__(name)
        self.add_input(input_node)
        self.sharpness = sharpness
        self.device = device or torch.device('cpu')
        
        if learnable:
            self.threshold = torch.nn.Parameter(
                torch.tensor(initial_threshold, device=self.device)
            )
        else:
            self.threshold = torch.tensor(initial_threshold, device=self.device)
    
    def forward(self, **kwargs) -> TensorLike:
        input_val = self._inputs[0].forward(**kwargs)
        
        if isinstance(input_val, dict):
            raise TypeError("ThresholdProjectionNode 期望张量输入")
        
        x = input_val.to(self.device)
        result = torch.sigmoid(self.sharpness * (x - self.threshold))
        
        self._cached_output = result
        return result


class DensityProjectionNode(Node):
    """
    密度投影节点
    
    用于拓扑优化中的密度方法，实现 SIMP (Solid Isotropic Material with Penalization) 惩罚。
    """
    
    def __init__(
        self,
        name: str,
        input_node: Node,
        penalty: float = 3.0,
        min_density: float = 0.0,
        max_density: float = 1.0,
        device: Optional[torch.device] = None
    ):
        """
        Args:
            name: 节点名称
            input_node: 输入节点
            penalty: SIMP 惩罚因子（通常为 3）
            min_density: 最小密度（空材料）
            max_density: 最大密度（实体材料）
            device: 计算设备
        """
        super().__init__(name)
        self.add_input(input_node)
        self.penalty = penalty
        self.min_density = min_density
        self.max_density = max_density
        self.device = device or torch.device('cpu')
    
    def forward(self, **kwargs) -> TensorLike:
        """
        SIMP 惩罚: ρ^p
        将中间密度值推向 0 或 1
        """
        input_val = self._inputs[0].forward(**kwargs)
        
        if isinstance(input_val, dict):
            raise TypeError("DensityProjectionNode 期望张量输入")
        
        x = input_val.to(self.device)
        
        # 首先截断到 [0, 1]
        x_clamped = torch.clamp(x, 0.0, 1.0)
        
        # 应用 SIMP 惩罚
        # ρ_eff = ρ_min + (ρ_max - ρ_min) * ρ^p
        result = self.min_density + (self.max_density - self.min_density) * torch.pow(x_clamped, self.penalty)
        
        self._cached_output = result
        return result


class ErosionProjectionNode(Node):
    """
    腐蚀投影节点
    
    用于实现制造约束中的最小特征尺寸。
    通过腐蚀操作确保实体区域的最小尺寸。
    """
    
    def __init__(
        self,
        name: str,
        input_node: Node,
        radius: int = 2,
        threshold: float = 0.5,
        sharpness: float = 10.0,
        device: Optional[torch.device] = None
    ):
        """
        Args:
            name: 节点名称
            input_node: 输入节点
            radius: 腐蚀半径（像素）
            threshold: 投影阈值
            sharpness: 投影锐度
            device: 计算设备
        """
        super().__init__(name)
        self.add_input(input_node)
        self.radius = radius
        self.threshold = threshold
        self.sharpness = sharpness
        self.device = device or torch.device('cpu')
        
        # 创建圆形结构元素
        self._kernel = self._create_circular_kernel()
    
    def _create_circular_kernel(self) -> torch.Tensor:
        """创建圆形结构元素"""
        k = 2 * self.radius + 1
        y, x = torch.meshgrid(
            torch.arange(k, dtype=torch.float32),
            torch.arange(k, dtype=torch.float32),
            indexing='ij'
        )
        center = self.radius
        dist = torch.sqrt((x - center)**2 + (y - center)**2)
        kernel = (dist <= self.radius).float()
        return kernel.unsqueeze(0).unsqueeze(0).to(self.device)
    
    def forward(self, **kwargs) -> TensorLike:
        """执行腐蚀投影"""
        input_val = self._inputs[0].forward(**kwargs)
        
        if isinstance(input_val, dict):
            raise TypeError("ErosionProjectionNode 期望张量输入")
        
        x = input_val.to(self.device)
        original_dim = x.dim()
        if original_dim == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif original_dim == 3:
            x = x.unsqueeze(0)
        
        # 先投影
        x_proj = torch.sigmoid(self.sharpness * (x - self.threshold))
        
        # 执行腐蚀（最小值滤波）
        k = 2 * self.radius + 1
        eroded = -F.max_pool2d(-x_proj, k, stride=1, padding=self.radius)
        
        if original_dim == 2:
            eroded = eroded.squeeze(0).squeeze(0)
        elif original_dim == 3:
            eroded = eroded.squeeze(0)
        
        self._cached_output = eroded
        return eroded


class DilationProjectionNode(Node):
    """
    膨胀投影节点
    
    用于实现制造约束中的最小孔洞尺寸。
    通过膨胀操作确保孔洞的最小尺寸。
    """
    
    def __init__(
        self,
        name: str,
        input_node: Node,
        radius: int = 2,
        threshold: float = 0.5,
        sharpness: float = 10.0,
        device: Optional[torch.device] = None
    ):
        """
        Args:
            name: 节点名称
            input_node: 输入节点
            radius: 膨胀半径（像素）
            threshold: 投影阈值
            sharpness: 投影锐度
            device: 计算设备
        """
        super().__init__(name)
        self.add_input(input_node)
        self.radius = radius
        self.threshold = threshold
        self.sharpness = sharpness
        self.device = device or torch.device('cpu')
        
        self._kernel = self._create_circular_kernel()
    
    def _create_circular_kernel(self) -> torch.Tensor:
        """创建圆形结构元素"""
        k = 2 * self.radius + 1
        y, x = torch.meshgrid(
            torch.arange(k, dtype=torch.float32),
            torch.arange(k, dtype=torch.float32),
            indexing='ij'
        )
        center = self.radius
        dist = torch.sqrt((x - center)**2 + (y - center)**2)
        kernel = (dist <= self.radius).float()
        return kernel.unsqueeze(0).unsqueeze(0).to(self.device)
    
    def forward(self, **kwargs) -> TensorLike:
        """执行膨胀投影"""
        input_val = self._inputs[0].forward(**kwargs)
        
        if isinstance(input_val, dict):
            raise TypeError("DilationProjectionNode 期望张量输入")
        
        x = input_val.to(self.device)
        original_dim = x.dim()
        if original_dim == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif original_dim == 3:
            x = x.unsqueeze(0)
        
        # 先投影
        x_proj = torch.sigmoid(self.sharpness * (x - self.threshold))
        
        # 执行膨胀（最大值滤波）
        k = 2 * self.radius + 1
        dilated = F.max_pool2d(x_proj, k, stride=1, padding=self.radius)
        
        if original_dim == 2:
            dilated = dilated.squeeze(0).squeeze(0)
        elif original_dim == 3:
            dilated = dilated.squeeze(0)
        
        self._cached_output = dilated
        return dilated


class CombinedProjectionNode(Node):
    """
    组合投影节点
    
    结合滤波和投影操作，实现平滑后的投影。
    常用流程: 高斯滤波 -> Sigmoid 投影
    """
    
    def __init__(
        self,
        name: str,
        input_node: Node,
        filter_radius: int = 2,
        threshold: float = 0.5,
        sharpness: float = 10.0,
        projection_type: str = 'sigmoid',
        device: Optional[torch.device] = None
    ):
        """
        Args:
            name: 节点名称
            input_node: 输入节点
            filter_radius: 滤波半径
            threshold: 投影阈值
            sharpness: 投影锐度
            projection_type: 投影类型
            device: 计算设备
        """
        super().__init__(name)
        self.add_input(input_node)
        self.filter_radius = filter_radius
        self.threshold = threshold
        self.sharpness = sharpness
        self.projection_type = projection_type
        self.device = device or torch.device('cpu')
        
        # 创建高斯滤波核
        self._filter_kernel = self._create_gaussian_kernel()
    
    def _create_gaussian_kernel(self) -> torch.Tensor:
        """创建高斯滤波核"""
        k = 2 * self.filter_radius + 1
        sigma = self.filter_radius / 2.0
        
        x = torch.arange(k, dtype=torch.float32, device=self.device) - (k - 1) / 2
        gauss_1d = torch.exp(-x**2 / (2 * sigma**2))
        gauss_1d = gauss_1d / gauss_1d.sum()
        kernel = gauss_1d.unsqueeze(1) @ gauss_1d.unsqueeze(0)
        return kernel.unsqueeze(0).unsqueeze(0)
    
    def forward(self, **kwargs) -> TensorLike:
        """执行滤波后投影"""
        input_val = self._inputs[0].forward(**kwargs)
        
        if isinstance(input_val, dict):
            raise TypeError("CombinedProjectionNode 期望张量输入")
        
        x = input_val.to(self.device)
        original_dim = x.dim()
        if original_dim == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif original_dim == 3:
            x = x.unsqueeze(0)
        
        batch, channels, height, width = x.shape
        
        # 高斯滤波
        kernel = self._filter_kernel.expand(channels, -1, -1, -1)
        k = 2 * self.filter_radius + 1
        x_filtered = F.conv2d(x, kernel, padding=self.filter_radius, groups=channels)
        
        # 投影
        if self.projection_type == 'sigmoid':
            result = torch.sigmoid(self.sharpness * (x_filtered - self.threshold))
        elif self.projection_type == 'tanh':
            result = (torch.tanh(self.sharpness * (x_filtered - self.threshold)) + 1) / 2
        else:
            result = torch.clamp(x_filtered, 0.0, 1.0)
        
        if original_dim == 2:
            result = result.squeeze(0).squeeze(0)
        elif original_dim == 3:
            result = result.squeeze(0)
        
        self._cached_output = result
        return result

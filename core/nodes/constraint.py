"""
约束节点模块

提供各种约束计算，用于限制设计的可行性。
在光子学逆向设计中，约束用于:
1. 最小特征尺寸约束（制造约束）
2. 曲率约束
3. 连通性约束
4. 对称性约束
5. 体积约束
"""

import torch
import torch.nn.functional as F
from typing import Optional, Union, Callable, Tuple, List
from ..node import Node
from ..utils.typing import TensorLike


class ConstraintNode(Node):
    """
    约束节点基类
    
    计算约束值（通常为标量），用于优化中的惩罚项或障碍函数。
    """
    
    def __init__(
        self,
        name: str,
        input_node: Node,
        constraint_type: str = 'volume',
        target_value: float = 0.5,
        weight: float = 1.0,
        device: Optional[torch.device] = None
    ):
        """
        初始化约束节点
        
        Args:
            name: 节点名称
            input_node: 输入节点
            constraint_type: 约束类型
            target_value: 目标值
            weight: 约束权重
            device: 计算设备
        """
        super().__init__(name)
        self.add_input(input_node)
        self.constraint_type = constraint_type
        self.target_value = target_value
        self.weight = weight
        self.device = device or torch.device('cpu')
    
    def forward(self, **kwargs) -> TensorLike:
        """计算约束值"""
        input_val = self._inputs[0].forward(**kwargs)
        
        if isinstance(input_val, dict):
            raise TypeError("ConstraintNode 期望张量输入")
        
        x = input_val.to(self.device)
        
        if self.constraint_type == 'volume':
            result = self._volume_constraint(x)
        elif self.constraint_type == 'binary':
            result = self._binary_constraint(x)
        elif self.constraint_type == 'smoothness':
            result = self._smoothness_constraint(x)
        else:
            raise ValueError(f"不支持的约束类型: {self.constraint_type}")
        
        self._cached_output = result
        return result
    
    def _volume_constraint(self, x: torch.Tensor) -> torch.Tensor:
        """体积约束: |mean(x) - target|"""
        return torch.abs(x.mean() - self.target_value) * self.weight
    
    def _binary_constraint(self, x: torch.Tensor) -> torch.Tensor:
        """
        二值化约束: 惩罚中间值
        x * (1 - x) 在 x=0.5 时最大，在 x=0 或 x=1 时为 0
        """
        return (x * (1 - x)).mean() * self.weight * 4  # 乘以 4 归一化
    
    def _smoothness_constraint(self, x: torch.Tensor) -> torch.Tensor:
        """平滑约束: 惩罚高频变化"""
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            x = x.unsqueeze(0)
        
        # 计算梯度
        grad_x = x[:, :, :, 1:] - x[:, :, :, :-1]
        grad_y = x[:, :, 1:, :] - x[:, :, :-1, :]
        
        # L2 范数
        smoothness = (grad_x**2).mean() + (grad_y**2).mean()
        return smoothness * self.weight


class VolumeConstraintNode(ConstraintNode):
    """体积约束节点 - 便捷类"""
    
    def __init__(
        self,
        name: str,
        input_node: Node,
        target_fraction: float = 0.5,
        weight: float = 1.0,
        **kwargs
    ):
        """
        Args:
            name: 节点名称
            input_node: 输入节点
            target_fraction: 目标体积分数
            weight: 约束权重
        """
        super().__init__(
            name, input_node,
            constraint_type='volume',
            target_value=target_fraction,
            weight=weight,
            **kwargs
        )


class BinaryConstraintNode(ConstraintNode):
    """二值化约束节点 - 便捷类"""
    
    def __init__(
        self,
        name: str,
        input_node: Node,
        weight: float = 1.0,
        **kwargs
    ):
        super().__init__(
            name, input_node,
            constraint_type='binary',
            weight=weight,
            **kwargs
        )


class MinimumFeatureSizeNode(Node):
    """
    最小特征尺寸约束节点
    
    确保设计的最小特征尺寸满足制造要求。
    使用形态学操作检测过小的特征。
    """
    
    def __init__(
        self,
        name: str,
        input_node: Node,
        min_feature_size: float = 5.0,
        resolution: float = 1.0,
        threshold: float = 0.5,
        weight: float = 1.0,
        device: Optional[torch.device] = None
    ):
        """
        Args:
            name: 节点名称
            input_node: 输入节点
            min_feature_size: 最小特征尺寸（物理单位）
            resolution: 网格分辨率（每像素的物理尺寸）
            threshold: 二值化阈值
            weight: 约束权重
            device: 计算设备
        """
        super().__init__(name)
        self.add_input(input_node)
        self.min_feature_size = min_feature_size
        self.resolution = resolution
        self.threshold = threshold
        self.weight = weight
        self.device = device or torch.device('cpu')
        
        # 计算形态学操作的半径（像素）
        self.radius = max(1, int(min_feature_size / resolution / 2))
    
    def forward(self, **kwargs) -> TensorLike:
        """
        计算最小特征尺寸约束
        
        返回违反约束的程度：
        - 0 表示满足约束
        - > 0 表示违反约束的程度
        """
        input_val = self._inputs[0].forward(**kwargs)
        
        if isinstance(input_val, dict):
            raise TypeError("MinimumFeatureSizeNode 期望张量输入")
        
        x = input_val.to(self.device)
        original_dim = x.dim()
        if original_dim == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif original_dim == 3:
            x = x.unsqueeze(0)
        
        # 二值化
        x_binary = (x >= self.threshold).float()
        
        k = 2 * self.radius + 1
        
        # 检测过小的实体区域：先腐蚀再膨胀，与原图比较
        eroded = -F.max_pool2d(-x_binary, k, stride=1, padding=self.radius)
        dilated_after_erosion = F.max_pool2d(eroded, k, stride=1, padding=self.radius)
        small_solid = x_binary - dilated_after_erosion  # 过小的实体区域
        
        # 检测过小的孔洞：先膨胀再腐蚀，与原图比较
        dilated = F.max_pool2d(x_binary, k, stride=1, padding=self.radius)
        eroded_after_dilation = -F.max_pool2d(-dilated, k, stride=1, padding=self.radius)
        small_holes = eroded_after_dilation - x_binary  # 过小的孔洞
        
        # 总违反程度
        violation = small_solid.abs().mean() + small_holes.abs().mean()
        
        result = violation * self.weight
        self._cached_output = result
        return result


class CurvatureConstraintNode(Node):
    """
    曲率约束节点
    
    限制设计边界的最大曲率，确保可制造性。
    高曲率区域可能导致制造困难。
    """
    
    def __init__(
        self,
        name: str,
        input_node: Node,
        max_curvature: float = 0.1,
        threshold: float = 0.5,
        weight: float = 1.0,
        device: Optional[torch.device] = None
    ):
        """
        Args:
            name: 节点名称
            input_node: 输入节点
            max_curvature: 最大允许曲率
            threshold: 边界检测阈值
            weight: 约束权重
            device: 计算设备
        """
        super().__init__(name)
        self.add_input(input_node)
        self.max_curvature = max_curvature
        self.threshold = threshold
        self.weight = weight
        self.device = device or torch.device('cpu')
    
    def forward(self, **kwargs) -> TensorLike:
        """计算曲率约束"""
        input_val = self._inputs[0].forward(**kwargs)
        
        if isinstance(input_val, dict):
            raise TypeError("CurvatureConstraintNode 期望张量输入")
        
        x = input_val.to(self.device)
        original_dim = x.dim()
        if original_dim == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif original_dim == 3:
            x = x.unsqueeze(0)
        
        # 计算梯度
        grad_x = x[:, :, :, 1:] - x[:, :, :, :-1]
        grad_y = x[:, :, 1:, :] - x[:, :, :-1, :]
        
        # 填充以保持形状
        grad_x = F.pad(grad_x, (0, 1, 0, 0), mode='replicate')
        grad_y = F.pad(grad_y, (0, 0, 0, 1), mode='replicate')
        
        # 计算二阶导数
        grad_xx = grad_x[:, :, :, 1:] - grad_x[:, :, :, :-1]
        grad_yy = grad_y[:, :, 1:, :] - grad_y[:, :, :-1, :]
        grad_xy = grad_x[:, :, 1:, :] - grad_x[:, :, :-1, :]
        
        # 填充
        grad_xx = F.pad(grad_xx, (0, 1, 0, 0), mode='replicate')
        grad_yy = F.pad(grad_yy, (0, 0, 0, 1), mode='replicate')
        grad_xy = F.pad(grad_xy, (0, 0, 0, 1), mode='replicate')
        
        # 计算曲率: κ = |∇²f| / (1 + |∇f|²)^(3/2)
        grad_magnitude_sq = grad_x**2 + grad_y**2 + 1e-8
        laplacian = grad_xx + grad_yy
        
        curvature = laplacian.abs() / (1 + grad_magnitude_sq).pow(1.5)
        
        # 边界权重：只在边界附近计算曲率
        boundary_mask = (x - self.threshold).abs() < 0.2
        boundary_mask = boundary_mask.float()
        
        # 加权曲率
        weighted_curvature = (curvature * boundary_mask).sum() / (boundary_mask.sum() + 1e-8)
        
        # 惩罚超过最大曲率的区域
        violation = F.relu(curvature - self.max_curvature)
        result = (violation * boundary_mask).mean() * self.weight
        
        self._cached_output = result
        return result


class ConnectivityConstraintNode(Node):
    """
    连通性约束节点
    
    确保设计区域的连通性。
    """
    
    def __init__(
        self,
        name: str,
        input_node: Node,
        threshold: float = 0.5,
        weight: float = 1.0,
        device: Optional[torch.device] = None
    ):
        """
        Args:
            name: 节点名称
            input_node: 输入节点
            threshold: 二值化阈值
            weight: 约束权重
            device: 计算设备
        """
        super().__init__(name)
        self.add_input(input_node)
        self.threshold = threshold
        self.weight = weight
        self.device = device or torch.device('cpu')
    
    def forward(self, **kwargs) -> TensorLike:
        """
        计算连通性约束
        
        使用形态学操作估计连通性。
        完整实现需要连通分量分析，这里使用简化版本。
        """
        input_val = self._inputs[0].forward(**kwargs)
        
        if isinstance(input_val, dict):
            raise TypeError("ConnectivityConstraintNode 期望张量输入")
        
        x = input_val.to(self.device)
        original_dim = x.dim()
        if original_dim == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif original_dim == 3:
            x = x.unsqueeze(0)
        
        # 二值化
        x_binary = (x >= self.threshold).float()
        
        # 使用形态学开闭运算检测断裂区域
        k = 3
        
        # 开运算检测断裂
        eroded = -F.max_pool2d(-x_binary, k, stride=1, padding=1)
        opened = F.max_pool2d(eroded, k, stride=1, padding=1)
        
        # 碎片检测：开运算后消失的区域
        fragments = x_binary - opened
        
        # 惩罚碎片
        result = fragments.abs().mean() * self.weight
        
        self._cached_output = result
        return result


class SymmetryConstraintNode(Node):
    """
    对称性约束节点
    
    确保设计满足指定的对称性。
    """
    
    def __init__(
        self,
        name: str,
        input_node: Node,
        symmetry_type: str = 'horizontal',  # 'horizontal', 'vertical', 'rotational', 'quad'
        weight: float = 1.0,
        device: Optional[torch.device] = None
    ):
        """
        Args:
            name: 节点名称
            input_node: 输入节点
            symmetry_type: 对称类型
            weight: 约束权重
            device: 计算设备
        """
        super().__init__(name)
        self.add_input(input_node)
        self.symmetry_type = symmetry_type
        self.weight = weight
        self.device = device or torch.device('cpu')
    
    def forward(self, **kwargs) -> TensorLike:
        """计算对称性约束"""
        input_val = self._inputs[0].forward(**kwargs)
        
        if isinstance(input_val, dict):
            raise TypeError("SymmetryConstraintNode 期望张量输入")
        
        x = input_val.to(self.device)
        
        if self.symmetry_type == 'horizontal':
            # 水平对称（关于垂直轴）
            x_flipped = torch.flip(x, dims=[-1])  # 水平翻转
            asymmetry = (x - x_flipped).abs().mean()
            
        elif self.symmetry_type == 'vertical':
            # 垂直对称（关于水平轴）
            x_flipped = torch.flip(x, dims=[-2])  # 垂直翻转
            asymmetry = (x - x_flipped).abs().mean()
            
        elif self.symmetry_type == 'rotational':
            # 旋转对称（180度）
            x_rotated = torch.flip(x, dims=[-2, -1])  # 旋转180度
            asymmetry = (x - x_rotated).abs().mean()
            
        elif self.symmetry_type == 'quad':
            # 四重对称
            x_h = torch.flip(x, dims=[-1])
            x_v = torch.flip(x, dims=[-2])
            x_r = torch.flip(x, dims=[-2, -1])
            asymmetry = ((x - x_h).abs() + (x - x_v).abs() + (x - x_r).abs()).mean() / 3
            
        else:
            raise ValueError(f"不支持的对称类型: {self.symmetry_type}")
        
        result = asymmetry * self.weight
        self._cached_output = result
        return result


class PerimeterConstraintNode(Node):
    """
    周长约束节点
    
    限制设计的边界周长，用于控制复杂度。
    """
    
    def __init__(
        self,
        name: str,
        input_node: Node,
        target_perimeter: Optional[float] = None,
        max_perimeter: Optional[float] = None,
        threshold: float = 0.5,
        weight: float = 1.0,
        device: Optional[torch.device] = None
    ):
        """
        Args:
            name: 节点名称
            input_node: 输入节点
            target_perimeter: 目标周长（可选）
            max_perimeter: 最大周长（可选）
            threshold: 边界检测阈值
            weight: 约束权重
            device: 计算设备
        """
        super().__init__(name)
        self.add_input(input_node)
        self.target_perimeter = target_perimeter
        self.max_perimeter = max_perimeter
        self.threshold = threshold
        self.weight = weight
        self.device = device or torch.device('cpu')
    
    def forward(self, **kwargs) -> TensorLike:
        """计算周长约束"""
        input_val = self._inputs[0].forward(**kwargs)
        
        if isinstance(input_val, dict):
            raise TypeError("PerimeterConstraintNode 期望张量输入")
        
        x = input_val.to(self.device)
        original_dim = x.dim()
        if original_dim == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif original_dim == 3:
            x = x.unsqueeze(0)
        
        # 计算边界
        # 使用梯度检测边界
        grad_x = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs()
        grad_y = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs()
        
        # 填充
        grad_x = F.pad(grad_x, (0, 1, 0, 0), mode='replicate')
        grad_y = F.pad(grad_y, (0, 0, 0, 1), mode='replicate')
        
        # 边界强度
        boundary = grad_x + grad_y
        
        # 估计周长
        perimeter = boundary.sum()
        
        # 计算约束
        if self.target_perimeter is not None:
            result = (perimeter - self.target_perimeter).abs() * self.weight
        elif self.max_perimeter is not None:
            result = F.relu(perimeter - self.max_perimeter) * self.weight
        else:
            result = perimeter * self.weight
        
        self._cached_output = result
        return result


class ManufacturingConstraintNode(Node):
    """
    综合制造约束节点
    
    结合多种制造约束，提供统一的约束接口。
    """
    
    def __init__(
        self,
        name: str,
        input_node: Node,
        min_feature_size: float = 5.0,
        resolution: float = 1.0,
        max_curvature: float = 0.1,
        threshold: float = 0.5,
        weights: Optional[dict] = None,
        device: Optional[torch.device] = None
    ):
        """
        Args:
            name: 节点名称
            input_node: 输入节点
            min_feature_size: 最小特征尺寸
            resolution: 网格分辨率
            max_curvature: 最大曲率
            threshold: 二值化阈值
            weights: 各约束的权重字典
            device: 计算设备
        """
        super().__init__(name)
        self.add_input(input_node)
        self.min_feature_size = min_feature_size
        self.resolution = resolution
        self.max_curvature = max_curvature
        self.threshold = threshold
        self.device = device or torch.device('cpu')
        
        # 默认权重
        self.weights = weights or {
            'feature_size': 1.0,
            'curvature': 0.5,
            'binary': 0.3
        }
        
        # 创建子约束节点
        self.feature_constraint = MinimumFeatureSizeNode(
            f"{name}_feature",
            input_node,
            min_feature_size=min_feature_size,
            resolution=resolution,
            threshold=threshold,
            weight=self.weights['feature_size'],
            device=device
        )
        
        self.curvature_constraint = CurvatureConstraintNode(
            f"{name}_curvature",
            input_node,
            max_curvature=max_curvature,
            threshold=threshold,
            weight=self.weights['curvature'],
            device=device
        )
        
        self.binary_constraint = BinaryConstraintNode(
            f"{name}_binary",
            input_node,
            weight=self.weights['binary'],
            device=device
        )
    
    def forward(self, **kwargs) -> TensorLike:
        """计算综合制造约束"""
        input_val = self._inputs[0].forward(**kwargs)
        
        if isinstance(input_val, dict):
            raise TypeError("ManufacturingConstraintNode 期望张量输入")
        
        # 计算各约束
        feature_violation = self.feature_constraint.forward(**kwargs)
        curvature_violation = self.curvature_constraint.forward(**kwargs)
        binary_violation = self.binary_constraint.forward(**kwargs)
        
        # 总约束
        result = feature_violation + curvature_violation + binary_violation
        
        self._cached_output = result
        return result


class GradientPenaltyNode(Node):
    """
    梯度惩罚节点
    
    惩罚设计变量的梯度，用于平滑或锐化控制。
    """
    
    def __init__(
        self,
        name: str,
        input_node: Node,
        penalty_type: str = 'total_variation',  # 'total_variation', 'l2', 'laplacian'
        weight: float = 1.0,
        device: Optional[torch.device] = None
    ):
        """
        Args:
            name: 节点名称
            input_node: 输入节点
            penalty_type: 惩罚类型
            weight: 惩罚权重
            device: 计算设备
        """
        super().__init__(name)
        self.add_input(input_node)
        self.penalty_type = penalty_type
        self.weight = weight
        self.device = device or torch.device('cpu')
    
    def forward(self, **kwargs) -> TensorLike:
        """计算梯度惩罚"""
        input_val = self._inputs[0].forward(**kwargs)
        
        if isinstance(input_val, dict):
            raise TypeError("GradientPenaltyNode 期望张量输入")
        
        x = input_val.to(self.device)
        original_dim = x.dim()
        if original_dim == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif original_dim == 3:
            x = x.unsqueeze(0)
        
        # 计算梯度
        grad_x = x[:, :, :, 1:] - x[:, :, :, :-1]
        grad_y = x[:, :, 1:, :] - x[:, :, :-1, :]
        
        if self.penalty_type == 'total_variation':
            # TV 范数: |∇x|
            penalty = grad_x.abs().mean() + grad_y.abs().mean()
            
        elif self.penalty_type == 'l2':
            # L2 范数: ||∇x||²
            penalty = (grad_x**2).mean() + (grad_y**2).mean()
            
        elif self.penalty_type == 'laplacian':
            # Laplacian 惩罚
            grad_xx = grad_x[:, :, :, 1:] - grad_x[:, :, :, :-1]
            grad_yy = grad_y[:, :, 1:, :] - grad_y[:, :, :-1, :]
            laplacian = F.pad(grad_xx, (0, 1, 0, 0)) + F.pad(grad_yy, (0, 0, 0, 1))
            penalty = laplacian.abs().mean()
            
        else:
            raise ValueError(f"不支持的惩罚类型: {self.penalty_type}")
        
        result = penalty * self.weight
        self._cached_output = result
        return result

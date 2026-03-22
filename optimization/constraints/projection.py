"""
投影约束模块

实现设计变量的投影约束，确保设计满足特定几何约束。

主要功能:
1. 二值化投影 - 将灰度设计投影到二元设计
2. 对称性投影 - 强制设计满足对称性
3. 连通性投影 - 确保设计的连通性
4. 制造约束投影 - 满足特定制造工艺要求

参考文献:
- Wang et al. (2011). "Projection filters in topology optimization"
- Sigmund (2007). "Morphology-based black and white filters"
"""

from typing import Dict, Optional, Tuple, List, Union, Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import numpy as np


@dataclass
class ProjectionConfig:
    """投影约束配置"""
    # 投影类型
    projection_type: str = "heaviside"  # 'heaviside', 'smooth_heaviside', 'error_function'
    
    # 投影参数
    threshold: float = 0.5
    sharpness: float = 10.0
    
    # 对称性
    enforce_symmetry: bool = False
    symmetry_type: str = "mirror"  # 'mirror', 'rotational', 'translational'
    symmetry_axis: str = "x"  # 'x', 'y', 'both'
    
    # 连通性
    enforce_connectivity: bool = False
    min_components: int = 1
    max_components: int = 1
    
    # 边界约束
    fixed_boundary: bool = False
    boundary_width: int = 5
    
    # 权重
    weight: float = 1.0


class HeavisideProjection(nn.Module):
    """
    Heaviside 投影
    
    实现平滑的阶跃函数投影，将连续设计投影到接近二元的分布。
    """
    
    def __init__(
        self,
        threshold: float = 0.5,
        sharpness: float = 10.0,
        smooth: bool = True,
    ):
        super().__init__()
        
        self.threshold = threshold
        self.sharpness = sharpness
        self.smooth = smooth
    
    def forward(self, rho: Tensor) -> Tensor:
        """应用 Heaviside 投影"""
        if rho.dim() == 2:
            rho = rho.unsqueeze(0).unsqueeze(0)
        
        beta = self.sharpness
        eta = self.threshold
        
        if self.smooth:
            # 平滑 Heaviside 投影
            numerator = torch.tanh(beta * eta) + torch.tanh(beta * (rho - eta))
            denominator = torch.tanh(beta * eta) + torch.tanh(beta * (1 - eta))
            return numerator / denominator
        else:
            # 硬阈值
            return (rho >= eta).float()
    
    def gradient(self, rho: Tensor) -> Tensor:
        """计算投影的梯度（用于链式法则）"""
        beta = self.sharpness
        eta = self.threshold
        
        # d/dρ of smooth Heaviside
        numerator_grad = beta * (1 - torch.tanh(beta * (rho - eta)) ** 2)
        denominator = torch.tanh(beta * eta) + torch.tanh(beta * (1 - eta))
        
        return numerator_grad / denominator


class ErrorFunctionProjection(nn.Module):
    """
    误差函数投影
    
    使用误差函数实现更平滑的投影。
    """
    
    def __init__(
        self,
        threshold: float = 0.5,
        scale: float = 5.0,
    ):
        super().__init__()
        
        self.threshold = threshold
        self.scale = scale
    
    def forward(self, rho: Tensor) -> Tensor:
        """应用误差函数投影"""
        if rho.dim() == 2:
            rho = rho.unsqueeze(0).unsqueeze(0)
        
        # 使用 torch 的误差函数近似
        # erf(z) ≈ tanh(√(4/π) * z)
        scaled = self.scale * (rho - self.threshold)
        projected = 0.5 * (1 + torch.tanh(1.128 * scaled))
        
        return projected


class SymmetryProjection(nn.Module):
    """
    对称性投影
    
    强制设计满足指定的对称性约束。
    """
    
    def __init__(
        self,
        symmetry_type: str = "mirror",
        symmetry_axis: str = "x",
        strength: float = 1.0,
    ):
        super().__init__()
        
        self.symmetry_type = symmetry_type
        self.symmetry_axis = symmetry_axis
        self.strength = strength
    
    def forward(self, rho: Tensor) -> Tensor:
        """应用对称性投影"""
        if rho.dim() == 2:
            rho = rho.unsqueeze(0).unsqueeze(0)
        
        if self.symmetry_type == "mirror":
            return self._mirror_symmetry(rho)
        elif self.symmetry_type == "rotational":
            return self._rotational_symmetry(rho)
        elif self.symmetry_type == "translational":
            return self._translational_symmetry(rho)
        else:
            return rho
    
    def _mirror_symmetry(self, rho: Tensor) -> Tensor:
        """镜像对称"""
        if self.symmetry_axis == "x":
            # 沿 x 轴（垂直轴）对称
            flipped = torch.flip(rho, dims=[-1])
            return self.strength * 0.5 * (rho + flipped) + (1 - self.strength) * rho
        elif self.symmetry_axis == "y":
            # 沿 y 轴（水平轴）对称
            flipped = torch.flip(rho, dims=[-2])
            return self.strength * 0.5 * (rho + flipped) + (1 - self.strength) * rho
        elif self.symmetry_axis == "both":
            # 双轴对称
            flipped_x = torch.flip(rho, dims=[-1])
            flipped_y = torch.flip(rho, dims=[-2])
            flipped_xy = torch.flip(rho, dims=[-1, -2])
            return self.strength * 0.25 * (rho + flipped_x + flipped_y + flipped_xy) + (1 - self.strength) * rho
        else:
            return rho
    
    def _rotational_symmetry(self, rho: Tensor) -> Tensor:
        """旋转对称（180度）"""
        rotated = torch.rot90(rho, k=2, dims=[-2, -1])
        return self.strength * 0.5 * (rho + rotated) + (1 - self.strength) * rho
    
    def _translational_symmetry(self, rho: Tensor, period: int = 10) -> Tensor:
        """平移对称（周期性）"""
        # 简化实现：平均平移版本
        shifted = torch.roll(rho, shifts=period, dims=-1)
        return self.strength * 0.5 * (rho + shifted) + (1 - self.strength) * rho
    
    def compute_violation(self, rho: Tensor) -> Tensor:
        """计算对称性违反程度"""
        projected = self.forward(rho)
        violation = torch.abs(rho - projected)
        return violation.mean(dim=(1, 2, 3))


class ConnectivityProjection(nn.Module):
    """
    连通性投影
    
    确保设计满足连通性要求，避免孤立区域。
    """
    
    def __init__(
        self,
        min_components: int = 1,
        max_components: int = 1,
        threshold: float = 0.5,
    ):
        super().__init__()
        
        self.min_components = min_components
        self.max_components = max_components
        self.threshold = threshold
    
    def forward(self, rho: Tensor) -> Tensor:
        """
        应用连通性投影
        
        注意：这是一个软约束，通过添加惩罚项实现
        """
        # 连通性投影通常需要迭代算法
        # 这里返回原始设计，约束在损失函数中计算
        return rho
    
    def compute_violation(self, rho: Tensor) -> Tensor:
        """计算连通性违反"""
        binary = (rho > self.threshold).float()
        
        # 使用连通分量分析
        num_components = self._count_components(binary)
        
        # 违反约束
        violation = torch.zeros(rho.shape[0], device=rho.device)
        
        for i, n in enumerate(num_components):
            if n < self.min_components:
                violation[i] = self.min_components - n
            elif n > self.max_components:
                violation[i] = n - self.max_components
        
        return violation
    
    def _count_components(self, binary: Tensor) -> List[int]:
        """统计连通分量数量（简化实现）"""
        components = []
        
        for i in range(binary.shape[0]):
            # 使用形态学方法近似
            img = binary[i, 0].cpu().numpy()
            
            # 简化：使用像素计数估计
            # 实际应使用连通分量标记算法
            filled_area = img.sum()
            perimeter = self._estimate_perimeter(img)
            
            # 粗略估计分量数
            if perimeter > 0:
                estimated = max(1, int(filled_area / (perimeter * 2)))
            else:
                estimated = 0
            
            components.append(estimated)
        
        return components
    
    def _estimate_perimeter(self, img: np.ndarray) -> float:
        """估计周长"""
        # 使用梯度近似周长
        grad_x = np.abs(np.diff(img, axis=1)).sum()
        grad_y = np.abs(np.diff(img, axis=0)).sum()
        return grad_x + grad_y


class BoundaryConstraint(nn.Module):
    """
    边界约束
    
    固定设计区域的边界值。
    """
    
    def __init__(
        self,
        boundary_width: int = 5,
        boundary_value: float = 0.0,
        constraint_type: str = "fixed",  # 'fixed', 'periodic', 'reflecting'
    ):
        super().__init__()
        
        self.boundary_width = boundary_width
        self.boundary_value = boundary_value
        self.constraint_type = constraint_type
    
    def forward(self, rho: Tensor) -> Tensor:
        """应用边界约束"""
        if rho.dim() == 2:
            rho = rho.unsqueeze(0).unsqueeze(0)
        
        if self.constraint_type == "fixed":
            return self._apply_fixed_boundary(rho)
        elif self.constraint_type == "periodic":
            return self._apply_periodic_boundary(rho)
        else:
            return rho
    
    def _apply_fixed_boundary(self, rho: Tensor) -> Tensor:
        """固定边界值"""
        w = self.boundary_width
        val = self.boundary_value
        
        rho[:, :, :w, :] = val
        rho[:, :, -w:, :] = val
        rho[:, :, :, :w] = val
        rho[:, :, :, -w:] = val
        
        return rho
    
    def _apply_periodic_boundary(self, rho: Tensor) -> Tensor:
        """周期性边界"""
        w = self.boundary_width
        
        # 复制对边值
        rho[:, :, :w, :] = rho[:, :, -2*w:-w, :]
        rho[:, :, -w:, :] = rho[:, :, w:2*w, :]
        rho[:, :, :, :w] = rho[:, :, :, -2*w:-w]
        rho[:, :, :, -w:] = rho[:, :, :, w:2*w]
        
        return rho
    
    def compute_violation(self, rho: Tensor) -> Tensor:
        """计算边界约束违反"""
        w = self.boundary_width
        
        boundary = torch.cat([
            rho[:, :, :w, :].flatten(1),
            rho[:, :, -w:, :].flatten(1),
            rho[:, :, :, :w].flatten(1),
            rho[:, :, :, -w:].flatten(1),
        ], dim=1)
        
        violation = torch.abs(boundary - self.boundary_value)
        return violation.mean(dim=1)


class ManufacturingProjection(nn.Module):
    """
    制造约束投影
    
    确保设计满足特定制造工艺的要求。
    """
    
    def __init__(
        self,
        process_type: str = "lithography",  # 'lithography', 'etching', '3d_print'
        min_feature_size: float = 5.0,
        aspect_ratio_limit: float = 10.0,
        overhang_limit: float = 45.0,  # degrees
    ):
        super().__init__()
        
        self.process_type = process_type
        self.min_feature_size = min_feature_size
        self.aspect_ratio_limit = aspect_ratio_limit
        self.overhang_limit = overhang_limit
    
    def forward(self, rho: Tensor) -> Tensor:
        """应用制造约束投影"""
        if self.process_type == "lithography":
            return self._lithography_projection(rho)
        elif self.process_type == "etching":
            return self._etching_projection(rho)
        else:
            return rho
    
    def _lithography_projection(self, rho: Tensor) -> Tensor:
        """
        光刻制造约束
        
        限制最小特征尺寸，确保可制造性
        """
        # 使用形态学开闭运算
        kernel_size = int(self.min_feature_size)
        
        # 开运算
        eroded = -F.max_pool2d(-rho, kernel_size, stride=1, padding=kernel_size // 2)
        opened = F.max_pool2d(eroded, kernel_size, stride=1, padding=kernel_size // 2)
        
        # 闭运算
        dilated = F.max_pool2d(rho, kernel_size, stride=1, padding=kernel_size // 2)
        closed = -F.max_pool2d(-dilated, kernel_size, stride=1, padding=kernel_size // 2)
        
        # 组合
        return 0.5 * (opened + closed)
    
    def _etching_projection(self, rho: Tensor) -> Tensor:
        """
        刻蚀制造约束
        
        限制深宽比和悬挂角度
        """
        # 简化实现：限制水平方向的变化率
        grad_x = torch.abs(rho[:, :, :, 1:] - rho[:, :, :, :-1])
        grad_y = torch.abs(rho[:, :, 1:, :] - rho[:, :, :-1, :])
        
        # 限制梯度过大的区域
        max_grad = 1.0 / self.aspect_ratio_limit
        
        # 平滑处理
        smoothed = self._lithography_projection(rho)
        
        return smoothed
    
    def compute_violation(self, rho: Tensor) -> Dict[str, Tensor]:
        """计算制造约束违反"""
        violations = {}
        
        # 最小特征尺寸违反
        violations['min_feature'] = self._check_min_feature(rho)
        
        # 深宽比违反
        violations['aspect_ratio'] = self._check_aspect_ratio(rho)
        
        return violations
    
    def _check_min_feature(self, rho: Tensor) -> Tensor:
        """检查最小特征尺寸"""
        kernel_size = int(self.min_feature_size)
        
        # 检测过小特征
        eroded = -F.max_pool2d(-rho, kernel_size, stride=1, padding=kernel_size // 2)
        dilated = F.max_pool2d(eroded, kernel_size, stride=1, padding=kernel_size // 2)
        
        violation = torch.abs(rho - dilated)
        return violation.mean(dim=(1, 2, 3))
    
    def _check_aspect_ratio(self, rho: Tensor) -> Tensor:
        """检查深宽比"""
        grad_x = torch.abs(rho[:, :, :, 1:] - rho[:, :, :, :-1])
        grad_y = torch.abs(rho[:, :, 1:, :] - rho[:, :, :-1, :])
        
        max_grad = torch.maximum(
            F.pad(grad_x, (0, 1), value=0),
            F.pad(grad_y, (0, 0, 0, 1), value=0)
        )
        
        violation = F.relu(max_grad - 1.0 / self.aspect_ratio_limit)
        return violation.mean(dim=(1, 2, 3))


class ProjectionConstraint(nn.Module):
    """
    综合投影约束
    
    组合多种投影约束。
    """
    
    def __init__(self, config: Optional[ProjectionConfig] = None):
        super().__init__()
        self.config = config or ProjectionConfig()
        
        # Heaviside 投影
        self.heaviside = HeavisideProjection(
            self.config.threshold,
            self.config.sharpness
        )
        
        # 对称性投影
        if self.config.enforce_symmetry:
            self.symmetry = SymmetryProjection(
                self.config.symmetry_type,
                self.config.symmetry_axis
            )
        else:
            self.symmetry = None
        
        # 边界约束
        if self.config.fixed_boundary:
            self.boundary = BoundaryConstraint(self.config.boundary_width)
        else:
            self.boundary = None
    
    def forward(self, rho: Tensor) -> Tensor:
        """应用所有投影约束"""
        # Heaviside 投影
        projected = self.heaviside(rho)
        
        # 对称性投影
        if self.symmetry is not None:
            projected = self.symmetry(projected)
        
        # 边界约束
        if self.boundary is not None:
            projected = self.boundary(projected)
        
        return projected
    
    def compute_loss(self, rho: Tensor) -> Tensor:
        """计算约束损失"""
        projected = self.forward(rho)
        
        # 投影前后差异作为约束违反
        violation = torch.abs(rho - projected)
        
        return violation.mean() * self.config.weight


# ============================================================================
# 便捷函数
# ============================================================================

def create_projection(
    projection_type: str = "heaviside",
    threshold: float = 0.5,
    sharpness: float = 10.0,
    **kwargs
) -> nn.Module:
    """创建投影模块"""
    if projection_type == "heaviside":
        return HeavisideProjection(threshold, sharpness)
    elif projection_type == "error_function":
        return ErrorFunctionProjection(threshold, sharpness / 10)
    elif projection_type == "symmetry":
        return SymmetryProjection(**kwargs)
    elif projection_type == "manufacturing":
        return ManufacturingProjection(**kwargs)
    else:
        return HeavisideProjection(threshold, sharpness)


def project_to_binary(
    design: Union[Tensor, np.ndarray],
    threshold: float = 0.5
) -> Union[Tensor, np.ndarray]:
    """
    将设计投影到二元分布
    
    Args:
        design: 设计变量
        threshold: 阈值
        
    Returns:
        二元设计
    """
    is_numpy = isinstance(design, np.ndarray)
    if is_numpy:
        design = torch.from_numpy(design).float()
    
    projection = HeavisideProjection(threshold, sharpness=100.0, smooth=False)
    binary = projection(design)
    
    if is_numpy:
        return binary.squeeze().numpy()
    return binary.squeeze()

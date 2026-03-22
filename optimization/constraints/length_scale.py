"""
长度尺度约束模块

实现光子学器件设计中的最小特征尺寸约束。
确保设计可制造，避免过于细微的结构。

主要方法:
1. 形态学过滤 - 腐蚀/膨胀操作
2. Helmholtz 过滤 - 偏微分方程平滑
3. 投影方法 - Heaviside 投影
4. 灵敏度过滤 - 基于灵敏度的过滤

参考文献:
- Bourdin (2001). "Filters in topology optimization"
- Guest et al. (2004). "Achieving minimum length scale in topology optimization"
- Lazarov & Sigmund (2011). "Filters in topology optimization based on Helmholtz-type differential equations"
"""

from typing import Dict, Optional, Tuple, List, Union
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import numpy as np


@dataclass
class LengthScaleConfig:
    """长度尺度约束配置"""
    # 最小特征尺寸 (像素)
    min_feature_size: float = 5.0
    
    # 过滤半径 (像素)
    filter_radius: float = 3.0
    
    # 过滤类型
    filter_type: str = "helmholtz"  # 'helmholtz', 'density', 'morphology', 'sensitivity'
    
    # Helmholtz 参数
    helmholtz_beta: float = 1.0
    
    # 投影参数
    projection_threshold: float = 0.5
    projection_sharpness: float = 10.0
    
    # 约束权重
    weight: float = 1.0
    
    # 是否使用连续投影
    continuation: bool = True
    continuation_steps: int = 10


class HelmholtzFilter(nn.Module):
    """
    Helmholtz 过滤器
    
    通过求解 Helmholtz 方程实现平滑过滤：
    -r² ∇²ρ_filtered + ρ_filtered = ρ
    
    其中 r 是过滤半径。
    """
    
    def __init__(self, radius: float = 3.0, beta: float = 1.0):
        super().__init__()
        
        self.radius = radius
        self.beta = beta
        
        # Jacobi 迭代参数
        self.max_iterations = 50
        self.tolerance = 1e-6
    
    def forward(self, rho: Tensor) -> Tensor:
        """
        应用 Helmholtz 过滤
        
        Args:
            rho: 输入密度场 [B, 1, H, W] 或 [H, W]
            
        Returns:
            过滤后的密度场
        """
        if rho.dim() == 2:
            rho = rho.unsqueeze(0).unsqueeze(0)
        
        # 初始化
        rho_filtered = rho.clone()
        
        # Jacobi 迭代
        r_sq = self.radius ** 2
        
        for _ in range(self.max_iterations):
            # 拉普拉斯算子
            laplacian = self._compute_laplacian(rho_filtered)
            
            # 更新
            rho_new = (rho + r_sq * laplacian) / (1 + 4 * r_sq * self.beta)
            
            # 检查收敛
            if torch.abs(rho_new - rho_filtered).max() < self.tolerance:
                break
            
            rho_filtered = rho_new
        
        return rho_filtered
    
    def _compute_laplacian(self, x: Tensor) -> Tensor:
        """计算拉普拉斯算子（有限差分）"""
        laplacian = (
            F.pad(x, (1, 1, 0, 0), mode='replicate')[:, :, :, 2:] +
            F.pad(x, (1, 1, 0, 0), mode='replicate')[:, :, :, :-2] +
            F.pad(x, (0, 0, 1, 1), mode='replicate')[:, :, 2:, :] +
            F.pad(x, (0, 0, 1, 1), mode='replicate')[:, :, :-2, :] -
            4 * x
        )
        return laplacian


class DensityFilter(nn.Module):
    """
    密度过滤器
    
    基于卷积的密度过滤，每个点的新值为邻居的加权平均。
    """
    
    def __init__(self, radius: float = 3.0):
        super().__init__()
        
        self.radius = int(radius)
        self.kernel = None
    
    def _create_kernel(self, device: torch.device, dtype: torch.dtype) -> Tensor:
        """创建过滤核"""
        r = self.radius
        size = 2 * r + 1
        
        # 距离权重
        y, x = torch.meshgrid(
            torch.arange(size, dtype=dtype),
            torch.arange(size, dtype=dtype),
            indexing='ij'
        )
        
        dist = torch.sqrt((y - r) ** 2 + (x - r) ** 2)
        kernel = (r - dist).clamp(min=0)
        kernel = kernel / kernel.sum()
        
        return kernel.view(1, 1, size, size).to(device)
    
    def forward(self, rho: Tensor) -> Tensor:
        """应用密度过滤"""
        if rho.dim() == 2:
            rho = rho.unsqueeze(0).unsqueeze(0)
        
        if self.kernel is None:
            self.kernel = self._create_kernel(rho.device, rho.dtype)
        
        # 卷积过滤
        filtered = F.conv2d(rho, self.kernel, padding=self.radius)
        
        return filtered


class MorphologyFilter(nn.Module):
    """
    形态学过滤器
    
    使用开/闭运算实现最小特征尺寸控制。
    """
    
    def __init__(self, radius: float = 3.0):
        super().__init__()
        
        self.radius = int(radius)
        self.kernel = None
    
    def _create_kernel(self, device: torch.device) -> Tensor:
        """创建圆形结构元素"""
        r = self.radius
        size = 2 * r + 1
        
        y, x = torch.meshgrid(
            torch.arange(size, device=device),
            torch.arange(size, device=device),
            indexing='ij'
        )
        
        dist = torch.sqrt((y - r) ** 2 + (x - r) ** 2)
        kernel = (dist <= r).float()
        
        return kernel.unsqueeze(0).unsqueeze(0)
    
    def forward(self, rho: Tensor) -> Tensor:
        """
        应用形态学过滤（开-闭运算）
        
        开运算 = 腐蚀 -> 膨胀：去除小突起
        闭运算 = 膨胀 -> 腐蚀：填充小孔洞
        """
        if rho.dim() == 2:
            rho = rho.unsqueeze(0).unsqueeze(0)
        
        if self.kernel is None:
            self.kernel = self._create_kernel(rho.device)
        
        # 开运算
        opened = self._dilate(self._erode(rho))
        
        # 闭运算
        closed = self._erode(self._dilate(rho))
        
        # 平均
        filtered = 0.5 * (opened + closed)
        
        return filtered
    
    def _erode(self, x: Tensor) -> Tensor:
        """腐蚀操作"""
        return -F.max_pool2d(
            -x, 
            self.kernel.shape[-1], 
            stride=1, 
            padding=self.radius
        )
    
    def _dilate(self, x: Tensor) -> Tensor:
        """膨胀操作"""
        return F.max_pool2d(
            x, 
            self.kernel.shape[-1], 
            stride=1, 
            padding=self.radius
        )


class HeavisideProjection(nn.Module):
    """
    Heaviside 投影
    
    将连续密度场投影到接近二元的分布，
    实现清晰的边界和长度尺度控制。
    
    η_β(ρ) = (tanh(β·η) + tanh(β·(ρ-η))) / (tanh(β·η) + tanh(β·(1-η)))
    
    其中 β 控制锐度，η 是阈值。
    """
    
    def __init__(
        self,
        threshold: float = 0.5,
        sharpness: float = 10.0,
        continuation: bool = True,
        max_sharpness: float = 100.0,
    ):
        super().__init__()
        
        self.threshold = threshold
        self.sharpness = sharpness
        self.max_sharpness = max_sharpness
        self.continuation = continuation
        self.current_sharpness = sharpness
    
    def forward(self, rho: Tensor) -> Tensor:
        """应用 Heaviside 投影"""
        beta = self.current_sharpness
        eta = self.threshold
        
        numerator = torch.tanh(beta * eta) + torch.tanh(beta * (rho - eta))
        denominator = torch.tanh(beta * eta) + torch.tanh(beta * (1 - eta))
        
        return numerator / denominator
    
    def increase_sharpness(self, factor: float = 1.5):
        """增加投影锐度（延续策略）"""
        if self.continuation:
            self.current_sharpness = min(
                self.current_sharpness * factor,
                self.max_sharpness
            )
    
    def reset_sharpness(self):
        """重置锐度"""
        self.current_sharpness = self.sharpness


class SensitivityFilter(nn.Module):
    """
    灵敏度过滤器
    
    过滤目标函数对设计变量的灵敏度，
    防止棋盘格模式和尺寸过小的特征。
    """
    
    def __init__(self, radius: float = 3.0):
        super().__init__()
        
        self.radius = int(radius)
        self.density_filter = DensityFilter(radius)
    
    def forward(self, sensitivity: Tensor, rho: Tensor) -> Tensor:
        """
        过滤灵敏度
        
        Args:
            sensitivity: 目标函数对设计变量的灵敏度
            rho: 当前密度场
            
        Returns:
            过滤后的灵敏度
        """
        # 加权平均
        rho_filtered = self.density_filter(rho)
        
        # 灵敏度过滤
        weighted_sensitivity = sensitivity * rho
        filtered_weighted = self.density_filter(weighted_sensitivity)
        
        # 归一化
        filtered_sensitivity = filtered_weighted / (rho_filtered + 1e-8)
        
        return filtered_sensitivity


class LengthScaleConstraint(nn.Module):
    """
    长度尺度约束
    
    综合使用过滤和投影实现最小特征尺寸控制。
    """
    
    def __init__(self, config: Optional[LengthScaleConfig] = None):
        super().__init__()
        self.config = config or LengthScaleConfig()
        
        # 选择过滤类型
        if self.config.filter_type == "helmholtz":
            self.filter = HelmholtzFilter(
                self.config.filter_radius,
                self.config.helmholtz_beta
            )
        elif self.config.filter_type == "density":
            self.filter = DensityFilter(self.config.filter_radius)
        elif self.config.filter_type == "morphology":
            self.filter = MorphologyFilter(self.config.filter_radius)
        else:
            self.filter = DensityFilter(self.config.filter_radius)
        
        # 投影
        self.projection = HeavisideProjection(
            self.config.projection_threshold,
            self.config.projection_sharpness,
            self.config.continuation
        )
    
    def forward(
        self,
        design: Tensor,
        return_filtered: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        应用长度尺度约束
        
        Args:
            design: 设计变量
            return_filtered: 是否返回过滤后的结果
            
        Returns:
            约束后的设计变量（和可选的过滤结果）
        """
        if design.dim() == 2:
            design = design.unsqueeze(0).unsqueeze(0)
        
        # 过滤
        filtered = self.filter(design)
        
        # 投影
        projected = self.projection(filtered)
        
        # 计算约束违反（检测过小特征）
        violation = self._detect_small_features(projected)
        
        if return_filtered:
            return projected, filtered, violation
        
        return projected
    
    def _detect_small_features(self, design: Tensor) -> Tensor:
        """检测过小的特征"""
        # 使用形态学开运算检测小突起
        kernel_size = int(self.config.min_feature_size)
        
        eroded = -F.max_pool2d(
            -design,
            kernel_size,
            stride=1,
            padding=kernel_size // 2
        )
        dilated = F.max_pool2d(
            eroded,
            kernel_size,
            stride=1,
            padding=kernel_size // 2
        )
        
        # 开运算与原设计的差异表示小特征
        small_features = design - dilated
        
        # 小孔洞检测（使用闭运算）
        dilated_first = F.max_pool2d(
            design,
            kernel_size,
            stride=1,
            padding=kernel_size // 2
        )
        eroded_then = -F.max_pool2d(
            -dilated_first,
            kernel_size,
            stride=1,
            padding=kernel_size // 2
        )
        
        small_holes = eroded_then - design
        
        # 总违规
        violation = small_features.abs() + small_holes.abs()
        
        return violation
    
    def compute_constraint(self, design: Tensor) -> Dict[str, Tensor]:
        """计算约束值"""
        projected, filtered, violation = self.forward(design, return_filtered=True)
        
        return {
            'filtered': filtered,
            'projected': projected,
            'violation': violation,
            'violation_sum': violation.sum(dim=(2, 3)),
            'violation_mean': violation.mean(dim=(2, 3)),
        }
    
    def compute_loss(self, design: Tensor) -> Tensor:
        """计算约束损失"""
        result = self.compute_constraint(design)
        return result['violation_sum'] * self.config.weight
    
    def continuation_step(self):
        """延续策略：增加投影锐度"""
        self.projection.increase_sharpness()


class RobustLengthScaleConstraint(LengthScaleConstraint):
    """
    鲁棒长度尺度约束
    
    考虑制造误差的长度尺度约束。
    """
    
    def __init__(
        self,
        config: Optional[LengthScaleConfig] = None,
        erosion_radius: float = 1.0,
        dilation_radius: float = 1.0,
    ):
        super().__init__(config)
        
        self.erosion_radius = int(erosion_radius)
        self.dilation_radius = int(dilation_radius)
    
    def forward(
        self,
        design: Tensor,
        return_all: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor]]:
        """
        计算鲁棒设计
        
        Args:
            design: 设计变量
            return_all: 是否返回所有变体
            
        Returns:
            设计变体（原始、腐蚀、膨胀）
        """
        # 标准过滤和投影
        nominal = super().forward(design)
        
        # 腐蚀变体
        eroded = self._erode_design(nominal)
        
        # 膨胀变体
        dilated = self._dilate_design(nominal)
        
        if return_all:
            return nominal, eroded, dilated
        
        # 返回最坏情况
        return nominal
    
    def _erode_design(self, design: Tensor) -> Tensor:
        """腐蚀设计"""
        return -F.max_pool2d(
            -design,
            2 * self.erosion_radius + 1,
            stride=1,
            padding=self.erosion_radius
        )
    
    def _dilate_design(self, design: Tensor) -> Tensor:
        """膨胀设计"""
        return F.max_pool2d(
            design,
            2 * self.dilation_radius + 1,
            stride=1,
            padding=self.dilation_radius
        )
    
    def compute_robust_loss(
        self,
        design: Tensor,
        objective_fn: callable
    ) -> Tensor:
        """计算鲁棒损失"""
        nominal, eroded, dilated = self.forward(design, return_all=True)
        
        # 评估三种情况
        obj_nominal = objective_fn(nominal)
        obj_eroded = objective_fn(eroded)
        obj_dilated = objective_fn(dilated)
        
        # 最坏情况
        worst_case = torch.maximum(
            torch.maximum(obj_nominal, obj_eroded),
            obj_dilated
        )
        
        return worst_case


# ============================================================================
# 便捷函数
# ============================================================================

def create_length_scale_constraint(
    min_feature_size: float = 5.0,
    filter_type: str = "helmholtz",
    **kwargs
) -> LengthScaleConstraint:
    """创建长度尺度约束"""
    config = LengthScaleConfig(
        min_feature_size=min_feature_size,
        filter_type=filter_type,
        **kwargs
    )
    return LengthScaleConstraint(config)


def apply_density_filter(
    design: Union[Tensor, np.ndarray],
    radius: float = 3.0
) -> Union[Tensor, np.ndarray]:
    """
    应用密度过滤
    
    Args:
        design: 设计变量
        radius: 过滤半径
        
    Returns:
        过滤后的设计
    """
    is_numpy = isinstance(design, np.ndarray)
    if is_numpy:
        design = torch.from_numpy(design).float()
    
    filter_obj = DensityFilter(radius)
    filtered = filter_obj(design)
    
    if is_numpy:
        return filtered.squeeze().numpy()
    return filtered.squeeze()


def check_min_feature_size(
    design: Union[Tensor, np.ndarray],
    min_size: float = 5.0
) -> Tuple[float, bool]:
    """
    检查设计是否满足最小特征尺寸要求
    
    Args:
        design: 设计变量
        min_size: 最小特征尺寸（像素）
        
    Returns:
        violation: 违反程度
        satisfied: 是否满足约束
    """
    if isinstance(design, np.ndarray):
        design = torch.from_numpy(design).float()
    
    constraint = LengthScaleConstraint(LengthScaleConfig(min_feature_size=min_size))
    result = constraint.compute_constraint(design)
    
    violation = result['violation_sum'].item()
    satisfied = violation < 1e-6
    
    return violation, satisfied

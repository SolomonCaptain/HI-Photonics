"""
曲率约束模块

实现光子学器件设计中的曲率相关约束，确保设计的可制造性。

主要功能:
1. 曲率半径约束 - 确保曲线/边界不过于尖锐
2. 最小特征尺寸约束 - 通过曲率间接实现
3. 边界平滑度约束 - 限制边界的高频振荡

应用场景:
- 波导弯曲设计
- 谐振腔边界优化
- 自由曲面光学设计

参考文献:
- Lazarov et al. (2016). "Length scale control in topology optimization"
- Wang et al. (2020). "Manufacturability in nanophotonics inverse design"
"""

from typing import Dict, Optional, Tuple, List, Union
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import numpy as np


@dataclass
class CurvatureConfig:
    """曲率约束配置"""
    # 最小曲率半径 (像素)
    min_radius: float = 5.0
    
    # 最大曲率
    max_curvature: float = 0.2
    
    # 网格分辨率 (μm/pixel)
    resolution: float = 0.01
    
    # 约束权重
    weight: float = 1.0
    
    # 计算方法
    method: str = "finite_difference"  # 'finite_difference', 'morphology'
    
    # 边界处理
    boundary_mode: str = "reflect"  # 'reflect', 'constant', 'periodic'


class CurvatureConstraint(nn.Module):
    """
    曲率约束
    
    确保设计边界的曲率不超过制造限制。
    """
    
    def __init__(self, config: Optional[CurvatureConfig] = None):
        super().__init__()
        self.config = config or CurvatureConfig()
        
        # 创建 Sobel 滤波器用于梯度计算
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))
    
    def forward(self, design: Tensor) -> Dict[str, Tensor]:
        """
        计算曲率约束
        
        Args:
            design: 设计变量 [B, 1, H, W] 或 [H, W]
            
        Returns:
            约束字典
        """
        if design.dim() == 2:
            design = design.unsqueeze(0).unsqueeze(0)
        elif design.dim() == 3:
            design = design.unsqueeze(1)
        
        # 计算梯度
        grad_x = F.conv2d(design, self.sobel_x, padding=1)
        grad_y = F.conv2d(design, self.sobel_y, padding=1)
        
        # 计算二阶导数
        grad_xx = F.conv2d(grad_x, self.sobel_x, padding=1)
        grad_yy = F.conv2d(grad_y, self.sobel_y, padding=1)
        grad_xy = F.conv2d(grad_x, self.sobel_y, padding=1)
        
        # 曲率计算
        # κ = (f_xx * f_y^2 - 2 * f_x * f_y * f_xy + f_yy * f_x^2) / (f_x^2 + f_y^2)^(3/2)
        grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
        
        numerator = (grad_xx * grad_y ** 2 - 
                    2 * grad_x * grad_y * grad_xy + 
                    grad_yy * grad_x ** 2)
        denominator = grad_mag ** 3 + 1e-8
        
        curvature = torch.abs(numerator / denominator)
        
        # 约束违反
        violation = F.relu(curvature - self.config.max_curvature)
        
        # 只在边界附近计算（梯度大的区域）
        boundary_mask = grad_mag > 0.1
        boundary_violation = violation * boundary_mask.float()
        
        constraints = {
            'curvature': curvature,
            'curvature_violation': boundary_violation,
            'curvature_max': curvature.amax(dim=(2, 3)),
            'curvature_mean': boundary_violation.mean(dim=(2, 3)),
        }
        
        # 总约束值
        constraints['total'] = constraints['curvature_mean'] * self.config.weight
        
        return constraints
    
    def compute_loss(self, design: Tensor) -> Tensor:
        """计算曲率约束损失"""
        constraints = self.forward(design)
        return constraints['total'].mean()


class MinimumRadiusConstraint(nn.Module):
    """
    最小曲率半径约束
    
    确保设计中的弯曲结构满足最小半径要求。
    """
    
    def __init__(
        self,
        min_radius: float = 5.0,
        resolution: float = 0.01,
        weight: float = 1.0,
    ):
        super().__init__()
        
        self.min_radius = min_radius
        self.resolution = resolution
        self.weight = weight
        
        # 最小曲率 = 1 / 最小半径
        self.max_curvature = 1.0 / (min_radius * resolution)
    
    def forward(self, design: Tensor) -> Dict[str, Tensor]:
        """
        计算最小半径约束
        
        Args:
            design: 设计变量
            
        Returns:
            约束字典
        """
        if design.dim() == 2:
            design = design.unsqueeze(0).unsqueeze(0)
        
        # 计算边界曲率
        curvature = self._compute_boundary_curvature(design)
        
        # 违反约束的曲率
        violation = F.relu(curvature - self.max_curvature)
        
        constraints = {
            'curvature': curvature,
            'violation': violation,
            'max_curvature': curvature.amax(dim=(2, 3)),
            'violation_sum': violation.sum(dim=(2, 3)),
        }
        
        constraints['total'] = constraints['violation_sum'] * self.weight
        
        return constraints
    
    def _compute_boundary_curvature(self, design: Tensor) -> Tensor:
        """计算边界曲率"""
        # 使用形态学方法提取边界
        kernel_size = 3
        
        # 腐蚀
        eroded = -F.max_pool2d(-design, kernel_size, stride=1, padding=kernel_size // 2)
        
        # 边界
        boundary = design - eroded
        
        # 计算边界曲率
        # 使用水平集方法
        grad_x = self._gradient_x(boundary)
        grad_y = self._gradient_y(boundary)
        
        grad_xx = self._gradient_x(grad_x)
        grad_yy = self._gradient_y(grad_y)
        grad_xy = self._gradient_y(grad_x)
        
        grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
        
        numerator = torch.abs(grad_xx * grad_y ** 2 - 
                             2 * grad_x * grad_y * grad_xy + 
                             grad_yy * grad_x ** 2)
        denominator = grad_mag ** 3 + 1e-8
        
        curvature = numerator / denominator
        
        # 只保留边界区域的曲率
        boundary_mask = boundary > 0.01
        curvature = curvature * boundary_mask.float()
        
        return curvature
    
    def _gradient_x(self, x: Tensor) -> Tensor:
        """计算 x 方向梯度"""
        return x[:, :, :, 2:] - x[:, :, :, :-2]
    
    def _gradient_y(self, x: Tensor) -> Tensor:
        """计算 y 方向梯度"""
        return x[:, :, 2:, :] - x[:, :, :-2, :]


class BoundarySmoothnessConstraint(nn.Module):
    """
    边界平滑度约束
    
    限制边界的高频振荡，确保平滑过渡。
    """
    
    def __init__(
        self,
        smoothness_weight: float = 0.1,
        high_freq_penalty: float = 1.0,
        weight: float = 1.0,
    ):
        super().__init__()
        
        self.smoothness_weight = smoothness_weight
        self.high_freq_penalty = high_freq_penalty
        self.weight = weight
    
    def forward(self, design: Tensor) -> Dict[str, Tensor]:
        """
        计算边界平滑度约束
        
        Args:
            design: 设计变量
            
        Returns:
            约束字典
        """
        if design.dim() == 2:
            design = design.unsqueeze(0).unsqueeze(0)
        
        # 提取边界
        boundary = self._extract_boundary(design)
        
        # 计算边界的傅里叶变换
        fft = torch.fft.fft2(boundary)
        fft_shifted = torch.fft.fftshift(fft)
        
        # 高频能量
        H, W = design.shape[-2:]
        center_h, center_w = H // 2, W // 2
        
        # 创建高通滤波器
        y, x = torch.meshgrid(
            torch.arange(H, device=design.device),
            torch.arange(W, device=design.device),
            indexing='ij'
        )
        
        distance = torch.sqrt((y - center_h) ** 2 + (x - center_w) ** 2)
        high_pass = (distance > min(H, W) // 4).float()
        
        # 高频能量
        high_freq_energy = (torch.abs(fft_shifted) ** 2 * high_pass).sum(dim=(2, 3))
        total_energy = (torch.abs(fft_shifted) ** 2).sum(dim=(2, 3)) + 1e-8
        
        # 高频比例
        high_freq_ratio = high_freq_energy / total_energy
        
        # 梯度平滑度
        grad_x = design[:, :, :, 1:] - design[:, :, :, :-1]
        grad_y = design[:, :, 1:, :] - design[:, :, :-1, :]
        
        smoothness = torch.sqrt(grad_x ** 2 + grad_y ** 2).mean(dim=(2, 3))
        
        constraints = {
            'high_freq_ratio': high_freq_ratio,
            'smoothness': smoothness,
            'high_freq_violation': F.relu(high_freq_ratio - 0.1),
        }
        
        constraints['total'] = (
            self.high_freq_penalty * constraints['high_freq_violation'] +
            self.smoothness_weight * smoothness
        ) * self.weight
        
        return constraints
    
    def _extract_boundary(self, design: Tensor) -> Tensor:
        """提取边界"""
        kernel_size = 3
        
        dilated = F.max_pool2d(design, kernel_size, stride=1, padding=kernel_size // 2)
        eroded = -F.max_pool2d(-design, kernel_size, stride=1, padding=kernel_size // 2)
        
        boundary = dilated - eroded
        return boundary


class WaveguideBendConstraint(nn.Module):
    """
    波导弯曲约束
    
    专门用于波导设计的曲率约束。
    """
    
    def __init__(
        self,
        min_radius: float = 5.0,  # μm
        max_loss_db: float = 0.1,  # dB per 90° bend
        resolution: float = 0.01,  # μm/pixel
        weight: float = 1.0,
    ):
        super().__init__()
        
        self.min_radius = min_radius
        self.max_loss_db = max_loss_db
        self.resolution = resolution
        self.weight = weight
    
    def forward(self, design: Tensor, path: Optional[Tensor] = None) -> Dict[str, Tensor]:
        """
        计算波导弯曲约束
        
        Args:
            design: 设计变量
            path: 波导路径坐标 (可选)
            
        Returns:
            约束字典
        """
        if design.dim() == 2:
            design = design.unsqueeze(0).unsqueeze(0)
        
        # 如果没有提供路径，自动检测
        if path is None:
            path = self._detect_waveguide_path(design)
        
        # 计算路径曲率
        curvature = self._compute_path_curvature(path)
        
        # 转换为曲率半径 (像素单位)
        radius_pixels = 1.0 / (curvature + 1e-8)
        radius_um = radius_pixels * self.resolution
        
        # 约束违反
        radius_violation = F.relu(self.min_radius - radius_um)
        
        # 估算弯曲损耗
        # 简化模型：损耗 ∝ exp(-radius / characteristic_length)
        characteristic_length = 10.0  # μm
        estimated_loss = torch.exp(-radius_um / characteristic_length)
        
        constraints = {
            'curvature': curvature,
            'radius_um': radius_um,
            'min_radius': radius_um.amin(dim=-1),
            'radius_violation': radius_violation.mean(dim=-1),
            'estimated_loss_db': estimated_loss.sum(dim=-1) * 0.1,  # 简化
        }
        
        constraints['total'] = constraints['radius_violation'] * self.weight
        
        return constraints
    
    def _detect_waveguide_path(self, design: Tensor) -> Tensor:
        """检测波导路径（骨架化）"""
        # 二值化
        binary = (design > 0.5).float()
        
        # 使用距离变换找中心线
        # 简化实现：返回高梯度区域
        grad_x = binary[:, :, :, 2:] - binary[:, :, :, :-2]
        grad_y = binary[:, :, 2:, :] - binary[:, :, :-2, :]
        
        grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2)
        
        # 返回边界点
        return grad_mag
    
    def _compute_path_curvature(self, path: Tensor) -> Tensor:
        """计算路径曲率"""
        # 简化实现
        if path.dim() == 4:
            path = path.squeeze(1)
        
        # 二阶差分近似曲率
        if path.shape[-1] > 2:
            d2 = path[:, :, 2:] - 2 * path[:, :, 1:-1] + path[:, :, :-2]
            curvature = torch.abs(d2).mean(dim=-1)
        else:
            curvature = torch.zeros(path.shape[0], device=path.device)
        
        return curvature


# ============================================================================
# 便捷函数
# ============================================================================

def create_curvature_constraint(
    min_radius: float = 5.0,
    resolution: float = 0.01,
    **kwargs
) -> CurvatureConstraint:
    """创建曲率约束"""
    config = CurvatureConfig(
        min_radius=min_radius,
        resolution=resolution,
        **kwargs
    )
    return CurvatureConstraint(config)


def compute_curvature_violation(
    design: Union[Tensor, np.ndarray],
    min_radius: float = 5.0,
    resolution: float = 0.01,
) -> float:
    """
    计算曲率约束违反程度
    
    Args:
        design: 设计变量
        min_radius: 最小曲率半径 (μm)
        resolution: 网格分辨率 (μm/pixel)
        
    Returns:
        约束违反值
    """
    if isinstance(design, np.ndarray):
        design = torch.from_numpy(design).float()
    
    constraint = MinimumRadiusConstraint(min_radius, resolution)
    result = constraint(design)
    
    return result['total'].item()

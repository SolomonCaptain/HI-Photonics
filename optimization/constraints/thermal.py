"""
热学约束模块

实现光子学器件设计中的热学约束，包括：
1. 温度分布约束
2. 热应力约束
3. 热光效应约束
4. 散热结构优化

参考文献:
- Maldovan (2013). "Sound and heat revolutions in phononics"
- Tian et al. (2020). "Thermal effects in silicon photonic devices"
"""

from typing import Dict, Optional, Tuple, List, Union
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import numpy as np


@dataclass
class ThermalConfig:
    """热学约束配置"""
    # 材料参数
    thermal_conductivity: float = 148.0    # W/(m·K) 硅的热导率
    specific_heat: float = 700.0           # J/(kg·K) 硅的比热容
    density: float = 2330.0                # kg/m³ 硅的密度
    thermo_optic_coeff: float = 1.86e-4    # K⁻¹ 硅的热光系数
    
    # 环境参数
    ambient_temperature: float = 300.0     # K 环境温度
    max_temperature: float = 350.0         # K 最高允许温度
    
    # 几何参数
    domain_size: Tuple[float, float] = (10.0, 10.0)  # μm
    grid_size: Tuple[int, int] = (100, 100)
    
    # 热源参数
    heat_generation_rate: float = 1e6      # W/m³ 热源强度
    
    # 对流换热
    convection_coefficient: float = 10.0   # W/(m²·K)
    
    # 约束权重
    weight: float = 1.0


class HeatEquationSolver(nn.Module):
    """
    热传导方程求解器
    
    求解稳态热传导方程：
    -∇·(k∇T) = Q
    
    其中 k 是热导率，T 是温度，Q 是热源。
    """
    
    def __init__(
        self,
        domain_size: Tuple[float, float] = (10.0, 10.0),
        grid_size: Tuple[int, int] = (100, 100),
        thermal_conductivity: float = 148.0,
    ):
        super().__init__()
        
        self.Lx, self.Ly = domain_size
        self.Nx, self.Ny = grid_size
        self.k = thermal_conductivity
        
        # 网格间距
        self.dx = self.Lx * 1e-6 / self.Nx  # 转换为米
        self.dy = self.Ly * 1e-6 / self.Ny
        
        # 构建有限差分算子
        self._setup_laplacian()
    
    def _setup_laplacian(self):
        """设置拉普拉斯算子"""
        # 5点差分格式的系数
        self.register_buffer(
            'laplacian_kernel',
            torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3)
        )
    
    def forward(
        self,
        heat_source: Tensor,
        thermal_conductivity: Optional[Tensor] = None,
        boundary_temp: float = 300.0,
        max_iterations: int = 1000,
        tolerance: float = 1e-6,
    ) -> Tensor:
        """
        求解稳态热传导方程
        
        Args:
            heat_source: 热源分布 [B, 1, H, W] 或 [H, W]
            thermal_conductivity: 空间变化的热导率（可选）
            boundary_temp: 边界温度
            max_iterations: 最大迭代次数
            tolerance: 收敛容差
            
        Returns:
            温度分布
        """
        if heat_source.dim() == 2:
            heat_source = heat_source.unsqueeze(0).unsqueeze(0)
        
        B = heat_source.shape[0]
        
        # 初始温度场
        T = torch.full_like(heat_source, boundary_temp)
        
        # 使用 Jacobi 迭代
        k = self.k if thermal_conductivity is None else thermal_conductivity
        
        for _ in range(max_iterations):
            T_old = T.clone()
            
            # 拉普拉斯算子
            laplacian = F.conv2d(T, self.laplacian_kernel, padding=1)
            laplacian = laplacian / (self.dx ** 2)  # 归一化
            
            # 更新温度
            # T_new = T + dt * (k * ∇²T + Q) / (ρ * c)
            # 稳态: k * ∇²T + Q = 0
            # Jacobi: T_new = (Σ T_neighbors + Q * dx² / k) / 4
            
            T_new = 0.25 * (
                F.pad(T[:, :, :, 1:], (0, 1, 0, 0), mode='replicate') +
                F.pad(T[:, :, :, :-1], (1, 0, 0, 0), mode='replicate') +
                F.pad(T[:, :, 1:, :], (0, 0, 0, 1), mode='replicate') +
                F.pad(T[:, :, :-1, :], (0, 0, 1, 0), mode='replicate') +
                heat_source * self.dx ** 2 / k
            )
            
            # 边界条件
            T = self._apply_boundary(T_new, boundary_temp)
            
            # 检查收敛
            if torch.abs(T - T_old).max() < tolerance:
                break
        
        return T
    
    def _apply_boundary(self, T: Tensor, boundary_temp: float) -> Tensor:
        """应用边界条件"""
        # Dirichlet 边界
        T[:, :, 0, :] = boundary_temp
        T[:, :, -1, :] = boundary_temp
        T[:, :, :, 0] = boundary_temp
        T[:, :, :, -1] = boundary_temp
        return T


class TemperatureConstraint(nn.Module):
    """
    温度约束
    
    确保器件温度不超过允许范围。
    """
    
    def __init__(
        self,
        max_temperature: float = 350.0,
        ambient_temperature: float = 300.0,
        weight: float = 1.0,
    ):
        super().__init__()
        
        self.max_temp = max_temperature
        self.ambient_temp = ambient_temperature
        self.weight = weight
        
        self.solver = None  # 延迟初始化
    
    def forward(
        self,
        design: Tensor,
        heat_source: Optional[Tensor] = None,
        solver: Optional[HeatEquationSolver] = None,
    ) -> Dict[str, Tensor]:
        """
        计算温度约束
        
        Args:
            design: 设计变量（用于确定热导率分布）
            heat_source: 热源分布
            solver: 热方程求解器
            
        Returns:
            约束字典
        """
        if heat_source is None:
            # 默认热源：与设计成正比（简化模型）
            heat_source = design * 1e6  # W/m³
        
        if solver is None:
            H, W = design.shape[-2:]
            solver = HeatEquationSolver(grid_size=(H, W))
        
        # 计算温度分布
        temperature = solver(heat_source, boundary_temp=self.ambient_temp)
        
        # 温度超过限制的违反
        violation = F.relu(temperature - self.max_temp)
        
        constraints = {
            'temperature': temperature,
            'max_temperature': temperature.amax(dim=(2, 3)),
            'mean_temperature': temperature.mean(dim=(2, 3)),
            'violation': violation,
            'violation_sum': violation.sum(dim=(2, 3)),
        }
        
        constraints['total'] = constraints['violation_sum'] * self.weight
        
        return constraints
    
    def compute_loss(self, design: Tensor, heat_source: Optional[Tensor] = None) -> Tensor:
        """计算约束损失"""
        constraints = self.forward(design, heat_source)
        return constraints['total'].mean()


class ThermalStressConstraint(nn.Module):
    """
    热应力约束
    
    计算由温度梯度引起的热应力。
    """
    
    def __init__(
        self,
        thermal_expansion_coeff: float = 2.6e-6,  # K⁻¹ 硅的热膨胀系数
        youngs_modulus: float = 170e9,            # Pa 硅的杨氏模量
        poissons_ratio: float = 0.28,             # 硅的泊松比
        max_stress: float = 1e9,                  # Pa 最大允许应力
        weight: float = 1.0,
    ):
        super().__init__()
        
        self.alpha = thermal_expansion_coeff
        self.E = youngs_modulus
        self.nu = poissons_ratio
        self.max_stress = max_stress
        self.weight = weight
    
    def forward(
        self,
        temperature: Tensor,
        reference_temp: float = 300.0,
    ) -> Dict[str, Tensor]:
        """
        计算热应力约束
        
        Args:
            temperature: 温度分布
            reference_temp: 参考温度（无应力状态）
            
        Returns:
            约束字典
        """
        if temperature.dim() == 2:
            temperature = temperature.unsqueeze(0).unsqueeze(0)
        
        # 温度变化
        delta_T = temperature - reference_temp
        
        # 温度梯度
        grad_x = (temperature[:, :, :, 2:] - temperature[:, :, :, :-2]) / 2
        grad_y = (temperature[:, :, 2:, :] - temperature[:, :, :-2, :]) / 2
        
        # 填充边界
        grad_x = F.pad(grad_x, (1, 1, 0, 0), mode='replicate')
        grad_y = F.pad(grad_y, (0, 0, 1, 1), mode='replicate')
        
        # 热应力（简化模型）
        # σ = E * α * ΔT / (1 - ν)
        stress = self.E * self.alpha * delta_T / (1 - self.nu)
        
        # Von Mises 应力（平面应力假设）
        sigma_xx = stress
        sigma_yy = stress
        sigma_xy = self.E * self.alpha * torch.sqrt(grad_x ** 2 + grad_y ** 2) / (2 * (1 + self.nu))
        
        von_mises = torch.sqrt(
            sigma_xx ** 2 + sigma_yy ** 2 - sigma_xx * sigma_yy + 3 * sigma_xy ** 2
        )
        
        # 应力违反
        violation = F.relu(von_mises - self.max_stress)
        
        constraints = {
            'thermal_stress': von_mises,
            'max_stress': von_mises.amax(dim=(2, 3)),
            'violation': violation,
            'violation_sum': violation.sum(dim=(2, 3)),
        }
        
        constraints['total'] = constraints['violation_sum'] * self.weight
        
        return constraints


class ThermoOpticConstraint(nn.Module):
    """
    热光效应约束
    
    考虑温度变化对折射率的影响。
    """
    
    def __init__(
        self,
        thermo_optic_coeff: float = 1.86e-4,  # K⁻¹ 硅的热光系数
        reference_temp: float = 300.0,
        wavelength: float = 1.55,             # μm
        max_index_change: float = 0.01,       # 最大允许折射率变化
        weight: float = 1.0,
    ):
        super().__init__()
        
        self.dn_dT = thermo_optic_coeff
        self.T_ref = reference_temp
        self.wavelength = wavelength
        self.max_delta_n = max_index_change
        self.weight = weight
    
    def forward(
        self,
        temperature: Tensor,
        base_refractive_index: float = 3.48,
    ) -> Dict[str, Tensor]:
        """
        计算热光效应约束
        
        Args:
            temperature: 温度分布
            base_refractive_index: 基础折射率
            
        Returns:
            约束字典
        """
        if temperature.dim() == 2:
            temperature = temperature.unsqueeze(0).unsqueeze(0)
        
        # 温度变化
        delta_T = temperature - self.T_ref
        
        # 折射率变化: Δn = dn/dT * ΔT
        delta_n = self.dn_dT * delta_T
        
        # 新的折射率分布
        n_new = base_refractive_index + delta_n
        
        # 相位变化
        # Δφ = 2π * Δn * L / λ
        phase_change = 2 * np.pi * delta_n * (self.wavelength * 1e-6) / (self.wavelength * 1e-6)
        
        # 约束违反
        violation = F.relu(torch.abs(delta_n) - self.max_delta_n)
        
        constraints = {
            'refractive_index': n_new,
            'index_change': delta_n,
            'phase_change': phase_change,
            'max_index_change': torch.abs(delta_n).amax(dim=(2, 3)),
            'violation': violation,
            'violation_sum': violation.sum(dim=(2, 3)),
        }
        
        constraints['total'] = constraints['violation_sum'] * self.weight
        
        return constraints


class HeatDissipationConstraint(nn.Module):
    """
    散热结构约束
    
    优化设计以改善散热性能。
    """
    
    def __init__(
        self,
        convection_coefficient: float = 10.0,  # W/(m²·K)
        ambient_temperature: float = 300.0,
        min_heat_transfer: float = 1e-4,       # W
        weight: float = 1.0,
    ):
        super().__init__()
        
        self.h = convection_coefficient
        self.T_ambient = ambient_temperature
        self.min_Q = min_heat_transfer
        self.weight = weight
    
    def forward(
        self,
        design: Tensor,
        temperature: Tensor,
    ) -> Dict[str, Tensor]:
        """
        计算散热约束
        
        Args:
            design: 设计变量
            temperature: 温度分布
            
        Returns:
            约束字典
        """
        if design.dim() == 2:
            design = design.unsqueeze(0).unsqueeze(0)
        if temperature.dim() == 2:
            temperature = temperature.unsqueeze(0).unsqueeze(0)
        
        # 表面积（设计边界）
        # 使用梯度近似表面积
        grad_x = torch.abs(design[:, :, :, 1:] - design[:, :, :, :-1])
        grad_y = torch.abs(design[:, :, 1:, :] - design[:, :, :-1, :])
        
        surface_area = (grad_x.sum(dim=(2, 3)) + grad_y.sum(dim=(2, 3))) * 1e-12  # 转换为 m²
        
        # 散热量 Q = h * A * (T - T_ambient)
        heat_transfer = self.h * surface_area * (temperature.mean(dim=(2, 3)) - self.T_ambient)
        
        # 热阻
        thermal_resistance = (temperature.amax(dim=(2, 3)) - self.T_ambient) / (heat_transfer + 1e-10)
        
        # 约束：散热能力不足
        violation = F.relu(self.min_Q - heat_transfer)
        
        constraints = {
            'surface_area': surface_area,
            'heat_transfer': heat_transfer,
            'thermal_resistance': thermal_resistance,
            'violation': violation,
        }
        
        constraints['total'] = violation * self.weight
        
        return constraints


class ThermalConstraint(nn.Module):
    """
    综合热学约束
    
    组合温度、应力、热光效应等约束。
    """
    
    def __init__(self, config: Optional[ThermalConfig] = None):
        super().__init__()
        self.config = config or ThermalConfig()
        
        # 热方程求解器
        self.solver = HeatEquationSolver(
            domain_size=self.config.domain_size,
            grid_size=self.config.grid_size,
            thermal_conductivity=self.config.thermal_conductivity,
        )
        
        # 子约束
        self.temp_constraint = TemperatureConstraint(
            max_temperature=self.config.max_temperature,
            ambient_temperature=self.config.ambient_temperature,
        )
        
        self.stress_constraint = ThermalStressConstraint()
        
        self.thermo_optic_constraint = ThermoOpticConstraint(
            thermo_optic_coeff=self.config.thermo_optic_coeff,
            reference_temp=self.config.ambient_temperature,
        )
    
    def forward(
        self,
        design: Tensor,
        heat_source: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        计算综合热学约束
        
        Args:
            design: 设计变量
            heat_source: 热源分布
            
        Returns:
            约束字典
        """
        if heat_source is None:
            heat_source = design * self.config.heat_generation_rate
        
        # 求解温度场
        temperature = self.solver(heat_source, boundary_temp=self.config.ambient_temperature)
        
        # 各项约束
        temp_constraints = self.temp_constraint(design, heat_source, self.solver)
        stress_constraints = self.stress_constraint(temperature, self.config.ambient_temperature)
        thermo_optic_constraints = self.thermo_optic_constraint(temperature)
        
        # 合并结果
        constraints = {
            'temperature': temperature,
            'temperature_constraints': temp_constraints,
            'stress_constraints': stress_constraints,
            'thermo_optic_constraints': thermo_optic_constraints,
        }
        
        # 总约束
        constraints['total'] = (
            temp_constraints['total'] +
            stress_constraints['total'] +
            thermo_optic_constraints['total']
        ) * self.config.weight
        
        return constraints
    
    def compute_loss(self, design: Tensor, heat_source: Optional[Tensor] = None) -> Tensor:
        """计算约束损失"""
        constraints = self.forward(design, heat_source)
        return constraints['total'].mean()


# ============================================================================
# 便捷函数
# ============================================================================

def create_thermal_constraint(
    max_temperature: float = 350.0,
    ambient_temperature: float = 300.0,
    **kwargs
) -> ThermalConstraint:
    """创建热学约束"""
    config = ThermalConfig(
        max_temperature=max_temperature,
        ambient_temperature=ambient_temperature,
        **kwargs
    )
    return ThermalConstraint(config)


def compute_temperature_field(
    heat_source: Union[Tensor, np.ndarray],
    domain_size: Tuple[float, float] = (10.0, 10.0),
    boundary_temp: float = 300.0,
) -> Union[Tensor, np.ndarray]:
    """
    计算温度分布
    
    Args:
        heat_source: 热源分布
        domain_size: 计算域大小 (μm)
        boundary_temp: 边界温度 (K)
        
    Returns:
        温度分布
    """
    is_numpy = isinstance(heat_source, np.ndarray)
    if is_numpy:
        heat_source = torch.from_numpy(heat_source).float()
    
    H, W = heat_source.shape[-2:]
    solver = HeatEquationSolver(domain_size=domain_size, grid_size=(H, W))
    
    temperature = solver(heat_source, boundary_temp=boundary_temp)
    
    if is_numpy:
        return temperature.squeeze().numpy()
    return temperature.squeeze()


def estimate_thermal_index_change(
    temperature: Union[Tensor, np.ndarray, float],
    thermo_optic_coeff: float = 1.86e-4,
    reference_temp: float = 300.0,
) -> float:
    """
    估算热光效应引起的折射率变化
    
    Args:
        temperature: 温度 (K)
        thermo_optic_coeff: 热光系数 (K⁻¹)
        reference_temp: 参考温度 (K)
        
    Returns:
        折射率变化
    """
    if isinstance(temperature, (Tensor, np.ndarray)):
        delta_T = float(temperature.mean()) - reference_temp
    else:
        delta_T = temperature - reference_temp
    
    return thermo_optic_coeff * delta_T
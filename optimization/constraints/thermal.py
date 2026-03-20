"""
热效应约束模块

实现热-光耦合约束，考虑温度对光子器件性能的影响。
包括:
1. 热光效应 - 温度导致的折射率变化
2. 热膨胀效应 - 温度导致的结构变形
3. 热源分布 - 光吸收产生的热量
4. 稳态热分析 - 热传导方程求解
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum


class ThermalModelType(Enum):
    """热分析模型类型"""
    STEADY_STATE = "steady_state"       # 稳态热分析
    TRANSIENT = "transient"             # 瞬态热分析
    ANALYTICAL = "analytical"           # 解析模型


@dataclass
class ThermalProperties:
    """材料热属性数据类"""
    name: str
    thermo_optic_coeff: float  # K⁻¹ (热光系数 dn/dT)
    thermal_conductivity: float  # W/(m·K)
    specific_heat: float  # J/(kg·K)
    density: float  # kg/m³
    thermal_expansion_coeff: float  # K⁻¹ (热膨胀系数)
    reference_temperature: float = 300.0  # K (参考温度)


# 常用光子学材料的热属性
THERMAL_PROPERTIES: Dict[str, ThermalProperties] = {
    "silicon": ThermalProperties(
        name="silicon",
        thermo_optic_coeff=1.86e-4,  # K⁻¹
        thermal_conductivity=148.0,  # W/(m·K)
        specific_heat=700.0,  # J/(kg·K)
        density=2329.0,  # kg/m³
        thermal_expansion_coeff=2.6e-6,  # K⁻¹
    ),
    "silicon_dioxide": ThermalProperties(
        name="silicon_dioxide",
        thermo_optic_coeff=1.0e-5,  # K⁻¹
        thermal_conductivity=1.38,  # W/(m·K)
        specific_heat=703.0,  # J/(kg·K)
        density=2200.0,  # kg/m³
        thermal_expansion_coeff=0.55e-6,  # K⁻¹
    ),
    "silicon_nitride": ThermalProperties(
        name="silicon_nitride",
        thermo_optic_coeff=2.45e-5,  # K⁻¹
        thermal_conductivity=30.0,  # W/(m·K)
        specific_heat=700.0,  # J/(kg·K)
        density=3100.0,  # kg/m³
        thermal_expansion_coeff=3.3e-6,  # K⁻¹
    ),
    "gaas": ThermalProperties(
        name="gaas",
        thermo_optic_coeff=2.67e-4,  # K⁻¹
        thermal_conductivity=46.0,  # W/(m·K)
        specific_heat=327.0,  # J/(kg·K)
        density=5320.0,  # kg/m³
        thermal_expansion_coeff=5.73e-6,  # K⁻¹
    ),
    "inp": ThermalProperties(
        name="inp",
        thermo_optic_coeff=2.0e-4,  # K⁻¹
        thermal_conductivity=68.0,  # W/(m·K)
        specific_heat=310.0,  # J/(kg·K)
        density=4810.0,  # kg/m³
        thermal_expansion_coeff=4.5e-6,  # K⁻¹
    ),
    "polymer": ThermalProperties(
        name="polymer",
        thermo_optic_coeff=-1.0e-4,  # K⁻¹ (负温度系数)
        thermal_conductivity=0.2,  # W/(m·K)
        specific_heat=1500.0,  # J/(kg·K)
        density=1200.0,  # kg/m³
        thermal_expansion_coeff=50e-6,  # K⁻¹
    ),
    "air": ThermalProperties(
        name="air",
        thermo_optic_coeff=0.0,
        thermal_conductivity=0.026,  # W/(m·K)
        specific_heat=1005.0,  # J/(kg·K)
        density=1.225,  # kg/m³
        thermal_expansion_coeff=0.0,  # 气体热膨胀由状态方程决定
    ),
    "gold": ThermalProperties(
        name="gold",
        thermo_optic_coeff=0.0,  # 金属热光效应不同
        thermal_conductivity=317.0,  # W/(m·K)
        specific_heat=129.0,  # J/(kg·K)
        density=19300.0,  # kg/m³
        thermal_expansion_coeff=14.2e-6,  # K⁻¹
    ),
}


class ThermoOpticEffect(nn.Module):
    """
    热光效应模型
    
    计算温度变化导致的折射率变化:
    n(T) = n(T₀) + (dn/dT)(T - T₀)
    """
    
    def __init__(
        self,
        material: str = "silicon",
        reference_temperature: float = 300.0,  # K
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            material: 材料名称
            reference_temperature: 参考温度 (K)
            device: 计算设备
        """
        super().__init__()
        
        if material not in THERMAL_PROPERTIES:
            raise ValueError(f"未知材料: {material}，可用: {list(THERMAL_PROPERTIES.keys())}")
        
        self.properties = THERMAL_PROPERTIES[material]
        self.reference_temperature = reference_temperature
        self.device = device or torch.device('cpu')
    
    def compute_refractive_index_change(
        self,
        temperature: torch.Tensor,
        base_refractive_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        计算温度导致的折射率变化
        
        Args:
            temperature: 温度场 (K)
            base_refractive_index: 基础折射率（可选）
            
        Returns:
            折射率变化 Δn 或新折射率
        """
        delta_T = temperature - self.reference_temperature
        delta_n = self.properties.thermo_optic_coeff * delta_T
        
        if base_refractive_index is not None:
            return base_refractive_index + delta_n
        return delta_n
    
    def compute_phase_shift(
        self,
        temperature: torch.Tensor,
        wavelength: float,  # μm
        propagation_length: float,  # μm
    ) -> torch.Tensor:
        """
        计算温度导致的相移
        
        Args:
            temperature: 温度场 (K)
            wavelength: 波长 (μm)
            propagation_length: 传播长度 (μm)
            
        Returns:
            相移 (rad)
        """
        delta_n = self.compute_refractive_index_change(temperature)
        
        # Δφ = 2π/λ × Δn × L
        phase_shift = 2 * torch.pi / wavelength * delta_n * propagation_length
        
        return phase_shift
    
    def forward(
        self,
        temperature: torch.Tensor,
        base_n: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """前向传播"""
        return self.compute_refractive_index_change(temperature, base_n)


class ThermalExpansion(nn.Module):
    """
    热膨胀模型
    
    计算温度变化导致的结构变形:
    L(T) = L₀ × (1 + α × ΔT)
    """
    
    def __init__(
        self,
        material: str = "silicon",
        reference_temperature: float = 300.0,  # K
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            material: 材料名称
            reference_temperature: 参考温度 (K)
            device: 计算设备
        """
        super().__init__()
        
        if material not in THERMAL_PROPERTIES:
            raise ValueError(f"未知材料: {material}")
        
        self.properties = THERMAL_PROPERTIES[material]
        self.reference_temperature = reference_temperature
        self.device = device or torch.device('cpu')
    
    def compute_dimensional_change(
        self,
        temperature: torch.Tensor,
        base_dimension: float,
    ) -> torch.Tensor:
        """
        计算尺寸变化
        
        Args:
            temperature: 温度 (K)
            base_dimension: 基础尺寸
            
        Returns:
            变化后的尺寸
        """
        delta_T = temperature - self.reference_temperature
        
        # 线性热膨胀
        expansion_factor = 1 + self.properties.thermal_expansion_coeff * delta_T
        
        return base_dimension * expansion_factor
    
    def compute_strain_field(
        self,
        temperature_field: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算应变场
        
        Args:
            temperature_field: 温度场 [H, W]
            
        Returns:
            热应变场
        """
        delta_T = temperature_field - self.reference_temperature
        strain = self.properties.thermal_expansion_coeff * delta_T
        
        return strain


class HeatConductionSolver(nn.Module):
    """
    热传导方程求解器
    
    求解稳态热传导方程:
    ∇·(k∇T) + Q = 0
    
    或瞬态热传导方程:
    ρc ∂T/∂t = ∇·(k∇T) + Q
    """
    
    def __init__(
        self,
        grid_size: Tuple[int, int],
        physical_size: Tuple[float, float],  # μm
        thermal_conductivity: float = 148.0,  # W/(m·K) 硅
        specific_heat: float = 700.0,  # J/(kg·K)
        density: float = 2329.0,  # kg/m³
        boundary_temperature: float = 300.0,  # K
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            grid_size: 网格尺寸 (H, W)
            physical_size: 物理尺寸 (μm)
            thermal_conductivity: 热导率
            specific_heat: 比热容
            density: 密度
            boundary_temperature: 边界温度
            device: 计算设备
        """
        super().__init__()
        
        self.grid_size = grid_size
        self.physical_size = physical_size
        self.k = thermal_conductivity
        self.c = specific_heat
        self.rho = density
        self.T_boundary = boundary_temperature
        self.device = device or torch.device('cpu')
        
        # 计算网格间距
        self.dx = physical_size[0] / grid_size[0] * 1e-6  # 转换为 m
        self.dy = physical_size[1] / grid_size[1] * 1e-6
        
        # 热扩散率
        self.alpha = self.k / (self.rho * self.c)
        
        # 初始化温度场
        self.register_buffer(
            'temperature',
            torch.full(grid_size, boundary_temperature, device=self.device)
        )
    
    def solve_steady_state(
        self,
        heat_source: torch.Tensor,
        num_iterations: int = 1000,
        convergence_threshold: float = 1e-6,
    ) -> torch.Tensor:
        """
        求解稳态热传导方程
        
        使用迭代法求解:
        T_{i,j} = (T_{i+1,j} + T_{i-1,j} + T_{i,j+1} + T_{i,j-1} + Q*dx²/k) / 4
        
        Args:
            heat_source: 热源分布 [H, W] (W/m³)
            num_iterations: 最大迭代次数
            convergence_threshold: 收敛阈值
            
        Returns:
            稳态温度场
        """
        H, W = self.grid_size
        T = self.temperature.clone()
        Q = heat_source
        
        # 归一化热源
        Q_normalized = Q * self.dx**2 / self.k
        
        for iteration in range(num_iterations):
            T_old = T.clone()
            
            # 边界条件（Dirichlet）
            T_padded = F.pad(T.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='replicate')
            
            # Laplacian
            T_new = (
                T_padded[0, 0, 2:, 1:-1] +  # T_{i+1,j}
                T_padded[0, 0, :-2, 1:-1] +  # T_{i-1,j}
                T_padded[0, 0, 1:-1, 2:] +  # T_{i,j+1}
                T_padded[0, 0, 1:-1, :-2] -  # T_{i,j-1}
                Q_normalized
            ) / 4
            
            # 边界保持恒定
            T_new = self._apply_boundary_conditions(T_new)
            
            # 检查收敛
            max_change = (T_new - T_old).abs().max()
            T = T_new
            
            if max_change < convergence_threshold:
                break
        
        self.temperature = T
        return T
    
    def solve_transient(
        self,
        heat_source: torch.Tensor,
        time_steps: int = 100,
        dt: Optional[float] = None,
    ) -> torch.Tensor:
        """
        求解瞬态热传导方程
        
        使用显式差分格式:
        T^{n+1} = T^n + αΔt∇²T^n + QΔt/(ρc)
        
        Args:
            heat_source: 热源分布 [H, W] (W/m³)
            time_steps: 时间步数
            dt: 时间步长（自动计算稳定性条件）
            
        Returns:
            温度场时间序列 [time_steps, H, W]
        """
        if dt is None:
            # 稳定性条件: dt < dx²/(4α)
            dt = 0.25 * min(self.dx, self.dy)**2 / self.alpha * 0.9
        
        T = self.temperature.clone()
        Q = heat_source
        
        # 时间序列存储
        T_history = [T.clone()]
        
        for _ in range(time_steps):
            # Laplacian
            T_padded = F.pad(T.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='replicate')
            
            laplacian = (
                T_padded[0, 0, 2:, 1:-1] +
                T_padded[0, 0, :-2, 1:-1] +
                T_padded[0, 0, 1:-1, 2:] +
                T_padded[0, 0, 1:-1, :-2] -
                4 * T
            ) / (self.dx**2)
            
            # 更新温度
            T = T + dt * (
                self.alpha * laplacian +
                Q / (self.rho * self.c)
            )
            
            # 边界条件
            T = self._apply_boundary_conditions(T)
            
            T_history.append(T.clone())
        
        return torch.stack(T_history)
    
    def _apply_boundary_conditions(self, T: torch.Tensor) -> torch.Tensor:
        """应用边界条件"""
        # Dirichlet 边界条件
        T[0, :] = self.T_boundary
        T[-1, :] = self.T_boundary
        T[:, 0] = self.T_boundary
        T[:, -1] = self.T_boundary
        return T


class HeatSourceModel(nn.Module):
    """
    热源模型
    
    计算光吸收产生的热量分布。
    """
    
    def __init__(
        self,
        absorption_coefficient: float = 0.01,  # 1/μm
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            absorption_coefficient: 吸收系数
            device: 计算设备
        """
        super().__init__()
        self.absorption_coeff = absorption_coefficient
        self.device = device or torch.device('cpu')
    
    def compute_absorption_heat(
        self,
        field_intensity: torch.Tensor,  # |E|² 分布
        material_density: torch.Tensor,  # 材料密度场
    ) -> torch.Tensor:
        """
        计算吸收热
        
        Q = α × I × ρ_material
        
        Args:
            field_intensity: 光场强度分布
            material_density: 材料密度（0-1）
            
        Returns:
            热源分布 (W/m³)
        """
        # 简化模型：吸收正比于光强和材料密度
        Q = self.absorption_coeff * field_intensity * material_density
        
        # 转换单位（假设 field_intensity 归一化）
        Q = Q * 1e12  # 缩放到 W/m³ 量级
        
        return Q
    
    def compute_heater_heat(
        self,
        heater_power: float,  # W
        heater_region: torch.Tensor,  # 加热器区域掩码
    ) -> torch.Tensor:
        """
        计算加热器产生的热量
        
        Args:
            heater_power: 加热器功率
            heater_region: 加热器区域掩码
            
        Returns:
            热源分布
        """
        # 均匀分布热量
        heater_area = heater_region.sum()
        Q = torch.zeros_like(heater_region)
        Q[heater_region > 0.5] = heater_power / heater_area
        
        return Q


class ThermalConstraint(nn.Module):
    """
    热效应约束模块
    
    综合热-光耦合约束。
    """
    
    def __init__(
        self,
        material: str = "silicon",
        max_temperature: float = 400.0,  # K
        max_temperature_gradient: float = 10.0,  # K/μm
        grid_size: Tuple[int, int] = (100, 100),
        physical_size: Tuple[float, float] = (10.0, 10.0),  # μm
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            material: 材料名称
            max_temperature: 最大允许温度
            max_temperature_gradient: 最大温度梯度
            grid_size: 网格尺寸
            physical_size: 物理尺寸
            device: 计算设备
        """
        super().__init__()
        
        if material not in THERMAL_PROPERTIES:
            raise ValueError(f"未知材料: {material}")
        
        self.properties = THERMAL_PROPERTIES[material]
        self.max_temperature = max_temperature
        self.max_temp_gradient = max_temperature_gradient
        self.device = device or torch.device('cpu')
        
        # 子模块
        self.thermo_optic = ThermoOpticEffect(material, device=device)
        self.thermal_expansion = ThermalExpansion(material, device=device)
        self.heat_solver = HeatConductionSolver(
            grid_size=grid_size,
            physical_size=physical_size,
            thermal_conductivity=self.properties.thermal_conductivity,
            specific_heat=self.properties.specific_heat,
            density=self.properties.density,
            device=device,
        )
        self.heat_source = HeatSourceModel(device=device)
    
    def forward(
        self,
        design: torch.Tensor,
        heat_source: torch.Tensor,
        base_refractive_index: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        计算热效应约束
        
        Args:
            design: 设计参数
            heat_source: 热源分布
            base_refractive_index: 基础折射率
            
        Returns:
            约束字典
        """
        # 求解温度场
        temperature = self.heat_solver.solve_steady_state(heat_source)
        
        # 热光效应
        delta_n = self.thermo_optic.compute_refractive_index_change(temperature)
        
        # 温度约束
        temp_violation = F.relu(temperature - self.max_temperature)
        
        # 温度梯度约束
        grad_x = temperature[1:, :] - temperature[:-1, :]
        grad_y = temperature[:, 1:] - temperature[:, :-1]
        temp_gradient = torch.sqrt(grad_x[:-1, :]**2 + grad_y[:, :-1]**2)
        gradient_violation = F.relu(temp_gradient - self.max_temp_gradient)
        
        constraints = {
            'temperature_field': temperature,
            'refractive_index_change': delta_n,
            'max_temperature': temperature.max(),
            'temperature_violation': temp_violation.mean(),
            'gradient_violation': gradient_violation.mean(),
            'thermal_phase_shift': self.thermo_optic.compute_phase_shift(
                temperature, wavelength=1.55, propagation_length=1.0
            ).mean(),
        }
        
        # 综合热约束损失
        constraints['thermal_loss'] = (
            temp_violation.mean() + 
            gradient_violation.mean() +
            delta_n.abs().mean()  # 惩罚大的折射率变化
        )
        
        return constraints
    
    def compute_thermal_performance_variation(
        self,
        design: torch.Tensor,
        heat_source: torch.Tensor,
        base_performance: torch.Tensor,
        performance_fn,
    ) -> Dict[str, torch.Tensor]:
        """
        计算热致性能变化
        
        Args:
            design: 设计参数
            heat_source: 热源分布
            base_performance: 基准性能
            performance_fn: 性能评估函数
            
        Returns:
            性能变化字典
        """
        # 求解温度场
        temperature = self.heat_solver.solve_steady_state(heat_source)
        
        # 计算折射率变化
        delta_n = self.thermo_optic.compute_refractive_index_change(temperature)
        
        # 更新设计（考虑折射率变化）
        # 这里简化处理，实际需要更新仿真器中的材料属性
        
        # 性能变化
        performance_variation = {
            'delta_n_mean': delta_n.mean(),
            'delta_n_max': delta_n.abs().max(),
            'temperature_rise': temperature.max() - 300.0,
        }
        
        return performance_variation


class ThermalOptimizationConstraint(nn.Module):
    """
    热优化约束
    
    用于在逆向设计中考虑热效应的约束。
    """
    
    def __init__(
        self,
        material: str = "silicon",
        operating_temperature_range: Tuple[float, float] = (280.0, 360.0),  # K
        thermal_stability_threshold: float = 0.1,  # 性能变化阈值
        weight: float = 1.0,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            material: 材料名称
            operating_temperature_range: 工作温度范围
            thermal_stability_threshold: 热稳定性阈值
            weight: 约束权重
            device: 计算设备
        """
        super().__init__()
        
        self.thermo_optic = ThermoOpticEffect(material, device=device)
        self.temp_range = operating_temperature_range
        self.stability_threshold = thermal_stability_threshold
        self.weight = weight
        self.device = device or torch.device('cpu')
    
    def forward(
        self,
        design: torch.Tensor,
        temperature_variation: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        计算热优化约束
        
        Args:
            design: 设计参数
            temperature_variation: 温度变化范围（用于快速评估）
            
        Returns:
            约束字典
        """
        if temperature_variation is None:
            temperature_variation = self.temp_range[1] - self.temp_range[0]
        
        # 计算温度范围内的折射率变化
        delta_n = self.thermo_optic.properties.thermo_optic_coeff * temperature_variation
        
        # 相位变化（假设特征尺寸）
        feature_size = design.shape[-1] * 0.01  # μm
        phase_variation = 2 * torch.pi / 1.55 * delta_n * feature_size
        
        # 热稳定性约束
        stability_violation = F.relu(phase_variation.abs() - self.stability_threshold)
        
        # 设计敏感度：对温度变化敏感的区域
        grad_x = design[:, 1:] - design[:, :-1] if design.dim() == 2 else design[0, :, 1:] - design[0, :, :-1]
        grad_y = design[1:, :] - design[:-1, :] if design.dim() == 2 else design[0, 1:, :] - design[0, :-1, :]
        
        # 高梯度区域更容易受热效应影响
        sensitivity_map = torch.zeros_like(design if design.dim() == 2 else design[0])
        sensitivity_map[:, :-1] = sensitivity_map[:, :-1] + grad_x.abs()
        sensitivity_map[:-1, :] = sensitivity_map[:-1, :] + grad_y.abs()
        
        constraints = {
            'delta_n_range': delta_n,
            'phase_variation': phase_variation,
            'stability_violation': stability_violation * self.weight,
            'thermal_sensitivity': sensitivity_map.mean(),
            'sensitivity_map': sensitivity_map,
        }
        
        return constraints


# 便捷函数
def get_thermal_properties(material: str) -> ThermalProperties:
    """获取材料热属性"""
    if material not in THERMAL_PROPERTIES:
        raise ValueError(f"未知材料: {material}，可用: {list(THERMAL_PROPERTIES.keys())}")
    return THERMAL_PROPERTIES[material]


def compute_thermal_phase_shift(
    material: str,
    temperature_change: float,
    wavelength: float = 1.55,  # μm
    length: float = 100.0,  # μm
) -> float:
    """
    计算热致相移
    
    Args:
        material: 材料名称
        temperature_change: 温度变化 (K)
        wavelength: 波长 (μm)
        length: 长度 (μm)
        
    Returns:
        相移 (rad)
    """
    props = get_thermal_properties(material)
    delta_n = props.thermo_optic_coeff * temperature_change
    phase_shift = 2 * torch.pi / wavelength * delta_n * length
    return phase_shift.item() if isinstance(phase_shift, torch.Tensor) else phase_shift

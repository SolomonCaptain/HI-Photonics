"""
材料色散约束模块

实现材料色散模型，支持波长依赖的折射率计算。
包括:
1. Sellmeier 方程 - 用于介质材料（Si, SiO2, SiN 等）
2. Drude-Lorentz 模型 - 用于金属材料
3. Cauchy 方程 - 简化色散模型
4. 多波长仿真约束
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum


class DispersionModel(Enum):
    """色散模型类型"""
    SELLMEIER = "sellmeier"
    DRUDE_LORENTZ = "drude_lorentz"
    CAUCHY = "cauchy"
    CONSTANT = "constant"


@dataclass
class MaterialProperties:
    """材料属性数据类"""
    name: str
    model_type: DispersionModel
    # Sellmeier 系数: n² - 1 = Σ B_i λ² / (λ² - C_i)
    sellmeier_B: Optional[Tuple[float, ...]] = None
    sellmeier_C: Optional[Tuple[float, ...]] = None  # 单位: μm²
    # Drude-Lorentz 系数
    plasma_freq: Optional[float] = None  # 等离子频率 (rad/s)
    damping_freq: Optional[float] = None  # 阻尼频率
    lorentz_strengths: Optional[Tuple[float, ...]] = None
    lorentz_freqs: Optional[Tuple[float, ...]] = None
    # Cauchy 系数: n = A + B/λ² + C/λ⁴
    cauchy_A: Optional[float] = None
    cauchy_B: Optional[float] = None
    cauchy_C: Optional[float] = None
    # 常数折射率
    constant_n: Optional[float] = None
    # 温度系数
    thermo_optic_coeff: float = 1.86e-4  # K⁻¹ (硅的默认值)


# 常用光子学材料数据库
MATERIAL_DATABASE: Dict[str, MaterialProperties] = {
    "silicon": MaterialProperties(
        name="silicon",
        model_type=DispersionModel.SELLMEIER,
        sellmeier_B=(10.6684293, 0.0030434748, 1.54133408),
        sellmeier_C=(0.301516485, 1.13475115, 1104.0),  # μm²
        thermo_optic_coeff=1.86e-4,  # K⁻¹
    ),
    "silicon_dioxide": MaterialProperties(
        name="silicon_dioxide",
        model_type=DispersionModel.SELLMEIER,
        sellmeier_B=(0.6961663, 0.4079426, 0.8974794),
        sellmeier_C=(0.0684043**2, 0.1162414**2, 9.896161**2),  # μm²
        thermo_optic_coeff=1.0e-5,  # K⁻¹
    ),
    "silicon_nitride": MaterialProperties(
        name="silicon_nitride",
        model_type=DispersionModel.SELLMEIER,
        sellmeier_B=(2.8939, 0.0, 0.0),
        sellmeier_C=(0.13967**2, 0.0, 0.0),  # μm²
        thermo_optic_coeff=2.45e-5,  # K⁻¹
    ),
    "gaas": MaterialProperties(
        name="gaas",
        model_type=DispersionModel.SELLMEIER,
        sellmeier_B=(7.1, 3.78, 1.3),
        sellmeier_C=(0.327, 0.675, 10.0),  # μm²
        thermo_optic_coeff=2.67e-4,  # K⁻¹
    ),
    "inp": MaterialProperties(
        name="inp",
        model_type=DispersionModel.SELLMEIER,
        sellmeier_B=(7.955, 2.692, 0.0),
        sellmeier_C=(0.278, 0.356, 0.0),  # μm²
        thermo_optic_coeff=2.0e-4,  # K⁻¹
    ),
    "aluminum": MaterialProperties(
        name="aluminum",
        model_type=DispersionModel.DRUDE_LORENTZ,
        plasma_freq=2.24e16,  # rad/s
        damping_freq=1.22e14,  # rad/s
        lorentz_strengths=(0.523, 0.227, 0.050),
        lorentz_freqs=(2.275e15, 5.175e15, 1.0e16),  # rad/s
    ),
    "gold": MaterialProperties(
        name="gold",
        model_type=DispersionModel.DRUDE_LORENTZ,
        plasma_freq=1.37e16,  # rad/s
        damping_freq=6.45e13,  # rad/s
        lorentz_strengths=(0.76, 0.024, 0.01),
        lorentz_freqs=(4.47e15, 6.55e15, 1.0e16),  # rad/s
    ),
    "silver": MaterialProperties(
        name="silver",
        model_type=DispersionModel.DRUDE_LORENTZ,
        plasma_freq=1.37e16,  # rad/s
        damping_freq=2.73e13,  # rad/s
        lorentz_strengths=(0.45, 0.21, 0.04),
        lorentz_freqs=(5.7e15, 8.3e15, 1.2e16),  # rad/s
    ),
    "air": MaterialProperties(
        name="air",
        model_type=DispersionModel.CONSTANT,
        constant_n=1.000293,
        thermo_optic_coeff=0.0,
    ),
    "polymer": MaterialProperties(
        name="polymer",
        model_type=DispersionModel.CAUCHY,
        cauchy_A=1.52,
        cauchy_B=0.005,
        cauchy_C=0.0,
        thermo_optic_coeff=-1.0e-4,  # K⁻¹ (负温度系数)
    ),
}


class DispersionCalculator(nn.Module):
    """
    色散计算器
    
    根据波长计算材料的折射率和介电常数。
    支持多种色散模型和材料数据库。
    """
    
    def __init__(
        self,
        material: str = "silicon",
        wavelength_range: Tuple[float, float] = (1.2, 1.7),  # μm
        num_wavelengths: int = 11,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            material: 材料名称（从材料数据库中选择）
            wavelength_range: 波长范围 (μm)
            num_wavelengths: 波长采样点数
            device: 计算设备
        """
        super().__init__()
        
        if material not in MATERIAL_DATABASE:
            raise ValueError(f"未知材料: {material}，可用材料: {list(MATERIAL_DATABASE.keys())}")
        
        self.material = MATERIAL_DATABASE[material]
        self.device = device or torch.device('cpu')
        
        # 波长采样
        self.register_buffer(
            'wavelengths',
            torch.linspace(wavelength_range[0], wavelength_range[1], num_wavelengths)
        )
    
    def compute_refractive_index(self, wavelengths: torch.Tensor) -> torch.Tensor:
        """
        计算折射率
        
        Args:
            wavelengths: 波长张量 (μm)
            
        Returns:
            折射率张量
        """
        model = self.material.model_type
        
        if model == DispersionModel.SELLMEIER:
            return self._sellmeier(wavelengths)
        elif model == DispersionModel.DRUDE_LORENTZ:
            return self._drude_lorentz(wavelengths)
        elif model == DispersionModel.CAUCHY:
            return self._cauchy(wavelengths)
        elif model == DispersionModel.CONSTANT:
            return torch.full_like(wavelengths, self.material.constant_n)
        else:
            raise ValueError(f"未知色散模型: {model}")
    
    def _sellmeier(self, wavelengths: torch.Tensor) -> torch.Tensor:
        """
        Sellmeier 方程
        
        n²(λ) - 1 = Σᵢ Bᵢ λ² / (λ² - Cᵢ)
        
        Args:
            wavelengths: 波长 (μm)
            
        Returns:
            折射率
        """
        lam = wavelengths
        lam_sq = lam ** 2
        
        n_sq_minus_1 = torch.zeros_like(lam)
        
        for B, C in zip(self.material.sellmeier_B, self.material.sellmeier_C):
            n_sq_minus_1 = n_sq_minus_1 + B * lam_sq / (lam_sq - C)
        
        n = torch.sqrt(n_sq_minus_1 + 1)
        return n
    
    def _drude_lorentz(self, wavelengths: torch.Tensor) -> torch.Tensor:
        """
        Drude-Lorentz 模型
        
        ε(ω) = ε∞ - ωₚ²/(ω² + iγω) + Σⱼ fⱼωⱼ²/(ωⱼ² - ω² - iΓⱼω)
        
        Args:
            wavelengths: 波长 (μm)
            
        Returns:
            复折射率
        """
        c = 299792458 * 1e6  # μm/s
        omega = 2 * torch.pi * c / wavelengths  # 角频率
        
        # Drude 项
        omega_p = self.material.plasma_freq
        gamma = self.material.damping_freq
        
        epsilon = 1.0 - omega_p**2 / (omega**2 + 1j * gamma * omega)
        
        # Lorentz 项
        if self.material.lorentz_strengths is not None:
            for f_j, omega_j in zip(self.material.lorentz_strengths, self.material.lorentz_freqs):
                epsilon = epsilon + f_j * omega_j**2 / (omega_j**2 - omega**2 - 1j * 0.01 * omega_j * omega)
        
        # 复折射率
        n_complex = torch.sqrt(epsilon)
        
        # 返回实部（用于约束计算）
        return n_complex.real
    
    def _cauchy(self, wavelengths: torch.Tensor) -> torch.Tensor:
        """
        Cauchy 方程
        
        n(λ) = A + B/λ² + C/λ⁴
        
        Args:
            wavelengths: 波长 (μm)
            
        Returns:
            折射率
        """
        lam = wavelengths
        n = (self.material.cauchy_A + 
             self.material.cauchy_B / lam**2 + 
             self.material.cauchy_C / lam**4)
        return n
    
    def compute_permittivity(self, wavelengths: torch.Tensor) -> torch.Tensor:
        """计算介电常数 ε = n²"""
        n = self.compute_refractive_index(wavelengths)
        return n ** 2
    
    def compute_group_index(self, wavelengths: torch.Tensor) -> torch.Tensor:
        """
        计算群折射率
        
        n_g = n - λ (dn/dλ)
        """
        wavelengths = wavelengths.requires_grad_(True)
        n = self.compute_refractive_index(wavelengths)
        
        dn_dlambda = torch.autograd.grad(
            n.sum(), wavelengths, create_graph=True
        )[0]
        
        n_g = n - wavelengths * dn_dlambda
        return n_g.detach()
    
    def compute_gvd(self, wavelengths: torch.Tensor) -> torch.Tensor:
        """
        计算群速度色散 (GVD)
        
        β₂ = d²β/dω² = λ³/(2πc²) d²n/dλ²
        
        Returns:
            GVD (ps²/km)
        """
        wavelengths = wavelengths.requires_grad_(True)
        n = self.compute_refractive_index(wavelengths)
        
        # 一阶导数
        dn_dlambda = torch.autograd.grad(n.sum(), wavelengths, create_graph=True)[0]
        
        # 二阶导数
        d2n_dlambda2 = torch.autograd.grad(dn_dlambda.sum(), wavelengths, create_graph=True)[0]
        
        # GVD 计算
        c = 299792458 * 1e6  # μm/s
        lam = wavelengths.detach()
        gvd = lam**3 / (2 * torch.pi * c**2) * d2n_dlambda2
        
        # 转换为 ps²/km
        gvd_ps2_km = gvd * 1e9  # 近似转换
        
        return gvd_ps2_km.detach()


class DispersionConstraint(nn.Module):
    """
    色散约束模块
    
    确保设计在多个波长下满足性能约束。
    考虑材料色散对器件性能的影响。
    """
    
    def __init__(
        self,
        material: str = "silicon",
        wavelengths: List[float] = None,
        bandwidth_constraint: str = 'average',  # 'average', 'min', 'weighted'
        weight: float = 1.0,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            material: 材料名称
            wavelengths: 波长列表 (μm)，默认 [1.3, 1.55, 1.7]
            bandwidth_constraint: 带宽约束类型
            weight: 约束权重
            device: 计算设备
        """
        super().__init__()
        
        self.calculator = DispersionCalculator(material, device=device)
        self.bandwidth_constraint = bandwidth_constraint
        self.weight = weight
        self.device = device or torch.device('cpu')
        
        # 默认波长（通信波段）
        if wavelengths is None:
            wavelengths = [1.30, 1.55, 1.70]
        
        self.register_buffer(
            'wavelengths',
            torch.tensor(wavelengths, dtype=torch.float32)
        )
        
        # 预计算折射率
        with torch.no_grad():
            self.register_buffer(
                'refractive_indices',
                self.calculator.compute_refractive_index(self.wavelengths)
            )
    
    def forward(
        self,
        permittivity_field: torch.Tensor,
        reference_wavelength_idx: int = 1,  # 默认以 1550nm 为参考
    ) -> Dict[str, torch.Tensor]:
        """
        计算色散约束
        
        Args:
            permittivity_field: 设计区域的介电常数分布 [H, W] 或 [B, H, W]
            reference_wavelength_idx: 参考波长索引
            
        Returns:
            约束字典，包含各波长的约束值
        """
        if permittivity_field.dim() == 2:
            permittivity_field = permittivity_field.unsqueeze(0)
        
        batch_size = permittivity_field.size(0)
        
        # 获取参考波长的折射率
        n_ref = self.refractive_indices[reference_wavelength_idx]
        
        constraints = {}
        total_violation = torch.zeros(batch_size, device=self.device)
        
        for i, (wavelength, n) in enumerate(zip(self.wavelengths, self.refractive_indices)):
            if i == reference_wavelength_idx:
                continue
            
            # 计算折射率差异导致的相位失配
            # Δn = n(λ) - n(λ_ref)
            delta_n = n - n_ref
            
            # 相位失配约束
            # 对于宽带器件，需要考虑不同波长下的性能一致性
            phase_mismatch = delta_n * permittivity_field.mean(dim=(1, 2)).sqrt()
            
            constraints[f'wavelength_{wavelength:.2f}um'] = phase_mismatch.abs()
            total_violation = total_violation + phase_mismatch.abs()
        
        # 根据带宽约束类型计算总约束
        if self.bandwidth_constraint == 'average':
            constraints['dispersion_total'] = total_violation / (len(self.wavelengths) - 1)
        elif self.bandwidth_constraint == 'min':
            constraints['dispersion_total'] = total_violation.min()
        elif self.bandwidth_constraint == 'weighted':
            # 中心波长权重更高
            weights = torch.softmax(-torch.abs(self.wavelengths - 1.55), dim=0)
            weights = weights[self.wavelengths != self.wavelengths[reference_wavelength_idx]]
            constraints['dispersion_total'] = (total_violation * weights).sum()
        
        constraints['dispersion_total'] = constraints['dispersion_total'] * self.weight
        
        return constraints
    
    def get_permittivity_at_wavelength(
        self,
        wavelength_idx: int
    ) -> torch.Tensor:
        """获取指定波长的折射率"""
        return self.refractive_indices[wavelength_idx] ** 2


class MultiWavelengthSimulator(nn.Module):
    """
    多波长仿真包装器
    
    在多个波长下评估设计，考虑材料色散。
    """
    
    def __init__(
        self,
        base_simulator,  # SimulatorInterface
        materials: Dict[str, str],  # region_name -> material_name
        wavelengths: List[float],
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            base_simulator: 基础仿真器实例
            materials: 各区域的材料映射
            wavelengths: 波长列表 (μm)
            device: 计算设备
        """
        super().__init__()
        self.simulator = base_simulator
        self.materials = materials
        self.device = device or torch.device('cpu')
        
        # 创建各材料的色散计算器
        self.dispersion_calculators = nn.ModuleDict({
            region: DispersionCalculator(material, device=device)
            for region, material in materials.items()
        })
        
        self.register_buffer(
            'wavelengths',
            torch.tensor(wavelengths, dtype=torch.float32)
        )
    
    def run_multiband(
        self,
        design_params: torch.Tensor,
        **kwargs
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        在多个波长下运行仿真
        
        Args:
            design_params: 设计参数
            
        Returns:
            各波长的仿真结果字典
        """
        results = {}
        
        for wavelength in self.wavelengths:
            # 更新材料属性
            permittivity_map = self._update_permittivity(design_params, wavelength)
            
            # 运行仿真
            result = self.simulator.run(permittivity_map, **kwargs)
            results[f'lambda_{wavelength:.2f}um'] = result
        
        return results
    
    def _update_permittivity(
        self,
        design_params: torch.Tensor,
        wavelength: torch.Tensor
    ) -> torch.Tensor:
        """根据波长更新介电常数"""
        # 简化实现：假设设计参数表示材料比例
        # 实际实现需要根据具体仿真器调整
        
        updated_params = design_params.clone()
        
        for region, calculator in self.dispersion_calculators.items():
            n = calculator.compute_refractive_index(wavelength.unsqueeze(0))
            # 更新介电常数
            # 这里需要根据实际设计参数的编码方式调整
        
        return updated_params


class WavelengthDemuxConstraint(nn.Module):
    """
    波长解复用器专用色散约束
    
    确保不同波长通道之间的隔离度。
    """
    
    def __init__(
        self,
        target_wavelengths: List[float],  # 各通道的目标波长
        channel_assignments: Dict[int, int],  # wavelength_idx -> output_port
        isolation_target: float = 20.0,  # dB
        weight: float = 1.0,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            target_wavelengths: 目标波长列表 (μm)
            channel_assignments: 波长到输出端口的映射
            isolation_target: 目标隔离度 (dB)
            weight: 约束权重
            device: 计算设备
        """
        super().__init__()
        
        self.target_wavelengths = target_wavelengths
        self.channel_assignments = channel_assignments
        self.isolation_target = isolation_target
        self.weight = weight
        self.device = device or torch.device('cpu')
    
    def forward(
        self,
        port_powers: Dict[str, torch.Tensor],  # port_name -> power_at_each_wavelength
    ) -> Dict[str, torch.Tensor]:
        """
        计算波长解复用约束
        
        Args:
            port_powers: 各端口的功率分布 {port_name: [powers_at_wavelengths]}
            
        Returns:
            约束字典
        """
        constraints = {}
        total_violation = torch.tensor(0.0, device=self.device)
        
        for wl_idx, target_wl in enumerate(self.target_wavelengths):
            target_port = self.channel_assignments[wl_idx]
            target_port_name = f'port_{target_port}'
            
            # 目标端口的功率
            target_power = port_powers[target_port_name][wl_idx]
            
            # 其他端口的串扰
            crosstalk = torch.tensor(0.0, device=self.device)
            for port_name, powers in port_powers.items():
                if port_name != target_port_name:
                    crosstalk = crosstalk + powers[wl_idx]
            
            # 隔离度约束：目标功率 >> 串扰
            # isolation_dB = 10 * log10(target_power / crosstalk)
            isolation_ratio = target_power / (crosstalk + 1e-8)
            isolation_db = 10 * torch.log10(isolation_ratio + 1e-8)
            
            # 违反约束
            violation = F.relu(self.isolation_target - isolation_db)
            constraints[f'isolation_wl_{target_wl:.2f}um'] = violation
            total_violation = total_violation + violation
        
        constraints['crosstalk_total'] = total_violation * self.weight
        
        return constraints


# 便捷函数
def get_material_index(material: str, wavelength: float) -> float:
    """
    获取材料在指定波长下的折射率
    
    Args:
        material: 材料名称
        wavelength: 波长 (μm)
        
    Returns:
        折射率
    """
    calculator = DispersionCalculator(material)
    with torch.no_grad():
        n = calculator.compute_refractive_index(torch.tensor([wavelength]))
    return n.item()


def list_available_materials() -> List[str]:
    """列出可用的材料"""
    return list(MATERIAL_DATABASE.keys())

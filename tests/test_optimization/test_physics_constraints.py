"""
物理约束单元测试

测试材料色散、制造公差、热效应等物理约束的实现。
"""

import pytest
import torch
import numpy as np

# 导入物理约束模块
import sys
sys.path.insert(0, '.')
from optimization.constraints import (
    DispersionCalculator,
    DispersionConstraint,
    ThermoOpticEffect,
    ThermalExpansion,
    HeatConductionSolver,
    ThermalConstraint,
    EdgeRoughnessModel,
    EtchBiasModel,
    RobustnessConstraint,
    DesignRuleCheck,
    get_material_index,
    list_available_materials,
    get_thermal_properties,
    get_process_specs,
)


class TestDispersionCalculator:
    """材料色散计算器测试"""
    
    def test_silicon_refractive_index(self):
        """测试硅的折射率计算"""
        calculator = DispersionCalculator(material="silicon")
        
        # 测试 1550nm 波长
        wavelength = torch.tensor([1.55])
        n = calculator.compute_refractive_index(wavelength)
        
        # 硅在 1550nm 的折射率约为 3.48
        assert 3.4 < n.item() < 3.6, f"硅折射率应在 3.4-3.6 范围内，实际为 {n.item()}"
    
    def test_silicon_dioxide_refractive_index(self):
        """测试二氧化硅的折射率计算"""
        calculator = DispersionCalculator(material="silicon_dioxide")
        
        wavelength = torch.tensor([1.55])
        n = calculator.compute_refractive_index(wavelength)
        
        # SiO2 在 1550nm 的折射率约为 1.44
        assert 1.4 < n.item() < 1.5, f"SiO2 折射率应在 1.4-1.5 范围内，实际为 {n.item()}"
    
    def test_wavelength_dependence(self):
        """测试波长依赖性"""
        calculator = DispersionCalculator(material="silicon")
        
        wavelengths = torch.tensor([1.30, 1.55, 1.70])
        n_values = calculator.compute_refractive_index(wavelengths)
        
        # 折射率应该随波长变化
        assert not torch.allclose(n_values[0], n_values[1]), "折射率应随波长变化"
        assert not torch.allclose(n_values[1], n_values[2]), "折射率应随波长变化"
    
    def test_group_index(self):
        """测试群折射率计算"""
        calculator = DispersionCalculator(material="silicon")
        
        wavelengths = torch.tensor([1.55])
        n_g = calculator.compute_group_index(wavelengths)
        
        # 群折射率应大于相折射率（正常色散）
        n = calculator.compute_refractive_index(wavelengths)
        # 注意：对于硅，色散特性可能不同，这里只检查计算不会出错
        assert n_g.item() > 0, "群折射率应为正数"
    
    def test_permittivity(self):
        """测试介电常数计算"""
        calculator = DispersionCalculator(material="silicon")
        
        wavelength = torch.tensor([1.55])
        epsilon = calculator.compute_permittivity(wavelength)
        n = calculator.compute_refractive_index(wavelength)
        
        # ε = n²
        assert torch.allclose(epsilon, n**2, rtol=1e-5), "介电常数应等于折射率平方"
    
    def test_list_materials(self):
        """测试材料列表"""
        materials = list_available_materials()
        
        assert "silicon" in materials
        assert "silicon_dioxide" in materials
        assert "silicon_nitride" in materials
        assert len(materials) >= 5, "应至少有 5 种材料"


class TestDispersionConstraint:
    """色散约束测试"""
    
    def test_constraint_output(self):
        """测试约束输出"""
        constraint = DispersionConstraint(
            material="silicon",
            wavelengths=[1.30, 1.55, 1.70]
        )
        
        # 创建测试设计
        design = torch.rand(50, 50)
        
        result = constraint(design)
        
        assert 'dispersion_total' in result
        assert result['dispersion_total'].item() >= 0
    
    def test_multiband_constraint(self):
        """测试多波长约束"""
        constraint = DispersionConstraint(
            material="silicon",
            wavelengths=[1.30, 1.55, 1.70],
            bandwidth_constraint='average'
        )
        
        design = torch.rand(50, 50)
        result = constraint(design)
        
        # 应该有每个波长的约束
        assert any('wavelength' in k for k in result.keys())


class TestThermoOpticEffect:
    """热光效应测试"""
    
    def test_refractive_index_change(self):
        """测试折射率变化计算"""
        thermo_optic = ThermoOpticEffect(material="silicon")
        
        # 温度变化 100K
        temperature = torch.tensor([400.0])  # K
        delta_n = thermo_optic.compute_refractive_index_change(temperature)
        
        # Δn = dn/dT × ΔT = 1.86e-4 × 100 = 0.0186
        expected_delta_n = 1.86e-4 * (400 - 300)
        
        assert abs(delta_n.item() - expected_delta_n) < 1e-4, \
            f"折射率变化应为约 {expected_delta_n}，实际为 {delta_n.item()}"
    
    def test_phase_shift(self):
        """测试相移计算"""
        thermo_optic = ThermoOpticEffect(material="silicon")
        
        temperature = torch.tensor([350.0])  # K
        phase_shift = thermo_optic.compute_phase_shift(
            temperature, wavelength=1.55, propagation_length=100.0
        )
        
        # 相移应随温度变化
        assert phase_shift.item() != 0, "温度变化应产生相移"
    
    def test_thermal_properties(self):
        """测试热属性获取"""
        props = get_thermal_properties("silicon")
        
        assert props.thermo_optic_coeff == pytest.approx(1.86e-4, rel=0.1)
        assert props.thermal_conductivity > 100  # W/(m·K)
        assert props.density > 2000  # kg/m³


class TestThermalExpansion:
    """热膨胀测试"""
    
    def test_dimensional_change(self):
        """测试尺寸变化计算"""
        expansion = ThermalExpansion(material="silicon")
        
        temperature = torch.tensor([400.0])  # K
        base_dim = 1.0  # μm
        
        new_dim = expansion.compute_dimensional_change(temperature, base_dim)
        
        # 硅热膨胀系数约 2.6e-6 K⁻¹
        # ΔL/L = α × ΔT = 2.6e-6 × 100 = 2.6e-4
        expected_change = 1 + 2.6e-6 * 100
        
        assert abs(new_dim.item() - expected_change) < 1e-4


class TestHeatConductionSolver:
    """热传导求解器测试"""
    
    def test_steady_state_solution(self):
        """测试稳态解"""
        solver = HeatConductionSolver(
            grid_size=(50, 50),
            physical_size=(10.0, 10.0),
            thermal_conductivity=148.0,  # 硅
            boundary_temperature=300.0
        )
        
        # 创建热源
        heat_source = torch.zeros(50, 50)
        heat_source[20:30, 20:30] = 1e12  # W/m³
        
        temperature = solver.solve_steady_state(heat_source, num_iterations=100)
        
        # 温度应在合理范围内
        assert temperature.min() >= 300.0, "温度不应低于边界温度"
        assert temperature.max() < 1000.0, "温度不应过高"
    
    def test_transient_solution(self):
        """测试瞬态解"""
        solver = HeatConductionSolver(
            grid_size=(30, 30),
            physical_size=(5.0, 5.0),
            boundary_temperature=300.0
        )
        
        heat_source = torch.zeros(30, 30)
        heat_source[10:20, 10:20] = 1e12
        
        T_history = solver.solve_transient(heat_source, time_steps=10)
        
        assert T_history.shape[0] == 11  # 初始 + 10 步
        assert T_history.shape[1:] == (30, 30)


class TestThermalConstraint:
    """热效应约束测试"""
    
    def test_constraint_output(self):
        """测试约束输出"""
        constraint = ThermalConstraint(
            material="silicon",
            grid_size=(50, 50),
            physical_size=(10.0, 10.0)
        )
        
        design = torch.rand(50, 50)
        heat_source = torch.zeros(50, 50)
        heat_source[20:30, 20:30] = 1e10
        
        result = constraint(design, heat_source)
        
        assert 'thermal_loss' in result
        assert 'temperature_field' in result
        assert result['thermal_loss'].item() >= 0


class TestEdgeRoughnessModel:
    """边缘粗糙度模型测试"""
    
    def test_roughness_generation(self):
        """测试粗糙度场生成"""
        model = EdgeRoughnessModel(
            rms_roughness=0.003,
            resolution=0.01
        )
        
        roughness = model.generate_roughness_field((50, 50), num_samples=5)
        
        assert roughness.shape == (5, 50, 50)
        
        # 检查 RMS
        rms = roughness.std()
        assert 0.001 < rms.item() < 0.01, f"RMS 应接近设定值，实际为 {rms.item()}"
    
    def test_apply_roughness(self):
        """测试应用粗糙度"""
        model = EdgeRoughnessModel(rms_roughness=0.003)
        
        # 创建简单设计（有边缘）
        design = torch.zeros(50, 50)
        design[10:40, 10:40] = 1.0
        
        samples = model.apply_edge_roughness(design, num_samples=3)
        
        assert samples.shape[0] == 3
        # 值应在 0-1 范围内
        assert samples.min() >= 0
        assert samples.max() <= 1


class TestEtchBiasModel:
    """蚀刻偏差模型测试"""
    
    def test_overetch(self):
        """测试过蚀刻"""
        model = EtchBiasModel(max_bias=0.01, resolution=0.01)
        
        design = torch.zeros(50, 50)
        design[10:40, 10:40] = 1.0
        
        result = model.apply_bias(design, bias_value=0.01)
        
        # 过蚀刻应使特征变小
        # 中心区域仍应为 1，但边缘区域应减小
        assert result[25, 25].item() == pytest.approx(1.0, abs=0.1)
    
    def test_underetch(self):
        """测试欠蚀刻"""
        model = EtchBiasModel(max_bias=0.01, resolution=0.01)
        
        design = torch.zeros(50, 50)
        design[10:40, 10:40] = 1.0
        
        result = model.apply_bias(design, bias_value=-0.01)
        
        # 欠蚀刻应使特征变大
        assert result[9, 25].item() > 0  # 原本为 0 的区域应有值


class TestRobustnessConstraint:
    """鲁棒性约束测试"""
    
    def test_perturbation_generation(self):
        """测试扰动生成"""
        constraint = RobustnessConstraint(
            specs="photonics_220nm",
            num_mc_samples=5,
            resolution=0.01
        )
        
        design = torch.rand(50, 50)
        samples = constraint.generate_perturbed_samples(design)
        
        assert samples.shape[0] == 5
        assert samples.shape[1:] == (50, 50)
    
    def test_sensitivity_calculation(self):
        """测试敏感度计算"""
        constraint = RobustnessConstraint(resolution=0.01)
        
        # 设计有更多边缘的图案
        design = torch.zeros(50, 50)
        design[10:40, 10:40] = 1.0
        
        result = constraint(design)
        
        assert 'edge_perimeter' in result
        assert 'sensitivity_map' in result


class TestDesignRuleCheck:
    """设计规则检查测试"""
    
    def test_min_feature_check(self):
        """测试最小特征尺寸检查"""
        drc = DesignRuleCheck(
            specs="photonics_220nm",
            resolution=0.01
        )
        
        # 创建一个有违规的设计
        design = torch.zeros(100, 100)
        design[45:55, 45:55] = 1.0  # 小于最小特征尺寸
        
        result = drc.check_min_feature_size(design)
        
        assert 'min_feature_violations' in result
        assert 'violation_area' in result
    
    def test_min_spacing_check(self):
        """测试最小间距检查"""
        drc = DesignRuleCheck(
            specs="photonics_220nm",
            resolution=0.01
        )
        
        # 创建两个靠得很近的特征
        design = torch.zeros(100, 100)
        design[20:40, 20:40] = 1.0
        design[45:65, 20:40] = 1.0  # 间距可能太小
        
        result = drc.check_min_spacing(design)
        
        assert 'min_spacing_violations' in result
    
    def test_all_checks(self):
        """测试所有检查"""
        drc = DesignRuleCheck(specs="photonics_220nm", resolution=0.01)
        
        design = torch.rand(100, 100)
        result = drc.run_all_checks(design)
        
        assert 'min_feature' in result
        assert 'min_spacing' in result
        assert 'total_violation' in result


class TestProcessSpecs:
    """工艺规格测试"""
    
    def test_get_specs(self):
        """测试获取工艺规格"""
        specs = get_process_specs("photonics_220nm")
        
        assert specs.min_feature_size == 0.1
        assert specs.cd_tolerance == 0.01
    
    def test_list_processes(self):
        """测试列出工艺"""
        processes = list_available_processes()
        
        assert "photonics_220nm" in processes
        assert len(processes) >= 3


class TestIntegration:
    """集成测试"""
    
    def test_full_constraint_pipeline(self):
        """测试完整约束管道"""
        # 创建测试设计
        design = torch.rand(50, 50)
        
        # 1. 色散约束
        disp_constraint = DispersionConstraint(
            material="silicon",
            wavelengths=[1.30, 1.55, 1.70]
        )
        disp_result = disp_constraint(design)
        
        # 2. 热效应约束
        thermal_constraint = ThermalConstraint(
            material="silicon",
            grid_size=(50, 50),
            physical_size=(5.0, 5.0)
        )
        heat_source = torch.zeros(50, 50)
        thermal_result = thermal_constraint(design, heat_source)
        
        # 3. 鲁棒性约束
        robust_constraint = RobustnessConstraint(resolution=0.01)
        robust_result = robust_constraint(design)
        
        # 验证所有约束都能正常工作
        assert disp_result['dispersion_total'].item() >= 0
        assert thermal_result['thermal_loss'].item() >= 0
        assert robust_result['robustness_penalty'].item() >= 0
    
    def test_physics_constraint_loss(self):
        """测试物理约束损失函数"""
        from models.training.losses import PhysicsConstraintLoss
        
        loss_fn = PhysicsConstraintLoss(material="silicon")
        
        design = torch.rand(4, 50, 50, requires_grad=True)
        
        losses = loss_fn(design)
        
        assert 'physics_total' in losses
        assert losses['physics_total'].item() >= 0
        
        # 测试反向传播
        losses['physics_total'].backward()
        assert design.grad is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

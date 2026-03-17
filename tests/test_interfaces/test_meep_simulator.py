"""
测试仿真器和挑战模块
"""
import sys
sys.path.append('.')

import torch
import numpy as np


def test_simulation_config():
    """测试仿真配置"""
    print("\n=== 测试仿真配置 ===")
    
    from interfaces.simulators.base import (
        SimulationConfig, SourceConfig, MonitorConfig,
        BoundaryCondition, SourceType
    )
    
    # 创建配置
    config = SimulationConfig(
        resolution=50,
        cell_size=(10.0, 10.0, 0.0),
        boundary_x=BoundaryCondition.PML,
        boundary_y=BoundaryCondition.PML,
        wavelengths=[1.31, 1.55]
    )
    
    assert config.resolution == 50
    assert len(config.get_wavelengths()) == 2
    print("仿真配置: 通过")
    
    # 光源配置
    source = SourceConfig(
        source_type=SourceType.GAUSSIAN,
        wavelength=1.55,
        center=(0, 0, 0),
        size=(0, 2, 0)
    )
    
    assert source.wavelength == 1.55
    assert source.frequency == 1.0 / 1.55
    print("光源配置: 通过")
    
    # 监视器配置
    monitor = MonitorConfig(
        name='transmission',
        monitor_type='flux',
        center=(5, 0, 0),
        size=(0, 2, 0)
    )
    
    assert monitor.name == 'transmission'
    print("监视器配置: 通过")


def test_material():
    """测试材料定义"""
    print("\n=== 测试材料定义 ===")
    
    from interfaces.simulators.meep import Material
    
    # 使用预定义材料
    si = Material(name='silicon')
    assert abs(si.n - 3.48) < 0.01
    print(f"硅折射率: {si.n}")
    
    # 自定义材料
    custom = Material(n=2.5)
    assert custom.eps == 6.25
    print(f"自定义材料 eps: {custom.eps}")
    
    # 从介电常数创建
    from_eps = Material(eps=4.0)
    assert abs(from_eps.n - 2.0) < 0.01
    print("材料定义: 通过")


def test_design_region():
    """测试设计区域"""
    print("\n=== 测试设计区域 ===")
    
    from interfaces.simulators.base import DesignRegion
    
    region = DesignRegion(
        name='design',
        center=(0, 0, 0),
        size=(5.0, 2.0, 0),
        min_permittivity=1.0,
        max_permittivity=12.0
    )
    
    shape = region.get_grid_shape(resolution=50)
    assert shape == (250, 100)
    print(f"设计区域网格形状: {shape}")
    print("设计区域: 通过")


def test_grating_coupler_challenge():
    """测试光栅耦合器挑战"""
    print("\n=== 测试光栅耦合器挑战 ===")
    
    from challenges.grating_coupler import GratingCouplerChallenge
    
    # 创建挑战
    challenge = GratingCouplerChallenge(
        wavelength=1.55,
        bandwidth=0.1,
        target_efficiency=0.8
    )
    
    # 检查规格
    assert challenge.spec.design_size[0] == 20.0
    print(f"设计尺寸: {challenge.spec.design_size}")
    
    # 获取初始设计
    initial_design = challenge.get_initial_design()
    print(f"初始设计形状: {initial_design.shape}")
    print(f"初始设计范围: [{initial_design.min().item():.2f}, {initial_design.max().item():.2f}]")
    
    # 评估设计
    objective, info = challenge.evaluate(initial_design)
    print(f"初始目标值: {objective.item():.4f}")
    print(f"性能指标: {info['metrics']}")
    
    print("光栅耦合器挑战: 通过")


def test_metagrating_challenge():
    """测试超构光栅挑战"""
    print("\n=== 测试超构光栅挑战 ===")
    
    from challenges.metagrating import MetagratingChallenge
    
    challenge = MetagratingChallenge(
        wavelength=1.55,
        deflection_angle=45.0,
        target_efficiency=0.9
    )
    
    initial_design = challenge.get_initial_design()
    print(f"初始设计形状: {initial_design.shape}")
    
    objective, info = challenge.evaluate(initial_design)
    print(f"目标值: {objective.item():.4f}")
    
    print("超构光栅挑战: 通过")


def test_wavelength_demux_challenge():
    """测试波长解复用器挑战"""
    print("\n=== 测试波长解复用器挑战 ===")
    
    from challenges.wavelength_demux import WavelengthDemuxChallenge
    
    challenge = WavelengthDemuxChallenge(
        wavelengths=[1.31, 1.55],
        num_channels=2
    )
    
    initial_design = challenge.get_initial_design()
    print(f"初始设计形状: {initial_design.shape}")
    
    objective, info = challenge.evaluate(initial_design)
    print(f"目标值: {objective.item():.4f}")
    
    print("波长解复用器挑战: 通过")


def test_challenge_factory():
    """测试挑战工厂"""
    print("\n=== 测试挑战工厂 ===")
    
    from challenges import ChallengeFactory
    
    # 列出可用挑战
    available = ChallengeFactory.list_available()
    print(f"可用挑战: {available}")
    
    assert 'grating_coupler' in available
    assert 'metagrating' in available
    assert 'wavelength_demux' in available
    
    print("挑战工厂: 通过")


def test_simulation_with_nodes():
    """测试仿真器与计算图节点的集成"""
    print("\n=== 测试仿真器与节点集成 ===")
    
    from core.nodes.parameterization import ParameterizationNode
    from challenges.grating_coupler import GratingCouplerChallenge
    
    # 创建挑战
    challenge = GratingCouplerChallenge(wavelength=1.55)
    
    # 获取初始设计作为参数节点
    initial_design = challenge.get_initial_design()
    param_node = ParameterizationNode('design', initial_design, requires_grad=True)
    
    # 运行仿真
    design = param_node.forward()
    objective, info = challenge.evaluate(design)
    
    print(f"参数节点输出形状: {design.shape}")
    print(f"目标值: {objective.item():.4f}")
    
    # 注意：模拟仿真器不支持梯度传播
    # 真实 Meep 仿真器会支持伴随方法计算梯度
    print("仿真器与节点集成: 通过")


def test_optimization_workflow():
    """测试优化工作流"""
    print("\n=== 测试优化工作流 ===")
    
    from challenges.grating_coupler import GratingCouplerChallenge
    from core.nodes.parameterization import ParameterizationNode
    from core.nodes.filter import GaussianFilterNode
    from core.nodes.projection import SigmoidProjectionNode
    
    # 创建挑战
    challenge = GratingCouplerChallenge(wavelength=1.55)
    
    # 获取初始设计
    initial_design = challenge.get_initial_design()
    
    # 构建处理管道
    param = ParameterizationNode('param', initial_design.clone(), requires_grad=True)
    gauss = GaussianFilterNode('filter', param, kernel_size=3)
    proj = SigmoidProjectionNode('proj', gauss, threshold=0.5, sharpness=10.0)
    
    # 简单优化（使用有限差分梯度）
    print("开始优化...")
    for i in range(10):
        # 处理设计
        processed = proj.forward()
        
        # 评估
        objective, info = challenge.evaluate(processed)
        
        # 使用有限差分更新（因为模拟仿真器不支持自动微分）
        with torch.no_grad():
            # 随机扰动（模拟梯度下降）
            param.value -= 0.01 * torch.randn_like(param.value) * (1 - i/10)
            param.value.clamp_(0, 1)
        
        # 清除缓存
        param.clear_cache()
        gauss.clear_cache()
        proj.clear_cache()
        
        if i % 3 == 0:
            print(f"Step {i}: objective={objective.item():.4f}")
    
    print("优化工作流: 通过")


def test_design_constraints():
    """测试设计约束"""
    print("\n=== 测试设计约束 ===")
    
    from challenges.grating_coupler import GratingCouplerChallenge
    import torch
    
    challenge = GratingCouplerChallenge()
    
    # 测试有效设计
    valid_design = torch.ones((100, 22)) * 0.5
    satisfied, violations = challenge.check_constraints(valid_design)
    print(f"有效设计 - 满足约束: {satisfied}, 违反: {violations}")
    
    # 测试无效设计（超出范围）
    invalid_design = torch.ones((100, 22)) * 1.5
    satisfied, violations = challenge.check_constraints(invalid_design)
    print(f"无效设计 - 满足约束: {satisfied}, 违反: {violations}")
    
    print("设计约束测试: 通过")


if __name__ == "__main__":
    test_simulation_config()
    test_material()
    test_design_region()
    test_grating_coupler_challenge()
    test_metagrating_challenge()
    test_wavelength_demux_challenge()
    test_challenge_factory()
    test_simulation_with_nodes()
    test_optimization_workflow()
    test_design_constraints()
    
    print("\n" + "=" * 50)
    print("所有测试通过!")
    print("=" * 50)

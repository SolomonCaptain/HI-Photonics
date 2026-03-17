"""
Meep 仿真器和设计挑战使用示例

本示例展示如何使用：
1. 仿真器配置
2. 设计挑战定义
3. 优化工作流
"""

import sys
from pathlib import Path
# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import matplotlib.pyplot as plt


def example_basic_config():
    """基础配置示例"""
    print("=" * 60)
    print("示例 1: 基础仿真配置")
    print("=" * 60)
    
    from interfaces.simulators.base import (
        SimulationConfig, SourceConfig, MonitorConfig,
        DesignRegion, BoundaryCondition, SourceType
    )
    
    # 创建仿真配置
    config = SimulationConfig(
        resolution=50,  # 50 像素/微米
        cell_size=(10.0, 5.0, 0.0),  # 10um x 5um 的 2D 区域
        boundary_x=BoundaryCondition.PML,
        boundary_y=BoundaryCondition.PML,
        pml_thickness=1.0,
        wavelengths=[1.31, 1.55],  # 两个波长
        simulation_time=200
    )
    
    print(f"仿真单元尺寸: {config.cell_size}")
    print(f"分辨率: {config.resolution} 像素/微米")
    print(f"波长: {config.get_wavelengths()} 微米")
    
    # 创建光源配置
    source = SourceConfig(
        source_type=SourceType.GAUSSIAN,
        wavelength=1.55,
        center=(0, 0, 0),
        size=(0, 2, 0),
        direction=1
    )
    print(f"\n光源类型: {source.source_type.value}")
    print(f"光源位置: {source.center}")
    
    # 创建监视器配置
    monitor = MonitorConfig(
        name='transmission',
        monitor_type='flux',
        center=(4, 0, 0),
        size=(0, 2, 0)
    )
    print(f"\n监视器名称: {monitor.name}")
    print(f"监视器类型: {monitor.monitor_type}")
    
    # 创建设计区域
    design_region = DesignRegion(
        name='design',
        center=(0, 0, 0),
        size=(5.0, 2.0, 0),
        min_permittivity=1.0,
        max_permittivity=12.0
    )
    print(f"\n设计区域尺寸: {design_region.size}")
    print(f"设计区域网格: {design_region.get_grid_shape(config.resolution)}")


def example_grating_coupler():
    """光栅耦合器设计示例"""
    print("\n" + "=" * 60)
    print("示例 2: 光栅耦合器设计")
    print("=" * 60)
    
    from challenges.grating_coupler import GratingCouplerChallenge
    
    # 创建光栅耦合器挑战
    challenge = GratingCouplerChallenge(
        wavelength=1.55,
        bandwidth=0.1,
        fiber_angle=10.0,
        target_efficiency=0.8
    )
    
    print(f"设计挑战: {challenge.name}")
    print(f"设计尺寸: {challenge.spec.design_size} 微米")
    print(f"目标效率: {challenge.target.metrics.get('efficiency', 'N/A')}")
    
    # 获取初始设计
    initial_design = challenge.get_initial_design()
    print(f"\n初始设计形状: {initial_design.shape}")
    print(f"初始设计范围: [{initial_design.min():.2f}, {initial_design.max():.2f}]")
    
    # 评估初始设计
    objective, info = challenge.evaluate(initial_design)
    print(f"\n初始目标值: {objective.item():.4f}")
    print(f"性能指标: {info['metrics']}")
    
    # 检查约束
    satisfied, violations = challenge.check_constraints(initial_design)
    print(f"\n约束满足: {satisfied}")
    if violations:
        print(f"违反项: {violations}")


def example_metagrating():
    """超构光栅设计示例"""
    print("\n" + "=" * 60)
    print("示例 3: 超构光栅设计")
    print("=" * 60)
    
    from challenges.metagrating import MetagratingChallenge
    
    # 创建超构光栅挑战
    challenge = MetagratingChallenge(
        wavelength=1.55,
        deflection_angle=45.0,
        period=1.0,
        target_efficiency=0.9
    )
    
    print(f"设计挑战: {challenge.name}")
    print(f"偏转角度: {challenge.deflection_angle} 度")
    print(f"光栅周期: {challenge.period} 微米")
    
    # 获取初始设计
    initial_design = challenge.get_initial_design()
    print(f"\n初始设计形状: {initial_design.shape}")
    
    # 评估
    objective, info = challenge.evaluate(initial_design)
    print(f"目标值: {objective.item():.4f}")


def example_wavelength_demux():
    """波长解复用器设计示例"""
    print("\n" + "=" * 60)
    print("示例 4: 波长解复用器设计")
    print("=" * 60)
    
    from challenges.wavelength_demux import WavelengthDemuxChallenge
    
    # 创建波长解复用器挑战
    challenge = WavelengthDemuxChallenge(
        wavelengths=[1.31, 1.55],
        num_channels=2,
        target_crosstalk=-20.0
    )
    
    print(f"设计挑战: {challenge.name}")
    print(f"波长通道: {challenge.wavelengths} 微米")
    print(f"目标串扰: {challenge.target_crosstalk} dB")
    
    # 获取初始设计
    initial_design = challenge.get_initial_design()
    print(f"\n初始设计形状: {initial_design.shape}")
    
    # 评估
    objective, info = challenge.evaluate(initial_design)
    print(f"目标值: {objective.item():.4f}")
    
    # 计算详细指标
    metrics = challenge.compute_metrics(info.get('result', {}))
    print(f"详细指标: {metrics}")


def example_optimization_pipeline():
    """优化管道示例"""
    print("\n" + "=" * 60)
    print("示例 5: 优化管道")
    print("=" * 60)
    
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
    smooth = GaussianFilterNode('smooth', param, kernel_size=5)
    proj = SigmoidProjectionNode('proj', smooth, threshold=0.5, sharpness=10.0)
    
    print("优化管道结构:")
    print("  param (设计参数) -> smooth (高斯滤波) -> proj (Sigmoid 投影)")
    
    # 简单优化循环
    print("\n开始优化...")
    best_objective = float('inf')
    best_design = None
    
    for i in range(20):
        # 前向传播
        processed = proj.forward()
        
        # 评估
        objective, info = challenge.evaluate(processed)
        
        # 记录最佳设计
        if objective.item() < best_objective:
            best_objective = objective.item()
            best_design = processed.clone()
        
        # 简单更新（实际应用中应使用梯度下降）
        with torch.no_grad():
            # 随机扰动模拟优化
            param.value -= 0.005 * torch.randn_like(param.value) * (1 - i/20)
            param.value.clamp_(0, 1)
        
        # 清除缓存
        param.clear_cache()
        smooth.clear_cache()
        proj.clear_cache()
        
        if i % 5 == 0:
            print(f"  Step {i:2d}: objective = {objective.item():.4f}")
    
    print(f"\n最佳目标值: {best_objective:.4f}")


def example_challenge_factory():
    """挑战工厂示例"""
    print("\n" + "=" * 60)
    print("示例 6: 挑战工厂")
    print("=" * 60)
    
    from challenges import ChallengeFactory
    
    # 列出可用挑战
    available = ChallengeFactory.list_available()
    print(f"可用挑战: {available}")
    
    # 使用工厂创建挑战
    for name in available:
        challenge = ChallengeFactory.create(name)
        print(f"\n创建挑战: {challenge.name}")
        print(f"  设计尺寸: {challenge.spec.design_size}")


def example_material_database():
    """材料数据库示例"""
    print("\n" + "=" * 60)
    print("示例 7: 材料数据库")
    print("=" * 60)
    
    from interfaces.simulators.meep import Material
    
    # 使用预定义材料
    print("预定义材料:")
    for name in ['silicon', 'sio2', 'sin', 'gaas']:
        mat = Material(name=name)
        print(f"  {name}: n = {mat.n:.2f}, eps = {mat.eps:.2f}")
    
    # 自定义材料
    custom = Material(n=2.5)
    print(f"\n自定义材料: n = {custom.n}, eps = {custom.eps}")
    
    # 注册新材料
    Material.register('custom_material', 3.0)
    new_mat = Material(name='custom_material')
    print(f"新材料: n = {new_mat.n}")


def main():
    """运行所有示例"""
    example_basic_config()
    example_grating_coupler()
    example_metagrating()
    example_wavelength_demux()
    example_optimization_pipeline()
    example_challenge_factory()
    example_material_database()
    
    print("\n" + "=" * 60)
    print("所有示例完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()

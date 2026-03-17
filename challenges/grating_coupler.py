"""
光栅耦合器设计挑战

实现光纤到芯片的光栅耦合器优化。
"""

from typing import Dict, Optional, Tuple, List
import torch
import numpy as np

from .base import (
    DesignChallenge, DesignSpec, PerformanceTarget,
    register_challenge
)
from interfaces.simulators.base import (
    SimulationConfig, SourceConfig, MonitorConfig,
    BoundaryCondition, SourceType
)


@register_challenge('grating_coupler')
class GratingCouplerChallenge(DesignChallenge):
    """
    光栅耦合器设计挑战
    
    目标：设计一个高效的光栅耦合器，将光纤光耦合到硅波导中。
    """
    
    def __init__(
        self,
        wavelength: float = 1.55,
        bandwidth: float = 0.1,
        fiber_angle: float = 10.0,
        target_efficiency: float = 0.8,
        **kwargs
    ):
        """
        Args:
            wavelength: 中心波长（微米）
            bandwidth: 带宽（微米）
            fiber_angle: 光纤角度（度）
            target_efficiency: 目标耦合效率
        """
        self.wavelength = wavelength
        self.bandwidth = bandwidth
        self.fiber_angle = fiber_angle
        self.target_efficiency = target_efficiency
        
        # 设计规格
        spec = DesignSpec(
            design_size=(20.0, 0.22),  # 20um 长, 220nm 厚
            resolution=100,
            min_eps=1.0,
            max_eps=12.0,  # 硅
            min_feature_size=0.1,
            wavelengths=[wavelength - bandwidth/2, wavelength, wavelength + bandwidth/2]
        )
        
        # 性能目标
        target = PerformanceTarget(
            metrics={'efficiency': target_efficiency},
            weights={'efficiency': 1.0},
            constraints={'efficiency': (0.0, 1.0)}
        )
        
        super().__init__('grating_coupler', spec, target, **kwargs)
    
    def setup_simulator(self):
        """设置仿真器"""
        from interfaces.simulators.meep import MeepSimulator, MEEP_AVAILABLE
        
        if not MEEP_AVAILABLE:
            # 返回模拟仿真器用于测试
            return MockGratingSimulator(self.spec)
        
        config = SimulationConfig(
            resolution=self.spec.resolution,
            cell_size=(self.spec.design_size[0] + 4, 6, 0),
            boundary_x=BoundaryCondition.PML,
            boundary_y=BoundaryCondition.PML,
            pml_thickness=1.0,
            wavelengths=self.spec.wavelengths,
            simulation_time=200
        )
        
        sim = MeepSimulator(config=config, device=self.device)
        
        # 添加设计区域
        from interfaces.simulators.base import DesignRegion
        design_region = DesignRegion(
            name='grating',
            center=(0, 0, 0),
            size=self.spec.design_size,
            min_permittivity=self.spec.min_eps,
            max_permittivity=self.spec.max_eps
        )
        sim.add_design_region(design_region)
        
        # 添加光源（光纤输入）
        source = SourceConfig(
            source_type=SourceType.GAUSSIAN,
            wavelength=self.wavelength,
            center=(0, 3.0, 0),
            size=(3.0, 0, 0),
            angle=self.fiber_angle,
            direction=-1
        )
        sim.add_source(source)
        
        # 添加监视器
        transmission = MonitorConfig(
            name='transmission',
            monitor_type='flux',
            center=(-self.spec.design_size[0]/2 - 1, 0, 0),
            size=(0, 1.0, 0)
        )
        sim.add_monitor(transmission)
        
        reflection = MonitorConfig(
            name='reflection',
            monitor_type='flux',
            center=(0, 4.0, 0),
            size=(3.0, 0, 0)
        )
        sim.add_monitor(reflection)
        
        return sim
    
    def compute_objective(self, result: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算目标函数"""
        # 目标：最大化透射效率
        if 'transmission' in result:
            efficiency = result['transmission'].mean()
            # 负号因为我们要最小化
            return -efficiency
        
        # 如果没有透射数据，返回大值
        return torch.tensor(1.0, device=self.device)
    
    def get_initial_design(self) -> torch.Tensor:
        """获取初始设计"""
        shape = self.spec.get_grid_shape()
        
        # 初始化为均匀光栅
        design = torch.zeros(shape, device=self.device)
        
        # 创建简单的光栅图案
        period_pixels = int(0.67 * self.spec.resolution)  # 典型周期
        duty_cycle = 0.5
        
        for i in range(0, shape[0], period_pixels):
            end = min(i + int(period_pixels * duty_cycle), shape[0])
            design[i:end, :] = 1.0
        
        return design
    
    def compute_efficiency(
        self,
        design_params: torch.Tensor
    ) -> Dict[str, float]:
        """计算耦合效率"""
        objective, info = self.evaluate(design_params)
        return {
            'efficiency': -objective.item(),
            'transmission': info.get('metrics', {}).get('transmission', 0),
            'reflection': info.get('metrics', {}).get('reflection', 0)
        }


class MockGratingSimulator:
    """模拟仿真器（用于无 Meep 环境测试）"""
    
    def __init__(self, spec: DesignSpec):
        self.spec = spec
    
    def run(self, design_params, **kwargs):
        """模拟运行"""
        import torch
        
        # 简单的模拟：效率与设计的均匀性相关
        if isinstance(design_params, torch.Tensor):
            params = design_params.detach().cpu().numpy()
        else:
            params = design_params
        
        # 模拟透射率
        # 更好的设计（更接近周期结构）效率更高
        uniformity = 1.0 - np.std(params)
        fill_factor = np.mean(params)
        
        # 模拟效率曲线
        efficiency = uniformity * (1 - 2 * abs(fill_factor - 0.5)) * 0.8
        
        return {
            'transmission': torch.tensor([efficiency] * len(self.spec.wavelengths)),
            'reflection': torch.tensor([1 - efficiency] * len(self.spec.wavelengths))
        }
    
    def compute_gradient(self, design_params, objective_grad, **kwargs):
        """模拟梯度"""
        if isinstance(design_params, torch.Tensor):
            shape = design_params.shape
            device = design_params.device
        else:
            shape = design_params.shape
            device = torch.device('cpu')
        
        # 随机梯度（模拟）
        return torch.randn(shape, device=device) * 0.1

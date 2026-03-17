"""
超构光栅设计挑战

实现高效的超构光栅偏转器设计。
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


@register_challenge('metagrating')
class MetagratingChallenge(DesignChallenge):
    """
    超构光栅设计挑战
    
    目标：设计一个超构光栅，将入射光以特定角度偏转。
    """
    
    def __init__(
        self,
        wavelength: float = 1.55,
        deflection_angle: float = 45.0,
        period: float = 1.0,
        target_efficiency: float = 0.9,
        **kwargs
    ):
        """
        Args:
            wavelength: 波长（微米）
            deflection_angle: 偏转角度（度）
            period: 光栅周期（微米）
            target_efficiency: 目标衍射效率
        """
        self.wavelength = wavelength
        self.deflection_angle = deflection_angle
        self.period = period
        self.target_efficiency = target_efficiency
        
        # 设计规格
        spec = DesignSpec(
            design_size=(period, 0.5),  # 单个周期
            resolution=100,
            min_eps=1.0,
            max_eps=12.0,
            min_feature_size=0.05,
            wavelengths=[wavelength]
        )
        
        # 性能目标
        target = PerformanceTarget(
            metrics={'diffraction_efficiency': target_efficiency},
            weights={'diffraction_efficiency': 1.0},
            constraints={'diffraction_efficiency': (0.0, 1.0)}
        )
        
        super().__init__('metagrating', spec, target, **kwargs)
    
    def setup_simulator(self):
        """设置仿真器"""
        from interfaces.simulators.meep import MeepSimulator, MEEP_AVAILABLE
        
        if not MEEP_AVAILABLE:
            return MockMetagratingSimulator(self.spec, self.deflection_angle)
        
        config = SimulationConfig(
            resolution=self.spec.resolution,
            cell_size=(self.period, 2.0, 0),
            boundary_x=BoundaryCondition.BLOCH,
            boundary_y=BoundaryCondition.PML,
            pml_thickness=0.5,
            wavelengths=self.spec.wavelengths,
            simulation_time=100
        )
        
        sim = MeepSimulator(config=config, device=self.device)
        
        # 设计区域
        from interfaces.simulators.base import DesignRegion
        design_region = DesignRegion(
            name='metagrating',
            center=(0, 0, 0),
            size=self.spec.design_size,
            min_permittivity=self.spec.min_eps,
            max_permittivity=self.spec.max_eps
        )
        sim.add_design_region(design_region)
        
        # 光源
        source = SourceConfig(
            source_type=SourceType.PLANE_WAVE,
            wavelength=self.wavelength,
            center=(0, -0.5, 0),
            size=(self.period, 0, 0),
            direction=1
        )
        sim.add_source(source)
        
        # 监视器
        # 目标衍射级次监视器
        transmission = MonitorConfig(
            name='diffraction',
            monitor_type='flux',
            center=(0, 0.5, 0),
            size=(self.period, 0, 0)
        )
        sim.add_monitor(transmission)
        
        return sim
    
    def compute_objective(self, result: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算目标函数"""
        if 'diffraction' in result:
            efficiency = result['diffraction'].mean()
            return -efficiency
        return torch.tensor(1.0, device=self.device)
    
    def get_initial_design(self) -> torch.Tensor:
        """获取初始设计"""
        shape = self.spec.get_grid_shape()
        
        # 初始化为均匀结构
        design = torch.ones(shape, device=self.device) * 0.5
        
        return design


class MockMetagratingSimulator:
    """模拟超构光栅仿真器"""
    
    def __init__(self, spec: DesignSpec, deflection_angle: float):
        self.spec = spec
        self.deflection_angle = deflection_angle
    
    def run(self, design_params, **kwargs):
        """模拟运行"""
        import torch
        
        if isinstance(design_params, torch.Tensor):
            params = design_params.detach().cpu().numpy()
        else:
            params = design_params
        
        # 模拟衍射效率
        # 更复杂的设计（更多变化）效率更高
        complexity = np.std(params) * 2
        efficiency = min(0.95, 0.5 + complexity)
        
        return {
            'diffraction': torch.tensor([efficiency])
        }
    
    def compute_gradient(self, design_params, objective_grad, **kwargs):
        """模拟梯度"""
        if isinstance(design_params, torch.Tensor):
            shape = design_params.shape
            device = design_params.device
        else:
            shape = design_params.shape
            device = torch.device('cpu')
        
        return torch.randn(shape, device=device) * 0.1

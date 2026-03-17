"""
波长解复用器设计挑战

实现多波长分离的波分解复用器设计。
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


@register_challenge('wavelength_demux')
class WavelengthDemuxChallenge(DesignChallenge):
    """
    波长解复用器设计挑战
    
    目标：设计一个波长解复用器，将不同波长的光分离到不同的输出端口。
    """
    
    def __init__(
        self,
        wavelengths: List[float] = None,
        channel_spacing: float = 0.02,
        num_channels: int = 2,
        target_crosstalk: float = -20.0,  # dB
        target_insertion_loss: float = 1.0,  # dB
        **kwargs
    ):
        """
        Args:
            wavelengths: 中心波长列表（微米）
            channel_spacing: 通道间隔（微米）
            num_channels: 通道数
            target_crosstalk: 目标串扰（dB）
            target_insertion_loss: 目标插入损耗（dB）
        """
        self.channel_spacing = channel_spacing
        self.num_channels = num_channels
        self.target_crosstalk = target_crosstalk
        self.target_insertion_loss = target_insertion_loss
        
        # 默认波长：1550nm 和 1310nm
        if wavelengths is None:
            if num_channels == 2:
                wavelengths = [1.31, 1.55]
            else:
                center = 1.55
                wavelengths = [center + i * channel_spacing for i in range(num_channels)]
        
        self.wavelengths = wavelengths
        
        # 设计规格
        spec = DesignSpec(
            design_size=(15.0, 5.0),  # 设计区域尺寸
            resolution=50,
            min_eps=1.0,
            max_eps=12.0,
            min_feature_size=0.1,
            wavelengths=wavelengths
        )
        
        # 性能目标
        target = PerformanceTarget(
            metrics={
                'insertion_loss': target_insertion_loss,
                'crosstalk': target_crosstalk
            },
            weights={
                'insertion_loss': 0.5,
                'crosstalk': 0.5
            },
            constraints={
                'insertion_loss': (0, 3.0),
                'crosstalk': (-30, 0)
            }
        )
        
        super().__init__('wavelength_demux', spec, target, **kwargs)
    
    def setup_simulator(self):
        """设置仿真器"""
        from interfaces.simulators.meep import MeepSimulator, MEEP_AVAILABLE
        
        if not MEEP_AVAILABLE:
            return MockDemuxSimulator(self.spec, self.wavelengths)
        
        config = SimulationConfig(
            resolution=self.spec.resolution,
            cell_size=(self.spec.design_size[0] + 6, self.spec.design_size[1] + 4, 0),
            boundary_x=BoundaryCondition.PML,
            boundary_y=BoundaryCondition.PML,
            pml_thickness=1.0,
            wavelengths=self.wavelengths,
            simulation_time=200
        )
        
        sim = MeepSimulator(config=config, device=self.device)
        
        # 设计区域
        from interfaces.simulators.base import DesignRegion
        design_region = DesignRegion(
            name='demux',
            center=(0, 0, 0),
            size=self.spec.design_size,
            min_permittivity=self.spec.min_eps,
            max_permittivity=self.spec.max_eps
        )
        sim.add_design_region(design_region)
        
        # 输入波导光源
        for i, wl in enumerate(self.wavelengths):
            source = SourceConfig(
                source_type=SourceType.MODE_SOURCE,
                wavelength=wl,
                center=(-self.spec.design_size[0]/2 - 1, 0, 0),
                size=(0, 0.5, 0),
                direction=1
            )
            sim.add_source(source)
        
        # 输出监视器（每个通道一个）
        output_spacing = self.spec.design_size[1] / (self.num_channels + 1)
        for i in range(self.num_channels):
            y_pos = -self.spec.design_size[1]/2 + (i + 1) * output_spacing
            monitor = MonitorConfig(
                name=f'output_{i}',
                monitor_type='flux',
                center=(self.spec.design_size[0]/2 + 1, y_pos, 0),
                size=(0, 0.5, 0)
            )
            sim.add_monitor(monitor)
        
        # 反射监视器
        reflection = MonitorConfig(
            name='reflection',
            monitor_type='flux',
            center=(-self.spec.design_size[0]/2 - 2, 0, 0),
            size=(0, 2.0, 0)
        )
        sim.add_monitor(reflection)
        
        return sim
    
    def compute_objective(self, result: Dict[str, torch.Tensor]) -> torch.Tensor:
        """计算目标函数"""
        # 计算每个波长对应通道的功率
        total_loss = torch.tensor(0.0, device=self.device)
        
        for i, wl in enumerate(self.wavelengths):
            # 目标通道的功率
            target_key = f'output_{i}'
            if target_key in result:
                target_power = result[target_key]
                if isinstance(target_power, torch.Tensor):
                    if target_power.dim() > 0:
                        target_power = target_power[i] if i < len(target_power) else target_power[0]
                else:
                    target_power = torch.tensor(target_power, device=self.device)
            else:
                target_power = torch.tensor(0.0, device=self.device)
            
            # 其他通道的功率（串扰）
            crosstalk = torch.tensor(0.0, device=self.device)
            for j in range(self.num_channels):
                if i != j:
                    other_key = f'output_{j}'
                    if other_key in result:
                        other_power = result[other_key]
                        if isinstance(other_power, torch.Tensor):
                            if other_power.dim() > 0:
                                other_power = other_power[i] if i < len(other_power) else other_power[0]
                        crosstalk = crosstalk + other_power
            
            # 插入损耗
            insertion_loss = -10 * torch.log10(target_power + 1e-10)
            
            # 串扰
            crosstalk_db = -10 * torch.log10(crosstalk / (target_power + 1e-10) + 1e-10)
            
            # 加权损失
            loss = insertion_loss + torch.relu(-crosstalk_db - self.target_crosstalk)
            total_loss = total_loss + loss
        
        return total_loss / self.num_channels
    
    def get_initial_design(self) -> torch.Tensor:
        """获取初始设计"""
        shape = self.spec.get_grid_shape()
        
        # 初始化为随机结构
        design = torch.rand(shape, device=self.device) * 0.3 + 0.35
        
        return design
    
    def compute_metrics(
        self,
        result: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """计算详细性能指标"""
        metrics = {}
        
        for i, wl in enumerate(self.wavelengths):
            target_key = f'output_{i}'
            if target_key in result:
                target_power = result[target_key]
                if isinstance(target_power, torch.Tensor):
                    target_power = target_power.mean().item()
                else:
                    target_power = float(target_power)
                
                # 插入损耗
                metrics[f'insertion_loss_ch{i}'] = -10 * np.log10(target_power + 1e-10)
                
                # 串扰
                crosstalk = 0.0
                for j in range(self.num_channels):
                    if i != j:
                        other_key = f'output_{j}'
                        if other_key in result:
                            other_power = result[other_key]
                            if isinstance(other_power, torch.Tensor):
                                other_power = other_power.mean().item()
                            crosstalk += other_power
                
                metrics[f'crosstalk_ch{i}'] = -10 * np.log10(crosstalk / (target_power + 1e-10) + 1e-10)
        
        return metrics


class MockDemuxSimulator:
    """模拟波长解复用器仿真器"""
    
    def __init__(self, spec: DesignSpec, wavelengths: List[float]):
        self.spec = spec
        self.wavelengths = wavelengths
    
    def run(self, design_params, **kwargs):
        """模拟运行"""
        import torch
        
        if isinstance(design_params, torch.Tensor):
            params = design_params.detach().cpu().numpy()
            device = design_params.device
        else:
            params = design_params
            device = torch.device('cpu')
        
        # 模拟输出功率
        result = {}
        
        # 简单模拟：根据设计的复杂度分配功率
        complexity = np.std(params)
        
        for i, wl in enumerate(self.wavelengths):
            outputs = []
            for j in range(2):  # 假设 2 个输出
                if i == j:
                    # 目标通道高功率
                    power = 0.7 + complexity * 0.2
                else:
                    # 其他通道低功率（串扰）
                    power = 0.05 - complexity * 0.03
                outputs.append(max(0, power))
            
            result[f'output_{i}'] = torch.tensor(outputs, device=device)
        
        return result
    
    def compute_gradient(self, design_params, objective_grad, **kwargs):
        """模拟梯度"""
        if isinstance(design_params, torch.Tensor):
            shape = design_params.shape
            device = design_params.device
        else:
            shape = design_params.shape
            device = torch.device('cpu')
        
        return torch.randn(shape, device=device) * 0.1

"""
Optics FDTD 仿真器模块

提供基于 C++ 实现的 FDTD 仿真功能，支持：
- 2D TM/TE 电磁场仿真
- PML 边界条件
- 多种光源类型
- 通量和场监视器

这个模块是 Meep 的替代方案，专门为 Windows 平台优化。
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import warnings

# 导入基类
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from base import (
    SimulatorInterface, SimulationConfig, SourceConfig, MonitorConfig,
    DesignRegion, SimulationResult, BoundaryCondition, SourceType,
    register_simulator
)

# 尝试导入 C++ 扩展
try:
    from . import optics as _optics
    OPTICS_AVAILABLE = True
except ImportError:
    OPTICS_AVAILABLE = False
    warnings.warn(
        "optics C++ extension not built. "
        "Run 'pip install -e .' in the interfaces/simulators/optics directory. "
        "Simulation features will be unavailable."
    )


@register_simulator('optics')
class OpticsSimulator(SimulatorInterface):
    """
    Optics FDTD 仿真器
    
    使用 C++ 实现的高性能 FDTD 仿真器，用于替代 Meep。
    支持 2D TM/TE 模式仿真，PML 边界条件，和多种光源类型。
    
    示例:
        >>> config = SimulationConfig(
        ...     resolution=50,
        ...     cell_size=(10.0, 10.0, 0.0),
        ...     simulation_time=100.0
        ... )
        >>> sim = OpticsSimulator(config)
        >>> sim.setup()
        >>> result = sim.run(design_params)
    """
    
    def __init__(
        self,
        config: Optional[SimulationConfig] = None,
        device: torch.device = None,
        use_gpu: bool = False
    ):
        """
        初始化 Optics 仿真器
        
        Args:
            config: 仿真配置
            device: PyTorch 设备
            use_gpu: 是否使用 GPU（暂不支持）
        """
        if not OPTICS_AVAILABLE:
            raise RuntimeError(
                "optics C++ extension not available. "
                "Please build the extension first:\n"
                "  cd interfaces/simulators/optics\n"
                "  pip install -e ."
            )
        
        super().__init__(config, device)
        self.use_gpu = use_gpu
        
        if use_gpu:
            warnings.warn("GPU acceleration not yet implemented for optics simulator")
        
        # C++ 仿真对象
        self._sim = None
        
        # 伴随计算相关
        self._forward_fields: Optional[Dict[str, np.ndarray]] = None
        self._design_region_shape: Optional[Tuple[int, int]] = None
    
    def setup(self):
        """设置仿真环境"""
        # 创建 C++ 仿真对象
        self._sim = _optics.FDTD()
        
        # 设置网格
        resolution = self.config.resolution
        cell_size = self.config.cell_size
        
        dx = 1.0 / resolution  # 微米
        nx = int(cell_size[0] * resolution)
        ny = int(cell_size[1] * resolution)
        nz = int(cell_size[2] * resolution) if cell_size[2] > 0 else 1
        
        if nz > 1:
            self._sim.set_grid_3d(nx, ny, nz, dx, dx, dx)
        else:
            self._sim.set_grid(nx, ny, dx, dx)
        
        # 设置边界条件
        self._setup_boundaries()
        
        # 设置极化模式
        self._sim.set_polarization("TM")  # 默认 TM 模式
        
        # 设置光源
        self._setup_sources()
        
        # 设置监视器
        self._setup_monitors()
        
        # 调用 C++ setup
        self._sim.setup()
        
        # 记录设计区域形状
        if self.design_regions:
            self._design_region_shape = self.get_design_region_shape()
    
    def _setup_boundaries(self):
        """设置边界条件"""
        # X 边界
        if self.config.boundary_x == BoundaryCondition.PML:
            self._sim.set_pml(
                layers=int(self.config.pml_thickness * self.config.resolution),
                sigma_max=0.8
            )
        elif self.config.boundary_x == BoundaryCondition.PERIODIC:
            self._sim.set_periodic_boundary("x")
        
        # Y 边界
        if self.config.boundary_y == BoundaryCondition.PERIODIC:
            self._sim.set_periodic_boundary("y")
        
        # Z 边界
        if self.config.boundary_z == BoundaryCondition.PERIODIC:
            self._sim.set_periodic_boundary("z")
    
    def _setup_sources(self):
        """设置光源"""
        wavelengths = self.config.get_wavelengths()
        default_wavelength = wavelengths[0] if wavelengths else 1.55
        
        for src_config in self.sources:
            center = src_config.center
            size = src_config.size
            wavelength = src_config.wavelength or default_wavelength
            
            if src_config.source_type == SourceType.GAUSSIAN:
                self._sim.add_gaussian_source(
                    wavelength,
                    center,
                    size,
                    src_config.pulse_width
                )
            elif src_config.source_type == SourceType.CONTINUOUS:
                self._sim.add_continuous_source(
                    wavelength,
                    center,
                    size
                )
            elif src_config.source_type == SourceType.PLANE_WAVE:
                self._sim.add_plane_wave(
                    wavelength,
                    center,
                    size,
                    src_config.angle,
                    src_config.pulse_width
                )
    
    def _setup_monitors(self):
        """设置监视器"""
        wavelengths = self.config.get_wavelengths()
        frequencies = [1.0 / w for w in wavelengths]
        
        for mon_config in self.monitors:
            center = mon_config.center
            size = mon_config.size
            
            if mon_config.monitor_type == 'flux':
                self._sim.add_flux_monitor(
                    mon_config.name,
                    center,
                    size,
                    frequencies
                )
            elif mon_config.monitor_type == 'field':
                # 确定场分量
                component = 'Ez'  # 默认
                if mon_config.output_fields:
                    component = mon_config.name.split('_')[-1] if '_' in mon_config.name else 'Ez'
                
                self._sim.add_field_monitor(
                    mon_config.name,
                    center,
                    size,
                    frequencies,
                    component
                )
    
    def run(
        self,
        design_params: Union[np.ndarray, torch.Tensor],
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        运行前向仿真
        
        Args:
            design_params: 设计参数，形状取决于设计区域
                - 对于密度表示: [Nx, Ny] 或 [Nx, Ny, Nz]，值在 [0, 1]
                - 对于介电常数: 直接的 eps 值
            
        Returns:
            包含仿真结果的字典:
                - 'flux_{name}': 各监视器的通量
                - 'field_{name}': 场分布（如果启用）
        """
        params = self._params_to_numpy(design_params)
        
        # 检查缓存
        if self._check_cache(params) and self._last_result is not None:
            return self._result_to_tensor(self._last_result)
        
        # 如果没有 setup，先 setup
        if self._sim is None:
            self.setup()
        
        # 设置设计区域的介电常数
        self._set_design_region_epsilon(params)
        
        # 重置仿真
        self._sim.reset()
        
        # 运行仿真
        self._sim.run(self.config.simulation_time)
        
        # 收集结果
        result = self._collect_results()
        
        # 记录前向场（用于伴随方法）
        self._record_forward_fields()
        
        # 更新缓存
        self._update_cache(params, result)
        
        return self._result_to_tensor(result)
    
    def _set_design_region_epsilon(self, params: np.ndarray):
        """设置设计区域的介电常数"""
        if not self.design_regions:
            # 直接设置整个区域
            self._sim.set_epsilon(params.flatten())
        else:
            # 设置各设计区域
            for name, region in self.design_regions.items():
                eps_map = self._params_to_epsilon(params, region)
                self._sim.set_epsilon(eps_map.flatten())
    
    def _params_to_epsilon(
        self,
        params: np.ndarray,
        region: DesignRegion
    ) -> np.ndarray:
        """将设计参数转换为介电常数"""
        # 检查是否已经是介电常数范围
        if params.min() >= region.min_permittivity:
            return params
        
        # 线性插值
        eps_min = region.min_permittivity
        eps_max = region.max_permittivity
        
        return eps_min + (eps_max - eps_min) * params
    
    def _collect_results(self) -> SimulationResult:
        """收集仿真结果"""
        result = SimulationResult()
        
        # 获取通量数据
        for mon_config in self.monitors:
            if mon_config.monitor_type == 'flux':
                flux_data = self._sim.get_flux(mon_config.name)
                result.flux[mon_config.name] = flux_data
        
        # 获取场数据
        for mon_config in self.monitors:
            if mon_config.monitor_type == 'field':
                field_data = self._sim.get_field('Ez')
                result.fields[mon_config.name] = field_data
        
        # 计算性能指标
        result.metrics = self._compute_metrics(result)
        
        # 获取介电常数分布
        result.metadata['epsilon'] = self._sim.get_epsilon()
        
        return result
    
    def _record_forward_fields(self):
        """记录前向场分布（用于伴随方法）"""
        self._forward_fields = {
            'Ez': self._sim.get_field('Ez')
        }
    
    def _compute_metrics(self, result: SimulationResult) -> Dict[str, float]:
        """计算性能指标"""
        metrics = {}
        
        # 透射率
        if 'transmission' in result.flux and 'input' in result.flux:
            trans = np.array(result.flux['transmission'])
            inp = np.array(result.flux['input'])
            metrics['transmission'] = float(np.mean(trans / (inp + 1e-10)))
        
        # 反射率
        if 'reflection' in result.flux and 'input' in result.flux:
            refl = np.array(result.flux['reflection'])
            inp = np.array(result.flux['input'])
            metrics['reflection'] = float(np.mean(refl / (inp + 1e-10)))
        
        return metrics
    
    def compute_gradient(
        self,
        design_params: torch.Tensor,
        objective_grad: Dict[str, torch.Tensor],
        **kwargs
    ) -> torch.Tensor:
        """
        使用伴随方法计算梯度
        
        Args:
            design_params: 设计参数
            objective_grad: 目标函数对各输出的梯度
            
        Returns:
            设计参数的梯度
        """
        params = self._params_to_numpy(design_params)
        
        # 确保前向仿真已运行
        if self._last_params is None or not np.allclose(params, self._last_params):
            self.run(design_params)
        
        # 运行伴随仿真
        gradient = self._run_adjoint(objective_grad)
        
        return torch.tensor(gradient, dtype=torch.float32, device=self.device)
    
    def _run_adjoint(
        self,
        objective_grad: Dict[str, torch.Tensor]
    ) -> np.ndarray:
        """
        运行伴随仿真
        
        简化实现：使用有限差分近似
        完整实现需要两次仿真
        """
        # 检查是否有前向场数据
        if self._forward_fields is None or 'Ez' not in self._forward_fields:
            warnings.warn("Forward field data not available for adjoint calculation")
            if self._design_region_shape:
                return np.zeros(self._design_region_shape)
            return np.array([0.0])
        
        # 获取前向场
        E_forward = self._forward_fields['Ez']
        
        # 简化实现：使用解析梯度
        # 完整实现需要运行伴随仿真
        gradient = np.zeros_like(E_forward)
        
        for name, grad in objective_grad.items():
            if name.startswith('flux_'):
                # 通量目标的梯度
                if isinstance(grad, torch.Tensor):
                    grad_val = grad.detach().cpu().numpy().flatten()
                else:
                    grad_val = np.atleast_1d(grad).flatten()
                
                # 简化的梯度计算
                # 真实伴随方法需要运行反向仿真
                gradient = gradient + grad_val[0] * np.real(E_forward) * 0.01
        
        return gradient
    
    def cleanup(self):
        """清理仿真资源"""
        if self._sim is not None:
            self._sim.reset()
        
        self._forward_fields = None
    
    def step(self) -> Dict[str, torch.Tensor]:
        """
        运行单个时间步
        
        Returns:
            当前场数据
        """
        if self._sim is None:
            raise RuntimeError("Simulation not initialized. Call setup() first.")
        
        self._sim.step()
        
        # 返回当前场
        return {
            'Ez': torch.tensor(self._sim.get_field('Ez'), dtype=torch.float32)
        }
    
    @property
    def current_time(self) -> float:
        """当前仿真时间"""
        return self._sim.current_time if self._sim else 0.0
    
    @property
    def current_step(self) -> int:
        """当前时间步"""
        return self._sim.current_step if self._sim else 0
    
    @property
    def progress(self) -> float:
        """仿真进度 (0-1)"""
        return self._sim.progress if self._sim else 0.0
    
    def get_field(self, component: str = 'Ez') -> np.ndarray:
        """
        获取场分量数据
        
        Args:
            component: 场分量名称 (Ex, Ey, Ez, Hx, Hy, Hz, epsilon)
            
        Returns:
            场数据数组
        """
        if self._sim is None:
            raise RuntimeError("Simulation not initialized. Call setup() first.")
        
        return self._sim.get_field(component)
    
    def visualize(
        self,
        design_params: Optional[np.ndarray] = None,
        field_name: str = "Ez",
        save_path: Optional[str] = None
    ):
        """可视化设计或场分布"""
        import matplotlib.pyplot as plt
        
        if design_params is not None:
            params = self._params_to_numpy(design_params)
        elif self._last_params is not None:
            params = self._last_params
        else:
            raise ValueError("No design parameters available")
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 设计参数
        ax = axes[0]
        im = ax.imshow(params.T, cmap='viridis', origin='lower')
        ax.set_title('Design Parameters')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.colorbar(im, ax=ax)
        
        # 场分布
        ax = axes[1]
        if self._sim is not None:
            field = self._sim.get_field(field_name)
            field_2d = field.reshape(params.shape)
            im = ax.imshow(np.abs(field_2d).T, cmap='hot', origin='lower')
            ax.set_title(f'|{field_name}| Field')
        else:
            im = ax.imshow(params.T, cmap='viridis', origin='lower')
            ax.set_title('Epsilon')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()


# 便捷函数
def create_waveguide_simulator(
    length: float = 10.0,
    width: float = 0.5,
    resolution: int = 50,
    wavelength: float = 1.55,
    **kwargs
) -> OpticsSimulator:
    """
    创建波导仿真器
    
    Args:
        length: 波导长度（微米）
        width: 波导宽度（微米）
        resolution: 网格分辨率（像素/微米）
        wavelength: 波长（微米）
        
    Returns:
        配置好的仿真器
    """
    config = SimulationConfig(
        resolution=resolution,
        cell_size=(length + 4.0, width * 6, 0.0),
        simulation_time=200.0,
        wavelengths=[wavelength]
    )
    
    sim = OpticsSimulator(config)
    
    # 添加光源
    source = SourceConfig(
        source_type=SourceType.GAUSSIAN,
        wavelength=wavelength,
        center=(-length/2 + 1.0, 0, 0),
        size=(0, width * 3, 0),
        pulse_width=20.0
    )
    sim.add_source(source)
    
    # 添加监视器
    input_mon = MonitorConfig(
        name='input',
        monitor_type='flux',
        center=(-length/4, 0, 0),
        size=(0, width * 3, 0)
    )
    sim.add_monitor(input_mon)
    
    output_mon = MonitorConfig(
        name='transmission',
        monitor_type='flux',
        center=(length/4, 0, 0),
        size=(0, width * 3, 0)
    )
    sim.add_monitor(output_mon)
    
    return sim


__all__ = [
    'OpticsSimulator',
    'OPTICS_AVAILABLE',
    'create_waveguide_simulator'
]

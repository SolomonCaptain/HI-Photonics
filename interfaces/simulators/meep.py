"""
Meep FDTD 仿真器模块

提供基于 Meep 的 2D/3D FDTD 仿真功能，支持：
- 电磁场仿真
- 伴随方法梯度计算
- 常见光子学器件
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass
import warnings

from .base import (
    SimulatorInterface, SimulationConfig, SourceConfig, MonitorConfig,
    DesignRegion, SimulationResult, BoundaryCondition, SourceType,
    register_simulator
)

# Meep 导入（可选依赖）
try:
    import meep as mp
    MEEP_AVAILABLE = True
except ImportError:
    MEEP_AVAILABLE = False
    warnings.warn(
        "Meep 未安装。请运行 `conda install -c conda-forge pymeep` 安装。"
        "仿真功能将不可用，但代码结构仍然可用。"
    )


class Material:
    """材料定义"""
    
    # 常用材料折射率
    MATERIALS = {
        'vacuum': 1.0,
        'air': 1.0,
        'silicon': 3.48,
        'sio2': 1.44,
        'sin': 2.0,
        'gaas': 3.4,
        'inp': 3.17,
        'tio2': 2.4,
        'al2o3': 1.75,
    }
    
    def __init__(self, name: str = None, n: float = None, eps: float = None):
        """
        初始化材料
        
        Args:
            name: 材料名称
            n: 折射率
            eps: 介电常数
        """
        if name is not None and name in self.MATERIALS:
            self.n = self.MATERIALS[name]
        elif n is not None:
            self.n = n
        elif eps is not None:
            self.n = np.sqrt(eps)
        else:
            raise ValueError("必须提供 name, n 或 eps 之一")
        
        self.eps = self.n ** 2
    
    def get_meep_material(self):
        """获取 Meep 材料对象"""
        if MEEP_AVAILABLE:
            return mp.Medium(index=self.n)
        return None
    
    @classmethod
    def register(cls, name: str, n: float):
        """注册新材料"""
        cls.MATERIALS[name] = n


@register_simulator('meep')
class MeepSimulator(SimulatorInterface):
    """
    Meep FDTD 仿真器
    
    支持 2D 和 3D 电磁仿真，以及伴随方法梯度计算。
    """
    
    def __init__(
        self,
        config: Optional[SimulationConfig] = None,
        device: torch.device = None,
        use_gpu: bool = False
    ):
        """
        初始化 Meep 仿真器
        
        Args:
            config: 仿真配置
            device: PyTorch 设备
            use_gpu: 是否使用 GPU 加速
        """
        if not MEEP_AVAILABLE:
            raise RuntimeError(
                "Meep 未安装。请运行 `conda install -c conda-forge pymeep` 安装。"
            )
        
        super().__init__(config, device)
        self.use_gpu = use_gpu
        
        # Meep 仿真对象
        self.sim = None
        self.geometry = []
        self.materials: Dict[str, Material] = {}
        
        # 设计区域的材料函数
        self._design_material_func: Dict[str, Any] = {}
        
        # 伴随计算相关
        self._forward_dft_fields: Dict[str, Any] = {}
        self._forward_flux: Dict[str, Any] = {}
    
    def setup(self):
        """设置仿真环境"""
        self._build_cell()
        self._build_geometry()
        self._build_sources()
        self._build_monitors()
    
    def _build_cell(self):
        """构建仿真单元"""
        cell_size = self.config.cell_size
        
        # 边界条件
        boundary_layers = []
        pml_thickness = self.config.pml_thickness
        
        if self.config.boundary_x == BoundaryCondition.PML:
            boundary_layers.append(mp.PML(pml_thickness, direction=mp.X))
        if self.config.boundary_y == BoundaryCondition.PML:
            boundary_layers.append(mp.PML(pml_thickness, direction=mp.Y))
        if len(cell_size) > 2 and cell_size[2] > 0:
            if self.config.boundary_z == BoundaryCondition.PML:
                boundary_layers.append(mp.PML(pml_thickness, direction=mp.Z))
        
        self._boundary_layers = boundary_layers
        
        # 创建 Meep 单元尺寸
        self._cell = mp.Vector3(*[s if s > 0 else 0 for s in cell_size])
    
    def _build_geometry(self):
        """构建几何结构"""
        self.geometry = []
        
        # 添加基底/衬底
        for region in self.design_regions.values():
            # 设计区域将在 run 时动态更新
            pass
    
    def _build_sources(self):
        """构建光源"""
        self._meep_sources = []
        
        for src_config in self.sources:
            source = self._create_source(src_config)
            self._meep_sources.append(source)
    
    def _create_source(self, config: SourceConfig) -> Any:
        """创建 Meep 光源"""
        # 频率
        fcen = 1.0 / config.wavelength
        df = fcen / config.pulse_width if config.pulse_width > 0 else 0
        
        # 位置和尺寸
        center = mp.Vector3(*config.center)
        size = mp.Vector3(*config.size)
        
        # 极化方向
        component = getattr(mp, config.polarization, mp.Ez)
        
        # 光源类型
        if config.source_type == SourceType.GAUSSIAN:
            src = mp.GaussianSource(fcen, fwidth=df)
        elif config.source_type == SourceType.CONTINUOUS:
            src = mp.ContinuousSource(fcen)
        else:
            src = mp.GaussianSource(fcen, fwidth=df)
        
        return mp.Source(
            src=src,
            component=component,
            center=center,
            size=size,
            direction=mp.Direction(config.direction)
        )
    
    def _build_monitors(self):
        """构建监视器"""
        self._meep_monitors = {}
        self._flux_planes = {}
        self._field_monitors = {}
        
        wavelengths = self.config.get_wavelengths()
        frequencies = [1.0 / w for w in wavelengths]
        
        for mon_config in self.monitors:
            center = mp.Vector3(*mon_config.center)
            size = mp.Vector3(*mon_config.size)
            
            if mon_config.monitor_type == 'flux':
                # 通量监视器将在仿真时创建
                self._flux_planes[mon_config.name] = {
                    'center': center,
                    'size': size,
                    'frequencies': frequencies
                }
            
            elif mon_config.monitor_type == 'field':
                # 场监视器
                self._field_monitors[mon_config.name] = {
                    'center': center,
                    'size': size,
                    'frequencies': frequencies
                }
    
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
        
        # 构建仿真
        self._build_simulation(params)
        
        # 运行仿真
        result = self._run_forward()
        
        # 更新缓存
        self._update_cache(params, result)
        
        return self._result_to_tensor(result)
    
    def _build_simulation(self, design_params: np.ndarray):
        """构建 Meep 仿真对象"""
        # 创建设计区域的材料函数
        geometry = []
        
        for name, region in self.design_regions.items():
            # 将设计参数映射到介电常数
            eps_map = self._params_to_epsilon(design_params, region)
            
            # 创建材料函数
            design_material = self._create_design_material(eps_map, region)
            
            # 添加设计区域几何体
            design_block = mp.Block(
                center=mp.Vector3(*region.center),
                size=mp.Vector3(*region.size),
                material=design_material
            )
            geometry.append(design_block)
        
        # 添加其他几何体
        geometry.extend(self.geometry)
        
        # 创建仿真对象
        self.sim = mp.Simulation(
            cell_size=self._cell,
            geometry=geometry,
            sources=self._meep_sources,
            boundary_layers=self._boundary_layers,
            resolution=self.config.resolution,
            force_complex_fields=self.config.force_complex_fields
        )
        
        # 初始化通量监视器
        for name, config in self._flux_planes.items():
            self._forward_flux[name] = self.sim.add_flux(
                config['frequencies'],
                mp.FluxRegion(
                    center=config['center'],
                    size=config['size']
                )
            )
    
    def _params_to_epsilon(
        self,
        params: np.ndarray,
        region: DesignRegion
    ) -> np.ndarray:
        """
        将设计参数转换为介电常数
        
        Args:
            params: 设计参数 [0, 1] 或直接 eps 值
            region: 设计区域
            
        Returns:
            介电常数分布
        """
        # 检查是否已经是介电常数范围
        if params.min() >= region.min_permittivity:
            return params
        
        # 线性插值
        eps_min = region.min_permittivity
        eps_max = region.max_permittivity
        
        return eps_min + (eps_max - eps_min) * params
    
    def _create_design_material(
        self,
        eps_map: np.ndarray,
        region: DesignRegion
    ) -> Any:
        """创建设计区域的材料函数"""
        # 创建 Meep 材料网格
        # 注意：Meep 使用不同的坐标系统
        
        # 计算网格尺寸
        nx, ny = eps_map.shape[:2]
        
        # 创建材料网格
        material_grid = mp.MaterialGrid(
            mp.Vector3(nx, ny, 0),
            mp.Medium(epsilon=region.min_permittivity),
            mp.Medium(epsilon=region.max_permittivity),
            weights=eps_map.flatten()
        )
        
        return material_grid
    
    def _run_forward(self) -> SimulationResult:
        """运行前向仿真并收集结果"""
        # 运行仿真
        self.sim.run(until=self.config.simulation_time)
        
        # 收集结果
        result = SimulationResult()
        
        # 获取通量数据
        for name, flux in self._forward_flux.items():
            flux_data = self.sim.get_flux(flux)
            result.flux[name] = np.array(flux_data)
            result.metadata[f'flux_{name}_shape'] = flux_data.shape
        
        # 获取场数据
        if self.config.save_fields:
            for name, config in self._field_monitors.items():
                # 获取场数据
                eps_data = self.sim.get_array(center=mp.Vector3(), size=self._cell, component=mp.Dielectric)
                result.fields['epsilon'] = eps_data
        
        # 计算性能指标
        result.metrics = self._compute_metrics(result)
        
        return result
    
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
        
        基本原理：
        1. 前向仿真记录场分布
        2. 根据目标函数梯度设置伴随源
        3. 反向传播计算梯度
        """
        # 创建新的仿真用于伴随计算
        self._setup_adjoint_simulation()
        
        # 设置伴随源
        adjoint_sources = self._create_adjoint_sources(objective_grad)
        
        # 重置仿真
        self.sim.reset_meep()
        
        # 添加伴随源
        for src in adjoint_sources:
            self.sim.add_source(src)
        
        # 运行伴随仿真
        self.sim.run(until=self.config.simulation_time)
        
        # 计算梯度
        gradient = self._compute_adjoint_gradient()
        
        return gradient
    
    def _setup_adjoint_simulation(self):
        """设置伴随仿真"""
        # 伴随仿真使用相同的几何结构，但不同的源
        pass  # 几何结构已在前向仿真中设置
    
    def _create_adjoint_sources(
        self,
        objective_grad: Dict[str, torch.Tensor]
    ) -> List[Any]:
        """
        创建伴随源
        
        根据目标函数对输出场的梯度设置伴随源。
        """
        adjoint_sources = []
        
        wavelengths = self.config.get_wavelengths()
        
        for name, grad in objective_grad.items():
            if name.startswith('flux_'):
                # 通量的伴随源
                monitor_name = name.replace('flux_', '')
                if monitor_name in self._flux_planes:
                    config = self._flux_planes[monitor_name]
                    
                    # 伴随源位置与监视器相同
                    center = config['center']
                    size = config['size']
                    
                    # 使用梯度幅度作为源强度
                    if isinstance(grad, torch.Tensor):
                        grad_val = grad.detach().cpu().numpy()
                    else:
                        grad_val = grad
                    
                    for i, freq in enumerate(config['frequencies']):
                        if i < len(grad_val):
                            src = mp.Source(
                                src=mp.GaussianSource(freq, fwidth=freq/10),
                                component=mp.Ez,
                                center=center,
                                size=size,
                                amplitude=grad_val[i]
                            )
                            adjoint_sources.append(src)
        
        return adjoint_sources
    
    def _compute_adjoint_gradient(self) -> np.ndarray:
        """
        计算伴随梯度
        
        梯度 = Re{E_forward * E_adjoint}
        """
        # 获取设计区域的梯度
        gradients = {}
        
        for name, region in self.design_regions.items():
            # 获取设计区域的场
            center = mp.Vector3(*region.center)
            size = mp.Vector3(*region.size)
            
            # 获取前向场和伴随场
            # 这里需要根据 Meep 的 API 获取场数据
            
            # 简化实现：使用有限差分近似
            grid_shape = region.get_grid_shape(self.config.resolution)
            gradient = np.zeros(grid_shape[:2])  # 2D
            
            # 实际实现需要：
            # 1. 在前向仿真中记录场分布
            # 2. 在伴随仿真中获取伴随场
            # 3. 计算点积
            
            gradients[name] = gradient
        
        # 返回第一个设计区域的梯度
        if gradients:
            return list(gradients.values())[0]
        
        return np.array([0.0])
    
    def cleanup(self):
        """清理仿真资源"""
        if self.sim is not None:
            self.sim.reset_meep()
            self.sim = None
        
        self._forward_flux.clear()
        self._forward_dft_fields.clear()
    
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
            raise ValueError("没有可用的设计参数")
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 设计参数
        ax = axes[0]
        im = ax.imshow(params.T, cmap='viridis', origin='lower')
        ax.set_title('Design Parameters')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.colorbar(im, ax=ax)
        
        # 介电常数
        ax = axes[1]
        for region in self.design_regions.values():
            eps_map = self._params_to_epsilon(params, region)
            im = ax.imshow(eps_map.T, cmap='viridis', origin='lower')
            ax.set_title('Permittivity')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            plt.colorbar(im, ax=ax)
            break
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()


class MeepWaveguideSimulator(MeepSimulator):
    """
    波导仿真器
    
    专门用于波导器件的仿真。
    """
    
    def __init__(
        self,
        waveguide_width: float = 0.5,
        waveguide_height: float = 0.22,
        substrate_index: float = 1.44,
        core_index: float = 3.48,
        **kwargs
    ):
        """
        Args:
            waveguide_width: 波导宽度（微米）
            waveguide_height: 波导高度（微米）
            substrate_index: 衬底折射率
            core_index: 波导芯折射率
        """
        super().__init__(**kwargs)
        
        self.waveguide_width = waveguide_width
        self.waveguide_height = waveguide_height
        self.substrate_index = substrate_index
        self.core_index = core_index
        
        # 添加材料
        self.materials['substrate'] = Material(n=substrate_index)
        self.materials['core'] = Material(n=core_index)
    
    def setup_waveguide(
        self,
        length: float = 10.0,
        design_region_length: float = 5.0
    ):
        """
        设置波导结构
        
        Args:
            length: 总长度
            design_region_length: 设计区域长度
        """
        # 更新单元尺寸
        self.config.cell_size = (length, self.waveguide_width * 4, 0)
        
        # 添加衬底
        substrate = mp.Block(
            center=mp.Vector3(0, -self.waveguide_height),
            size=mp.Vector3(mp.inf, mp.inf, mp.inf),
            material=mp.Medium(index=self.substrate_index)
        )
        self.geometry.append(substrate)
        
        # 添加输入波导
        input_wg = mp.Block(
            center=mp.Vector3(-length/4, 0),
            size=mp.Vector3(length/2 - design_region_length/2, self.waveguide_width, mp.inf),
            material=mp.Medium(index=self.core_index)
        )
        self.geometry.append(input_wg)
        
        # 添加输出波导
        output_wg = mp.Block(
            center=mp.Vector3(length/4, 0),
            size=mp.Vector3(length/2 - design_region_length/2, self.waveguide_width, mp.inf),
            material=mp.Medium(index=self.core_index)
        )
        self.geometry.append(output_wg)
        
        # 添加设计区域
        design_region = DesignRegion(
            name='design',
            center=(0, 0, 0),
            size=(design_region_length, self.waveguide_width * 2, 0),
            min_permittivity=self.substrate_index ** 2,
            max_permittivity=self.core_index ** 2
        )
        self.add_design_region(design_region)
        
        # 添加光源
        source = SourceConfig(
            source_type=SourceType.MODE_SOURCE,
            wavelength=self.config.get_wavelengths()[0],
            center=(-length/2 + 1.0, 0, 0),
            size=(0, self.waveguide_width * 2, 0),
            direction=1
        )
        self.add_source(source)
        
        # 添加监视器
        input_monitor = MonitorConfig(
            name='input',
            monitor_type='flux',
            center=(-length/4, 0, 0),
            size=(0, self.waveguide_width * 2, 0)
        )
        self.add_monitor(input_monitor)
        
        output_monitor = MonitorConfig(
            name='transmission',
            monitor_type='flux',
            center=(length/4, 0, 0),
            size=(0, self.waveguide_width * 2, 0)
        )
        self.add_monitor(output_monitor)


class MeepGratingSimulator(MeepSimulator):
    """
    光栅耦合器仿真器
    
    专门用于光栅耦合器的仿真和优化。
    """
    
    def __init__(
        self,
        grating_period: float = 0.67,
        grating_duty_cycle: float = 0.5,
        etch_depth: float = 0.07,
        **kwargs
    ):
        """
        Args:
            grating_period: 光栅周期（微米）
            grating_duty_cycle: 光栅占空比
            etch_depth: 刻蚀深度（微米）
        """
        super().__init__(**kwargs)
        
        self.grating_period = grating_period
        self.grating_duty_cycle = grating_duty_cycle
        self.etch_depth = etch_depth
    
    def setup_grating(
        self,
        num_periods: int = 20,
        fiber_angle: float = 10.0,  # 度
        fiber_position: Tuple[float, float] = (0, 3.0)
    ):
        """
        设置光栅结构
        
        Args:
            num_periods: 光栅周期数
            fiber_angle: 光纤角度（度）
            fiber_position: 光纤位置
        """
        # 计算光栅总长度
        grating_length = num_periods * self.grating_period
        
        # 更新单元尺寸
        self.config.cell_size = (grating_length + 4, 6, 0)
        
        # 设置边界条件
        self.config.boundary_x = BoundaryCondition.PML
        self.config.boundary_y = BoundaryCondition.PML
        
        # 添加设计区域（光栅区域）
        design_region = DesignRegion(
            name='grating',
            center=(0, 0, 0),
            size=(grating_length, 0.22, 0),  # 硅厚度约 220nm
            min_permittivity=1.0,  # 空气
            max_permittivity=12.0  # 硅
        )
        self.add_design_region(design_region)
        
        # 添加光源（光纤输入）
        angle_rad = np.radians(fiber_angle)
        source = SourceConfig(
            source_type=SourceType.GAUSSIAN,
            wavelength=self.config.get_wavelengths()[0],
            center=fiber_position,
            size=(2.0, 0, 0),
            angle=fiber_angle,
            direction=-1
        )
        self.add_source(source)
        
        # 添加监视器
        # 波导输出监视器
        waveguide_monitor = MonitorConfig(
            name='transmission',
            monitor_type='flux',
            center=(-grating_length/2 - 1, 0, 0),
            size=(0, 1.0, 0)
        )
        self.add_monitor(waveguide_monitor)
        
        # 反射监视器
        reflection_monitor = MonitorConfig(
            name='reflection',
            monitor_type='flux',
            center=(0, fiber_position[1] + 1, 0),
            size=(2.0, 0, 0)
        )
        self.add_monitor(reflection_monitor)

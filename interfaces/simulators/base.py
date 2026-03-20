"""
仿真器基类模块

定义仿真器的标准接口和通用功能。
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Any, List, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import torch
import numpy as np
from pathlib import Path


class SimulationType(Enum):
    """仿真类型"""
    FDTD_2D = "fdtd_2d"
    FDTD_3D = "fdtd_3d"
    MODE_2D = "mode_2d"
    RCWA = "rcwa"
    FEM = "fem"


class BoundaryCondition(Enum):
    """边界条件类型"""
    PML = "pml"           # 完美匹配层
    PERIODIC = "periodic"  # 周期边界
    PMC = "pmc"           # 完美磁导体
    PEC = "pec"           # 完美电导体
    BLOCH = "bloch"       # Bloch 边界


class SourceType(Enum):
    """光源类型"""
    GAUSSIAN = "gaussian"
    CONTINUOUS = "continuous"
    PLANE_WAVE = "plane_wave"
    MODE_SOURCE = "mode_source"
    DIPOLE = "dipole"


@dataclass
class SimulationConfig:
    """仿真配置"""
    # 网格和尺寸
    resolution: int = 50  # 像素/微米
    cell_size: Tuple[float, float, float] = (10.0, 10.0, 0.0)  # (x, y, z) 微米
    
    # 边界条件
    boundary_x: BoundaryCondition = BoundaryCondition.PML
    boundary_y: BoundaryCondition = BoundaryCondition.PML
    boundary_z: BoundaryCondition = BoundaryCondition.PML
    pml_thickness: float = 1.0  # PML 厚度（微米）
    
    # 仿真时间
    simulation_time: float = 100.0  # 时间步长单位
    
    # 波长设置
    wavelengths: Optional[List[float]] = None  # 波长列表（微米）
    wavelength_range: Optional[Tuple[float, float]] = None  # 波长范围
    num_wavelengths: int = 1
    
    # 精度设置
    force_complex_fields: bool = False
    accuracy: int = 2  # Meep 精度等级
    
    # 并行设置
    num_processes: int = 1
    
    # 输出设置
    output_dir: Optional[str] = None
    save_fields: bool = False
    verbose: bool = True
    
    # ========== 多物理场约束配置 ==========
    
    # 材料色散设置
    enable_dispersion: bool = False  # 是否启用材料色散
    material: str = "silicon"  # 主材料
    dispersion_model: str = "sellmeier"  # 色散模型类型
    
    # 热效应设置
    enable_thermal: bool = False  # 是否启用热效应
    reference_temperature: float = 300.0  # 参考温度 (K)
    max_temperature: float = 400.0  # 最大允许温度 (K)
    thermal_conductivity: Optional[float] = None  # 热导率（使用材料默认值）
    
    # 制造公差设置
    enable_manufacturing_tolerance: bool = False  # 是否启用制造公差
    process_node: str = "photonics_220nm"  # 工艺节点
    min_featureSize: float = 0.1  # 最小特征尺寸 (μm)
    cd_tolerance: float = 0.01  # 关键尺寸公差 (μm)
    
    # 鲁棒性优化设置
    num_mc_samples: int = 10  # 蒙特卡洛样本数
    robustness_weight: float = 0.1  # 鲁棒性权重
    
    def get_wavelengths(self) -> List[float]:
        """获取波长列表"""
        if self.wavelengths is not None:
            return self.wavelengths
        if self.wavelength_range is not None:
            return list(np.linspace(
                self.wavelength_range[0],
                self.wavelength_range[1],
                self.num_wavelengths
            ))
        return [1.55]  # 默认 1550nm


@dataclass
class SourceConfig:
    """光源配置"""
    source_type: SourceType = SourceType.GAUSSIAN
    wavelength: float = 1.55  # 微米
    frequency: Optional[float] = None  # 频率（自动从波长计算）
    
    # 位置和尺寸
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    size: Tuple[float, float, float] = (0.0, 2.0, 0.0)
    
    # 方向
    polarization: str = "Ez"  # 电场极化方向
    direction: int = 1  # +1 或 -1
    
    # 高斯光源参数
    pulse_width: float = 10.0  # 脉冲宽度（时间单位）
    
    # 平面波参数
    angle: float = 0.0  # 入射角度（度）
    
    def __post_init__(self):
        if self.frequency is None:
            self.frequency = 1.0 / self.wavelength


@dataclass
class MonitorConfig:
    """监视器配置"""
    name: str
    monitor_type: str  # 'field', 'flux', 'mode', 'energy'
    
    # 位置和尺寸
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    size: Tuple[float, float, float] = (0.0, 2.0, 0.0)
    
    # 频率点
    frequencies: Optional[List[float]] = None
    wavelengths: Optional[List[float]] = None
    
    # 输出
    output_fields: bool = False


@dataclass
class DesignRegion:
    """设计区域定义"""
    name: str
    center: Tuple[float, float, float]
    size: Tuple[float, float, float]
    
    # 材料约束
    min_permittivity: float = 1.0
    max_permittivity: float = 12.0
    
    # 网格
    grid_shape: Optional[Tuple[int, int, int]] = None
    
    def get_grid_shape(self, resolution: int) -> Tuple[int, int, int]:
        """根据分辨率计算网格形状"""
        if self.grid_shape is not None:
            return self.grid_shape
        return tuple(int(s * resolution) for s in self.size if s > 0)


@dataclass
class SimulationResult:
    """仿真结果"""
    # 通量/功率数据
    flux: Dict[str, np.ndarray] = field(default_factory=dict)
    
    # 场分布
    fields: Dict[str, np.ndarray] = field(default_factory=dict)
    
    # 模式数据
    modes: Dict[str, Any] = field(default_factory=dict)
    
    # 器件性能指标
    metrics: Dict[str, float] = field(default_factory=dict)
    
    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_tensor_dict(self, device: torch.device = None) -> Dict[str, torch.Tensor]:
        """转换为 PyTorch 张量"""
        result = {}
        for key, value in self.flux.items():
            result[key] = torch.tensor(value, dtype=torch.float32, device=device)
        for key, value in self.fields.items():
            result[key] = torch.tensor(value, dtype=torch.float32, device=device)
        return result


class SimulatorInterface(ABC):
    """
    仿真器接口基类
    
    定义所有仿真器必须实现的标准接口。
    """
    
    def __init__(
        self,
        config: Optional[SimulationConfig] = None,
        device: torch.device = None
    ):
        """
        初始化仿真器
        
        Args:
            config: 仿真配置
            device: PyTorch 设备
        """
        self.config = config or SimulationConfig()
        self.device = device or torch.device('cpu')
        
        # 光源和监视器
        self.sources: List[SourceConfig] = []
        self.monitors: List[MonitorConfig] = []
        
        # 设计区域
        self.design_regions: Dict[str, DesignRegion] = {}
        
        # 缓存
        self._last_params: Optional[np.ndarray] = None
        self._last_result: Optional[SimulationResult] = None
        
        # 伴随方法相关
        self._forward_fields: Optional[Dict[str, np.ndarray]] = None
        self._adjoint_sources: Optional[List[Any]] = None
    
    @abstractmethod
    def run(
        self,
        design_params: Union[np.ndarray, torch.Tensor],
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        运行前向仿真
        
        Args:
            design_params: 设计参数（介电常数分布或密度）
            
        Returns:
            包含仿真结果的字典
        """
        pass
    
    @abstractmethod
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
        pass
    
    def add_source(self, source: SourceConfig):
        """添加光源"""
        self.sources.append(source)
    
    def add_monitor(self, monitor: MonitorConfig):
        """添加监视器"""
        self.monitors.append(monitor)
    
    def add_design_region(self, region: DesignRegion):
        """添加设计区域"""
        self.design_regions[region.name] = region
    
    def setup(self):
        """
        设置仿真环境
        
        在运行仿真前调用，用于初始化网格、材料等。
        """
        pass
    
    def cleanup(self):
        """
        清理仿真环境
        
        仿真完成后调用，释放资源。
        """
        pass
    
    def _params_to_numpy(
        self,
        design_params: Union[np.ndarray, torch.Tensor]
    ) -> np.ndarray:
        """将设计参数转换为 numpy 数组"""
        if isinstance(design_params, torch.Tensor):
            return design_params.detach().cpu().numpy()
        return design_params
    
    def _result_to_tensor(
        self,
        result: SimulationResult
    ) -> Dict[str, torch.Tensor]:
        """将仿真结果转换为张量"""
        return result.to_tensor_dict(self.device)
    
    def _check_cache(self, design_params: np.ndarray) -> bool:
        """检查是否有缓存结果"""
        if self._last_params is None:
            return False
        return np.allclose(design_params, self._last_params)
    
    def _update_cache(self, design_params: np.ndarray, result: SimulationResult):
        """更新缓存"""
        self._last_params = design_params.copy()
        self._last_result = result
    
    def get_design_region_shape(self, name: str = None) -> Tuple[int, ...]:
        """
        获取设计区域的网格形状
        
        Args:
            name: 设计区域名称，None 表示第一个
            
        Returns:
            网格形状元组
        """
        if name is None:
            if not self.design_regions:
                raise ValueError("没有定义设计区域")
            name = list(self.design_regions.keys())[0]
        
        region = self.design_regions[name]
        return region.get_grid_shape(self.config.resolution)
    
    def get_design_region_size(self, name: str = None) -> Tuple[float, ...]:
        """获取设计区域的物理尺寸"""
        if name is None:
            if not self.design_regions:
                raise ValueError("没有定义设计区域")
            name = list(self.design_regions.keys())[0]
        
        return self.design_regions[name].size
    
    def validate_params(self, design_params: Union[np.ndarray, torch.Tensor]) -> bool:
        """验证设计参数的有效性"""
        params = self._params_to_numpy(design_params)
        
        # 检查值范围
        for region in self.design_regions.values():
            if params.min() < region.min_permittivity:
                return False
            if params.max() > region.max_permittivity:
                return False
        
        return True
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """获取上次仿真的性能指标"""
        if self._last_result is None:
            return {}
        return self._last_result.metrics.copy()
    
    def visualize(
        self,
        design_params: Optional[np.ndarray] = None,
        field_name: str = "Ez",
        save_path: Optional[str] = None
    ):
        """
        可视化设计或场分布
        
        Args:
            design_params: 设计参数，None 使用上次的结果
            field_name: 要可视化的场分量
            save_path: 保存路径
        """
        pass
    
    # ========== 多物理场约束支持 ==========
    
    def get_material_permittivity(
        self,
        wavelength: float,
        temperature: Optional[float] = None,
    ) -> float:
        """
        获取材料在指定波长和温度下的介电常数
        
        Args:
            wavelength: 波长 (μm)
            temperature: 温度 (K)，None 使用参考温度
            
        Returns:
            介电常数
        """
        # 默认实现返回常数折射率
        # 子类应覆盖此方法以支持色散和热光效应
        n = 3.48  # 硅在 1550nm 的折射率
        
        if temperature is not None and self.config.enable_thermal:
            # 热光效应修正
            thermo_optic_coeff = 1.86e-4  # K⁻¹
            delta_T = temperature - self.config.reference_temperature
            n = n + thermo_optic_coeff * delta_T
        
        return n ** 2
    
    def apply_thermal_effects(
        self,
        design_params: Union[np.ndarray, torch.Tensor],
        temperature_field: Optional[Union[np.ndarray, torch.Tensor]] = None,
        heat_source: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        应用热效应修正到设计参数
        
        Args:
            design_params: 设计参数（介电常数分布）
            temperature_field: 温度场（可选）
            heat_source: 热源分布（可选，用于计算温度场）
            
        Returns:
            修正后的设计参数
        """
        if not self.config.enable_thermal:
            return design_params
        
        # 简化实现：子类应覆盖此方法
        # 实际实现需要求解热传导方程并更新介电常数
        return design_params
    
    def apply_manufacturing_variations(
        self,
        design_params: Union[np.ndarray, torch.Tensor],
        variation_type: str = 'combined',
        num_samples: int = 1,
    ) -> List[Union[np.ndarray, torch.Tensor]]:
        """
        应用制造公差变化
        
        Args:
            design_params: 设计参数
            variation_type: 变化类型 ('roughness', 'bias', 'combined')
            num_samples: 样本数
            
        Returns:
            变化后的设计参数样本列表
        """
        if not self.config.enable_manufacturing_tolerance:
            return [design_params]
        
        # 简化实现：子类应覆盖此方法
        return [design_params] * num_samples
    
    def run_multiband(
        self,
        design_params: Union[np.ndarray, torch.Tensor],
        wavelengths: Optional[List[float]] = None,
        **kwargs
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        在多个波长下运行仿真（考虑材料色散）
        
        Args:
            design_params: 设计参数
            wavelengths: 波长列表，None 使用配置中的波长
            
        Returns:
            各波长的仿真结果字典
        """
        if wavelengths is None:
            wavelengths = self.config.get_wavelengths()
        
        results = {}
        for wl in wavelengths:
            # 更新材料属性
            permittivity = self.get_material_permittivity(wl)
            
            # 运行仿真
            result = self.run(design_params, wavelength=wl, **kwargs)
            results[f'lambda_{wl:.2f}um'] = result
        
        return results
    
    def compute_robust_performance(
        self,
        design_params: Union[np.ndarray, torch.Tensor],
        objective_fn,
        num_samples: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        计算鲁棒性能（考虑制造公差）
        
        Args:
            design_params: 设计参数
            objective_fn: 目标函数
            num_samples: 样本数，None 使用配置中的值
            
        Returns:
            鲁棒性能指标字典
        """
        if num_samples is None:
            num_samples = self.config.num_mc_samples
        
        # 生成制造变化样本
        samples = self.apply_manufacturing_variations(
            design_params,
            variation_type='combined',
            num_samples=num_samples
        )
        
        # 评估各样本性能
        performances = []
        for sample in samples:
            result = self.run(sample)
            perf = objective_fn(result)
            performances.append(perf)
        
        performances = torch.stack(performances)
        
        return {
            'mean_performance': performances.mean(dim=0),
            'std_performance': performances.std(dim=0),
            'min_performance': performances.min(dim=0)[0],
            'max_performance': performances.max(dim=0)[0],
            'robustness_score': performances.mean(dim=0) - self.config.robustness_weight * performances.std(dim=0),
        }
    
    def validate_design_rules(
        self,
        design_params: Union[np.ndarray, torch.Tensor],
    ) -> Dict[str, float]:
        """
        验证设计规则
        
        Args:
            design_params: 设计参数
            
        Returns:
            设计规则检查结果
        """
        results = {
            'valid': True,
            'min_feature_violation': 0.0,
            'min_spacing_violation': 0.0,
            'total_violation': 0.0,
        }
        
        if not self.config.enable_manufacturing_tolerance:
            return results
        
        # 简化实现：子类应覆盖此方法
        return results


class SimulatorFactory:
    """仿真器工厂类"""
    
    _simulators: Dict[str, type] = {}
    
    @classmethod
    def register(cls, name: str, simulator_class: type):
        """注册仿真器"""
        cls._simulators[name] = simulator_class
    
    @classmethod
    def create(
        cls,
        name: str,
        config: Optional[SimulationConfig] = None,
        **kwargs
    ) -> SimulatorInterface:
        """创建仿真器实例"""
        if name not in cls._simulators:
            raise ValueError(f"未知的仿真器类型: {name}，可用: {list(cls._simulators.keys())}")
        return cls._simulators[name](config=config, **kwargs)
    
    @classmethod
    def list_available(cls) -> List[str]:
        """列出可用的仿真器"""
        return list(cls._simulators.keys())


def register_simulator(name: str):
    """仿真器注册装饰器"""
    def decorator(cls):
        SimulatorFactory.register(name, cls)
        return cls
    return decorator

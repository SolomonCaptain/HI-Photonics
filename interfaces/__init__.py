"""
接口模块

提供与外部仿真器、代工厂、可视化工具的接口。
"""

from .simulators.base import (
    SimulatorInterface,
    SimulatorFactory,
    register_simulator,
    SimulationConfig,
    SourceConfig,
    MonitorConfig,
    DesignRegion,
    SimulationResult,
    SimulationType,
    BoundaryCondition,
    SourceType
)

# Meep 仿真器（可选）
try:
    from .simulators.meep import (
        MeepSimulator,
        MeepWaveguideSimulator,
        MeepGratingSimulator,
        Material,
        MEEP_AVAILABLE
    )
except ImportError:
    MEEP_AVAILABLE = False
    MeepSimulator = None
    MeepWaveguideSimulator = None
    MeepGratingSimulator = None
    Material = None

__all__ = [
    # 仿真器基类
    'SimulatorInterface',
    'SimulatorFactory',
    'register_simulator',
    
    # 配置类
    'SimulationConfig',
    'SourceConfig',
    'MonitorConfig',
    'DesignRegion',
    'SimulationResult',
    
    # 枚举
    'SimulationType',
    'BoundaryCondition',
    'SourceType',
    
    # 材料
    'Material',
    
    # Meep 仿真器
    'MeepSimulator',
    'MeepWaveguideSimulator',
    'MeepGratingSimulator',
    'MEEP_AVAILABLE',
]

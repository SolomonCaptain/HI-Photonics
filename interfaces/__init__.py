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

# Optics FDTD 仿真器（C++ 实现）
try:
    from .simulators.optics import (
        OpticsSimulator,
        OPTICS_AVAILABLE,
        create_waveguide_simulator
    )
except ImportError:
    OPTICS_AVAILABLE = False
    OpticsSimulator = None
    create_waveguide_simulator = None

# RCWA 仿真器
try:
    from .simulators.rcwa import (
        RCWASimulator,
        RCWAConfig,
        RCWA_AVAILABLE
    )
except ImportError:
    RCWA_AVAILABLE = False
    RCWASimulator = None
    RCWAConfig = None

# Foundry 接口（可选）
try:
    from .foundry.design_rules import (
        DesignRuleChecker,
        DesignRule,
        RuleType,
        Violation
    )
except ImportError:
    DesignRuleChecker = None
    DesignRule = None
    RuleType = None
    Violation = None

try:
    from .foundry.gds import (
        GDSExporter,
        GDSLayer,
        GDSConfig
    )
except ImportError:
    GDSExporter = None
    GDSLayer = None
    GDSConfig = None

# 可视化模块
from .visualization.field import (
    FieldVisualizer,
    FieldPlotConfig,
    plot_field,
    plot_intensity,
    create_field_visualizer
)

from .visualization.structure import (
    StructureVisualizer,
    StructurePlotConfig,
    plot_design,
    plot_binary_structure,
    create_structure_visualizer
)

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
    
    # Optics 仿真器
    'OpticsSimulator',
    'OPTICS_AVAILABLE',
    'create_waveguide_simulator',
    
    # RCWA 仿真器
    'RCWASimulator',
    'RCWAConfig',
    'RCWA_AVAILABLE',
    
    # Foundry 接口
    'DesignRuleChecker',
    'DesignRule',
    'RuleType',
    'Violation',
    'GDSExporter',
    'GDSLayer',
    'GDSConfig',
    
    # 可视化 - 场分布
    'FieldVisualizer',
    'FieldPlotConfig',
    'plot_field',
    'plot_intensity',
    'create_field_visualizer',
    
    # 可视化 - 结构
    'StructureVisualizer',
    'StructurePlotConfig',
    'plot_design',
    'plot_binary_structure',
    'create_structure_visualizer',
]

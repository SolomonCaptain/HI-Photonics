"""
HI-Photonics: 光子学逆向设计框架

一个基于深度学习的光子学器件逆向设计平台，整合了：
- 多种逆向设计模型（TNN, MDN, CGAN, PINN, GNN, HiLAB）
- LLM 智能助手（自然语言设计意图解析、工作流配置推荐）
- 完整的数据模块（生成器、加载器、预处理、增强）
- FDTD/RCWA 仿真器接口（Optics C++ FDTD 已编译）
- 贝叶斯优化和多物理场约束
- 完整的工作流管道

使用示例:
    >>> from hi_photonics import create_pipeline, run_quick_design
    >>> 
    >>> # 快速逆向设计
    >>> result = run_quick_design(
    ...     challenge_name="grating_coupler",
    ...     target_performance={"efficiency": 0.8},
    ...     num_iterations=50
    ... )
    >>> 
    >>> # 完整管道
    >>> pipeline = create_pipeline(
    ...     challenge_name="metagrating",
    ...     model_type="hilab",
    ...     num_epochs=100
    ... )
    >>> result = pipeline.run()
"""

__version__ = "0.3.0"
__author__ = "HI-Photonics Team"

# 核心模块
from core import (
    # 基础类型
    Node, Graph, Params, TensorLike,
    # 标准节点
    ParameterizationNode, SimulationNode, ObjectiveNode
)

# 模型模块
from models import (
    # 基类
    BaseModel, ModelConfig,
    SurrogateModel, InverseModel, GenerativeModel,
    # TNN
    TandemNetwork, create_tnn_for_challenge,
    # MDN
    MDN, MDNTandemNetwork, create_mdn_for_challenge,
    # CGAN
    CGAN, WGAN_GP, create_cgan_for_challenge,
    # PINN
    PhysicsInformedNet, MaxwellPINN, create_pinn_for_photonics,
    # HiLAB
    HiLABEngine, create_hilab_for_challenge,
    # 训练工具
    get_loss, get_metric, get_default_callbacks
)

# 数据模块
from data import (
    # 数据集
    PhotonicsDataset, HDF5Dataset, SyntheticDataset,
    # 数据加载
    create_dataloaders, save_dataset_to_hdf5,
    # 预处理
    DataAugmentation
)

# 挑战模块
from challenges import (
    # 基类
    DesignChallenge, DesignSpec, PerformanceTarget,
    # 工厂
    ChallengeFactory, register_challenge,
    # 具体挑战
    GratingCouplerChallenge, MetagratingChallenge, WavelengthDemuxChallenge
)

# 接口模块
from interfaces import (
    # 仿真器基类
    SimulatorInterface, SimulatorFactory, register_simulator,
    SimulationConfig, SourceConfig, MonitorConfig,
    # Meep 仿真器
    MEEP_AVAILABLE,
    # Optics 仿真器
    OPTICS_AVAILABLE
)

# 优化模块
from optimization import (
    # 约束
    DispersionConstraint, ThermalConstraint, RobustnessConstraint
)

# 工作流模块
from workflows import (
    # 管道
    DesignPipeline, PipelineConfig, create_pipeline, run_quick_design,
    # 调度
    TaskDispatcher, TaskStatus, submit_task, get_task_result
)


def get_version() -> str:
    """获取版本号"""
    return __version__


def list_available_simulators() -> list:
    """列出可用的仿真器"""
    simulators = []
    if MEEP_AVAILABLE:
        simulators.append('meep')
    if OPTICS_AVAILABLE:
        simulators.append('optics')
    simulators.extend(SimulatorFactory.list_available())
    return list(set(simulators))


def list_available_challenges() -> list:
    """列出可用的挑战"""
    return ChallengeFactory.list_available()


def quick_start(
    challenge_name: str = "grating_coupler",
    model_type: str = "hilab",
    target: dict = None,
    **kwargs
) -> dict:
    """
    快速开始逆向设计
    
    Args:
        challenge_name: 挑战名称
        model_type: 模型类型
        target: 目标性能
        **kwargs: 其他参数
        
    Returns:
        设计结果
    """
    if target is None:
        challenge = ChallengeFactory.create(challenge_name)
        target = challenge.get_default_target()
    
    return run_quick_design(
        challenge_name=challenge_name,
        target_performance=target,
        **kwargs
    )


__all__ = [
    # 版本信息
    '__version__',
    'get_version',
    
    # 核心模块
    'Node', 'Graph', 'Params', 'TensorLike',
    'ParameterizationNode', 'SimulationNode', 'ObjectiveNode',
    
    # 模型基类
    'BaseModel', 'ModelConfig',
    'SurrogateModel', 'InverseModel', 'GenerativeModel',
    
    # TNN
    'TandemNetwork', 'create_tnn_for_challenge',
    
    # MDN
    'MDN', 'MDNTandemNetwork', 'create_mdn_for_challenge',
    
    # CGAN
    'CGAN', 'WGAN_GP', 'create_cgan_for_challenge',
    
    # PINN
    'PhysicsInformedNet', 'MaxwellPINN', 'create_pinn_for_photonics',
    
    # HiLAB
    'HiLABEngine', 'create_hilab_for_challenge',
    
    # 训练工具
    'get_loss', 'get_metric', 'get_default_callbacks',
    
    # 数据
    'PhotonicsDataset', 'HDF5Dataset', 'SyntheticDataset',
    'create_dataloaders', 'save_dataset_to_hdf5',
    'DataAugmentation',
    
    # 挑战
    'DesignChallenge', 'DesignSpec', 'PerformanceTarget',
    'ChallengeFactory', 'register_challenge',
    'GratingCouplerChallenge', 'MetagratingChallenge', 'WavelengthDemuxChallenge',
    
    # 接口
    'SimulatorInterface', 'SimulatorFactory', 'register_simulator',
    'SimulationConfig', 'SourceConfig', 'MonitorConfig',
    'MEEP_AVAILABLE', 'OPTICS_AVAILABLE',
    
    # 优化
    'DispersionConstraint', 'ThermalConstraint', 'RobustnessConstraint',
    
    # 工作流
    'DesignPipeline', 'PipelineConfig', 'create_pipeline', 'run_quick_design',
    'TaskDispatcher', 'TaskStatus', 'submit_task', 'get_task_result',
    
    # 便捷函数
    'list_available_simulators',
    'list_available_challenges',
    'quick_start',
]
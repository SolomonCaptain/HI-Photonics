"""
HI-Photonics 模型模块

提供深度学习模型用于光子学逆向设计。
"""

from models.base import (
    BaseModel,
    ModelConfig,
    SurrogateModel,
    InverseModel,
    GenerativeModel
)

# Safetensor 工具
from models.safetensor_utils import (
    check_safetensors_available,
    convert_torch_to_safetensors,
    convert_safetensors_to_torch,
    load_safetensors_metadata,
    get_safetensors_info,
    batch_convert_to_safetensors,
    validate_safetensors_file
)

from models.inverse.tnn import (
    TandemNetwork,
    TandemNetworkConfig,
    ForwardNetwork,
    ForwardNetworkConfig,
    InverseNetwork,
    InverseNetworkConfig,
    create_tnn_for_challenge
)

from models.inverse.mdn import (
    MDN,
    MDNConfig,
    GaussianMixtureDistribution,
    GaussianMixtureParameters,
    MDNTandemNetwork,
    create_mdn_for_challenge
)

from models.inverse.cgan import (
    CGAN,
    CGANConfig,
    ConditionalGenerator,
    ConditionalDiscriminator,
    GeneratorConfig,
    DiscriminatorConfig,
    WGAN_GP,
    create_cgan_for_challenge
)

from models.inverse.pinn import (
    PhysicsInformedNet,
    SirenNet,
    MaxwellPINN,
    PhotonicsPINN,
    PINNSolver,
    PINNConfig,
    MaxwellConfig,
    PhysicsLossConfig,
    FourierFeatures,
    Sine,
    create_pinn_for_photonics
)

from models.training.losses import (
    BaseLoss,
    PerformanceLoss,
    DesignLoss,
    TandemLoss,
    PhysicsInformedLoss,
    ContrastiveLoss,
    MDNLoss,
    MDNRegularizedLoss,
    GANLoss,
    GradientPenaltyLoss,
    ConditionalConsistencyLoss,
    CGANCombinedLoss,
    PDEResidualLoss,
    HelmholtzLoss,
    MaxwellLoss,
    BoundaryConditionLoss,
    PINNCombinedLoss,
    get_loss
)

from models.training.metrics import (
    BaseMetric,
    MSE,
    MAE,
    R2Score,
    RMSE,
    MAPE,
    ThresholdAccuracy,
    DesignQualityMetric,
    InverseDesignMetric,
    MetricsCollection,
    PerformanceTracker,
    get_metric
)

from models.training.callbacks import (
    Callback,
    EarlyStopping,
    ModelCheckpoint,
    LearningRateScheduler,
    TrainingLogger,
    GradientClipping,
    ProgressBar,
    CallbackList,
    get_default_callbacks
)

# HiLAB 混合逆向设计框架
from models.inverse.hilab import (
    # 配置类
    VAEEncoderConfig,
    VAEDecoderConfig,
    VAEConfig,
    BayesianOptimizerConfig,
    HiLABConfig,
    # 核心类
    VAEEncoder,
    VAEDecoder,
    VAE,
    HiLABEngine,
    # 工厂函数
    create_vae_for_challenge,
    create_hilab_for_challenge
)

__all__ = [
    # Base classes
    'BaseModel',
    'ModelConfig',
    'SurrogateModel',
    'InverseModel',
    'GenerativeModel',
    
    # Safetensor utilities
    'check_safetensors_available',
    'convert_torch_to_safetensors',
    'convert_safetensors_to_torch',
    'load_safetensors_metadata',
    'get_safetensors_info',
    'batch_convert_to_safetensors',
    'validate_safetensors_file',
    
    # TNN
    'TandemNetwork',
    'TandemNetworkConfig',
    'ForwardNetwork',
    'ForwardNetworkConfig',
    'InverseNetwork',
    'InverseNetworkConfig',
    'create_tnn_for_challenge',
    
    # MDN
    'MDN',
    'MDNConfig',
    'GaussianMixtureDistribution',
    'GaussianMixtureParameters',
    'MDNTandemNetwork',
    'create_mdn_for_challenge',
    
    # CGAN
    'CGAN',
    'CGANConfig',
    'ConditionalGenerator',
    'ConditionalDiscriminator',
    'GeneratorConfig',
    'DiscriminatorConfig',
    'WGAN_GP',
    'create_cgan_for_challenge',
    
    # PINN
    'PhysicsInformedNet',
    'SirenNet',
    'MaxwellPINN',
    'PhotonicsPINN',
    'PINNSolver',
    'PINNConfig',
    'MaxwellConfig',
    'PhysicsLossConfig',
    'FourierFeatures',
    'Sine',
    'create_pinn_for_photonics',
    
    # Losses
    'BaseLoss',
    'PerformanceLoss',
    'DesignLoss',
    'TandemLoss',
    'PhysicsInformedLoss',
    'ContrastiveLoss',
    'MDNLoss',
    'MDNRegularizedLoss',
    'GANLoss',
    'GradientPenaltyLoss',
    'ConditionalConsistencyLoss',
    'CGANCombinedLoss',
    'PDEResidualLoss',
    'HelmholtzLoss',
    'MaxwellLoss',
    'BoundaryConditionLoss',
    'PINNCombinedLoss',
    'get_loss',
    
    # Metrics
    'BaseMetric',
    'MSE',
    'MAE',
    'R2Score',
    'RMSE',
    'MAPE',
    'ThresholdAccuracy',
    'DesignQualityMetric',
    'InverseDesignMetric',
    'MetricsCollection',
    'PerformanceTracker',
    'get_metric',
    
    # Callbacks
    'Callback',
    'EarlyStopping',
    'ModelCheckpoint',
    'LearningRateScheduler',
    'TrainingLogger',
    'GradientClipping',
    'ProgressBar',
    'CallbackList',
    'get_default_callbacks',
    
    # HiLAB
    'VAEEncoderConfig',
    'VAEDecoderConfig',
    'VAEConfig',
    'BayesianOptimizerConfig',
    'HiLABConfig',
    'VAEEncoder',
    'VAEDecoder',
    'VAE',
    'BayesianOptimizer',
    'HiLABEngine',
    'create_vae_for_challenge',
    'create_hilab_for_challenge'
]

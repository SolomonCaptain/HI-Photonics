"""
数据生成器模块

提供多种数据生成策略用于训练和测试逆向设计模型。
"""

from data.generators.base import (
    DataGenerator,
    GeneratorConfig,
    GenerationResult
)

from data.generators.random_sampling import (
    RandomSamplingGenerator,
    RandomSamplingConfig
)

from data.generators.active_learning import (
    ActiveLearningGenerator,
    ActiveLearningConfig,
    AcquisitionFunction
)

from data.generators.multi_fidelity import (
    MultiFidelityGenerator,
    MultiFidelityConfig,
    FidelityLevel
)

__all__ = [
    # 基类
    'DataGenerator',
    'GeneratorConfig',
    'GenerationResult',
    
    # 随机采样
    'RandomSamplingGenerator',
    'RandomSamplingConfig',
    
    # 主动学习
    'ActiveLearningGenerator',
    'ActiveLearningConfig',
    'AcquisitionFunction',
    
    # 多保真度
    'MultiFidelityGenerator',
    'MultiFidelityConfig',
    'FidelityLevel'
]

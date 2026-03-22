"""
数据预处理模块

提供数据归一化和增强功能。
"""

from data.preprocess.normalization import (
    Normalizer,
    NormalizationStats,
    MultiFieldNormalizer,
    BatchNormalizer,
    create_normalizer,
    normalize_data
)

from data.preprocess.augmentation import (
    AugmentationBase,
    AugmentationConfig,
    HorizontalFlip,
    VerticalFlip,
    Rotation,
    GaussianNoise,
    RandomCrop,
    Mixup,
    Cutout,
    RandomErasing,
    PhysicsAwareAugmentation,
    CompositeAugmentation,
    DataAugmenter,
    create_augmenter
)

__all__ = [
    # 归一化
    'Normalizer',
    'NormalizationStats',
    'MultiFieldNormalizer',
    'BatchNormalizer',
    'create_normalizer',
    'normalize_data',
    
    # 数据增强
    'AugmentationBase',
    'AugmentationConfig',
    'HorizontalFlip',
    'VerticalFlip',
    'Rotation',
    'GaussianNoise',
    'RandomCrop',
    'Mixup',
    'Cutout',
    'RandomErasing',
    'PhysicsAwareAugmentation',
    'CompositeAugmentation',
    'DataAugmenter',
    'create_augmenter'
]
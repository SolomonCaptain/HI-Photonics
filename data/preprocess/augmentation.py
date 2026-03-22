"""
数据增强模块

提供光子学设计数据增强方法，支持几何变换和物理约束增强。
"""

from typing import Optional, Tuple, List, Union, Callable, Dict, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
import torch


class AugmentationBase(ABC):
    """数据增强基类"""
    
    @abstractmethod
    def __call__(
        self,
        design: Union[np.ndarray, torch.Tensor],
        performance: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> Tuple[Union[np.ndarray, torch.Tensor], Optional[Union[np.ndarray, torch.Tensor]]]:
        """应用增强"""
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


@dataclass
class AugmentationConfig:
    """数据增强配置"""
    # 几何变换
    horizontal_flip: bool = False
    vertical_flip: bool = False
    rotation_90: bool = False
    rotation_180: bool = False
    rotation_270: bool = False
    
    # 噪声注入
    gaussian_noise: bool = False
    noise_std: float = 0.01
    
    # 随机裁剪/填充
    random_crop: bool = False
    crop_ratio: Tuple[float, float] = (0.9, 1.0)
    
    # 插值增强
    mixup: bool = False
    mixup_alpha: float = 0.2
    
    # 概率控制
    apply_probability: float = 0.5


class HorizontalFlip(AugmentationBase):
    """水平翻转"""
    
    def __init__(self, axis: int = -1):
        self.axis = axis
    
    def __call__(
        self,
        design: Union[np.ndarray, torch.Tensor],
        performance: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> Tuple[Union[np.ndarray, torch.Tensor], Optional[Union[np.ndarray, torch.Tensor]]]:
        if isinstance(design, torch.Tensor):
            design = torch.flip(design, dims=[self.axis])
        else:
            design = np.flip(design, axis=self.axis)
        return design, performance


class VerticalFlip(AugmentationBase):
    """垂直翻转"""
    
    def __init__(self, axis: int = -2):
        self.axis = axis
    
    def __call__(
        self,
        design: Union[np.ndarray, torch.Tensor],
        performance: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> Tuple[Union[np.ndarray, torch.Tensor], Optional[Union[np.ndarray, torch.Tensor]]]:
        if isinstance(design, torch.Tensor):
            design = torch.flip(design, dims=[self.axis])
        else:
            design = np.flip(design, axis=self.axis)
        return design, performance


class Rotation(AugmentationBase):
    """旋转增强"""
    
    def __init__(self, k: int = 1, axes: Tuple[int, int] = (-2, -1)):
        """
        Args:
            k: 旋转次数（每次 90 度）
            axes: 旋转轴
        """
        self.k = k
        self.axes = axes
    
    def __call__(
        self,
        design: Union[np.ndarray, torch.Tensor],
        performance: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> Tuple[Union[np.ndarray, torch.Tensor], Optional[Union[np.ndarray, torch.Tensor]]]:
        if isinstance(design, torch.Tensor):
            design = torch.rot90(design, k=self.k, dims=self.axes)
        else:
            design = np.rot90(design, k=self.k, axes=self.axes)
        return design, performance


class GaussianNoise(AugmentationBase):
    """高斯噪声注入"""
    
    def __init__(
        self,
        std: float = 0.01,
        mean: float = 0.0,
        clip: bool = True,
        clip_range: Tuple[float, float] = (0.0, 1.0)
    ):
        self.std = std
        self.mean = mean
        self.clip = clip
        self.clip_range = clip_range
    
    def __call__(
        self,
        design: Union[np.ndarray, torch.Tensor],
        performance: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> Tuple[Union[np.ndarray, torch.Tensor], Optional[Union[np.ndarray, torch.Tensor]]]:
        is_tensor = isinstance(design, torch.Tensor)
        
        if is_tensor:
            noise = torch.randn_like(design) * self.std + self.mean
            design = design + noise
            if self.clip:
                design = torch.clamp(design, self.clip_range[0], self.clip_range[1])
        else:
            noise = np.random.randn(*design.shape) * self.std + self.mean
            design = design + noise
            if self.clip:
                design = np.clip(design, self.clip_range[0], self.clip_range[1])
        
        return design, performance


class RandomCrop(AugmentationBase):
    """随机裁剪"""
    
    def __init__(
        self,
        crop_size: Optional[Tuple[int, ...]] = None,
        crop_ratio: Tuple[float, float] = (0.9, 1.0),
        pad_mode: str = "constant",
        pad_value: float = 0.0
    ):
        self.crop_size = crop_size
        self.crop_ratio = crop_ratio
        self.pad_mode = pad_mode
        self.pad_value = pad_value
    
    def __call__(
        self,
        design: Union[np.ndarray, torch.Tensor],
        performance: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> Tuple[Union[np.ndarray, torch.Tensor], Optional[Union[np.ndarray, torch.Tensor]]]:
        is_tensor = isinstance(design, torch.Tensor)
        
        if is_tensor:
            shape = design.shape
        else:
            shape = design.shape
        
        # 确定裁剪大小
        if self.crop_size is not None:
            crop_shape = self.crop_size
        else:
            ratio = np.random.uniform(*self.crop_ratio)
            crop_shape = tuple(int(s * ratio) for s in shape)
        
        # 生成随机偏移
        offsets = tuple(
            np.random.randint(0, max(1, s - c + 1))
            for s, c in zip(shape, crop_shape)
        )
        
        # 裁剪
        if is_tensor:
            # 使用切片
            slices = [slice(None)] * len(shape)
            for i, (o, c) in enumerate(zip(offsets, crop_shape)):
                slices[i] = slice(o, o + c)
            design = design[tuple(slices)]
            
            # 填充回原始大小
            pad_width = []
            for s, c, o in zip(shape, crop_shape, offsets):
                pad_before = o
                pad_after = s - o - c
                pad_width.extend([pad_before, pad_after])
            
            if any(p > 0 for p in pad_width):
                design = torch.nn.functional.pad(
                    design, pad_width[::-1],
                    mode=self.pad_mode,
                    value=self.pad_value
                )
        else:
            slices = tuple(slice(o, o + c) for o, c in zip(offsets, crop_shape))
            design = design[slices]
            
            # 填充
            pad_width = []
            for s, c, o in zip(shape, crop_shape, offsets):
                pad_before = o
                pad_after = s - o - c
                pad_width.append((pad_before, pad_after))
            
            design = np.pad(design, pad_width, mode=self.pad_mode, constant_values=self.pad_value)
        
        return design, performance


class Mixup(AugmentationBase):
    """Mixup 数据增强"""
    
    def __init__(
        self,
        alpha: float = 0.2,
        beta: float = None
    ):
        self.alpha = alpha
        self.beta = beta or alpha
    
    def __call__(
        self,
        design: Union[np.ndarray, torch.Tensor],
        performance: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> Tuple[Union[np.ndarray, torch.Tensor], Optional[Union[np.ndarray, torch.Tensor]]]:
        # Mixup 需要另一个样本，这里返回 lambda 值供外部使用
        # 实际使用时需要在 DataLoader 中实现
        return design, performance
    
    def get_lambda(self) -> float:
        """获取 mixup lambda 值"""
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.beta)
        else:
            lam = 1.0
        return lam
    
    def mix(
        self,
        design1: Union[np.ndarray, torch.Tensor],
        design2: Union[np.ndarray, torch.Tensor],
        performance1: Optional[Union[np.ndarray, torch.Tensor]] = None,
        performance2: Optional[Union[np.ndarray, torch.Tensor]] = None,
        lam: Optional[float] = None
    ) -> Tuple[Union[np.ndarray, torch.Tensor], Optional[Union[np.ndarray, torch.Tensor]]]:
        """
        混合两个样本
        
        Args:
            design1: 第一个设计
            design2: 第二个设计
            performance1: 第一个性能
            performance2: 第二个性能
            lam: 混合系数
            
        Returns:
            混合后的 (设计, 性能)
        """
        if lam is None:
            lam = self.get_lambda()
        
        is_tensor = isinstance(design1, torch.Tensor)
        
        if is_tensor:
            mixed_design = lam * design1 + (1 - lam) * design2
            if performance1 is not None and performance2 is not None:
                mixed_performance = lam * performance1 + (1 - lam) * performance2
            else:
                mixed_performance = None
        else:
            mixed_design = lam * design1 + (1 - lam) * design2
            if performance1 is not None and performance2 is not None:
                mixed_performance = lam * performance1 + (1 - lam) * performance2
            else:
                mixed_performance = None
        
        return mixed_design, mixed_performance


class Cutout(AugmentationBase):
    """Cutout 增强"""
    
    def __init__(
        self,
        num_holes: int = 1,
        hole_size: Union[int, Tuple[int, ...]] = 16,
        fill_value: float = 0.0
    ):
        self.num_holes = num_holes
        self.hole_size = hole_size if isinstance(hole_size, tuple) else (hole_size,) * 2
        self.fill_value = fill_value
    
    def __call__(
        self,
        design: Union[np.ndarray, torch.Tensor],
        performance: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> Tuple[Union[np.ndarray, torch.Tensor], Optional[Union[np.ndarray, torch.Tensor]]]:
        is_tensor = isinstance(design, torch.Tensor)
        design = design.clone() if is_tensor else design.copy()
        
        shape = design.shape[-2:] if len(design.shape) >= 2 else design.shape
        
        for _ in range(self.num_holes):
            # 随机位置
            y = np.random.randint(0, max(1, shape[-2] - self.hole_size[0]))
            x = np.random.randint(0, max(1, shape[-1] - self.hole_size[1]))
            
            # 应用 cutout
            if is_tensor:
                design[..., y:y + self.hole_size[0], x:x + self.hole_size[1]] = self.fill_value
            else:
                design[..., y:y + self.hole_size[0], x:x + self.hole_size[1]] = self.fill_value
        
        return design, performance


class RandomErasing(AugmentationBase):
    """随机擦除增强"""
    
    def __init__(
        self,
        probability: float = 0.5,
        scale_range: Tuple[float, float] = (0.02, 0.33),
        ratio_range: Tuple[float, float] = (0.3, 3.3),
        value: float = 0.0
    ):
        self.probability = probability
        self.scale_range = scale_range
        self.ratio_range = ratio_range
        self.value = value
    
    def __call__(
        self,
        design: Union[np.ndarray, torch.Tensor],
        performance: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> Tuple[Union[np.ndarray, torch.Tensor], Optional[Union[np.ndarray, torch.Tensor]]]:
        if np.random.rand() > self.probability:
            return design, performance
        
        is_tensor = isinstance(design, torch.Tensor)
        design = design.clone() if is_tensor else design.copy()
        
        shape = design.shape[-2:] if len(design.shape) >= 2 else design.shape
        area = shape[-2] * shape[-1]
        
        # 随机擦除区域
        for _ in range(10):  # 最多尝试 10 次
            target_area = np.random.uniform(*self.scale_range) * area
            aspect_ratio = np.random.uniform(*self.ratio_range)
            
            h = int(round(np.sqrt(target_area * aspect_ratio)))
            w = int(round(np.sqrt(target_area / aspect_ratio)))
            
            if w < shape[-1] and h < shape[-2]:
                y = np.random.randint(0, shape[-2] - h)
                x = np.random.randint(0, shape[-1] - w)
                
                if is_tensor:
                    design[..., y:y + h, x:x + w] = self.value
                else:
                    design[..., y:y + h, x:x + w] = self.value
                break
        
        return design, performance


class PhysicsAwareAugmentation(AugmentationBase):
    """
    物理感知增强
    
    根据光子学设计的物理特性进行增强，确保增强后的设计仍然有效。
    """
    
    def __init__(
        self,
        min_feature_size: float = 0.1,
        smooth_kernel_size: int = 3,
        constraint_func: Optional[Callable] = None
    ):
        """
        Args:
            min_feature_size: 最小特征尺寸
            smooth_kernel_size: 平滑核大小
            constraint_func: 物理约束函数
        """
        self.min_feature_size = min_feature_size
        self.smooth_kernel_size = smooth_kernel_size
        self.constraint_func = constraint_func
    
    def __call__(
        self,
        design: Union[np.ndarray, torch.Tensor],
        performance: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> Tuple[Union[np.ndarray, torch.Tensor], Optional[Union[np.ndarray, torch.Tensor]]]:
        is_tensor = isinstance(design, torch.Tensor)
        
        # 应用平滑
        if self.smooth_kernel_size > 1:
            design = self._apply_smooth(design, is_tensor)
        
        # 应用阈值化确保最小特征尺寸
        design = self._enforce_min_feature(design, is_tensor)
        
        # 检查约束
        if self.constraint_func is not None:
            if not self.constraint_func(design):
                # 如果约束不满足，返回原始设计
                return design, performance
        
        return design, performance
    
    def _apply_smooth(
        self,
        design: Union[np.ndarray, torch.Tensor],
        is_tensor: bool
    ) -> Union[np.ndarray, torch.Tensor]:
        """应用平滑滤波"""
        if is_tensor:
            import torch.nn.functional as F
            
            # 创建平滑核
            k = self.smooth_kernel_size
            kernel = torch.ones(1, 1, k, k) / (k * k)
            kernel = kernel.to(design.device)
            
            # 确保是 4D
            if design.ndim == 2:
                design = design.unsqueeze(0).unsqueeze(0)
            elif design.ndim == 3:
                design = design.unsqueeze(0)
            
            # 应用卷积
            smoothed = F.conv2d(design, kernel, padding=k // 2)
            
            # 恢复原始形状
            return smoothed.squeeze()
        else:
            from scipy.ndimage import uniform_filter
            
            smoothed = uniform_filter(design, size=self.smooth_kernel_size)
            return smoothed
    
    def _enforce_min_feature(
        self,
        design: Union[np.ndarray, torch.Tensor],
        is_tensor: bool
    ) -> Union[np.ndarray, torch.Tensor]:
        """强制最小特征尺寸"""
        # 简化实现：使用形态学操作
        # 实际应用中可能需要更复杂的算法
        return design


class CompositeAugmentation(AugmentationBase):
    """组合增强"""
    
    def __init__(
        self,
        augmentations: List[AugmentationBase],
        probabilities: Optional[List[float]] = None
    ):
        """
        Args:
            augmentations: 增强方法列表
            probabilities: 各增强方法的概率
        """
        self.augmentations = augmentations
        self.probabilities = probabilities or [0.5] * len(augmentations)
    
    def __call__(
        self,
        design: Union[np.ndarray, torch.Tensor],
        performance: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> Tuple[Union[np.ndarray, torch.Tensor], Optional[Union[np.ndarray, torch.Tensor]]]:
        for aug, prob in zip(self.augmentations, self.probabilities):
            if np.random.rand() < prob:
                design, performance = aug(design, performance)
        return design, performance
    
    def add_augmentation(
        self,
        augmentation: AugmentationBase,
        probability: float = 0.5
    ) -> 'CompositeAugmentation':
        """添加增强方法"""
        self.augmentations.append(augmentation)
        self.probabilities.append(probability)
        return self


class DataAugmenter:
    """
    数据增强器
    
    提供高级 API 进行数据增强。
    """
    
    def __init__(self, config: Optional[AugmentationConfig] = None):
        self.config = config or AugmentationConfig()
        self._build_augmentations()
    
    def _build_augmentations(self) -> None:
        """构建增强方法"""
        self.augmentations = []
        
        cfg = self.config
        
        if cfg.horizontal_flip:
            self.augmentations.append(HorizontalFlip())
        
        if cfg.vertical_flip:
            self.augmentations.append(VerticalFlip())
        
        if cfg.rotation_90:
            self.augmentations.append(Rotation(k=1))
        if cfg.rotation_180:
            self.augmentations.append(Rotation(k=2))
        if cfg.rotation_270:
            self.augmentations.append(Rotation(k=3))
        
        if cfg.gaussian_noise:
            self.augmentations.append(GaussianNoise(std=cfg.noise_std))
        
        if cfg.random_crop:
            self.augmentations.append(RandomCrop(crop_ratio=cfg.crop_ratio))
    
    def __call__(
        self,
        design: Union[np.ndarray, torch.Tensor],
        performance: Optional[Union[np.ndarray, torch.Tensor]] = None,
        apply_all: bool = False
    ) -> Tuple[Union[np.ndarray, torch.Tensor], Optional[Union[np.ndarray, torch.Tensor]]]:
        """
        应用数据增强
        
        Args:
            design: 设计参数
            performance: 性能指标
            apply_all: 是否应用所有增强（而非随机选择）
            
        Returns:
            增强后的 (设计, 性能)
        """
        if apply_all:
            for aug in self.augmentations:
                design, performance = aug(design, performance)
        else:
            for aug in self.augmentations:
                if np.random.rand() < self.config.apply_probability:
                    design, performance = aug(design, performance)
        
        return design, performance
    
    def augment_batch(
        self,
        designs: Union[np.ndarray, torch.Tensor],
        performances: Optional[Union[np.ndarray, torch.Tensor]] = None
    ) -> Tuple[Union[np.ndarray, torch.Tensor], Optional[Union[np.ndarray, torch.Tensor]]]:
        """批量增强"""
        augmented_designs = []
        augmented_performances = [] if performances is not None else None
        
        for i in range(len(designs)):
            design = designs[i]
            perf = performances[i] if performances is not None else None
            
            aug_design, aug_perf = self(design, perf)
            
            augmented_designs.append(aug_design)
            if augmented_performances is not None:
                augmented_performances.append(aug_perf)
        
        if isinstance(designs, torch.Tensor):
            augmented_designs = torch.stack(augmented_designs)
            if augmented_performances is not None:
                augmented_performances = torch.stack(augmented_performances)
        else:
            augmented_designs = np.stack(augmented_designs)
            if augmented_performances is not None:
                augmented_performances = np.stack(augmented_performances)
        
        return augmented_designs, augmented_performances


def create_augmenter(
    horizontal_flip: bool = True,
    vertical_flip: bool = False,
    gaussian_noise: bool = True,
    noise_std: float = 0.01,
    probability: float = 0.5
) -> DataAugmenter:
    """
    创建数据增强器的便捷函数
    
    Args:
        horizontal_flip: 水平翻转
        vertical_flip: 垂直翻转
        gaussian_noise: 高斯噪声
        noise_std: 噪声标准差
        probability: 应用概率
        
    Returns:
        数据增强器
    """
    config = AugmentationConfig(
        horizontal_flip=horizontal_flip,
        vertical_flip=vertical_flip,
        gaussian_noise=gaussian_noise,
        noise_std=noise_std,
        apply_probability=probability
    )
    return DataAugmenter(config)

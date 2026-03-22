"""
数据归一化模块

提供多种归一化和反归一化方法，支持保存和恢复归一化参数。
"""

from typing import Optional, Tuple, Union, Dict, Any, Literal
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import torch
import json


@dataclass
class NormalizationStats:
    """归一化统计信息"""
    mean: np.ndarray
    std: np.ndarray
    min_val: Optional[np.ndarray] = None
    max_val: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {
            'mean': self.mean.tolist(),
            'std': self.std.tolist()
        }
        if self.min_val is not None:
            result['min_val'] = self.min_val.tolist()
        if self.max_val is not None:
            result['max_val'] = self.max_val.tolist()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NormalizationStats':
        """从字典创建"""
        return cls(
            mean=np.array(data['mean']),
            std=np.array(data['std']),
            min_val=np.array(data['min_val']) if 'min_val' in data else None,
            max_val=np.array(data['max_val']) if 'max_val' in data else None
        )


class Normalizer:
    """
    数据归一化器基类
    
    支持多种归一化方法：
    - z-score: (x - mean) / std
    - min-max: (x - min) / (max - min)
    - robust: (x - median) / IQR
    - log: log(x + 1)
    """
    
    def __init__(
        self,
        method: Literal["zscore", "minmax", "robust", "log", "none"] = "zscore",
        eps: float = 1e-8,
        clip: bool = False,
        clip_range: Tuple[float, float] = (-5.0, 5.0)
    ):
        """
        Args:
            method: 归一化方法
            eps: 数值稳定性常数
            clip: 是否裁剪归一化后的值
            clip_range: 裁剪范围
        """
        self.method = method
        self.eps = eps
        self.clip = clip
        self.clip_range = clip_range
        
        self._stats: Optional[NormalizationStats] = None
        self._fitted = False
    
    def fit(self, data: Union[np.ndarray, torch.Tensor]) -> 'Normalizer':
        """
        计算归一化参数
        
        Args:
            data: 训练数据
            
        Returns:
            self
        """
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        
        if self.method == "none":
            self._stats = NormalizationStats(
                mean=np.zeros(data.shape[1:] if data.ndim > 1 else 1),
                std=np.ones(data.shape[1:] if data.ndim > 1 else 1)
            )
        elif self.method == "zscore":
            self._stats = NormalizationStats(
                mean=data.mean(axis=0),
                std=data.std(axis=0) + self.eps
            )
        elif self.method == "minmax":
            self._stats = NormalizationStats(
                mean=np.zeros(data.shape[1:] if data.ndim > 1 else 1),
                std=np.ones(data.shape[1:] if data.ndim > 1 else 1),
                min_val=data.min(axis=0),
                max_val=data.max(axis=0)
            )
        elif self.method == "robust":
            median = np.median(data, axis=0)
            q1 = np.percentile(data, 25, axis=0)
            q3 = np.percentile(data, 75, axis=0)
            iqr = q3 - q1 + self.eps
            
            self._stats = NormalizationStats(
                mean=median,
                std=iqr,
                min_val=q1,
                max_val=q3
            )
        elif self.method == "log":
            self._stats = NormalizationStats(
                mean=np.zeros(data.shape[1:] if data.ndim > 1 else 1),
                std=np.ones(data.shape[1:] if data.ndim > 1 else 1)
            )
        else:
            raise ValueError(f"未知的归一化方法: {self.method}")
        
        self._fitted = True
        return self
    
    def transform(
        self,
        data: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        应用归一化
        
        Args:
            data: 输入数据
            
        Returns:
            归一化后的数据
        """
        if not self._fitted:
            raise RuntimeError("归一化器尚未拟合，请先调用 fit()")
        
        is_tensor = isinstance(data, torch.Tensor)
        if is_tensor:
            device = data.device
            data = data.cpu().numpy()
        
        if self.method == "none":
            result = data
        elif self.method == "zscore":
            result = (data - self._stats.mean) / self._stats.std
        elif self.method == "minmax":
            range_val = self._stats.max_val - self._stats.min_val + self.eps
            result = (data - self._stats.min_val) / range_val
        elif self.method == "robust":
            result = (data - self._stats.mean) / self._stats.std
        elif self.method == "log":
            result = np.log(data + 1)
        else:
            raise ValueError(f"未知的归一化方法: {self.method}")
        
        if self.clip:
            result = np.clip(result, self.clip_range[0], self.clip_range[1])
        
        if is_tensor:
            result = torch.from_numpy(result).float().to(device)
        
        return result
    
    def inverse_transform(
        self,
        data: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        反归一化
        
        Args:
            data: 归一化后的数据
            
        Returns:
            原始尺度的数据
        """
        if not self._fitted:
            raise RuntimeError("归一化器尚未拟合，请先调用 fit()")
        
        is_tensor = isinstance(data, torch.Tensor)
        if is_tensor:
            device = data.device
            data = data.cpu().numpy()
        
        if self.method == "none":
            result = data
        elif self.method == "zscore":
            result = data * self._stats.std + self._stats.mean
        elif self.method == "minmax":
            range_val = self._stats.max_val - self._stats.min_val + self.eps
            result = data * range_val + self._stats.min_val
        elif self.method == "robust":
            result = data * self._stats.std + self._stats.mean
        elif self.method == "log":
            result = np.exp(data) - 1
        else:
            raise ValueError(f"未知的归一化方法: {self.method}")
        
        if is_tensor:
            result = torch.from_numpy(result).float().to(device)
        
        return result
    
    def fit_transform(
        self,
        data: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """拟合并转换"""
        return self.fit(data).transform(data)
    
    def get_stats(self) -> NormalizationStats:
        """获取归一化统计信息"""
        if not self._fitted:
            raise RuntimeError("归一化器尚未拟合")
        return self._stats
    
    def save(self, filepath: Union[str, Path]) -> None:
        """保存归一化参数"""
        if not self._fitted:
            raise RuntimeError("归一化器尚未拟合")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'method': self.method,
            'eps': self.eps,
            'clip': self.clip,
            'clip_range': self.clip_range,
            'stats': self._stats.to_dict()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'Normalizer':
        """加载归一化参数"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        normalizer = cls(
            method=data['method'],
            eps=data['eps'],
            clip=data['clip'],
            clip_range=tuple(data['clip_range'])
        )
        normalizer._stats = NormalizationStats.from_dict(data['stats'])
        normalizer._fitted = True
        
        return normalizer


class MultiFieldNormalizer:
    """
    多字段归一化器
    
    支持对多个字段分别使用不同的归一化方法。
    """
    
    def __init__(self):
        self._normalizers: Dict[str, Normalizer] = {}
        self._fitted = False
    
    def add_field(
        self,
        name: str,
        method: Literal["zscore", "minmax", "robust", "log", "none"] = "zscore",
        **kwargs
    ) -> 'MultiFieldNormalizer':
        """
        添加字段归一化器
        
        Args:
            name: 字段名
            method: 归一化方法
            **kwargs: 传递给 Normalizer 的参数
            
        Returns:
            self
        """
        self._normalizers[name] = Normalizer(method=method, **kwargs)
        return self
    
    def fit(
        self,
        data: Dict[str, Union[np.ndarray, torch.Tensor]]
    ) -> 'MultiFieldNormalizer':
        """
        拟合所有字段
        
        Args:
            data: 字段名到数据的映射
            
        Returns:
            self
        """
        for name, field_data in data.items():
            if name not in self._normalizers:
                self._normalizers[name] = Normalizer()
            self._normalizers[name].fit(field_data)
        
        self._fitted = True
        return self
    
    def transform(
        self,
        data: Dict[str, Union[np.ndarray, torch.Tensor]]
    ) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        """转换所有字段"""
        if not self._fitted:
            raise RuntimeError("归一化器尚未拟合")
        
        return {
            name: self._normalizers[name].transform(field_data)
            for name, field_data in data.items()
            if name in self._normalizers
        }
    
    def inverse_transform(
        self,
        data: Dict[str, Union[np.ndarray, torch.Tensor]]
    ) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        """反转换所有字段"""
        if not self._fitted:
            raise RuntimeError("归一化器尚未拟合")
        
        return {
            name: self._normalizers[name].inverse_transform(field_data)
            for name, field_data in data.items()
            if name in self._normalizers
        }
    
    def save(self, filepath: Union[str, Path]) -> None:
        """保存所有归一化参数"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'fields': {
                name: {
                    'method': norm.method,
                    'eps': norm.eps,
                    'clip': norm.clip,
                    'clip_range': norm.clip_range,
                    'stats': norm._stats.to_dict()
                }
                for name, norm in self._normalizers.items()
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'MultiFieldNormalizer':
        """加载归一化参数"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        normalizer = cls()
        
        for name, field_data in data['fields'].items():
            norm = Normalizer(
                method=field_data['method'],
                eps=field_data['eps'],
                clip=field_data['clip'],
                clip_range=tuple(field_data['clip_range'])
            )
            norm._stats = NormalizationStats.from_dict(field_data['stats'])
            norm._fitted = True
            normalizer._normalizers[name] = norm
        
        normalizer._fitted = True
        return normalizer


class BatchNormalizer:
    """
    批量归一化器
    
    支持增量更新归一化统计信息。
    """
    
    def __init__(
        self,
        method: Literal["zscore", "minmax"] = "zscore",
        eps: float = 1e-8,
        momentum: float = 0.1
    ):
        """
        Args:
            method: 归一化方法
            eps: 数值稳定性常数
            momentum: 动量系数（用于增量更新）
        """
        self.method = method
        self.eps = eps
        self.momentum = momentum
        
        self._running_mean: Optional[np.ndarray] = None
        self._running_var: Optional[np.ndarray] = None
        self._running_min: Optional[np.ndarray] = None
        self._running_max: Optional[np.ndarray] = None
        self._count = 0
        self._fitted = False
    
    def partial_fit(
        self,
        data: Union[np.ndarray, torch.Tensor]
    ) -> 'BatchNormalizer':
        """
        增量拟合
        
        Args:
            data: 新数据批次
            
        Returns:
            self
        """
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        
        batch_mean = data.mean(axis=0)
        batch_var = data.var(axis=0)
        batch_count = len(data)
        
        if self._running_mean is None:
            self._running_mean = batch_mean
            self._running_var = batch_var
            self._count = batch_count
        else:
            # 增量更新均值和方差
            delta = batch_mean - self._running_mean
            total_count = self._count + batch_count
            
            self._running_mean = (
                self._count * self._running_mean + batch_count * batch_mean
            ) / total_count
            
            # 增量方差更新（Welford's algorithm 简化版）
            self._running_var = (
                self._count * self._running_var + batch_count * batch_var +
                self._count * batch_count * delta ** 2 / total_count
            ) / total_count
            
            self._count = total_count
        
        if self.method == "minmax":
            batch_min = data.min(axis=0)
            batch_max = data.max(axis=0)
            
            if self._running_min is None:
                self._running_min = batch_min
                self._running_max = batch_max
            else:
                self._running_min = np.minimum(self._running_min, batch_min)
                self._running_max = np.maximum(self._running_max, batch_max)
        
        self._fitted = True
        return self
    
    def transform(
        self,
        data: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """应用归一化"""
        if not self._fitted:
            raise RuntimeError("归一化器尚未拟合")
        
        is_tensor = isinstance(data, torch.Tensor)
        if is_tensor:
            device = data.device
            data = data.cpu().numpy()
        
        if self.method == "zscore":
            result = (data - self._running_mean) / (np.sqrt(self._running_var) + self.eps)
        elif self.method == "minmax":
            range_val = self._running_max - self._running_min + self.eps
            result = (data - self._running_min) / range_val
        else:
            raise ValueError(f"未知的方法: {self.method}")
        
        if is_tensor:
            result = torch.from_numpy(result).float().to(device)
        
        return result
    
    def inverse_transform(
        self,
        data: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """反归一化"""
        if not self._fitted:
            raise RuntimeError("归一化器尚未拟合")
        
        is_tensor = isinstance(data, torch.Tensor)
        if is_tensor:
            device = data.device
            data = data.cpu().numpy()
        
        if self.method == "zscore":
            result = data * (np.sqrt(self._running_var) + self.eps) + self._running_mean
        elif self.method == "minmax":
            range_val = self._running_max - self._running_min + self.eps
            result = data * range_val + self._running_min
        else:
            raise ValueError(f"未知的方法: {self.method}")
        
        if is_tensor:
            result = torch.from_numpy(result).float().to(device)
        
        return result


def create_normalizer(
    method: Literal["zscore", "minmax", "robust", "log", "none"] = "zscore",
    **kwargs
) -> Normalizer:
    """
    创建归一化器的便捷函数
    
    Args:
        method: 归一化方法
        **kwargs: 传递给 Normalizer 的参数
        
    Returns:
        归一化器实例
    """
    return Normalizer(method=method, **kwargs)


def normalize_data(
    data: Union[np.ndarray, torch.Tensor],
    method: Literal["zscore", "minmax", "robust", "log"] = "zscore"
) -> Tuple[Union[np.ndarray, torch.Tensor], Normalizer]:
    """
    快速归一化数据
    
    Args:
        data: 输入数据
        method: 归一化方法
        
    Returns:
        (归一化后的数据, 归一化器)
    """
    normalizer = Normalizer(method=method)
    normalized = normalizer.fit_transform(data)
    return normalized, normalizer

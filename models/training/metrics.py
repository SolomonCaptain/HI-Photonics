"""
评估指标模块

提供光子学逆向设计的评估指标。
"""

from typing import Dict, List, Optional, Union, Any, Callable
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np


@dataclass
class MetricResult:
    """指标结果"""
    name: str
    value: float
    higher_is_better: bool = True
    details: Optional[Dict[str, Any]] = None


class BaseMetric(nn.Module):
    """指标基类"""
    
    def __init__(self, name: str, higher_is_better: bool = True):
        super().__init__()
        self.name = name
        self.higher_is_better = higher_is_better
    
    def forward(self, pred: Tensor, target: Tensor, **kwargs) -> MetricResult:
        """计算指标"""
        raise NotImplementedError


class MSE(BaseMetric):
    """均方误差"""
    
    def __init__(self):
        super().__init__('mse', higher_is_better=False)
    
    def forward(self, pred: Tensor, target: Tensor, **kwargs) -> MetricResult:
        value = F.mse_loss(pred, target).item()
        return MetricResult(self.name, value, self.higher_is_better)


class MAE(BaseMetric):
    """平均绝对误差"""
    
    def __init__(self):
        super().__init__('mae', higher_is_better=False)
    
    def forward(self, pred: Tensor, target: Tensor, **kwargs) -> MetricResult:
        value = F.l1_loss(pred, target).item()
        return MetricResult(self.name, value, self.higher_is_better)


class R2Score(BaseMetric):
    """决定系数 R²"""
    
    def __init__(self):
        super().__init__('r2', higher_is_better=True)
    
    def forward(self, pred: Tensor, target: Tensor, **kwargs) -> MetricResult:
        ss_res = ((target - pred) ** 2).sum()
        ss_tot = ((target - target.mean()) ** 2).sum()
        value = (1 - ss_res / (ss_tot + 1e-8)).item()
        return MetricResult(self.name, value, self.higher_is_better)


class RMSE(BaseMetric):
    """均方根误差"""
    
    def __init__(self):
        super().__init__('rmse', higher_is_better=False)
    
    def forward(self, pred: Tensor, target: Tensor, **kwargs) -> MetricResult:
        value = torch.sqrt(F.mse_loss(pred, target)).item()
        return MetricResult(self.name, value, self.higher_is_better)


class MAPE(BaseMetric):
    """平均绝对百分比误差"""
    
    def __init__(self, epsilon: float = 1e-8):
        super().__init__('mape', higher_is_better=False)
        self.epsilon = epsilon
    
    def forward(self, pred: Tensor, target: Tensor, **kwargs) -> MetricResult:
        value = (torch.abs(target - pred) / (torch.abs(target) + self.epsilon)).mean().item() * 100
        return MetricResult(self.name, value, self.higher_is_better)


class ThresholdAccuracy(BaseMetric):
    """阈值准确率"""
    
    def __init__(self, threshold: float = 0.05):
        super().__init__(f'accuracy@{threshold}', higher_is_better=True)
        self.threshold = threshold
    
    def forward(self, pred: Tensor, target: Tensor, **kwargs) -> MetricResult:
        correct = (torch.abs(pred - target) < self.threshold).float().mean().item()
        return MetricResult(self.name, correct, self.higher_is_better)


class DesignQualityMetric(BaseMetric):
    """
    设计质量指标
    
    评估生成设计的物理质量。
    """
    
    def __init__(self):
        super().__init__('design_quality', higher_is_better=True)
    
    def forward(self, design: Tensor, target: Optional[Tensor] = None, **kwargs) -> MetricResult:
        """
        计算设计质量
        
        Args:
            design: 设计参数 [B, H, W]
            target: 目标设计（可选）
            
        Returns:
            设计质量指标
        """
        details = {}
        
        # 二值化程度
        binary_score = 1 - 2 * torch.min(design, 1 - design).mean().item()
        details['binary_score'] = binary_score
        
        # 平滑度
        dx = torch.abs(design[:, :, 1:] - design[:, :, :-1]).mean().item()
        dy = torch.abs(design[:, 1:, :] - design[:, :-1, :]).mean().item()
        smoothness = 1 - (dx + dy) / 2
        details['smoothness'] = smoothness
        
        # 体积分数
        volume_fraction = design.mean().item()
        details['volume_fraction'] = volume_fraction
        
        # 综合质量分数
        quality = 0.4 * binary_score + 0.3 * smoothness + 0.3 * (1 - abs(volume_fraction - 0.5))
        
        return MetricResult(self.name, quality, self.higher_is_better, details)


class InverseDesignMetric(BaseMetric):
    """
    逆向设计指标
    
    评估逆向设计的成功率。
    """
    
    def __init__(self, tolerance: float = 0.05):
        super().__init__('inverse_success_rate', higher_is_better=True)
        self.tolerance = tolerance
    
    def forward(
        self,
        pred_performance: Tensor,
        target_performance: Tensor,
        design: Optional[Tensor] = None,
        **kwargs
    ) -> MetricResult:
        """
        计算逆向设计成功率
        
        Args:
            pred_performance: 预测性能
            target_performance: 目标性能
            design: 设计参数（可选，用于额外质量评估）
            
        Returns:
            成功率指标
        """
        # 性能匹配度
        perf_error = torch.abs(pred_performance - target_performance)
        success = (perf_error < self.tolerance).all(dim=1).float().mean().item()
        
        details = {
            'success_rate': success,
            'mean_error': perf_error.mean().item(),
            'max_error': perf_error.max().item()
        }
        
        # 如果提供了设计，添加设计质量评估
        if design is not None:
            design_quality = DesignQualityMetric()
            quality_result = design_quality(design)
            details['design_quality'] = quality_result.value
        
        return MetricResult(self.name, success, self.higher_is_better, details)


class MetricsCollection:
    """
    指标集合
    
    管理多个指标的批量计算。
    """
    
    def __init__(self, metrics: Optional[List[BaseMetric]] = None):
        self.metrics = metrics or [
            MSE(),
            MAE(),
            R2Score(),
            RMSE()
        ]
    
    def add_metric(self, metric: BaseMetric):
        """添加指标"""
        self.metrics.append(metric)
    
    def compute(
        self,
        pred: Tensor,
        target: Tensor,
        **kwargs
    ) -> Dict[str, MetricResult]:
        """
        计算所有指标
        
        Args:
            pred: 预测值
            target: 目标值
            
        Returns:
            指标结果字典
        """
        results = {}
        for metric in self.metrics:
            result = metric(pred, target, **kwargs)
            results[metric.name] = result
        
        return results
    
    def compute_summary(
        self,
        pred: Tensor,
        target: Tensor,
        **kwargs
    ) -> Dict[str, float]:
        """
        计算指标摘要
        
        Returns:
            指标名到值的映射
        """
        results = self.compute(pred, target, **kwargs)
        return {name: result.value for name, result in results.items()}


class PerformanceTracker:
    """
    性能追踪器
    
    追踪训练过程中的指标变化。
    """
    
    def __init__(self):
        self.history: Dict[str, List[float]] = {}
        self.best: Dict[str, float] = {}
        self.best_epoch: Dict[str, int] = {}
    
    def update(
        self,
        metrics: Dict[str, MetricResult],
        epoch: int
    ):
        """
        更新追踪器
        
        Args:
            metrics: 指标结果字典
            epoch: 当前轮次
        """
        for name, result in metrics.items():
            # 更新历史
            if name not in self.history:
                self.history[name] = []
            self.history[name].append(result.value)
            
            # 更新最佳值
            if name not in self.best:
                self.best[name] = result.value
                self.best_epoch[name] = epoch
            else:
                is_better = (
                    (result.higher_is_better and result.value > self.best[name]) or
                    (not result.higher_is_better and result.value < self.best[name])
                )
                if is_better:
                    self.best[name] = result.value
                    self.best_epoch[name] = epoch
    
    def get_best(self, name: str) -> tuple:
        """获取最佳值"""
        return self.best.get(name, float('inf')), self.best_epoch.get(name, 0)
    
    def get_history(self, name: str) -> List[float]:
        """获取历史记录"""
        return self.history.get(name, [])
    
    def is_improving(self, name: str, current_value: float) -> bool:
        """检查是否在改进"""
        if name not in self.best:
            return True
        
        # 假设大多数指标越小越好
        return current_value < self.best[name]


# 指标工厂
METRIC_REGISTRY = {
    'mse': MSE,
    'mae': MAE,
    'r2': R2Score,
    'rmse': RMSE,
    'mape': MAPE,
    'accuracy': ThresholdAccuracy,
    'design_quality': DesignQualityMetric,
    'inverse_success': InverseDesignMetric
}


def get_metric(name: str, **kwargs) -> BaseMetric:
    """获取指标"""
    if name not in METRIC_REGISTRY:
        raise ValueError(f"Unknown metric: {name}")
    return METRIC_REGISTRY[name](**kwargs)


def get_default_metrics() -> MetricsCollection:
    """获取默认指标集合"""
    return MetricsCollection()

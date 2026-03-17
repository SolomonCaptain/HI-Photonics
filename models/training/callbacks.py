"""
训练回调模块

提供训练过程中的回调功能，如早停、模型保存、学习率调度等。
"""

from typing import Dict, Optional, Any, List, Callable
from pathlib import Path
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from torch.optim import Optimizer


class Callback(ABC):
    """回调基类"""
    
    def on_train_begin(self, **kwargs):
        """训练开始时调用"""
        pass
    
    def on_train_end(self, **kwargs):
        """训练结束时调用"""
        pass
    
    def on_epoch_begin(self, epoch: int, **kwargs):
        """每个轮次开始时调用"""
        pass
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, float], **kwargs):
        """每个轮次结束时调用"""
        pass
    
    def on_batch_begin(self, batch: int, **kwargs):
        """每个批次开始时调用"""
        pass
    
    def on_batch_end(self, batch: int, logs: Dict[str, float], **kwargs):
        """每个批次结束时调用"""
        pass


class EarlyStopping(Callback):
    """
    早停回调
    
    当监控指标在指定轮次内没有改进时停止训练。
    """
    
    def __init__(
        self,
        monitor: str = 'val_loss',
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'min',
        restore_best_weights: bool = True
    ):
        """
        Args:
            monitor: 监控的指标名称
            patience: 耐心值（允许无改进的轮次数）
            min_delta: 最小改进阈值
            mode: 'min' 或 'max'，指标应该减小还是增大
            restore_best_weights: 是否恢复最佳权重
        """
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        self.wait = 0
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.best_weights = None
        self.best_epoch = 0
        self.stopped_epoch = 0
    
    def on_train_begin(self, model: nn.Module, **kwargs):
        """初始化"""
        self.wait = 0
        self.best_value = float('inf') if self.mode == 'min' else float('-inf')
        self.best_weights = None
        self.stopped_epoch = 0
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, float], model: nn.Module, **kwargs):
        """检查是否应该停止"""
        current = logs.get(self.monitor)
        if current is None:
            return
        
        if self.mode == 'min':
            improved = current < self.best_value - self.min_delta
        else:
            improved = current > self.best_value + self.min_delta
        
        if improved:
            self.best_value = current
            self.wait = 0
            self.best_epoch = epoch
            if self.restore_best_weights:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                return True  # 停止训练
        
        return False
    
    def on_train_end(self, **kwargs):
        """打印早停信息"""
        if self.stopped_epoch > 0:
            print(f"\nEarly stopping at epoch {self.stopped_epoch + 1}")
            print(f"Best {self.monitor}: {self.best_value:.4f} at epoch {self.best_epoch + 1}")


class ModelCheckpoint(Callback):
    """
    模型检查点回调
    
    定期保存模型。
    """
    
    def __init__(
        self,
        filepath: str,
        monitor: str = 'val_loss',
        save_best_only: bool = True,
        mode: str = 'min',
        save_every: int = 0  # 0 表示不定期保存
    ):
        """
        Args:
            filepath: 保存路径模板，支持 {epoch}, {val_loss} 等占位符
            monitor: 监控的指标
            save_best_only: 是否只保存最佳模型
            mode: 'min' 或 'max'
            save_every: 每隔多少轮保存一次（0 表示不定期保存）
        """
        self.filepath = Path(filepath)
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.mode = mode
        self.save_every = save_every
        
        self.best_value = float('inf') if mode == 'min' else float('-inf')
    
    def on_epoch_end(
        self,
        epoch: int,
        logs: Dict[str, float],
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        **kwargs
    ):
        """保存模型"""
        current = logs.get(self.monitor)
        
        # 检查是否应该保存
        should_save = False
        
        if self.save_best_only and current is not None:
            if self.mode == 'min':
                if current < self.best_value:
                    self.best_value = current
                    should_save = True
            else:
                if current > self.best_value:
                    self.best_value = current
                    should_save = True
        elif self.save_every > 0 and (epoch + 1) % self.save_every == 0:
            should_save = True
        
        if should_save:
            # 生成文件名
            filename = str(self.filepath).format(
                epoch=epoch + 1,
                **{k: f"{v:.4f}" for k, v in logs.items()}
            )
            
            # 保存模型
            self.filepath.parent.mkdir(parents=True, exist_ok=True)
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'logs': logs
            }
            if optimizer is not None:
                checkpoint['optimizer_state_dict'] = optimizer.state_dict()
            
            torch.save(checkpoint, filename)
            print(f"Model saved to {filename}")


class LearningRateScheduler(Callback):
    """
    学习率调度回调
    
    支持多种学习率调整策略。
    """
    
    def __init__(
        self,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        monitor: Optional[str] = None
    ):
        """
        Args:
            scheduler: PyTorch 学习率调度器
            monitor: 监控的指标（用于 ReduceLROnPlateau）
        """
        self.scheduler = scheduler
        self.monitor = monitor
    
    def on_epoch_end(self, logs: Dict[str, float], **kwargs):
        """更新学习率"""
        if self.monitor is not None:
            # ReduceLROnPlateau 需要传入监控指标
            metric = logs.get(self.monitor)
            if metric is not None:
                self.scheduler.step(metric)
        else:
            self.scheduler.step()
    
    def get_lr(self) -> float:
        """获取当前学习率"""
        return self.scheduler.get_last_lr()[0]


class TrainingLogger(Callback):
    """
    训练日志回调
    
    记录训练过程中的指标和输出。
    """
    
    def __init__(
        self,
        log_file: Optional[str] = None,
        log_every: int = 1
    ):
        """
        Args:
            log_file: 日志文件路径（可选）
            log_every: 每隔多少轮输出一次日志
        """
        self.log_file = log_file
        self.log_every = log_every
        self.history: Dict[str, List[float]] = {}
    
    def on_train_begin(self, **kwargs):
        """初始化历史记录"""
        self.history = {}
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, float], **kwargs):
        """记录日志"""
        # 更新历史
        for key, value in logs.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(value)
        
        # 输出日志
        if (epoch + 1) % self.log_every == 0:
            log_str = f"Epoch {epoch + 1}: " + " | ".join([f"{k}={v:.4f}" for k, v in logs.items()])
            print(log_str)
            
            if self.log_file:
                with open(self.log_file, 'a') as f:
                    f.write(log_str + '\n')


class GradientClipping(Callback):
    """
    梯度裁剪回调
    
    防止梯度爆炸。
    """
    
    def __init__(
        self,
        max_norm: float = 1.0,
        norm_type: float = 2.0
    ):
        """
        Args:
            max_norm: 最大梯度范数
            norm_type: 范数类型
        """
        self.max_norm = max_norm
        self.norm_type = norm_type
    
    def on_batch_end(self, model: nn.Module, **kwargs):
        """裁剪梯度"""
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            self.max_norm,
            self.norm_type
        )


class ProgressBar(Callback):
    """
    进度条回调
    
    显示训练进度。
    """
    
    def __init__(self, verbose: int = 1):
        """
        Args:
            verbose: 详细程度（0: 无输出, 1: 进度条）
        """
        self.verbose = verbose
    
    def on_epoch_begin(self, epoch: int, total_epochs: int = 0, **kwargs):
        """显示轮次信息"""
        if self.verbose > 0:
            print(f"\nEpoch {epoch + 1}/{total_epochs}")
    
    def on_batch_end(self, batch: int, total_batches: int = 0, logs: Dict[str, float] = None, **kwargs):
        """显示批次进度"""
        if self.verbose > 0 and total_batches > 0:
            progress = (batch + 1) / total_batches
            bar_length = 30
            filled = int(bar_length * progress)
            bar = '=' * filled + '-' * (bar_length - filled)
            
            log_str = f"[{bar}] {progress:.0%}"
            if logs:
                log_str += " - " + " | ".join([f"{k}: {v:.4f}" for k, v in logs.items()])
            
            print(f"\r{log_str}", end='', flush=True)


class CallbackList:
    """
    回调列表
    
    管理多个回调的统一调用。
    """
    
    def __init__(self, callbacks: Optional[List[Callback]] = None):
        self.callbacks = callbacks or []
    
    def append(self, callback: Callback):
        """添加回调"""
        self.callbacks.append(callback)
    
    def on_train_begin(self, **kwargs):
        """调用所有回调的训练开始方法"""
        for callback in self.callbacks:
            callback.on_train_begin(**kwargs)
    
    def on_train_end(self, **kwargs):
        """调用所有回调的训练结束方法"""
        for callback in self.callbacks:
            callback.on_train_end(**kwargs)
    
    def on_epoch_begin(self, epoch: int, **kwargs):
        """调用所有回调的轮次开始方法"""
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, **kwargs)
    
    def on_epoch_end(self, epoch: int, logs: Dict[str, float], **kwargs) -> bool:
        """调用所有回调的轮次结束方法"""
        should_stop = False
        for callback in self.callbacks:
            result = callback.on_epoch_end(epoch, logs, **kwargs)
            if result is True:
                should_stop = True
        return should_stop
    
    def on_batch_begin(self, batch: int, **kwargs):
        """调用所有回调的批次开始方法"""
        for callback in self.callbacks:
            callback.on_batch_begin(batch, **kwargs)
    
    def on_batch_end(self, batch: int, logs: Dict[str, float], **kwargs):
        """调用所有回调的批次结束方法"""
        for callback in self.callbacks:
            callback.on_batch_end(batch, logs, **kwargs)


def get_default_callbacks(
    checkpoint_path: Optional[str] = None,
    patience: int = 10,
    monitor: str = 'val_loss'
) -> CallbackList:
    """
    获取默认回调列表
    
    Args:
        checkpoint_path: 检查点保存路径
        patience: 早停耐心值
        monitor: 监控指标
        
    Returns:
        回调列表
    """
    callbacks = [
        TrainingLogger(),
        EarlyStopping(monitor=monitor, patience=patience)
    ]
    
    if checkpoint_path:
        callbacks.append(ModelCheckpoint(checkpoint_path, monitor=monitor))
    
    return CallbackList(callbacks)

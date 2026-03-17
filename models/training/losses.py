"""
损失函数模块

提供光子学逆向设计专用的损失函数。
"""

from typing import Dict, Optional, List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class BaseLoss(nn.Module):
    """损失函数基类"""
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
    
    def _reduce(self, loss: Tensor) -> Tensor:
        """应用归约"""
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")


class PerformanceLoss(BaseLoss):
    """
    性能预测损失
    
    用于前向网络训练，计算预测性能与真实性能的差异。
    """
    
    def __init__(
        self,
        loss_type: str = 'mse',
        weights: Optional[Dict[str, float]] = None,
        reduction: str = 'mean'
    ):
        super().__init__(reduction)
        self.loss_type = loss_type
        self.weights = weights or {}
        
        self.loss_fn = {
            'mse': nn.MSELoss(reduction='none'),
            'mae': nn.L1Loss(reduction='none'),
            'huber': nn.HuberLoss(reduction='none'),
            'smooth_l1': nn.SmoothL1Loss(reduction='none')
        }[loss_type]
    
    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        feature_names: Optional[List[str]] = None
    ) -> Tensor:
        """
        计算性能损失
        
        Args:
            pred: 预测性能 [B, D]
            target: 目标性能 [B, D]
            feature_names: 性能指标名称列表（用于加权）
            
        Returns:
            损失值
        """
        loss = self.loss_fn(pred, target)
        
        # 应用特征权重
        if self.weights and feature_names:
            weight_tensor = torch.ones(1, loss.size(1), device=loss.device)
            for i, name in enumerate(feature_names):
                if name in self.weights:
                    weight_tensor[0, i] = self.weights[name]
            loss = loss * weight_tensor
        
        return self._reduce(loss)


class DesignLoss(BaseLoss):
    """
    设计参数损失
    
    用于逆向网络训练，计算生成设计与参考设计的差异。
    """
    
    def __init__(
        self,
        loss_type: str = 'bce',
        reduction: str = 'mean'
    ):
        super().__init__(reduction)
        self.loss_type = loss_type
        
        self.loss_fn = {
            'mse': nn.MSELoss(reduction='none'),
            'bce': nn.BCELoss(reduction='none'),
            'mae': nn.L1Loss(reduction='none')
        }[loss_type]
    
    def forward(self, design: Tensor, target: Optional[Tensor] = None) -> Tensor:
        """
        计算设计损失
        
        Args:
            design: 设计参数 [B, H, W] 或 [B, C, H, W]
            target: 目标设计（可选）
            
        Returns:
            损失值
        """
        if target is None:
            # 无目标时，使用正则化
            return self._regularization_loss(design)
        
        return self._reduce(self.loss_fn(design, target))
    
    def _regularization_loss(self, design: Tensor) -> Tensor:
        """设计正则化损失"""
        # 鼓励二值化
        binary_loss = torch.min(
            torch.abs(design),
            torch.abs(1 - design)
        ).mean()
        
        return binary_loss


class TandemLoss(nn.Module):
    """
    串联网络损失
    
    组合多种损失用于 TNN 训练。
    """
    
    def __init__(
        self,
        performance_weight: float = 1.0,
        design_weight: float = 0.0,
        regularization_weight: float = 0.0,
        diversity_weight: float = 0.0
    ):
        super().__init__()
        self.performance_weight = performance_weight
        self.design_weight = design_weight
        self.regularization_weight = regularization_weight
        self.diversity_weight = diversity_weight
        
        self.perf_loss = PerformanceLoss()
        self.design_loss = DesignLoss()
    
    def forward(
        self,
        pred_performance: Tensor,
        target_performance: Tensor,
        design: Tensor,
        design_gt: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        """
        计算串联损失
        
        Args:
            pred_performance: 预测性能
            target_performance: 目标性能
            design: 生成的设计
            design_gt: 真实设计（可选）
            
        Returns:
            损失字典
        """
        losses = {}
        
        # 性能重建损失
        losses['performance'] = self.perf_loss(pred_performance, target_performance)
        
        total_loss = self.performance_weight * losses['performance']
        
        # 设计损失
        if self.design_weight > 0 and design_gt is not None:
            losses['design'] = self.design_loss(design, design_gt)
            total_loss += self.design_weight * losses['design']
        
        # 正则化损失
        if self.regularization_weight > 0:
            losses['regularization'] = self._compute_regularization(design)
            total_loss += self.regularization_weight * losses['regularization']
        
        # 多样性损失
        if self.diversity_weight > 0:
            losses['diversity'] = self._compute_diversity(design)
            total_loss += self.diversity_weight * losses['diversity']
        
        losses['total'] = total_loss
        
        return losses
    
    def _compute_regularization(self, design: Tensor) -> Tensor:
        """计算正则化损失"""
        # 平滑度正则化
        dx = torch.abs(design[:, :, 1:] - design[:, :, :-1])
        dy = torch.abs(design[:, 1:, :] - design[:, :-1, :])
        smoothness = dx.mean() + dy.mean()
        
        # 二值化正则化
        binary = torch.min(torch.abs(design), torch.abs(1 - design)).mean()
        
        return smoothness + 0.1 * binary
    
    def _compute_diversity(self, design: Tensor) -> Tensor:
        """计算多样性损失（负值，鼓励多样性）"""
        design_flat = design.view(design.size(0), -1)
        # 计算样本间距离
        dist = torch.pdist(design_flat)
        # 返回负距离，最小化时增加多样性
        return -dist.mean()


class PhysicsInformedLoss(BaseLoss):
    """
    物理信息损失
    
    将物理约束融入损失函数。
    """
    
    def __init__(
        self,
        physics_weight: float = 0.1,
        reduction: str = 'mean'
    ):
        super().__init__(reduction)
        self.physics_weight = physics_weight
    
    def forward(
        self,
        design: Tensor,
        physics_constraints: Optional[Dict[str, Tensor]] = None
    ) -> Tensor:
        """
        计算物理约束损失
        
        Args:
            design: 设计参数
            physics_constraints: 物理约束字典
            
        Returns:
            物理约束损失
        """
        if physics_constraints is None:
            return torch.tensor(0.0, device=design.device)
        
        loss = torch.tensor(0.0, device=design.device)
        
        # 体积约束
        if 'volume_fraction' in physics_constraints:
            target_vol = physics_constraints['volume_fraction']
            actual_vol = design.mean()
            loss += F.mse_loss(actual_vol, target_vol)
        
        # 对称性约束
        if 'symmetry' in physics_constraints:
            sym_type = physics_constraints['symmetry']
            if sym_type == 'horizontal':
                loss += F.mse_loss(design, torch.flip(design, dims=[-1]))
            elif sym_type == 'vertical':
                loss += F.mse_loss(design, torch.flip(design, dims=[-2]))
        
        # 连续性约束
        if 'continuity' in physics_constraints:
            grad_x = torch.abs(design[:, :, 1:] - design[:, :, :-1])
            grad_y = torch.abs(design[:, 1:, :] - design[:, :-1, :])
            loss += (grad_x.mean() + grad_y.mean())
        
        return self.physics_weight * loss


class ContrastiveLoss(nn.Module):
    """
    对比损失
    
    用于学习设计空间的语义结构。
    """
    
    def __init__(
        self,
        margin: float = 1.0,
        distance: str = 'cosine'
    ):
        super().__init__()
        self.margin = margin
        self.distance = distance
    
    def forward(
        self,
        anchor: Tensor,
        positive: Tensor,
        negative: Tensor
    ) -> Tensor:
        """
        计算对比损失
        
        Args:
            anchor: 锚点特征
            positive: 正样本特征
            negative: 负样本特征
            
        Returns:
            对比损失
        """
        if self.distance == 'cosine':
            pos_dist = 1 - F.cosine_similarity(anchor, positive)
            neg_dist = 1 - F.cosine_similarity(anchor, negative)
        else:  # euclidean
            pos_dist = F.pairwise_distance(anchor, positive)
            neg_dist = F.pairwise_distance(anchor, negative)
        
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()


# 损失函数工厂
LOSS_REGISTRY = {
    'mse': nn.MSELoss,
    'mae': nn.L1Loss,
    'bce': nn.BCELoss,
    'huber': nn.HuberLoss,
    'performance': PerformanceLoss,
    'design': DesignLoss,
    'tandem': TandemLoss,
    'physics': PhysicsInformedLoss,
    'contrastive': ContrastiveLoss
}


def get_loss(name: str, **kwargs) -> nn.Module:
    """获取损失函数"""
    if name not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss function: {name}")
    return LOSS_REGISTRY[name](**kwargs)

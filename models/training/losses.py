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


class MDNLoss(BaseLoss):
    """
    混合密度网络损失函数
    
    计算负对数似然损失，用于 MDN 训练。
    
    数学公式:
        L = -log p(d|c) = -log sum_k π_k * N(d; μ_k, σ_k²)
    
    使用 log-sum-exp 技巧确保数值稳定性。
    """
    
    def __init__(
        self,
        epsilon: float = 1e-8,
        reduction: str = 'mean'
    ):
        """
        Args:
            epsilon: 数值稳定性常数
            reduction: 归约方式
        """
        super().__init__(reduction)
        self.epsilon = epsilon
    
    def forward(
        self,
        pi: Tensor,
        mu: Tensor,
        sigma: Tensor,
        target: Tensor
    ) -> Tensor:
        """
        计算负对数似然损失
        
        Args:
            pi: 混合权重 [B, K]
            mu: 均值 [B, K, D]
            sigma: 标准差 [B, K, D]
            target: 目标设计 [B, D]
            
        Returns:
            负对数似然损失
        """
        import math
        
        # 确保目标形状正确
        if target.dim() == 2:
            target = target.unsqueeze(1).expand(-1, pi.size(1), -1)  # [B, K, D]
        
        # 计算每个高斯分量的对数概率
        # log N(x; μ, σ²) = -0.5 * log(2π) - log(σ) - 0.5 * ((x-μ)/σ)²
        log_2pi = math.log(2 * math.pi)
        
        log_prob_components = (
            -0.5 * log_2pi 
            - torch.log(sigma + self.epsilon) 
            - 0.5 * ((target - mu) / (sigma + self.epsilon)) ** 2
        )  # [B, K, D]
        
        # 对设计维度求和
        log_prob_components = log_prob_components.sum(dim=-1)  # [B, K]
        
        # 使用 log-sum-exp 计算加权对数概率
        log_pi = torch.log(pi + self.epsilon)
        log_prob = torch.logsumexp(log_pi + log_prob_components, dim=-1)  # [B]
        
        # 负对数似然
        nll = -log_prob
        
        return self._reduce(nll)


class MDNRegularizedLoss(BaseLoss):
    """
    带正则化的 MDN 损失函数
    
    在负对数似然基础上添加:
    1. 分量平衡损失：鼓励各分量被均匀使用
    2. 熵正则化：控制分布的不确定性
    """
    
    def __init__(
        self,
        balance_weight: float = 0.01,
        entropy_weight: float = 0.001,
        epsilon: float = 1e-8,
        reduction: str = 'mean'
    ):
        """
        Args:
            balance_weight: 分量平衡权重
            entropy_weight: 熵正则化权重
            epsilon: 数值稳定性常数
            reduction: 归约方式
        """
        super().__init__(reduction)
        self.mdn_loss = MDNLoss(epsilon, 'none')
        self.balance_weight = balance_weight
        self.entropy_weight = entropy_weight
    
    def forward(
        self,
        pi: Tensor,
        mu: Tensor,
        sigma: Tensor,
        target: Tensor
    ) -> Tensor:
        """
        计算带正则化的损失
        
        Args:
            pi: 混合权重 [B, K]
            mu: 均值 [B, K, D]
            sigma: 标准差 [B, K, D]
            target: 目标设计 [B, D]
            
        Returns:
            总损失
        """
        # 基础 NLL 损失
        nll_loss = self.mdn_loss(pi, mu, sigma, target)
        
        # 分量平衡损失：鼓励各分量平均使用
        # 使用平均权重分布作为目标
        avg_pi = pi.mean(dim=0)  # [K]
        target_pi = torch.ones_like(avg_pi) / pi.size(1)
        balance_loss = F.kl_div(
            torch.log(avg_pi + 1e-8),
            target_pi,
            reduction='sum'
        )
        
        # 熵正则化：防止分布过于尖锐
        entropy = -torch.sum(pi * torch.log(pi + 1e-8), dim=-1).mean()
        entropy_loss = -entropy  # 最大化熵
        
        # 总损失
        total_loss = (
            nll_loss 
            + self.balance_weight * balance_loss 
            + self.entropy_weight * entropy_loss
        )
        
        return self._reduce(total_loss)


class GANLoss(BaseLoss):
    """
    生成对抗网络损失函数
    
    支持多种 GAN 变体:
    - 标准 GAN (BCE 损失)
    - LSGAN (最小二乘)
    - WGAN (Wasserstein)
    - Hinge 损失
    """
    
    def __init__(
        self,
        gan_type: str = 'gan',
        reduction: str = 'mean'
    ):
        """
        Args:
            gan_type: GAN 类型 ('gan', 'lsgan', 'wgan', 'hinge')
            reduction: 归约方式
        """
        super().__init__(reduction)
        self.gan_type = gan_type.lower()
    
    def discriminator_loss(
        self,
        real_validity: Tensor,
        fake_validity: Tensor
    ) -> Tensor:
        """
        计算判别器损失
        
        Args:
            real_validity: 真实样本的判别分数 [B, 1]
            fake_validity: 生成样本的判别分数 [B, 1]
            
        Returns:
            判别器损失
        """
        if self.gan_type == 'gan':
            # 标准 GAN: -log(D(x)) - log(1 - D(G(z)))
            real_loss = F.binary_cross_entropy_with_logits(
                real_validity, torch.ones_like(real_validity), reduction='none'
            )
            fake_loss = F.binary_cross_entropy_with_logits(
                fake_validity, torch.zeros_like(fake_validity), reduction='none'
            )
            loss = real_loss + fake_loss
            
        elif self.gan_type == 'lsgan':
            # LSGAN: (D(x) - 1)^2 + D(G(z))^2
            real_loss = F.mse_loss(
                real_validity, torch.ones_like(real_validity), reduction='none'
            )
            fake_loss = F.mse_loss(
                fake_validity, torch.zeros_like(fake_validity), reduction='none'
            )
            loss = real_loss + fake_loss
            
        elif self.gan_type == 'wgan':
            # WGAN: -D(x) + D(G(z))
            loss = -real_validity + fake_validity
            
        elif self.gan_type == 'hinge':
            # Hinge: max(0, 1 - D(x)) + max(0, 1 + D(G(z)))
            real_loss = F.relu(1.0 - real_validity)
            fake_loss = F.relu(1.0 + fake_validity)
            loss = real_loss + fake_loss
        else:
            raise ValueError(f"Unknown GAN type: {self.gan_type}")
        
        return self._reduce(loss)
    
    def generator_loss(
        self,
        fake_validity: Tensor,
        target_is_real: bool = True
    ) -> Tensor:
        """
        计算生成器损失
        
        Args:
            fake_validity: 生成样本的判别分数 [B, 1]
            target_is_real: 目标是真实样本（默认 True）
            
        Returns:
            生成器损失
        """
        if self.gan_type == 'gan':
            target = torch.ones_like(fake_validity) if target_is_real else torch.zeros_like(fake_validity)
            loss = F.binary_cross_entropy_with_logits(fake_validity, target, reduction='none')
            
        elif self.gan_type == 'lsgan':
            target = torch.ones_like(fake_validity) if target_is_real else torch.zeros_like(fake_validity)
            loss = F.mse_loss(fake_validity, target, reduction='none')
            
        elif self.gan_type == 'wgan':
            # 最大化 D(G(z)) 等价于最小化 -D(G(z))
            loss = -fake_validity if target_is_real else fake_validity
            
        elif self.gan_type == 'hinge':
            # Hinge 生成器损失: -D(G(z))
            loss = -fake_validity
        else:
            raise ValueError(f"Unknown GAN type: {self.gan_type}")
        
        return self._reduce(loss)


class GradientPenaltyLoss(nn.Module):
    """
    梯度惩罚损失
    
    用于 WGAN-GP，强制判别器满足 Lipschitz 约束。
    """
    
    def __init__(self, lambda_gp: float = 10.0):
        """
        Args:
            lambda_gp: 梯度惩罚系数
        """
        super().__init__()
        self.lambda_gp = lambda_gp
    
    def forward(
        self,
        discriminator: nn.Module,
        real_data: Tensor,
        fake_data: Tensor,
        condition: Optional[Tensor] = None
    ) -> Tensor:
        """
        计算梯度惩罚
        
        Args:
            discriminator: 判别器网络
            real_data: 真实数据 [B, ...]
            fake_data: 生成数据 [B, ...]
            condition: 条件向量 [B, C]（可选）
            
        Returns:
            梯度惩罚损失
        """
        batch_size = real_data.size(0)
        
        # 随机插值因子
        alpha = torch.rand(batch_size, 1, device=real_data.device)
        
        # 对空间维度进行广播
        for _ in range(real_data.dim() - 2):
            alpha = alpha.unsqueeze(-1)
        
        # 插值样本
        interpolates = alpha * real_data + (1 - alpha) * fake_data
        interpolates.requires_grad_(True)
        
        # 判别器前向传播
        if condition is not None:
            disc_output = discriminator(interpolates, condition)
        else:
            disc_output = discriminator(interpolates)
        
        # 计算梯度
        gradients = torch.autograd.grad(
            outputs=disc_output,
            inputs=interpolates,
            grad_outputs=torch.ones_like(disc_output),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # 计算梯度范数
        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        
        # 梯度惩罚: (||∇||_2 - 1)^2
        gradient_penalty = ((gradient_norm - 1) ** 2).mean()
        
        return self.lambda_gp * gradient_penalty


class ConditionalConsistencyLoss(BaseLoss):
    """
    条件一致性损失
    
    确保生成的设计满足给定的条件约束。
    通过代理模型评估生成性能与目标条件的差异。
    """
    
    def __init__(
        self,
        weight: float = 1.0,
        loss_type: str = 'mse',
        reduction: str = 'mean'
    ):
        """
        Args:
            weight: 损失权重
            loss_type: 损失类型 ('mse', 'mae')
            reduction: 归约方式
        """
        super().__init__(reduction)
        self.weight = weight
        self.loss_fn = nn.MSELoss(reduction='none') if loss_type == 'mse' else nn.L1Loss(reduction='none')
    
    def forward(
        self,
        generated_performance: Tensor,
        target_condition: Tensor
    ) -> Tensor:
        """
        计算条件一致性损失
        
        Args:
            generated_performance: 生成设计的预测性能 [B, D]
            target_condition: 目标条件 [B, D]
            
        Returns:
            条件一致性损失
        """
        loss = self.loss_fn(generated_performance, target_condition)
        return self.weight * self._reduce(loss)


class CGANCombinedLoss(nn.Module):
    """
    CGAN 组合损失
    
    组合对抗损失、条件一致性损失和正则化损失。
    用于 CGAN 生成器训练。
    """
    
    def __init__(
        self,
        gan_type: str = 'wgan-gp',
        condition_weight: float = 1.0,
        diversity_weight: float = 0.0,
        lambda_gp: float = 10.0
    ):
        """
        Args:
            gan_type: GAN 类型
            condition_weight: 条件一致性权重
            diversity_weight: 多样性损失权重
            lambda_gp: 梯度惩罚系数
        """
        super().__init__()
        self.gan_loss = GANLoss(gan_type)
        self.gp_loss = GradientPenaltyLoss(lambda_gp)
        self.condition_weight = condition_weight
        self.diversity_weight = diversity_weight
    
    def forward(
        self,
        fake_validity: Tensor,
        generated_performance: Optional[Tensor] = None,
        target_condition: Optional[Tensor] = None,
        designs: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        """
        计算组合损失
        
        Args:
            fake_validity: 判别器对生成样本的评分
            generated_performance: 生成设计的预测性能
            target_condition: 目标条件
            designs: 生成的设计（用于多样性计算）
            
        Returns:
            损失字典
        """
        losses = {}
        
        # 对抗损失
        losses['gan'] = self.gan_loss.generator_loss(fake_validity)
        total = losses['gan']
        
        # 条件一致性损失
        if generated_performance is not None and target_condition is not None:
            losses['condition'] = F.mse_loss(generated_performance, target_condition)
            total += self.condition_weight * losses['condition']
        
        # 多样性损失
        if designs is not None and self.diversity_weight > 0:
            losses['diversity'] = self._compute_diversity(designs)
            total += self.diversity_weight * losses['diversity']
        
        losses['total'] = total
        
        return losses
    
    def _compute_diversity(self, designs: Tensor) -> Tensor:
        """计算多样性损失（鼓励不同样本间差异）"""
        if designs.size(0) < 2:
            return torch.tensor(0.0, device=designs.device)
        
        designs_flat = designs.view(designs.size(0), -1)
        # 计算样本间距离
        dist = torch.pdist(designs_flat)
        # 返回负距离，最小化时增加多样性
        return -dist.mean()
    
    def compute_gp(
        self,
        discriminator: nn.Module,
        real_designs: Tensor,
        fake_designs: Tensor,
        conditions: Tensor
    ) -> Tensor:
        """计算梯度惩罚"""
        return self.gp_loss(discriminator, real_designs, fake_designs, conditions)


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
    'contrastive': ContrastiveLoss,
    'mdn': MDNLoss,
    'mdn_regularized': MDNRegularizedLoss,
    'gan': GANLoss,
    'gradient_penalty': GradientPenaltyLoss,
    'conditional_consistency': ConditionalConsistencyLoss,
    'cgan_combined': CGANCombinedLoss
}


def get_loss(name: str, **kwargs) -> nn.Module:
    """获取损失函数"""
    if name not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss function: {name}")
    return LOSS_REGISTRY[name](**kwargs)

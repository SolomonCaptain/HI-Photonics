"""
损失函数模块

提供光子学逆向设计专用的损失函数。
"""

from typing import Dict, Optional, List, Union, Any, Tuple, Callable
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


class PDEResidualLoss(BaseLoss):
    """
    PDE 残差损失
    
    计算偏微分方程残差，用于 PINN 训练。
    支持多种常用 PDE。
    """
    
    def __init__(
        self,
        pde_type: str = 'helmholtz',
        reduction: str = 'mean'
    ):
        """
        Args:
            pde_type: PDE 类型 ('helmholtz', 'laplace', 'heat', 'wave')
            reduction: 归约方式
        """
        super().__init__(reduction)
        self.pde_type = pde_type
    
    def forward(
        self,
        residual: Tensor,
        **kwargs
    ) -> Tensor:
        """
        计算 PDE 残差损失
        
        Args:
            residual: PDE 残差
            **kwargs: 额外参数
            
        Returns:
            残差损失
        """
        return self._reduce(residual ** 2)


class HelmholtzLoss(nn.Module):
    """
    Helmholtz 方程损失
    
    (∇² + k²)u = f
    
    用于电磁波和谐振腔问题。
    """
    
    def __init__(
        self,
        k: float = 1.0,
        weight: float = 1.0
    ):
        """
        Args:
            k: 波数
            weight: 损失权重
        """
        super().__init__()
        self.k = k
        self.weight = weight
    
    def forward(
        self,
        u: Tensor,
        laplacian: Tensor,
        f: Optional[Tensor] = None
    ) -> Tensor:
        """
        计算 Helmholtz 残差
        
        Args:
            u: 场值
            laplacian: 场的拉普拉斯
            f: 源项（可选）
            
        Returns:
            Helmholtz 残差损失
        """
        k2 = self.k ** 2
        
        if f is None:
            residual = laplacian + k2 * u
        else:
            residual = laplacian + k2 * u - f
        
        return self.weight * (residual ** 2).mean()


class MaxwellLoss(nn.Module):
    """
    Maxwell 方程损失
    
    频域 Maxwell 方程:
    ∇ × E = -jωμH
    ∇ × H = jωεE + J
    
    用于电磁场问题。
    """
    
    def __init__(
        self,
        omega: float = 1.0,
        epsilon: float = 1.0,
        mu: float = 1.0,
        weight: float = 1.0
    ):
        """
        Args:
            omega: 角频率
            epsilon: 介电常数
            mu: 磁导率
            weight: 损失权重
        """
        super().__init__()
        self.omega = omega
        self.epsilon = epsilon
        self.mu = mu
        self.weight = weight
    
    def forward(
        self,
        curl_E: Tensor,
        curl_H: Tensor,
        E: Tensor,
        H: Tensor,
        J: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        """
        计算 Maxwell 方程残差
        
        Args:
            curl_E: E 的旋度
            curl_H: H 的旋度
            E: 电场
            H: 磁场
            J: 电流密度（可选）
            
        Returns:
            残差字典
        """
        # ∇ × E + jωμH = 0
        residual_E = curl_E + 1j * self.omega * self.mu * H
        
        # ∇ × H - jωεE = J
        if J is None:
            residual_H = curl_H - 1j * self.omega * self.epsilon * E
        else:
            residual_H = curl_H - 1j * self.omega * self.epsilon * E - J
        
        loss_E = (residual_E.abs() ** 2).mean()
        loss_H = (residual_H.abs() ** 2).mean()
        
        total_loss = self.weight * (loss_E + loss_H)
        
        return {
            'maxwell_E': loss_E,
            'maxwell_H': loss_H,
            'total': total_loss
        }


class BoundaryConditionLoss(BaseLoss):
    """
    边界条件损失
    
    支持多种边界条件:
    - Dirichlet: u = g
    - Neumann: ∂u/∂n = g
    - Robin: αu + β∂u/∂n = g
    - PML: 吸收边界条件
    """
    
    def __init__(
        self,
        bc_type: str = 'dirichlet',
        weight: float = 1.0,
        reduction: str = 'mean'
    ):
        """
        Args:
            bc_type: 边界条件类型
            weight: 损失权重
            reduction: 归约方式
        """
        super().__init__(reduction)
        self.bc_type = bc_type.lower()
        self.weight = weight
    
    def forward(
        self,
        pred: Tensor,
        target: Tensor,
        normal_grad: Optional[Tensor] = None
    ) -> Tensor:
        """
        计算边界条件损失
        
        Args:
            pred: 预测值
            target: 目标值
            normal_grad: 法向梯度（Neumann/Robin 需要）
            
        Returns:
            边界条件损失
        """
        if self.bc_type == 'dirichlet':
            loss = (pred - target) ** 2
        
        elif self.bc_type == 'neumann':
            if normal_grad is None:
                raise ValueError("normal_grad required for Neumann BC")
            loss = (normal_grad - target) ** 2
        
        elif self.bc_type == 'robin':
            if normal_grad is None:
                raise ValueError("normal_grad required for Robin BC")
            # αu + β∂u/∂n = g
            # 假设 target 是 (alpha, beta, g) 的元组
            alpha, beta, g = target
            loss = (alpha * pred + beta * normal_grad - g) ** 2
        
        else:
            raise ValueError(f"Unknown BC type: {self.bc_type}")
        
        return self.weight * self._reduce(loss)


class PINNCombinedLoss(nn.Module):
    """
    PINN 组合损失
    
    组合 PDE 残差损失、边界条件损失和数据损失。
    支持自适应权重调整。
    """
    
    def __init__(
        self,
        physics_weight: float = 1.0,
        bc_weight: float = 1.0,
        data_weight: float = 1.0,
        adaptive_weights: bool = False,
        update_freq: int = 100
    ):
        """
        Args:
            physics_weight: 物理损失权重
            bc_weight: 边界条件权重
            data_weight: 数据损失权重
            adaptive_weights: 是否使用自适应权重
            update_freq: 权重更新频率
        """
        super().__init__()
        self.physics_weight = physics_weight
        self.bc_weight = bc_weight
        self.data_weight = data_weight
        self.adaptive_weights = adaptive_weights
        self.update_freq = update_freq
        
        # 初始化可学习权重
        if adaptive_weights:
            self.log_weights = nn.ParameterDict({
                'physics': nn.Parameter(torch.tensor(0.0)),
                'bc': nn.Parameter(torch.tensor(0.0)),
                'data': nn.Parameter(torch.tensor(0.0))
            })
        
        # 损失函数
        self.pde_loss = PDEResidualLoss()
        self.bc_loss = BoundaryConditionLoss()
    
    def forward(
        self,
        physics_residual: Optional[Tensor] = None,
        bc_pred: Optional[Tensor] = None,
        bc_target: Optional[Tensor] = None,
        data_pred: Optional[Tensor] = None,
        data_target: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        """
        计算组合损失
        
        Args:
            physics_residual: PDE 残差
            bc_pred: 边界预测值
            bc_target: 边界目标值
            data_pred: 数据预测值
            data_target: 数据目标值
            
        Returns:
            损失字典
        """
        losses = {}
        total = torch.tensor(0.0)
        
        # 物理损失
        if physics_residual is not None:
            weight = self._get_weight('physics')
            losses['physics'] = self.pde_loss(physics_residual)
            total = total + weight * losses['physics']
        
        # 边界条件损失
        if bc_pred is not None and bc_target is not None:
            weight = self._get_weight('bc')
            losses['bc'] = self.bc_loss(bc_pred, bc_target)
            total = total + weight * losses['bc']
        
        # 数据损失
        if data_pred is not None and data_target is not None:
            weight = self._get_weight('data')
            losses['data'] = F.mse_loss(data_pred, data_target)
            total = total + weight * losses['data']
        
        losses['total'] = total
        
        return losses
    
    def _get_weight(self, key: str) -> Tensor:
        """获取损失权重"""
        if self.adaptive_weights:
            # λ = exp(-log_weight)
            return torch.exp(-self.log_weights[key])
        else:
            weights = {
                'physics': self.physics_weight,
                'bc': self.bc_weight,
                'data': self.data_weight
            }
            return torch.tensor(weights.get(key, 1.0))
    
    def update_weights_adaptive(
        self,
        losses: Dict[str, Tensor],
        model: nn.Module
    ):
        """
        自适应更新权重
        
        基于梯度平衡策略更新损失权重。
        """
        if not self.adaptive_weights:
            return
        
        grad_norms = {}
        
        for key, loss in losses.items():
            if key == 'total':
                continue
            
            # 计算梯度范数
            grad = torch.autograd.grad(
                loss, model.parameters(),
                retain_graph=True,
                create_graph=False
            )
            grad_norm = torch.norm(torch.cat([g.flatten() for g in grad if g is not None]))
            grad_norms[key] = grad_norm
        
        # 更新权重
        total_grad_norm = sum(grad_norms.values())
        for key in grad_norms:
            # λ_i = ||∇L_i|| / Σ_j ||∇L_j||
            new_weight = grad_norms[key] / (total_grad_norm + 1e-8)
            # 更新 log 权重
            self.log_weights[key].data = -torch.log(new_weight + 1e-8)


# ============================================================================
# VAE 专用损失函数
# ============================================================================

class VAEReconstructionLoss(BaseLoss):
    """
    VAE 重建损失

    支持多种重建损失类型:
    - MSE (均方误差)
    - BCE (二元交叉熵)
    - L1 (平均绝对误差)
    - Focal (焦点损失，用于不平衡数据)

    适用于光子学设计参数的重建任务。
    """

    def __init__(
        self,
        loss_type: str = 'mse',
        reduction: str = 'mean',
        focal_gamma: float = 2.0
    ):
        """
        Args:
            loss_type: 损失类型 ('mse', 'bce', 'l1', 'focal')
            reduction: 归约方式
            focal_gamma: 焦点损失的 gamma 参数
        """
        super().__init__(reduction)
        self.loss_type = loss_type
        self.focal_gamma = focal_gamma

    def forward(
        self,
        recon_x: Tensor,
        x: Tensor
    ) -> Tensor:
        """
        计算重建损失

        Args:
            recon_x: 重建的设计参数 [B, H, W] 或 [B, C, H, W]
            x: 原始设计参数

        Returns:
            重建损失
        """
        if self.loss_type == 'mse':
            loss = (recon_x - x) ** 2
        elif self.loss_type == 'bce':
            # 数值稳定性
            recon_x = torch.clamp(recon_x, min=1e-7, max=1 - 1e-7)
            loss = -(x * torch.log(recon_x) + (1 - x) * torch.log(1 - recon_x))
        elif self.loss_type == 'l1':
            loss = torch.abs(recon_x - x)
        elif self.loss_type == 'focal':
            # Focal Loss for imbalanced reconstruction
            bce = F.binary_cross_entropy(recon_x, x, reduction='none')
            pt = torch.exp(-bce)
            loss = (1 - pt) ** self.focal_gamma * bce
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        return self._reduce(loss)


class KLDivergenceLoss(BaseLoss):
    """
    KL 散度损失

    计算 q(z|x) 和 p(z) 之间的 KL 散度:
    KL(q(z|x) || p(z)) = -0.5 * Σ(1 + log(σ²) - μ² - σ²)

    其中 p(z) 是标准正态分布 N(0, I)。

    支持:
    - 标准 KL 散度
    - 闭环 KL 散度（更稳定）
    - KL 退火（Cyclical Annealing）
    """

    def __init__(
        self,
        reduction: str = 'mean',
        closed_form: bool = True,
        unit_variance: bool = True
    ):
        """
        Args:
            reduction: 归约方式
            closed_form: 是否使用闭环形式（更稳定）
            unit_variance: 先验是否为单位方差
        """
        super().__init__(reduction)
        self.closed_form = closed_form
        self.unit_variance = unit_variance

    def forward(
        self,
        mu: Tensor,
        logvar: Tensor
    ) -> Tensor:
        """
        计算 KL 散度

        Args:
            mu: 潜在空间均值 [B, latent_dim]
            logvar: 潜在空间对数方差 [B, latent_dim]

        Returns:
            KL 散度
        """
        if self.closed_form:
            # 闭环形式，数值更稳定
            # KL = -0.5 * Σ(1 + log(σ²) - μ² - σ²)
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        else:
            # 使用重参数化采样估计
            std = torch.exp(0.5 * logvar)
            z = mu + std * torch.randn_like(std)

            # 计算对数概率比
            log_pz = -0.5 * (z ** 2).sum(dim=-1)  # N(0, I)
            log_qz = -0.5 * (((z - mu) / std) ** 2 + logvar).sum(dim=-1)  # N(mu, σ²)

            kl = log_qz - log_pz

        return self._reduce(kl)


class BetaVAELoss(nn.Module):
    """
    β-VAE 损失函数

    通过调整 β 参数控制潜在空间的解耦程度:
    L = L_recon + β * L_KL

    β > 1: 鼓励解耦的潜在表示
    β = 1: 标准 VAE
    β < 1: 更好的重建质量

    支持:
    - 固定 β
    - 线性预热
    - 周期性退火
    """

    def __init__(
        self,
        beta: float = 1.0,
        recon_type: str = 'mse',
        warmup_epochs: int = 0,
        cyclical_annealing: bool = False,
        cyclical_period: int = 10,
        reduction: str = 'mean'
    ):
        """
        Args:
            beta: KL 损失权重
            recon_type: 重建损失类型
            warmup_epochs: β 预热的 epoch 数
            cyclical_annealing: 是否使用周期性退火
            cyclical_period: 周期性退火的周期
            reduction: 归约方式
        """
        super().__init__()
        self.beta = beta
        self.warmup_epochs = warmup_epochs
        self.cyclical_annealing = cyclical_annealing
        self.cyclical_period = cyclical_period

        self.recon_loss = VAEReconstructionLoss(recon_type, reduction)
        self.kl_loss = KLDivergenceLoss(reduction)

        self.current_epoch = 0

    def forward(
        self,
        x: Tensor,
        recon_x: Tensor,
        mu: Tensor,
        logvar: Tensor
    ) -> Dict[str, Tensor]:
        """
        计算 β-VAE 损失

        Args:
            x: 原始输入
            recon_x: 重建输入
            mu: 潜在均值
            logvar: 潜在对数方差

        Returns:
            损失字典
        """
        # 重建损失
        recon = self.recon_loss(recon_x, x)

        # KL 散度
        kl = self.kl_loss(mu, logvar)

        # 获取当前 β
        current_beta = self.get_beta()

        # 总损失
        total = recon + current_beta * kl

        return {
            'recon': recon,
            'kl': kl,
            'beta': torch.tensor(current_beta),
            'total': total
        }

    def get_beta(self) -> float:
        """获取当前的 β 值"""
        if self.cyclical_annealing:
            # 周期性退火
            cycle = self.current_epoch % self.cyclical_period
            return self.beta * min(1.0, cycle / (self.cyclical_period / 2))
        elif self.current_epoch < self.warmup_epochs:
            # 线性预热
            return self.beta * (self.current_epoch + 1) / self.warmup_epochs
        return self.beta

    def step(self):
        """推进一个 epoch"""
        self.current_epoch += 1


class VAELatentRegularization(nn.Module):
    """
    VAE 潜在空间正则化

    对潜在空间施加额外约束:
    1. 容量约束 (Capacity Constraint): 限制信息瓶颈
    2. MMD 约束 (Maximum Mean Discrepancy): 匹配先验分布
    3. 对抗约束 (Adversarial): 对抗性正则化
    """

    def __init__(
        self,
        reg_type: str = 'capacity',
        capacity_start: float = 0.0,
        capacity_end: float = 25.0,
        capacity_epochs: int = 100,
        mmd_kernel: str = 'rbf',
        mmd_bandwidth: float = 1.0
    ):
        """
        Args:
            reg_type: 正则化类型 ('capacity', 'mmd', 'adversarial')
            capacity_start: 容量约束起始值
            capacity_end: 容量约束结束值
            capacity_epochs: 容量增长的 epoch 数
            mmd_kernel: MMD 核函数类型
            mmd_bandwidth: MMD 核带宽
        """
        super().__init__()
        self.reg_type = reg_type
        self.capacity_start = capacity_start
        self.capacity_end = capacity_end
        self.capacity_epochs = capacity_epochs
        self.mmd_kernel = mmd_kernel
        self.mmd_bandwidth = mmd_bandwidth

        self.current_epoch = 0

    def forward(
        self,
        z: Tensor,
        mu: Tensor,
        logvar: Tensor
    ) -> Tensor:
        """
        计算正则化损失

        Args:
            z: 潜在向量 [B, latent_dim]
            mu: 潜在均值
            logvar: 潜在对数方差

        Returns:
            正则化损失
        """
        if self.reg_type == 'capacity':
            return self._capacity_loss(mu, logvar)
        elif self.reg_type == 'mmd':
            return self._mmd_loss(z)
        elif self.reg_type == 'adversarial':
            return self._adversarial_loss(mu)
        else:
            raise ValueError(f"Unknown regularization type: {self.reg_type}")

    def _capacity_loss(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        容量约束损失

        KL 项的权重会随着容量 C(t) 变化:
        L_KL' = |KL(q(z|x) || p(z)) - C(t)|

        这鼓励模型逐渐增加潜在空间的使用。
        """
        # 计算 KL 散度
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)

        # 计算当前容量
        if self.current_epoch < self.capacity_epochs:
            capacity = self.capacity_start + \
                       (self.capacity_end - self.capacity_start) * \
                       self.current_epoch / self.capacity_epochs
        else:
            capacity = self.capacity_end

        # 容量约束损失
        loss = torch.abs(kl.mean() - capacity)

        return loss

    def _mmd_loss(self, z: Tensor) -> Tensor:
        """
        MMD (Maximum Mean Discrepancy) 损失

        计算潜在分布与先验分布之间的 MMD。
        比 KL 散度更平滑，适用于复杂分布。
        """
        batch_size = z.size(0)

        # 从先验采样
        z_prior = torch.randn_like(z)

        # 计算核矩阵
        if self.mmd_kernel == 'rbf':
            K_zz = self._rbf_kernel(z, z)
            K_pp = self._rbf_kernel(z_prior, z_prior)
            K_zp = self._rbf_kernel(z, z_prior)
        else:
            raise ValueError(f"Unknown kernel: {self.mmd_kernel}")

        # MMD 估计
        mmd = K_zz.mean() + K_pp.mean() - 2 * K_zp.mean()

        return mmd

    def _rbf_kernel(self, x: Tensor, y: Tensor) -> Tensor:
        """计算 RBF 核矩阵"""
        xx = (x ** 2).sum(dim=-1, keepdim=True)
        yy = (y ** 2).sum(dim=-1, keepdim=True)
        dist = xx + yy.T - 2 * x @ y.T

        return torch.exp(-dist / (2 * self.mmd_bandwidth ** 2))

    def _adversarial_loss(self, mu: Tensor) -> Tensor:
        """
        对抗性正则化损失

        鼓励潜在分布与先验分布难以区分。
        （需要配合判别器网络使用）
        """
        # 简化版本：直接鼓励均值接近 0
        return mu.pow(2).mean()

    def step(self):
        """推进一个 epoch"""
        self.current_epoch += 1


class VAETotalLoss(nn.Module):
    """
    VAE 组合损失函数

    组合多种损失用于 VAE 训练:
    1. 重建损失
    2. KL 散度
    3. 潜在空间正则化
    4. 感知损失（可选）

    支持自适应权重调整和多种训练策略。
    """

    def __init__(
        self,
        recon_type: str = 'mse',
        beta: float = 1.0,
        warmup_epochs: int = 0,
        reg_type: Optional[str] = None,
        reg_weight: float = 0.1,
        perceptual_weight: float = 0.0,
        adaptive_weights: bool = False
    ):
        """
        Args:
            recon_type: 重建损失类型
            beta: KL 损失权重
            warmup_epochs: β 预热 epoch 数
            reg_type: 正则化类型（None 表示不使用）
            reg_weight: 正则化权重
            perceptual_weight: 感知损失权重
            adaptive_weights: 是否使用自适应权重
        """
        super().__init__()

        self.beta = beta
        self.warmup_epochs = warmup_epochs
        self.reg_weight = reg_weight
        self.perceptual_weight = perceptual_weight
        self.adaptive_weights = adaptive_weights

        # 损失组件
        self.recon_loss = VAEReconstructionLoss(recon_type)
        self.kl_loss = KLDivergenceLoss()
        self.reg_loss = VAELatentRegularization(reg_type) if reg_type else None

        # 自适应权重
        if adaptive_weights:
            self.log_weights = nn.ParameterDict({
                'recon': nn.Parameter(torch.tensor(0.0)),
                'kl': nn.Parameter(torch.tensor(0.0))
            })

        self.current_epoch = 0

    def forward(
        self,
        x: Tensor,
        recon_x: Tensor,
        mu: Tensor,
        logvar: Tensor,
        z: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        """
        计算组合损失

        Args:
            x: 原始输入
            recon_x: 重建输入
            mu: 潜在均值
            logvar: 潜在对数方差
            z: 潜在向量（可选，用于某些正则化）

        Returns:
            损失字典
        """
        losses = {}

        # 重建损失
        losses['recon'] = self.recon_loss(recon_x, x)

        # KL 散度
        losses['kl'] = self.kl_loss(mu, logvar)

        # 获取权重
        if self.adaptive_weights:
            w_recon = torch.exp(-self.log_weights['recon'])
            w_kl = torch.exp(-self.log_weights['kl'])
        else:
            w_recon = 1.0
            current_beta = self._get_beta()
            w_kl = current_beta

        # 总损失
        total = w_recon * losses['recon'] + w_kl * losses['kl']

        # 正则化损失
        if self.reg_loss is not None and z is not None:
            losses['reg'] = self.reg_loss(z, mu, logvar)
            total = total + self.reg_weight * losses['reg']

        # 感知损失（简化版）
        if self.perceptual_weight > 0:
            losses['perceptual'] = self._compute_perceptual_loss(x, recon_x)
            total = total + self.perceptual_weight * losses['perceptual']

        losses['total'] = total

        # 保存当前 beta 用于日志
        losses['beta'] = torch.tensor(self._get_beta())

        return losses

    def _get_beta(self) -> float:
        """获取当前 β 值"""
        if self.current_epoch < self.warmup_epochs:
            return self.beta * (self.current_epoch + 1) / self.warmup_epochs
        return self.beta

    def _compute_perceptual_loss(self, x: Tensor, recon_x: Tensor) -> Tensor:
        """
        计算感知损失

        使用简单的梯度损失作为感知损失的替代。
        """
        # 水平梯度
        grad_x_h = torch.abs(x[:, :, 1:] - x[:, :, :-1])
        grad_r_h = torch.abs(recon_x[:, :, 1:] - recon_x[:, :, :-1])

        # 垂直梯度
        grad_x_v = torch.abs(x[:, 1:, :] - x[:, :-1, :])
        grad_r_v = torch.abs(recon_x[:, 1:, :] - recon_x[:, :-1, :])

        # 梯度损失
        loss_h = F.mse_loss(grad_r_h, grad_x_h)
        loss_v = F.mse_loss(grad_r_v, grad_x_v)

        return loss_h + loss_v

    def step(self):
        """推进一个 epoch"""
        self.current_epoch += 1
        if self.reg_loss is not None:
            self.reg_loss.step()


class VAEScheduler:
    """
    VAE 训练调度器

    管理 β 和容量等参数的动态调整:
    - 线性/周期性预热
    - 容量增长
    - 自适应调整
    """

    def __init__(
        self,
        loss_fn: VAETotalLoss,
        schedule_type: str = 'linear',
        max_epochs: int = 100
    ):
        """
        Args:
            loss_fn: VAE 损失函数
            schedule_type: 调度类型 ('linear', 'cyclical', 'capacity')
            max_epochs: 最大训练 epoch 数
        """
        self.loss_fn = loss_fn
        self.schedule_type = schedule_type
        self.max_epochs = max_epochs
        self.current_epoch = 0

    def step(self):
        """推进一个 epoch"""
        self.current_epoch += 1
        self.loss_fn.step()

    def get_current_beta(self) -> float:
        """获取当前 β 值"""
        return self.loss_fn._get_beta()

    def state_dict(self) -> Dict[str, Any]:
        """获取状态字典"""
        return {
            'current_epoch': self.current_epoch,
            'schedule_type': self.schedule_type,
            'max_epochs': self.max_epochs
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """加载状态字典"""
        self.current_epoch = state_dict['current_epoch']
        self.schedule_type = state_dict['schedule_type']
        self.max_epochs = state_dict['max_epochs']


# ============================================================================
# 物理约束损失函数
# ============================================================================

class DispersionLoss(nn.Module):
    """
    材料色散损失
    
    惩罚设计在多个波长下的性能不一致性。
    """
    
    def __init__(
        self,
        material: str = "silicon",
        wavelengths: Optional[List[float]] = None,
        reference_wavelength: float = 1.55,
        weight: float = 1.0
    ):
        """
        Args:
            material: 材料名称
            wavelengths: 波长列表 (μm)
            reference_wavelength: 参考波长
            weight: 损失权重
        """
        super().__init__()
        self.material = material
        self.reference_wavelength = reference_wavelength
        self.weight = weight
        
        # 默认波长范围（通信波段）
        if wavelengths is None:
            wavelengths = [1.30, 1.55, 1.70]
        self.wavelengths = wavelengths
        
        # 材料色散参数
        self._init_material_params()
    
    def _init_material_params(self):
        """初始化材料 Sellmeier 参数"""
        # 硅的 Sellmeier 系数
        if self.material == "silicon":
            self.register_buffer('sellmeier_B', 
                torch.tensor([10.6684293, 0.0030434748, 1.54133408]))
            self.register_buffer('sellmeier_C',
                torch.tensor([0.301516485, 1.13475115, 1104.0]))
        elif self.material == "silicon_dioxide":
            self.register_buffer('sellmeier_B',
                torch.tensor([0.6961663, 0.4079426, 0.8974794]))
            self.register_buffer('sellmeier_C',
                torch.tensor([0.004672, 0.013512, 97.934]))
        else:
            # 默认硅氮
            self.register_buffer('sellmeier_B', torch.tensor([2.8939, 0.0, 0.0]))
            self.register_buffer('sellmeier_C', torch.tensor([0.01951, 0.0, 0.0]))
    
    def _compute_refractive_index(self, wavelength: Tensor) -> Tensor:
        """计算给定波长下的折射率"""
        lam_sq = wavelength ** 2
        n_sq_minus_1 = torch.tensor(0.0, device=wavelength.device)
        
        for B, C in zip(self.sellmeier_B, self.sellmeier_C):
            n_sq_minus_1 = n_sq_minus_1 + B * lam_sq / (lam_sq - C)
        
        return torch.sqrt(n_sq_minus_1 + 1)
    
    def forward(
        self,
        design: Tensor,
        performance_dict: Optional[Dict[str, Tensor]] = None,
    ) -> Dict[str, Tensor]:
        """
        计算色散损失
        
        Args:
            design: 设计参数 [B, H, W] 或 [B, C, H, W]
            performance_dict: 各波长的性能字典（可选）
            
        Returns:
            损失字典
        """
        # 计算参考波长折射率
        n_ref = self._compute_refractive_index(
            torch.tensor(self.reference_wavelength, device=design.device)
        )
        
        losses = {}
        total_loss = torch.tensor(0.0, device=design.device)
        
        # 计算各波长的相位失配
        for wl in self.wavelengths:
            if wl == self.reference_wavelength:
                continue
            
            n_wl = self._compute_refractive_index(
                torch.tensor(wl, device=design.device)
            )
            delta_n = (n_wl - n_ref).abs()
            
            # 相位失配惩罚
            phase_mismatch = delta_n * design.mean() * 2 * torch.pi / wl
            losses[f'phase_mismatch_{wl:.2f}um'] = phase_mismatch
            total_loss = total_loss + phase_mismatch
        
        # 如果有性能数据，计算性能一致性损失
        if performance_dict is not None:
            performances = list(performance_dict.values())
            if len(performances) > 1:
                perf_stack = torch.stack(performances)
                perf_std = perf_stack.std()
                losses['performance_variance'] = perf_std
                total_loss = total_loss + perf_std
        
        losses['dispersion_total'] = total_loss * self.weight
        
        return losses


class ThermalLoss(nn.Module):
    """
    热效应损失
    
    惩罚设计对温度变化的敏感性。
    """
    
    def __init__(
        self,
        material: str = "silicon",
        thermo_optic_coeff: float = 1.86e-4,  # K⁻¹
        temperature_range: Tuple[float, float] = (280.0, 360.0),  # K
        reference_temperature: float = 300.0,  # K
        weight: float = 1.0
    ):
        """
        Args:
            material: 材料名称
            thermo_optic_coeff: 热光系数
            temperature_range: 工作温度范围
            reference_temperature: 参考温度
            weight: 损失权重
        """
        super().__init__()
        self.material = material
        self.thermo_optic_coeff = thermo_optic_coeff
        self.temp_range = temperature_range
        self.reference_temp = reference_temperature
        self.weight = weight
    
    def forward(
        self,
        design: Tensor,
        temperature_field: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        计算热效应损失
        
        Args:
            design: 设计参数
            temperature_field: 温度场（可选）
            
        Returns:
            损失字典
        """
        # 温度变化范围
        delta_T_max = self.temp_range[1] - self.reference_temp
        delta_T_min = self.temp_range[0] - self.reference_temp
        
        # 最大折射率变化
        max_delta_n = self.thermo_optic_coeff * max(abs(delta_T_max), abs(delta_T_min))
        
        # 计算热敏感度（边缘敏感度更高）
        if design.dim() == 4:
            design = design.squeeze(1)
        
        # 计算梯度（边缘检测）
        if design.dim() == 3:
            grad_x = (design[:, :, 1:] - design[:, :, :-1]).abs()
            grad_y = (design[:, 1:, :] - design[:, :-1, :]).abs()
        else:
            grad_x = (design[:, 1:] - design[:, :-1]).abs()
            grad_y = (design[1:, :] - design[:-1, :]).abs()
        
        edge_density = (grad_x.mean() + grad_y.mean()) / 2
        
        # 热稳定性损失
        thermal_sensitivity = max_delta_n * edge_density
        
        losses = {
            'thermal_sensitivity': thermal_sensitivity,
            'edge_density': edge_density,
            'max_delta_n': torch.tensor(max_delta_n, device=design.device),
            'thermal_total': thermal_sensitivity * self.weight
        }
        
        # 如果有温度场，计算额外的热约束
        if temperature_field is not None:
            temp_violation = F.relu(temperature_field - self.temp_range[1])
            losses['temperature_violation'] = temp_violation.mean()
            losses['thermal_total'] = losses['thermal_total'] + temp_violation.mean()
        
        return losses


class RobustnessLoss(nn.Module):
    """
    鲁棒性损失
    
    惩罚设计对制造公差的敏感性。
    """
    
    def __init__(
        self,
        cd_tolerance: float = 0.01,  # μm
        edge_roughness_rms: float = 0.003,  # μm
        resolution: float = 0.01,  # μm/pixel
        num_mc_samples: int = 5,
        weight: float = 1.0
    ):
        """
        Args:
            cd_tolerance: 关键尺寸公差
            edge_roughness_rms: 边缘粗糙度 RMS
            resolution: 网格分辨率
            num_mc_samples: 蒙特卡洛样本数
            weight: 损失权重
        """
        super().__init__()
        self.cd_tolerance = cd_tolerance
        self.roughness_rms = edge_roughness_rms
        self.resolution = resolution
        self.num_mc_samples = num_mc_samples
        self.weight = weight
    
    def forward(
        self,
        design: Tensor,
        perturbed_samples: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        计算鲁棒性损失
        
        Args:
            design: 设计参数
            perturbed_samples: 扰动样本（可选）
            
        Returns:
            损失字典
        """
        # 确保维度正确
        if design.dim() == 2:
            design = design.unsqueeze(0).unsqueeze(0)
        elif design.dim() == 3:
            design = design.unsqueeze(0)
        
        # Sobel 边缘检测
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype=torch.float32, device=design.device
        ).view(1, 1, 3, 3)
        
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            dtype=torch.float32, device=design.device
        ).view(1, 1, 3, 3)
        
        grad_x = F.conv2d(design, sobel_x, padding=1)
        grad_y = F.conv2d(design, sobel_y, padding=1)
        
        edge_strength = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
        
        # 边缘周长
        perimeter = edge_strength.sum()
        
        # 边缘密度
        edge_density = edge_strength.mean()
        
        # 鲁棒性指标
        roughness_factor = self.roughness_rms / self.resolution
        robustness_penalty = edge_density * roughness_factor
        
        losses = {
            'edge_perimeter': perimeter,
            'edge_density': edge_density,
            'robustness_penalty': robustness_penalty,
            'robustness_total': robustness_penalty * self.weight
        }
        
        # 如果有扰动样本，计算性能方差
        if perturbed_samples is not None:
            if perturbed_samples.dim() == 4:
                # [num_samples, B, H, W] -> 计算方差
                variance = perturbed_samples.var(dim=0).mean()
                losses['design_variance'] = variance
                losses['robustness_total'] = losses['robustness_total'] + variance
        
        return losses


class PhysicsConstraintLoss(nn.Module):
    """
    综合物理约束损失
    
    组合色散、热效应和鲁棒性约束。
    """
    
    def __init__(
        self,
        material: str = "silicon",
        # 色散参数
        wavelengths: Optional[List[float]] = None,
        reference_wavelength: float = 1.55,
        dispersion_weight: float = 0.5,
        # 热效应参数
        temperature_range: Tuple[float, float] = (280.0, 360.0),
        thermo_optic_coeff: float = 1.86e-4,
        thermal_weight: float = 0.5,
        # 鲁棒性参数
        cd_tolerance: float = 0.01,
        edge_roughness_rms: float = 0.003,
        resolution: float = 0.01,
        robustness_weight: float = 1.0,
        # 其他
        adaptive_weights: bool = False
    ):
        """
        Args:
            material: 材料名称
            wavelengths: 波长列表
            reference_wavelength: 参考波长
            dispersion_weight: 色散损失权重
            temperature_range: 温度范围
            thermo_optic_coeff: 热光系数
            thermal_weight: 热效应损失权重
            cd_tolerance: 尺寸公差
            edge_roughness_rms: 边缘粗糙度
            resolution: 网格分辨率
            robustness_weight: 鲁棒性损失权重
            adaptive_weights: 是否使用自适应权重
        """
        super().__init__()
        
        self.dispersion_loss = DispersionLoss(
            material=material,
            wavelengths=wavelengths,
            reference_wavelength=reference_wavelength,
            weight=dispersion_weight
        )
        
        self.thermal_loss = ThermalLoss(
            material=material,
            thermo_optic_coeff=thermo_optic_coeff,
            temperature_range=temperature_range,
            weight=thermal_weight
        )
        
        self.robustness_loss = RobustnessLoss(
            cd_tolerance=cd_tolerance,
            edge_roughness_rms=edge_roughness_rms,
            resolution=resolution,
            weight=robustness_weight
        )
        
        self.adaptive_weights = adaptive_weights
        
        if adaptive_weights:
            self.log_weights = nn.ParameterDict({
                'dispersion': nn.Parameter(torch.tensor(0.0)),
                'thermal': nn.Parameter(torch.tensor(0.0)),
                'robustness': nn.Parameter(torch.tensor(0.0))
            })
    
    def forward(
        self,
        design: Tensor,
        performance_dict: Optional[Dict[str, Tensor]] = None,
        temperature_field: Optional[Tensor] = None,
        perturbed_samples: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        计算综合物理约束损失
        
        Args:
            design: 设计参数
            performance_dict: 各波长性能（可选）
            temperature_field: 温度场（可选）
            perturbed_samples: 扰动样本（可选）
            
        Returns:
            损失字典
        """
        losses = {}
        
        # 计算各约束损失
        disp_losses = self.dispersion_loss(design, performance_dict)
        therm_losses = self.thermal_loss(design, temperature_field)
        robust_losses = self.robustness_loss(design, perturbed_samples)
        
        # 合并损失
        for k, v in disp_losses.items():
            losses[f'dispersion_{k}'] = v
        for k, v in therm_losses.items():
            losses[f'thermal_{k}'] = v
        for k, v in robust_losses.items():
            losses[f'robustness_{k}'] = v
        
        # 计算总损失
        if self.adaptive_weights:
            w_disp = torch.exp(-self.log_weights['dispersion'])
            w_therm = torch.exp(-self.log_weights['thermal'])
            w_robust = torch.exp(-self.log_weights['robustness'])
            
            total = (
                w_disp * disp_losses['dispersion_total'] +
                w_therm * therm_losses['thermal_total'] +
                w_robust * robust_losses['robustness_total']
            )
        else:
            total = (
                disp_losses['dispersion_total'] +
                therm_losses['thermal_total'] +
                robust_losses['robustness_total']
            )
        
        losses['physics_total'] = total
        
        return losses


# ============================================================================
# TNN 专用损失函数 - 解决"平均设计"问题
# ============================================================================

class InverseContrastiveLoss(nn.Module):
    """
    逆向设计对比学习损失
    
    解决 TNN 的"平均设计"问题：
    - 同一目标性能的不同设计应该彼此远离（避免平均化）
    - 相似目标性能的设计应该在潜在空间中形成有意义的结构
    
    核心思想：
    1. 正样本对：不同目标性能 -> 设计应该不同
    2. 负样本对：相同目标性能 -> 设计应该多样化（不聚集到均值）
    3. 特征对比：在逆向网络的潜在空间中进行对比学习
    """
    
    def __init__(
        self,
        temperature: float = 0.1,
        hard_negative_weight: float = 0.5,
        design_diversity_weight: float = 1.0,
        feature_dim: Optional[int] = None
    ):
        """
        Args:
            temperature: 温度参数，控制对比损失的锐度
            hard_negative_weight: 困难负样本权重
            design_diversity_weight: 设计多样性权重
            feature_dim: 特征维度（用于投影头）
        """
        super().__init__()
        self.temperature = temperature
        self.hard_negative_weight = hard_negative_weight
        self.design_diversity_weight = design_diversity_weight
        
        # 可选的投影头（将设计投影到对比空间）
        self.projection_head = None
        if feature_dim is not None:
            self.projection_head = nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.ReLU(),
                nn.Linear(feature_dim, feature_dim // 2)
            )
    
    def forward(
        self,
        designs: Tensor,
        target_performances: Tensor,
        design_features: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        """
        计算对比学习损失
        
        Args:
            designs: 生成的设计 [B, H, W] 或 [B, D]
            target_performances: 目标性能 [B, P]
            design_features: 设计特征（来自逆向网络中间层）[B, F]
            
        Returns:
            损失字典
        """
        batch_size = designs.size(0)
        
        # 展平设计
        designs_flat = designs.view(batch_size, -1)
        
        # 使用特征或原始设计
        if design_features is not None:
            features = design_features
            if self.projection_head is not None:
                features = self.projection_head(features)
        else:
            features = designs_flat
        
        # 归一化特征
        features = F.normalize(features, dim=1)
        
        # 计算性能相似度矩阵
        perf_normalized = F.normalize(target_performances, dim=1)
        perf_similarity = torch.mm(perf_normalized, perf_normalized.t())  # [B, B]
        
        # 计算设计相似度矩阵
        design_similarity = torch.mm(features, features.t())  # [B, B]
        
        # 创建掩码：区分正负样本
        # 正样本：性能相似度高
        # 负样本：性能相似度低
        positive_mask = (perf_similarity > 0.9).float()  # 性能非常相似
        negative_mask = (perf_similarity < 0.5).float()   # 性能差异较大
        
        # 移除对角线（自身）
        identity_mask = torch.eye(batch_size, device=designs.device)
        positive_mask = positive_mask * (1 - identity_mask)
        negative_mask = negative_mask * (1 - identity_mask)
        
        # ========== 核心对比损失 ==========
        # 对于相同目标性能，设计应该多样化（避免聚集到均值）
        design_diversity_loss = torch.tensor(0.0, device=designs.device)
        if positive_mask.sum() > 0:
            # 正样本对的设计相似度应该低（多样性）
            pos_design_sim = design_similarity * positive_mask
            design_diversity_loss = pos_design_sim.sum() / (positive_mask.sum() + 1e-8)
        
        # 对于不同目标性能，设计应该有明显差异
        performance_distinction_loss = torch.tensor(0.0, device=designs.device)
        if negative_mask.sum() > 0:
            # 负样本对的设计相似度可以稍高，但不能太高
            neg_design_sim = design_similarity * negative_mask
            # 使用 hinge loss: max(0, sim - margin)
            margin = 0.3
            performance_distinction_loss = F.relu(neg_design_sim - margin).sum() / (negative_mask.sum() + 1e-8)
        
        # ========== InfoNCE 风格损失 ==========
        # 将每个样本作为 anchor，找正负样本
        infonce_loss = torch.tensor(0.0, device=designs.device)
        if batch_size > 1:
            # 使用性能相似度作为正负样本的软标签
            logits = design_similarity / self.temperature
            
            # 创建目标分布：相似性能应该有相似的设计
            target_distribution = F.softmax(perf_similarity / self.temperature, dim=1)
            
            # 交叉熵损失
            log_probs = F.log_softmax(logits, dim=1)
            infonce_loss = -(target_distribution * log_probs).sum(dim=1).mean()
        
        # ========== 总损失 ==========
        total_loss = (
            self.design_diversity_weight * design_diversity_loss +
            0.5 * performance_distinction_loss +
            0.5 * infonce_loss
        )
        
        return {
            'design_diversity': design_diversity_loss,
            'performance_distinction': performance_distinction_loss,
            'infonce': infonce_loss,
            'contrastive_total': total_loss
        }


class DesignSharpnessLoss(nn.Module):
    """
    设计锐度损失
    
    惩罚"模糊"的平均设计，鼓励生成清晰、二值化的设计。
    
    原理：
    - 平均设计通常在每个位置都是 0.5 左右的值
    - 好的设计应该在边界处有锐利的过渡
    - 使用梯度强度和二值化程度来衡量锐度
    """
    
    def __init__(
        self,
        sharpness_weight: float = 1.0,
        binary_weight: float = 0.5,
        edge_weight: float = 0.3,
        target_sharpness: float = 0.8
    ):
        """
        Args:
            sharpness_weight: 锐度损失权重
            binary_weight: 二值化损失权重
            edge_weight: 边缘清晰度权重
            target_sharpness: 目标锐度值
        """
        super().__init__()
        self.sharpness_weight = sharpness_weight
        self.binary_weight = binary_weight
        self.edge_weight = edge_weight
        self.target_sharpness = target_sharpness
    
    def forward(self, designs: Tensor) -> Dict[str, Tensor]:
        """
        计算设计锐度损失
        
        Args:
            designs: 设计参数 [B, H, W]
            
        Returns:
            损失字典
        """
        if designs.dim() == 2:
            designs = designs.unsqueeze(0)
        
        batch_size = designs.size(0)
        
        # ========== 二值化程度 ==========
        # 好的设计应该接近 0 或 1，而不是中间值
        # 使用熵来衡量：entropy = -p*log(p) - (1-p)*log(1-p)
        # 最大熵在 p=0.5，最小熵在 p=0 或 p=1
        p = torch.clamp(designs, min=1e-7, max=1 - 1e-7)
        entropy = -p * torch.log(p) - (1 - p) * torch.log(1 - p)
        max_entropy = -0.5 * torch.log(torch.tensor(0.5)) - 0.5 * torch.log(torch.tensor(0.5))
        normalized_entropy = entropy / max_entropy  # 归一化到 [0, 1]
        
        # 二值化损失：熵越高（越接近 0.5），惩罚越大
        binary_loss = normalized_entropy.mean()
        
        # ========== 梯度强度（锐度） ==========
        # 清晰的设计应该有强的梯度
        grad_x = torch.abs(designs[:, :, 1:] - designs[:, :, :-1])
        grad_y = torch.abs(designs[:, 1:, :] - designs[:, :-1, :])
        
        grad_magnitude = torch.sqrt(grad_x[:, :, :-1].pow(2) + grad_y[:, :-1, :].pow(2) + 1e-8)
        avg_grad = grad_magnitude.mean()
        
        # 锐度损失：梯度应该足够强
        sharpness_loss = F.relu(self.target_sharpness - avg_grad)
        
        # ========== 边缘清晰度 ==========
        # 使用 Sobel 算子检测边缘
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype=torch.float32, device=designs.device
        ).view(1, 1, 3, 3)
        
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            dtype=torch.float32, device=designs.device
        ).view(1, 1, 3, 3)
        
        designs_4d = designs.unsqueeze(1)  # [B, 1, H, W]
        edge_x = F.conv2d(designs_4d, sobel_x, padding=1)
        edge_y = F.conv2d(designs_4d, sobel_y, padding=1)
        edge_strength = torch.sqrt(edge_x.pow(2) + edge_y.pow(2) + 1e-8)
        
        # 边缘应该集中而不是分散
        edge_concentration = edge_strength.std()  # 边缘强度方差大意味着有清晰的边界
        edge_loss = -edge_concentration  # 最大化方差
        
        # ========== 总损失 ==========
        total_loss = (
            self.sharpness_weight * sharpness_loss +
            self.binary_weight * binary_loss +
            self.edge_weight * edge_loss
        )
        
        return {
            'sharpness': sharpness_loss,
            'binary': binary_loss,
            'edge_clarity': edge_loss,
            'avg_gradient': avg_grad,
            'sharpness_total': total_loss
        }


class DiversityPreservingLoss(nn.Module):
    """
    多样性保持损失
    
    在训练过程中保持设计空间的多样性，避免所有设计塌缩到均值。
    
    方法：
    1. 批次内多样性：同一批次的设计应该彼此不同
    2. 时间多样性：同一目标性能在不同训练步应该产生不同的设计
    3. 模式覆盖：确保覆盖设计空间的不同区域
    """
    
    def __init__(
        self,
        batch_diversity_weight: float = 1.0,
        mode_coverage_weight: float = 0.5,
        num_modes: int = 4,
        temperature: float = 1.0
    ):
        """
        Args:
            batch_diversity_weight: 批次多样性权重
            mode_coverage_weight: 模式覆盖权重
            num_modes: 模式数量（设计空间的区域数）
            temperature: 温度参数
        """
        super().__init__()
        self.batch_diversity_weight = batch_diversity_weight
        self.mode_coverage_weight = mode_coverage_weight
        self.num_modes = num_modes
        self.temperature = temperature
        
        # 可学习的模式中心
        self.mode_centers = None
        self.initialized = False
    
    def _init_mode_centers(self, designs: Tensor):
        """初始化模式中心"""
        if self.mode_centers is None or self.mode_centers.size(0) != self.num_modes:
            # 使用 K-means 风格初始化
            designs_flat = designs.view(designs.size(0), -1)
            indices = torch.randperm(designs_flat.size(0))[:self.num_modes]
            self.mode_centers = nn.Parameter(designs_flat[indices].clone())
            self.initialized = True
    
    def forward(
        self,
        designs: Tensor,
        update_centers: bool = True
    ) -> Dict[str, Tensor]:
        """
        计算多样性保持损失
        
        Args:
            designs: 设计参数 [B, H, W]
            update_centers: 是否更新模式中心
            
        Returns:
            损失字典
        """
        batch_size = designs.size(0)
        designs_flat = designs.view(batch_size, -1)
        
        # 初始化模式中心
        if not self.initialized:
            self._init_mode_centers(designs)
        
        # ========== 批次内多样性 ==========
        # 计算批次内设计两两距离
        dist_matrix = torch.cdist(designs_flat, designs_flat, p=2)
        
        # 移除对角线
        mask = 1 - torch.eye(batch_size, device=designs.device)
        dist_matrix = dist_matrix * mask
        
        # 多样性损失：最小化平均距离的负值（最大化平均距离）
        avg_dist = dist_matrix.sum() / (mask.sum() + 1e-8)
        batch_diversity_loss = -avg_dist
        
        # ========== 模式覆盖损失 ==========
        # 设计应该均匀分布到各个模式
        mode_coverage_loss = torch.tensor(0.0, device=designs.device)
        
        if self.mode_centers is not None:
            # 计算每个设计到各模式中心的距离
            centers = self.mode_centers.to(designs.device)
            center_dists = torch.cdist(designs_flat, centers, p=2)  # [B, num_modes]
            
            # 软分配：设计属于各模式的概率
            mode_probs = F.softmax(-center_dists / self.temperature, dim=1)  # [B, num_modes]
            
            # 模式使用率：各模式被分配的设计比例
            mode_usage = mode_probs.sum(dim=0) / batch_size  # [num_modes]
            
            # 理想情况：均匀使用各模式
            target_usage = torch.ones(self.num_modes, device=designs.device) / self.num_modes
            
            # KL 散度作为模式覆盖损失
            mode_coverage_loss = F.kl_div(
                torch.log(mode_usage + 1e-8),
                target_usage,
                reduction='sum'
            )
            
            # 更新模式中心（移动平均）
            if update_centers and self.training:
                with torch.no_grad():
                    # 每个模式的加权平均
                    for i in range(self.num_modes):
                        weights = mode_probs[:, i].unsqueeze(1)
                        new_center = (weights * designs_flat).sum(dim=0) / (weights.sum() + 1e-8)
                        # 移动平均更新
                        self.mode_centers.data[i] = 0.9 * self.mode_centers.data[i] + 0.1 * new_center.to(self.mode_centers.device)
        
        # ========== 总损失 ==========
        total_loss = (
            self.batch_diversity_weight * batch_diversity_loss +
            self.mode_coverage_weight * mode_coverage_loss
        )
        
        return {
            'batch_diversity': batch_diversity_loss,
            'avg_distance': avg_dist,
            'mode_coverage': mode_coverage_loss,
            'diversity_total': total_loss
        }


class TNNAntiAverageLoss(nn.Module):
    """
    TNN 反平均损失
    
    专门针对 TNN 逆向设计的"平均设计"问题的综合损失函数。
    
    问题分析：
    1. 一对多映射：同一性能可能有多个不同的设计
    2. MSE 损失倾向于让所有设计收敛到均值
    3. 缺乏对设计多样性的明确约束
    
    解决方案：
    1. 对比学习：让设计形成有意义的结构
    2. 锐度约束：鼓励清晰的设计
    3. 多样性保持：避免塌缩
    """
    
    def __init__(
        self,
        contrastive_weight: float = 1.0,
        sharpness_weight: float = 0.5,
        diversity_weight: float = 0.3,
        temperature: float = 0.1,
        target_sharpness: float = 0.8,
        num_modes: int = 4
    ):
        """
        Args:
            contrastive_weight: 对比学习损失权重
            sharpness_weight: 锐度损失权重
            diversity_weight: 多样性损失权重
            temperature: 对比学习温度
            target_sharpness: 目标锐度
            num_modes: 模式数量
        """
        super().__init__()
        
        self.contrastive_weight = contrastive_weight
        self.sharpness_weight = sharpness_weight
        self.diversity_weight = diversity_weight
        
        self.contrastive_loss = InverseContrastiveLoss(temperature=temperature)
        self.sharpness_loss = DesignSharpnessLoss(target_sharpness=target_sharpness)
        self.diversity_loss = DiversityPreservingLoss(num_modes=num_modes)
    
    def forward(
        self,
        designs: Tensor,
        target_performances: Tensor,
        design_features: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        """
        计算反平均损失
        
        Args:
            designs: 生成的设计 [B, H, W]
            target_performances: 目标性能 [B, P]
            design_features: 设计特征（可选）
            
        Returns:
            损失字典
        """
        losses = {}
        
        # 对比学习损失
        if self.contrastive_weight > 0:
            contrastive = self.contrastive_loss(designs, target_performances, design_features)
            for k, v in contrastive.items():
                losses[f'contrastive_{k}'] = v
        
        # 锐度损失
        if self.sharpness_weight > 0:
            sharpness = self.sharpness_loss(designs)
            for k, v in sharpness.items():
                losses[f'sharpness_{k}'] = v
        
        # 多样性损失
        if self.diversity_weight > 0:
            diversity = self.diversity_loss(designs)
            for k, v in diversity.items():
                losses[f'diversity_{k}'] = v
        
        # 总损失
        total = torch.tensor(0.0, device=designs.device)
        
        if self.contrastive_weight > 0:
            total = total + self.contrastive_weight * losses['contrastive_contrastive_total']
        if self.sharpness_weight > 0:
            total = total + self.sharpness_weight * losses['sharpness_sharpness_total']
        if self.diversity_weight > 0:
            total = total + self.diversity_weight * losses['diversity_diversity_total']
        
        losses['anti_average_total'] = total
        
        return losses


class OptimalDesignGuidanceLoss(nn.Module):
    """
    最优设计引导损失
    
    引导逆向网络学习生成"最优"设计，而不仅仅是"满足性能"的设计。
    
    核心思想：
    1. 高性能区域引导：让设计向高透射率区域集中
    2. 物理合理性：确保设计满足物理约束
    3. 多目标平衡：在多个性能指标间找到平衡
    """
    
    def __init__(
        self,
        performance_weight: float = 1.0,
        physical_weight: float = 0.5,
        smoothness_weight: float = 0.1,
        min_feature_size: float = 0.1
    ):
        """
        Args:
            performance_weight: 性能引导权重
            physical_weight: 物理约束权重
            smoothness_weight: 平滑度权重
            min_feature_size: 最小特征尺寸
        """
        super().__init__()
        self.performance_weight = performance_weight
        self.physical_weight = physical_weight
        self.smoothness_weight = smoothness_weight
        self.min_feature_size = min_feature_size
    
    def forward(
        self,
        designs: Tensor,
        predicted_performances: Tensor,
        target_performances: Tensor
    ) -> Dict[str, Tensor]:
        """
        计算最优设计引导损失
        
        Args:
            designs: 设计参数 [B, H, W]
            predicted_performances: 预测性能 [B, P]
            target_performances: 目标性能 [B, P]
            
        Returns:
            损失字典
        """
        # ========== 性能引导 ==========
        # 鼓励预测性能超过目标（在可接受的范围内）
        performance_margin = predicted_performances - target_performances
        # 对于效率类指标，超过目标是好的
        # 对于损耗类指标，低于目标是好的
        # 这里假设第一个指标是效率
        performance_guidance = F.relu(-performance_margin[:, 0])  # 惩罚效率低于目标
        performance_guidance_loss = performance_guidance.mean()
        
        # ========== 物理合理性 ==========
        # 体积约束：确保设计不是全空或全满
        volume = designs.mean(dim=(1, 2))
        target_volume = torch.tensor(0.5, device=designs.device)
        volume_loss = F.mse_loss(volume, target_volume.expand_as(volume))
        
        # 连通性约束：避免孤立的小区域
        if designs.dim() == 2:
            designs = designs.unsqueeze(0)
        
        # 使用形态学操作近似检测孤立区域
        kernel_size = max(3, int(self.min_feature_size / 0.01))  # 假设分辨率为 0.01
        kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        
        # 平均池化后的差异（检测小特征）
        designs_4d = designs.unsqueeze(1)
        pooled = F.avg_pool2d(designs_4d, kernel_size, stride=1, padding=kernel_size // 2)
        isolation = (designs_4d - pooled).abs()
        isolation_loss = isolation.mean()
        
        physical_loss = volume_loss + 0.5 * isolation_loss
        
        # ========== 平滑度约束 ==========
        # 鼓励平滑的边界（但不能过于平滑）
        grad_x = torch.abs(designs[:, :, 1:] - designs[:, :, :-1])
        grad_y = torch.abs(designs[:, 1:, :] - designs[:, :-1, :])
        smoothness = (grad_x.mean() + grad_y.mean()) / 2
        
        # 平滑度应该在合理范围内
        target_smoothness = torch.tensor(0.1, device=designs.device)
        smoothness_loss = F.mse_loss(smoothness, target_smoothness)
        
        # ========== 总损失 ==========
        total = (
            self.performance_weight * performance_guidance_loss +
            self.physical_weight * physical_loss +
            self.smoothness_weight * smoothness_loss
        )
        
        return {
            'performance_guidance': performance_guidance_loss,
            'volume': volume_loss,
            'isolation': isolation_loss,
            'smoothness': smoothness_loss,
            'guidance_total': total
        }


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
    'cgan_combined': CGANCombinedLoss,
    'pde_residual': PDEResidualLoss,
    'helmholtz': HelmholtzLoss,
    'maxwell': MaxwellLoss,
    'boundary_condition': BoundaryConditionLoss,
    'pinn_combined': PINNCombinedLoss,
    # VAE 损失函数
    'vae_recon': VAEReconstructionLoss,
    'kl_divergence': KLDivergenceLoss,
    'beta_vae': BetaVAELoss,
    'vae_latent_reg': VAELatentRegularization,
    'vae_total': VAETotalLoss,
    # 物理约束损失函数
    'dispersion': DispersionLoss,
    'thermal': ThermalLoss,
    'robustness': RobustnessLoss,
    'physics_constraint': PhysicsConstraintLoss,
    # TNN 专用损失函数（解决"平均设计"问题）
    'inverse_contrastive': InverseContrastiveLoss,
    'design_sharpness': DesignSharpnessLoss,
    'diversity_preserving': DiversityPreservingLoss,
    'tnn_anti_average': TNNAntiAverageLoss,
    'optimal_design_guidance': OptimalDesignGuidanceLoss,
}


def get_loss(name: str, **kwargs) -> nn.Module:
    """获取损失函数"""
    if name not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss function: {name}")
    return LOSS_REGISTRY[name](**kwargs)

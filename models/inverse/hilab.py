"""
HiLAB: Hybrid Inverse-Design Framework
混合逆向设计框架

结合 VAE 潜在空间学习、伴随方法部分优化和贝叶斯优化，
实现高效的光子学逆向设计。

核心组件:
1. VAE (Variational Autoencoder): 学习设计的潜在空间表示
2. Bayesian Optimizer: 在潜在空间中进行高效优化
3. Adjoint-based refinement: 可选的伴随方法精细化

参考文献:
- Marzban et al., "HiLAB: A Hybrid Inverse-Design Framework", arXiv:2505.17491, 2025
- https://arxiv.org/abs/2505.17491
"""

from typing import Dict, Optional, Tuple, List, Union, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np

from models.base import BaseModel, ModelConfig, GenerativeModel


# ============================================================================
# 配置类
# ============================================================================

@dataclass
class VAEEncoderConfig(ModelConfig):
    """VAE 编码器配置"""
    name: str = "vae_encoder"

    # 网络架构
    hidden_channels: List[int] = field(default_factory=lambda: [32, 64, 128, 256])
    hidden_dims: List[int] = field(default_factory=lambda: [512, 256])
    kernel_size: int = 3
    padding: int = 1
    stride: int = 2

    # 输入输出
    design_shape: Tuple[int, int] = (200, 22)  # (H, W)
    latent_dim: int = 32  # 潜在空间维度

    # 正则化
    dropout_rate: float = 0.1
    batch_norm: bool = True

    # 激活函数
    activation: str = "relu"


@dataclass
class VAEDecoderConfig(ModelConfig):
    """VAE 解码器配置"""
    name: str = "vae_decoder"

    # 网络架构
    hidden_dims: List[int] = field(default_factory=lambda: [256, 512])
    hidden_channels: List[int] = field(default_factory=lambda: [128, 64, 32, 1])
    kernel_size: int = 4
    stride: int = 2
    padding: int = 1

    # 输入输出
    latent_dim: int = 32
    design_shape: Tuple[int, int] = (200, 22)

    # 正则化
    dropout_rate: float = 0.1
    batch_norm: bool = True

    # 激活函数
    activation: str = "relu"
    output_activation: str = "sigmoid"


@dataclass
class VAEConfig(ModelConfig):
    """VAE 完整配置"""
    name: str = "vae"

    # 子模块配置
    encoder_config: VAEEncoderConfig = field(default_factory=VAEEncoderConfig)
    decoder_config: VAEDecoderConfig = field(default_factory=VAEDecoderConfig)

    # 损失权重
    recon_weight: float = 1.0  # 重建损失权重
    kl_weight: float = 0.001   # KL 散度权重 (β-VAE)
    beta_warmup_epochs: int = 0  # β 预热轮数

    # 潜在空间
    latent_dim: int = 32


@dataclass
class BayesianOptimizerConfig(ModelConfig):
    """贝叶斯优化器配置"""
    name: str = "bayesian_optimizer"

    # 高斯过程参数
    kernel_type: str = "rbf"  # 'rbf', 'matern', 'spectral'
    kernel_lengthscale: float = 1.0
    kernel_variance: float = 1.0
    noise_variance: float = 1e-4

    # Matern 核参数
    matern_nu: float = 2.5  # 1.5 或 2.5

    # 采集函数
    acquisition_type: str = "ei"  # 'ei', 'ucb', 'pi', 'kg'
    ucb_beta: float = 2.0  # UCB 的 β 参数
    xi: float = 0.01  # EI/PI 的探索参数

    # 优化参数
    n_initial_samples: int = 10  # 初始随机采样数
    n_optimization_steps: int = 100  # 优化步数
    n_restarts: int = 10  # 采集函数优化的重启次数

    # 潜在空间边界
    latent_bounds: Tuple[float, float] = (-3.0, 3.0)  # 潜在空间采样边界


@dataclass
class HiLABConfig(ModelConfig):
    """HiLAB 完整配置"""
    name: str = "hilab"

    # 子模块配置
    vae_config: VAEConfig = field(default_factory=VAEConfig)
    optimizer_config: BayesianOptimizerConfig = field(default_factory=BayesianOptimizerConfig)

    # 逆向设计参数
    performance_dim: int = 3
    design_shape: Tuple[int, int] = (200, 22)

    # 优化流程控制
    use_adjoint_refinement: bool = False  # 是否使用伴随方法精细化
    adjoint_iterations: int = 50  # 伴随优化迭代次数
    adjoint_lr: float = 0.01  # 伴随优化学习率

    # 早停
    early_stopping_patience: int = 20
    convergence_threshold: float = 1e-4


# ============================================================================
# 激活函数工厂
# ============================================================================

def get_activation(name: str, **kwargs) -> nn.Module:
    """获取激活函数"""
    activations = {
        'relu': nn.ReLU,
        'leaky_relu': lambda: nn.LeakyReLU(kwargs.get('negative_slope', 0.2)),
        'gelu': nn.GELU,
        'silu': nn.SiLU,
        'tanh': nn.Tanh,
        'sigmoid': nn.Sigmoid,
        'none': nn.Identity
    }
    if name.lower() not in activations:
        raise ValueError(f"Unknown activation: {name}")
    act_class = activations[name.lower()]
    return act_class() if name.lower() not in ['leaky_relu'] else act_class


# ============================================================================
# VAE 编码器
# ============================================================================

class VAEEncoder(nn.Module):
    """
    VAE 编码器

    将设计参数映射到潜在空间的均值和对数方差。

    架构:
        输入: [B, 1, H, W] 设计参数网格
          ↓
        ConvEncoder (多层卷积下采样)
          ↓
        FC Layers
          ↓
        μ: [B, latent_dim] 均值
        log_σ²: [B, latent_dim] 对数方差
    """

    def __init__(self, config: Optional[VAEEncoderConfig] = None):
        super().__init__()
        self.config = config or VAEEncoderConfig()

        # 保存关键参数
        self.design_shape = self.config.design_shape
        self.latent_dim = self.config.latent_dim

        # 构建编码器
        self.encoder = self._build_encoder()
        self.fc_mu = nn.Linear(self.config.hidden_dims[-1], self.latent_dim)
        self.fc_logvar = nn.Linear(self.config.hidden_dims[-1], self.latent_dim)

        # 初始化权重
        self._init_weights()

    def _build_encoder(self) -> nn.Module:
        """构建卷积编码器"""
        layers = []
        in_channels = 1
        config = self.config

        for i, out_channels in enumerate(config.hidden_channels):
            # 卷积层
            layers.append(nn.Conv2d(
                in_channels, out_channels,
                kernel_size=config.kernel_size,
                stride=config.stride if i > 0 else 1,
                padding=config.padding
            ))

            # 批归一化
            if config.batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))

            # 激活函数
            layers.append(get_activation(config.activation))

            # Dropout
            if config.dropout_rate > 0:
                layers.append(nn.Dropout2d(config.dropout_rate))

            in_channels = out_channels

        self.conv_out = nn.Sequential(*layers)

        # 计算展平后的维度
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, *self.design_shape)
            dummy_output = self.conv_out(dummy_input)
            self.flattened_dim = dummy_output.view(1, -1).size(1)

        # 全连接层
        fc_layers = []
        in_dim = self.flattened_dim
        for out_dim in config.hidden_dims:
            fc_layers.append(nn.Linear(in_dim, out_dim))
            if config.batch_norm:
                fc_layers.append(nn.BatchNorm1d(out_dim))
            fc_layers.append(get_activation(config.activation))
            if config.dropout_rate > 0:
                fc_layers.append(nn.Dropout(config.dropout_rate))
            in_dim = out_dim

        self.fc = nn.Sequential(*fc_layers)

        return nn.Sequential(self.conv_out, nn.Flatten(), self.fc)

    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        编码输入到潜在空间

        Args:
            x: 设计参数 [B, H, W] 或 [B, 1, H, W]

        Returns:
            mu: 潜在空间均值 [B, latent_dim]
            logvar: 潜在空间对数方差 [B, latent_dim]
        """
        # 确保输入维度正确
        if x.dim() == 3:
            x = x.unsqueeze(1)

        # 编码
        h = self.encoder(x)

        # 输出均值和方差
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        return mu, logvar


# ============================================================================
# VAE 解码器
# ============================================================================

class VAEDecoder(nn.Module):
    """
    VAE 解码器

    从潜在空间向量重建设计参数。

    架构:
        输入: [B, latent_dim] 潜在向量
          ↓
        FC Layers (扩展维度)
          ↓
        Reshape [B, C, H', W']
          ↓
        ConvDecoder (转置卷积上采样)
          ↓
        输出: [B, 1, H, W] 设计参数
    """

    def __init__(self, config: Optional[VAEDecoderConfig] = None):
        super().__init__()
        self.config = config or VAEDecoderConfig()

        # 保存关键参数
        self.latent_dim = self.config.latent_dim
        self.design_shape = self.config.design_shape

        # 构建解码器
        self.fc = self._build_fc()
        self.decoder = self._build_decoder()

        # 初始化权重
        self._init_weights()

    def _build_fc(self) -> nn.Module:
        """构建前置全连接层"""
        layers = []
        in_dim = self.latent_dim

        for out_dim in self.config.hidden_dims:
            layers.append(nn.Linear(in_dim, out_dim))
            if self.config.batch_norm:
                layers.append(nn.BatchNorm1d(out_dim))
            layers.append(get_activation(self.config.activation))
            if self.config.dropout_rate > 0:
                layers.append(nn.Dropout(self.config.dropout_rate))
            in_dim = out_dim

        self.fc_output_dim = in_dim
        return nn.Sequential(*layers)

    def _build_decoder(self) -> nn.Module:
        """构建转置卷积解码器"""
        config = self.config

        # 计算解码器输入尺寸
        h, w = self.design_shape
        n_upsamples = len(config.hidden_channels) - 1
        self.init_h = max(h // (2 ** n_upsamples), 1)
        self.init_w = max(w // (2 ** n_upsamples), 1)

        # 计算初始通道数
        init_channels = self.fc_output_dim // (self.init_h * self.init_w)
        init_channels = max(init_channels, config.hidden_channels[0])

        # 调整全连接层输出维度
        self.fc_adjust = nn.Linear(
            self.fc_output_dim,
            init_channels * self.init_h * self.init_w
        )

        # 构建转置卷积层
        layers = []
        in_channels = init_channels

        for i, out_channels in enumerate(config.hidden_channels[:-1]):
            # 转置卷积
            layers.append(nn.ConvTranspose2d(
                in_channels, out_channels,
                kernel_size=config.kernel_size,
                stride=config.stride,
                padding=config.padding,
                output_padding=1
            ))

            # 批归一化
            if config.batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))

            # 激活函数
            layers.append(get_activation(config.activation))

            in_channels = out_channels

        # 最终输出层
        layers.append(nn.Conv2d(in_channels, 1, kernel_size=3, padding=1))

        # 输出激活函数
        layers.append(get_activation(config.output_activation))

        return nn.Sequential(*layers)

    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, z: Tensor) -> Tensor:
        """
        从潜在向量解码设计

        Args:
            z: 潜在向量 [B, latent_dim]

        Returns:
            design: 设计参数 [B, H, W]
        """
        # 全连接层扩展
        x = self.fc(z)
        x = self.fc_adjust(x)

        # 重塑为特征图
        x = x.view(x.size(0), -1, self.init_h, self.init_w)

        # 解码
        design = self.decoder(x)

        # 调整到目标尺寸
        design = F.interpolate(design, size=self.design_shape, mode='bilinear', align_corners=False)

        # 移除通道维度
        design = design.squeeze(1)

        return design


# ============================================================================
# VAE 完整模型
# ============================================================================

class VAE(GenerativeModel):
    """
    变分自编码器 (Variational Autoencoder)

    学习设计参数的潜在空间表示，支持:
    - 无监督预训练
    - 潜在空间采样
    - 设计插值

    使用示例:
    ```python
    # 创建 VAE
    config = VAEConfig(latent_dim=32, design_shape=(200, 22))
    vae = VAE(config)

    # 训练
    vae.train_model(train_loader, epochs=100)

    # 编码
    mu, logvar = vae.encode(design)
    z = vae.reparameterize(mu, logvar)

    # 解码
    design_recon = vae.decode(z)

    # 从先验采样
    z_sample = vae.sample_prior(batch_size=10)
    design_new = vae.decode(z_sample)
    ```
    """

    def __init__(self, config: Optional[VAEConfig] = None):
        config = config or VAEConfig()
        super().__init__(config)
        self.config = config

        # 保存关键参数
        self.latent_dim = config.latent_dim
        self.design_shape = config.encoder_config.design_shape

        # 创建编码器和解码器
        self.encoder = VAEEncoder(config.encoder_config)
        self.decoder = VAEDecoder(config.decoder_config)

        # 当前 epoch（用于 β 预热）
        self.current_epoch = 0

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """编码输入到潜在空间"""
        return self.encoder(x)

    def decode(self, z: Tensor) -> Tensor:
        """从潜在向量解码"""
        return self.decoder(z)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        重参数化采样

        z = μ + σ * ε, ε ~ N(0, I)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: Tensor, noise: Optional[Tensor] = None) -> Tensor:
        """
        前向传播: 编码 -> 重参数化 -> 解码

        Args:
            x: 设计参数 [B, H, W]
            noise: 随机噪声（可选，用于条件生成）

        Returns:
            recon_x: 重建的设计参数 [B, H, W]
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x

    def sample_prior(self, batch_size: int) -> Tensor:
        """从先验分布 N(0, I) 采样"""
        return torch.randn(batch_size, self.latent_dim, device=self.device)

    def generate(self, num_samples: int = 1) -> Tensor:
        """从先验采样生成新设计"""
        z = self.sample_prior(num_samples)
        return self.decode(z)

    def interpolate(
        self,
        design1: Tensor,
        design2: Tensor,
        num_steps: int = 10
    ) -> Tensor:
        """
        在潜在空间中插值

        Args:
            design1: 起始设计 [B, H, W]
            design2: 目标设计 [B, H, W]
            num_steps: 插值步数

        Returns:
            插值设计 [num_steps, H, W]
        """
        # 编码
        mu1, _ = self.encode(design1)
        mu2, _ = self.encode(design2)

        # 线性插值
        alphas = torch.linspace(0, 1, num_steps, device=self.device)
        designs = []

        for alpha in alphas:
            z = (1 - alpha) * mu1 + alpha * mu2
            design = self.decode(z)
            designs.append(design)

        return torch.cat(designs, dim=0)

    def compute_loss(
        self,
        x: Tensor,
        recon_x: Tensor,
        mu: Tensor,
        logvar: Tensor,
        epoch: int = 0
    ) -> Dict[str, Tensor]:
        """
        计算 VAE 损失

        L = L_recon + β * L_KL

        其中 β 可以动态调整（β-VAE）
        """
        # 重建损失 (MSE)
        recon_loss = F.mse_loss(recon_x, x, reduction='sum') / x.size(0)

        # KL 散度
        # KL(q(z|x) || p(z)) = -0.5 * Σ(1 + log(σ²) - μ² - σ²)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_loss / x.size(0)

        # β 预热
        beta = self._get_beta(epoch)

        # 总损失
        total_loss = self.config.recon_weight * recon_loss + beta * kl_loss

        return {
            'recon': recon_loss,
            'kl': kl_loss,
            'beta': torch.tensor(beta),
            'total': total_loss
        }

    def _get_beta(self, epoch: int) -> float:
        """获取当前的 β 值（支持预热）"""
        if epoch < self.config.beta_warmup_epochs:
            # 线性预热
            return self.config.kl_weight * (epoch + 1) / self.config.beta_warmup_epochs
        return self.config.kl_weight

    def train_model(
        self,
        train_loader,
        val_loader=None,
        epochs: int = 100,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        patience: int = 15,
        save_path: Optional[str] = None
    ) -> Dict[str, List[float]]:
        """
        训练 VAE

        Args:
            train_loader: 训练数据
            val_loader: 验证数据
            epochs: 训练轮数
            lr: 学习率
            weight_decay: 权重衰减
            patience: 早停耐心值
            save_path: 模型保存路径

        Returns:
            训练历史
        """
        self.train()
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=patience // 3
        )

        best_val_loss = float('inf')
        patience_counter = 0
        history = {'train_loss': [], 'val_loss': [], 'recon_loss': [], 'kl_loss': []}

        for epoch in range(epochs):
            self.current_epoch = epoch

            # 训练阶段
            train_loss = 0.0
            recon_total = 0.0
            kl_total = 0.0

            for batch in train_loader:
                if isinstance(batch, dict):
                    x = batch['design'].to(self.device)
                else:
                    x = batch.to(self.device) if batch.dim() == 4 else batch[0].to(self.device)

                optimizer.zero_grad()

                # 前向传播
                mu, logvar = self.encode(x)
                z = self.reparameterize(mu, logvar)
                recon_x = self.decode(z)

                # 计算损失
                losses = self.compute_loss(x, recon_x, mu, logvar, epoch)

                losses['total'].backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

                optimizer.step()

                train_loss += losses['total'].item()
                recon_total += losses['recon'].item()
                kl_total += losses['kl'].item()

            train_loss /= len(train_loader)
            recon_total /= len(train_loader)
            kl_total /= len(train_loader)

            history['train_loss'].append(train_loss)
            history['recon_loss'].append(recon_total)
            history['kl_loss'].append(kl_total)

            # 验证阶段
            if val_loader is not None:
                val_loss, val_recon, val_kl = self._validate(val_loader, epoch)
                history['val_loss'].append(val_loss)

                scheduler.step(val_loss)

                # 早停检查
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    if save_path:
                        self.save(save_path)
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch + 1}")
                        break

                if (epoch + 1) % 10 == 0:
                    beta = self._get_beta(epoch)
                    print(f"Epoch {epoch + 1}: loss={train_loss:.4f}, recon={recon_total:.4f}, "
                          f"kl={kl_total:.4f}, beta={beta:.4f}, val_loss={val_loss:.4f}")
            else:
                if (epoch + 1) % 10 == 0:
                    beta = self._get_beta(epoch)
                    print(f"Epoch {epoch + 1}: loss={train_loss:.4f}, recon={recon_total:.4f}, "
                          f"kl={kl_total:.4f}, beta={beta:.4f}")

        return history

    def _validate(self, val_loader, epoch: int) -> Tuple[float, float, float]:
        """验证"""
        self.eval()
        val_loss = 0.0
        val_recon = 0.0
        val_kl = 0.0

        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, dict):
                    x = batch['design'].to(self.device)
                else:
                    x = batch.to(self.device) if batch.dim() == 4 else batch[0].to(self.device)

                mu, logvar = self.encode(x)
                z = self.reparameterize(mu, logvar)
                recon_x = self.decode(z)

                losses = self.compute_loss(x, recon_x, mu, logvar, epoch)

                val_loss += losses['total'].item()
                val_recon += losses['recon'].item()
                val_kl += losses['kl'].item()

        self.train()
        return val_loss / len(val_loader), val_recon / len(val_loader), val_kl / len(val_loader)

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            'name': self.config.name,
            'device': str(self.device),
            'latent_dim': self.latent_dim,
            'design_shape': self.design_shape,
            'encoder_parameters': sum(p.numel() for p in self.encoder.parameters()),
            'decoder_parameters': sum(p.numel() for p in self.decoder.parameters()),
            'total_parameters': self.count_parameters()
        }


# ============================================================================
# 便捷函数
# ============================================================================

def create_vae_for_challenge(
    challenge_name: str,
    latent_dim: int = 32,
    device: str = 'auto'
) -> VAE:
    """
    为特定挑战创建 VAE

    Args:
        challenge_name: 挑战名称
        latent_dim: 潜在空间维度
        device: 计算设备

    Returns:
        配置好的 VAE
    """
    from challenges import ChallengeFactory

    # 获取挑战以确定设计形状
    challenge = ChallengeFactory.create(challenge_name)
    design_shape = challenge.spec.get_grid_shape()

    encoder_config = VAEEncoderConfig(
        design_shape=design_shape,
        latent_dim=latent_dim,
        device=device
    )

    decoder_config = VAEDecoderConfig(
        latent_dim=latent_dim,
        design_shape=design_shape,
        device=device
    )

    config = VAEConfig(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        latent_dim=latent_dim,
        device=device
    )

    return VAE(config)


# ============================================================================
# HiLAB 主引擎
# ============================================================================

class HiLABEngine(GenerativeModel):
    """
    HiLAB 混合逆向设计引擎

    整合 VAE 潜在空间学习和贝叶斯优化，实现高效的光子学逆向设计。

    工作流程:
    1. VAE 训练: 学习设计空间的低维潜在表示
    2. 代理模型训练: 在潜在空间中建立 z -> performance 的映射
    3. 贝叶斯优化: 在潜在空间中搜索满足目标性能的最优点
    4. 解码生成: 将潜在向量解码为设计参数

    参考文献:
    - Marzban et al., "HiLAB: A Hybrid Inverse-Design Framework", arXiv:2505.17491, 2025

    使用示例:
    ```python
    # 创建 HiLAB 引擎
    config = HiLABConfig(
        vae_config=VAEConfig(latent_dim=32, design_shape=(200, 22)),
        optimizer_config=BayesianOptimizerConfig(acquisition_type='ei')
    )
    engine = HiLABEngine(config)

    # 训练 VAE
    engine.train_vae(train_loader, val_loader, epochs=100)

    # 训练代理模型
    engine.train_surrogate(design_performance_pairs, epochs=50)

    # 逆向设计
    target_performance = torch.tensor([0.95, 0.8, 0.1])
    design = engine.inverse_design(target_performance, n_iterations=50)
    ```
    """

    def __init__(self, config: Optional[HiLABConfig] = None):
        config = config or HiLABConfig()
        super().__init__(config)
        self.config = config

        # 核心组件
        self.vae = VAE(config.vae_config)
        self.surrogate: Optional[nn.Module] = None

        # 贝叶斯优化器配置（延迟创建）
        self._bo_config = config.optimizer_config

        # 设计空间信息
        self.latent_dim = config.vae_config.latent_dim
        self.design_shape = config.vae_config.encoder_config.design_shape
        self.performance_dim = config.performance_dim

        # 优化历史
        self.optimization_history: List[Dict[str, Any]] = []

    def train_vae(
        self,
        train_loader,
        val_loader=None,
        epochs: int = 100,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        patience: int = 15,
        save_path: Optional[str] = None
    ) -> Dict[str, List[float]]:
        """
        训练 VAE 学习设计空间的潜在表示

        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epochs: 训练轮数
            lr: 学习率
            weight_decay: 权重衰减
            patience: 早停耐心值
            save_path: 模型保存路径

        Returns:
            训练历史
        """
        return self.vae.train_model(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            patience=patience,
            save_path=save_path
        )

    def build_surrogate(
        self,
        hidden_dims: List[int] = None,
        activation: str = 'relu',
        dropout: float = 0.1
    ) -> nn.Module:
        """
        构建潜在空间代理模型

        代理模型将潜在向量映射到性能指标:
        z -> performance

        Args:
            hidden_dims: 隐藏层维度列表
            activation: 激活函数
            dropout: Dropout 比率

        Returns:
            代理模型
        """
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        layers = []
        in_dim = self.latent_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(get_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, self.performance_dim))

        self.surrogate = nn.Sequential(*layers).to(self.device)

        return self.surrogate

    def train_surrogate(
        self,
        data_loader,
        val_loader=None,
        epochs: int = 50,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        patience: int = 10
    ) -> Dict[str, List[float]]:
        """
        训练代理模型

        使用 VAE 编码的设计数据训练代理模型，
        建立 z -> performance 的映射。

        Args:
            data_loader: 数据加载器，提供 (design, performance) 对
            val_loader: 验证数据加载器
            epochs: 训练轮数
            lr: 学习率
            weight_decay: 权重衰减
            patience: 早停耐心值

        Returns:
            训练历史
        """
        if self.surrogate is None:
            self.build_surrogate()

        self.surrogate.train()
        optimizer = torch.optim.AdamW(
            self.surrogate.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=patience // 3
        )

        history = {'train_loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            # 训练
            train_loss = 0.0
            for batch in data_loader:
                if isinstance(batch, dict):
                    designs = batch['design'].to(self.device)
                    performances = batch['performance'].to(self.device)
                elif isinstance(batch, (list, tuple)):
                    designs, performances = batch[0].to(self.device), batch[1].to(self.device)
                else:
                    continue

                # 编码到潜在空间
                with torch.no_grad():
                    mu, _ = self.vae.encode(designs)

                # 预测性能
                optimizer.zero_grad()
                pred_perf = self.surrogate(mu)

                # 损失
                loss = F.mse_loss(pred_perf, performances)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.surrogate.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(data_loader)
            history['train_loss'].append(train_loss)

            # 验证
            if val_loader is not None:
                val_loss = self._validate_surrogate(val_loader)
                history['val_loss'].append(val_loss)
                scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Surrogate early stopping at epoch {epoch + 1}")
                        break

                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch + 1}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
            else:
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch + 1}: train_loss={train_loss:.6f}")

        return history

    def _validate_surrogate(self, val_loader) -> float:
        """验证代理模型"""
        self.surrogate.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, dict):
                    designs = batch['design'].to(self.device)
                    performances = batch['performance'].to(self.device)
                elif isinstance(batch, (list, tuple)):
                    designs, performances = batch[0].to(self.device), batch[1].to(self.device)
                else:
                    continue

                mu, _ = self.vae.encode(designs)
                pred_perf = self.surrogate(mu)
                val_loss += F.mse_loss(pred_perf, performances).item()

        self.surrogate.train()
        return val_loss / len(val_loader)

    def forward(self, condition: Tensor, noise: Optional[Tensor] = None) -> Tensor:
        """
        条件生成：根据目标性能生成设计

        Args:
            condition: 目标性能 [B, performance_dim]
            noise: 随机噪声（可选）

        Returns:
            设计参数 [B, H, W]
        """
        # 执行逆向设计
        designs = []
        for i in range(condition.size(0)):
            target = condition[i:i+1]
            z = self._optimize_latent(target)
            design = self.vae.decode(z)
            designs.append(design)

        return torch.cat(designs, dim=0)

    def inverse_design(
        self,
        target_performance: Union[Tensor, np.ndarray],
        n_iterations: int = 50,
        n_initial: int = 10,
        verbose: bool = True,
        return_history: bool = False
    ) -> Union[Tensor, Tuple[Tensor, List[Dict]]]:
        """
        逆向设计：寻找满足目标性能的设计参数

        这是 HiLAB 的核心方法，通过以下步骤实现:
        1. 在 VAE 潜在空间中定义目标函数
        2. 使用贝叶斯优化搜索最优潜在向量
        3. 将最优潜在向量解码为设计参数

        Args:
            target_performance: 目标性能指标 [B, D] 或 [D]
            n_iterations: 贝叶斯优化迭代次数
            n_initial: 初始随机采样数
            verbose: 是否打印进度
            return_history: 是否返回优化历史

        Returns:
            design: 设计参数 [B, H, W]
            history: 优化历史（可选）
        """
        if self.surrogate is None:
            raise RuntimeError("Surrogate model not trained. Call train_surrogate() first.")

        # 确保目标性能是批量形式
        if isinstance(target_performance, np.ndarray):
            target_performance = torch.from_numpy(target_performance).float().to(self.device)

        if target_performance.dim() == 1:
            target_performance = target_performance.unsqueeze(0)

        batch_size = target_performance.size(0)
        designs = []
        histories = []

        for i in range(batch_size):
            target = target_performance[i:i+1]

            # 定义目标函数：负性能误差（因为 BO 最大化）
            def objective(z: np.ndarray) -> float:
                z_tensor = torch.from_numpy(z).float().to(self.device).unsqueeze(0)

                self.surrogate.eval()
                with torch.no_grad():
                    pred_perf = self.surrogate(z_tensor)

                # 负 MSE（越大越好）
                error = F.mse_loss(pred_perf, target).item()
                return -error

            # 创建贝叶斯优化器
            from optimization.solvers.bayesian import BayesianOptimizer

            bo = BayesianOptimizer(
                dim=self.latent_dim,
                bounds=self._bo_config.latent_bounds,
                kernel_type=self._bo_config.kernel_type,
                kernel_lengthscale=self._bo_config.kernel_lengthscale,
                kernel_variance=self._bo_config.kernel_variance,
                noise_variance=self._bo_config.noise_variance,
                acquisition_type=self._bo_config.acquisition_type,
                ucb_beta=self._bo_config.ucb_beta,
                xi=self._bo_config.xi,
                n_restarts=self._bo_config.n_restarts
            )

            # 执行优化
            best_z, best_neg_error = bo.optimize(
                objective=objective,
                n_iterations=n_iterations,
                n_initial=n_initial,
                verbose=verbose
            )

            # 记录历史
            iter_history = {
                'target': target.cpu().numpy(),
                'best_z': best_z,
                'best_error': -best_neg_error,
                'observations': list(zip(bo.get_observations()[0], bo.get_observations()[1]))
            }
            histories.append(iter_history)
            self.optimization_history.append(iter_history)

            # 解码
            best_z_tensor = torch.from_numpy(best_z).float().to(self.device).unsqueeze(0)
            design = self.vae.decode(best_z_tensor)
            designs.append(design)

        designs = torch.cat(designs, dim=0)

        if return_history:
            return designs, histories
        return designs

    def inverse_design_with_refinement(
        self,
        target_performance: Union[Tensor, np.ndarray],
        simulator: Callable,
        n_iterations: int = 50,
        adjoint_iterations: int = 50,
        adjoint_lr: float = 0.01,
        verbose: bool = True
    ) -> Tensor:
        """
        带伴随方法精细化的逆向设计

        在贝叶斯优化结果基础上，使用伴随方法进行精细化优化。

        Args:
            target_performance: 目标性能
            simulator: 仿真器函数
            n_iterations: 贝叶斯优化迭代次数
            adjoint_iterations: 伴随优化迭代次数
            adjoint_lr: 伴随优化学习率
            verbose: 是否打印进度

        Returns:
            精细化后的设计参数
        """
        # 先执行贝叶斯优化
        design, history = self.inverse_design(
            target_performance,
            n_iterations=n_iterations,
            verbose=verbose,
            return_history=True
        )

        if not self.config.use_adjoint_refinement:
            return design

        # 伴随方法精细化
        design = design.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([design], lr=adjoint_lr)

        target_tensor = target_performance if isinstance(target_performance, Tensor) else \
            torch.from_numpy(target_performance).float().to(self.device)

        if target_tensor.dim() == 1:
            target_tensor = target_tensor.unsqueeze(0)

        for i in range(adjoint_iterations):
            optimizer.zero_grad()

            # 通过仿真器计算性能
            pred_performance = simulator(design)

            # 计算损失
            loss = F.mse_loss(pred_performance, target_tensor)

            # 反向传播
            loss.backward()
            optimizer.step()

            # 投影到有效范围
            with torch.no_grad():
                design.data = torch.clamp(design.data, 0, 1)

            if verbose and (i + 1) % 10 == 0:
                print(f"Adjoint refinement {i + 1}/{adjoint_iterations}: loss={loss.item():.6f}")

        return design.detach()

    def sample_diverse(
        self,
        target_performance: Union[Tensor, np.ndarray],
        n_samples: int = 5,
        n_iterations: int = 30,
        diversity_weight: float = 0.1
    ) -> Tensor:
        """
        采样多样化的设计

        生成多个满足相同目标性能但具有不同结构的设计。

        Args:
            target_performance: 目标性能
            n_samples: 采样数量
            n_iterations: 每个样本的优化迭代次数
            diversity_weight: 多样性权重

        Returns:
            多样化设计集合 [n_samples, H, W]
        """
        if isinstance(target_performance, np.ndarray):
            target_performance = torch.from_numpy(target_performance).float().to(self.device)

        if target_performance.dim() == 1:
            target_performance = target_performance.unsqueeze(0)

        designs = []

        for _ in range(n_samples):
            # 从先验采样初始点
            z_init = self.vae.sample_prior(1)
            z = z_init.clone().requires_grad_(True)

            optimizer = torch.optim.Adam([z], lr=0.1)

            for _ in range(n_iterations):
                optimizer.zero_grad()

                # 预测性能
                pred_perf = self.surrogate(z)

                # 性能损失
                perf_loss = F.mse_loss(pred_perf, target_performance)

                # 多样性损失（与已有设计的差异）
                if len(designs) > 0:
                    current_design = self.vae.decode(z)
                    existing_designs = torch.cat(designs, dim=0)
                    # 计算与已有设计的相似度
                    sim = F.cosine_similarity(
                        current_design.view(1, -1),
                        existing_designs.view(len(designs), -1)
                    ).mean()
                    div_loss = sim  # 最小化相似度
                else:
                    div_loss = torch.tensor(0.0, device=self.device)

                # 总损失
                loss = perf_loss + diversity_weight * div_loss
                loss.backward()
                optimizer.step()

            # 解码最终设计
            with torch.no_grad():
                design = self.vae.decode(z.detach())
            designs.append(design)

        return torch.cat(designs, dim=0)

    def evaluate_design(
        self,
        design: Tensor,
        simulator: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        评估设计性能

        Args:
            design: 设计参数 [B, H, W]
            simulator: 仿真器函数（可选，用于精确评估）

        Returns:
            评估结果字典
        """
        results = {}

        # 使用代理模型预测性能
        if self.surrogate is not None:
            with torch.no_grad():
                mu, logvar = self.vae.encode(design)
                pred_performance = self.surrogate(mu)
            results['predicted_performance'] = pred_performance.cpu().numpy()

        # 使用仿真器精确评估
        if simulator is not None:
            actual_performance = simulator(design)
            results['actual_performance'] = actual_performance.cpu().numpy() if \
                isinstance(actual_performance, Tensor) else actual_performance

        # 计算重建误差
        with torch.no_grad():
            mu, logvar = self.vae.encode(design)
            z = self.vae.reparameterize(mu, logvar)
            recon_design = self.vae.decode(z)
            recon_error = F.mse_loss(recon_design, design).item()
        results['reconstruction_error'] = recon_error

        return results

    def get_latent_embedding(self, design: Tensor) -> Tuple[Tensor, Tensor]:
        """
        获取设计的潜在空间嵌入

        Args:
            design: 设计参数 [B, H, W]

        Returns:
            mu: 潜在空间均值 [B, latent_dim]
            logvar: 潜在空间对数方差 [B, latent_dim]
        """
        with torch.no_grad():
            mu, logvar = self.vae.encode(design)
        return mu, logvar

    def interpolate_designs(
        self,
        design1: Tensor,
        design2: Tensor,
        num_steps: int = 10
    ) -> Tensor:
        """
        在潜在空间中插值两个设计

        Args:
            design1: 起始设计
            design2: 目标设计
            num_steps: 插值步数

        Returns:
            插值设计序列
        """
        return self.vae.interpolate(design1, design2, num_steps)

    def save_engine(self, path: str):
        """保存整个引擎"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # 保存 VAE
        self.vae.save(path / 'vae.pt')

        # 保存代理模型
        if self.surrogate is not None:
            torch.save(self.surrogate.state_dict(), path / 'surrogate.pt')

        # 保存配置
        import json
        config_dict = {
            'latent_dim': self.latent_dim,
            'design_shape': list(self.design_shape),
            'performance_dim': self.performance_dim
        }
        with open(path / 'config.json', 'w') as f:
            json.dump(config_dict, f)

    def load_engine(self, path: str):
        """加载引擎"""
        path = Path(path)

        # 加载 VAE
        self.vae.load(path / 'vae.pt')

        # 加载代理模型
        if (path / 'surrogate.pt').exists():
            if self.surrogate is None:
                self.build_surrogate()
            self.surrogate.load_state_dict(torch.load(path / 'surrogate.pt', map_location=self.device))

    def get_model_info(self) -> Dict[str, Any]:
        """获取引擎信息"""
        return {
            'name': self.config.name,
            'device': str(self.device),
            'latent_dim': self.latent_dim,
            'design_shape': self.design_shape,
            'performance_dim': self.performance_dim,
            'vae_parameters': self.vae.count_parameters(),
            'surrogate_parameters': sum(p.numel() for p in self.surrogate.parameters()) if self.surrogate else 0,
            'use_adjoint_refinement': self.config.use_adjoint_refinement
        }


# ============================================================================
# 便捷工厂函数
# ============================================================================

def create_hilab_for_challenge(
    challenge_name: str,
    latent_dim: int = 32,
    performance_dim: int = 3,
    device: str = 'auto'
) -> HiLABEngine:
    """
    为特定挑战创建 HiLAB 引擎

    Args:
        challenge_name: 挑战名称
        latent_dim: 潜在空间维度
        performance_dim: 性能指标维度
        device: 计算设备

    Returns:
        配置好的 HiLAB 引擎
    """
    from challenges import ChallengeFactory

    # 获取挑战以确定设计形状
    challenge = ChallengeFactory.create(challenge_name)
    design_shape = challenge.spec.get_grid_shape()

    # 配置 VAE
    encoder_config = VAEEncoderConfig(
        design_shape=design_shape,
        latent_dim=latent_dim
    )

    decoder_config = VAEDecoderConfig(
        latent_dim=latent_dim,
        design_shape=design_shape
    )

    vae_config = VAEConfig(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        latent_dim=latent_dim
    )

    # 配置贝叶斯优化器
    optimizer_config = BayesianOptimizerConfig()

    # 完整配置
    config = HiLABConfig(
        vae_config=vae_config,
        optimizer_config=optimizer_config,
        performance_dim=performance_dim,
        design_shape=design_shape
    )

    return HiLABEngine(config)

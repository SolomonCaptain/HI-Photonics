"""
Conditional Generative Adversarial Network (CGAN)

条件生成对抗网络用于光子学逆向设计。
从性能目标条件生成多样化的设计参数。

核心组件:
- ConditionalGenerator: 条件生成器
- ConditionalDiscriminator: 条件判别器
- CGAN: 标准 CGAN
- WGAN_GP: Wasserstein GAN + 梯度惩罚

参考文献:
- Mirza & Osindero, "Conditional Generative Adversarial Nets", 2014
- Arjovsky et al., "Wasserstein GAN", 2017
- Gulrajani et al., "Improved Training of Wasserstein GANs", 2017
"""

from typing import Dict, Optional, Tuple, List, Union, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Adam, RMSprop
from torch.optim.lr_scheduler import LambdaLR
import numpy as np

from models.base import BaseModel, ModelConfig, GenerativeModel


# ============================================================================
# 配置类
# ============================================================================

@dataclass
class GeneratorConfig(ModelConfig):
    """生成器配置"""
    name: str = "conditional_generator"
    
    # 输入维度
    latent_dim: int = 128           # 噪声向量维度
    condition_dim: int = 3          # 条件向量维度（性能指标）
    
    # 输出维度
    design_shape: Tuple[int, int] = (200, 22)  # 设计形状 (H, W)
    
    # 网络架构
    hidden_dims: List[int] = field(default_factory=lambda: [256, 512, 1024])
    hidden_channels: List[int] = field(default_factory=lambda: [512, 256, 128, 64])
    kernel_size: int = 4
    stride: int = 2
    padding: int = 1
    
    # 正则化
    dropout_rate: float = 0.0
    batch_norm: bool = True
    spectral_norm: bool = False
    
    # 激活函数
    activation: str = "leaky_relu"
    output_activation: str = "tanh"  # 'tanh', 'sigmoid', 'none'


@dataclass
class DiscriminatorConfig(ModelConfig):
    """判别器配置"""
    name: str = "conditional_discriminator"
    
    # 输入维度
    design_shape: Tuple[int, int] = (200, 22)
    condition_dim: int = 3
    
    # 网络架构
    hidden_channels: List[int] = field(default_factory=lambda: [64, 128, 256, 512])
    hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 1])
    kernel_size: int = 4
    stride: int = 2
    padding: int = 1
    
    # 正则化
    dropout_rate: float = 0.0
    batch_norm: bool = True
    spectral_norm: bool = True  # 谱归一化稳定训练
    
    # 激活函数
    activation: str = "leaky_relu"
    
    # 条件融合方式
    condition_fusion: str = "concat"  # 'concat', 'film', 'attention'


@dataclass
class CGANConfig(ModelConfig):
    """CGAN 完整配置"""
    name: str = "cgan"
    
    # 子网络配置
    generator_config: GeneratorConfig = field(default_factory=GeneratorConfig)
    discriminator_config: DiscriminatorConfig = field(default_factory=DiscriminatorConfig)
    
    # 训练配置
    n_critic: int = 5  # 判别器更新次数（每更新一次生成器）
    lambda_gp: float = 10.0  # 梯度惩罚系数
    
    # 优化器配置
    g_lr: float = 1e-4
    d_lr: float = 4e-4
    beta1: float = 0.0
    beta2: float = 0.9
    
    # GAN 类型
    gan_type: str = "wgan-gp"  # 'gan', 'lsgan', 'wgan', 'wgan-gp'


# ============================================================================
# 工具函数
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


def apply_sn(module: nn.Module, use_sn: bool) -> nn.Module:
    """应用谱归一化"""
    if use_sn and hasattr(module, 'weight'):
        return nn.utils.spectral_norm(module)
    return module


# ============================================================================
# 条件生成器
# ============================================================================

class ConditionalGenerator(GenerativeModel):
    """
    条件生成器
    
    从噪声向量和条件向量生成设计参数。
    
    架构:
        噪声 z [B, latent_dim]
        条件 c [B, condition_dim]
          ↓
        Concat [z, c]
          ↓
        FC Layers (扩展维度)
          ↓
        Reshape [B, C, H', W']
          ↓
        Transposed Conv Blocks
          ↓
        设计 [B, 1, H, W]
    """
    
    def __init__(self, config: Optional[GeneratorConfig] = None):
        config = config or GeneratorConfig()
        super().__init__(config)
        self.config = config
        
        self.latent_dim = config.latent_dim
        self.condition_dim = config.condition_dim
        self.design_shape = config.design_shape
        
        # 构建网络
        self.fc = self._build_fc()
        self.decoder = self._build_decoder()
        
        # 初始化权重
        self._init_weights()
    
    def _build_fc(self) -> nn.Module:
        """构建全连接层"""
        layers = []
        in_dim = self.latent_dim + self.condition_dim
        
        for i, out_dim in enumerate(self.config.hidden_dims):
            linear = nn.Linear(in_dim, out_dim)
            linear = apply_sn(linear, self.config.spectral_norm)
            layers.append(linear)
            
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
        layers = []
        config = self.config
        
        # 计算初始空间维度
        h, w = self.design_shape
        num_upsamples = len(config.hidden_channels) - 1
        self.init_h = max(h // (2 ** num_upsamples), 1)
        self.init_w = max(w // (2 ** num_upsamples), 1)
        
        # 计算初始通道数
        init_channels = self.fc_output_dim // (self.init_h * self.init_w)
        init_channels = max(init_channels, config.hidden_channels[0])
        
        # 调整全连接输出
        self.fc_adjust = nn.Linear(
            self.fc_output_dim,
            init_channels * self.init_h * self.init_w
        )
        
        # 构建转置卷积层
        in_channels = init_channels
        for i, out_channels in enumerate(config.hidden_channels):
            # 转置卷积
            conv = nn.ConvTranspose2d(
                in_channels, out_channels,
                kernel_size=config.kernel_size,
                stride=config.stride if i < len(config.hidden_channels) - 1 else 1,
                padding=config.padding,
                output_padding=1 if i < len(config.hidden_channels) - 1 else 0
            )
            conv = apply_sn(conv, config.spectral_norm)
            layers.append(conv)
            
            # 归一化
            if config.batch_norm and i < len(config.hidden_channels) - 1:
                layers.append(nn.BatchNorm2d(out_channels))
            
            # 激活函数
            if i < len(config.hidden_channels) - 1:
                layers.append(get_activation(config.activation))
            
            in_channels = out_channels
        
        # 输出层
        output_conv = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)
        output_conv = apply_sn(output_conv, config.spectral_norm)
        layers.append(output_conv)
        
        # 输出激活函数
        layers.append(get_activation(config.output_activation))
        
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.ConvTranspose2d, nn.Conv2d)):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(
        self,
        condition: Tensor,
        noise: Optional[Tensor] = None,
        **kwargs
    ) -> Tensor:
        """
        条件生成
        
        Args:
            condition: 条件向量 [B, condition_dim]
            noise: 噪声向量 [B, latent_dim]（可选，自动生成）
            
        Returns:
            设计参数 [B, H, W]
        """
        batch_size = condition.size(0)
        
        # 生成噪声
        if noise is None:
            noise = torch.randn(batch_size, self.latent_dim, device=condition.device)
        
        # 拼接噪声和条件
        x = torch.cat([noise, condition], dim=1)
        
        # 全连接层
        x = self.fc(x)
        x = self.fc_adjust(x)
        
        # 重塑为特征图
        x = x.view(batch_size, -1, self.init_h, self.init_w)
        
        # 解码
        design = self.decoder(x)
        
        # 调整到目标尺寸
        design = F.interpolate(design, size=self.design_shape, mode='bilinear', align_corners=False)
        
        # 移除通道维度
        design = design.squeeze(1)
        
        return design
    
    def sample(
        self,
        condition: Tensor,
        num_samples: int = 1,
        **kwargs
    ) -> Tensor:
        """
        从条件采样多个设计
        
        Args:
            condition: 条件向量 [B, condition_dim]
            num_samples: 每个条件生成的设计数量
            
        Returns:
            设计参数 [B * num_samples, H, W]
        """
        batch_size = condition.size(0)
        
        # 扩展条件
        condition_expanded = condition.repeat_interleave(num_samples, dim=0)
        
        # 生成噪声
        noise = torch.randn(batch_size * num_samples, self.latent_dim, device=condition.device)
        
        return self.forward(condition_expanded, noise)
    
    def encode_condition(self, condition: Tensor) -> Tensor:
        """编码条件向量（用于条件注入）"""
        return condition


# ============================================================================
# 条件判别器
# ============================================================================

class ConditionalDiscriminator(BaseModel):
    """
    条件判别器
    
    判断设计的真实性，同时考虑条件匹配。
    
    架构:
        设计 d [B, 1, H, W]
          ↓
        Conv Blocks (下采样)
          ↓
        Flatten
          ↓
        条件 c [B, condition_dim]
          ↓
        Concat (特征 + 条件)
          ↓
        FC Layers
          ↓
        真实性分数 [B, 1]
    """
    
    def __init__(self, config: Optional[DiscriminatorConfig] = None):
        config = config or DiscriminatorConfig()
        super().__init__(config)
        self.config = config
        
        self.design_shape = config.design_shape
        self.condition_dim = config.condition_dim
        
        # 构建网络
        self.encoder = self._build_encoder()
        self.fc = self._build_fc()
        
        # 条件融合模块
        if config.condition_fusion == "film":
            self.condition_film = self._build_film()
        elif config.condition_fusion == "attention":
            self.condition_attention = self._build_attention()
        
        # 初始化权重
        self._init_weights()
    
    def _build_encoder(self) -> nn.Module:
        """构建卷积编码器"""
        layers = []
        config = self.config
        in_channels = 1
        
        for i, out_channels in enumerate(config.hidden_channels):
            # 卷积层
            conv = nn.Conv2d(
                in_channels, out_channels,
                kernel_size=config.kernel_size,
                stride=config.stride,
                padding=config.padding
            )
            conv = apply_sn(conv, config.spectral_norm)
            layers.append(conv)
            
            # 归一化
            if config.batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            
            # 激活函数
            layers.append(get_activation(config.activation))
            
            # Dropout
            if config.dropout_rate > 0:
                layers.append(nn.Dropout2d(config.dropout_rate))
            
            in_channels = out_channels
        
        return nn.Sequential(*layers)
    
    def _build_fc(self) -> nn.Module:
        """构建全连接层"""
        layers = []
        
        # 计算编码器输出维度
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, *self.design_shape)
            dummy_output = self.encoder(dummy_input)
            self.encoder_output_dim = dummy_output.view(1, -1).size(1)
        
        # 条件融合后的输入维度
        in_dim = self.encoder_output_dim + self.condition_dim
        
        for i, out_dim in enumerate(config.hidden_dims if (config := self.config) else [512, 256, 1]):
            linear = nn.Linear(in_dim, out_dim)
            linear = apply_sn(linear, self.config.spectral_norm)
            layers.append(linear)
            
            if i < len(self.config.hidden_dims) - 1:
                if self.config.batch_norm:
                    layers.append(nn.BatchNorm1d(out_dim))
                layers.append(get_activation(self.config.activation))
                if self.config.dropout_rate > 0:
                    layers.append(nn.Dropout(self.config.dropout_rate))
            
            in_dim = out_dim
        
        return nn.Sequential(*layers)
    
    def _build_film(self) -> nn.Module:
        """构建 FiLM 条件调制层"""
        # FiLM: Feature-wise Linear Modulation
        return nn.Sequential(
            nn.Linear(self.condition_dim, self.config.hidden_channels[-1] * 2),
            nn.LeakyReLU(0.2)
        )
    
    def _build_attention(self) -> nn.Module:
        """构建注意力条件融合"""
        return nn.MultiheadAttention(
            embed_dim=self.config.hidden_channels[-1],
            num_heads=4,
            batch_first=True
        )
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(
        self,
        design: Tensor,
        condition: Tensor,
        **kwargs
    ) -> Tensor:
        """
        判断设计的真实性
        
        Args:
            design: 设计参数 [B, H, W] 或 [B, 1, H, W]
            condition: 条件向量 [B, condition_dim]
            
        Returns:
            真实性分数 [B, 1]
        """
        # 确保设计维度正确
        if design.dim() == 3:
            design = design.unsqueeze(1)
        
        # 编码设计
        features = self.encoder(design)
        
        # 条件融合
        if self.config.condition_fusion == "film":
            features = self._apply_film(features, condition)
        elif self.config.condition_fusion == "attention":
            features = self._apply_attention(features, condition)
        
        # 展平
        features = features.view(features.size(0), -1)
        
        # 拼接条件
        x = torch.cat([features, condition], dim=1)
        
        # 全连接层
        output = self.fc(x)
        
        return output
    
    def _apply_film(self, features: Tensor, condition: Tensor) -> Tensor:
        """应用 FiLM 调制"""
        film_params = self.condition_film(condition)
        gamma, beta = film_params.chunk(2, dim=1)
        
        # 重塑为空间维度
        gamma = gamma.view(features.size(0), -1, 1, 1)
        beta = beta.view(features.size(0), -1, 1, 1)
        
        return gamma * features + beta
    
    def _apply_attention(self, features: Tensor, condition: Tensor) -> Tensor:
        """应用注意力融合"""
        # 将条件转换为查询向量
        condition = condition.unsqueeze(1)
        
        # 将特征转换为键值对
        b, c, h, w = features.shape
        features_flat = features.view(b, c, h * w).permute(0, 2, 1)
        
        # 注意力计算
        attended, _ = self.condition_attention(condition, features_flat, features_flat)
        
        return features


# ============================================================================
# CGAN 完整模型
# ============================================================================

class CGAN(BaseModel):
    """
    条件生成对抗网络
    
    组合生成器和判别器，实现端到端训练。
    
    训练策略:
    1. 交替训练判别器和生成器
    2. 支持多种 GAN 变体: GAN, LSGAN, WGAN, WGAN-GP
    3. 梯度惩罚稳定训练
    
    使用示例:
    ```python
    # 创建 CGAN
    config = CGANConfig()
    cgan = CGAN(config)
    
    # 训练
    cgan.train(dataloader, epochs=100)
    
    # 生成设计
    target_perf = torch.tensor([[0.85, 0.1, 0.05]])
    designs = cgan.generate(target_perf, num_samples=10)
    ```
    """
    
    def __init__(self, config: Optional[CGANConfig] = None):
        config = config or CGANConfig()
        super().__init__(config)
        self.config = config
        
        # 创建生成器和判别器
        self.generator = ConditionalGenerator(config.generator_config)
        self.discriminator = ConditionalDiscriminator(config.discriminator_config)
        
        # 确保配置一致
        assert config.generator_config.design_shape == config.discriminator_config.design_shape
        assert config.generator_config.condition_dim == config.discriminator_config.condition_dim
        
        self.design_shape = config.generator_config.design_shape
        self.condition_dim = config.generator_config.condition_dim
        self.latent_dim = config.generator_config.latent_dim
        
        # 训练状态
        self.g_trained = False
        self.d_trained = False
    
    def generate(
        self,
        condition: Tensor,
        num_samples: int = 1,
        noise: Optional[Tensor] = None
    ) -> Tensor:
        """
        从条件生成设计
        
        Args:
            condition: 条件向量 [B, condition_dim]
            num_samples: 每个条件生成的设计数量
            noise: 自定义噪声（可选）
            
        Returns:
            设计参数 [B * num_samples, H, W]
        """
        self.generator.eval()
        with torch.no_grad():
            if num_samples == 1:
                return self.generator(condition, noise)
            else:
                return self.generator.sample(condition, num_samples)
    
    def discriminate(
        self,
        design: Tensor,
        condition: Tensor
    ) -> Tensor:
        """
        判断设计的真实性
        
        Args:
            design: 设计参数
            condition: 条件向量
            
        Returns:
            真实性分数
        """
        return self.discriminator(design, condition)
    
    def train_model(
        self,
        dataloader,
        val_dataloader=None,
        epochs: int = 100,
        save_path: Optional[str] = None,
        log_interval: int = 10,
        **kwargs
    ) -> Dict[str, List[float]]:
        """
        训练 CGAN
        
        Args:
            dataloader: 训练数据加载器
            val_dataloader: 验证数据加载器
            epochs: 训练轮数
            save_path: 模型保存路径
            log_interval: 日志输出间隔
            
        Returns:
            训练历史
        """
        # 初始化优化器
        g_optimizer = Adam(
            self.generator.parameters(),
            lr=self.config.g_lr,
            betas=(self.config.beta1, self.config.beta2)
        )
        d_optimizer = Adam(
            self.discriminator.parameters(),
            lr=self.config.d_lr,
            betas=(self.config.beta1, self.config.beta2)
        )
        
        # 学习率调度
        scheduler = LambdaLR(
            [g_optimizer, d_optimizer],
            lr_lambda=lambda epoch: 1.0 - epoch / epochs
        )
        
        history = {'g_loss': [], 'd_loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            g_loss_epoch = 0.0
            d_loss_epoch = 0.0
            n_batches = 0
            
            for batch in dataloader:
                real_designs = batch['design'].to(self.device)
                conditions = batch['performance'].to(self.device)
                batch_size = real_designs.size(0)
                
                # ---------------------
                # 训练判别器
                # ---------------------
                for _ in range(self.config.n_critic):
                    d_optimizer.zero_grad()
                    
                    # 真实设计
                    real_validity = self.discriminator(real_designs, conditions)
                    
                    # 生成假设计
                    noise = torch.randn(batch_size, self.latent_dim, device=self.device)
                    fake_designs = self.generator(conditions, noise)
                    fake_validity = self.discriminator(fake_designs.detach(), conditions)
                    
                    # 计算判别器损失
                    d_loss = self._compute_d_loss(
                        real_validity, fake_validity,
                        real_designs, fake_designs,
                        conditions
                    )
                    
                    d_loss.backward()
                    d_optimizer.step()
                
                # ---------------------
                # 训练生成器
                # ---------------------
                g_optimizer.zero_grad()
                
                noise = torch.randn(batch_size, self.latent_dim, device=self.device)
                fake_designs = self.generator(conditions, noise)
                fake_validity = self.discriminator(fake_designs, conditions)
                
                # 计算生成器损失
                g_loss = self._compute_g_loss(fake_validity, fake_designs, conditions)
                
                g_loss.backward()
                g_optimizer.step()
                
                g_loss_epoch += g_loss.item()
                d_loss_epoch += d_loss.item()
                n_batches += 1
            
            # 更新学习率
            scheduler.step()
            
            # 记录历史
            history['g_loss'].append(g_loss_epoch / n_batches)
            history['d_loss'].append(d_loss_epoch / n_batches)
            
            # 验证
            if val_dataloader is not None:
                val_loss = self._validate(val_dataloader)
                history['val_loss'].append(val_loss)
            
            # 日志输出
            if (epoch + 1) % log_interval == 0:
                msg = f"Epoch [{epoch + 1}/{epochs}] "
                msg += f"G_loss: {history['g_loss'][-1]:.4f} "
                msg += f"D_loss: {history['d_loss'][-1]:.4f}"
                if val_dataloader is not None:
                    msg += f" Val_loss: {val_loss:.4f}"
                print(msg)
            
            # 保存模型
            if save_path and (epoch + 1) % 50 == 0:
                self.save(f"{save_path}_epoch{epoch + 1}.pth")
        
        self.g_trained = True
        self.d_trained = True
        
        if save_path:
            self.save(f"{save_path}_final.pth")
        
        return history
    
    def _compute_d_loss(
        self,
        real_validity: Tensor,
        fake_validity: Tensor,
        real_designs: Tensor,
        fake_designs: Tensor,
        conditions: Tensor
    ) -> Tensor:
        """计算判别器损失"""
        gan_type = self.config.gan_type
        
        if gan_type == 'gan':
            # 标准 GAN 损失
            real_loss = F.binary_cross_entropy_with_logits(
                real_validity, torch.ones_like(real_validity)
            )
            fake_loss = F.binary_cross_entropy_with_logits(
                fake_validity, torch.zeros_like(fake_validity)
            )
            d_loss = real_loss + fake_loss
            
        elif gan_type == 'lsgan':
            # LSGAN 损失
            real_loss = F.mse_loss(real_validity, torch.ones_like(real_validity))
            fake_loss = F.mse_loss(fake_validity, torch.zeros_like(fake_validity))
            d_loss = real_loss + fake_loss
            
        elif gan_type in ['wgan', 'wgan-gp']:
            # Wasserstein 损失
            d_loss = -real_validity.mean() + fake_validity.mean()
            
            # 梯度惩罚
            if gan_type == 'wgan-gp':
                gp = self._compute_gradient_penalty(real_designs, fake_designs, conditions)
                d_loss += self.config.lambda_gp * gp
        else:
            raise ValueError(f"Unknown GAN type: {gan_type}")
        
        return d_loss
    
    def _compute_g_loss(
        self,
        fake_validity: Tensor,
        fake_designs: Tensor,
        conditions: Tensor
    ) -> Tensor:
        """计算生成器损失"""
        gan_type = self.config.gan_type
        
        if gan_type == 'gan':
            g_loss = F.binary_cross_entropy_with_logits(
                fake_validity, torch.ones_like(fake_validity)
            )
        elif gan_type == 'lsgan':
            g_loss = F.mse_loss(fake_validity, torch.ones_like(fake_validity))
        elif gan_type in ['wgan', 'wgan-gp']:
            g_loss = -fake_validity.mean()
        else:
            raise ValueError(f"Unknown GAN type: {gan_type}")
        
        return g_loss
    
    def _compute_gradient_penalty(
        self,
        real_designs: Tensor,
        fake_designs: Tensor,
        conditions: Tensor
    ) -> Tensor:
        """计算梯度惩罚"""
        batch_size = real_designs.size(0)
        
        # 随机插值
        alpha = torch.rand(batch_size, 1, 1, device=self.device)
        interpolates = alpha * real_designs.unsqueeze(1) + (1 - alpha) * fake_designs.unsqueeze(1)
        interpolates = interpolates.squeeze(1)
        interpolates.requires_grad_(True)
        
        # 计算判别器输出
        disc_interpolates = self.discriminator(interpolates, conditions)
        
        # 计算梯度
        gradients = torch.autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(disc_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        # 计算梯度惩罚
        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        gradient_penalty = ((gradient_norm - 1) ** 2).mean()
        
        return gradient_penalty
    
    def _validate(self, val_dataloader) -> float:
        """验证模型"""
        self.generator.eval()
        self.discriminator.eval()
        
        total_loss = 0.0
        n_batches = 0
        
        with torch.no_grad():
            for batch in val_dataloader:
                conditions = batch['performance'].to(self.device)
                real_designs = batch['design'].to(self.device)
                
                # 生成设计
                fake_designs = self.generator(conditions)
                
                # 计算重建损失
                loss = F.mse_loss(fake_designs, real_designs)
                total_loss += loss.item()
                n_batches += 1
        
        self.generator.train()
        self.discriminator.train()
        
        return total_loss / n_batches if n_batches > 0 else 0.0
    
    def save(self, path: Union[str, Path], **kwargs) -> None:
        """保存模型"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'config': self.config.__dict__,
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'g_trained': self.g_trained,
            'd_trained': self.d_trained
        }
        
        torch.save(checkpoint, path)
    
    def load(self, path: Union[str, Path], **kwargs) -> None:
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.generator.load_state_dict(checkpoint['generator'])
        self.discriminator.load_state_dict(checkpoint['discriminator'])
        self.g_trained = checkpoint.get('g_trained', False)
        self.d_trained = checkpoint.get('d_trained', False)
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            'name': self.config.name,
            'device': str(self.device),
            'design_shape': self.design_shape,
            'condition_dim': self.condition_dim,
            'latent_dim': self.latent_dim,
            'generator_parameters': self.generator.count_parameters(),
            'discriminator_parameters': self.discriminator.count_parameters(),
            'gan_type': self.config.gan_type,
            'trained': self.g_trained and self.d_trained
        }


# ============================================================================
# WGAN-GP 便捷类
# ============================================================================

class WGAN_GP(CGAN):
    """
    Wasserstein GAN with Gradient Penalty
    
    WGAN-GP 的便捷封装，使用更稳定的训练配置。
    """
    
    def __init__(
        self,
        design_shape: Tuple[int, int] = (200, 22),
        condition_dim: int = 3,
        latent_dim: int = 128,
        **kwargs
    ):
        # 配置 WGAN-GP
        gen_config = GeneratorConfig(
            design_shape=design_shape,
            condition_dim=condition_dim,
            latent_dim=latent_dim,
            output_activation='tanh',  # WGAN 使用 tanh
            **{k: v for k, v in kwargs.items() if k in GeneratorConfig.__dataclass_fields__}
        )
        
        disc_config = DiscriminatorConfig(
            design_shape=design_shape,
            condition_dim=condition_dim,
            spectral_norm=False,  # WGAN-GP 不需要谱归一化
            **{k: v for k, v in kwargs.items() if k in DiscriminatorConfig.__dataclass_fields__}
        )
        
        config = CGANConfig(
            generator_config=gen_config,
            discriminator_config=disc_config,
            gan_type='wgan-gp',
            n_critic=5,
            lambda_gp=10.0,
            g_lr=1e-4,
            d_lr=4e-4
        )
        
        super().__init__(config)


# ============================================================================
# 便捷函数
# ============================================================================

def create_cgan_for_challenge(
    challenge_name: str,
    condition_dim: int = 3,
    latent_dim: int = 128,
    gan_type: str = 'wgan-gp',
    device: str = 'auto'
) -> CGAN:
    """
    为特定挑战创建 CGAN
    
    Args:
        challenge_name: 挑战名称
        condition_dim: 条件维度
        latent_dim: 潜在空间维度
        gan_type: GAN 类型
        device: 计算设备
        
    Returns:
        配置好的 CGAN
    """
    from challenges import ChallengeFactory
    
    # 获取挑战以确定设计形状
    challenge = ChallengeFactory.create(challenge_name)
    design_shape = challenge.spec.get_grid_shape()
    
    # 创建配置
    gen_config = GeneratorConfig(
        design_shape=design_shape,
        condition_dim=condition_dim,
        latent_dim=latent_dim,
        device=device
    )
    
    disc_config = DiscriminatorConfig(
        design_shape=design_shape,
        condition_dim=condition_dim,
        device=device
    )
    
    config = CGANConfig(
        generator_config=gen_config,
        discriminator_config=disc_config,
        gan_type=gan_type,
        device=device
    )
    
    return CGAN(config)

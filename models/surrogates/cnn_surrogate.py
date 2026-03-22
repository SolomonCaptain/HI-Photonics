"""
CNN Surrogate Model - 卷积神经网络代理模型

用于光子学器件的快速性能预测。
输入: 设计图案 (2D/3D permittivity field)
输出: 性能指标 (transmission, reflection, bandwidth, etc.)

架构特点:
1. 多尺度特征提取 (ResNet-style)
2. 注意力机制增强关键区域感知
3. 不确定性量化 (可选)
4. 支持多种输出格式

参考文献:
- Peurifoy et al. (2018). "Nanophotonic particle simulation and inverse design"
- Nadell et al. (2019). "Deep learning for nanophotonics"
"""

from typing import Dict, Optional, Tuple, List, Union, Any
from dataclasses import dataclass, field
from pathlib import Path
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.base import BaseModel, ModelConfig, SurrogateModel


# ============================================================================
# 配置类
# ============================================================================

@dataclass
class CNNSurrogateConfig(ModelConfig):
    """CNN代理模型配置"""
    name: str = "cnn_surrogate"
    
    # 输入维度
    input_channels: int = 1              # 输入通道数 (1=灰度, 3=RGB或多波长)
    input_height: int = 200              # 输入高度
    input_width: int = 200               # 输入宽度
    
    # 输出维度
    output_dim: int = 3                  # 性能指标数量 (e.g., transmission, bandwidth, crosstalk)
    output_names: List[str] = field(default_factory=lambda: ['transmission', 'bandwidth', 'crosstalk'])
    
    # 编码器架构
    encoder_type: str = "resnet"         # 'resnet', 'unet', 'attention'
    base_channels: int = 64              # 基础通道数
    num_blocks: int = 4                  # 残差块数量
    block_layers: int = 2                # 每个块的层数
    
    # 注意力机制
    use_attention: bool = True           # 是否使用注意力
    attention_reduction: int = 8         # 注意力降维比例
    
    # 不确定性量化
    predict_uncertainty: bool = True     # 是否预测不确定性
    uncertainty_type: str = "heteroscedastic"  # 'heteroscedastic', 'ensemble'
    
    # 正则化
    dropout_rate: float = 0.1
    batch_norm: bool = True
    spectral_norm: bool = False          # 谱归一化
    
    # 激活函数
    activation: str = "relu"             # 'relu', 'leaky_relu', 'gelu', 'silu'
    
    # 池化
    pooling_type: str = "adaptive"       # 'adaptive', 'global_avg', 'global_max'
    
    # 多尺度
    use_multiscale: bool = True          # 是否使用多尺度特征


# ============================================================================
# 基础模块
# ============================================================================

class ConvBlock(nn.Module):
    """卷积块: Conv -> Norm -> Activation -> Dropout"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        groups: int = 1,
        use_batch_norm: bool = True,
        activation: str = "relu",
        dropout_rate: float = 0.0,
        spectral_norm: bool = False,
    ):
        super().__init__()
        
        # 卷积层
        conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, groups=groups, bias=not use_batch_norm
        )
        if spectral_norm:
            conv = nn.utils.spectral_norm(conv)
        
        self.conv = conv
        
        # 归一化
        self.norm = nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity()
        
        # 激活函数
        self.activation = self._get_activation(activation)
        
        # Dropout
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()
    
    def _get_activation(self, name: str) -> nn.Module:
        activations = {
            'relu': nn.ReLU(inplace=True),
            'leaky_relu': nn.LeakyReLU(0.2, inplace=True),
            'gelu': nn.GELU(),
            'silu': nn.SiLU(inplace=True),
        }
        return activations.get(name, nn.ReLU(inplace=True))
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class ResidualBlock(nn.Module):
    """残差块: 支持 BasicBlock 和 Bottleneck"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expansion: int = 1,
        use_batch_norm: bool = True,
        activation: str = "relu",
        dropout_rate: float = 0.0,
        spectral_norm: bool = False,
    ):
        super().__init__()
        
        self.expansion = expansion
        mid_channels = out_channels // expansion if expansion > 1 else out_channels
        
        # 主路径
        self.conv1 = ConvBlock(
            in_channels, mid_channels, 3, stride, 1,
            use_batch_norm=use_batch_norm, activation=activation,
            dropout_rate=dropout_rate, spectral_norm=spectral_norm
        )
        
        self.conv2 = ConvBlock(
            mid_channels, out_channels, 3, 1, 1,
            use_batch_norm=use_batch_norm, activation='none',  # 最后一层无激活
            dropout_rate=0.0, spectral_norm=spectral_norm
        )
        
        # 残差连接
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity()
            )
        
        self.activation = nn.ReLU(inplace=True) if activation == 'relu' else nn.GELU()
    
    def forward(self, x: Tensor) -> Tensor:
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.conv2(out)
        
        out = out + identity
        out = self.activation(out)
        
        return out


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation 注意力模块"""
    
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: Tensor) -> Tensor:
        b, c, _, _ = x.shape
        # Squeeze
        y = self.squeeze(x).view(b, c)
        # Excitation
        y = self.excitation(y).view(b, c, 1, 1)
        # Scale
        return x * y


class SelfAttention(nn.Module):
    """自注意力模块 (用于捕获全局依赖)"""
    
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        
        self.channels = channels
        reduced_channels = channels // reduction
        
        self.query = nn.Conv2d(channels, reduced_channels, 1)
        self.key = nn.Conv2d(channels, reduced_channels, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
    
    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        
        # Query, Key, Value
        q = self.query(x).view(b, -1, h * w).permute(0, 2, 1)  # [B, HW, C']
        k = self.key(x).view(b, -1, h * w)                     # [B, C', HW]
        v = self.value(x).view(b, -1, h * w)                   # [B, C, HW]
        
        # Attention weights
        attention = torch.bmm(q, k)                            # [B, HW, HW]
        attention = F.softmax(attention, dim=-1)
        
        # Apply attention
        out = torch.bmm(v, attention.permute(0, 2, 1))         # [B, C, HW]
        out = out.view(b, c, h, w)
        
        # Residual connection with learnable weight
        return self.gamma * out + x


# ============================================================================
# 编码器
# ============================================================================

class ResNetEncoder(nn.Module):
    """ResNet风格编码器"""
    
    def __init__(
        self,
        in_channels: int,
        base_channels: int = 64,
        num_blocks: int = 4,
        use_attention: bool = True,
        attention_reduction: int = 8,
        use_batch_norm: bool = True,
        activation: str = "relu",
        dropout_rate: float = 0.0,
        spectral_norm: bool = False,
    ):
        super().__init__()
        
        self.use_attention = use_attention
        
        # 初始卷积
        self.stem = nn.Sequential(
            ConvBlock(
                in_channels, base_channels, 7, 2, 3,
                use_batch_norm=use_batch_norm, activation=activation,
                spectral_norm=spectral_norm
            ),
            nn.MaxPool2d(3, 2, 1)
        )
        
        # 残差块组
        channels = base_channels
        self.blocks = nn.ModuleList()
        self.attentions = nn.ModuleList()
        
        for i in range(num_blocks):
            stride = 2 if i > 0 else 1
            self.blocks.append(
                ResidualBlock(
                    channels, channels * 2, stride,
                    use_batch_norm=use_batch_norm, activation=activation,
                    dropout_rate=dropout_rate, spectral_norm=spectral_norm
                )
            )
            channels *= 2
            
            # 注意力模块
            if use_attention:
                self.attentions.append(
                    SqueezeExcitation(channels, attention_reduction)
                )
            else:
                self.attentions.append(nn.Identity())
        
        self.out_channels = channels
    
    def forward(self, x: Tensor) -> Tuple[Tensor, List[Tensor]]:
        features = []
        
        x = self.stem(x)
        
        for block, attention in zip(self.blocks, self.attentions):
            x = block(x)
            x = attention(x)
            features.append(x)
        
        return x, features


class MultiScaleEncoder(nn.Module):
    """多尺度编码器"""
    
    def __init__(
        self,
        in_channels: int,
        base_channels: int = 64,
        num_blocks: int = 4,
        use_batch_norm: bool = True,
        activation: str = "relu",
    ):
        super().__init__()
        
        # 不同尺度的分支
        self.branches = nn.ModuleList([
            self._make_branch(in_channels, base_channels, scale, num_blocks // 2,
                             use_batch_norm, activation)
            for scale in [1, 2, 4]
        ])
        
        # 特征融合
        self.fusion = nn.Sequential(
            ConvBlock(base_channels * 3, base_channels * 2, 1,
                     use_batch_norm=use_batch_norm, activation=activation),
            ConvBlock(base_channels * 2, base_channels * 2, 3,
                     use_batch_norm=use_batch_norm, activation=activation),
        )
        
        self.out_channels = base_channels * 2
    
    def _make_branch(
        self,
        in_channels: int,
        base_channels: int,
        scale: int,
        num_blocks: int,
        use_batch_norm: bool,
        activation: str,
    ) -> nn.Sequential:
        layers = []
        
        # 下采样
        if scale > 1:
            layers.append(nn.AvgPool2d(scale))
        
        layers.append(ConvBlock(in_channels, base_channels, 3,
                               use_batch_norm=use_batch_norm, activation=activation))
        
        for _ in range(num_blocks - 1):
            layers.append(ConvBlock(base_channels, base_channels, 3,
                                   use_batch_norm=use_batch_norm, activation=activation))
        
        return nn.Sequential(*layers)
    
    def forward(self, x: Tensor) -> Tuple[Tensor, List[Tensor]]:
        features = []
        
        # 各分支处理
        branch_outputs = []
        for branch in self.branches:
            out = branch(x)
            # 上采样到统一尺寸
            out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
            branch_outputs.append(out)
            features.append(out)
        
        # 融合
        x = torch.cat(branch_outputs, dim=1)
        x = self.fusion(x)
        
        return x, features


# ============================================================================
# 输出头
# ============================================================================

class PredictionHead(nn.Module):
    """预测头"""
    
    def __init__(
        self,
        in_channels: int,
        output_dim: int,
        hidden_dim: int = 256,
        predict_uncertainty: bool = True,
        use_batch_norm: bool = True,
        dropout_rate: float = 0.5,
    ):
        super().__init__()
        
        self.predict_uncertainty = predict_uncertainty
        
        # 全局池化
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # 预测网络
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, hidden_dim),
            nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
        )
        
        # 均值预测
        self.mean_head = nn.Linear(hidden_dim // 2, output_dim)
        
        # 不确定性预测 (log variance)
        if predict_uncertainty:
            self.var_head = nn.Linear(hidden_dim // 2, output_dim)
    
    def forward(self, x: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        x = self.pool(x)
        x = self.fc(x)
        
        mean = self.mean_head(x)
        
        if self.predict_uncertainty:
            log_var = self.var_head(x)
            # 限制方差范围
            log_var = torch.clamp(log_var, min=-10, max=10)
            return mean, log_var
        
        return mean, None


# ============================================================================
# 主模型
# ============================================================================

class CNNSurrogate(SurrogateModel):
    """
    CNN代理模型
    
    从设计图案预测性能指标，支持不确定性量化。
    """
    
    def __init__(self, config: Optional[CNNSurrogateConfig] = None):
        super().__init__(config or CNNSurrogateConfig())
        self.config: CNNSurrogateConfig = self.config
        
        # 编码器
        if self.config.use_multiscale:
            self.encoder = MultiScaleEncoder(
                in_channels=self.config.input_channels,
                base_channels=self.config.base_channels,
                num_blocks=self.config.num_blocks,
                use_batch_norm=self.config.batch_norm,
                activation=self.config.activation,
            )
        else:
            self.encoder = ResNetEncoder(
                in_channels=self.config.input_channels,
                base_channels=self.config.base_channels,
                num_blocks=self.config.num_blocks,
                use_attention=self.config.use_attention,
                attention_reduction=self.config.attention_reduction,
                use_batch_norm=self.config.batch_norm,
                activation=self.config.activation,
                dropout_rate=self.config.dropout_rate,
                spectral_norm=self.config.spectral_norm,
            )
        
        # 预测头
        self.head = PredictionHead(
            in_channels=self.encoder.out_channels,
            output_dim=self.config.output_dim,
            predict_uncertainty=self.config.predict_uncertainty,
            use_batch_norm=self.config.batch_norm,
            dropout_rate=self.config.dropout_rate,
        )
        
        # 输出名称映射
        self.output_names = self.config.output_names
    
    def forward(
        self,
        design: Tensor,
        return_uncertainty: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        从设计图案预测性能
        
        Args:
            design: 设计图案 [B, C, H, W] 或 [B, H, W] 或 [H, W]
            return_uncertainty: 是否返回不确定性
            
        Returns:
            performance: 性能指标 [B, output_dim]
            uncertainty: (可选) 不确定性 [B, output_dim]
        """
        # 确保输入维度正确
        if design.dim() == 2:
            design = design.unsqueeze(0).unsqueeze(0)
        elif design.dim() == 3:
            design = design.unsqueeze(1)
        
        # 编码
        features, _ = self.encoder(design)
        
        # 预测
        mean, log_var = self.head(features)
        
        if return_uncertainty and log_var is not None:
            return mean, torch.exp(log_var)
        
        return mean
    
    def compute_loss(
        self,
        output: Tensor,
        target: Tensor,
        log_var: Optional[Tensor] = None,
        **kwargs
    ) -> Tensor:
        """
        计算损失
        
        支持普通MSE损失和异方差不确定性损失
        """
        if log_var is not None:
            # 异方差不确定性损失
            # Loss = 0.5 * exp(-log_var) * (target - output)^2 + 0.5 * log_var
            precision = torch.exp(-log_var)
            loss = 0.5 * precision * (target - output) ** 2 + 0.5 * log_var
            return loss.mean()
        else:
            return F.mse_loss(output, target)
    
    def compute_metrics(
        self,
        output: Tensor,
        target: Tensor,
        **kwargs
    ) -> Dict[str, float]:
        """计算评估指标"""
        with torch.no_grad():
            mse = F.mse_loss(output, target).item()
            mae = F.l1_loss(output, target).item()
            
            # R²
            ss_res = ((target - output) ** 2).sum()
            ss_tot = ((target - target.mean()) ** 2).sum()
            r2 = 1 - ss_res / (ss_tot + 1e-8)
            
            # 各输出的相对误差
            relative_errors = {}
            for i, name in enumerate(self.output_names):
                rel_err = torch.abs(output[:, i] - target[:, i]) / (torch.abs(target[:, i]) + 1e-8)
                relative_errors[f'rel_err_{name}'] = rel_err.mean().item()
            
            return {
                'mse': mse,
                'mae': mae,
                'r2': r2.item(),
                **relative_errors
            }
    
    def predict_with_uncertainty(
        self,
        design: Tensor,
        num_samples: int = 100,
        return_all: bool = False
    ) -> Tuple[Tensor, Tensor]:
        """
        使用MC Dropout进行不确定性估计
        
        Args:
            design: 设计图案
            num_samples: MC采样次数
            return_all: 是否返回所有采样结果
            
        Returns:
            mean: 预测均值
            std: 预测标准差
        """
        self.train()  # 启用dropout
        
        predictions = []
        for _ in range(num_samples):
            with torch.no_grad():
                pred = self.forward(design)
                predictions.append(pred)
        
        predictions = torch.stack(predictions)
        
        mean = predictions.mean(dim=0)
        std = predictions.std(dim=0)
        
        if return_all:
            return mean, std, predictions
        
        return mean, std
    
    def get_feature_representation(self, design: Tensor) -> Tensor:
        """获取设计的特征表示"""
        if design.dim() == 2:
            design = design.unsqueeze(0).unsqueeze(0)
        elif design.dim() == 3:
            design = design.unsqueeze(1)
        
        features, _ = self.encoder(design)
        return self.head.pool(features).flatten(1)


# ============================================================================
# 便捷函数
# ============================================================================

def create_cnn_surrogate(
    input_shape: Tuple[int, int, int] = (1, 200, 200),
    output_dim: int = 3,
    **kwargs
) -> CNNSurrogate:
    """
    创建CNN代理模型
    
    Args:
        input_shape: 输入形状 (C, H, W)
        output_dim: 输出维度
        **kwargs: 其他配置参数
        
    Returns:
        CNN代理模型实例
    """
    config = CNNSurrogateConfig(
        input_channels=input_shape[0],
        input_height=input_shape[1],
        input_width=input_shape[2],
        output_dim=output_dim,
        **kwargs
    )
    return CNNSurrogate(config)

"""
DeepONet - Deep Operator Network

学习算子映射，从函数空间到函数空间的映射。
适用于学习物理场响应，如 Maxwell 方程求解。

核心思想:
- Branch Network: 编码输入函数 (如源场、设计参数)
- Trunk Network: 编码输出位置 (如空间坐标)
- 输出: 两网络的点积，得到该位置的物理量

应用场景:
- 快速电磁场求解
- 多物理场耦合预测
- 参数化器件响应

参考文献:
- Lu et al. (2021). "Learning nonlinear operators via DeepONet"
- Wang et al. (2022). "Learning the solution operator of parametric PDEs"
"""

from typing import Dict, Optional, Tuple, List, Union, Any, Callable
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
class DeepONetConfig(ModelConfig):
    """DeepONet配置"""
    name: str = "deeponet"
    
    # Branch Network (输入函数编码器)
    branch_input_dim: int = 100          # 输入函数采样点数
    branch_hidden_dims: List[int] = field(default_factory=lambda: [128, 128, 128])
    branch_output_dim: int = 64          # Branch输出维度 (p)
    
    # Trunk Network (位置编码器)
    trunk_input_dim: int = 2             # 空间坐标维度 (x, y) 或 (x, y, z)
    trunk_hidden_dims: List[int] = field(default_factory=lambda: [128, 128, 128])
    trunk_output_dim: int = 64           # Trunk输出维度 (必须等于 branch_output_dim)
    
    # 输出配置
    output_fields: List[str] = field(default_factory=lambda: ['E_field'])
    num_outputs: int = 1                 # 输出场数量
    
    # 网络类型
    branch_type: str = "mlp"             # 'mlp', 'cnn', 'attention'
    trunk_type: str = "mlp"              # 'mlp', 'siren', 'fourier'
    
    # 激活函数
    activation: str = "gelu"             # 'relu', 'tanh', 'gelu', 'silu'
    
    # 正则化
    dropout_rate: float = 0.0
    batch_norm: bool = False
    
    # SIREN 特定参数 (用于 Trunk Network)
    siren_omega: float = 30.0            # SIREN 频率
    siren_hidden_omega: float = 30.0
    
    # Fourier 特征参数
    fourier_sigma: float = 10.0          # Fourier 特征标准差
    fourier_modes: int = 256             # Fourier 模式数
    
    # 物理约束
    use_physics_loss: bool = True        # 是否使用物理约束损失
    
    # 归一化
    normalize_input: bool = True
    normalize_output: bool = True


# ============================================================================
# 基础网络模块
# ============================================================================

class MLP(nn.Module):
    """多层感知机"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        activation: str = "gelu",
        dropout_rate: float = 0.0,
        batch_norm: bool = False,
        final_activation: bool = False,
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            layers.append(self._get_activation(activation))
            
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))
        
        if final_activation:
            layers.append(self._get_activation(activation))
        
        self.net = nn.Sequential(*layers)
    
    def _get_activation(self, name: str) -> nn.Module:
        activations = {
            'relu': nn.ReLU(inplace=True),
            'tanh': nn.Tanh(),
            'gelu': nn.GELU(),
            'silu': nn.SiLU(inplace=True),
            'sigmoid': nn.Sigmoid(),
        }
        return activations.get(name, nn.GELU())
    
    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class SIRENLayer(nn.Module):
    """SIREN层: sin(ω * (Wx + b))"""
    
    def __init__(self, in_features: int, out_features: int, omega: float = 30.0):
        super().__init__()
        
        self.linear = nn.Linear(in_features, out_features)
        self.omega = omega
        
        # SIREN 初始化
        with torch.no_grad():
            bound = math.sqrt(6.0 / in_features) / omega
            self.linear.weight.uniform_(-bound, bound)
            self.linear.bias.uniform_(-bound, bound)
    
    def forward(self, x: Tensor) -> Tensor:
        return torch.sin(self.omega * self.linear(x))


class SIREN(nn.Module):
    """SIREN网络: 正弦激活的网络，适合表示高频信号"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        omega: float = 30.0,
        hidden_omega: float = 30.0,
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            # 第一层使用 omega，后续层使用 hidden_omega
            layer_omega = omega if i == 0 else hidden_omega
            layers.append(SIRENLayer(prev_dim, hidden_dim, layer_omega))
            prev_dim = hidden_dim
        
        # 输出层 (无激活)
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class FourierFeatures(nn.Module):
    """Fourier特征映射: 增强对高频信号的表示能力"""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        sigma: float = 10.0,
        learnable: bool = True,
    ):
        super().__init__()
        
        # Fourier 系数
        self.B = nn.Parameter(
            torch.randn(input_dim, output_dim // 2) * sigma,
            requires_grad=learnable
        )
        
        self.output_dim = output_dim
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, input_dim] 或 [B, N, input_dim]
        Returns:
            [B, output_dim] 或 [B, N, output_dim]
        """
        # Fourier 特征: [sin(2πBx), cos(2πBx)]
        x_proj = 2 * math.pi * torch.matmul(x, self.B)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class FourierMLP(nn.Module):
    """带Fourier特征的MLP"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        fourier_sigma: float = 10.0,
        fourier_modes: int = 256,
        activation: str = "gelu",
    ):
        super().__init__()
        
        # Fourier 特征
        self.fourier = FourierFeatures(
            input_dim, fourier_modes, fourier_sigma
        )
        
        # MLP
        layers = []
        prev_dim = fourier_modes
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self._get_activation(activation))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.net = nn.Sequential(*layers)
    
    def _get_activation(self, name: str) -> nn.Module:
        activations = {
            'relu': nn.ReLU(inplace=True),
            'tanh': nn.Tanh(),
            'gelu': nn.GELU(),
            'silu': nn.SiLU(inplace=True),
        }
        return activations.get(name, nn.GELU())
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.fourier(x)
        return self.net(x)


class CNNBranch(nn.Module):
    """CNN Branch Network: 用于处理图像/场输入"""
    
    def __init__(
        self,
        input_channels: int,
        hidden_channels: List[int],
        output_dim: int,
        input_height: int = 64,
        input_width: int = 64,
        activation: str = "gelu",
    ):
        super().__init__()
        
        layers = []
        prev_channels = input_channels
        
        for hidden_channel in hidden_channels:
            layers.extend([
                nn.Conv2d(prev_channels, hidden_channel, 3, padding=1),
                nn.BatchNorm2d(hidden_channel),
                self._get_activation(activation),
                nn.MaxPool2d(2),
            ])
            prev_channels = hidden_channel
        
        self.conv = nn.Sequential(*layers)
        
        # 计算展平后的维度
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, input_height, input_width)
            flat_dim = self.conv(dummy).numel()
        
        self.fc = nn.Linear(flat_dim, output_dim)
    
    def _get_activation(self, name: str) -> nn.Module:
        activations = {
            'relu': nn.ReLU(inplace=True),
            'tanh': nn.Tanh(),
            'gelu': nn.GELU(),
            'silu': nn.SiLU(inplace=True),
        }
        return activations.get(name, nn.GELU())
    
    def forward(self, x: Tensor) -> Tensor:
        # x: [B, C, H, W]
        x = self.conv(x)
        x = x.flatten(1)
        return self.fc(x)


class AttentionBranch(nn.Module):
    """Attention Branch Network: 用于处理变长输入"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.output_proj = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x: Tensor) -> Tensor:
        # x: [B, N, input_dim]
        x = self.input_proj(x)
        x = self.transformer(x)
        # 全局平均池化
        x = x.mean(dim=1)
        return self.output_proj(x)


# ============================================================================
# Branch Network
# ============================================================================

class BranchNetwork(nn.Module):
    """Branch Network: 编码输入函数"""
    
    def __init__(self, config: DeepONetConfig):
        super().__init__()
        
        self.config = config
        self.normalize = config.normalize_input
        
        if config.branch_type == "mlp":
            self.net = MLP(
                input_dim=config.branch_input_dim,
                hidden_dims=config.branch_hidden_dims,
                output_dim=config.branch_output_dim,
                activation=config.activation,
                dropout_rate=config.dropout_rate,
                batch_norm=config.batch_norm,
            )
        elif config.branch_type == "cnn":
            self.net = CNNBranch(
                input_channels=1,
                hidden_channels=[32, 64, 128],
                output_dim=config.branch_output_dim,
                activation=config.activation,
            )
        elif config.branch_type == "attention":
            self.net = AttentionBranch(
                input_dim=1,
                hidden_dim=config.branch_hidden_dims[0],
                output_dim=config.branch_output_dim,
            )
        else:
            raise ValueError(f"Unknown branch type: {config.branch_type}")
        
        # 输入归一化参数
        if self.normalize:
            self.register_buffer('input_mean', torch.zeros(config.branch_input_dim))
            self.register_buffer('input_std', torch.ones(config.branch_input_dim))
    
    def forward(self, u: Tensor) -> Tensor:
        """
        Args:
            u: 输入函数采样点 [B, N] 或 [B, C, H, W]
        Returns:
            [B, p] 编码后的特征
        """
        if self.normalize and u.dim() == 2:
            u = (u - self.input_mean) / (self.input_std + 1e-8)
        
        return self.net(u)


# ============================================================================
# Trunk Network
# ============================================================================

class TrunkNetwork(nn.Module):
    """Trunk Network: 编码输出位置"""
    
    def __init__(self, config: DeepONetConfig):
        super().__init__()
        
        self.config = config
        
        if config.trunk_type == "mlp":
            self.net = MLP(
                input_dim=config.trunk_input_dim,
                hidden_dims=config.trunk_hidden_dims,
                output_dim=config.trunk_output_dim,
                activation=config.activation,
                dropout_rate=config.dropout_rate,
                batch_norm=config.batch_norm,
            )
        elif config.trunk_type == "siren":
            self.net = SIREN(
                input_dim=config.trunk_input_dim,
                hidden_dims=config.trunk_hidden_dims,
                output_dim=config.trunk_output_dim,
                omega=config.siren_omega,
                hidden_omega=config.siren_hidden_omega,
            )
        elif config.trunk_type == "fourier":
            self.net = FourierMLP(
                input_dim=config.trunk_input_dim,
                hidden_dims=config.trunk_hidden_dims,
                output_dim=config.trunk_output_dim,
                fourier_sigma=config.fourier_sigma,
                fourier_modes=config.fourier_modes,
                activation=config.activation,
            )
        else:
            raise ValueError(f"Unknown trunk type: {config.trunk_type}")
    
    def forward(self, y: Tensor) -> Tensor:
        """
        Args:
            y: 输出位置 [B, M, D] 或 [M, D] 或 [B, D]
        Returns:
            [B, M, p] 或 [M, p] 或 [B, p] 编码后的特征
        """
        return self.net(y)


# ============================================================================
# DeepONet 主模型
# ============================================================================

class DeepONet(SurrogateModel):
    """
    Deep Operator Network
    
    学习算子 G: u -> G(u)，其中 u 是输入函数
    G(u)(y) = Σᵢ branch(u)ᵢ × trunk(y)ᵢ
    """
    
    def __init__(self, config: Optional[DeepONetConfig] = None):
        super().__init__(config or DeepONetConfig())
        self.config: DeepONetConfig = self.config
        
        # Branch 和 Trunk 网络
        self.branch = BranchNetwork(self.config)
        self.trunk = TrunkNetwork(self.config)
        
        # 输出归一化参数
        if self.config.normalize_output:
            self.register_buffer('output_mean', torch.zeros(self.config.num_outputs))
            self.register_buffer('output_std', torch.ones(self.config.num_outputs))
        
        # Bias (可选)
        self.bias = nn.Parameter(torch.zeros(self.config.num_outputs))
    
    def forward(
        self,
        u: Tensor,
        y: Optional[Tensor] = None,
    ) -> Tensor:
        """
        计算算子在给定位置的输出
        
        Args:
            u: 输入函数 [B, N] 或 [B, C, H, W]
            y: 输出位置 [B, M, D] 或 [M, D]，若为None则使用所有采样点
            
        Returns:
            G(u)(y): [B, M] 或 [B, M, num_outputs]
        """
        # Branch: [B, p]
        b = self.branch(u)
        
        # Trunk: [B, M, p] 或 [M, p]
        if y is None:
            raise ValueError("Output locations y must be provided")
        
        t = self.trunk(y)
        
        # 处理维度
        if t.dim() == 2 and b.dim() == 2:
            # t: [M, p], b: [B, p] -> output: [B, M]
            output = torch.einsum('bp,mp->bm', b, t)
        elif t.dim() == 3:
            # t: [B, M, p], b: [B, p] -> output: [B, M]
            output = torch.einsum('bp,bmp->bm', b, t)
        else:
            output = (b * t).sum(dim=-1)
        
        # 添加偏置
        output = output + self.bias
        
        # 输出反归一化
        if self.config.normalize_output:
            output = output * self.output_std + self.output_mean
        
        return output
    
    def forward_field(
        self,
        u: Tensor,
        grid: Tensor,
    ) -> Tensor:
        """
        在整个空间网格上计算场
        
        Args:
            u: 输入函数 [B, N]
            grid: 空间网格 [H, W, D] 或 [H, W]
            
        Returns:
            输出场 [B, H, W]
        """
        # Branch: [B, p]
        b = self.branch(u)
        
        # 展平网格
        original_shape = grid.shape[:2]
        grid_flat = grid.reshape(-1, grid.shape[-1])  # [H*W, D]
        
        # Trunk: [H*W, p]
        t = self.trunk(grid_flat)
        
        # 计算: [B, H*W]
        output = torch.einsum('bp,np->bn', b, t)
        
        # 重塑为空间场
        output = output.reshape(-1, *original_shape)
        
        return output
    
    def compute_loss(
        self,
        u: Tensor,
        y: Tensor,
        target: Tensor,
        **kwargs
    ) -> Tensor:
        """
        计算损失
        
        Args:
            u: 输入函数
            y: 输出位置
            target: 目标值
        """
        output = self.forward(u, y)
        return F.mse_loss(output, target)
    
    def compute_physics_loss(
        self,
        u: Tensor,
        y: Tensor,
        physics_fn: Callable,
        **kwargs
    ) -> Tensor:
        """
        计算物理约束损失
        
        Args:
            u: 输入函数
            y: 输出位置
            physics_fn: 物理约束函数，返回PDE残差
        """
        y.requires_grad_(True)
        output = self.forward(u, y)
        
        # 计算物理残差
        residual = physics_fn(output, y, u)
        
        return (residual ** 2).mean()
    
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
            
            # 相对 L2 误差
            rel_l2 = (output - target).norm() / (target.norm() + 1e-8)
            
            return {
                'mse': mse,
                'mae': mae,
                'r2': r2.item(),
                'rel_l2': rel_l2.item(),
            }
    
    def set_normalization(
        self,
        input_mean: Tensor,
        input_std: Tensor,
        output_mean: Optional[Tensor] = None,
        output_std: Optional[Tensor] = None,
    ):
        """设置归一化参数"""
        if self.config.normalize_input:
            self.input_mean = input_mean
            self.input_std = input_std
        
        if self.config.normalize_output and output_mean is not None:
            self.output_mean = output_mean
            self.output_std = output_std


# ============================================================================
# 变分 DeepONet (Variational DeepONet)
# ============================================================================

class VariationalDeepONet(DeepONet):
    """
    变分 DeepONet: 支持不确定性量化
    """
    
    def __init__(self, config: Optional[DeepONetConfig] = None):
        super().__init__(config)
        
        # 额外的方差预测头
        self.var_branch = nn.Sequential(
            nn.Linear(self.config.branch_output_dim, self.config.branch_output_dim),
            nn.GELU(),
            nn.Linear(self.config.branch_output_dim, self.config.branch_output_dim),
        )
    
    def forward(
        self,
        u: Tensor,
        y: Tensor,
        return_variance: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        计算输出和方差
        """
        # Branch 特征
        b = self.branch(u)
        
        # 方差特征
        b_var = self.var_branch(b)
        
        # Trunk 特征
        t = self.trunk(y)
        
        # 均值
        if t.dim() == 2:
            mean = torch.einsum('bp,mp->bm', b, t) + self.bias
            log_var = torch.einsum('bp,mp->bm', b_var, t)
        else:
            mean = torch.einsum('bp,bmp->bm', b, t) + self.bias
            log_var = torch.einsum('bp,bmp->bm', b_var, t)
        
        if return_variance:
            return mean, torch.exp(log_var)
        
        return mean
    
    def compute_loss(
        self,
        u: Tensor,
        y: Tensor,
        target: Tensor,
        **kwargs
    ) -> Tensor:
        """异方差不确定性损失"""
        mean, var = self.forward(u, y, return_variance=True)
        
        # 负对数似然
        loss = 0.5 * torch.log(var + 1e-8) + 0.5 * (target - mean) ** 2 / var
        
        return loss.mean()


# ============================================================================
# 分支 DeepONet (Branched DeepONet)
# ============================================================================

class BranchedDeepONet(BaseModel):
    """
    分支 DeepONet: 处理多场输出
    
    每个输出场有独立的 Branch 和 Trunk 网络
    """
    
    def __init__(self, config: Optional[DeepONetConfig] = None):
        super().__init__(config or DeepONetConfig())
        self.config: DeepONetConfig = self.config
        
        self.num_outputs = self.config.num_outputs
        self.output_fields = self.config.output_fields
        
        # 每个输出场的网络
        self.branches = nn.ModuleList([
            BranchNetwork(self.config) for _ in range(self.num_outputs)
        ])
        
        self.trunks = nn.ModuleList([
            TrunkNetwork(self.config) for _ in range(self.num_outputs)
        ])
        
        self.biases = nn.Parameter(torch.zeros(self.num_outputs))
    
    def forward(
        self,
        u: Tensor,
        y: Tensor,
    ) -> Tensor:
        """
        计算所有输出场
        
        Returns:
            [B, M, num_outputs] 或 [B, num_outputs]
        """
        outputs = []
        
        for i in range(self.num_outputs):
            b = self.branches[i](u)
            t = self.trunks[i](y)
            
            if t.dim() == 2:
                out = torch.einsum('bp,mp->bm', b, t) + self.biases[i]
            else:
                out = torch.einsum('bp,bmp->bm', b, t) + self.biases[i]
            
            outputs.append(out)
        
        # [B, M, num_outputs]
        return torch.stack(outputs, dim=-1)


# ============================================================================
# 便捷函数
# ============================================================================

def create_deeponet(
    branch_input_dim: int,
    trunk_input_dim: int = 2,
    output_dim: int = 64,
    branch_type: str = "mlp",
    trunk_type: str = "fourier",
    **kwargs
) -> DeepONet:
    """
    创建 DeepONet 模型
    
    Args:
        branch_input_dim: Branch输入维度
        trunk_input_dim: Trunk输入维度 (空间坐标维度)
        output_dim: 隐藏层输出维度
        branch_type: Branch网络类型
        trunk_type: Trunk网络类型
        
    Returns:
        DeepONet 模型实例
    """
    config = DeepONetConfig(
        branch_input_dim=branch_input_dim,
        trunk_input_dim=trunk_input_dim,
        branch_output_dim=output_dim,
        trunk_output_dim=output_dim,
        branch_type=branch_type,
        trunk_type=trunk_type,
        **kwargs
    )
    return DeepONet(config)


def create_electromagnetic_deeponet(
    design_shape: Tuple[int, int] = (64, 64),
    num_wavelengths: int = 1,
    output_fields: List[str] = ['Ex', 'Ey', 'Ez'],
) -> BranchedDeepONet:
    """
    创建电磁场求解 DeepONet
    
    Args:
        design_shape: 设计区域形状
        num_wavelengths: 波长数量
        output_fields: 输出场名称列表
        
    Returns:
        BranchedDeepONet 模型实例
    """
    config = DeepONetConfig(
        branch_input_dim=design_shape[0] * design_shape[1] * num_wavelengths,
        trunk_input_dim=3,  # (x, y, z) 或 (x, y, wavelength)
        branch_output_dim=128,
        trunk_output_dim=128,
        branch_hidden_dims=[256, 256, 128],
        trunk_hidden_dims=[256, 256, 128],
        output_fields=output_fields,
        num_outputs=len(output_fields),
        trunk_type="fourier",
        fourier_sigma=10.0,
        fourier_modes=256,
    )
    return BranchedDeepONet(config)

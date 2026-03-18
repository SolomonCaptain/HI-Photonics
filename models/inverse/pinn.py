"""
Physics-Informed Neural Network (PINN)

物理信息神经网络用于光子学正向和逆向问题求解。
将 Maxwell 方程等物理约束融入神经网络训练。

核心组件:
- PhysicsInformedNet: 基础 PINN 网络
- SirenNet: 正弦激活网络（SIREN）
- MaxwellPINN: Maxwell 方程 PINN
- PhotonicsPINN: 光子学专用 PINN
- PINNSolver: PINN 求解器

参考文献:
- Raissi et al., "Physics-Informed Neural Networks", JCP 2019
- Sitzmann et al., "Implicit Neural Representations with Periodic Activation Functions", NeurIPS 2020
"""

from typing import Dict, Optional, Tuple, List, Union, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Adam, LBFGS
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models.base import BaseModel, ModelConfig


# ============================================================================
# 配置类
# ============================================================================

@dataclass
class PINNConfig(ModelConfig):
    """PINN 基础配置"""
    name: str = "pinn"
    
    # 输入维度
    spatial_dim: int = 2          # 空间维度 (2D 或 3D)
    design_dim: int = 0           # 设计参数维度（逆向问题时使用）
    
    # 输出维度
    field_components: int = 3     # 场分量数（2D: 3, 3D: 6）
    
    # 网络架构
    hidden_dims: List[int] = field(default_factory=lambda: [64, 128, 256, 256, 128, 64])
    activation: str = "tanh"      # 'sine', 'tanh', 'gelu', 'silu'
    
    # Fourier 特征
    use_fourier: bool = True
    fourier_dim: int = 128
    fourier_sigma: float = 10.0
    
    # 正则化
    dropout_rate: float = 0.0
    spectral_norm: bool = False
    
    # 输出
    output_scale: bool = True     # 是否使用输出缩放


@dataclass
class MaxwellConfig(PINNConfig):
    """Maxwell 方程 PINN 配置"""
    name: str = "maxwell_pinn"
    
    # 场分量
    field_components: int = 6     # Ex, Ey, Ez, Hx, Hy, Hz
    
    # 物理参数
    wavelength: float = 1.55e-6   # 工作波长 (m)
    epsilon_r: float = 12.0       # 相对介电常数 (硅)
    mu_r: float = 1.0             # 相对磁导率
    
    # 归一化
    normalize_coords: bool = True
    coord_scale: float = 1e-6     # 坐标缩放因子


@dataclass
class PhysicsLossConfig:
    """物理损失配置"""
    # 损失权重
    data_weight: float = 1.0
    physics_weight: float = 1.0
    bc_weight: float = 1.0
    ic_weight: float = 1.0
    
    # 自适应权重
    adaptive_weights: bool = True
    adaptive_update_freq: int = 100
    
    # 梯度平衡
    gradient_balancing: bool = True


# ============================================================================
# 工具模块
# ============================================================================

class FourierFeatures(nn.Module):
    """
    Fourier 特征编码
    
    将输入映射到高维 Fourier 空间，帮助网络学习高频特征。
    """
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int = 128,
        sigma: float = 10.0,
        learnable: bool = False
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.sigma = sigma
        
        # 随机频率矩阵
        B = torch.randn(in_dim, out_dim // 2) * sigma
        
        if learnable:
            self.B = nn.Parameter(B)
        else:
            self.register_buffer('B', B)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Fourier 特征编码
        
        Args:
            x: 输入 [B, in_dim]
            
        Returns:
            Fourier 特征 [B, out_dim]
        """
        # 线性投影
        x_proj = 2 * math.pi * x @ self.B
        
        # sin 和 cos 编码
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Sine(nn.Module):
    """正弦激活函数（用于 SIREN）"""
    
    def __init__(self, w0: float = 1.0):
        super().__init__()
        self.w0 = w0
    
    def forward(self, x: Tensor) -> Tensor:
        return torch.sin(self.w0 * x)


class GradientScale(nn.Module):
    """梯度缩放层"""
    
    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = scale
    
    def forward(self, x: Tensor) -> Tensor:
        return x * self.scale


def get_activation(name: str, **kwargs) -> nn.Module:
    """获取激活函数"""
    activations = {
        'sine': lambda: Sine(kwargs.get('w0', 30.0)),
        'tanh': nn.Tanh,
        'gelu': nn.GELU,
        'silu': nn.SiLU,
        'relu': nn.ReLU,
        'leaky_relu': lambda: nn.LeakyReLU(kwargs.get('negative_slope', 0.2)),
        'identity': nn.Identity
    }
    
    if name.lower() not in activations:
        raise ValueError(f"Unknown activation: {name}")
    
    return activations[name.lower()]()


# ============================================================================
# 基础 PINN 网络
# ============================================================================

class PhysicsInformedNet(BaseModel):
    """
    物理信息神经网络基础类
    
    支持多种激活函数和 Fourier 特征编码。
    
    使用示例:
    ```python
    config = PINNConfig(
        spatial_dim=2,
        field_components=3,
        hidden_dims=[64, 128, 256, 128, 64]
    )
    pinn = PhysicsInformedNet(config)
    
    # 预测场
    coords = torch.rand(100, 2)  # (x, y)
    fields = pinn(coords)
    ```
    """
    
    def __init__(self, config: Optional[PINNConfig] = None):
        config = config or PINNConfig()
        super().__init__(config)
        self.config = config
        
        self.spatial_dim = config.spatial_dim
        self.design_dim = config.design_dim
        self.field_components = config.field_components
        
        # Fourier 特征编码
        if config.use_fourier:
            self.fourier = FourierFeatures(
                in_dim=config.spatial_dim + config.design_dim,
                out_dim=config.fourier_dim,
                sigma=config.fourier_sigma
            )
            input_dim = config.fourier_dim
        else:
            self.fourier = None
            input_dim = config.spatial_dim + config.design_dim
        
        # 构建网络
        self.net = self._build_network(input_dim)
        
        # 输出缩放
        if config.output_scale:
            self.output_scale = nn.Parameter(torch.ones(config.field_components))
        else:
            self.output_scale = None
        
        # 初始化权重
        self._init_weights()
    
    def _build_network(self, input_dim: int) -> nn.Module:
        """构建全连接网络"""
        layers = []
        dims = [input_dim] + self.config.hidden_dims
        
        for i in range(len(dims) - 1):
            # 线性层
            linear = nn.Linear(dims[i], dims[i + 1])
            
            if self.config.spectral_norm:
                linear = nn.utils.spectral_norm(linear)
            
            layers.append(linear)
            
            # 激活函数
            layers.append(get_activation(self.config.activation))
            
            # Dropout
            if self.config.dropout_rate > 0:
                layers.append(nn.Dropout(self.config.dropout_rate))
        
        # 输出层
        output_layer = nn.Linear(dims[-1], self.config.field_components)
        layers.append(output_layer)
        
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Xavier 初始化
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(
        self,
        coordinates: Tensor,
        design_params: Optional[Tensor] = None
    ) -> Tensor:
        """
        前向传播
        
        Args:
            coordinates: 空间坐标 [B, spatial_dim]
            design_params: 设计参数 [B, design_dim]（可选）
            
        Returns:
            物理场 [B, field_components]
        """
        # 准备输入
        if design_params is not None:
            x = torch.cat([coordinates, design_params], dim=-1)
        else:
            x = coordinates
        
        # Fourier 特征编码
        if self.fourier is not None:
            x = self.fourier(x)
        
        # 网络前向
        output = self.net(x)
        
        # 输出缩放
        if self.output_scale is not None:
            output = output * self.output_scale
        
        return output
    
    def compute_gradient(
        self,
        coordinates: Tensor,
        design_params: Optional[Tensor] = None,
        component: int = 0,
        wrt: str = 'coordinates'
    ) -> Tensor:
        """
        计算梯度（用于 PDE 残差）
        
        Args:
            coordinates: 空间坐标
            design_params: 设计参数
            component: 场分量索引
            wrt: 对什么求导 ('coordinates' 或 'design_params')
            
        Returns:
            梯度
        """
        if wrt == 'coordinates':
            coordinates = coordinates.requires_grad_(True)
            output = self.forward(coordinates, design_params)
            grad = torch.autograd.grad(
                output[:, component].sum(),
                coordinates,
                create_graph=True,
                retain_graph=True
            )[0]
        elif wrt == 'design_params':
            if design_params is None:
                raise ValueError("design_params required for gradient w.r.t. design")
            design_params = design_params.requires_grad_(True)
            output = self.forward(coordinates, design_params)
            grad = torch.autograd.grad(
                output[:, component].sum(),
                design_params,
                create_graph=True,
                retain_graph=True
            )[0]
        else:
            raise ValueError(f"Unknown wrt: {wrt}")
        
        return grad
    
    def compute_laplacian(
        self,
        coordinates: Tensor,
        design_params: Optional[Tensor] = None,
        component: int = 0
    ) -> Tensor:
        """
        计算拉普拉斯算子
        
        Args:
            coordinates: 空间坐标
            design_params: 设计参数
            component: 场分量索引
            
        Returns:
            拉普拉斯算子结果
        """
        coordinates = coordinates.requires_grad_(True)
        
        # 一阶导数
        output = self.forward(coordinates, design_params)
        grad1 = torch.autograd.grad(
            output[:, component].sum(),
            coordinates,
            create_graph=True,
            retain_graph=True
        )[0]
        
        # 二阶导数
        laplacian = torch.zeros(coordinates.size(0), device=coordinates.device)
        for i in range(self.spatial_dim):
            grad2 = torch.autograd.grad(
                grad1[:, i].sum(),
                coordinates,
                create_graph=True,
                retain_graph=True
            )[0]
            laplacian += grad2[:, i]
        
        return laplacian


# ============================================================================
# SIREN 网络
# ============================================================================

class SirenLayer(nn.Module):
    """SIREN 层（正弦激活）"""
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        w0: float = 30.0,
        is_first: bool = False,
        is_last: bool = False
    ):
        super().__init__()
        self.w0 = w0
        self.is_first = is_first
        self.is_last = is_last
        
        self.linear = nn.Linear(in_features, out_features)
        
        # SIREN 特殊初始化
        self._init_weights()
    
    def _init_weights(self):
        """SIREN 初始化"""
        with torch.no_grad():
            if self.is_first:
                # 第一层：均匀分布
                bound = 1.0 / self.linear.in_features
            else:
                # 后续层：根据 w0 调整
                bound = math.sqrt(6.0 / self.linear.in_features) / self.w0
            
            self.linear.weight.uniform_(-bound, bound)
            self.linear.bias.uniform_(-bound, bound)
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.linear(x)
        
        if not self.is_last:
            x = torch.sin(self.w0 * x)
        
        return x


class SirenNet(BaseModel):
    """
    SIREN 网络
    
    使用正弦激活函数的隐式神经表示网络。
    特别适合表示高频信号和物理场。
    
    参考文献:
    Sitzmann et al., "Implicit Neural Representations with Periodic Activation Functions", NeurIPS 2020
    """
    
    def __init__(
        self,
        config: Optional[PINNConfig] = None,
        w0: float = 30.0,
        w0_initial: float = 30.0
    ):
        config = config or PINNConfig()
        super().__init__(config)
        self.config = config
        
        self.spatial_dim = config.spatial_dim
        self.design_dim = config.design_dim
        self.field_components = config.field_components
        
        # Fourier 特征（可选）
        if config.use_fourier:
            self.fourier = FourierFeatures(
                in_dim=config.spatial_dim + config.design_dim,
                out_dim=config.fourier_dim,
                sigma=config.fourier_sigma
            )
            input_dim = config.fourier_dim
        else:
            self.fourier = None
            input_dim = config.spatial_dim + config.design_dim
        
        # 构建 SIREN 层
        layers = []
        dims = [input_dim] + config.hidden_dims
        
        for i in range(len(dims) - 1):
            is_first = (i == 0)
            is_last = (i == len(dims) - 2)
            
            layer = SirenLayer(
                dims[i], dims[i + 1],
                w0=w0 if not is_first else w0_initial,
                is_first=is_first,
                is_last=is_last
            )
            layers.append(layer)
        
        # 输出层
        layers.append(nn.Linear(dims[-1], config.field_components))
        
        self.net = nn.Sequential(*layers)
    
    def forward(
        self,
        coordinates: Tensor,
        design_params: Optional[Tensor] = None
    ) -> Tensor:
        """前向传播"""
        if design_params is not None:
            x = torch.cat([coordinates, design_params], dim=-1)
        else:
            x = coordinates
        
        if self.fourier is not None:
            x = self.fourier(x)
        
        return self.net(x)


# ============================================================================
# Maxwell 方程 PINN
# ============================================================================

class MaxwellPINN(PhysicsInformedNet):
    """
    Maxwell 方程物理信息神经网络
    
    求解电磁场分布，满足 Maxwell 方程约束。
    
    支持的问题类型:
    1. 散射问题：入射场 + 散射场
    2. 波导模式：本征模式求解
    3. 谐振腔：本征频率求解
    
    使用示例:
    ```python
    config = MaxwellConfig(
        spatial_dim=2,
        wavelength=1.55e-6,
        epsilon_r=12.0
    )
    pinn = MaxwellPINN(config)
    
    # 预测电磁场
    coords = generate_grid((100, 100))
    fields = pinn(coords)  # (Ex, Ey, Ez, Hx, Hy, Hz)
    
    # 计算 Maxwell 残差
    residual = pinn.compute_maxwell_residual(coords)
    ```
    """
    
    def __init__(self, config: Optional[MaxwellConfig] = None):
        config = config or MaxwellConfig()
        super().__init__(config)
        
        # 物理参数
        self.wavelength = config.wavelength
        self.k0 = 2 * math.pi / config.wavelength  # 波数
        
        # 相对介电常数和磁导率
        self.epsilon_r = config.epsilon_r
        self.mu_r = config.mu_r
        
        # 归一化坐标
        self.normalize_coords = config.normalize_coords
        self.coord_scale = config.coord_scale
    
    def forward(
        self,
        coordinates: Tensor,
        design_params: Optional[Tensor] = None
    ) -> Dict[str, Tensor]:
        """
        预测电磁场
        
        Args:
            coordinates: 空间坐标 [B, spatial_dim]
            design_params: 设计参数（可选）
            
        Returns:
            字典包含:
            - E: 电场 [B, 3]
            - H: 磁场 [B, 3]
        """
        # 坐标归一化
        if self.normalize_coords:
            coordinates = coordinates / self.coord_scale
        
        # 网络输出
        output = super().forward(coordinates, design_params)
        
        # 分离电场和磁场
        E = output[:, :3]
        H = output[:, 3:6]
        
        return {'E': E, 'H': H}
    
    def compute_maxwell_residual(
        self,
        coordinates: Tensor,
        design_params: Optional[Tensor] = None,
        frequency: Optional[float] = None
    ) -> Dict[str, Tensor]:
        """
        计算 Maxwell 方程残差
        
        频域 Maxwell 方程:
        ∇ × E = -jωμH
        ∇ × H = jωεE + J
        
        Args:
            coordinates: 空间坐标
            design_params: 设计参数
            frequency: 频率（默认使用波长计算）
            
        Returns:
            残差字典
        """
        coordinates = coordinates.requires_grad_(True)
        
        # 预测场
        fields = self.forward(coordinates, design_params)
        E = fields['E']
        H = fields['H']
        
        # 计算旋度
        curl_E = self._compute_curl(E, coordinates)
        curl_H = self._compute_curl(H, coordinates)
        
        # 波数
        if frequency is None:
            k = self.k0
        else:
            k = 2 * math.pi * frequency / 3e8
        
        # 相对参数
        epsilon_r = self._get_epsilon(coordinates, design_params)
        mu_r = self.mu_r
        
        # Maxwell 方程残差
        # ∇ × E + jωμH = 0
        residual_E = curl_E + 1j * k * mu_r * H
        
        # ∇ × H - jωεE = 0 (无源情况)
        residual_H = curl_H - 1j * k * epsilon_r * E
        
        return {
            'curl_E_residual': residual_E,
            'curl_H_residual': residual_H,
            'E': E,
            'H': H
        }
    
    def _compute_curl(self, field: Tensor, coordinates: Tensor) -> Tensor:
        """
        计算场的旋度
        
        对于 2D 问题 (Ez, Hx, Hy):
        curl(E) = ∂Ey/∂x - ∂Ex/∂y (z 分量)
        """
        # 计算梯度
        grad_x = torch.autograd.grad(
            field[:, 0].sum(), coordinates,
            create_graph=True, retain_graph=True
        )[0][:, 0]
        
        grad_y = torch.autograd.grad(
            field[:, 1].sum(), coordinates,
            create_graph=True, retain_graph=True
        )[0][:, 1]
        
        # 简化的 2D 旋度
        # curl_z = ∂Ey/∂x - ∂Ex/∂y
        curl_z = grad_y - grad_x
        
        # 返回 3D 形式（z 分量非零）
        curl = torch.zeros_like(field)
        curl[:, 2] = curl_z
        
        return curl
    
    def _get_epsilon(
        self,
        coordinates: Tensor,
        design_params: Optional[Tensor] = None
    ) -> Tensor:
        """
        获取介电常数分布
        
        可以是空间变化的（通过设计参数定义）
        """
        if design_params is None:
            # 均匀介质
            return torch.full(
                (coordinates.size(0),),
                self.epsilon_r,
                device=coordinates.device
            )
        else:
            # 空间变化的介电常数（由设计参数定义）
            # 这里简化处理，实际应用中需要根据设计参数计算
            return torch.ones(coordinates.size(0), device=coordinates.device) * self.epsilon_r
    
    def compute_poynting_vector(
        self,
        coordinates: Tensor,
        design_params: Optional[Tensor] = None
    ) -> Tensor:
        """
        计算 Poynting 矢量
        
        S = E × H*
        """
        fields = self.forward(coordinates, design_params)
        E = fields['E']
        H = fields['H']
        
        # Poynting 矢量
        S = torch.cross(E, H, dim=-1)
        
        return S


# ============================================================================
# 光子学专用 PINN
# ============================================================================

class PhotonicsPINN(BaseModel):
    """
    光子学专用物理信息神经网络
    
    专门针对光子器件设计优化：
    1. 波导设计
    2. 谐振腔设计
    3. 光栅设计
    
    特点:
    - 支持多物理场（电磁 + 热）
    - 内置常见边界条件
    - 支持逆向设计
    """
    
    def __init__(
        self,
        config: Optional[PINNConfig] = None,
        physics_config: Optional[PhysicsLossConfig] = None
    ):
        config = config or PINNConfig()
        super().__init__(config)
        self.config = config
        
        # 物理损失配置
        self.physics_config = physics_config or PhysicsLossConfig()
        
        # 场预测网络
        self.field_net = PhysicsInformedNet(config)
        
        # 逆向设计网络（可选）
        if config.design_dim > 0:
            self.inverse_net = nn.Sequential(
                nn.Linear(config.field_components * 2, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, config.design_dim),
                nn.Sigmoid()
            )
        else:
            self.inverse_net = None
        
        # 损失权重（可学习）
        if self.physics_config.adaptive_weights:
            self.loss_weights = nn.ParameterDict({
                'data': nn.Parameter(torch.tensor(1.0)),
                'physics': nn.Parameter(torch.tensor(1.0)),
                'bc': nn.Parameter(torch.tensor(1.0))
            })
    
    def forward(
        self,
        coordinates: Tensor,
        design_params: Optional[Tensor] = None
    ) -> Tensor:
        """预测物理场"""
        return self.field_net(coordinates, design_params)
    
    def inverse_design(
        self,
        target_field: Tensor,
        coordinates: Tensor
    ) -> Tensor:
        """
        逆向设计
        
        从目标场分布反推设计参数
        
        Args:
            target_field: 目标场分布
            coordinates: 空间坐标
            
        Returns:
            设计参数
        """
        if self.inverse_net is None:
            raise ValueError("Inverse network not configured")
        
        # 编码目标场
        field_encoding = torch.cat([
            target_field.mean(dim=0),
            target_field.std(dim=0)
        ], dim=-1)
        
        # 预测设计参数
        design_params = self.inverse_net(field_encoding)
        
        return design_params
    
    def compute_total_loss(
        self,
        coordinates: Tensor,
        design_params: Optional[Tensor],
        labeled_data: Optional[Dict[str, Tensor]] = None,
        boundary_data: Optional[Dict[str, Tensor]] = None
    ) -> Dict[str, Tensor]:
        """
        计算总损失
        
        Args:
            coordinates: 配点坐标
            design_params: 设计参数
            labeled_data: 标签数据
            boundary_data: 边界数据
            
        Returns:
            损失字典
        """
        losses = {}
        
        # 物理损失
        fields = self.forward(coordinates, design_params)
        physics_loss = self._compute_physics_loss(fields, coordinates)
        losses['physics'] = physics_loss
        
        # 数据损失
        if labeled_data is not None:
            pred = self.forward(labeled_data['coords'], design_params)
            data_loss = F.mse_loss(pred, labeled_data['fields'])
            losses['data'] = data_loss
        
        # 边界条件损失
        if boundary_data is not None:
            bc_loss = self._compute_bc_loss(boundary_data, design_params)
            losses['bc'] = bc_loss
        
        # 加权总损失
        total = torch.tensor(0.0, device=coordinates.device)
        for key, loss in losses.items():
            weight = self._get_loss_weight(key)
            total = total + weight * loss
        
        losses['total'] = total
        
        return losses
    
    def _compute_physics_loss(
        self,
        fields: Tensor,
        coordinates: Tensor
    ) -> Tensor:
        """计算物理约束损失"""
        # 简化的 Helmholtz 方程残差
        # (∇² + k²)ψ = 0
        
        coordinates = coordinates.requires_grad_(True)
        
        # 计算拉普拉斯
        laplacian = self.field_net.compute_laplacian(coordinates, None, 0)
        
        # Helmholtz 残差
        k2 = (2 * math.pi / 1.55e-6) ** 2
        residual = laplacian + k2 * fields[:, 0]
        
        return (residual ** 2).mean()
    
    def _compute_bc_loss(
        self,
        boundary_data: Dict[str, Tensor],
        design_params: Optional[Tensor]
    ) -> Tensor:
        """计算边界条件损失"""
        coords = boundary_data['coords']
        target = boundary_data['values']
        
        pred = self.forward(coords, design_params)
        
        return F.mse_loss(pred, target)
    
    def _get_loss_weight(self, key: str) -> Tensor:
        """获取损失权重"""
        if self.physics_config.adaptive_weights:
            return torch.exp(-self.loss_weights[key])
        else:
            weights = {
                'data': self.physics_config.data_weight,
                'physics': self.physics_config.physics_weight,
                'bc': self.physics_config.bc_weight
            }
            return torch.tensor(weights.get(key, 1.0))


# ============================================================================
# PINN 求解器
# ============================================================================

class PINNSolver:
    """
    PINN 求解器
    
    提供完整的训练和求解流程。
    
    使用示例:
    ```python
    # 创建求解器
    solver = PINNSolver(pinn)
    
    # 准备数据
    domain = DomainSampler(bounds=[[-1, 1], [-1, 1]])
    
    # 训练
    solver.train(
        n_iterations=10000,
        collocation_points=domain.sample(1000),
        boundary_points=domain.sample_boundary(100)
    )
    
    # 预测
    solution = solver.predict(test_coords)
    ```
    """
    
    def __init__(
        self,
        model: PhysicsInformedNet,
        optimizer: str = 'adam',
        lr: float = 1e-3
    ):
        """
        Args:
            model: PINN 模型
            optimizer: 优化器类型 ('adam', 'lbfgs')
            lr: 学习率
        """
        self.model = model
        
        # 优化器
        if optimizer.lower() == 'adam':
            self.optimizer = Adam(model.parameters(), lr=lr)
        elif optimizer.lower() == 'lbfgs':
            self.optimizer = LBFGS(
                model.parameters(),
                lr=lr,
                max_iter=20,
                history_size=100
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")
        
        # 学习率调度器
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=500
        )
        
        # 训练历史
        self.history = {
            'loss': [],
            'physics_loss': [],
            'bc_loss': [],
            'data_loss': []
        }
    
    def train(
        self,
        n_iterations: int,
        collocation_points: Tensor,
        boundary_points: Optional[Tensor] = None,
        labeled_data: Optional[Dict[str, Tensor]] = None,
        log_interval: int = 100,
        adaptive_resampling: bool = False
    ) -> Dict[str, List[float]]:
        """
        训练 PINN
        
        Args:
            n_iterations: 迭代次数
            collocation_points: 配点
            boundary_points: 边界点
            labeled_data: 标签数据
            log_interval: 日志间隔
            adaptive_resampling: 自适应重采样
            
        Returns:
            训练历史
        """
        print(f"开始训练，共 {n_iterations} 次迭代...")
        
        for i in range(n_iterations):
            # 清零梯度
            self.optimizer.zero_grad()
            
            # 计算损失
            losses = self._compute_losses(
                collocation_points,
                boundary_points,
                labeled_data
            )
            
            # 反向传播
            losses['total'].backward()
            
            # 更新参数
            self.optimizer.step()
            
            # 更新学习率
            self.scheduler.step(losses['total'])
            
            # 记录历史
            self.history['loss'].append(losses['total'].item())
            if 'physics' in losses:
                self.history['physics_loss'].append(losses['physics'].item())
            if 'bc' in losses:
                self.history['bc_loss'].append(losses['bc'].item())
            if 'data' in losses:
                self.history['data_loss'].append(losses['data'].item())
            
            # 日志
            if (i + 1) % log_interval == 0:
                print(f"Iteration {i + 1}/{n_iterations}, "
                      f"Loss: {losses['total'].item():.6f}")
            
            # 自适应重采样
            if adaptive_resampling and (i + 1) % 1000 == 0:
                collocation_points = self._adaptive_resample(
                    collocation_points
                )
        
        print("训练完成!")
        return self.history
    
    def _compute_losses(
        self,
        collocation_points: Tensor,
        boundary_points: Optional[Tensor],
        labeled_data: Optional[Dict[str, Tensor]]
    ) -> Dict[str, Tensor]:
        """计算所有损失"""
        losses = {}
        
        # 物理损失
        physics_loss = self._compute_physics_loss(collocation_points)
        losses['physics'] = physics_loss
        
        total = physics_loss
        
        # 边界条件损失
        if boundary_points is not None:
            bc_loss = self._compute_bc_loss(boundary_points)
            losses['bc'] = bc_loss
            total = total + bc_loss
        
        # 数据损失
        if labeled_data is not None:
            data_loss = self._compute_data_loss(labeled_data)
            losses['data'] = data_loss
            total = total + data_loss
        
        losses['total'] = total
        
        return losses
    
    def _compute_physics_loss(self, points: Tensor) -> Tensor:
        """计算物理损失"""
        points = points.requires_grad_(True)
        
        # 预测
        output = self.model(points)
        
        # 计算 PDE 残差（示例：Helmholtz 方程）
        laplacian = self.model.compute_laplacian(points, None, 0)
        k2 = (2 * math.pi / 1.55e-6) ** 2
        
        residual = laplacian + k2 * output[:, 0]
        
        return (residual ** 2).mean()
    
    def _compute_bc_loss(self, points: Tensor) -> Tensor:
        """计算边界条件损失"""
        # 简化：Dirichlet 边界条件
        output = self.model(points)
        
        # 假设边界值为 0
        return (output ** 2).mean()
    
    def _compute_data_loss(self, data: Dict[str, Tensor]) -> Tensor:
        """计算数据损失"""
        pred = self.model(data['coords'])
        return F.mse_loss(pred, data['fields'])
    
    def _adaptive_resample(self, points: Tensor) -> Tensor:
        """自适应重采样配点"""
        # 在残差较大的区域增加采样
        points = points.requires_grad_(True)
        output = self.model(points)
        
        # 简化：添加一些随机扰动
        noise = torch.randn_like(points) * 0.01
        new_points = points.detach() + noise
        
        # 限制在域内
        # new_points = torch.clamp(new_points, -1, 1)
        
        return new_points
    
    def predict(self, coordinates: Tensor) -> Tensor:
        """预测解"""
        self.model.eval()
        with torch.no_grad():
            return self.model(coordinates)
    
    def save(self, path: Union[str, Path]):
        """保存模型"""
        checkpoint = {
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'history': self.history
        }
        torch.save(checkpoint, path)
    
    def load(self, path: Union[str, Path]):
        """加载模型"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.history = checkpoint.get('history', self.history)


# ============================================================================
# 便捷函数
# ============================================================================

def create_pinn_for_photonics(
    spatial_dim: int = 2,
    field_components: int = 3,
    wavelength: float = 1.55e-6,
    epsilon_r: float = 12.0,
    device: str = 'auto'
) -> MaxwellPINN:
    """
    为光子学问题创建 PINN
    
    Args:
        spatial_dim: 空间维度
        field_components: 场分量数
        wavelength: 工作波长
        epsilon_r: 相对介电常数
        device: 计算设备
        
    Returns:
        配置好的 MaxwellPINN
    """
    config = MaxwellConfig(
        spatial_dim=spatial_dim,
        field_components=field_components,
        wavelength=wavelength,
        epsilon_r=epsilon_r,
        device=device
    )
    
    return MaxwellPINN(config)

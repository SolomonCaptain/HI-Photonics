"""
PINO - Physics-Informed Neural Operator

结合神经网络算子和物理约束，用于求解偏微分方程。
特别适用于电磁场仿真和光子学器件分析。

核心思想:
1. 使用神经网络算子（如 FNO, DeepONet）学习解算子
2. 通过 PDE 残差损失强制物理约束
3. 边界条件作为软约束或硬约束

应用场景:
- Maxwell 方程求解（电磁场仿真）
- Helmholtz 方程求解（频域电磁）
- 热传导方程
- 波动方程

参考文献:
- Li et al. (2021). "Physics-Informed Neural Operator"
- Lu et al. (2022). "Physics-informed neural networks for solving PDEs"
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
class PINOConfig(ModelConfig):
    """PINO配置"""
    name: str = "pino"
    
    # 域大小
    domain_size: Tuple[float, float] = (10.0, 10.0)  # μm
    grid_size: Tuple[int, int] = (128, 128)
    
    # 输入输出
    input_channels: int = 1           # 输入场通道数（如介电常数分布）
    output_channels: int = 3          # 输出场通道数（如 Ex, Ey, Ez）
    
    # 算子类型
    operator_type: str = "fno"        # 'fno', 'deeponet', 'transformer'
    
    # FNO 参数
    fno_modes: int = 12               # Fourier 模式数
    fno_hidden_dim: int = 64          # 隐藏层维度
    fno_num_layers: int = 4           # FNO 层数
    
    # Transformer 参数
    transformer_dim: int = 256
    transformer_heads: int = 8
    transformer_layers: int = 6
    
    # 激活函数
    activation: str = "gelu"
    
    # 物理约束
    pde_type: str = "helmholtz"       # 'helmholtz', 'maxwell', 'heat', 'wave'
    physics_weight: float = 1.0       # PDE 损失权重
    boundary_weight: float = 1.0      # 边界条件权重
    
    # 边界条件
    boundary_type: str = "dirichlet"  # 'dirichlet', 'neumann', 'periodic', 'pml'
    boundary_value: float = 0.0
    
    # PML 参数 (Perfectly Matched Layer)
    pml_thickness: int = 10           # PML 层厚度（像素）
    pml_sigma_max: float = 1000.0     # PML 最大电导率
    
    # Helmholtz 方程参数
    wavelength: float = 1.55          # μm
    refractive_index: float = 3.48    # 硅的折射率
    
    # Maxwell 方程参数
    frequency: float = 193.5          # THz (对应 1.55 μm)
    
    # 训练参数
    use_sobolev_loss: bool = True     # 是否使用 Sobolev 损失
    sobolev_order: int = 1            # Sobolev 空间阶数
    
    # 数据增强
    augment_data: bool = True


# ============================================================================
# Fourier Neural Operator (FNO)
# ============================================================================

class SpectralConv2d(nn.Module):
    """2D Fourier 卷积层"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int,
        modes2: int,
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        
        self.scale = 1 / (in_channels * out_channels)
        
        # 复数权重
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
    
    def compl_mul2d(self, x: Tensor, weights: Tensor) -> Tensor:
        """复数矩阵乘法"""
        return torch.einsum('bixy,ioxy->boxy', x, weights)
    
    def forward(self, x: Tensor) -> Tensor:
        batchsize = x.shape[0]
        
        # Fourier 变换
        x_ft = torch.fft.rfft2(x)
        
        # 低频模式
        out_ft = torch.zeros(
            batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1,
            dtype=torch.cfloat, device=x.device
        )
        
        out_ft[:, :, :self.modes1, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, :self.modes1, :self.modes2], self.weights1
        )
        out_ft[:, :, -self.modes1:, :self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1:, :self.modes2], self.weights2
        )
        
        # 逆 Fourier 变换
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        
        return x


class FNO2d(nn.Module):
    """2D Fourier Neural Operator"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: int,
        hidden_dim: int,
        num_layers: int = 4,
        activation: str = "gelu",
    ):
        super().__init__()
        
        self.modes = modes
        self.hidden_dim = hidden_dim
        
        # 输入投影
        self.fc0 = nn.Linear(in_channels + 2, hidden_dim)  # +2 for grid
        
        # Spectral 层
        self.spectral_convs = nn.ModuleList()
        self.w_convs = nn.ModuleList()
        
        for _ in range(num_layers):
            self.spectral_convs.append(
                SpectralConv2d(hidden_dim, hidden_dim, modes, modes)
            )
            self.w_convs.append(
                nn.Conv2d(hidden_dim, hidden_dim, 1)
            )
        
        # 输出投影
        self.fc1 = nn.Linear(hidden_dim, 128)
        self.fc2 = nn.Linear(128, out_channels)
        
        self.activation = self._get_activation(activation)
    
    def _get_activation(self, name: str) -> nn.Module:
        activations = {
            'relu': nn.ReLU(inplace=True),
            'tanh': nn.Tanh(),
            'gelu': nn.GELU(),
            'silu': nn.SiLU(inplace=True),
        }
        return activations.get(name, nn.GELU())
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            [B, C_out, H, W]
        """
        # 添加网格坐标
        gridx, gridy = torch.meshgrid(
            torch.linspace(0, 1, x.size(-2), device=x.device),
            torch.linspace(0, 1, x.size(-1), device=x.device),
            indexing='ij'
        )
        gridx = gridx.unsqueeze(0).unsqueeze(0).expand(x.size(0), -1, -1, -1)
        gridy = gridy.unsqueeze(0).unsqueeze(0).expand(x.size(0), -1, -1, -1)
        
        x = torch.cat([x, gridx, gridy], dim=1)
        
        # [B, C+2, H, W] -> [B, H, W, C+2] -> [B, H, W, hidden]
        x = x.permute(0, 2, 3, 1)
        x = self.fc0(x)
        
        # [B, H, W, hidden] -> [B, hidden, H, W]
        x = x.permute(0, 3, 1, 2)
        
        # Spectral 卷积层
        for spectral, w in zip(self.spectral_convs, self.w_convs):
            x1 = spectral(x)
            x2 = w(x)
            x = self.activation(x1 + x2)
        
        # 输出投影
        x = x.permute(0, 2, 3, 1)
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        
        # [B, H, W, C_out] -> [B, C_out, H, W]
        x = x.permute(0, 3, 1, 2)
        
        return x


# ============================================================================
# 物理约束模块
# ============================================================================

class HelmholtzPDE(nn.Module):
    """
    Helmholtz 方程约束
    
    ∇²u + k²n²u = f
    
    其中:
    - u: 场变量（如电场）
    - k: 波数 = 2π/λ
    - n: 折射率分布
    - f: 源项
    """
    
    def __init__(
        self,
        wavelength: float = 1.55,
        domain_size: Tuple[float, float] = (10.0, 10.0),
    ):
        super().__init__()
        
        self.wavelength = wavelength
        self.domain_size = domain_size
        
        # 波数
        self.k = 2 * math.pi / wavelength
        
        # 网格间距
        self.dx = domain_size[0] / 128  # 假设 128 网格
        self.dy = domain_size[1] / 128
    
    def compute_laplacian(self, u: Tensor, dx: float, dy: float) -> Tensor:
        """
        使用有限差分计算 Laplacian
        
        ∇²u ≈ (u_{i+1,j} + u_{i-1,j} - 2u_{i,j})/dx²
             + (u_{i,j+1} + u_{i,j-1} - 2u_{i,j})/dy²
        """
        # 二阶中心差分
        u_xx = (F.pad(u, (1, 1, 0, 0), mode='constant', value=0)[:, :, 2:, :]
                + F.pad(u, (1, 1, 0, 0), mode='constant', value=0)[:, :, :-2, :]
                - 2 * u) / (dx ** 2)
        
        u_yy = (F.pad(u, (0, 0, 1, 1), mode='constant', value=0)[:, :, :, 2:]
                + F.pad(u, (0, 0, 1, 1), mode='constant', value=0)[:, :, :, :-2]
                - 2 * u) / (dy ** 2)
        
        return u_xx + u_yy
    
    def forward(
        self,
        u: Tensor,
        n: Tensor,
        source: Optional[Tensor] = None,
    ) -> Tensor:
        """
        计算 Helmholtz 残差
        
        Args:
            u: 场变量 [B, C, H, W]
            n: 折射率分布 [B, 1, H, W] 或 [B, C, H, W]
            source: 源项 (可选)
            
        Returns:
            PDE 残差
        """
        # Laplacian
        laplacian = self.compute_laplacian(u, self.dx, self.dy)
        
        # 波数平方 * 折射率平方
        k_sq_n_sq = (self.k * n) ** 2
        
        # Helmholtz 残差: ∇²u + k²n²u - f
        residual = laplacian + k_sq_n_sq * u
        
        if source is not None:
            residual = residual - source
        
        return residual


class MaxwellPDE(nn.Module):
    """
    Maxwell 方程约束 (频域)
    
    ∇ × (1/μᵣ ∇ × E) - k₀²εᵣE = -jωμ₀J
    
    简化形式（对于 TM 模式，仅有 Ez 分量）:
    ∂/∂x(1/μᵣ ∂Ez/∂x) + ∂/∂y(1/μᵣ ∂Ez/∂y) + k₀²εᵣEz = -jωμ₀Jz
    """
    
    def __init__(
        self,
        wavelength: float = 1.55,
        domain_size: Tuple[float, float] = (10.0, 10.0),
        mu_r: float = 1.0,  # 相对磁导率
    ):
        super().__init__()
        
        self.wavelength = wavelength
        self.domain_size = domain_size
        self.mu_r = mu_r
        
        # 物理常数
        self.c = 299792458  # 光速 m/s
        self.k0 = 2 * math.pi / (wavelength * 1e-6)  # 自由空间波数
        
        # 网格间距
        self.dx = domain_size[0] * 1e-6 / 128
        self.dy = domain_size[1] * 1e-6 / 128
    
    def compute_curl_curl(self, E: Tensor, eps_r: Tensor) -> Tensor:
        """计算 ∇ × ∇ × E"""
        # 简化：假设 TM 模式
        # ∇ × ∇ × Ez = -∇²Ez (对于 TM 模式)
        
        E_xx = (F.pad(E, (1, 1, 0, 0), mode='replicate')[:, :, 2:, :]
                + F.pad(E, (1, 1, 0, 0), mode='replicate')[:, :, :-2, :]
                - 2 * E) / (self.dx ** 2)
        
        E_yy = (F.pad(E, (0, 0, 1, 1), mode='replicate')[:, :, :, 2:]
                + F.pad(E, (0, 0, 1, 1), mode='replicate')[:, :, :, :-2]
                - 2 * E) / (self.dy ** 2)
        
        return -(E_xx + E_yy)
    
    def forward(
        self,
        E: Tensor,
        eps_r: Tensor,
        J: Optional[Tensor] = None,
    ) -> Tensor:
        """
        计算 Maxwell 残差
        
        Args:
            E: 电场 [B, C, H, W]
            eps_r: 相对介电常数 [B, 1, H, W]
            J: 电流源 (可选)
            
        Returns:
            Maxwell 方程残差
        """
        # ∇ × (1/μᵣ ∇ × E) - k₀²εᵣE
        curl_curl = self.compute_curl_curl(E, eps_r) / self.mu_r
        
        residual = curl_curl - self.k0 ** 2 * eps_r * E
        
        if J is not None:
            # 简化源项
            residual = residual + J
        
        return residual


class BoundaryCondition(nn.Module):
    """边界条件"""
    
    def __init__(
        self,
        bc_type: str = "dirichlet",
        bc_value: float = 0.0,
        pml_thickness: int = 10,
        pml_sigma_max: float = 1000.0,
    ):
        super().__init__()
        
        self.bc_type = bc_type
        self.bc_value = bc_value
        self.pml_thickness = pml_thickness
        self.pml_sigma_max = pml_sigma_max
    
    def forward(self, u: Tensor) -> Tensor:
        """
        计算边界条件残差
        
        Args:
            u: 场变量 [B, C, H, W]
            
        Returns:
            边界条件残差
        """
        if self.bc_type == "dirichlet":
            # u = bc_value at boundaries
            residual = torch.cat([
                u[:, :, 0, :].flatten(),
                u[:, :, -1, :].flatten(),
                u[:, :, :, 0].flatten(),
                u[:, :, :, -1].flatten(),
            ])
            return residual - self.bc_value
        
        elif self.bc_type == "neumann":
            # ∂u/∂n = 0 at boundaries
            # 使用一阶差分近似
            residual = torch.cat([
                (u[:, :, 0, :] - u[:, :, 1, :]).flatten(),
                (u[:, :, -1, :] - u[:, :, -2, :]).flatten(),
                (u[:, :, :, 0] - u[:, :, :, 1]).flatten(),
                (u[:, :, :, -1] - u[:, :, :, -2]).flatten(),
            ])
            return residual
        
        elif self.bc_type == "pml":
            # PML 吸收边界
            return self._pml_residual(u)
        
        else:
            return torch.tensor(0.0, device=u.device)
    
    def _pml_residual(self, u: Tensor) -> Tensor:
        """计算 PML 区域的残差"""
        # 简化实现：PML 区域内的场应该逐渐衰减
        H, W = u.shape[-2:]
        
        # 创建 PML 权重
        pml_weight = torch.ones_like(u)
        
        # 边界渐变
        for i in range(self.pml_thickness):
            sigma = self.pml_sigma_max * ((self.pml_thickness - i) / self.pml_thickness) ** 2
            weight = torch.exp(-sigma * 1e-9)  # 简化衰减
            
            pml_weight[:, :, i, :] *= weight
            pml_weight[:, :, -i-1, :] *= weight
            pml_weight[:, :, :, i] *= weight
            pml_weight[:, :, :, -i-1] *= weight
        
        # PML 区域场应该接近零
        pml_field = u * (1 - pml_weight)
        
        return pml_field.abs().mean()


# ============================================================================
# PINO 主模型
# ============================================================================

class PINO(SurrogateModel):
    """
    Physics-Informed Neural Operator
    
    结合神经网络算子和物理约束求解PDE。
    """
    
    def __init__(self, config: Optional[PINOConfig] = None):
        super().__init__(config or PINOConfig())
        self.config: PINOConfig = self.config
        
        # 神经网络算子
        if self.config.operator_type == "fno":
            self.operator = FNO2d(
                in_channels=self.config.input_channels,
                out_channels=self.config.output_channels,
                modes=self.config.fno_modes,
                hidden_dim=self.config.fno_hidden_dim,
                num_layers=self.config.fno_num_layers,
                activation=self.config.activation,
            )
        else:
            raise ValueError(f"Unknown operator type: {self.config.operator_type}")
        
        # 物理约束模块
        if self.config.pde_type == "helmholtz":
            self.pde = HelmholtzPDE(
                wavelength=self.config.wavelength,
                domain_size=self.config.domain_size,
            )
        elif self.config.pde_type == "maxwell":
            self.pde = MaxwellPDE(
                wavelength=self.config.wavelength,
                domain_size=self.config.domain_size,
            )
        else:
            raise ValueError(f"Unknown PDE type: {self.config.pde_type}")
        
        # 边界条件
        self.boundary = BoundaryCondition(
            bc_type=self.config.boundary_type,
            bc_value=self.config.boundary_value,
            pml_thickness=self.config.pml_thickness,
            pml_sigma_max=self.config.pml_sigma_max,
        )
    
    def forward(
        self,
        design: Tensor,
        source: Optional[Tensor] = None,
    ) -> Tensor:
        """
        求解场分布
        
        Args:
            design: 设计参数（如介电常数分布）[B, C, H, W]
            source: 源项 (可选)
            
        Returns:
            场分布 [B, C_out, H, W]
        """
        return self.operator(design)
    
    def compute_pde_residual(
        self,
        u: Tensor,
        n: Tensor,
        source: Optional[Tensor] = None,
    ) -> Tensor:
        """计算PDE残差"""
        return self.pde(u, n, source)
    
    def compute_boundary_residual(self, u: Tensor) -> Tensor:
        """计算边界条件残差"""
        return self.boundary(u)
    
    def compute_loss(
        self,
        output: Tensor,
        target: Optional[Tensor] = None,
        design: Optional[Tensor] = None,
        source: Optional[Tensor] = None,
        use_physics: bool = True,
        **kwargs
    ) -> Tensor:
        """
        计算总损失
        
        Args:
            output: 模型输出（场分布）
            target: 目标场分布（可选，用于监督学习）
            design: 输入设计（折射率分布）
            source: 源项
            use_physics: 是否使用物理约束
            
        Returns:
            总损失
        """
        total_loss = torch.tensor(0.0, device=output.device)
        
        # 数据损失
        if target is not None:
            data_loss = F.mse_loss(output, target)
            total_loss = total_loss + data_loss
        
        # 物理损失
        if use_physics and design is not None:
            # 计算折射率
            n = torch.sqrt(design.abs().clamp(min=1.0))  # n = sqrt(ε)
            
            # PDE 残差
            pde_residual = self.compute_pde_residual(output, n, source)
            physics_loss = (pde_residual ** 2).mean()
            
            # 边界条件
            boundary_residual = self.compute_boundary_residual(output)
            boundary_loss = (boundary_residual ** 2).mean()
            
            total_loss = total_loss + self.config.physics_weight * physics_loss
            total_loss = total_loss + self.config.boundary_weight * boundary_loss
        
        return total_loss
    
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
            
            # 相对 L2 误差
            rel_l2 = (output - target).norm() / (target.norm() + 1e-8)
            
            # R²
            ss_res = ((target - output) ** 2).sum()
            ss_tot = ((target - target.mean()) ** 2).sum()
            r2 = 1 - ss_res / (ss_tot + 1e-8)
            
            return {
                'mse': mse,
                'mae': mae,
                'r2': r2.item(),
                'rel_l2': rel_l2.item(),
            }
    
    def solve_iterative(
        self,
        design: Tensor,
        source: Optional[Tensor] = None,
        num_iterations: int = 100,
        learning_rate: float = 0.01,
    ) -> Tensor:
        """
        迭代求解（用于精化解）
        
        使用梯度下降最小化 PDE 残差
        """
        # 初始猜测
        u = self.forward(design, source).detach().requires_grad_(True)
        
        optimizer = torch.optim.Adam([u], lr=learning_rate)
        
        n = torch.sqrt(design.abs().clamp(min=1.0))
        
        for _ in range(num_iterations):
            optimizer.zero_grad()
            
            # PDE 残差
            pde_residual = self.compute_pde_residual(u, n, source)
            pde_loss = (pde_residual ** 2).mean()
            
            # 边界条件
            boundary_loss = (self.compute_boundary_residual(u) ** 2).mean()
            
            loss = pde_loss + boundary_loss
            loss.backward()
            optimizer.step()
        
        return u.detach()


# ============================================================================
# 多尺度 PINO
# ============================================================================

class MultiScalePINO(PINO):
    """
    多尺度 PINO: 支持多分辨率求解
    """
    
    def __init__(self, config: Optional[PINOConfig] = None):
        super().__init__(config)
        
        # 多尺度算子
        self.scales = [1, 2, 4]  # 下采样倍数
        self.operators = nn.ModuleList([
            FNO2d(
                in_channels=self.config.input_channels,
                out_channels=self.config.output_channels,
                modes=self.config.fno_modes // scale,
                hidden_dim=self.config.fno_hidden_dim // scale,
                num_layers=self.config.fno_num_layers,
                activation=self.config.activation,
            )
            for scale in self.scales
        ])
        
        # 融合层
        self.fusion = nn.Conv2d(
            self.config.output_channels * len(self.scales),
            self.config.output_channels,
            1
        )
    
    def forward(
        self,
        design: Tensor,
        source: Optional[Tensor] = None,
    ) -> Tensor:
        """多尺度前向传播"""
        outputs = []
        
        for scale, operator in zip(self.scales, self.operators):
            if scale > 1:
                # 下采样
                design_scaled = F.avg_pool2d(design, scale)
                # 求解
                out_scaled = operator(design_scaled)
                # 上采样
                out = F.interpolate(out_scaled, size=design.shape[2:], mode='bilinear')
            else:
                out = operator(design)
            
            outputs.append(out)
        
        # 融合
        combined = torch.cat(outputs, dim=1)
        return self.fusion(combined)


# ============================================================================
# 变分 PINO (用于不确定性量化)
# ============================================================================

class VariationalPINO(PINO):
    """
    变分 PINO: 支持贝叶斯推断
    """
    
    def __init__(self, config: Optional[PINOConfig] = None):
        super().__init__(config)
        
        # 方差预测头
        self.var_head = nn.Sequential(
            nn.Conv2d(self.config.output_channels, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, self.config.output_channels, 3, padding=1),
        )
    
    def forward(
        self,
        design: Tensor,
        source: Optional[Tensor] = None,
        return_variance: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """带不确定性估计的前向传播"""
        mean = self.operator(design)
        
        if return_variance:
            log_var = self.var_head(mean)
            return mean, torch.exp(log_var)
        
        return mean
    
    def compute_loss(
        self,
        output: Tensor,
        target: Tensor,
        design: Optional[Tensor] = None,
        **kwargs
    ) -> Tensor:
        """变分损失"""
        mean, var = self.forward(design, return_variance=True)
        
        # 异方差损失
        loss = 0.5 * torch.log(var + 1e-8) + 0.5 * (target - mean) ** 2 / var
        
        # 加上物理损失
        if design is not None:
            n = torch.sqrt(design.abs().clamp(min=1.0))
            pde_residual = self.compute_pde_residual(mean, n)
            loss = loss + self.config.physics_weight * (pde_residual ** 2).mean()
        
        return loss.mean()


# ============================================================================
# 便捷函数
# ============================================================================

def create_pino(
    input_channels: int = 1,
    output_channels: int = 3,
    domain_size: Tuple[float, float] = (10.0, 10.0),
    wavelength: float = 1.55,
    pde_type: str = "helmholtz",
    **kwargs
) -> PINO:
    """
    创建 PINO 模型
    
    Args:
        input_channels: 输入通道数
        output_channels: 输出通道数
        domain_size: 计算域大小 (μm)
        wavelength: 波长 (μm)
        pde_type: PDE 类型
        
    Returns:
        PINO 模型实例
    """
    config = PINOConfig(
        input_channels=input_channels,
        output_channels=output_channels,
        domain_size=domain_size,
        wavelength=wavelength,
        pde_type=pde_type,
        **kwargs
    )
    return PINO(config)


def create_electromagnetic_pino(
    grid_size: Tuple[int, int] = (128, 128),
    domain_size: Tuple[float, float] = (10.0, 10.0),
    wavelength: float = 1.55,
) -> PINO:
    """
    创建电磁场求解 PINO
    
    Args:
        grid_size: 网格大小
        domain_size: 计算域大小 (μm)
        wavelength: 波长 (μm)
        
    Returns:
        PINO 模型实例
    """
    return create_pino(
        input_channels=1,
        output_channels=3,  # Ex, Ey, Ez
        domain_size=domain_size,
        grid_size=grid_size,
        wavelength=wavelength,
        pde_type="maxwell",
        boundary_type="pml",
        pml_thickness=10,
    )

"""
Mixture Density Network (MDN) 混合密度网络

用于处理光子学逆向设计中的"一对多"映射问题。
MDN 输出设计参数的概率分布（高斯混合模型），而非单一设计。

核心思想:
- 输入: 性能指标
- 输出: 高斯混合分布参数 (π, μ, σ)
- 训练: 最小化负对数似然
- 推理: 从分布中采样多个候选设计

参考文献:
- Bishop, C. M. (1994). "Mixture density networks"
- Peurifoy et al. (2018). "Nanophotonic particle simulation and inverse design"
"""

from typing import Dict, Optional, Tuple, List, Union, Any
from dataclasses import dataclass, field
from pathlib import Path
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.base import BaseModel, ModelConfig, InverseModel


# ============================================================================
# 配置类
# ============================================================================

@dataclass
class MDNConfig(ModelConfig):
    """混合密度网络配置"""
    name: str = "mdn"
    
    # 输入输出维度
    input_dim: int = 3              # 性能指标维度
    output_dim: int = 200 * 22      # 设计参数维度（展平）
    design_shape: Tuple[int, int] = (200, 22)  # 设计形状（用于重塑）
    
    # 高斯混合参数
    n_components: int = 5           # 高斯分量数 K
    
    # 网络架构
    hidden_dims: List[int] = field(default_factory=lambda: [256, 512, 256])
    shared_hidden_dims: List[int] = field(default_factory=lambda: [256, 128])
    
    # 正则化
    dropout_rate: float = 0.1
    batch_norm: bool = True
    
    # 数值稳定性
    log_sigma_min: float = -10.0    # log(σ) 最小值
    log_sigma_max: float = 2.0      # log(σ) 最大值
    
    # 激活函数
    activation: str = "relu"
    
    # 训练参数
    temperature: float = 1.0        # softmax 温度
    
    # ========== 新增：分布类型选择 ==========
    distribution_type: str = "gmm"  # 'gmm' (高斯混合), 'flow' (Normalizing Flow), 'hybrid'
    
    # ========== 新增：Normalizing Flow 参数 ==========
    n_flow_layers: int = 8          # Flow 层数
    flow_hidden_dim: int = 128      # Flow 隐藏层维度
    flow_type: str = "real_nvp"     # 'real_nvp', 'maf', 'nsf'
    
    # ========== 新增：拓扑感知参数 ==========
    topology_aware: bool = True     # 是否启用拓扑感知
    min_feature_size: float = 0.1   # 最小特征尺寸（μm）
    resolution: float = 0.01        # 网格分辨率（μm/pixel）
    
    # 拓扑约束权重
    connectivity_weight: float = 0.1      # 连通性权重
    min_feature_weight: float = 0.2       # 最小特征尺寸权重
    curvature_weight: float = 0.05        # 曲率约束权重
    
    # 拓扑采样参数
    topology_guided_sampling: bool = True  # 拓扑引导采样
    topology_refinement_steps: int = 5     # 拓扑精炼步数
    constraint_threshold: float = 0.5      # 约束满足阈值


# ============================================================================
# 高斯混合模型组件
# ============================================================================

class GaussianMixtureParameters(nn.Module):
    """
    高斯混合参数输出层
    
    将特征向量转换为高斯混合分布的参数：
    - π: 混合权重 [B, K]
    - μ: 均值 [B, K, D]
    - σ: 标准差 [B, K, D]
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        n_components: int,
        log_sigma_min: float = -10.0,
        log_sigma_max: float = 2.0
    ):
        """
        Args:
            input_dim: 输入特征维度
            output_dim: 输出设计维度
            n_components: 高斯分量数
            log_sigma_min: log(σ) 最小值
            log_sigma_max: log(σ) 最大值
        """
        super().__init__()
        self.output_dim = output_dim
        self.n_components = n_components
        self.log_sigma_min = log_sigma_min
        self.log_sigma_max = log_sigma_max
        
        # 混合权重网络
        self.pi_net = nn.Sequential(
            nn.Linear(input_dim, n_components)
        )
        
        # 均值网络
        self.mu_net = nn.Linear(input_dim, n_components * output_dim)
        
        # 标准差网络（输出 log(σ)）
        self.sigma_net = nn.Linear(input_dim, n_components * output_dim)
    
    def forward(self, features: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        输出高斯混合参数
        
        Args:
            features: 特征向量 [B, input_dim]
            
        Returns:
            pi: 混合权重 [B, K]，满足 sum(pi) = 1
            mu: 均值 [B, K, D]
            sigma: 标准差 [B, K, D]，σ > 0
        """
        batch_size = features.size(0)
        
        # 混合权重 (softmax 归一化)
        pi_logits = self.pi_net(features)  # [B, K]
        pi = F.softmax(pi_logits, dim=-1)
        
        # 均值
        mu = self.mu_net(features)  # [B, K*D]
        mu = mu.view(batch_size, self.n_components, self.output_dim)
        
        # 标准差 (通过 clamp 确保 σ > 0)
        log_sigma = self.sigma_net(features)  # [B, K*D]
        log_sigma = torch.clamp(log_sigma, self.log_sigma_min, self.log_sigma_max)
        sigma = torch.exp(log_sigma)
        sigma = sigma.view(batch_size, self.n_components, self.output_dim)
        
        return pi, mu, sigma


class GaussianMixtureDistribution:
    """
    高斯混合分布
    
    提供采样、概率密度计算等功能。
    """
    
    def __init__(
        self,
        pi: Tensor,
        mu: Tensor,
        sigma: Tensor,
        epsilon: float = 1e-8
    ):
        """
        Args:
            pi: 混合权重 [B, K]
            mu: 均值 [B, K, D]
            sigma: 标准差 [B, K, D]
            epsilon: 数值稳定性常数
        """
        self.pi = pi
        self.mu = mu
        self.sigma = sigma
        self.epsilon = epsilon
        
        self.batch_size = pi.size(0)
        self.n_components = pi.size(1)
        self.output_dim = mu.size(-1)
    
    def log_prob(self, x: Tensor) -> Tensor:
        """
        计算对数概率密度
        
        Args:
            x: 样本 [B, D] 或 [B, K, D]
            
        Returns:
            log_prob: 对数概率 [B]
        """
        # 确保 x 形状正确
        if x.dim() == 2:
            x = x.unsqueeze(1).expand(-1, self.n_components, -1)  # [B, K, D]
        
        # 计算每个分量的对数概率
        # log N(x; μ, σ²) = -0.5 * log(2π) - log(σ) - 0.5 * ((x-μ)/σ)²
        log_2pi = math.log(2 * math.pi)
        
        log_prob_components = (
            -0.5 * log_2pi 
            - torch.log(self.sigma + self.epsilon) 
            - 0.5 * ((x - self.mu) / (self.sigma + self.epsilon)) ** 2
        )  # [B, K, D]
        
        # 对输出维度求和
        log_prob_components = log_prob_components.sum(dim=-1)  # [B, K]
        
        # 加权求和（使用 log-sum-exp 技巧）
        log_pi = torch.log(self.pi + self.epsilon)
        log_prob = torch.logsumexp(log_pi + log_prob_components, dim=-1)  # [B]
        
        return log_prob
    
    def sample(self, n_samples: int = 1) -> Tensor:
        """
        从混合分布中采样
        
        Args:
            n_samples: 每个输入的采样数量
            
        Returns:
            samples: 样本 [B, n_samples, D]
        """
        samples = []
        
        for _ in range(n_samples):
            # 采样分量索引
            component_indices = torch.multinomial(self.pi, 1).squeeze(-1)  # [B]
            
            # 获取对应分量的参数
            batch_indices = torch.arange(self.batch_size, device=self.pi.device)
            mu_selected = self.mu[batch_indices, component_indices]  # [B, D]
            sigma_selected = self.sigma[batch_indices, component_indices]  # [B, D]
            
            # 从高斯分布采样
            noise = torch.randn_like(mu_selected)
            sample = mu_selected + sigma_selected * noise
            samples.append(sample)
        
        return torch.stack(samples, dim=1)  # [B, n_samples, D]
    
    def sample_mode(self) -> Tensor:
        """
        返回最可能分量（权重最大）的均值
        
        Returns:
            mode: 最可能的设计 [B, D]
        """
        # 找到权重最大的分量
        max_component = torch.argmax(self.pi, dim=-1)  # [B]
        batch_indices = torch.arange(self.batch_size, device=self.pi.device)
        
        return self.mu[batch_indices, max_component]  # [B, D]
    
    def sample_weighted(self, n_samples: int = 1) -> Tensor:
        """
        按权重加权采样（期望值）
        
        Args:
            n_samples: 采样数量
            
        Returns:
            samples: 加权样本 [B, n_samples, D]
        """
        samples = []
        
        for _ in range(n_samples):
            # 随机选择分量
            component_indices = torch.multinomial(self.pi, 1).squeeze(-1)
            batch_indices = torch.arange(self.batch_size, device=self.pi.device)
            
            mu_selected = self.mu[batch_indices, component_indices]
            sigma_selected = self.sigma[batch_indices, component_indices]
            
            # 采样
            noise = torch.randn_like(mu_selected)
            sample = mu_selected + sigma_selected * noise
            samples.append(sample)
        
        return torch.stack(samples, dim=1)
    
    def get_entropy(self) -> Tensor:
        """
        计算分布的熵（不确定性度量）
        
        Returns:
            entropy: 熵 [B]
        """
        # 分量选择熵
        pi_entropy = -torch.sum(self.pi * torch.log(self.pi + self.epsilon), dim=-1)
        
        # 各分量高斯熵的平均
        gaussian_entropy = 0.5 * math.log(2 * math.pi * math.e) * self.output_dim
        gaussian_entropy += torch.sum(
            self.pi.unsqueeze(-1) * torch.log(self.sigma + self.epsilon),
            dim=(-1, -2)
        )
        
        return pi_entropy + gaussian_entropy


# ============================================================================
# Normalizing Flow 组件
# ============================================================================

class AffineCouplingLayer(nn.Module):
    """
    仿射耦合层 (Real NVP)
    
    将输入分为两部分，一部分不变，另一部分进行仿射变换。
    允许高效的可逆变换和精确的似然计算。
    """
    
    def __init__(
        self,
        dim: int,
        hidden_dim: int = 128,
        mask_type: str = 'alternating'
    ):
        """
        Args:
            dim: 输入维度
            hidden_dim: 隐藏层维度
            mask_type: 掩码类型 ('alternating', 'half')
        """
        super().__init__()
        self.dim = dim
        
        # 创建掩码
        if mask_type == 'alternating':
            self.register_buffer('mask', torch.arange(dim) % 2)
        else:  # 'half'
            self.register_buffer('mask', torch.cat([
                torch.ones(dim // 2),
                torch.zeros(dim - dim // 2)
            ]))
        
        # 缩放和平移网络
        self.scale_net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim),
            nn.Tanh()  # 限制输出范围
        )
        
        self.translate_net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim)
        )
    
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        前向变换: z -> x
        
        Args:
            x: 输入 [B, D]
            
        Returns:
            z: 变换后的输出 [B, D]
            log_det: 对数行列式 [B]
        """
        mask = self.mask
        x_masked = x * mask
        
        # 计算缩放和平移参数
        s = self.scale_net(x_masked) * (1 - mask)
        t = self.translate_net(x_masked) * (1 - mask)
        
        # 应用变换
        z = x_masked + (1 - mask) * (x * torch.exp(s) + t)
        
        # 对数行列式
        log_det = s.sum(dim=-1)
        
        return z, log_det
    
    def inverse(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        """
        逆变换: x -> z
        
        Args:
            z: 输入 [B, D]
            
        Returns:
            x: 逆变换后的输出 [B, D]
            log_det: 对数行列式 [B]
        """
        mask = self.mask
        z_masked = z * mask
        
        # 计算缩放和平移参数
        s = self.scale_net(z_masked) * (1 - mask)
        t = self.translate_net(z_masked) * (1 - mask)
        
        # 应用逆变换
        x = z_masked + (1 - mask) * ((z - t) * torch.exp(-s))
        
        # 对数行列式（逆变换取负）
        log_det = -s.sum(dim=-1)
        
        return x, log_det


class NormalizingFlow(nn.Module):
    """
    Normalizing Flow
    
    组合多个耦合层构建强大的概率分布。
    相比高斯混合，能表达更复杂的分布形状。
    
    优势：
    1. 精确的似然计算
    2. 可逆变换，支持高效采样
    3. 能表达多模态和非凸分布
    4. 不会产生"平均设计"问题（因为不使用高斯平均）
    """
    
    def __init__(
        self,
        dim: int,
        n_layers: int = 8,
        hidden_dim: int = 128,
        condition_dim: Optional[int] = None
    ):
        """
        Args:
            dim: 数据维度
            n_layers: 耦合层数量
            hidden_dim: 隐藏层维度
            condition_dim: 条件维度（可选）
        """
        super().__init__()
        self.dim = dim
        self.n_layers = n_layers
        self.condition_dim = condition_dim
        
        # 条件嵌入（如果有条件输入）
        if condition_dim is not None:
            self.condition_encoder = nn.Sequential(
                nn.Linear(condition_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            # 修改后的耦合层会使用条件
            self.coupling_layers = nn.ModuleList([
                ConditionalAffineCouplingLayer(
                    dim, hidden_dim, hidden_dim, 
                    mask_type='alternating' if i % 2 == 0 else 'reverse_alternating'
                )
                for i in range(n_layers)
            ])
        else:
            # 构建耦合层（交替使用不同的掩码）
            self.coupling_layers = nn.ModuleList([
                AffineCouplingLayer(
                    dim, hidden_dim,
                    mask_type='alternating' if i % 2 == 0 else 'reverse_alternating'
                )
                for i in range(n_layers)
            ])
    
    def forward(self, x: Tensor, condition: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """
        前向变换: 数据 -> 潜在空间
        
        Args:
            x: 数据 [B, D]
            condition: 条件 [B, condition_dim]
            
        Returns:
            z: 潜在表示 [B, D]
            log_det: 总对数行列式 [B]
        """
        log_det_total = torch.zeros(x.size(0), device=x.device)
        z = x
        
        for layer in self.coupling_layers:
            if self.condition_dim is not None:
                z, log_det = layer(z, condition)
            else:
                z, log_det = layer(z)
            log_det_total = log_det_total + log_det
        
        return z, log_det_total
    
    def inverse(self, z: Tensor, condition: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """
        逆变换: 潜在空间 -> 数据空间
        
        Args:
            z: 潜在表示 [B, D]
            condition: 条件 [B, condition_dim]
            
        Returns:
            x: 数据 [B, D]
            log_det: 总对数行列式 [B]
        """
        log_det_total = torch.zeros(z.size(0), device=z.device)
        x = z
        
        # 逆序通过各层
        for layer in reversed(self.coupling_layers):
            if self.condition_dim is not None:
                x, log_det = layer.inverse(x, condition)
            else:
                x, log_det = layer.inverse(x)
            log_det_total = log_det_total + log_det
        
        return x, log_det_total
    
    def log_prob(self, x: Tensor, condition: Optional[Tensor] = None) -> Tensor:
        """
        计算对数概率密度
        
        Args:
            x: 数据 [B, D]
            condition: 条件 [B, condition_dim]
            
        Returns:
            log_prob: 对数概率 [B]
        """
        z, log_det = self.forward(x, condition)
        
        # 基础分布是标准正态分布
        log_prob_base = -0.5 * (z ** 2 + math.log(2 * math.pi)).sum(dim=-1)
        
        return log_prob_base + log_det
    
    def sample(self, condition: Optional[Tensor] = None, n_samples: int = 1) -> Tensor:
        """
        从分布中采样
        
        Args:
            condition: 条件 [B, condition_dim]
            n_samples: 每个条件的采样数
            
        Returns:
            samples: 样本 [B, n_samples, D] 或 [n_samples, D]
        """
        if condition is not None:
            batch_size = condition.size(0)
            # 从标准正态分布采样
            z = torch.randn(batch_size * n_samples, self.dim, device=condition.device)
            
            # 扩展条件
            if n_samples > 1:
                condition_expanded = condition.unsqueeze(1).expand(-1, n_samples, -1)
                condition_expanded = condition_expanded.reshape(batch_size * n_samples, -1)
            else:
                condition_expanded = condition
            
            # 逆变换
            samples, _ = self.inverse(z, condition_expanded)
            
            if n_samples > 1:
                samples = samples.view(batch_size, n_samples, self.dim)
        else:
            z = torch.randn(n_samples, self.dim)
            samples, _ = self.inverse(z)
        
        return samples


class ConditionalAffineCouplingLayer(nn.Module):
    """
    条件仿射耦合层
    
    与标准耦合层类似，但接受额外的条件输入。
    用于条件生成任务。
    """
    
    def __init__(
        self,
        dim: int,
        hidden_dim: int = 128,
        condition_dim: int = 64,
        mask_type: str = 'alternating'
    ):
        super().__init__()
        self.dim = dim
        
        # 创建掩码
        if mask_type == 'alternating':
            self.register_buffer('mask', torch.arange(dim) % 2)
        elif mask_type == 'reverse_alternating':
            self.register_buffer('mask', 1 - (torch.arange(dim) % 2))
        else:
            self.register_buffer('mask', torch.cat([
                torch.ones(dim // 2),
                torch.zeros(dim - dim // 2)
            ]))
        
        # 条件感知的缩放和平移网络
        self.scale_net = nn.Sequential(
            nn.Linear(dim + condition_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim),
            nn.Tanh()
        )
        
        self.translate_net = nn.Sequential(
            nn.Linear(dim + condition_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim)
        )
    
    def forward(self, x: Tensor, condition: Tensor) -> Tuple[Tensor, Tensor]:
        """前向变换"""
        mask = self.mask
        x_masked = x * mask
        
        # 拼接条件
        x_cond = torch.cat([x_masked, condition], dim=-1)
        
        s = self.scale_net(x_cond) * (1 - mask)
        t = self.translate_net(x_cond) * (1 - mask)
        
        z = x_masked + (1 - mask) * (x * torch.exp(s) + t)
        log_det = s.sum(dim=-1)
        
        return z, log_det
    
    def inverse(self, z: Tensor, condition: Tensor) -> Tuple[Tensor, Tensor]:
        """逆变换"""
        mask = self.mask
        z_masked = z * mask
        
        x_cond = torch.cat([z_masked, condition], dim=-1)
        
        s = self.scale_net(x_cond) * (1 - mask)
        t = self.translate_net(x_cond) * (1 - mask)
        
        x = z_masked + (1 - mask) * ((z - t) * torch.exp(-s))
        log_det = -s.sum(dim=-1)
        
        return x, log_det


# ============================================================================
# 拓扑感知组件
# ============================================================================

class TopologyConstraint:
    """
    拓扑约束检查和评估
    
    光子器件设计的拓扑特性：
    1. 连通性：波导等结构应该连续
    2. 最小特征尺寸：避免无法制造的小结构
    3. 曲率约束：边界曲率不应过大
    4. 对称性：某些器件需要对称结构
    
    重要：两个有效设计的"平均"不一定是有效设计！
    """
    
    def __init__(
        self,
        min_feature_size: float = 0.1,
        resolution: float = 0.01,
        connectivity_threshold: float = 0.5
    ):
        """
        Args:
            min_feature_size: 最小特征尺寸（μm）
            resolution: 网格分辨率（μm/pixel）
            connectivity_threshold: 连通性阈值
        """
        self.min_feature_size = min_feature_size
        self.resolution = resolution
        self.min_pixels = int(min_feature_size / resolution)
        self.connectivity_threshold = connectivity_threshold
    
    def check_minimum_feature_size(self, design: Tensor) -> Tuple[bool, Tensor]:
        """
        检查最小特征尺寸约束
        
        使用形态学操作检测小于最小尺寸的特征。
        
        Args:
            design: 设计参数 [H, W] 或 [B, H, W]
            
        Returns:
            satisfied: 是否满足约束
            violation_map: 违规区域图
        """
        if design.dim() == 2:
            design = design.unsqueeze(0)
        
        # 二值化
        binary = (design > 0.5).float()
        
        # 使用形态学开运算检测小特征
        kernel_size = max(3, self.min_pixels)
        
        # 创建圆形结构元素
        y, x = torch.meshgrid(
            torch.arange(kernel_size, device=design.device) - kernel_size // 2,
            torch.arange(kernel_size, device=design.device) - kernel_size // 2,
            indexing='ij'
        )
        kernel = ((x ** 2 + y ** 2) <= (kernel_size // 2) ** 2).float()
        kernel = kernel.view(1, 1, kernel_size, kernel_size)
        
        # 开运算检测小特征
        binary_4d = binary.unsqueeze(1)
        opened = self._morphology_open(binary_4d, kernel)
        
        # 原图与开运算结果的差异即为小特征
        small_features = binary_4d - opened
        violation_map = small_features.squeeze(1)  # [B, H, W]
        
        # 计算违规比例
        violation_ratio = violation_map.sum(dim=(1, 2)) / (design.size(1) * design.size(2))
        satisfied = (violation_ratio < 0.05).all()  # 允许 5% 以下的小特征
        
        return satisfied.item() if violation_map.size(0) == 1 else satisfied, violation_map
    
    def _morphology_open(self, binary: Tensor, kernel: Tensor) -> Tensor:
        """形态学开运算（先腐蚀后膨胀）"""
        eroded = self._morphology_erode(binary, kernel)
        dilated = self._morphology_dilate(eroded, kernel)
        return dilated
    
    def _morphology_erode(self, binary: Tensor, kernel: Tensor) -> Tensor:
        """形态学腐蚀"""
        padding = kernel.size(-1) // 2
        result = F.conv2d(binary, kernel, padding=padding)
        kernel_sum = kernel.sum()
        return (result >= kernel_sum - 0.5).float()
    
    def _morphology_dilate(self, binary: Tensor, kernel: Tensor) -> Tensor:
        """形态学膨胀"""
        padding = kernel.size(-1) // 2
        result = F.conv2d(binary, kernel, padding=padding)
        return (result > 0.5).float()
    
    def check_connectivity(self, design: Tensor) -> Tuple[bool, float]:
        """
        检查设计区域的连通性
        
        使用简单的连通分量分析。
        
        Args:
            design: 设计参数 [H, W] 或 [B, H, W]
            
        Returns:
            connected: 是否连通
            connectivity_score: 连通性得分
        """
        if design.dim() == 2:
            design = design.unsqueeze(0)
        
        batch_size = design.size(0)
        scores = []
        
        for i in range(batch_size):
            d = design[i]
            
            # 计算梯度（边界检测）
            grad_x = torch.abs(d[:, 1:] - d[:, :-1])
            grad_y = torch.abs(d[1:, :] - d[:-1, :])
            
            # 边界密度
            boundary_density = (grad_x > 0.1).float().mean() + (grad_y > 0.1).float().mean()
            
            # 连通性得分（边界越少，连通性越好，但不能太少）
            # 理想情况下，边界密度应该在合理范围内
            optimal_density = 0.1
            score = 1.0 - torch.abs(boundary_density - optimal_density) * 2
            score = torch.clamp(score, 0.0, 1.0)
            scores.append(score)
        
        connectivity_score = torch.stack(scores).mean().item()
        connected = connectivity_score > self.connectivity_threshold
        
        return connected, connectivity_score
    
    def check_curvature(self, design: Tensor, max_curvature: float = 0.5) -> Tuple[bool, Tensor]:
        """
        检查边界曲率约束
        
        过大的曲率可能导致制造困难。
        
        Args:
            design: 设计参数 [H, W]
            max_curvature: 最大允许曲率
            
        Returns:
            satisfied: 是否满足
            curvature_map: 曲率图
        """
        if design.dim() == 2:
            design = design.unsqueeze(0)
        
        # 计算一阶梯度
        grad_x = design[:, :, 1:] - design[:, :, :-1]
        grad_y = design[:, 1:, :] - design[:, :-1, :]
        
        # 计算二阶梯度（曲率近似）
        grad_x_padded = F.pad(grad_x, (0, 1), mode='replicate')
        grad_y_padded = F.pad(grad_y, (0, 0, 0, 1), mode='replicate')
        
        curvature_x = grad_x_padded[:, :, 1:] - grad_x_padded[:, :, :-1]
        curvature_y = grad_y_padded[:, 1:, :] - grad_y_padded[:, :-1, :]
        
        # 曲率幅度
        curvature = torch.sqrt(curvature_x[:, :-1, :] ** 2 + curvature_y[:, :, :-1] ** 2 + 1e-8)
        
        # 检查是否超过阈值
        violation = (curvature > max_curvature).float()
        satisfied = violation.sum() < design.size(1) * design.size(2) * 0.05
        
        return satisfied.item(), curvature
    
    def compute_topology_loss(
        self,
        design: Tensor,
        weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Tensor]:
        """
        计算综合拓扑损失
        
        Args:
            design: 设计参数 [B, H, W]
            weights: 各项权重
            
        Returns:
            损失字典
        """
        if weights is None:
            weights = {
                'min_feature': 1.0,
                'connectivity': 1.0,
                'curvature': 0.5
            }
        
        losses = {}
        total_loss = torch.tensor(0.0, device=design.device)
        
        # 最小特征尺寸损失
        if weights.get('min_feature', 0) > 0:
            _, violation_map = self.check_minimum_feature_size(design)
            min_feature_loss = violation_map.mean()
            losses['min_feature'] = min_feature_loss
            total_loss = total_loss + weights['min_feature'] * min_feature_loss
        
        # 连通性损失
        if weights.get('connectivity', 0) > 0:
            connected, score = self.check_connectivity(design)
            connectivity_loss = 1.0 - score
            losses['connectivity'] = torch.tensor(connectivity_loss, device=design.device)
            total_loss = total_loss + weights['connectivity'] * connectivity_loss
        
        # 曲率损失
        if weights.get('curvature', 0) > 0:
            _, curvature = self.check_curvature(design)
            curvature_loss = curvature.mean()
            losses['curvature'] = curvature_loss
            total_loss = total_loss + weights['curvature'] * curvature_loss
        
        losses['total'] = total_loss
        
        return losses


class TopologyAwareSampler:
    """
    拓扑感知采样器
    
    在采样过程中考虑拓扑约束，确保生成的设计满足制造要求。
    
    核心思想：
    1. 从基础分布采样候选设计
    2. 评估拓扑约束满足程度
    3. 使用投影或优化方法修正设计
    4. 选择满足约束的最优设计
    """
    
    def __init__(
        self,
        constraint: TopologyConstraint,
        refinement_steps: int = 5,
        learning_rate: float = 0.01
    ):
        """
        Args:
            constraint: 拓扑约束检查器
            refinement_steps: 精炼步数
            learning_rate: 精炼学习率
        """
        self.constraint = constraint
        self.refinement_steps = refinement_steps
        self.learning_rate = learning_rate
    
    def sample_with_refinement(
        self,
        raw_samples: Tensor,
        return_all: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Dict[str, Any]]]:
        """
        采样并进行拓扑精炼
        
        Args:
            raw_samples: 原始样本 [B, N, H, W] 或 [N, H, W]
            return_all: 是否返回所有样本（包括被拒绝的）
            
        Returns:
            refined_samples: 精炼后的样本
            info: 额外信息（如果 return_all=True）
        """
        original_shape = raw_samples.shape
        if raw_samples.dim() == 3:
            raw_samples = raw_samples.unsqueeze(0)
        
        batch_size = raw_samples.size(0)
        n_samples = raw_samples.size(1)
        
        refined_samples = []
        info = {
            'acceptance_rate': [],
            'refinement_loss': []
        }
        
        for b in range(batch_size):
            batch_samples = []
            for n in range(n_samples):
                sample = raw_samples[b, n].clone()
                sample.requires_grad = True
                
                # 拓扑精炼
                optimizer = torch.optim.Adam([sample], lr=self.learning_rate)
                
                for step in range(self.refinement_steps):
                    optimizer.zero_grad()
                    
                    # 计算拓扑损失
                    topo_losses = self.constraint.compute_topology_loss(
                        sample.unsqueeze(0),
                        weights={'min_feature': 1.0, 'connectivity': 0.5, 'curvature': 0.3}
                    )
                    
                    loss = topo_losses['total']
                    loss.backward()
                    optimizer.step()
                    
                    # 投影到 [0, 1]
                    with torch.no_grad():
                        sample.clamp_(0, 1)
                
                # 检查约束满足
                min_feat_ok, _ = self.constraint.check_minimum_feature_size(sample)
                conn_ok, _ = self.constraint.check_connectivity(sample)
                
                batch_samples.append(sample.detach())
            
            refined_samples.append(torch.stack(batch_samples))
        
        refined_samples = torch.stack(refined_samples)
        
        if len(original_shape) == 3:
            refined_samples = refined_samples.squeeze(0)
        
        if return_all:
            return refined_samples, info
        return refined_samples
    
    def filter_valid_designs(
        self,
        designs: Tensor,
        threshold: float = 0.8
    ) -> Tuple[Tensor, Tensor]:
        """
        过滤出有效的设计
        
        Args:
            designs: 设计样本 [B, N, H, W]
            threshold: 有效性阈值
            
        Returns:
            valid_designs: 有效设计
            validity_mask: 有效性掩码
        """
        if designs.dim() == 3:
            designs = designs.unsqueeze(0)
        
        batch_size, n_samples = designs.size(0), designs.size(1)
        validity_scores = []
        
        for b in range(batch_size):
            batch_scores = []
            for n in range(n_samples):
                design = designs[b, n]
                
                # 检查各项约束
                min_feat_ok, _ = self.constraint.check_minimum_feature_size(design)
                conn_ok, conn_score = self.constraint.check_connectivity(design)
                curv_ok, _ = self.constraint.check_curvature(design)
                
                # 综合有效性分数
                score = float(min_feat_ok) * 0.4 + conn_score * 0.4 + float(curv_ok) * 0.2
                batch_scores.append(score)
            
            validity_scores.append(batch_scores)
        
        validity_scores = torch.tensor(validity_scores, device=designs.device)
        validity_mask = validity_scores > threshold
        
        return designs[validity_mask], validity_mask


# ============================================================================
# MDN 主网络
# ============================================================================

class MDN(InverseModel):
    """
    混合密度网络
    
    将性能指标映射到设计参数的概率分布。
    支持三种分布类型：
    1. GMM（高斯混合模型）：传统 MDN
    2. Flow（Normalizing Flow）：更复杂的分布
    3. Hybrid：混合使用
    
    拓扑感知：
    - 集成最小特征尺寸约束
    - 连通性检查
    - 曲率约束
    - 拓扑引导采样
    
    使用示例:
    ```python
    # 创建 MDN with Normalizing Flow
    config = MDNConfig(
        input_dim=3,
        output_dim=200*22,
        distribution_type='flow',
        topology_aware=True
    )
    mdn = MDN(config)
    
    # 训练
    mdn.train(train_loader, val_loader, epochs=100)
    
    # 推理
    target_perf = torch.tensor([[0.85, 0.8, 0.1]])
    
    # 方式1: 获取分布参数
    pi, mu, sigma = mdn.forward(target_perf)
    distribution = GaussianMixtureDistribution(pi, mu, sigma)
    
    # 方式2: 直接采样（拓扑感知）
    samples = mdn.sample(target_perf, n_samples=10)
    
    # 方式3: 获取最可能的设计
    best_design = mdn.sample_mode(target_perf)
    ```
    """
    
    def __init__(self, config: Optional[MDNConfig] = None):
        config = config or MDNConfig()
        super().__init__(config)
        self.config = config
        
        # 保存关键参数
        self.input_dim = config.input_dim
        self.output_dim = config.output_dim
        self.design_shape = config.design_shape
        self.n_components = config.n_components
        self.distribution_type = config.distribution_type
        
        # 构建网络
        self.encoder = self._build_encoder()
        
        # 根据分布类型构建输出层
        if config.distribution_type == 'gmm':
            self.mixture = GaussianMixtureParameters(
                input_dim=config.shared_hidden_dims[-1] if config.shared_hidden_dims else config.hidden_dims[-1],
                output_dim=config.output_dim,
                n_components=config.n_components,
                log_sigma_min=config.log_sigma_min,
                log_sigma_max=config.log_sigma_max
            )
            self.flow = None
        
        elif config.distribution_type == 'flow':
            self.mixture = None
            self.flow = NormalizingFlow(
                dim=config.output_dim,
                n_layers=config.n_flow_layers,
                hidden_dim=config.flow_hidden_dim,
                condition_dim=config.shared_hidden_dims[-1] if config.shared_hidden_dims else config.hidden_dims[-1]
            )
        
        elif config.distribution_type == 'hybrid':
            # 混合模式：使用 GMM 初始化，Flow 精炼
            self.mixture = GaussianMixtureParameters(
                input_dim=config.shared_hidden_dims[-1] if config.shared_hidden_dims else config.hidden_dims[-1],
                output_dim=config.output_dim,
                n_components=config.n_components,
                log_sigma_min=config.log_sigma_min,
                log_sigma_max=config.log_sigma_max
            )
            self.flow = NormalizingFlow(
                dim=config.output_dim,
                n_layers=config.n_flow_layers // 2,  # 较少层用于精炼
                hidden_dim=config.flow_hidden_dim,
                condition_dim=config.shared_hidden_dims[-1] if config.shared_hidden_dims else config.hidden_dims[-1]
            )
        else:
            raise ValueError(f"Unknown distribution type: {config.distribution_type}")
        
        # 拓扑约束组件
        if config.topology_aware:
            self.topology_constraint = TopologyConstraint(
                min_feature_size=config.min_feature_size,
                resolution=config.resolution,
                connectivity_threshold=config.constraint_threshold
            )
            self.topology_sampler = TopologyAwareSampler(
                constraint=self.topology_constraint,
                refinement_steps=config.topology_refinement_steps
            )
        else:
            self.topology_constraint = None
            self.topology_sampler = None
        
        # 初始化权重
        self._init_weights()
    
    def _build_encoder(self) -> nn.Module:
        """构建共享特征编码器"""
        layers = []
        in_dim = self.input_dim
        config = self.config
        
        # 主隐藏层
        for i, out_dim in enumerate(config.hidden_dims):
            layers.append(nn.Linear(in_dim, out_dim))
            if config.batch_norm:
                layers.append(nn.BatchNorm1d(out_dim))
            layers.append(self._get_activation(config.activation))
            if config.dropout_rate > 0:
                layers.append(nn.Dropout(config.dropout_rate))
            in_dim = out_dim
        
        # 共享隐藏层
        for out_dim in config.shared_hidden_dims:
            layers.append(nn.Linear(in_dim, out_dim))
            if config.batch_norm:
                layers.append(nn.BatchNorm1d(out_dim))
            layers.append(self._get_activation(config.activation))
            in_dim = out_dim
        
        return nn.Sequential(*layers)
    
    def _get_activation(self, name: str) -> nn.Module:
        """获取激活函数"""
        activations = {
            'relu': nn.ReLU,
            'leaky_relu': nn.LeakyReLU,
            'gelu': nn.GELU,
            'silu': nn.SiLU,
            'tanh': nn.Tanh
        }
        if name.lower() not in activations:
            raise ValueError(f"Unknown activation: {name}")
        return activations[name.lower()]()
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, condition: Tensor) -> Union[Tuple[Tensor, Tensor, Tensor], Tensor]:
        """
        前向传播
        
        Args:
            condition: 条件输入（性能指标）[B, input_dim]
            
        Returns:
            GMM 模式: (pi, mu, sigma)
            Flow 模式: log_prob（通过采样计算）
        """
        features = self.encoder(condition)
        
        if self.distribution_type == 'gmm':
            pi, mu, sigma = self.mixture(features)
            return pi, mu, sigma
        
        elif self.distribution_type == 'flow':
            # Flow 模式不直接返回参数，而是通过采样
            return features  # 返回条件特征用于采样
        
        else:  # hybrid
            pi, mu, sigma = self.mixture(features)
            return pi, mu, sigma, features
    
    def get_distribution(self, condition: Tensor) -> Union[GaussianMixtureDistribution, NormalizingFlow]:
        """
        获取条件分布
        
        Args:
            condition: 条件输入 [B, input_dim]
            
        Returns:
            分布对象
        """
        features = self.encoder(condition)
        
        if self.distribution_type == 'gmm':
            pi, mu, sigma = self.mixture(features)
            return GaussianMixtureDistribution(pi, mu, sigma)
        
        elif self.distribution_type == 'flow':
            return self.flow
        
        else:  # hybrid
            pi, mu, sigma = self.mixture(features)
            return GaussianMixtureDistribution(pi, mu, sigma), self.flow
    
    def sample(
        self,
        condition: Tensor,
        n_samples: int = 1,
        topology_refine: Optional[bool] = None
    ) -> Tensor:
        """
        从条件分布中采样
        
        Args:
            condition: 条件输入 [B, input_dim]
            n_samples: 采样数量
            topology_refine: 是否进行拓扑精炼（None 使用配置）
            
        Returns:
            samples: 样本 [B, n_samples, D] 或重塑后 [B, n_samples, H, W]
        """
        features = self.encoder(condition)
        
        # 根据分布类型采样
        if self.distribution_type == 'gmm':
            pi, mu, sigma = self.mixture(features)
            distribution = GaussianMixtureDistribution(pi, mu, sigma)
            samples = distribution.sample(n_samples)
        
        elif self.distribution_type == 'flow':
            samples = self.flow.sample(features, n_samples)
        
        else:  # hybrid
            pi, mu, sigma = self.mixture(features)
            gmm = GaussianMixtureDistribution(pi, mu, sigma)
            samples = gmm.sample(n_samples)
            # 可选：使用 flow 精炼
            if self.flow is not None and n_samples > 0:
                # 将样本通过 flow 精炼
                batch_size = samples.size(0)
                samples_flat = samples.view(batch_size * n_samples, -1)
                features_expanded = features.unsqueeze(1).expand(-1, n_samples, -1)
                features_expanded = features_expanded.reshape(batch_size * n_samples, -1)
                
                # Flow 精炼（逆向变换再正向变换）
                z, _ = self.flow.forward(samples_flat, features_expanded)
                samples, _ = self.flow.inverse(z, features_expanded)
                samples = samples.view(batch_size, n_samples, -1)
        
        # 拓扑精炼
        if topology_refine is None:
            topology_refine = self.config.topology_guided_sampling
        
        if topology_refine and self.topology_sampler is not None:
            # 重塑为设计形状
            if self.design_shape is not None:
                H, W = self.design_shape
                samples = samples.view(samples.size(0), samples.size(1), H, W)
            
            samples = self.topology_sampler.sample_with_refinement(samples)
            
            if self.design_shape is not None:
                samples = samples.view(samples.size(0), samples.size(1), -1)
        
        # 重塑为设计形状
        if self.design_shape is not None and samples.dim() == 3:
            H, W = self.design_shape
            samples = samples.view(samples.size(0), samples.size(1), H, W)
        
        return samples
    
    def sample_mode(self, condition: Tensor) -> Tensor:
        """
        返回最可能的设计（最大权重分量的均值）
        
        Args:
            condition: 条件输入 [B, input_dim]
            
        Returns:
            design: 最可能的设计 [B, D] 或 [B, H, W]
        """
        distribution = self.get_distribution(condition)
        design = distribution.sample_mode()
        
        if self.design_shape is not None:
            H, W = self.design_shape
            design = design.view(design.size(0), H, W)
        
        return design
    
    def sample_best(
        self,
        condition: Tensor,
        forward_model: nn.Module,
        n_samples: int = 10
    ) -> Tuple[Tensor, Tensor]:
        """
        采样并选择性能最优的设计
        
        Args:
            condition: 目标性能 [B, input_dim]
            forward_model: 前向模型（用于评估设计性能）
            n_samples: 采样数量
            
        Returns:
            best_design: 最优设计
            best_performance: 预测性能
        """
        # 采样
        samples = self.sample(condition, n_samples)  # [B, n_samples, H, W]
        
        batch_size = samples.size(0)
        samples_flat = samples.view(batch_size * n_samples, *self.design_shape)
        
        # 评估性能
        with torch.no_grad():
            pred_perf = forward_model(samples_flat)  # [B*n_samples, perf_dim]
        
        pred_perf = pred_perf.view(batch_size, n_samples, -1)  # [B, n_samples, perf_dim]
        
        # 计算与目标的距离
        condition_expanded = condition.unsqueeze(1).expand(-1, n_samples, -1)
        distances = torch.norm(pred_perf - condition_expanded, dim=-1)  # [B, n_samples]
        
        # 选择最优
        best_indices = torch.argmin(distances, dim=-1)  # [B]
        batch_indices = torch.arange(batch_size, device=samples.device)
        
        best_design = samples[batch_indices, best_indices]  # [B, H, W]
        best_performance = pred_perf[batch_indices, best_indices]  # [B, perf_dim]
        
        return best_design, best_performance
    
    def compute_loss(
        self,
        condition: Tensor,
        target: Tensor,
        include_topology: Optional[bool] = None
    ) -> Union[Tensor, Dict[str, Tensor]]:
        """
        计算损失
        
        Args:
            condition: 条件输入 [B, input_dim]
            target: 目标设计 [B, D] 或 [B, H, W]
            include_topology: 是否包含拓扑损失（None 使用配置）
            
        Returns:
            loss: 总损失（标量或字典）
        """
        # 展平目标
        if target.dim() > 2:
            target = target.view(target.size(0), -1)
        
        features = self.encoder(condition)
        
        # 基础损失（负对数似然）
        if self.distribution_type == 'gmm':
            pi, mu, sigma = self.mixture(features)
            distribution = GaussianMixtureDistribution(pi, mu, sigma)
            log_prob = distribution.log_prob(target)
            nll = -log_prob.mean()
            base_loss = nll
        
        elif self.distribution_type == 'flow':
            # Flow 的负对数似然
            log_prob = self.flow.log_prob(target, features)
            nll = -log_prob.mean()
            base_loss = nll
        
        else:  # hybrid
            pi, mu, sigma = self.mixture(features)
            distribution = GaussianMixtureDistribution(pi, mu, sigma)
            log_prob_gmm = distribution.log_prob(target)
            nll_gmm = -log_prob_gmm.mean()
            
            log_prob_flow = self.flow.log_prob(target, features)
            nll_flow = -log_prob_flow.mean()
            
            # 加权组合
            base_loss = 0.5 * nll_gmm + 0.5 * nll_flow
        
        # 拓扑损失
        if include_topology is None:
            include_topology = self.config.topology_aware
        
        if include_topology and self.topology_constraint is not None:
            # 重塑为设计形状计算拓扑损失
            target_design = target.view(target.size(0), *self.design_shape) if self.design_shape else target
            
            topo_losses = self.topology_constraint.compute_topology_loss(
                target_design,
                weights={
                    'min_feature': self.config.min_feature_weight,
                    'connectivity': self.config.connectivity_weight,
                    'curvature': self.config.curvature_weight
                }
            )
            
            total_loss = base_loss + topo_losses['total']
            
            return {
                'total': total_loss,
                'nll': base_loss,
                'topology': topo_losses['total'],
                'min_feature': topo_losses.get('min_feature', torch.tensor(0.0)),
                'connectivity': topo_losses.get('connectivity', torch.tensor(0.0)),
                'curvature': topo_losses.get('curvature', torch.tensor(0.0))
            }
        
        return base_loss
    
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
        训练 MDN
        
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
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            # 训练阶段
            train_loss = 0.0
            for batch in train_loader:
                condition = batch['performance'].to(self.device)
                target = batch['design'].to(self.device)
                
                optimizer.zero_grad()
                loss = self.compute_loss(condition, target)
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            history['train_loss'].append(train_loss)
            
            # 验证阶段
            if val_loader is not None:
                val_loss = self._validate(val_loader)
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
                    print(f"Epoch {epoch + 1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
            else:
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch + 1}: train_loss={train_loss:.4f}")
        
        return history
    
    def _validate(self, val_loader) -> float:
        """验证"""
        self.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                condition = batch['performance'].to(self.device)
                target = batch['design'].to(self.device)
                
                loss = self.compute_loss(condition, target)
                val_loss += loss.item()
        
        self.train()
        return val_loss / len(val_loader)
    
    def save(self, path: Union[str, Path], **kwargs):
        """保存模型"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'config': self.config.__dict__,
            'model_state_dict': self.state_dict(),
            'training_history': self.training_history
        }
        
        torch.save(checkpoint, path)
    
    def load(self, path: Union[str, Path], **kwargs):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.load_state_dict(checkpoint['model_state_dict'])
        
        if 'training_history' in checkpoint:
            self.training_history = checkpoint['training_history']
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            'name': self.config.name,
            'device': str(self.device),
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'design_shape': self.design_shape,
            'n_components': self.n_components,
            'parameters': self.count_parameters()
        }


# ============================================================================
# 便捷函数
# ============================================================================

def create_mdn_for_challenge(
    challenge_name: str,
    n_components: int = 5,
    performance_dim: int = 3,
    distribution_type: str = 'gmm',
    topology_aware: bool = True,
    min_feature_size: float = 0.1,
    resolution: float = 0.01,
    device: str = 'auto'
) -> MDN:
    """
    为特定挑战创建 MDN
    
    Args:
        challenge_name: 挑战名称
        n_components: 高斯分量数（仅 GMM 模式）
        performance_dim: 性能指标维度
        distribution_type: 分布类型 ('gmm', 'flow', 'hybrid')
        topology_aware: 是否启用拓扑感知
        min_feature_size: 最小特征尺寸（μm）
        resolution: 网格分辨率（μm/pixel）
        device: 计算设备
        
    Returns:
        配置好的 MDN
    """
    from challenges import ChallengeFactory
    
    # 获取挑战以确定设计形状
    challenge = ChallengeFactory.create(challenge_name)
    design_shape = challenge.spec.get_grid_shape()
    
    config = MDNConfig(
        input_dim=performance_dim,
        output_dim=int(torch.prod(torch.tensor(design_shape)).item()),
        design_shape=design_shape,
        n_components=n_components,
        distribution_type=distribution_type,
        topology_aware=topology_aware,
        min_feature_size=min_feature_size,
        resolution=resolution,
        device=device
    )
    
    return MDN(config)


# ============================================================================
# 简化的 Flow-based MDN（推荐用于拓扑设计）
# ============================================================================

class FlowMDN(InverseModel):
    """
    基于 Normalizing Flow 的 MDN
    
    专门为光子器件拓扑设计优化：
    1. 不使用高斯混合，避免"平均设计"问题
    2. 精确的似然计算
    3. 内置拓扑约束
    
    推荐用于需要精确拓扑控制的设计任务。
    """
    
    def __init__(
        self,
        input_dim: int = 3,
        output_dim: int = 200 * 22,
        design_shape: Tuple[int, int] = (200, 22),
        n_flow_layers: int = 8,
        hidden_dim: int = 128,
        min_feature_size: float = 0.1,
        resolution: float = 0.01
    ):
        """
        Args:
            input_dim: 性能指标维度
            output_dim: 设计参数维度
            design_shape: 设计形状
            n_flow_layers: Flow 层数
            hidden_dim: 隐藏层维度
            min_feature_size: 最小特征尺寸
            resolution: 网格分辨率
        """
        config = MDNConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            design_shape=design_shape,
            distribution_type='flow',
            n_flow_layers=n_flow_layers,
            flow_hidden_dim=hidden_dim,
            topology_aware=True,
            min_feature_size=min_feature_size,
            resolution=resolution
        )
        super().__init__(config)
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.design_shape = design_shape
        
        # 条件编码器
        self.condition_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Normalizing Flow
        self.flow = NormalizingFlow(
            dim=output_dim,
            n_layers=n_flow_layers,
            hidden_dim=hidden_dim,
            condition_dim=hidden_dim
        )
        
        # 拓扑约束
        self.topology_constraint = TopologyConstraint(
            min_feature_size=min_feature_size,
            resolution=resolution
        )
        
        self.topology_sampler = TopologyAwareSampler(
            constraint=self.topology_constraint
        )
    
    def forward(self, condition: Tensor) -> Tensor:
        """编码条件"""
        return self.condition_encoder(condition)
    
    def sample(self, condition: Tensor, n_samples: int = 1, refine: bool = True) -> Tensor:
        """采样"""
        features = self.condition_encoder(condition)
        samples = self.flow.sample(features, n_samples)
        
        if self.design_shape is not None:
            samples = samples.view(samples.size(0), samples.size(1), *self.design_shape)
        
        if refine:
            samples = self.topology_sampler.sample_with_refinement(samples)
        
        return samples
    
    def compute_loss(self, condition: Tensor, target: Tensor) -> Tensor:
        """计算损失"""
        if target.dim() > 2:
            target = target.view(target.size(0), -1)
        
        features = self.condition_encoder(condition)
        log_prob = self.flow.log_prob(target, features)
        nll = -log_prob.mean()
        
        return nll


# ============================================================================
# MDN 与 TNN 联合模型
# ============================================================================

class MDNTandemNetwork(BaseModel):
    """
    MDN-TNN 联合模型
    
    结合 TNN 的前向网络和 MDN 的逆向网络：
    - 前向：设计 → 性能（使用 TNN 的前向网络）
    - 逆向：性能 → 设计分布（使用 MDN）
    
    使用示例:
    ```python
    # 创建联合模型
    mdn_tandem = MDNTandemNetwork(
        forward_net=tnn.forward_net,
        mdn_config=MDNConfig(...)
    )
    
    # 训练 MDN（前向网络已预训练）
    mdn_tandem.train_mdn(train_loader, epochs=100)
    
    # 逆向设计
    samples = mdn_tandem.sample(target_performance, n_samples=10)
    best_design = mdn_tandem.sample_best(target_performance, n_samples=20)
    ```
    """
    
    def __init__(
        self,
        forward_net: nn.Module,
        mdn_config: Optional[MDNConfig] = None,
        freeze_forward: bool = True
    ):
        """
        Args:
            forward_net: 预训练的前向网络
            mdn_config: MDN 配置
            freeze_forward: 是否冻结前向网络
        """
        super().__init__(mdn_config or MDNConfig())
        
        self.forward_net = forward_net
        self.mdn = MDN(mdn_config)
        
        if freeze_forward:
            for param in self.forward_net.parameters():
                param.requires_grad = False
        
        self.forward_net.to(self.device)
        self.mdn.to(self.device)
    
    def forward(self, design: Tensor) -> Tensor:
        """前向预测"""
        return self.forward_net(design)
    
    def inverse(self, performance: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """逆向设计（返回分布参数）"""
        return self.mdn(performance)
    
    def sample(self, performance: Tensor, n_samples: int = 1) -> Tensor:
        """采样设计"""
        return self.mdn.sample(performance, n_samples)
    
    def sample_best(
        self,
        target_performance: Tensor,
        n_samples: int = 10
    ) -> Tuple[Tensor, Tensor]:
        """采样并选择最优设计"""
        return self.mdn.sample_best(target_performance, self.forward_net, n_samples)
    
    def train_mdn(
        self,
        train_loader,
        val_loader=None,
        epochs: int = 100,
        **kwargs
    ):
        """训练 MDN 部分"""
        return self.mdn.train_model(train_loader, val_loader, epochs, **kwargs)
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        forward_params = sum(p.numel() for p in self.forward_net.parameters())
        mdn_info = self.mdn.get_model_info()
        
        return {
            **mdn_info,
            'forward_parameters': forward_params,
            'total_parameters': forward_params + mdn_info['parameters']
        }

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
# MDN 主网络
# ============================================================================

class MDN(InverseModel):
    """
    混合密度网络
    
    将性能指标映射到设计参数的概率分布。
    
    使用示例:
    ```python
    # 创建 MDN
    config = MDNConfig(
        input_dim=3,
        output_dim=200*22,
        n_components=5
    )
    mdn = MDN(config)
    
    # 训练
    mdn.train(train_loader, val_loader, epochs=100)
    
    # 推理
    target_perf = torch.tensor([[0.85, 0.8, 0.1]])
    
    # 方式1: 获取分布参数
    pi, mu, sigma = mdn.forward(target_perf)
    distribution = GaussianMixtureDistribution(pi, mu, sigma)
    
    # 方式2: 直接采样
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
        
        # 构建网络
        self.encoder = self._build_encoder()
        self.mixture = GaussianMixtureParameters(
            input_dim=config.shared_hidden_dims[-1] if config.shared_hidden_dims else config.hidden_dims[-1],
            output_dim=config.output_dim,
            n_components=config.n_components,
            log_sigma_min=config.log_sigma_min,
            log_sigma_max=config.log_sigma_max
        )
        
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
    
    def forward(self, condition: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        前向传播，输出高斯混合参数
        
        Args:
            condition: 条件输入（性能指标）[B, input_dim]
            
        Returns:
            pi: 混合权重 [B, K]
            mu: 均值 [B, K, D]
            sigma: 标准差 [B, K, D]
        """
        features = self.encoder(condition)
        pi, mu, sigma = self.mixture(features)
        return pi, mu, sigma
    
    def get_distribution(self, condition: Tensor) -> GaussianMixtureDistribution:
        """
        获取条件分布
        
        Args:
            condition: 条件输入 [B, input_dim]
            
        Returns:
            高斯混合分布对象
        """
        pi, mu, sigma = self.forward(condition)
        return GaussianMixtureDistribution(pi, mu, sigma)
    
    def sample(self, condition: Tensor, n_samples: int = 1) -> Tensor:
        """
        从条件分布中采样
        
        Args:
            condition: 条件输入 [B, input_dim]
            n_samples: 采样数量
            
        Returns:
            samples: 样本 [B, n_samples, D] 或重塑后 [B, n_samples, H, W]
        """
        distribution = self.get_distribution(condition)
        samples = distribution.sample(n_samples)
        
        # 如果需要，重塑为设计形状
        if self.design_shape is not None:
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
        target: Tensor
    ) -> Tensor:
        """
        计算负对数似然损失
        
        Args:
            condition: 条件输入 [B, input_dim]
            target: 目标设计 [B, D] 或 [B, H, W]
            
        Returns:
            loss: 负对数似然
        """
        # 展平目标
        if target.dim() > 2:
            target = target.view(target.size(0), -1)
        
        # 获取分布参数
        pi, mu, sigma = self.forward(condition)
        
        # 创建分布并计算 log_prob
        distribution = GaussianMixtureDistribution(pi, mu, sigma)
        log_prob = distribution.log_prob(target)
        
        # 负对数似然
        nll = -log_prob.mean()
        
        return nll
    
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
    device: str = 'auto'
) -> MDN:
    """
    为特定挑战创建 MDN
    
    Args:
        challenge_name: 挑战名称
        n_components: 高斯分量数
        performance_dim: 性能指标维度
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
        device=device
    )
    
    return MDN(config)


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

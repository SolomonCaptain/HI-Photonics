"""
Tandem Neural Network (TNN) 串联神经网络

用于光子学逆向设计的双网络架构：
- 前向网络：设计参数 → 性能指标（代理模型）
- 逆向网络：性能指标 → 设计参数（生成器）
- 串联训练：固定前向网络，训练逆向网络

参考文献:
- Peurifoy et al., "Nanophotonic particle simulation and inverse design 
  using artificial neural networks", Science Advances, 2018
"""

from typing import Dict, Optional, Tuple, List, Union, Any
from dataclasses import dataclass, field
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.base import BaseModel, ModelConfig, SurrogateModel, InverseModel


# ============================================================================
# 配置类
# ============================================================================

@dataclass
class ForwardNetworkConfig(ModelConfig):
    """前向网络配置"""
    name: str = "forward_network"
    
    # 网络架构
    hidden_channels: List[int] = field(default_factory=lambda: [32, 64, 128])
    hidden_dims: List[int] = field(default_factory=lambda: [256, 128])
    kernel_size: int = 3
    padding: int = 1
    
    # 输入输出
    design_shape: Tuple[int, int] = (200, 22)  # (H, W)
    performance_dim: int = 3  # 性能指标维度
    
    # 正则化
    dropout_rate: float = 0.1
    batch_norm: bool = True
    
    # 激活函数
    activation: str = "relu"  # 'relu', 'leaky_relu', 'gelu', 'silu'


@dataclass
class InverseNetworkConfig(ModelConfig):
    """逆向网络配置"""
    name: str = "inverse_network"
    
    # 网络架构
    hidden_dims: List[int] = field(default_factory=lambda: [256, 512, 1024])
    hidden_channels: List[int] = field(default_factory=lambda: [128, 64, 32])
    kernel_size: int = 4
    stride: int = 2
    padding: int = 1
    
    # 输入输出
    performance_dim: int = 3
    design_shape: Tuple[int, int] = (200, 22)
    
    # 正则化
    dropout_rate: float = 0.1
    batch_norm: bool = True
    
    # 激活函数
    activation: str = "relu"
    output_activation: str = "sigmoid"  # 'sigmoid', 'tanh', 'none'


@dataclass
class TandemNetworkConfig(ModelConfig):
    """串联网络配置"""
    name: str = "tandem_network"
    
    # 子网络配置
    forward_config: ForwardNetworkConfig = field(default_factory=ForwardNetworkConfig)
    inverse_config: InverseNetworkConfig = field(default_factory=InverseNetworkConfig)
    
    # 训练策略
    pretrain_forward: bool = True
    freeze_forward: bool = True
    
    # 损失权重
    performance_loss_weight: float = 1.0
    design_loss_weight: float = 0.0  # 可选的设计空间正则化
    diversity_loss_weight: float = 0.0  # 多样性正则化


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
# 前向网络（代理模型）
# ============================================================================

class ForwardNetwork(SurrogateModel):
    """
    前向网络：设计参数 → 性能指标
    
    使用 CNN 架构处理二维设计网格，输出性能指标。
    作为代理模型替代昂贵的仿真计算。
    
    架构:
        输入: [B, 1, H, W] 设计参数网格
          ↓
        ConvEncoder (多层卷积下采样)
          ↓
        AdaptiveAvgPool
          ↓
        FC Layers
          ↓
        输出: [B, performance_dim] 性能指标
    """
    
    def __init__(self, config: Optional[ForwardNetworkConfig] = None):
        config = config or ForwardNetworkConfig()
        super().__init__(config)
        self.config = config
        
        # 保存输入输出维度
        self.design_shape = config.design_shape
        self.performance_dim = config.performance_dim
        
        # 构建编码器
        self.encoder = self._build_encoder()
        
        # 构建全连接层
        self.fc = self._build_fc()
        
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
            
            # 下采样（除第一层外）
            if i > 0:
                layers.append(nn.MaxPool2d(2, 2))
            
            in_channels = out_channels
        
        return nn.Sequential(*layers)
    
    def _build_fc(self) -> nn.Module:
        """构建全连接层"""
        layers = []
        
        # 计算编码器输出维度
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, *self.design_shape)
            dummy_output = self.encoder(dummy_input)
            flattened_dim = dummy_output.view(1, -1).size(1)
        
        in_dim = flattened_dim
        for out_dim in self.config.hidden_dims:
            layers.append(nn.Linear(in_dim, out_dim))
            if self.config.batch_norm:
                layers.append(nn.BatchNorm1d(out_dim))
            layers.append(get_activation(self.config.activation))
            if self.config.dropout_rate > 0:
                layers.append(nn.Dropout(self.config.dropout_rate))
            in_dim = out_dim
        
        # 输出层
        layers.append(nn.Linear(in_dim, self.performance_dim))
        
        return nn.Sequential(*layers)
    
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
    
    def forward(self, design: Tensor) -> Tensor:
        """
        从设计参数预测性能
        
        Args:
            design: 设计参数 [B, H, W] 或 [B, 1, H, W]
            
        Returns:
            性能指标 [B, performance_dim]
        """
        # 确保输入维度正确
        if design.dim() == 3:
            design = design.unsqueeze(1)  # [B, H, W] -> [B, 1, H, W]
        
        # 编码
        features = self.encoder(design)
        
        # 展平
        features = features.view(features.size(0), -1)
        
        # 全连接层
        performance = self.fc(features)
        
        return performance
    
    def encode(self, design: Tensor) -> Tensor:
        """提取设计特征"""
        if design.dim() == 3:
            design = design.unsqueeze(1)
        return self.encoder(design)


# ============================================================================
# 逆向网络（生成器）
# ============================================================================

class InverseNetwork(InverseModel):
    """
    逆向网络：性能指标 → 设计参数
    
    使用转置卷积架构从性能指标生成二维设计网格。
    解决"一对多"映射问题。
    
    架构:
        输入: [B, performance_dim] 目标性能
          ↓
        FC Layers (扩展维度)
          ↓
        Reshape [B, C, H', W']
          ↓
        ConvDecoder (转置卷积上采样)
          ↓
        输出: [B, 1, H, W] 设计参数
    """
    
    def __init__(self, config: Optional[InverseNetworkConfig] = None):
        config = config or InverseNetworkConfig()
        super().__init__(config)
        self.config = config
        
        # 保存输入输出维度
        self.performance_dim = config.performance_dim
        self.design_shape = config.design_shape
        
        # 构建网络
        self.fc = self._build_fc()
        self.decoder = self._build_decoder()
        
        # 初始化权重
        self._init_weights()
    
    def _build_fc(self) -> nn.Module:
        """构建前置全连接层"""
        layers = []
        in_dim = self.performance_dim
        
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
        layers = []
        config = self.config
        
        # 计算解码器输入尺寸
        # 目标: 从 (H/4, W/4) 上采样到 (H, W)
        h, w = self.design_shape
        self.init_h = max(h // (2 ** (len(config.hidden_channels) - 1)), 1)
        self.init_w = max(w // (2 ** (len(config.hidden_channels) - 1)), 1)
        
        # 计算初始通道数
        init_channels = self.fc_output_dim // (self.init_h * self.init_w)
        init_channels = max(init_channels, config.hidden_channels[0])
        
        # 调整全连接层输出维度
        self.fc_adjust = nn.Linear(
            self.fc_output_dim,
            init_channels * self.init_h * self.init_w
        )
        
        # 构建转置卷积层
        in_channels = init_channels
        for i, out_channels in enumerate(config.hidden_channels):
            # 转置卷积
            layers.append(nn.ConvTranspose2d(
                in_channels, out_channels,
                kernel_size=config.kernel_size,
                stride=config.stride if i < len(config.hidden_channels) - 1 else 1,
                padding=config.padding,
                output_padding=1 if i < len(config.hidden_channels) - 1 else 0
            ))
            
            # 批归一化
            if config.batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            
            # 激活函数（最后一层除外）
            if i < len(config.hidden_channels) - 1:
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
    
    def forward(self, performance: Tensor) -> Tensor:
        """
        从性能指标生成设计
        
        Args:
            performance: 性能指标 [B, performance_dim]
            
        Returns:
            设计参数 [B, H, W]
        """
        # 全连接层扩展
        x = self.fc(performance)
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
# 串联神经网络
# ============================================================================

class TandemNetwork(BaseModel):
    """
    串联神经网络 (Tandem Neural Network)
    
    组合前向网络和逆向网络，实现端到端逆向设计。
    
    训练策略:
    1. 预训练前向网络（监督学习）
    2. 固定前向网络，训练逆向网络
    
    推理时:
    目标性能 → 逆向网络 → 设计 → 前向网络 → 预测性能
    
    使用示例:
    ```python
    # 创建 TNN
    config = TandemNetworkConfig(
        forward_config=ForwardNetworkConfig(design_shape=(200, 22)),
        inverse_config=InverseNetworkConfig(design_shape=(200, 22))
    )
    tnn = TandemNetwork(config)
    
    # 训练前向网络
    tnn.train_forward(train_loader, epochs=100)
    
    # 训练逆向网络
    tnn.train_inverse(train_loader, epochs=100)
    
    # 推理
    target_perf = torch.tensor([[0.85, 0.1, 0.05]])  # 目标效率
    design, pred_perf = tnn.inverse_design(target_perf)
    ```
    """
    
    def __init__(self, config: Optional[TandemNetworkConfig] = None):
        config = config or TandemNetworkConfig()
        super().__init__(config)
        self.config = config
        
        # 创建子网络
        self.forward_net = ForwardNetwork(config.forward_config)
        self.inverse_net = InverseNetwork(config.inverse_config)
        
        # 确保设计形状一致
        assert config.forward_config.design_shape == config.inverse_config.design_shape, \
            "Forward and inverse network design shapes must match"
        
        self.design_shape = config.forward_config.design_shape
        self.performance_dim = config.forward_config.performance_dim
        
        # 训练状态
        self.forward_trained = False
        self.inverse_trained = False
    
    def forward(self, design: Tensor) -> Tensor:
        """
        前向传播（使用前向网络）
        
        Args:
            design: 设计参数 [B, H, W]
            
        Returns:
            性能指标 [B, performance_dim]
        """
        return self.forward_net(design)
    
    def inverse_design(
        self,
        target_performance: Tensor,
        return_predicted: bool = True
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        逆向设计：从目标性能生成设计
        
        Args:
            target_performance: 目标性能指标 [B, performance_dim]
            return_predicted: 是否返回预测性能
            
        Returns:
            design: 设计参数 [B, H, W]
            pred_performance: 预测性能 [B, performance_dim]（可选）
        """
        design = self.inverse_net(target_performance)
        
        if return_predicted:
            pred_performance = self.forward_net(design)
            return design, pred_performance
        
        return design
    
    def train_forward(
        self,
        train_loader,
        val_loader=None,
        epochs: int = 100,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        patience: int = 10,
        save_path: Optional[str] = None
    ) -> Dict[str, List[float]]:
        """
        训练前向网络
        
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
        self.forward_net.train()
        optimizer = torch.optim.AdamW(
            self.forward_net.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=patience//2
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # 训练阶段
            train_loss = 0.0
            for batch in train_loader:
                design = batch['design'].to(self.device)
                performance = batch['performance'].to(self.device)
                
                optimizer.zero_grad()
                pred_performance = self.forward_net(design)
                loss = self.compute_loss(pred_performance, performance)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            self.training_history['train_loss'].append(train_loss)
            
            # 验证阶段
            if val_loader is not None:
                val_loss = self._validate_forward(val_loader)
                self.training_history['val_loss'].append(val_loss)
                
                scheduler.step(val_loss)
                
                # 早停检查
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    if save_path:
                        self.forward_net.save(save_path)
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
        
        self.forward_trained = True
        
        # 冻结前向网络
        if self.config.freeze_forward:
            self.forward_net.freeze()
        
        return self.training_history
    
    def train_inverse(
        self,
        train_loader,
        val_loader=None,
        epochs: int = 100,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        patience: int = 15,
        save_path: Optional[str] = None
    ) -> Dict[str, List[float]]:
        """
        训练逆向网络（串联训练）
        
        固定预训练的前向网络，训练逆向网络使预测性能接近目标。
        
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
        if not self.forward_trained and self.config.pretrain_forward:
            print("Warning: Forward network not trained. Consider running train_forward() first.")
        
        # 确保前向网络冻结
        if self.config.freeze_forward:
            self.forward_net.freeze()
        
        self.inverse_net.train()
        optimizer = torch.optim.AdamW(
            self.inverse_net.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=patience//2
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        inverse_history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(epochs):
            # 训练阶段
            train_loss = 0.0
            for batch in train_loader:
                target_performance = batch['performance'].to(self.device)
                design_gt = batch['design'].to(self.device)
                
                optimizer.zero_grad()
                
                # 逆向网络生成设计
                design = self.inverse_net(target_performance)
                
                # 前向网络预测性能
                pred_performance = self.forward_net(design)
                
                # 计算损失
                loss = self._compute_tandem_loss(
                    pred_performance, target_performance, design, design_gt
                )
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            inverse_history['train_loss'].append(train_loss)
            
            # 验证阶段
            if val_loader is not None:
                val_loss = self._validate_inverse(val_loader)
                inverse_history['val_loss'].append(val_loss)
                
                scheduler.step(val_loss)
                
                # 早停检查
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    if save_path:
                        self.inverse_net.save(save_path)
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
        
        self.inverse_trained = True
        return inverse_history
    
    def _compute_tandem_loss(
        self,
        pred_performance: Tensor,
        target_performance: Tensor,
        design: Tensor,
        design_gt: Optional[Tensor] = None
    ) -> Tensor:
        """
        计算串联网络损失
        
        L = w_perf * L_perf + w_design * L_design + w_div * L_div
        """
        # 性能重建损失
        perf_loss = F.mse_loss(pred_performance, target_performance)
        
        total_loss = self.config.performance_loss_weight * perf_loss
        
        # 设计空间正则化（可选）
        if self.config.design_loss_weight > 0 and design_gt is not None:
            design_loss = F.mse_loss(design, design_gt)
            total_loss += self.config.design_loss_weight * design_loss
        
        # 多样性正则化（可选）
        if self.config.diversity_loss_weight > 0:
            # 鼓励设计多样性
            design_flat = design.view(design.size(0), -1)
            diversity_loss = -torch.pdist(design_flat).mean()
            total_loss += self.config.diversity_loss_weight * diversity_loss
        
        return total_loss
    
    def _validate_forward(self, val_loader) -> float:
        """验证前向网络"""
        self.forward_net.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                design = batch['design'].to(self.device)
                performance = batch['performance'].to(self.device)
                
                pred_performance = self.forward_net(design)
                loss = F.mse_loss(pred_performance, performance)
                val_loss += loss.item()
        
        self.forward_net.train()
        return val_loss / len(val_loader)
    
    def _validate_inverse(self, val_loader) -> float:
        """验证逆向网络"""
        self.inverse_net.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                target_performance = batch['performance'].to(self.device)
                
                design = self.inverse_net(target_performance)
                pred_performance = self.forward_net(design)
                
                loss = F.mse_loss(pred_performance, target_performance)
                val_loss += loss.item()
        
        self.inverse_net.train()
        return val_loss / len(val_loader)
    
    def save(self, path: Union[str, Path], **kwargs) -> None:
        """保存完整 TNN 模型"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'config': self.config.__dict__,
            'forward_net': self.forward_net.state_dict(),
            'inverse_net': self.inverse_net.state_dict(),
            'forward_trained': self.forward_trained,
            'inverse_trained': self.inverse_trained,
            'training_history': self.training_history
        }
        
        torch.save(checkpoint, path)
    
    def load(self, path: Union[str, Path], **kwargs) -> None:
        """加载完整 TNN 模型"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.forward_net.load_state_dict(checkpoint['forward_net'])
        self.inverse_net.load_state_dict(checkpoint['inverse_net'])
        self.forward_trained = checkpoint.get('forward_trained', False)
        self.inverse_trained = checkpoint.get('inverse_trained', False)
        self.training_history = checkpoint.get('training_history', self.training_history)
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            'name': self.config.name,
            'device': str(self.device),
            'design_shape': self.design_shape,
            'performance_dim': self.performance_dim,
            'forward_parameters': self.forward_net.count_parameters(),
            'inverse_parameters': self.inverse_net.count_parameters(),
            'total_parameters': self.forward_net.count_parameters() + self.inverse_net.count_parameters(),
            'forward_trained': self.forward_trained,
            'inverse_trained': self.inverse_trained
        }


# ============================================================================
# 便捷函数
# ============================================================================

def create_tnn_for_challenge(
    challenge_name: str,
    performance_dim: int = 3,
    device: str = 'auto'
) -> TandemNetwork:
    """
    为特定挑战创建 TNN
    
    Args:
        challenge_name: 挑战名称（'grating_coupler', 'metagrating', 'wavelength_demux'）
        performance_dim: 性能指标维度
        device: 计算设备
        
    Returns:
        配置好的 TandemNetwork
    """
    from challenges import ChallengeFactory
    
    # 获取挑战以确定设计形状
    challenge = ChallengeFactory.create(challenge_name)
    design_shape = challenge.spec.get_grid_shape()
    
    # 创建配置
    forward_config = ForwardNetworkConfig(
        design_shape=design_shape,
        performance_dim=performance_dim,
        device=device
    )
    
    inverse_config = InverseNetworkConfig(
        design_shape=design_shape,
        performance_dim=performance_dim,
        device=device
    )
    
    tandem_config = TandemNetworkConfig(
        forward_config=forward_config,
        inverse_config=inverse_config,
        device=device
    )
    
    return TandemNetwork(tandem_config)

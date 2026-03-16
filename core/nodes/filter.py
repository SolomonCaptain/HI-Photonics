"""
滤波器节点模块

提供各种滤波操作，用于平滑设计变量、边缘检测等。
在光子学逆向设计中，滤波器常用于:
1. 平滑设计变量以获得可制造的结构
2. 实现最小特征尺寸约束
3. 边缘锐化或平滑
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Union, Callable
from ..node import Node
from ..utils.typing import TensorLike


class FilterNode(Node):
    """
    滤波器节点基类
    
    对输入张量应用滤波操作，保持梯度可传播。
    """
    
    def __init__(
        self, 
        name: str, 
        input_node: Node,
        filter_type: str = 'gaussian',
        kernel_size: int = 3,
        sigma: Optional[float] = None,
        padding: str = 'same',
        device: Optional[torch.device] = None
    ):
        """
        初始化滤波器节点
        
        Args:
            name: 节点名称
            input_node: 输入节点
            filter_type: 滤波器类型 ('gaussian', 'mean', 'median', 'sobel', 'laplacian', 'dilation', 'erosion')
            kernel_size: 核大小 (奇数)
            sigma: 高斯滤波器的标准差 (仅用于高斯滤波)
            padding: 填充方式 ('same', 'valid', 'reflect', 'replicate')
            device: 计算设备
        """
        super().__init__(name)
        self.add_input(input_node)
        self.filter_type = filter_type
        self.kernel_size = kernel_size
        self.sigma = sigma or (kernel_size / 6.0)  # 默认 sigma
        self.padding = padding
        self.device = device or torch.device('cpu')
        
        # 预计算滤波核
        self._kernel = self._create_kernel()
    
    def _create_kernel(self) -> torch.Tensor:
        """根据滤波类型创建核"""
        k = self.kernel_size
        
        if self.filter_type == 'gaussian':
            # 创建 1D 高斯核
            x = torch.arange(k, dtype=torch.float32, device=self.device) - (k - 1) / 2
            gauss_1d = torch.exp(-x**2 / (2 * self.sigma**2))
            gauss_1d = gauss_1d / gauss_1d.sum()
            # 创建 2D 高斯核
            kernel = gauss_1d.unsqueeze(1) @ gauss_1d.unsqueeze(0)
            kernel = kernel.unsqueeze(0).unsqueeze(0)  # shape: (1, 1, k, k)
            
        elif self.filter_type == 'mean':
            kernel = torch.ones(1, 1, k, k, dtype=torch.float32, device=self.device) / (k * k)
            
        elif self.filter_type == 'sobel':
            # Sobel 边缘检测核 (水平和垂直组合)
            # 水平 Sobel
            sobel_x = torch.tensor([
                [-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]
            ], dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
            # 垂直 Sobel
            sobel_y = torch.tensor([
                [-1, -2, -1],
                [0, 0, 0],
                [1, 2, 1]
            ], dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
            # 如果 kernel_size > 3，需要调整
            if k != 3:
                sobel_x = F.interpolate(sobel_x, size=(k, k), mode='bilinear', align_corners=True)
                sobel_y = F.interpolate(sobel_y, size=(k, k), mode='bilinear', align_corners=True)
            kernel = torch.cat([sobel_x, sobel_y], dim=0)  # shape: (2, 1, k, k)
            
        elif self.filter_type == 'laplacian':
            # Laplacian 边缘检测核
            if k == 3:
                kernel = torch.tensor([
                    [0, 1, 0],
                    [1, -4, 1],
                    [0, 1, 0]
                ], dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(0)
            else:
                # 更大的 Laplacian 核
                kernel = torch.ones(1, 1, k, k, dtype=torch.float32, device=self.device)
                center = k // 2
                kernel[0, 0, center, center] = -(k * k - 1)
            
        elif self.filter_type == 'dilation':
            # 膨胀核 (最大值滤波)
            kernel = torch.ones(1, 1, k, k, dtype=torch.float32, device=self.device)
            
        elif self.filter_type == 'erosion':
            # 腐蚀核 (最小值滤波)
            kernel = torch.ones(1, 1, k, k, dtype=torch.float32, device=self.device)
            
        else:
            raise ValueError(f"不支持的滤波器类型: {self.filter_type}")
        
        return kernel
    
    def _get_padding_size(self) -> int:
        """计算填充大小"""
        if self.padding == 'same':
            return self.kernel_size // 2
        return 0
    
    def forward(self, **kwargs) -> TensorLike:
        """
        执行滤波操作
        
        Returns:
            滤波后的张量
        """
        input_val = self._inputs[0].forward(**kwargs)
        
        if isinstance(input_val, dict):
            raise TypeError("FilterNode 期望张量输入，不支持字典输入")
        
        x = input_val.to(self.device)
        
        # 确保输入是 4D (batch, channel, height, width)
        original_dim = x.dim()
        if original_dim == 2:
            x = x.unsqueeze(0).unsqueeze(0)  # (H, W) -> (1, 1, H, W)
        elif original_dim == 3:
            x = x.unsqueeze(0)  # (C, H, W) -> (1, C, H, W)
        
        batch, channels, height, width = x.shape
        
        # 处理不同滤波类型
        if self.filter_type in ['gaussian', 'mean', 'laplacian']:
            # 标准卷积滤波
            # 扩展核以匹配通道数
            kernel = self._kernel.expand(channels, -1, -1, -1)
            pad = self._get_padding_size()
            
            if self.padding == 'same':
                x_padded = F.pad(x, (pad, pad, pad, pad), mode='reflect')
            elif self.padding == 'reflect':
                x_padded = F.pad(x, (pad, pad, pad, pad), mode='reflect')
            elif self.padding == 'replicate':
                x_padded = F.pad(x, (pad, pad, pad, pad), mode='replicate')
            else:
                x_padded = x
            
            # 分组卷积
            result = F.conv2d(x_padded, kernel, groups=channels)
            
        elif self.filter_type == 'sobel':
            # Sobel 边缘检测: 计算 x 和 y 方向的梯度幅度
            kernel = self._kernel  # shape: (2, 1, k, k)
            pad = self._get_padding_size()
            
            if self.padding in ['same', 'reflect']:
                x_padded = F.pad(x, (pad, pad, pad, pad), mode='reflect')
            else:
                x_padded = x
            
            # 对每个通道应用 Sobel
            results = []
            for c in range(channels):
                x_c = x_padded[:, c:c+1, :, :]
                grad_x = F.conv2d(x_c, kernel[0:1])
                grad_y = F.conv2d(x_c, kernel[1:2])
                # 计算梯度幅度
                magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
                results.append(magnitude)
            result = torch.cat(results, dim=1)
            
        elif self.filter_type == 'dilation':
            # 最大值滤波 (膨胀)
            pad = self._get_padding_size()
            result = F.max_pool2d(x, self.kernel_size, stride=1, padding=pad)
            
        elif self.filter_type == 'erosion':
            # 最小值滤波 (腐蚀)
            pad = self._get_padding_size()
            result = -F.max_pool2d(-x, self.kernel_size, stride=1, padding=pad)
        
        else:
            raise ValueError(f"未知滤波类型: {self.filter_type}")
        
        # 恢复原始维度
        if original_dim == 2:
            result = result.squeeze(0).squeeze(0)
        elif original_dim == 3:
            result = result.squeeze(0)
        
        self._cached_output = result
        return result


class GaussianFilterNode(FilterNode):
    """高斯滤波器节点 - 便捷类"""
    
    def __init__(
        self, 
        name: str, 
        input_node: Node,
        kernel_size: int = 3,
        sigma: Optional[float] = None,
        **kwargs
    ):
        super().__init__(
            name, input_node, 
            filter_type='gaussian',
            kernel_size=kernel_size,
            sigma=sigma,
            **kwargs
        )


class SobelFilterNode(FilterNode):
    """Sobel 边缘检测节点 - 便捷类"""
    
    def __init__(self, name: str, input_node: Node, **kwargs):
        super().__init__(
            name, input_node,
            filter_type='sobel',
            kernel_size=3,
            **kwargs
        )


class DilationFilterNode(FilterNode):
    """膨胀滤波器节点 - 便捷类"""
    
    def __init__(
        self, 
        name: str, 
        input_node: Node,
        kernel_size: int = 3,
        **kwargs
    ):
        super().__init__(
            name, input_node,
            filter_type='dilation',
            kernel_size=kernel_size,
            **kwargs
        )


class ErosionFilterNode(FilterNode):
    """腐蚀滤波器节点 - 便捷类"""
    
    def __init__(
        self, 
        name: str, 
        input_node: Node,
        kernel_size: int = 3,
        **kwargs
    ):
        super().__init__(
            name, input_node,
            filter_type='erosion',
            kernel_size=kernel_size,
            **kwargs
        )


class BilateralFilterNode(Node):
    """
    双边滤波器节点
    
    保持边缘的同时平滑区域，用于保持结构锐度的平滑操作。
    """
    
    def __init__(
        self,
        name: str,
        input_node: Node,
        kernel_size: int = 5,
        sigma_spatial: float = 1.0,
        sigma_range: float = 0.1,
        device: Optional[torch.device] = None
    ):
        """
        Args:
            name: 节点名称
            input_node: 输入节点
            kernel_size: 核大小
            sigma_spatial: 空间标准差
            sigma_range: 值域标准差
            device: 计算设备
        """
        super().__init__(name)
        self.add_input(input_node)
        self.kernel_size = kernel_size
        self.sigma_spatial = sigma_spatial
        self.sigma_range = sigma_range
        self.device = device or torch.device('cpu')
        
        # 预计算空间权重
        self._spatial_kernel = self._create_spatial_kernel()
    
    def _create_spatial_kernel(self) -> torch.Tensor:
        """创建空间高斯核"""
        k = self.kernel_size
        x = torch.arange(k, dtype=torch.float32, device=self.device) - (k - 1) / 2
        gauss_1d = torch.exp(-x**2 / (2 * self.sigma_spatial**2))
        kernel = gauss_1d.unsqueeze(1) @ gauss_1d.unsqueeze(0)
        return kernel
    
    def forward(self, **kwargs) -> TensorLike:
        """执行双边滤波"""
        input_val = self._inputs[0].forward(**kwargs)
        
        if isinstance(input_val, dict):
            raise TypeError("BilateralFilterNode 期望张量输入")
        
        x = input_val.to(self.device)
        original_dim = x.dim()
        if original_dim == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif original_dim == 3:
            x = x.unsqueeze(0)
        
        batch, channels, height, width = x.shape
        k = self.kernel_size
        pad = k // 2
        
        # 使用 unfold 提取局部块
        x_padded = F.pad(x, (pad, pad, pad, pad), mode='reflect')
        
        # 提取局部块: (batch, channels, height, width, k, k)
        patches = x_padded.unfold(2, k, 1).unfold(3, k, 1)
        
        # 计算值域权重
        center = x.unsqueeze(-1).unsqueeze(-1)  # (batch, channels, height, width, 1, 1)
        range_weights = torch.exp(-(patches - center)**2 / (2 * self.sigma_range**2))
        
        # 组合空间和值域权重
        spatial_weights = self._spatial_kernel.view(1, 1, 1, 1, k, k)
        combined_weights = spatial_weights * range_weights
        
        # 加权平均
        result = (patches * combined_weights).sum(dim=(-2, -1)) / combined_weights.sum(dim=(-2, -1))
        
        if original_dim == 2:
            result = result.squeeze(0).squeeze(0)
        elif original_dim == 3:
            result = result.squeeze(0)
        
        self._cached_output = result
        return result


class MorphologicalOpeningNode(Node):
    """
    形态学开运算节点
    
    先腐蚀后膨胀，用于去除小的前景对象（噪点）。
    """
    
    def __init__(
        self,
        name: str,
        input_node: Node,
        kernel_size: int = 3,
        device: Optional[torch.device] = None
    ):
        super().__init__(name)
        self.add_input(input_node)
        self.kernel_size = kernel_size
        self.device = device or torch.device('cpu')
    
    def forward(self, **kwargs) -> TensorLike:
        input_val = self._inputs[0].forward(**kwargs)
        
        if isinstance(input_val, dict):
            raise TypeError("MorphologicalOpeningNode 期望张量输入")
        
        x = input_val.to(self.device)
        original_dim = x.dim()
        if original_dim == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif original_dim == 3:
            x = x.unsqueeze(0)
        
        k = self.kernel_size
        pad = k // 2
        
        # 先腐蚀
        eroded = -F.max_pool2d(-x, k, stride=1, padding=pad)
        # 再膨胀
        result = F.max_pool2d(eroded, k, stride=1, padding=pad)
        
        if original_dim == 2:
            result = result.squeeze(0).squeeze(0)
        elif original_dim == 3:
            result = result.squeeze(0)
        
        self._cached_output = result
        return result


class MorphologicalClosingNode(Node):
    """
    形态学闭运算节点
    
    先膨胀后腐蚀，用于填充小的孔洞。
    """
    
    def __init__(
        self,
        name: str,
        input_node: Node,
        kernel_size: int = 3,
        device: Optional[torch.device] = None
    ):
        super().__init__(name)
        self.add_input(input_node)
        self.kernel_size = kernel_size
        self.device = device or torch.device('cpu')
    
    def forward(self, **kwargs) -> TensorLike:
        input_val = self._inputs[0].forward(**kwargs)
        
        if isinstance(input_val, dict):
            raise TypeError("MorphologicalClosingNode 期望张量输入")
        
        x = input_val.to(self.device)
        original_dim = x.dim()
        if original_dim == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif original_dim == 3:
            x = x.unsqueeze(0)
        
        k = self.kernel_size
        pad = k // 2
        
        # 先膨胀
        dilated = F.max_pool2d(x, k, stride=1, padding=pad)
        # 再腐蚀
        result = -F.max_pool2d(-dilated, k, stride=1, padding=pad)
        
        if original_dim == 2:
            result = result.squeeze(0).squeeze(0)
        elif original_dim == 3:
            result = result.squeeze(0)
        
        self._cached_output = result
        return result

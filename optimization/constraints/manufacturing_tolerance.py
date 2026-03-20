"""
制造公差约束模块

实现制造公差相关的约束和鲁棒性优化。
包括:
1. 边缘粗糙度约束
2. 曝光/蚀刻误差约束
3. 尺寸偏差约束
4. 鲁棒性优化约束
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum
import math


class ToleranceType(Enum):
    """公差类型"""
    EDGE_ROUGHNESS = "edge_roughness"       # 边缘粗糙度
    CD_VARIATION = "cd_variation"           # 关键尺寸偏差
    OVERETCH = "overetch"                   # 过蚀刻
    UNDERETCH = "underetch"                 # 欠蚀刻
    LINEWIDTH = "linewidth"                 # 线宽偏差
    THICKNESS = "thickness"                 # 厚度偏差
    ALIGNMENT = "alignment"                 # 对准偏差


@dataclass
class ManufacturingSpecs:
    """制造规格数据类"""
    # 关键尺寸公差
    cd_tolerance: float = 0.01  # μm (±10nm)
    linewidth_tolerance: float = 0.01  # μm
    thickness_tolerance: float = 0.005  # μm (±5nm)
    
    # 边缘粗糙度
    edge_roughness_rms: float = 0.002  # μm (2nm RMS)
    
    # 对准精度
    alignment_tolerance: float = 0.02  # μm (±20nm)
    
    # 最小特征尺寸
    min_feature_size: float = 0.1  # μm (100nm)
    
    # 制造过程参数
    etch_bias: float = 0.01  # μm (蚀刻偏差)
    sidewall_angle: float = 88.0  # 度 (侧壁角度)
    
    # 设计规则
    min_gap: float = 0.1  # μm
    min_line: float = 0.1  # μm


# 典型工艺节点的制造规格
PROCESS_SPECS: Dict[str, ManufacturingSpecs] = {
    "180nm": ManufacturingSpecs(
        cd_tolerance=0.02,
        linewidth_tolerance=0.02,
        edge_roughness_rms=0.005,
        min_feature_size=0.18,
        min_gap=0.18,
        min_line=0.18,
    ),
    "90nm": ManufacturingSpecs(
        cd_tolerance=0.01,
        linewidth_tolerance=0.01,
        edge_roughness_rms=0.003,
        min_feature_size=0.09,
        min_gap=0.09,
        min_line=0.09,
    ),
    "45nm": ManufacturingSpecs(
        cd_tolerance=0.005,
        linewidth_tolerance=0.005,
        edge_roughness_rms=0.002,
        min_feature_size=0.045,
        min_gap=0.045,
        min_line=0.045,
    ),
    "photonics_220nm": ManufacturingSpecs(
        cd_tolerance=0.01,
        linewidth_tolerance=0.01,
        edge_roughness_rms=0.003,
        min_feature_size=0.1,
        min_gap=0.1,
        min_line=0.1,
        thickness_tolerance=0.01,  # 220nm SOI 平台
    ),
}


class EdgeRoughnessModel(nn.Module):
    """
    边缘粗糙度模型
    
    模拟制造过程中产生的边缘粗糙度（LER/LWR）。
    使用自相关函数建模边缘粗糙度。
    """
    
    def __init__(
        self,
        rms_roughness: float = 0.003,  # μm (3nm RMS)
        correlation_length: float = 0.02,  # μm
        resolution: float = 0.01,  # μm/pixel
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            rms_roughness: RMS 粗糙度
            correlation_length: 相关长度
            resolution: 网格分辨率
            device: 计算设备
        """
        super().__init__()
        self.rms_roughness = rms_roughness
        self.correlation_length = correlation_length
        self.resolution = resolution
        self.device = device or torch.device('cpu')
    
    def generate_roughness_field(
        self,
        shape: Tuple[int, int],
        num_samples: int = 1,
    ) -> torch.Tensor:
        """
        生成粗糙度场
        
        使用指数相关函数: R(τ) = σ² exp(-τ/ξ)
        
        Args:
            shape: 空间形状 (H, W)
            num_samples: 样本数
            
        Returns:
            粗糙度场 [num_samples, H, W]
        """
        H, W = shape
        
        # 生成高斯白噪声
        noise = torch.randn(num_samples, H, W, device=self.device)
        
        # 创建高斯滤波器进行空间相关
        # 滤波器尺寸基于相关长度
        filter_size = int(6 * self.correlation_length / self.resolution)
        filter_size = max(3, filter_size if filter_size % 2 == 1 else filter_size + 1)
        
        # 高斯核
        x = torch.linspace(-filter_size // 2, filter_size // 2, filter_size, device=self.device)
        y = torch.linspace(-filter_size // 2, filter_size // 2, filter_size, device=self.device)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        
        sigma_filter = self.correlation_length / self.resolution
        gaussian_kernel = torch.exp(-(X**2 + Y**2) / (2 * sigma_filter**2))
        gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
        gaussian_kernel = gaussian_kernel.view(1, 1, filter_size, filter_size)
        
        # 应用滤波器
        noise_padded = F.pad(noise.unsqueeze(1), (filter_size//2,) * 4, mode='reflect')
        roughness = F.conv2d(noise_padded, gaussian_kernel)
        roughness = roughness.squeeze(1)
        
        # 归一化到目标 RMS
        roughness = roughness / (roughness.std() + 1e-8) * self.rms_roughness
        
        return roughness
    
    def apply_edge_roughness(
        self,
        design: torch.Tensor,
        num_samples: int = 5,
    ) -> torch.Tensor:
        """
        对设计应用边缘粗糙度
        
        Args:
            design: 设计参数 [H, W] 或 [B, H, W]
            num_samples: 蒙特卡洛样本数
            
        Returns:
            带粗糙度的设计样本 [num_samples, H, W]
        """
        if design.dim() == 2:
            design = design.unsqueeze(0)
        
        # 取第一个样本
        design_2d = design[0]
        H, W = design_2d.shape
        
        # 生成粗糙度场
        roughness = self.generate_roughness_field((H, W), num_samples)
        
        # 计算设计梯度（边缘位置）
        grad_x = design_2d[:, 1:] - design_2d[:, :-1]
        grad_y = design_2d[1:, :] - design_2d[:-1, :]
        
        grad_x = F.pad(grad_x.unsqueeze(0).unsqueeze(0), (0, 1, 0, 0), mode='replicate')
        grad_y = F.pad(grad_y.unsqueeze(0).unsqueeze(0), (0, 0, 1, 0), mode='replicate')
        
        # 梯度模
        grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8).squeeze()
        
        # 只在边缘附近应用粗糙度
        edge_mask = (grad_mag > 0.1).float()
        
        # 扩展设计并应用粗糙度
        designs_with_roughness = design_2d.unsqueeze(0).expand(num_samples, -1, -1).clone()
        
        for i in range(num_samples):
            # 粗糙度扰动（在边缘梯度方向）
            perturbation = roughness[i] * edge_mask * torch.sign(grad_x.squeeze() + grad_y.squeeze() + 1e-8)
            designs_with_roughness[i] = torch.clamp(designs_with_roughness[i] + perturbation, 0, 1)
        
        return designs_with_roughness


class EtchBiasModel(nn.Module):
    """
    蚀刻偏差模型
    
    模拟过蚀刻或欠蚀刻导致的设计偏差。
    """
    
    def __init__(
        self,
        max_bias: float = 0.01,  # μm
        bias_type: str = 'isotropic',  # 'isotropic', 'anisotropic'
        resolution: float = 0.01,  # μm/pixel
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            max_bias: 最大偏差
            bias_type: 偏差类型
            resolution: 网格分辨率
            device: 计算设备
        """
        super().__init__()
        self.max_bias = max_bias
        self.bias_type = bias_type
        self.resolution = resolution
        self.device = device or torch.device('cpu')
    
    def apply_bias(
        self,
        design: torch.Tensor,
        bias_value: Optional[float] = None,
    ) -> torch.Tensor:
        """
        应用蚀刻偏差
        
        Args:
            design: 设计参数 [H, W]
            bias_value: 偏差值（正值=过蚀刻，负值=欠蚀刻）
            
        Returns:
            带偏差的设计
        """
        if bias_value is None:
            bias_value = self.max_bias
        
        if design.dim() == 2:
            design = design.unsqueeze(0).unsqueeze(0)
        elif design.dim() == 3:
            design = design.unsqueeze(0)
        
        # 偏差转换为像素数
        bias_pixels = abs(bias_value) / self.resolution
        
        if bias_value > 0:  # 过蚀刻 - 腐蚀
            result = self._erode(design, bias_pixels)
        else:  # 欠蚀刻 - 膨胀
            result = self._dilate(design, bias_pixels)
        
        return result.squeeze()
    
    def _erode(self, x: torch.Tensor, radius: float) -> torch.Tensor:
        """形态学腐蚀"""
        k = 2 * int(radius) + 1
        return -F.max_pool2d(-x, k, stride=1, padding=int(radius))
    
    def _dilate(self, x: torch.Tensor, radius: float) -> torch.Tensor:
        """形态学膨胀"""
        k = 2 * int(radius) + 1
        return F.max_pool2d(x, k, stride=1, padding=int(radius))
    
    def generate_bias_samples(
        self,
        design: torch.Tensor,
        num_samples: int = 5,
        bias_range: Tuple[float, float] = (-0.01, 0.01),
    ) -> torch.Tensor:
        """
        生成不同偏差的样本
        
        Args:
            design: 设计参数
            num_samples: 样本数
            bias_range: 偏差范围 (μm)
            
        Returns:
            偏差样本 [num_samples, H, W]
        """
        if design.dim() == 2:
            H, W = design.shape
        else:
            H, W = design.shape[-2], design.shape[-1]
        
        samples = []
        biases = torch.linspace(bias_range[0], bias_range[1], num_samples)
        
        for bias in biases:
            sample = self.apply_bias(design, bias.item())
            samples.append(sample)
        
        return torch.stack(samples)


class RobustnessConstraint(nn.Module):
    """
    鲁棒性约束模块
    
    确保设计在制造公差范围内性能稳定。
    使用蒙特卡洛方法评估性能方差。
    """
    
    def __init__(
        self,
        specs: Union[str, ManufacturingSpecs] = "photonics_220nm",
        num_mc_samples: int = 10,
        resolution: float = 0.01,  # μm/pixel
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            specs: 制造规格（字符串或规格对象）
            num_mc_samples: 蒙特卡洛样本数
            resolution: 网格分辨率
            device: 计算设备
        """
        super().__init__()
        
        if isinstance(specs, str):
            if specs not in PROCESS_SPECS:
                raise ValueError(f"未知工艺: {specs}，可用: {list(PROCESS_SPECS.keys())}")
            self.specs = PROCESS_SPECS[specs]
        else:
            self.specs = specs
        
        self.num_mc_samples = num_mc_samples
        self.resolution = resolution
        self.device = device or torch.device('cpu')
        
        # 初始化子模型
        self.edge_roughness_model = EdgeRoughnessModel(
            rms_roughness=self.specs.edge_roughness_rms,
            resolution=resolution,
            device=device,
        )
        
        self.etch_bias_model = EtchBiasModel(
            max_bias=self.specs.etch_bias,
            resolution=resolution,
            device=device,
        )
    
    def generate_perturbed_samples(
        self,
        design: torch.Tensor,
        perturbation_types: List[str] = ['roughness', 'bias'],
    ) -> torch.Tensor:
        """
        生成带制造扰动的样本
        
        Args:
            design: 设计参数 [H, W]
            perturbation_types: 扰动类型列表
            
        Returns:
            扰动样本 [num_samples, H, W]
        """
        samples = []
        
        for _ in range(self.num_mc_samples):
            sample = design.clone()
            
            if 'roughness' in perturbation_types:
                rough_samples = self.edge_roughness_model.apply_edge_roughness(
                    sample.unsqueeze(0) if sample.dim() == 2 else sample,
                    num_samples=1
                )
                sample = rough_samples[0]
            
            if 'bias' in perturbation_types:
                # 随机偏差值
                bias = torch.empty(1, device=self.device).uniform_(
                    -self.specs.cd_tolerance,
                    self.specs.cd_tolerance
                ).item()
                sample = self.etch_bias_model.apply_bias(sample, bias)
            
            samples.append(sample)
        
        return torch.stack(samples)
    
    def compute_robustness_loss(
        self,
        design: torch.Tensor,
        performance_fn,  # Callable[[torch.Tensor], torch.Tensor]
        target_performance: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        计算鲁棒性损失
        
        Args:
            design: 设计参数
            performance_fn: 性能评估函数
            target_performance: 目标性能（可选）
            
        Returns:
            鲁棒性损失字典
        """
        # 生成扰动样本
        perturbed_samples = self.generate_perturbed_samples(design)
        
        # 评估各样本性能
        performances = []
        for i in range(perturbed_samples.size(0)):
            perf = performance_fn(perturbed_samples[i])
            performances.append(perf)
        
        performances = torch.stack(performances)
        
        # 计算统计量
        mean_performance = performances.mean(dim=0)
        std_performance = performances.std(dim=0)
        
        # 性能退化
        if target_performance is not None:
            degradation = F.relu(target_performance - mean_performance)
        else:
            # 使用第一个样本作为参考
            degradation = F.relu(performances[0] - mean_performance)
        
        # 鲁棒性损失
        losses = {
            'performance_mean': mean_performance,
            'performance_std': std_performance,
            'performance_variance': (std_performance ** 2).mean(),
            'degradation': degradation.mean() if degradation.numel() > 1 else degradation,
            'robustness_penalty': std_performance.mean(),
        }
        
        return losses
    
    def forward(
        self,
        design: torch.Tensor,
        weight: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播：计算设计对扰动的敏感度
        
        Args:
            design: 设计参数
            weight: 约束权重
            
        Returns:
            敏感度指标
        """
        # 计算设计梯度敏感度
        if design.dim() == 2:
            design = design.unsqueeze(0).unsqueeze(0)
        elif design.dim() == 3:
            design = design.unsqueeze(0)
        
        # Sobel 滤波器检测边缘
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype=torch.float32, device=self.device
        ).view(1, 1, 3, 3)
        
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            dtype=torch.float32, device=self.device
        ).view(1, 1, 3, 3)
        
        grad_x = F.conv2d(design, sobel_x, padding=1)
        grad_y = F.conv2d(design, sobel_y, padding=1)
        
        # 边缘强度
        edge_strength = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
        
        # 边缘周长
        perimeter = edge_strength.sum()
        
        # 敏感度指标
        sensitivity = {
            'edge_perimeter': perimeter * weight,
            'edge_density': edge_strength.mean() * weight,
            'sensitivity_map': edge_strength.squeeze(),
        }
        
        return sensitivity


class CDVariationConstraint(nn.Module):
    """
    关键尺寸（CD）偏差约束
    
    确保设计的特征尺寸在制造公差范围内。
    """
    
    def __init__(
        self,
        target_cd: float,
        tolerance: float = 0.01,  # μm
        resolution: float = 0.01,  # μm/pixel
        weight: float = 1.0,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            target_cd: 目标关键尺寸 (μm)
            tolerance: 尺寸公差 (μm)
            resolution: 网格分辨率
            weight: 约束权重
            device: 计算设备
        """
        super().__init__()
        self.target_cd = target_cd
        self.tolerance = tolerance
        self.resolution = resolution
        self.weight = weight
        self.device = device or torch.device('cpu')
    
    def forward(self, design: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        计算 CD 偏差约束
        
        Args:
            design: 设计参数 [H, W]
            
        Returns:
            约束字典
        """
        if design.dim() == 2:
            design = design.unsqueeze(0).unsqueeze(0)
        elif design.dim() == 3:
            design = design.unsqueeze(0)
        
        # 检测特征尺寸
        # 使用腐蚀操作检测可制造的线条
        tolerance_pixels = int(self.tolerance / self.resolution)
        k = 2 * tolerance_pixels + 1
        
        # 检测过小的实体区域
        eroded = -F.max_pool2d(-design, k, stride=1, padding=tolerance_pixels)
        small_features = design - eroded
        
        # 检测过小的间隙
        dilated = F.max_pool2d(design, k, stride=1, padding=tolerance_pixels)
        small_gaps = dilated - design
        
        # 计算违反程度
        feature_violation = small_features.abs().mean()
        gap_violation = small_gaps.abs().mean()
        
        return {
            'cd_violation': (feature_violation + gap_violation) * self.weight,
            'feature_violation': feature_violation,
            'gap_violation': gap_violation,
        }


class DesignRuleCheck(nn.Module):
    """
    设计规则检查（DRC）模块
    
    检查设计是否满足制造设计规则。
    """
    
    def __init__(
        self,
        specs: Union[str, ManufacturingSpecs] = "photonics_220nm",
        resolution: float = 0.01,  # μm/pixel
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            specs: 制造规格
            resolution: 网格分辨率
            device: 计算设备
        """
        super().__init__()
        
        if isinstance(specs, str):
            self.specs = PROCESS_SPECS[specs]
        else:
            self.specs = specs
        
        self.resolution = resolution
        self.device = device or torch.device('cpu')
    
    def check_min_feature_size(
        self,
        design: torch.Tensor,
        threshold: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        """
        检查最小特征尺寸
        
        Args:
            design: 设计参数
            threshold: 二值化阈值
            
        Returns:
            检查结果
        """
        min_feature_pixels = int(self.specs.min_feature_size / self.resolution)
        k = 2 * min_feature_pixels + 1
        
        if design.dim() == 2:
            design = design.unsqueeze(0).unsqueeze(0)
        
        binary = (design >= threshold).float()
        
        # 开运算检测小特征
        eroded = -F.max_pool2d(-binary, k, stride=1, padding=min_feature_pixels)
        opened = F.max_pool2d(eroded, k, stride=1, padding=min_feature_pixels)
        
        violations = (binary - opened).abs()
        
        return {
            'min_feature_violations': violations,
            'violation_count': (violations > 0.1).float().sum(),
            'violation_area': violations.mean(),
        }
    
    def check_min_spacing(
        self,
        design: torch.Tensor,
        threshold: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        """
        检查最小间距
        
        Args:
            design: 设计参数
            threshold: 二值化阈值
            
        Returns:
            检查结果
        """
        min_gap_pixels = int(self.specs.min_gap / self.resolution)
        k = 2 * min_gap_pixels + 1
        
        if design.dim() == 2:
            design = design.unsqueeze(0).unsqueeze(0)
        
        binary = (design >= threshold).float()
        inverted = 1 - binary
        
        # 检测过小的间隙（在反转图像中检测小特征）
        eroded = -F.max_pool2d(-inverted, k, stride=1, padding=min_gap_pixels)
        opened = F.max_pool2d(eroded, k, stride=1, padding=min_gap_pixels)
        
        violations = (inverted - opened).abs()
        
        return {
            'min_spacing_violations': violations,
            'violation_count': (violations > 0.1).float().sum(),
            'violation_area': violations.mean(),
        }
    
    def run_all_checks(
        self,
        design: torch.Tensor,
        threshold: float = 0.5,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        运行所有设计规则检查
        
        Args:
            design: 设计参数
            threshold: 二值化阈值
            
        Returns:
            所有检查结果
        """
        results = {
            'min_feature': self.check_min_feature_size(design, threshold),
            'min_spacing': self.check_min_spacing(design, threshold),
        }
        
        # 计算总违反率
        total_violation = (
            results['min_feature']['violation_area'] + 
            results['min_spacing']['violation_area']
        )
        results['total_violation'] = total_violation
        
        return results


# 便捷函数
def get_process_specs(process_name: str) -> ManufacturingSpecs:
    """获取工艺规格"""
    if process_name not in PROCESS_SPECS:
        raise ValueError(f"未知工艺: {process_name}，可用: {list(PROCESS_SPECS.keys())}")
    return PROCESS_SPECS[process_name]


def list_available_processes() -> List[str]:
    """列出可用的工艺节点"""
    return list(PROCESS_SPECS.keys())

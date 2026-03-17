"""
光子学数据加载管道

提供用于训练逆向设计模型的数据加载器。
"""

from typing import Dict, Optional, Tuple, List, Union, Any, Callable
from dataclasses import dataclass
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import h5py


@dataclass
class PhotonicsDataSample:
    """光子学数据样本"""
    design: torch.Tensor          # 设计参数 [H, W]
    performance: torch.Tensor      # 性能指标 [performance_dim]
    spec: Optional[Dict] = None    # 设计规格
    metadata: Optional[Dict] = None  # 元数据


class PhotonicsDataset(Dataset):
    """
    光子学数据集
    
    存储和管理设计-性能数据对。
    """
    
    def __init__(
        self,
        designs: np.ndarray,
        performances: np.ndarray,
        specs: Optional[List[Dict]] = None,
        transform: Optional[Callable] = None,
        normalize: bool = True
    ):
        """
        Args:
            designs: 设计参数数组 [N, H, W]
            performances: 性能指标数组 [N, performance_dim]
            specs: 设计规格列表
            transform: 数据变换函数
            normalize: 是否归一化
        """
        self.designs = torch.from_numpy(designs).float()
        self.performances = torch.from_numpy(performances).float()
        self.specs = specs or [None] * len(designs)
        self.transform = transform
        
        # 归一化
        self.normalize = normalize
        if normalize:
            self._compute_normalization()
    
    def _compute_normalization(self):
        """计算归一化参数"""
        self.design_mean = self.designs.mean()
        self.design_std = self.designs.std() + 1e-8
        self.perf_mean = self.performances.mean(dim=0)
        self.perf_std = self.performances.std(dim=0) + 1e-8
    
    def __len__(self) -> int:
        return len(self.designs)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        design = self.designs[idx]
        performance = self.performances[idx]
        
        # 归一化
        if self.normalize:
            design = (design - self.design_mean) / self.design_std
            performance = (performance - self.perf_mean) / self.perf_std
        
        # 应用变换
        if self.transform:
            design = self.transform(design)
        
        return {
            'design': design,
            'performance': performance,
            'spec': self.specs[idx]
        }
    
    def get_raw(self, idx: int) -> PhotonicsDataSample:
        """获取原始数据（未归一化）"""
        return PhotonicsDataSample(
            design=self.designs[idx],
            performance=self.performances[idx],
            spec=self.specs[idx]
        )
    
    def split(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42
    ) -> Tuple['PhotonicsDataset', 'PhotonicsDataset', 'PhotonicsDataset']:
        """
        划分数据集
        
        Args:
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            seed: 随机种子
            
        Returns:
            (train_dataset, val_dataset, test_dataset)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
        
        total_size = len(self)
        train_size = int(total_size * train_ratio)
        val_size = int(total_size * val_ratio)
        test_size = total_size - train_size - val_size
        
        generator = torch.Generator().manual_seed(seed)
        train_dataset, val_dataset, test_dataset = random_split(
            self,
            [train_size, val_size, test_size],
            generator=generator
        )
        
        return train_dataset, val_dataset, test_dataset


class HDF5Dataset(Dataset):
    """
    HDF5 格式数据集
    
    支持大规模数据的延迟加载。
    """
    
    def __init__(
        self,
        filepath: str,
        design_key: str = 'designs',
        performance_key: str = 'performances',
        transform: Optional[Callable] = None
    ):
        """
        Args:
            filepath: HDF5 文件路径
            design_key: 设计数据在文件中的键名
            performance_key: 性能数据在文件中的键名
            transform: 数据变换函数
        """
        self.filepath = Path(filepath)
        self.design_key = design_key
        self.performance_key = performance_key
        self.transform = transform
        
        # 检查文件是否存在
        if not self.filepath.exists():
            raise FileNotFoundError(f"HDF5 file not found: {filepath}")
        
        # 获取数据集大小
        with h5py.File(self.filepath, 'r') as f:
            self.length = f[design_key].shape[0]
            self.design_shape = f[design_key].shape[1:]
            self.performance_dim = f[performance_key].shape[1]
    
    def __len__(self) -> int:
        return self.length
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        with h5py.File(self.filepath, 'r') as f:
            design = torch.from_numpy(f[self.design_key][idx]).float()
            performance = torch.from_numpy(f[self.performance_key][idx]).float()
        
        if self.transform:
            design = self.transform(design)
        
        return {
            'design': design,
            'performance': performance
        }


class SyntheticDataset(Dataset):
    """
    合成数据集
    
    用于测试和快速原型开发。
    """
    
    def __init__(
        self,
        num_samples: int = 1000,
        design_shape: Tuple[int, int] = (200, 22),
        performance_dim: int = 3,
        noise_level: float = 0.1,
        seed: int = 42
    ):
        """
        Args:
            num_samples: 样本数量
            design_shape: 设计形状
            performance_dim: 性能维度
            noise_level: 噪声水平
            seed: 随机种子
        """
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        self.num_samples = num_samples
        self.design_shape = design_shape
        self.performance_dim = performance_dim
        
        # 生成随机设计
        self.designs = torch.rand(num_samples, *design_shape)
        
        # 生成模拟性能（基于设计的简单函数）
        self.performances = self._generate_performance(noise_level)
    
    def _generate_performance(self, noise_level: float) -> torch.Tensor:
        """根据设计生成模拟性能"""
        performances = []
        
        for i in range(self.num_samples):
            design = self.designs[i]
            
            # 模拟性能指标
            # 1. 平均填充因子 -> 效率
            efficiency = design.mean().item()
            
            # 2. 均匀性 -> 带宽
            uniformity = 1 - design.std().item()
            
            # 3. 复杂度 -> 损耗
            complexity = torch.abs(design[1:, :] - design[:-1, :]).mean().item()
            loss = complexity * 0.5
            
            performances.append([efficiency, uniformity, loss])
        
        perf = torch.tensor(performances)
        
        # 添加噪声
        if noise_level > 0:
            perf += torch.randn_like(perf) * noise_level
        
        return perf
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'design': self.designs[idx],
            'performance': self.performances[idx]
        }


class DataAugmentation:
    """
    数据增强
    
    提供设计参数的数据增强方法。
    """
    
    def __init__(
        self,
        horizontal_flip: bool = True,
        vertical_flip: bool = False,
        rotation: bool = False,
        noise_injection: float = 0.0
    ):
        """
        Args:
            horizontal_flip: 水平翻转
            vertical_flip: 垂直翻转
            rotation: 旋转（90度）
            noise_injection: 噪声注入强度
        """
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.rotation = rotation
        self.noise_injection = noise_injection
    
    def __call__(self, design: torch.Tensor) -> torch.Tensor:
        """应用数据增强"""
        # 水平翻转
        if self.horizontal_flip and torch.rand(1).item() > 0.5:
            design = torch.flip(design, dims=[-1])
        
        # 垂直翻转
        if self.vertical_flip and torch.rand(1).item() > 0.5:
            design = torch.flip(design, dims=[-2])
        
        # 旋转
        if self.rotation and torch.rand(1).item() > 0.5:
            k = torch.randint(1, 4, (1,)).item()
            design = torch.rot90(design, k, dims=[-2, -1])
        
        # 噪声注入
        if self.noise_injection > 0:
            noise = torch.randn_like(design) * self.noise_injection
            design = design + noise
            design = torch.clamp(design, 0, 1)
        
        return design


def create_dataloaders(
    dataset: Dataset,
    batch_size: int = 32,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    num_workers: int = 0,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建数据加载器
    
    Args:
        dataset: 数据集
        batch_size: 批次大小
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        num_workers: 工作进程数
        seed: 随机种子
        
    Returns:
        (train_loader, val_loader, test_loader)
    """
    # 划分数据集
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size
    
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=generator
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def save_dataset_to_hdf5(
    dataset: Dataset,
    filepath: str,
    design_key: str = 'designs',
    performance_key: str = 'performances',
    metadata: Optional[Dict] = None
):
    """
    将数据集保存为 HDF5 格式
    
    Args:
        dataset: 数据集
        filepath: 保存路径
        design_key: 设计数据键名
        performance_key: 性能数据键名
        metadata: 元数据
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # 收集所有数据
    designs = []
    performances = []
    
    for i in range(len(dataset)):
        sample = dataset[i]
        designs.append(sample['design'].numpy())
        performances.append(sample['performance'].numpy())
    
    designs = np.stack(designs)
    performances = np.stack(performances)
    
    # 保存到 HDF5
    with h5py.File(filepath, 'w') as f:
        f.create_dataset(design_key, data=designs, compression='gzip')
        f.create_dataset(performance_key, data=performances, compression='gzip')
        
        if metadata:
            for key, value in metadata.items():
                f.attrs[key] = value
    
    print(f"Dataset saved to {filepath}")

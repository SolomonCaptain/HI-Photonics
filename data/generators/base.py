"""
数据生成器基类

定义所有数据生成策略的统一接口。
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List, Callable, Union
from dataclasses import dataclass, field
from pathlib import Path
import torch
import numpy as np


@dataclass
class GeneratorConfig:
    """数据生成器配置基类"""
    name: str = "base_generator"
    seed: int = 42
    device: str = "auto"
    
    def get_device(self) -> torch.device:
        """获取计算设备"""
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)
    
    def get_rng(self) -> np.random.Generator:
        """获取随机数生成器"""
        return np.random.default_rng(self.seed)


@dataclass
class GenerationResult:
    """数据生成结果"""
    designs: np.ndarray          # 设计参数 [N, ...]
    performances: np.ndarray     # 性能指标 [N, performance_dim]
    metadata: Dict[str, Any] = field(default_factory=dict)
    uncertainties: Optional[np.ndarray] = None  # 不确定性估计 [N, performance_dim]
    
    def __len__(self) -> int:
        return len(self.designs)
    
    def to_tensors(self, device: Optional[torch.device] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """转换为 PyTorch 张量"""
        designs = torch.from_numpy(self.designs).float()
        performances = torch.from_numpy(self.performances).float()
        
        if device is not None:
            designs = designs.to(device)
            performances = performances.to(device)
        
        return designs, performances
    
    def split(
        self,
        ratios: Tuple[float, ...] = (0.8, 0.1, 0.1),
        seed: int = 42
    ) -> List['GenerationResult']:
        """
        划分结果集
        
        Args:
            ratios: 各部分比例
            seed: 随机种子
            
        Returns:
            划分后的结果列表
        """
        assert abs(sum(ratios) - 1.0) < 1e-6, "比例之和必须为 1"
        
        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(self))
        
        results = []
        start = 0
        for ratio in ratios:
            end = start + int(len(self) * ratio)
            idx = indices[start:end]
            
            result = GenerationResult(
                designs=self.designs[idx],
                performances=self.performances[idx],
                metadata=self.metadata.copy(),
                uncertainties=self.uncertainties[idx] if self.uncertainties is not None else None
            )
            results.append(result)
            start = end
        
        # 处理剩余样本
        if start < len(self):
            idx = indices[start:]
            results[-1].designs = np.concatenate([results[-1].designs, self.designs[idx]])
            results[-1].performances = np.concatenate([results[-1].performances, self.performances[idx]])
            if self.uncertainties is not None:
                results[-1].uncertainties = np.concatenate([results[-1].uncertainties, self.uncertainties[idx]])
        
        return results


class DataGenerator(ABC):
    """
    数据生成器基类
    
    定义数据生成的统一接口，支持：
    - 批量生成
    - 增量生成
    - 条件生成
    """
    
    def __init__(self, config: Optional[GeneratorConfig] = None):
        self.config = config or GeneratorConfig()
        self.device = self.config.get_device()
        self.rng = self.config.get_rng()
        self._generation_count = 0
    
    @abstractmethod
    def generate(
        self,
        num_samples: int,
        **kwargs
    ) -> GenerationResult:
        """
        生成数据
        
        Args:
            num_samples: 生成样本数量
            
        Returns:
            生成结果
        """
        pass
    
    def generate_batch(
        self,
        total_samples: int,
        batch_size: int = 1000,
        callback: Optional[Callable[[int, GenerationResult], None]] = None
    ) -> GenerationResult:
        """
        批量生成数据
        
        Args:
            total_samples: 总样本数
            batch_size: 每批样本数
            callback: 每批完成后的回调函数
            
        Returns:
            合并后的生成结果
        """
        designs_list = []
        performances_list = []
        uncertainties_list = []
        metadata = {}
        
        num_batches = (total_samples + batch_size - 1) // batch_size
        
        for i in range(num_batches):
            current_batch_size = min(batch_size, total_samples - i * batch_size)
            result = self.generate(current_batch_size)
            
            designs_list.append(result.designs)
            performances_list.append(result.performances)
            
            if result.uncertainties is not None:
                uncertainties_list.append(result.uncertainties)
            
            if i == 0:
                metadata = result.metadata.copy()
            
            if callback is not None:
                callback(i + 1, result)
        
        # 合并结果
        final_result = GenerationResult(
            designs=np.concatenate(designs_list, axis=0),
            performances=np.concatenate(performances_list, axis=0),
            metadata=metadata,
            uncertainties=np.concatenate(uncertainties_list, axis=0) if uncertainties_list else None
        )
        
        self._generation_count += total_samples
        
        return final_result
    
    def generate_incremental(
        self,
        num_samples: int,
        existing_result: Optional[GenerationResult] = None
    ) -> GenerationResult:
        """
        增量生成数据
        
        在已有数据基础上生成新数据。
        
        Args:
            num_samples: 新增样本数
            existing_result: 已有数据
            
        Returns:
            合并后的结果
        """
        new_result = self.generate(num_samples)
        
        if existing_result is None:
            return new_result
        
        return GenerationResult(
            designs=np.concatenate([existing_result.designs, new_result.designs], axis=0),
            performances=np.concatenate([existing_result.performances, new_result.performances], axis=0),
            metadata={**existing_result.metadata, **new_result.metadata},
            uncertainties=np.concatenate([existing_result.uncertainties, new_result.uncertainties], axis=0)
            if existing_result.uncertainties is not None and new_result.uncertainties is not None
            else None
        )
    
    def reset(self, seed: Optional[int] = None) -> None:
        """
        重置生成器状态
        
        Args:
            seed: 新的随机种子
        """
        if seed is not None:
            self.config.seed = seed
        self.rng = self.config.get_rng()
        self._generation_count = 0
    
    def get_generation_count(self) -> int:
        """获取已生成的总样本数"""
        return self._generation_count


class ConditionalGenerator(DataGenerator):
    """
    条件生成器基类
    
    支持基于条件的数据生成。
    """
    
    @abstractmethod
    def generate_conditional(
        self,
        conditions: np.ndarray,
        **kwargs
    ) -> GenerationResult:
        """
        条件生成
        
        Args:
            conditions: 条件数组 [N, condition_dim]
            
        Returns:
            生成结果
        """
        pass
    
    def generate(
        self,
        num_samples: int,
        conditions: Optional[np.ndarray] = None,
        **kwargs
    ) -> GenerationResult:
        """
        生成数据（支持条件）
        
        Args:
            num_samples: 生成样本数
            conditions: 可选的条件数组
            
        Returns:
            生成结果
        """
        if conditions is not None:
            return self.generate_conditional(conditions, **kwargs)
        return self._generate_unconditional(num_samples, **kwargs)
    
    @abstractmethod
    def _generate_unconditional(
        self,
        num_samples: int,
        **kwargs
    ) -> GenerationResult:
        """无条件生成"""
        pass


class SimulatorBasedGenerator(DataGenerator):
    """
    基于仿真器的数据生成器
    
    使用仿真器评估生成的设计。
    """
    
    def __init__(
        self,
        config: Optional[GeneratorConfig] = None,
        simulator: Optional[Callable] = None
    ):
        super().__init__(config)
        self.simulator = simulator
    
    def set_simulator(self, simulator: Callable) -> None:
        """设置仿真器"""
        self.simulator = simulator
    
    def evaluate_designs(
        self,
        designs: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        """
        评估设计
        
        Args:
            designs: 设计参数数组
            
        Returns:
            性能指标数组
        """
        if self.simulator is None:
            raise ValueError("仿真器未设置，请调用 set_simulator() 或在初始化时提供")
        
        performances = []
        for design in designs:
            perf = self.simulator(design, **kwargs)
            performances.append(perf)
        
        return np.array(performances)


def save_generation_result(
    result: GenerationResult,
    filepath: Union[str, Path],
    format: str = "hdf5"
) -> None:
    """
    保存生成结果
    
    Args:
        result: 生成结果
        filepath: 保存路径
        format: 保存格式 ('hdf5', 'npz', 'pt')
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if format == "hdf5":
        import h5py
        with h5py.File(filepath, 'w') as f:
            f.create_dataset('designs', data=result.designs, compression='gzip')
            f.create_dataset('performances', data=result.performances, compression='gzip')
            if result.uncertainties is not None:
                f.create_dataset('uncertainties', data=result.uncertainties, compression='gzip')
            for key, value in result.metadata.items():
                f.attrs[key] = str(value)
    
    elif format == "npz":
        data = {
            'designs': result.designs,
            'performances': result.performances,
        }
        if result.uncertainties is not None:
            data['uncertainties'] = result.uncertainties
        np.savez(filepath, **data)
    
    elif format == "pt":
        torch.save({
            'designs': torch.from_numpy(result.designs),
            'performances': torch.from_numpy(result.performances),
            'uncertainties': torch.from_numpy(result.uncertainties) if result.uncertainties is not None else None,
            'metadata': result.metadata
        }, filepath)
    
    else:
        raise ValueError(f"不支持的格式: {format}")


def load_generation_result(
    filepath: Union[str, Path],
    format: str = "auto"
) -> GenerationResult:
    """
    加载生成结果
    
    Args:
        filepath: 文件路径
        format: 文件格式 ('auto', 'hdf5', 'npz', 'pt')
        
    Returns:
        生成结果
    """
    filepath = Path(filepath)
    
    if format == "auto":
        suffix = filepath.suffix.lower()
        if suffix in ['.h5', '.hdf5']:
            format = "hdf5"
        elif suffix == '.npz':
            format = "npz"
        elif suffix in ['.pt', '.pth']:
            format = "pt"
        else:
            raise ValueError(f"无法推断文件格式: {suffix}")
    
    if format == "hdf5":
        import h5py
        with h5py.File(filepath, 'r') as f:
            designs = f['designs'][:]
            performances = f['performances'][:]
            uncertainties = f['uncertainties'][:] if 'uncertainties' in f else None
            metadata = dict(f.attrs)
        
        return GenerationResult(
            designs=designs,
            performances=performances,
            uncertainties=uncertainties,
            metadata=metadata
        )
    
    elif format == "npz":
        data = np.load(filepath)
        return GenerationResult(
            designs=data['designs'],
            performances=data['performances'],
            uncertainties=data.get('uncertainties'),
            metadata={}
        )
    
    elif format == "pt":
        data = torch.load(filepath, weights_only=False)
        return GenerationResult(
            designs=data['designs'].numpy(),
            performances=data['performances'].numpy(),
            uncertainties=data['uncertainties'].numpy() if data.get('uncertainties') is not None else None,
            metadata=data.get('metadata', {})
        )
    
    else:
        raise ValueError(f"不支持的格式: {format}")

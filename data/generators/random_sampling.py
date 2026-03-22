"""
随机采样数据生成器

提供多种随机采样策略用于生成初始训练数据。
"""

from typing import Optional, Tuple, List, Callable, Union, Literal
from dataclasses import dataclass
import numpy as np
import torch

from data.generators.base import (
    DataGenerator,
    GeneratorConfig,
    GenerationResult,
    SimulatorBasedGenerator
)


@dataclass
class RandomSamplingConfig(GeneratorConfig):
    """随机采样配置"""
    name: str = "random_sampling"
    
    # 设计空间参数
    design_shape: Tuple[int, ...] = (200, 22)
    design_bounds: Tuple[float, float] = (0.0, 1.0)
    performance_dim: int = 3
    
    # 采样策略
    sampling_method: Literal["uniform", "latin_hypercube", "sobol", "halton"] = "uniform"
    
    # 约束
    constraint_func: Optional[Callable[[np.ndarray], bool]] = None
    max_rejection_attempts: int = 10000
    
    # 连续性
    continuous: bool = True
    discretization_levels: int = 100


class RandomSamplingGenerator(SimulatorBasedGenerator):
    """
    随机采样数据生成器
    
    支持多种采样策略：
    - 均匀随机采样
    - 拉丁超立方采样 (LHS)
    - Sobol 序列
    - Halton 序列
    """
    
    def __init__(
        self,
        config: Optional[RandomSamplingConfig] = None,
        simulator: Optional[Callable] = None
    ):
        super().__init__(config or RandomSamplingConfig(), simulator)
    
    @property
    def sampling_config(self) -> RandomSamplingConfig:
        return self.config
    
    def generate(
        self,
        num_samples: int,
        **kwargs
    ) -> GenerationResult:
        """
        生成随机设计样本
        
        Args:
            num_samples: 样本数量
            
        Returns:
            生成结果
        """
        method = self.sampling_config.sampling_method
        
        # 生成设计参数
        if method == "uniform":
            designs = self._uniform_sampling(num_samples)
        elif method == "latin_hypercube":
            designs = self._latin_hypercube_sampling(num_samples)
        elif method == "sobol":
            designs = self._sobol_sampling(num_samples)
        elif method == "halton":
            designs = self._halton_sampling(num_samples)
        else:
            raise ValueError(f"未知的采样方法: {method}")
        
        # 应用约束
        if self.sampling_config.constraint_func is not None:
            designs = self._apply_constraints(designs)
        
        # 评估性能
        if self.simulator is not None:
            performances = self.evaluate_designs(designs)
        else:
            performances = self._dummy_performance(designs)
        
        return GenerationResult(
            designs=designs,
            performances=performances,
            metadata={
                'method': method,
                'design_shape': self.sampling_config.design_shape,
                'bounds': self.sampling_config.design_bounds,
                'seed': self.config.seed
            }
        )
    
    def _uniform_sampling(self, num_samples: int) -> np.ndarray:
        """均匀随机采样"""
        low, high = self.sampling_config.design_bounds
        shape = (num_samples,) + self.sampling_config.design_shape
        
        if self.sampling_config.continuous:
            return self.rng.uniform(low, high, shape)
        else:
            levels = self.sampling_config.discretization_levels
            return self.rng.integers(0, levels, shape) / (levels - 1) * (high - low) + low
    
    def _latin_hypercube_sampling(self, num_samples: int) -> np.ndarray:
        """拉丁超立方采样"""
        design_dim = np.prod(self.sampling_config.design_shape)
        
        # 生成 LHS 样本
        samples = np.zeros((num_samples, design_dim))
        
        for i in range(design_dim):
            # 随机排列
            perm = self.rng.permutation(num_samples)
            # 在每个区间内随机采样
            offsets = self.rng.uniform(0, 1, num_samples)
            samples[:, i] = (perm + offsets) / num_samples
        
        # 打乱样本顺序
        self.rng.shuffle(samples)
        
        # 重塑为目标形状
        samples = samples.reshape((num_samples,) + self.sampling_config.design_shape)
        
        # 映射到目标范围
        low, high = self.sampling_config.design_bounds
        samples = samples * (high - low) + low
        
        return samples
    
    def _sobol_sampling(self, num_samples: int) -> np.ndarray:
        """Sobol 序列采样"""
        try:
            from scipy.stats import qmc
        except ImportError:
            # 回退到均匀采样
            return self._uniform_sampling(num_samples)
        
        design_dim = np.prod(self.sampling_config.design_shape)
        
        # 创建 Sobol 生成器
        sampler = qmc.Sobol(d=design_dim, scramble=True, seed=self.config.seed)
        
        # 生成样本
        samples = sampler.random(num_samples)
        
        # 重塑为目标形状
        samples = samples.reshape((num_samples,) + self.sampling_config.design_shape)
        
        # 映射到目标范围
        low, high = self.sampling_config.design_bounds
        samples = samples * (high - low) + low
        
        return samples
    
    def _halton_sampling(self, num_samples: int) -> np.ndarray:
        """Halton 序列采样"""
        try:
            from scipy.stats import qmc
        except ImportError:
            # 回退到均匀采样
            return self._uniform_sampling(num_samples)
        
        design_dim = np.prod(self.sampling_config.design_shape)
        
        # 创建 Halton 生成器
        sampler = qmc.Halton(d=design_dim, scramble=True, seed=self.config.seed)
        
        # 生成样本
        samples = sampler.random(num_samples)
        
        # 重塑为目标形状
        samples = samples.reshape((num_samples,) + self.sampling_config.design_shape)
        
        # 映射到目标范围
        low, high = self.sampling_config.design_bounds
        samples = samples * (high - low) + low
        
        return samples
    
    def _apply_constraints(self, designs: np.ndarray) -> np.ndarray:
        """应用约束条件"""
        constraint_func = self.sampling_config.constraint_func
        max_attempts = self.sampling_config.max_rejection_attempts
        
        valid_designs = []
        attempts = 0
        
        for design in designs:
            if constraint_func(design):
                valid_designs.append(design)
            attempts += 1
            
            if attempts >= max_attempts:
                break
        
        # 如果有效样本不足，补充随机采样
        while len(valid_designs) < len(designs) and attempts < max_attempts:
            new_design = self._uniform_sampling(1)[0]
            if constraint_func(new_design):
                valid_designs.append(new_design)
            attempts += 1
        
        if len(valid_designs) < len(designs):
            raise RuntimeError(
                f"无法在 {max_attempts} 次尝试内生成足够的有效样本。"
                f"需要 {len(designs)}，得到 {len(valid_designs)}"
            )
        
        return np.array(valid_designs[:len(designs)])
    
    def _dummy_performance(self, designs: np.ndarray) -> np.ndarray:
        """生成模拟性能指标（无仿真器时使用）"""
        num_samples = len(designs)
        perf_dim = self.sampling_config.performance_dim
        
        performances = np.zeros((num_samples, perf_dim))
        
        for i, design in enumerate(designs):
            # 模拟性能指标
            # 1. 平均值 -> 效率
            performances[i, 0] = design.mean()
            
            # 2. 均匀性 -> 带宽
            performances[i, 1] = 1 - design.std()
            
            # 3. 复杂度 -> 损耗
            if design.ndim >= 2:
                complexity = np.abs(np.diff(design, axis=0)).mean()
            else:
                complexity = np.abs(np.diff(design)).mean()
            performances[i, 2] = complexity * 0.5
        
        # 添加噪声
        performances += self.rng.normal(0, 0.05, performances.shape)
        
        return performances
    
    def generate_with_diversity(
        self,
        num_samples: int,
        min_distance: float = 0.1,
        max_iterations: int = 10
    ) -> GenerationResult:
        """
        生成具有多样性的样本
        
        使用最大化最小距离策略生成分散的样本。
        
        Args:
            num_samples: 样本数量
            min_distance: 样本间的最小距离
            max_iterations: 最大迭代次数
            
        Returns:
            生成结果
        """
        # 初始生成更多样本
        candidates = self.generate(num_samples * 3)
        candidate_designs = candidates.designs
        
        selected_indices = [0]
        
        for _ in range(num_samples - 1):
            best_idx = None
            best_min_dist = -1
            
            for i in range(len(candidate_designs)):
                if i in selected_indices:
                    continue
                
                # 计算到已选样本的最小距离
                min_dist = float('inf')
                for j in selected_indices:
                    dist = np.linalg.norm(
                        candidate_designs[i].flatten() - candidate_designs[j].flatten()
                    )
                    min_dist = min(min_dist, dist)
                
                if min_dist > best_min_dist:
                    best_min_dist = min_dist
                    best_idx = i
            
            if best_idx is not None and best_min_dist >= min_distance:
                selected_indices.append(best_idx)
            
            if len(selected_indices) >= num_samples:
                break
        
        # 如果不足，补充随机样本
        while len(selected_indices) < num_samples:
            idx = self.rng.integers(0, len(candidate_designs))
            if idx not in selected_indices:
                selected_indices.append(idx)
        
        return GenerationResult(
            designs=candidate_designs[selected_indices],
            performances=candidates.performances[selected_indices],
            metadata={
                **candidates.metadata,
                'diversity_filter': True,
                'min_distance': min_distance
            }
        )


def create_random_generator(
    design_shape: Tuple[int, ...] = (200, 22),
    method: Literal["uniform", "latin_hypercube", "sobol", "halton"] = "uniform",
    bounds: Tuple[float, float] = (0.0, 1.0),
    seed: int = 42,
    simulator: Optional[Callable] = None
) -> RandomSamplingGenerator:
    """
    创建随机采样生成器的便捷函数
    
    Args:
        design_shape: 设计形状
        method: 采样方法
        bounds: 设计参数范围
        seed: 随机种子
        simulator: 仿真器函数
        
    Returns:
        配置好的生成器
    """
    config = RandomSamplingConfig(
        design_shape=design_shape,
        sampling_method=method,
        design_bounds=bounds,
        seed=seed
    )
    return RandomSamplingGenerator(config, simulator)

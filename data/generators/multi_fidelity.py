"""
多保真度数据生成器

支持不同精度级别的仿真数据生成，
实现低成本高保真数据的策略性获取。
"""

from typing import Optional, Tuple, List, Callable, Union, Literal, Dict, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np
import torch

from data.generators.base import (
    DataGenerator,
    GeneratorConfig,
    GenerationResult,
    SimulatorBasedGenerator
)


class FidelityLevel(Enum):
    """保真度级别"""
    LOW = 0        # 低保真度（快速近似）
    MEDIUM = 1     # 中保真度
    HIGH = 2       # 高保真度（精确仿真）
    EXPERIMENT = 3  # 实验数据


@dataclass
class FidelityInfo:
    """保真度信息"""
    level: FidelityLevel
    name: str
    cost: float              # 相对计算成本
    accuracy: float          # 相对精度 (0-1)
    simulator: Optional[Callable] = None
    bias_model: Optional[Callable] = None  # 偏差模型 (low -> high)


@dataclass
class MultiFidelityConfig(GeneratorConfig):
    """多保真度生成配置"""
    name: str = "multi_fidelity"
    
    # 设计空间参数
    design_shape: Tuple[int, ...] = (200, 22)
    design_bounds: Tuple[float, float] = (0.0, 1.0)
    performance_dim: int = 3
    
    # 保真度配置
    fidelity_levels: List[FidelityLevel] = None  # type: ignore
    fidelity_costs: Dict[int, float] = None  # type: ignore
    
    # 分配策略
    allocation_strategy: Literal["fixed", "adaptive", "cascaded"] = "cascaded"
    high_fidelity_ratio: float = 0.2    # 高保真数据比例
    cascade_threshold: float = 0.1      # 级联选择阈值
    
    # 预算约束
    total_budget: Optional[float] = None  # 总计算预算
    max_high_fidelity_samples: int = 1000
    
    # 偏差校正
    use_bias_correction: bool = True
    num_calibration_samples: int = 50
    
    def __post_init__(self):
        if self.fidelity_levels is None:
            self.fidelity_levels = [FidelityLevel.LOW, FidelityLevel.HIGH]
        if self.fidelity_costs is None:
            self.fidelity_costs = {
                FidelityLevel.LOW.value: 1.0,
                FidelityLevel.MEDIUM.value: 10.0,
                FidelityLevel.HIGH.value: 100.0,
                FidelityLevel.EXPERIMENT.value: 1000.0
            }


@dataclass
class MultiFidelityResult(GenerationResult):
    """多保真度生成结果"""
    fidelity_labels: np.ndarray = None  # type: ignore  # 保真度标签 [N]
    corrected_performances: Optional[np.ndarray] = None  # 校正后的性能
    
    def __post_init__(self):
        if self.fidelity_labels is None:
            self.fidelity_labels = np.zeros(len(self.designs), dtype=int)


class MultiFidelityGenerator(SimulatorBasedGenerator):
    """
    多保真度数据生成器
    
    支持在多个精度级别上生成数据，并使用偏差校正
    将低保真数据映射到高保真空间。
    """
    
    def __init__(
        self,
        config: Optional[MultiFidelityConfig] = None,
        simulators: Optional[Dict[int, Callable]] = None
    ):
        super().__init__(config or MultiFidelityConfig())
        self._simulators: Dict[int, Callable] = simulators or {}
        self._bias_models: Dict[int, Callable] = {}
        self._calibration_data: Dict[int, GenerationResult] = {}
    
    @property
    def mf_config(self) -> MultiFidelityConfig:
        return self.config
    
    def set_simulator(
        self,
        fidelity_level: FidelityLevel,
        simulator: Callable
    ) -> None:
        """设置特定保真度的仿真器"""
        self._simulators[fidelity_level.value] = simulator
    
    def set_bias_model(
        self,
        from_level: FidelityLevel,
        to_level: FidelityLevel,
        model: Callable
    ) -> None:
        """设置偏差校正模型"""
        key = (from_level.value, to_level.value)
        self._bias_models[key] = model
    
    def generate(
        self,
        num_samples: int,
        fidelity_level: Optional[FidelityLevel] = None,
        **kwargs
    ) -> MultiFidelityResult:
        """
        生成数据
        
        Args:
            num_samples: 样本数量
            fidelity_level: 指定保真度级别（None 则自动分配）
            
        Returns:
            多保真度生成结果
        """
        if fidelity_level is not None:
            return self._generate_at_fidelity(num_samples, fidelity_level)
        
        strategy = self.mf_config.allocation_strategy
        
        if strategy == "fixed":
            return self._fixed_allocation(num_samples)
        elif strategy == "adaptive":
            return self._adaptive_allocation(num_samples)
        elif strategy == "cascaded":
            return self._cascaded_allocation(num_samples)
        else:
            raise ValueError(f"未知的分配策略: {strategy}")
    
    def _generate_at_fidelity(
        self,
        num_samples: int,
        fidelity_level: FidelityLevel
    ) -> MultiFidelityResult:
        """在指定保真度生成数据"""
        # 生成设计
        designs = self._generate_designs(num_samples)
        
        # 获取仿真器
        simulator = self._simulators.get(fidelity_level.value)
        
        if simulator is not None:
            performances = self._evaluate_with_simulator(designs, simulator)
        else:
            performances = self._dummy_performance(
                designs,
                fidelity_level=fidelity_level
            )
        
        return MultiFidelityResult(
            designs=designs,
            performances=performances,
            fidelity_labels=np.full(num_samples, fidelity_level.value, dtype=int),
            metadata={
                'fidelity_level': fidelity_level.name,
                'num_samples': num_samples
            }
        )
    
    def _generate_designs(self, num_samples: int) -> np.ndarray:
        """生成设计参数"""
        low, high = self.mf_config.design_bounds
        shape = (num_samples,) + self.mf_config.design_shape
        return self.rng.uniform(low, high, shape)
    
    def _evaluate_with_simulator(
        self,
        designs: np.ndarray,
        simulator: Callable
    ) -> np.ndarray:
        """使用仿真器评估设计"""
        performances = []
        for design in designs:
            perf = simulator(design)
            performances.append(perf)
        return np.array(performances)
    
    def _fixed_allocation(self, num_samples: int) -> MultiFidelityResult:
        """固定比例分配"""
        high_ratio = self.mf_config.high_fidelity_ratio
        
        num_high = int(num_samples * high_ratio)
        num_low = num_samples - num_high
        
        # 生成低保真数据
        low_result = self._generate_at_fidelity(num_low, FidelityLevel.LOW)
        
        # 生成高保真数据
        high_result = self._generate_at_fidelity(num_high, FidelityLevel.HIGH)
        
        # 合并结果
        return self._merge_results([low_result, high_result])
    
    def _adaptive_allocation(
        self,
        num_samples: int,
        uncertainty_threshold: float = 0.1
    ) -> MultiFidelityResult:
        """自适应分配"""
        # 首先生成所有低保真数据
        designs = self._generate_designs(num_samples)
        
        # 评估低保真
        low_simulator = self._simulators.get(FidelityLevel.LOW.value)
        if low_simulator is not None:
            low_performances = self._evaluate_with_simulator(designs, low_simulator)
        else:
            low_performances = self._dummy_performance(designs, FidelityLevel.LOW)
        
        # 估计不确定性（简化版）
        uncertainties = self._estimate_uncertainties(designs, low_performances)
        
        # 选择高不确定性样本进行高保真仿真
        high_fidelity_mask = uncertainties > uncertainty_threshold
        
        performances = low_performances.copy()
        fidelity_labels = np.full(num_samples, FidelityLevel.LOW.value, dtype=int)
        
        if high_fidelity_mask.any():
            high_designs = designs[high_fidelity_mask]
            high_simulator = self._simulators.get(FidelityLevel.HIGH.value)
            
            if high_simulator is not None:
                high_performances = self._evaluate_with_simulator(high_designs, high_simulator)
            else:
                high_performances = self._dummy_performance(high_designs, FidelityLevel.HIGH)
            
            performances[high_fidelity_mask] = high_performances
            fidelity_labels[high_fidelity_mask] = FidelityLevel.HIGH.value
        
        return MultiFidelityResult(
            designs=designs,
            performances=performances,
            fidelity_labels=fidelity_labels,
            metadata={
                'strategy': 'adaptive',
                'high_fidelity_count': high_fidelity_mask.sum()
            }
        )
    
    def _cascaded_allocation(self, num_samples: int) -> MultiFidelityResult:
        """级联分配"""
        threshold = self.mf_config.cascade_threshold
        
        # 初始候选池
        candidate_pool_size = min(num_samples * 5, 10000)
        candidates = self._generate_designs(candidate_pool_size)
        
        # 第一级：低保真评估
        low_simulator = self._simulators.get(FidelityLevel.LOW.value)
        if low_simulator is not None:
            low_performances = self._evaluate_with_simulator(candidates, low_simulator)
        else:
            low_performances = self._dummy_performance(candidates, FidelityLevel.LOW)
        
        # 筛选有潜力的候选
        potential_scores = self._compute_potential_scores(low_performances)
        potential_mask = potential_scores > np.percentile(potential_scores, 50)
        
        # 第二级：中保真评估
        medium_candidates = candidates[potential_mask]
        medium_performances_low = low_performances[potential_mask]
        
        medium_simulator = self._simulators.get(FidelityLevel.MEDIUM.value)
        if medium_simulator is not None:
            medium_performances = self._evaluate_with_simulator(medium_candidates, medium_simulator)
        else:
            medium_performances = self._dummy_performance(medium_candidates, FidelityLevel.MEDIUM)
        
        # 最终筛选
        final_scores = self._compute_potential_scores(medium_performances)
        top_indices = np.argsort(final_scores)[-num_samples:]
        
        # 第三级：高保真评估
        final_designs = medium_candidates[top_indices]
        
        high_simulator = self._simulators.get(FidelityLevel.HIGH.value)
        if high_simulator is not None:
            final_performances = self._evaluate_with_simulator(final_designs, high_simulator)
        else:
            final_performances = self._dummy_performance(final_designs, FidelityLevel.HIGH)
        
        return MultiFidelityResult(
            designs=final_designs,
            performances=final_performances,
            fidelity_labels=np.full(len(final_designs), FidelityLevel.HIGH.value, dtype=int),
            metadata={
                'strategy': 'cascaded',
                'initial_pool_size': candidate_pool_size,
                'medium_pool_size': len(medium_candidates)
            }
        )
    
    def _estimate_uncertainties(
        self,
        designs: np.ndarray,
        performances: np.ndarray
    ) -> np.ndarray:
        """估计不确定性"""
        # 简化实现：基于性能方差
        # 实际应用中可以使用代理模型的预测不确定性
        perf_std = performances.std(axis=1)
        return perf_std / (perf_std.max() + 1e-8)
    
    def _compute_potential_scores(self, performances: np.ndarray) -> np.ndarray:
        """计算潜力分数"""
        # 假设性能越高越好
        # 对于多目标，使用加权和
        if performances.ndim == 1:
            return performances
        
        # 简单加权：第一个指标权重最高
        weights = np.array([0.5, 0.3, 0.2])[:performances.shape[1]]
        weights = weights / weights.sum()
        
        return performances @ weights
    
    def _merge_results(
        self,
        results: List[MultiFidelityResult]
    ) -> MultiFidelityResult:
        """合并多个结果"""
        designs = np.concatenate([r.designs for r in results])
        performances = np.concatenate([r.performances for r in results])
        fidelity_labels = np.concatenate([r.fidelity_labels for r in results])
        
        metadata = {}
        for i, r in enumerate(results):
            metadata.update(r.metadata)
        
        return MultiFidelityResult(
            designs=designs,
            performances=performances,
            fidelity_labels=fidelity_labels,
            metadata=metadata
        )
    
    def _dummy_performance(
        self,
        designs: np.ndarray,
        fidelity_level: FidelityLevel = FidelityLevel.LOW
    ) -> np.ndarray:
        """生成模拟性能指标"""
        num_samples = len(designs)
        perf_dim = self.mf_config.performance_dim
        
        performances = np.zeros((num_samples, perf_dim))
        
        for i, design in enumerate(designs):
            performances[i, 0] = design.mean()
            performances[i, 1] = 1 - design.std()
            if design.ndim >= 2:
                performances[i, 2] = np.abs(np.diff(design, axis=0)).mean() * 0.5
            else:
                performances[i, 2] = np.abs(np.diff(design)).mean() * 0.5
        
        # 根据保真度添加噪声
        noise_levels = {
            FidelityLevel.LOW: 0.1,
            FidelityLevel.MEDIUM: 0.05,
            FidelityLevel.HIGH: 0.01,
            FidelityLevel.EXPERIMENT: 0.005
        }
        noise = noise_levels.get(fidelity_level, 0.1)
        performances += self.rng.normal(0, noise, performances.shape)
        
        return performances
    
    def calibrate_bias(
        self,
        num_samples: int = 50
    ) -> Dict[int, np.ndarray]:
        """
        校准偏差
        
        使用配对样本估计低保真到高保真的偏差。
        
        Args:
            num_samples: 校准样本数
            
        Returns:
            偏差统计信息
        """
        designs = self._generate_designs(num_samples)
        calibration_data = {}
        
        for level in [FidelityLevel.LOW, FidelityLevel.HIGH]:
            result = self._generate_at_fidelity(num_samples, level)
            self._calibration_data[level.value] = result
        
        # 计算偏差
        low_perf = self._calibration_data[FidelityLevel.LOW.value].performances
        high_perf = self._calibration_data[FidelityLevel.HIGH.value].performances
        
        bias = high_perf - low_perf
        calibration_data['bias_mean'] = bias.mean(axis=0)
        calibration_data['bias_std'] = bias.std(axis=0)
        
        return calibration_data
    
    def apply_bias_correction(
        self,
        result: MultiFidelityResult
    ) -> MultiFidelityResult:
        """
        应用偏差校正
        
        将低保真数据校正到高保真空间。
        """
        if not self.mf_config.use_bias_correction:
            return result
        
        corrected_performances = result.performances.copy()
        
        # 检查是否有校准数据
        if not self._calibration_data:
            self.calibrate_bias(self.mf_config.num_calibration_samples)
        
        if 'bias_mean' in self._calibration_data or hasattr(self, '_bias_stats'):
            # 获取偏差统计
            if 'bias_mean' in self._calibration_data:
                bias_mean = self._calibration_data['bias_mean']
            else:
                bias_mean = np.zeros(self.mf_config.performance_dim)
            
            # 校正低保真数据
            low_mask = result.fidelity_labels == FidelityLevel.LOW.value
            corrected_performances[low_mask] += bias_mean
        
        return MultiFidelityResult(
            designs=result.designs,
            performances=result.performances,
            fidelity_labels=result.fidelity_labels,
            corrected_performances=corrected_performances,
            metadata={
                **result.metadata,
                'bias_corrected': True
            }
        )
    
    def get_budget_allocation(
        self,
        total_budget: float
    ) -> Dict[FidelityLevel, int]:
        """
        根据预算计算各保真度的样本数量
        
        Args:
            total_budget: 总计算预算
            
        Returns:
            各保真度的样本数量
        """
        allocation = {}
        remaining_budget = total_budget
        
        costs = self.mf_config.fidelity_costs
        
        # 从低到高分配
        for level in [FidelityLevel.LOW, FidelityLevel.MEDIUM, FidelityLevel.HIGH]:
            cost = costs.get(level.value, 1.0)
            
            if level == FidelityLevel.LOW:
                # 低保真：尽可能多
                num = int(remaining_budget * 0.5 / cost)
            elif level == FidelityLevel.MEDIUM:
                # 中保真：适中
                num = int(remaining_budget * 0.3 / cost)
            else:
                # 高保真：根据剩余预算
                num = int(remaining_budget / cost)
            
            allocation[level] = num
            remaining_budget -= num * cost
        
        return allocation


def create_multi_fidelity_generator(
    design_shape: Tuple[int, ...] = (200, 22),
    fidelity_levels: Optional[List[FidelityLevel]] = None,
    allocation_strategy: Literal["fixed", "adaptive", "cascaded"] = "cascaded",
    seed: int = 42
) -> MultiFidelityGenerator:
    """
    创建多保真度生成器的便捷函数
    
    Args:
        design_shape: 设计形状
        fidelity_levels: 保真度级别列表
        allocation_strategy: 分配策略
        seed: 随机种子
        
    Returns:
        配置好的生成器
    """
    config = MultiFidelityConfig(
        design_shape=design_shape,
        fidelity_levels=fidelity_levels or [FidelityLevel.LOW, FidelityLevel.HIGH],
        allocation_strategy=allocation_strategy,
        seed=seed
    )
    return MultiFidelityGenerator(config)

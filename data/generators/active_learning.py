"""
主动学习数据生成器

基于代理模型不确定性选择最有价值的样本进行标注。
"""

from typing import Optional, Tuple, List, Callable, Union, Literal
from dataclasses import dataclass
from enum import Enum
import numpy as np
import torch
import torch.nn as nn

from data.generators.base import (
    DataGenerator,
    GeneratorConfig,
    GenerationResult,
    SimulatorBasedGenerator
)


class AcquisitionFunction(Enum):
    """采集函数类型"""
    UNCERTAINTY = "uncertainty"           # 不确定性采样
    EXPECTED_IMPROVEMENT = "ei"           # 期望改进
    UPPER_CONFIDENCE = "ucb"              # 上置信界
    THOMPSON_SAMPLING = "thompson"        # 汤普森采样
    DIVERSITY = "diversity"               # 多样性采样
    BALANCED = "balanced"                 # 平衡采样


@dataclass
class ActiveLearningConfig(GeneratorConfig):
    """主动学习配置"""
    name: str = "active_learning"
    
    # 设计空间参数
    design_shape: Tuple[int, ...] = (200, 22)
    design_bounds: Tuple[float, float] = (0.0, 1.0)
    performance_dim: int = 3
    
    # 采集函数配置
    acquisition_function: AcquisitionFunction = AcquisitionFunction.UNCERTAINTY
    exploration_weight: float = 2.0       # UCB 探索权重
    target_performance: Optional[np.ndarray] = None  # 目标性能（用于 EI）
    
    # 候选池配置
    candidate_pool_size: int = 10000
    num_candidates_to_select: int = 100
    
    # 代理模型配置
    surrogate_model: Optional[nn.Module] = None
    ensemble_size: int = 5                # 集成模型数量
    uncertainty_threshold: float = 0.1    # 不确定性阈值
    
    # 迭代配置
    max_iterations: int = 100
    convergence_threshold: float = 0.001


class UncertaintyEstimator:
    """
    不确定性估计器
    
    支持多种不确定性估计方法：
    - MC Dropout
    - Deep Ensemble
    - Bootstrapping
    """
    
    def __init__(
        self,
        model: nn.Module,
        method: Literal["mc_dropout", "ensemble", "bootstrap"] = "mc_dropout",
        num_samples: int = 20
    ):
        self.model = model
        self.method = method
        self.num_samples = num_samples
    
    def estimate(
        self,
        designs: np.ndarray,
        device: Optional[torch.device] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        估计预测均值和不确定性
        
        Args:
            designs: 设计参数数组
            device: 计算设备
            
        Returns:
            (mean_predictions, uncertainties)
        """
        if device is None:
            device = next(self.model.parameters()).device
        
        x = torch.from_numpy(designs).float().to(device)
        
        if self.method == "mc_dropout":
            return self._mc_dropout_estimate(x)
        elif self.method == "ensemble":
            return self._ensemble_estimate(x)
        else:
            return self._bootstrap_estimate(x)
    
    def _mc_dropout_estimate(
        self,
        x: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray]:
        """MC Dropout 不确定性估计"""
        self.model.train()  # 启用 dropout
        
        predictions = []
        with torch.no_grad():
            for _ in range(self.num_samples):
                pred = self.model(x)
                predictions.append(pred.cpu().numpy())
        
        predictions = np.array(predictions)
        mean = predictions.mean(axis=0)
        std = predictions.std(axis=0)
        
        return mean, std
    
    def _ensemble_estimate(
        self,
        x: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray]:
        """集成模型不确定性估计"""
        # 假设模型有 ensemble 属性
        if hasattr(self.model, 'ensemble'):
            predictions = []
            with torch.no_grad():
                for member in self.model.ensemble:
                    pred = member(x)
                    predictions.append(pred.cpu().numpy())
            
            predictions = np.array(predictions)
            mean = predictions.mean(axis=0)
            std = predictions.std(axis=0)
            
            return mean, std
        else:
            # 回退到 MC Dropout
            return self._mc_dropout_estimate(x)
    
    def _bootstrap_estimate(
        self,
        x: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Bootstrap 不确定性估计"""
        # 简化实现，使用 MC Dropout
        return self._mc_dropout_estimate(x)


class ActiveLearningGenerator(SimulatorBasedGenerator):
    """
    主动学习数据生成器
    
    使用代理模型的不确定性指导样本选择，
    优先选择模型预测最不确定的样本进行仿真。
    """
    
    def __init__(
        self,
        config: Optional[ActiveLearningConfig] = None,
        simulator: Optional[Callable] = None
    ):
        super().__init__(config or ActiveLearningConfig(), simulator)
        self._labeled_data: Optional[GenerationResult] = None
        self._uncertainty_estimator: Optional[UncertaintyEstimator] = None
        self._iteration_count = 0
    
    @property
    def al_config(self) -> ActiveLearningConfig:
        return self.config
    
    def set_surrogate_model(self, model: nn.Module) -> None:
        """设置代理模型"""
        self.al_config.surrogate_model = model
        self._uncertainty_estimator = UncertaintyEstimator(model)
    
    def set_labeled_data(self, data: GenerationResult) -> None:
        """设置已标注数据"""
        self._labeled_data = data
    
    def generate(
        self,
        num_samples: int,
        **kwargs
    ) -> GenerationResult:
        """
        使用主动学习策略生成样本
        
        Args:
            num_samples: 样本数量
            
        Returns:
            生成结果
        """
        # 生成候选池
        candidates = self._generate_candidate_pool()
        
        # 选择最有价值的样本
        selected_indices = self._select_samples(candidates, num_samples)
        
        # 提取选中的设计
        selected_designs = candidates[selected_indices]
        
        # 评估性能
        if self.simulator is not None:
            performances = self.evaluate_designs(selected_designs)
        else:
            performances = self._dummy_performance(selected_designs)
        
        # 更新已标注数据
        new_data = GenerationResult(
            designs=selected_designs,
            performances=performances,
            metadata={
                'iteration': self._iteration_count,
                'acquisition_function': self.al_config.acquisition_function.value,
                'selected_indices': selected_indices.tolist()
            }
        )
        
        if self._labeled_data is not None:
            self._labeled_data = GenerationResult(
                designs=np.concatenate([self._labeled_data.designs, new_data.designs]),
                performances=np.concatenate([self._labeled_data.performances, new_data.performances]),
                metadata=self._labeled_data.metadata
            )
        else:
            self._labeled_data = new_data
        
        self._iteration_count += 1
        
        return new_data
    
    def _generate_candidate_pool(self) -> np.ndarray:
        """生成候选池"""
        pool_size = self.al_config.candidate_pool_size
        low, high = self.al_config.design_bounds
        shape = (pool_size,) + self.al_config.design_shape
        
        return self.rng.uniform(low, high, shape)
    
    def _select_samples(
        self,
        candidates: np.ndarray,
        num_samples: int
    ) -> np.ndarray:
        """
        选择样本
        
        Args:
            candidates: 候选设计池
            num_samples: 需要选择的数量
            
        Returns:
            选中的索引数组
        """
        acq_func = self.al_config.acquisition_function
        
        if acq_func == AcquisitionFunction.UNCERTAINTY:
            scores = self._uncertainty_scores(candidates)
        elif acq_func == AcquisitionFunction.EXPECTED_IMPROVEMENT:
            scores = self._expected_improvement_scores(candidates)
        elif acq_func == AcquisitionFunction.UPPER_CONFIDENCE:
            scores = self._ucb_scores(candidates)
        elif acq_func == AcquisitionFunction.THOMPSON_SAMPLING:
            scores = self._thompson_sampling_scores(candidates)
        elif acq_func == AcquisitionFunction.DIVERSITY:
            scores = self._diversity_scores(candidates)
        elif acq_func == AcquisitionFunction.BALANCED:
            scores = self._balanced_scores(candidates)
        else:
            raise ValueError(f"未知的采集函数: {acq_func}")
        
        # 选择得分最高的样本
        selected_indices = np.argsort(scores)[-num_samples:]
        
        return selected_indices
    
    def _uncertainty_scores(self, candidates: np.ndarray) -> np.ndarray:
        """计算不确定性分数"""
        if self._uncertainty_estimator is None:
            # 没有代理模型，使用随机分数
            return self.rng.random(len(candidates))
        
        _, uncertainties = self._uncertainty_estimator.estimate(
            candidates,
            device=self.device
        )
        
        # 使用总不确定性作为分数
        return uncertainties.sum(axis=-1)
    
    def _expected_improvement_scores(self, candidates: np.ndarray) -> np.ndarray:
        """计算期望改进分数"""
        if self._uncertainty_estimator is None or self._labeled_data is None:
            return self.rng.random(len(candidates))
        
        target = self.al_config.target_performance
        if target is None:
            # 使用当前最佳性能作为目标
            target = self._labeled_data.performances.max(axis=0)
        
        means, stds = self._uncertainty_estimator.estimate(
            candidates,
            device=self.device
        )
        
        # 计算 EI
        target = np.array(target)
        z = (target - means) / (stds + 1e-8)
        
        # EI = (target - mean) * Phi(z) + std * phi(z)
        from scipy.stats import norm
        ei = (target - means) * norm.cdf(z) + stds * norm.pdf(z)
        
        return ei.sum(axis=-1)
    
    def _ucb_scores(self, candidates: np.ndarray) -> np.ndarray:
        """计算上置信界分数"""
        if self._uncertainty_estimator is None:
            return self.rng.random(len(candidates))
        
        means, stds = self._uncertainty_estimator.estimate(
            candidates,
            device=self.device
        )
        
        beta = self.al_config.exploration_weight
        ucb = means + beta * stds
        
        return ucb.sum(axis=-1)
    
    def _thompson_sampling_scores(self, candidates: np.ndarray) -> np.ndarray:
        """Thompson 采样分数"""
        if self._uncertainty_estimator is None:
            return self.rng.random(len(candidates))
        
        means, stds = self._uncertainty_estimator.estimate(
            candidates,
            device=self.device
        )
        
        # 从后验分布采样
        samples = self.rng.normal(means, stds)
        
        return samples.sum(axis=-1)
    
    def _diversity_scores(self, candidates: np.ndarray) -> np.ndarray:
        """计算多样性分数"""
        if self._labeled_data is None:
            return self.rng.random(len(candidates))
        
        # 计算到已标注数据的最小距离
        labeled = self._labeled_data.designs
        
        scores = np.zeros(len(candidates))
        for i, candidate in enumerate(candidates):
            distances = np.linalg.norm(
                labeled.reshape(len(labeled), -1) - candidate.flatten(),
                axis=1
            )
            scores[i] = distances.min()
        
        return scores
    
    def _balanced_scores(self, candidates: np.ndarray) -> np.ndarray:
        """计算平衡分数（不确定性 + 多样性）"""
        uncertainty_scores = self._uncertainty_scores(candidates)
        diversity_scores = self._diversity_scores(candidates)
        
        # 归一化
        uncertainty_scores = (uncertainty_scores - uncertainty_scores.min()) / (
            uncertainty_scores.max() - uncertainty_scores.min() + 1e-8
        )
        diversity_scores = (diversity_scores - diversity_scores.min()) / (
            diversity_scores.max() - diversity_scores.min() + 1e-8
        )
        
        # 平衡组合
        return 0.5 * uncertainty_scores + 0.5 * diversity_scores
    
    def _dummy_performance(self, designs: np.ndarray) -> np.ndarray:
        """生成模拟性能指标"""
        num_samples = len(designs)
        perf_dim = self.al_config.performance_dim
        
        performances = np.zeros((num_samples, perf_dim))
        
        for i, design in enumerate(designs):
            performances[i, 0] = design.mean()
            performances[i, 1] = 1 - design.std()
            if design.ndim >= 2:
                performances[i, 2] = np.abs(np.diff(design, axis=0)).mean() * 0.5
            else:
                performances[i, 2] = np.abs(np.diff(design)).mean() * 0.5
        
        performances += self.rng.normal(0, 0.05, performances.shape)
        
        return performances
    
    def run_active_learning_loop(
        self,
        initial_data: GenerationResult,
        num_iterations: int = 10,
        samples_per_iteration: int = 50,
        model_update_callback: Optional[Callable[[GenerationResult], nn.Module]] = None,
        early_stopping_threshold: Optional[float] = None
    ) -> List[GenerationResult]:
        """
        运行主动学习循环
        
        Args:
            initial_data: 初始已标注数据
            num_iterations: 迭代次数
            samples_per_iteration: 每次迭代选择的样本数
            model_update_callback: 模型更新回调
            early_stopping_threshold: 早停阈值
            
        Returns:
            每次迭代的生成结果列表
        """
        self.set_labeled_data(initial_data)
        results = []
        
        for i in range(num_iterations):
            # 生成新样本
            new_data = self.generate(samples_per_iteration)
            results.append(new_data)
            
            # 检查早停条件
            if early_stopping_threshold is not None:
                if self._check_convergence(new_data, early_stopping_threshold):
                    print(f"在迭代 {i+1} 收敛，提前停止")
                    break
            
            # 更新模型
            if model_update_callback is not None:
                new_model = model_update_callback(self._labeled_data)
                self.set_surrogate_model(new_model)
        
        return results
    
    def _check_convergence(
        self,
        new_data: GenerationResult,
        threshold: float
    ) -> bool:
        """检查是否收敛"""
        if self._uncertainty_estimator is None:
            return False
        
        _, uncertainties = self._uncertainty_estimator.estimate(
            new_data.designs,
            device=self.device
        )
        
        mean_uncertainty = uncertainties.mean()
        
        return mean_uncertainty < threshold
    
    def get_labeled_data(self) -> Optional[GenerationResult]:
        """获取所有已标注数据"""
        return self._labeled_data
    
    def reset(self, seed: Optional[int] = None) -> None:
        """重置生成器状态"""
        super().reset(seed)
        self._labeled_data = None
        self._iteration_count = 0


def create_active_learning_generator(
    design_shape: Tuple[int, ...] = (200, 22),
    acquisition_function: AcquisitionFunction = AcquisitionFunction.UNCERTAINTY,
    candidate_pool_size: int = 10000,
    seed: int = 42,
    simulator: Optional[Callable] = None
) -> ActiveLearningGenerator:
    """
    创建主动学习生成器的便捷函数
    
    Args:
        design_shape: 设计形状
        acquisition_function: 采集函数类型
        candidate_pool_size: 候选池大小
        seed: 随机种子
        simulator: 仿真器函数
        
    Returns:
        配置好的生成器
    """
    config = ActiveLearningConfig(
        design_shape=design_shape,
        acquisition_function=acquisition_function,
        candidate_pool_size=candidate_pool_size,
        seed=seed
    )
    return ActiveLearningGenerator(config, simulator)

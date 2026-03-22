"""
进化算法求解器

实现基于种群的进化优化算法，用于光子学器件设计。
支持多种进化策略，适用于非凸、多模态优化问题。

主要算法:
1. 遗传算法 (GA) - 经典进化算法
2. 差分进化 (DE) - 高效的实数编码进化算法
3. CMA-ES - 协方差矩阵自适应进化策略
4. 粒子群优化 (PSO) - 群智能优化算法

特点:
- 无需梯度信息
- 全局搜索能力强
- 适合离散/混合优化
- 支持多目标优化

参考文献:
- Goldberg (1989). "Genetic Algorithms"
- Storn & Price (1997). "Differential Evolution"
- Hansen (2006). "The CMA Evolution Strategy"
"""

from typing import Dict, Optional, Tuple, List, Union, Callable, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import numpy as np


# ============================================================================
# 配置类
# ============================================================================

@dataclass
class EvolutionaryConfig:
    """进化算法配置基类"""
    name: str = "evolutionary"
    
    # 种群参数
    population_size: int = 50
    max_generations: int = 100
    
    # 收敛参数
    tolerance: float = 1e-6
    patience: int = 20
    
    # 并行化
    n_workers: int = 1
    
    # 随机种子
    seed: Optional[int] = None
    
    # 日志
    verbose: bool = True


@dataclass
class GAConfig(EvolutionaryConfig):
    """遗传算法配置"""
    name: str = "ga"
    
    # 选择
    selection_method: str = "tournament"    # 'tournament', 'roulette', 'rank'
    tournament_size: int = 3
    
    # 交叉
    crossover_prob: float = 0.8
    crossover_method: str = "uniform"       # 'uniform', 'single_point', 'two_point'
    
    # 变异
    mutation_prob: float = 0.1
    mutation_rate: float = 0.01             # 每个基因的变异概率
    mutation_scale: float = 0.1             # 变异幅度
    
    # 精英保留
    elitism: bool = True
    elite_size: int = 2


@dataclass
class DEConfig(EvolutionaryConfig):
    """差分进化配置"""
    name: str = "de"
    
    # 变异策略: DE/rand/1, DE/best/1, DE/rand/2, DE/best/2, DE/current-to-best/1
    mutation_strategy: str = "rand/1"
    
    # 缩放因子
    F: float = 0.8                          # 变异步长
    
    # 交叉
    crossover_prob: float = 0.9             # CR
    crossover_method: str = "binomial"      # 'binomial', 'exponential'
    
    # 自适应
    adaptive: bool = True
    F_range: Tuple[float, float] = (0.5, 1.0)
    CR_range: Tuple[float, float] = (0.5, 1.0)


@dataclass
class CMAESConfig(EvolutionaryConfig):
    """CMA-ES配置"""
    name: str = "cmaes"
    
    # 初始步长
    sigma0: float = 0.5
    
    # 种群大小 (默认自动计算)
    population_size: int = 0                # 0 表示自动
    
    # 学习率
    cc: float = 0.0                         # 0 表示自动
    cs: float = 0.0
    c1: float = 0.0
    cmu: float = 0.0
    
    # 重启策略
    restarts: int = 0
    restart_interval: int = 50


@dataclass
class PSOConfig(EvolutionaryConfig):
    """粒子群优化配置"""
    name: str = "pso"
    
    # 粒子参数
    inertia_weight: float = 0.729           # 惯性权重 w
    cognitive_weight: float = 1.494         # 认知权重 c1
    social_weight: float = 1.494            # 社会权重 c2
    
    # 速度限制
    velocity_scale: float = 0.5             # 速度上限相对搜索空间的比例
    
    # 拓扑
    topology: str = "global"                # 'global', 'ring', 'von_neumann'
    neighbors: int = 3
    
    # 自适应
    adaptive_inertia: bool = True
    inertia_min: float = 0.4
    inertia_max: float = 0.9


# ============================================================================
# 基础进化算法类
# ============================================================================

class EvolutionaryOptimizer(ABC):
    """进化算法基类"""
    
    def __init__(self, config: Optional[EvolutionaryConfig] = None):
        self.config = config or EvolutionaryConfig()
        
        if self.config.seed is not None:
            np.random.seed(self.config.seed)
            random.seed(self.config.seed)
            torch.manual_seed(self.config.seed)
        
        self.generation = 0
        self.population = None
        self.fitness = None
        self.best_individual = None
        self.best_fitness = float('inf')
        self.history: Dict[str, List[float]] = {
            'best_fitness': [],
            'mean_fitness': [],
            'std_fitness': [],
        }
    
    @abstractmethod
    def initialize(self, dim: int, bounds: Tuple[float, float]) -> np.ndarray:
        """初始化种群"""
        pass
    
    @abstractmethod
    def evolve(
        self,
        fitness_fn: Callable,
        bounds: Tuple[float, float],
    ) -> np.ndarray:
        """进化一代"""
        pass
    
    def evaluate(
        self,
        population: np.ndarray,
        fitness_fn: Callable,
    ) -> np.ndarray:
        """评估种群适应度"""
        fitness = np.array([fitness_fn(ind) for ind in population])
        return fitness
    
    def check_convergence(self) -> bool:
        """检查收敛"""
        if len(self.history['best_fitness']) < self.config.patience:
            return False
        
        recent = self.history['best_fitness'][-self.config.patience:]
        return (max(recent) - min(recent)) < self.config.tolerance
    
    def reset(self):
        """重置优化器状态"""
        self.generation = 0
        self.population = None
        self.fitness = None
        self.best_individual = None
        self.best_fitness = float('inf')
        self.history = {'best_fitness': [], 'mean_fitness': [], 'std_fitness': []}


# ============================================================================
# 遗传算法
# ============================================================================

class GeneticAlgorithm(EvolutionaryOptimizer):
    """遗传算法"""
    
    def __init__(self, config: Optional[GAConfig] = None):
        super().__init__(config or GAConfig())
        self.config: GAConfig = self.config
    
    def initialize(self, dim: int, bounds: Tuple[float, float]) -> np.ndarray:
        """初始化种群"""
        self.dim = dim
        self.bounds = bounds
        
        # 随机初始化
        self.population = np.random.uniform(
            bounds[0], bounds[1],
            size=(self.config.population_size, dim)
        )
        
        return self.population
    
    def evolve(
        self,
        fitness_fn: Callable,
        bounds: Tuple[float, float],
    ) -> np.ndarray:
        """进化一代"""
        # 评估适应度（最小化问题，取负）
        self.fitness = self.evaluate(self.population, fitness_fn)
        
        # 更新最佳个体
        best_idx = np.argmin(self.fitness)
        if self.fitness[best_idx] < self.best_fitness:
            self.best_fitness = self.fitness[best_idx]
            self.best_individual = self.population[best_idx].copy()
        
        # 记录历史
        self.history['best_fitness'].append(self.best_fitness)
        self.history['mean_fitness'].append(np.mean(self.fitness))
        self.history['std_fitness'].append(np.std(self.fitness))
        
        # 选择
        selected = self._selection()
        
        # 交叉
        offspring = self._crossover(selected)
        
        # 变异
        offspring = self._mutation(offspring, bounds)
        
        # 精英保留
        if self.config.elitism:
            elite_indices = np.argsort(self.fitness)[:self.config.elite_size]
            offspring[:self.config.elite_size] = self.population[elite_indices]
        
        self.population = offspring
        self.generation += 1
        
        return self.population
    
    def _selection(self) -> np.ndarray:
        """选择操作"""
        pop_size = self.config.population_size
        selected = np.zeros_like(self.population)
        
        for i in range(pop_size):
            if self.config.selection_method == "tournament":
                # 锦标赛选择
                tournament_indices = np.random.choice(pop_size, self.config.tournament_size, replace=False)
                tournament_fitness = self.fitness[tournament_indices]
                winner_idx = tournament_indices[np.argmin(tournament_fitness)]
                selected[i] = self.population[winner_idx]
            
            elif self.config.selection_method == "roulette":
                # 轮盘赌选择（适用于最大化，这里反转）
                fitness_neg = -self.fitness
                fitness_shifted = fitness_neg - fitness_neg.min() + 1e-8
                probs = fitness_shifted / fitness_shifted.sum()
                selected[i] = self.population[np.random.choice(pop_size, p=probs)]
            
            elif self.config.selection_method == "rank":
                # 排名选择
                ranks = np.argsort(np.argsort(self.fitness))
                probs = (ranks + 1) / (ranks + 1).sum()
                selected[i] = self.population[np.random.choice(pop_size, p=probs)]
        
        return selected
    
    def _crossover(self, selected: np.ndarray) -> np.ndarray:
        """交叉操作"""
        pop_size = self.config.population_size
        offspring = selected.copy()
        
        for i in range(0, pop_size - 1, 2):
            if np.random.random() < self.config.crossover_prob:
                parent1, parent2 = selected[i], selected[i + 1]
                
                if self.config.crossover_method == "uniform":
                    # 均匀交叉
                    mask = np.random.random(self.dim) < 0.5
                    offspring[i] = np.where(mask, parent1, parent2)
                    offspring[i + 1] = np.where(mask, parent2, parent1)
                
                elif self.config.crossover_method == "single_point":
                    # 单点交叉
                    point = np.random.randint(1, self.dim)
                    offspring[i] = np.concatenate([parent1[:point], parent2[point:]])
                    offspring[i + 1] = np.concatenate([parent2[:point], parent1[point:]])
                
                elif self.config.crossover_method == "two_point":
                    # 两点交叉
                    points = sorted(np.random.choice(range(1, self.dim), 2, replace=False))
                    offspring[i] = np.concatenate([
                        parent1[:points[0]],
                        parent2[points[0]:points[1]],
                        parent1[points[1]:]
                    ])
                    offspring[i + 1] = np.concatenate([
                        parent2[:points[0]],
                        parent1[points[0]:points[1]],
                        parent2[points[1]:]
                    ])
        
        return offspring
    
    def _mutation(self, offspring: np.ndarray, bounds: Tuple[float, float]) -> np.ndarray:
        """变异操作"""
        mutation_mask = np.random.random(offspring.shape) < self.config.mutation_rate
        
        # 高斯变异
        mutation = np.random.normal(0, self.config.mutation_scale, offspring.shape)
        offspring = offspring + mutation_mask * mutation
        
        # 边界处理
        offspring = np.clip(offspring, bounds[0], bounds[1])
        
        return offspring


# ============================================================================
# 差分进化
# ============================================================================

class DifferentialEvolution(EvolutionaryOptimizer):
    """差分进化算法"""
    
    def __init__(self, config: Optional[DEConfig] = None):
        super().__init__(config or DEConfig())
        self.config: DEConfig = self.config
        
        # 自适应参数
        self.F_current = self.config.F
        self.CR_current = self.config.crossover_prob
    
    def initialize(self, dim: int, bounds: Tuple[float, float]) -> np.ndarray:
        """初始化种群"""
        self.dim = dim
        self.bounds = bounds
        
        self.population = np.random.uniform(
            bounds[0], bounds[1],
            size=(self.config.population_size, dim)
        )
        
        return self.population
    
    def evolve(
        self,
        fitness_fn: Callable,
        bounds: Tuple[float, float],
    ) -> np.ndarray:
        """进化一代"""
        pop_size = self.config.population_size
        
        # 评估适应度
        self.fitness = self.evaluate(self.population, fitness_fn)
        
        # 更新最佳
        best_idx = np.argmin(self.fitness)
        if self.fitness[best_idx] < self.best_fitness:
            self.best_fitness = self.fitness[best_idx]
            self.best_individual = self.population[best_idx].copy()
        
        # 记录历史
        self.history['best_fitness'].append(self.best_fitness)
        self.history['mean_fitness'].append(np.mean(self.fitness))
        self.history['std_fitness'].append(np.std(self.fitness))
        
        # 自适应参数调整
        if self.config.adaptive:
            self._adapt_parameters()
        
        # 生成试验向量
        new_population = np.zeros_like(self.population)
        
        for i in range(pop_size):
            # 变异
            mutant = self._mutation(i)
            
            # 交叉
            trial = self._crossover(self.population[i], mutant)
            
            # 选择
            trial_fitness = fitness_fn(trial)
            if trial_fitness <= self.fitness[i]:
                new_population[i] = trial
            else:
                new_population[i] = self.population[i]
        
        self.population = new_population
        self.generation += 1
        
        return self.population
    
    def _mutation(self, target_idx: int) -> np.ndarray:
        """差分变异"""
        pop_size = self.config.population_size
        strategy = self.config.mutation_strategy
        
        # 随机选择个体（排除目标个体）
        candidates = [i for i in range(pop_size) if i != target_idx]
        
        if strategy == "rand/1":
            r1, r2, r3 = np.random.choice(candidates, 3, replace=False)
            mutant = self.population[r1] + self.F_current * (
                self.population[r2] - self.population[r3]
            )
        
        elif strategy == "best/1":
            r1, r2 = np.random.choice(candidates, 2, replace=False)
            mutant = self.best_individual + self.F_current * (
                self.population[r1] - self.population[r2]
            )
        
        elif strategy == "rand/2":
            r1, r2, r3, r4, r5 = np.random.choice(candidates, 5, replace=False)
            mutant = self.population[r1] + self.F_current * (
                self.population[r2] - self.population[r3]
            ) + self.F_current * (
                self.population[r4] - self.population[r5]
            )
        
        elif strategy == "best/2":
            r1, r2, r3, r4 = np.random.choice(candidates, 4, replace=False)
            mutant = self.best_individual + self.F_current * (
                self.population[r1] - self.population[r2]
            ) + self.F_current * (
                self.population[r3] - self.population[r4]
            )
        
        elif strategy == "current-to-best/1":
            r1, r2 = np.random.choice(candidates, 2, replace=False)
            mutant = self.population[target_idx] + self.F_current * (
                self.best_individual - self.population[target_idx]
            ) + self.F_current * (
                self.population[r1] - self.population[r2]
            )
        
        else:
            raise ValueError(f"Unknown mutation strategy: {strategy}")
        
        # 边界处理
        mutant = np.clip(mutant, self.bounds[0], self.bounds[1])
        
        return mutant
    
    def _crossover(self, target: np.ndarray, mutant: np.ndarray) -> np.ndarray:
        """交叉操作"""
        trial = target.copy()
        
        if self.config.crossover_method == "binomial":
            # 二项式交叉
            cross_points = np.random.random(self.dim) < self.CR_current
            # 确保至少一个维度来自变异个体
            cross_points[np.random.randint(self.dim)] = True
            trial[cross_points] = mutant[cross_points]
        
        elif self.config.crossover_method == "exponential":
            # 指数交叉
            n = np.random.randint(self.dim)
            L = 0
            while np.random.random() < self.CR_current and L < self.dim:
                trial[n] = mutant[n]
                n = (n + 1) % self.dim
                L += 1
        
        return trial
    
    def _adapt_parameters(self):
        """自适应参数调整"""
        # 根据成功率调整 F 和 CR
        self.F_current = np.random.uniform(*self.config.F_range)
        self.CR_current = np.random.uniform(*self.config.CR_range)


# ============================================================================
# CMA-ES
# ============================================================================

class CMAES(EvolutionaryOptimizer):
    """CMA-ES 进化策略"""
    
    def __init__(self, config: Optional[CMAESConfig] = None):
        super().__init__(config or CMAESConfig())
        self.config: CMAESConfig = self.config
    
    def initialize(self, dim: int, bounds: Tuple[float, float]) -> np.ndarray:
        """初始化"""
        self.dim = dim
        self.bounds = bounds
        
        # 自动计算种群大小
        if self.config.population_size == 0:
            self.lambda_ = 4 + int(3 * np.log(dim))
        else:
            self.lambda_ = self.config.population_size
        
        # 选择参数
        self.mu = self.lambda_ // 2
        
        # 重组权重
        weights_prime = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = weights_prime / weights_prime.sum()
        self.mueff = 1 / (self.weights ** 2).sum()
        
        # 步长控制参数
        self.cs = (self.mueff + 2) / (dim + self.mueff + 5)
        self.ds = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (dim + 1)) - 1) + self.cs
        
        # 协方差矩阵更新参数
        self.cc = (4 + self.mueff / dim) / (dim + 4 + 2 * self.mueff / dim)
        self.c1 = 2 / ((dim + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((dim + 2) ** 2 + self.mueff))
        
        # 初始化状态变量
        self.mean = np.random.uniform(bounds[0], bounds[1], dim)
        self.sigma = self.config.sigma0
        self.C = np.eye(dim)  # 协方差矩阵
        self.pc = np.zeros(dim)  # 进化路径
        self.ps = np.zeros(dim)  # 共轭进化路径
        
        # 初始种群
        self.population = self._sample_population()
        
        return self.population
    
    def _sample_population(self) -> np.ndarray:
        """采样新种群"""
        # 使用 Cholesky 分解采样
        try:
            A = np.linalg.cholesky(self.C)
        except np.linalg.LinAlgError:
            # 如果 C 不是正定的，强制修正
            eigvals, eigvecs = np.linalg.eigh(self.C)
            eigvals = np.maximum(eigvals, 1e-10)
            A = eigvecs @ np.diag(np.sqrt(eigvals))
        
        z = np.random.randn(self.lambda_, self.dim)
        y = z @ A.T
        population = self.mean + self.sigma * y
        
        # 边界处理
        population = np.clip(population, self.bounds[0], self.bounds[1])
        
        return population
    
    def evolve(
        self,
        fitness_fn: Callable,
        bounds: Tuple[float, float],
    ) -> np.ndarray:
        """进化一代"""
        # 评估适应度
        self.fitness = self.evaluate(self.population, fitness_fn)
        
        # 排序
        sorted_indices = np.argsort(self.fitness)
        
        # 更新最佳
        if self.fitness[sorted_indices[0]] < self.best_fitness:
            self.best_fitness = self.fitness[sorted_indices[0]]
            self.best_individual = self.population[sorted_indices[0]].copy()
        
        # 记录历史
        self.history['best_fitness'].append(self.best_fitness)
        self.history['mean_fitness'].append(np.mean(self.fitness))
        self.history['std_fitness'].append(np.std(self.fitness))
        
        # 更新均值
        y_sorted = (self.population[sorted_indices] - self.mean) / self.sigma
        y_mean = self.weights @ y_sorted[:self.mu]
        self.mean = self.mean + self.sigma * y_mean
        
        # 更新进化路径
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * y_mean
        
        # 更新协方差矩阵进化路径
        hsig = (np.sum(self.ps ** 2) / (1 - (1 - self.cs) ** (2 * (self.generation + 1))) < 
                2 + 4 / (self.dim + 1)) * 1.0
        self.pc = (1 - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * y_mean
        
        # 更新协方差矩阵
        artmp = y_sorted[:self.mu]
        self.C = ((1 - self.c1 - self.cmu) * self.C + 
                  self.c1 * np.outer(self.pc, self.pc) + 
                  self.cmu * np.sum([self.weights[i] * np.outer(artmp[i], artmp[i]) 
                                     for i in range(self.mu)], axis=0))
        
        # 更新步长
        self.sigma = self.sigma * np.exp((self.cs / self.ds) * (np.linalg.norm(self.ps) / np.sqrt(self.dim) - 1))
        
        # 限制步长
        self.sigma = np.clip(self.sigma, 1e-20, 1e10)
        
        # 采样新种群
        self.population = self._sample_population()
        self.generation += 1
        
        return self.population


# ============================================================================
# 粒子群优化
# ============================================================================

class ParticleSwarmOptimization(EvolutionaryOptimizer):
    """粒子群优化算法"""
    
    def __init__(self, config: Optional[PSOConfig] = None):
        super().__init__(config or PSOConfig())
        self.config: PSOConfig = self.config
        
        self.velocities = None
        self.personal_best = None
        self.personal_best_fitness = None
        self.global_best = None
        self.global_best_fitness = float('inf')
    
    def initialize(self, dim: int, bounds: Tuple[float, float]) -> np.ndarray:
        """初始化粒子群"""
        self.dim = dim
        self.bounds = bounds
        pop_size = self.config.population_size
        
        # 初始化位置
        self.population = np.random.uniform(bounds[0], bounds[1], (pop_size, dim))
        
        # 初始化速度
        v_max = self.config.velocity_scale * (bounds[1] - bounds[0])
        self.velocities = np.random.uniform(-v_max, v_max, (pop_size, dim))
        
        # 初始化个体最优
        self.personal_best = self.population.copy()
        self.personal_best_fitness = np.full(pop_size, float('inf'))
        
        # 全局最优
        self.global_best = None
        self.global_best_fitness = float('inf')
        
        return self.population
    
    def evolve(
        self,
        fitness_fn: Callable,
        bounds: Tuple[float, float],
    ) -> np.ndarray:
        """进化一代"""
        pop_size = self.config.population_size
        
        # 评估适应度
        self.fitness = self.evaluate(self.population, fitness_fn)
        
        # 更新个体最优
        improved = self.fitness < self.personal_best_fitness
        self.personal_best[improved] = self.population[improved]
        self.personal_best_fitness[improved] = self.fitness[improved]
        
        # 更新全局最优
        best_idx = np.argmin(self.personal_best_fitness)
        if self.personal_best_fitness[best_idx] < self.global_best_fitness:
            self.global_best = self.personal_best[best_idx].copy()
            self.global_best_fitness = self.personal_best_fitness[best_idx]
        
        # 更新最佳记录
        if self.global_best_fitness < self.best_fitness:
            self.best_fitness = self.global_best_fitness
            self.best_individual = self.global_best.copy()
        
        # 记录历史
        self.history['best_fitness'].append(self.best_fitness)
        self.history['mean_fitness'].append(np.mean(self.fitness))
        self.history['std_fitness'].append(np.std(self.fitness))
        
        # 自适应惯性权重
        if self.config.adaptive_inertia:
            progress = self.generation / self.config.max_generations
            w = self.config.inertia_max - (self.config.inertia_max - self.config.inertia_min) * progress
        else:
            w = self.config.inertia_weight
        
        # 更新速度和位置
        r1, r2 = np.random.random((2, pop_size, self.dim))
        
        cognitive = self.config.cognitive_weight * r1 * (self.personal_best - self.population)
        social = self.config.social_weight * r2 * (self._get_neighbors_best() - self.population)
        
        self.velocities = w * self.velocities + cognitive + social
        
        # 限制速度
        v_max = self.config.velocity_scale * (bounds[1] - bounds[0])
        self.velocities = np.clip(self.velocities, -v_max, v_max)
        
        # 更新位置
        self.population = self.population + self.velocities
        self.population = np.clip(self.population, bounds[0], bounds[1])
        
        self.generation += 1
        
        return self.population
    
    def _get_neighbors_best(self) -> np.ndarray:
        """获取邻居最优位置"""
        pop_size = self.config.population_size
        
        if self.config.topology == "global":
            return np.tile(self.global_best, (pop_size, 1))
        
        elif self.config.topology == "ring":
            neighbors_best = np.zeros_like(self.population)
            for i in range(pop_size):
                # 环形拓扑邻居
                neighbor_indices = [(i + j) % pop_size for j in range(-self.config.neighbors, self.config.neighbors + 1)]
                best_neighbor_idx = neighbor_indices[np.argmin(self.personal_best_fitness[neighbor_indices])]
                neighbors_best[i] = self.personal_best[best_neighbor_idx]
            return neighbors_best
        
        else:
            return np.tile(self.global_best, (pop_size, 1))


# ============================================================================
# 优化器工厂
# ============================================================================

def create_evolutionary_optimizer(
    name: str = "de",
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> EvolutionaryOptimizer:
    """
    创建进化算法优化器
    
    Args:
        name: 算法名称 ('ga', 'de', 'cmaes', 'pso')
        config: 配置字典
        **kwargs: 其他参数
        
    Returns:
        进化算法优化器实例
    """
    optimizers = {
        'ga': GeneticAlgorithm,
        'genetic': GeneticAlgorithm,
        'de': DifferentialEvolution,
        'differential_evolution': DifferentialEvolution,
        'cmaes': CMAES,
        'cma-es': CMAES,
        'pso': ParticleSwarmOptimization,
        'particle_swarm': ParticleSwarmOptimization,
    }
    
    if name not in optimizers:
        raise ValueError(f"Unknown optimizer: {name}, available: {list(optimizers.keys())}")
    
    return optimizers[name](config=config if config else None, **kwargs)


# ============================================================================
# 进化算法运行器
# ============================================================================

class EvolutionaryRunner:
    """进化算法运行器"""
    
    def __init__(
        self,
        optimizer: EvolutionaryOptimizer,
        fitness_fn: Callable,
        dim: int,
        bounds: Tuple[float, float] = (0.0, 1.0),
        verbose: bool = True,
    ):
        self.optimizer = optimizer
        self.fitness_fn = fitness_fn
        self.dim = dim
        self.bounds = bounds
        self.verbose = verbose
    
    def run(
        self,
        max_generations: Optional[int] = None,
        callback: Optional[Callable] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """运行优化"""
        max_gen = max_generations or self.optimizer.config.max_generations
        
        # 初始化
        self.optimizer.initialize(self.dim, self.bounds)
        
        for gen in range(max_gen):
            # 进化
            self.optimizer.evolve(self.fitness_fn, self.bounds)
            
            # 回调
            if callback is not None:
                callback(self.optimizer.population, self.optimizer.fitness, gen)
            
            # 打印进度
            if self.verbose and gen % 10 == 0:
                print(f"Generation {gen}: best = {self.optimizer.best_fitness:.6e}, "
                      f"mean = {self.optimizer.history['mean_fitness'][-1]:.6e}")
            
            # 检查收敛
            if self.optimizer.check_convergence():
                if self.verbose:
                    print(f"Converged at generation {gen}")
                break
        
        info = {
            'generation': self.optimizer.generation,
            'best_fitness': self.optimizer.best_fitness,
            'best_individual': self.optimizer.best_individual,
            'history': self.optimizer.history,
        }
        
        return self.optimizer.best_individual, info

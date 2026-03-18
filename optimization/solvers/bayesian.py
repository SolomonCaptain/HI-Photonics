"""
贝叶斯优化求解器

实现基于高斯过程的贝叶斯优化，用于在低维潜在空间中
进行高效的全局优化。

主要组件:
1. 高斯过程回归模型
2. 核函数 (RBF, Matern, Spectral)
3. 采集函数 (EI, UCB, PI, KG)
"""

from typing import Dict, Optional, Tuple, List, Union, Callable, Any
from dataclasses import dataclass, field
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


# ============================================================================
# 核函数
# ============================================================================

class KernelFunction:
    """核函数基类"""

    def __init__(self, lengthscale: float = 1.0, variance: float = 1.0):
        self.lengthscale = lengthscale
        self.variance = variance

    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def diagonal(self, X: np.ndarray) -> np.ndarray:
        """计算核矩阵的对角元素"""
        return np.full(len(X), self.variance)


class RBFKernel(KernelFunction):
    """
    径向基函数核 (高斯核)

    k(x, x') = σ² * exp(-||x - x'||² / (2 * l²))

    其中 σ² 是方差，l 是长度尺度
    """

    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)

        # 计算平方距离
        X1_sq = np.sum(X1 ** 2, axis=1, keepdims=True)
        X2_sq = np.sum(X2 ** 2, axis=1, keepdims=True)
        sq_dist = X1_sq + X2_sq.T - 2 * X1 @ X2.T

        # RBF 核
        K = self.variance * np.exp(-0.5 * sq_dist / (self.lengthscale ** 2))

        return K


class MaternKernel(KernelFunction):
    """
    Matern 核

    k(x, x') = σ² * 2^(1-ν) / Γ(ν) * (√(2ν) r / l)^ν * K_ν(√(2ν) r / l)

    其中 r = ||x - x'||, ν 控制平滑度 (常用 1.5 或 2.5)
    """

    def __init__(self, lengthscale: float = 1.0, variance: float = 1.0, nu: float = 2.5):
        super().__init__(lengthscale, variance)
        self.nu = nu

    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        from scipy.special import kv, gamma

        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)

        # 计算距离
        X1_sq = np.sum(X1 ** 2, axis=1, keepdims=True)
        X2_sq = np.sum(X2 ** 2, axis=1, keepdims=True)
        dist = np.sqrt(np.maximum(X1_sq + X2_sq.T - 2 * X1 @ X2.T, 1e-12))

        # 避免除零
        dist_nz = np.where(dist == 0, 1e-12, dist)

        # 归一化距离
        scaled_dist = np.sqrt(2 * self.nu) * dist_nz / self.lengthscale

        if self.nu == 0.5:
            # Matern 1/2 (指数核)
            K = self.variance * np.exp(-scaled_dist)
        elif self.nu == 1.5:
            # Matern 3/2
            K = self.variance * (1 + scaled_dist) * np.exp(-scaled_dist)
        elif self.nu == 2.5:
            # Matern 5/2
            K = self.variance * (1 + scaled_dist + scaled_dist ** 2 / 3) * np.exp(-scaled_dist)
        else:
            # 通用 Matern
            K = self.variance * (2 ** (1 - self.nu) / gamma(self.nu) *
                                 scaled_dist ** self.nu * kv(self.nu, scaled_dist))

        # 处理零距离点
        K[dist == 0] = self.variance

        return K


class SpectralMixtureKernel(KernelFunction):
    """
    谱混合核

    组合多个高斯核以捕获不同频率的模式:
    k(x, x') = Σ_q σ_q² * exp(-2π² τ² / l_q²) * cos(2π τ μ_q)

    其中 τ = ||x - x'||, Q 是混合数
    """

    def __init__(
        self,
        n_mixtures: int = 4,
        lengthscales: Optional[List[float]] = None,
        variances: Optional[List[float]] = None,
        frequencies: Optional[List[float]] = None
    ):
        self.n_mixtures = n_mixtures

        # 默认参数
        self.lengthscales = lengthscales or [1.0] * n_mixtures
        self.variances = variances or [1.0 / n_mixtures] * n_mixtures
        self.frequencies = frequencies or [0.1 * i for i in range(n_mixtures)]

    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)

        # 计算距离
        X1_sq = np.sum(X1 ** 2, axis=1, keepdims=True)
        X2_sq = np.sum(X2 ** 2, axis=1, keepdims=True)
        dist = np.sqrt(np.maximum(X1_sq + X2_sq.T - 2 * X1 @ X2.T, 1e-12))

        K = np.zeros((X1.shape[0], X2.shape[0]))

        for q in range(self.n_mixtures):
            l_q = self.lengthscales[q]
            var_q = self.variances[q]
            freq_q = self.frequencies[q]

            # 高斯部分
            gauss = np.exp(-2 * np.pi ** 2 * dist ** 2 / l_q ** 2)
            # 余弦部分
            cos = np.cos(2 * np.pi * freq_q * dist)

            K += var_q * gauss * cos

        return K


# ============================================================================
# 高斯过程回归
# ============================================================================

class GaussianProcessRegressor:
    """
    高斯过程回归模型

    使用核函数定义先验分布，通过训练数据更新后验分布。

    使用示例:
    ```python
    # 创建 GP
    kernel = RBFKernel(lengthscale=1.0, variance=1.0)
    gp = GaussianProcessRegressor(kernel, noise_variance=1e-4)

    # 训练
    gp.fit(X_train, y_train)

    # 预测
    y_mean, y_std = gp.predict(X_test)
    ```
    """

    def __init__(
        self,
        kernel: Optional[KernelFunction] = None,
        noise_variance: float = 1e-4,
        normalize_y: bool = True
    ):
        """
        Args:
            kernel: 核函数
            noise_variance: 观测噪声方差
            normalize_y: 是否标准化目标值
        """
        self.kernel = kernel or RBFKernel()
        self.noise_variance = noise_variance
        self.normalize_y = normalize_y

        # 训练数据
        self.X_train = None
        self.y_train = None

        # 预计算的量
        self.K_inv = None  # K^{-1}
        self.alpha = None  # K^{-1} y
        self.y_mean = 0.0
        self.y_std = 1.0

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        训练高斯过程

        Args:
            X: 输入 [N, D]
            y: 目标 [N, 1] 或 [N]
        """
        X = np.atleast_2d(X)
        y = np.atleast_1d(y).flatten()

        self.X_train = X
        self.y_train = y

        # 标准化目标值
        if self.normalize_y:
            self.y_mean = y.mean()
            self.y_std = y.std() + 1e-8
            y_normalized = (y - self.y_mean) / self.y_std
        else:
            y_normalized = y

        # 计算核矩阵
        K = self.kernel(X, X)

        # 添加噪声
        K += self.noise_variance * np.eye(len(X))

        # 确保 K 正定
        K = self._ensure_positive_definite(K)

        # 预计算
        try:
            self.K_inv = np.linalg.inv(K)
        except np.linalg.LinAlgError:
            # 使用伪逆作为备选
            self.K_inv = np.linalg.pinv(K)

        self.alpha = self.K_inv @ y_normalized

        return self

    def predict(
        self,
        X: np.ndarray,
        return_std: bool = True,
        return_cov: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        预测新点的均值和方差

        Args:
            X: 测试输入 [M, D]
            return_std: 是否返回标准差
            return_cov: 是否返回协方差

        Returns:
            y_mean: 预测均值 [M]
            y_std: 预测标准差 [M] (可选)
            y_cov: 预测协方差 [M, M] (可选)
        """
        if self.X_train is None:
            raise RuntimeError("GP not fitted. Call fit() first.")

        X = np.atleast_2d(X)

        # 核矩阵
        K_star = self.kernel(X, self.X_train)  # [M, N]
        K_star_star = self.kernel(X, X)  # [M, M]

        # 预测均值
        y_mean = K_star @ self.alpha
        y_mean = y_mean * self.y_std + self.y_mean

        if not return_std and not return_cov:
            return y_mean

        # 预测方差
        v = self.K_inv @ K_star.T  # [N, M]
        y_cov = K_star_star - K_star @ v

        # 添加噪声
        y_cov += self.noise_variance * np.eye(len(X))

        # 确保正定
        y_cov = self._ensure_positive_definite(y_cov)

        y_var = np.diag(y_cov)
        y_std = np.sqrt(np.maximum(y_var, 1e-10))

        if return_cov:
            return y_mean, y_std, y_cov
        return y_mean, y_std

    def _ensure_positive_definite(self, K: np.ndarray) -> np.ndarray:
        """确保矩阵正定"""
        # 对称化
        K = (K + K.T) / 2

        # 检查特征值
        eigvals = np.linalg.eigvalsh(K)
        if eigvals.min() < 1e-10:
            # 添加小的对角偏移
            K += (1e-10 - eigvals.min() + 1e-10) * np.eye(len(K))

        return K

    def log_marginal_likelihood(self) -> float:
        """计算对数边际似然"""
        if self.X_train is None:
            return 0.0

        N = len(self.y_train)
        y_normalized = (self.y_train - self.y_mean) / self.y_std

        # log p(y|X) = -0.5 * y^T K^{-1} y - 0.5 * log|K| - N/2 log(2π)
        K = self.kernel(self.X_train, self.X_train)
        K += self.noise_variance * np.eye(N)
        K = self._ensure_positive_definite(K)

        try:
            sign, logdet = np.linalg.slogdet(K)
            if sign <= 0:
                logdet = N * np.log(1e-10)

            lml = -0.5 * y_normalized @ self.alpha - 0.5 * logdet - 0.5 * N * np.log(2 * np.pi)
        except:
            lml = -np.inf

        return lml


# ============================================================================
# 采集函数
# ============================================================================

class AcquisitionFunction:
    """采集函数基类"""

    def __call__(
        self,
        X: np.ndarray,
        gp: GaussianProcessRegressor,
        y_best: float
    ) -> np.ndarray:
        raise NotImplementedError


class ExpectedImprovement(AcquisitionFunction):
    """
    期望改进 (Expected Improvement)

    EI(x) = E[max(f(x) - f(x⁺), 0)]

    = (μ(x) - f(x⁺) - ξ) * Φ(Z) + σ(x) * φ(Z)

    其中 Z = (μ(x) - f(x⁺) - ξ) / σ(x)
    """

    def __init__(self, xi: float = 0.01):
        """
        Args:
            xi: 探索-利用平衡参数
        """
        self.xi = xi

    def __call__(
        self,
        X: np.ndarray,
        gp: GaussianProcessRegressor,
        y_best: float
    ) -> np.ndarray:
        X = np.atleast_2d(X)

        mu, sigma = gp.predict(X, return_std=True)

        # 避免除零
        sigma = np.maximum(sigma, 1e-10)

        # 计算 Z
        with np.errstate(divide='warn'):
            imp = mu - y_best - self.xi
            Z = imp / sigma

        # 计算 EI
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)

        # 将负值设为零
        ei = np.maximum(ei, 0)

        return ei


class UpperConfidenceBound(AcquisitionFunction):
    """
    上置信界 (Upper Confidence Bound)

    UCB(x) = μ(x) + β * σ(x)

    其中 β 控制探索程度
    """

    def __init__(self, beta: float = 2.0):
        """
        Args:
            beta: 探索参数
        """
        self.beta = beta

    def __call__(
        self,
        X: np.ndarray,
        gp: GaussianProcessRegressor,
        y_best: float
    ) -> np.ndarray:
        X = np.atleast_2d(X)

        mu, sigma = gp.predict(X, return_std=True)

        return mu + self.beta * sigma


class ProbabilityOfImprovement(AcquisitionFunction):
    """
    改进概率 (Probability of Improvement)

    PI(x) = Φ((μ(x) - f(x⁺) - ξ) / σ(x))
    """

    def __init__(self, xi: float = 0.01):
        self.xi = xi

    def __call__(
        self,
        X: np.ndarray,
        gp: GaussianProcessRegressor,
        y_best: float
    ) -> np.ndarray:
        X = np.atleast_2d(X)

        mu, sigma = gp.predict(X, return_std=True)
        sigma = np.maximum(sigma, 1e-10)

        with np.errstate(divide='warn'):
            Z = (mu - y_best - self.xi) / sigma

        return norm.cdf(Z)


class KnowledgeGradient(AcquisitionFunction):
    """
    知识梯度 (Knowledge Gradient)

    KG 量化了在当前最佳点获得的信息价值。
    比较复杂，这里使用近似实现。
    """

    def __init__(self, n_samples: int = 100):
        self.n_samples = n_samples

    def __call__(
        self,
        X: np.ndarray,
        gp: GaussianProcessRegressor,
        y_best: float
    ) -> np.ndarray:
        X = np.atleast_2d(X)
        M = X.shape[0]

        mu, sigma = gp.predict(X, return_std=True)

        # 采样近似 KG
        kg = np.zeros(M)

        for i in range(M):
            # 采样候选值
            samples = np.random.normal(mu[i], sigma[i], self.n_samples)

            # 计算期望的最佳改进
            kg[i] = np.mean(np.maximum(samples - y_best, 0))

        return kg


# ============================================================================
# 贝叶斯优化器
# ============================================================================

class BayesianOptimizer:
    """
    贝叶斯优化器

    使用高斯过程作为代理模型，通过采集函数引导搜索。

    使用示例:
    ```python
    # 定义目标函数
    def objective(x):
        return -(x ** 2).sum()  # 最大化

    # 创建优化器
    optimizer = BayesianOptimizer(
        dim=10,
        bounds=(-3, 3),
        acquisition_type='ei'
    )

    # 优化
    best_x, best_y = optimizer.optimize(objective, n_iterations=50)
    ```
    """

    def __init__(
        self,
        dim: int,
        bounds: Tuple[float, float] = (-3.0, 3.0),
        kernel_type: str = 'rbf',
        kernel_lengthscale: float = 1.0,
        kernel_variance: float = 1.0,
        noise_variance: float = 1e-4,
        acquisition_type: str = 'ei',
        ucb_beta: float = 2.0,
        xi: float = 0.01,
        n_restarts: int = 10
    ):
        """
        Args:
            dim: 搜索空间维度
            bounds: 每个维度的边界 (low, high)
            kernel_type: 核函数类型 ('rbf', 'matern', 'spectral')
            kernel_lengthscale: 核长度尺度
            kernel_variance: 核方差
            noise_variance: 观测噪声方差
            acquisition_type: 采集函数类型 ('ei', 'ucb', 'pi', 'kg')
            ucb_beta: UCB 的 beta 参数
            xi: EI/PI 的 xi 参数
            n_restarts: 采集函数优化的重启次数
        """
        self.dim = dim
        self.bounds = bounds
        self.n_restarts = n_restarts

        # 创建核函数
        if kernel_type == 'rbf':
            self.kernel = RBFKernel(kernel_lengthscale, kernel_variance)
        elif kernel_type == 'matern':
            self.kernel = MaternKernel(kernel_lengthscale, kernel_variance, nu=2.5)
        elif kernel_type == 'spectral':
            self.kernel = SpectralMixtureKernel(n_mixtures=4)
        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}")

        # 创建高斯过程
        self.gp = GaussianProcessRegressor(self.kernel, noise_variance)

        # 创建采集函数
        if acquisition_type == 'ei':
            self.acquisition = ExpectedImprovement(xi)
        elif acquisition_type == 'ucb':
            self.acquisition = UpperConfidenceBound(ucb_beta)
        elif acquisition_type == 'pi':
            self.acquisition = ProbabilityOfImprovement(xi)
        elif acquisition_type == 'kg':
            self.acquisition = KnowledgeGradient()
        else:
            raise ValueError(f"Unknown acquisition type: {acquisition_type}")

        # 数据存储
        self.X_observed = []
        self.y_observed = []

    def suggest_next(self) -> np.ndarray:
        """
        建议下一个评估点

        Returns:
            x_next: 建议的下一个点 [D]
        """
        if len(self.X_observed) < 2:
            # 数据不足，随机采样
            return self._random_sample()

        # 更新 GP
        X = np.array(self.X_observed)
        y = np.array(self.y_observed)
        self.gp.fit(X, y)

        # 当前最佳值
        y_best = np.max(y)

        # 优化采集函数
        x_next = self._optimize_acquisition(y_best)

        return x_next

    def observe(self, x: np.ndarray, y: float):
        """
        记录观测值

        Args:
            x: 评估点
            y: 目标值
        """
        self.X_observed.append(np.atleast_1d(x).flatten())
        self.y_observed.append(y)

    def optimize(
        self,
        objective: Callable[[np.ndarray], float],
        n_iterations: int,
        n_initial: int = 5,
        verbose: bool = True
    ) -> Tuple[np.ndarray, float]:
        """
        执行贝叶斯优化

        Args:
            objective: 目标函数 (最大化)
            n_iterations: 迭代次数
            n_initial: 初始随机采样数
            verbose: 是否打印进度

        Returns:
            best_x: 最优点
            best_y: 最优值
        """
        # 初始随机采样
        if len(self.X_observed) < n_initial:
            n_init_needed = n_initial - len(self.X_observed)
            if verbose:
                print(f"Initial sampling: {n_init_needed} points")

            for _ in range(n_init_needed):
                x = self._random_sample()
                y = objective(x)
                self.observe(x, y)

        # 贝叶斯优化循环
        for i in range(n_iterations):
            # 建议
            x_next = self.suggest_next()

            # 评估
            y_next = objective(x_next)
            self.observe(x_next, y_next)

            if verbose and (i + 1) % 10 == 0:
                best_y = np.max(self.y_observed)
                print(f"Iteration {i + 1}/{n_iterations}: best_y = {best_y:.6f}")

        # 返回最佳结果
        best_idx = np.argmax(self.y_observed)
        best_x = np.array(self.X_observed[best_idx])
        best_y = self.y_observed[best_idx]

        return best_x, best_y

    def _random_sample(self) -> np.ndarray:
        """在边界内随机采样"""
        return np.random.uniform(self.bounds[0], self.bounds[1], self.dim)

    def _optimize_acquisition(self, y_best: float) -> np.ndarray:
        """优化采集函数"""
        best_x = None
        best_acq = -np.inf

        # 多起点优化
        for _ in range(self.n_restarts):
            # 随机起点
            x0 = self._random_sample()

            # 定义负采集函数（用于最小化）
            def neg_acq(x):
                return -self.acquisition(x.reshape(1, -1), self.gp, y_best)[0]

            # 边界约束
            bounds_list = [self.bounds] * self.dim

            # 优化
            result = minimize(
                neg_acq,
                x0,
                method='L-BFGS-B',
                bounds=bounds_list
            )

            if -result.fun > best_acq:
                best_acq = -result.fun
                best_x = result.x

        return best_x if best_x is not None else self._random_sample()

    def get_best(self) -> Tuple[np.ndarray, float]:
        """获取当前最佳观测"""
        if len(self.y_observed) == 0:
            return self._random_sample(), -np.inf

        best_idx = np.argmax(self.y_observed)
        return np.array(self.X_observed[best_idx]), self.y_observed[best_idx]

    def get_observations(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取所有观测"""
        if len(self.X_observed) == 0:
            return np.array([]).reshape(0, self.dim), np.array([])
        return np.array(self.X_observed), np.array(self.y_observed)

    def reset(self):
        """重置优化器"""
        self.X_observed = []
        self.y_observed = []
        self.gp = GaussianProcessRegressor(self.kernel, self.gp.noise_variance)


# ============================================================================
# PyTorch 版本 (用于梯度优化)
# ============================================================================

class GaussianProcessRegressorTorch(nn.Module):
    """
    PyTorch 版本的高斯过程回归

    支持端到端梯度计算，可用于神经网络集成。
    """

    def __init__(
        self,
        dim: int,
        lengthscale: float = 1.0,
        variance: float = 1.0,
        noise_variance: float = 1e-4
    ):
        super().__init__()
        self.dim = dim

        # 可学习参数
        self.log_lengthscale = nn.Parameter(torch.log(torch.tensor(lengthscale)))
        self.log_variance = nn.Parameter(torch.log(torch.tensor(variance)))
        self.log_noise = nn.Parameter(torch.log(torch.tensor(noise_variance)))

        # 训练数据缓存
        self.register_buffer('X_train', None)
        self.register_buffer('y_train', None)
        self.register_buffer('K_inv', None)
        self.register_buffer('alpha', None)

    @property
    def lengthscale(self) -> Tensor:
        return torch.exp(self.log_lengthscale)

    @property
    def variance(self) -> Tensor:
        return torch.exp(self.log_variance)

    @property
    def noise_variance(self) -> Tensor:
        return torch.exp(self.log_noise)

    def rbf_kernel(self, X1: Tensor, X2: Tensor) -> Tensor:
        """RBF 核"""
        # 计算平方距离
        X1_sq = (X1 ** 2).sum(dim=-1, keepdim=True)
        X2_sq = (X2 ** 2).sum(dim=-1, keepdim=True)
        sq_dist = X1_sq + X2_sq.T - 2 * X1 @ X2.T

        # RBF
        K = self.variance * torch.exp(-0.5 * sq_dist / (self.lengthscale ** 2))

        return K

    def fit(self, X: Tensor, y: Tensor):
        """训练 GP"""
        self.X_train = X
        self.y_train = y

        # 计算核矩阵
        K = self.rbf_kernel(X, X)
        K = K + self.noise_variance * torch.eye(len(X), device=X.device)

        # Cholesky 分解
        L = torch.linalg.cholesky(K)
        self.K_inv = torch.cholesky_inverse(L)
        self.alpha = self.K_inv @ y

    def predict(self, X: Tensor, return_std: bool = True) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """预测"""
        if self.X_train is None:
            raise RuntimeError("GP not fitted")

        K_star = self.rbf_kernel(X, self.X_train)
        K_star_star = self.rbf_kernel(X, X)

        # 均值
        mu = K_star @ self.alpha

        if not return_std:
            return mu

        # 方差
        v = self.K_inv @ K_star.T
        cov = K_star_star - K_star @ v
        var = torch.diag(cov) + self.noise_variance
        std = torch.sqrt(torch.clamp(var, min=1e-10))

        return mu, std

"""
梯度优化求解器

实现基于梯度的优化算法，用于光子学器件的逆向设计。
支持多种优化器、约束处理和自适应策略。

主要组件:
1. 梯度下降变体 (SGD, Adam, L-BFGS)
2. 投影梯度下降 (用于约束优化)
3. 伴随方法 (高效梯度计算)
4. 自适应步长策略

参考文献:
- Jensen & Sigmund (2011). "Topology optimization"
- Molesky et al. (2018). "Inverse design in nanophotonics"
"""

from typing import Dict, Optional, Tuple, List, Union, Callable, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Optimizer

import numpy as np
from scipy.optimize import minimize, OptimizeResult


# ============================================================================
# 配置类
# ============================================================================

@dataclass
class OptimizerConfig:
    """优化器配置基类"""
    max_iterations: int = 1000
    tolerance: float = 1e-6
    verbose: bool = True
    
    # 收敛判断
    patience: int = 50              # 早停耐心值
    min_delta: float = 1e-8         # 最小改进阈值


@dataclass
class GradientDescentConfig(OptimizerConfig):
    """梯度下降配置"""
    name: str = "gradient_descent"
    learning_rate: float = 0.01
    momentum: float = 0.9
    nesterov: bool = False
    weight_decay: float = 0.0


@dataclass
class AdamConfig(OptimizerConfig):
    """Adam配置"""
    name: str = "adam"
    learning_rate: float = 0.001
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    weight_decay: float = 0.0
    amsgrad: bool = False


@dataclass
class LBFGSConfig(OptimizerConfig):
    """L-BFGS配置"""
    name: str = "lbfgs"
    learning_rate: float = 1.0
    max_history_size: int = 10
    max_line_search_steps: int = 20
    line_search_tol: float = 1e-7


@dataclass
class TopologyOptConfig(OptimizerConfig):
    """拓扑优化配置"""
    name: str = "topology_opt"
    learning_rate: float = 0.01
    
    # 密度过滤
    filter_radius: float = 2.0          # 过滤半径（像素）
    filter_type: str = "helmholtz"      # 'helmholtz', 'density', 'sensitivity'
    
    # 投影参数
    threshold: float = 0.5              # 投影阈值
    sharpness: float = 10.0             # 投影锐度 (β)
    
    # 延续策略
    continuation_steps: int = 10        # 延续步数
    threshold_increment: float = 0.05   # 阈值增量


# ============================================================================
# 基础优化器类
# ============================================================================

class GradientBasedOptimizer(ABC):
    """梯度优化器基类"""
    
    def __init__(self, config: Optional[OptimizerConfig] = None):
        self.config = config or OptimizerConfig()
        self.iteration = 0
        self.history: Dict[str, List[float]] = {
            'loss': [],
            'gradient_norm': [],
            'step_size': [],
        }
        self.best_loss = float('inf')
        self.best_params = None
        self.patience_counter = 0
    
    @abstractmethod
    def step(
        self,
        params: Tensor,
        grad: Tensor,
        **kwargs
    ) -> Tensor:
        """
        执行一步优化
        
        Args:
            params: 当前参数
            grad: 梯度
            
        Returns:
            更新后的参数
        """
        pass
    
    def reset(self):
        """重置优化器状态"""
        self.iteration = 0
        self.history = {'loss': [], 'gradient_norm': [], 'step_size': []}
        self.best_loss = float('inf')
        self.best_params = None
        self.patience_counter = 0
    
    def check_convergence(self, loss: float) -> bool:
        """检查收敛"""
        if loss < self.best_loss - self.config.min_delta:
            self.best_loss = loss
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            return self.patience_counter >= self.config.patience


# ============================================================================
# 梯度下降
# ============================================================================

class GradientDescent(GradientBasedOptimizer):
    """带动量的梯度下降"""
    
    def __init__(self, config: Optional[GradientDescentConfig] = None):
        super().__init__(config or GradientDescentConfig())
        self.config: GradientDescentConfig = self.config
        self.velocity = None
    
    def step(
        self,
        params: Tensor,
        grad: Tensor,
        **kwargs
    ) -> Tensor:
        # 初始化速度
        if self.velocity is None:
            self.velocity = torch.zeros_like(params)
        
        # 权重衰减
        if self.config.weight_decay > 0:
            grad = grad + self.config.weight_decay * params
        
        # Nesterov 动量
        if self.config.nesterov:
            grad = grad + self.config.momentum * self.velocity
        
        # 更新速度
        self.velocity = self.config.momentum * self.velocity - self.config.learning_rate * grad
        
        # 更新参数
        new_params = params + self.velocity
        
        self.iteration += 1
        return new_params


# ============================================================================
# Adam 优化器
# ============================================================================

class Adam(GradientBasedOptimizer):
    """Adam 优化器"""
    
    def __init__(self, config: Optional[AdamConfig] = None):
        super().__init__(config or AdamConfig())
        self.config: AdamConfig = self.config
        self.m = None  # 一阶矩估计
        self.v = None  # 二阶矩估计
        self.v_max = None  # AMSGrad 的最大二阶矩
    
    def step(
        self,
        params: Tensor,
        grad: Tensor,
        **kwargs
    ) -> Tensor:
        # 初始化矩估计
        if self.m is None:
            self.m = torch.zeros_like(params)
            self.v = torch.zeros_like(params)
            if self.config.amsgrad:
                self.v_max = torch.zeros_like(params)
        
        # 权重衰减
        if self.config.weight_decay > 0:
            grad = grad + self.config.weight_decay * params
        
        # 更新矩估计
        self.m = self.config.beta1 * self.m + (1 - self.config.beta1) * grad
        self.v = self.config.beta2 * self.v + (1 - self.config.beta2) * grad ** 2
        
        # 偏差校正
        m_hat = self.m / (1 - self.config.beta1 ** (self.iteration + 1))
        v_hat = self.v / (1 - self.config.beta2 ** (self.iteration + 1))
        
        # AMSGrad
        if self.config.amsgrad:
            self.v_max = torch.maximum(self.v_max, v_hat)
            v_hat = self.v_max
        
        # 更新参数
        new_params = params - self.config.learning_rate * m_hat / (torch.sqrt(v_hat) + self.config.epsilon)
        
        self.iteration += 1
        return new_params


# ============================================================================
# L-BFGS 优化器
# ============================================================================

class LBFGS(GradientBasedOptimizer):
    """L-BFGS 优化器（有限内存拟牛顿法）"""
    
    def __init__(self, config: Optional[LBFGSConfig] = None):
        super().__init__(config or LBFGSConfig())
        self.config: LBFGSConfig = self.config
        
        # 历史存储
        self.s_history = []  # 参数差
        self.y_history = []  # 梯度差
        self.rho_history = []  # 缩放因子
    
    def step(
        self,
        params: Tensor,
        grad: Tensor,
        loss_fn: Optional[Callable] = None,
        **kwargs
    ) -> Tensor:
        """
        L-BFGS 更新步骤
        
        Args:
            params: 当前参数
            grad: 梯度
            loss_fn: 损失函数（用于线搜索）
        """
        # 计算 L-BFGS 方向
        direction = self._compute_direction(grad)
        
        # 线搜索
        step_size = self._line_search(params, grad, direction, loss_fn)
        
        # 更新参数
        new_params = params + step_size * direction
        
        # 更新历史
        if self.previous_params is not None:
            s = new_params - self.previous_params
            y = grad - self.previous_grad
            
            if torch.dot(s, y) > 1e-10:  # 曲率条件
                self.s_history.append(s)
                self.y_history.append(y)
                self.rho_history.append(1.0 / torch.dot(y, s))
                
                # 限制历史大小
                if len(self.s_history) > self.config.max_history_size:
                    self.s_history.pop(0)
                    self.y_history.pop(0)
                    self.rho_history.pop(0)
        
        self.previous_params = params.clone()
        self.previous_grad = grad.clone()
        
        self.iteration += 1
        return new_params
    
    def _compute_direction(self, grad: Tensor) -> Tensor:
        """计算 L-BFGS 搜索方向"""
        q = grad.clone()
        
        alpha_list = []
        
        # 两循环递推
        for s, y, rho in reversed(list(zip(self.s_history, self.y_history, self.rho_history))):
            alpha = rho * torch.dot(s, q)
            alpha_list.append(alpha)
            q = q - alpha * y
        
        # 初始 Hessian 近似
        if len(self.s_history) > 0:
            gamma = torch.dot(self.s_history[-1], self.y_history[-1]) / torch.dot(self.y_history[-1], self.y_history[-1])
        else:
            gamma = 1.0
        
        r = gamma * q
        
        # 第二个循环
        for (s, y, rho), alpha in zip(zip(self.s_history, self.y_history, self.rho_history), reversed(alpha_list)):
            beta = rho * torch.dot(y, r)
            r = r + (alpha - beta) * s
        
        return -r
    
    def _line_search(
        self,
        params: Tensor,
        grad: Tensor,
        direction: Tensor,
        loss_fn: Optional[Callable],
    ) -> float:
        """回溯线搜索"""
        if loss_fn is None:
            return self.config.learning_rate
        
        alpha = self.config.learning_rate
        c1 = 1e-4  # Armijo 条件参数
        
        f0 = loss_fn(params).item()
        grad_dot_dir = torch.dot(grad, direction).item()
        
        for _ in range(self.config.max_line_search_steps):
            new_params = params + alpha * direction
            f_new = loss_fn(new_params).item()
            
            # Armijo 条件
            if f_new <= f0 + c1 * alpha * grad_dot_dir:
                return alpha
            
            alpha *= 0.5
        
        return alpha
    
    def reset(self):
        super().reset()
        self.s_history = []
        self.y_history = []
        self.rho_history = []
        self.previous_params = None
        self.previous_grad = None


# ============================================================================
# 拓扑优化器
# ============================================================================

class TopologyOptimizer(GradientBasedOptimizer):
    """
    拓扑优化器
    
    使用伴随方法进行高效的梯度计算，
    结合密度过滤和投影实现清晰的拓扑结构。
    """
    
    def __init__(self, config: Optional[TopologyOptConfig] = None):
        super().__init__(config or TopologyOptConfig())
        self.config: TopologyOptConfig = self.config
        
        # 密度过滤
        self.filter_kernel = None
        
        # 延续参数
        self.current_threshold = 0.5
    
    def setup_filter(self, shape: Tuple[int, int], device: torch.device):
        """设置密度过滤核"""
        H, W = shape
        r = int(self.config.filter_radius)
        
        # 创建圆形核
        y, x = torch.meshgrid(
            torch.arange(-r, r + 1, dtype=torch.float32),
            torch.arange(-r, r + 1, dtype=torch.float32),
            indexing='ij'
        )
        
        # 距离权重
        dist = torch.sqrt(x ** 2 + y ** 2)
        kernel = (r - dist).clamp(min=0)
        kernel = kernel / kernel.sum()
        
        self.filter_kernel = kernel.unsqueeze(0).unsqueeze(0).to(device)
    
    def density_filter(self, rho: Tensor) -> Tensor:
        """
        密度过滤
        
        平滑密度场以避免棋盘格模式
        """
        if self.filter_kernel is None:
            self.setup_filter(rho.shape[-2:], rho.device)
        
        # 应用卷积过滤
        rho_filtered = F.conv2d(
            rho.unsqueeze(0).unsqueeze(0),
            self.filter_kernel,
            padding=int(self.config.filter_radius)
        )
        
        return rho_filtered.squeeze()
    
    def projection(self, rho: Tensor, threshold: Optional[float] = None) -> Tensor:
        """
        Heaviside 投影
        
        将密度场投影到接近二元分布
        
        η_β(ρ) = (tanh(β·η) + tanh(β·(ρ-η))) / (tanh(β·η) + tanh(β·(1-η)))
        """
        if threshold is None:
            threshold = self.current_threshold
        
        beta = self.config.sharpness
        eta = threshold
        
        numerator = torch.tanh(beta * eta) + torch.tanh(beta * (rho - eta))
        denominator = torch.tanh(beta * eta) + torch.tanh(beta * (1 - eta))
        
        return numerator / denominator
    
    def step(
        self,
        params: Tensor,
        grad: Tensor,
        **kwargs
    ) -> Tensor:
        """拓扑优化更新步骤"""
        # 密度过滤
        rho_filtered = self.density_filter(params)
        
        # 计算过滤后的梯度（链式法则）
        # dJ/dρ_filtered 需要通过过滤的反向传播
        grad_filtered = self._filter_gradient(grad)
        
        # 使用 Adam 更新
        if not hasattr(self, '_adam'):
            self._adam = Adam(AdamConfig(learning_rate=self.config.learning_rate))
        
        # 在过滤空间更新
        new_rho_filtered = self._adam.step(rho_filtered, grad_filtered)
        
        # 投影回 [0, 1]
        new_params = new_rho_filtered.clamp(0, 1)
        
        # 延续策略：逐渐增加阈值
        if self.iteration > 0 and self.iteration % (self.config.max_iterations // self.config.continuation_steps) == 0:
            self.current_threshold = min(0.5 + self.config.threshold_increment * 
                                         (self.iteration // (self.config.max_iterations // self.config.continuation_steps)), 0.99)
        
        self.iteration += 1
        return new_params
    
    def _filter_gradient(self, grad: Tensor) -> Tensor:
        """通过密度过滤反向传播梯度"""
        # 密度过滤的伴随：由于是卷积，反向传播是转置卷积
        grad_filtered = F.conv_transpose2d(
            grad.unsqueeze(0).unsqueeze(0),
            self.filter_kernel,
            padding=int(self.config.filter_radius)
        )
        return grad_filtered.squeeze()
    
    def get_discretized_design(self, params: Tensor) -> Tensor:
        """获取离散化设计（接近二元）"""
        rho_proj = self.projection(params)
        return (rho_proj > 0.5).float()


# ============================================================================
# 投影梯度下降 (约束优化)
# ============================================================================

class ProjectedGradientDescent(GradientBasedOptimizer):
    """
    投影梯度下降
    
    用于约束优化，在每步后投影到可行域。
    """
    
    def __init__(
        self,
        config: Optional[AdamConfig] = None,
        bounds: Optional[Tuple[float, float]] = (0.0, 1.0),
        constraints: Optional[List[Callable]] = None,
    ):
        super().__init__(config or AdamConfig())
        self.config: AdamConfig = self.config
        self.bounds = bounds
        self.constraints = constraints or []
        self._adam = Adam(self.config)
    
    def project(self, params: Tensor) -> Tensor:
        """投影到可行域"""
        # Box 约束
        if self.bounds is not None:
            params = params.clamp(self.bounds[0], self.bounds[1])
        
        # 额外约束（迭代投影）
        for constraint_fn in self.constraints:
            params = constraint_fn(params)
        
        return params
    
    def step(
        self,
        params: Tensor,
        grad: Tensor,
        **kwargs
    ) -> Tensor:
        # 梯度更新
        new_params = self._adam.step(params, grad)
        
        # 投影到可行域
        new_params = self.project(new_params)
        
        self.iteration += 1
        return new_params


# ============================================================================
# 伴随方法求解器
# ============================================================================

class AdjointSolver:
    """
    伴随方法求解器
    
    高效计算目标函数对大量设计变量的梯度。
    
    原理：
    对于目标 J(ρ) 和约束向量 c(ρ) = 0，
    伴随方程：∂L/∂u = 0（L是拉格朗日函数）
    梯度：dJ/dρ = ∂J/∂ρ + λ^T ∂c/∂ρ
    """
    
    def __init__(
        self,
        forward_solver: Callable,
        objective_fn: Callable,
        constraint_fns: Optional[List[Callable]] = None,
    ):
        """
        Args:
            forward_solver: 正向问题求解器
            objective_fn: 目标函数
            constraint_fns: 约束函数列表
        """
        self.forward_solver = forward_solver
        self.objective_fn = objective_fn
        self.constraint_fns = constraint_fns or []
    
    def solve_adjoint(
        self,
        design: Tensor,
        state: Tensor,
        objective_grad: Tensor,
    ) -> Tensor:
        """
        求解伴随方程
        
        Args:
            design: 设计变量
            state: 正向求解的状态变量
            objective_grad: 目标对状态的梯度
            
        Returns:
            伴随变量（拉格朗日乘子）
        """
        # 构建伴随系统矩阵
        # 这需要知道正向问题的具体形式
        # 这里提供通用框架
        
        # 对于线性系统 A(design) @ state = b
        # 伴随方程: A^T @ λ = -∂J/∂state
        
        # 使用自动微分求解（通用方法）
        state_adjoint = torch.autograd.grad(
            outputs=state.sum(),
            inputs=design,
            create_graph=True,
            allow_unused=True
        )[0]
        
        return state_adjoint
    
    def compute_gradient(
        self,
        design: Tensor,
        **kwargs
    ) -> Tuple[float, Tensor]:
        """
        计算目标值和梯度
        
        Returns:
            objective: 目标函数值
            gradient: 对设计变量的梯度
        """
        # 正向求解
        state = self.forward_solver(design)
        
        # 目标函数
        objective = self.objective_fn(state, design)
        
        # 使用自动微分计算梯度
        if isinstance(objective, Tensor):
            gradient = torch.autograd.grad(
                outputs=objective,
                inputs=design,
                create_graph=False,
                allow_unused=True
            )[0]
            objective_val = objective.item()
        else:
            objective_val = objective
            gradient = torch.zeros_like(design)
        
        return objective_val, gradient


# ============================================================================
# 优化器工厂
# ============================================================================

def create_optimizer(
    name: str = "adam",
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> GradientBasedOptimizer:
    """
    创建优化器
    
    Args:
        name: 优化器名称
        config: 配置字典
        **kwargs: 其他参数
        
    Returns:
        优化器实例
    """
    optimizers = {
        'sgd': GradientDescent,
        'gradient_descent': GradientDescent,
        'adam': Adam,
        'lbfgs': LBFGS,
        'topology': TopologyOptimizer,
        'projected': ProjectedGradientDescent,
    }
    
    if name not in optimizers:
        raise ValueError(f"Unknown optimizer: {name}, available: {list(optimizers.keys())}")
    
    return optimizers[name](config=config if config else None, **kwargs)


# ============================================================================
# 优化运行器
# ============================================================================

class OptimizationRunner:
    """
    优化运行器
    
    封装优化过程，提供日志、检查点等功能。
    """
    
    def __init__(
        self,
        optimizer: GradientBasedOptimizer,
        objective_fn: Callable,
        constraint_fns: Optional[List[Callable]] = None,
        verbose: bool = True,
    ):
        self.optimizer = optimizer
        self.objective_fn = objective_fn
        self.constraint_fns = constraint_fns or []
        self.verbose = verbose
    
    def run(
        self,
        initial_params: Tensor,
        max_iterations: Optional[int] = None,
        callback: Optional[Callable] = None,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        """
        运行优化
        
        Args:
            initial_params: 初始参数
            max_iterations: 最大迭代次数
            callback: 回调函数
            
        Returns:
            optimized_params: 优化后的参数
            info: 优化信息字典
        """
        params = initial_params.clone()
        max_iter = max_iterations or self.optimizer.config.max_iterations
        
        for i in range(max_iter):
            params.requires_grad_(True)
            
            # 计算目标
            objective = self.objective_fn(params)
            
            # 计算梯度
            grad = torch.autograd.grad(objective, params)[0]
            
            # 约束梯度（如果需要）
            for constraint_fn in self.constraint_fns:
                constraint_val = constraint_fn(params)
                constraint_grad = torch.autograd.grad(constraint_val.sum(), params)[0]
                grad = grad + constraint_grad
            
            # 更新参数
            with torch.no_grad():
                params = self.optimizer.step(params.detach(), grad)
            
            # 记录历史
            self.optimizer.history['loss'].append(objective.item())
            self.optimizer.history['gradient_norm'].append(grad.norm().item())
            
            # 回调
            if callback is not None:
                callback(params, objective, i)
            
            # 打印进度
            if self.verbose and i % 10 == 0:
                print(f"Iteration {i}: loss = {objective.item():.6e}, grad_norm = {grad.norm().item():.6e}")
            
            # 检查收敛
            if self.optimizer.check_convergence(objective.item()):
                if self.verbose:
                    print(f"Converged at iteration {i}")
                break
        
        info = {
            'iterations': i + 1,
            'final_loss': objective.item(),
            'history': self.optimizer.history,
        }
        
        return params, info

"""
优化模块

提供光子学逆向设计中的优化求解器和约束条件。
"""

from . import constraints
from .constraints import (
    # 色散约束
    DispersionConstraint,
    DispersionCalculator,
    MultiWavelengthSimulator,
    
    # 制造公差约束
    RobustnessConstraint,
    DesignRuleCheck,
    EdgeRoughnessModel,
    
    # 热效应约束
    ThermalConstraint,
    ThermoOpticEffect,
    HeatConductionSolver,
)

# 求解器
from .solvers.bayesian import (
    BayesianOptimizer,
    GaussianProcessRegressor,
    ExpectedImprovement,
    UpperConfidenceBound,
)

__all__ = [
    'constraints',
    
    # 色散约束
    'DispersionConstraint',
    'DispersionCalculator',
    'MultiWavelengthSimulator',
    
    # 制造公差约束
    'RobustnessConstraint',
    'DesignRuleCheck',
    'EdgeRoughnessModel',
    
    # 热效应约束
    'ThermalConstraint',
    'ThermoOpticEffect',
    'HeatConductionSolver',
    
    # 贝叶斯优化器
    'BayesianOptimizer',
    'GaussianProcessRegressor',
    'ExpectedImprovement',
    'UpperConfidenceBound',
]

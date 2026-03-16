"""
HI-Photonics 核心模块

提供计算图框架和各类节点实现。
"""

# 核心基类
from .node import Node
from .graph import Graph

# 类型定义
from .utils.typing import TensorLike, Params

# 节点实现
from .nodes.parameterization import ParameterizationNode
from .nodes.simulation import SimulationNode
from .nodes.objective import ObjectiveNode

# 滤波器节点
from .nodes.filter import (
    FilterNode,
    GaussianFilterNode,
    SobelFilterNode,
    DilationFilterNode,
    ErosionFilterNode,
    BilateralFilterNode,
    MorphologicalOpeningNode,
    MorphologicalClosingNode,
)

# 投影节点
from .nodes.projection import (
    ProjectionNode,
    SigmoidProjectionNode,
    HeavisideProjectionNode,
    TanhProjectionNode,
    SoftmaxProjectionNode,
    ThresholdProjectionNode,
    DensityProjectionNode,
    ErosionProjectionNode,
    DilationProjectionNode,
    CombinedProjectionNode,
)

# 约束节点
from .nodes.constraint import (
    ConstraintNode,
    VolumeConstraintNode,
    BinaryConstraintNode,
    MinimumFeatureSizeNode,
    CurvatureConstraintNode,
    ConnectivityConstraintNode,
    SymmetryConstraintNode,
    PerimeterConstraintNode,
    ManufacturingConstraintNode,
    GradientPenaltyNode,
)

# 组合节点
from .nodes.composite import (
    CompositeNode,
    PipelineNode,
    MergeNode,
    SplitNode,
    SelectorNode,
    LambdaNode,
    CachedNode,
    ResidualNode,
    BranchNode,
    WeightedSumNode,
    DesignPipelineNode,
)

__all__ = [
    # 核心基类
    'Node',
    'Graph',
    
    # 类型定义
    'TensorLike',
    'Params',
    
    # 基础节点
    'ParameterizationNode',
    'SimulationNode',
    'ObjectiveNode',
    
    # 滤波器节点
    'FilterNode',
    'GaussianFilterNode',
    'SobelFilterNode',
    'DilationFilterNode',
    'ErosionFilterNode',
    'BilateralFilterNode',
    'MorphologicalOpeningNode',
    'MorphologicalClosingNode',
    
    # 投影节点
    'ProjectionNode',
    'SigmoidProjectionNode',
    'HeavisideProjectionNode',
    'TanhProjectionNode',
    'SoftmaxProjectionNode',
    'ThresholdProjectionNode',
    'DensityProjectionNode',
    'ErosionProjectionNode',
    'DilationProjectionNode',
    'CombinedProjectionNode',
    
    # 约束节点
    'ConstraintNode',
    'VolumeConstraintNode',
    'BinaryConstraintNode',
    'MinimumFeatureSizeNode',
    'CurvatureConstraintNode',
    'ConnectivityConstraintNode',
    'SymmetryConstraintNode',
    'PerimeterConstraintNode',
    'ManufacturingConstraintNode',
    'GradientPenaltyNode',
    
    # 组合节点
    'CompositeNode',
    'PipelineNode',
    'MergeNode',
    'SplitNode',
    'SelectorNode',
    'LambdaNode',
    'CachedNode',
    'ResidualNode',
    'BranchNode',
    'WeightedSumNode',
    'DesignPipelineNode',
]

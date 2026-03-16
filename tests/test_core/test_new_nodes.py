"""
测试新实现的节点类型：滤波器、投影、约束、组合节点
"""
import torch
import sys
sys.path.append('.')

from core.node import Node
from core.nodes.parameterization import ParameterizationNode
from core.nodes.filter import (
    FilterNode, GaussianFilterNode, SobelFilterNode,
    DilationFilterNode, ErosionFilterNode,
    MorphologicalOpeningNode, MorphologicalClosingNode,
    BilateralFilterNode
)
from core.nodes.projection import (
    ProjectionNode, SigmoidProjectionNode, HeavisideProjectionNode,
    TanhProjectionNode, DensityProjectionNode,
    CombinedProjectionNode
)
from core.nodes.constraint import (
    ConstraintNode, VolumeConstraintNode, BinaryConstraintNode,
    MinimumFeatureSizeNode, CurvatureConstraintNode,
    SymmetryConstraintNode, GradientPenaltyNode
)
from core.nodes.composite import (
    PipelineNode, MergeNode, SplitNode, LambdaNode,
    WeightedSumNode, DesignPipelineNode
)
from core.graph import Graph


def test_filter_nodes():
    """测试滤波器节点"""
    print("\n=== 测试滤波器节点 ===")
    
    # 创建一个简单的测试图像
    x = torch.randn(1, 1, 32, 32)
    param_node = ParameterizationNode('input', x.squeeze(0).squeeze(0), requires_grad=False)
    
    # 测试高斯滤波
    gauss = GaussianFilterNode('gauss', param_node, kernel_size=5, sigma=1.0)
    out = gauss.forward()
    assert out.shape == (32, 32), f"高斯滤波输出形状错误: {out.shape}"
    print("高斯滤波: 通过")
    
    # 测试 Sobel 边缘检测
    sobel = SobelFilterNode('sobel', param_node)
    out = sobel.forward()
    assert out.shape == (32, 32), f"Sobel 输出形状错误: {out.shape}"
    print("Sobel 边缘检测: 通过")
    
    # 测试膨胀
    dilation = DilationFilterNode('dilation', param_node, kernel_size=3)
    out = dilation.forward()
    assert out.shape == (32, 32), f"膨胀输出形状错误: {out.shape}"
    print("膨胀滤波: 通过")
    
    # 测试腐蚀
    erosion = ErosionFilterNode('erosion', param_node, kernel_size=3)
    out = erosion.forward()
    assert out.shape == (32, 32), f"腐蚀输出形状错误: {out.shape}"
    print("腐蚀滤波: 通过")
    
    # 测试形态学开运算
    opening = MorphologicalOpeningNode('opening', param_node, kernel_size=3)
    out = opening.forward()
    assert out.shape == (32, 32), f"开运算输出形状错误: {out.shape}"
    print("形态学开运算: 通过")
    
    # 测试形态学闭运算
    closing = MorphologicalClosingNode('closing', param_node, kernel_size=3)
    out = closing.forward()
    assert out.shape == (32, 32), f"闭运算输出形状错误: {out.shape}"
    print("形态学闭运算: 通过")
    
    print("滤波器节点测试: 全部通过")


def test_projection_nodes():
    """测试投影节点"""
    print("\n=== 测试投影节点 ===")
    
    x = torch.linspace(-2, 2, 100)
    param_node = ParameterizationNode('input', x, requires_grad=True)
    
    # 测试 Sigmoid 投影
    sigmoid = SigmoidProjectionNode('sigmoid', param_node, threshold=0.0, sharpness=10.0)
    out = sigmoid.forward()
    assert out.min() >= 0 and out.max() <= 1, "Sigmoid 输出应在 [0, 1] 范围内"
    print("Sigmoid 投影: 通过")
    
    # 测试 Heaviside 投影
    heaviside = HeavisideProjectionNode('heaviside', param_node, threshold=0.0, sharpness=10.0)
    out = heaviside.forward()
    assert out.shape == x.shape, "Heaviside 输出形状应与输入相同"
    print("Heaviside 投影: 通过")
    
    # 测试 Tanh 投影
    tanh = TanhProjectionNode('tanh', param_node, threshold=0.0, sharpness=10.0)
    out = tanh.forward()
    assert out.min() >= 0 and out.max() <= 1, "Tanh 投影输出应在 [0, 1] 范围内"
    print("Tanh 投影: 通过")
    
    # 测试密度投影 (SIMP)
    density = DensityProjectionNode('density', param_node, penalty=3.0)
    out = density.forward()
    # 输入范围是 [-2, 2]，截断后是 [0, 1]
    assert out.min() >= 0, "密度投影输出应 >= 0"
    print("密度投影 (SIMP): 通过")
    
    # 测试组合投影 (需要 2D 输入)
    x_2d = torch.linspace(-2, 2, 100).view(10, 10)
    param_2d = ParameterizationNode('input_2d', x_2d, requires_grad=True)
    combined = CombinedProjectionNode(
        'combined', param_2d,
        filter_radius=1, threshold=0.0, sharpness=10.0
    )
    out = combined.forward()
    assert out.shape == x_2d.shape, "组合投影输出形状应与输入相同"
    print("组合投影: 通过")
    
    print("投影节点测试: 全部通过")


def test_constraint_nodes():
    """测试约束节点"""
    print("\n=== 测试约束节点 ===")
    
    # 创建一个 2D 设计变量
    x = torch.rand(32, 32) * 0.5 + 0.25  # 在 [0.25, 0.75] 范围内
    param_node = ParameterizationNode('input', x, requires_grad=True)
    
    # 测试体积约束
    volume = VolumeConstraintNode('volume', param_node, target_fraction=0.5)
    out = volume.forward()
    assert out.dim() == 0, "体积约束应返回标量"
    print(f"体积约束值: {out.item():.4f}")
    
    # 测试二值化约束
    binary = BinaryConstraintNode('binary', param_node, weight=1.0)
    out = binary.forward()
    assert out.dim() == 0, "二值化约束应返回标量"
    print(f"二值化约束值: {out.item():.4f}")
    
    # 测试最小特征尺寸约束
    min_feature = MinimumFeatureSizeNode(
        'min_feature', param_node,
        min_feature_size=3.0, resolution=1.0
    )
    out = min_feature.forward()
    assert out.dim() == 0, "最小特征尺寸约束应返回标量"
    print(f"最小特征尺寸约束值: {out.item():.4f}")
    
    # 测试曲率约束
    curvature = CurvatureConstraintNode('curvature', param_node, max_curvature=0.5)
    out = curvature.forward()
    assert out.dim() == 0, "曲率约束应返回标量"
    print(f"曲率约束值: {out.item():.4f}")
    
    # 测试对称性约束
    sym_h = SymmetryConstraintNode('sym_h', param_node, symmetry_type='horizontal')
    out = sym_h.forward()
    assert out.dim() == 0, "对称性约束应返回标量"
    print(f"水平对称约束值: {out.item():.4f}")
    
    sym_v = SymmetryConstraintNode('sym_v', param_node, symmetry_type='vertical')
    out = sym_v.forward()
    print(f"垂直对称约束值: {out.item():.4f}")
    
    # 测试梯度惩罚
    tv = GradientPenaltyNode('tv', param_node, penalty_type='total_variation')
    out = tv.forward()
    assert out.dim() == 0, "梯度惩罚应返回标量"
    print(f"TV 梯度惩罚值: {out.item():.4f}")
    
    print("约束节点测试: 全部通过")


def test_composite_nodes():
    """测试组合节点"""
    print("\n=== 测试组合节点 ===")
    
    # 创建输入节点
    x1 = torch.randn(16, 16)
    x2 = torch.randn(16, 16)
    node1 = ParameterizationNode('node1', x1, requires_grad=False)
    node2 = ParameterizationNode('node2', x2, requires_grad=False)
    
    # 测试 MergeNode (concat)
    merge = MergeNode('merge', [node1, node2], merge_type='concat', dim=0)
    out = merge.forward()
    assert out.shape == (32, 16), f"Merge (concat) 输出形状错误: {out.shape}"
    print("MergeNode (concat): 通过")
    
    # 测试 MergeNode (sum)
    merge_sum = MergeNode('merge_sum', [node1, node2], merge_type='sum')
    out = merge_sum.forward()
    assert out.shape == (16, 16), f"Merge (sum) 输出形状错误: {out.shape}"
    print("MergeNode (sum): 通过")
    
    # 测试 SplitNode
    x_concat = torch.randn(32, 16)
    concat_node = ParameterizationNode('concat', x_concat, requires_grad=False)
    split = SplitNode('split', concat_node, split_type='chunk', num_chunks=2, dim=0)
    out = split.forward()
    assert len(out) == 2, "Split 应返回 2 个张量"
    assert out[0].shape == (16, 16), f"Split 输出形状错误: {out[0].shape}"
    print("SplitNode: 通过")
    
    # 测试 LambdaNode
    lambda_node = LambdaNode('lambda', node1, fn=lambda x: x * 2 + 1)
    out = lambda_node.forward()
    expected = x1 * 2 + 1
    assert torch.allclose(out, expected), "LambdaNode 输出不正确"
    print("LambdaNode: 通过")
    
    # 测试 WeightedSumNode
    weighted = WeightedSumNode('weighted', [node1, node2], weights=[0.7, 0.3])
    out = weighted.forward()
    assert out.shape == (16, 16), f"WeightedSum 输出形状错误: {out.shape}"
    print("WeightedSumNode: 通过")
    
    print("组合节点测试: 全部通过")


def test_design_pipeline():
    """测试设计管道节点"""
    print("\n=== 测试设计管道节点 ===")
    
    # 创建设计变量
    x = torch.rand(32, 32)
    param = ParameterizationNode('design', x, requires_grad=True)
    
    # 创建设计管道
    pipeline = DesignPipelineNode(
        'pipeline',
        param,
        filter_type='gaussian',
        filter_radius=2,
        projection_type='sigmoid',
        threshold=0.5,
        sharpness=10.0
    )
    
    out = pipeline.forward()
    assert out.shape == (32, 32), f"设计管道输出形状错误: {out.shape}"
    assert out.min() >= 0 and out.max() <= 1, "投影后输出应在 [0, 1] 范围内"
    print(f"设计管道输出范围: [{out.min().item():.4f}, {out.max().item():.4f}]")
    
    # 测试梯度传播
    out.sum().backward()
    assert param.value.grad is not None, "梯度应该传播到参数节点"
    print(f"梯度已传播: param.value.grad.sum() = {param.value.grad.sum().item():.4f}")
    
    print("设计管道节点测试: 通过")


def test_gradient_flow():
    """测试梯度流"""
    print("\n=== 测试梯度流 ===")
    
    # 创建参数
    x = torch.rand(16, 16, requires_grad=True)
    param = ParameterizationNode('param', x)
    
    # 构建处理管道
    gauss = GaussianFilterNode('gauss', param, kernel_size=3)
    proj = SigmoidProjectionNode('proj', gauss, threshold=0.5, sharpness=5.0)
    constraint = VolumeConstraintNode('constraint', proj, target_fraction=0.5)
    
    # 前向传播
    loss = constraint.forward()
    print(f"约束损失值: {loss.item():.4f}")
    
    # 反向传播
    loss.backward()
    
    # 检查梯度
    assert param.value.grad is not None, "梯度应该传播到参数节点"
    print(f"参数梯度范数: {param.value.grad.norm().item():.4f}")
    
    print("梯度流测试: 通过")


def test_full_optimization():
    """测试完整优化流程"""
    print("\n=== 测试完整优化流程 ===")
    
    # 创建设计变量
    x = torch.rand(32, 32) * 0.5 + 0.25
    param = ParameterizationNode('design', x, requires_grad=True)
    
    # 构建处理管道
    pipeline = DesignPipelineNode(
        'pipeline', param,
        filter_type='gaussian',
        filter_radius=2,
        projection_type='sigmoid',
        threshold=0.5,
        sharpness=10.0
    )
    
    # 定义目标函数: 最小化与目标图案的差异
    target = torch.zeros(32, 32)
    target[8:24, 8:24] = 1.0  # 中间正方形
    
    def loss_fn(design):
        return ((design - target) ** 2).mean()
    
    objective = LambdaNode('objective', pipeline, loss_fn)
    
    # 添加约束
    volume_constraint = VolumeConstraintNode('volume', pipeline, target_fraction=0.25)
    
    # 优化
    optimizer = torch.optim.Adam([param.value], lr=0.1)
    
    print("开始优化...")
    for i in range(50):
        optimizer.zero_grad()
        
        # 前向传播
        loss = objective.forward()
        constraint_val = volume_constraint.forward()
        
        # 总损失
        total_loss = loss + 0.1 * constraint_val
        
        # 反向传播
        total_loss.backward()
        
        optimizer.step()
        
        # 清除缓存
        pipeline._projection_node.clear_cache()
        if pipeline._filter_node:
            pipeline._filter_node.clear_cache()
        param.clear_cache()
        
        if i % 10 == 0:
            print(f"Step {i}: loss={loss.item():.4f}, constraint={constraint_val.item():.4f}")
    
    final_design = pipeline.forward()
    print(f"最终设计均值: {final_design.mean().item():.4f}")
    print("完整优化流程测试: 通过")


if __name__ == "__main__":
    test_filter_nodes()
    test_projection_nodes()
    test_constraint_nodes()
    test_composite_nodes()
    test_design_pipeline()
    test_gradient_flow()
    test_full_optimization()
    
    print("\n" + "=" * 50)
    print("所有测试通过!")
    print("=" * 50)

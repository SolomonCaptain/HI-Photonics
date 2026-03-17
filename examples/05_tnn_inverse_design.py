"""
TNN（串联神经网络）逆向设计示例

本示例展示如何使用 TNN 进行光子学器件的逆向设计：
1. 创建合成数据集
2. 训练前向网络（代理模型）
3. 训练逆向网络
4. 使用 TNN 进行逆向设计
5. 集成到计算图框架
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# 导入 TNN 相关模块
from models.inverse.tnn import (
    TandemNetwork,
    TandemNetworkConfig,
    ForwardNetworkConfig,
    InverseNetworkConfig,
    create_tnn_for_challenge
)
from models.training.losses import TandemLoss, PerformanceLoss
from models.training.metrics import MetricsCollection, R2Score, MAE, MSE
from models.training.callbacks import EarlyStopping, TrainingLogger, ModelCheckpoint

# 导入数据加载器
from data.loaders.pipeline import (
    PhotonicsDataset,
    SyntheticDataset,
    create_dataloaders,
    DataAugmentation
)

# 导入计算图模块
from core.node import Node
from core.graph import Graph
from core.nodes.parameterization import ParameterizationNode
from core.nodes.neural_network import TNNNode, TargetPerformanceNode


def create_synthetic_training_data(
    num_samples: int = 5000,
    design_shape: tuple = (100, 22),
    performance_dim: int = 3,
    noise_level: float = 0.05,
    seed: int = 42
):
    """
    创建合成训练数据
    
    模拟光栅耦合器的设计-性能关系。
    """
    print(f"Creating synthetic dataset with {num_samples} samples...")
    
    dataset = SyntheticDataset(
        num_samples=num_samples,
        design_shape=design_shape,
        performance_dim=performance_dim,
        noise_level=noise_level,
        seed=seed
    )
    
    return dataset


def train_tnn_example():
    """
    完整的 TNN 训练示例
    """
    print("=" * 60)
    print("TNN Training Example for Photonic Inverse Design")
    print("=" * 60)
    
    # 配置参数
    design_shape = (100, 22)  # 设计形状
    performance_dim = 3       # 性能指标维度
    batch_size = 32
    epochs = 50
    
    # 1. 创建数据集
    print("\n[Step 1] Creating dataset...")
    dataset = create_synthetic_training_data(
        num_samples=3000,
        design_shape=design_shape,
        performance_dim=performance_dim
    )
    
    # 划分数据集
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset,
        batch_size=batch_size,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )
    
    print(f"  Train: {len(train_loader.dataset)} samples")
    print(f"  Val: {len(val_loader.dataset)} samples")
    print(f"  Test: {len(test_loader.dataset)} samples")
    
    # 2. 创建 TNN 模型
    print("\n[Step 2] Creating TNN model...")
    
    forward_config = ForwardNetworkConfig(
        design_shape=design_shape,
        performance_dim=performance_dim,
        hidden_channels=[32, 64, 128],
        hidden_dims=[256, 128],
        dropout_rate=0.1
    )
    
    inverse_config = InverseNetworkConfig(
        design_shape=design_shape,
        performance_dim=performance_dim,
        hidden_dims=[256, 512],
        hidden_channels=[128, 64, 32],
        dropout_rate=0.1
    )
    
    tandem_config = TandemNetworkConfig(
        forward_config=forward_config,
        inverse_config=inverse_config,
        pretrain_forward=True,
        freeze_forward=True
    )
    
    tnn = TandemNetwork(tandem_config)
    print(f"  Model info: {tnn.get_model_info()}")
    
    # 3. 训练前向网络
    print("\n[Step 3] Training forward network...")
    forward_history = tnn.train_forward(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs // 2,
        lr=1e-3,
        patience=10
    )
    
    print(f"  Final train loss: {forward_history['train_loss'][-1]:.4f}")
    if forward_history['val_loss']:
        print(f"  Final val loss: {forward_history['val_loss'][-1]:.4f}")
    
    # 4. 训练逆向网络
    print("\n[Step 4] Training inverse network...")
    inverse_history = tnn.train_inverse(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=1e-4,
        patience=15
    )
    
    print(f"  Final train loss: {inverse_history['train_loss'][-1]:.4f}")
    if inverse_history['val_loss']:
        print(f"  Final val loss: {inverse_history['val_loss'][-1]:.4f}")
    
    # 5. 评估模型
    print("\n[Step 5] Evaluating model...")
    metrics = MetricsCollection()
    
    tnn.eval()
    all_pred = []
    all_target = []
    
    with torch.no_grad():
        for batch in test_loader:
            design = batch['design']
            performance = batch['performance']
            pred_perf = tnn.forward(design)
            all_pred.append(pred_perf)
            all_target.append(performance)
    
    all_pred = torch.cat(all_pred, dim=0)
    all_target = torch.cat(all_target, dim=0)
    
    results = metrics.compute_summary(all_pred, all_target)
    print("  Forward network metrics:")
    for name, value in results.items():
        print(f"    {name}: {value:.4f}")
    
    # 6. 逆向设计测试
    print("\n[Step 6] Testing inverse design...")
    
    # 设置目标性能
    target_efficiency = 0.85
    target_uniformity = 0.8
    target_loss = 0.1
    target_performance = torch.tensor([[target_efficiency, target_uniformity, target_loss]])
    
    # 生成设计
    design, pred_performance = tnn.inverse_design(target_performance)
    
    print(f"  Target performance: {target_performance[0].tolist()}")
    print(f"  Predicted performance: {pred_performance[0].tolist()}")
    print(f"  Design shape: {design.shape}")
    print(f"  Design range: [{design.min().item():.3f}, {design.max().item():.3f}]")
    
    # 7. 保存模型
    print("\n[Step 7] Saving model...")
    save_path = project_root / "models" / "pretrained" / "tnn_example.pth"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    tnn.save(save_path)
    print(f"  Model saved to: {save_path}")
    
    return tnn, design, pred_performance


def graph_integration_example(tnn):
    """
    计算图集成示例
    """
    print("\n" + "=" * 60)
    print("Graph Integration Example")
    print("=" * 60)
    
    design_shape = tnn.design_shape
    
    # 示例 1: 前向预测
    print("\n[Example 1] Forward prediction using graph...")
    
    # 创建设计参数节点
    initial_design = torch.rand(design_shape) * 0.5 + 0.25
    param_node = ParameterizationNode('param', initial_design, requires_grad=False)
    
    # 创建 TNN 前向节点
    forward_node = TNNNode(
        name='tnn_forward',
        tnn_model=tnn,
        mode='forward',
        input_node=param_node
    )
    
    # 创建计算图
    graph = Graph([forward_node])
    
    # 执行前向预测
    performance = graph.forward()[0]
    print(f"  Input design shape: {initial_design.shape}")
    print(f"  Predicted performance: {performance[0].tolist()}")
    
    # 示例 2: 逆向设计
    print("\n[Example 2] Inverse design using graph...")
    
    # 创建目标性能节点
    target_perf = torch.tensor([[0.85, 0.8, 0.1]])
    target_node = TargetPerformanceNode('target', target_perf)
    
    # 创建 TNN 逆向节点
    inverse_node = TNNNode(
        name='tnn_inverse',
        tnn_model=tnn,
        mode='inverse',
        input_node=target_node
    )
    
    # 创建计算图
    graph = Graph([inverse_node])
    
    # 执行逆向设计
    design, pred_perf = graph.forward()[0]
    print(f"  Target performance: {target_perf[0].tolist()}")
    print(f"  Generated design shape: {design.shape}")
    print(f"  Predicted performance: {pred_perf[0].tolist()}")
    
    return design, pred_perf


def visualize_results(tnn, design, pred_performance):
    """
    可视化结果
    """
    print("\n" + "=" * 60)
    print("Visualizing Results")
    print("=" * 60)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. 设计可视化
    ax = axes[0, 0]
    im = ax.imshow(design.squeeze().cpu().numpy(), cmap='viridis', aspect='auto')
    ax.set_title('Generated Design')
    ax.set_xlabel('Width (pixels)')
    ax.set_ylabel('Length (pixels)')
    plt.colorbar(im, ax=ax, label='Material density')
    
    # 2. 设计直方图
    ax = axes[0, 1]
    ax.hist(design.cpu().numpy().flatten(), bins=50, alpha=0.7, edgecolor='black')
    ax.set_title('Design Distribution')
    ax.set_xlabel('Material density')
    ax.set_ylabel('Count')
    ax.axvline(x=0.5, color='r', linestyle='--', label='Binary threshold')
    ax.legend()
    
    # 3. 性能对比
    ax = axes[1, 0]
    target = torch.tensor([[0.85, 0.8, 0.1]])
    pred = pred_performance[0].cpu().numpy()
    target_np = target[0].cpu().numpy()
    
    x = np.arange(3)
    width = 0.35
    
    ax.bar(x - width/2, target_np, width, label='Target', alpha=0.7)
    ax.bar(x + width/2, pred, width, label='Predicted', alpha=0.7)
    
    ax.set_title('Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(['Efficiency', 'Uniformity', 'Loss'])
    ax.set_ylabel('Value')
    ax.legend()
    
    # 4. 训练历史（如果有）
    ax = axes[1, 1]
    if tnn.training_history['train_loss']:
        epochs = range(1, len(tnn.training_history['train_loss']) + 1)
        ax.plot(epochs, tnn.training_history['train_loss'], 'b-', label='Train Loss')
        if tnn.training_history['val_loss']:
            ax.plot(epochs, tnn.training_history['val_loss'], 'r-', label='Val Loss')
        ax.set_title('Training History')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.set_yscale('log')
    else:
        ax.text(0.5, 0.5, 'No training history available', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Training History')
    
    plt.tight_layout()
    
    # 保存图像
    save_path = project_root / "examples" / "outputs" / "tnn_example_results.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Figure saved to: {save_path}")
    
    plt.close()


def main():
    """主函数"""
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 设置 matplotlib 后端（避免 GUI 问题）
    import matplotlib
    matplotlib.use('Agg')
    
    print("\n" + "=" * 60)
    print("TNN (Tandem Neural Network) Inverse Design Example")
    print("=" * 60)
    
    # 训练 TNN
    tnn, design, pred_performance = train_tnn_example()
    
    # 计算图集成
    design, pred_performance = graph_integration_example(tnn)
    
    # 可视化结果
    visualize_results(tnn, design, pred_performance)
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

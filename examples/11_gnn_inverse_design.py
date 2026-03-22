"""
GNN（图神经网络）逆向设计示例

本示例展示如何使用 GNN 进行光子学器件的逆向设计：
1. 将设计网格转换为图结构
2. 训练图神经网络代理模型
3. 使用 GNN 进行性能预测和逆向设计
4. 可视化图结构和注意力权重

GNN 的优势：
- 捕获空间拓扑关系
- 处理不规则结构
- 可解释性（通过注意力机制）

依赖要求：
- pip install torch-geometric torch-sparse torch-scatter
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

# 检查 PyTorch Geometric 是否可用
try:
    import torch_geometric
    from torch_geometric.nn import GCNConv, GATConv, SAGEConv
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False
    print("=" * 60)
    print("警告: PyTorch Geometric 未安装")
    print("=" * 60)
    print("请运行以下命令安装:")
    print("  pip install torch-geometric torch-sparse torch-scatter")
    print()
    print("或参考官方文档:")
    print("  https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html")
    print("=" * 60)
    print()
    print("示例将以演示模式运行（跳过实际训练）")
    print()

if PYG_AVAILABLE:
    # 导入 GNN 相关模块
    from models.inverse.gnn import (
        PhotonicsGNN,
        GNNConfig,
        GraphBuildConfig,
        GraphBuilder,
        GraphEncoder,
        GraphDecoder,
        GlobalPooling
    )

    # 导入数据加载器
    from data.loaders.pipeline import (
        SyntheticDataset,
        create_dataloaders
    )

    # 导入训练相关
    from models.training.metrics import MetricsCollection, R2Score, MAE, MSE
    from models.training.callbacks import EarlyStopping, TrainingLogger

    # 导入可视化
    from interfaces.visualization import plot_design, plot_intensity


def create_graph_training_data(
    num_samples: int = 2000,
    design_shape: tuple = (50, 50),
    performance_dim: int = 3,
    noise_level: float = 0.05,
    seed: int = 42
):
    """
    创建用于 GNN 训练的合成数据
    
    使用较小的设计网格以适应图结构表示。
    """
    print(f"Creating synthetic dataset for GNN...")
    print(f"  Design shape: {design_shape}")
    print(f"  Num samples: {num_samples}")
    
    dataset = SyntheticDataset(
        num_samples=num_samples,
        design_shape=design_shape,
        performance_dim=performance_dim,
        noise_level=noise_level,
        seed=seed
    )
    
    return dataset


def visualize_graph_structure(design, graph_builder, save_path=None):
    """
    可视化图结构
    
    展示设计网格如何转换为图结构。
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 原始设计
    ax = axes[0]
    im = ax.imshow(design, cmap='viridis', aspect='auto')
    ax.set_title('Original Design')
    ax.set_xlabel('Width')
    ax.set_ylabel('Height')
    plt.colorbar(im, ax=ax)
    
    # 图节点分布
    ax = axes[1]
    h, w = design.shape
    
    # 采样部分边进行可视化
    node_features, edge_index, edge_features = graph_builder(
        torch.from_numpy(design).float().unsqueeze(0)
    )
    
    # 绘制节点位置
    y_coords = np.repeat(np.arange(h), w) / h
    x_coords = np.tile(np.arange(w), h) / w
    
    ax.scatter(x_coords, y_coords, c=design.flatten(), 
               cmap='viridis', s=10, alpha=0.6)
    
    # 采样边
    num_edges = edge_index.shape[1]
    sample_indices = np.random.choice(num_edges, min(500, num_edges), replace=False)
    
    for idx in sample_indices:
        src, dst = edge_index[:, idx].numpy()
        src_y, src_x = src // w, src % w
        dst_y, dst_x = dst // w, dst % w
        ax.plot([src_x/w, dst_x/w], [src_y/h, dst_y/h], 
                'gray', alpha=0.1, linewidth=0.5)
    
    ax.set_title('Graph Structure')
    ax.set_xlabel('Width (normalized)')
    ax.set_ylabel('Height (normalized)')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    
    # 节点特征分布
    ax = axes[2]
    features = node_features.numpy()
    ax.hist(features.flatten(), bins=50, alpha=0.7, edgecolor='black')
    ax.set_title('Node Feature Distribution')
    ax.set_xlabel('Feature Value')
    ax.set_ylabel('Count')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved graph structure visualization to {save_path}")
    
    plt.close()


def visualize_attention_weights(
    design, 
    model, 
    graph_builder,
    save_path=None
):
    """
    可视化注意力权重
    
    展示 GNN 如何关注设计中的不同区域。
    """
    model.eval()
    
    with torch.no_grad():
        # 构建图
        design_tensor = torch.from_numpy(design).float().unsqueeze(0)
        node_features, edge_index, edge_features = graph_builder(design_tensor)
        batch = torch.zeros(node_features.shape[0], dtype=torch.long)
        
        # 获取注意力权重
        # 注意：需要修改模型以返回注意力权重
        # 这里使用简化的可视化方法
        
        h, w = design.shape
        
        # 获取图嵌入
        graph_embedding = model.encoder(node_features, edge_index, batch)
        
        # 使用节点特征的范数作为重要性指标
        node_importance = torch.norm(node_features, dim=1).numpy()
        importance_map = node_importance.reshape(h, w)
        
        # 归一化
        importance_map = (importance_map - importance_map.min()) / \
                        (importance_map.max() - importance_map.min() + 1e-8)
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # 原始设计
    ax = axes[0]
    im = ax.imshow(design, cmap='viridis', aspect='auto')
    ax.set_title('Original Design')
    plt.colorbar(im, ax=ax)
    
    # 重要性热图
    ax = axes[1]
    im = ax.imshow(importance_map, cmap='hot', aspect='auto')
    ax.set_title('Node Importance (Attention)')
    plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved attention visualization to {save_path}")
    
    plt.close()


def train_gnn_surrogate(
    train_loader,
    val_loader,
    config,
    num_epochs=50,
    learning_rate=1e-3,
    device='cpu',
    verbose=True
):
    """
    训练 GNN 代理模型
    
    将设计网格转换为图，使用 GNN 预测性能。
    """
    print("\n" + "=" * 60)
    print("Training GNN Surrogate Model")
    print("=" * 60)
    
    # 创建模型
    model = PhotonicsGNN(config)
    model = model.to(device)
    
    # 打印模型信息
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    print(f"Conv type: {config.conv_type}")
    print(f"Num layers: {config.num_layers}")
    print(f"Hidden dim: {config.hidden_dim}")
    
    # 优化器和损失
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()
    
    # 指标
    metrics = MetricsCollection([
        MSE(),
        MAE(),
        R2Score()
    ])
    
    # 训练循环
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
            designs = batch['design'].to(device)
            targets = batch['performance'].to(device)
            
            optimizer.zero_grad()
            
            # 前向传播
            predictions = model(designs)
            loss = criterion(predictions, targets)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1
        
        avg_train_loss = train_loss / num_batches
        train_losses.append(avg_train_loss)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                designs = batch['design'].to(device)
                targets = batch['performance'].to(device)
                
                predictions = model(designs)
                loss = criterion(predictions, targets)
                
                val_loss += loss.item()
                val_batches += 1
                
                all_preds.append(predictions.cpu())
                all_targets.append(targets.cpu())
        
        avg_val_loss = val_loss / val_batches
        val_losses.append(avg_val_loss)
        
        # 计算指标
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        metric_results = metrics.compute(all_preds, all_targets)
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {avg_train_loss:.6f}")
            print(f"  Val Loss: {avg_val_loss:.6f}")
            print(f"  Val R²: {metric_results['r2']:.4f}")
            print(f"  Val MAE: {metric_results['mae']:.6f}")
        
        # 早停
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # 保存最佳模型
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    # 恢复最佳模型
    model.load_state_dict(best_model_state)
    
    print(f"\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    
    return model, {'train_losses': train_losses, 'val_losses': val_losses}


def inverse_design_with_gnn(
    model,
    target_performance,
    design_shape,
    num_iterations=100,
    learning_rate=0.1,
    device='cpu',
    verbose=True
):
    """
    使用 GNN 进行逆向设计
    
    通过优化输入设计来匹配目标性能。
    """
    print("\n" + "=" * 60)
    print("GNN-based Inverse Design")
    print("=" * 60)
    print(f"Target performance: {target_performance}")
    
    model.eval()
    
    # 初始化设计（随机）
    design = torch.rand(1, *design_shape, requires_grad=True, device=device)
    
    # 目标性能
    target = torch.tensor([target_performance], device=device)
    
    # 优化器
    optimizer = torch.optim.Adam([design], lr=learning_rate)
    
    # 记录优化过程
    history = {
        'loss': [],
        'predicted_performance': []
    }
    
    for i in range(num_iterations):
        optimizer.zero_grad()
        
        # 预测性能
        predicted = model(design)
        
        # 计算损失
        loss = torch.nn.functional.mse_loss(predicted, target)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 约束设计范围
        with torch.no_grad():
            design.data = torch.clamp(design.data, 0, 1)
        
        # 记录
        history['loss'].append(loss.item())
        history['predicted_performance'].append(predicted.detach().cpu().numpy()[0])
        
        if verbose and (i + 1) % 20 == 0:
            print(f"Iteration {i+1}: Loss = {loss.item():.6f}")
    
    # 最终结果
    final_design = design.detach().cpu().numpy()[0]
    final_performance = model(design).detach().cpu().numpy()[0]
    
    print(f"\nFinal predicted performance: {final_performance}")
    print(f"Target performance: {target_performance}")
    
    # 计算误差
    errors = np.abs(final_performance - np.array(target_performance))
    print(f"Absolute errors: {errors}")
    
    return final_design, final_performance, history


def visualize_training_history(history, save_path=None):
    """可视化训练历史"""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    epochs = range(1, len(history['train_losses']) + 1)
    ax.plot(epochs, history['train_losses'], 'b-', label='Train Loss', linewidth=2)
    ax.plot(epochs, history['val_losses'], 'r-', label='Val Loss', linewidth=2)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (MSE)')
    ax.set_title('Training History')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved training history to {save_path}")
    
    plt.close()


def visualize_optimization_history(history, target_performance, save_path=None):
    """可视化优化历史"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 损失曲线
    ax = axes[0]
    ax.plot(history['loss'], 'b-', linewidth=2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss (MSE)')
    ax.set_title('Optimization Loss')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # 性能收敛
    ax = axes[1]
    predicted = np.array(history['predicted_performance'])
    for i in range(predicted.shape[1]):
        ax.plot(predicted[:, i], label=f'Perf {i+1}', linewidth=2)
        ax.axhline(y=target_performance[i], color=f'C{i}', 
                   linestyle='--', alpha=0.7, label=f'Target {i+1}')
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Performance')
    ax.set_title('Performance Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved optimization history to {save_path}")
    
    plt.close()


def compare_gnn_architectures(
    train_loader,
    val_loader,
    design_shape,
    performance_dim,
    device='cpu'
):
    """
    比较不同 GNN 架构的性能
    
    比较 GCN, GAT, GraphSAGE 三种架构。
    """
    print("\n" + "=" * 60)
    print("Comparing GNN Architectures")
    print("=" * 60)
    
    architectures = ['gcn', 'gat', 'graphsage']
    results = {}
    
    for arch in architectures:
        print(f"\n--- Testing {arch.upper()} ---")
        
        config = GNNConfig(
            node_feature_dim=3,  # 值 + 位置
            hidden_dim=64,
            num_layers=3,
            conv_type=arch,
            num_heads=4,
            output_dim=performance_dim,
            dropout=0.1,
            graph_build_config=GraphBuildConfig(
                method='grid',
                connectivity=4,
                use_position=True
            )
        )
        
        model, history = train_gnn_surrogate(
            train_loader, val_loader, config,
            num_epochs=30,  # 较少 epoch 用于比较
            device=device,
            verbose=False
        )
        
        results[arch] = {
            'model': model,
            'history': history,
            'final_val_loss': min(history['val_losses'])
        }
        
        print(f"  Final val loss: {results[arch]['final_val_loss']:.6f}")
    
    # 打印比较结果
    print("\n" + "-" * 40)
    print("Architecture Comparison Results:")
    print("-" * 40)
    for arch, result in results.items():
        print(f"  {arch.upper():12s}: {result['final_val_loss']:.6f}")
    
    # 找出最佳架构
    best_arch = min(results.keys(), key=lambda x: results[x]['final_val_loss'])
    print(f"\nBest architecture: {best_arch.upper()}")
    
    return results


def main():
    """主函数"""
    print("=" * 60)
    print("GNN-based Photonics Inverse Design Example")
    print("=" * 60)
    
    if not PYG_AVAILABLE:
        print("\n演示模式: PyTorch Geometric 未安装")
        print("-" * 40)
        print("\nGNN 示例需要 PyTorch Geometric 库。")
        print("\n安装命令:")
        print("  pip install torch-geometric torch-sparse torch-scatter")
        print("\n或使用 conda:")
        print("  conda install pyg -c pyg")
        print("\n安装后重新运行此示例。")
        print("\n" + "=" * 60)
        return
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 参数
    design_shape = (30, 30)  # 较小的网格以加快训练
    performance_dim = 3
    num_samples = 1000
    
    # 创建数据
    dataset = create_graph_training_data(
        num_samples=num_samples,
        design_shape=design_shape,
        performance_dim=performance_dim
    )
    
    # 创建数据加载器
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset, batch_size=16, train_ratio=0.7, val_ratio=0.15
    )
    
    # 输出目录
    output_dir = Path(__file__).parent / 'outputs' / 'gnn_example'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 配置 GNN
    config = GNNConfig(
        node_feature_dim=3,  # 值 + (x, y) 位置
        hidden_dim=64,
        num_layers=4,
        conv_type='gat',  # 使用图注意力网络
        num_heads=4,
        output_dim=performance_dim,
        dropout=0.1,
        layer_norm=True,
        graph_build_config=GraphBuildConfig(
            method='grid',
            connectivity=4,
            use_position=True,
            use_distance=True
        )
    )
    
    # 训练模型
    model, history = train_gnn_surrogate(
        train_loader, val_loader, config,
        num_epochs=50,
        learning_rate=1e-3,
        device=device
    )
    
    # 可视化训练历史
    visualize_training_history(
        history, 
        save_path=output_dir / 'training_history.png'
    )
    
    # 可视化图结构
    sample_design = dataset[0]['design'].numpy()
    graph_builder = GraphBuilder(config.graph_build_config)
    
    visualize_graph_structure(
        sample_design,
        graph_builder,
        save_path=output_dir / 'graph_structure.png'
    )
    
    # 可视化注意力权重
    model = model.to('cpu')  # 移回 CPU 进行可视化
    visualize_attention_weights(
        sample_design,
        model,
        graph_builder,
        save_path=output_dir / 'attention_weights.png'
    )
    
    # 逆向设计
    target_performance = [0.7, 0.6, 0.3]  # 目标性能
    
    final_design, predicted_perf, opt_history = inverse_design_with_gnn(
        model, target_performance, design_shape,
        num_iterations=100, device='cpu'
    )
    
    # 可视化逆向设计结果
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    ax = axes[0]
    im = ax.imshow(final_design, cmap='viridis', aspect='auto')
    ax.set_title('Designed Structure')
    plt.colorbar(im, ax=ax)
    
    ax = axes[1]
    labels = [f'Perf {i+1}' for i in range(performance_dim)]
    x = np.arange(len(labels))
    width = 0.35
    
    ax.bar(x - width/2, target_performance, width, label='Target', color='blue', alpha=0.7)
    ax.bar(x + width/2, predicted_perf, width, label='Predicted', color='orange', alpha=0.7)
    
    ax.set_ylabel('Performance')
    ax.set_title('Target vs Predicted Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'inverse_design_result.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved inverse design result to {output_dir / 'inverse_design_result.png'}")
    
    # 可视化优化历史
    visualize_optimization_history(
        opt_history, target_performance,
        save_path=output_dir / 'optimization_history.png'
    )
    
    # 比较不同架构
    print("\n" + "=" * 60)
    print("Architecture Comparison")
    print("=" * 60)
    
    results = compare_gnn_architectures(
        train_loader, val_loader,
        design_shape, performance_dim,
        device=device
    )
    
    print("\n" + "=" * 60)
    print("Example completed!")
    print("=" * 60)
    print(f"\nOutput files saved to: {output_dir}")
    print("  - training_history.png")
    print("  - graph_structure.png")
    print("  - attention_weights.png")
    print("  - inverse_design_result.png")
    print("  - optimization_history.png")


if __name__ == "__main__":
    main()

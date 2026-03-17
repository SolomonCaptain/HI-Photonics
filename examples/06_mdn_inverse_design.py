"""
MDN（混合密度网络）逆向设计示例

本示例展示如何使用 MDN 进行光子学逆向设计：
1. 创建和训练 MDN
2. 从条件分布中采样多个候选设计
3. 选择最优设计
4. 与 TNN 对比

MDN 相比 TNN 的优势：
- 处理一对多映射问题
- 输出设计的概率分布
- 提供不确定性估计
- 生成多样化的候选设计
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

from models.inverse.mdn import (
    MDN, MDNConfig,
    GaussianMixtureDistribution,
    MDNTandemNetwork
)
from models.inverse.tnn import ForwardNetwork, ForwardNetworkConfig
from models.training.losses import MDNLoss, MDNRegularizedLoss
from data.loaders.pipeline import SyntheticDataset


def set_seed(seed: int = 42):
    """设置随机种子"""
    torch.manual_seed(seed)
    np.random.seed(seed)


def generate_multimodal_data(
    n_samples: int = 5000,
    design_shape: tuple = (50, 10),
    performance_dim: int = 3,
    n_modes: int = 3
):
    """
    生成多模态数据（同一性能对应多个设计）
    
    这模拟了光子学逆向设计中的"一对多"映射问题。
    """
    H, W = design_shape
    design_dim = H * W
    
    designs = []
    performances = []
    
    samples_per_mode = n_samples // n_modes
    
    for mode in range(n_modes):
        # 为每个模式创建不同的基础设计
        base_design = torch.zeros(H, W)
        
        # 不同模式有不同的结构特征
        if mode == 0:
            # 模式1：中央光栅
            base_design[H//3:2*H//3, :] = torch.linspace(0.3, 0.7, W)
        elif mode == 1:
            # 模式2：边缘增强
            base_design[:, :W//3] = 0.6
            base_design[:, 2*W//3:] = 0.6
        else:
            # 模式3：棋盘格
            for i in range(0, H, 5):
                for j in range(0, W, 3):
                    base_design[i:i+3, j:j+2] = 0.5 + 0.3 * ((i+j) % 2)
        
        # 为该模式生成变化
        for _ in range(samples_per_mode):
            # 添加随机扰动
            noise = torch.randn(H, W) * 0.1
            design = torch.clamp(base_design + noise, 0, 1)
            
            # 模拟性能计算（简化版）
            performance = torch.zeros(performance_dim)
            performance[0] = 0.7 + 0.2 * design.mean() + 0.1 * torch.randn(1)
            performance[1] = 0.5 + 0.3 * design.std() + 0.1 * torch.randn(1)
            performance[2] = 0.1 * mode + 0.05 * torch.randn(1)
            
            designs.append(design)
            performances.append(performance)
    
    designs = torch.stack(designs)
    performances = torch.stack(performances)
    
    # 打乱数据
    indices = torch.randperm(len(designs))
    designs = designs[indices]
    performances = performances[indices]
    
    return designs, performances


def train_mdn_example():
    """MDN 训练示例"""
    print("=" * 60)
    print("MDN 训练示例")
    print("=" * 60)
    
    set_seed(42)
    
    # 参数设置
    design_shape = (50, 10)
    performance_dim = 3
    n_components = 5
    
    # 生成数据
    print("\n生成多模态训练数据...")
    designs, performances = generate_multimodal_data(
        n_samples=5000,
        design_shape=design_shape,
        performance_dim=performance_dim,
        n_modes=3
    )
    
    # 划分数据集
    n_train = int(0.8 * len(designs))
    train_designs = designs[:n_train]
    train_performances = performances[:n_train]
    val_designs = designs[n_train:]
    val_performances = performances[n_train:]
    
    # 创建数据加载器
    train_dataset = TensorDataset(train_performances, train_designs)
    val_dataset = TensorDataset(val_performances, val_designs)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # 创建 MDN
    config = MDNConfig(
        input_dim=performance_dim,
        output_dim=design_shape[0] * design_shape[1],
        design_shape=design_shape,
        n_components=n_components,
        hidden_dims=[256, 512, 256],
        dropout_rate=0.1
    )
    
    mdn = MDN(config)
    print(f"\nMDN 配置:")
    print(f"  - 输入维度: {config.input_dim}")
    print(f"  - 输出维度: {config.output_dim}")
    print(f"  - 高斯分量数: {config.n_components}")
    print(f"  - 参数数量: {mdn.count_parameters():,}")
    
    # 训练
    print("\n开始训练...")
    history = mdn.train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=50,
        lr=1e-3,
        patience=10
    )
    
    return mdn, history, design_shape, performance_dim


def sample_and_visualize(mdn, design_shape, performance_dim):
    """采样和可视化"""
    print("\n" + "=" * 60)
    print("采样和可视化")
    print("=" * 60)
    
    mdn.eval()
    
    # 目标性能
    target_perf = torch.tensor([[0.85, 0.6, 0.1]])
    
    # 获取分布参数
    pi, mu, sigma = mdn(target_perf)
    print(f"\n目标性能: {target_perf[0].tolist()}")
    print(f"混合权重: {pi[0].detach().numpy()}")
    
    # 创建分布对象
    distribution = mdn.get_distribution(target_perf)
    
    # 计算熵（不确定性）
    entropy = distribution.get_entropy()
    print(f"分布熵: {entropy.item():.4f}")
    
    # 采样多个设计
    n_samples = 9
    samples = mdn.sample(target_perf, n_samples=n_samples)
    
    # 可视化
    H, W = design_shape
    
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    fig.suptitle('MDN 采样的候选设计', fontsize=14)
    
    for i, ax in enumerate(axes.flat):
        design = samples[0, i].detach().numpy()
        im = ax.imshow(design, cmap='gray', vmin=0, vmax=1)
        ax.set_title(f'样本 {i+1}')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    plt.tight_layout()
    plt.savefig('examples/outputs/mdn_samples.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\n采样设计已保存至 examples/outputs/mdn_samples.png")
    
    # 可视化分布参数
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 混合权重
    ax = axes[0]
    ax.bar(range(pi.size(1)), pi[0].detach().numpy())
    ax.set_xlabel('分量索引')
    ax.set_ylabel('权重')
    ax.set_title('混合权重 π')
    
    # 均值（第一个分量）
    ax = axes[1]
    mu_reshaped = mu[0, 0].detach().numpy().reshape(H, W)
    im = ax.imshow(mu_reshaped, cmap='viridis')
    ax.set_title('分量0的均值 μ')
    plt.colorbar(im, ax=ax)
    
    # 标准差（第一个分量）
    ax = axes[2]
    sigma_reshaped = sigma[0, 0].detach().numpy().reshape(H, W)
    im = ax.imshow(sigma_reshaped, cmap='hot')
    ax.set_title('分量0的标准差 σ')
    plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig('examples/outputs/mdn_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("分布参数已保存至 examples/outputs/mdn_distribution.png")


def compare_with_tnn():
    """与 TNN 对比"""
    print("\n" + "=" * 60)
    print("MDN vs TNN 对比")
    print("=" * 60)
    
    design_shape = (50, 10)
    performance_dim = 3
    
    # 创建 TNN 前向网络（简化版）
    forward_config = ForwardNetworkConfig(
        design_shape=design_shape,
        performance_dim=performance_dim,
        hidden_dims=[128, 256, 128]
    )
    forward_net = ForwardNetwork(forward_config)
    
    # 创建 MDN
    mdn_config = MDNConfig(
        input_dim=performance_dim,
        output_dim=design_shape[0] * design_shape[1],
        design_shape=design_shape,
        n_components=5
    )
    mdn = MDN(mdn_config)
    
    # 创建联合模型
    mdn_tandem = MDNTandemNetwork(forward_net, mdn_config)
    
    print("\n对比分析:")
    print("-" * 40)
    print("| 特性         | TNN          | MDN          |")
    print("|--------------|--------------|--------------|")
    print("| 输出类型     | 单一设计     | 概率分布     |")
    print("| 多解处理     | 否           | 是           |")
    print("| 不确定性     | 无           | 有           |")
    print("| 采样多样性   | 单一解       | 多样化解     |")
    print("| 训练复杂度   | 低           | 中等         |")
    
    # 目标性能
    target_perf = torch.tensor([[0.85, 0.6, 0.1]])
    
    # MDN 采样
    mdn_samples = mdn.sample(target_perf, n_samples=10)
    mdn_best = mdn.sample_mode(target_perf)
    
    print(f"\n目标性能: {target_perf[0].tolist()}")
    print(f"MDN 采样数量: 10")
    print(f"MDN 最可能设计形状: {mdn_best.shape}")
    
    # 计算采样多样性
    sample_diversity = torch.std(mdn_samples.view(10, -1), dim=0).mean()
    print(f"采样多样性指标: {sample_diversity.item():.4f}")


def graph_integration_example():
    """计算图集成示例"""
    print("\n" + "=" * 60)
    print("计算图集成示例")
    print("=" * 60)
    
    from core.nodes.neural_network import MDNNode, TargetPerformanceNode
    
    design_shape = (50, 10)
    performance_dim = 3
    
    # 创建 MDN
    config = MDNConfig(
        input_dim=performance_dim,
        output_dim=design_shape[0] * design_shape[1],
        design_shape=design_shape,
        n_components=5
    )
    mdn = MDN(config)
    
    # 创建节点
    target_perf = torch.tensor([[0.85, 0.6, 0.1]])
    target_node = TargetPerformanceNode('target', target_perf)
    mdn_node = MDNNode('mdn', mdn, target_node)
    
    # 方式1: 获取分布参数
    pi, mu, sigma = mdn_node.forward(mode='params')
    print(f"\n方式1 - 分布参数:")
    print(f"  混合权重: {pi.shape}")
    print(f"  均值: {mu.shape}")
    print(f"  标准差: {sigma.shape}")
    
    # 方式2: 采样
    samples = mdn_node.forward(mode='sample', n_samples=5)
    print(f"\n方式2 - 采样设计:")
    print(f"  样本形状: {samples.shape}")
    
    # 方式3: 最可能设计
    best = mdn_node.forward(mode='mode')
    print(f"\n方式3 - 最可能设计:")
    print(f"  设计形状: {best.shape}")
    
    print("\n计算图集成完成!")


def main():
    """主函数"""
    # 确保输出目录存在
    import os
    os.makedirs('examples/outputs', exist_ok=True)
    
    # 训练 MDN
    mdn, history, design_shape, performance_dim = train_mdn_example()
    
    # 采样和可视化
    sample_and_visualize(mdn, design_shape, performance_dim)
    
    # 与 TNN 对比
    compare_with_tnn()
    
    # 计算图集成
    graph_integration_example()
    
    print("\n" + "=" * 60)
    print("MDN 示例完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()

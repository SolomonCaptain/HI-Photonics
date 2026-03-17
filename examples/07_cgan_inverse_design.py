"""
CGAN 条件生成对抗网络逆向设计示例

本示例展示如何使用 CGAN 进行光子器件逆向设计：
1. 创建 CGAN 模型
2. 使用合成数据训练
3. 从目标性能生成多样化设计
4. 集成到计算图框架

CGAN 相比 TNN 的优势：
- 可以为同一目标生成多个不同的候选设计
- 生成的设计更加真实和多样化
- 通过对抗训练提升设计质量
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

from models.inverse.cgan import (
    CGAN, CGANConfig,
    GeneratorConfig, DiscriminatorConfig,
    ConditionalGenerator, ConditionalDiscriminator,
    WGAN_GP
)
from models.training.losses import GANLoss, GradientPenaltyLoss, CGANCombinedLoss
from core.nodes.neural_network import CGANNode, TargetPerformanceNode


def create_synthetic_dataset(
    num_samples: int = 1000,
    design_shape: tuple = (64, 16),
    condition_dim: int = 3,
    seed: int = 42
):
    """
    创建合成数据集用于演示
    
    生成随机设计和对应的"性能"指标
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 生成随机设计（0-1之间的连续值）
    designs = torch.rand(num_samples, *design_shape)
    
    # 模拟性能计算（基于设计的简单函数）
    # 实际应用中应使用真实仿真器
    performances = torch.zeros(num_samples, condition_dim)
    
    for i in range(num_samples):
        design = designs[i]
        # 性能1: 平均密度
        performances[i, 0] = design.mean()
        # 性能2: 空间变化（梯度范数）
        dx = torch.abs(design[:, 1:] - design[:, :-1])
        performances[i, 1] = dx.mean()
        # 性能3: 中心区域密度
        center = design.shape[1] // 4
        performances[i, 2] = design[:, center:center*3].mean()
    
    # 归一化性能到 [0, 1]
    performances = (performances - performances.min(dim=0)[0]) / (
        performances.max(dim=0)[0] - performances.min(dim=0)[0] + 1e-8
    )
    
    return designs, performances


def train_cgan_example():
    """CGAN 训练示例"""
    print("=" * 60)
    print("CGAN 条件生成对抗网络训练示例")
    print("=" * 60)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    # 参数设置
    design_shape = (64, 16)
    condition_dim = 3
    latent_dim = 64
    batch_size = 32
    epochs = 50
    
    # 创建合成数据
    print("\n创建合成数据集...")
    designs, performances = create_synthetic_dataset(
        num_samples=1000,
        design_shape=design_shape,
        condition_dim=condition_dim
    )
    
    # 创建数据加载器
    dataset = TensorDataset(designs, performances)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    print(f"数据集大小: {len(dataset)}")
    print(f"设计形状: {designs.shape}")
    print(f"性能形状: {performances.shape}")
    
    # 创建 CGAN 模型
    print("\n创建 CGAN 模型...")
    
    # 配置生成器
    gen_config = GeneratorConfig(
        latent_dim=latent_dim,
        condition_dim=condition_dim,
        design_shape=design_shape,
        hidden_dims=[256, 512],
        hidden_channels=[256, 128, 64],
        output_activation='sigmoid'
    )
    
    # 配置判别器
    disc_config = DiscriminatorConfig(
        design_shape=design_shape,
        condition_dim=condition_dim,
        hidden_channels=[64, 128, 256],
        hidden_dims=[512, 256, 1],
        spectral_norm=True
    )
    
    # 配置 CGAN
    cgan_config = CGANConfig(
        generator_config=gen_config,
        discriminator_config=disc_config,
        gan_type='wgan-gp',
        n_critic=5,
        lambda_gp=10.0,
        g_lr=1e-4,
        d_lr=4e-4
    )
    
    cgan = CGAN(cgan_config).to(device)
    
    print(f"\n生成器参数: {cgan.generator.count_parameters():,}")
    print(f"判别器参数: {cgan.discriminator.count_parameters():,}")
    
    # 训练模型
    print("\n开始训练...")
    history = cgan.train_model(
        dataloader,
        epochs=epochs,
        log_interval=10
    )
    
    # 绘制训练曲线
    plot_training_history(history)
    
    return cgan, designs, performances


def test_generation(cgan, performances, device):
    """测试生成功能"""
    print("\n" + "=" * 60)
    print("测试设计生成")
    print("=" * 60)
    
    cgan.eval()
    
    # 选择几个测试条件
    test_conditions = performances[:5].to(device)
    
    print("\n目标性能条件:")
    print(test_conditions.cpu().numpy())
    
    # 生成设计
    with torch.no_grad():
        # 单样本生成
        designs_single = cgan.generate(test_conditions, num_samples=1)
        print(f"\n单样本生成形状: {designs_single.shape}")
        
        # 多样本生成
        designs_multi = cgan.generate(test_conditions, num_samples=3)
        print(f"多样本生成形状: {designs_multi.shape}")
    
    # 可视化生成的设计
    visualize_generated_designs(designs_single, designs_multi)
    
    return designs_single, designs_multi


def test_graph_integration(cgan, device):
    """测试计算图集成"""
    print("\n" + "=" * 60)
    print("测试计算图集成")
    print("=" * 60)
    
    # 创建性能目标节点
    target_perf = torch.tensor([[0.5, 0.3, 0.7]], device=device)
    perf_node = TargetPerformanceNode('target_perf', target_perf)
    
    # 创建 CGAN 节点
    cgan_node = CGANNode('cgan_designer', cgan, perf_node, device=device)
    
    # 生成单个设计
    design = cgan_node.forward()
    print(f"\n生成设计形状: {design.shape}")
    print(f"设计值范围: [{design.min().item():.3f}, {design.max().item():.3f}]")
    
    # 生成多个候选设计
    designs = cgan_node.forward(num_samples=5)
    print(f"\n多样本生成形状: {designs.shape}")
    
    # 评估真实性
    validity = cgan_node.discriminate(design[:1], target_perf)
    print(f"\n判别器评分: {validity.item():.3f}")
    
    # 潜在空间插值
    interpolation = cgan_node.get_latent_interpolation(
        target_perf, num_steps=5
    )
    print(f"\n插值设计序列形状: {interpolation.shape}")
    
    return cgan_node


def test_wgan_gp_variant():
    """测试 WGAN-GP 便捷类"""
    print("\n" + "=" * 60)
    print("测试 WGAN-GP 便捷类")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 使用便捷类创建 WGAN-GP
    wgan = WGAN_GP(
        design_shape=(64, 16),
        condition_dim=3,
        latent_dim=64
    ).to(device)
    
    print(f"\nWGAN-GP 模型信息:")
    info = wgan.get_model_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # 测试生成
    condition = torch.rand(2, 3, device=device)
    designs = wgan.generate(condition, num_samples=3)
    print(f"\n生成设计形状: {designs.shape}")
    
    return wgan


def plot_training_history(history):
    """绘制训练历史"""
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # 损失曲线
        ax = axes[0]
        ax.plot(history['g_loss'], label='Generator Loss')
        ax.plot(history['d_loss'], label='Discriminator Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Losses')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 判别器损失细节
        ax = axes[1]
        ax.plot(history['d_loss'], label='D Loss', color='orange')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Discriminator Loss')
        ax.set_title('Discriminator Loss Detail')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图像
        output_dir = Path(__file__).parent / 'outputs'
        output_dir.mkdir(exist_ok=True)
        plt.savefig(output_dir / 'cgan_training_history.png', dpi=150)
        print(f"\n训练曲线已保存到: {output_dir / 'cgan_training_history.png'}")
        plt.close()
    except Exception as e:
        print(f"绘图失败: {e}")


def visualize_generated_designs(designs_single, designs_multi):
    """可视化生成的设计"""
    try:
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        
        # 单样本设计
        for i in range(min(5, designs_single.size(0))):
            ax = axes[0, i]
            design = designs_single[i].cpu().numpy()
            im = ax.imshow(design, cmap='gray', aspect='auto', vmin=0, vmax=1)
            ax.set_title(f'Single {i+1}')
            ax.set_xlabel('Width')
            ax.set_ylabel('Height')
        
        # 多样本设计（第一个条件的多个样本）
        for i in range(min(5, designs_multi.size(0))):
            ax = axes[1, i]
            design = designs_multi[i].cpu().numpy()
            im = ax.imshow(design, cmap='gray', aspect='auto', vmin=0, vmax=1)
            ax.set_title(f'Multi {i+1}')
            ax.set_xlabel('Width')
            ax.set_ylabel('Height')
        
        plt.suptitle('CGAN Generated Designs', fontsize=12)
        plt.tight_layout()
        
        # 保存图像
        output_dir = Path(__file__).parent / 'outputs'
        output_dir.mkdir(exist_ok=True)
        plt.savefig(output_dir / 'cgan_generated_designs.png', dpi=150)
        print(f"\n生成设计可视化已保存到: {output_dir / 'cgan_generated_designs.png'}")
        plt.close()
    except Exception as e:
        print(f"绘图失败: {e}")


def compare_with_tnn():
    """比较 CGAN 与 TNN 的特点"""
    print("\n" + "=" * 60)
    print("CGAN vs TNN 特点比较")
    print("=" * 60)
    
    comparison = """
    | 特性            | TNN                          | CGAN                         |
    |-----------------|------------------------------|------------------------------|
    | 映射方式        | 确定性映射                   | 概率性生成                   |
    | 一对多问题      | 单一解                       | 多样化解                     |
    | 训练稳定性      | 监督学习，稳定               | 对抗训练，需技巧             |
    | 设计多样性      | 低                           | 高                           |
    | 计算开销        | 低                           | 中等                         |
    | 推荐场景        | 快速原型设计                 | 探索性设计空间搜索           |
    
    使用建议:
    1. 如果需要快速获得单一设计方案，使用 TNN
    2. 如果需要探索多种可能的设计，使用 CGAN
    3. 可以结合使用：CGAN 生成候选，TNN 精调
    """
    print(comparison)


def main():
    """主函数"""
    print("CGAN 条件生成对抗网络逆向设计示例")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. 训练 CGAN
    cgan, designs, performances = train_cgan_example()
    
    # 2. 测试生成
    test_generation(cgan, performances, device)
    
    # 3. 测试计算图集成
    test_graph_integration(cgan, device)
    
    # 4. 测试 WGAN-GP 变体
    test_wgan_gp_variant()
    
    # 5. 比较 CGAN 与 TNN
    compare_with_tnn()
    
    print("\n" + "=" * 60)
    print("示例完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()

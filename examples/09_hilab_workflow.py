"""
HiLAB 混合逆向设计工作流示例

本示例展示如何使用 HiLAB 进行光子学器件的逆向设计：
1. 创建合成数据集
2. 训练 VAE 学习设计空间的潜在表示
3. 训练代理模型建立 z -> performance 映射
4. 使用贝叶斯优化在潜在空间中搜索最优设计
5. 评估生成设计的性能
6. 可视化结果

参考文献:
- Marzban et al., "HiLAB: A Hybrid Inverse-Design Framework", arXiv:2505.17491, 2025
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

# 导入 HiLAB 相关模块
from models.inverse.hilab import (
    VAE,
    VAEConfig,
    VAEEncoderConfig,
    VAEDecoderConfig,
    BayesianOptimizerConfig,
    HiLABConfig,
    HiLABEngine,
    create_hilab_for_challenge
)

from models.training.losses import (
    BetaVAELoss,
    VAETotalLoss,
    VAEScheduler
)

from models.training.metrics import MetricsCollection, R2Score, MAE, MSE
from models.training.callbacks import EarlyStopping, TrainingLogger

# 导入数据加载器
from data.loaders.pipeline import (
    PhotonicsDataset,
    SyntheticDataset,
    create_dataloaders
)


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


def train_vae_example(train_loader, val_loader, design_shape, latent_dim=32, epochs=50):
    """
    VAE 训练示例
    
    展示如何训练 VAE 学习设计空间的潜在表示。
    """
    print("\n" + "=" * 60)
    print("Step 1: Training VAE")
    print("=" * 60)
    
    # 配置 VAE
    encoder_config = VAEEncoderConfig(
        design_shape=design_shape,
        latent_dim=latent_dim,
        hidden_channels=[32, 64, 128],
        hidden_dims=[256, 128],
        dropout_rate=0.1
    )
    
    decoder_config = VAEDecoderConfig(
        latent_dim=latent_dim,
        design_shape=design_shape,
        hidden_dims=[128, 256],
        hidden_channels=[64, 32, 16, 1],
        dropout_rate=0.1
    )
    
    vae_config = VAEConfig(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        latent_dim=latent_dim,
        recon_weight=1.0,
        kl_weight=0.001,  # β-VAE
        beta_warmup_epochs=10
    )
    
    # 创建 VAE
    vae = VAE(vae_config)
    print(f"VAE parameters: {vae.count_parameters():,}")
    
    # 训练 VAE
    history = vae.train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=1e-3,
        weight_decay=1e-5,
        patience=15
    )
    
    print(f"Final train loss: {history['train_loss'][-1]:.4f}")
    if history['val_loss']:
        print(f"Final val loss: {history['val_loss'][-1]:.4f}")
    
    return vae, history


def train_surrogate_example(engine, train_loader, val_loader, epochs=30):
    """
    代理模型训练示例
    
    展示如何在潜在空间中训练 z -> performance 映射。
    """
    print("\n" + "=" * 60)
    print("Step 2: Training Surrogate Model")
    print("=" * 60)
    
    # 构建代理模型
    surrogate = engine.build_surrogate(
        hidden_dims=[128, 64, 32],
        activation='relu',
        dropout=0.1
    )
    
    print(f"Surrogate parameters: {sum(p.numel() for p in surrogate.parameters()):,}")
    
    # 训练代理模型
    history = engine.train_surrogate(
        data_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=1e-3,
        weight_decay=1e-5,
        patience=10
    )
    
    print(f"Final train loss: {history['train_loss'][-1]:.4f}")
    if history['val_loss']:
        print(f"Final val loss: {history['val_loss'][-1]:.4f}")
    
    return surrogate, history


def bayesian_optimization_example(engine, target_performance, n_iterations=30):
    """
    贝叶斯优化示例
    
    展示如何在潜在空间中使用贝叶斯优化进行逆向设计。
    """
    print("\n" + "=" * 60)
    print("Step 3: Bayesian Optimization in Latent Space")
    print("=" * 60)
    
    print(f"Target performance: {target_performance.tolist()}")
    
    # 执行逆向设计
    design, history = engine.inverse_design(
        target_performance=target_performance,
        n_iterations=n_iterations,
        n_initial=10,
        verbose=True,
        return_history=True
    )
    
    print(f"\nOptimization completed:")
    print(f"  Best error: {history[0]['best_error']:.6f}")
    print(f"  Design shape: {design.shape}")
    print(f"  Design range: [{design.min().item():.3f}, {design.max().item():.3f}]")
    
    return design, history


def diverse_sampling_example(engine, target_performance, n_samples=5):
    """
    多样化采样示例
    
    展示如何生成多个满足相同目标的不同设计。
    """
    print("\n" + "=" * 60)
    print("Step 4: Diverse Design Sampling")
    print("=" * 60)
    
    # 生成多样化设计
    designs = engine.sample_diverse(
        target_performance=target_performance,
        n_samples=n_samples,
        n_iterations=20,
        diversity_weight=0.1
    )
    
    print(f"Generated {n_samples} diverse designs")
    print(f"Design shape: {designs.shape}")
    
    return designs


def latent_interpolation_example(engine, design1, design2, num_steps=10):
    """
    潜在空间插值示例
    
    展示如何在两个设计之间进行平滑插值。
    """
    print("\n" + "=" * 60)
    print("Step 5: Latent Space Interpolation")
    print("=" * 60)
    
    # 在潜在空间中插值
    interpolated = engine.interpolate_designs(
        design1=design1,
        design2=design2,
        num_steps=num_steps
    )
    
    print(f"Generated {num_steps} interpolated designs")
    print(f"Interpolation shape: {interpolated.shape}")
    
    return interpolated


def evaluate_design_example(engine, design, simulator=None):
    """
    设计评估示例
    """
    print("\n" + "=" * 60)
    print("Step 6: Design Evaluation")
    print("=" * 60)
    
    results = engine.evaluate_design(design, simulator)
    
    print("Evaluation results:")
    print(f"  Reconstruction error: {results['reconstruction_error']:.6f}")
    if 'predicted_performance' in results:
        print(f"  Predicted performance: {results['predicted_performance'][0]}")
    
    return results


def visualize_vae_results(vae, history, test_loader):
    """
    可视化 VAE 训练结果
    """
    print("\n" + "=" * 60)
    print("Visualizing VAE Results")
    print("=" * 60)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. 训练损失曲线
    ax = axes[0, 0]
    epochs = range(1, len(history['train_loss']) + 1)
    ax.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    if history['val_loss']:
        ax.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax.set_title('VAE Training Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    # 2. 重建损失曲线
    ax = axes[0, 1]
    ax.plot(epochs, history['recon_loss'], 'g-', label='Reconstruction Loss', linewidth=2)
    ax.set_title('Reconstruction Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. KL 散度曲线
    ax = axes[0, 2]
    ax.plot(epochs, history['kl_loss'], 'm-', label='KL Divergence', linewidth=2)
    ax.set_title('KL Divergence')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('KL Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4-6. 重建示例
    vae.eval()
    with torch.no_grad():
        batch = next(iter(test_loader))
        if isinstance(batch, dict):
            x = batch['design'][:3].to(vae.device)
        else:
            x = batch[0][:3].to(vae.device)
        
        recon_x = vae(x)
    
    for i in range(3):
        ax = axes[1, i]
        # 显示原始设计 vs 重建设计
        combined = torch.cat([x[i], recon_x[i]], dim=1).cpu().numpy()
        im = ax.imshow(combined, cmap='viridis', aspect='auto')
        ax.axvline(x=x.shape[2], color='r', linestyle='--', linewidth=2)
        ax.set_title(f'Sample {i+1}: Original | Reconstructed')
        ax.set_xlabel('Width (pixels)')
        ax.set_ylabel('Length (pixels)')
        plt.colorbar(im, ax=ax, shrink=0.8)
    
    plt.tight_layout()
    
    save_path = project_root / "examples" / "outputs" / "hilab_vae_results.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"VAE results saved to: {save_path}")
    
    plt.close()


def visualize_optimization_results(design, history, target_performance):
    """
    可视化贝叶斯优化结果
    """
    print("\n" + "=" * 60)
    print("Visualizing Optimization Results")
    print("=" * 60)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. 生成的设计
    ax = axes[0]
    im = ax.imshow(design.squeeze().cpu().numpy(), cmap='viridis', aspect='auto')
    ax.set_title('Generated Design')
    ax.set_xlabel('Width (pixels)')
    ax.set_ylabel('Length (pixels)')
    plt.colorbar(im, ax=ax, label='Material density')
    
    # 2. 设计直方图
    ax = axes[1]
    ax.hist(design.cpu().numpy().flatten(), bins=50, alpha=0.7, edgecolor='black')
    ax.set_title('Design Distribution')
    ax.set_xlabel('Material density')
    ax.set_ylabel('Count')
    ax.axvline(x=0.5, color='r', linestyle='--', label='Binary threshold')
    ax.legend()
    
    # 3. 优化历史
    ax = axes[2]
    if history and 'observations' in history[0]:
        observations = history[0]['observations']
        x_obs = np.arange(len(observations))
        y_obs = [-y for _, y in observations]  # 转回误差（之前是负误差）
        ax.plot(x_obs, y_obs, 'b.-', label='Observation Error')
        ax.axhline(y=history[0]['best_error'], color='r', linestyle='--', 
                   label=f'Best Error: {history[0]["best_error"]:.4f}')
    ax.set_title('Optimization History')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Performance Error')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = project_root / "examples" / "outputs" / "hilab_optimization_results.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Optimization results saved to: {save_path}")
    
    plt.close()


def visualize_diverse_designs(designs):
    """
    可视化多样化设计
    """
    print("\n" + "=" * 60)
    print("Visualizing Diverse Designs")
    print("=" * 60)
    
    n_samples = designs.shape[0]
    fig, axes = plt.subplots(1, n_samples, figsize=(4 * n_samples, 4))
    
    if n_samples == 1:
        axes = [axes]
    
    for i in range(n_samples):
        ax = axes[i]
        im = ax.imshow(designs[i].cpu().numpy(), cmap='viridis', aspect='auto')
        ax.set_title(f'Design {i+1}')
        ax.set_xlabel('Width (pixels)')
        ax.set_ylabel('Length (pixels)')
        plt.colorbar(im, ax=ax, shrink=0.8)
    
    plt.tight_layout()
    
    save_path = project_root / "examples" / "outputs" / "hilab_diverse_designs.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Diverse designs saved to: {save_path}")
    
    plt.close()


def visualize_latent_space(vae, test_loader, n_samples=1000):
    """
    可视化潜在空间
    """
    print("\n" + "=" * 60)
    print("Visualizing Latent Space")
    print("=" * 60)
    
    vae.eval()
    
    # 收集潜在向量
    all_mu = []
    all_logvar = []
    
    with torch.no_grad():
        for batch in test_loader:
            if isinstance(batch, dict):
                x = batch['design'].to(vae.device)
            else:
                x = batch[0].to(vae.device)
            
            mu, logvar = vae.encode(x)
            all_mu.append(mu.cpu())
            all_logvar.append(logvar.cpu())
            
            if len(all_mu) * x.shape[0] >= n_samples:
                break
    
    all_mu = torch.cat(all_mu, dim=0)[:n_samples]
    all_logvar = torch.cat(all_logvar, dim=0)[:n_samples]
    
    # 如果 latent_dim > 2，使用 PCA 降维
    latent_dim = all_mu.shape[1]
    
    if latent_dim > 2:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        mu_2d = pca.fit_transform(all_mu.numpy())
    else:
        mu_2d = all_mu.numpy()
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 1. 潜在空间散点图
    ax = axes[0]
    sc = ax.scatter(mu_2d[:, 0], mu_2d[:, 1], alpha=0.5, c=range(len(mu_2d)), cmap='viridis')
    ax.set_title('Latent Space Distribution')
    ax.set_xlabel('Latent Dimension 1')
    ax.set_ylabel('Latent Dimension 2')
    plt.colorbar(sc, ax=ax, label='Sample Index')
    
    # 2. 潜在空间方差分布
    ax = axes[1]
    std = torch.exp(0.5 * all_logvar)
    ax.boxplot(std.numpy(), showfliers=False)
    ax.set_title('Latent Variable Uncertainty')
    ax.set_xlabel('Latent Dimension')
    ax.set_ylabel('Standard Deviation')
    
    plt.tight_layout()
    
    save_path = project_root / "examples" / "outputs" / "hilab_latent_space.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Latent space visualization saved to: {save_path}")
    
    plt.close()


def main():
    """主函数"""
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 设置 matplotlib 后端
    import matplotlib
    matplotlib.use('Agg')
    
    print("\n" + "=" * 60)
    print("HiLAB Hybrid Inverse Design Workflow Example")
    print("=" * 60)
    
    # 配置参数
    design_shape = (100, 22)
    performance_dim = 3
    latent_dim = 16
    batch_size = 32
    
    # 1. 创建数据集
    print("\n[Step 1] Creating dataset...")
    dataset = create_synthetic_training_data(
        num_samples=2000,
        design_shape=design_shape,
        performance_dim=performance_dim
    )
    
    train_loader, val_loader, test_loader = create_dataloaders(
        dataset,
        batch_size=batch_size,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )
    
    print(f"Train: {len(train_loader.dataset)} samples")
    print(f"Val: {len(val_loader.dataset)} samples")
    print(f"Test: {len(test_loader.dataset)} samples")
    
    # 2. 创建 HiLAB 引擎
    print("\n[Step 2] Creating HiLAB engine...")
    
    encoder_config = VAEEncoderConfig(
        design_shape=design_shape,
        latent_dim=latent_dim,
        hidden_channels=[32, 64, 128],
        hidden_dims=[256, 128]
    )
    
    decoder_config = VAEDecoderConfig(
        latent_dim=latent_dim,
        design_shape=design_shape,
        hidden_dims=[128, 256],
        hidden_channels=[64, 32, 16, 1]
    )
    
    vae_config = VAEConfig(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        latent_dim=latent_dim,
        kl_weight=0.001,
        beta_warmup_epochs=10
    )
    
    optimizer_config = BayesianOptimizerConfig(
        acquisition_type='ei',
        kernel_type='rbf',
        n_restarts=5
    )
    
    hilab_config = HiLABConfig(
        vae_config=vae_config,
        optimizer_config=optimizer_config,
        performance_dim=performance_dim,
        design_shape=design_shape,
        use_adjoint_refinement=False
    )
    
    engine = HiLABEngine(hilab_config)
    print(f"Engine info: {engine.get_model_info()}")
    
    # 3. 训练 VAE
    vae, vae_history = train_vae_example(
        train_loader, val_loader, design_shape, latent_dim, epochs=30
    )
    
    # 4. 训练代理模型
    surrogate, surrogate_history = train_surrogate_example(
        engine, train_loader, val_loader, epochs=20
    )
    
    # 5. 贝叶斯优化逆向设计
    target_performance = torch.tensor([[0.85, 0.80, 0.10]])
    design, opt_history = bayesian_optimization_example(
        engine, target_performance, n_iterations=25
    )
    
    # 6. 评估设计
    results = evaluate_design_example(engine, design)
    
    # 7. 多样化采样
    diverse_designs = diverse_sampling_example(engine, target_performance, n_samples=3)
    
    # 8. 潜在空间可视化
    visualize_latent_space(engine.vae, test_loader)
    
    # 9. 可视化结果
    visualize_vae_results(engine.vae, vae_history, test_loader)
    visualize_optimization_results(design, opt_history, target_performance)
    visualize_diverse_designs(diverse_designs)
    
    # 10. 保存引擎
    print("\n" + "=" * 60)
    print("Saving HiLAB Engine")
    print("=" * 60)
    
    save_path = project_root / "models" / "pretrained" / "hilab_example"
    engine.save_engine(str(save_path))
    print(f"Engine saved to: {save_path}")
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)
    
    return engine, design, results


if __name__ == "__main__":
    main()

"""
HiLAB 单元测试

测试 HiLAB 各组件的功能正确性，包括：
- VAE 编码器/解码器
- VAE 完整模型
- HiLAB 引擎
- VAE 损失函数
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from models.inverse.hilab import (
    VAEEncoder,
    VAEDecoder,
    VAE,
    VAEEncoderConfig,
    VAEDecoderConfig,
    VAEConfig,
    BayesianOptimizerConfig,
    HiLABConfig,
    HiLABEngine
)

from models.training.losses import (
    VAEReconstructionLoss,
    KLDivergenceLoss,
    BetaVAELoss,
    VAELatentRegularization,
    VAETotalLoss
)

from data.loaders.pipeline import SyntheticDataset, create_dataloaders


class TestVAEEncoder:
    """VAE 编码器测试"""
    
    @pytest.fixture
    def config(self):
        return VAEEncoderConfig(
            design_shape=(50, 11),
            latent_dim=16,
            hidden_channels=[16, 32],
            hidden_dims=[64, 32]
        )
    
    @pytest.fixture
    def encoder(self, config):
        return VAEEncoder(config)
    
    def test_encoder_output_shape(self, encoder, config):
        """测试编码器输出形状"""
        batch_size = 8
        x = torch.rand(batch_size, *config.design_shape)
        
        mu, logvar = encoder(x)
        
        assert mu.shape == (batch_size, config.latent_dim)
        assert logvar.shape == (batch_size, config.latent_dim)
    
    def test_encoder_with_channel_dim(self, encoder, config):
        """测试带通道维度的输入"""
        batch_size = 4
        x = torch.rand(batch_size, 1, *config.design_shape)
        
        mu, logvar = encoder(x)
        
        assert mu.shape == (batch_size, config.latent_dim)
    
    def test_encoder_gradient_flow(self, encoder, config):
        """测试编码器梯度流动"""
        x = torch.rand(2, *config.design_shape, requires_grad=True)
        
        mu, logvar = encoder(x)
        loss = mu.sum() + logvar.sum()
        loss.backward()
        
        assert x.grad is not None
    
    def test_logvar_range(self, encoder, config):
        """测试对数方差范围（应该合理）"""
        x = torch.rand(4, *config.design_shape)
        
        mu, logvar = encoder(x)
        
        # 对数方差不应该太大或太小
        assert logvar.max() < 100
        assert logvar.min() > -100


class TestVAEDecoder:
    """VAE 解码器测试"""
    
    @pytest.fixture
    def config(self):
        return VAEDecoderConfig(
            latent_dim=16,
            design_shape=(50, 11),
            hidden_dims=[32, 64],
            hidden_channels=[32, 16, 1]
        )
    
    @pytest.fixture
    def decoder(self, config):
        return VAEDecoder(config)
    
    def test_decoder_output_shape(self, decoder, config):
        """测试解码器输出形状"""
        batch_size = 8
        z = torch.randn(batch_size, config.latent_dim)
        
        output = decoder(z)
        
        assert output.shape == (batch_size, *config.design_shape)
    
    def test_decoder_output_range(self, decoder, config):
        """测试解码器输出范围（Sigmoid 应该在 [0, 1]）"""
        z = torch.randn(4, config.latent_dim)
        
        output = decoder(z)
        
        assert output.min() >= 0
        assert output.max() <= 1
    
    def test_decoder_gradient_flow(self, decoder, config):
        """测试解码器梯度流动"""
        z = torch.randn(2, config.latent_dim, requires_grad=True)
        
        output = decoder(z)
        loss = output.sum()
        loss.backward()
        
        assert z.grad is not None
    
    def test_decoder_deterministic(self, decoder, config):
        """测试解码器确定性（相同输入应产生相同输出）"""
        z = torch.randn(1, config.latent_dim)
        
        decoder.eval()
        output1 = decoder(z)
        output2 = decoder(z)
        
        assert torch.allclose(output1, output2)


class TestVAE:
    """VAE 完整模型测试"""
    
    @pytest.fixture
    def config(self):
        encoder_config = VAEEncoderConfig(
            design_shape=(50, 11),
            latent_dim=16,
            hidden_channels=[16, 32],
            hidden_dims=[64, 32]
        )
        
        decoder_config = VAEDecoderConfig(
            latent_dim=16,
            design_shape=(50, 11),
            hidden_dims=[32, 64],
            hidden_channels=[32, 16, 1]
        )
        
        return VAEConfig(
            encoder_config=encoder_config,
            decoder_config=decoder_config,
            latent_dim=16,
            recon_weight=1.0,
            kl_weight=0.001
        )
    
    @pytest.fixture
    def vae(self, config):
        return VAE(config)
    
    def test_vae_forward(self, vae, config):
        """测试 VAE 前向传播"""
        batch_size = 4
        x = torch.rand(batch_size, *config.encoder_config.design_shape)
        
        recon_x = vae(x)
        
        assert recon_x.shape == x.shape
    
    def test_vae_encode_decode(self, vae, config):
        """测试编码-解码流程"""
        batch_size = 4
        x = torch.rand(batch_size, *config.encoder_config.design_shape)
        
        mu, logvar = vae.encode(x)
        z = vae.reparameterize(mu, logvar)
        recon_x = vae.decode(z)
        
        assert z.shape == (batch_size, config.latent_dim)
        assert recon_x.shape == x.shape
    
    def test_vae_sample_prior(self, vae, config):
        """测试从先验采样"""
        batch_size = 10
        z = vae.sample_prior(batch_size)
        
        assert z.shape == (batch_size, config.latent_dim)
    
    def test_vae_generate(self, vae):
        """测试生成新设计"""
        num_samples = 5
        designs = vae.generate(num_samples)
        
        assert designs.shape[0] == num_samples
    
    def test_vae_interpolate(self, vae, config):
        """测试设计插值"""
        design1 = torch.rand(1, *config.encoder_config.design_shape)
        design2 = torch.rand(1, *config.encoder_config.design_shape)
        num_steps = 5
        
        interpolated = vae.interpolate(design1, design2, num_steps)
        
        assert interpolated.shape[0] == num_steps
    
    def test_vae_compute_loss(self, vae, config):
        """测试损失计算"""
        batch_size = 4
        x = torch.rand(batch_size, *config.encoder_config.design_shape)
        
        mu, logvar = vae.encode(x)
        z = vae.reparameterize(mu, logvar)
        recon_x = vae.decode(z)
        
        losses = vae.compute_loss(x, recon_x, mu, logvar)
        
        assert 'recon' in losses
        assert 'kl' in losses
        assert 'total' in losses
        assert losses['recon'].item() >= 0
        assert losses['kl'].item() >= 0
    
    def test_vae_training_short(self, vae):
        """测试简短训练"""
        dataset = SyntheticDataset(
            num_samples=50,
            design_shape=vae.design_shape,
            performance_dim=3
        )
        
        train_loader, _, _ = create_dataloaders(
            dataset, batch_size=16, train_ratio=1.0, val_ratio=0.0, test_ratio=0.0
        )
        
        history = vae.train_model(
            train_loader=train_loader,
            epochs=2,
            lr=1e-3
        )
        
        assert 'train_loss' in history
        assert len(history['train_loss']) > 0
    
    def test_vae_save_load(self, vae, tmp_path):
        """测试模型保存和加载"""
        save_path = tmp_path / "vae_test.pt"
        
        vae.save(save_path)
        assert save_path.exists()
        
        new_vae = VAE(vae.config)
        new_vae.load(save_path)
        
        x = torch.rand(1, *vae.design_shape)
        output1 = vae(x)
        output2 = new_vae(x)
        
        assert torch.allclose(output1, output2, atol=1e-5)


class TestHiLABEngine:
    """HiLAB 引擎测试"""
    
    @pytest.fixture
    def config(self):
        encoder_config = VAEEncoderConfig(
            design_shape=(50, 11),
            latent_dim=8,
            hidden_channels=[16, 32],
            hidden_dims=[64, 32]
        )
        
        decoder_config = VAEDecoderConfig(
            latent_dim=8,
            design_shape=(50, 11),
            hidden_dims=[32, 64],
            hidden_channels=[32, 16, 1]
        )
        
        vae_config = VAEConfig(
            encoder_config=encoder_config,
            decoder_config=decoder_config,
            latent_dim=8
        )
        
        optimizer_config = BayesianOptimizerConfig(
            acquisition_type='ei',
            n_restarts=3
        )
        
        return HiLABConfig(
            vae_config=vae_config,
            optimizer_config=optimizer_config,
            performance_dim=3,
            design_shape=(50, 11)
        )
    
    @pytest.fixture
    def engine(self, config):
        return HiLABEngine(config)
    
    def test_engine_creation(self, engine, config):
        """测试引擎创建"""
        assert engine.latent_dim == config.vae_config.latent_dim
        assert engine.performance_dim == config.performance_dim
    
    def test_build_surrogate(self, engine):
        """测试代理模型构建"""
        surrogate = engine.build_surrogate(
            hidden_dims=[32, 16],
            activation='relu'
        )
        
        assert surrogate is not None
        
        # 测试代理模型前向传播
        z = torch.randn(4, engine.latent_dim)
        output = surrogate(z)
        
        assert output.shape == (4, engine.performance_dim)
    
    def test_engine_forward(self, engine):
        """测试引擎前向传播"""
        # 先训练 VAE 和代理模型
        dataset = SyntheticDataset(
            num_samples=50,
            design_shape=engine.design_shape,
            performance_dim=engine.performance_dim
        )
        
        train_loader, _, _ = create_dataloaders(
            dataset, batch_size=16, train_ratio=1.0, val_ratio=0.0, test_ratio=0.0
        )
        
        # 训练 VAE
        engine.train_vae(train_loader, epochs=2)
        
        # 训练代理模型
        engine.train_surrogate(train_loader, epochs=2)
        
        # 测试条件生成
        condition = torch.rand(2, engine.performance_dim)
        designs = engine(condition)
        
        assert designs.shape[0] == 2
        assert designs.shape[1:] == engine.design_shape
    
    def test_inverse_design(self, engine):
        """测试逆向设计"""
        # 准备数据
        dataset = SyntheticDataset(
            num_samples=50,
            design_shape=engine.design_shape,
            performance_dim=engine.performance_dim
        )
        
        train_loader, _, _ = create_dataloaders(
            dataset, batch_size=16, train_ratio=1.0, val_ratio=0.0, test_ratio=0.0
        )
        
        # 训练
        engine.train_vae(train_loader, epochs=2)
        engine.train_surrogate(train_loader, epochs=2)
        
        # 逆向设计
        target = torch.tensor([[0.7, 0.6, 0.2]])
        design = engine.inverse_design(target, n_iterations=5, verbose=False)
        
        assert design.shape == (1, *engine.design_shape)
    
    def test_inverse_design_with_history(self, engine):
        """测试带历史记录的逆向设计"""
        dataset = SyntheticDataset(
            num_samples=50,
            design_shape=engine.design_shape,
            performance_dim=engine.performance_dim
        )
        
        train_loader, _, _ = create_dataloaders(
            dataset, batch_size=16, train_ratio=1.0, val_ratio=0.0, test_ratio=0.0
        )
        
        engine.train_vae(train_loader, epochs=2)
        engine.train_surrogate(train_loader, epochs=2)
        
        target = torch.tensor([[0.7, 0.6, 0.2]])
        design, history = engine.inverse_design(
            target, n_iterations=5, verbose=False, return_history=True
        )
        
        assert len(history) == 1
        assert 'best_z' in history[0]
        assert 'best_error' in history[0]
    
    def test_sample_diverse(self, engine):
        """测试多样化采样"""
        dataset = SyntheticDataset(
            num_samples=50,
            design_shape=engine.design_shape,
            performance_dim=engine.performance_dim
        )
        
        train_loader, _, _ = create_dataloaders(
            dataset, batch_size=16, train_ratio=1.0, val_ratio=0.0, test_ratio=0.0
        )
        
        engine.train_vae(train_loader, epochs=2)
        engine.train_surrogate(train_loader, epochs=2)
        
        target = torch.tensor([[0.7, 0.6, 0.2]])
        designs = engine.sample_diverse(target, n_samples=3, n_iterations=5)
        
        assert designs.shape[0] == 3
    
    def test_evaluate_design(self, engine):
        """测试设计评估"""
        dataset = SyntheticDataset(
            num_samples=50,
            design_shape=engine.design_shape,
            performance_dim=engine.performance_dim
        )
        
        train_loader, _, _ = create_dataloaders(
            dataset, batch_size=16, train_ratio=1.0, val_ratio=0.0, test_ratio=0.0
        )
        
        engine.train_vae(train_loader, epochs=2)
        engine.train_surrogate(train_loader, epochs=2)
        
        design = torch.rand(1, *engine.design_shape)
        results = engine.evaluate_design(design)
        
        assert 'reconstruction_error' in results
        assert results['reconstruction_error'] >= 0
    
    def test_latent_embedding(self, engine):
        """测试潜在空间嵌入"""
        dataset = SyntheticDataset(
            num_samples=20,
            design_shape=engine.design_shape,
            performance_dim=engine.performance_dim
        )
        
        train_loader, _, _ = create_dataloaders(
            dataset, batch_size=16, train_ratio=1.0, val_ratio=0.0, test_ratio=0.0
        )
        
        engine.train_vae(train_loader, epochs=2)
        
        design = torch.rand(4, *engine.design_shape)
        mu, logvar = engine.get_latent_embedding(design)
        
        assert mu.shape == (4, engine.latent_dim)
        assert logvar.shape == (4, engine.latent_dim)
    
    def test_engine_save_load(self, engine, tmp_path):
        """测试引擎保存和加载"""
        save_path = tmp_path / "hilab_test"
        
        engine.save_engine(str(save_path))
        assert (Path(save_path) / "vae.pt").exists()
        
        new_engine = HiLABEngine(engine.config)
        new_engine.load_engine(str(save_path))
        
        assert new_engine.latent_dim == engine.latent_dim


class TestVAELosses:
    """VAE 损失函数测试"""
    
    def test_vae_reconstruction_loss_mse(self):
        """测试 MSE 重建损失"""
        loss_fn = VAEReconstructionLoss(loss_type='mse')
        
        x = torch.rand(8, 50, 11)
        recon_x = torch.rand(8, 50, 11)
        
        loss = loss_fn(recon_x, x)
        
        assert loss.dim() == 0
        assert loss.item() >= 0
    
    def test_vae_reconstruction_loss_bce(self):
        """测试 BCE 重建损失"""
        loss_fn = VAEReconstructionLoss(loss_type='bce')
        
        x = torch.rand(8, 50, 11)
        recon_x = torch.rand(8, 50, 11)
        
        loss = loss_fn(recon_x, x)
        
        assert loss.dim() == 0
        assert loss.item() >= 0
    
    def test_vae_reconstruction_loss_l1(self):
        """测试 L1 重建损失"""
        loss_fn = VAEReconstructionLoss(loss_type='l1')
        
        x = torch.rand(8, 50, 11)
        recon_x = torch.rand(8, 50, 11)
        
        loss = loss_fn(recon_x, x)
        
        assert loss.dim() == 0
        assert loss.item() >= 0
    
    def test_kl_divergence_loss(self):
        """测试 KL 散度损失"""
        loss_fn = KLDivergenceLoss()
        
        mu = torch.randn(8, 16)
        logvar = torch.randn(8, 16)
        
        loss = loss_fn(mu, logvar)
        
        assert loss.dim() == 0
        # KL 散度应该非负
        assert loss.item() >= 0
    
    def test_kl_divergence_zero_mean_unit_var(self):
        """测试 KL 散度在零均值单位方差时应该较小"""
        loss_fn = KLDivergenceLoss()
        
        mu = torch.zeros(8, 16)
        logvar = torch.zeros(8, 16)  # var = 1
        
        loss = loss_fn(mu, logvar)
        
        # KL(N(0,1) || N(0,1)) = 0
        assert loss.item() < 0.1
    
    def test_beta_vae_loss(self):
        """测试 β-VAE 损失"""
        loss_fn = BetaVAELoss(beta=0.5, recon_type='mse')
        
        x = torch.rand(8, 50, 11)
        recon_x = torch.rand(8, 50, 11)
        mu = torch.randn(8, 16)
        logvar = torch.randn(8, 16)
        
        losses = loss_fn(x, recon_x, mu, logvar)
        
        assert 'recon' in losses
        assert 'kl' in losses
        assert 'beta' in losses
        assert 'total' in losses
    
    def test_beta_vae_warmup(self):
        """测试 β 预热"""
        loss_fn = BetaVAELoss(beta=1.0, warmup_epochs=10)
        
        # 初始 β 应该较小
        beta_0 = loss_fn.get_beta()
        
        # 推进 epoch
        for _ in range(5):
            loss_fn.step()
        
        beta_5 = loss_fn.get_beta()
        
        # β 应该增加
        assert beta_5 > beta_0
    
    def test_vae_latent_regularization_capacity(self):
        """测试容量约束正则化"""
        loss_fn = VAELatentRegularization(
            reg_type='capacity',
            capacity_start=0.0,
            capacity_end=10.0,
            capacity_epochs=100
        )
        
        mu = torch.randn(8, 16)
        logvar = torch.randn(8, 16)
        z = torch.randn(8, 16)
        
        loss = loss_fn(z, mu, logvar)
        
        assert loss.dim() == 0
    
    def test_vae_latent_regularization_mmd(self):
        """测试 MMD 正则化"""
        loss_fn = VAELatentRegularization(
            reg_type='mmd',
            mmd_kernel='rbf'
        )
        
        mu = torch.randn(8, 16)
        logvar = torch.randn(8, 16)
        z = torch.randn(8, 16)
        
        loss = loss_fn(z, mu, logvar)
        
        assert loss.dim() == 0
    
    def test_vae_total_loss(self):
        """测试 VAE 组合损失"""
        loss_fn = VAETotalLoss(
            recon_type='mse',
            beta=0.5,
            warmup_epochs=5
        )
        
        x = torch.rand(8, 50, 11)
        recon_x = torch.rand(8, 50, 11)
        mu = torch.randn(8, 16)
        logvar = torch.randn(8, 16)
        z = torch.randn(8, 16)
        
        losses = loss_fn(x, recon_x, mu, logvar, z)
        
        assert 'recon' in losses
        assert 'kl' in losses
        assert 'total' in losses


class TestVAEWithLosses:
    """测试 VAE 与损失函数的集成"""
    
    def test_vae_with_beta_vae_loss(self):
        """测试 VAE 使用 β-VAE 损失训练"""
        encoder_config = VAEEncoderConfig(
            design_shape=(50, 11),
            latent_dim=8,
            hidden_channels=[16, 32],
            hidden_dims=[32]
        )
        
        decoder_config = VAEDecoderConfig(
            latent_dim=8,
            design_shape=(50, 11),
            hidden_dims=[32],
            hidden_channels=[16, 1]
        )
        
        vae_config = VAEConfig(
            encoder_config=encoder_config,
            decoder_config=decoder_config,
            latent_dim=8
        )
        
        vae = VAE(vae_config)
        loss_fn = BetaVAELoss(beta=0.001, recon_type='mse')
        
        optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
        
        # 训练几个批次
        for _ in range(3):
            x = torch.rand(4, 50, 11)
            
            optimizer.zero_grad()
            mu, logvar = vae.encode(x)
            z = vae.reparameterize(mu, logvar)
            recon_x = vae.decode(z)
            
            losses = loss_fn(x, recon_x, mu, logvar)
            losses['total'].backward()
            
            optimizer.step()
        
        # 验证模型仍然工作
        x_test = torch.rand(1, 50, 11)
        recon = vae(x_test)
        
        assert recon.shape == x_test.shape


class TestBayesianOptimizerConfig:
    """贝叶斯优化器配置测试"""
    
    def test_config_defaults(self):
        """测试默认配置"""
        config = BayesianOptimizerConfig()
        
        assert config.kernel_type == 'rbf'
        assert config.acquisition_type == 'ei'
        assert config.n_restarts == 10
    
    def test_config_custom(self):
        """测试自定义配置"""
        config = BayesianOptimizerConfig(
            kernel_type='matern',
            acquisition_type='ucb',
            ucb_beta=3.0,
            latent_bounds=(-5.0, 5.0)
        )
        
        assert config.kernel_type == 'matern'
        assert config.acquisition_type == 'ucb'
        assert config.ucb_beta == 3.0
        assert config.latent_bounds == (-5.0, 5.0)


class TestHiLABConfig:
    """HiLAB 配置测试"""
    
    def test_config_defaults(self):
        """测试默认配置"""
        config = HiLABConfig()
        
        assert config.name == 'hilab'
        assert config.performance_dim == 3
        assert config.use_adjoint_refinement == False
    
    def test_config_nested(self):
        """测试嵌套配置"""
        vae_config = VAEConfig(latent_dim=64)
        optimizer_config = BayesianOptimizerConfig(acquisition_type='ucb')
        
        config = HiLABConfig(
            vae_config=vae_config,
            optimizer_config=optimizer_config,
            performance_dim=5
        )
        
        assert config.vae_config.latent_dim == 64
        assert config.optimizer_config.acquisition_type == 'ucb'
        assert config.performance_dim == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

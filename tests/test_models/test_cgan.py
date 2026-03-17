"""
CGAN 模型单元测试

测试条件生成对抗网络的核心功能。
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys

# 添加项目根目录
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from models.inverse.cgan import (
    CGAN, CGANConfig,
    GeneratorConfig, DiscriminatorConfig,
    ConditionalGenerator, ConditionalDiscriminator,
    WGAN_GP
)
from models.training.losses import GANLoss, GradientPenaltyLoss, CGANCombinedLoss


class TestGeneratorConfig:
    """测试生成器配置"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = GeneratorConfig()
        
        assert config.latent_dim == 128
        assert config.condition_dim == 3
        assert config.design_shape == (200, 22)
        assert config.output_activation == 'tanh'
    
    def test_custom_config(self):
        """测试自定义配置"""
        config = GeneratorConfig(
            latent_dim=64,
            condition_dim=5,
            design_shape=(100, 20),
            output_activation='sigmoid'
        )
        
        assert config.latent_dim == 64
        assert config.condition_dim == 5
        assert config.design_shape == (100, 20)
        assert config.output_activation == 'sigmoid'


class TestDiscriminatorConfig:
    """测试判别器配置"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = DiscriminatorConfig()
        
        assert config.design_shape == (200, 22)
        assert config.condition_dim == 3
        assert config.spectral_norm == True
    
    def test_condition_fusion_options(self):
        """测试条件融合选项"""
        for fusion in ['concat', 'film', 'attention']:
            config = DiscriminatorConfig(condition_fusion=fusion)
            assert config.condition_fusion == fusion


class TestConditionalGenerator:
    """测试条件生成器"""
    
    @pytest.fixture
    def generator(self):
        """创建测试生成器"""
        config = GeneratorConfig(
            latent_dim=64,
            condition_dim=3,
            design_shape=(64, 16),
            hidden_dims=[128, 256],
            hidden_channels=[128, 64]
        )
        return ConditionalGenerator(config)
    
    def test_initialization(self, generator):
        """测试初始化"""
        assert generator.latent_dim == 64
        assert generator.condition_dim == 3
        assert generator.design_shape == (64, 16)
    
    def test_forward_shape(self, generator):
        """测试前向传播形状"""
        batch_size = 4
        condition = torch.randn(batch_size, 3)
        
        output = generator(condition)
        
        assert output.shape == (batch_size, 64, 16)
    
    def test_forward_with_noise(self, generator):
        """测试使用自定义噪声"""
        batch_size = 4
        condition = torch.randn(batch_size, 3)
        noise = torch.randn(batch_size, 64)
        
        output = generator(condition, noise)
        
        assert output.shape == (batch_size, 64, 16)
    
    def test_sample_multiple(self, generator):
        """测试多样本生成"""
        batch_size = 2
        condition = torch.randn(batch_size, 3)
        num_samples = 5
        
        outputs = generator.sample(condition, num_samples)
        
        assert outputs.shape == (batch_size * num_samples, 64, 16)
    
    def test_output_range_sigmoid(self):
        """测试 sigmoid 输出范围"""
        config = GeneratorConfig(
            design_shape=(32, 8),
            output_activation='sigmoid'
        )
        gen = ConditionalGenerator(config)
        
        condition = torch.randn(4, 3)
        output = gen(condition)
        
        assert output.min() >= 0
        assert output.max() <= 1
    
    def test_output_range_tanh(self):
        """测试 tanh 输出范围"""
        config = GeneratorConfig(
            design_shape=(32, 8),
            output_activation='tanh'
        )
        gen = ConditionalGenerator(config)
        
        condition = torch.randn(4, 3)
        output = gen(condition)
        
        assert output.min() >= -1
        assert output.max() <= 1
    
    def test_count_parameters(self, generator):
        """测试参数计数"""
        count = generator.count_parameters()
        
        assert count > 0
        assert isinstance(count, int)


class TestConditionalDiscriminator:
    """测试条件判别器"""
    
    @pytest.fixture
    def discriminator(self):
        """创建测试判别器"""
        config = DiscriminatorConfig(
            design_shape=(64, 16),
            condition_dim=3,
            hidden_channels=[32, 64, 128],
            hidden_dims=[256, 128, 1]
        )
        return ConditionalDiscriminator(config)
    
    def test_initialization(self, discriminator):
        """测试初始化"""
        assert discriminator.design_shape == (64, 16)
        assert discriminator.condition_dim == 3
    
    def test_forward_shape(self, discriminator):
        """测试前向传播形状"""
        batch_size = 4
        design = torch.rand(batch_size, 64, 16)
        condition = torch.randn(batch_size, 3)
        
        output = discriminator(design, condition)
        
        assert output.shape == (batch_size, 1)
    
    def test_forward_with_channel_dim(self, discriminator):
        """测试带通道维度的输入"""
        batch_size = 4
        design = torch.rand(batch_size, 1, 64, 16)  # 带通道维度
        condition = torch.randn(batch_size, 3)
        
        output = discriminator(design, condition)
        
        assert output.shape == (batch_size, 1)


class TestCGAN:
    """测试完整 CGAN 模型"""
    
    @pytest.fixture
    def cgan(self):
        """创建测试 CGAN"""
        gen_config = GeneratorConfig(
            latent_dim=32,
            condition_dim=3,
            design_shape=(32, 8),
            hidden_dims=[64, 128],
            hidden_channels=[64, 32]
        )
        
        disc_config = DiscriminatorConfig(
            design_shape=(32, 8),
            condition_dim=3,
            hidden_channels=[16, 32, 64],
            hidden_dims=[128, 64, 1]
        )
        
        config = CGANConfig(
            generator_config=gen_config,
            discriminator_config=disc_config,
            gan_type='wgan-gp'
        )
        
        return CGAN(config)
    
    def test_initialization(self, cgan):
        """测试初始化"""
        assert cgan.latent_dim == 32
        assert cgan.condition_dim == 3
        assert cgan.design_shape == (32, 8)
    
    def test_generate_shape(self, cgan):
        """测试生成形状"""
        batch_size = 4
        condition = torch.randn(batch_size, 3)
        
        designs = cgan.generate(condition)
        
        assert designs.shape == (batch_size, 32, 8)
    
    def test_generate_multiple(self, cgan):
        """测试多样本生成"""
        batch_size = 2
        condition = torch.randn(batch_size, 3)
        num_samples = 3
        
        designs = cgan.generate(condition, num_samples)
        
        assert designs.shape == (batch_size * num_samples, 32, 8)
    
    def test_discriminate(self, cgan):
        """测试判别功能"""
        design = torch.rand(4, 32, 8)
        condition = torch.randn(4, 3)
        
        validity = cgan.discriminate(design, condition)
        
        assert validity.shape == (4, 1)
    
    def test_train_model(self, cgan, tmp_path):
        """测试训练流程"""
        # 创建简单数据集
        designs = torch.rand(20, 32, 8)
        performances = torch.rand(20, 3)
        
        dataset = torch.utils.data.TensorDataset(designs, performances)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=4)
        
        # 简短训练
        history = cgan.train_model(
            dataloader,
            epochs=2,
            log_interval=1
        )
        
        assert 'g_loss' in history
        assert 'd_loss' in history
        assert len(history['g_loss']) == 2
    
    def test_save_load(self, cgan, tmp_path):
        """测试保存和加载"""
        # 保存
        save_path = tmp_path / 'cgan_test.pth'
        cgan.save(save_path)
        
        assert save_path.exists()
        
        # 加载
        cgan_loaded = CGAN(cgan.config)
        cgan_loaded.load(save_path)
        
        # 验证
        condition = torch.randn(2, 3)
        original_output = cgan.generate(condition)
        loaded_output = cgan_loaded.generate(condition)
        
        assert torch.allclose(original_output, loaded_output, atol=1e-5)
    
    def test_get_model_info(self, cgan):
        """测试获取模型信息"""
        info = cgan.get_model_info()
        
        assert 'name' in info
        assert 'design_shape' in info
        assert 'condition_dim' in info
        assert 'latent_dim' in info
        assert 'generator_parameters' in info
        assert 'discriminator_parameters' in info


class TestWGANGP:
    """测试 WGAN-GP 便捷类"""
    
    def test_initialization(self):
        """测试初始化"""
        wgan = WGAN_GP(
            design_shape=(64, 16),
            condition_dim=3,
            latent_dim=64
        )
        
        assert wgan.design_shape == (64, 16)
        assert wgan.condition_dim == 3
        assert wgan.config.gan_type == 'wgan-gp'
    
    def test_generate(self):
        """测试生成"""
        wgan = WGAN_GP(design_shape=(32, 8))
        
        condition = torch.randn(4, 3)
        designs = wgan.generate(condition)
        
        assert designs.shape == (4, 32, 8)


class TestGANLoss:
    """测试 GAN 损失函数"""
    
    def test_gan_loss(self):
        """测试标准 GAN 损失"""
        loss_fn = GANLoss(gan_type='gan')
        
        real = torch.randn(4, 1)
        fake = torch.randn(4, 1)
        
        d_loss = loss_fn.discriminator_loss(real, fake)
        g_loss = loss_fn.generator_loss(fake)
        
        assert d_loss.item() > 0
        assert g_loss.item() > 0
    
    def test_wgan_loss(self):
        """测试 Wasserstein 损失"""
        loss_fn = GANLoss(gan_type='wgan')
        
        real = torch.randn(4, 1)
        fake = torch.randn(4, 1)
        
        d_loss = loss_fn.discriminator_loss(real, fake)
        g_loss = loss_fn.generator_loss(fake)
        
        assert isinstance(d_loss, torch.Tensor)
        assert isinstance(g_loss, torch.Tensor)
    
    def test_lsgan_loss(self):
        """测试 LSGAN 损失"""
        loss_fn = GANLoss(gan_type='lsgan')
        
        real = torch.randn(4, 1)
        fake = torch.randn(4, 1)
        
        d_loss = loss_fn.discriminator_loss(real, fake)
        g_loss = loss_fn.generator_loss(fake)
        
        assert d_loss.item() >= 0
        assert g_loss.item() >= 0
    
    def test_hinge_loss(self):
        """测试 Hinge 损失"""
        loss_fn = GANLoss(gan_type='hinge')
        
        real = torch.randn(4, 1)
        fake = torch.randn(4, 1)
        
        d_loss = loss_fn.discriminator_loss(real, fake)
        g_loss = loss_fn.generator_loss(fake)
        
        assert d_loss.item() >= 0


class TestGradientPenalty:
    """测试梯度惩罚"""
    
    def test_gradient_penalty(self):
        """测试梯度惩罚计算"""
        gp_fn = GradientPenaltyLoss(lambda_gp=10.0)
        
        # 创建简单判别器
        class SimpleDisc(nn.Module):
            def forward(self, x, c):
                return x.mean(dim=(1, 2), keepdim=True)
        
        disc = SimpleDisc()
        real = torch.rand(4, 32, 8)
        fake = torch.rand(4, 32, 8)
        condition = torch.randn(4, 3)
        
        gp = gp_fn(disc, real, fake, condition)
        
        assert gp.item() >= 0


class TestCGANCombinedLoss:
    """测试 CGAN 组合损失"""
    
    def test_combined_loss(self):
        """测试组合损失计算"""
        loss_fn = CGANCombinedLoss(
            gan_type='wgan-gp',
            condition_weight=1.0
        )
        
        fake_validity = torch.randn(4, 1)
        generated_perf = torch.rand(4, 3)
        target_cond = torch.rand(4, 3)
        
        losses = loss_fn(
            fake_validity,
            generated_perf,
            target_cond
        )
        
        assert 'gan' in losses
        assert 'condition' in losses
        assert 'total' in losses
    
    def test_diversity_loss(self):
        """测试多样性损失"""
        loss_fn = CGANCombinedLoss(
            gan_type='wgan-gp',
            diversity_weight=0.1
        )
        
        fake_validity = torch.randn(4, 1)
        designs = torch.rand(4, 32, 8)
        
        losses = loss_fn(fake_validity, designs=designs)
        
        assert 'diversity' in losses


class TestIntegration:
    """集成测试"""
    
    def test_full_training_cycle(self, tmp_path):
        """测试完整训练周期"""
        # 创建 CGAN
        gen_config = GeneratorConfig(
            latent_dim=32,
            condition_dim=3,
            design_shape=(32, 8),
            hidden_dims=[64],
            hidden_channels=[32]
        )
        
        disc_config = DiscriminatorConfig(
            design_shape=(32, 8),
            condition_dim=3,
            hidden_channels=[16, 32],
            hidden_dims=[64, 1]
        )
        
        config = CGANConfig(
            generator_config=gen_config,
            discriminator_config=disc_config,
            gan_type='wgan-gp',
            n_critic=2
        )
        
        cgan = CGAN(config)
        
        # 创建数据
        designs = torch.rand(16, 32, 8)
        performances = torch.rand(16, 3)
        
        dataset = torch.utils.data.TensorDataset(designs, performances)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=4)
        
        # 训练
        history = cgan.train_model(
            dataloader,
            epochs=3,
            save_path=str(tmp_path / 'cgan'),
            log_interval=1
        )
        
        # 验证训练完成
        assert len(history['g_loss']) == 3
        
        # 测试生成
        condition = torch.randn(2, 3)
        designs = cgan.generate(condition, num_samples=3)
        
        assert designs.shape == (6, 32, 8)
        
        # 测试保存
        cgan.save(tmp_path / 'final_model.pth')
        assert (tmp_path / 'final_model.pth').exists()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

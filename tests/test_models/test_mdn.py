"""
MDN（混合密度网络）单元测试
"""

import pytest
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from models.inverse.mdn import (
    MDN, MDNConfig,
    GaussianMixtureDistribution,
    GaussianMixtureParameters,
    MDNTandemNetwork
)
from models.training.losses import MDNLoss, MDNRegularizedLoss


class TestGaussianMixtureParameters:
    """测试高斯混合参数层"""
    
    def test_output_shapes(self):
        """测试输出形状"""
        batch_size = 8
        input_dim = 32
        output_dim = 100
        n_components = 5
        
        layer = GaussianMixtureParameters(input_dim, output_dim, n_components)
        features = torch.randn(batch_size, input_dim)
        
        pi, mu, sigma = layer(features)
        
        assert pi.shape == (batch_size, n_components)
        assert mu.shape == (batch_size, n_components, output_dim)
        assert sigma.shape == (batch_size, n_components, output_dim)
    
    def test_pi_normalization(self):
        """测试混合权重归一化"""
        layer = GaussianMixtureParameters(32, 100, 5)
        features = torch.randn(8, 32)
        
        pi, _, _ = layer(features)
        
        # 检查权重和为1
        assert torch.allclose(pi.sum(dim=-1), torch.ones(8), atol=1e-5)
        
        # 检查所有权重非负
        assert (pi >= 0).all()
    
    def test_sigma_positive(self):
        """测试标准差为正"""
        layer = GaussianMixtureParameters(32, 100, 5)
        features = torch.randn(8, 32)
        
        _, _, sigma = layer(features)
        
        assert (sigma > 0).all()


class TestGaussianMixtureDistribution:
    """测试高斯混合分布"""
    
    def test_log_prob_shape(self):
        """测试对数概率形状"""
        batch_size = 4
        n_components = 3
        output_dim = 50
        
        pi = torch.softmax(torch.randn(batch_size, n_components), dim=-1)
        mu = torch.randn(batch_size, n_components, output_dim)
        sigma = torch.exp(torch.randn(batch_size, n_components, output_dim))
        
        distribution = GaussianMixtureDistribution(pi, mu, sigma)
        x = torch.randn(batch_size, output_dim)
        
        log_prob = distribution.log_prob(x)
        
        assert log_prob.shape == (batch_size,)
    
    def test_sample_shape(self):
        """测试采样形状"""
        batch_size = 4
        n_components = 3
        output_dim = 50
        n_samples = 10
        
        pi = torch.softmax(torch.randn(batch_size, n_components), dim=-1)
        mu = torch.randn(batch_size, n_components, output_dim)
        sigma = torch.exp(torch.randn(batch_size, n_components, output_dim))
        
        distribution = GaussianMixtureDistribution(pi, mu, sigma)
        samples = distribution.sample(n_samples)
        
        assert samples.shape == (batch_size, n_samples, output_dim)
    
    def test_sample_mode_shape(self):
        """测试众数采样形状"""
        batch_size = 4
        n_components = 3
        output_dim = 50
        
        pi = torch.softmax(torch.randn(batch_size, n_components), dim=-1)
        mu = torch.randn(batch_size, n_components, output_dim)
        sigma = torch.exp(torch.randn(batch_size, n_components, output_dim))
        
        distribution = GaussianMixtureDistribution(pi, mu, sigma)
        mode = distribution.sample_mode()
        
        assert mode.shape == (batch_size, output_dim)
    
    def test_entropy_shape(self):
        """测试熵形状"""
        batch_size = 4
        n_components = 3
        output_dim = 50
        
        pi = torch.softmax(torch.randn(batch_size, n_components), dim=-1)
        mu = torch.randn(batch_size, n_components, output_dim)
        sigma = torch.exp(torch.randn(batch_size, n_components, output_dim))
        
        distribution = GaussianMixtureDistribution(pi, mu, sigma)
        entropy = distribution.get_entropy()
        
        assert entropy.shape == (batch_size,)
        assert (entropy >= 0).all()  # 熵非负


class TestMDN:
    """测试 MDN 网络"""
    
    @pytest.fixture
    def mdn_config(self):
        """MDN 配置"""
        return MDNConfig(
            input_dim=3,
            output_dim=100,
            design_shape=(10, 10),
            n_components=5,
            hidden_dims=[64, 128, 64]
        )
    
    @pytest.fixture
    def mdn(self, mdn_config):
        """MDN 模型"""
        return MDN(mdn_config)
    
    def test_forward_shape(self, mdn):
        """测试前向传播形状"""
        batch_size = 8
        condition = torch.randn(batch_size, 3)
        
        pi, mu, sigma = mdn(condition)
        
        assert pi.shape == (batch_size, 5)
        assert mu.shape == (batch_size, 5, 100)
        assert sigma.shape == (batch_size, 5, 100)
    
    def test_sample_shape(self, mdn):
        """测试采样形状"""
        batch_size = 4
        n_samples = 10
        condition = torch.randn(batch_size, 3)
        
        samples = mdn.sample(condition, n_samples)
        
        # 应该重塑为设计形状
        assert samples.shape == (batch_size, n_samples, 10, 10)
    
    def test_sample_mode_shape(self, mdn):
        """测试众数采样形状"""
        batch_size = 4
        condition = torch.randn(batch_size, 3)
        
        design = mdn.sample_mode(condition)
        
        assert design.shape == (batch_size, 10, 10)
    
    def test_compute_loss(self, mdn):
        """测试损失计算"""
        condition = torch.randn(8, 3)
        target = torch.randn(8, 10, 10)
        
        loss = mdn.compute_loss(condition, target)
        
        assert loss.ndim == 0  # 标量
        assert not torch.isnan(loss)
    
    def test_get_distribution(self, mdn):
        """测试获取分布"""
        condition = torch.randn(4, 3)
        
        distribution = mdn.get_distribution(condition)
        
        assert isinstance(distribution, GaussianMixtureDistribution)
        assert distribution.batch_size == 4
        assert distribution.n_components == 5
    
    def test_model_info(self, mdn):
        """测试模型信息"""
        info = mdn.get_model_info()
        
        assert 'name' in info
        assert 'parameters' in info
        assert info['n_components'] == 5


class TestMDNLoss:
    """测试 MDN 损失函数"""
    
    def test_mdn_loss_shape(self):
        """测试 MDN 损失形状"""
        loss_fn = MDNLoss()
        
        batch_size = 8
        n_components = 5
        output_dim = 100
        
        pi = torch.softmax(torch.randn(batch_size, n_components), dim=-1)
        mu = torch.randn(batch_size, n_components, output_dim)
        sigma = torch.exp(torch.randn(batch_size, n_components, output_dim))
        target = torch.randn(batch_size, output_dim)
        
        loss = loss_fn(pi, mu, sigma, target)
        
        assert loss.ndim == 0
        assert not torch.isnan(loss)
    
    def test_mdn_loss_gradient(self):
        """测试 MDN 损失梯度"""
        loss_fn = MDNLoss()
        
        batch_size = 4
        n_components = 3
        output_dim = 50
        
        pi = torch.softmax(torch.randn(batch_size, n_components, dim=-1), requires_grad=True)
        mu = torch.randn(batch_size, n_components, output_dim, requires_grad=True)
        sigma = torch.exp(torch.randn(batch_size, n_components, output_dim))
        sigma.requires_grad_(True)
        target = torch.randn(batch_size, output_dim)
        
        loss = loss_fn(pi, mu, sigma, target)
        loss.backward()
        
        assert pi.grad is not None
        assert mu.grad is not None
        assert sigma.grad is not None
    
    def test_regularized_loss(self):
        """测试正则化损失"""
        loss_fn = MDNRegularizedLoss(balance_weight=0.01, entropy_weight=0.001)
        
        batch_size = 8
        n_components = 5
        output_dim = 100
        
        pi = torch.softmax(torch.randn(batch_size, n_components), dim=-1)
        mu = torch.randn(batch_size, n_components, output_dim)
        sigma = torch.exp(torch.randn(batch_size, n_components, output_dim))
        target = torch.randn(batch_size, output_dim)
        
        loss = loss_fn(pi, mu, sigma, target)
        
        assert loss.ndim == 0
        assert not torch.isnan(loss)


class TestMDNTandemNetwork:
    """测试 MDN-TNN 联合模型"""
    
    def test_tandem_creation(self):
        """测试联合模型创建"""
        # 创建简单的前向网络
        forward_net = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(100, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 3)
        )
        
        mdn_config = MDNConfig(
            input_dim=3,
            output_dim=100,
            design_shape=(10, 10),
            n_components=3
        )
        
        tandem = MDNTandemNetwork(forward_net, mdn_config)
        
        assert tandem.forward_net is not None
        assert tandem.mdn is not None
    
    def test_sample_best(self):
        """测试最优采样"""
        # 创建前向网络
        forward_net = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(100, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 3)
        )
        
        mdn_config = MDNConfig(
            input_dim=3,
            output_dim=100,
            design_shape=(10, 10),
            n_components=3
        )
        
        tandem = MDNTandemNetwork(forward_net, mdn_config)
        
        target_perf = torch.randn(2, 3)
        best_design, best_perf = tandem.sample_best(target_perf, n_samples=5)
        
        assert best_design.shape == (2, 10, 10)
        assert best_perf.shape == (2, 3)


class TestMDNTraining:
    """测试 MDN 训练"""
    
    def test_training_step(self):
        """测试训练步骤"""
        config = MDNConfig(
            input_dim=3,
            output_dim=100,
            design_shape=(10, 10),
            n_components=3,
            hidden_dims=[32, 64, 32]
        )
        
        mdn = MDN(config)
        optimizer = torch.optim.Adam(mdn.parameters(), lr=1e-3)
        
        # 创建模拟数据
        performances = torch.randn(32, 3)
        designs = torch.randn(32, 10, 10)
        
        # 训练步骤
        optimizer.zero_grad()
        loss = mdn.compute_loss(performances, designs)
        loss.backward()
        optimizer.step()
        
        assert not torch.isnan(loss)
    
    def test_overfit_small_data(self):
        """测试小数据过拟合"""
        config = MDNConfig(
            input_dim=2,
            output_dim=20,
            design_shape=(4, 5),
            n_components=2,
            hidden_dims=[32, 32]
        )
        
        mdn = MDN(config)
        
        # 小数据集
        performances = torch.randn(10, 2)
        designs = torch.randn(10, 4, 5)
        
        dataset = TensorDataset(performances, designs)
        loader = DataLoader(dataset, batch_size=10)
        
        # 训练几轮
        optimizer = torch.optim.Adam(mdn.parameters(), lr=1e-2)
        initial_loss = None
        final_loss = None
        
        for epoch in range(20):
            for perf, des in loader:
                optimizer.zero_grad()
                loss = mdn.compute_loss(perf, des)
                loss.backward()
                optimizer.step()
                
                if epoch == 0:
                    initial_loss = loss.item()
                final_loss = loss.item()
        
        # 损失应该下降
        assert final_loss < initial_loss


class TestNumericalStability:
    """测试数值稳定性"""
    
    def test_extreme_sigma_values(self):
        """测试极端 sigma 值"""
        config = MDNConfig(
            input_dim=3,
            output_dim=50,
            n_components=3,
            log_sigma_min=-5.0,
            log_sigma_max=1.0
        )
        
        mdn = MDN(config)
        
        # 极端输入
        extreme_input = torch.randn(4, 3) * 100
        
        pi, mu, sigma = mdn(extreme_input)
        
        # sigma 应该在合理范围内
        assert (sigma > 0).all()
        assert not torch.isnan(sigma).any()
        assert not torch.isinf(sigma).any()
    
    def test_small_sigma(self):
        """测试小 sigma 值"""
        loss_fn = MDNLoss(epsilon=1e-8)
        
        pi = torch.tensor([[0.5, 0.5]])
        mu = torch.zeros(1, 2, 10)
        sigma = torch.ones(1, 2, 10) * 0.001  # 很小的 sigma
        target = torch.zeros(1, 10)
        
        loss = loss_fn(pi, mu, sigma, target)
        
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

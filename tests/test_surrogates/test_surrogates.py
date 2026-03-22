"""
代理模型测试

测试 CNN Surrogate, DeepONet, PINO 等代理模型的功能正确性。
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


# ============================================================================
# CNN Surrogate Tests
# ============================================================================

class TestCNNSurrogate:
    """CNN 代理模型测试"""
    
    @pytest.fixture
    def config(self):
        from models.surrogates.cnn_surrogate import CNNSurrogateConfig
        return CNNSurrogateConfig(
            input_channels=1,
            input_height=64,
            input_width=64,
            output_dim=3,
            base_channels=32,
            num_blocks=2,
        )
    
    @pytest.fixture
    def model(self, config):
        from models.surrogates.cnn_surrogate import CNNSurrogate
        return CNNSurrogate(config)
    
    def test_forward_output_shape(self, model, config):
        """测试输出形状"""
        batch_size = 4
        design = torch.rand(batch_size, config.input_channels, 
                           config.input_height, config.input_width)
        
        output = model(design)
        
        assert output.shape == (batch_size, config.output_dim)
    
    def test_forward_without_batch(self, model, config):
        """测试无批次维度输入"""
        design = torch.rand(config.input_height, config.input_width)
        
        output = model(design)
        
        assert output.dim() == 1
        assert output.size(0) == config.output_dim
    
    def test_uncertainty_prediction(self, model, config):
        """测试不确定性预测"""
        batch_size = 2
        design = torch.rand(batch_size, config.input_channels,
                           config.input_height, config.input_width)
        
        mean, uncertainty = model(design, return_uncertainty=True)
        
        assert mean.shape == (batch_size, config.output_dim)
        assert uncertainty.shape == (batch_size, config.output_dim)
        assert (uncertainty >= 0).all()
    
    def test_gradient_flow(self, model, config):
        """测试梯度流动"""
        batch_size = 2
        design = torch.rand(batch_size, config.input_channels,
                           config.input_height, config.input_width, 
                           requires_grad=True)
        
        output = model(design)
        loss = output.sum()
        loss.backward()
        
        assert design.grad is not None
    
    def test_compute_loss(self, model, config):
        """测试损失计算"""
        batch_size = 4
        output = torch.rand(batch_size, config.output_dim)
        target = torch.rand(batch_size, config.output_dim)
        log_var = torch.zeros(batch_size, config.output_dim)
        
        # MSE 损失
        loss_mse = model.compute_loss(output, target)
        assert loss_mse.item() >= 0
        
        # 不确定性损失
        loss_uncertainty = model.compute_loss(output, target, log_var)
        assert loss_uncertainty.item() >= 0
    
    def test_compute_metrics(self, model, config):
        """测试指标计算"""
        batch_size = 10
        output = torch.rand(batch_size, config.output_dim)
        target = torch.rand(batch_size, config.output_dim)
        
        metrics = model.compute_metrics(output, target)
        
        assert 'mse' in metrics
        assert 'mae' in metrics
        assert 'r2' in metrics
        assert metrics['mse'] >= 0
        assert metrics['mae'] >= 0
    
    def test_mc_dropout_uncertainty(self, model, config):
        """测试 MC Dropout 不确定性估计"""
        design = torch.rand(1, config.input_channels,
                           config.input_height, config.input_width)
        
        mean, std = model.predict_with_uncertainty(design, num_samples=10)
        
        assert mean.shape == (1, config.output_dim)
        assert std.shape == (1, config.output_dim)


# ============================================================================
# DeepONet Tests
# ============================================================================

class TestDeepONet:
    """DeepONet 测试"""
    
    @pytest.fixture
    def config(self):
        from models.surrogates.deeponet import DeepONetConfig
        return DeepONetConfig(
            branch_input_dim=100,
            trunk_input_dim=2,
            branch_output_dim=32,
            trunk_output_dim=32,
            branch_hidden_dims=[64, 64],
            trunk_hidden_dims=[64, 64],
        )
    
    @pytest.fixture
    def model(self, config):
        from models.surrogates.deeponet import DeepONet
        return DeepONet(config)
    
    def test_forward_output_shape(self, model, config):
        """测试输出形状"""
        batch_size = 4
        u = torch.rand(batch_size, config.branch_input_dim)  # 输入函数
        y = torch.rand(10, config.trunk_input_dim)  # 输出位置
        
        output = model(u, y)
        
        assert output.shape == (batch_size, 10)
    
    def test_branch_network(self, model, config):
        """测试 Branch 网络"""
        batch_size = 4
        u = torch.rand(batch_size, config.branch_input_dim)
        
        b = model.branch(u)
        
        assert b.shape == (batch_size, config.branch_output_dim)
    
    def test_trunk_network(self, model, config):
        """测试 Trunk 网络"""
        y = torch.rand(10, config.trunk_input_dim)
        
        t = model.trunk(y)
        
        assert t.shape == (10, config.trunk_output_dim)
    
    def test_forward_field(self, model, config):
        """测试场求解"""
        batch_size = 2
        u = torch.rand(batch_size, config.branch_input_dim)
        
        # 创建网格
        H, W = 16, 16
        grid = torch.stack(torch.meshgrid(
            torch.linspace(0, 1, H),
            torch.linspace(0, 1, W),
            indexing='ij'
        ), dim=-1)
        
        output = model.forward_field(u, grid)
        
        assert output.shape == (batch_size, H, W)
    
    def test_gradient_flow(self, model, config):
        """测试梯度流动"""
        u = torch.rand(2, config.branch_input_dim, requires_grad=True)
        y = torch.rand(5, config.trunk_input_dim)
        
        output = model(u, y)
        loss = output.sum()
        loss.backward()
        
        assert u.grad is not None


class TestVariationalDeepONet:
    """变分 DeepONet 测试"""
    
    @pytest.fixture
    def model(self):
        from models.surrogates.deeponet import VariationalDeepONet, DeepONetConfig
        config = DeepONetConfig(
            branch_input_dim=50,
            trunk_input_dim=2,
            branch_output_dim=16,
            trunk_output_dim=16,
        )
        return VariationalDeepONet(config)
    
    def test_variance_prediction(self, model):
        """测试方差预测"""
        u = torch.rand(2, 50)
        y = torch.rand(5, 2)
        
        mean, var = model(u, y, return_variance=True)
        
        assert mean.shape == (2, 5)
        assert var.shape == (2, 5)
        assert (var >= 0).all()


class TestBranchedDeepONet:
    """分支 DeepONet 测试"""
    
    @pytest.fixture
    def model(self):
        from models.surrogates.deeponet import BranchedDeepONet, DeepONetConfig
        config = DeepONetConfig(
            branch_input_dim=50,
            trunk_input_dim=2,
            branch_output_dim=16,
            trunk_output_dim=16,
            output_fields=['Ex', 'Ey', 'Ez'],
            num_outputs=3,
        )
        return BranchedDeepONet(config)
    
    def test_multi_field_output(self, model):
        """测试多场输出"""
        u = torch.rand(2, 50)
        y = torch.rand(5, 2)
        
        output = model(u, y)
        
        assert output.shape == (2, 5, 3)  # [batch, positions, fields]


# ============================================================================
# PINO Tests
# ============================================================================

class TestPINO:
    """PINO 测试"""
    
    @pytest.fixture
    def config(self):
        from models.surrogates.pino import PINOConfig
        return PINOConfig(
            input_channels=1,
            output_channels=1,
            domain_size=(5.0, 5.0),
            grid_size=(32, 32),
            fno_modes=8,
            fno_hidden_dim=32,
            fno_num_layers=2,
            pde_type="helmholtz",
            wavelength=1.55,
        )
    
    @pytest.fixture
    def model(self, config):
        from models.surrogates.pino import PINO
        return PINO(config)
    
    def test_forward_output_shape(self, model, config):
        """测试输出形状"""
        batch_size = 2
        design = torch.rand(batch_size, config.input_channels,
                           *config.grid_size)
        
        output = model(design)
        
        assert output.shape == (batch_size, config.output_channels,
                               *config.grid_size)
    
    def test_pde_residual(self, model, config):
        """测试 PDE 残差计算"""
        batch_size = 2
        design = torch.rand(batch_size, config.input_channels, *config.grid_size)
        u = torch.rand(batch_size, config.output_channels, *config.grid_size)
        
        residual = model.compute_pde_residual(u, design)
        
        assert residual.shape == u.shape
    
    def test_physics_loss(self, model, config):
        """测试物理损失"""
        design = torch.rand(2, config.input_channels, *config.grid_size)
        
        output = model(design)
        loss = model.compute_loss(output, design=design, use_physics=True)
        
        assert loss.item() >= 0
    
    def test_gradient_flow(self, model, config):
        """测试梯度流动"""
        design = torch.rand(2, config.input_channels, *config.grid_size,
                           requires_grad=True)
        
        output = model(design)
        loss = output.sum()
        loss.backward()
        
        assert design.grad is not None


class TestFNO2d:
    """FNO 2D 测试"""
    
    @pytest.fixture
    def model(self):
        from models.surrogates.pino import FNO2d
        return FNO2d(
            in_channels=1,
            out_channels=1,
            modes=8,
            hidden_dim=32,
            num_layers=2,
        )
    
    def test_forward_shape(self, model):
        """测试输出形状"""
        x = torch.rand(2, 1, 32, 32)
        output = model(x)
        assert output.shape == (2, 1, 32, 32)
    
    def test_different_sizes(self, model):
        """测试不同输入尺寸"""
        x = torch.rand(2, 1, 64, 64)
        output = model(x)
        assert output.shape == (2, 1, 64, 64)


class TestHelmholtzPDE:
    """Helmholtz 方程测试"""
    
    @pytest.fixture
    def pde(self):
        from models.surrogates.pino import HelmholtzPDE
        return HelmholtzPDE(wavelength=1.55, domain_size=(5.0, 5.0))
    
    def test_laplacian_computation(self, pde):
        """测试拉普拉斯计算"""
        u = torch.rand(1, 1, 32, 32)
        laplacian = pde.compute_laplacian(u, pde.dx, pde.dy)
        
        assert laplacian.shape == u.shape
    
    def test_residual_shape(self, pde):
        """测试残差形状"""
        u = torch.rand(1, 1, 32, 32)
        n = torch.ones(1, 1, 32, 32) * 3.48  # 硅的折射率
        
        residual = pde(u, n)
        
        assert residual.shape == u.shape


class TestMultiScalePINO:
    """多尺度 PINO 测试"""
    
    @pytest.fixture
    def model(self):
        from models.surrogates.pino import MultiScalePINO, PINOConfig
        config = PINOConfig(
            input_channels=1,
            output_channels=1,
            grid_size=(32, 32),
            fno_modes=8,
            fno_hidden_dim=16,
        )
        return MultiScalePINO(config)
    
    def test_forward_shape(self, model):
        """测试输出形状"""
        design = torch.rand(2, 1, 32, 32)
        output = model(design)
        
        assert output.shape == (2, 1, 32, 32)


# ============================================================================
# Model Saving/Loading Tests
# ============================================================================

class TestModelPersistence:
    """模型持久化测试"""
    
    def test_cnn_save_load(self, tmp_path):
        """测试 CNN 代理模型保存加载"""
        from models.surrogates.cnn_surrogate import CNNSurrogate, CNNSurrogateConfig
        
        config = CNNSurrogateConfig(
            input_channels=1,
            input_height=32,
            input_width=32,
            output_dim=3,
            base_channels=16,
        )
        
        model = CNNSurrogate(config)
        
        # 保存
        save_path = tmp_path / "cnn_model.pt"
        model.save(save_path)
        
        assert save_path.exists()
        
        # 加载
        new_model = CNNSurrogate(config)
        new_model.load(save_path)
        
        # 验证参数一致
        for (name1, param1), (name2, param2) in zip(
            model.named_parameters(), new_model.named_parameters()
        ):
            assert torch.allclose(param1, param2)
    
    def test_deeponet_save_load(self, tmp_path):
        """测试 DeepONet 保存加载"""
        from models.surrogates.deeponet import DeepONet, DeepONetConfig
        
        config = DeepONetConfig(
            branch_input_dim=50,
            trunk_input_dim=2,
            branch_output_dim=16,
            trunk_output_dim=16,
        )
        
        model = DeepONet(config)
        
        # 保存
        save_path = tmp_path / "deeponet_model.pt"
        model.save(save_path)
        
        assert save_path.exists()
        
        # 加载
        new_model = DeepONet(config)
        new_model.load(save_path)
        
        # 验证推理一致
        u = torch.rand(2, 50)
        y = torch.rand(5, 2)
        
        with torch.no_grad():
            output1 = model(u, y)
            output2 = new_model(u, y)
        
        assert torch.allclose(output1, output2, atol=1e-5)


# ============================================================================
# Convenience Functions Tests
# ============================================================================

class TestConvenienceFunctions:
    """便捷函数测试"""
    
    def test_create_cnn_surrogate(self):
        """测试创建 CNN 代理模型"""
        from models.surrogates.cnn_surrogate import create_cnn_surrogate
        
        model = create_cnn_surrogate(
            input_shape=(1, 32, 32),
            output_dim=3,
        )
        
        x = torch.rand(2, 1, 32, 32)
        output = model(x)
        
        assert output.shape == (2, 3)
    
    def test_create_deeponet(self):
        """测试创建 DeepONet"""
        from models.surrogates.deeponet import create_deeponet
        
        model = create_deeponet(
            branch_input_dim=100,
            trunk_input_dim=2,
            output_dim=32,
        )
        
        u = torch.rand(2, 100)
        y = torch.rand(5, 2)
        output = model(u, y)
        
        assert output.shape == (2, 5)
    
    def test_create_pino(self):
        """测试创建 PINO"""
        from models.surrogates.pino import create_pino
        
        model = create_pino(
            input_channels=1,
            output_channels=1,
            wavelength=1.55,
        )
        
        design = torch.rand(2, 1, 32, 32)
        output = model(design)
        
        assert output.dim() == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
PINN (物理信息神经网络) 单元测试

测试内容:
1. 配置类测试
2. 网络初始化测试
3. 前向传播测试
4. 梯度计算测试
5. 损失函数测试
6. 求解器测试
7. 节点集成测试
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import pytest
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict

from models.inverse.pinn import (
    PhysicsInformedNet,
    SirenNet,
    MaxwellPINN,
    PhotonicsPINN,
    PINNSolver,
    PINNConfig,
    MaxwellConfig,
    PhysicsLossConfig,
    FourierFeatures,
    Sine,
    create_pinn_for_photonics
)
from models.training.losses import (
    PDEResidualLoss,
    HelmholtzLoss,
    MaxwellLoss,
    BoundaryConditionLoss,
    PINNCombinedLoss
)
from core.nodes.neural_network import (
    PINNNode,
    CoordinateNode,
    create_pinn_node_for_photonics
)


# ============================================================================
# 固定装置
# ============================================================================

@pytest.fixture
def device():
    """获取计算设备"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def basic_config():
    """基础配置"""
    return PINNConfig(
        spatial_dim=2,
        field_components=1,
        hidden_dims=[32, 64, 32],
        activation='tanh'
    )


@pytest.fixture
def maxwell_config():
    """Maxwell 配置"""
    return MaxwellConfig(
        spatial_dim=2,
        field_components=6,
        wavelength=1.55e-6,
        epsilon_r=12.0,
        hidden_dims=[32, 64, 64, 32]
    )


# ============================================================================
# 配置类测试
# ============================================================================

class TestPINNConfig:
    """PINN 配置测试"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = PINNConfig()
        
        assert config.spatial_dim == 2
        assert config.design_dim == 0
        assert config.field_components == 3
        assert config.activation == 'tanh'
        assert config.use_fourier == True
    
    def test_custom_config(self):
        """测试自定义配置"""
        config = PINNConfig(
            spatial_dim=3,
            design_dim=10,
            field_components=6,
            hidden_dims=[64, 128, 256],
            activation='sine'
        )
        
        assert config.spatial_dim == 3
        assert config.design_dim == 10
        assert config.field_components == 6
        assert config.hidden_dims == [64, 128, 256]
        assert config.activation == 'sine'
    
    def test_maxwell_config(self):
        """测试 Maxwell 配置"""
        config = MaxwellConfig()
        
        assert config.wavelength == 1.55e-6
        assert config.epsilon_r == 12.0
        assert config.mu_r == 1.0
        assert config.field_components == 6


# ============================================================================
# 工具模块测试
# ============================================================================

class TestFourierFeatures:
    """Fourier 特征测试"""
    
    def test_output_shape(self):
        """测试输出形状"""
        fourier = FourierFeatures(in_dim=2, out_dim=128)
        x = torch.randn(10, 2)
        
        output = fourier(x)
        
        assert output.shape == (10, 128)
    
    def test_different_sigma(self):
        """测试不同 sigma"""
        for sigma in [1.0, 5.0, 10.0]:
            fourier = FourierFeatures(in_dim=3, out_dim=64, sigma=sigma)
            x = torch.randn(10, 3)
            
            output = fourier(x)
            
            assert output.shape == (10, 64)


class TestSine:
    """正弦激活函数测试"""
    
    def test_forward(self):
        """测试前向传播"""
        sine = Sine(w0=30.0)
        x = torch.randn(10, 5)
        
        output = sine(x)
        
        # 验证输出范围 [-1, 1]
        assert output.min() >= -1.0
        assert output.max() <= 1.0


# ============================================================================
# 网络初始化测试
# ============================================================================

class TestPhysicsInformedNet:
    """物理信息神经网络测试"""
    
    def test_initialization(self, basic_config, device):
        """测试初始化"""
        pinn = PhysicsInformedNet(basic_config).to(device)
        
        # 验证参数
        total_params = sum(p.numel() for p in pinn.parameters())
        assert total_params > 0
    
    def test_forward_shape(self, basic_config, device):
        """测试前向传播形状"""
        pinn = PhysicsInformedNet(basic_config).to(device)
        
        batch_size = 100
        coords = torch.randn(batch_size, 2).to(device)
        
        output = pinn(coords)
        
        assert output.shape == (batch_size, 1)
    
    def test_forward_with_design(self, device):
        """测试带设计参数的前向传播"""
        config = PINNConfig(
            spatial_dim=2,
            design_dim=5,
            field_components=3
        )
        pinn = PhysicsInformedNet(config).to(device)
        
        batch_size = 50
        coords = torch.randn(batch_size, 2).to(device)
        design = torch.randn(batch_size, 5).to(device)
        
        output = pinn(coords, design)
        
        assert output.shape == (batch_size, 3)
    
    def test_gradient_computation(self, basic_config, device):
        """测试梯度计算"""
        pinn = PhysicsInformedNet(basic_config).to(device)
        
        coords = torch.randn(10, 2, requires_grad=True).to(device)
        
        output = pinn(coords)
        grad = torch.autograd.grad(
            output.sum(), coords, create_graph=True
        )[0]
        
        assert grad.shape == (10, 2)
    
    def test_laplacian_computation(self, basic_config, device):
        """测试拉普拉斯计算"""
        pinn = PhysicsInformedNet(basic_config).to(device)
        
        coords = torch.randn(10, 2).to(device)
        laplacian = pinn.compute_laplacian(coords, None, 0)
        
        assert laplacian.shape == (10,)


class TestSirenNet:
    """SIREN 网络测试"""
    
    def test_initialization(self, basic_config, device):
        """测试初始化"""
        siren = SirenNet(basic_config, w0=30.0).to(device)
        
        total_params = sum(p.numel() for p in siren.parameters())
        assert total_params > 0
    
    def test_forward_shape(self, basic_config, device):
        """测试前向传播形状"""
        siren = SirenNet(basic_config, w0=30.0).to(device)
        
        batch_size = 100
        coords = torch.randn(batch_size, 2).to(device)
        
        output = siren(coords)
        
        assert output.shape == (batch_size, 1)


class TestMaxwellPINN:
    """Maxwell PINN 测试"""
    
    def test_initialization(self, maxwell_config, device):
        """测试初始化"""
        pinn = MaxwellPINN(maxwell_config).to(device)
        
        total_params = sum(p.numel() for p in pinn.parameters())
        assert total_params > 0
    
    def test_forward_shape(self, maxwell_config, device):
        """测试前向传播形状"""
        pinn = MaxwellPINN(maxwell_config).to(device)
        
        batch_size = 50
        coords = torch.randn(batch_size, 2).to(device)
        
        fields = pinn(coords)
        
        assert 'E' in fields
        assert 'H' in fields
        assert fields['E'].shape == (batch_size, 3)
        assert fields['H'].shape == (batch_size, 3)
    
    def test_maxwell_residual(self, maxwell_config, device):
        """测试 Maxwell 残差计算"""
        pinn = MaxwellPINN(maxwell_config).to(device)
        
        coords = torch.randn(20, 2).to(device)
        coords.requires_grad_(True)
        
        residual = pinn.compute_maxwell_residual(coords)
        
        assert 'curl_E_residual' in residual
        assert 'curl_H_residual' in residual
    
    def test_poynting_vector(self, maxwell_config, device):
        """测试 Poynting 矢量计算"""
        pinn = MaxwellPINN(maxwell_config).to(device)
        
        coords = torch.randn(20, 2).to(device)
        
        S = pinn.compute_poynting_vector(coords)
        
        assert S.shape == (20, 3)


class TestPhotonicsPINN:
    """光子学 PINN 测试"""
    
    def test_initialization(self, device):
        """测试初始化"""
        config = PINNConfig(
            spatial_dim=2,
            design_dim=5,
            field_components=3
        )
        pinn = PhotonicsPINN(config).to(device)
        
        total_params = sum(p.numel() for p in pinn.parameters())
        assert total_params > 0
    
    def test_inverse_design(self, device):
        """测试逆向设计"""
        config = PINNConfig(
            spatial_dim=2,
            design_dim=5,
            field_components=3
        )
        pinn = PhotonicsPINN(config).to(device)
        
        target_field = torch.randn(10, 3).to(device)
        coords = torch.randn(10, 2).to(device)
        
        design = pinn.inverse_design(target_field, coords)
        
        assert design.shape == (1, 5)
        assert (design >= 0).all() and (design <= 1).all()
    
    def test_total_loss(self, device):
        """测试总损失计算"""
        config = PINNConfig(
            spatial_dim=2,
            field_components=1
        )
        physics_config = PhysicsLossConfig()
        pinn = PhotonicsPINN(config, physics_config).to(device)
        
        coords = torch.randn(50, 2).to(device)
        labeled_data = {
            'coords': torch.randn(10, 2).to(device),
            'fields': torch.randn(10, 1).to(device)
        }
        boundary_data = {
            'coords': torch.randn(20, 2).to(device),
            'values': torch.randn(20, 1).to(device)
        }
        
        losses = pinn.compute_total_loss(
            coords, None,
            labeled_data=labeled_data,
            boundary_data=boundary_data
        )
        
        assert 'total' in losses
        assert losses['total'].item() >= 0


# ============================================================================
# 损失函数测试
# ============================================================================

class TestPhysicsLosses:
    """物理损失函数测试"""
    
    def test_pde_residual_loss(self):
        """测试 PDE 残差损失"""
        loss_fn = PDEResidualLoss()
        
        residual = torch.randn(100, 1)
        loss = loss_fn(residual)
        
        assert loss.item() >= 0
    
    def test_helmholtz_loss(self):
        """测试 Helmholtz 损失"""
        loss_fn = HelmholtzLoss(k=1.0, weight=1.0)
        
        u = torch.randn(100, 1)
        laplacian = torch.randn(100, 1)
        
        loss = loss_fn(u, laplacian)
        
        assert loss.item() >= 0
    
    def test_maxwell_loss(self):
        """测试 Maxwell 损失"""
        loss_fn = MaxwellLoss(omega=1.0, epsilon=1.0, mu=1.0)
        
        curl_E = torch.randn(100, 3)
        curl_H = torch.randn(100, 3)
        E = torch.randn(100, 3)
        H = torch.randn(100, 3)
        
        losses = loss_fn(curl_E, curl_H, E, H)
        
        assert 'maxwell_E' in losses
        assert 'maxwell_H' in losses
        assert 'total' in losses
    
    def test_boundary_condition_loss_dirichlet(self):
        """测试 Dirichlet 边界条件损失"""
        loss_fn = BoundaryConditionLoss(bc_type='dirichlet')
        
        pred = torch.randn(100, 1)
        target = torch.zeros(100, 1)
        
        loss = loss_fn(pred, target)
        
        assert loss.item() >= 0
    
    def test_pinn_combined_loss(self):
        """测试 PINN 组合损失"""
        loss_fn = PINNCombinedLoss(
            physics_weight=1.0,
            bc_weight=1.0,
            data_weight=1.0,
            adaptive_weights=True
        )
        
        physics_residual = torch.randn(100, 1)
        bc_pred = torch.randn(50, 1)
        bc_target = torch.zeros(50, 1)
        data_pred = torch.randn(30, 1)
        data_target = torch.randn(30, 1)
        
        losses = loss_fn(
            physics_residual=physics_residual,
            bc_pred=bc_pred,
            bc_target=bc_target,
            data_pred=data_pred,
            data_target=data_target
        )
        
        assert 'physics' in losses
        assert 'bc' in losses
        assert 'data' in losses
        assert 'total' in losses


# ============================================================================
# 求解器测试
# ============================================================================

class TestPINNSolver:
    """PINN 求解器测试"""
    
    def test_initialization(self, basic_config, device):
        """测试初始化"""
        pinn = PhysicsInformedNet(basic_config).to(device)
        solver = PINNSolver(pinn, optimizer='adam', lr=1e-3)
        
        assert solver.model is pinn
        assert solver.optimizer is not None
    
    def test_train_step(self, basic_config, device):
        """测试训练步骤"""
        pinn = PhysicsInformedNet(basic_config).to(device)
        solver = PINNSolver(pinn, optimizer='adam', lr=1e-3)
        
        collocation = torch.randn(100, 2).to(device)
        boundary = torch.randn(20, 2).to(device)
        
        history = solver.train(
            n_iterations=10,
            collocation_points=collocation,
            boundary_points=boundary,
            log_interval=10
        )
        
        assert len(history['loss']) == 10
    
    def test_predict(self, basic_config, device):
        """测试预测"""
        pinn = PhysicsInformedNet(basic_config).to(device)
        solver = PINNSolver(pinn)
        
        coords = torch.randn(50, 2).to(device)
        prediction = solver.predict(coords)
        
        assert prediction.shape == (50, 1)
    
    def test_save_load(self, basic_config, device, tmp_path):
        """测试保存和加载"""
        pinn = PhysicsInformedNet(basic_config).to(device)
        solver = PINNSolver(pinn)
        
        # 保存
        save_path = str(tmp_path / 'pinn_checkpoint.pt')
        solver.save(save_path)
        
        # 加载
        solver.load(save_path)
        
        assert solver.model is not None


# ============================================================================
# 节点集成测试
# ============================================================================

class TestPINNNode:
    """PINN 节点测试"""
    
    def test_initialization(self, basic_config, device):
        """测试初始化"""
        pinn = PhysicsInformedNet(basic_config).to(device)
        coords = CoordinateNode('coords', bounds=[(-1, 1), (-1, 1)])
        node = PINNNode('pinn', pinn, coords, device=str(device))
        
        assert node.pinn is pinn
    
    def test_forward(self, basic_config, device):
        """测试前向传播"""
        pinn = PhysicsInformedNet(basic_config).to(device)
        coords = CoordinateNode(
            'coords',
            bounds=[(-1, 1), (-1, 1)],
            resolution=(10, 10)
        )
        node = PINNNode('pinn', pinn, coords, device=str(device))
        
        output = node.forward()
        
        assert output.shape[1] == 1
        assert output.shape[0] == 100  # 10 x 10
    
    def test_compute_residual(self, basic_config, device):
        """测试残差计算"""
        pinn = PhysicsInformedNet(basic_config).to(device)
        coords = CoordinateNode('coords', bounds=[(-1, 1), (-1, 1)])
        node = PINNNode('pinn', pinn, coords, device=str(device))
        
        sample_coords = torch.randn(20, 2).to(device)
        residual = node.compute_residual(sample_coords)
        
        assert 'residual' in residual
    
    def test_physics_loss(self, basic_config, device):
        """测试物理损失"""
        pinn = PhysicsInformedNet(basic_config).to(device)
        coords = CoordinateNode('coords', bounds=[(-1, 1), (-1, 1)])
        node = PINNNode('pinn', pinn, coords, device=str(device))
        
        sample_coords = torch.randn(20, 2).to(device)
        loss = node.compute_physics_loss(sample_coords)
        
        assert loss.item() >= 0


class TestCoordinateNode:
    """坐标节点测试"""
    
    def test_initialization(self):
        """测试初始化"""
        node = CoordinateNode(
            'coords',
            bounds=[(-1, 1), (-1, 1)],
            resolution=(50, 50)
        )
        
        assert node.bounds == [(-1, 1), (-1, 1)]
        assert node.resolution == (50, 50)
    
    def test_forward_shape(self):
        """测试前向传播形状"""
        node = CoordinateNode(
            'coords',
            bounds=[(-1, 1), (-1, 1)],
            resolution=(50, 50)
        )
        
        coords = node.forward()
        
        assert coords.shape == (2500, 2)
    
    def test_random_sampling(self):
        """测试随机采样"""
        node = CoordinateNode(
            'coords',
            bounds=[(-1, 1), (-1, 1)]
        )
        
        samples = node.sample_random(100)
        
        assert samples.shape == (100, 2)
        assert samples[:, 0].min() >= -1
        assert samples[:, 0].max() <= 1
    
    def test_boundary_sampling(self):
        """测试边界采样"""
        node = CoordinateNode(
            'coords',
            bounds=[(-1, 1), (-1, 1)]
        )
        
        boundary = node.sample_boundary(25)
        
        # 边界点应该在边界上
        on_boundary = (
            (boundary[:, 0].abs() - 1.0).abs() < 0.01 |
            (boundary[:, 1].abs() - 1.0).abs() < 0.01
        )
        assert on_boundary.all()


# ============================================================================
# 便捷函数测试
# ============================================================================

class TestConvenienceFunctions:
    """便捷函数测试"""
    
    def test_create_pinn_for_photonics(self, device):
        """测试创建光子学 PINN"""
        pinn = create_pinn_for_photonics(
            spatial_dim=2,
            field_components=3,
            wavelength=1.55e-6,
            epsilon_r=12.0,
            device=str(device)
        )
        
        assert isinstance(pinn, MaxwellPINN)
    
    def test_create_pinn_node_for_photonics(self, device):
        """测试创建光子学 PINN 节点"""
        node = create_pinn_node_for_photonics(
            spatial_dim=2,
            field_components=3,
            device=str(device)
        )
        
        assert isinstance(node, PINNNode)


# ============================================================================
# 端到端测试
# ============================================================================

class TestEndToEnd:
    """端到端测试"""
    
    def test_training_workflow(self, device):
        """测试完整训练流程"""
        # 创建 PINN
        config = PINNConfig(
            spatial_dim=2,
            field_components=1,
            hidden_dims=[32, 64, 32]
        )
        pinn = PhysicsInformedNet(config).to(device)
        
        # 创建求解器
        solver = PINNSolver(pinn, lr=1e-3)
        
        # 准备数据
        collocation = torch.randn(500, 2).to(device)
        boundary = torch.cat([
            torch.stack([torch.full((50,), -1.0), torch.rand(50)], dim=1),
            torch.stack([torch.full((50,), 1.0), torch.rand(50)], dim=1),
        ], dim=0).to(device)
        
        # 训练
        history = solver.train(
            n_iterations=100,
            collocation_points=collocation,
            boundary_points=boundary,
            log_interval=100
        )
        
        # 验证损失下降
        assert history['loss'][-1] < history['loss'][0]
    
    def test_inverse_design_workflow(self, device):
        """测试逆向设计流程"""
        # 创建 PINN
        config = PINNConfig(
            spatial_dim=2,
            design_dim=3,
            field_components=2
        )
        pinn = PhotonicsPINN(config).to(device)
        
        # 创建节点
        coords = CoordinateNode('coords', bounds=[(-1, 1), (-1, 1)])
        node = PINNNode('pinn', pinn, coords, device=str(device))
        
        # 生成目标
        target_design = torch.tensor([[0.3, 0.7, 0.5]])
        target_coords = coords.sample_random(50).to(device)
        
        with torch.no_grad():
            target_field = pinn(target_coords, target_design.to(device))
        
        # 逆向设计
        recovered = node.inverse_design(
            target_field=target_field,
            coordinates=target_coords,
            n_iterations=100,
            lr=0.01
        )
        
        # 验证设计参数范围
        assert (recovered >= 0).all() and (recovered <= 1).all()


# ============================================================================
# 运行测试
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])

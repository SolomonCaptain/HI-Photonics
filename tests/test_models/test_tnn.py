"""
TNN 单元测试

测试 TNN 各组件的功能正确性。
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from models.inverse.tnn import (
    ForwardNetwork,
    InverseNetwork,
    TandemNetwork,
    ForwardNetworkConfig,
    InverseNetworkConfig,
    TandemNetworkConfig
)
from models.training.losses import TandemLoss, PerformanceLoss
from models.training.metrics import MSE, MAE, R2Score
from data.loaders.pipeline import SyntheticDataset, create_dataloaders


class TestForwardNetwork:
    """前向网络测试"""
    
    @pytest.fixture
    def config(self):
        return ForwardNetworkConfig(
            design_shape=(100, 22),
            performance_dim=3,
            hidden_channels=[16, 32],
            hidden_dims=[64, 32]
        )
    
    @pytest.fixture
    def model(self, config):
        return ForwardNetwork(config)
    
    def test_forward_output_shape(self, model, config):
        """测试输出形状"""
        batch_size = 8
        design = torch.rand(batch_size, *config.design_shape)
        
        output = model(design)
        
        assert output.shape == (batch_size, config.performance_dim)
    
    def test_forward_without_batch(self, model, config):
        """测试无批次维度输入"""
        design = torch.rand(*config.design_shape)
        
        output = model(design)
        
        assert output.dim() == 1
        assert output.size(0) == config.performance_dim
    
    def test_forward_with_channel_dim(self, model, config):
        """测试带通道维度的输入"""
        batch_size = 4
        design = torch.rand(batch_size, 1, *config.design_shape)
        
        output = model(design)
        
        assert output.shape == (batch_size, config.performance_dim)
    
    def test_gradient_flow(self, model, config):
        """测试梯度流动"""
        design = torch.rand(2, *config.design_shape, requires_grad=True)
        
        output = model(design)
        loss = output.sum()
        loss.backward()
        
        assert design.grad is not None
        assert design.grad.shape == design.shape
    
    def test_count_parameters(self, model):
        """测试参数计数"""
        num_params = model.count_parameters()
        
        assert num_params > 0
        assert isinstance(num_params, int)


class TestInverseNetwork:
    """逆向网络测试"""
    
    @pytest.fixture
    def config(self):
        return InverseNetworkConfig(
            design_shape=(100, 22),
            performance_dim=3,
            hidden_dims=[128, 256],
            hidden_channels=[64, 32]
        )
    
    @pytest.fixture
    def model(self, config):
        return InverseNetwork(config)
    
    def test_inverse_output_shape(self, model, config):
        """测试输出形状"""
        batch_size = 8
        performance = torch.rand(batch_size, config.performance_dim)
        
        output = model(performance)
        
        assert output.shape == (batch_size, *config.design_shape)
    
    def test_inverse_output_range(self, model, config):
        """测试输出范围（Sigmoid 应该在 [0, 1]）"""
        performance = torch.rand(4, config.performance_dim)
        
        output = model(performance)
        
        assert output.min() >= 0
        assert output.max() <= 1
    
    def test_gradient_flow(self, model, config):
        """测试梯度流动"""
        performance = torch.rand(2, config.performance_dim, requires_grad=True)
        
        output = model(performance)
        loss = output.sum()
        loss.backward()
        
        assert performance.grad is not None


class TestTandemNetwork:
    """串联网络测试"""
    
    @pytest.fixture
    def config(self):
        forward_config = ForwardNetworkConfig(
            design_shape=(50, 11),
            performance_dim=3,
            hidden_channels=[16, 32],
            hidden_dims=[64, 32]
        )
        
        inverse_config = InverseNetworkConfig(
            design_shape=(50, 11),
            performance_dim=3,
            hidden_dims=[64, 128],
            hidden_channels=[32, 16]
        )
        
        return TandemNetworkConfig(
            forward_config=forward_config,
            inverse_config=inverse_config
        )
    
    @pytest.fixture
    def model(self, config):
        return TandemNetwork(config)
    
    def test_forward_pass(self, model, config):
        """测试前向传播"""
        batch_size = 4
        design = torch.rand(batch_size, *config.forward_config.design_shape)
        
        performance = model(design)
        
        assert performance.shape == (batch_size, config.forward_config.performance_dim)
    
    def test_inverse_design(self, model, config):
        """测试逆向设计"""
        batch_size = 2
        target_performance = torch.rand(batch_size, config.forward_config.performance_dim)
        
        design, pred_performance = model.inverse_design(target_performance)
        
        assert design.shape == (batch_size, *config.forward_config.design_shape)
        assert pred_performance.shape == (batch_size, config.forward_config.performance_dim)
    
    def test_inverse_design_without_predicted(self, model, config):
        """测试逆向设计（不返回预测性能）"""
        target_performance = torch.rand(2, config.forward_config.performance_dim)
        
        design = model.inverse_design(target_performance, return_predicted=False)
        
        assert design.shape[0] == 2
        assert design.shape[1:] == config.forward_config.design_shape
    
    def test_freeze_forward(self, model):
        """测试冻结前向网络"""
        model.forward_net.freeze()
        
        for param in model.forward_net.parameters():
            assert not param.requires_grad
    
    def test_model_save_load(self, model, tmp_path):
        """测试模型保存和加载"""
        save_path = tmp_path / "tnn_test.pth"
        
        # 保存
        model.save(save_path)
        assert save_path.exists()
        
        # 创建新模型并加载
        config = model.config
        new_model = TandemNetwork(config)
        new_model.load(save_path)
        
        # 验证加载正确
        design = torch.rand(1, *config.forward_config.design_shape)
        output1 = model(design)
        output2 = new_model(design)
        
        assert torch.allclose(output1, output2, atol=1e-5)
    
    def test_training_pipeline(self, model):
        """测试训练管道"""
        # 创建合成数据
        dataset = SyntheticDataset(
            num_samples=100,
            design_shape=model.design_shape,
            performance_dim=model.performance_dim
        )
        
        train_loader, val_loader, _ = create_dataloaders(
            dataset, batch_size=16, train_ratio=0.8, val_ratio=0.2, test_ratio=0.0
        )
        
        # 简短训练
        history = model.train_forward(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=2,
            lr=1e-3
        )
        
        assert 'train_loss' in history
        assert len(history['train_loss']) > 0


class TestLosses:
    """损失函数测试"""
    
    def test_performance_loss(self):
        """测试性能损失"""
        loss_fn = PerformanceLoss()
        
        pred = torch.rand(8, 3)
        target = torch.rand(8, 3)
        
        loss = loss_fn(pred, target)
        
        assert loss.dim() == 0  # 标量
        assert loss.item() >= 0
    
    def test_tandem_loss(self):
        """测试串联损失"""
        loss_fn = TandemLoss(
            performance_weight=1.0,
            design_weight=0.5
        )
        
        pred_perf = torch.rand(4, 3)
        target_perf = torch.rand(4, 3)
        design = torch.rand(4, 50, 11)
        design_gt = torch.rand(4, 50, 11)
        
        losses = loss_fn(pred_perf, target_perf, design, design_gt)
        
        assert 'performance' in losses
        assert 'design' in losses
        assert 'total' in losses


class TestMetrics:
    """评估指标测试"""
    
    def test_mse(self):
        """测试 MSE"""
        metric = MSE()
        
        pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        target = torch.tensor([[1.1, 2.1], [2.9, 4.1]])
        
        result = metric(pred, target)
        
        assert result.name == 'mse'
        assert result.value > 0
    
    def test_r2_score(self):
        """测试 R² 分数"""
        metric = R2Score()
        
        # 完美预测
        pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        target = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        
        result = metric(pred, target)
        
        assert abs(result.value - 1.0) < 1e-5


class TestDataset:
    """数据集测试"""
    
    def test_synthetic_dataset(self):
        """测试合成数据集"""
        dataset = SyntheticDataset(
            num_samples=100,
            design_shape=(50, 11),
            performance_dim=3
        )
        
        assert len(dataset) == 100
        
        sample = dataset[0]
        assert 'design' in sample
        assert 'performance' in sample
        assert sample['design'].shape == (50, 11)
        assert sample['performance'].shape == (3,)
    
    def test_dataloader(self):
        """测试数据加载器"""
        dataset = SyntheticDataset(
            num_samples=100,
            design_shape=(50, 11),
            performance_dim=3
        )
        
        train_loader, val_loader, test_loader = create_dataloaders(
            dataset, batch_size=16
        )
        
        batch = next(iter(train_loader))
        assert batch['design'].shape[0] <= 16
        assert batch['performance'].shape[1] == 3


class TestGraphIntegration:
    """计算图集成测试"""
    
    def test_tnn_node(self):
        """测试 TNN 节点"""
        from core.nodes.neural_network import TNNNode, TargetPerformanceNode
        
        # 创建 TNN
        forward_config = ForwardNetworkConfig(
            design_shape=(50, 11),
            performance_dim=3,
            hidden_channels=[16, 32],
            hidden_dims=[64]
        )
        
        inverse_config = InverseNetworkConfig(
            design_shape=(50, 11),
            performance_dim=3,
            hidden_dims=[64, 128],
            hidden_channels=[32, 16]
        )
        
        tandem_config = TandemNetworkConfig(
            forward_config=forward_config,
            inverse_config=inverse_config
        )
        
        tnn = TandemNetwork(tandem_config)
        
        # 测试逆向节点
        target = torch.tensor([[0.8, 0.7, 0.1]])
        target_node = TargetPerformanceNode('target', target)
        
        inverse_node = TNNNode(
            name='tnn_inverse',
            tnn_model=tnn,
            mode='inverse',
            input_node=target_node
        )
        
        design, pred_perf = inverse_node.forward()
        
        assert design.shape == (1, 50, 11)
        assert pred_perf.shape == (1, 3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

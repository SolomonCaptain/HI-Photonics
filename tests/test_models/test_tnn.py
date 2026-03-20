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
from models.training.losses import (
    TandemLoss, PerformanceLoss,
    InverseContrastiveLoss,
    DesignSharpnessLoss,
    DiversityPreservingLoss,
    TNNAntiAverageLoss,
    OptimalDesignGuidanceLoss
)
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


class TestAntiAverageLosses:
    """反平均设计损失函数测试"""
    
    def test_inverse_contrastive_loss(self):
        """测试逆向设计对比学习损失"""
        loss_fn = InverseContrastiveLoss(temperature=0.1)
        
        batch_size = 8
        designs = torch.rand(batch_size, 50, 11)
        performances = torch.rand(batch_size, 3)
        
        losses = loss_fn(designs, performances)
        
        assert 'contrastive_total' in losses
        assert losses['contrastive_total'].dim() == 0
        print(f"Contrastive loss: {losses['contrastive_total'].item():.4f}")
    
    def test_contrastive_loss_with_similar_performances(self):
        """测试相同性能的设计应该被推远"""
        loss_fn = InverseContrastiveLoss(temperature=0.1, design_diversity_weight=1.0)
        
        # 创建相同性能的设计
        performances = torch.tensor([
            [0.8, 0.1, 0.1],
            [0.8, 0.1, 0.1],
            [0.8, 0.1, 0.1],
            [0.5, 0.3, 0.2],
            [0.5, 0.3, 0.2],
        ])
        
        # 设计稍微不同
        designs = torch.rand(5, 50, 11)
        
        losses = loss_fn(designs, performances)
        
        # 相同性能的设计应该有高多样性损失
        print(f"Design diversity loss: {losses['design_diversity'].item():.4f}")
        assert losses['design_diversity'].item() >= 0
    
    def test_design_sharpness_loss(self):
        """测试设计锐度损失"""
        loss_fn = DesignSharpnessLoss(target_sharpness=0.8)
        
        # 创建模糊设计（接近 0.5）
        blurry_design = torch.ones(4, 50, 11) * 0.5
        
        # 创建锐利设计（二值化）
        sharp_design = (torch.rand(4, 50, 11) > 0.5).float()
        
        blurry_losses = loss_fn(blurry_design)
        sharp_losses = loss_fn(sharp_design)
        
        # 模糊设计应该有更高的二值化损失
        assert blurry_losses['binary'].item() > sharp_losses['binary'].item()
        print(f"Blurry binary loss: {blurry_losses['binary'].item():.4f}")
        print(f"Sharp binary loss: {sharp_losses['binary'].item():.4f}")
    
    def test_diversity_preserving_loss(self):
        """测试多样性保持损失"""
        loss_fn = DiversityPreservingLoss(num_modes=4)
        
        # 创建多样化设计
        diverse_designs = torch.rand(16, 50, 11)
        
        # 创建塌缩设计（都相似）
        collapsed_designs = torch.ones(16, 50, 11) * 0.5 + torch.randn(16, 50, 11) * 0.01
        
        diverse_losses = loss_fn(diverse_designs)
        collapsed_losses = loss_fn(collapsed_designs)
        
        # 多样化设计的平均距离应该更大
        assert diverse_losses['avg_distance'].item() > collapsed_losses['avg_distance'].item()
        print(f"Diverse avg distance: {diverse_losses['avg_distance'].item():.4f}")
        print(f"Collapsed avg distance: {collapsed_losses['avg_distance'].item():.4f}")
    
    def test_tnn_anti_average_loss(self):
        """测试综合反平均损失"""
        loss_fn = TNNAntiAverageLoss(
            contrastive_weight=0.5,
            sharpness_weight=0.3,
            diversity_weight=0.2
        )
        
        designs = torch.rand(8, 50, 11)
        performances = torch.rand(8, 3)
        
        losses = loss_fn(designs, performances)
        
        assert 'anti_average_total' in losses
        print(f"Anti-average total loss: {losses['anti_average_total'].item():.4f}")
    
    def test_optimal_design_guidance_loss(self):
        """测试最优设计引导损失"""
        loss_fn = OptimalDesignGuidanceLoss()
        
        designs = torch.rand(4, 50, 11)
        pred_perf = torch.tensor([[0.7, 0.2, 0.1], [0.8, 0.1, 0.1], [0.6, 0.3, 0.1], [0.9, 0.05, 0.05]])
        target_perf = torch.tensor([[0.8, 0.1, 0.1], [0.8, 0.1, 0.1], [0.8, 0.1, 0.1], [0.8, 0.1, 0.1]])
        
        losses = loss_fn(designs, pred_perf, target_perf)
        
        assert 'guidance_total' in losses
        print(f"Guidance total loss: {losses['guidance_total'].item():.4f}")


class TestTNNWithAntiAverageLoss:
    """TNN 集成反平均损失测试"""
    
    def test_tnn_config_with_anti_average(self):
        """测试 TNN 配置支持反平均损失参数"""
        config = TandemNetworkConfig(
            contrastive_loss_weight=0.5,
            sharpness_loss_weight=0.3,
            diversity_preserve_weight=0.2,
            guidance_loss_weight=0.1
        )
        
        assert config.contrastive_loss_weight == 0.5
        assert config.sharpness_loss_weight == 0.3
        assert config.diversity_preserve_weight == 0.2
        assert config.guidance_loss_weight == 0.1
    
    def test_tnn_init_with_anti_average_losses(self):
        """测试 TNN 初始化反平均损失组件"""
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
            inverse_config=inverse_config,
            contrastive_loss_weight=0.5,
            sharpness_loss_weight=0.3,
            diversity_preserve_weight=0.2
        )
        
        tnn = TandemNetwork(tandem_config)
        
        # 检查损失函数已初始化
        assert tnn.contrastive_loss is not None
        assert tnn.sharpness_loss is not None
        assert tnn.diversity_loss is not None
    
    def test_tnn_compute_tandem_loss_with_anti_average(self):
        """测试 TNN 计算损失包含反平均组件"""
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
            inverse_config=inverse_config,
            contrastive_loss_weight=0.5,
            sharpness_loss_weight=0.3,
            diversity_preserve_weight=0.2
        )
        
        tnn = TandemNetwork(tandem_config)
        
        # 创建测试数据
        pred_perf = torch.rand(4, 3)
        target_perf = torch.rand(4, 3)
        designs = torch.rand(4, 50, 11)
        
        total_loss, loss_dict = tnn._compute_tandem_loss(pred_perf, target_perf, designs)
        
        # 检查损失字典包含反平均损失
        assert 'total' in loss_dict
        assert 'performance' in loss_dict
        assert total_loss.dim() == 0
        
        print(f"Total loss: {total_loss.item():.4f}")
        print(f"Performance loss: {loss_dict['performance'].item():.4f}")
    
    def test_tnn_design_metrics(self):
        """测试 TNN 设计评估指标"""
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
        
        # 创建测试设计
        designs = torch.rand(8, 50, 11)
        
        # 测试多样性分数
        diversity_score = tnn.get_design_diversity_score(designs)
        assert isinstance(diversity_score, float)
        print(f"Diversity score: {diversity_score:.4f}")
        
        # 测试锐度分数
        sharpness = tnn.get_design_sharpness_score(designs)
        assert 'binary_score' in sharpness
        assert 'sharpness_score' in sharpness
        assert 'is_sharp' in sharpness
        print(f"Binary score: {sharpness['binary_score']:.4f}")
        print(f"Sharpness score: {sharpness['sharpness_score']:.4f}")
    
    def test_tnn_get_model_info_with_anti_average(self):
        """测试模型信息包含反平均配置"""
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
            inverse_config=inverse_config,
            contrastive_loss_weight=0.5,
            sharpness_loss_weight=0.3
        )
        
        tnn = TandemNetwork(tandem_config)
        info = tnn.get_model_info()
        
        assert 'anti_average_config' in info
        assert info['anti_average_config']['contrastive_weight'] == 0.5
        assert info['anti_average_config']['sharpness_weight'] == 0.3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

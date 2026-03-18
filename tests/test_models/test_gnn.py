"""
GNN 单元测试

测试 GNN 各组件的功能正确性。
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from models.inverse.gnn import (
    GraphBuildConfig,
    GNNConfig,
    GraphBuilder,
    GraphConvLayer,
    GraphAttentionLayer,
    GraphSAGELayer,
    GlobalPooling,
    GraphEncoder,
    GraphDecoder,
    PhotonicsGNN,
    GraphBatch,
    create_gnn_for_challenge
)


class TestGraphBuildConfig:
    """图构建配置测试"""

    def test_default_config(self):
        """测试默认配置"""
        config = GraphBuildConfig()

        assert config.method == "grid"
        assert config.connectivity == 4
        assert config.use_position is True
        assert config.use_gradient is False

    def test_custom_config(self):
        """测试自定义配置"""
        config = GraphBuildConfig(
            method="knn",
            connectivity=8,
            use_gradient=True,
            k_neighbors=16
        )

        assert config.method == "knn"
        assert config.connectivity == 8
        assert config.use_gradient is True
        assert config.k_neighbors == 16


class TestGNNConfig:
    """GNN 配置测试"""

    def test_default_config(self):
        """测试默认配置"""
        config = GNNConfig()

        assert config.hidden_dim == 128
        assert config.num_layers == 4
        assert config.conv_type == "gcn"
        assert config.pooling == "mean"
        assert config.output_dim == 3

    def test_custom_config(self):
        """测试自定义配置"""
        config = GNNConfig(
            hidden_dim=256,
            num_layers=6,
            conv_type="gat",
            num_heads=8,
            pooling="attention",
            design_shape=(100, 50)
        )

        assert config.hidden_dim == 256
        assert config.num_layers == 6
        assert config.conv_type == "gat"
        assert config.num_heads == 8
        assert config.pooling == "attention"
        assert config.design_shape == (100, 50)


class TestGraphBuilder:
    """图构建器测试"""

    @pytest.fixture
    def grid_config(self):
        return GraphBuildConfig(
            method="grid",
            connectivity=4,
            use_position=True,
            use_gradient=False
        )

    @pytest.fixture
    def knn_config(self):
        return GraphBuildConfig(
            method="knn",
            k_neighbors=8,
            use_position=True
        )

    @pytest.fixture
    def superpixel_config(self):
        return GraphBuildConfig(
            method="superpixel",
            num_superpixels=50
        )

    def test_grid_graph_shape(self, grid_config):
        """测试网格图形状"""
        builder = GraphBuilder(grid_config)
        design = torch.rand(4, 20, 10)

        node_features, edge_index, edge_features = builder(design)

        # 节点数 = batch_size * H * W
        assert node_features.size(0) == 4 * 20 * 10
        # 节点特征维度: 1 (设计值) + 2 (位置) = 3
        assert node_features.size(1) == 3
        # 边索引形状
        assert edge_index.size(0) == 2

    def test_knn_graph_shape(self, knn_config):
        """测试 KNN 图形状"""
        builder = GraphBuilder(knn_config)
        design = torch.rand(2, 15, 15)

        node_features, edge_index, edge_features = builder(design)

        # 节点数 = batch_size * H * W
        assert node_features.size(0) == 2 * 15 * 15
        # 每个节点有 k_neighbors 条出边
        assert edge_index.size(1) >= 2 * 15 * 15 * knn_config.k_neighbors

    def test_superpixel_graph_shape(self, superpixel_config):
        """测试超像素图形状"""
        builder = GraphBuilder(superpixel_config)
        design = torch.rand(2, 40, 30)

        node_features, edge_index, edge_features = builder(design)

        # 节点数应该远小于像素数
        assert node_features.size(0) < 2 * 40 * 30
        # 节点特征维度：均值、标准差、最大值、最小值、中心位置、区域大小
        assert node_features.size(1) == 10

    def test_batch_indices(self, grid_config):
        """测试批次索引"""
        builder = GraphBuilder(grid_config)
        design = torch.rand(3, 10, 10)

        batch = builder.get_batch_indices(design)

        # 批次索引长度
        assert batch.size(0) == 3 * 10 * 10
        # 批次索引范围
        assert batch.min() == 0
        assert batch.max() == 2

    def test_8_connectivity(self):
        """测试 8 邻域连接"""
        config = GraphBuildConfig(connectivity=8)
        builder = GraphBuilder(config)
        design = torch.rand(1, 5, 5)

        _, edge_index, _ = builder(design)

        # 8 邻域比 4 邻域边更多
        # 5x5 网格，4 邻域: 40 条有向边
        # 8 邻域: 更多边
        assert edge_index.size(1) > 80  # 双向边


class TestGraphConvLayer:
    """图卷积层测试"""

    @pytest.fixture
    def layer(self):
        return GraphConvLayer(
            in_channels=64,
            out_channels=128,
            dropout=0.1,
            layer_norm=True,
            residual=False
        )

    def test_forward_shape(self, layer):
        """测试输出形状"""
        num_nodes = 100
        num_edges = 300

        x = torch.rand(num_nodes, 64)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))

        output = layer(x, edge_index)

        assert output.shape == (num_nodes, 128)

    def test_residual_connection(self):
        """测试残差连接"""
        layer = GraphConvLayer(
            in_channels=64,
            out_channels=64,
            residual=True
        )

        x = torch.rand(50, 64, requires_grad=True)
        edge_index = torch.randint(0, 50, (2, 200))

        output = layer(x, edge_index)

        assert output.shape == (50, 64)

    def test_gradient_flow(self, layer):
        """测试梯度流动"""
        x = torch.rand(50, 64, requires_grad=True)
        edge_index = torch.randint(0, 50, (2, 150))

        output = layer(x, edge_index)
        loss = output.sum()
        loss.backward()

        assert x.grad is not None


class TestGraphAttentionLayer:
    """图注意力层测试"""

    @pytest.fixture
    def layer(self):
        return GraphAttentionLayer(
            in_channels=64,
            out_channels=128,
            num_heads=4,
            dropout=0.1,
            concat=True,
            residual=False
        )

    def test_forward_shape(self, layer):
        """测试输出形状"""
        num_nodes = 100
        num_edges = 300

        x = torch.rand(num_nodes, 64)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))

        output = layer(x, edge_index)

        assert output.shape == (num_nodes, 128)

    def test_return_attention(self, layer):
        """测试返回注意力权重"""
        x = torch.rand(50, 64)
        edge_index = torch.randint(0, 50, (2, 150))

        output, attention = layer(x, edge_index, return_attention=True)

        assert output.shape == (50, 128)
        assert attention is not None


class TestGraphSAGELayer:
    """GraphSAGE 层测试"""

    @pytest.fixture
    def layer(self):
        return GraphSAGELayer(
            in_channels=64,
            out_channels=128,
            aggr='mean',
            dropout=0.1,
            residual=False
        )

    def test_forward_shape(self, layer):
        """测试输出形状"""
        num_nodes = 100
        num_edges = 300

        x = torch.rand(num_nodes, 64)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))

        output = layer(x, edge_index)

        assert output.shape == (num_nodes, 128)

    def test_different_aggregations(self):
        """测试不同聚合方式"""
        for aggr in ['mean', 'max']:
            layer = GraphSAGELayer(
                in_channels=32,
                out_channels=64,
                aggr=aggr
            )

            x = torch.rand(50, 32)
            edge_index = torch.randint(0, 50, (2, 100))

            output = layer(x, edge_index)
            assert output.shape == (50, 64)


class TestGlobalPooling:
    """全局池化层测试"""

    def test_mean_pooling(self):
        """测试平均池化"""
        pooling = GlobalPooling(pooling='mean')

        x = torch.rand(100, 64)
        batch = torch.cat([torch.zeros(50), torch.ones(50)]).long()

        output = pooling(x, batch)

        assert output.shape == (2, 64)

    def test_sum_pooling(self):
        """测试求和池化"""
        pooling = GlobalPooling(pooling='sum')

        x = torch.ones(100, 64)
        batch = torch.cat([torch.zeros(50), torch.ones(50)]).long()

        output = pooling(x, batch)

        assert output.shape == (2, 64)
        # 每个图 50 个节点，每个节点特征为 1
        assert torch.allclose(output, torch.ones(2, 64) * 50)

    def test_max_pooling(self):
        """测试最大池化"""
        pooling = GlobalPooling(pooling='max')

        x = torch.rand(100, 64)
        batch = torch.cat([torch.zeros(50), torch.ones(50)]).long()

        output = pooling(x, batch)

        assert output.shape == (2, 64)

    def test_attention_pooling(self):
        """测试注意力池化"""
        pooling = GlobalPooling(pooling='attention', in_channels=64)

        x = torch.rand(100, 64)
        batch = torch.cat([torch.zeros(50), torch.ones(50)]).long()

        output = pooling(x, batch)

        assert output.shape == (2, 64)


class TestGraphEncoder:
    """图编码器测试"""

    @pytest.fixture
    def config(self):
        return GNNConfig(
            node_feature_dim=3,
            hidden_dim=64,
            num_layers=2,
            conv_type='gcn',
            pooling='mean'
        )

    def test_encode_shape(self, config):
        """测试编码形状"""
        encoder = GraphEncoder(config)

        num_nodes = 100
        x = torch.rand(num_nodes, 3)
        edge_index = torch.randint(0, num_nodes, (2, 300))
        batch = torch.zeros(num_nodes, dtype=torch.long)

        embedding = encoder(x, edge_index, batch)

        assert embedding.shape == (1, 64)

    def test_gat_encoder(self):
        """测试 GAT 编码器"""
        config = GNNConfig(
            node_feature_dim=3,
            hidden_dim=64,
            num_layers=2,
            conv_type='gat',
            num_heads=4,
            pooling='mean'
        )

        encoder = GraphEncoder(config)

        num_nodes = 50
        x = torch.rand(num_nodes, 3)
        edge_index = torch.randint(0, num_nodes, (2, 150))
        batch = torch.zeros(num_nodes, dtype=torch.long)

        embedding = encoder(x, edge_index, batch)

        assert embedding.shape == (1, 64)

    def test_graphsage_encoder(self):
        """测试 GraphSAGE 编码器"""
        config = GNNConfig(
            node_feature_dim=3,
            hidden_dim=64,
            num_layers=2,
            conv_type='graphsage',
            pooling='mean'
        )

        encoder = GraphEncoder(config)

        num_nodes = 50
        x = torch.rand(num_nodes, 3)
        edge_index = torch.randint(0, num_nodes, (2, 150))
        batch = torch.zeros(num_nodes, dtype=torch.long)

        embedding = encoder(x, edge_index, batch)

        assert embedding.shape == (1, 64)

    def test_return_node_features(self, config):
        """测试返回节点特征"""
        encoder = GraphEncoder(config)

        num_nodes = 50
        x = torch.rand(num_nodes, 3)
        edge_index = torch.randint(0, num_nodes, (2, 150))
        batch = torch.zeros(num_nodes, dtype=torch.long)

        embedding, node_features = encoder(x, edge_index, batch, return_node_features=True)

        assert embedding.shape == (1, 64)
        assert node_features.shape == (num_nodes, 64)


class TestGraphDecoder:
    """图解码器测试"""

    @pytest.fixture
    def config(self):
        return GNNConfig(
            hidden_dim=64,
            output_dim=3,
            design_shape=(20, 10)
        )

    def test_performance_decoder_shape(self, config):
        """测试性能解码器形状"""
        decoder = GraphDecoder(config, mode='performance')

        batch_size = 4
        graph_embedding = torch.rand(batch_size, 64)

        output = decoder(graph_embedding)

        assert output.shape == (batch_size, 3)

    def test_design_decoder_shape(self, config):
        """测试设计解码器形状"""
        decoder = GraphDecoder(config, mode='design')

        batch_size = 4
        graph_embedding = torch.rand(batch_size, 64)

        output = decoder(graph_embedding)

        assert output.shape == (batch_size, 20, 10)
        # 输出应该在 [0, 1] 范围内
        assert output.min() >= 0
        assert output.max() <= 1


class TestPhotonicsGNN:
    """PhotonicsGNN 主模型测试"""

    @pytest.fixture
    def config(self):
        return GNNConfig(
            hidden_dim=64,
            num_layers=2,
            conv_type='gcn',
            pooling='mean',
            output_dim=3,
            design_shape=(20, 10)
        )

    @pytest.fixture
    def model(self, config):
        return PhotonicsGNN(config)

    def test_predict_performance_shape(self, model, config):
        """测试性能预测形状"""
        batch_size = 4
        design = torch.rand(batch_size, *config.design_shape)

        performance = model.predict_performance(design)

        assert performance.shape == (batch_size, config.output_dim)

    def test_encode_design_shape(self, model, config):
        """测试设计编码形状"""
        batch_size = 4
        design = torch.rand(batch_size, *config.design_shape)

        embedding = model.encode_design(design)

        assert embedding.shape == (batch_size, config.hidden_dim)

    def test_inverse_design_shape(self, model, config):
        """测试逆向设计形状"""
        batch_size = 2
        target_performance = torch.rand(batch_size, config.output_dim)

        design, predicted_perf = model.inverse_design(
            target_performance,
            n_iterations=10,
            lr=0.1
        )

        assert design.shape == (batch_size, *config.design_shape)
        assert predicted_perf.shape == (batch_size, config.output_dim)
        # 设计应该在 [0, 1] 范围内
        assert design.min() >= 0
        assert design.max() <= 1

    def test_gat_model(self):
        """测试 GAT 模型"""
        config = GNNConfig(
            hidden_dim=64,
            num_layers=2,
            conv_type='gat',
            num_heads=4,
            pooling='mean',
            output_dim=3,
            design_shape=(15, 15)
        )

        model = PhotonicsGNN(config)
        design = torch.rand(2, 15, 15)

        performance = model.predict_performance(design)

        assert performance.shape == (2, 3)

    def test_graphsage_model(self):
        """测试 GraphSAGE 模型"""
        config = GNNConfig(
            hidden_dim=64,
            num_layers=2,
            conv_type='graphsage',
            pooling='mean',
            output_dim=3,
            design_shape=(15, 15)
        )

        model = PhotonicsGNN(config)
        design = torch.rand(2, 15, 15)

        performance = model.predict_performance(design)

        assert performance.shape == (2, 3)

    def test_superpixel_model(self):
        """测试超像素图模型"""
        graph_config = GraphBuildConfig(
            method="superpixel",
            num_superpixels=30
        )
        config = GNNConfig(
            hidden_dim=64,
            num_layers=2,
            conv_type='gcn',
            pooling='mean',
            output_dim=3,
            design_shape=(40, 30),
            graph_build_config=graph_config
        )

        model = PhotonicsGNN(config)
        design = torch.rand(2, 40, 30)

        performance = model.predict_performance(design)

        assert performance.shape == (2, 3)

    def test_gradient_flow(self, model, config):
        """测试梯度流动"""
        design = torch.rand(2, *config.design_shape, requires_grad=True)

        performance = model.predict_performance(design)
        loss = performance.sum()
        loss.backward()

        assert design.grad is not None

    def test_count_parameters(self, model):
        """测试参数计数"""
        num_params = model.count_parameters()

        assert num_params > 0
        assert isinstance(num_params, int)

    def test_get_model_info(self, model):
        """测试获取模型信息"""
        info = model.get_model_info()

        assert 'name' in info
        assert 'device' in info
        assert 'parameters' in info


class TestGraphBatch:
    """GraphBatch 数据类测试"""

    def test_basic_operations(self):
        """测试基本操作"""
        node_features = torch.rand(100, 16)
        edge_index = torch.randint(0, 100, (2, 300))
        batch = torch.cat([torch.zeros(50), torch.ones(50)]).long()

        graph_batch = GraphBatch(
            node_features=node_features,
            edge_index=edge_index,
            batch=batch
        )

        assert graph_batch.num_graphs == 2

    def test_to_device(self):
        """测试设备转移"""
        node_features = torch.rand(100, 16)
        edge_index = torch.randint(0, 100, (2, 300))
        batch = torch.zeros(100, dtype=torch.long)

        graph_batch = GraphBatch(
            node_features=node_features,
            edge_index=edge_index,
            batch=batch
        )

        # 测试 CPU 转移
        cpu_batch = graph_batch.to(torch.device('cpu'))
        assert cpu_batch.node_features.device == torch.device('cpu')


class TestCreateGNNForChallenge:
    """便捷创建函数测试"""

    def test_create_for_grating_coupler(self):
        """测试为光栅耦合器创建 GNN"""
        try:
            gnn = create_gnn_for_challenge(
                'grating_coupler',
                hidden_dim=64,
                num_layers=2,
                conv_type='gcn',
                output_dim=3
            )

            assert isinstance(gnn, PhotonicsGNN)
            assert gnn.config.hidden_dim == 64
        except ImportError:
            pytest.skip("Challenge module not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

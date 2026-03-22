"""
Graph Neural Network (GNN) 图神经网络

用于光子学逆向设计的图神经网络架构：
- 将设计网格转换为图结构
- 通过消息传递学习拓扑特征
- 支持性能预测和逆向设计

核心组件:
- GraphBuilder: 设计网格到图的转换
- GraphConvLayer: 图卷积层 (GCN)
- GraphAttentionLayer: 图注意力层 (GAT)
- GraphSAGELayer: GraphSAGE 层
- GlobalPooling: 图池化层
- GraphEncoder: 图编码器
- GraphDecoder: 图解码器
- PhotonicsGNN: 完整 GNN 模型

参考文献:
- Kipf & Welling, "Semi-Supervised Classification with Graph Convolutional Networks", ICLR 2017
- Veličković et al., "Graph Attention Networks", ICLR 2018
- Hamilton et al., "Inductive Representation Learning on Large Graphs", NeurIPS 2017
- Jiang et al., "Graph Neural Networks for Electromagnetic Modeling", 2022
"""

from typing import Dict, Optional, Tuple, List, Union, Any
from dataclasses import dataclass, field
from pathlib import Path
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.base import BaseModel, ModelConfig, SurrogateModel, InverseModel

# PyTorch Geometric 依赖（可选）
try:
    from torch_sparse import SparseTensor
    from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool, global_max_pool, global_add_pool
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False
    warnings.warn(
        "PyTorch Geometric 未安装。请运行 `pip install torch-geometric torch-sparse` 安装。"
        "GNN 功能将不可用，但代码结构仍然可用。",
        UserWarning
    )


# ============================================================================
# 配置类
# ============================================================================

@dataclass
class GraphBuildConfig(ModelConfig):
    """图构建配置"""
    name: str = "graph_builder"

    # 构建方式
    method: str = "grid"  # 'grid', 'superpixel', 'knn'

    # 邻接设置
    connectivity: int = 4  # 4 或 8 邻域

    # 节点特征
    use_position: bool = True  # 包含位置特征
    use_gradient: bool = False  # 包含梯度特征
    position_dim: int = 2  # 位置维度

    # 边特征
    use_distance: bool = True  # 包含距离特征
    use_boundary: bool = False  # 包含边界强度

    # 超像素设置（method='superpixel' 时使用）
    num_superpixels: int = 100

    # KNN 设置（method='knn' 时使用）
    k_neighbors: int = 8


@dataclass
class GNNConfig(ModelConfig):
    """GNN 网络配置"""
    name: str = "photonics_gnn"

    # 输入
    node_feature_dim: int = 16  # 节点特征维度
    edge_feature_dim: int = 4  # 边特征维度

    # 编码器
    hidden_dim: int = 128
    num_layers: int = 4
    conv_type: str = "gcn"  # 'gcn', 'gat', 'graphsage'
    num_heads: int = 4  # GAT 头数

    # 池化
    pooling: str = "mean"  # 'mean', 'sum', 'max', 'attention'

    # 输出
    output_dim: int = 3  # 性能维度

    # 设计形状（用于逆向设计）
    design_shape: Optional[Tuple[int, int]] = None

    # 正则化
    dropout: float = 0.1
    layer_norm: bool = True

    # 图构建配置
    graph_build_config: GraphBuildConfig = field(default_factory=GraphBuildConfig)


# ============================================================================
# 图构建器
# ============================================================================

class GraphBuilder(nn.Module):
    """
    图构建器

    将设计网格转换为图结构，包括节点特征和边索引。

    转换方式:
    1. grid: 将每个像素作为节点，邻域像素连接为边
    2. superpixel: 将超像素区域作为节点
    3. knn: 基于 K 近邻构建图

    使用示例:
    ```python
    config = GraphBuildConfig(connectivity=4, use_position=True)
    builder = GraphBuilder(config)

    design = torch.rand(1, 200, 22)  # 设计网格
    node_features, edge_index, edge_features = builder(design)
    ```
    """

    def __init__(self, config: Optional[GraphBuildConfig] = None):
        super().__init__()
        self.config = config or GraphBuildConfig()

    def forward(
        self,
        design: Tensor,
        return_edge_features: bool = True
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """
        构建图

        Args:
            design: 设计网格 [B, H, W] 或 [H, W]
            return_edge_features: 是否返回边特征

        Returns:
            node_features: 节点特征 [N, F]
            edge_index: 边索引 [2, E]
            edge_features: 边特征 [E, F_e]（可选）
        """
        if design.dim() == 2:
            design = design.unsqueeze(0)

        batch_size, h, w = design.shape

        # 根据方法选择构建方式
        if self.config.method == "grid":
            return self._build_grid_graph(design, h, w, return_edge_features)
        elif self.config.method == "knn":
            return self._build_knn_graph(design, return_edge_features)
        elif self.config.method == "superpixel":
            return self._build_superpixel_graph(design, h, w, return_edge_features)
        else:
            raise ValueError(f"Unknown graph build method: {self.config.method}")

    def _build_grid_graph(
        self,
        design: Tensor,
        h: int,
        w: int,
        return_edge_features: bool
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """构建网格图"""
        batch_size = design.size(0)
        device = design.device

        # 节点特征
        node_features_list = []

        for b in range(batch_size):
            design_b = design[b]  # [H, W]

            # 基础特征：设计值
            features = [design_b.flatten().unsqueeze(1)]  # [N, 1]

            # 位置特征
            if self.config.use_position:
                # 归一化坐标
                y_coords = torch.arange(h, device=device, dtype=torch.float32) / h
                x_coords = torch.arange(w, device=device, dtype=torch.float32) / w

                y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
                pos_features = torch.stack([
                    y_grid.flatten(),
                    x_grid.flatten()
                ], dim=1)  # [N, 2]
                features.append(pos_features)

            # 梯度特征
            if self.config.use_gradient:
                # 计算 Sobel 梯度
                gradient_x = self._compute_gradient(design_b, direction='x')
                gradient_y = self._compute_gradient(design_b, direction='y')
                gradient_mag = torch.sqrt(gradient_x ** 2 + gradient_y ** 2)

                grad_features = torch.stack([
                    gradient_x.flatten(),
                    gradient_y.flatten(),
                    gradient_mag.flatten()
                ], dim=1)  # [N, 3]
                features.append(grad_features)

            node_features = torch.cat(features, dim=1)  # [N, F]
            node_features_list.append(node_features)

        # 合并批次
        all_node_features = torch.cat(node_features_list, dim=0)  # [B*N, F]

        # 构建边索引
        num_nodes_per_graph = h * w
        edge_index = self._build_edge_index(
            h, w, batch_size, num_nodes_per_graph, device
        )

        # 边特征
        edge_features = None
        if return_edge_features:
            edge_features = self._compute_edge_features(
                design, edge_index, h, w, batch_size
            )

        return all_node_features, edge_index, edge_features

    def _build_edge_index(
        self,
        h: int,
        w: int,
        batch_size: int,
        num_nodes_per_graph: int,
        device: torch.device
    ) -> Tensor:
        """构建边索引"""
        edges = []

        for b in range(batch_size):
            offset = b * num_nodes_per_graph

            for i in range(h):
                for j in range(w):
                    node_id = i * w + j + offset

                    # 4-邻域连接
                    if i > 0:  # 上
                        edges.append([node_id, (i-1) * w + j + offset])
                    if i < h - 1:  # 下
                        edges.append([node_id, (i+1) * w + j + offset])
                    if j > 0:  # 左
                        edges.append([node_id, i * w + (j-1) + offset])
                    if j < w - 1:  # 右
                        edges.append([node_id, i * w + (j+1) + offset])

                    # 8-邻域连接
                    if self.config.connectivity == 8:
                        if i > 0 and j > 0:  # 左上
                            edges.append([node_id, (i-1) * w + (j-1) + offset])
                        if i > 0 and j < w - 1:  # 右上
                            edges.append([node_id, (i-1) * w + (j+1) + offset])
                        if i < h - 1 and j > 0:  # 左下
                            edges.append([node_id, (i+1) * w + (j-1) + offset])
                        if i < h - 1 and j < w - 1:  # 右下
                            edges.append([node_id, (i+1) * w + (j+1) + offset])

        edge_index = torch.tensor(edges, device=device, dtype=torch.long).t().contiguous()

        # 添加反向边（无向图）
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

        return edge_index

    def _compute_edge_features(
        self,
        design: Tensor,
        edge_index: Tensor,
        h: int,
        w: int,
        batch_size: int
    ) -> Tensor:
        """计算边特征"""
        device = design.device
        num_nodes_per_graph = h * w

        features_list = []

        src, dst = edge_index[0], edge_index[1]
        num_edges = edge_index.size(1)

        # 距离特征
        if self.config.use_distance:
            # 计算节点间的位置距离
            src_batch = src // num_nodes_per_graph
            dst_batch = dst // num_nodes_per_graph

            src_local = src % num_nodes_per_graph
            dst_local = dst % num_nodes_per_graph

            src_y, src_x = src_local // w, src_local % w
            dst_y, dst_x = dst_local // w, dst_local % w

            # 归一化距离
            distances = torch.sqrt(
                ((src_y.float() - dst_y.float()) / h) ** 2 +
                ((src_x.float() - dst_x.float()) / w) ** 2
            )
            features_list.append(distances.unsqueeze(1))

        # 边界强度
        if self.config.use_boundary:
            # 计算相邻节点的值差异
            design_flat = design.flatten()
            src_vals = design_flat[src]
            dst_vals = design_flat[dst]
            boundary_strength = torch.abs(src_vals - dst_vals).unsqueeze(1)
            features_list.append(boundary_strength)

        if features_list:
            return torch.cat(features_list, dim=1)
        else:
            # 默认返回全1特征
            return torch.ones(edge_index.size(1), 1, device=device)

    def _compute_gradient(self, design: Tensor, direction: str) -> Tensor:
        """计算设计网格的梯度"""
        if direction == 'x':
            kernel = torch.tensor([[-1, 0, 1]], dtype=torch.float32, device=design.device)
            kernel = kernel.view(1, 1, 1, 3)
        else:  # 'y'
            kernel = torch.tensor([[-1], [0], [1]], dtype=torch.float32, device=design.device)
            kernel = kernel.view(1, 1, 3, 1)

        design_4d = design.unsqueeze(0).unsqueeze(0)
        gradient = F.conv2d(design_4d, kernel, padding=1)
        return gradient.squeeze(0).squeeze(0)

    def _build_knn_graph(
        self,
        design: Tensor,
        return_edge_features: bool
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """构建 KNN 图"""
        batch_size, h, w = design.shape
        device = design.device
        num_nodes_per_graph = h * w

        node_features_list = []
        edge_index_list = []
        edge_features_list = []

        for b in range(batch_size):
            design_b = design[b].flatten()  # [N]

            # 节点位置
            y_coords = torch.arange(h, device=device, dtype=torch.float32)
            x_coords = torch.arange(w, device=device, dtype=torch.float32)
            y_grid, x_grid = torch.meshgrid(y_coords, x_coords, indexing='ij')
            positions = torch.stack([y_grid.flatten(), x_grid.flatten()], dim=1)  # [N, 2]

            # 节点特征
            features = [design_b.unsqueeze(1)]
            if self.config.use_position:
                features.append(positions / torch.tensor([h, w], device=device, dtype=torch.float32))
            node_features = torch.cat(features, dim=1)
            node_features_list.append(node_features)

            # KNN 边
            from torch_geometric.nn import knn_graph
            edge_index_b = knn_graph(positions, k=self.config.k_neighbors, loop=False)
            offset = b * num_nodes_per_graph
            edge_index_b = edge_index_b + offset
            edge_index_list.append(edge_index_b)

            if return_edge_features:
                src, dst = edge_index_b
                distances = torch.norm(positions[src % num_nodes_per_graph] -
                                      positions[dst % num_nodes_per_graph], dim=1)
                edge_features_list.append(distances.unsqueeze(1))

        all_node_features = torch.cat(node_features_list, dim=0)
        all_edge_index = torch.cat(edge_index_list, dim=1)

        edge_features = None
        if return_edge_features:
            edge_features = torch.cat(edge_features_list, dim=0)

        return all_node_features, all_edge_index, edge_features

    def _build_superpixel_graph(
        self,
        design: Tensor,
        h: int,
        w: int,
        return_edge_features: bool
    ) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        """
        构建超像素图

        将设计网格分割为超像素区域，每个区域作为一个节点。
        使用简单的基于均匀网格的分割方法，避免额外的依赖。
        """
        batch_size = design.size(0)
        device = design.device

        # 计算网格分割
        num_superpixels = self.config.num_superpixels
        # 计算合适的网格大小
        grid_h = max(1, int(np.sqrt(num_superpixels * h / w)))
        grid_w = max(1, int(num_superpixels / grid_h))

        node_features_list = []
        edge_index_list = []
        edge_features_list = []

        # 计算每个超像素的边界
        region_h = h // grid_h
        region_w = w // grid_w

        for b in range(batch_size):
            design_b = design[b]  # [H, W]
            offset = b * (grid_h * grid_w)

            # 节点特征：每个超像素区域的统计特征
            node_features_b = []

            for i in range(grid_h):
                for j in range(grid_w):
                    # 区域边界
                    y_start = i * region_h
                    y_end = min((i + 1) * region_h, h)
                    x_start = j * region_w
                    x_end = min((j + 1) * region_w, w)

                    # 提取区域
                    region = design_b[y_start:y_end, x_start:x_end]

                    # 计算区域特征
                    features = []

                    # 均值
                    features.append(region.mean().unsqueeze(0))

                    # 标准差
                    features.append(region.std().unsqueeze(0))

                    # 最大值和最小值
                    features.append(region.max().unsqueeze(0))
                    features.append(region.min().unsqueeze(0))

                    # 区域中心位置
                    center_y = (y_start + y_end) / 2 / h
                    center_x = (x_start + x_end) / 2 / w
                    features.append(torch.tensor([center_y, center_x], device=device))

                    # 区域大小
                    features.append(torch.tensor([(y_end - y_start) / h, (x_end - x_start) / w], device=device))

                    node_features_b.append(torch.cat(features))

            node_features_b = torch.stack(node_features_b)  # [num_nodes, F]
            node_features_list.append(node_features_b)

            # 构建边索引：相邻超像素连接
            edges_b = []
            for i in range(grid_h):
                for j in range(grid_w):
                    node_id = i * grid_w + j

                    # 4-邻域连接
                    if i > 0:  # 上
                        edges_b.append([node_id + offset, (i-1) * grid_w + j + offset])
                    if i < grid_h - 1:  # 下
                        edges_b.append([node_id + offset, (i+1) * grid_w + j + offset])
                    if j > 0:  # 左
                        edges_b.append([node_id + offset, i * grid_w + (j-1) + offset])
                    if j < grid_w - 1:  # 右
                        edges_b.append([node_id + offset, i * grid_w + (j+1) + offset])

            edge_index_b = torch.tensor(edges_b, device=device, dtype=torch.long).t().contiguous()
            edge_index_b = torch.cat([edge_index_b, edge_index_b.flip(0)], dim=1)
            edge_index_list.append(edge_index_b)

            # 边特征
            if return_edge_features:
                src, dst = edge_index_b[0], edge_index_b[1]
                # 超像素之间的值差异
                src_features = node_features_b[src % (grid_h * grid_w)]
                dst_features = node_features_b[dst % (grid_h * grid_w)]
                edge_feat = torch.abs(src_features[:, :4] - dst_features[:, :4])  # 使用前4个特征
                edge_features_list.append(edge_feat)

        all_node_features = torch.cat(node_features_list, dim=0)
        all_edge_index = torch.cat(edge_index_list, dim=1)

        edge_features = None
        if return_edge_features:
            edge_features = torch.cat(edge_features_list, dim=0)

        return all_node_features, all_edge_index, edge_features

    def get_batch_indices(self, design: Tensor) -> Tensor:
        """获取每个节点对应的批次索引"""
        if design.dim() == 2:
            design = design.unsqueeze(0)

        batch_size, h, w = design.shape

        # 根据图构建方法计算每图的节点数
        if self.config.method == "superpixel":
            num_superpixels = self.config.num_superpixels
            grid_h = max(1, int(np.sqrt(num_superpixels * h / w)))
            grid_w = max(1, int(num_superpixels / grid_h))
            num_nodes_per_graph = grid_h * grid_w
        else:
            num_nodes_per_graph = h * w

        batch_indices = torch.arange(batch_size, device=design.device)
        batch_indices = batch_indices.repeat_interleave(num_nodes_per_graph)

        return batch_indices


# ============================================================================
# 图卷积层
# ============================================================================

class GraphConvLayer(nn.Module):
    """
    图卷积层

    实现 GCN 风格的消息传递:
    H^{l+1} = σ(D^{-1/2} A D^{-1/2} H^{l} W^{l})

    使用示例:
    ```python
    conv = GraphConvLayer(64, 128, dropout=0.1)
    x = conv(x, edge_index)
    ```
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        layer_norm: bool = True,
        residual: bool = True
    ):
        super().__init__()
        self.conv = GCNConv(in_channels, out_channels)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm = nn.LayerNorm(out_channels) if layer_norm else nn.Identity()
        self.residual = residual and (in_channels == out_channels)

        if residual and in_channels != out_channels:
            self.res_proj = nn.Linear(in_channels, out_channels)
        else:
            self.res_proj = None

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """
        前向传播

        Args:
            x: 节点特征 [N, in_channels]
            edge_index: 边索引 [2, E]

        Returns:
            更新后的节点特征 [N, out_channels]
        """
        identity = x

        # 图卷积
        x = self.conv(x, edge_index)
        x = self.norm(x)
        x = F.relu(x)
        x = self.dropout(x)

        # 残差连接
        if self.residual:
            if self.res_proj is not None:
                identity = self.res_proj(identity)
            x = x + identity

        return x


# ============================================================================
# 图注意力层
# ============================================================================

class GraphAttentionLayer(nn.Module):
    """
    图注意力层 (GAT)

    使用注意力机制聚合邻居信息:
    α_{ij} = softmax(LeakyReLU(a^T [Wh_i || Wh_j]))
    h_i' = σ(Σ_{j∈N(i)} α_{ij} W h_j)

    使用示例:
    ```python
    attn = GraphAttentionLayer(64, 128, num_heads=4, dropout=0.1)
    x = attn(x, edge_index)
    ```
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_heads: int = 4,
        dropout: float = 0.0,
        concat: bool = True,
        layer_norm: bool = True,
        residual: bool = True
    ):
        super().__init__()
        self.num_heads = num_heads
        self.concat = concat

        # GAT 卷积
        self.conv = GATConv(
            in_channels,
            out_channels // num_heads if concat else out_channels,
            heads=num_heads,
            concat=concat,
            dropout=dropout,
            add_self_loops=True
        )

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # 输出维度
        out_dim = out_channels if concat else out_channels

        self.norm = nn.LayerNorm(out_dim) if layer_norm else nn.Identity()
        self.residual = residual

        if residual and in_channels != out_dim:
            self.res_proj = nn.Linear(in_channels, out_dim)
        else:
            self.res_proj = None

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        return_attention: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        前向传播

        Args:
            x: 节点特征 [N, in_channels]
            edge_index: 边索引 [2, E]
            return_attention: 是否返回注意力权重

        Returns:
            更新后的节点特征 [N, out_channels]
            注意力权重 [E, num_heads]（可选）
        """
        identity = x

        # 图注意力卷积
        x, attention = self.conv(x, edge_index, return_attention_weights=True)
        x = self.norm(x)
        x = F.elu(x)
        x = self.dropout(x)

        # 残差连接
        if self.residual:
            if self.res_proj is not None:
                identity = self.res_proj(identity)
            x = x + identity

        if return_attention:
            return x, attention
        return x


# ============================================================================
# GraphSAGE 层
# ============================================================================

class GraphSAGELayer(nn.Module):
    """
    GraphSAGE 层

    使用采样和聚合策略进行图卷积:
    h_i^{l+1} = σ(W * CONCAT(h_i^l, AGGREGATE({h_j^l, j ∈ N(i)})))

    支持多种聚合方式:
    - mean: 平均聚合
    - max: 最大池化聚合
    - lstm: LSTM 聚合

    使用示例:
    ```python
    sage = GraphSAGELayer(64, 128, aggr='mean', dropout=0.1)
    x = sage(x, edge_index)
    ```
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        aggr: str = 'mean',
        dropout: float = 0.0,
        layer_norm: bool = True,
        residual: bool = True
    ):
        super().__init__()
        self.aggr = aggr
        self.residual = residual

        # 使用 PyG 的 SAGEConv
        from torch_geometric.nn import SAGEConv

        self.conv = SAGEConv(
            in_channels,
            out_channels,
            aggr=aggr,
            normalize=True
        )

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm = nn.LayerNorm(out_channels) if layer_norm else nn.Identity()

        if residual and in_channels != out_channels:
            self.res_proj = nn.Linear(in_channels, out_channels)
        else:
            self.res_proj = None

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """
        前向传播

        Args:
            x: 节点特征 [N, in_channels]
            edge_index: 边索引 [2, E]

        Returns:
            更新后的节点特征 [N, out_channels]
        """
        identity = x

        # GraphSAGE 卷积
        x = self.conv(x, edge_index)
        x = self.norm(x)
        x = F.relu(x)
        x = self.dropout(x)

        # 残差连接
        if self.residual:
            if self.res_proj is not None:
                identity = self.res_proj(identity)
            x = x + identity

        return x


# ============================================================================
# 图池化层
# ============================================================================

class GlobalPooling(nn.Module):
    """
    全局图池化层

    将节点特征聚合为图级别表示:
    h_G = READOUT({h_i | i ∈ V})

    支持多种池化方式:
    - mean: 平均池化
    - sum: 求和池化
    - max: 最大池化
    - attention: 注意力池化

    使用示例:
    ```python
    pool = GlobalPooling(pooling='mean')
    graph_embedding = pool(node_features, batch_indices)
    ```
    """

    def __init__(
        self,
        pooling: str = 'mean',
        in_channels: Optional[int] = None,
        attention_heads: int = 1
    ):
        super().__init__()
        self.pooling = pooling

        if pooling == 'attention':
            if in_channels is None:
                raise ValueError("in_channels required for attention pooling")
            self.attention = nn.Sequential(
                nn.Linear(in_channels, in_channels // 4),
                nn.Tanh(),
                nn.Linear(in_channels // 4, 1)
            )

    def forward(self, x: Tensor, batch: Tensor) -> Tensor:
        """
        池化操作

        Args:
            x: 节点特征 [N, F]
            batch: 批次索引 [N]，指示每个节点属于哪个图

        Returns:
            图嵌入 [B, F]
        """
        if self.pooling == 'mean':
            return global_mean_pool(x, batch)
        elif self.pooling == 'sum':
            return global_add_pool(x, batch)
        elif self.pooling == 'max':
            return global_max_pool(x, batch)
        elif self.pooling == 'attention':
            return self._attention_pooling(x, batch)
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")

    def _attention_pooling(self, x: Tensor, batch: Tensor) -> Tensor:
        """注意力池化"""
        # 计算注意力权重
        attention_scores = self.attention(x)  # [N, 1]

        # 应用 softmax 归一化（按图）
        num_graphs = batch.max().item() + 1
        attention_weights = torch.zeros_like(attention_scores)

        for i in range(num_graphs):
            mask = batch == i
            attention_weights[mask] = F.softmax(attention_scores[mask], dim=0)

        # 加权求和
        weighted_x = x * attention_weights
        return global_add_pool(weighted_x, batch)


# ============================================================================
# 图编码器
# ============================================================================

class GraphEncoder(nn.Module):
    """
    图编码器

    将图数据编码为图嵌入向量。

    架构:
        输入: 节点特征 [N, F], 边索引 [2, E]
          ↓
        [NodeFeatureProjector] 特征投影
          ↓
        [GraphConvLayer] × L 层 或 [GraphAttentionLayer] × K 层
          ↓
        [GlobalPooling] 图池化
          ↓
        输出: 图嵌入 [B, hidden_dim]

    使用示例:
    ```python
    config = GNNConfig(hidden_dim=128, num_layers=4, conv_type='gat')
    encoder = GraphEncoder(config)

    node_features, edge_index, batch = ...  # 图数据
    graph_embedding = encoder(node_features, edge_index, batch)
    ```
    """

    def __init__(self, config: GNNConfig):
        super().__init__()
        self.config = config

        # 节点特征投影
        self.node_projector = nn.Sequential(
            nn.Linear(config.node_feature_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim) if config.layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )

        # 构建卷积层
        self.conv_layers = nn.ModuleList()

        for i in range(config.num_layers):
            if config.conv_type == 'gcn':
                layer = GraphConvLayer(
                    in_channels=config.hidden_dim,
                    out_channels=config.hidden_dim,
                    dropout=config.dropout,
                    layer_norm=config.layer_norm,
                    residual=True
                )
            elif config.conv_type == 'gat':
                layer = GraphAttentionLayer(
                    in_channels=config.hidden_dim,
                    out_channels=config.hidden_dim,
                    num_heads=config.num_heads,
                    dropout=config.dropout,
                    layer_norm=config.layer_norm,
                    residual=True
                )
            elif config.conv_type == 'graphsage':
                layer = GraphSAGELayer(
                    in_channels=config.hidden_dim,
                    out_channels=config.hidden_dim,
                    aggr='mean',
                    dropout=config.dropout,
                    layer_norm=config.layer_norm,
                    residual=True
                )
            else:
                raise ValueError(f"Unknown conv type: {config.conv_type}")

            self.conv_layers.append(layer)

        # 全局池化
        self.pooling = GlobalPooling(
            pooling=config.pooling,
            in_channels=config.hidden_dim
        )

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        batch: Tensor,
        return_node_features: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        编码图

        Args:
            x: 节点特征 [N, F]
            edge_index: 边索引 [2, E]
            batch: 批次索引 [N]
            return_node_features: 是否返回节点特征

        Returns:
            graph_embedding: 图嵌入 [B, hidden_dim]
            node_features: 节点特征 [N, hidden_dim]（可选）
        """
        # 特征投影
        x = self.node_projector(x)

        # 图卷积层
        for conv in self.conv_layers:
            x = conv(x, edge_index)

        # 全局池化
        graph_embedding = self.pooling(x, batch)

        if return_node_features:
            return graph_embedding, x

        return graph_embedding


# ============================================================================
# 图解码器
# ============================================================================

class GraphDecoder(nn.Module):
    """
    图解码器

    从图嵌入生成设计网格或性能预测。

    支持:
    1. 性能预测头: 图嵌入 → 性能指标
    2. 设计生成头: 图嵌入 → 设计网格

    使用示例:
    ```python
    # 性能预测
    decoder = GraphDecoder(config, mode='performance')
    performance = decoder(graph_embedding)

    # 设计生成
    decoder = GraphDecoder(config, mode='design', design_shape=(200, 22))
    design = decoder(graph_embedding)
    ```
    """

    def __init__(
        self,
        config: GNNConfig,
        mode: str = 'performance'
    ):
        super().__init__()
        self.config = config
        self.mode = mode

        if mode == 'performance':
            # 性能预测头
            self.head = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim // 2),
                nn.LayerNorm(config.hidden_dim // 2) if config.layer_norm else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim // 2, config.output_dim)
            )

        elif mode == 'design':
            if config.design_shape is None:
                raise ValueError("design_shape required for design generation mode")

            h, w = config.design_shape
            self.design_shape = config.design_shape

            # 设计生成头
            self.head = nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim * 2),
                nn.LayerNorm(config.hidden_dim * 2) if config.layer_norm else nn.Identity(),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim * 2, h * w),
                nn.Sigmoid()
            )

        else:
            raise ValueError(f"Unknown decoder mode: {mode}")

    def forward(self, graph_embedding: Tensor) -> Tensor:
        """
        解码

        Args:
            graph_embedding: 图嵌入 [B, hidden_dim]

        Returns:
            performance 模式: 性能指标 [B, output_dim]
            design 模式: 设计网格 [B, H, W]
        """
        output = self.head(graph_embedding)

        if self.mode == 'design':
            output = output.view(-1, *self.design_shape)

        return output


# ============================================================================
# PhotonicsGNN 主模型
# ============================================================================

class PhotonicsGNN(BaseModel):
    """
    光子学图神经网络

    完整的 GNN 模型，支持:
    1. 性能预测: 设计网格 → 图构建 → 编码 → 性能预测
    2. 逆向设计: 性能目标 → 逆向网络 → 设计网格（需配合逆向模型）

    架构:
        设计网格 [B, H, W]
              ↓
        GraphBuilder (构建图)
              ↓
        节点特征 + 边索引
              ↓
        GraphEncoder (编码)
              ↓
        图嵌入 [B, hidden_dim]
              ↓
        GraphDecoder (解码)
              ↓
        性能预测 [B, output_dim] 或 设计生成 [B, H, W]

    使用示例:
    ```python
    # 创建 GNN
    config = GNNConfig(
        hidden_dim=128,
        num_layers=4,
        conv_type='gat',
        output_dim=3,
        design_shape=(200, 22)
    )
    gnn = PhotonicsGNN(config)

    # 性能预测
    design = torch.rand(4, 200, 22)
    performance = gnn.predict_performance(design)

    # 逆向设计（需要训练逆向网络）
    target_perf = torch.tensor([[0.85, 0.1, 0.05]])
    design = gnn.inverse_design(target_perf)
    ```
    """

    def __init__(self, config: Optional[GNNConfig] = None):
        config = config or GNNConfig()
        super().__init__(config)
        self.config = config

        # 图构建器
        self.graph_builder = GraphBuilder(config.graph_build_config)

        # 图编码器
        self.encoder = GraphEncoder(config)

        # 性能预测解码器
        self.performance_decoder = GraphDecoder(config, mode='performance')

        # 设计生成解码器（可选）
        self.design_decoder = None
        if config.design_shape is not None:
            self.design_decoder = GraphDecoder(config, mode='design')

        # 特征投影（将原始节点特征投影到配置维度）
        node_feature_dim = self._compute_node_feature_dim()
        self.feature_projection = nn.Linear(node_feature_dim, config.node_feature_dim)

    def _compute_node_feature_dim(self) -> int:
        """计算节点特征维度"""
        dim = 1  # 设计值

        if self.config.graph_build_config.use_position:
            dim += self.config.graph_build_config.position_dim

        if self.config.graph_build_config.use_gradient:
            dim += 3  # gradient_x, gradient_y, gradient_mag

        return dim

    def forward(self, design: Tensor) -> Tensor:
        """
        前向传播（性能预测）

        Args:
            design: 设计网格 [B, H, W]

        Returns:
            性能预测 [B, output_dim]
        """
        return self.predict_performance(design)

    def predict_performance(self, design: Tensor) -> Tensor:
        """
        预测设计性能

        Args:
            design: 设计网格 [B, H, W]

        Returns:
            性能预测 [B, output_dim]
        """
        # 构建图
        node_features, edge_index, _ = self.graph_builder(design)

        # 投影节点特征
        node_features = self.feature_projection(node_features)

        # 获取批次索引
        batch = self.graph_builder.get_batch_indices(design)

        # 编码
        graph_embedding = self.encoder(node_features, edge_index, batch)

        # 解码性能
        performance = self.performance_decoder(graph_embedding)

        return performance

    def encode_design(self, design: Tensor) -> Tensor:
        """
        编码设计为图嵌入

        Args:
            design: 设计网格 [B, H, W]

        Returns:
            图嵌入 [B, hidden_dim]
        """
        # 构建图
        node_features, edge_index, _ = self.graph_builder(design)

        # 投影节点特征
        node_features = self.feature_projection(node_features)

        # 获取批次索引
        batch = self.graph_builder.get_batch_indices(design)

        # 编码
        graph_embedding = self.encoder(node_features, edge_index, batch)

        return graph_embedding

    def inverse_design(
        self,
        target_performance: Tensor,
        n_iterations: int = 100,
        lr: float = 0.1
    ) -> Tuple[Tensor, Tensor]:
        """
        逆向设计：从目标性能优化设计

        Args:
            target_performance: 目标性能 [B, output_dim]
            n_iterations: 优化迭代次数
            lr: 学习率

        Returns:
            design: 优化后的设计 [B, H, W]
            predicted_performance: 预测性能 [B, output_dim]
        """
        if self.config.design_shape is None:
            raise ValueError("design_shape must be set for inverse design")

        batch_size = target_performance.size(0)
        device = target_performance.device

        # 初始化设计
        design = torch.rand(
            batch_size, *self.config.design_shape,
            device=device, requires_grad=True
        )

        optimizer = torch.optim.Adam([design], lr=lr)

        for _ in range(n_iterations):
            optimizer.zero_grad()

            # 预测性能
            pred_perf = self.predict_performance(design)

            # 计算损失
            loss = F.mse_loss(pred_perf, target_performance)

            # 反向传播
            loss.backward()
            optimizer.step()

            # 限制设计范围
            with torch.no_grad():
                design.data = design.data.clamp(0, 1)

        # 最终预测
        with torch.no_grad():
            final_design = design.detach()
            final_perf = self.predict_performance(final_design)

        return final_design, final_perf

    def compute_loss(
        self,
        output: Tensor,
        target: Tensor,
        **kwargs
    ) -> Tensor:
        """
        计算损失

        Args:
            output: 预测性能
            target: 目标性能

        Returns:
            损失值
        """
        return F.mse_loss(output, target)

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            'name': self.config.name,
            'device': str(self.device),
            'hidden_dim': self.config.hidden_dim,
            'num_layers': self.config.num_layers,
            'conv_type': self.config.conv_type,
            'pooling': self.config.pooling,
            'design_shape': self.config.design_shape,
            'parameters': self.count_parameters()
        }


# ============================================================================
# 便捷创建函数
# ============================================================================

def create_gnn_for_challenge(
    challenge_name: str,
    hidden_dim: int = 128,
    num_layers: int = 4,
    conv_type: str = 'gat',
    output_dim: int = 3,
    device: str = 'auto'
) -> PhotonicsGNN:
    """
    为特定挑战创建 GNN

    Args:
        challenge_name: 挑战名称
        hidden_dim: 隐藏层维度
        num_layers: 层数
        conv_type: 卷积类型 ('gcn', 'gat')
        output_dim: 输出维度
        device: 计算设备

    Returns:
        配置好的 PhotonicsGNN
    """
    from challenges import ChallengeFactory

    # 获取挑战以确定设计形状
    challenge = ChallengeFactory.create(challenge_name)
    design_shape = challenge.spec.get_grid_shape()

    # 创建配置
    config = GNNConfig(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        conv_type=conv_type,
        output_dim=output_dim,
        design_shape=design_shape,
        device=device
    )

    return PhotonicsGNN(config)


# ============================================================================
# 数据类（用于批量处理）
# ============================================================================

class GraphBatch:
    """
    图批次数据容器

    用于组织一批图数据，方便传递给模型。
    """

    def __init__(
        self,
        node_features: Tensor,
        edge_index: Tensor,
        batch: Tensor,
        edge_features: Optional[Tensor] = None,
        y: Optional[Tensor] = None
    ):
        self.node_features = node_features
        self.edge_index = edge_index
        self.batch = batch
        self.edge_features = edge_features
        self.y = y

    def to(self, device: torch.device) -> 'GraphBatch':
        """移动到指定设备"""
        return GraphBatch(
            node_features=self.node_features.to(device),
            edge_index=self.edge_index.to(device),
            batch=self.batch.to(device),
            edge_features=self.edge_features.to(device) if self.edge_features is not None else None,
            y=self.y.to(device) if self.y is not None else None
        )

        @property

        def num_graphs(self) -> int:

            """图的数量"""

            return self.batch.max().item() + 1

    

    

    # ============================================================================

    # 计算图节点集成

    # ============================================================================

    

    class GNNNode:

        """

        GNN 计算图节点

    

        将 GNN 模型集成到计算图框架中，支持自动微分和梯度传播。

    

        使用示例:

        ```python

        # 创建 GNN 节点

        config = GNNConfig(hidden_dim=128, output_dim=3)

        gnn_node = GNNNode("performance_predictor", config)

    

        # 添加到计算图

        design_node = ...  # 设计参数节点

        gnn_node.add_input(design_node)

    

        # 执行前向计算

        performance = gnn_node.forward()

        ```

        """

    

        def __init__(

            self,

            name: str,

            config: Optional[GNNConfig] = None,

            params: Optional[Dict[str, Any]] = None

        ):

            """

            初始化 GNN 节点

    

            Args:

                name: 节点名称

                config: GNN 配置

                params: 额外参数

            """

            self.name = name

            self.config = config or GNNConfig()

            self.params = params or {}

    

            # 创建 GNN 模型

            self._model = PhotonicsGNN(self.config)

    

            # 输入输出节点列表

            self._inputs: List['GNNNode'] = []

            self._outputs: List['GNNNode'] = []

    

            # 缓存输出

            self._cached_output: Optional[Tensor] = None

    

        def add_input(self, node: 'GNNNode') -> None:

            """添加输入节点"""

            self._inputs.append(node)

            node._outputs.append(self)

    

        def add_output(self, node: 'GNNNode') -> None:

            """添加输出节点"""

            self._outputs.append(node)

            node._inputs.append(self)

    

        def forward(self, design: Optional[Tensor] = None, **kwargs) -> Tensor:

            """

            执行前向计算

    

            Args:

                design: 设计网格（可选，如果没有输入节点）

                **kwargs: 额外参数

    

            Returns:

                预测的性能或生成的设计

            """

            # 如果没有提供设计，从输入节点获取

            if design is None:

                if not self._inputs:

                    raise ValueError("No design provided and no input nodes")

                design = self._inputs[0]._cached_output

    

            if design is None:

                raise ValueError("Design tensor is None")

    

            # 根据模式选择前向方式

            mode = self.params.get('mode', 'performance')

    

            if mode == 'performance':

                self._cached_output = self._model.predict_performance(design)

            elif mode == 'encode':

                self._cached_output = self._model.encode_design(design)

            elif mode == 'inverse':

                target = self.params.get('target_performance')

                if target is None:

                    raise ValueError("target_performance required for inverse mode")

                self._cached_output, _ = self._model.inverse_design(target)

            else:

                self._cached_output = self._model(design)

    

            return self._cached_output

    

        def backward(self, grad_output: Tensor) -> None:

            """反向传播梯度"""

            if isinstance(self._cached_output, Tensor):

                torch.autograd.backward(self._cached_output, grad_output)

    

        def clear_cache(self) -> None:

            """清理缓存"""

            self._cached_output = None

    

        def load_weights(self, path: Union[str, Path]) -> None:

            """加载模型权重"""

            self._model.load(path)

    

        def save_weights(self, path: Union[str, Path]) -> None:

            """保存模型权重"""

            self._model.save(path)

    

        def to(self, device: torch.device) -> 'GNNNode':

            """移动到指定设备"""

            self._model = self._model.to(device)

            return self

    

        @property

        def model(self) -> PhotonicsGNN:

            """获取底层模型"""

            return self._model

    

        def __repr__(self) -> str:

            return f"GNNNode(name={self.name}, config={self.config.name})"

    

    

    

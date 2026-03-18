"""
神经网络计算节点

将深度学习模型集成到计算图框架中。
"""

from typing import Dict, Optional, Any, Union, Tuple
from pathlib import Path
import torch
import torch.nn as nn
from torch import Tensor

from core.node import Node
from core.utils.typing import TensorLike


class NeuralNetworkNode(Node):
    """
    神经网络节点基类
    
    封装 PyTorch 模型，使其可作为计算图节点使用。
    """
    
    def __init__(
        self,
        name: str,
        model: nn.Module,
        input_node: Optional[Node] = None,
        device: str = 'auto'
    ):
        """
        Args:
            name: 节点名称
            model: PyTorch 模型
            input_node: 输入节点
            device: 计算设备
        """
        super().__init__(name)
        self.model = model
        
        # 设置设备
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        
        # 连接输入节点
        if input_node is not None:
            self.add_input(input_node)
    
    def forward(self, **kwargs) -> TensorLike:
        """前向传播"""
        if self._inputs:
            input_data = self._inputs[0].forward(**kwargs)
            return self._run_model(input_data)
        raise ValueError("No input node connected")
    
    def _run_model(self, input_data: Tensor) -> Tensor:
        """运行模型"""
        self.model.eval()
        with torch.no_grad():
            if isinstance(input_data, Tensor):
                input_data = input_data.to(self.device)
            output = self.model(input_data)
        return output
    
    def train_mode(self):
        """设置为训练模式"""
        self.model.train()
    
    def eval_mode(self):
        """设置为评估模式"""
        self.model.eval()
    
    def load_weights(self, path: Union[str, Path]):
        """加载模型权重"""
        checkpoint = torch.load(path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
    
    def save_weights(self, path: Union[str, Path]):
        """保存模型权重"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)


class TNNForwardNode(NeuralNetworkNode):
    """
    TNN 前向网络节点
    
    作为仿真代理，快速预测设计性能。
    
    使用示例:
    ```python
    from models.inverse.tnn import ForwardNetwork, ForwardNetworkConfig
    from core.nodes.neural_network import TNNForwardNode
    
    # 创建前向网络
    config = ForwardNetworkConfig(design_shape=(200, 22), performance_dim=3)
    forward_net = ForwardNetwork(config)
    
    # 创建节点
    param_node = ParameterizationNode('param', initial_design)
    surrogate_node = TNNForwardNode('surrogate', forward_net, param_node)
    
    # 获取预测性能
    predicted_perf = surrogate_node.forward()
    ```
    """
    
    def __init__(
        self,
        name: str,
        model: nn.Module,
        design_node: Node,
        device: str = 'auto'
    ):
        """
        Args:
            name: 节点名称
            model: 前向网络模型
            design_node: 设计参数节点
            device: 计算设备
        """
        super().__init__(name, model, design_node, device)
    
    def forward(self, **kwargs) -> TensorLike:
        """
        预测设计性能
        
        Returns:
            性能指标 [B, performance_dim]
        """
        design = self._inputs[0].forward(**kwargs)
        return self._run_model(design)


class TNNInverseNode(NeuralNetworkNode):
    """
    TNN 逆向网络节点
    
    从性能目标生成设计参数。
    
    使用示例:
    ```python
    from models.inverse.tnn import InverseNetwork, InverseNetworkConfig
    from core.nodes.neural_network import TNNInverseNode
    
    # 创建逆向网络
    config = InverseNetworkConfig(design_shape=(200, 22), performance_dim=3)
    inverse_net = InverseNetwork(config)
    
    # 创建目标性能节点
    target_perf = TargetPerformanceNode('target', torch.tensor([[0.85, 0.1, 0.05]]))
    
    # 创建逆向节点
    inverse_node = TNNInverseNode('inverse', inverse_net, target_perf)
    
    # 生成设计
    design = inverse_node.forward()
    ```
    """
    
    def __init__(
        self,
        name: str,
        model: nn.Module,
        performance_node: Node,
        device: str = 'auto'
    ):
        """
        Args:
            name: 节点名称
            model: 逆向网络模型
            performance_node: 性能目标节点
            device: 计算设备
        """
        super().__init__(name, model, performance_node, device)
    
    def forward(self, **kwargs) -> TensorLike:
        """
        生成设计参数
        
        Returns:
            设计参数 [B, H, W]
        """
        target_performance = self._inputs[0].forward(**kwargs)
        return self._run_model(target_performance)


class TNNNode(Node):
    """
    完整 TNN 节点
    
    封装完整的串联神经网络，支持：
    1. 前向预测：设计 → 性能
    2. 逆向设计：性能 → 设计
    
    使用示例:
    ```python
    from models.inverse.tnn import TandemNetwork, TandemNetworkConfig
    from core.nodes.neural_network import TNNNode
    
    # 创建 TNN
    config = TandemNetworkConfig(
        forward_config=ForwardNetworkConfig(design_shape=(200, 22)),
        inverse_config=InverseNetworkConfig(design_shape=(200, 22))
    )
    tnn = TandemNetwork(config)
    
    # 方式1: 作为前向代理
    design_node = ParameterizationNode('design', initial_design)
    forward_node = TNNNode('tnn_forward', tnn, mode='forward', input_node=design_node)
    performance = forward_node.forward()
    
    # 方式2: 作为逆向生成器
    perf_node = TargetPerformanceNode('target', target_performance)
    inverse_node = TNNNode('tnn_inverse', tnn, mode='inverse', input_node=perf_node)
    design = inverse_node.forward()
    ```
    """
    
    def __init__(
        self,
        name: str,
        tnn_model,  # TandemNetwork 类型
        mode: str = 'forward',
        input_node: Optional[Node] = None,
        device: str = 'auto'
    ):
        """
        Args:
            name: 节点名称
            tnn_model: TandemNetwork 模型
            mode: 运行模式 ('forward' 或 'inverse')
            input_node: 输入节点
            device: 计算设备
        """
        super().__init__(name)
        self.tnn = tnn_model
        self.mode = mode
        
        # 设置设备
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.tnn.to(self.device)
        
        # 连接输入节点
        if input_node is not None:
            self.add_input(input_node)
    
    def forward(self, **kwargs) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        根据模式执行前向或逆向计算
        
        Returns:
            forward 模式: 性能指标
            inverse 模式: (设计参数, 预测性能)
        """
        if not self._inputs:
            raise ValueError("No input node connected")
        
        input_data = self._inputs[0].forward(**kwargs)
        
        if self.mode == 'forward':
            return self._forward_pass(input_data)
        elif self.mode == 'inverse':
            return self._inverse_pass(input_data)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
    
    def _forward_pass(self, design: Tensor) -> Tensor:
        """前向预测"""
        self.tnn.eval()
        with torch.no_grad():
            design = design.to(self.device)
            performance = self.tnn.forward(design)
        return performance
    
    def _inverse_pass(self, target_performance: Tensor) -> Tuple[Tensor, Tensor]:
        """逆向设计"""
        self.tnn.eval()
        with torch.no_grad():
            target_performance = target_performance.to(self.device)
            design, pred_performance = self.tnn.inverse_design(target_performance)
        return design, pred_performance
    
    def set_mode(self, mode: str):
        """设置运行模式"""
        if mode not in ['forward', 'inverse']:
            raise ValueError(f"Invalid mode: {mode}")
        self.mode = mode
    
    def load_pretrained(self, path: Union[str, Path]):
        """加载预训练模型"""
        self.tnn.load(path)
    
    def save_model(self, path: Union[str, Path]):
        """保存模型"""
        self.tnn.save(path)


class TargetPerformanceNode(Node):
    """
    目标性能节点
    
    存储并提供目标性能指标。
    """
    
    def __init__(
        self,
        name: str,
        target_performance: Tensor,
        requires_grad: bool = False
    ):
        """
        Args:
            name: 节点名称
            target_performance: 目标性能指标
            requires_grad: 是否需要梯度
        """
        super().__init__(name)
        if requires_grad:
            self.target = torch.nn.Parameter(target_performance, requires_grad=True)
        else:
            self.target = target_performance.detach().clone()
    
    def forward(self, **kwargs) -> TensorLike:
        """返回目标性能"""
        return self.target
    
    def set_target(self, target: Tensor):
        """设置新的目标性能"""
        self.target = target.detach().clone()


class SurrogateSimulationNode(Node):
    """
    代理仿真节点
    
    使用神经网络代理替代真实仿真，支持：
    1. 快速性能预测
    2. 梯度计算（通过神经网络）
    3. 不确定性估计（可选）
    """
    
    def __init__(
        self,
        name: str,
        surrogate_model: nn.Module,
        design_node: Node,
        uncertainty_model: Optional[nn.Module] = None,
        device: str = 'auto'
    ):
        """
        Args:
            name: 节点名称
            surrogate_model: 代理模型
            design_node: 设计参数节点
            uncertainty_model: 不确定性模型（可选）
            device: 计算设备
        """
        super().__init__(name)
        self.surrogate = surrogate_model
        self.uncertainty_model = uncertainty_model
        
        # 设置设备
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.surrogate.to(self.device)
        if self.uncertainty_model is not None:
            self.uncertainty_model.to(self.device)
        
        self.add_input(design_node)
    
    def forward(self, **kwargs) -> Dict[str, Tensor]:
        """
        执行代理仿真
        
        Returns:
            包含性能预测和可选不确定性的字典
        """
        design = self._inputs[0].forward(**kwargs)
        design = design.to(self.device)
        
        self.surrogate.eval()
        with torch.no_grad():
            performance = self.surrogate(design)
        
        result = {'performance': performance}
        
        # 计算不确定性
        if self.uncertainty_model is not None:
            self.uncertainty_model.eval()
            with torch.no_grad():
                uncertainty = self.uncertainty_model(design)
            result['uncertainty'] = uncertainty
        
        self._cached_output = result
        return result
    
    def backward(self, grad_output: Tensor):
        """
        反向传播（支持梯度优化）
        
        由于代理模型是可微的，可以直接计算梯度。
        """
        if self._cached_output is None:
            return
        
        # 使用代理模型的梯度
        design = self._inputs[0]._cached_output
        design = design.to(self.device)
        design.requires_grad_(True)
        
        performance = self.surrogate(design)
        
        # 计算设计参数的梯度
        performance.backward(grad_output.to(self.device))
        
        if design.grad is not None:
            # 将梯度传递给设计节点
            self._inputs[0].backward(design.grad)


# 便捷创建函数

def create_tnn_node_for_challenge(
    challenge_name: str,
    mode: str = 'forward',
    pretrained_path: Optional[str] = None,
    device: str = 'auto'
) -> TNNNode:
    """
    为特定挑战创建 TNN 节点
    
    Args:
        challenge_name: 挑战名称
        mode: 运行模式
        pretrained_path: 预训练模型路径
        device: 计算设备
        
    Returns:
        配置好的 TNNNode
    """
    from models.inverse.tnn import create_tnn_for_challenge
    
    tnn = create_tnn_for_challenge(challenge_name, device=device)
    
    if pretrained_path:
        tnn.load(pretrained_path)
    
    return TNNNode(
        name=f'tnn_{challenge_name}',
        tnn_model=tnn,
        mode=mode,
        device=device
    )


class MDNNode(Node):
    """
    混合密度网络节点
    
    封装 MDN 模型，支持：
    1. 获取设计分布参数
    2. 从分布中采样
    3. 选择最优设计
    
    使用示例:
    ```python
    from models.inverse.mdn import MDN, MDNConfig
    from core.nodes.neural_network import MDNNode
    
    # 创建 MDN
    config = MDNConfig(
        input_dim=3,
        output_dim=200*22,
        design_shape=(200, 22),
        n_components=5
    )
    mdn = MDN(config)
    
    # 创建目标性能节点
    target_perf = TargetPerformanceNode('target', torch.tensor([[0.85, 0.8, 0.1]]))
    
    # 创建 MDN 节点
    mdn_node = MDNNode('mdn', mdn, target_perf)
    
    # 方式1: 获取分布参数
    pi, mu, sigma = mdn_node.forward(mode='params')
    
    # 方式2: 采样多个设计
    samples = mdn_node.forward(mode='sample', n_samples=10)
    
    # 方式3: 获取最可能的设计
    best_design = mdn_node.forward(mode='mode')
    ```
    """
    
    def __init__(
        self,
        name: str,
        mdn_model,  # MDN 类型
        performance_node: Optional[Node] = None,
        forward_model: Optional[nn.Module] = None,
        device: str = 'auto'
    ):
        """
        Args:
            name: 节点名称
            mdn_model: MDN 模型
            performance_node: 性能目标节点
            forward_model: 前向模型（用于 sample_best 模式）
            device: 计算设备
        """
        super().__init__(name)
        self.mdn = mdn_model
        self.forward_model = forward_model
        
        # 设置设备
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.mdn.to(self.device)
        if self.forward_model is not None:
            self.forward_model.to(self.device)
        
        # 连接输入节点
        if performance_node is not None:
            self.add_input(performance_node)
    
    def forward(
        self,
        mode: str = 'sample',
        n_samples: int = 1,
        **kwargs
    ) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor]]:
        """
        根据模式执行不同操作
        
        Args:
            mode: 运行模式
                - 'params': 返回分布参数 (pi, mu, sigma)
                - 'sample': 采样多个设计
                - 'mode': 返回最可能的设计
                - 'best': 采样并选择最优设计
            n_samples: 采样数量（仅 sample 和 best 模式）
            
        Returns:
            params 模式: (pi, mu, sigma)
            sample 模式: samples [B, n_samples, H, W]
            mode 模式: best_design [B, H, W]
            best 模式: (best_design, best_performance)
        """
        if not self._inputs:
            raise ValueError("No performance node connected")
        
        target_performance = self._inputs[0].forward(**kwargs)
        target_performance = target_performance.to(self.device)
        
        if mode == 'params':
            return self._get_params(target_performance)
        elif mode == 'sample':
            return self._sample(target_performance, n_samples)
        elif mode == 'mode':
            return self._get_mode(target_performance)
        elif mode == 'best':
            return self._sample_best(target_performance, n_samples)
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    def _get_params(self, performance: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """获取分布参数"""
        self.mdn.eval()
        with torch.no_grad():
            pi, mu, sigma = self.mdn(performance)
        return pi, mu, sigma
    
    def _sample(self, performance: Tensor, n_samples: int) -> Tensor:
        """采样设计"""
        self.mdn.eval()
        with torch.no_grad():
            samples = self.mdn.sample(performance, n_samples)
        return samples
    
    def _get_mode(self, performance: Tensor) -> Tensor:
        """获取最可能的设计"""
        self.mdn.eval()
        with torch.no_grad():
            design = self.mdn.sample_mode(performance)
        return design
    
    def _sample_best(
        self,
        performance: Tensor,
        n_samples: int
    ) -> Tuple[Tensor, Tensor]:
        """采样并选择最优设计"""
        if self.forward_model is None:
            raise ValueError("Forward model required for 'best' mode")
        
        self.mdn.eval()
        self.forward_model.eval()
        
        with torch.no_grad():
            best_design, best_perf = self.mdn.sample_best(
                performance, self.forward_model, n_samples
            )
        
        return best_design, best_perf
    
    def get_distribution(self, performance: Optional[Tensor] = None):
        """
        获取高斯混合分布对象
        
        Args:
            performance: 性能目标（如果未提供则使用输入节点）
            
        Returns:
            GaussianMixtureDistribution 对象
        """
        if performance is None:
            if not self._inputs:
                raise ValueError("No performance provided or connected")
            performance = self._inputs[0].forward()
        
        performance = performance.to(self.device)
        return self.mdn.get_distribution(performance)
    
    def load_pretrained(self, path: Union[str, Path]):
        """加载预训练模型"""
        self.mdn.load(path)
    
    def save_model(self, path: Union[str, Path]):
        """保存模型"""
        self.mdn.save(path)


def create_mdn_node_for_challenge(
    challenge_name: str,
    n_components: int = 5,
    performance_dim: int = 3,
    pretrained_path: Optional[str] = None,
    forward_model: Optional[nn.Module] = None,
    device: str = 'auto'
) -> MDNNode:
    """
    为特定挑战创建 MDN 节点
    
    Args:
        challenge_name: 挑战名称
        n_components: 高斯分量数
        performance_dim: 性能指标维度
        pretrained_path: 预训练模型路径
        forward_model: 前向模型（可选）
        device: 计算设备
        
    Returns:
        配置好的 MDNNode
    """
    from models.inverse.mdn import create_mdn_for_challenge
    
    mdn = create_mdn_for_challenge(
        challenge_name,
        n_components=n_components,
        performance_dim=performance_dim,
        device=device
    )
    
    if pretrained_path:
        mdn.load(pretrained_path)
    
    return MDNNode(
        name=f'mdn_{challenge_name}',
        mdn_model=mdn,
        forward_model=forward_model,
        device=device
    )


class CGANNode(Node):
    """
    条件生成对抗网络节点
    
    封装 CGAN 模型，支持：
    1. 条件生成：从性能目标生成多样化设计
    2. 多样本生成：为同一目标生成多个候选设计
    3. 判别评估：评估设计的真实性
    
    使用示例:
    ```python
    from models.inverse.cgan import CGAN, CGANConfig
    from core.nodes.neural_network import CGANNode
    
    # 创建 CGAN
    config = CGANConfig(
        generator_config=GeneratorConfig(design_shape=(200, 22)),
        discriminator_config=DiscriminatorConfig(design_shape=(200, 22))
    )
    cgan = CGAN(config)
    
    # 创建目标性能节点
    target_perf = TargetPerformanceNode('target', torch.tensor([[0.85, 0.1, 0.05]]))
    
    # 创建 CGAN 节点
    cgan_node = CGANNode('cgan', cgan, target_perf)
    
    # 方式1: 生成单个设计
    design = cgan_node.forward()
    
    # 方式2: 生成多个候选设计
    designs = cgan_node.forward(num_samples=10)
    
    # 方式3: 评估设计真实性
    validity = cgan_node.discriminate(design, target_perf.forward())
    ```
    """
    
    def __init__(
        self,
        name: str,
        cgan_model,  # CGAN 类型
        performance_node: Optional[Node] = None,
        forward_model: Optional[nn.Module] = None,
        device: str = 'auto'
    ):
        """
        Args:
            name: 节点名称
            cgan_model: CGAN 模型
            performance_node: 性能目标节点
            forward_model: 前向模型（用于评估生成质量）
            device: 计算设备
        """
        super().__init__(name)
        self.cgan = cgan_model
        self.forward_model = forward_model
        
        # 设置设备
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.cgan.to(self.device)
        if self.forward_model is not None:
            self.forward_model.to(self.device)
        
        # 连接输入节点
        if performance_node is not None:
            self.add_input(performance_node)
    
    def forward(
        self,
        num_samples: int = 1,
        noise: Optional[Tensor] = None,
        **kwargs
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        生成设计
        
        Args:
            num_samples: 每个条件生成的设计数量
            noise: 自定义噪声（可选）
            
        Returns:
            单样本模式: design [B, H, W]
            多样本模式: 如果有 forward_model 则返回 (designs, performances)
        """
        if not self._inputs:
            raise ValueError("No performance node connected")
        
        condition = self._inputs[0].forward(**kwargs)
        condition = condition.to(self.device)
        
        # 生成设计
        designs = self.cgan.generate(condition, num_samples, noise)
        
        # 如果有前向模型，评估生成设计的性能
        if self.forward_model is not None and num_samples > 1:
            performances = self._evaluate_designs(designs)
            return designs, performances
        
        return designs
    
    def _evaluate_designs(self, designs: Tensor) -> Tensor:
        """评估生成设计的性能"""
        self.forward_model.eval()
        with torch.no_grad():
            designs = designs.to(self.device)
            performances = self.forward_model(designs)
        return performances
    
    def discriminate(
        self,
        design: Tensor,
        condition: Optional[Tensor] = None
    ) -> Tensor:
        """
        评估设计的真实性
        
        Args:
            design: 设计参数
            condition: 条件向量（如果为 None 则使用输入节点）
            
        Returns:
            真实性分数
        """
        if condition is None:
            if not self._inputs:
                raise ValueError("No condition provided or connected")
            condition = self._inputs[0].forward()
        
        design = design.to(self.device)
        condition = condition.to(self.device)
        
        return self.cgan.discriminate(design, condition)
    
    def sample_best(
        self,
        num_samples: int = 10,
        **kwargs
    ) -> Tuple[Tensor, Tensor]:
        """
        生成多个设计并选择最优
        
        Args:
            num_samples: 采样数量
            
        Returns:
            (best_design, best_performance)
        """
        if self.forward_model is None:
            raise ValueError("Forward model required for sample_best")
        
        designs, performances = self.forward(num_samples=num_samples, **kwargs)
        
        # 选择最接近目标的设计
        if self._inputs:
            target = self._inputs[0].forward(**kwargs).to(self.device)
            # 扩展目标以匹配样本数
            target = target.repeat_interleave(num_samples, dim=0)
            # 计算与目标的距离
            distances = torch.norm(performances - target, dim=-1)
            best_idx = distances.argmin()
            
            batch_size = target.size(0) // num_samples
            design_shape = designs.shape[1:]
            
            designs = designs.view(batch_size, num_samples, *design_shape)
            performances = performances.view(batch_size, num_samples, -1)
            
            best_design = designs[0, best_idx % num_samples]
            best_perf = performances[0, best_idx % num_samples]
            
            return best_design, best_perf
        
        return designs[0], performances[0]
    
    def get_latent_interpolation(
        self,
        condition: Tensor,
        start_noise: Optional[Tensor] = None,
        end_noise: Optional[Tensor] = None,
        num_steps: int = 10
    ) -> Tensor:
        """
        在潜在空间中插值生成设计序列
        
        Args:
            condition: 条件向量
            start_noise: 起始噪声
            end_noise: 终止噪声
            num_steps: 插值步数
            
        Returns:
            插值设计序列 [num_steps, H, W]
        """
        condition = condition.to(self.device)
        
        if start_noise is None:
            start_noise = torch.randn(1, self.cgan.latent_dim, device=self.device)
        if end_noise is None:
            end_noise = torch.randn(1, self.cgan.latent_dim, device=self.device)
        
        # 线性插值
        alphas = torch.linspace(0, 1, num_steps, device=self.device)
        designs = []
        
        self.cgan.generator.eval()
        with torch.no_grad():
            for alpha in alphas:
                noise = (1 - alpha) * start_noise + alpha * end_noise
                design = self.cgan.generator(condition, noise)
                designs.append(design)
        
        return torch.cat(designs, dim=0)
    
    def load_pretrained(self, path: Union[str, Path]):
        """加载预训练模型"""
        self.cgan.load(path)
    
    def save_model(self, path: Union[str, Path]):
        """保存模型"""
        self.cgan.save(path)


def create_cgan_node_for_challenge(
    challenge_name: str,
    condition_dim: int = 3,
    latent_dim: int = 128,
    gan_type: str = 'wgan-gp',
    pretrained_path: Optional[str] = None,
    forward_model: Optional[nn.Module] = None,
    device: str = 'auto'
) -> CGANNode:
    """
    为特定挑战创建 CGAN 节点
    
    Args:
        challenge_name: 挑战名称
        condition_dim: 条件维度
        latent_dim: 潜在空间维度
        gan_type: GAN 类型
        pretrained_path: 预训练模型路径
        forward_model: 前向模型（可选）
        device: 计算设备
        
    Returns:
        配置好的 CGANNode
    """
    from models.inverse.cgan import create_cgan_for_challenge
    
    cgan = create_cgan_for_challenge(
        challenge_name,
        condition_dim=condition_dim,
        latent_dim=latent_dim,
        gan_type=gan_type,
        device=device
    )
    
    if pretrained_path:
        cgan.load(pretrained_path)
    
    return CGANNode(
        name=f'cgan_{challenge_name}',
        cgan_model=cgan,
        forward_model=forward_model,
        device=device
    )


class PINNNode(Node):
    """
    物理信息神经网络节点
    
    封装 PINN 模型，支持：
    1. 正向问题：从设计参数预测物理场
    2. 逆向问题：从目标场分布反推设计参数
    3. 物理约束：自动满足 PDE 约束
    
    使用示例:
    ```python
    from models.inverse.pinn import MaxwellPINN, MaxwellConfig
    from core.nodes.neural_network import PINNNode
    
    # 创建 PINN
    config = MaxwellConfig(spatial_dim=2, wavelength=1.55e-6)
    pinn = MaxwellPINN(config)
    
    # 创建坐标节点
    coords = CoordinateNode('coords', resolution=(100, 100))
    
    # 创建 PINN 节点
    pinn_node = PINNNode('pinn', pinn, coords)
    
    # 预测场分布
    fields = pinn_node.forward()
    
    # 计算 Maxwell 残差
    residual = pinn_node.compute_residual()
    
    # 逆向设计
    design = pinn_node.inverse_design(target_field)
    ```
    """
    
    def __init__(
        self,
        name: str,
        pinn_model,  # PhysicsInformedNet 或 MaxwellPINN
        coordinate_node: Optional[Node] = None,
        design_node: Optional[Node] = None,
        device: str = 'auto'
    ):
        """
        Args:
            name: 节点名称
            pinn_model: PINN 模型
            coordinate_node: 坐标节点（提供空间坐标）
            design_node: 设计参数节点（逆向问题时使用）
            device: 计算设备
        """
        super().__init__(name)
        self.pinn = pinn_model
        
        # 设置设备
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.pinn.to(self.device)
        
        # 连接输入节点
        if coordinate_node is not None:
            self.add_input(coordinate_node)
        if design_node is not None:
            self.add_input(design_node)
        
        # 缓存
        self._cached_fields = None
    
    def forward(
        self,
        coordinates: Optional[Tensor] = None,
        design_params: Optional[Tensor] = None,
        **kwargs
    ) -> Union[Tensor, Dict[str, Tensor]]:
        """
        预测物理场
        
        Args:
            coordinates: 空间坐标（如果为 None 则使用输入节点）
            design_params: 设计参数（可选）
            
        Returns:
            物理场（张量或字典）
        """
        # 获取坐标
        if coordinates is None:
            if not self._inputs:
                raise ValueError("No coordinate input provided")
            coordinates = self._inputs[0].forward(**kwargs)
        
        coordinates = coordinates.to(self.device)
        
        # 获取设计参数
        if design_params is None and len(self._inputs) > 1:
            design_params = self._inputs[1].forward(**kwargs)
        
        if design_params is not None:
            design_params = design_params.to(self.device)
        
        # 预测场
        fields = self.pinn(coordinates, design_params)
        
        self._cached_fields = fields
        return fields
    
    def compute_residual(
        self,
        coordinates: Optional[Tensor] = None,
        **kwargs
    ) -> Dict[str, Tensor]:
        """
        计算 PDE 残差
        
        Args:
            coordinates: 空间坐标
            
        Returns:
            残差字典
        """
        if coordinates is None:
            if not self._inputs:
                raise ValueError("No coordinate input provided")
            coordinates = self._inputs[0].forward(**kwargs)
        
        coordinates = coordinates.to(self.device)
        
        # 如果是 MaxwellPINN，使用专用方法
        if hasattr(self.pinn, 'compute_maxwell_residual'):
            return self.pinn.compute_maxwell_residual(coordinates)
        
        # 否则计算通用 PDE 残差
        return self._compute_generic_residual(coordinates)
    
    def _compute_generic_residual(
        self,
        coordinates: Tensor
    ) -> Dict[str, Tensor]:
        """计算通用 PDE 残差"""
        coordinates = coordinates.requires_grad_(True)
        
        # 预测场
        fields = self.pinn(coordinates)
        
        # 计算拉普拉斯（示例：Helmholtz 方程）
        if hasattr(self.pinn, 'compute_laplacian'):
            laplacian = self.pinn.compute_laplacian(coordinates, None, 0)
            k2 = (2 * 3.14159 / 1.55e-6) ** 2
            residual = laplacian + k2 * fields[:, 0]
        else:
            residual = torch.zeros_like(fields[:, 0])
        
        return {
            'residual': residual,
            'fields': fields
        }
    
    def compute_physics_loss(
        self,
        coordinates: Optional[Tensor] = None,
        **kwargs
    ) -> Tensor:
        """
        计算物理约束损失
        
        Args:
            coordinates: 空间坐标
            
        Returns:
            物理损失
        """
        residual_dict = self.compute_residual(coordinates, **kwargs)
        
        total_loss = torch.tensor(0.0, device=self.device)
        
        for key, value in residual_dict.items():
            if 'residual' in key.lower():
                total_loss = total_loss + (value ** 2).mean()
        
        return total_loss
    
    def inverse_design(
        self,
        target_field: Tensor,
        coordinates: Tensor,
        n_iterations: int = 1000,
        lr: float = 0.01
    ) -> Tensor:
        """
        逆向设计：从目标场分布反推设计参数
        
        Args:
            target_field: 目标场分布
            coordinates: 空间坐标
            n_iterations: 优化迭代次数
            lr: 学习率
            
        Returns:
            优化后的设计参数
        """
        # 初始化设计参数
        design_params = torch.rand(
            1, self.pinn.design_dim,
            device=self.device, requires_grad=True
        )
        
        optimizer = torch.optim.Adam([design_params], lr=lr)
        
        target_field = target_field.to(self.device)
        coordinates = coordinates.to(self.device)
        
        for i in range(n_iterations):
            optimizer.zero_grad()
            
            # 预测场
            pred_field = self.pinn(coordinates, design_params)
            
            # 计算损失
            loss = F.mse_loss(pred_field, target_field)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 限制设计参数范围
            with torch.no_grad():
                design_params.clamp_(0, 1)
        
        return design_params.detach()
    
    def train_pinn(
        self,
        collocation_points: Tensor,
        boundary_points: Optional[Tensor] = None,
        labeled_data: Optional[Dict[str, Tensor]] = None,
        n_iterations: int = 10000,
        log_interval: int = 100
    ) -> Dict[str, List[float]]:
        """
        训练 PINN
        
        Args:
            collocation_points: 配点
            boundary_points: 边界点
            labeled_data: 标签数据
            n_iterations: 迭代次数
            log_interval: 日志间隔
            
        Returns:
            训练历史
        """
        from models.inverse.pinn import PINNSolver
        
        solver = PINNSolver(self.pinn)
        
        history = solver.train(
            n_iterations=n_iterations,
            collocation_points=collocation_points.to(self.device),
            boundary_points=boundary_points.to(self.device) if boundary_points else None,
            labeled_data={k: v.to(self.device) for k, v in labeled_data.items()} if labeled_data else None,
            log_interval=log_interval
        )
        
        return history
    
    def get_field_gradient(
        self,
        coordinates: Tensor,
        component: int = 0
    ) -> Tensor:
        """
        计算场的梯度
        
        Args:
            coordinates: 空间坐标
            component: 场分量索引
            
        Returns:
            场梯度
        """
        coordinates = coordinates.to(self.device).requires_grad_(True)
        
        fields = self.pinn(coordinates)
        
        grad = torch.autograd.grad(
            fields[:, component].sum(),
            coordinates,
            create_graph=True
        )[0]
        
        return grad
    
    def load_pretrained(self, path: Union[str, Path]):
        """加载预训练模型"""
        checkpoint = torch.load(path, map_location=self.device)
        if 'model_state' in checkpoint:
            self.pinn.load_state_dict(checkpoint['model_state'])
        else:
            self.pinn.load_state_dict(checkpoint)
    
    def save_model(self, path: Union[str, Path]):
        """保存模型"""
        torch.save(self.pinn.state_dict(), path)


def create_pinn_node_for_photonics(
    spatial_dim: int = 2,
    field_components: int = 3,
    wavelength: float = 1.55e-6,
    epsilon_r: float = 12.0,
    coordinate_node: Optional[Node] = None,
    pretrained_path: Optional[str] = None,
    device: str = 'auto'
) -> PINNNode:
    """
    为光子学问题创建 PINN 节点
    
    Args:
        spatial_dim: 空间维度
        field_components: 场分量数
        wavelength: 工作波长
        epsilon_r: 相对介电常数
        coordinate_node: 坐标节点
        pretrained_path: 预训练模型路径
        device: 计算设备
        
    Returns:
        配置好的 PINNNode
    """
    from models.inverse.pinn import create_pinn_for_photonics
    
    pinn = create_pinn_for_photonics(
        spatial_dim=spatial_dim,
        field_components=field_components,
        wavelength=wavelength,
        epsilon_r=epsilon_r,
        device=device
    )
    
    if pretrained_path:
        pinn.load_state_dict(torch.load(pretrained_path, map_location=device))
    
    return PINNNode(
        name='pinn_photonics',
        pinn_model=pinn,
        coordinate_node=coordinate_node,
        device=device
    )


class CoordinateNode(Node):
    """
    坐标节点
    
    生成空间坐标网格，用于 PINN 输入。
    """
    
    def __init__(
        self,
        name: str,
        bounds: Optional[List[Tuple[float, float]]] = None,
        resolution: Optional[Tuple[int, ...]] = None,
        device: str = 'auto'
    ):
        """
        Args:
            name: 节点名称
            bounds: 各维度的边界 [(min, max), ...]
            resolution: 各维度的分辨率
            device: 计算设备
        """
        super().__init__(name)
        
        self.bounds = bounds or [(-1, 1), (-1, 1)]
        self.resolution = resolution or (100, 100)
        
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # 预生成坐标网格
        self._generate_grid()
    
    def _generate_grid(self):
        """生成坐标网格"""
        grids = []
        for i, (bound, res) in enumerate(zip(self.bounds, self.resolution)):
            grid = torch.linspace(bound[0], bound[1], res, device=self.device)
            grids.append(grid)
        
        # 创建网格
        mesh = torch.meshgrid(*grids, indexing='ij')
        
        # 展平为坐标点
        self._coords = torch.stack([m.flatten() for m in mesh], dim=-1)
        
        # 保存网格形状
        self._grid_shape = self.resolution
    
    def forward(self, **kwargs) -> Tensor:
        """
        获取坐标
        
        Returns:
            坐标张量 [N, spatial_dim]
        """
        return self._coords
    
    def get_grid(self) -> Tensor:
        """获取网格形式的坐标"""
        return self._coords.view(*self._grid_shape, -1)
    
    def sample_random(self, n_points: int) -> Tensor:
        """随机采样坐标点"""
        coords = torch.zeros(n_points, len(self.bounds), device=self.device)
        
        for i, (bound, _) in enumerate(zip(self.bounds, self.resolution)):
            coords[:, i] = torch.rand(n_points, device=self.device) * (bound[1] - bound[0]) + bound[0]
        
        return coords
    
    def sample_boundary(self, n_points_per_side: int = 50) -> Tensor:
        """采样边界点"""
        boundary_points = []
        
        for dim, (bound, _) in enumerate(zip(self.bounds, self.resolution)):
            for boundary_val in bound:
                points = torch.rand(n_points_per_side, len(self.bounds), device=self.device)
                points[:, dim] = boundary_val
                
                # 设置其他维度的范围
                for other_dim, other_bound in enumerate(self.bounds):
                    if other_dim != dim:
                        points[:, other_dim] = (
                            points[:, other_dim] * (other_bound[1] - other_bound[0]) + other_bound[0]
                        )
                
                boundary_points.append(points)
        
        return torch.cat(boundary_points, dim=0)

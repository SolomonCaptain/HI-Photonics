# HI-Photonics API 参考

> 版本: 0.1.0 | 最后更新: 2026-03-21

本文档提供 HI-Photonics 框架的完整 API 参考。

## 目录

- [统一入口模块](#统一入口模块)
- [核心模块 (core)](#核心模块-core)
- [模型模块 (models)](#模型模块-models)
- [工作流模块 (workflows)](#工作流模块-workflows)
- [挑战模块 (challenges)](#挑战模块-challenges)
- [接口模块 (interfaces)](#接口模块-interfaces)
- [优化模块 (optimization)](#优化模块-optimization)
- [数据模块 (data)](#数据模块-data)
- [REST API](#rest-api)

---

## 统一入口模块

### hi_photonics

主入口模块，导出所有公共 API。

```python
import hi_photonics
```

#### 版本信息

```python
hi_photonics.__version__  # "0.1.0"
hi_photonics.get_version() -> str
```

#### 便捷函数

```python
hi_photonics.list_available_simulators() -> List[str]
hi_photonics.list_available_challenges() -> List[str]
hi_photonics.quick_start(
    challenge_name: str = "grating_coupler",
    model_type: str = "hilab",
    target: dict = None,
    **kwargs
) -> dict
```

---

## 核心模块 (core)

### Node

节点基类，所有计算节点的基础。

```python
from hi_photonics import Node

class Node:
    def __init__(self, name: str):
        self.name = name
        self._inputs: List[Node] = []
        self._outputs: List[Node] = []
        self._cached_output: Optional[TensorLike] = None
    
    def add_input(self, node: 'Node') -> 'Node':
        """添加输入节点"""
    
    def add_output(self, node: 'Node') -> 'Node':
        """添加输出节点"""
    
    def forward(self, **kwargs) -> TensorLike:
        """前向传播（子类实现）"""
    
    def __call__(self, **kwargs) -> TensorLike:
        """调用节点"""
    
    def reset(self):
        """重置缓存"""
```

### Graph

计算图管理器。

```python
from hi_photonics import Graph

class Graph:
    def __init__(self, output_nodes: List[Node]):
        """
        Args:
            output_nodes: 输出节点列表
        """
    
    def forward(self, **inputs) -> Dict[str, TensorLike]:
        """
        执行前向传播
        
        Args:
            **inputs: 输入数据
            
        Returns:
            输出节点名称到结果的映射
        """
    
    def topological_sort(self) -> List[Node]:
        """返回拓扑排序后的节点列表"""
    
    def reset(self):
        """重置所有节点缓存"""
    
    def visualize(self, save_path: str = None):
        """可视化计算图"""
```

### 类型定义

```python
from hi_photonics import TensorLike, Params

# TensorLike: Union[torch.Tensor, np.ndarray]
# Params: Dict[str, Any]
```

### 标准节点

#### ParameterizationNode

```python
from hi_photonics import ParameterizationNode

node = ParameterizationNode(
    name="param",
    shape=(100, 22),        # 设计形状
    init_method="random",   # 初始化方法
    bounds=(0.0, 1.0),      # 值范围
    requires_grad=True      # 是否可训练
)
```

#### SimulationNode

```python
from hi_photonics import SimulationNode

node = SimulationNode(
    name="sim",
    simulator=simulator,    # SimulatorInterface 实例
    design_node=param_node  # 输入设计节点
)
```

#### ObjectiveNode

```python
from hi_photonics import ObjectiveNode

node = ObjectiveNode(
    name="objective",
    objective_fn=lambda result: result['efficiency'],
    simulation_node=sim_node
)
```

---

## 模型模块 (models)

### 基类

#### BaseModel

```python
from hi_photonics import BaseModel, ModelConfig

class BaseModel(nn.Module):
    config: ModelConfig
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
    
    def forward(self, *args, **kwargs) -> Tensor:
        """前向传播"""
    
    def save(self, path: Union[str, Path]):
        """保存模型"""
    
    def load(self, path: Union[str, Path]):
        """加载模型"""
    
    def count_parameters(self) -> int:
        """返回参数数量"""
    
    def get_device(self) -> torch.device:
        """返回当前设备"""
```

#### ModelConfig

```python
@dataclass
class ModelConfig:
    name: str = "base_model"
    device: str = "auto"  # "auto", "cuda", "cpu"
```

### TandemNetwork (TNN)

```python
from hi_photonics import TandemNetwork, create_tnn_for_challenge

# 使用工厂函数
model = create_tnn_for_challenge(
    challenge_name="grating_coupler",
    hidden_dim=256
)

# 直接创建
from hi_photonics import TandemNetworkConfig

config = TandemNetworkConfig(
    name="tnn",
    design_shape=(100, 22),
    performance_dim=3,
    hidden_dim=256
)
model = TandemNetwork(config)

# 训练
model.fit(train_loader, val_loader, epochs=100)

# 逆向设计
design = model.inverse(performance_target)
```

### MDN (混合密度网络)

```python
from hi_photonics import MDN, MDNTandemNetwork, create_mdn_for_challenge

model = create_mdn_for_challenge(
    challenge_name="grating_coupler",
    num_components=5
)

# 获取多个设计解
designs = model.sample_designs(
    performance_target,
    num_samples=10
)
```

### CGAN (条件生成对抗网络)

```python
from hi_photonics import CGAN, create_cgan_for_challenge

model = create_cgan_for_challenge(
    challenge_name="grating_coupler",
    latent_dim=128
)

# 训练
model.train(
    train_loader,
    epochs=1000,
    n_critic=5  # critic 更新次数
)

# 生成设计
design = model.generate(performance_target)
```

### HiLABEngine

```python
from hi_photonics import HiLABEngine, create_hilab_for_challenge

# 使用工厂函数
engine = create_hilab_for_challenge(
    challenge_name="grating_coupler",
    latent_dim=32,
    performance_dim=3
)

# 训练 VAE
engine.train_vae(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=100,
    lr=1e-3
)

# 训练代理模型
engine.train_surrogate(
    data_loader=data_loader,
    epochs=50
)

# 逆向设计
result = engine.inverse_design(
    target_performance=torch.tensor([[0.9, 0.8, 0.1]]),
    n_iterations=30
)

# 获取设计信息
info = engine.get_model_info()
```

### 训练工具

#### 损失函数

```python
from hi_photonics import get_loss

# 获取损失函数
loss_fn = get_loss("mse")
loss_fn = get_loss("mdn")
loss_fn = get_loss("gan")
loss_fn = get_loss("physics_informed")

# 直接导入
from hi_photonics import (
    BaseLoss,
    PerformanceLoss,
    DesignLoss,
    TandemLoss,
    MDNLoss,
    GANLoss,
    PhysicsInformedLoss
)
```

#### 评估指标

```python
from hi_photonics import get_metric

metric = get_metric("mse")
metric = get_metric("r2")
metric = get_metric("mae")

# 直接导入
from hi_photonics import (
    MSE, MAE, R2Score, RMSE,
    DesignQualityMetric,
    InverseDesignMetric
)
```

#### 回调函数

```python
from hi_photonics import get_default_callbacks

callbacks = get_default_callbacks(
    early_stopping=True,
    checkpoint=True,
    progress_bar=True
)

# 直接导入
from hi_photonics import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateScheduler,
    ProgressBar
)
```

---

## 工作流模块 (workflows)

### DesignPipeline

```python
from hi_photonics import DesignPipeline, PipelineConfig

config = PipelineConfig(
    name="my_design",
    challenge_name="grating_coupler",
    model_type="hilab",
    num_epochs=100,
    num_iterations=50,
    batch_size=32,
    learning_rate=1e-3,
    output_dir="outputs",
    use_simulation=True,
    use_constraints=True,
    verbose=True
)

pipeline = DesignPipeline(config)

# 设置
pipeline.setup()

# 训练模型
history = pipeline.train_model()

# 逆向设计
result = pipeline.inverse_design(
    target_performance=[0.9, 0.8, 0.1]
)

# 运行完整流程
result = pipeline.run()

# 保存/加载
pipeline.save("my_pipeline.pkl")
pipeline.load("my_pipeline.pkl")
```

### create_pipeline

```python
from hi_photonics import create_pipeline

pipeline = create_pipeline(
    challenge_name="grating_coupler",
    model_type="hilab",
    num_epochs=100,
    num_iterations=50,
    output_dir="outputs"
)

result = pipeline.run()
```

### run_quick_design

```python
from hi_photonics import run_quick_design

result = run_quick_design(
    challenge_name="grating_coupler",
    target_performance={"efficiency": 0.8},
    num_iterations=50,
    model_type="hilab"
)
```

### TaskDispatcher

```python
from hi_photonics import TaskDispatcher, submit_task, get_task_result

# 使用便捷函数
task_id = submit_task(
    fn=long_running_task,
    args=(arg1, arg2),
    kwargs={"key": "value"},
    task_name="my_task"
)

result = get_task_result(task_id)

# 使用类
dispatcher = TaskDispatcher(max_workers=4)

task_id = dispatcher.submit(
    fn=long_running_task,
    args=(arg1, arg2)
)

status = dispatcher.get_status(task_id)
result = dispatcher.get_result(task_id)
```

---

## 挑战模块 (challenges)

### DesignChallenge

```python
from hi_photonics import DesignChallenge, DesignSpec, PerformanceTarget

class MyChallenge(DesignChallenge):
    def __init__(self):
        super().__init__(
            name="my_challenge",
            spec=DesignSpec(
                design_shape=(100, 22),
                performance_dim=3,
                parameter_range=(0.0, 1.0)
            )
        )
    
    def setup_simulator(self) -> SimulatorInterface:
        """配置仿真器"""
        return self.simulator
    
    def compute_objective(self, result: SimulationResult) -> Tensor:
        """计算目标函数值"""
        return torch.tensor(result.metrics['efficiency'])
    
    def get_initial_design(self) -> Tensor:
        """获取初始设计"""
        return torch.rand(self.spec.design_shape)
    
    def get_default_target(self) -> Dict:
        """获取默认目标"""
        return {"efficiency": 0.8}
```

### ChallengeFactory

```python
from hi_photonics import ChallengeFactory, register_challenge

# 列出可用挑战
challenges = ChallengeFactory.list_available()
# ['grating_coupler', 'metagrating', 'wavelength_demux']

# 创建挑战
challenge = ChallengeFactory.create("grating_coupler")

# 注册自定义挑战
@register_challenge("my_challenge")
class MyChallenge(DesignChallenge):
    ...
```

### 内置挑战

```python
from hi_photonics import (
    GratingCouplerChallenge,
    MetagratingChallenge,
    WavelengthDemuxChallenge
)

# 光栅耦合器
grating = GratingCouplerChallenge(
    wavelength=1.55,
    fiber_angle=10.0,
    num_periods=20
)

# 超光栅
meta = MetagratingChallenge(
    wavelength=1.55,
    period=0.5,
    num_periods=50
)

# 波长解复用器
demux = WavelengthDemuxChallenge(
    wavelengths=[1.31, 1.55],
    channels=2
)
```

---

## 接口模块 (interfaces)

### SimulatorInterface

```python
from hi_photonics import SimulatorInterface, SimulationConfig

class MySimulator(SimulatorInterface):
    def __init__(self, config: SimulationConfig):
        super().__init__(config)
    
    def run(self, design_params: Tensor, **kwargs) -> Dict[str, Tensor]:
        """运行前向仿真"""
        ...
    
    def compute_gradient(
        self, 
        design_params: Tensor,
        objective_grad: Dict[str, Tensor],
        **kwargs
    ) -> Tensor:
        """计算伴随梯度"""
        ...
```

### SimulationConfig

```python
from hi_photonics import SimulationConfig, BoundaryCondition

config = SimulationConfig(
    resolution=50,              # 像素/微米
    cell_size=(10.0, 10.0, 0.0),  # (x, y, z) 微米
    boundary_x=BoundaryCondition.PML,
    boundary_y=BoundaryCondition.PML,
    pml_thickness=1.0,          # PML 厚度 (微米)
    simulation_time=100.0,      # 时间步长
    wavelengths=[1.55],         # 波长列表 (微米)
    force_complex_fields=False,
    accuracy=2
)
```

### SourceConfig

```python
from hi_photonics import SourceConfig, SourceType

source = SourceConfig(
    source_type=SourceType.GAUSSIAN,
    wavelength=1.55,            # 微米
    center=(0.0, 0.0, 0.0),     # 位置
    size=(0.0, 2.0, 0.0),       # 尺寸
    polarization="Ez",          # 极化
    direction=1,                # 方向
    pulse_width=10.0            # 脉宽
)
```

### MonitorConfig

```python
from hi_photonics import MonitorConfig

monitor = MonitorConfig(
    name="transmission",
    monitor_type="flux",        # 'flux', 'field', 'mode', 'energy'
    center=(5.0, 0.0, 0.0),
    size=(0.0, 2.0, 0.0),
    wavelengths=[1.55]
)
```

### SimulatorFactory

```python
from hi_photonics import SimulatorFactory, register_simulator

# 列出可用仿真器
simulators = SimulatorFactory.list_available()

# 创建仿真器
sim = SimulatorFactory.create("meep", config=config)

# 注册自定义仿真器
@register_simulator("my_simulator")
class MySimulator(SimulatorInterface):
    ...
```

---

## 优化模块 (optimization)

### 约束条件

```python
from hi_photonics import (
    DispersionConstraint,
    ThermalConstraint,
    RobustnessConstraint
)

# 色散约束
dispersion = DispersionConstraint(
    wavelengths=[1.31, 1.55],
    max_variation=0.1
)

# 热效应约束
thermal = ThermalConstraint(
    temperature_range=(280.0, 360.0),
    thermo_optic_coeff=1.86e-4
)

# 鲁棒性约束
robustness = RobustnessConstraint(
    cd_tolerance=0.01,
    edge_roughness_rms=0.003
)
```

### BayesianOptimizer

```python
from hi_photonics import (
    BayesianOptimizer,
    GaussianProcessRegressor,
    ExpectedImprovement,
    UpperConfidenceBound
)

# 创建高斯过程
gp = GaussianProcessRegressor(
    kernel="rbf",
    normalize_y=True
)

# 创建采集函数
acquisition = ExpectedImprovement(xi=0.01)
# 或
acquisition = UpperConfidenceBound(beta=2.0)

# 创建优化器
optimizer = BayesianOptimizer(
    surrogate=gp,
    acquisition=acquisition,
    bounds=[(0, 1)] * 32  # 搜索边界
)

# 优化
optimizer.tell(X_initial, y_initial)
for i in range(50):
    x_next = optimizer.ask()
    y_next = evaluate(x_next)
    optimizer.tell(x_next, y_next)
    
best_x, best_y = optimizer.get_best()
```

---

## 数据模块 (data)

### 数据集

```python
from hi_photonics import (
    PhotonicsDataset,
    HDF5Dataset,
    SyntheticDataset
)

# 合成数据集
dataset = SyntheticDataset(
    num_samples=1000,
    design_shape=(100, 22),
    performance_dim=3,
    noise_level=0.1,
    seed=42
)

# HDF5 数据集
dataset = HDF5Dataset(
    filepath="data.h5",
    design_key="designs",
    performance_key="performances"
)

# 自定义数据集
dataset = PhotonicsDataset(
    designs=designs_array,        # [N, H, W]
    performances=performances_array,  # [N, D]
    normalize=True
)
```

### 数据加载器

```python
from hi_photonics import create_dataloaders

train_loader, val_loader, test_loader = create_dataloaders(
    dataset=dataset,
    batch_size=32,
    train_ratio=0.8,
    val_ratio=0.1,
    num_workers=0,
    seed=42
)
```

### 数据增强

```python
from hi_photonics import DataAugmentation

augmentation = DataAugmentation(
    horizontal_flip=True,
    vertical_flip=False,
    rotation=False,
    noise_injection=0.02
)

# 应用增强
augmented_design = augmentation(design)
```

### 数据保存

```python
from hi_photonics import save_dataset_to_hdf5

save_dataset_to_hdf5(
    dataset=dataset,
    filepath="output.h5",
    metadata={"version": "1.0"}
)
```

---

## REST API

### 基础 URL

```
http://localhost:8000
```

### 端点

#### 获取节点定义

```
GET /workflow/nodes
```

响应:
```json
{
  "parameterization": {...},
  "simulation": {...},
  "model_train": {...}
}
```

#### 执行工作流

```
POST /workflow/execute
```

请求体:
```json
{
  "nodes": [...],
  "edges": [...]
}
```

#### 创建管道

```
POST /workflow/pipeline/create?challenge_name=grating_coupler&model_type=hilab
```

响应:
```json
{
  "pipeline_id": "pipeline_123",
  "status": "created"
}
```

#### 运行管道

```
POST /workflow/pipeline/{pipeline_id}/run
```

请求体:
```json
{
  "target_performance": [0.9, 0.8, 0.1]
}
```

#### 训练模型

```
POST /workflow/train?challenge_name=grating_coupler&model_type=hilab
```

请求体:
```json
{
  "epochs": 100,
  "batchSize": 32
}
```

#### 逆向设计

```
POST /workflow/inverse-design
```

请求体:
```json
{
  "challenge_name": "grating_coupler",
  "target_performance": [0.9, 0.8, 0.1],
  "model_type": "hilab"
}
```

#### 仿真设计

```
POST /workflow/simulate
```

请求体:
```json
{
  "design": [[...]],
  "simulator_type": "optics",
  "config": {
    "resolution": 50
  }
}
```

#### 获取任务状态

```
GET /workflow/task/{task_id}/status
```

响应:
```json
{
  "task_id": "task_123",
  "status": "completed",
  "result": {...},
  "progress": 1.0,
  "duration": 12.5
}
```

---

## 错误处理

### 异常类型

```python
class HIPhotonicsError(Exception):
    """基础异常"""

class ModelError(HIPhotonicsError):
    """模型相关错误"""

class SimulationError(HIPhotonicsError):
    """仿真相关错误"""

class ConfigError(HIPhotonicsError):
    """配置相关错误"""

class DataError(HIPhotonicsError):
    """数据相关错误"""
```

### 错误响应格式

```json
{
  "detail": "Error message",
  "error_code": "MODEL_NOT_FOUND",
  "timestamp": "2026-03-21T12:00:00Z"
}
```

---

## 类型提示

框架使用 Python 类型提示：

```python
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from torch import Tensor
import numpy as np

# 常用类型
TensorLike = Union[Tensor, np.ndarray]
Params = Dict[str, Any]
Shape = Tuple[int, ...]
```
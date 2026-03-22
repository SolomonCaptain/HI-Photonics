# HI-Photonics API 参考

> 版本: 0.2.0 | 最后更新: 2026-03-22

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
- [LLM 模块 (llm)](#llm-模块-llm)
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
    
    def save(self, path: Union[str, Path], 
             format: Literal["torch", "safetensors"] = "torch",
             metadata: Optional[Dict[str, Any]] = None):
        """
        保存模型
        
        Args:
            path: 保存路径
            format: 保存格式
                - "torch": PyTorch 格式 (.pt/.pth)
                - "safetensors": Safetensors 格式 (.safetensors)
            metadata: 可选的元数据（仅 safetensors 格式）
        
        Examples:
            # 保存为 PyTorch 格式
            model.save("model.pt")
            
            # 保存为 Safetensors 格式（推荐用于 Windows）
            model.save("model.safetensors", format="safetensors")
        """
    
    def load(self, path: Union[str, Path], 
             format: Literal["torch", "safetensors", "auto"] = "auto"):
        """
        加载模型
        
        Args:
            path: 模型路径
            format: 加载格式
                - "torch": PyTorch 格式
                - "safetensors": Safetensors 格式
                - "auto": 根据文件扩展名自动检测
        """
    
    def count_parameters(self) -> int:
        """返回参数数量"""
    
    def get_device(self) -> torch.device:
        """返回当前设备"""
```

#### Safetensors 工具

```python
from hi_photonics import (
    convert_torch_to_safetensors,
    convert_safetensors_to_torch,
    get_safetensors_info,
    validate_safetensors_file
)

# 转换格式
convert_torch_to_safetensors("model.pt", "model.safetensors")
convert_safetensors_to_torch("model.safetensors", "model.pt")

# 获取模型信息
info = get_safetensors_info("model.safetensors")
# {
#   "format": "pt",
#   "metadata": {...},
#   "tensors": {"weight_name": {"dtype": "F32", "shape": [128, 256]}}
# }

# 验证文件
is_valid = validate_safetensors_file("model.safetensors")
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

## LLM 模块 (llm)

LLM 模块提供基于大语言模型的智能逆向设计辅助功能。

### 配置管理

```python
from llm import (
    LLMConfig,
    EmbeddingConfig,
    QdrantConfig,
    RAGConfig,
    LLMAssistantConfig,
    get_config,
    reload_config
)

# 加载配置（从环境变量）
config = get_config()

# 自定义配置
llm_config = LLMConfig(
    provider="openai",  # "openai" | "azure"
    model="gpt-4",
    api_key="sk-...",
    temperature=0.7
)

embedding_config = EmbeddingConfig(
    provider="openai",
    model="text-embedding-3-small"
)

qdrant_config = QdrantConfig(
    host="localhost",
    port=6333,
    collection_name="photonics_knowledge"
)

assistant_config = LLMAssistantConfig(
    llm=llm_config,
    embedding=embedding_config,
    qdrant=qdrant_config
)
```

### LLMAssistant

统一入口类，提供完整的 LLM 增强逆向设计功能。

```python
from llm import LLMAssistant, LLMAssistantConfig

# 创建助手
assistant = LLMAssistant(config)

# 异步上下文管理器
async with LLMAssistant() as assistant:
    # 解析意图
    intent = await assistant.parse_intent("设计1550nm光栅耦合器")

    # 生成工作流
    workflow = await assistant.generate_workflow(intent)

    # 解释结果
    report = await assistant.explain_results(design, simulation)

    # 对话
    response = await assistant.chat("如何提高耦合效率？")
```

### LLMClient

LLM API 客户端，支持同步和流式调用。

```python
from llm import LLMClient, LLMConfig

client = LLMClient(LLMConfig(
    provider="openai",
    model="gpt-4"
))

# 同步对话
response = await client.chat([
    {"role": "user", "content": "什么是光栅耦合器？"}
])

# 流式对话
async for chunk in client.chat_stream(messages):
    print(chunk, end="", flush=True)

# 带系统提示的对话
response = await client.chat_with_system(
    user_message="设计1550nm耦合器",
    system_prompt="你是光子学设计专家..."
)

# 关闭客户端
await client.close()
```

### RAGService

检索增强生成服务，支持知识库检索。

```python
from llm import RAGService, RAGConfig

rag = RAGService(RAGConfig())

# 初始化（创建集合）
rag.initialize()

# 索引文档
documents = [
    {"content": "光栅耦合器是...", "metadata": {"source": "docs"}},
    {"content": "FDTD仿真方法是...", "metadata": {"source": "tutorial"}}
]
await rag.index_documents(documents)

# 索引目录
await rag.index_directory(Path("llm/knowledge"))

# 检索
context = await rag.retrieve("如何设计光栅耦合器", top_k=5)
print(context.formatted_context)

# 删除集合
await rag.delete_collection()

# 关闭连接
await rag.close()
```

### QdrantService

Qdrant 向量数据库服务。

```python
from llm import QdrantService, QdrantConfig

qdrant = QdrantService(QdrantConfig(
    host="localhost",
    port=6333,
    collection_name="my_collection"
))

# 创建集合
qdrant.create_collection(vector_size=1536)

# 插入向量
qdrant.upsert_vectors(
    ids=["doc1", "doc2"],
    vectors=[embedding1, embedding2],
    payloads=[{"text": "..."}, {"text": "..."}]
)

# 搜索
results = qdrant.search(query_vector, limit=5)

# 删除集合
qdrant.delete_collection()
```

### EmbeddingClient

文本嵌入客户端。

```python
from llm import EmbeddingClient, EmbeddingConfig

embedding = EmbeddingClient(EmbeddingConfig(
    provider="openai",
    model="text-embedding-3-small"
))

# 获取嵌入向量
vectors = await embedding.embed(["文本1", "文本2"])

# 获取嵌入维度
dim = embedding.get_embedding_dimension()  # 1536
```

### PromptOrchestrator

提示词编排器，管理提示词模板和 LLM 调用。

```python
from llm import PromptOrchestrator, LLMAssistantConfig

orchestrator = PromptOrchestrator(LLMAssistantConfig())

# 解析设计意图
intent = await orchestrator.parse_intent(
    user_input="设计一个1550nm的光栅耦合器，效率目标80%",
    use_rag=True
)
# DesignIntent(
#   device_type="grating_coupler",
#   target_specs={"wavelength": 1550, "efficiency": 0.8},
#   confidence=0.9
# )

# 生成工作流配置
workflow = await orchestrator.generate_workflow_config(intent)
# WorkflowSuggestion(
#   pipeline_name="grating_coupler_hilab",
#   challenge="grating_coupler",
#   model_config={"type": "hilab", "latent_dim": 32},
#   rationale="HiLab适合追求最优设计..."
# )

# 解释结果
report = await orchestrator.explain_results(
    design_result={"efficiency": 0.85},
    simulation_result={"transmission": 0.82},
    detail_level="normal"
)

# 对话
response = await orchestrator.chat(
    message="如何提高设计效率？",
    history=[{"role": "user", "content": "..."}],
    use_rag=True
)
```

### 数据模型

```python
from llm import DesignIntent, WorkflowSuggestion, DesignReport

# 设计意图
intent = DesignIntent(
    device_type="grating_coupler",
    target_specs={"wavelength": 1550, "efficiency": 0.8},
    constraints={"fabrication": "standard"},
    preferences={"model_preference": "hilab"},
    confidence=0.9,
    clarification_needed=[]
)

# 工作流建议
workflow = WorkflowSuggestion(
    pipeline_name="grating_coupler_hilab",
    challenge="grating_coupler",
    model_config={"type": "hilab"},
    training_config={"epochs": 100},
    optimization_config={"num_iterations": 50},
    rationale="选择理由",
    alternatives=["tnn", "mdn"]
)

# 设计报告
report = DesignReport(
    summary="设计达到了80%的耦合效率目标",
    metrics_analysis={"efficiency": {"target": 0.8, "achieved": 0.82}},
    design_features=["周期结构", "渐变槽深"],
    potential_issues=["制造公差敏感"],
    suggestions=["考虑添加金属反射层"]
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
GET /api/workflow/nodes
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
POST /api/workflow/execute
```

请求体:
```json
{
  "nodes": [...],
  "edges": [...]
}
```

响应:
```json
[
  {
    "node_id": "node_1",
    "status": "success",
    "output": {...},
    "duration": 0.5
  }
]
```

#### 执行单个节点

```
POST /api/workflow/execute-node
```

请求体:
```json
{
  "node": {...},
  "inputs": {...}
}
```

#### 创建管道

```
POST /api/workflow/pipeline/create?challenge_name=grating_coupler&model_type=hilab
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
POST /api/workflow/pipeline/{pipeline_id}/run
```

请求体:
```json
{
  "target_performance": [0.9, 0.8, 0.1]
}
```

---

### 模型训练 API

#### 完整训练（支持 Safetensors）

```
POST /api/models/train-full
```

请求体:
```json
{
  "model_type": "tnn",
  "challenge_name": "grating_coupler",
  "num_samples": 3000,
  "design_shape": [100, 22],
  "performance_dim": 3,
  "epochs": 50,
  "batch_size": 32,
  "learning_rate": 0.001,
  "save_format": "safetensors",
  "model_name": "my_model"
}
```

响应:
```json
{
  "training_id": "uuid",
  "model_type": "tnn",
  "status": "started"
}
```

#### 获取训练状态

```
GET /api/models/train/{training_id}
```

响应:
```json
{
  "training_id": "uuid",
  "model_type": "tnn",
  "status": "completed",
  "model_path": "op_models/pretrained/tnn_20260321.safetensors",
  "metrics": {
    "final_val_loss": 0.015,
    "epochs": 100,
    "parameters": 125000
  }
}
```

#### 使用已加载模型执行逆向设计

```
POST /api/models/inverse-design-loaded
```

请求体:
```json
{
  "model_path": "tnn_20260321",
  "target_performance": [0.9, 0.8, 0.1],
  "num_samples": 1,
  "model_type": "tnn"
}
```

响应:
```json
{
  "design": [[...]],
  "predicted_performance": [0.89, 0.79, 0.11],
  "target_performance": [0.9, 0.8, 0.1],
  "model_path": "op_models/pretrained/tnn_20260321_forward.safetensors"
}
```

#### 列出预训练模型

```
GET /api/models/pretrained/list
```

响应:
```json
[
  {
    "id": "tnn_20260321",
    "path": "op_models/pretrained/tnn_20260321",
    "format": "safetensors",
    "model_type": "tnn",
    "name": "tnn_windows_trained",
    "saved_at": "2026-03-21T12:00:00",
    "platform": "Windows",
    "design_shape": [100, 22],
    "performance_dim": 3
  }
]
```

---

### 资源管理 API

#### 列出资产

```
GET /api/resources/assets?category=inputs&assetType=spectrum
```

响应:
```json
[
  {
    "id": "asset_1",
    "name": "光栅耦合器光谱",
    "type": "spectrum",
    "category": "inputs",
    "size": 128000,
    "created_at": "2026-03-21T10:00:00",
    "updated_at": "2026-03-21T10:00:00"
  }
]
```

#### 上传资产

```
POST /api/resources/assets
Content-Type: multipart/form-data
```

#### 列出模型

```
GET /api/resources/models?modelType=tnn&pretrainedOnly=true
```

#### 保存工作流

```
POST /api/resources/workflows
```

请求体:
```json
{
  "name": "我的逆向设计工作流",
  "nodes": [...],
  "edges": [...],
  "description": "使用 TNN 进行逆向设计",
  "tags": ["tnn", "inverse-design"]
}
```

#### 列出模板

```
GET /api/resources/templates?category=基础
```

响应:
```json
[
  {
    "id": "tpl_basic_inverse",
    "name": "基础逆向设计",
    "description": "使用神经网络进行基础逆向设计",
    "category": "基础",
    "difficulty": "beginner"
  }
]
```

#### 获取模板详情

```
GET /api/resources/templates/{template_id}
```

---

### 系统信息 API

#### 健康检查

```
GET /health
```

响应:
```json
{
  "status": "healthy"
}
```

---

### LLM API

#### 对话

```
POST /api/llm/chat
```

请求体:
```json
{
  "message": "设计一个1550nm的光栅耦合器",
  "history": [
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好！我是光子学设计助手..."}
  ],
  "use_rag": true
}
```

响应:
```json
{
  "response": "好的，我可以帮你设计1550nm的光栅耦合器...",
  "success": true,
  "error": null
}
```

#### 流式对话

```
POST /api/llm/chat/stream
```

返回 SSE 流式响应：
```
data: {"content": "好的"}
data: {"content": "，我"}
data: {"content": "可以帮你..."}
data: [DONE]
```

#### 解析设计意图

```
POST /api/llm/parse-intent
```

请求体:
```json
{
  "user_input": "设计一个1550nm的光栅耦合器，效率目标80%",
  "use_rag": true
}
```

响应:
```json
{
  "device_type": "grating_coupler",
  "target_specs": {
    "wavelength": 1550,
    "efficiency": 0.8
  },
  "constraints": {},
  "preferences": {},
  "confidence": 0.9,
  "clarification_needed": [],
  "success": true
}
```

#### 生成工作流配置

```
POST /api/llm/generate-workflow
```

请求体:
```json
{
  "intent": {
    "device_type": "grating_coupler",
    "target_specs": {"wavelength": 1550, "efficiency": 0.8}
  },
  "use_rag": true
}
```

响应:
```json
{
  "pipeline_name": "grating_coupler_hilab",
  "challenge": "grating_coupler",
  "model_config": {
    "type": "hilab",
    "latent_dim": 32
  },
  "training_config": {
    "epochs": 100,
    "batch_size": 32
  },
  "optimization_config": {
    "num_iterations": 50
  },
  "rationale": "HiLab 适合追求最优设计...",
  "alternatives": ["tnn", "mdn"],
  "success": true
}
```

#### 解释设计结果

```
POST /api/llm/explain-results
```

请求体:
```json
{
  "design_result": {
    "efficiency": 0.85,
    "design_path": "outputs/design.pt"
  },
  "simulation_result": {
    "transmission": 0.82,
    "reflection": 0.05
  },
  "detail_level": "normal"
}
```

响应:
```json
{
  "summary": "设计达到了80%的耦合效率目标，实际效率为85%...",
  "metrics_analysis": {
    "efficiency": {"target": 0.8, "achieved": 0.85}
  },
  "design_features": ["周期性结构", "渐变槽深"],
  "potential_issues": ["制造公差敏感"],
  "suggestions": ["考虑添加金属反射层提高效率"],
  "success": true
}
```

#### LLM 健康检查

```
GET /api/llm/health
```

响应:
```json
{
  "status": "healthy",
  "assistant_initialized": true
}
```

#### 系统信息

```
GET /api/system/info
```

响应:
```json
{
  "python_version": "3.13.9",
  "torch_version": "2.0.0",
  "cuda_available": true,
  "cuda_version": "12.0",
  "device_count": 1
}
```

#### 目录信息

```
GET /api/resources/directories/{category}
```

响应:
```json
{
  "category": "inputs",
  "path": "F:/repo/HI-Photonics/inputs",
  "subdirectories": ["datasets", "spectra", "gds", "structures"],
  "total_size": 1024000,
  "file_count": 10
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

class LLMError(HIPhotonicsError):
    """LLM 相关错误"""

class RAGError(HIPhotonicsError):
    """RAG 服务相关错误"""
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
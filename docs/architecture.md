# HI-Photonics 架构文档

> 版本: 0.1.0 | 最后更新: 2026-03-21

## 概述

HI-Photonics 是一个模块化的光子学逆向设计框架，采用分层架构设计，支持多种深度学习模型和仿真器的集成。

## 系统架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              用户界面层                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     React Frontend (TypeScript)                      │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐   │   │
│  │  │工作流编辑│ │节点配置  │ │执行监控  │ │结果可视化│ │模型管理  │   │   │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │ HTTP/WebSocket
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              API 服务层                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        FastAPI Backend                                │   │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐                  │   │
│  │  │ workflow.py  │ │  models.py   │ │  schemas.py  │                  │   │
│  │  │ 工作流路由   │ │ 模型路由     │ │ 数据模型     │                  │   │
│  │  └──────────────┘ └──────────────┘ └──────────────┘                  │   │
│  │  ┌──────────────────────────────────────────────────────────────┐    │   │
│  │  │                    WorkflowService                             │    │   │
│  │  │  • 节点执行  • 管道管理  • 任务调度  • 结果缓存             │    │   │
│  │  └──────────────────────────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              应用核心层                                       │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                          hi_photonics                                  │ │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │ │
│  │  │                      Workflows Module                            │  │ │
│  │  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │  │ │
│  │  │  │ DesignPipeline  │  │ TaskDispatcher  │  │  Templates      │  │  │ │
│  │  │  │ 设计管道        │  │ 任务调度        │  │ 工作流模板      │  │  │ │
│  │  │  └─────────────────┘  └─────────────────┘  └─────────────────┘  │  │ │
│  │  └─────────────────────────────────────────────────────────────────┘  │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
          │                    │                    │                    │
          ▼                    ▼                    ▼                    ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│   Core 模块     │ │  Models 模块    │ │ Challenges 模块 │ │ Optimization    │
│                 │ │                 │ │                 │ │     模块        │
│ ┌─────────────┐ │ │ ┌─────────────┐ │ │ ┌─────────────┐ │ │ ┌─────────────┐ │
│ │   Graph     │ │ │ │  BaseModel  │ │ │ │ DesignSpec  │ │ │ │ Constraints │ │
│ │   计算图    │ │ │ │  模型基类   │ │ │ │ 设计规格    │ │ │ │   约束条件  │ │
│ └─────────────┘ │ │ └─────────────┘ │ │ └─────────────┘ │ │ └─────────────┘ │
│ ┌─────────────┐ │ │ ┌─────────────┐ │ │ ┌─────────────┐ │ │ ┌─────────────┐ │
│ │    Node     │ │ │ │   Inverse   │ │ │ │ Performance │ │ │ │  Solvers    │ │
│ │   节点基类  │ │ │ │   Models    │ │ │ │   Target    │ │ │ │   求解器    │ │
│ └─────────────┘ │ │ └─────────────┘ │ │ └─────────────┘ │ │ └─────────────┘ │
│ ┌─────────────┐ │ │ ┌─────────────┐ │ │ ┌─────────────┐ │ └─────────────────┘
│ │   Nodes/    │ │ │ │  Training   │ │ │ │  Challenge  │ │
│ │  节点实现   │ │ │ │   Tools     │ │ │ │   Factory   │ │
│ └─────────────┘ │ │ └─────────────┘ │ │ └─────────────┘ │
└─────────────────┘ └─────────────────┘ └─────────────────┘
          │                    │                    │
          └────────────────────┼────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              接口适配层                                       │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                           Interfaces Module                            │ │
│  │  ┌─────────────────────────────────────────────────────────────────┐  │ │
│  │  │                      Simulators                                  │  │ │
│  │  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐ │  │ │
│  │  │  │  Meep    │  │  RCWA    │  │ Lumerical│  │ Optics (C++ FDTD)│ │  │ │
│  │  │  │  FDTD    │  │  严格耦合│  │  商业软件│  │  自研仿真器      │ │  │ │
│  │  │  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘ │  │ │
│  │  └─────────────────────────────────────────────────────────────────┘  │ │
│  │  ┌──────────────────────────┐  ┌──────────────────────────────────┐   │ │
│  │  │      Foundry 接口        │  │       Visualization 可视化       │   │ │
│  │  │  • Design Rules 设计规则 │  │  • Field Plot 场分布            │   │ │
│  │  │  • GDS Export 版图导出   │  │  • Structure 结构可视化         │   │ │
│  │  └──────────────────────────┘  └──────────────────────────────────┘   │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              数据存储层                                       │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                            Data Module                                 │ │
│  │  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐    │ │
│  │  │    Generators    │  │     Loaders      │  │    Preprocess    │    │ │
│  │  │  数据生成器      │  │   数据加载器     │  │   数据预处理     │    │ │
│  │  │  • Random        │  │  • HDF5          │  │  • Normalize     │    │ │
│  │  │  • Active Learn  │  │  • Pipeline      │  │  • Augment       │    │ │
│  │  │  • Multi-Fidelity│  │                  │  │                  │    │ │
│  │  └──────────────────┘  └──────────────────┘  └──────────────────┘    │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 核心模块详解

### 1. Core 模块 - 计算图框架

计算图框架是整个系统的核心，负责管理设计流程的执行。

#### 1.1 Graph 类

```python
class Graph:
    """计算图管理器"""
    
    def __init__(self, output_nodes: List[Node]):
        self.output_nodes = output_nodes
        self._topo_order = None
    
    def forward(self, **inputs) -> Dict[str, TensorLike]:
        """执行前向传播"""
        
    def topological_sort(self) -> List[Node]:
        """拓扑排序确定执行顺序"""
```

**职责:**
- 管理节点依赖关系
- 拓扑排序确定执行顺序
- 执行前向/反向传播
- 缓存中间结果

#### 1.2 Node 类层次

```
Node (抽象基类)
├── ParameterizationNode    # 参数化
├── SimulationNode          # 仿真
├── ObjectiveNode           # 目标函数
├── FilterNode              # 滤波器
│   ├── GaussianFilterNode
│   ├── SobelFilterNode
│   └── MorphologicalFilterNode
├── ProjectionNode          # 投影
│   ├── SigmoidProjectionNode
│   ├── HeavisideProjectionNode
│   └── DensityProjectionNode
├── ConstraintNode          # 约束
│   ├── VolumeConstraintNode
│   ├── ManufacturingConstraintNode
│   └── PhysicsConstraintNode
└── CompositeNode           # 组合节点
    ├── PipelineNode
    └── DesignPipelineNode
```

### 2. Models 模块 - 深度学习模型

#### 2.1 模型基类

```python
class BaseModel(nn.Module):
    """所有模型的基类"""
    config: ModelConfig
    
    def forward(self, *args, **kwargs) -> Tensor:
        """前向传播"""
    
    def save(self, path: Path):
        """保存模型"""
    
    def load(self, path: Path):
        """加载模型"""

class SurrogateModel(BaseModel):
    """代理模型: 设计 → 性能"""
    def forward(self, design: Tensor) -> Tensor:
        return performance

class InverseModel(BaseModel):
    """逆向模型: 性能 → 设计"""
    def forward(self, performance: Tensor) -> Tensor:
        return design

class GenerativeModel(BaseModel):
    """生成模型: 条件 → 设计"""
    def sample(self, condition: Tensor) -> Tensor:
        return design
```

#### 2.2 模型继承关系

```
BaseModel
├── SurrogateModel
│   ├── CNNSurrogate
│   ├── DeepONet
│   └── PINO
├── InverseModel
│   ├── TandemNetwork (TNN)
│   ├── MDNTandemNetwork
│   └── InverseNetwork
└── GenerativeModel
    ├── VAE (HiLab)
    ├── CGAN
    └── HiLABEngine
```

#### 2.3 HiLab 架构

```
HiLABEngine
├── VAE (变分自编码器)
│   ├── Encoder: 设计 → 潜在向量
│   └── Decoder: 潜在向量 → 设计
├── Surrogate (代理模型)
│   └── 潜在向量 → 性能预测
└── BayesianOptimizer
    ├── GaussianProcessRegressor
    └── AcquisitionFunction (EI/UCB/PI)

工作流程:
1. VAE 训练: 学习设计空间的低维表示
2. 代理训练: 建立潜在向量到性能的映射
3. 贝叶斯优化: 在潜在空间中搜索最优设计
4. 解码生成: 将最优潜在向量解码为设计
```

### 3. Interfaces 模块 - 外部接口

#### 3.1 仿真器接口

```python
class SimulatorInterface(ABC):
    """仿真器接口基类"""
    
    @abstractmethod
    def run(self, design_params: Tensor, **kwargs) -> Dict:
        """运行前向仿真"""
    
    @abstractmethod
    def compute_gradient(self, design_params: Tensor, 
                        objective_grad: Dict) -> Tensor:
        """计算伴随梯度"""
    
    def add_source(self, source: SourceConfig):
        """添加光源"""
    
    def add_monitor(self, monitor: MonitorConfig):
        """添加监视器"""
```

#### 3.2 仿真器实现

| 仿真器 | 实现语言 | 平台支持 | 特点 |
|--------|----------|----------|------|
| Meep | Python/C++ | Linux/macOS | 开源 FDTD，功能完整 |
| Optics | C++/pybind11 | Windows/Linux | 自研，轻量级 |
| RCWA | Python | 跨平台 | 严格耦合波分析 |
| Lumerical | Python API | 跨平台 | 商业软件接口 |

### 4. Workflows 模块 - 工作流管理

#### 4.1 DesignPipeline

```python
class DesignPipeline:
    """设计管道"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.challenge = ChallengeFactory.create(config.challenge_name)
        self.model = None
        self.simulator = None
    
    def setup(self):
        """初始化管道组件"""
    
    def train_model(self) -> Dict:
        """训练模型"""
    
    def inverse_design(self, target: List[float]) -> Dict:
        """执行逆向设计"""
    
    def run(self) -> Dict:
        """运行完整流程"""
```

#### 4.2 TaskDispatcher

```python
class TaskDispatcher:
    """异步任务调度器"""
    
    def submit(self, fn: Callable, **kwargs) -> str:
        """提交任务"""
    
    def get_result(self, task_id: str) -> TaskResult:
        """获取结果"""
    
    def list_tasks(self) -> List[TaskInfo]:
        """列出所有任务"""
```

### 5. Challenges 模块 - 设计挑战

```python
class DesignChallenge(ABC):
    """设计挑战基类"""
    
    def __init__(self, name: str, spec: DesignSpec):
        self.name = name
        self.spec = spec
    
    @abstractmethod
    def setup_simulator(self) -> SimulatorInterface:
        """配置仿真器"""
    
    @abstractmethod
    def compute_objective(self, result: SimulationResult) -> Tensor:
        """计算目标函数"""
    
    def get_initial_design(self) -> Tensor:
        """获取初始设计"""
    
    def evaluate(self, design: Tensor) -> Dict:
        """评估设计性能"""
```

## 数据流

### 逆向设计数据流

```
目标性能 ──────────────────────────────────────────────────────────────┐
                                                                        │
                                                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              逆向设计流程                                    │
│                                                                             │
│  目标性能 ──► 模型推理 ──► 候选设计 ──► 仿真验证 ──► 性能评估 ──► 结果    │
│                  │              │              │                          │
│                  │              │              │                          │
│                  ▼              ▼              ▼                          │
│             [TNN/MDN/     [Meep/Optics]   [目标函数]                      │
│              CGAN/HiLab]                                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 工作流执行数据流

```
用户请求 ──► API层 ──► WorkflowService ──► DesignPipeline ──► 结果
                           │                    │
                           │                    ├─► Challenge
                           │                    ├─► Model
                           │                    ├─► Simulator
                           │                    └─► Optimizer
                           │
                           └─► TaskDispatcher (异步)
```

## 配置管理

### PipelineConfig

```python
@dataclass
class PipelineConfig:
    name: str
    challenge_name: str
    model_type: str = "hilab"
    
    # 训练参数
    num_epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    
    # 优化参数
    num_iterations: int = 50
    
    # 功能开关
    use_simulation: bool = True
    use_constraints: bool = True
    
    # 输出设置
    output_dir: str = "outputs"
    verbose: bool = True
```

### SimulationConfig

```python
@dataclass
class SimulationConfig:
    resolution: int = 50  # 像素/微米
    cell_size: Tuple[float, float, float] = (10.0, 10.0, 0.0)
    boundary_x: BoundaryCondition = BoundaryCondition.PML
    boundary_y: BoundaryCondition = BoundaryCondition.PML
    pml_thickness: float = 1.0
    simulation_time: float = 100.0
    wavelengths: Optional[List[float]] = None
```

## 扩展机制

### 注册系统

框架使用工厂模式和装饰器实现组件注册：

```python
# 注册挑战
@register_challenge("my_challenge")
class MyChallenge(DesignChallenge):
    ...

# 注册仿真器
@register_simulator("my_simulator")
class MySimulator(SimulatorInterface):
    ...

# 注册节点
@register_node("my_node")
class MyNode(Node):
    ...
```

### 插件架构

```
hi_photonics/
├── plugins/
│   ├── __init__.py
│   ├── custom_models/      # 自定义模型
│   ├── custom_simulators/  # 自定义仿真器
│   └── custom_challenges/  # 自定义挑战
└── plugin_manager.py
```

## 性能优化

### 1. 计算图优化
- 节点缓存：避免重复计算
- 拓扑排序：确定最优执行顺序
- 并行执行：独立节点并行处理

### 2. 模型优化
- 混合精度训练 (AMP)
- 梯度累积
- 分布式训练支持

### 3. 仿真优化
- 结果缓存
- 多进程并行
- GPU 加速

## 安全考虑

### 输入验证
- Pydantic 模型验证 API 输入
- 类型检查和范围验证
- 防止注入攻击

### 资源管理
- 任务超时控制
- 内存使用限制
- 并发连接限制

## 部署架构

### 开发环境
```
Frontend (Vite) ──► Backend (Uvicorn) ──► Local Storage
```

### 生产环境
```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Nginx     │────►│   Gunicorn  │────►│   Redis     │
│   反向代理  │     │   WSGI 服务器│     │   缓存      │
└─────────────┘     └─────────────┘     └─────────────┘
                           │
                    ┌──────┴──────┐
                    ▼             ▼
              ┌──────────┐  ┌──────────┐
              │ Worker 1 │  │ Worker 2 │
              │ 任务执行 │  │ 任务执行 │
              └──────────┘  └──────────┘
```

## 未来规划

### 短期 (v0.2.0)
- 完善代理模型
- 增强可视化功能
- 优化工作流性能

### 中期 (v0.3.0)
- 多目标优化支持
- 分布式仿真
- 模型解释性工具

### 长期 (v1.0.0)
- 云平台部署
- 协作设计功能
- 自动超参数优化

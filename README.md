# HI-Photonics

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/SolomonCaptain/HI-Photonics)

---

## 项目概述 

HI-Photonics 是一个基于深度学习的**光子学逆向设计框架**，旨在通过神经网络模型实现高效的光子器件设计与优化。该项目整合了多种先进的深度学习方法，包括 Tandem Network、混合密度网络(MDN)、条件生成对抗网络(CGAN)、物理信息神经网络(PINN) 和图神经网络(GNN)。

**技术栈:**
- Python 3.13
- PyTorch 2.10.0 (CUDA 13.0)
- NumPy, Matplotlib
- Meep (可选，用于 FDTD 仿真)

## 目录结构

```
HI-Photonics/
├── core/               # 核心计算图框架
│   ├── graph.py        # 计算图管理与拓扑排序
│   ├── node.py         # 节点基类
│   └── nodes/          # 各类节点实现
│       ├── parameterization.py  # 参数化节点
│       ├── simulation.py        # 仿真节点
│       ├── objective.py         # 目标函数节点
│       ├── filter.py            # 滤波器节点
│       ├── projection.py        # 投影节点
│       ├── constraint.py        # 约束节点
│       └── composite.py         # 组合节点
├── models/             # 深度学习模型
│   ├── base.py         # 模型基类 (BaseModel, SurrogateModel, InverseModel)
│   ├── inverse/        # 逆向设计模型
│   │   ├── tnn.py      # Tandem Network
│   │   ├── mdn.py      # Mixture Density Network
│   │   ├── cgan.py     # Conditional GAN
│   │   ├── pinn.py     # Physics-Informed Neural Network
│   │   ├── gnn.py      # Graph Neural Network
│   │   └── hilab.py    # HiLab 混合逆向设计框架 (VAE + 贝叶斯优化)
│   ├── surrogates/     # 代理模型
│   │   ├── cnn_surrogate.py
│   │   ├── deeponet.py
│   │   └── pino.py
│   ├── interpret/      # 模型可解释性
│   └── training/       # 训练工具
│       ├── losses.py   # 损失函数
│       ├── metrics.py  # 评估指标
│       └── callbacks.py # 训练回调
├── challenges/         # 设计挑战定义
│   ├── base.py         # 挑战基类 (DesignChallenge)
│   ├── grating_coupler.py    # 光栅耦合器
│   ├── metagrating.py        # 超光栅
│   └── wavelength_demux.py   # 波长解复用器
├── interfaces/         # 外部接口
│   ├── simulators/     # 仿真器接口
│   │   ├── base.py     # 仿真器基类
│   │   ├── meep.py     # Meep FDTD
│   │   ├── lumerical.py # Lumerical
│   │   └── rcwa.py     # RCWA
│   ├── foundry/        # 代工厂接口
│   │   ├── design_rules.py
│   │   └── gds.py
│   └── visualization/  # 可视化工具
├── optimization/       # 优化算法
│   ├── constraints/    # 约束处理
│   └── solvers/        # 求解器
│       ├── gradient_based.py
│       ├── evolutionary.py
│       └── bayesian.py  # 贝叶斯优化 (GP + EI/UCB/PI)
├── data/               # 数据管理
│   ├── generators/     # 数据生成器
│   ├── loaders/        # 数据加载器
│   └── preprocess/     # 数据预处理
├── workflows/          # 工作流管理
│   ├── pipeline.py
│   ├── dispatcher.py
│   └── templates/
├── examples/           # 示例代码
│   ├── 01_basic_design.ipynb
│   ├── 02_mdn_vs_tnn_comparison.ipynb
│   ├── 03_hilab_workflow.ipynb
│   ├── 04_meep_simulation.py
│   ├── 05_tnn_inverse_design.py
│   ├── 06_mdn_inverse_design.py
│   ├── 07_cgan_inverse_design.py
│   ├── 08_pinn_inverse_design.py
│   └── 09_hilab_workflow.py
├── tests/              # 测试代码
├── docs/               # 文档
└── environment/        # 环境配置
    ├── Win64/
    └── Ubuntu/
```

## 核心概念

### 1. 计算图框架 (core/)

项目采用**计算图**架构管理设计流程：

```python
from core import Graph, ParameterizationNode, SimulationNode, ObjectiveNode

# 构建计算图
param_node = ParameterizationNode(...)
sim_node = SimulationNode(inputs=[param_node], ...)
obj_node = ObjectiveNode(inputs=[sim_node], ...)

graph = Graph(output_nodes=[obj_node])
outputs = graph.forward()
```

**节点类型:**
- `ParameterizationNode`: 设计参数表示
- `SimulationNode`: 仿真计算
- `ObjectiveNode`: 目标函数计算
- `FilterNode`: 高斯/形态学滤波
- `ProjectionNode`: Sigmoid/Heaviside 投影
- `ConstraintNode`: 体积/对称性/制造约束

### 2. 设计挑战 (challenges/)

每个设计问题继承 `DesignChallenge` 基类：

```python
from challenges import DesignChallenge, DesignSpec, PerformanceTarget

class MyChallenge(DesignChallenge):
    def setup_simulator(self) -> SimulatorInterface:
        ...
    
    def compute_objective(self, result) -> torch.Tensor:
        ...
    
    def get_initial_design(self) -> torch.Tensor:
        ...
```

**内置挑战:**
- `GratingCouplerChallenge`: 光栅耦合器设计
- `MetagratingChallenge`: 超光栅设计
- `WavelengthDemuxChallenge`: 波长解复用器设计

### 3. 模型体系 (models/)

**模型基类层次:**
```
BaseModel (nn.Module)
├── SurrogateModel     # 正向模型: 设计 → 性能
├── InverseModel       # 逆向模型: 性能 → 设计
└── GenerativeModel    # 生成模型: 条件 → 设计
```

**主要模型:**

| 模型 | 用途 | 特点 |
|------|------|------|
| TandemNetwork (TNN) | 逆向设计 | 级联正向-逆向网络 |
| MDN | 多解逆向设计 | 输出混合高斯分布 |
| CGAN | 条件生成 | 支持多样性和模式覆盖 |
| PINN | 物理约束设计 | 融入 Maxwell 方程 |
| GNN | 结构化设计 | 处理图结构数据 |
| HiLab | 混合逆向设计 | VAE + 贝叶斯优化，高效潜在空间搜索 |

### 5. HiLab 混合逆向设计框架

HiLab 结合 VAE 潜在空间学习与贝叶斯优化，实现高效的光子学逆向设计：

```python
from models.inverse.hilab import HiLABEngine, HiLABConfig, VAEConfig

# 创建 HiLAB 引擎
config = HiLABConfig(
    vae_config=VAEConfig(latent_dim=32, design_shape=(100, 22)),
    performance_dim=3
)
engine = HiLABEngine(config)

# 训练 VAE 学习设计空间
engine.train_vae(train_loader, epochs=100)

# 训练代理模型
engine.train_surrogate(data_loader, epochs=50)

# 逆向设计
target = torch.tensor([[0.95, 0.8, 0.1]])
design = engine.inverse_design(target, n_iterations=30)
```

**工作流程:**
1. VAE 训练: 学习设计空间的低维潜在表示
2. 代理模型: 建立潜在向量到性能的映射
3. 贝叶斯优化: 在潜在空间中搜索最优设计
4. 解码生成: 将潜在向量解码为设计参数
| HiLab | 混合逆向设计 | VAE + 贝叶斯优化，高效潜在空间搜索 |

### 4. 仿真器接口 (interfaces/simulators/)

```python
from interfaces import MeepSimulator, SimulationConfig

config = SimulationConfig(
    wavelength=1.55,
    design_region=DesignRegion(...)
)
simulator = MeepSimulator(config)
result = simulator.run(design_params)
```

## 构建与运行

### 环境配置

```powershell
# Windows (Conda)
conda env create -f environment/Win64/environment.yml
conda activate HI_Photonics

# Ubuntu (Conda)
conda env create -f environment/Ubuntu/environment.yml
conda activate HI_Photonics
```

### 运行测试

```powershell
# 运行基础设计测试
python -m tests.test_core.test_simple_design

# 使用 pytest (推荐)
pytest tests/ -v
```

### 示例运行

```powershell
# 运行 TNN 逆向设计示例
python examples/05_tnn_inverse_design.py

# 运行 MDN 逆向设计示例
python examples/06_mdn_inverse_design.py

# 运行 CGAN 逆向设计示例
python examples/07_cgan_inverse_design.py

# 运行 PINN 逆向设计示例
python examples/08_pinn_inverse_design.py

# 运行 HiLab 混合逆向设计示例
python examples/09_hilab_workflow.py

# 运行 Meep 仿真示例 (需安装 Meep)
python examples/04_meep_simulation.py
```

## 开发约定

### 代码风格

- 使用 Python 3.13 类型提示
- 遵循 PEP 8 规范
- 使用 dataclass 定义配置类
- 抽象基类使用 `ABC` 和 `@abstractmethod`

### 模型开发

新增模型应继承适当的基类：

```python
from models.base import InverseModel, ModelConfig

@dataclass
class MyModelConfig(ModelConfig):
    hidden_dim: int = 128
    ...

class MyModel(InverseModel):
    def __init__(self, config: MyModelConfig):
        super().__init__(config)
        ...
    
    def forward(self, performance: torch.Tensor) -> torch.Tensor:
        ...
```

### 测试规范

- 测试文件放在 `tests/` 对应子目录
- 命名规范: `test_<module_name>.py`
- 使用 Mock 仿真器进行单元测试

### 注册机制

使用工厂模式注册新组件：

```python
# 注册挑战
from challenges import register_challenge

@register_challenge("my_challenge")
class MyChallenge(DesignChallenge):
    ...

# 注册仿真器
from interfaces import register_simulator

@register_simulator("my_simulator")
class MySimulator(SimulatorInterface):
    ...
```

## 关键依赖

| 包 | 版本 | 用途 |
|---|------|------|
| torch | 2.10.0+cu130 | 深度学习框架 |
| numpy | 2.3.5 | 数值计算 |
| matplotlib | 3.10.8 | 可视化 |
| meep | 可选 | FDTD 仿真 |

## CI/CD

GitHub Actions 配置位于 `.github/workflows/run-tests.yml`：
- 触发: push/PR 到 main/master/develop
- Python 版本: 3.13
- 自动运行 `tests/test_core/test_simple_design`

## 常见任务

### 添加新的设计挑战

1. 在 `challenges/` 创建新文件
2. 继承 `DesignChallenge` 并实现抽象方法
3. 在 `challenges/__init__.py` 导出
4. 使用 `@register_challenge` 注册

### 添加新的神经网络模型

1. 在 `models/inverse/` 或 `models/surrogates/` 创建文件
2. 继承 `BaseModel`/`InverseModel`/`SurrogateModel`
3. 实现 `forward()` 方法
4. 在 `models/__init__.py` 导出
5. 添加对应损失函数到 `models/training/losses.py`

### 集成新的仿真器

1. 在 `interfaces/simulators/` 创建文件
2. 继承 `SimulatorInterface`
3. 使用 `@register_simulator` 注册
4. 在 `interfaces/__init__.py` 条件导出

## 注意事项

- Windows 系统使用 PowerShell 命令语法
- Meep 仅在 Linux/macOS 可用，Windows 需使用 WSL
- GPU 加速需要 CUDA 13.0 兼容显卡
- 大规模仿真建议使用远程服务器

---

## 关于开发者

个人博客：[http://blog.istaroth.xin/tags/hi-photonics](http://blog.istaroth.xin/tags/hi-photonics)

邮箱：[zou12345@qq.com](mailto:zou12345@qq.com)
# HI-Photonics

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/SolomonCaptain/HI-Photonics)

---

## 项目概述

HI-Photonics 是一个基于深度学习的**光子学逆向设计框架**，旨在通过神经网络模型实现高效的光子器件设计与优化。该项目整合了多种先进的深度学习方法，包括 Tandem Network、混合密度网络(MDN)、条件生成对抗网络(CGAN)、物理信息神经网络(PINN) 和 HiLab 混合逆向设计框架。

**核心特性:**
- 多种逆向设计模型（TNN, MDN, CGAN, PINN, HiLab）
- Safetensors 模型格式支持（Windows 友好）
- 完整的工作流管道和任务调度
- FDTD/RCWA 仿真器接口
- 贝叶斯优化和多物理场约束
- FastAPI 后端 + React 前端
- Windows 平台优化支持

**技术栈:**
- Python 3.9+
- PyTorch 2.0+ (CUDA 13.0)
- NumPy, SciPy, Matplotlib
- Meep / Optics FDTD (可选，用于仿真)
- FastAPI + React (Web 界面)

## 快速开始

### 安装

```bash
# 克隆仓库
git clone https://github.com/SolomonCaptain/HI-Photonics.git
cd HI-Photonics

# 安装依赖
pip install -e .

# 或者使用 conda
conda env create -f environment/Win64/environment.yml
conda activate HI_Photonics
```

### 基本使用

```python
from hi_photonics import create_pipeline, run_quick_design

# 方式1: 快速逆向设计
result = run_quick_design(
    challenge_name="grating_coupler",
    target_performance={"efficiency": 0.8},
    num_iterations=50
)

# 方式2: 完整管道
pipeline = create_pipeline(
    challenge_name="metagrating",
    model_type="hilab",
    num_epochs=100
)
result = pipeline.run()

# 方式3: 使用命令行
# hi-photonics run -c grating_coupler -m hilab
```

## 目录结构

```
HI-Photonics/
├── hi_photonics/       # 统一入口模块
│   ├── __init__.py     # 导出所有公共 API
│   └── cli.py          # 命令行接口
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
│   ├── base.py         # 模型基类
│   ├── inverse/        # 逆向设计模型
│   │   ├── tnn.py      # Tandem Network
│   │   ├── mdn.py      # Mixture Density Network
│   │   ├── cgan.py     # Conditional GAN
│   │   ├── pinn.py     # Physics-Informed Neural Network
│   │   ├── gnn.py      # Graph Neural Network
│   │   └── hilab.py    # HiLab 混合框架
│   ├── surrogates/     # 代理模型
│   ├── interpret/      # 模型可解释性
│   └── training/       # 训练工具
│       ├── losses.py   # 损失函数库
│       ├── metrics.py  # 评估指标
│       └── callbacks.py # 训练回调
├── challenges/         # 设计挑战定义
│   ├── base.py         # 挑战基类
│   ├── grating_coupler.py    # 光栅耦合器
│   ├── metagrating.py        # 超光栅
│   └── wavelength_demux.py   # 波长解复用器
├── interfaces/         # 外部接口
│   ├── simulators/     # 仿真器接口
│   │   ├── base.py     # 仿真器基类
│   │   ├── meep.py     # Meep FDTD
│   │   ├── rcwa.py     # RCWA
│   │   └── optics/     # C++ FDTD 实现
│   ├── foundry/        # 代工厂接口
│   └── visualization/  # 可视化工具
├── optimization/       # 优化算法
│   ├── constraints/    # 约束处理
│   └── solvers/        # 求解器
│       └── bayesian.py # 贝叶斯优化
├── data/               # 数据管理
│   ├── generators/     # 数据生成器
│   ├── loaders/        # 数据加载器
│   └── preprocess/     # 数据预处理
├── workflows/          # 工作流管理
│   ├── pipeline.py     # 设计管道
│   └── dispatcher.py   # 任务调度
├── api/                # FastAPI 后端
│   ├── main.py         # API 入口
│   ├── routers/        # 路由模块
│   └── services/       # 服务层
├── frontend/           # React 前端
│   └── src/            # TypeScript 源码
├── examples/           # 示例代码
├── tests/              # 测试代码
└── docs/               # 文档
```

## 核心概念

### 1. 统一入口 (hi_photonics/)

所有公共 API 通过 `hi_photonics` 模块导出：

```python
from hi_photonics import (
    # 核心类
    Node, Graph, TensorLike,
    
    # 模型
    TandemNetwork, MDN, CGAN, HiLABEngine,
    create_tnn_for_challenge,
    create_hilab_for_challenge,
    
    # 工作流
    DesignPipeline, create_pipeline, run_quick_design,
    
    # 挑战
    ChallengeFactory, list_available_challenges,
    
    # 工具
    get_loss, get_metric, get_default_callbacks,
)
```

### 2. 计算图框架 (core/)

项目采用**计算图**架构管理设计流程：

```python
from hi_photonics import Node, Graph

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
| HiLab | 混合逆向设计 | VAE + 贝叶斯优化 |

**模型保存格式:**

框架支持两种模型保存格式：

```python
# PyTorch 格式 (默认)
model.save("model.pt")

# Safetensors 格式 (推荐用于 Windows)
model.save("model.safetensors", format="safetensors")

# 格式转换
from hi_photonics import convert_torch_to_safetensors
convert_torch_to_safetensors("model.pt", "model.safetensors")
```

**Windows 平台优化:**

框架针对 Windows 平台进行了优化：
- 自动调整 DataLoader 的 `num_workers=0`
- 使用 `file_system` 多进程共享策略
- 支持 Safetensors 格式避免 pickle 兼容性问题
- 混合精度训练 (AMP) 减少内存占用

### 4. 工作流管道 (workflows/)

完整的设计流程管道：

```python
from hi_photonics import DesignPipeline, PipelineConfig

config = PipelineConfig(
    name="my_design",
    challenge_name="grating_coupler",
    model_type="hilab",
    num_epochs=100,
    num_iterations=50,
    use_simulation=True,
    use_constraints=True
)

pipeline = DesignPipeline(config)
result = pipeline.run()
```

**管道功能:**
- 自动数据生成/加载
- 模型训练和管理
- 逆向设计优化
- 仿真验证
- 结果保存和可视化

### 5. 设计挑战 (challenges/)

每个设计问题继承 `DesignChallenge` 基类：

```python
from hi_photonics import DesignChallenge, DesignSpec, PerformanceTarget

class MyChallenge(DesignChallenge):
    def setup_simulator(self) -> SimulatorInterface:
        ...
    
    def compute_objective(self, result) -> torch.Tensor:
        ...
    
    def get_initial_design(self) -> torch.Tensor:
        ...
```

**内置挑战:**
- `grating_coupler` - 光栅耦合器设计
- `metagrating` - 超光栅设计
- `wavelength_demux` - 波长解复用器设计

### 6. HiLab 混合逆向设计框架

HiLab 结合 VAE 潜在空间学习与贝叶斯优化：

```python
from hi_photonics import HiLABEngine, create_hilab_for_challenge

# 使用工厂函数创建
engine = create_hilab_for_challenge(
    "grating_coupler",
    latent_dim=32
)

# 训练 VAE
engine.train_vae(train_loader, epochs=100)

# 逆向设计
target = torch.tensor([[0.95, 0.8, 0.1]])
design = engine.inverse_design(target, n_iterations=30)
```

## 命令行工具

安装后可使用 `hi-photonics` 命令：

```bash
# 显示帮助
hi-photonics --help

# 列出资源
hi-photonics list challenges
hi-photonics list simulators
hi-photonics list models

# 运行逆向设计
hi-photonics run -c grating_coupler -m hilab -t "[0.9, 0.8, 0.1]"

# 训练模型
hi-photonics train -c grating_coupler -m hilab -e 100

# 启动 API 服务
hi-photonics api --port 8000
```

## API 服务

### 启动服务

```bash
# 使用命令行
hi-photonics api --port 8000

# 或直接运行
cd api && uvicorn main:app --reload
```

### API 端点

| 端点 | 方法 | 描述 |
|------|------|------|
| `/workflow/nodes` | GET | 获取节点定义 |
| `/workflow/execute` | POST | 执行工作流 |
| `/workflow/pipeline/create` | POST | 创建设计管道 |
| `/workflow/pipeline/{id}/run` | POST | 运行管道 |
| `/workflow/train` | POST | 训练模型 |
| `/workflow/inverse-design` | POST | 逆向设计 |
| `/workflow/simulate` | POST | 仿真设计 |

### 示例请求

```python
import requests

# 创建管道
response = requests.post(
    "http://localhost:8000/workflow/pipeline/create",
    params={
        "challenge_name": "grating_coupler",
        "model_type": "hilab"
    }
)
pipeline_id = response.json()["pipeline_id"]

# 运行设计
response = requests.post(
    f"http://localhost:8000/workflow/pipeline/{pipeline_id}/run",
    json={"target_performance": [0.9, 0.8, 0.1]}
)
result = response.json()
```

## 前端界面

React 前端提供可视化工作流编辑器：

```bash
cd frontend
npm install
npm run dev
```

**功能:**
- 拖拽式工作流编辑
- 节点配置面板
- 实时执行状态
- 结果可视化

## 开发指南

### 添加新的设计挑战

```python
# challenges/my_challenge.py
from hi_photonics import DesignChallenge, register_challenge

@register_challenge("my_challenge")
class MyChallenge(DesignChallenge):
    def __init__(self):
        super().__init__(
            name="my_challenge",
            spec=DesignSpec(...)
        )
    
    def setup_simulator(self):
        ...
    
    def compute_objective(self, result):
        ...
```

### 添加新的模型

```python
# models/inverse/my_model.py
from hi_photonics import InverseModel, ModelConfig

@dataclass
class MyModelConfig(ModelConfig):
    hidden_dim: int = 128

class MyModel(InverseModel):
    def __init__(self, config: MyModelConfig):
        super().__init__(config)
        ...
    
    def forward(self, performance: torch.Tensor) -> torch.Tensor:
        ...
```

### 添加新的仿真器

```python
# interfaces/simulators/my_simulator.py
from hi_photonics import SimulatorInterface, register_simulator

@register_simulator("my_simulator")
class MySimulator(SimulatorInterface):
    def run(self, design_params, **kwargs):
        ...
    
    def compute_gradient(self, design_params, objective_grad, **kwargs):
        ...
```

## 测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行特定模块测试
pytest tests/test_models/ -v

# 带覆盖率
pytest tests/ --cov=hi_photonics --cov-report=html
```

## 示例

| 文件 | 描述 |
|------|------|
| `examples/01_basic_design.ipynb` | 基础设计流程 |
| `examples/02_mdn_vs_tnn_comparison.ipynb` | MDN vs TNN 对比 |
| `examples/03_hilab_workflow.ipynb` | HiLab 工作流 |
| `examples/04_meep_simulation.py` | Meep 仿真 |
| `examples/05_tnn_inverse_design.py` | TNN 逆向设计 |
| `examples/06_mdn_inverse_design.py` | MDN 逆向设计 |
| `examples/07_cgan_inverse_design.py` | CGAN 逆向设计 |
| `examples/08_pinn_inverse_design.py` | PINN 逆向设计 |
| `examples/09_hilab_workflow.py` | HiLab 完整流程 |
| `examples/10_windows_training_safetensors.py` | Windows 训练 + Safetensors |

## 关键依赖

| 包 | 版本 | 用途 |
|---|------|------|
| torch | >=2.0 | 深度学习框架 |
| safetensors | >=0.4.0 | 模型序列化（跨平台） |
| numpy | >=1.20 | 数值计算 |
| scipy | >=1.7 | 科学计算 |
| matplotlib | >=3.5 | 可视化 |
| h5py | >=3.0 | 数据存储 |
| fastapi | >=0.100 | API 服务 |
| pydantic | >=2.0 | 数据验证 |

## CI/CD

GitHub Actions 配置位于 `.github/workflows/run-tests.yml`：
- 触发: push/PR 到 main/master/develop
- Python 版本: 3.9, 3.10, 3.11, 3.12
- 自动运行测试套件

## 版本历史

### v0.1.0 (当前)
- 完整的模型实现 (TNN, MDN, CGAN, PINN, HiLab)
- Safetensors 模型格式支持
- Windows 平台优化训练
- 工作流管道和任务调度
- FastAPI 后端 + React 前端
- 资源管理系统 (inputs/outputs/op_models)
- 命令行工具
- C++ FDTD 仿真器 (Windows 兼容)

## 贡献指南

1. Fork 仓库
2. 创建功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送分支 (`git push origin feature/amazing-feature`)
5. 创建 Pull Request

## 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

## 联系方式

- 博客: [http://blog.istaroth.xin/tags/hi-photonics](http://blog.istaroth.xin/tags/hi-photonics)
- 邮箱: [zou12345@qq.com](mailto:zou12345@qq.com)
- Issues: [GitHub Issues](https://github.com/SolomonCaptain/HI-Photonics/issues)

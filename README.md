# HI-Photonics

---
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/SolomonCaptain/HI-Photonics)

## 项目概述

HI-Photonics 是一个基于深度学习的光子学逆向设计框架，旨在通过神经网络和伴随方法实现光子器件的自动化设计优化。项目结合了物理仿真、机器学习代理模型和拓扑优化技术。

## 核心架构

### 计算图框架 (`core/`)

项目采用基于计算图的架构，核心组件包括：

- **Node (节点)**: 所有计算节点的抽象基类，支持前向计算、反向传播和缓存机制
- **Graph (图)**: 管理计算图的拓扑排序和执行顺序
- **节点类型**:
  - `ParameterizationNode`: 设计参数节点，作为图的根节点
  - `SimulationNode`: 仿真节点
  - `ObjectiveNode`: 目标函数节点，计算标量损失值
  - `ConstraintNode`: 约束节点

```python
# 典型使用模式
from core.node import Node
from core.graph import Graph
from core.nodes.parameterization import ParameterizationNode
from core.nodes.objective import ObjectiveNode

# 1. 定义设计参数
d1 = ParameterizationNode('d1', torch.tensor([0.1], requires_grad=True))

# 2. 构建计算图
# ... 添加仿真节点和目标节点 ...

# 3. 执行优化
graph = Graph([objective])
graph.forward()
graph.backward([torch.tensor(1.0)])
```

## 目录结构

```
HI-Photonics/
├── core/                    # 核心计算图框架
│   ├── node.py             # 节点基类
│   ├── graph.py            # 计算图管理
│   ├── nodes/              # 各类节点实现
│   │   ├── parameterization.py
│   │   ├── simulation.py
│   │   ├── objective.py
│   │   └── constraint.py
│   └── utils/              # 工具函数
│       ├── adjoint.py      # 伴随方法
│       └── autodiff.py     # 自动微分
│
├── models/                  # 神经网络模型
│   ├── base.py             # 模型基类
│   ├── inverse/            # 逆向设计模型
│   │   ├── mdn.py          # 混合密度网络
│   │   ├── tnn.py          # Transformer 网络
│   │   ├── cgan.py         # 条件生成对抗网络
│   │   ├── pinn.py         # 物理信息神经网络
│   │   └── hilab.py        # HiLab 工作流
│   ├── surrogates/         # 代理模型
│   │   ├── cnn_surrogate.py
│   │   ├── deeponet.py
│   │   └── pino.py
│   ├── interpret/          # 可解释性分析
│   └── training/           # 训练相关
│       ├── losses.py
│       ├── metrics.py
│       └── callbacks.py
│
├── challenges/              # 设计挑战定义
│   ├── base.py
│   ├── grating_coupler.py  # 光栅耦合器
│   ├── metagrating.py      # 超构光栅
│   └── wavelength_demux.py # 波长解复用器
│
├── optimization/            # 优化算法
│   ├── solvers/
│   │   ├── gradient_based.py
│   │   ├── bayesian.py
│   │   └── evolutionary.py
│   └── constraints/
│       ├── curvature.py
│       ├── length_scale.py
│       └── projection.py
│
├── data/                    # 数据处理
│   ├── generators/         # 数据生成器
│   │   ├── active_learning.py
│   │   ├── multi_fidelity.py
│   │   └── random_sampling.py
│   ├── loaders/            # 数据加载
│   └── preprocess/         # 预处理
│
├── interfaces/              # 外部接口
│   ├── simulators/         # 仿真器接口
│   │   ├── base.py
│   │   ├── lumerical.py
│   │   ├── meep.py
│   │   └── rcwa.py
│   ├── foundry/            # 代工厂接口
│   │   ├── design_rules.py
│   │   └── gds.py
│   └── visualization/      # 可视化
│
├── workflows/               # 工作流管理
│   ├── pipeline.py
│   ├── dispatcher.py
│   └── templates/
│
├── tests/                   # 测试
│   └── test_core/
│       └── test_simple_design.py
│
└── examples/                # Jupyter 示例
    ├── 01_basic_design.ipynb
    ├── 02_mdn_vs_tnn_comparison.ipynb
    └── 03_hilab_workflow.ipynb
```

## 技术栈

- **Python**: 3.13
- **PyTorch**: 2.10.0+cu130 (CUDA 13.0)
- **NumPy**: 2.3.5
- **Matplotlib**: 3.10.8
- **其他依赖**: networkx, sympy, pillow

## 构建与运行

### 环境设置

```bash
# 使用 Conda 创建环境 (Windows)
conda env create -f environment/Win64/environment.yml
conda activate HI_Photonics

# 或手动安装核心依赖
pip install torch numpy matplotlib
```

### 运行测试

```bash
# 运行基础测试
python -m tests.test_core.test_simple_design

# 设置 Matplotlib 后端 (无显示环境)
export MPLBACKEND=Agg  # Linux/Mac
set MPLBACKEND=Agg     # Windows
```

### 安装项目包

```bash
pip install -e .
```

## 开发规范

### 代码风格

- 使用类型注解 (`typing` 模块)
- 遵循 Python 命名约定：
  - 类名: PascalCase (如 `ParameterizationNode`)
  - 函数/方法: snake_case (如 `forward`, `add_input`)
  - 私有属性: 前缀下划线 (如 `_inputs`, `_cached_output`)
- 使用抽象基类 (`ABC`, `@abstractmethod`) 定义接口

### 节点开发规范

创建自定义节点时需：

1. 继承 `Node` 基类
2. 实现 `forward()` 方法
3. 通过 `add_input()` 添加依赖节点
4. 使用 `_cached_output` 缓存计算结果

```python
class CustomNode(Node):
    def __init__(self, name: str, input_node: Node):
        super().__init__(name)
        self.add_input(input_node)
    
    def forward(self, **kwargs) -> torch.Tensor:
        input_val = self._inputs[0].forward(**kwargs)
        result = self._compute(input_val)
        self._cached_output = result
        return result
```

### 仿真器接口

实现自定义仿真器需继承 `SimulatorInterface`:

```python
class CustomSimulator(SimulatorInterface):
    def run(self, design_params: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        # 执行仿真，返回物理量字典
        pass
    
    def compute_gradient(self, design_params, objective_grad, **kwargs) -> torch.Tensor:
        # 伴随法计算梯度
        pass
```

## CI/CD

项目使用 GitHub Actions 进行持续集成：

- 触发条件: push/PR 到 `main`, `master`, `develop` 分支
- 测试环境: Ubuntu + Python 3.13
- 配置文件: `.github/workflows/run-tests.yml`

## 注意事项

1. **Windows 环境**: 项目支持 Windows，注意路径分隔符和 PowerShell 语法
2. **GPU 加速**: 默认使用 CUDA 13.0，确保显卡驱动兼容
3. **仿真器依赖**: Lumerical, Meep 等外部仿真器需单独安装
4. **内存管理**: 大规模仿真注意清理节点缓存 (`clear_cache=True`)
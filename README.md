# HI-Photonics

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/SolomonCaptain/HI-Photonics)

---

## 项目概述

HI-Photonics 是一个光子学逆向设计框架，基于 PyTorch 构建计算图架构，支持多种仿真后端（Meep、Lumerical、RCWA）和深度学习模型（CGAN、HiLab、MDN、PINN、TNN）进行光子器件的自动化设计与优化。

**核心特性：**
- 节点式计算图框架，支持自动微分和伴随方法
- 多仿真器统一接口（Meep FDTD、RCWA 等）
- 内置多种设计挑战（光栅耦合器、超构光栅、波长解复用器）
- 丰富的约束和投影节点（最小特征尺寸、曲率约束、制造约束等）
- 支持多保真度和主动学习数据生成

## 目录结构

```
HI-Photonics/
├── core/                    # 核心计算图框架
│   ├── node.py             # 节点基类
│   ├── graph.py            # 计算图管理
│   ├── nodes/              # 各类节点实现
│   │   ├── parameterization.py   # 设计参数节点
│   │   ├── simulation.py         # 仿真节点
│   │   ├── objective.py          # 目标函数节点
│   │   ├── filter.py             # 滤波器节点（高斯、Sobel、形态学等）
│   │   ├── projection.py         # 投影节点（Sigmoid、Heaviside等）
│   │   ├── constraint.py         # 约束节点（最小特征尺寸、曲率等）
│   │   └── composite.py          # 组合节点（Pipeline、Merge等）
│   └── utils/              # 工具函数
│       ├── adjoint.py      # 伴随方法实现
│       ├── autodiff.py     # 自动微分
│       └── typing.py       # 类型定义
├── challenges/             # 设计挑战定义
│   ├── base.py             # 挑战基类（DesignChallenge, DesignSpec）
│   ├── grating_coupler.py  # 光栅耦合器
│   ├── metagrating.py      # 超构光栅
│   └── wavelength_demux.py # 波长解复用器
├── interfaces/             # 外部接口
│   ├── simulators/         # 仿真器接口
│   │   ├── base.py         # 仿真器基类（SimulatorInterface）
│   │   ├── meep.py         # Meep FDTD 仿真器
│   │   ├── lumerical.py    # Lumerical 接口
│   │   └── rcwa.py         # RCWA 仿真器
│   ├── foundry/            # 代工厂接口
│   │   ├── design_rules.py # 设计规则
│   │   └── gds.py          # GDS 文件处理
│   └── visualization/      # 可视化
│       ├── field.py        # 场分布可视化
│       └── structure.py    # 结构可视化
├── models/                 # 神经网络模型
│   ├── inverse/            # 逆向设计模型
│   │   ├── cgan.py         # 条件生成对抗网络
│   │   ├── hilab.py        # HiLab 模型
│   │   ├── mdn.py          # 混合密度网络
│   │   ├── pinn.py         # 物理信息神经网络
│   │   └── tnn.py          # 张量神经网络
│   ├── surrogates/         # 代理模型
│   │   ├── cnn_surrogate.py
│   │   ├── deeponet.py
│   │   └── pino.py
│   ├── interpret/          # 可解释性
│   └── training/           # 训练工具
│       ├── callbacks.py
│       ├── losses.py
│       └── metrics.py
├── optimization/           # 优化算法
│   ├── constraints/        # 约束处理
│   └── solvers/            # 求解器
│       ├── gradient_based.py   # 梯度优化
│       ├── evolutionary.py     # 进化算法
│       └── bayesian.py         # 贝叶斯优化
├── data/                   # 数据处理
│   ├── generators/         # 数据生成器
│   ├── loaders/            # 数据加载器
│   └── preprocess/         # 预处理
├── workflows/              # 工作流
│   ├── pipeline.py         # 设计管道
│   ├── dispatcher.py       # 任务调度
│   └── templates/          # 模板
├── tests/                  # 测试
│   ├── test_core/
│   └── test_interfaces/
├── examples/               # 示例代码
│   └── 04_meep_simulation.py
├── environment/            # 环境配置
│   ├── Win64/environment.yml
│   └── Ubuntu/environment.yml
└── docs/                   # 文档
```

## 核心架构

### 计算图框架

项目采用节点式计算图架构，所有计算单元继承自 `Node` 基类：

```python
from core.node import Node
from core.graph import Graph

class Node(ABC):
    def __init__(self, name: str, params: Optional[Params] = None):
        self.name = name
        self._inputs: List['Node'] = []
        self._outputs: List['Node'] = []
        self._cached_output: Optional[TensorLike] = None
    
    @abstractmethod
    def forward(self, **kwargs) -> TensorLike:
        pass
    
    def backward(self, grad_output: torch.Tensor):
        pass
```

### 典型设计流程

```python
from challenges import ChallengeFactory
from core.nodes.parameterization import ParameterizationNode
from core.nodes.filter import GaussianFilterNode
from core.nodes.projection import SigmoidProjectionNode
from core.graph import Graph

# 1. 创建设计挑战
challenge = ChallengeFactory.create('grating_coupler')

# 2. 构建处理管道
param = ParameterizationNode('param', challenge.get_initial_design(), requires_grad=True)
smooth = GaussianFilterNode('smooth', param, kernel_size=5)
proj = SigmoidProjectionNode('proj', smooth, threshold=0.5)

# 3. 评估设计
processed = proj.forward()
objective, info = challenge.evaluate(processed)

# 4. 优化循环
optimizer = torch.optim.Adam([param.value], lr=0.01)
for i in range(100):
    optimizer.zero_grad()
    design = proj.forward()
    loss, _ = challenge.evaluate(design)
    loss.backward()
    optimizer.step()
```

## 构建与运行

### 环境安装

```bash
# Windows (Conda)
conda env create -f environment/Win64/environment.yml
conda activate HI_Photonics

# Ubuntu (Conda)
conda env create -f environment/Ubuntu/environment.yml
conda activate HI_Photonics
```

### 运行测试

```bash
# 设置 matplotlib 后端（无 GUI 环境）
set MPLBACKEND=Agg

# 运行测试
python -m tests.test_core.test_simple_design
```

### 运行示例

```bash
python examples/04_meep_simulation.py
```

## 开发规范

### 代码风格

- 使用 Python 3.13 特性
- 类型注解：使用 `typing` 模块和 `torch.Tensor` 类型
- 文档字符串：使用中文，遵循 Google 风格
- 命名约定：
  - 类名：PascalCase（如 `GratingCouplerChallenge`）
  - 函数/方法：snake_case（如 `compute_objective`）
  - 私有成员：前缀下划线（如 `_inputs`）

### 注册模式

使用装饰器注册组件：

```python
# 注册挑战
@register_challenge('my_challenge')
class MyChallenge(DesignChallenge):
    pass

# 注册仿真器
@register_simulator('my_simulator')
class MySimulator(SimulatorInterface):
    pass

# 使用工厂创建
challenge = ChallengeFactory.create('my_challenge')
simulator = SimulatorFactory.create('my_simulator')
```

### 节点开发

创建自定义节点：

```python
from core.node import Node

class MyCustomNode(Node):
    def __init__(self, name: str, input_node: Node, **params):
        super().__init__(name, params)
        self.add_input(input_node)
    
    def forward(self, **kwargs) -> torch.Tensor:
        input_data = self._inputs[0].forward(**kwargs)
        # 处理逻辑
        return result
```

## 可用设计挑战

| 挑战名称 | 类 | 描述 |
|---------|-----|------|
| `grating_coupler` | `GratingCouplerChallenge` | 光纤到芯片光栅耦合器 |
| `metagrating` | `MetagratingChallenge` | 超构光栅（偏转光束） |
| `wavelength_demux` | `WavelengthDemuxChallenge` | 波长解复用器 |

## 仿真器支持

| 仿真器 | 状态 | 描述 |
|--------|------|------|
| Meep | 可选 | 开源 FDTD，需 `conda install -c conda-forge pymeep` |
| Mock | 内置 | 模拟仿真器，用于无 Meep 环境测试 |
| Lumerical | 规划中 | 商业 FDTD/FEM |
| RCWA | 规划中 | 严格耦合波分析 |

## 常用材料

```python
from interfaces.simulators.meep import Material

# 预定义材料
si = Material(name='silicon')      # n = 3.48
sio2 = Material(name='sio2')       # n = 1.44
sin = Material(name='sin')         # n = 2.0

# 自定义材料
custom = Material(n=2.5)
```

## 依赖关系

**核心依赖：**
- Python 3.13
- PyTorch 2.10+ (CUDA 13.0)
- NumPy 2.3+
- Matplotlib 3.10+

**可选依赖：**
- pymeep: FDTD 仿真（conda install -c conda-forge pymeep）
- h5py: HDF5 数据加载

## 注意事项

1. **Windows 兼容性**：Meep 在 Windows 上需要 WSL 或特殊配置，建议使用 Mock 仿真器进行开发测试
2. **GPU 支持**：PyTorch 配置为 CUDA 13.0，确保显卡驱动兼容
3. **缓存机制**：节点使用 `_cached_output` 缓存结果，优化时需调用 `clear_cache()` 或使用 `graph.forward(clear_cache=True)`
4. **伴随方法**：仿真器实现 `compute_gradient()` 方法支持梯度计算，用于拓扑优化

---

## 关于开发者

个人博客：[http://blog.istaroth.xin/tags/hi-photonics](http://blog.istaroth.xin/tags/hi-photonics)

邮箱：[zou12345@qq.com](mailto:zou12345@qq.com)
# LLM 集成逆向设计优化方案

> 版本: 0.1.0 | 创建日期: 2026-03-22

## 概述

本文档探讨将大语言模型 (LLM) 和提示词工程集成到 HI-Photonics 逆向设计框架中的可行性、具体方案和实施建议。

---

## 1. 当前逆向设计流程分析

### 1.1 现有工作流

```
用户输入目标性能 → 选择模型 → 配置参数 → 执行逆向设计 → 仿真验证 → 结果评估
```

### 1.2 主要痛点

| 痛点 | 描述 | 影响程度 |
|------|------|----------|
| **专业门槛高** | 用户需要理解光子学、深度学习、优化算法等多个领域知识 | 高 |
| **参数配置复杂** | 模型选择、超参数调优需要丰富经验 | 高 |
| **结果解读困难** | 设计结果需要专业知识才能正确解读 | 中 |
| **工作流编排繁琐** | 手动组合节点、配置依赖关系容易出错 | 中 |
| **调试效率低** | 问题定位需要查阅多处文档和代码 | 中 |

---

## 2. LLM 集成可行性分析

### 2.1 技术可行性

| 维度 | 评估 | 说明 |
|------|------|------|
| **API 可用性** | ✅ 高 | OpenAI、Claude、国产大模型 API 成熟稳定 |
| **领域知识** | ⚠️ 中 | 通用 LLM 对光子学专业知识有限，需 RAG 增强 |
| **代码生成** | ✅ 高 | LLM 在代码生成和解释方面表现出色 |
| **自然语言理解** | ✅ 高 | 可准确理解用户意图并转换为结构化输入 |
| **推理能力** | ✅ 高 | 可进行多步骤推理和工作流规划 |

### 2.2 集成场景矩阵

```
┌─────────────────────────────────────────────────────────────────────┐
│                    LLM 集成场景价值矩阵                              │
├──────────────────┬──────────────┬──────────────┬────────────────────┤
│     场景         │  实现难度    │   价值贡献   │     优先级         │
├──────────────────┼──────────────┼──────────────┼────────────────────┤
│ 自然语言目标解析 │    低        │     高       │     P0             │
│ 工作流自动编排   │    中        │     高       │     P0             │
│ 模型选择建议     │    低        │     中       │     P1             │
│ 参数配置优化     │    中        │     高       │     P1             │
│ 结果解释报告     │    低        │     高       │     P0             │
│ 代码生成辅助     │    中        │     中       │     P2             │
│ 知识问答系统     │    中        │     中       │     P2             │
│ 多轮设计优化     │    高        │     高       │     P1             │
└──────────────────┴──────────────┴──────────────┴────────────────────┘
```

---

## 3. 集成方案设计

### 3.1 系统架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            LLM 增强逆向设计系统                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        用户交互层 (LLM 接口)                          │   │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐                  │   │
│  │  │ 自然语言输入 │ │ 对话式交互   │ │ 智能建议     │                  │   │
│  │  └──────────────┘ └──────────────┘ └──────────────┘                  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        LLM 服务层                                    │   │
│  │  ┌─────────────────────────────────────────────────────────────┐    │   │
│  │  │                    PromptOrchestrator                        │    │   │
│  │  │  • 意图识别  • 实体抽取  • 上下文管理  • 响应解析           │    │   │
│  │  └─────────────────────────────────────────────────────────────┘    │   │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐                  │   │
│  │  │ Prompt模板库 │ │ RAG 知识库   │ │ 上下文缓存   │                  │   │
│  │  └──────────────┘ └──────────────┘ └──────────────┘                  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      HI-Photonics 核心层                             │   │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐                  │   │
│  │  │ Workflows    │ │ Models       │ │ Challenges   │                  │   │
│  │  └──────────────┘ └──────────────┘ └──────────────┘                  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 核心组件设计

#### 3.2.1 PromptOrchestrator (提示词编排器)

```python
class PromptOrchestrator:
    """LLM 提示词编排和响应处理"""
    
    def __init__(self, llm_client, knowledge_base):
        self.llm = llm_client
        self.knowledge_base = knowledge_base
        self.context = ConversationContext()
        self.templates = PromptTemplateLibrary()
    
    def parse_design_intent(self, user_input: str) -> DesignIntent:
        """解析用户设计意图"""
        
    def generate_workflow_config(self, intent: DesignIntent) -> dict:
        """生成工作流配置"""
        
    def explain_results(self, results: dict) -> str:
        """解释设计结果"""
        
    def suggest_improvements(self, results: dict) -> List[Suggestion]:
        """提出改进建议"""
```

#### 3.2.2 RAG 知识库设计

```
knowledge_base/
├── photonics/
│   ├── devices.json          # 器件知识 (grating_coupler, waveguide, etc.)
│   ├── materials.json        # 材料属性
│   ├── physics.json          # 物理原理
│   └── design_rules.json     # 设计规则
├── models/
│   ├── tnn.md                # TNN 模型说明
│   ├── mdn.md                # MDN 模型说明
│   ├── cgan.md               # CGAN 模型说明
│   ├── pinn.md               # PINN 模型说明
│   └── hilab.md              # HiLab 模型说明
├── workflows/
│   ├── templates.json        # 工作流模板
│   └── best_practices.md     # 最佳实践
└── examples/
    ├── design_cases.json     # 设计案例
    └── troubleshooting.md    # 故障排除
```

---

## 4. 提示词工程应用

### 4.1 核心提示词模板

#### 4.1.1 意图解析提示词

```
你是一个光子学逆向设计助手。分析用户的自然语言输入，提取结构化设计意图。

用户输入: {user_input}

请输出以下 JSON 格式:
{{
    "device_type": "器件类型 (grating_coupler/metagrating/wavelength_demux)",
    "target_specs": {{
        "wavelength": "目标波长 (nm)",
        "efficiency": "目标效率 (%)",
        "bandwidth": "带宽要求 (nm)",
        "other_metrics": "其他指标"
    }},
    "constraints": {{
        "fabrication": "制造约束",
        "size": "尺寸约束",
        "materials": "材料约束"
    }},
    "preferences": {{
        "model_preference": "模型偏好 (tnm/mdn/cgan/pinn/hilab)",
        "priority": "优先级 (efficiency/speed/robustness)"
    }},
    "confidence": 0.0-1.0,
    "clarification_needed": ["需要用户澄清的问题"]
}}

参考知识:
{retrieved_knowledge}
```

#### 4.1.2 工作流配置生成提示词

```
基于以下设计意图，生成 HI-Photonics 工作流配置:

设计意图: {design_intent}

可用模型:
- TNN: 适合快速原型，一对一映射
- MDN: 处理一对多映射，提供不确定性
- CGAN: 生成多样化设计
- PINN: 物理约束，适合数据稀缺场景
- HiLab: VAE + 贝叶斯优化，高质量设计

请生成工作流配置 JSON:
{{
    "pipeline_name": "...",
    "challenge": "...",
    "model_config": {{...}},
    "training_config": {{...}},
    "optimization_config": {{...}},
    "rationale": "选择理由"
}}
```

#### 4.1.3 结果解释提示词

```
解释以下光子学逆向设计结果:

设计结果: {design_result}
仿真验证: {simulation_result}

请提供:
1. 性能指标解读 (目标达成情况)
2. 设计特征分析 (结构特点)
3. 潜在问题提示
4. 改进建议

输出格式:
- 使用通俗易懂的语言
- 对于专业术语提供简要解释
- 突出关键指标和风险点
```

### 4.2 提示词优化策略

| 策略 | 描述 | 示例 |
|------|------|------|
| **Few-shot Learning** | 提供示例引导输出格式 | 给出 2-3 个已解析的案例 |
| **Chain-of-Thought** | 引导逐步推理 | "首先分析器件类型，然后确定目标..." |
| **Self-Consistency** | 多次采样取共识 | 对同一输入生成多个解析结果 |
| **RAG 增强** | 检索相关领域知识 | 注入器件相关文档片段 |

---

## 5. 具体应用场景

### 5.1 场景一: 自然语言目标输入

**用户输入:**
> "我想设计一个工作在 1550nm 的光栅耦合器，效率最好能到 70% 以上，带宽大概 100nm 左右"

**LLM 处理流程:**

```python
# Step 1: 意图解析
intent = orchestrator.parse_design_intent(user_input)
# 输出:
# {
#     "device_type": "grating_coupler",
#     "target_specs": {
#         "wavelength": 1550,
#         "efficiency": ">70%",
#         "bandwidth": "~100nm"
#     },
#     "confidence": 0.92
# }

# Step 2: 工作流配置生成
config = orchestrator.generate_workflow_config(intent)

# Step 3: 确认对话
if intent.confidence < 0.8:
    clarifications = orchestrator.ask_clarification(intent)
    # "您希望侧重效率还是带宽？是否需要考虑制造约束？"
```

### 5.2 场景二: 智能模型选择

**系统提示词:**
```
你是光子学逆向设计专家。根据以下条件推荐最合适的模型:

器件类型: {device_type}
数据量: {data_availability}
设计复杂度: {complexity}
物理约束: {physics_constraints}
用户优先级: {priority}

模型能力对照表:
{model_capabilities}

请推荐模型并解释理由。
```

**输出示例:**
> "对于光栅耦合器设计，推荐使用 **HiLab** 框架:
> - VAE 可以学习复杂的设计空间分布
> - 贝叶斯优化能有效搜索高效率设计
> - 支持多目标优化 (效率 vs 带宽)
> 
> 如果您希望快速获得初步结果，也可以先尝试 **TNN** 进行快速原型设计。"

### 5.3 场景三: 多轮设计优化

```
用户: 设计一个光栅耦合器，效率 > 70%
系统: [执行设计] 结果: 效率 68%，带宽 80nm

用户: 效率不够高，能优化吗？
系统: 分析结果发现效率略低的原因可能是:
      1. 光栅周期可能需要调整
      2. 当前设计未充分利用刻蚀深度
      
      建议尝试:
      - 使用 HiLab 的贝叶斯优化继续搜索
      - 放宽制造约束以探索更大设计空间
      
      是否执行优化迭代？

用户: 好的，用 HiLab 优化
系统: [执行优化] 结果: 效率 72%，带宽 95nm
      已达到您的目标！
```

### 5.4 场景四: 结果报告生成

**LLM 生成的报告结构:**

```markdown
# 光栅耦合器设计报告

## 设计目标
- 中心波长: 1550 nm
- 目标效率: >70%
- 目标带宽: ~100 nm

## 设计结果
| 指标 | 目标 | 实际 | 达成率 |
|------|------|------|--------|
| 效率 | >70% | 72.3% | ✅ |
| 带宽 | ~100nm | 95nm | ✅ |
| 插入损耗 | - | 1.4 dB | - |

## 设计特征分析
设计呈现出典型的倾斜光栅结构:
- 光栅周期: 620 nm
- 占空比: 0.55
- 刻蚀深度: 70 nm

## 物理机制
入射光通过光栅的衍射效应被耦合进波导...
[自动生成的物理机制解释]

## 建议
1. 可进一步优化刻蚀深度以提升效率
2. 当前设计符合标准 CMOS 工艺要求
```

---

## 6. API 接口设计

### 6.1 LLM 服务 API

```python
# api/routers/llm.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List

router = APIRouter(prefix="/llm", tags=["llm"])

class ParseIntentRequest(BaseModel):
    user_input: str
    context: Optional[dict] = None

class ParseIntentResponse(BaseModel):
    intent: dict
    confidence: float
    clarification_questions: List[str]

class GenerateConfigRequest(BaseModel):
    intent: dict
    preferences: Optional[dict] = None

class GenerateConfigResponse(BaseModel):
    config: dict
    rationale: str
    alternatives: List[dict]

class ExplainResultsRequest(BaseModel):
    design_result: dict
    simulation_result: dict
    detail_level: str = "normal"  # brief/normal/detailed

class ExplainResultsResponse(BaseModel):
    summary: str
    metrics_analysis: dict
    suggestions: List[str]

@router.post("/parse-intent", response_model=ParseIntentResponse)
async def parse_design_intent(request: ParseIntentRequest):
    """解析用户自然语言输入为结构化设计意图"""
    pass

@router.post("/generate-config", response_model=GenerateConfigResponse)
async def generate_workflow_config(request: GenerateConfigRequest):
    """基于设计意图生成工作流配置"""
    pass

@router.post("/explain-results", response_model=ExplainResultsResponse)
async def explain_design_results(request: ExplainResultsRequest):
    """解释设计结果并生成报告"""
    pass

@router.post("/suggest-improvements")
async def suggest_improvements(results: dict):
    """基于结果提出改进建议"""
    pass
```

### 6.2 LLM 服务层

```python
# api/services/llm_service.py

from typing import Optional, List, Dict
import httpx
from core.config import settings

class LLMService:
    """LLM 服务封装"""
    
    def __init__(self, provider: str = "openai"):
        self.provider = provider
        self.api_key = settings.LLM_API_KEY
        self.model = settings.LLM_MODEL
        self.orchestrator = PromptOrchestrator()
        
    async def chat_completion(
        self,
        messages: List[Dict],
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> str:
        """调用 LLM API"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens
                }
            )
            return response.json()["choices"][0]["message"]["content"]
    
    async def parse_with_rag(
        self,
        user_input: str,
        knowledge_queries: List[str]
    ) -> Dict:
        """使用 RAG 增强的解析"""
        # 1. 检索相关知识
        knowledge = await self.retrieve_knowledge(knowledge_queries)
        
        # 2. 构建增强提示词
        prompt = self.orchestrator.build_intent_prompt(
            user_input, knowledge
        )
        
        # 3. 调用 LLM
        response = await self.chat_completion([
            {"role": "system", "content": INTENT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ])
        
        # 4. 解析响应
        return self.orchestrator.parse_intent_response(response)
```

---

## 7. 实施路线图

### Phase 1: 基础集成 (P0)

- [ ] 创建 `api/routers/llm.py` LLM 路由
- [ ] 实现 `PromptOrchestrator` 核心类
- [ ] 构建提示词模板库
- [ ] 实现意图解析 API
- [ ] 实现结果解释 API

### Phase 2: 知识增强 (P1)

- [ ] 构建光子学知识库
- [ ] 实现 RAG 检索模块
- [ ] 模型文档向量化存储
- [ ] 设计案例库构建

### Phase 3: 智能优化 (P1)

- [ ] 多轮对话上下文管理
- [ ] 自动参数调优建议
- [ ] 设计迭代策略生成
- [ ] 错误诊断和修复建议

### Phase 4: 深度集成 (P2)

- [ ] 前端 LLM 聊天界面
- [ ] 工作流可视化 + LLM 控制
- [ ] 自动报告生成
- [ ] 语音交互支持

---

## 8. 风险与对策

| 风险 | 影响 | 对策 |
|------|------|------|
| LLM 幻觉导致错误配置 | 高 | 严格输出格式校验 + 人工确认 |
| 领域知识不足 | 中 | RAG 增强 + 领域微调 |
| API 调用成本 | 中 | 缓存策略 + 本地模型备选 |
| 响应延迟 | 中 | 异步处理 + 流式输出 |
| 数据安全 | 高 | 敏感数据本地处理 + 私有部署选项 |

---

## 9. 成本估算

### API 调用成本 (以 OpenAI GPT-4 为例)

| 场景 | Token 消耗 | 单次成本 |
|------|-----------|----------|
| 意图解析 | ~1000 tokens | ~$0.03 |
| 配置生成 | ~2000 tokens | ~$0.06 |
| 结果解释 | ~1500 tokens | ~$0.045 |
| 单次完整流程 | ~5000 tokens | ~$0.15 |

### 成本优化策略

1. **使用更经济的模型** (GPT-3.5, Claude Instant)
2. **本地部署开源模型** (Llama 3, Qwen)
3. **提示词缓存** 减少重复调用
4. **批量处理** 合并请求

---

## 10. 推荐技术栈

### LLM 提供商选择

| 提供商 | 模型 | 优势 | 推荐场景 |
|--------|------|------|----------|
| OpenAI | GPT-4/GPT-4o | 能力最强，API 稳定 | 生产环境 |
| Anthropic | Claude 3.5 | 长上下文，推理强 | 复杂分析 |
| 阿里云 | Qwen | 中文支持好，性价比高 | 国内部署 |
| 本地 | Llama 3/Ollama | 无成本，数据安全 | 开发/敏感场景 |

### 推荐库

```toml
[project.dependencies]
openai = ">=1.0.0"
anthropic = ">=0.18.0"
langchain = ">=0.1.0"
langchain-community = ">=0.0.20"
chromadb = ">=0.4.0"          # 向量数据库
tiktoken = ">=0.5.0"          # Token 计数
```

---

## 11. 结论

将 LLM 集成到 HI-Photonics 逆向设计框架是**可行且高价值**的。通过提示词工程和 RAG 技术，可以显著降低用户使用门槛，提升设计效率。

**核心价值:**
1. **降低专业门槛** - 自然语言交互取代复杂配置
2. **提升设计效率** - 智能推荐和自动化工作流
3. **增强结果可用性** - 自动解释和改进建议
4. **知识沉淀** - RAG 知识库持续积累

**建议优先实施:**
1. 意图解析 + 工作流配置生成 (核心价值)
2. 结果解释报告生成 (用户最需要)
3. RAG 知识库构建 (长期价值)

---

## 附录 A: 提示词模板示例

### A.1 完整意图解析模板

```python
INTENT_SYSTEM_PROMPT = """
你是一个专业的光子学逆向设计助手，帮助用户将自然语言描述转换为结构化的设计意图。

你的职责:
1. 准确识别用户想要设计的器件类型
2. 提取性能目标参数
3. 识别设计约束条件
4. 评估输入完整性，必要时提出澄清问题

输出要求:
- 使用 JSON 格式
- 数值使用标准单位 (nm, %, dB 等)
- 置信度反映信息完整性
- 对不确定的信息提出澄清问题
"""

INTENT_USER_PROMPT = """
用户输入: {user_input}

参考知识:
{knowledge}

请解析为结构化设计意图 JSON。
"""
```

### A.2 模型选择提示词

```python
MODEL_SELECTION_PROMPT = """
作为光子学设计专家，根据以下条件推荐最适合的逆向设计模型:

设计需求:
- 器件类型: {device_type}
- 目标性能: {target_specs}
- 数据可用性: {data_availability}
- 计算资源: {compute_resources}
- 优先级: {priority}

可用模型及其特点:
1. TNN (串联神经网络)
   - 优点: 训练快，推理快，适合快速原型
   - 缺点: 一对一映射，无法处理多解问题
   - 适用: 设计空间简单，有充足训练数据

2. MDN (混合密度网络)
   - 优点: 处理一对多映射，提供不确定性估计
   - 缺点: 训练相对复杂
   - 适用: 设计空间有多个局部最优解

3. CGAN (条件生成对抗网络)
   - 优点: 生成多样化设计，可以探索设计空间
   - 缺点: 训练不稳定，需要调参经验
   - 适用: 需要多样化设计选项

4. PINN (物理信息神经网络)
   - 优点: 物理约束保证，数据需求少
   - 缺点: 训练慢，需要物理方程
   - 适用: 数据稀缺，物理规律明确

5. HiLab (VAE + 贝叶斯优化)
   - 优点: 高质量设计，支持多目标优化
   - 缺点: 计算成本高
   - 适用: 追求最优设计

请推荐最合适的模型并解释理由，格式如下:
{{
    "recommended_model": "...",
    "rationale": "...",
    "alternatives": ["...", "..."],
    "config_suggestions": {{...}}
}}
"""
```

---

## 附录 B: 代码实现参考

### B.1 PromptOrchestrator 实现框架

```python
# api/services/prompt_orchestrator.py

import json
from typing import Dict, List, Optional
from dataclasses import dataclass
import re

@dataclass
class DesignIntent:
    device_type: str
    target_specs: Dict
    constraints: Dict
    preferences: Dict
    confidence: float
    clarification_needed: List[str]

class PromptOrchestrator:
    """提示词编排器"""
    
    def __init__(self, llm_service, knowledge_base):
        self.llm = llm_service
        self.kb = knowledge_base
        self.templates = self._load_templates()
    
    def _load_templates(self) -> Dict:
        """加载提示词模板"""
        return {
            "intent_parse": INTENT_SYSTEM_PROMPT,
            "config_gen": CONFIG_GEN_PROMPT,
            "result_explain": RESULT_EXPLAIN_PROMPT,
            "model_select": MODEL_SELECTION_PROMPT
        }
    
    async def parse_design_intent(
        self, 
        user_input: str,
        context: Optional[Dict] = None
    ) -> DesignIntent:
        """解析设计意图"""
        
        # 1. 检索相关知识
        queries = self._extract_keywords(user_input)
        knowledge = await self.kb.retrieve(queries)
        
        # 2. 构建提示词
        prompt = self.templates["intent_parse"].format(
            user_input=user_input,
            knowledge=knowledge
        )
        
        # 3. 调用 LLM
        response = await self.llm.chat_completion([
            {"role": "system", "content": INTENT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ])
        
        # 4. 解析响应
        intent_dict = self._parse_json_response(response)
        
        return DesignIntent(
            device_type=intent_dict.get("device_type"),
            target_specs=intent_dict.get("target_specs", {}),
            constraints=intent_dict.get("constraints", {}),
            preferences=intent_dict.get("preferences", {}),
            confidence=intent_dict.get("confidence", 0.0),
            clarification_needed=intent_dict.get("clarification_needed", [])
        )
    
    async def generate_workflow_config(
        self,
        intent: DesignIntent,
        preferences: Optional[Dict] = None
    ) -> Dict:
        """生成工作流配置"""
        
        prompt = self.templates["config_gen"].format(
            device_type=intent.device_type,
            target_specs=json.dumps(intent.target_specs),
            constraints=json.dumps(intent.constraints),
            preferences=json.dumps(preferences or intent.preferences)
        )
        
        response = await self.llm.chat_completion([
            {"role": "system", "content": CONFIG_GEN_PROMPT},
            {"role": "user", "content": prompt}
        ])
        
        return self._parse_json_response(response)
    
    async def explain_results(
        self,
        design_result: Dict,
        simulation_result: Dict,
        detail_level: str = "normal"
    ) -> Dict:
        """解释设计结果"""
        
        prompt = self.templates["result_explain"].format(
            design_result=json.dumps(design_result, indent=2),
            simulation_result=json.dumps(simulation_result, indent=2),
            detail_level=detail_level
        )
        
        response = await self.llm.chat_completion([
            {"role": "system", "content": RESULT_EXPLAIN_PROMPT},
            {"role": "user", "content": prompt}
        ])
        
        return {
            "explanation": response,
            "summary": self._extract_summary(response)
        }
    
    def _parse_json_response(self, response: str) -> Dict:
        """从 LLM 响应中提取 JSON"""
        # 尝试找到 JSON 块
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        # 尝试解析整个响应
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {"raw_response": response}
    
    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词用于知识检索"""
        keywords = []
        device_keywords = [
            "grating coupler", "metagrating", "wavelength demux",
            "waveguide", "splitter", "resonator"
        ]
        for kw in device_keywords:
            if kw.lower() in text.lower():
                keywords.append(kw)
        return keywords
    
    def _extract_summary(self, explanation: str) -> str:
        """从解释中提取摘要"""
        # 简单实现: 取前 200 字符
        return explanation[:200] + "..." if len(explanation) > 200 else explanation
```

---

## 变更历史

| 日期 | 版本 | 变更内容 |
|------|------|----------|
| 2026-03-22 | 0.1.0 | 初始版本 |

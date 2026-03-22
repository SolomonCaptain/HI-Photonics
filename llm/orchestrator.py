"""
PromptOrchestrator - 提示词编排器

管理和编排提示词模板，协调 LLM 和 RAG 服务。
"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field, asdict
from pathlib import Path
import json
import re

from llm.config import LLMAssistantConfig
from llm.llm_client import LLMClient, ChatMessage, ChatResponse
from llm.embedding_client import EmbeddingClient
from llm.qdrant_service import QdrantService
from llm.rag_service import RAGService, RetrievedContext


# ============================================
# 数据模型
# ============================================

@dataclass
class DesignIntent:
    """设计意图"""
    device_type: str = ""
    target_specs: Dict[str, Any] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)
    preferences: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    clarification_needed: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class WorkflowSuggestion:
    """工作流建议"""
    pipeline_name: str
    challenge: str
    model_config: Dict[str, Any] = field(default_factory=dict)
    training_config: Dict[str, Any] = field(default_factory=dict)
    optimization_config: Dict[str, Any] = field(default_factory=dict)
    rationale: str = ""
    alternatives: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class DesignReport:
    """设计报告"""
    summary: str
    metrics_analysis: Dict[str, Any] = field(default_factory=dict)
    design_features: List[str] = field(default_factory=list)
    potential_issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    raw_response: str = ""


# ============================================
# 系统提示词
# ============================================

INTENT_SYSTEM_PROMPT = """你是一个专业的光子学逆向设计助手，帮助用户将自然语言描述转换为结构化的设计意图。

你的职责:
1. 准确识别用户想要设计的器件类型
2. 提取性能目标参数
3. 识别设计约束条件
4. 评估输入完整性，必要时提出澄清问题

支持的器件类型:
- grating_coupler: 光栅耦合器
- metagrating: 超构光栅
- wavelength_demux: 波分复用器

输出格式要求:
```json
{
    "device_type": "器件类型",
    "target_specs": {
        "wavelength": "目标波长(nm)",
        "efficiency": "目标效率(%)",
        "bandwidth": "带宽要求(nm)",
        "other_metrics": "其他指标"
    },
    "constraints": {
        "fabrication": "制造约束",
        "size": "尺寸约束",
        "materials": "材料约束"
    },
    "preferences": {
        "model_preference": "模型偏好(tnn/mdn/cgan/pinn/hilab)",
        "priority": "优先级(efficiency/speed/robustness)"
    },
    "confidence": 0.0-1.0,
    "clarification_needed": ["需要用户澄清的问题"]
}
```

请严格按照 JSON 格式输出，不要添加其他内容。"""

WORKFLOW_SYSTEM_PROMPT = """你是光子学逆向设计专家，根据设计意图推荐最合适的工作流配置。

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

输出格式要求:
```json
{
    "pipeline_name": "管道名称",
    "challenge": "设计挑战名称",
    "model_config": {
        "type": "模型类型",
        "hidden_dims": [256, 512, 256],
        "latent_dim": 64
    },
    "training_config": {
        "epochs": 100,
        "batch_size": 32,
        "learning_rate": 0.001
    },
    "optimization_config": {
        "num_iterations": 50,
        "method": "bayesian"
    },
    "rationale": "选择理由",
    "alternatives": ["备选方案"]
}
```"""

EXPLAIN_SYSTEM_PROMPT = """你是一个光子学设计结果解释专家，帮助用户理解设计结果。

请分析以下内容:
1. 性能指标解读 - 目标达成情况如何
2. 设计特征分析 - 结构有什么特点
3. 潜在问题 - 可能存在的风险
4. 改进建议 - 如何进一步优化

输出要求:
- 使用通俗易懂的语言
- 对于专业术语提供简要解释
- 突出关键指标和风险点
- 使用 Markdown 格式"""


# ============================================
# PromptOrchestrator
# ============================================

class PromptOrchestrator:
    """
    提示词编排器
    
    管理提示词模板，协调 LLM 和 RAG 服务。
    
    示例:
        orchestrator = PromptOrchestrator(config)
        intent = await orchestrator.parse_intent("设计一个1550nm的光栅耦合器")
    """
    
    def __init__(
        self,
        config: Optional[LLMAssistantConfig] = None,
        llm_client: Optional[LLMClient] = None,
        rag_service: Optional[RAGService] = None
    ):
        self.config = config or LLMAssistantConfig()
        self.llm = llm_client or LLMClient(self.config.llm)
        self.rag = rag_service
        
        self._prompts_dir = self.config.prompts_dir
    
    async def _ensure_rag(self) -> RAGService:
        """确保 RAG 服务可用"""
        if self.rag is None:
            self.rag = RAGService(self.config)
        return self.rag
    
    async def parse_intent(
        self,
        user_input: str,
        use_rag: bool = True
    ) -> DesignIntent:
        """
        解析用户设计意图
        
        Args:
            user_input: 用户自然语言输入
            use_rag: 是否使用 RAG 增强
            
        Returns:
            DesignIntent 对象
        """
        # 构建提示词
        prompt = f"用户输入: {user_input}\n\n请分析用户的设计意图并输出 JSON。"
        
        # RAG 增强
        if use_rag:
            rag = await self._ensure_rag()
            context = await rag.retrieve(user_input, top_k=3)
            if context.formatted_context:
                prompt = f"参考知识:\n{context.formatted_context}\n\n{prompt}"
        
        # 调用 LLM
        response = await self.llm.chat_with_system(
            user_message=prompt,
            system_prompt=INTENT_SYSTEM_PROMPT
        )
        
        # 解析响应
        return self._parse_intent_response(response.content)
    
    def _parse_intent_response(self, response: str) -> DesignIntent:
        """解析意图响应"""
        # 尝试提取 JSON
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_match = re.search(r'\{[\s\S]*\}', response)
            json_str = json_match.group(0) if json_match else "{}"
        
        try:
            data = json.loads(json_str)
            return DesignIntent(
                device_type=data.get("device_type", ""),
                target_specs=data.get("target_specs", {}),
                constraints=data.get("constraints", {}),
                preferences=data.get("preferences", {}),
                confidence=data.get("confidence", 0.0),
                clarification_needed=data.get("clarification_needed", [])
            )
        except json.JSONDecodeError:
            return DesignIntent(confidence=0.0, clarification_needed=["无法解析输入，请重新描述"])
    
    async def generate_workflow_config(
        self,
        intent: DesignIntent,
        use_rag: bool = True
    ) -> WorkflowSuggestion:
        """
        生成工作流配置建议
        
        Args:
            intent: 设计意图
            use_rag: 是否使用 RAG 增强
            
        Returns:
            WorkflowSuggestion 对象
        """
        prompt = f"""设计意图:
- 器件类型: {intent.device_type}
- 目标规格: {json.dumps(intent.target_specs, ensure_ascii=False)}
- 约束条件: {json.dumps(intent.constraints, ensure_ascii=False)}
- 用户偏好: {json.dumps(intent.preferences, ensure_ascii=False)}

请推荐最合适的工作流配置并输出 JSON。"""
        
        # RAG 增强
        if use_rag:
            rag = await self._ensure_rag()
            context = await rag.retrieve(
                f"{intent.device_type} {intent.preferences.get('model_preference', '')}",
                top_k=3
            )
            if context.formatted_context:
                prompt = f"参考知识:\n{context.formatted_context}\n\n{prompt}"
        
        response = await self.llm.chat_with_system(
            user_message=prompt,
            system_prompt=WORKFLOW_SYSTEM_PROMPT
        )
        
        return self._parse_workflow_response(response.content)
    
    def _parse_workflow_response(self, response: str) -> WorkflowSuggestion:
        """解析工作流响应"""
        json_match = re.search(r'```json\s*([\s\S]*?)\s*```', response)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_match = re.search(r'\{[\s\S]*\}', response)
            json_str = json_match.group(0) if json_match else "{}"
        
        try:
            data = json.loads(json_str)
            return WorkflowSuggestion(
                pipeline_name=data.get("pipeline_name", ""),
                challenge=data.get("challenge", ""),
                model_config=data.get("model_config", {}),
                training_config=data.get("training_config", {}),
                optimization_config=data.get("optimization_config", {}),
                rationale=data.get("rationale", ""),
                alternatives=data.get("alternatives", [])
            )
        except json.JSONDecodeError:
            return WorkflowSuggestion(pipeline_name="", rationale="无法解析配置建议")
    
    async def explain_results(
        self,
        design_result: Dict[str, Any],
        simulation_result: Dict[str, Any],
        detail_level: str = "normal"
    ) -> DesignReport:
        """
        解释设计结果
        
        Args:
            design_result: 设计结果
            simulation_result: 仿真结果
            detail_level: 详细程度 (brief/normal/detailed)
            
        Returns:
            DesignReport 对象
        """
        prompt = f"""设计结果:
{json.dumps(design_result, ensure_ascii=False, indent=2)}

仿真验证结果:
{json.dumps(simulation_result, ensure_ascii=False, indent=2)}

请解释这些结果。详细程度: {detail_level}"""
        
        response = await self.llm.chat_with_system(
            user_message=prompt,
            system_prompt=EXPLAIN_SYSTEM_PROMPT
        )
        
        return self._parse_report_response(response.content)
    
    def _parse_report_response(self, response: str) -> DesignReport:
        """解析报告响应"""
        # 提取各部分内容
        summary = ""
        metrics_analysis = {}
        design_features = []
        potential_issues = []
        suggestions = []
        
        # 简单解析
        lines = response.split("\n")
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            if "性能指标" in line or "指标解读" in line:
                current_section = "metrics"
            elif "设计特征" in line:
                current_section = "features"
            elif "潜在问题" in line or "风险" in line:
                current_section = "issues"
            elif "改进建议" in line or "建议" in line:
                current_section = "suggestions"
            elif line.startswith(("-", "•", "*", "1.", "2.", "3.", "4.", "5.")):
                item = line.lstrip("-•*123456789. ").strip()
                if current_section == "features":
                    design_features.append(item)
                elif current_section == "issues":
                    potential_issues.append(item)
                elif current_section == "suggestions":
                    suggestions.append(item)
            elif not summary and len(line) > 20:
                summary = line
        
        if not summary:
            summary = response[:200] + "..." if len(response) > 200 else response
        
        return DesignReport(
            summary=summary,
            metrics_analysis=metrics_analysis,
            design_features=design_features,
            potential_issues=potential_issues,
            suggestions=suggestions,
            raw_response=response
        )
    
    async def chat(
        self,
        message: str,
        history: Optional[List[Dict]] = None,
        use_rag: bool = True
    ) -> str:
        """
        对话接口
        
        Args:
            message: 用户消息
            history: 对话历史
            use_rag: 是否使用 RAG 增强
            
        Returns:
            LLM 响应
        """
        messages = history or []
        messages.append({"role": "user", "content": message})
        
        # RAG 增强
        if use_rag:
            rag = await self._ensure_rag()
            context = await rag.retrieve(message, top_k=3)
            if context.formatted_context:
                system_prompt = f"你是光子学逆向设计助手。以下是一些相关知识:\n\n{context.formatted_context}"
            else:
                system_prompt = "你是光子学逆向设计助手，帮助用户进行光子器件设计。"
        else:
            system_prompt = "你是光子学逆向设计助手，帮助用户进行光子器件设计。"
        
        response = await self.llm.chat_with_system(
            user_message=message,
            system_prompt=system_prompt
        )
        
        return response.content


# ============================================
# LLMAssistant - 统一入口
# ============================================

class LLMAssistant:
    """
    LLM 助手统一入口
    
    提供完整的 LLM 增强逆向设计辅助功能。
    
    示例:
        assistant = LLMAssistant()
        
        # 解析意图
        intent = await assistant.parse_intent("设计1550nm光栅耦合器")
        
        # 生成配置
        workflow = await assistant.generate_workflow(intent)
        
        # 解释结果
        report = await assistant.explain_results(design, simulation)
    """
    
    def __init__(self, config: Optional[LLMAssistantConfig] = None):
        self.config = config or LLMAssistantConfig()
        self._orchestrator: Optional[PromptOrchestrator] = None
        self._rag: Optional[RAGService] = None
    
    @property
    def orchestrator(self) -> PromptOrchestrator:
        if self._orchestrator is None:
            self._orchestrator = PromptOrchestrator(self.config)
        return self._orchestrator
    
    async def initialize(self, index_knowledge: bool = False) -> bool:
        """
        初始化助手
        
        Args:
            index_knowledge: 是否索引知识库
            
        Returns:
            是否成功
        """
        if index_knowledge:
            rag = RAGService(self.config)
            rag.initialize()
            
            knowledge_dir = self.config.knowledge_dir
            if knowledge_dir.exists():
                await rag.index_directory(knowledge_dir)
            
            self._rag = rag
        
        return True
    
    async def parse_intent(self, user_input: str) -> DesignIntent:
        """解析用户设计意图"""
        return await self.orchestrator.parse_intent(user_input)
    
    async def generate_workflow(self, intent: DesignIntent) -> WorkflowSuggestion:
        """生成工作流配置"""
        return await self.orchestrator.generate_workflow_config(intent)
    
    async def explain_results(
        self,
        design_result: Dict,
        simulation_result: Dict
    ) -> DesignReport:
        """解释设计结果"""
        return await self.orchestrator.explain_results(design_result, simulation_result)
    
    async def chat(self, message: str, history: Optional[List[Dict]] = None) -> str:
        """对话接口"""
        return await self.orchestrator.chat(message, history)
    
    async def close(self):
        """关闭连接"""
        if self._orchestrator and self._orchestrator.llm:
            await self._orchestrator.llm.close()
        if self._rag:
            await self._rag.close()
    
    async def __aenter__(self):
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

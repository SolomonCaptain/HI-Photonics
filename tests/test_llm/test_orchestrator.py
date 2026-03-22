"""
LLM 编排器测试
"""

import pytest
from pathlib import Path
import sys
from unittest.mock import MagicMock, AsyncMock, patch

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from llm.orchestrator import (
    PromptOrchestrator,
    DesignIntent,
    WorkflowSuggestion,
    DesignReport,
    LLMAssistant
)


class TestDesignIntent:
    """DesignIntent 测试"""
    
    def test_create_intent(self):
        """测试创建意图"""
        intent = DesignIntent(
            device_type="grating_coupler",
            target_specs={"wavelength": 1550, "efficiency": 0.8},
            constraints={"fabrication": "silicon_photonics"},
            preferences={"model_preference": "hilab"},
            confidence=0.9,
            clarification_needed=[]
        )
        
        assert intent.device_type == "grating_coupler"
        assert intent.target_specs["wavelength"] == 1550
        assert intent.constraints["fabrication"] == "silicon_photonics"
        assert intent.confidence == 0.9
    
    def test_to_dict(self):
        """测试转换为字典"""
        intent = DesignIntent(
            device_type="metagrating",
            target_specs={"wavelength": 1310},
            confidence=0.85
        )
        
        result = intent.to_dict()
        
        assert result["device_type"] == "metagrating"
        assert result["target_specs"]["wavelength"] == 1310
        assert result["confidence"] == 0.85


class TestWorkflowSuggestion:
    """WorkflowSuggestion 测试"""
    
    def test_create_suggestion(self):
        """测试创建建议"""
        suggestion = WorkflowSuggestion(
            pipeline_name="inverse_design",
            challenge="grating_coupler",
            model_config={"type": "hilab", "latent_dim": 64},
            training_config={"epochs": 100, "batch_size": 32},
            optimization_config={"num_iterations": 50},
            rationale="适合高质量设计",
            alternatives=["mdn", "tnn"]
        )
        
        assert suggestion.pipeline_name == "inverse_design"
        assert suggestion.challenge == "grating_coupler"
        assert suggestion.model_config["type"] == "hilab"
        assert len(suggestion.alternatives) == 2
    
    def test_to_dict(self):
        """测试转换为字典"""
        suggestion = WorkflowSuggestion(
            pipeline_name="test",
            challenge="test_challenge"
        )
        
        result = suggestion.to_dict()
        
        assert result["pipeline_name"] == "test"
        assert result["challenge"] == "test_challenge"


class TestDesignReport:
    """DesignReport 测试"""
    
    def test_create_report(self):
        """测试创建报告"""
        report = DesignReport(
            summary="设计已完成，效率达到 85%",
            metrics_analysis={"efficiency": "良好", "bandwidth": "符合要求"},
            design_features=["周期结构", "渐变光栅"],
            potential_issues=["边缘耦合效率较低"],
            suggestions=["增加周期数", "优化刻蚀深度"],
            raw_response="Full response..."
        )
        
        assert report.summary == "设计已完成，效率达到 85%"
        assert "efficiency" in report.metrics_analysis
        assert len(report.design_features) == 2
        assert len(report.potential_issues) == 1
        assert len(report.suggestions) == 2


class TestPromptOrchestrator:
    """PromptOrchestrator 测试"""
    
    def test_init(self):
        """测试初始化"""
        orchestrator = PromptOrchestrator()
        assert orchestrator is not None
        assert orchestrator.llm is not None
    
    def test_init_with_config(self):
        """测试使用配置初始化"""
        from llm.config import LLMAssistantConfig
        config = LLMAssistantConfig()
        orchestrator = PromptOrchestrator(config=config)
        
        assert orchestrator.config is not None
    
    def test_parse_intent_response(self):
        """测试解析意图响应"""
        orchestrator = PromptOrchestrator()
        
        # 测试 JSON 格式响应
        response = '''
        ```json
        {
            "device_type": "grating_coupler",
            "target_specs": {"wavelength": 1550},
            "constraints": {},
            "preferences": {},
            "confidence": 0.9,
            "clarification_needed": []
        }
        ```
        '''
        
        intent = orchestrator._parse_intent_response(response)
        
        assert intent.device_type == "grating_coupler"
        assert intent.target_specs["wavelength"] == 1550
        assert intent.confidence == 0.9
    
    def test_parse_intent_response_plain_json(self):
        """测试解析纯 JSON 响应"""
        orchestrator = PromptOrchestrator()
        
        response = '{"device_type": "metagrating", "target_specs": {}, "confidence": 0.8}'
        
        intent = orchestrator._parse_intent_response(response)
        
        assert intent.device_type == "metagrating"
        assert intent.confidence == 0.8
    
    def test_parse_intent_response_invalid(self):
        """测试解析无效响应"""
        orchestrator = PromptOrchestrator()
        
        response = "This is not a valid JSON response"
        
        intent = orchestrator._parse_intent_response(response)
        
        # 应该返回默认的 DesignIntent
        assert intent is not None
        assert intent.confidence == 0.0
    
    def test_parse_workflow_response(self):
        """测试解析工作流响应"""
        orchestrator = PromptOrchestrator()
        
        response = '''
        ```json
        {
            "pipeline_name": "inverse_design",
            "challenge": "grating_coupler",
            "model_config": {"type": "hilab"},
            "training_config": {"epochs": 100},
            "optimization_config": {},
            "rationale": "测试理由",
            "alternatives": ["mdn"]
        }
        ```
        '''
        
        suggestion = orchestrator._parse_workflow_response(response)
        
        assert suggestion.pipeline_name == "inverse_design"
        assert suggestion.challenge == "grating_coupler"
        assert suggestion.model_config["type"] == "hilab"
    
    def test_parse_report_response(self):
        """测试解析报告响应"""
        orchestrator = PromptOrchestrator()
        
        response = """
        设计已完成，效率达到预期目标。
        
        性能指标解读：
        - 效率: 85%，达到目标
        - 带宽: 100nm，符合要求
        
        设计特征：
        - 周期性结构
        - 渐变光栅设计
        
        潜在问题：
        - 边缘耦合效率略低
        
        改进建议：
        - 增加周期数
        - 优化刻蚀深度
        """
        
        report = orchestrator._parse_report_response(response)
        
        assert report.summary is not None
        assert len(report.design_features) > 0
        assert len(report.suggestions) > 0
    
    @pytest.mark.asyncio
    async def test_parse_intent_async(self):
        """测试异步解析意图"""
        from llm.config import LLMConfig, LLMAssistantConfig
        
        # 创建 mock LLM 客户端
        mock_llm = AsyncMock()
        mock_llm.chat_with_system = AsyncMock(return_value=MagicMock(
            content='{"device_type": "grating_coupler", "confidence": 0.9}'
        ))
        
        orchestrator = PromptOrchestrator(llm_client=mock_llm)
        
        intent = await orchestrator.parse_intent("设计一个光栅耦合器", use_rag=False)
        
        assert intent is not None
    
    @pytest.mark.asyncio
    async def test_generate_workflow_config_async(self):
        """测试异步生成工作流配置"""
        mock_llm = AsyncMock()
        mock_llm.chat_with_system = AsyncMock(return_value=MagicMock(
            content='{"pipeline_name": "test", "challenge": "test"}'
        ))
        
        orchestrator = PromptOrchestrator(llm_client=mock_llm)
        
        intent = DesignIntent(device_type="grating_coupler")
        suggestion = await orchestrator.generate_workflow_config(intent, use_rag=False)
        
        assert suggestion is not None


class TestLLMAssistant:
    """LLMAssistant 测试"""
    
    def test_init(self):
        """测试初始化"""
        assistant = LLMAssistant()
        assert assistant is not None
    
    def test_orchestrator_property(self):
        """测试 orchestrator 属性"""
        assistant = LLMAssistant()
        orchestrator = assistant.orchestrator
        
        assert orchestrator is not None
        assert isinstance(orchestrator, PromptOrchestrator)
    
    @pytest.mark.asyncio
    async def test_initialize(self):
        """测试初始化方法"""
        assistant = LLMAssistant()
        result = await assistant.initialize(index_knowledge=False)
        
        assert result is True
    
    @pytest.mark.asyncio
    async def test_context_manager(self):
        """测试上下文管理器"""
        async with LLMAssistant() as assistant:
            assert assistant is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "asyncio"])

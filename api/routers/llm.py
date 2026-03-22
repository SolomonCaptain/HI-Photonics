"""
LLM API 路由

提供 LLM 助手相关的 API 端点
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import asyncio
import json

from llm.orchestrator import LLMAssistant, DesignIntent, WorkflowSuggestion, DesignReport
from llm.config import LLMAssistantConfig

router = APIRouter(prefix="/llm", tags=["LLM"])

# 全局 LLM 助手实例
_assistant: Optional[LLMAssistant] = None


def get_assistant() -> LLMAssistant:
    """获取 LLM 助手实例"""
    global _assistant
    if _assistant is None:
        config = LLMAssistantConfig()
        _assistant = LLMAssistant(config)
    return _assistant


# ===== 请求/响应模型 =====

class ChatMessage(BaseModel):
    """聊天消息"""
    role: str  # 'user' | 'assistant'
    content: str


class ChatRequest(BaseModel):
    """聊天请求"""
    message: str
    history: Optional[List[ChatMessage]] = None
    use_rag: bool = True


class ChatResponse(BaseModel):
    """聊天响应"""
    response: str
    success: bool = True
    error: Optional[str] = None


class ParseIntentRequest(BaseModel):
    """解析意图请求"""
    user_input: str
    use_rag: bool = True


class IntentResponse(BaseModel):
    """意图响应"""
    device_type: str
    target_specs: Dict[str, Any]
    constraints: Dict[str, Any]
    preferences: Dict[str, Any]
    confidence: float
    clarification_needed: List[str]
    success: bool = True
    error: Optional[str] = None


class GenerateWorkflowRequest(BaseModel):
    """生成工作流请求"""
    intent: IntentResponse
    use_rag: bool = True


class WorkflowResponse(BaseModel):
    """工作流响应"""
    pipeline_name: str
    challenge: str
    model_configuration: Dict[str, Any]
    training_config: Dict[str, Any]
    optimization_config: Dict[str, Any]
    rationale: str
    alternatives: List[str]
    success: bool = True
    error: Optional[str] = None


class ExplainResultsRequest(BaseModel):
    """解释结果请求"""
    design_result: Dict[str, Any]
    simulation_result: Dict[str, Any]
    detail_level: str = "normal"


class ReportResponse(BaseModel):
    """报告响应"""
    summary: str
    metrics_analysis: Dict[str, Any]
    design_features: List[str]
    potential_issues: List[str]
    suggestions: List[str]
    raw_response: str
    success: bool = True
    error: Optional[str] = None


# ===== API 端点 =====

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    与 LLM 助手对话
    
    支持多轮对话，可启用 RAG 知识增强
    """
    try:
        assistant = get_assistant()
        history = None
        if request.history:
            history = [{"role": m.role, "content": m.content} for m in request.history]
        
        response = await assistant.chat(
            message=request.message,
            history=history
        )
        
        return ChatResponse(response=response, success=True)
    except Exception as e:
        return ChatResponse(response="", success=False, error=str(e))


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    流式聊天接口
    
    返回 SSE 流式响应
    """
    async def generate():
        try:
            assistant = get_assistant()
            history = None
            if request.history:
                history = [{"role": m.role, "content": m.content} for m in request.history]
            
            # 获取 LLM 客户端进行流式调用
            from llm.config import LLMConfig
            from llm.llm_client import LLMClient
            
            config = LLMConfig()
            client = LLMClient(config)
            
            messages = history or []
            messages.append({"role": "user", "content": request.message})
            
            # 调用流式 API
            async for chunk in client.chat_stream(messages):
                yield f"data: {json.dumps({'content': chunk})}\n\n"
            
            yield "data: [DONE]\n\n"
            await client.close()
            
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )


@router.post("/parse-intent", response_model=IntentResponse)
async def parse_intent(request: ParseIntentRequest):
    """
    解析用户设计意图
    
    将自然语言描述转换为结构化的设计意图
    """
    try:
        assistant = get_assistant()
        intent = await assistant.parse_intent(request.user_input)
        
        return IntentResponse(
            device_type=intent.device_type,
            target_specs=intent.target_specs,
            constraints=intent.constraints,
            preferences=intent.preferences,
            confidence=intent.confidence,
            clarification_needed=intent.clarification_needed,
            success=True
        )
    except Exception as e:
        return IntentResponse(
            device_type="",
            target_specs={},
            constraints={},
            preferences={},
            confidence=0.0,
            clarification_needed=[],
            success=False,
            error=str(e)
        )


@router.post("/generate-workflow", response_model=WorkflowResponse)
async def generate_workflow(request: GenerateWorkflowRequest):
    """
    生成工作流配置
    
    根据设计意图生成推荐的工作流配置
    """
    try:
        assistant = get_assistant()
        
        # 转换为 DesignIntent
        intent = DesignIntent(
            device_type=request.intent.device_type,
            target_specs=request.intent.target_specs,
            constraints=request.intent.constraints,
            preferences=request.intent.preferences,
            confidence=request.intent.confidence,
            clarification_needed=request.intent.clarification_needed
        )
        
        workflow = await assistant.generate_workflow(intent)
        
        return WorkflowResponse(
            pipeline_name=workflow.pipeline_name,
            challenge=workflow.challenge,
            model_config=workflow.model_config,
            training_config=workflow.training_config,
            optimization_config=workflow.optimization_config,
            rationale=workflow.rationale,
            alternatives=workflow.alternatives,
            success=True
        )
    except Exception as e:
        return WorkflowResponse(
            pipeline_name="",
            challenge="",
            model_config={},
            training_config={},
            optimization_config={},
            rationale="",
            alternatives=[],
            success=False,
            error=str(e)
        )


@router.post("/explain-results", response_model=ReportResponse)
async def explain_results(request: ExplainResultsRequest):
    """
    解释设计结果
    
    分析设计结果和仿真验证结果，生成报告
    """
    try:
        assistant = get_assistant()
        report = await assistant.explain_results(
            design_result=request.design_result,
            simulation_result=request.simulation_result,
            detail_level=request.detail_level
        )
        
        return ReportResponse(
            summary=report.summary,
            metrics_analysis=report.metrics_analysis,
            design_features=report.design_features,
            potential_issues=report.potential_issues,
            suggestions=report.suggestions,
            raw_response=report.raw_response,
            success=True
        )
    except Exception as e:
        return ReportResponse(
            summary="",
            metrics_analysis={},
            design_features=[],
            potential_issues=[],
            suggestions=[],
            raw_response="",
            success=False,
            error=str(e)
        )


@router.get("/health")
async def llm_health():
    """LLM 服务健康检查"""
    try:
        assistant = get_assistant()
        return {
            "status": "healthy",
            "assistant_initialized": assistant._orchestrator is not None
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

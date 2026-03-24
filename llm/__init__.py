"""
HI-Photonics LLM 助手模块

提供基于 LLM 的智能逆向设计辅助功能：
- 自然语言设计意图解析
- 工作流自动配置
- 结果智能解释
- RAG 知识检索增强

使用方式:
    from llm import LLMAssistant, get_config
    
    config = get_config()
    assistant = LLMAssistant(config)
    
    # 解析用户意图
    intent = await assistant.parse_intent("设计一个1550nm的光栅耦合器")
    
    # 生成工作流配置
    workflow = await assistant.generate_workflow(intent)
"""

from llm.config import (
    LLMConfig,
    EmbeddingConfig,
    QdrantConfig,
    ChromaConfig,
    RAGConfig,
    LLMAssistantConfig,
    VectorDBType,
    get_config,
    reload_config,
)

from llm.vector_db_base import (
    VectorDBBase,
    KnowledgeDocument,
    SearchResult,
)

from llm.llm_client import LLMClient
from llm.embedding_client import EmbeddingClient
from llm.qdrant_service import QdrantService
from llm.chroma_service import ChromaService
from llm.rag_service import RAGService
from llm.orchestrator import PromptOrchestrator, LLMAssistant

__all__ = [
    # 配置
    "LLMConfig",
    "EmbeddingConfig",
    "QdrantConfig",
    "ChromaConfig",
    "RAGConfig",
    "LLMAssistantConfig",
    "VectorDBType",
    "get_config",
    "reload_config",
    # 向量数据库基类和数据类型
    "VectorDBBase",
    "KnowledgeDocument",
    "SearchResult",
    # 客户端
    "LLMClient",
    "EmbeddingClient",
    # 向量数据库服务
    "QdrantService",
    "ChromaService",
    "RAGService",
    # 核心服务
    "PromptOrchestrator",
    "LLMAssistant",
]

__version__ = "0.1.0"

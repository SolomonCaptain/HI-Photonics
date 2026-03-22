"""
LLM 助手配置模块

配置项通过 .env 文件管理，支持：
- 百炼云 LLM (qwen-plus)
- 硅基流动 Embedding (Qwen3-Embedding-8B)
- Qdrant 向量数据库
"""

from dataclasses import dataclass, field
from typing import Optional
import os
from pathlib import Path

# 尝试加载 .env 文件
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass


@dataclass
class LLMConfig:
    """LLM 配置 - 百炼云 qwen-plus"""
    base_url: str = field(default_factory=lambda: os.getenv(
        "LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    ))
    api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY", ""))
    model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "qwen-plus"))
    
    # 生成参数
    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: float = 0.9
    
    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)


@dataclass
class EmbeddingConfig:
    """Embedding 配置 - 硅基流动 Qwen3-Embedding-8B"""
    base_url: str = field(default_factory=lambda: os.getenv(
        "EMBEDDING_BASE_URL", "https://api.siliconflow.cn/v1"
    ))
    api_key: str = field(default_factory=lambda: os.getenv("EMBEDDING_API_KEY", ""))
    model: str = field(default_factory=lambda: os.getenv(
        "EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-8B"
    ))
    dimension: int = field(default_factory=lambda: int(os.getenv(
        "EMBEDDING_DIMENSION", "4096"
    )))
    
    # 批处理配置
    batch_size: int = 32
    
    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)


@dataclass
class QdrantConfig:
    """Qdrant 向量数据库配置"""
    host: str = field(default_factory=lambda: os.getenv("QDRANT_HOST", "localhost"))
    port: int = field(default_factory=lambda: int(os.getenv("QDRANT_PORT", "6333")))
    api_key: Optional[str] = field(default_factory=lambda: os.getenv("QDRANT_API_KEY") or None)
    collection: str = field(default_factory=lambda: os.getenv(
        "QDRANT_COLLECTION", "hi_photonics_knowledge"
    ))
    
    # 向量配置
    vector_size: int = 4096  # 与 Embedding 维度一致
    distance_metric: str = "Cosine"
    
    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"
    
    @property
    def grpc_url(self) -> str:
        return f"{self.host}:{self.port + 1}"


@dataclass
class RAGConfig:
    """RAG 检索配置"""
    # 检索参数
    top_k: int = 5
    score_threshold: float = 0.7
    
    # 分块参数
    chunk_size: int = 500
    chunk_overlap: int = 50


@dataclass
class LLMAssistantConfig:
    """LLM 助手完整配置"""
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    
    # 知识库路径
    knowledge_dir: Path = field(default_factory=lambda: Path(__file__).parent / "knowledge")
    prompts_dir: Path = field(default_factory=lambda: Path(__file__).parent / "prompts")
    
    @classmethod
    def from_env(cls) -> "LLMAssistantConfig":
        """从环境变量创建配置"""
        return cls()
    
    def validate(self) -> bool:
        """验证配置完整性"""
        errors = []
        
        if not self.llm.is_configured:
            errors.append("LLM API Key 未配置 (LLM_API_KEY)")
        
        if not self.embedding.is_configured:
            errors.append("Embedding API Key 未配置 (EMBEDDING_API_KEY)")
        
        if errors:
            print("配置警告:")
            for error in errors:
                print(f"  - {error}")
            return False
        
        return True


# 全局配置实例
_config: Optional[LLMAssistantConfig] = None


def get_config() -> LLMAssistantConfig:
    """获取全局配置实例"""
    global _config
    if _config is None:
        _config = LLMAssistantConfig.from_env()
    return _config


def reload_config() -> LLMAssistantConfig:
    """重新加载配置"""
    global _config
    _config = LLMAssistantConfig.from_env()
    return _config

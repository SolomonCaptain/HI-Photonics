"""
LLM 配置模块测试
"""

import pytest
import os
from pathlib import Path
import sys

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from llm.config import (
    LLMConfig,
    EmbeddingConfig,
    QdrantConfig,
    RAGConfig,
    LLMAssistantConfig,
    get_config,
    reload_config
)


class TestLLMConfig:
    """LLM 配置测试"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = LLMConfig()
        
        assert config.base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"
        assert config.model == "qwen-plus"
        assert config.temperature == 0.7
        assert config.max_tokens == 4096
        assert config.top_p == 0.9
    
    def test_is_configured_without_api_key(self):
        """测试未配置 API Key 的情况"""
        config = LLMConfig(api_key="")
        assert config.is_configured is False
    
    def test_is_configured_with_api_key(self):
        """测试已配置 API Key 的情况"""
        config = LLMConfig(api_key="test_api_key")
        assert config.is_configured is True
    
    def test_custom_config(self):
        """测试自定义配置"""
        config = LLMConfig(
            base_url="https://custom.api.com/v1",
            api_key="custom_key",
            model="custom-model",
            temperature=0.5,
            max_tokens=2048
        )
        
        assert config.base_url == "https://custom.api.com/v1"
        assert config.api_key == "custom_key"
        assert config.model == "custom-model"
        assert config.temperature == 0.5
        assert config.max_tokens == 2048


class TestEmbeddingConfig:
    """Embedding 配置测试"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = EmbeddingConfig()
        
        assert config.base_url == "https://api.siliconflow.cn/v1"
        assert config.model == "Qwen/Qwen3-Embedding-8B"
        assert config.dimension == 4096
        assert config.batch_size == 32
    
    def test_is_configured(self):
        """测试配置状态"""
        config_no_key = EmbeddingConfig(api_key="")
        assert config_no_key.is_configured is False
        
        config_with_key = EmbeddingConfig(api_key="test_key")
        assert config_with_key.is_configured is True


class TestQdrantConfig:
    """Qdrant 配置测试"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = QdrantConfig()
        
        # 这些值可能受环境变量影响，只检查类型
        assert isinstance(config.host, str)
        assert isinstance(config.port, int)
        assert isinstance(config.collection, str)
        assert config.vector_size == 4096
        assert config.distance_metric == "Cosine"
    
    def test_url_property(self):
        """测试 URL 属性"""
        config = QdrantConfig(host="192.168.1.1", port=6333)
        assert config.url == "http://192.168.1.1:6333"
    
    def test_grpc_url_property(self):
        """测试 gRPC URL 属性"""
        config = QdrantConfig(host="localhost", port=6333)
        assert config.grpc_url == "localhost:6334"


class TestRAGConfig:
    """RAG 配置测试"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = RAGConfig()
        
        assert config.top_k == 5
        assert config.score_threshold == 0.7
        assert config.chunk_size == 500
        assert config.chunk_overlap == 50


class TestLLMAssistantConfig:
    """LLM 助手完整配置测试"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = LLMAssistantConfig()
        
        assert isinstance(config.llm, LLMConfig)
        assert isinstance(config.embedding, EmbeddingConfig)
        assert isinstance(config.qdrant, QdrantConfig)
        assert isinstance(config.rag, RAGConfig)
    
    def test_from_env(self):
        """测试从环境变量创建"""
        config = LLMAssistantConfig.from_env()
        assert isinstance(config, LLMAssistantConfig)
    
    def test_validate_without_keys(self):
        """测试无 API Key 时的验证"""
        config = LLMAssistantConfig(
            llm=LLMConfig(api_key=""),
            embedding=EmbeddingConfig(api_key="")
        )
        
        result = config.validate()
        assert result is False
    
    def test_validate_with_keys(self):
        """测试有 API Key 时的验证"""
        config = LLMAssistantConfig(
            llm=LLMConfig(api_key="test_llm_key"),
            embedding=EmbeddingConfig(api_key="test_embedding_key")
        )
        
        result = config.validate()
        assert result is True
    
    def test_knowledge_dir_exists(self):
        """测试知识库目录"""
        config = LLMAssistantConfig()
        assert config.knowledge_dir.name == "knowledge"
    
    def test_prompts_dir_exists(self):
        """测试提示词目录"""
        config = LLMAssistantConfig()
        assert config.prompts_dir.name == "prompts"


class TestGlobalConfig:
    """全局配置测试"""
    
    def test_get_config(self):
        """测试获取全局配置"""
        config = get_config()
        assert isinstance(config, LLMAssistantConfig)
    
    def test_reload_config(self):
        """测试重新加载配置"""
        config1 = get_config()
        config2 = reload_config()
        
        # 两个配置应该是不同的实例
        assert config1 is not config2
        assert isinstance(config2, LLMAssistantConfig)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
LLM 客户端测试

使用 mock 模拟 API 响应，测试客户端功能。
"""

import pytest
from pathlib import Path
import sys
import json
from unittest.mock import AsyncMock, MagicMock, patch

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from llm.llm_client import LLMClient, ChatMessage, ChatResponse
from llm.config import LLMConfig


class TestChatMessage:
    """ChatMessage 测试"""
    
    def test_create_message(self):
        """测试创建消息"""
        msg = ChatMessage(role="user", content="Hello")
        
        assert msg.role == "user"
        assert msg.content == "Hello"
    
    def test_to_dict(self):
        """测试转换为字典"""
        msg = ChatMessage(role="assistant", content="Hi there!")
        result = msg.to_dict()
        
        assert result == {"role": "assistant", "content": "Hi there!"}


class TestChatResponse:
    """ChatResponse 测试"""
    
    def test_create_response(self):
        """测试创建响应"""
        response = ChatResponse(
            content="This is a response",
            role="assistant",
            model="qwen-plus",
            usage={"prompt_tokens": 10, "completion_tokens": 20},
            finish_reason="stop"
        )
        
        assert response.content == "This is a response"
        assert response.role == "assistant"
        assert response.model == "qwen-plus"
        assert response.usage["prompt_tokens"] == 10
    
    def test_from_api_response(self):
        """测试从 API 响应创建"""
        api_response = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Test response"
                },
                "finish_reason": "stop"
            }],
            "model": "qwen-plus",
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 10,
                "total_tokens": 15
            }
        }
        
        response = ChatResponse.from_api_response(api_response)
        
        assert response.content == "Test response"
        assert response.role == "assistant"
        assert response.model == "qwen-plus"
        assert response.finish_reason == "stop"


class TestLLMClient:
    """LLM 客户端测试"""
    
    def test_init_with_config(self):
        """测试使用配置初始化"""
        config = LLMConfig(
            api_key="test_key",
            model="test-model",
            temperature=0.5
        )
        client = LLMClient(config)
        
        assert client.config.api_key == "test_key"
        assert client.config.model == "test-model"
        assert client.config.temperature == 0.5
    
    def test_init_without_config(self):
        """测试不使用配置初始化"""
        client = LLMClient()
        assert client.config is not None
    
    def test_headers(self):
        """测试请求头"""
        config = LLMConfig(api_key="my_api_key")
        client = LLMClient(config)
        
        headers = client.headers
        
        assert headers["Authorization"] == "Bearer my_api_key"
        assert headers["Content-Type"] == "application/json"
    
    @pytest.mark.asyncio
    async def test_close(self):
        """测试关闭客户端"""
        client = LLMClient(LLMConfig(api_key="test"))
        
        # 初始化客户端
        _ = await client._get_client()
        assert client._client is not None
        
        # 关闭
        await client.close()
        assert client._client is None
    
    @pytest.mark.asyncio
    async def test_context_manager(self):
        """测试上下文管理器"""
        config = LLMConfig(api_key="test")
        
        async with LLMClient(config) as client:
            assert client._client is None or not client._client.is_closed
        
        # 退出后应该关闭
        # 注意：这里不能直接检查 _client，因为已经被设为 None
    
    @pytest.mark.asyncio
    async def test_chat_with_mock(self):
        """测试对话功能（使用 mock）"""
        config = LLMConfig(api_key="test_key", model="test-model")
        
        # 创建 mock 响应
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Mocked response"
                },
                "finish_reason": "stop"
            }],
            "model": "test-model",
            "usage": {"total_tokens": 10}
        }
        mock_response.raise_for_status = MagicMock()
        
        # 创建 mock 客户端
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.is_closed = False
        
        client = LLMClient(config)
        client._client = mock_client
        
        # 测试对话
        response = await client.chat("Hello")
        
        assert isinstance(response, ChatResponse)
        assert response.content == "Mocked response"
    
    @pytest.mark.asyncio
    async def test_chat_with_messages(self):
        """测试使用消息列表对话"""
        config = LLMConfig(api_key="test_key")
        
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {"role": "assistant", "content": "Response"},
                "finish_reason": "stop"
            }],
            "model": "qwen-plus",
            "usage": {}
        }
        mock_response.raise_for_status = MagicMock()
        
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.is_closed = False
        
        client = LLMClient(config)
        client._client = mock_client
        
        messages = [
            ChatMessage(role="system", content="You are helpful."),
            ChatMessage(role="user", content="Hi")
        ]
        
        response = await client.chat(messages)
        assert response.content == "Response"
    
    @pytest.mark.asyncio
    async def test_chat_with_dict_messages(self):
        """测试使用字典格式消息对话"""
        config = LLMConfig(api_key="test_key")
        
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {"role": "assistant", "content": "Dict response"},
                "finish_reason": "stop"
            }],
            "model": "qwen-plus",
            "usage": {}
        }
        mock_response.raise_for_status = MagicMock()
        
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.is_closed = False
        
        client = LLMClient(config)
        client._client = mock_client
        
        messages = [
            {"role": "user", "content": "Hello"}
        ]
        
        response = await client.chat(messages)
        assert response.content == "Dict response"
    
    @pytest.mark.asyncio
    async def test_chat_with_system_prompt(self):
        """测试带系统提示词的对话"""
        config = LLMConfig(api_key="test_key")
        
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{
                "message": {"role": "assistant", "content": "System response"},
                "finish_reason": "stop"
            }],
            "model": "qwen-plus",
            "usage": {}
        }
        mock_response.raise_for_status = MagicMock()
        
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.is_closed = False
        
        client = LLMClient(config)
        client._client = mock_client
        
        response = await client.chat_with_system(
            user_message="Test",
            system_prompt="You are an expert."
        )
        
        assert response.content == "System response"
    
    @pytest.mark.asyncio
    async def test_chat_stream_mock(self):
        """测试流式对话（简化版）"""
        config = LLMConfig(api_key="test_key")
        client = LLMClient(config)
        
        # 仅测试方法存在和配置正确
        assert hasattr(client, 'chat_stream')
        assert callable(client.chat_stream)
        
        # 测试通过 - 实际流式调用需要真实 API
        # 此测试仅验证方法签名


class TestLLMClientSync:
    """LLM 客户端同步接口测试"""
    
    def test_chat_sync_method_exists(self):
        """测试同步方法存在"""
        client = LLMClient(LLMConfig(api_key="test"))
        assert hasattr(client, 'chat_sync')
        assert callable(client.chat_sync)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "asyncio"])

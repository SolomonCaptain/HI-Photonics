"""
LLM 客户端 - 百炼云 qwen-plus

提供与百炼云 API 的交互接口，支持：
- 同步/异步调用
- 流式输出
- 对话历史管理
"""

from typing import Optional, List, Dict, Any, AsyncGenerator, Generator
from dataclasses import dataclass, field
import httpx
import json

from llm.config import LLMConfig


@dataclass
class ChatMessage:
    """对话消息"""
    role: str  # system, user, assistant
    content: str
    
    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class ChatResponse:
    """对话响应"""
    content: str
    role: str = "assistant"
    model: str = ""
    usage: Dict[str, int] = field(default_factory=dict)
    finish_reason: str = ""
    
    @classmethod
    def from_api_response(cls, data: Dict) -> "ChatResponse":
        """从 API 响应创建"""
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        return cls(
            content=message.get("content", ""),
            role=message.get("role", "assistant"),
            model=data.get("model", ""),
            usage=data.get("usage", {}),
            finish_reason=choice.get("finish_reason", "")
        )


class LLMClient:
    """
    LLM 客户端 - 百炼云 qwen-plus
    
    使用 OpenAI 兼容接口调用百炼云 API。
    
    示例:
        client = LLMClient(config)
        response = await client.chat("你好，请介绍一下光栅耦合器")
    """
    
    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self._client: Optional[httpx.AsyncClient] = None
    
    @property
    def headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }
    
    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(120.0, connect=30.0)
            )
        return self._client
    
    async def close(self):
        """关闭客户端"""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
    
    async def chat(
        self,
        messages: List[ChatMessage] | List[Dict] | str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        **kwargs
    ) -> ChatResponse:
        """
        发送对话请求
        
        Args:
            messages: 消息列表或单条消息字符串
            temperature: 温度参数
            max_tokens: 最大 token 数
            top_p: top_p 参数
            
        Returns:
            ChatResponse 对象
        """
        # 标准化消息格式
        if isinstance(messages, str):
            messages = [ChatMessage(role="user", content=messages)]
        elif isinstance(messages, list) and len(messages) > 0:
            if isinstance(messages[0], dict):
                messages = [ChatMessage(**m) for m in messages]
        
        payload = {
            "model": self.config.model,
            "messages": [m.to_dict() for m in messages],
            "temperature": temperature or self.config.temperature,
            "max_tokens": max_tokens or self.config.max_tokens,
            "top_p": top_p or self.config.top_p,
            **kwargs
        }
        
        client = await self._get_client()
        response = await client.post(
            f"{self.config.base_url}/chat/completions",
            headers=self.headers,
            json=payload
        )
        response.raise_for_status()
        
        return ChatResponse.from_api_response(response.json())
    
    async def chat_stream(
        self,
        messages: List[ChatMessage] | List[Dict] | str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        流式对话请求
        
        Yields:
            逐个 token 字符串
        """
        if isinstance(messages, str):
            messages = [ChatMessage(role="user", content=messages)]
        elif isinstance(messages, list) and len(messages) > 0:
            if isinstance(messages[0], dict):
                messages = [ChatMessage(**m) for m in messages]
        
        payload = {
            "model": self.config.model,
            "messages": [m.to_dict() for m in messages],
            "temperature": temperature or self.config.temperature,
            "max_tokens": max_tokens or self.config.max_tokens,
            "stream": True,
            **kwargs
        }
        
        client = await self._get_client()
        async with client.stream(
            "POST",
            f"{self.config.base_url}/chat/completions",
            headers=self.headers,
            json=payload
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            yield content
                    except json.JSONDecodeError:
                        continue
    
    async def chat_with_system(
        self,
        user_message: str,
        system_prompt: str,
        **kwargs
    ) -> ChatResponse:
        """带系统提示词的对话"""
        messages = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=user_message)
        ]
        return await self.chat(messages, **kwargs)
    
    def chat_sync(
        self,
        messages: List[ChatMessage] | List[Dict] | str,
        **kwargs
    ) -> ChatResponse:
        """同步对话接口（阻塞调用）"""
        import asyncio
        return asyncio.run(self.chat(messages, **kwargs))
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

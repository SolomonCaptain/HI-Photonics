"""
Embedding 客户端 - 硅基流动 Qwen3-Embedding-8B

提供文本嵌入向量的生成功能，用于 RAG 检索增强。
"""

from typing import Optional, List, Union
from dataclasses import dataclass
import httpx
import numpy as np

from llm.config import EmbeddingConfig


@dataclass
class EmbeddingResult:
    """嵌入结果"""
    embedding: np.ndarray
    model: str
    index: int = 0
    
    @property
    def dimension(self) -> int:
        return len(self.embedding)


class EmbeddingClient:
    """
    Embedding 客户端 - 硅基流动 Qwen3-Embedding-8B
    
    将文本转换为 4096 维向量，用于语义检索。
    
    示例:
        client = EmbeddingClient(config)
        embedding = await client.embed("光栅耦合器是一种重要的光学器件")
        print(f"向量维度: {embedding.dimension}")
    """
    
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or EmbeddingConfig()
        self._client: Optional[httpx.AsyncClient] = None
    
    @property
    def headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }
    
    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(60.0, connect=30.0)
            )
        return self._client
    
    async def close(self):
        """关闭客户端"""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
    
    async def embed(
        self,
        text: Union[str, List[str]],
        normalize: bool = True
    ) -> Union[EmbeddingResult, List[EmbeddingResult]]:
        """
        生成文本嵌入向量
        
        Args:
            text: 单个文本或文本列表
            normalize: 是否归一化向量
            
        Returns:
            EmbeddingResult 或 EmbeddingResult 列表
        """
        is_single = isinstance(text, str)
        texts = [text] if is_single else text
        
        client = await self._get_client()
        response = await client.post(
            f"{self.config.base_url}/embeddings",
            headers=self.headers,
            json={
                "model": self.config.model,
                "input": texts,
                "encoding_format": "float"
            }
        )
        response.raise_for_status()
        
        data = response.json()
        results = []
        
        for item in data.get("data", []):
            embedding = np.array(item["embedding"], dtype=np.float32)
            
            if normalize:
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
            
            results.append(EmbeddingResult(
                embedding=embedding,
                model=data.get("model", self.config.model),
                index=item.get("index", 0)
            ))
        
        return results[0] if is_single else results
    
    async def embed_batch(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        normalize: bool = True
    ) -> List[EmbeddingResult]:
        """
        批量生成嵌入向量
        
        Args:
            texts: 文本列表
            batch_size: 批次大小
            normalize: 是否归一化
            
        Returns:
            EmbeddingResult 列表
        """
        batch_size = batch_size or self.config.batch_size
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_results = await self.embed(batch, normalize=normalize)
            results.extend(batch_results)
        
        return results
    
    def embed_sync(
        self,
        text: Union[str, List[str]],
        normalize: bool = True
    ) -> Union[EmbeddingResult, List[EmbeddingResult]]:
        """同步嵌入接口（阻塞调用）"""
        import asyncio
        return asyncio.run(self.embed(text, normalize))
    
    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """计算余弦相似度"""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

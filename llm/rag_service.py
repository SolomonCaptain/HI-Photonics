"""
RAG 知识检索服务

结合 Embedding 和 Qdrant 实现知识检索增强生成。
"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from pathlib import Path
import re
import json

from llm.config import RAGConfig, LLMAssistantConfig
from llm.embedding_client import EmbeddingClient, EmbeddingResult
from llm.qdrant_service import QdrantService, KnowledgeDocument, SearchResult


@dataclass
class RetrievedContext:
    """检索到的上下文"""
    query: str
    documents: List[SearchResult]
    formatted_context: str


class TextSplitter:
    """文本分块器"""
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: Optional[List[str]] = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", "。", "，", " ", ""]
    
    def split(self, text: str) -> List[str]:
        """将文本分割成块"""
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            if end < len(text):
                # 尝试在分隔符处分割
                for sep in self.separators:
                    last_sep = text.rfind(sep, start, end)
                    if last_sep > start:
                        end = last_sep + len(sep)
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - self.chunk_overlap
            if start < 0:
                start = 0
            if start >= len(text):
                break
        
        return chunks


class RAGService:
    """
    RAG 知识检索服务
    
    提供知识库的索引和检索功能。
    
    示例:
        rag = RAGService(config)
        await rag.initialize()
        
        # 索引文档
        await rag.index_documents(documents)
        
        # 检索相关上下文
        context = await rag.retrieve("光栅耦合器的设计原理")
    """
    
    def __init__(
        self,
        config: Optional[LLMAssistantConfig] = None,
        embedding_client: Optional[EmbeddingClient] = None,
        qdrant_service: Optional[QdrantService] = None
    ):
        self.config = config or LLMAssistantConfig()
        self.rag_config = self.config.rag
        
        self.embedding_client = embedding_client or EmbeddingClient(self.config.embedding)
        self.qdrant_service = qdrant_service or QdrantService(self.config.qdrant)
        
        self.splitter = TextSplitter(
            chunk_size=self.rag_config.chunk_size,
            chunk_overlap=self.rag_config.chunk_overlap
        )
    
    def initialize(self, recreate: bool = False) -> bool:
        """
        初始化 RAG 服务
        
        Args:
            recreate: 是否重建向量数据库
            
        Returns:
            是否成功
        """
        return self.qdrant_service.initialize_collection(recreate=recreate)
    
    async def index_document(
        self,
        content: str,
        source: str = "",
        category: str = "",
        title: str = "",
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        索引单个文档
        
        Args:
            content: 文档内容
            source: 来源
            category: 分类
            title: 标题
            metadata: 额外元数据
            
        Returns:
            是否成功
        """
        # 分块
        chunks = self.splitter.split(content)
        
        if not chunks:
            return False
        
        # 生成嵌入向量
        embeddings = await self.embedding_client.embed_batch(chunks)
        
        # 创建文档并插入
        documents = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            doc = KnowledgeDocument(
                id=f"{source}_{i}_{hash(chunk) % 1000000}",
                content=chunk,
                embedding=embedding.embedding,
                source=source,
                category=category,
                title=title,
                metadata=metadata or {}
            )
            documents.append(doc)
        
        return self.qdrant_service.upsert(documents)
    
    async def index_documents(
        self,
        documents: List[Dict[str, Any]]
    ) -> int:
        """
        批量索引文档
        
        Args:
            documents: 文档列表，每个文档包含 content, source, category 等字段
            
        Returns:
            成功索引的文档数量
        """
        success_count = 0
        
        for doc in documents:
            success = await self.index_document(
                content=doc.get("content", ""),
                source=doc.get("source", ""),
                category=doc.get("category", ""),
                title=doc.get("title", ""),
                metadata=doc.get("metadata")
            )
            if success:
                success_count += 1
        
        return success_count
    
    async def index_directory(
        self,
        directory: Path,
        category: str = "",
        file_extensions: Optional[List[str]] = None
    ) -> int:
        """
        索引目录中的文档
        
        Args:
            directory: 目录路径
            category: 分类
            file_extensions: 文件扩展名过滤
            
        Returns:
            成功索引的文档数量
        """
        file_extensions = file_extensions or [".md", ".txt", ".json"]
        success_count = 0
        
        for ext in file_extensions:
            for file_path in directory.rglob(f"*{ext}"):
                try:
                    content = file_path.read_text(encoding="utf-8")
                    
                    # JSON 文件特殊处理
                    if ext == ".json":
                        data = json.loads(content)
                        if isinstance(data, list):
                            for item in data:
                                if isinstance(item, dict):
                                    await self.index_document(
                                        content=item.get("content", str(item)),
                                        source=str(file_path),
                                        category=item.get("category", category),
                                        title=item.get("title", file_path.stem)
                                    )
                        else:
                            await self.index_document(
                                content=content,
                                source=str(file_path),
                                category=category,
                                title=file_path.stem
                            )
                    else:
                        await self.index_document(
                            content=content,
                            source=str(file_path),
                            category=category,
                            title=file_path.stem
                        )
                    
                    success_count += 1
                except Exception as e:
                    print(f"索引文件 {file_path} 失败: {e}")
        
        return success_count
    
    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        filter_conditions: Optional[Dict] = None
    ) -> RetrievedContext:
        """
        检索相关上下文
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            score_threshold: 分数阈值
            filter_conditions: 过滤条件
            
        Returns:
            RetrievedContext 对象
        """
        top_k = top_k or self.rag_config.top_k
        score_threshold = score_threshold or self.rag_config.score_threshold
        
        # 生成查询向量
        query_embedding = await self.embedding_client.embed(query)
        
        # 检索相关文档
        results = self.qdrant_service.search(
            query_vector=query_embedding.embedding,
            top_k=top_k,
            score_threshold=score_threshold,
            filter_conditions=filter_conditions
        )
        
        # 格式化上下文
        formatted = self._format_context(results)
        
        return RetrievedContext(
            query=query,
            documents=results,
            formatted_context=formatted
        )
    
    def _format_context(self, results: List[SearchResult]) -> str:
        """格式化检索结果为上下文字符串"""
        if not results:
            return ""
        
        parts = []
        for i, result in enumerate(results, 1):
            source = result.metadata.get("source", "未知来源")
            parts.append(f"[文档 {i}] (来源: {source}, 相关度: {result.score:.2f})\n{result.content}")
        
        return "\n\n---\n\n".join(parts)
    
    async def close(self):
        """关闭连接"""
        await self.embedding_client.close()
        self.qdrant_service.disconnect()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

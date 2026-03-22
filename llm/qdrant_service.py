"""
Qdrant 向量数据库服务

提供向量存储、检索和管理功能，用于 RAG 知识检索。
"""

from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass, field
import uuid
import numpy as np

from qdrant_client import QdrantClient as QdrantClientSDK
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse

from llm.config import QdrantConfig


@dataclass
class KnowledgeDocument:
    """知识文档"""
    id: str
    content: str
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # 元数据字段
    source: str = ""
    category: str = ""
    title: str = ""


@dataclass
class SearchResult:
    """检索结果"""
    id: str
    content: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class QdrantService:
    """
    Qdrant 向量数据库服务
    
    提供知识向量的存储和检索功能。
    
    示例:
        service = QdrantService(config)
        await service.initialize()
        
        # 插入文档
        await service.upsert(documents)
        
        # 检索相关文档
        results = await service.search(query_embedding, top_k=5)
    """
    
    def __init__(self, config: Optional[QdrantConfig] = None):
        self.config = config or QdrantConfig()
        self._client: Optional[QdrantClientSDK] = None
    
    @property
    def collection_name(self) -> str:
        return self.config.collection
    
    def connect(self) -> bool:
        """
        连接到 Qdrant 服务器
        
        Returns:
            是否连接成功
        """
        try:
            self._client = QdrantClientSDK(
                host=self.config.host,
                port=self.config.port,
                api_key=self.config.api_key,
                timeout=60
            )
            # 测试连接
            self._client.get_collections()
            return True
        except Exception as e:
            print(f"连接 Qdrant 失败: {e}")
            return False
    
    def disconnect(self):
        """断开连接"""
        if self._client:
            self._client = None
    
    def initialize_collection(
        self,
        vector_size: Optional[int] = None,
        recreate: bool = False
    ) -> bool:
        """
        初始化集合
        
        Args:
            vector_size: 向量维度
            recreate: 是否重建集合
            
        Returns:
            是否成功
        """
        if not self._client:
            if not self.connect():
                return False
        
        vector_size = vector_size or self.config.vector_size
        
        try:
            # 检查集合是否存在
            collections = self._client.get_collections().collections
            collection_names = [c.name for c in collections]
            
            if self.collection_name in collection_names:
                if recreate:
                    self._client.delete_collection(self.collection_name)
                else:
                    return True  # 集合已存在
            
            # 创建新集合
            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE
                ),
                # 优化配置
                hnsw_config=models.HnswConfigDiff(
                    m=16,
                    ef_construct=100
                ),
                optimizers_config=models.OptimizersConfigDiff(
                    indexing_threshold=10000
                )
            )
            return True
            
        except Exception as e:
            print(f"初始化集合失败: {e}")
            return False
    
    def upsert(
        self,
        documents: Union[KnowledgeDocument, List[KnowledgeDocument]]
    ) -> bool:
        """
        插入或更新文档
        
        Args:
            documents: 单个文档或文档列表
            
        Returns:
            是否成功
        """
        if not self._client:
            if not self.connect():
                return False
        
        if isinstance(documents, KnowledgeDocument):
            documents = [documents]
        
        points = []
        for doc in documents:
            if doc.embedding is None:
                raise ValueError(f"文档 {doc.id} 缺少嵌入向量")
            
            points.append(models.PointStruct(
                id=doc.id,
                vector=doc.embedding.tolist(),
                payload={
                    "content": doc.content,
                    "source": doc.source,
                    "category": doc.category,
                    "title": doc.title,
                    **doc.metadata
                }
            ))
        
        try:
            self._client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            return True
        except Exception as e:
            print(f"插入文档失败: {e}")
            return False
    
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        filter_conditions: Optional[Dict] = None
    ) -> List[SearchResult]:
        """
        向量相似度检索
        
        Args:
            query_vector: 查询向量
            top_k: 返回结果数量
            score_threshold: 分数阈值
            filter_conditions: 过滤条件
            
        Returns:
            检索结果列表
        """
        if not self._client:
            if not self.connect():
                return []
        
        # 构建过滤条件
        query_filter = None
        if filter_conditions:
            must_conditions = []
            for key, value in filter_conditions.items():
                must_conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value)
                    )
                )
            if must_conditions:
                query_filter = models.Filter(must=must_conditions)
        
        try:
            results = self._client.search(
                collection_name=self.collection_name,
                query_vector=query_vector.tolist(),
                limit=top_k,
                score_threshold=score_threshold,
                query_filter=query_filter
            )
            
            return [
                SearchResult(
                    id=str(result.id),
                    content=result.payload.get("content", ""),
                    score=result.score,
                    metadata={
                        k: v for k, v in result.payload.items()
                        if k != "content"
                    }
                )
                for result in results
            ]
        except Exception as e:
            print(f"检索失败: {e}")
            return []
    
    def delete(self, ids: Union[str, List[str]]) -> bool:
        """
        删除文档
        
        Args:
            ids: 文档 ID 或 ID 列表
            
        Returns:
            是否成功
        """
        if not self._client:
            if not self.connect():
                return False
        
        if isinstance(ids, str):
            ids = [ids]
        
        try:
            self._client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(
                    points=ids
                )
            )
            return True
        except Exception as e:
            print(f"删除文档失败: {e}")
            return False
    
    def get_collection_info(self) -> Optional[Dict]:
        """获取集合信息"""
        if not self._client:
            if not self.connect():
                return None
        
        try:
            info = self._client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": info.status.value
            }
        except Exception as e:
            print(f"获取集合信息失败: {e}")
            return None
    
    def scroll(
        self,
        limit: int = 100,
        offset: Optional[str] = None
    ) -> tuple[List[KnowledgeDocument], Optional[str]]:
        """
        遍历集合中的文档
        
        Args:
            limit: 每次获取的数量
            offset: 偏移量
            
        Returns:
            (文档列表, 下一个偏移量)
        """
        if not self._client:
            if not self.connect():
                return [], None
        
        try:
            points, next_offset = self._client.scroll(
                collection_name=self.collection_name,
                limit=limit,
                offset=offset,
                with_vectors=False
            )
            
            documents = [
                KnowledgeDocument(
                    id=str(point.id),
                    content=point.payload.get("content", ""),
                    metadata={
                        k: v for k, v in point.payload.items()
                        if k != "content"
                    },
                    source=point.payload.get("source", ""),
                    category=point.payload.get("category", ""),
                    title=point.payload.get("title", "")
                )
                for point in points
            ]
            
            return documents, next_offset
        except Exception as e:
            print(f"遍历文档失败: {e}")
            return [], None
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

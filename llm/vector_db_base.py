"""
向量数据库抽象基类

定义统一的向量数据库接口，支持多种向量数据库实现。
"""

from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass, field
import numpy as np


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


class VectorDBBase(ABC):
    """
    向量数据库抽象基类
    
    定义所有向量数据库必须实现的接口。
    
    实现类:
    - QdrantService: Qdrant 向量数据库
    - ChromaService: Chroma 向量数据库
    """
    
    @abstractmethod
    def connect(self) -> bool:
        """
        连接到向量数据库
        
        Returns:
            是否连接成功
        """
        pass
    
    @abstractmethod
    def disconnect(self):
        """断开连接"""
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    def delete(self, ids: Union[str, List[str]]) -> bool:
        """
        删除文档
        
        Args:
            ids: 文档 ID 或 ID 列表
            
        Returns:
            是否成功
        """
        pass
    
    @abstractmethod
    def get_collection_info(self) -> Optional[Dict]:
        """
        获取集合信息
        
        Returns:
            集合信息字典
        """
        pass
    
    @abstractmethod
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
        pass
    
    @property
    @abstractmethod
    def db_type(self) -> str:
        """返回数据库类型标识"""
        pass
    
    @property
    @abstractmethod
    def collection_name(self) -> str:
        """返回集合名称"""
        pass
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

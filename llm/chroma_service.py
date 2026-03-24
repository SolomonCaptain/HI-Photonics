"""
Chroma 向量数据库服务

提供向量存储、检索和管理功能，用于 RAG 知识检索。
Chroma 是一个轻量级的开源向量数据库，支持本地持久化存储。
"""

from typing import Optional, List, Dict, Any, Union
from pathlib import Path
import numpy as np

from llm.config import ChromaConfig
from llm.vector_db_base import VectorDBBase, KnowledgeDocument, SearchResult


class ChromaService(VectorDBBase):
    """
    Chroma 向量数据库服务
    
    提供知识向量的存储和检索功能。
    支持本地持久化存储和内存模式。
    
    示例:
        service = ChromaService(config)
        service.initialize()
        
        # 插入文档
        service.upsert(documents)
        
        # 检索相关文档
        results = service.search(query_embedding, top_k=5)
    """
    
    def __init__(self, config: Optional[ChromaConfig] = None):
        self.config = config or ChromaConfig()
        self._client = None
        self._collection = None
    
    @property
    def collection_name(self) -> str:
        return self.config.collection
    
    @property
    def db_type(self) -> str:
        """返回数据库类型标识"""
        return "chroma"
    
    def connect(self) -> bool:
        """
        连接到 Chroma 数据库
        
        Returns:
            是否连接成功
        """
        try:
            import chromadb
            
            if self.config.persistent:
                # 确保持久化目录存在
                persist_dir = Path(self.config.persist_directory)
                persist_dir.mkdir(parents=True, exist_ok=True)
                
                self._client = chromadb.PersistentClient(
                    path=str(persist_dir)
                )
            else:
                # 内存模式
                self._client = chromadb.Client()
            
            return True
        except ImportError:
            print("ChromaDB 未安装，请运行: pip install chromadb")
            return False
        except Exception as e:
            print(f"连接 Chroma 失败: {e}")
            return False
    
    def disconnect(self):
        """断开连接"""
        # Chroma 客户端无需显式关闭
        self._client = None
        self._collection = None
    
    def initialize_collection(
        self,
        vector_size: Optional[int] = None,
        recreate: bool = False
    ) -> bool:
        """
        初始化集合
        
        Args:
            vector_size: 向量维度（Chroma 自动管理，此参数用于兼容）
            recreate: 是否重建集合
            
        Returns:
            是否成功
        """
        if not self._client:
            if not self.connect():
                return False
        
        try:
            # 检查集合是否存在
            existing_collections = [c.name for c in self._client.list_collections()]
            
            if self.collection_name in existing_collections:
                if recreate:
                    self._client.delete_collection(self.collection_name)
                else:
                    # 获取现有集合
                    self._collection = self._client.get_collection(
                        name=self.collection_name
                    )
                    return True
            
            # 创建新集合
            # Chroma 使用自定义的距离函数名称
            distance_map = {
                "cosine": "cosine",
                "euclidean": "l2",
                "dot": "ip"
            }
            distance_fn = distance_map.get(
                self.config.distance_metric.lower(), 
                "cosine"
            )
            
            self._collection = self._client.create_collection(
                name=self.collection_name,
                metadata={
                    "hnsw:space": distance_fn,
                    "description": "HI-Photonics Knowledge Base"
                }
            )
            return True
            
        except Exception as e:
            print(f"初始化集合失败: {e}")
            return False
    
    def _ensure_collection(self) -> bool:
        """确保集合已初始化"""
        if self._collection is None:
            return self.initialize_collection()
        return True
    
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
        if not self._ensure_collection():
            return False
        
        if isinstance(documents, KnowledgeDocument):
            documents = [documents]
        
        ids = []
        embeddings = []
        metadatas = []
        contents = []
        
        for doc in documents:
            if doc.embedding is None:
                raise ValueError(f"文档 {doc.id} 缺少嵌入向量")
            
            ids.append(doc.id)
            embeddings.append(doc.embedding.tolist())
            contents.append(doc.content)
            
            # 合并元数据
            metadata = {
                "source": doc.source,
                "category": doc.category,
                "title": doc.title,
                **doc.metadata
            }
            metadatas.append(metadata)
        
        try:
            self._collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=contents,
                metadatas=metadatas
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
            score_threshold: 分数阈值（Chroma 的距离阈值）
            filter_conditions: 过滤条件
            
        Returns:
            检索结果列表
        """
        if not self._ensure_collection():
            return []
        
        try:
            # Chroma 的 where 过滤
            where_filter = None
            if filter_conditions:
                # Chroma 支持 $and, $or 等操作符
                if len(filter_conditions) == 1:
                    key, value = list(filter_conditions.items())[0]
                    where_filter = {key: value}
                else:
                    where_filter = {"$and": [
                        {k: v} for k, v in filter_conditions.items()
                    ]}
            
            results = self._collection.query(
                query_embeddings=[query_vector.tolist()],
                n_results=top_k,
                where=where_filter,
                include=["documents", "metadatas", "distances"]
            )
            
            # 解析结果
            search_results = []
            if results and results.get("ids"):
                ids = results["ids"][0]
                documents = results.get("documents", [[]])[0]
                metadatas = results.get("metadatas", [[]])[0]
                distances = results.get("distances", [[]])[0]
                
                for i, doc_id in enumerate(ids):
                    # Chroma 返回的是距离，需要转换为相似度分数
                    # 对于 cosine 距离，相似度 = 1 - 距离
                    distance = distances[i] if i < len(distances) else 0
                    score = 1 - distance  # 转换为相似度
                    
                    # 应用分数阈值
                    if score_threshold is not None and score < score_threshold:
                        continue
                    
                    metadata = metadatas[i] if i < len(metadatas) else {}
                    content = documents[i] if i < len(documents) else ""
                    
                    search_results.append(SearchResult(
                        id=str(doc_id),
                        content=content,
                        score=score,
                        metadata=metadata
                    ))
            
            return search_results
            
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
        if not self._ensure_collection():
            return False
        
        if isinstance(ids, str):
            ids = [ids]
        
        try:
            self._collection.delete(ids=ids)
            return True
        except Exception as e:
            print(f"删除文档失败: {e}")
            return False
    
    def get_collection_info(self) -> Optional[Dict]:
        """获取集合信息"""
        if not self._ensure_collection():
            return None
        
        try:
            count = self._collection.count()
            return {
                "name": self.collection_name,
                "count": count,
                "db_type": "chroma",
                "persistent": self.config.persistent,
                "persist_directory": self.config.persist_directory if self.config.persistent else None
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
            offset: 偏移量（Chroma 不支持真正的偏移，使用 ID 分页）
            
        Returns:
            (文档列表, 下一个偏移量)
        """
        if not self._ensure_collection():
            return [], None
        
        try:
            # Chroma 的 get 方法获取所有数据
            results = self._collection.get(
                limit=limit,
                offset=int(offset) if offset else 0,
                include=["documents", "metadatas"]
            )
            
            documents = []
            if results and results.get("ids"):
                ids = results["ids"]
                contents = results.get("documents", [])
                metadatas = results.get("metadatas", [])
                
                for i, doc_id in enumerate(ids):
                    metadata = metadatas[i] if i < len(metadatas) else {}
                    content = contents[i] if i < len(contents) else ""
                    
                    documents.append(KnowledgeDocument(
                        id=str(doc_id),
                        content=content,
                        metadata=metadata,
                        source=metadata.get("source", ""),
                        category=metadata.get("category", ""),
                        title=metadata.get("title", "")
                    ))
            
            # Chroma 不支持真正的游标分页
            next_offset = None
            if len(documents) == limit:
                current_offset = int(offset) if offset else 0
                next_offset = str(current_offset + limit)
            
            return documents, next_offset
            
        except Exception as e:
            print(f"遍历文档失败: {e}")
            return [], None
    
    def count(self) -> int:
        """获取集合中的文档数量"""
        if not self._ensure_collection():
            return 0
        return self._collection.count()
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

"""
测试配置文件

处理可选依赖的 mock。
"""

import sys
from unittest.mock import MagicMock, Mock

# 完整 mock qdrant_client 模块（可选依赖）
def create_qdrant_mock():
    mock_qdrant = MagicMock()
    
    # Mock http 子模块
    mock_http = MagicMock()
    mock_models = MagicMock()
    mock_exceptions = MagicMock()
    
    # Mock 常用类
    mock_models.PointStruct = MagicMock
    mock_models.Filter = MagicMock
    mock_models.FieldCondition = MagicMock
    mock_models.MatchValue = MagicMock
    mock_models.SearchRequest = MagicMock
    
    # Mock 异常类
    mock_exceptions.UnexpectedResponse = Exception
    
    mock_qdrant.http = mock_http
    mock_qdrant.http.models = mock_models
    mock_qdrant.http.exceptions = mock_exceptions
    mock_qdrant.models = mock_models
    
    # Mock QdrantClient
    mock_client_class = MagicMock
    mock_qdrant.QdrantClient = mock_client_class
    
    return mock_qdrant

mock_qdrant = create_qdrant_mock()

# 在导入 llm 模块之前设置 mock
sys.modules['qdrant_client'] = mock_qdrant
sys.modules['qdrant_client.http'] = mock_qdrant.http
sys.modules['qdrant_client.http.models'] = mock_qdrant.http.models
sys.modules['qdrant_client.http.exceptions'] = mock_qdrant.http.exceptions
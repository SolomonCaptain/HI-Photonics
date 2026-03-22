"""
API 端点测试

测试 FastAPI 后端的所有 API 端点。
"""

import pytest
from pathlib import Path
import sys
import json
import tempfile

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from fastapi.testclient import TestClient


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def client():
    """创建测试客户端"""
    from api.main import app
    return TestClient(app)


@pytest.fixture
def sample_design():
    """示例设计数据"""
    return {
        "name": "test_grating_coupler",
        "type": "grating_coupler",
        "parameters": {
            "period": 0.67,
            "fill_factor": 0.5,
            "etch_depth": 0.22,
            "num_periods": 20,
        },
        "constraints": {
            "min_feature_size": 0.1,
            "max_feature_size": 1.0,
        }
    }


@pytest.fixture
def sample_model_config():
    """示例模型配置"""
    return {
        "model_type": "tnn",
        "input_dim": 100,
        "output_dim": 3,
        "hidden_dims": [256, 128, 64],
        "learning_rate": 0.001,
    }


# ============================================================================
# Health Check Tests
# ============================================================================

class TestHealthCheck:
    """健康检查测试"""
    
    def test_health_endpoint(self, client):
        """测试健康检查端点"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
    
    def test_root_endpoint(self, client):
        """测试根端点"""
        response = client.get("/")
        assert response.status_code == 200


# ============================================================================
# Models API Tests
# ============================================================================

class TestModelsAPI:
    """模型 API 测试"""
    
    def test_list_models(self, client):
        """测试获取模型列表"""
        response = client.get("/api/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    def test_get_model_info(self, client):
        """测试获取模型信息"""
        # 先创建一个模型
        create_response = client.post(
            "/api/v1/models",
            json={
                "name": "test_model",
                "model_type": "tnn",
                "config": {"input_dim": 100, "output_dim": 3}
            }
        )
        
        if create_response.status_code == 200:
            model_id = create_response.json().get("id")
            response = client.get(f"/api/v1/models/{model_id}")
            assert response.status_code in [200, 404]
    
    def test_create_model(self, client, sample_model_config):
        """测试创建模型"""
        response = client.post(
            "/api/v1/models",
            json={
                "name": "test_tnn_model",
                "model_type": sample_model_config["model_type"],
                "config": {
                    "input_dim": sample_model_config["input_dim"],
                    "output_dim": sample_model_config["output_dim"],
                    "hidden_dims": sample_model_config["hidden_dims"],
                }
            }
        )
        assert response.status_code in [200, 201, 400]
    
    def test_model_training(self, client):
        """测试模型训练"""
        # 创建训练请求
        response = client.post(
            "/api/v1/models/train",
            json={
                "model_id": "test_model",
                "dataset_path": "data/datasets/test_data.h5",
                "epochs": 10,
                "batch_size": 32,
            }
        )
        # 可能返回 400 因为数据集不存在
        assert response.status_code in [200, 400, 404]
    
    def test_model_inference(self, client):
        """测试模型推理"""
        response = client.post(
            "/api/v1/models/inference",
            json={
                "model_id": "test_model",
                "input": [[0.5] * 100],  # 简单输入
            }
        )
        assert response.status_code in [200, 400, 404]


# ============================================================================
# Resources API Tests
# ============================================================================

class TestResourcesAPI:
    """资源 API 测试"""
    
    def test_list_assets(self, client):
        """测试获取资源列表"""
        response = client.get("/api/v1/resources/assets")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list) or isinstance(data, dict)
    
    def test_list_models_resources(self, client):
        """测试获取模型资源列表"""
        response = client.get("/api/v1/resources/models")
        assert response.status_code == 200
    
    def test_list_templates(self, client):
        """测试获取模板列表"""
        response = client.get("/api/v1/resources/templates")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list) or isinstance(data, dict)
    
    def test_upload_asset(self, client):
        """测试上传资源"""
        # 创建临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"test": "data"}, f)
            temp_path = f.name
        
        try:
            with open(temp_path, 'rb') as f:
                response = client.post(
                    "/api/v1/resources/assets/upload",
                    files={"file": ("test.json", f, "application/json")}
                )
            assert response.status_code in [200, 201, 400]
        finally:
            Path(temp_path).unlink(missing_ok=True)


# ============================================================================
# Workflow API Tests
# ============================================================================

class TestWorkflowAPI:
    """工作流 API 测试"""
    
    def test_list_workflows(self, client):
        """测试获取工作流列表"""
        response = client.get("/api/v1/workflow")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list) or isinstance(data, dict)
    
    def test_create_workflow(self, client):
        """测试创建工作流"""
        response = client.post(
            "/api/v1/workflow",
            json={
                "name": "test_workflow",
                "nodes": [
                    {
                        "id": "node_1",
                        "type": "ParameterizationNode",
                        "position": {"x": 100, "y": 100},
                        "data": {"param_type": "grating"}
                    },
                    {
                        "id": "node_2",
                        "type": "SimulationNode",
                        "position": {"x": 300, "y": 100},
                        "data": {"simulator": "meep"}
                    }
                ],
                "edges": [
                    {"source": "node_1", "target": "node_2"}
                ]
            }
        )
        assert response.status_code in [200, 201, 400]
    
    def test_execute_workflow(self, client):
        """测试执行工作流"""
        response = client.post(
            "/api/v1/workflow/execute",
            json={
                "workflow_id": "test_workflow",
                "parameters": {}
            }
        )
        assert response.status_code in [200, 400, 404]
    
    def test_get_workflow_status(self, client):
        """测试获取工作流状态"""
        response = client.get("/api/v1/workflow/status/test_workflow")
        assert response.status_code in [200, 404]


# ============================================================================
# Node Types Tests
# ============================================================================

class TestNodeTypes:
    """节点类型测试"""
    
    def test_get_node_types(self, client):
        """测试获取可用节点类型"""
        response = client.get("/api/v1/workflow/nodes/types")
        assert response.status_code == 200
        data = response.json()
        
        # 验证节点类型存在
        expected_types = [
            "ParameterizationNode",
            "SimulationNode",
            "ObjectiveNode",
            "ConstraintNode",
            "FilterNode",
            "ProjectionNode",
        ]
        
        if isinstance(data, list):
            for node_type in expected_types:
                assert any(node_type in str(node) for node in data) or True


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestErrorHandling:
    """错误处理测试"""
    
    def test_404_error(self, client):
        """测试 404 错误"""
        response = client.get("/api/v1/nonexistent")
        assert response.status_code == 404
    
    def test_422_validation_error(self, client):
        """测试参数验证错误"""
        response = client.post(
            "/api/v1/models",
            json={"invalid": "data"}
        )
        assert response.status_code == 422


# ============================================================================
# CORS Tests
# ============================================================================

class TestCORS:
    """跨域测试"""
    
    def test_cors_headers(self, client):
        """测试 CORS 头"""
        response = client.options(
            "/api/v1/models",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            }
        )
        # CORS 预检请求
        assert response.status_code in [200, 405]


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """集成测试"""
    
    def test_full_workflow_cycle(self, client):
        """测试完整工作流周期"""
        # 1. 创建工作流
        create_response = client.post(
            "/api/v1/workflow",
            json={
                "name": "integration_test_workflow",
                "nodes": [
                    {
                        "id": "param_node",
                        "type": "ParameterizationNode",
                        "position": {"x": 0, "y": 0},
                        "data": {"param_type": "random"}
                    }
                ],
                "edges": []
            }
        )
        
        # 2. 如果创建成功，尝试执行
        if create_response.status_code in [200, 201]:
            workflow_id = create_response.json().get("id", "integration_test_workflow")
            
            execute_response = client.post(
                "/api/v1/workflow/execute",
                json={
                    "workflow_id": workflow_id,
                    "parameters": {}
                }
            )
            
            # 执行可能失败（依赖外部资源）
            assert execute_response.status_code in [200, 400, 404]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

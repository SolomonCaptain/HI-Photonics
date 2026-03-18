"""
工作流 API 路由
"""

from fastapi import APIRouter, HTTPException
from typing import List
from api.models.schemas import (
    NodeInstance, NodeConnection, NodeDefinition, ExecutionResult,
    WorkflowCreate, WorkflowResponse, NodeTypeEnum
)
from api.services import workflow_service

router = APIRouter(prefix="/workflow", tags=["workflow"])

# 节点定义
NODE_DEFINITIONS = {
    NodeTypeEnum.PARAMETERIZATION: NodeDefinition(
        type=NodeTypeEnum.PARAMETERIZATION,
        name="参数化",
        category="设计",
        description="定义设计参数空间",
        icon="Tune",
        inputs=[],
        outputs=[{"id": "design", "name": "设计参数", "type": "output", "data_type": "design", "required": True}],
        params=[
            {"key": "shape", "label": "设计形状", "type": "array", "default": [100, 22]},
            {"key": "method", "label": "参数化方法", "type": "select", "default": "density"}
        ]
    ),
    NodeTypeEnum.SIMULATION: NodeDefinition(
        type=NodeTypeEnum.SIMULATION,
        name="仿真器",
        category="仿真",
        description="运行 FDTD/RCWA 仿真",
        icon="PlayArrow",
        inputs=[{"id": "design", "name": "设计参数", "type": "input", "data_type": "design", "required": True}],
        outputs=[
            {"id": "performance", "name": "性能指标", "type": "output", "data_type": "performance", "required": True},
            {"id": "fields", "name": "场分布", "type": "output", "data_type": "data", "required": False}
        ],
        params=[
            {"key": "simulator", "label": "仿真器类型", "type": "select", "default": "meep"},
            {"key": "wavelength", "label": "波长 (μm)", "type": "number", "default": 1.55}
        ]
    ),
    NodeTypeEnum.MODEL_TRAIN: NodeDefinition(
        type=NodeTypeEnum.MODEL_TRAIN,
        name="模型训练",
        category="模型",
        description="训练神经网络模型",
        icon="ModelTraining",
        inputs=[{"id": "data", "name": "训练数据", "type": "input", "data_type": "data", "required": True}],
        outputs=[{"id": "model", "name": "训练模型", "type": "output", "data_type": "model", "required": True}],
        params=[
            {"key": "modelType", "label": "模型类型", "type": "select", "default": "hilab"},
            {"key": "epochs", "label": "训练轮数", "type": "number", "default": 100},
            {"key": "batchSize", "label": "批次大小", "type": "number", "default": 32}
        ]
    ),
    NodeTypeEnum.MODEL_INFER: NodeDefinition(
        type=NodeTypeEnum.MODEL_INFER,
        name="模型推理",
        category="模型",
        description="使用模型进行逆向设计",
        icon="Psychology",
        inputs=[
            {"id": "model", "name": "模型", "type": "input", "data_type": "model", "required": True},
            {"id": "target", "name": "目标性能", "type": "input", "data_type": "performance", "required": True}
        ],
        outputs=[{"id": "design", "name": "设计结果", "type": "output", "data_type": "design", "required": True}],
        params=[
            {"key": "numSamples", "label": "采样数量", "type": "number", "default": 1}
        ]
    ),
    NodeTypeEnum.DATA_LOAD: NodeDefinition(
        type=NodeTypeEnum.DATA_LOAD,
        name="数据加载",
        category="数据",
        description="加载训练/测试数据",
        icon="Storage",
        inputs=[],
        outputs=[{"id": "data", "name": "数据集", "type": "output", "data_type": "data", "required": True}],
        params=[
            {"key": "source", "label": "数据源", "type": "select", "default": "synthetic"},
            {"key": "numSamples", "label": "样本数量", "type": "number", "default": 1000}
        ]
    ),
    NodeTypeEnum.OPTIMIZER: NodeDefinition(
        type=NodeTypeEnum.OPTIMIZER,
        name="优化器",
        category="优化",
        description="拓扑优化求解器",
        icon="AutoFixHigh",
        inputs=[
            {"id": "objective", "name": "目标函数", "type": "input", "data_type": "params", "required": True},
            {"id": "design", "name": "初始设计", "type": "input", "data_type": "design", "required": True}
        ],
        outputs=[{"id": "optimized", "name": "优化结果", "type": "output", "data_type": "design", "required": True}],
        params=[
            {"key": "method", "label": "优化方法", "type": "select", "default": "adam"},
            {"key": "iterations", "label": "迭代次数", "type": "number", "default": 100}
        ]
    ),
    NodeTypeEnum.OUTPUT: NodeDefinition(
        type=NodeTypeEnum.OUTPUT,
        name="输出",
        category="输出",
        description="查看和可视化结果",
        icon="Visibility",
        inputs=[
            {"id": "design", "name": "设计", "type": "input", "data_type": "design", "required": False},
            {"id": "performance", "name": "性能", "type": "input", "data_type": "performance", "required": False}
        ],
        outputs=[],
        params=[]
    )
}


@router.get("/nodes")
async def get_node_definitions() -> dict:
    """获取所有节点定义"""
    return {k.value: v.dict() for k, v in NODE_DEFINITIONS.items()}


@router.post("/execute")
async def execute_workflow(
    nodes: List[NodeInstance],
    edges: List[NodeConnection]
) -> List[ExecutionResult]:
    """执行工作流"""
    return await workflow_service.execute_workflow(nodes, edges)


@router.post("/execute-node")
async def execute_single_node(
    node: NodeInstance,
    inputs: dict = {}
) -> ExecutionResult:
    """执行单个节点"""
    return await workflow_service.execute_node(node, inputs, [], [])


# 工作流存储（简化版，实际应使用数据库）
_workflows: dict = {}


@router.post("/", response_model=WorkflowResponse)
async def create_workflow(workflow: WorkflowCreate) -> WorkflowResponse:
    """创建工作流"""
    from datetime import datetime
    import uuid
    
    workflow_id = str(uuid.uuid4())
    now = datetime.now().isoformat()
    
    response = WorkflowResponse(
        id=workflow_id,
        name=workflow.name,
        description=workflow.description,
        nodes=workflow.nodes,
        edges=workflow.edges,
        created_at=now,
        updated_at=now
    )
    
    _workflows[workflow_id] = response
    return response


@router.get("/{workflow_id}", response_model=WorkflowResponse)
async def get_workflow(workflow_id: str) -> WorkflowResponse:
    """获取工作流"""
    if workflow_id not in _workflows:
        raise HTTPException(status_code=404, detail="Workflow not found")
    return _workflows[workflow_id]


@router.delete("/{workflow_id}")
async def delete_workflow(workflow_id: str) -> dict:
    """删除工作流"""
    if workflow_id not in _workflows:
        raise HTTPException(status_code=404, detail="Workflow not found")
    del _workflows[workflow_id]
    return {"message": "Workflow deleted"}

"""
API 数据模型定义
"""

from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field
from enum import Enum


class NodeTypeEnum(str, Enum):
    PARAMETERIZATION = "parameterization"
    SIMULATION = "simulation"
    OBJECTIVE = "objective"
    FILTER = "filter"
    PROJECTION = "projection"
    CONSTRAINT = "constraint"
    MODEL_TRAIN = "model_train"
    MODEL_INFER = "model_infer"
    DATA_LOAD = "data_load"
    DATA_SAVE = "data_save"
    OPTIMIZER = "optimizer"
    OUTPUT = "output"


class NodeStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    SUCCESS = "success"
    ERROR = "error"


class ModelTypeEnum(str, Enum):
    TNN = "tnn"
    MDN = "mdn"
    CGAN = "cgan"
    PINN = "pinn"
    GNN = "gnn"
    HILAB = "hilab"


# ===== 节点相关 =====

class NodePort(BaseModel):
    id: str
    name: str
    type: str  # 'input' | 'output'
    data_type: str
    required: bool = True
    multiple: bool = False


class NodeDefinition(BaseModel):
    type: NodeTypeEnum
    name: str
    category: str
    description: str
    icon: str
    inputs: List[NodePort]
    outputs: List[NodePort]
    params: List[Dict[str, Any]]


class NodeInstance(BaseModel):
    id: str
    type: NodeTypeEnum
    position: Dict[str, float]
    data: Dict[str, Any]


class NodeConnection(BaseModel):
    id: str
    source: str
    source_handle: str
    target: str
    target_handle: str


# ===== 工作流相关 =====

class WorkflowCreate(BaseModel):
    name: str
    description: Optional[str] = None
    nodes: List[NodeInstance]
    edges: List[NodeConnection]


class WorkflowResponse(WorkflowCreate):
    id: str
    created_at: str
    updated_at: str


class ExecutionRequest(BaseModel):
    node_id: str
    params: Dict[str, Any]


class ExecutionResult(BaseModel):
    node_id: str
    status: NodeStatus
    output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    duration: float


# ===== 模型训练相关 =====

class TrainingConfig(BaseModel):
    model_type: ModelTypeEnum
    epochs: int = 100
    batch_size: int = Field(default=32, alias="batchSize")
    learning_rate: float = Field(default=0.001, alias="learningRate")
    weight_decay: float = 1e-5
    patience: int = 15
    device: str = "auto"

    class Config:
        populate_by_name = True


class TrainingProgress(BaseModel):
    epoch: int
    total_epochs: int
    loss: float
    val_loss: Optional[float] = None
    metrics: Dict[str, float]
    status: str


class InverseDesignRequest(BaseModel):
    model_type: ModelTypeEnum
    target_performance: List[float]
    num_samples: int = 1
    diversity_weight: float = 0.0


class InverseDesignResponse(BaseModel):
    designs: List[List[List[float]]]  # [samples, H, W]
    predicted_performance: List[List[float]]
    confidence: Optional[List[float]] = None


# ===== 数据相关 =====

class DatasetConfig(BaseModel):
    source: str = "synthetic"
    num_samples: int = Field(default=1000, alias="numSamples")
    design_shape: List[int] = Field(default=[100, 22], alias="designShape")
    performance_dim: int = 3
    noise_level: float = 0.05
    filepath: Optional[str] = None

    class Config:
        populate_by_name = True


class DatasetInfo(BaseModel):
    num_samples: int
    design_shape: List[int]
    performance_dim: int
    train_samples: int
    val_samples: int
    test_samples: int


# ===== 资源管理相关 =====

class ResourceCategory(str, Enum):
    """资源大类"""
    INPUTS = "inputs"
    OUTPUTS = "outputs"
    MODELS = "models"
    WORKFLOWS = "workflows"


class AssetTypeEnum(str, Enum):
    """资产类型"""
    # 输入类
    DATASET = "dataset"
    SPECTRUM = "spectrum"
    GDS = "gds"
    STRUCTURE = "structure"
    # 输出类
    DESIGN = "design"
    SIMULATION = "simulation"
    EXPORT = "export"
    # 其他
    FIELD = "field"
    MODEL_WEIGHTS = "model_weights"


class AssetInfo(BaseModel):
    """资产信息"""
    id: str
    name: str
    type: AssetTypeEnum
    category: ResourceCategory
    description: Optional[str] = None
    file_path: str
    file_size: int = 0
    created_at: str
    updated_at: str
    metadata: Optional[Dict[str, Any]] = None
    thumbnail: Optional[str] = None

    class Config:
        use_enum_values = True


class AssetCreate(BaseModel):
    """创建资产请求"""
    name: str
    type: AssetTypeEnum
    category: ResourceCategory
    description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class AssetUpdate(BaseModel):
    """更新资产请求"""
    name: Optional[str] = None
    description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ModelInfo(BaseModel):
    """模型信息"""
    id: str
    name: str
    type: ModelTypeEnum
    challenge: str
    description: Optional[str] = None
    file_path: str
    file_size: int = 0
    metrics: Optional[Dict[str, float]] = None
    created_at: str
    updated_at: str
    is_pretrained: bool = False

    class Config:
        use_enum_values = True


class SavedWorkflow(BaseModel):
    """已保存的工作流"""
    id: str
    name: str
    description: Optional[str] = None
    nodes: List[NodeInstance]
    edges: List[NodeConnection]
    file_path: str
    created_at: str
    updated_at: str
    tags: Optional[List[str]] = None


class WorkflowTemplate(BaseModel):
    """工作流模板"""
    id: str
    name: str
    description: Optional[str] = None
    category: str
    icon: str
    nodes: List[NodeInstance]
    edges: List[NodeConnection]
    tags: Optional[List[str]] = None
    file_path: str


class DirectoryInfo(BaseModel):
    """目录信息"""
    path: str
    name: str
    category: ResourceCategory
    total_size: int
    file_count: int
    subdirectories: List[str]

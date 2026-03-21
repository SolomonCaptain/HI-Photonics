"""
HI-Photonics 工作流模块

提供完整的逆向设计工作流管道和任务调度功能。
"""

from workflows.pipeline import (
    # 枚举
    PipelineStage,
    PipelineStatus,
    # 配置
    PipelineConfig,
    StageResult,
    # 阶段基类
    PipelineStageBase,
    DataGenerationStage,
    DataLoadingStage,
    ModelTrainingStage,
    InverseDesignStage,
    SimulationStage,
    ValidationStage,
    ExportStage,
    # 主管道类
    DesignPipeline,
    # 便捷函数
    create_pipeline,
    run_quick_design
)

from workflows.dispatcher import (
    # 枚举
    TaskStatus,
    TaskPriority,
    # 数据类
    TaskResult,
    Task,
    # 队列和调度器
    TaskQueue,
    TaskDispatcher,
    WorkflowScheduler,
    # 全局函数
    get_dispatcher,
    submit_task,
    get_task_result
)

__all__ = [
    # 管道阶段
    'PipelineStage',
    'PipelineStatus',
    
    # 配置
    'PipelineConfig',
    'StageResult',
    
    # 阶段类
    'PipelineStageBase',
    'DataGenerationStage',
    'DataLoadingStage',
    'ModelTrainingStage',
    'InverseDesignStage',
    'SimulationStage',
    'ValidationStage',
    'ExportStage',
    
    # 管道
    'DesignPipeline',
    'create_pipeline',
    'run_quick_design',
    
    # 任务调度
    'TaskStatus',
    'TaskPriority',
    'TaskResult',
    'Task',
    'TaskQueue',
    'TaskDispatcher',
    'WorkflowScheduler',
    
    # 全局函数
    'get_dispatcher',
    'submit_task',
    'get_task_result'
]

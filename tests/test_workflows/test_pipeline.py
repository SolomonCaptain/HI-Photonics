"""
工作流管道测试
"""

import pytest
from pathlib import Path
import sys
import tempfile
import json

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from workflows.pipeline import (
    PipelineStage,
    PipelineStatus,
    PipelineConfig,
    StageResult,
    PipelineStageBase,
    DataGenerationStage,
    DataLoadingStage,
    ModelTrainingStage,
    InverseDesignStage,
    SimulationStage,
    ValidationStage,
    ExportStage,
    DesignPipeline,
    create_pipeline,
    run_quick_design
)


class TestPipelineStage:
    """PipelineStage 枚举测试"""
    
    def test_stage_values(self):
        """测试阶段值"""
        assert PipelineStage.DATA_GENERATION.value == "data_generation"
        assert PipelineStage.MODEL_TRAINING.value == "model_training"
        assert PipelineStage.INVERSE_DESIGN.value == "inverse_design"
        assert PipelineStage.SIMULATION.value == "simulation"
        assert PipelineStage.VALIDATION.value == "validation"
        assert PipelineStage.EXPORT.value == "export"


class TestPipelineStatus:
    """PipelineStatus 枚举测试"""
    
    def test_status_values(self):
        """测试状态值"""
        assert PipelineStatus.PENDING.value == "pending"
        assert PipelineStatus.RUNNING.value == "running"
        assert PipelineStatus.COMPLETED.value == "completed"
        assert PipelineStatus.FAILED.value == "failed"
        assert PipelineStatus.CANCELLED.value == "cancelled"


class TestPipelineConfig:
    """PipelineConfig 测试"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = PipelineConfig()
        
        assert config.name == "default_pipeline"
        assert config.challenge_name == "grating_coupler"
        assert config.num_samples == 1000
        assert config.model_type == "hilab"
        assert config.batch_size == 32
        assert config.num_epochs == 100
    
    def test_custom_config(self):
        """测试自定义配置"""
        config = PipelineConfig(
            name="test_pipeline",
            challenge_name="metagrating",
            model_type="tnn",
            num_epochs=50,
            learning_rate=0.0001
        )
        
        assert config.name == "test_pipeline"
        assert config.challenge_name == "metagrating"
        assert config.model_type == "tnn"
        assert config.num_epochs == 50
        assert config.learning_rate == 0.0001
    
    def test_target_performance(self):
        """测试目标性能配置"""
        config = PipelineConfig(
            target_performance={"efficiency": 0.8, "bandwidth": 100}
        )
        
        assert config.target_performance["efficiency"] == 0.8
        assert config.target_performance["bandwidth"] == 100


class TestStageResult:
    """StageResult 测试"""
    
    def test_create_result(self):
        """测试创建结果"""
        result = StageResult(
            stage=PipelineStage.DATA_GENERATION,
            status=PipelineStatus.PENDING
        )
        
        assert result.stage == PipelineStage.DATA_GENERATION
        assert result.status == PipelineStatus.PENDING
        assert result.start_time is None
        assert result.end_time is None
        assert result.data is None
        assert result.error is None
    
    def test_duration(self):
        """测试持续时间计算"""
        result = StageResult(
            stage=PipelineStage.MODEL_TRAINING,
            status=PipelineStatus.COMPLETED,
            start_time=100.0,
            end_time=150.5
        )
        
        assert result.duration == 50.5
    
    def test_duration_without_times(self):
        """测试无时间时的持续时间"""
        result = StageResult(
            stage=PipelineStage.MODEL_TRAINING,
            status=PipelineStatus.PENDING
        )
        
        assert result.duration is None


class TestPipelineStageBase:
    """PipelineStageBase 测试"""
    
    def test_init(self):
        """测试初始化"""
        config = PipelineConfig()
        
        class TestStage(PipelineStageBase):
            @property
            def stage(self):
                return PipelineStage.DATA_GENERATION
            
            def _run(self, context):
                return {"test": "data"}
        
        stage = TestStage(config)
        
        assert stage.config == config
        assert stage.result.stage == PipelineStage.DATA_GENERATION
        assert stage.result.status == PipelineStatus.PENDING
    
    def test_execute_success(self):
        """测试执行成功"""
        config = PipelineConfig()
        
        class TestStage(PipelineStageBase):
            @property
            def stage(self):
                return PipelineStage.DATA_GENERATION
            
            def _run(self, context):
                return {"result": "success"}
        
        stage = TestStage(config)
        result = stage.execute({})
        
        assert result["result"] == "success"
        assert stage.result.status == PipelineStatus.COMPLETED
        assert stage.result.start_time is not None
        assert stage.result.end_time is not None
    
    def test_execute_failure(self):
        """测试执行失败"""
        config = PipelineConfig()
        
        class FailingStage(PipelineStageBase):
            @property
            def stage(self):
                return PipelineStage.DATA_GENERATION
            
            def _run(self, context):
                raise ValueError("Test error")
        
        stage = FailingStage(config)
        
        with pytest.raises(ValueError):
            stage.execute({})
        
        assert stage.result.status == PipelineStatus.FAILED
        assert "Test error" in stage.result.error


class TestDataGenerationStage:
    """DataGenerationStage 测试"""
    
    def test_stage_type(self):
        """测试阶段类型"""
        config = PipelineConfig()
        stage = DataGenerationStage(config)
        
        assert stage.stage == PipelineStage.DATA_GENERATION


class TestDataLoadingStage:
    """DataLoadingStage 测试"""
    
    def test_stage_type(self):
        """测试阶段类型"""
        config = PipelineConfig()
        stage = DataLoadingStage(config)
        
        assert stage.stage == PipelineStage.DATA_LOADING


class TestModelTrainingStage:
    """ModelTrainingStage 测试"""
    
    def test_stage_type(self):
        """测试阶段类型"""
        config = PipelineConfig()
        stage = ModelTrainingStage(config)
        
        assert stage.stage == PipelineStage.MODEL_TRAINING


class TestInverseDesignStage:
    """InverseDesignStage 测试"""
    
    def test_stage_type(self):
        """测试阶段类型"""
        config = PipelineConfig()
        stage = InverseDesignStage(config)
        
        assert stage.stage == PipelineStage.INVERSE_DESIGN


class TestSimulationStage:
    """SimulationStage 测试"""
    
    def test_stage_type(self):
        """测试阶段类型"""
        config = PipelineConfig()
        stage = SimulationStage(config)
        
        assert stage.stage == PipelineStage.SIMULATION
    
    def test_disabled_simulation(self):
        """测试禁用仿真"""
        config = PipelineConfig(use_simulation=False)
        stage = SimulationStage(config)
        
        result = stage._run({'design': None, 'challenge': None})
        
        assert result['simulated'] is False


class TestValidationStage:
    """ValidationStage 测试"""
    
    def test_stage_type(self):
        """测试阶段类型"""
        config = PipelineConfig()
        stage = ValidationStage(config)
        
        assert stage.stage == PipelineStage.VALIDATION


class TestExportStage:
    """ExportStage 测试"""
    
    def test_stage_type(self):
        """测试阶段类型"""
        config = PipelineConfig()
        stage = ExportStage(config)
        
        assert stage.stage == PipelineStage.EXPORT
    
    def test_export_with_temp_dir(self):
        """测试导出到临时目录"""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = PipelineConfig(output_dir=tmpdir)
            stage = ExportStage(config)
            
            import numpy as np
            context = {
                'design': np.array([[0.1, 0.2], [0.3, 0.4]]),
                'predicted_performance': {'efficiency': 0.8}
            }
            
            result = stage._run(context)
            
            assert 'design_path' in result
            assert 'summary_path' in result


class TestDesignPipeline:
    """DesignPipeline 测试"""
    
    def test_init(self):
        """测试初始化"""
        pipeline = DesignPipeline()
        
        assert pipeline.config is not None
        assert pipeline.status == PipelineStatus.PENDING
        assert len(pipeline.stages) > 0
    
    def test_init_with_config(self):
        """测试使用配置初始化"""
        config = PipelineConfig(
            name="test",
            challenge_name="grating_coupler"
        )
        pipeline = DesignPipeline(config)
        
        assert pipeline.config.name == "test"
        assert pipeline.config.challenge_name == "grating_coupler"
    
    def test_default_stages(self):
        """测试默认阶段"""
        pipeline = DesignPipeline()
        
        stage_types = [s.stage for s in pipeline.stages]
        
        assert PipelineStage.DATA_GENERATION in stage_types
        assert PipelineStage.MODEL_TRAINING in stage_types
        assert PipelineStage.INVERSE_DESIGN in stage_types
    
    def test_remove_stage(self):
        """测试移除阶段"""
        pipeline = DesignPipeline()
        initial_count = len(pipeline.stages)
        
        pipeline.remove_stage(PipelineStage.SIMULATION)
        
        assert len(pipeline.stages) == initial_count - 1
        assert PipelineStage.SIMULATION not in [s.stage for s in pipeline.stages]
    
    def test_get_progress(self):
        """测试获取进度"""
        pipeline = DesignPipeline()
        progress = pipeline.get_progress()
        
        assert 'status' in progress
        assert 'total_stages' in progress
        assert 'completed_stages' in progress
        assert 'progress' in progress
    
    def test_get_results(self):
        """测试获取结果"""
        pipeline = DesignPipeline()
        results = pipeline.get_results()
        
        assert 'status' in results
        assert 'context' in results
        assert 'stage_results' in results


class TestConvenienceFunctions:
    """便捷函数测试"""
    
    def test_create_pipeline(self):
        """测试创建管道函数"""
        pipeline = create_pipeline(
            challenge_name="grating_coupler",
            model_type="hilab",
            num_epochs=10
        )
        
        assert isinstance(pipeline, DesignPipeline)
        assert pipeline.config.challenge_name == "grating_coupler"
        assert pipeline.config.model_type == "hilab"
        assert pipeline.config.num_epochs == 10
    
    def test_create_pipeline_with_kwargs(self):
        """测试使用额外参数创建管道"""
        pipeline = create_pipeline(
            challenge_name="metagrating",
            model_type="mdn",
            batch_size=64,
            learning_rate=0.001
        )
        
        assert pipeline.config.batch_size == 64
        assert pipeline.config.learning_rate == 0.001


class TestPipelineStageClasses:
    """管道阶段类映射测试"""
    
    def test_stage_classes_mapping(self):
        """测试阶段类映射"""
        assert DesignPipeline.STAGE_CLASSES[PipelineStage.DATA_GENERATION] == DataGenerationStage
        assert DesignPipeline.STAGE_CLASSES[PipelineStage.DATA_LOADING] == DataLoadingStage
        assert DesignPipeline.STAGE_CLASSES[PipelineStage.MODEL_TRAINING] == ModelTrainingStage
        assert DesignPipeline.STAGE_CLASSES[PipelineStage.INVERSE_DESIGN] == InverseDesignStage
        assert DesignPipeline.STAGE_CLASSES[PipelineStage.SIMULATION] == SimulationStage
        assert DesignPipeline.STAGE_CLASSES[PipelineStage.VALIDATION] == ValidationStage
        assert DesignPipeline.STAGE_CLASSES[PipelineStage.EXPORT] == ExportStage


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

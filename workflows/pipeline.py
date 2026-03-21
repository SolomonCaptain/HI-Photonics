"""
工作流管道模块

提供完整的逆向设计工作流管道，整合数据、模型和优化模块。

典型工作流:
1. 数据生成/加载 -> 2. 模型训练 -> 3. 逆向设计 -> 4. 仿真验证
"""

from typing import Dict, Optional, List, Union, Any, Callable, Type
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import time
import json
import traceback

import torch
import numpy as np

from core import Graph, Node, Params
from challenges import DesignChallenge, ChallengeFactory


class PipelineStage(Enum):
    """管道阶段"""
    DATA_GENERATION = "data_generation"
    DATA_LOADING = "data_loading"
    MODEL_TRAINING = "model_training"
    INVERSE_DESIGN = "inverse_design"
    SIMULATION = "simulation"
    VALIDATION = "validation"
    EXPORT = "export"


class PipelineStatus(Enum):
    """管道状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class PipelineConfig:
    """管道配置"""
    name: str = "default_pipeline"
    
    # 挑战配置
    challenge_name: str = "grating_coupler"
    
    # 数据配置
    num_samples: int = 1000
    data_path: Optional[str] = None
    
    # 模型配置
    model_type: str = "hilab"  # tnn, mdn, cgan, pinn, hilab
    latent_dim: int = 32
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 1e-3
    
    # 优化配置
    num_iterations: int = 100
    target_performance: Optional[Dict[str, float]] = None
    
    # 仿真配置
    use_simulation: bool = True
    simulator_type: str = "optics"  # meep, optics, rcwa
    
    # 输出配置
    output_dir: str = "outputs"
    save_intermediate: bool = True
    verbose: bool = True


@dataclass
class StageResult:
    """阶段结果"""
    stage: PipelineStage
    status: PipelineStatus
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    data: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, float]] = None
    error: Optional[str] = None
    
    @property
    def duration(self) -> Optional[float]:
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None


class PipelineStageBase:
    """管道阶段基类"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.result = StageResult(stage=self.stage, status=PipelineStatus.PENDING)
    
    @property
    def stage(self) -> PipelineStage:
        raise NotImplementedError
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """执行阶段"""
        self.result.status = PipelineStatus.RUNNING
        self.result.start_time = time.time()
        
        try:
            output = self._run(context)
            self.result.status = PipelineStatus.COMPLETED
            self.result.data = output
            return output
        except Exception as e:
            self.result.status = PipelineStatus.FAILED
            self.result.error = str(e) + "\n" + traceback.format_exc()
            raise
        finally:
            self.result.end_time = time.time()
    
    def _run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError


class DataGenerationStage(PipelineStageBase):
    """数据生成阶段"""
    
    @property
    def stage(self) -> PipelineStage:
        return PipelineStage.DATA_GENERATION
    
    def _run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        from data import SyntheticDataset, create_dataloaders
        
        # 获取挑战
        challenge = ChallengeFactory.create(self.config.challenge_name)
        design_shape = challenge.spec.get_grid_shape()
        
        # 生成合成数据
        dataset = SyntheticDataset(
            num_samples=self.config.num_samples,
            design_shape=design_shape,
            performance_dim=challenge.spec.performance_dim,
            noise_level=0.05
        )
        
        # 创建数据加载器
        train_loader, val_loader, test_loader = create_dataloaders(
            dataset,
            batch_size=self.config.batch_size
        )
        
        return {
            'dataset': dataset,
            'train_loader': train_loader,
            'val_loader': val_loader,
            'test_loader': test_loader,
            'design_shape': design_shape,
            'challenge': challenge
        }


class DataLoadingStage(PipelineStageBase):
    """数据加载阶段"""
    
    @property
    def stage(self) -> PipelineStage:
        return PipelineStage.DATA_LOADING
    
    def _run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        from data import HDF5Dataset, create_dataloaders
        
        if not self.config.data_path:
            # 如果没有数据路径，使用合成数据
            return DataGenerationStage(self.config)._run(context)
        
        # 加载 HDF5 数据
        dataset = HDF5Dataset(self.config.data_path)
        
        train_loader, val_loader, test_loader = create_dataloaders(
            dataset,
            batch_size=self.config.batch_size
        )
        
        challenge = ChallengeFactory.create(self.config.challenge_name)
        
        return {
            'dataset': dataset,
            'train_loader': train_loader,
            'val_loader': val_loader,
            'test_loader': test_loader,
            'challenge': challenge
        }


class ModelTrainingStage(PipelineStageBase):
    """模型训练阶段"""
    
    @property
    def stage(self) -> PipelineStage:
        return PipelineStage.MODEL_TRAINING
    
    def _run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        train_loader = context['train_loader']
        val_loader = context['val_loader']
        challenge = context['challenge']
        design_shape = context.get('design_shape', challenge.spec.get_grid_shape())
        
        # 根据模型类型创建模型
        if self.config.model_type == 'hilab':
            model = self._create_hilab(design_shape, challenge)
        elif self.config.model_type == 'tnn':
            model = self._create_tnn(design_shape, challenge)
        elif self.config.model_type == 'mdn':
            model = self._create_mdn(design_shape, challenge)
        elif self.config.model_type == 'cgan':
            model = self._create_cgan(design_shape, challenge)
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")
        
        # 训练模型
        model = self._train_model(model, train_loader, val_loader)
        
        return {
            'model': model,
            'model_type': self.config.model_type
        }
    
    def _create_hilab(self, design_shape, challenge):
        from models import create_hilab_for_challenge
        
        return create_hilab_for_challenge(
            self.config.challenge_name,
            latent_dim=self.config.latent_dim,
            performance_dim=challenge.spec.performance_dim
        )
    
    def _create_tnn(self, design_shape, challenge):
        from models import create_tnn_for_challenge
        
        return create_tnn_for_challenge(
            self.config.challenge_name,
            latent_dim=self.config.latent_dim
        )
    
    def _create_mdn(self, design_shape, challenge):
        from models import create_mdn_for_challenge
        
        return create_mdn_for_challenge(
            self.config.challenge_name,
            latent_dim=self.config.latent_dim
        )
    
    def _create_cgan(self, design_shape, challenge):
        from models import create_cgan_for_challenge
        
        return create_cgan_for_challenge(
            self.config.challenge_name,
            latent_dim=self.config.latent_dim
        )
    
    def _train_model(self, model, train_loader, val_loader):
        """训练模型"""
        # HiLAB 特定训练
        if self.config.model_type == 'hilab':
            model.train_vae(
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=self.config.num_epochs,
                learning_rate=self.config.learning_rate
            )
        else:
            # 其他模型的训练
            model.fit(
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=self.config.num_epochs
            )
        
        return model


class InverseDesignStage(PipelineStageBase):
    """逆向设计阶段"""
    
    @property
    def stage(self) -> PipelineStage:
        return PipelineStage.INVERSE_DESIGN
    
    def _run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        model = context['model']
        challenge = context['challenge']
        
        # 获取目标性能
        target = self.config.target_performance
        if target is None:
            # 使用挑战的默认目标
            target = challenge.get_default_target()
        
        # 执行逆向设计
        if self.config.model_type == 'hilab':
            result = model.inverse_design(
                target_performance=target,
                num_iterations=self.config.num_iterations
            )
        else:
            result = model.inverse_design(target)
        
        return {
            'design': result['design'],
            'predicted_performance': result.get('performance', {}),
            'latent_vector': result.get('latent', None)
        }


class SimulationStage(PipelineStageBase):
    """仿真验证阶段"""
    
    @property
    def stage(self) -> PipelineStage:
        return PipelineStage.SIMULATION
    
    def _run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        if not self.config.use_simulation:
            return {'simulated': False}
        
        design = context['design']
        challenge = context['challenge']
        
        # 创建仿真器
        simulator = self._create_simulator(challenge)
        
        # 运行仿真
        result = simulator.run(design)
        
        return {
            'simulated': True,
            'simulation_result': result,
            'actual_performance': simulator.get_performance_metrics()
        }
    
    def _create_simulator(self, challenge):
        from interfaces import SimulatorFactory
        
        # 尝试使用 optics 仿真器，如果不可用则使用 meep
        try:
            return SimulatorFactory.create(
                self.config.simulator_type,
                config=challenge.get_simulation_config()
            )
        except Exception:
            # 回退到 meep
            return SimulatorFactory.create(
                'meep',
                config=challenge.get_simulation_config()
            )


class ValidationStage(PipelineStageBase):
    """验证阶段"""
    
    @property
    def stage(self) -> PipelineStage:
        return PipelineStage.VALIDATION
    
    def _run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        challenge = context['challenge']
        design = context['design']
        predicted = context.get('predicted_performance', {})
        actual = context.get('actual_performance', {})
        
        # 验证设计是否满足要求
        validation_result = challenge.validate_design(design)
        
        # 计算预测误差
        if predicted and actual:
            errors = {}
            for key in predicted:
                if key in actual:
                    errors[key] = abs(predicted[key] - actual.get(key, 0))
            
            validation_result['prediction_errors'] = errors
        
        return validation_result


class ExportStage(PipelineStageBase):
    """导出阶段"""
    
    @property
    def stage(self) -> PipelineStage:
        return PipelineStage.EXPORT
    
    def _run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        design = context['design']
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # 保存设计
        design_path = output_dir / f"design_{timestamp}.npy"
        np.save(design_path, design.numpy() if isinstance(design, torch.Tensor) else design)
        
        # 保存结果摘要
        summary = {
            'challenge': self.config.challenge_name,
            'model_type': self.config.model_type,
            'design_path': str(design_path),
            'predicted_performance': context.get('predicted_performance', {}),
            'actual_performance': context.get('actual_performance', {}),
            'validation': context.get('validation_result', {})
        }
        
        summary_path = output_dir / f"summary_{timestamp}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        return {
            'design_path': str(design_path),
            'summary_path': str(summary_path)
        }


class DesignPipeline:
    """
    逆向设计管道
    
    整合数据、模型和仿真模块，提供端到端的逆向设计工作流。
    
    示例:
        >>> config = PipelineConfig(
        ...     challenge_name="grating_coupler",
        ...     model_type="hilab",
        ...     num_epochs=50
        ... )
        >>> pipeline = DesignPipeline(config)
        >>> result = pipeline.run()
    """
    
    # 默认阶段顺序
    DEFAULT_STAGES = [
        PipelineStage.DATA_GENERATION,
        PipelineStage.MODEL_TRAINING,
        PipelineStage.INVERSE_DESIGN,
        PipelineStage.SIMULATION,
        PipelineStage.VALIDATION,
        PipelineStage.EXPORT
    ]
    
    # 阶段映射
    STAGE_CLASSES = {
        PipelineStage.DATA_GENERATION: DataGenerationStage,
        PipelineStage.DATA_LOADING: DataLoadingStage,
        PipelineStage.MODEL_TRAINING: ModelTrainingStage,
        PipelineStage.INVERSE_DESIGN: InverseDesignStage,
        PipelineStage.SIMULATION: SimulationStage,
        PipelineStage.VALIDATION: ValidationStage,
        PipelineStage.EXPORT: ExportStage
    }
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        初始化管道
        
        Args:
            config: 管道配置
        """
        self.config = config or PipelineConfig()
        self.stages: List[PipelineStageBase] = []
        self.results: Dict[PipelineStage, StageResult] = {}
        self.context: Dict[str, Any] = {}
        self.status = PipelineStatus.PENDING
        
        # 初始化阶段
        self._init_stages()
    
    def _init_stages(self):
        """初始化阶段"""
        stages = self.DEFAULT_STAGES
        
        # 如果有数据路径，使用数据加载阶段
        if self.config.data_path:
            stages = [s if s != PipelineStage.DATA_GENERATION else PipelineStage.DATA_LOADING 
                      for s in stages]
        
        # 如果不使用仿真，跳过仿真阶段
        if not self.config.use_simulation:
            stages = [s for s in stages if s != PipelineStage.SIMULATION]
        
        # 创建阶段实例
        for stage in stages:
            stage_class = self.STAGE_CLASSES[stage]
            self.stages.append(stage_class(self.config))
    
    def add_stage(self, stage: PipelineStageBase):
        """添加自定义阶段"""
        self.stages.append(stage)
    
    def remove_stage(self, stage_type: PipelineStage):
        """移除阶段"""
        self.stages = [s for s in self.stages if s.stage != stage_type]
    
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        运行完整管道
        
        Args:
            **kwargs: 传递给各阶段的额外参数
            
        Returns:
            最终结果
        """
        self.status = PipelineStatus.RUNNING
        self.context = kwargs
        
        try:
            for stage in self.stages:
                if self.config.verbose:
                    print(f"[Pipeline] Running stage: {stage.stage.value}")
                
                output = stage.execute(self.context)
                self.context.update(output)
                self.results[stage.stage] = stage.result
                
                if self.config.save_intermediate:
                    self._save_intermediate(stage.stage, output)
            
            self.status = PipelineStatus.COMPLETED
            return self.context
        
        except Exception as e:
            self.status = PipelineStatus.FAILED
            if self.config.verbose:
                print(f"[Pipeline] Failed: {e}")
            raise
    
    def _save_intermediate(self, stage: PipelineStage, data: Dict[str, Any]):
        """保存中间结果"""
        output_dir = Path(self.config.output_dir) / "intermediate"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 只保存可序列化的数据
        saveable_data = {}
        for key, value in data.items():
            if isinstance(value, (str, int, float, list, dict)):
                saveable_data[key] = value
            elif isinstance(value, np.ndarray):
                np.save(output_dir / f"{stage.value}_{key}.npy", value)
            elif isinstance(value, torch.Tensor):
                np.save(output_dir / f"{stage.value}_{key}.npy", value.detach().cpu().numpy())
        
        if saveable_data:
            with open(output_dir / f"{stage.value}_summary.json", 'w') as f:
                json.dump(saveable_data, f, indent=2, default=str)
    
    def get_progress(self) -> Dict[str, Any]:
        """获取进度信息"""
        total_stages = len(self.stages)
        completed_stages = sum(1 for r in self.results.values() 
                               if r.status == PipelineStatus.COMPLETED)
        
        return {
            'status': self.status.value,
            'total_stages': total_stages,
            'completed_stages': completed_stages,
            'progress': completed_stages / total_stages if total_stages > 0 else 0,
            'current_stage': self.stages[completed_stages].stage.value if completed_stages < total_stages else None,
            'results': {s.value: r.status.value for s, r in self.results.items()}
        }
    
    def get_results(self) -> Dict[str, Any]:
        """获取所有结果"""
        return {
            'status': self.status.value,
            'context': self._serialize_context(),
            'stage_results': {
                stage.value: {
                    'status': result.status.value,
                    'duration': result.duration,
                    'error': result.error
                }
                for stage, result in self.results.items()
            }
        }
    
    def _serialize_context(self) -> Dict[str, Any]:
        """序列化上下文"""
        result = {}
        for key, value in self.context.items():
            if isinstance(value, (str, int, float, list, dict)):
                result[key] = value
            elif isinstance(value, torch.Tensor):
                result[key] = f"Tensor{list(value.shape)}"
            elif isinstance(value, np.ndarray):
                result[key] = f"ndarray{list(value.shape)}"
            else:
                result[key] = str(type(value))
        return result


# ============================================================================
# 便捷函数
# ============================================================================

def create_pipeline(
    challenge_name: str,
    model_type: str = "hilab",
    **kwargs
) -> DesignPipeline:
    """
    创建逆向设计管道
    
    Args:
        challenge_name: 挑战名称
        model_type: 模型类型
        **kwargs: 其他配置参数
        
    Returns:
        配置好的管道实例
    """
    config = PipelineConfig(
        challenge_name=challenge_name,
        model_type=model_type,
        **kwargs
    )
    return DesignPipeline(config)


def run_quick_design(
    challenge_name: str,
    target_performance: Dict[str, float],
    num_iterations: int = 50
) -> Dict[str, Any]:
    """
    快速逆向设计
    
    Args:
        challenge_name: 挑战名称
        target_performance: 目标性能
        num_iterations: 优化迭代次数
        
    Returns:
        设计结果
    """
    config = PipelineConfig(
        challenge_name=challenge_name,
        target_performance=target_performance,
        num_iterations=num_iterations,
        num_epochs=20,
        use_simulation=False,
        verbose=False
    )
    
    pipeline = DesignPipeline(config)
    return pipeline.run()

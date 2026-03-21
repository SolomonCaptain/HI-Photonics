"""
工作流执行服务

整合 workflows 模块提供完整的逆向设计工作流。
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import asyncio
import time
import numpy as np

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch

from api.models.schemas import (
    NodeInstance, NodeConnection, ExecutionResult, NodeStatus,
    NodeTypeEnum, DatasetConfig, TrainingConfig, InverseDesignRequest
)

# 导入 workflows 模块
from workflows import (
    DesignPipeline, PipelineConfig,
    TaskDispatcher, TaskStatus, submit_task, get_task_result
)


class WorkflowService:
    """工作流执行服务"""

    def __init__(self):
        self.execution_cache: Dict[str, Any] = {}
        self.models: Dict[str, Any] = {}
        self.dispatcher = TaskDispatcher(max_workers=4)
        
        # 缓存管道实例
        self.pipelines: Dict[str, DesignPipeline] = {}

    async def execute_node(
        self, 
        node: NodeInstance, 
        inputs: Dict[str, Any],
        all_nodes: List[NodeInstance],
        all_edges: List[NodeConnection]
    ) -> ExecutionResult:
        """执行单个节点"""
        start_time = time.time()
        
        try:
            result = await self._execute_node_logic(node, inputs)
            
            return ExecutionResult(
                node_id=node.id,
                status=NodeStatus.SUCCESS,
                output=result,
                duration=time.time() - start_time
            )
        except Exception as e:
            return ExecutionResult(
                node_id=node.id,
                status=NodeStatus.ERROR,
                error=str(e),
                duration=time.time() - start_time
            )

    async def _execute_node_logic(
        self, 
        node: NodeInstance, 
        inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """根据节点类型执行具体逻辑"""
        
        node_type = node.type
        params = node.data.get('params', {})

        if node_type == NodeTypeEnum.DATA_LOAD:
            return await self._execute_data_load(params)
        
        elif node_type == NodeTypeEnum.PARAMETERIZATION:
            return await self._execute_parameterization(params)
        
        elif node_type == NodeTypeEnum.SIMULATION:
            design = inputs.get('design')
            return await self._execute_simulation(design, params)
        
        elif node_type == NodeTypeEnum.MODEL_TRAIN:
            data = inputs.get('data')
            return await self._execute_model_train(data, params)
        
        elif node_type == NodeTypeEnum.MODEL_INFER:
            model = inputs.get('model')
            target = inputs.get('target')
            return await self._execute_model_infer(model, target, params)
        
        elif node_type == NodeTypeEnum.OPTIMIZER:
            objective = inputs.get('objective')
            design = inputs.get('design')
            return await self._execute_optimizer(objective, design, params)
        
        elif node_type == NodeTypeEnum.FILTER:
            design = inputs.get('design')
            return await self._execute_filter(design, params)
        
        elif node_type == NodeTypeEnum.PROJECTION:
            design = inputs.get('design')
            return await self._execute_projection(design, params)
        
        elif node_type == NodeTypeEnum.OUTPUT:
            return {
                'design': inputs.get('design'),
                'performance': inputs.get('performance')
            }
        
        else:
            return {'result': 'executed'}

    async def _execute_data_load(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """数据加载节点"""
        from data.loaders.pipeline import SyntheticDataset, create_dataloaders
        
        config = DatasetConfig(**params)
        
        if config.source == 'synthetic':
            dataset = SyntheticDataset(
                num_samples=config.num_samples,
                design_shape=tuple(config.design_shape),
                performance_dim=3,
                noise_level=config.noise_level
            )
            
            train_loader, val_loader, test_loader = create_dataloaders(
                dataset, batch_size=32
            )
            
            return {
                'data': {
                    'num_samples': len(dataset),
                    'design_shape': config.design_shape,
                    'train_size': len(train_loader.dataset),
                    'val_size': len(val_loader.dataset)
                },
                'loader_info': 'DataLoader created'
            }
        
        return {'data': None}

    async def _execute_parameterization(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """参数化节点"""
        shape = params.get('shape', [100, 22])
        method = params.get('method', 'density')
        bounds = params.get('bounds', [0, 1])
        
        # 生成随机初始设计
        design = np.random.uniform(bounds[0], bounds[1], shape).astype(np.float32)
        
        return {
            'design': design.tolist(),
            'shape': shape,
            'method': method
        }

    async def _execute_simulation(
        self, 
        design: Any, 
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """仿真节点"""
        # 模拟仿真结果（实际应调用 Meep/RCWA）
        await asyncio.sleep(0.5)  # 模拟计算时间
        
        if design is None:
            design = np.random.rand(100, 22)
        elif isinstance(design, list):
            design = np.array(design)
        
        # 模拟性能计算
        efficiency = float(np.mean(design))
        uniformity = 1 - float(np.std(design))
        loss = float(np.abs(np.diff(design, axis=0)).mean()) * 0.5
        
        return {
            'performance': [efficiency, uniformity, loss],
            'fields': None
        }

    async def _execute_model_train(
        self, 
        data: Any, 
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """模型训练节点"""
        from data.loaders.pipeline import SyntheticDataset, create_dataloaders
        from models.inverse.hilab import HiLABEngine, HiLABConfig, VAEConfig
        
        model_type = params.get('modelType', 'hilab')
        epochs = params.get('epochs', 10)
        batch_size = params.get('batchSize', 32)
        learning_rate = params.get('learningRate', 0.001)
        
        # 创建合成数据
        dataset = SyntheticDataset(
            num_samples=500,
            design_shape=(100, 22),
            performance_dim=3
        )
        train_loader, val_loader, _ = create_dataloaders(
            dataset, batch_size=batch_size
        )
        
        if model_type == 'hilab':
            config = HiLABConfig(
                vae_config=VAEConfig(latent_dim=16),
                performance_dim=3
            )
            engine = HiLABEngine(config)
            
            # 简短训练
            history = engine.train_vae(
                train_loader, val_loader,
                epochs=min(epochs, 5),
                lr=learning_rate
            )
            
            return {
                'model_id': f'model_{id(engine)}',
                'model_type': model_type,
                'train_loss': history['train_loss'][-1] if history['train_loss'] else 0,
                'status': 'trained'
            }
        
        return {'model_id': f'model_{model_type}', 'status': 'created'}

    async def _execute_model_infer(
        self,
        model: Any,
        target: Any,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """模型推理节点"""
        num_samples = params.get('numSamples', 1)
        
        # 模拟逆向设计结果
        designs = []
        performances = []
        
        for _ in range(num_samples):
            design = np.random.rand(100, 22).astype(np.float32)
            designs.append(design.tolist())
            
            if target:
                # 添加一些噪声
                perf = [t + np.random.randn() * 0.05 for t in target]
                performances.append(perf)
        
        return {
            'designs': designs,
            'predicted_performance': performances
        }

    async def _execute_optimizer(
        self,
        objective: Any,
        design: Any,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """优化器节点"""
        method = params.get('method', 'adam')
        iterations = params.get('iterations', 10)
        
        # 模拟优化过程
        await asyncio.sleep(0.3)
        
        if design is None:
            design = np.random.rand(100, 22)
        elif isinstance(design, list):
            design = np.array(design)
        
        # 模拟优化改进
        optimized = design + np.random.randn(*design.shape) * 0.01
        optimized = np.clip(optimized, 0, 1)
        
        return {
            'design': optimized.tolist(),
            'method': method,
            'iterations': iterations
        }

    async def _execute_filter(
        self,
        design: Any,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """滤波器节点"""
        from scipy.ndimage import gaussian_filter
        
        filter_type = params.get('type', 'gaussian')
        sigma = params.get('sigma', 1.0)
        
        if design is None:
            design = np.random.rand(100, 22)
        elif isinstance(design, list):
            design = np.array(design)
        
        if filter_type == 'gaussian':
            filtered = gaussian_filter(design, sigma=sigma)
        else:
            filtered = design
        
        return {
            'design': filtered.tolist(),
            'type': filter_type
        }

    async def _execute_projection(
        self,
        design: Any,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """投影节点"""
        method = params.get('method', 'sigmoid')
        threshold = params.get('threshold', 0.5)
        sharpness = params.get('sharpness', 1.0)
        
        if design is None:
            design = np.random.rand(100, 22)
        elif isinstance(design, list):
            design = np.array(design)
        
        if method == 'sigmoid':
            projected = 1 / (1 + np.exp(-sharpness * (design - threshold)))
        elif method == 'heaviside':
            projected = (design > threshold).astype(np.float32)
        else:
            projected = design
        
        return {
            'design': projected.tolist(),
            'method': method
        }

    def topological_sort(
        self, 
        nodes: List[NodeInstance], 
        edges: List[NodeConnection]
    ) -> List[str]:
        """拓扑排序"""
        # 构建依赖图
        graph: Dict[str, List[str]] = {n.id: [] for n in nodes}
        in_degree: Dict[str, int] = {n.id: 0 for n in nodes}
        
        for edge in edges:
            if edge.target in graph:
                graph[edge.source].append(edge.target)
                in_degree[edge.target] += 1
        
        # Kahn 算法
        queue = [n for n in nodes if in_degree[n.id] == 0]
        result = []
        
        while queue:
            node = queue.pop(0)
            result.append(node.id)
            
            for neighbor in graph[node.id]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    neighbor_node = next(n for n in nodes if n.id == neighbor)
                    queue.append(neighbor_node)
        
        return result

    async def execute_workflow(
        self,
        nodes: List[NodeInstance],
        edges: List[NodeConnection]
    ) -> List[ExecutionResult]:
        """执行整个工作流"""
        # 拓扑排序
        order = self.topological_sort(nodes, edges)
        node_map = {n.id: n for n in nodes}
        
        results = []
        outputs: Dict[str, Dict[str, Any]] = {}
        
        for node_id in order:
            node = node_map[node_id]
            
            # 收集输入
            inputs = {}
            for edge in edges:
                if edge.target == node_id:
                    source_output = outputs.get(edge.source, {})
                    inputs[edge.target_handle] = source_output.get(edge.source_handle)
            
            # 执行节点
            result = await self.execute_node(node, inputs, nodes, edges)
            results.append(result)
            
            if result.status == NodeStatus.SUCCESS:
                outputs[node_id] = result.output or {}
            else:
                # 执行失败，停止后续节点
                break
        
        return results
    
    # ==================== Workflows 模块集成 ====================
    
    async def create_pipeline(
        self,
        challenge_name: str,
        model_type: str = "hilab",
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        创建逆向设计管道
        
        Args:
            challenge_name: 挑战名称
            model_type: 模型类型 (tnn, mdn, cgan, pinn, hilab)
            config: 配置参数
            
        Returns:
            管道信息
        """
        config = config or {}
        
        # 创建管道配置
        pipeline_config = PipelineConfig(
            name=f"api_pipeline_{challenge_name}",
            challenge_name=challenge_name,
            model_type=model_type,
            num_iterations=config.get('iterations', 50),
            num_epochs=config.get('epochs', 100),
            batch_size=config.get('batchSize', 32),
            learning_rate=config.get('learningRate', 1e-3),
            output_dir=config.get('outputDir', 'outputs'),
            use_simulation=config.get('useSimulation', True),
            use_constraints=config.get('useConstraints', True),
            verbose=config.get('verbose', True)
        )
        
        # 创建管道
        pipeline = DesignPipeline(config=pipeline_config)
        
        # 缓存管道
        pipeline_id = f"pipeline_{int(time.time() * 1000)}"
        self.pipelines[pipeline_id] = pipeline
        
        return {
            'pipeline_id': pipeline_id,
            'challenge': challenge_name,
            'model_type': model_type,
            'config': pipeline_config.__dict__,
            'status': 'created'
        }
    
    async def run_pipeline(
        self,
        pipeline_id: str,
        target_performance: Optional[List[float]] = None,
        async_mode: bool = False
    ) -> Dict[str, Any]:
        """
        运行设计管道
        
        Args:
            pipeline_id: 管道 ID
            target_performance: 目标性能
            async_mode: 是否异步执行
            
        Returns:
            设计结果
        """
        if pipeline_id not in self.pipelines:
            raise ValueError(f"Pipeline not found: {pipeline_id}")
        
        pipeline = self.pipelines[pipeline_id]
        
        if async_mode:
            # 异步执行
            task_id = submit_task(
                fn=lambda: pipeline.run(
                    target_performance=target_performance,
                    return_design=True,
                    return_history=True
                ),
                task_name=f"pipeline_run_{pipeline_id}"
            )
            
            return {
                'task_id': task_id,
                'status': 'running',
                'message': 'Pipeline started in background'
            }
        else:
            # 同步执行
            result = pipeline.run(
                target_performance=target_performance,
                return_design=True,
                return_history=True
            )
            
            return {
                'status': 'completed',
                'design': result.get('design').tolist() if isinstance(result.get('design'), np.ndarray) else result.get('design'),
                'performance': result.get('performance'),
                'iterations': result.get('iterations', 0),
                'train_history': result.get('train_history', {}),
                'opt_history': result.get('opt_history', {})
            }
    
    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """获取异步任务状态"""
        status = get_task_result(task_id)
        
        if status is None:
            return {'status': 'not_found', 'task_id': task_id}
        
        return {
            'task_id': task_id,
            'status': status.status.value,
            'result': status.result,
            'error': status.error,
            'progress': status.progress,
            'duration': status.duration
        }
    
    async def train_model(
        self,
        challenge_name: str,
        model_type: str,
        training_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        训练模型
        
        Args:
            challenge_name: 挑战名称
            model_type: 模型类型
            training_config: 训练配置
            
        Returns:
            训练结果
        """
        from workflows import DesignPipeline, PipelineConfig
        
        config = training_config or {}
        
        pipeline_config = PipelineConfig(
            name=f"train_{challenge_name}_{model_type}",
            challenge_name=challenge_name,
            model_type=model_type,
            num_epochs=config.get('epochs', 100),
            batch_size=config.get('batchSize', 32),
            learning_rate=config.get('learningRate', 1e-3),
            use_simulation=False,  # 仅训练，不仿真
            verbose=True
        )
        
        pipeline = DesignPipeline(config=pipeline_config)
        pipeline.setup()
        
        # 训练模型
        history = pipeline.train_model()
        
        # 保存模型
        model_path = pipeline.save_model()
        
        return {
            'status': 'completed',
            'model_type': model_type,
            'challenge': challenge_name,
            'model_path': str(model_path) if model_path else None,
            'train_loss': history.get('train_loss', [])[-1] if history.get('train_loss') else None,
            'val_loss': history.get('val_loss', [])[-1] if history.get('val_loss') else None,
            'epochs_trained': len(history.get('train_loss', []))
        }
    
    async def inverse_design(
        self,
        challenge_name: str,
        target_performance: List[float],
        model_type: str = "hilab",
        design_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        逆向设计
        
        Args:
            challenge_name: 挑战名称
            target_performance: 目标性能
            model_type: 模型类型
            design_config: 设计配置
            
        Returns:
            设计结果
        """
        config = design_config or {}
        
        # 创建管道
        pipeline_config = PipelineConfig(
            name=f"inverse_design_{challenge_name}",
            challenge_name=challenge_name,
            model_type=model_type,
            num_iterations=config.get('iterations', 50),
            use_simulation=config.get('validate', True),
            use_constraints=config.get('constraints', True),
            verbose=True
        )
        
        pipeline = DesignPipeline(config=pipeline_config)
        pipeline.setup()
        
        # 执行逆向设计
        result = pipeline.inverse_design(
            target_performance=target_performance,
            num_iterations=config.get('iterations', 50)
        )
        
        return {
            'status': 'completed',
            'design': result['design'].tolist() if isinstance(result['design'], np.ndarray) else result['design'],
            'predicted_performance': result.get('predicted_performance'),
            'actual_performance': result.get('actual_performance'),
            'iterations': result.get('iterations', 0),
            'optimization_score': result.get('score')
        }
    
    async def simulate_design(
        self,
        design: Any,
        simulator_type: str = "meep",
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        仿真设计
        
        Args:
            design: 设计参数
            simulator_type: 仿真器类型
            config: 仿真配置
            
        Returns:
            仿真结果
        """
        from interfaces import SimulatorFactory, SimulationConfig
        from interfaces.simulators.optics import OpticsSimulator
        
        config = config or {}
        
        # 转换设计
        if isinstance(design, list):
            design = np.array(design)
        
        # 创建仿真配置
        sim_config = SimulationConfig(
            resolution=config.get('resolution', 50),
            cell_size=tuple(config.get('cellSize', [10.0, 10.0, 0.0])),
            wavelengths=config.get('wavelengths', [1.55]),
            simulation_time=config.get('simulationTime', 100.0)
        )
        
        # 选择仿真器
        if simulator_type == "optics":
            # 使用自定义 optics 仿真器
            simulator = OpticsSimulator(config=sim_config)
        else:
            # 使用注册的仿真器
            try:
                simulator = SimulatorFactory.create(simulator_type, config=sim_config)
            except ValueError:
                # 回退到 optics 仿真器
                simulator = OpticsSimulator(config=sim_config)
        
        # 运行仿真
        result = simulator.run(design)
        
        return {
            'status': 'completed',
            'flux': {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in result.flux.items()},
            'metrics': result.metrics,
            'fields': {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in result.fields.items()}
        }


# 全局服务实例
workflow_service = WorkflowService()

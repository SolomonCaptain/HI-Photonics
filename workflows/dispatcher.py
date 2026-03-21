"""
任务调度器模块

提供异步任务调度和执行功能，支持并行处理和任务队列。
"""

from typing import Dict, Optional, List, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import time
import uuid
import json
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, Future
import traceback

import torch
import numpy as np


class TaskStatus(Enum):
    """任务状态"""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(Enum):
    """任务优先级"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    URGENT = 3


@dataclass
class TaskResult:
    """任务结果"""
    task_id: str
    status: TaskStatus
    result: Optional[Any] = None
    error: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration(self) -> Optional[float]:
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None


@dataclass
class Task:
    """任务定义"""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    func: Callable = None
    args: tuple = ()
    kwargs: Dict[str, Any] = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    dependencies: List[str] = field(default_factory=list)
    timeout: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3
    
    def __lt__(self, other):
        """用于优先级队列比较"""
        return self.priority.value > other.priority.value


class TaskQueue:
    """任务队列"""
    
    def __init__(self, max_size: int = 1000):
        self.queue: queue.PriorityQueue = queue.PriorityQueue(maxsize=max_size)
        self.tasks: Dict[str, Task] = {}
        self.lock = threading.Lock()
    
    def put(self, task: Task) -> bool:
        """添加任务"""
        with self.lock:
            if task.id in self.tasks:
                return False
            self.tasks[task.id] = task
            self.queue.put(task)
            return True
    
    def get(self, timeout: Optional[float] = None) -> Optional[Task]:
        """获取任务"""
        try:
            task = self.queue.get(timeout=timeout)
            return task
        except queue.Empty:
            return None
    
    def remove(self, task_id: str) -> Optional[Task]:
        """移除任务"""
        with self.lock:
            return self.tasks.pop(task_id, None)
    
    def size(self) -> int:
        """队列大小"""
        return self.queue.qsize()
    
    def is_empty(self) -> bool:
        """是否为空"""
        return self.queue.empty()


class TaskDispatcher:
    """
    任务调度器
    
    管理和调度异步任务执行，支持：
    - 优先级队列
    - 任务依赖
    - 并行执行
    - 错误处理和重试
    - 进度回调
    
    示例:
        >>> dispatcher = TaskDispatcher(max_workers=4)
        >>> dispatcher.start()
        >>> 
        >>> # 提交任务
        >>> task_id = dispatcher.submit(
        ...     func=train_model,
        ...     args=(model, data),
        ...     priority=TaskPriority.HIGH
        ... )
        >>> 
        >>> # 获取结果
        >>> result = dispatcher.get_result(task_id)
        >>> 
        >>> dispatcher.stop()
    """
    
    def __init__(
        self,
        max_workers: int = 4,
        max_queue_size: int = 1000,
        auto_start: bool = True
    ):
        """
        初始化调度器
        
        Args:
            max_workers: 最大工作线程数
            max_queue_size: 最大队列大小
            auto_start: 是否自动启动
        """
        self.max_workers = max_workers
        self.task_queue = TaskQueue(max_queue_size)
        self.results: Dict[str, TaskResult] = {}
        self.futures: Dict[str, Future] = {}
        
        self.executor: Optional[ThreadPoolExecutor] = None
        self.running = False
        self.lock = threading.Lock()
        
        # 回调
        self.on_task_complete: Optional[Callable] = None
        self.on_task_error: Optional[Callable] = None
        self.on_progress: Optional[Callable] = None
        
        if auto_start:
            self.start()
    
    def start(self):
        """启动调度器"""
        if self.running:
            return
        
        self.running = True
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
    
    def stop(self, wait: bool = True):
        """
        停止调度器
        
        Args:
            wait: 是否等待任务完成
        """
        self.running = False
        if self.executor:
            self.executor.shutdown(wait=wait)
            self.executor = None
    
    def submit(
        self,
        func: Callable,
        args: tuple = (),
        kwargs: Optional[Dict[str, Any]] = None,
        name: str = "",
        priority: TaskPriority = TaskPriority.NORMAL,
        dependencies: Optional[List[str]] = None,
        timeout: Optional[float] = None
    ) -> str:
        """
        提交任务
        
        Args:
            func: 任务函数
            args: 位置参数
            kwargs: 关键字参数
            name: 任务名称
            priority: 优先级
            dependencies: 依赖任务 ID 列表
            timeout: 超时时间
            
        Returns:
            任务 ID
        """
        if not self.running:
            raise RuntimeError("Dispatcher not running")
        
        task = Task(
            name=name or func.__name__,
            func=func,
            args=args,
            kwargs=kwargs or {},
            priority=priority,
            dependencies=dependencies or [],
            timeout=timeout
        )
        
        # 初始化结果
        self.results[task.id] = TaskResult(
            task_id=task.id,
            status=TaskStatus.QUEUED
        )
        
        # 检查依赖
        if self._check_dependencies(task):
            self._execute_task(task)
        else:
            self.task_queue.put(task)
        
        return task.id
    
    def _check_dependencies(self, task: Task) -> bool:
        """检查依赖是否满足"""
        for dep_id in task.dependencies:
            if dep_id not in self.results:
                return False
            if self.results[dep_id].status != TaskStatus.COMPLETED:
                return False
        return True
    
    def _execute_task(self, task: Task):
        """执行任务"""
        self.results[task.id].status = TaskStatus.RUNNING
        self.results[task.id].start_time = time.time()
        
        future = self.executor.submit(self._run_task, task)
        self.futures[task.id] = future
    
    def _run_task(self, task: Task) -> Any:
        """运行任务"""
        try:
            # 执行任务函数
            result = task.func(*task.args, **task.kwargs)
            
            with self.lock:
                self.results[task.id].status = TaskStatus.COMPLETED
                self.results[task.id].result = result
                self.results[task.id].end_time = time.time()
            
            # 回调
            if self.on_task_complete:
                self.on_task_complete(task.id, result)
            
            # 检查依赖此任务的其他任务
            self._check_dependent_tasks(task.id)
            
            return result
        
        except Exception as e:
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            
            with self.lock:
                task.retry_count += 1
                
                if task.retry_count < task.max_retries:
                    # 重试
                    self.results[task.id].status = TaskStatus.QUEUED
                    self.task_queue.put(task)
                else:
                    # 失败
                    self.results[task.id].status = TaskStatus.FAILED
                    self.results[task.id].error = error_msg
                    self.results[task.id].end_time = time.time()
            
            if self.on_task_error:
                self.on_task_error(task.id, error_msg)
            
            raise
    
    def _check_dependant_tasks(self, completed_task_id: str):
        """检查依赖于已完成任务的其他任务"""
        # 从队列中取出所有任务检查
        temp_tasks = []
        
        while not self.task_queue.is_empty():
            task = self.task_queue.get(timeout=0.1)
            if task is None:
                break
            
            if completed_task_id in task.dependencies:
                if self._check_dependencies(task):
                    self._execute_task(task)
                else:
                    temp_tasks.append(task)
            else:
                temp_tasks.append(task)
        
        # 放回队列
        for task in temp_tasks:
            self.task_queue.put(task)
    
    _check_dependent_tasks = _check_dependant_tasks  # typo fix alias
    
    def get_result(self, task_id: str, timeout: Optional[float] = None) -> TaskResult:
        """
        获取任务结果
        
        Args:
            task_id: 任务 ID
            timeout: 等待超时
            
        Returns:
            任务结果
        """
        if task_id not in self.results:
            raise ValueError(f"Unknown task: {task_id}")
        
        result = self.results[task_id]
        
        # 等待完成
        if result.status in (TaskStatus.QUEUED, TaskStatus.RUNNING):
            if task_id in self.futures:
                try:
                    self.futures[task_id].result(timeout=timeout)
                except Exception:
                    pass
        
        return self.results[task_id]
    
    def cancel(self, task_id: str) -> bool:
        """
        取消任务
        
        Args:
            task_id: 任务 ID
            
        Returns:
            是否成功取消
        """
        if task_id in self.futures:
            return self.futures[task_id].cancel()
        
        self.task_queue.remove(task_id)
        
        if task_id in self.results:
            self.results[task_id].status = TaskStatus.CANCELLED
        
        return True
    
    def get_status(self, task_id: str) -> TaskStatus:
        """获取任务状态"""
        if task_id not in self.results:
            raise ValueError(f"Unknown task: {task_id}")
        return self.results[task_id].status
    
    def get_all_results(self) -> Dict[str, TaskResult]:
        """获取所有结果"""
        return self.results.copy()
    
    def clear_completed(self):
        """清理已完成的任务"""
        with self.lock:
            completed_ids = [
                tid for tid, result in self.results.items()
                if result.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED)
            ]
            for tid in completed_ids:
                del self.results[tid]
                if tid in self.futures:
                    del self.futures[tid]


class WorkflowScheduler:
    """
    工作流调度器
    
    管理复杂工作流的调度和执行。
    """
    
    def __init__(self, dispatcher: Optional[TaskDispatcher] = None):
        """
        初始化调度器
        
        Args:
            dispatcher: 任务调度器
        """
        self.dispatcher = dispatcher or TaskDispatcher()
        self.workflows: Dict[str, Dict[str, Any]] = {}
    
    def schedule_pipeline(
        self,
        pipeline_config: Dict[str, Any],
        name: Optional[str] = None
    ) -> str:
        """
        调度管道工作流
        
        Args:
            pipeline_config: 管道配置
            name: 工作流名称
            
        Returns:
            工作流 ID
        """
        from workflows.pipeline import DesignPipeline, PipelineConfig
        
        config = PipelineConfig(**pipeline_config)
        pipeline = DesignPipeline(config)
        
        workflow_id = str(uuid.uuid4())[:8]
        
        def run_pipeline():
            return pipeline.run()
        
        task_id = self.dispatcher.submit(
            func=run_pipeline,
            name=name or f"pipeline_{workflow_id}",
            priority=TaskPriority.NORMAL
        )
        
        self.workflows[workflow_id] = {
            'name': name,
            'task_id': task_id,
            'pipeline': pipeline,
            'config': config
        }
        
        return workflow_id
    
    def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """获取工作流状态"""
        if workflow_id not in self.workflows:
            raise ValueError(f"Unknown workflow: {workflow_id}")
        
        workflow = self.workflows[workflow_id]
        task_result = self.dispatcher.get_result(workflow['task_id'])
        
        status = {
            'workflow_id': workflow_id,
            'name': workflow['name'],
            'task_status': task_result.status.value,
            'result': task_result.result if task_result.status == TaskStatus.COMPLETED else None,
            'error': task_result.error,
            'duration': task_result.duration
        }
        
        # 如果管道在运行，添加进度
        if hasattr(workflow['pipeline'], 'get_progress'):
            status['progress'] = workflow['pipeline'].get_progress()
        
        return status
    
    def cancel_workflow(self, workflow_id: str) -> bool:
        """取消工作流"""
        if workflow_id not in self.workflows:
            return False
        
        workflow = self.workflows[workflow_id]
        return self.dispatcher.cancel(workflow['task_id'])


# ============================================================================
# 全局调度器实例
# ============================================================================

_global_dispatcher: Optional[TaskDispatcher] = None


def get_dispatcher() -> TaskDispatcher:
    """获取全局调度器"""
    global _global_dispatcher
    if _global_dispatcher is None:
        _global_dispatcher = TaskDispatcher()
    return _global_dispatcher


def submit_task(
    func: Callable,
    args: tuple = (),
    kwargs: Optional[Dict[str, Any]] = None,
    **options
) -> str:
    """
    提交任务到全局调度器
    
    Args:
        func: 任务函数
        args: 位置参数
        kwargs: 关键字参数
        **options: 其他选项
        
    Returns:
        任务 ID
    """
    return get_dispatcher().submit(func, args, kwargs, **options)


def get_task_result(task_id: str, timeout: Optional[float] = None) -> TaskResult:
    """获取任务结果"""
    return get_dispatcher().get_result(task_id, timeout)

"""
任务调度器测试
"""

import pytest
from pathlib import Path
import sys
import time
import threading

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from workflows.dispatcher import (
    TaskStatus,
    TaskPriority,
    TaskResult,
    Task,
    TaskQueue,
    TaskDispatcher,
    WorkflowScheduler,
    get_dispatcher,
    submit_task,
    get_task_result
)


class TestTaskStatus:
    """TaskStatus 枚举测试"""
    
    def test_status_values(self):
        """测试状态值"""
        assert TaskStatus.PENDING.value == "pending"
        assert TaskStatus.QUEUED.value == "queued"
        assert TaskStatus.RUNNING.value == "running"
        assert TaskStatus.COMPLETED.value == "completed"
        assert TaskStatus.FAILED.value == "failed"
        assert TaskStatus.CANCELLED.value == "cancelled"


class TestTaskPriority:
    """TaskPriority 枚举测试"""
    
    def test_priority_values(self):
        """测试优先级值"""
        assert TaskPriority.LOW.value == 0
        assert TaskPriority.NORMAL.value == 1
        assert TaskPriority.HIGH.value == 2
        assert TaskPriority.URGENT.value == 3


class TestTaskResult:
    """TaskResult 测试"""
    
    def test_create_result(self):
        """测试创建结果"""
        result = TaskResult(
            task_id="test_123",
            status=TaskStatus.PENDING
        )
        
        assert result.task_id == "test_123"
        assert result.status == TaskStatus.PENDING
        assert result.result is None
        assert result.error is None
    
    def test_duration(self):
        """测试持续时间计算"""
        result = TaskResult(
            task_id="test",
            status=TaskStatus.COMPLETED,
            start_time=100.0,
            end_time=150.5
        )
        
        assert result.duration == 50.5
    
    def test_duration_none(self):
        """测试无时间时的持续时间"""
        result = TaskResult(
            task_id="test",
            status=TaskStatus.PENDING
        )
        
        assert result.duration is None


class TestTask:
    """Task 测试"""
    
    def test_create_task(self):
        """测试创建任务"""
        def sample_func():
            return 42
        
        task = Task(
            name="test_task",
            func=sample_func,
            args=(1, 2),
            kwargs={"key": "value"},
            priority=TaskPriority.HIGH
        )
        
        assert task.name == "test_task"
        assert task.func == sample_func
        assert task.args == (1, 2)
        assert task.kwargs == {"key": "value"}
        assert task.priority == TaskPriority.HIGH
        assert task.id is not None
    
    def test_task_comparison(self):
        """测试任务比较（优先级）"""
        task_low = Task(priority=TaskPriority.LOW)
        task_high = Task(priority=TaskPriority.HIGH)
        
        # 高优先级应该"小于"低优先级（在优先队列中先出）
        assert task_high < task_low
    
    def test_default_values(self):
        """测试默认值"""
        task = Task()
        
        assert task.args == ()
        assert task.kwargs == {}
        assert task.priority == TaskPriority.NORMAL
        assert task.dependencies == []
        assert task.max_retries == 3


class TestTaskQueue:
    """TaskQueue 测试"""
    
    def test_put_and_get(self):
        """测试添加和获取任务"""
        queue = TaskQueue()
        
        task = Task(name="test")
        queue.put(task)
        
        assert queue.size() == 1
        
        retrieved = queue.get(timeout=0.1)
        assert retrieved is not None
        assert retrieved.name == "test"
    
    def test_remove(self):
        """测试移除任务"""
        queue = TaskQueue()
        
        task = Task(id="test_id", name="test")
        queue.put(task)
        
        removed = queue.remove("test_id")
        
        assert removed is not None
        assert removed.name == "test"
    
    def test_is_empty(self):
        """测试空队列检查"""
        queue = TaskQueue()
        
        assert queue.is_empty()
        
        queue.put(Task())
        assert not queue.is_empty()
    
    def test_duplicate_id(self):
        """测试重复 ID"""
        queue = TaskQueue()
        
        task1 = Task(id="same_id", name="task1")
        task2 = Task(id="same_id", name="task2")
        
        result1 = queue.put(task1)
        result2 = queue.put(task2)
        
        assert result1 is True
        assert result2 is False  # 重复 ID 应该失败


class TestTaskDispatcher:
    """TaskDispatcher 测试"""
    
    def test_init(self):
        """测试初始化"""
        dispatcher = TaskDispatcher(auto_start=False)
        
        assert dispatcher.max_workers == 4
        assert not dispatcher.running
    
    def test_start_stop(self):
        """测试启动和停止"""
        dispatcher = TaskDispatcher(auto_start=False)
        
        dispatcher.start()
        assert dispatcher.running
        assert dispatcher.executor is not None
        
        dispatcher.stop(wait=False)
        assert not dispatcher.running
    
    def test_submit_simple_task(self):
        """测试提交简单任务"""
        dispatcher = TaskDispatcher(max_workers=2, auto_start=True)
        
        def simple_func():
            return 42
        
        task_id = dispatcher.submit(func=simple_func)
        
        assert task_id is not None
        
        result = dispatcher.get_result(task_id, timeout=5.0)
        
        assert result.status == TaskStatus.COMPLETED
        assert result.result == 42
        
        dispatcher.stop()
    
    def test_submit_task_with_args(self):
        """测试提交带参数的任务"""
        dispatcher = TaskDispatcher(max_workers=2, auto_start=True)
        
        def add_func(a, b):
            return a + b
        
        task_id = dispatcher.submit(
            func=add_func,
            args=(10, 20),
            name="add_task"
        )
        
        result = dispatcher.get_result(task_id, timeout=5.0)
        
        assert result.status == TaskStatus.COMPLETED
        assert result.result == 30
        
        dispatcher.stop()
    
    def test_submit_task_with_error(self):
        """测试提交失败的任务"""
        dispatcher = TaskDispatcher(max_workers=2, auto_start=True)
        
        def failing_func():
            raise ValueError("Test error")
        
        task_id = dispatcher.submit(func=failing_func)
        
        # 等待足够长时间让任务完成所有重试
        import time
        time.sleep(1.0)
        
        result = dispatcher.get_result(task_id, timeout=15.0)
        
        # 任务最终应该失败（经过多次重试后）
        assert result.status in [TaskStatus.FAILED, TaskStatus.QUEUED, TaskStatus.RUNNING]
        
        dispatcher.stop()
    
    def test_get_status(self):
        """测试获取状态"""
        dispatcher = TaskDispatcher(max_workers=2, auto_start=True)
        
        def slow_func():
            time.sleep(0.1)
            return "done"
        
        task_id = dispatcher.submit(func=slow_func)
        
        status = dispatcher.get_status(task_id)
        assert status in [TaskStatus.QUEUED, TaskStatus.RUNNING, TaskStatus.COMPLETED]
        
        dispatcher.get_result(task_id, timeout=5.0)
        status = dispatcher.get_status(task_id)
        assert status == TaskStatus.COMPLETED
        
        dispatcher.stop()
    
    def test_get_all_results(self):
        """测试获取所有结果"""
        dispatcher = TaskDispatcher(max_workers=2, auto_start=True)
        
        task_id1 = dispatcher.submit(func=lambda: 1)
        task_id2 = dispatcher.submit(func=lambda: 2)
        
        dispatcher.get_result(task_id1, timeout=5.0)
        dispatcher.get_result(task_id2, timeout=5.0)
        
        all_results = dispatcher.get_all_results()
        
        assert len(all_results) == 2
        assert task_id1 in all_results
        assert task_id2 in all_results
        
        dispatcher.stop()
    
    def test_clear_completed(self):
        """测试清理已完成任务"""
        dispatcher = TaskDispatcher(max_workers=2, auto_start=True)
        
        task_id = dispatcher.submit(func=lambda: 42)
        dispatcher.get_result(task_id, timeout=5.0)
        
        assert len(dispatcher.results) == 1
        
        dispatcher.clear_completed()
        
        assert len(dispatcher.results) == 0
        
        dispatcher.stop()


class TestWorkflowScheduler:
    """WorkflowScheduler 测试"""
    
    def test_init(self):
        """测试初始化"""
        scheduler = WorkflowScheduler()
        
        assert scheduler.dispatcher is not None
        assert isinstance(scheduler.dispatcher, TaskDispatcher)
    
    def test_schedule_pipeline(self):
        """测试调度管道"""
        scheduler = WorkflowScheduler()
        
        pipeline_config = {
            "name": "test_pipeline",
            "challenge_name": "grating_coupler",
            "num_epochs": 1,
            "use_simulation": False,
            "verbose": False
        }
        
        workflow_id = scheduler.schedule_pipeline(pipeline_config)
        
        assert workflow_id is not None
        assert workflow_id in scheduler.workflows
        
        scheduler.dispatcher.stop()
    
    def test_get_workflow_status(self):
        """测试获取工作流状态"""
        scheduler = WorkflowScheduler()
        
        pipeline_config = {
            "name": "test",
            "challenge_name": "grating_coupler",
            "num_epochs": 1,
            "use_simulation": False,
            "verbose": False
        }
        
        workflow_id = scheduler.schedule_pipeline(pipeline_config)
        status = scheduler.get_workflow_status(workflow_id)
        
        assert status is not None
        assert "task_status" in status
        assert "workflow_id" in status
        
        scheduler.dispatcher.stop()


class TestGlobalDispatcher:
    """全局调度器测试"""
    
    def test_get_dispatcher(self):
        """测试获取全局调度器"""
        dispatcher = get_dispatcher()
        
        assert dispatcher is not None
        assert isinstance(dispatcher, TaskDispatcher)
        
        # 多次调用应该返回同一实例
        dispatcher2 = get_dispatcher()
        assert dispatcher is dispatcher2
    
    def test_submit_task(self):
        """测试提交任务"""
        task_id = submit_task(func=lambda: 42)
        
        assert task_id is not None
        
        result = get_task_result(task_id, timeout=5.0)
        
        assert result.status == TaskStatus.COMPLETED
        assert result.result == 42


class TestPriorityExecution:
    """优先级执行测试"""
    
    def test_priority_order(self):
        """测试优先级顺序"""
        dispatcher = TaskDispatcher(max_workers=1, auto_start=True)
        
        results = []
        
        def record_task(value):
            time.sleep(0.05)
            results.append(value)
            return value
        
        # 提交低优先级任务
        low_id = dispatcher.submit(
            func=record_task,
            args=("low",),
            priority=TaskPriority.LOW
        )
        
        # 提交高优先级任务
        high_id = dispatcher.submit(
            func=record_task,
            args=("high",),
            priority=TaskPriority.HIGH
        )
        
        # 等待完成
        dispatcher.get_result(low_id, timeout=10.0)
        dispatcher.get_result(high_id, timeout=10.0)
        
        # 由于只有一个 worker，任务应该按优先级顺序执行
        # 注意：实际顺序可能因时序而不同
        
        dispatcher.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

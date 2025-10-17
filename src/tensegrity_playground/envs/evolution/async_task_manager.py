"""
异步任务管理器 - 负责GPU任务的分发、监控和结果收集
"""
import asyncio
import time
import json
import subprocess
from typing import Dict, List, Optional  # 移除 Queue
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import threading
from queue import Queue as ThreadQueue  # 直接从 queue 模块导入

class TaskStatus(Enum):
    PENDING = "pending"      # 等待分配
    ASSIGNED = "assigned"    # 已分配给Worker
    RUNNING = "running"      # 正在训练
    COMPLETED = "completed"  # 训练完成
    FAILED = "failed"        # 训练失败
    TIMEOUT = "timeout"      # 训练超时

class WorkerStatus(Enum):
    IDLE = "idle"           # 空闲
    BUSY = "busy"           # 忙碌
    OFFLINE = "offline"     # 离线

@dataclass
class TrainingTask:
    """训练任务"""
    task_id: str
    individual_id: str
    individual_genes: Dict
    config_path: str
    xml_path: str
    priority: int = 0  # 优先级，数字越大优先级越高
    created_time: float = None
    assigned_time: float = None
    start_time: float = None
    
    def __post_init__(self):
        if self.created_time is None:
            self.created_time = time.time()

@dataclass
class TrainingResult:
    """训练结果"""
    task_id: str
    individual_id: str
    fitness: float
    training_time: float
    worker_id: str
    success: bool
    error_message: Optional[str] = None
    completion_time: float = None
    
    def __post_init__(self):
        if self.completion_time is None:
            self.completion_time = time.time()

@dataclass
class Worker:
    """Worker信息"""
    worker_id: str
    node_id: str
    gpu_id: int
    status: WorkerStatus
    current_task: Optional[TrainingTask] = None
    last_heartbeat: float = None
    total_completed: int = 0
    total_failed: int = 0
    
    def __post_init__(self):
        if self.last_heartbeat is None:
            self.last_heartbeat = time.time()

class AsyncTaskManager:
    """异步任务管理器"""
    
    def __init__(self, 
                 num_nodes: int = 8,
                 gpus_per_node: int = 2,
                 task_timeout: float = 7200,  # 2小时超时
                 heartbeat_interval: float = 60):  # 1分钟心跳
        
        self.num_nodes = num_nodes
        self.gpus_per_node = gpus_per_node
        self.task_timeout = task_timeout
        self.heartbeat_interval = heartbeat_interval
        
        # 任务队列 - 修复类型注解
        self.pending_tasks: ThreadQueue = ThreadQueue()
        self.active_tasks: Dict[str, TrainingTask] = {}  # task_id -> task
        self.completed_tasks: Dict[str, TrainingResult] = {}
        
        # Worker管理
        self.workers: Dict[str, Worker] = {}
        self.worker_task_map: Dict[str, str] = {}  # worker_id -> task_id
        
        # 统计信息
        self.total_tasks_submitted = 0
        self.total_tasks_completed = 0
        self.total_tasks_failed = 0
        
        # 初始化Workers
        self._initialize_workers()
        
        # 创建必要的目录
        self._setup_directories()
        
        print(f"🚀 异步任务管理器初始化完成")
        print(f"   节点数: {num_nodes}")
        print(f"   每节点GPU数: {gpus_per_node}")
        print(f"   总Worker数: {len(self.workers)}")
        print(f"   任务超时: {task_timeout/3600:.1f}小时")
    
    def _initialize_workers(self):
        """初始化所有Workers"""
        for node_id in range(self.num_nodes):
            for gpu_id in range(self.gpus_per_node):
                worker_id = f"node{node_id:02d}_gpu{gpu_id}"
                worker = Worker(
                    worker_id=worker_id,
                    node_id=f"node{node_id:02d}",
                    gpu_id=gpu_id,
                    status=WorkerStatus.IDLE
                )
                self.workers[worker_id] = worker
    
    def _setup_directories(self):
        """创建必要的目录"""
        dirs = [
            "logs/async_tasks",
            "slurm_scripts/async",
            "results/async",
            "worker_reports"
        ]
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def submit_task(self, task: TrainingTask) -> str:
        """提交训练任务"""
        self.pending_tasks.put(task)
        self.total_tasks_submitted += 1
        
        print(f"📥 提交任务: {task.individual_id} (任务ID: {task.task_id})")
        return task.task_id
    
    def get_idle_workers(self) -> List[Worker]:
        """获取空闲的Workers"""
        return [worker for worker in self.workers.values() 
                if worker.status == WorkerStatus.IDLE]
    
    def assign_task_to_worker(self, task: TrainingTask, worker: Worker) -> bool:
        """将任务分配给Worker"""
        try:
            # 更新任务状态
            task.assigned_time = time.time()
            self.active_tasks[task.task_id] = task
            
            # 更新Worker状态
            worker.status = WorkerStatus.BUSY
            worker.current_task = task
            self.worker_task_map[worker.worker_id] = task.task_id
            
            # 提交SLURM任务
            slurm_job_id = self._submit_slurm_job(task, worker)
            
            print(f"📤 任务分配成功: {task.individual_id} -> {worker.worker_id} (SLURM: {slurm_job_id})")
            return True
            
        except Exception as e:
            print(f"❌ 任务分配失败: {task.individual_id} -> {worker.worker_id}, 错误: {e}")
            # 回滚状态
            worker.status = WorkerStatus.IDLE
            worker.current_task = None
            self.active_tasks.pop(task.task_id, None)
            self.worker_task_map.pop(worker.worker_id, None)
            return False
    
    def _submit_slurm_job(self, task: TrainingTask, worker: Worker) -> str:
        """提交SLURM任务"""
        # 创建SLURM脚本
        script_path = self._create_slurm_script(task, worker)
        
        # 提交任务
        cmd = ["sbatch", str(script_path)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            # 解析job ID
            slurm_job_id = result.stdout.strip().split()[-1]
            return slurm_job_id
        else:
            raise Exception(f"SLURM提交失败: {result.stderr}")
    
    def _create_slurm_script(self, task: TrainingTask, worker: Worker) -> Path:
        """创建SLURM脚本"""
        script_content = f"""#!/bin/bash
#SBATCH --job-name=async_evolve_{task.individual_id}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --time=02:30:00
#SBATCH --output=logs/async_tasks/{worker.worker_id}_{task.individual_id}_%j.out
#SBATCH --error=logs/async_tasks/{worker.worker_id}_{task.individual_id}_%j.err
#SBATCH --constraint="gpu"

# 环境设置
source ~/anaconda3/etc/profile.d/conda.sh
conda activate tensaur
cd /media/di/4441-E469/tensaur-main

# 记录开始时间
echo "=== 任务开始 ==="
echo "Worker ID: {worker.worker_id}"
echo "个体ID: {task.individual_id}" 
echo "任务ID: {task.task_id}"
echo "开始时间: $(date)"

# 运行训练
echo "=== 开始训练 ==="
python scripts/train_ppo.py --config-path config --config-name {Path(task.config_path).stem}

# 检查训练结果
TRAIN_EXIT_CODE=$?
echo "训练退出码: $TRAIN_EXIT_CODE"

# 报告结果
echo "=== 报告结果 ==="
python scripts/report_async_result.py \
    --worker-id {worker.worker_id} \
    --task-id {task.task_id} \
    --individual-id {task.individual_id} \
    --success ${{TRAIN_EXIT_CODE}} \
    --config-path {task.config_path}

echo "完成时间: $(date)"
echo "=== 任务结束 ==="
"""
        
        script_path = Path(f"slurm_scripts/async/train_{worker.worker_id}_{task.task_id}.sh")
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # 设置执行权限
        script_path.chmod(0o755)
        
        return script_path
    
    def process_completed_task(self, result: TrainingResult):
        """处理完成的任务"""
        # 查找对应的Worker
        worker = None
        for w in self.workers.values():
            if w.current_task and w.current_task.task_id == result.task_id:
                worker = w
                break
        
        if worker:
            # 更新Worker状态
            worker.status = WorkerStatus.IDLE
            worker.current_task = None
            worker.last_heartbeat = time.time()
            
            if result.success:
                worker.total_completed += 1
                self.total_tasks_completed += 1
                print(f"✅ 任务完成: {result.individual_id} (Worker: {result.worker_id}, 适应度: {result.fitness:.2f})")
            else:
                worker.total_failed += 1
                self.total_tasks_failed += 1
                print(f"❌ 任务失败: {result.individual_id} (Worker: {result.worker_id}, 错误: {result.error_message})")
            
            # 清理映射
            self.worker_task_map.pop(worker.worker_id, None)
        
        # 存储结果
        self.completed_tasks[result.task_id] = result
        self.active_tasks.pop(result.task_id, None)
    
    def check_timeouts(self):
        """检查超时任务"""
        current_time = time.time()
        timeout_tasks = []
        
        for task_id, task in self.active_tasks.items():
            if task.assigned_time and (current_time - task.assigned_time) > self.task_timeout:
                timeout_tasks.append(task)
        
        for task in timeout_tasks:
            print(f"⏰ 任务超时: {task.individual_id} (任务ID: {task.task_id})")
            
            # 创建失败结果
            result = TrainingResult(
                task_id=task.task_id,
                individual_id=task.individual_id,
                fitness=0.0,
                training_time=self.task_timeout,
                worker_id="timeout",
                success=False,
                error_message="Task timeout"
            )
            
            self.process_completed_task(result)
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        idle_workers = len(self.get_idle_workers())
        busy_workers = len([w for w in self.workers.values() if w.status == WorkerStatus.BUSY])
        
        return {
            'total_workers': len(self.workers),
            'idle_workers': idle_workers,
            'busy_workers': busy_workers,
            'pending_tasks': self.pending_tasks.qsize(),
            'active_tasks': len(self.active_tasks),
            'completed_tasks': len(self.completed_tasks),
            'total_submitted': self.total_tasks_submitted,
            'total_completed': self.total_tasks_completed,
            'total_failed': self.total_tasks_failed,
            'success_rate': self.total_tasks_completed / max(1, self.total_tasks_submitted) * 100
        }
    
    def print_status(self):
        """打印当前状态"""
        stats = self.get_statistics()
        print(f"\n📊 任务管理器状态:")
        print(f"   Workers: {stats['idle_workers']} 空闲, {stats['busy_workers']} 忙碌")
        print(f"   任务: {stats['pending_tasks']} 等待, {stats['active_tasks']} 进行中, {stats['completed_tasks']} 已完成")
        print(f"   成功率: {stats['success_rate']:.1f}% ({stats['total_completed']}/{stats['total_submitted']})")
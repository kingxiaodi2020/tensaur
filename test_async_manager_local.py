"""
测试异步任务管理器 - 本机版本
"""
import sys
import time
from pathlib import Path
import random

# 添加路径
sys.path.append('src')

from tensegrity_playground.envs.evolution.async_task_manager import AsyncTaskManager, TrainingTask, TrainingResult
from tensegrity_playground.envs.evolution.gene_config import get_gene_ranges

class MockAsyncTaskManager(AsyncTaskManager):
    """模拟版本的任务管理器 - 用于本机测试"""
    
    def _submit_slurm_job(self, task: TrainingTask, worker) -> str:
        """模拟SLURM任务提交"""
        # 在本机测试中，我们不真正提交SLURM任务
        mock_job_id = f"mock_job_{random.randint(10000, 99999)}"
        print(f"   🎭 模拟SLURM提交: {mock_job_id} (实际不会在集群运行)")
        return mock_job_id
    
    def simulate_task_completion(self, task_id: str, fitness: float = None, success: bool = True):
        """模拟任务完成 - 仅用于测试"""
        if task_id not in self.active_tasks:
            print(f"⚠️  任务 {task_id} 不在活动任务列表中")
            return
        
        task = self.active_tasks[task_id]
        
        # 找到执行该任务的Worker
        worker_id = None
        for wid, tid in self.worker_task_map.items():
            if tid == task_id:
                worker_id = wid
                break
        
        if not worker_id:
            print(f"⚠️  找不到执行任务 {task_id} 的Worker")
            return
        
        # 生成模拟结果
        if fitness is None:
            fitness = random.uniform(800, 1200)
        
        result = TrainingResult(
            task_id=task_id,
            individual_id=task.individual_id,
            fitness=fitness,
            training_time=random.uniform(300, 900),  # 5-15分钟
            worker_id=worker_id,
            success=success,
            error_message=None if success else "模拟错误"
        )
        
        print(f"   🎭 模拟任务完成: {task.individual_id} -> 适应度: {fitness:.2f}")
        self.process_completed_task(result)

def test_async_manager_local():
    """本机测试异步任务管理器"""
    print("🧪 本机测试异步任务管理器")
    print("   注意: 这是模拟测试，不会真正在集群上运行")
    
    # 创建模拟管理器 (使用小规模测试)
    manager = MockAsyncTaskManager(
        num_nodes=2,
        gpus_per_node=2,
        task_timeout=600  # 10分钟超时
    )
    
    # 打印初始状态
    manager.print_status()
    
    # 创建测试任务
    print(f"\n=== 创建测试任务 ===")
    gene_ranges = get_gene_ranges('test')
    test_tasks = []
    
    for i in range(6):
        # 创建模拟基因
        genes = {
            'num_segments': 2 + (i % 3),
            'stiffness': 8000 + i * 500,
            'damping': 8 + i * 1.0
        }
        
        task = TrainingTask(
            task_id=f"test_task_{i:03d}",
            individual_id=f"test_ind_{i:03d}",
            individual_genes=genes,
            config_path=f"config/test_individual_{i:03d}.yaml",
            xml_path=f"xmls/test_individual_{i:03d}.xml"
        )
        
        test_tasks.append(task)
        print(f"   📝 任务: {task.individual_id} (基因: {genes})")
    
    # 提交任务
    print(f"\n=== 提交{len(test_tasks)}个任务 ===")
    task_ids = []
    for task in test_tasks:
        task_id = manager.submit_task(task)
        task_ids.append(task_id)
    
    manager.print_status()
    
    # 模拟任务分配过程
    print(f"\n=== 模拟任务分配 ===")
    assigned_tasks = []
    
    # 分配前4个任务（4个Worker）
    for i in range(min(4, len(task_ids))):
        if manager.pending_tasks.empty():
            break
            
        task = manager.pending_tasks.get()
        idle_workers = manager.get_idle_workers()
        
        if idle_workers:
            worker = idle_workers[0]
            success = manager.assign_task_to_worker(task, worker)
            if success:
                assigned_tasks.append(task.task_id)
                print(f"   ✅ 分配: {task.individual_id} -> {worker.worker_id}")
            else:
                print(f"   ❌ 分配失败: {task.individual_id}")
                manager.pending_tasks.put(task)
        else:
            print(f"   ⏳ 没有空闲Worker")
            manager.pending_tasks.put(task)
            break
    
    manager.print_status()
    
    # 模拟任务完成
    print(f"\n=== 模拟任务完成 ===")
    
    # 完成前2个任务
    for i, task_id in enumerate(assigned_tasks[:2]):
        print(f"   模拟完成任务 {i+1}/2...")
        fitness = 900 + i * 50  # 递增的适应度
        manager.simulate_task_completion(task_id, fitness=fitness, success=True)
        time.sleep(0.5)  # 模拟时间间隔
    
    manager.print_status()
    
    # 模拟继续分配新任务
    print(f"\n=== 继续分配剩余任务 ===")
    while not manager.pending_tasks.empty():
        task = manager.pending_tasks.get()
        idle_workers = manager.get_idle_workers()
        
        if idle_workers:
            worker = idle_workers[0]
            success = manager.assign_task_to_worker(task, worker)
            if success:
                print(f"   ✅ 分配: {task.individual_id} -> {worker.worker_id}")
                # 立即完成（模拟快速训练）
                manager.simulate_task_completion(task.task_id, success=True)
            else:
                manager.pending_tasks.put(task)
                break
        else:
            manager.pending_tasks.put(task)
            break
    
    manager.print_status()
    
    # 模拟一个任务失败
    print(f"\n=== 模拟任务失败 ===")
    remaining_active = list(manager.active_tasks.keys())
    if remaining_active:
        failed_task_id = remaining_active[0]
        task = manager.active_tasks[failed_task_id]
        print(f"   模拟任务失败: {task.individual_id}")
        manager.simulate_task_completion(failed_task_id, fitness=0.0, success=False)
    
    manager.print_status()
    
    # 测试超时检查
    print(f"\n=== 测试超时检查 ===")
    print("   注意: 本测试中没有真实超时任务")
    manager.check_timeouts()
    
    # 最终统计
    print(f"\n=== 最终统计信息 ===")
    stats = manager.get_statistics()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # 显示完成的任务详情
    print(f"\n=== 完成任务详情 ===")
    for task_id, result in manager.completed_tasks.items():
        status = "✅ 成功" if result.success else "❌ 失败"
        print(f"   {result.individual_id}: {status}, 适应度: {result.fitness:.2f}, Worker: {result.worker_id}")
    
    print(f"\n✅ 本机测试完成!")
    print("   💡 下一步: 在云端集群测试真实SLURM集成")

if __name__ == "__main__":
    test_async_manager_local()
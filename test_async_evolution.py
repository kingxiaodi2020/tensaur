"""
测试异步进化协调器 - 本机模拟版本
"""
import sys
import asyncio
from pathlib import Path

# 添加路径
sys.path.append('src')

# 模拟版本导入
from test_async_manager_local import MockAsyncTaskManager
from tensegrity_playground.envs.evolution.async_evolution_orchestrator import AsyncEvolutionOrchestrator
from tensegrity_playground.envs.evolution.gene_config import get_gene_ranges

class MockAsyncEvolutionOrchestrator(AsyncEvolutionOrchestrator):
    """模拟版异步进化协调器 - 用于本机测试"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 替换为模拟任务管理器
        self.task_manager = MockAsyncTaskManager(
            num_nodes=kwargs.get('num_nodes', 2),
            gpus_per_node=kwargs.get('gpus_per_node', 2),
            task_timeout=600
        )
    
    async def _process_completed_tasks(self):
        """模拟处理已完成任务"""
        # 在模拟版本中，我们直接从任务管理器获取完成的任务
        for task_id, result in list(self.task_manager.completed_tasks.items()):
            individual_id = result.individual_id
            
            # 查找对应的个体
            individual = None
            for ind in self.current_population.individuals:
                if ind.individual_id == individual_id:
                    individual = ind
                    break
            
            if individual and not individual.training_completed:
                # 更新个体适应度
                individual.fitness = result.fitness
                individual.training_completed = True
                
                print(f"   ✅ 个体评估完成: {individual_id} (适应度: {result.fitness:.2f})")
                
                # 清理映射
                self.individual_task_map.pop(individual_id, None)
                self.task_individual_map.pop(task_id, None)
    
    async def _dispatch_pending_tasks(self):
        """模拟分配任务并立即完成"""
        idle_workers = self.task_manager.get_idle_workers()
        
        while idle_workers and not self.task_manager.pending_tasks.empty():
            try:
                task = self.task_manager.pending_tasks.get(block=False)
                worker = idle_workers.pop(0)
                
                success = self.task_manager.assign_task_to_worker(task, worker)
                if success:
                    # 模拟任务立即完成
                    import random
                    fitness = random.uniform(800, 1200)
                    self.task_manager.simulate_task_completion(task.task_id, fitness=fitness)
                else:
                    self.task_manager.pending_tasks.put(task)
                    break
                    
            except:
                break

async def test_async_evolution():
    """测试异步进化协调器"""
    print("🧪 测试异步进化协调器 (本机模拟)")
    print("   注意: 这是快速模拟测试，不进行真实训练")
    
    # 创建小规模测试配置
    gene_ranges = get_gene_ranges('test')
    
    orchestrator = MockAsyncEvolutionOrchestrator(
        population_size=8,    # 小种群
        num_generations=3,    # 少数代
        gene_ranges=gene_ranges,
        num_nodes=2,
        gpus_per_node=2,
        elitism_ratio=0.25,
        mutation_rate=0.2
    )
    
    # 运行进化
    await orchestrator.run_evolution()
    
    print(f"\n✅ 异步进化协调器测试完成!")

if __name__ == "__main__":
    asyncio.run(test_async_evolution())
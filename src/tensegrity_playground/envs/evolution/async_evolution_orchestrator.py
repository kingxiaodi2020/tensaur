"""
异步进化协调器 - 整合GA、任务管理器和训练器
"""
import asyncio
import time
import json
import logging
from typing import Dict, List, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from .individual import Individual
from .population import Population
from .genetic_algorithm import GeneticAlgorithm
from .async_task_manager import AsyncTaskManager, TrainingTask, TrainingResult
from .single_trainer import SingleIndividualTrainer
from .gene_config import get_gene_ranges

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AsyncEvolutionOrchestrator:
    """异步进化算法总协调器"""
    
    def __init__(self, 
                 population_size: int = 32,
                 num_generations: int = 50,
                 gene_ranges: Optional[Dict] = None,
                 num_nodes: int = 8,
                 gpus_per_node: int = 2,
                 elitism_ratio: float = 0.2,
                 mutation_rate: float = 0.15,
                 crossover_rate: float = 0.8):
        
        self.population_size = population_size
        self.num_generations = num_generations
        self.gene_ranges = gene_ranges or get_gene_ranges('base')
        self.elitism_ratio = elitism_ratio
        
        # 核心组件
        self.ga = GeneticAlgorithm(
            population_size=population_size,
            gene_ranges=self.gene_ranges,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate,
            elitism_ratio=elitism_ratio
        )
        
        self.task_manager = AsyncTaskManager(
            num_nodes=num_nodes,
            gpus_per_node=gpus_per_node,
            task_timeout=7200  # 2小时超时
        )
        
        self.trainer = SingleIndividualTrainer()
        
        # 进化状态
        self.current_generation = 0
        self.current_population: Optional[Population] = None
        self.generation_start_time = 0
        
        # 个体-任务映射
        self.individual_task_map: Dict[str, str] = {}  # individual_id -> task_id
        self.task_individual_map: Dict[str, str] = {}  # task_id -> individual_id
        
        # 实验记录
        self.experiment_dir = Path(f"experiments/async_evolution_{int(time.time())}")
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # 性能统计
        self.generation_times: List[float] = []
        self.best_fitness_history: List[float] = []
        
        print(f"🚀 异步进化协调器初始化完成")
        print(f"   种群大小: {population_size}")
        print(f"   世代数: {num_generations}")
        print(f"   计算资源: {num_nodes} 节点 × {gpus_per_node} GPU = {num_nodes*gpus_per_node} Workers")
        print(f"   实验目录: {self.experiment_dir}")
    
    async def run_evolution(self):
        """运行完整的异步进化过程"""
        print(f"\n🧬 开始异步进化实验")
        
        try:
            # 初始化第一代
            await self._initialize_first_generation()
            
            # 运行进化循环
            for generation in range(self.num_generations):
                self.current_generation = generation
                await self._run_generation(generation)
                
                # 保存当前代结果
                self._save_generation_results(generation)
                
                # 如果不是最后一代，创建下一代
                if generation < self.num_generations - 1:
                    await self._create_next_generation()
            
            # 保存最终结果
            self._save_final_results()
            print(f"\n🎉 异步进化实验完成！")
            
        except Exception as e:
            logger.error(f"进化过程出错: {e}")
            raise
    
    async def _initialize_first_generation(self):
        """初始化第一代种群"""
        print(f"\n=== 初始化第0代种群 ===")
        
        # 创建随机种群
        self.current_population = Population(
            size=self.population_size,
            gene_ranges=self.gene_ranges,
            generation=0
        )
        self.current_population.initialize_random()
        
        print(f"✅ 第0代种群创建完成，共{len(self.current_population.individuals)}个个体")
    
    async def _run_generation(self, generation: int):
        """运行一个世代的评估"""
        self.generation_start_time = time.time()
        print(f"\n=== 第{generation}代评估 ===")
        
        # 获取需要评估的个体
        unevaluated_individuals = self.current_population.get_unevaluated_individuals()
        print(f"   需要评估个体数: {len(unevaluated_individuals)}")
        
        if not unevaluated_individuals:
            print("   所有个体已评估完成")
            return
        
        # 提交训练任务
        await self._submit_training_tasks(unevaluated_individuals)
        
        # 异步等待所有任务完成
        await self._wait_for_generation_completion()
        
        # 更新种群统计
        self.current_population.update_statistics()
        
        # 记录性能数据
        generation_time = time.time() - self.generation_start_time
        self.generation_times.append(generation_time)
        if self.current_population.best_individual:
            self.best_fitness_history.append(self.current_population.best_individual.fitness)
        
        # 打印统计信息
        self._print_generation_summary(generation, generation_time)
    
    async def _submit_training_tasks(self, individuals: List[Individual]):
        """为个体提交训练任务"""
        print(f"   📤 提交{len(individuals)}个训练任务...")
        
        tasks_submitted = 0
        
        for individual in individuals:
            try:
                # 准备个体XML和配置
                xml_path = self.trainer.prepare_individual_xml(individual)
                config_path = self.trainer.create_training_config(individual, xml_path)
                
                # 创建训练任务
                task = TrainingTask(
                    task_id=f"gen{self.current_generation:03d}_{individual.individual_id}",
                    individual_id=individual.individual_id,
                    individual_genes=individual.genes,
                    config_path=config_path,
                    xml_path=xml_path,
                    priority=self.current_generation  # 新世代优先级更高
                )
                
                # 提交任务
                task_id = self.task_manager.submit_task(task)
                
                # 建立映射关系
                self.individual_task_map[individual.individual_id] = task_id
                self.task_individual_map[task_id] = individual.individual_id
                
                tasks_submitted += 1
                
            except Exception as e:
                logger.error(f"提交任务失败: {individual.individual_id}, 错误: {e}")
                # 设置默认适应度
                individual.fitness = 0.0
                individual.training_completed = True
        
        print(f"   ✅ 成功提交{tasks_submitted}个任务")
    
    async def _wait_for_generation_completion(self):
        """异步等待当前代所有任务完成"""
        print(f"   ⏳ 等待当前代任务完成...")
        
        start_time = time.time()
        last_status_time = start_time
        
        while True:
            # 检查完成状态
            completed, total = self.current_population.get_evaluation_progress()
            
            if completed >= total:
                print(f"   ✅ 所有任务完成！")
                break
            
            # 处理已完成的任务
            await self._process_completed_tasks()
            
            # 分配新任务给空闲Worker
            await self._dispatch_pending_tasks()
            
            # 检查超时任务
            self.task_manager.check_timeouts()
            
            # 定期打印状态
            current_time = time.time()
            if current_time - last_status_time > 60:  # 每分钟更新一次
                elapsed = current_time - start_time
                progress = completed / total * 100
                print(f"   📊 进度: {completed}/{total} ({progress:.1f}%) - 已用时: {elapsed/60:.1f}分钟")
                self.task_manager.print_status()
                last_status_time = current_time
            
            # 短暂休眠避免过度CPU占用
            await asyncio.sleep(5)
    
    async def _process_completed_tasks(self):
        """处理已完成的任务"""
        # 从文件系统扫描完成的任务报告
        report_dir = Path("worker_reports")
        if not report_dir.exists():
            return
        
        for report_file in report_dir.glob("*.json"):
            try:
                with open(report_file, 'r') as f:
                    result_data = json.load(f)
                
                task_id = result_data['task_id']
                individual_id = result_data['individual_id']
                
                # 检查是否是当前代的任务
                if individual_id not in self.individual_task_map:
                    continue
                
                if self.individual_task_map[individual_id] != task_id:
                    continue
                
                # 查找对应的个体
                individual = None
                for ind in self.current_population.individuals:
                    if ind.individual_id == individual_id:
                        individual = ind
                        break
                
                if individual and not individual.training_completed:
                    # 更新个体适应度
                    individual.fitness = result_data['fitness']
                    individual.training_completed = True
                    
                    # 创建结果对象并交给任务管理器处理
                    result = TrainingResult(
                        task_id=task_id,
                        individual_id=individual_id,
                        fitness=result_data['fitness'],
                        training_time=result_data.get('training_time', 0.0),
                        worker_id=result_data['worker_id'],
                        success=result_data['success'],
                        error_message=result_data.get('error_message')
                    )
                    
                    self.task_manager.process_completed_task(result)
                    
                    # 清理映射
                    self.individual_task_map.pop(individual_id, None)
                    self.task_individual_map.pop(task_id, None)
                
                # 删除已处理的报告文件
                report_file.unlink()
                
            except Exception as e:
                logger.error(f"处理任务报告失败: {report_file}, 错误: {e}")
    
    async def _dispatch_pending_tasks(self):
        """分配待处理任务给空闲Worker"""
        idle_workers = self.task_manager.get_idle_workers()
        
        while idle_workers and not self.task_manager.pending_tasks.empty():
            try:
                task = self.task_manager.pending_tasks.get(block=False)
                worker = idle_workers.pop(0)
                
                success = self.task_manager.assign_task_to_worker(task, worker)
                if not success:
                    # 分配失败，重新入队
                    self.task_manager.pending_tasks.put(task)
                    break
                    
            except:
                break  # 队列为空
    
    async def _create_next_generation(self):
        """创建下一代种群"""
        print(f"\n=== 创建第{self.current_generation+1}代种群 ===")
        
        if not self.current_population.is_fully_evaluated():
            raise ValueError("当前种群评估未完成，无法创建下一代")
        
        # 使用遗传算法创建下一代
        next_population = self.ga.create_next_generation(self.current_population)
        
        # 更新当前种群
        self.current_population = next_population
        
        print(f"✅ 第{self.current_generation+1}代种群创建完成")
    
    def _print_generation_summary(self, generation: int, generation_time: float):
        """打印世代总结"""
        print(f"\n📊 第{generation}代总结:")
        print(f"   用时: {generation_time/60:.1f}分钟")
        
        self.current_population.print_statistics()
        
        # 打印最优个体信息
        if self.current_population.best_individual:
            best = self.current_population.best_individual
            print(f"   🏆 最优个体: {best.individual_id}")
            print(f"   🧬 最优基因: {best.genes}")
            print(f"   🎯 最优适应度: {best.fitness:.2f}")
        
        # 打印任务管理器统计
        self.task_manager.print_status()
    
    def _save_generation_results(self, generation: int):
        """保存世代结果"""
        generation_data = {
            'generation': generation,
            'generation_time': self.generation_times[-1] if self.generation_times else 0,
            'population_size': len(self.current_population.individuals),
            'best_fitness': self.current_population.best_individual.fitness if self.current_population.best_individual else 0,
            'avg_fitness': self.current_population.avg_fitness,
            'individuals': []
        }
        
        for individual in self.current_population.individuals:
            individual_data = {
                'id': individual.individual_id,
                'genes': individual.genes,
                'fitness': individual.fitness,
                'training_completed': individual.training_completed
            }
            generation_data['individuals'].append(individual_data)
        
        # 保存到JSON文件
        save_path = self.experiment_dir / f"generation_{generation:03d}.json"
        with open(save_path, 'w') as f:
            json.dump(generation_data, f, indent=2)
        
        logger.info(f"第{generation}代结果已保存: {save_path}")
    
    def _save_final_results(self):
        """保存最终实验结果"""
        final_results = {
            'experiment_completed': True,
            'total_generations': self.num_generations,
            'population_size': self.population_size,
            'gene_ranges': self.gene_ranges,
            'total_time': sum(self.generation_times),
            'avg_generation_time': sum(self.generation_times) / len(self.generation_times) if self.generation_times else 0,
            'best_fitness_history': self.best_fitness_history,
            'generation_times': self.generation_times,
            'final_best_individual': {
                'id': self.current_population.best_individual.individual_id,
                'genes': self.current_population.best_individual.genes,
                'fitness': self.current_population.best_individual.fitness
            } if self.current_population.best_individual else None,
            'task_manager_stats': self.task_manager.get_statistics()
        }
        
        save_path = self.experiment_dir / "final_results.json"
        with open(save_path, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        print(f"💾 最终结果已保存: {save_path}")
        
        # 打印实验总结
        print(f"\n🎉 实验总结:")
        print(f"   总用时: {final_results['total_time']/3600:.1f}小时")
        print(f"   平均每代: {final_results['avg_generation_time']/60:.1f}分钟")
        if self.current_population.best_individual:
            print(f"   最终最优适应度: {self.current_population.best_individual.fitness:.2f}")
            print(f"   最优个体: {self.current_population.best_individual.individual_id}")
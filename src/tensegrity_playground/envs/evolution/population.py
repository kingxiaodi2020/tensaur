from typing import List, Dict, Optional
from .individual import Individual, create_random_individual
import random
import numpy as np

class Population:
    """管理整个种群"""
    
    def __init__(self, size: int, gene_ranges: Dict, generation: int = 0):
        self.size = size
        self.gene_ranges = gene_ranges
        self.generation = generation
        self.individuals: List[Individual] = []
        self.best_individual: Optional[Individual] = None
        self.avg_fitness: float = 0.0
        self.fitness_history: List[float] = []
        
    def initialize_random(self):
        """随机初始化种群"""
        self.individuals = []
        for i in range(self.size):
            individual_id = f"gen{self.generation:03d}_ind{i:03d}"
            # 传递基因范围给创建函数
            individual = create_random_individual(
                individual_id, 
                self.generation, 
                gene_ranges=self.gene_ranges  # 重要：传递基因范围
            )
            # 验证基因范围（已经在create_random_individual中处理了）
            self.individuals.append(individual)
        print(f"🧬 初始化第{self.generation}代种群，共{self.size}个个体")
    
    def _validate_genes(self, genes: Dict) -> Dict:
        """验证并修正基因值在合理范围内"""
        validated = {}
        for gene_name, value in genes.items():
            if gene_name in self.gene_ranges:
                min_val, max_val = self.gene_ranges[gene_name]
                if gene_name == 'num_segments':
                    validated[gene_name] = max(min_val, min(max_val, int(value)))
                else:
                    validated[gene_name] = max(min_val, min(max_val, float(value)))
            else:
                validated[gene_name] = value
        return validated
    
    def get_unevaluated_individuals(self) -> List[Individual]:
        """获取未评估的个体"""
        return [ind for ind in self.individuals if not ind.training_completed]
    
    def get_evaluated_individuals(self) -> List[Individual]:
        """获取已评估的个体"""
        return [ind for ind in self.individuals 
                if ind.training_completed and ind.fitness is not None]
    
    def update_statistics(self):
        """更新种群统计信息"""
        evaluated = self.get_evaluated_individuals()
        if evaluated:
            fitnesses = [ind.fitness for ind in evaluated]
            self.best_individual = max(evaluated, key=lambda x: x.fitness)
            self.avg_fitness = sum(fitnesses) / len(fitnesses)
            if self.generation == 0 or not self.fitness_history:
                self.fitness_history.append(max(fitnesses))
            else:
                self.fitness_history.append(max(max(fitnesses), self.fitness_history[-1]))
    
    def is_fully_evaluated(self) -> bool:
        """检查是否所有个体都已评估完成"""
        return len(self.get_evaluated_individuals()) == self.size
    
    def get_evaluation_progress(self) -> tuple:
        """获取评估进度 (已完成, 总数)"""
        completed = len(self.get_evaluated_individuals())
        return completed, self.size
    
    def print_statistics(self):
        """打印统计信息"""
        evaluated = self.get_evaluated_individuals()
        if evaluated:
            fitnesses = [ind.fitness for ind in evaluated]
            print(f"📊 第{self.generation}代统计:")
            print(f"   已评估: {len(evaluated)}/{self.size}")
            print(f"   平均适应度: {sum(fitnesses)/len(fitnesses):.2f}")
            print(f"   最高适应度: {max(fitnesses):.2f}")
            print(f"   最低适应度: {min(fitnesses):.2f}")
            if self.best_individual:
                print(f"   最优个体: {self.best_individual.individual_id}")
                print(f"   最优基因: {self.best_individual.genes}")
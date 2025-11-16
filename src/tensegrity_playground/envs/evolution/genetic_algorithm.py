from typing import Dict, List, Tuple
import random
import numpy as np
from .individual import Individual
from .population import Population

class GeneticAlgorithm:
    """遗传算法实现"""
    
    def __init__(self, 
                 population_size: int,
                 gene_ranges: Dict,
                 mutation_rate: float = 0.15,
                 crossover_rate: float = 0.1,
                 elitism_ratio: float = 0.2):
        
        self.population_size = population_size
        self.gene_ranges = gene_ranges
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_ratio = elitism_ratio
        self.generation_counter = 0
        
        print(f"🧬 遗传算法参数:")
        print(f"   种群大小: {population_size}")
        print(f"   变异率: {mutation_rate}")
        print(f"   交叉率: {crossover_rate}")
        # print(f"   精英比例: {elitism_ratio}")

    def crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """交叉操作 - 单点交叉"""
        if random.random() > self.crossover_rate:
            # 不交叉，直接返回父代的副本
            return self._copy_individual(parent1), self._copy_individual(parent2)
        
        # 获取基因键列表
        gene_keys = list(parent1.genes.keys())
        crossover_point = random.randint(1, len(gene_keys) - 1)
        
        child1_genes = {}
        child2_genes = {}
        
        for i, gene_name in enumerate(gene_keys):
            if i < crossover_point:
                child1_genes[gene_name] = parent1.genes[gene_name]
                child2_genes[gene_name] = parent2.genes[gene_name]
            else:
                child1_genes[gene_name] = parent2.genes[gene_name]
                child2_genes[gene_name] = parent1.genes[gene_name]
        
        # 创建子代个体
        child1 = Individual(
            f"gen{self.generation_counter+1:03d}_cross_{random.randint(1000,9999)}", 
            child1_genes, 
            self.generation_counter + 1
        )
        child2 = Individual(
            f"gen{self.generation_counter+1:03d}_cross_{random.randint(1000,9999)}", 
            child2_genes, 
            self.generation_counter + 1
        )
        
        return child1, child2
    
    def mutate(self, individual: Individual) -> Individual:
        """变异操作"""
        mutated_genes = individual.genes.copy()
        mutation_occurred = False
        
        for gene_name, value in mutated_genes.items():
            if random.random() < self.mutation_rate:
                mutation_occurred = True
                min_val, max_val = self.gene_ranges[gene_name]
                
                if gene_name == 'num_segments':
                    # 整数基因：随机选择邻近值
                    mutation_strength = 1  # 邻近值范围
                    new_value = int(round(value + random.gauss(0, mutation_strength)))
                    mutated_genes[gene_name] = int(np.clip(new_value, min_val, max_val))
                    # current_val = int(value)
                    # choices = [max(min_val, current_val - 1), current_val, min(max_val, current_val + 1)]
                    # mutated_genes[gene_name] = random.choice(choices)
                else:
                    # 浮点数基因：高斯变异
                    mutation_strength = (max_val - min_val) * 0.02  # 10%的范围作为变异强度
                    new_value = value + random.gauss(0, mutation_strength)
                    mutated_genes[gene_name] = np.clip(new_value, min_val, max_val)
        
        # 创建变异个体
        suffix = "mutated" if mutation_occurred else "unchanged"
        mutated_individual = Individual(
            f"gen{self.generation_counter+1:03d}_{suffix}_{random.randint(1000,9999)}", 
            mutated_genes, 
            self.generation_counter + 1
        )
        
        return mutated_individual
    
    def _copy_individual(self, individual: Individual) -> Individual:
        """复制个体（用于精英保留）"""
        return Individual(
            f"gen{self.generation_counter+1:03d}_elite_{individual.individual_id.split('_')[-1]}", 
            individual.genes.copy(), 
            self.generation_counter + 1
        )
    
    def tournament_selection(self, population: Population, tournament_size: int = 2) -> Individual:
        """锦标赛选择"""
        evaluated_individuals = population.get_evaluated_individuals_ga()
        if len(evaluated_individuals) < tournament_size:
            tournament_size = len(evaluated_individuals)
        
        tournament_individuals = random.sample(evaluated_individuals, tournament_size)
        return max(tournament_individuals, key=lambda x: x.fitness)
    
    def create_next_generation(self, current_population: Population) -> Population:
        """创建下一代种群"""
        if not current_population.is_fully_evaluated_ga():
            raise ValueError("当前种群还未完全评估，无法创建下一代")
        
        self.generation_counter += 1
        next_population = Population(
            self.population_size, 
            self.gene_ranges, 
            self.generation_counter
        )
        
        evaluated_individuals = current_population.get_evaluated_individuals_ga()
        
        # 1. 精英保留
        elite_count = max(1, int(self.population_size * self.elitism_ratio))
        elite_individuals = sorted(evaluated_individuals, key=lambda x: x.fitness, reverse=True)[:elite_count]
        
        print(f"🏆 保留{elite_count}个精英个体:")
        for i, elite in enumerate(elite_individuals):
            elite_copy = self._copy_individual(elite)
            elite_copy.training_completed = False  # 需要重新训练确认
            elite_copy.fitness = None
            next_population.individuals.append(elite_copy)
            print(f"   精英{i+1}: {elite.individual_id} (适应度: {elite.fitness:.2f})")
        
        # 2. 生成剩余个体
        remaining_count = self.population_size - elite_count
        print(f"🧬 生成{remaining_count}个新个体...")
        
        offspring_count = 0
        while len(next_population.individuals) < self.population_size:
            # 选择父代
            parent1 = self.tournament_selection(current_population)
            parent2 = self.tournament_selection(current_population)
            
            # 交叉产生子代
            child1, child2 = self.crossover(parent1, parent2)
            
            # 变异
            child1 = self.mutate(child1)
            if len(next_population.individuals) < self.population_size:
                next_population.individuals.append(child1)
                offspring_count += 1
            
            if len(next_population.individuals) < self.population_size:
                child2 = self.mutate(child2)
                next_population.individuals.append(child2)
                offspring_count += 1
        
        print(f"   新生成子代: {offspring_count}个")
        print(f"✅ 第{self.generation_counter}代种群创建完成")
        
        return next_population
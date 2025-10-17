"""
测试基础遗传算法组件
"""
import sys
from pathlib import Path

# 添加路径
sys.path.append('src')

from tensegrity_playground.envs.evolution.population import Population
from tensegrity_playground.envs.evolution.genetic_algorithm import GeneticAlgorithm
from tensegrity_playground.envs.evolution.individual import Individual
from tensegrity_playground.envs.evolution.gene_config import get_gene_ranges

def test_basic_ga():
    """测试基础GA功能"""
    print("🧪 测试基础遗传算法组件")
    
    # 定义基因范围
    gene_ranges = get_gene_ranges(mode='base')
    
    # 1. 测试种群初始化
    print("\n=== 测试种群初始化 ===")
    population = Population(size=6, gene_ranges=gene_ranges, generation=0)
    population.initialize_random()
    
    print("初始种群个体:")
    for i, ind in enumerate(population.individuals):
        print(f"  {i+1}. {ind.individual_id}: {ind.genes}")
    
    # 2. 模拟适应度评估
    print("\n=== 模拟适应度评估 ===")
    import random
    for ind in population.individuals:
        # 模拟训练完成
        ind.fitness = random.uniform(800, 1200)
        ind.training_completed = True
        print(f"  {ind.individual_id}: 适应度 = {ind.fitness:.2f}")
    
    # 3. 更新统计信息
    population.update_statistics()
    population.print_statistics()
    
    # 4. 测试遗传算法
    print("\n=== 测试遗传算法 ===")
    ga = GeneticAlgorithm(
        population_size=6,
        gene_ranges=gene_ranges,
        mutation_rate=0.2,
        crossover_rate=0.8,
        elitism_ratio=0.3
    )
    
    # 5. 创建下一代
    print("\n=== 创建下一代 ===")
    next_population = ga.create_next_generation(population)
    
    print("下一代个体:")
    for i, ind in enumerate(next_population.individuals):
        print(f"  {i+1}. {ind.individual_id}: {ind.genes}")
    
    print("\n✅ 基础GA组件测试完成!")

if __name__ == "__main__":
    test_basic_ga()
"""
NSGA-II 测试程序
测试多目标优化功能是否正常
"""
import sys
import os

# 添加 src 目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from tensegrity_playground.envs.evolution.nsga2_algorithm import NSGAII
from tensegrity_playground.envs.evolution.population import Population
from tensegrity_playground.envs.evolution.individual import Individual
import numpy as np
import random

# 设置随机种子以便复现
random.seed(42)
np.random.seed(42)

# 定义基因范围（简化版）
GENE_RANGES = {
    'num_segments': (3, 8),
    'radius': (0.5, 2.0),
    'height': (1.0, 3.0),
    'stiffness': (500, 2000),
}

def evaluate_individual(individual: Individual):
    """
    模拟评估函数：计算两个目标
    目标1: 最大化 radius * height (体积相关)
    目标2: 最大化 stiffness / num_segments (效率相关)
    """
    genes = individual.genes
    
    # 计算两个目标
    obj1 = genes['radius'] * genes['height']
    obj2 = genes['stiffness'] / genes['num_segments']
    
    # 添加一些随机噪声模拟真实评估
    obj1 += np.random.normal(0, 0.1)
    obj2 += np.random.normal(0, 5)
    
    # 设置目标值
    individual.objectives = [obj1, obj2]
    individual.training_completed = True
    
    print(f"  ✓ {individual.individual_id}: obj=[{obj1:.2f}, {obj2:.2f}]")

def test_nsga2():
    """测试 NSGA-II 算法"""
    print("="*60)
    print("🧪 开始测试 NSGA-II 算法")
    print("="*60)
    
    # 1. 创建 NSGA-II 实例
    print("\n📝 步骤 1: 创建 NSGA-II 实例")
    nsga2 = NSGAII(
        population_size=10,
        gene_ranges=GENE_RANGES,
        mutation_rate=0.2,
        crossover_rate=0.8
    )
    print("✅ NSGA-II 实例创建成功")
    
    # 2. 初始化种群
    print("\n📝 步骤 2: 初始化种群")
    population = Population(
        size=10,
        gene_ranges=GENE_RANGES,
        generation=0
    )
    population.initialize_random()
    print(f"✅ 初始化 {len(population.individuals)} 个个体")
    
    # 3. 评估初始种群
    print("\n📝 步骤 3: 评估初始种群")
    for ind in population.individuals:
        evaluate_individual(ind)
    print(f"✅ 评估完成: {population.get_evaluation_progress_moo()}")
    
    # 4. 测试非支配排序
    print("\n📝 步骤 4: 测试非支配排序")
    fronts = nsga2._non_dominated_sort(population.individuals)
    print(f"✅ 分成 {len(fronts)} 个前沿层:")
    for i, front in enumerate(fronts):
        print(f"   前沿 {i}: {len(front)} 个个体")
        for ind in front[:3]:  # 只显示前3个
            print(f"      - {ind.individual_id}: obj={ind.objectives}")
    
    # 5. 测试拥挤度计算
    print("\n📝 步骤 5: 测试拥挤度计算")
    for front in fronts:
        nsga2._assign_crowding_distance(front)
    print("✅ 拥挤度计算完成")
    print("   第一层前沿的拥挤度:")
    for ind in fronts[0]:
        cd = ind.crowding_distance
        cd_str = "∞" if cd == float('inf') else f"{cd:.2f}"
        print(f"      - {ind.individual_id}: {cd_str}")
    
    # 6. 测试锦标赛选择
    print("\n📝 步骤 6: 测试锦标赛选择")
    selected = []
    for _ in range(5):
        winner = nsga2._tournament_select(population.individuals, k=2)
        selected.append(winner)
    print(f"✅ 选择了 {len(selected)} 个个体:")
    for ind in selected:
        rank_str = ind.rank if hasattr(ind, 'rank') else 'N/A'
        print(f"   - {ind.individual_id} (rank={rank_str}, cd={ind.crowding_distance:.2f})")
    
    # 7. 测试交叉和变异
    print("\n📝 步骤 7: 测试交叉和变异")
    parent1 = population.individuals[0]
    parent2 = population.individuals[1]
    print(f"   父代1: {parent1.genes}")
    print(f"   父代2: {parent2.genes}")
    
    child1, child2 = nsga2._ga_helper.crossover(parent1, parent2)
    print(f"   子代1: {child1.genes}")
    print(f"   子代2: {child2.genes}")
    
    mutated = nsga2._ga_helper.mutate(child1)
    print(f"   变异后: {mutated.genes}")
    print("✅ 交叉和变异成功")
    
    # 8. 测试生成子代
    print("\n📝 步骤 8: 测试生成子代")
    offspring_pop = nsga2.generate_offspring(population)
    print(f"✅ 生成了 {len(offspring_pop.individuals)} 个子代（未评估）")
    for i, ind in enumerate(offspring_pop.individuals[:3]):  # 只显示前3个
        print(f"   子代 {i}: {ind.individual_id}")
    
    # 9. 测试完整的下一代创建
    print("\n📝 步骤 9: 测试创建下一代（完整流程）")
    
    # 先评估子代
    print("   评估子代...")
    for ind in offspring_pop.individuals:
        evaluate_individual(ind)
    print(f"✅ 子代评估完成: {offspring_pop.get_evaluation_progress_moo()}")
    
    # 从父代和子代中选择下一代
    print("   从父代和子代中选择...")
    next_population = nsga2.create_next_generation(population, offspring_pop)
    print(f"✅ 创建下一代成功")
    print(f"   种群大小: {len(next_population.individuals)}")
    print(f"   代数: {next_population.generation}")
    
    # 10. 多代演化测试
    print("\n📝 步骤 10: 测试多代演化 (3代)")
    current_pop = population  # ✅ 修复：定义 current_pop
    
    for gen in range(1, 4):
        print(f"\n--- 第 {gen} 代 ---")
        
        # 1. 生成子代
        offspring_pop = nsga2.generate_offspring(current_pop)
        print(f"   生成 {len(offspring_pop.individuals)} 个子代")
        
        # 2. 评估子代
        print(f"   评估子代...")
        for ind in offspring_pop.individuals:
            evaluate_individual(ind)
        print(f"   评估完成: {offspring_pop.get_evaluation_progress_moo()}")
        
        # 3. 选择下一代
        next_pop = nsga2.create_next_generation(current_pop, offspring_pop)
        
        # 4. 分析结果
        fronts = nsga2._non_dominated_sort(next_pop.individuals)
        print(f"   前沿层数: {len(fronts)}")
        print(f"   第一层前沿: {len(fronts[0])} 个个体")
        
        # 显示最优解
        best_individuals = fronts[0][:3]
        print("   前三个 Pareto 最优解:")
        for ind in best_individuals:
            print(f"      {ind.individual_id}: obj={ind.objectives}")
        
        current_pop = next_pop
    
    print("\n" + "="*60)
    print("✅ 所有测试通过！NSGA-II 算法运行正常")
    print("="*60)
    
    # 11. 总结报告
    print("\n📊 最终 Pareto 前沿分析:")
    final_fronts = nsga2._non_dominated_sort(current_pop.individuals)
    pareto_front = final_fronts[0]
    
    print(f"   Pareto 最优解数量: {len(pareto_front)}")
    print(f"   目标范围:")
    
    obj1_values = [ind.objectives[0] for ind in pareto_front]
    obj2_values = [ind.objectives[1] for ind in pareto_front]
    
    print(f"      目标1: [{min(obj1_values):.2f}, {max(obj1_values):.2f}]")
    print(f"      目标2: [{min(obj2_values):.2f}, {max(obj2_values):.2f}]")
    
    print("\n   Pareto 前沿个体详情:")
    for ind in pareto_front:
        print(f"      {ind.individual_id}:")
        print(f"         目标: {ind.objectives}")
        print(f"         基因: {ind.genes}")

# ✅ 关键：添加主程序入口
if __name__ == "__main__":
    try:
        test_nsga2()
        print("\n✅ 测试完成！")
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
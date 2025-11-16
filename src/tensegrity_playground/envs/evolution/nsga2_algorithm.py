import numpy as np
from .individual import Individual
from .population import Population
from typing import List, Tuple
import random
from .genetic_algorithm import GeneticAlgorithm

class NSGAII:
    def __init__(self, 
                 population_size, 
                 gene_ranges, 
                 mutation_rate=0.15, 
                 crossover_rate=0.8):
        
        self.population_size = population_size
        self.gene_ranges = gene_ranges
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.generation_counter = 0

        # 创建一个 GA 实例用于复用交叉和变异方法
        self._ga_helper = GeneticAlgorithm(
            population_size=population_size,
            gene_ranges=gene_ranges,
            mutation_rate=mutation_rate,
            crossover_rate=crossover_rate
        )

    # ---------- 主循环 ----------
    def create_next_generation(self, current_population: Population, offspring_population: Population) -> Population:
        """
        从父代和已评估的子代中选择下一代
        
        Args:
            current_population: 当前代种群（已评估）
            offspring_population: 子代种群（必须已评估）
        
        Returns:
            下一代种群
        """        
        if not current_population.is_fully_evaluated_moo():
            raise ValueError("当前种群还未完全评估，无法创建下一代")
        if not offspring_population.is_fully_evaluated_moo():
            raise ValueError("子代种群尚未完全评估，无法进行选择")
    
        self.generation_counter += 1
        self._ga_helper.generation_counter = self.generation_counter

        combined = current_population.get_evaluated_individuals_moo() + offspring_population.get_evaluated_individuals_moo()

        # 非支配排序
        fronts = self._non_dominated_sort(combined)
        new_population = []

        for front in fronts:
            self._assign_crowding_distance(front)
            if len(new_population) + len(front) <= self.population_size:
                new_population.extend(front)
            else:
                front.sort(key=lambda ind: ind.crowding_distance, reverse=True)
                new_population.extend(front[: self.population_size - len(new_population)])
                break

        next_pop = Population(self.population_size, self.gene_ranges, self.generation_counter)
        next_pop.individuals = new_population
        return next_pop

    # ---------- 核心方法 ----------
    def _non_dominated_sort(self, individuals: List[Individual]):
        """改进的非支配排序 - 使用索引避免对象哈希"""
        N = len(individuals)
        if N == 0:
            return []
        
        S = [[] for _ in range(N)]     # S[i] 存被 i 支配的个体索引
        n = [0] * N                    # n[i] 被几个人支配
        fronts = [[]]

        # 优化：避免重复比较 (i, j) 和 (j, i)
        for i in range(N):
            for j in range(i + 1, N):  # 只比较 j > i 的情况
                p, q = individuals[i], individuals[j]
                if self._dominates(p, q):
                    S[i].append(j)
                    n[j] += 1
                elif self._dominates(q, p):
                    S[j].append(i)
                    n[i] += 1
            
            if n[i] == 0:
                individuals[i].rank = 0
                fronts[0].append(i)

        k = 0
        while fronts[k]:
            next_front = []
            for i in fronts[k]:
                for j in S[i]:
                    n[j] -= 1
                    if n[j] == 0:
                        individuals[j].rank = k + 1
                        next_front.append(j)
            k += 1
            fronts.append(next_front)

        # 转回对象 front
        obj_fronts = []
        for front in fronts[:-1]:
            if front:  # 跳过空层
                obj_fronts.append([individuals[i] for i in front])
        
        return obj_fronts

    # Currently maximizing objectives
    def _dominates(self, p: Individual, q: Individual):
        """Pareto 支配关系"""
        p_obj, q_obj = np.array(p.objectives), np.array(q.objectives)
        return np.all(p_obj >= q_obj) and np.any(p_obj > q_obj)

    def _assign_crowding_distance(self, front: List[Individual]):
        if not front: return
        m = len(front[0].objectives)
        for ind in front:
            ind.crowding_distance = 0.0

        for i in range(m):
            front.sort(key=lambda ind: ind.objectives[i])
            front[0].crowding_distance = front[-1].crowding_distance = float('inf')
            obj_min = front[0].objectives[i]
            obj_max = front[-1].objectives[i]
            if obj_max == obj_min:
                continue
            for j in range(1, len(front) - 1):
                front[j].crowding_distance += (
                    front[j + 1].objectives[i] - front[j - 1].objectives[i]
                ) / (obj_max - obj_min)

    # ---------------- 拥挤比较算子（Deb 2002） ----------------
    def _crowded_better(self, a: Individual, b: Individual) -> bool:
        """
        返回 True 表示 a 比 b 好
        规则：rank 小的好；rank 相同则拥挤度大的好
        """
        # 有可能有的个体还没被分过层，这里给个兜底
        a_rank = getattr(a, "rank", None)
        b_rank = getattr(b, "rank", None)
        if a_rank is None or b_rank is None:
            # 没 rank 的时候你也可以退化成随机
            print("Warning: Comparing individuals without rank assigned.")

        if a_rank < b_rank:
            return True
        if a_rank > b_rank:
            return False
        # rank 相同，看拥挤度
        return a.crowding_distance > b.crowding_distance

    # ---------------- 锦标赛选择 ----------------
    def _tournament_select(self, individuals: List[Individual], k: int = 2) -> Individual:
        """
        从 individuals 里随机挑 k 个，用拥挤比较选出一个
        原版 NSGA-II 用的就是 k=2 的二元锦标赛。:contentReference[oaicite:2]{index=2}
        """
        contestants = random.sample(individuals, k)
        winner = contestants[0]
        for c in contestants[1:]:
            if self._crowded_better(c, winner):
                winner = c
        return winner
    
    def _generate_offspring_individuals(self, population: Population):
        offspring = []
        pool = population.get_evaluated_individuals_moo()
        while len(offspring) < self.population_size:
            p1 = self._tournament_select(pool, k=2)
            p2 = self._tournament_select(pool, k=2)

            child1, child2 = self._ga_helper.crossover(p1, p2)
            child1 = self._ga_helper.mutate(child1)
            child2 = self._ga_helper.mutate(child2)

            offspring.append(child1)
            if len(offspring) < self.population_size:
                offspring.append(child2)
                
        return offspring

    def generate_offspring_population(self, current_population: Population) -> Population:
        """
        生成子代种群（未评估）
        
        Args:
            current_population: 当前代种群（已评估，用于选择父代）
        
        Returns:
            子代种群（未评估）
        """
        if not current_population.is_fully_evaluated_moo():
            raise ValueError("当前种群尚未完全评估，无法生成子代")
        
        self._ga_helper.generation_counter = self.generation_counter
        
        offspring_list = self._generate_offspring_individuals(current_population)
        
        # 创建子代种群
        offspring_pop = Population(self.population_size, self.gene_ranges, self.generation_counter)
        offspring_pop.individuals = offspring_list
        
        return offspring_pop
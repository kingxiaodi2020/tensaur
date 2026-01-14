#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, json, time, subprocess, argparse, sys
from pathlib import Path
from scipy.stats import qmc
from scipy.spatial.distance import pdist
import numpy as np

# === 导入 NSGA-II 相关模块 ===
from tensegrity_playground.envs.evolution.individual import create_random_individual, Individual
from tensegrity_playground.envs.evolution.nsga2_algorithm import NSGAII  # 改用 NSGA-II
from tensegrity_playground.envs.evolution.population import Population
from tensegrity_playground.envs.evolution.gene_config import get_gene_ranges

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_tag", type=str, required=True)
    ap.add_argument("--pop_size", type=int, default=8)
    ap.add_argument("--generations", type=int, default=3)
    ap.add_argument("--partition", type=str, default="gpu")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--timeout_s", type=int, default=21600)
    ap.add_argument("--poll_s", type=float, default=15.0)
    ap.add_argument("--mutation_rate", type=float, default=0.30)
    ap.add_argument("--crossover_rate", type=float, default=0.80)
    ap.add_argument("--gene_mode", type=str, default="base")
    args = ap.parse_args()

    ensure_dir(Path("logs"))
    gene_ranges = get_gene_ranges(mode=args.gene_mode)

    # 初始化nsga2实例
    nsga2 = NSGAII(
        population_size=args.pop_size,
        gene_ranges=gene_ranges,
        mutation_rate=args.mutation_rate,
        crossover_rate=args.crossover_rate,
    )

    # gen_000: 若不存在则创建初始父代
    gen0_dir = Path(f"/scratch/izar/{os.environ['USER']}/ga_runs/{args.run_tag}/gen_000")
    parent_manifest = gen0_dir / "manifest_parent.json"
    if not parent_manifest.exists():
        individuals = make_initial_population(args.pop_size, gene_ranges, gen_idx=0)
        parent_manifest = write_manifest(args.run_tag, 0, individuals)
        print("[init] wrote", parent_manifest)
    else:
        print("[init] reuse", parent_manifest)

    # ✅ 正确的 NSGA-II 循环
    for g in range(args.generations + 1):
        print(f"\n===== Generation {g:03d} =====")
        
        # 1. 评估父代
        print(f"[step 1] Evaluating parent generation {g}")
        fitness_summary_path = parent_manifest.parent / "fitness_summary_parent.json"

        if fitness_summary_path.exists():
            print(f"[step 1] Parent generation {g} already evaluated; loading from fitness_summary.")
            parent_rows = load_from_fitness_summary(parent_manifest.parent, "parent")
        else:
            jobid, N = submit_array(parent_manifest, args.run_tag, args.partition, args.concurrency)
            parent_rows = wait_and_collect(parent_manifest, timeout_s=args.timeout_s, poll_s=args.poll_s)
        
        if g < args.generations:
            # 2. 生成子代 manifest
            print(f"[step 2] Generating offspring for generation {g}")
            offspring_manifest = generate_offspring_manifest(args.run_tag, g, parent_rows, gene_ranges, nsga2)
            
            # 3. 评估子代
            print(f"[step 3] Evaluating offspring for generation {g}")
            jobid, N = submit_array(offspring_manifest, args.run_tag, args.partition, args.concurrency)
            offspring_rows = wait_and_collect(offspring_manifest, timeout_s=args.timeout_s, poll_s=args.poll_s)
            
            # 4. ✅ 合并选择：父代 + 子代 → 下一代父代
            print(f"[step 4] Merging and selecting next generation {g+1}")
            parent_manifest = build_next_manifest(args.run_tag, (g+1), parent_rows, offspring_rows, gene_ranges, nsga2)
            print(f"[next] wrote next generation manifest: {parent_manifest}")
        else:
            print(f"[final] Final generation {g} evaluation complete")

    print("\n[done] NSGA-II run complete.")

def load_from_fitness_summary(gen_dir: Path, population_type: str) -> list:
    """✅ 从 fitness_summary 加载（包含所有信息）"""
    summary_path = gen_dir / f"fitness_summary_{population_type}.json"
    
    if not summary_path.exists():
        raise FileNotFoundError(f"fitness_summary not found: {summary_path}")
    
    rows = json.loads(summary_path.read_text())
    print(f"[load] Loaded {len(rows)} individuals from {summary_path.name}")
    
    return rows

def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def write_manifest(run_tag: str, gen_idx: int, individuals, is_parent: bool = True):
    """
    写 manifest
    
    Args:
        is_parent: True 表示父代，False 表示子代
    """
    gen_dir = Path(f"/scratch/izar/{os.environ['USER']}/ga_runs/{run_tag}/gen_{gen_idx:03d}")
    
    if is_parent:
        manifest_path = gen_dir / "manifest_parent.json"  # ✅ 更清晰的文件名
    else:
        manifest_path = gen_dir / "offspring" / "manifest_offspring.json"  # ✅ 更清晰的文件名
    
    ensure_dir(manifest_path.parent)
    
    manifest = {
        "run_tag": run_tag,
        "generation": gen_idx,
        "population_size": len(individuals),
        "population_type": "parent" if is_parent else "offspring",
        "individuals": [
            {
                "individual_id": ind["individual_id"], 
                "genes": ind["genes"],
            } 
            for ind in individuals
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest_path

# def make_initial_population(pop_size: int, gene_ranges, gen_idx: int):
#     """创建初始种群（未评估）"""
#     individuals = []
#     for i in range(pop_size):
#         ind_id = f"gen{gen_idx:03d}_parent_{i:03d}"  # ✅ 明确标识为父代
#         ind = create_random_individual(individual_id=ind_id, generation=gen_idx, gene_ranges=gene_ranges)
#         individuals.append({
#             "individual_id": ind.individual_id, 
#             "genes": ind.genes,
#             "objectives": None
#         })
#     return individuals

def make_initial_population(pop_size: int, gene_ranges, gen_idx: int):
    """使用Maximin策略创建初始种群，最大化点之间的最小距离"""
    discrete_genes = {}
    continuous_genes = {}
    
    # 分离离散基因和连续基因
    for gene_name, (min_val, max_val) in gene_ranges.items():
        if gene_name == 'num_segments':
            discrete_genes[gene_name] = (min_val, max_val)
        else:
            continuous_genes[gene_name] = (min_val, max_val)
    
    individuals = []
    
    if continuous_genes:
        n_dims = len(continuous_genes)
        
        # ✅ 生成多个候选样本集，选择最优的（Maximin策略）
        best_samples = None
        best_min_dist = -1
        
        for _ in range(10):  # 尝试10次
            sampler = qmc.LatinHypercube(d=n_dims)
            candidate = sampler.random(n=pop_size)
            
            # 计算最小成对距离
            min_dist = pdist(candidate).min()
            
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_samples = candidate
        
        samples = best_samples
        gene_names = list(continuous_genes.keys())
        
        # ✅ 为每个个体分配基因值
        for i in range(pop_size):
            ind_id = f"gen{gen_idx:03d}_parent_{i:03d}"  # 保持 NSGA-II 的命名格式
            genes = {}
            
            # 连续基因：从 LHS 样本映射到实际范围
            for j, gene_name in enumerate(gene_names):
                min_val, max_val = continuous_genes[gene_name]
                genes[gene_name] = float(min_val + samples[i, j] * (max_val - min_val))
            
            # 离散基因：随机采样
            for gene_name, (min_val, max_val) in discrete_genes.items():
                genes[gene_name] = int(np.random.randint(int(min_val), int(max_val) + 1))
            
            individuals.append({
                "individual_id": ind_id,
                "genes": genes,
                "objectives": None  # NSGA-II 需要 objectives 字段
            })
    
    print(f"[init] Created {pop_size} individuals using LHS + Maximin (min_dist={best_min_dist:.4f})")
    
    return individuals
    
def submit_array(manifest_path: Path, run_tag: str, partition: str, concurrency: int):
    """提交 worker 数组任务（不变）"""
    N = len(json.loads(manifest_path.read_text())["individuals"])
    cmd = [
        "sbatch",
        "-p", partition,
        f"--array=0-{N-1}%{concurrency}",
        "--job-name", f"{run_tag}_gen{manifest_path.parent.name.split('_')[-1]}",
        "--output", "logs/workers/%x-%A_%a.out",
        "--export", f"ALL,MANIFEST_PATH={manifest_path},RUN_TAG={run_tag}",
        "scripts/run_worker.sh",
    ]
    print("[submit]", " ".join(cmd))
    out = subprocess.check_output(cmd, text=True).strip()
    jobid = out.split()[-1]
    print(f"[submit] Submitted array job {jobid} with {N} tasks")
    return jobid, N

def wait_and_collect(manifest_path: Path, timeout_s: int = 21600, poll_s: float = 15.0):
    """等待并收集结果 - ✅ 改为收集 objectives"""
    t0 = time.time()
    gen_dir = manifest_path.parent
    manifest = json.loads(manifest_path.read_text())
    N = len(manifest["individuals"])

    is_offspring = manifest.get("population_type") == "offspring"
    population_type = "offspring" if is_offspring else "parent"

    # ✅ 缓存：已成功解析的个体
    parsed_cache = {}

    def snapshot():
        rows, done, pending = [], [], []
        for i, item in enumerate(manifest["individuals"]):

            # ✅ 优先检查缓存
            if i in parsed_cache:
                rows.append(parsed_cache[i])
                done.append(i)
                continue  # ← 跳过重复读取

            # 只解析未缓存的
            f = gen_dir / f"ind_{i}" / "fitness.json"
            if f.exists():
                try:
                    d = json.loads(f.read_text())
                    objectives = d.get("objectives")
                    if objectives is None:
                        objectives = [float(d.get("fitness", -1e9))]
                    
                    result = {
                        "idx": i,
                        "individual_id": item["individual_id"],
                        "objectives": objectives,
                        "genes": item["genes"],
                        "path": str(f),
                        "status": "ok",
                    }
                    parsed_cache[i] = result
                    rows.append(result)
                    done.append(i)
                except Exception as e:
                    print(f"[warn] Failed to parse {f}: {e}")
                    pending.append(i)
            else:
                pending.append(i)
        return rows, done, pending

    rows, done, pending = snapshot()
    while pending and (time.time() - t0) < timeout_s:
        print(f"[wait] done={len(done)}/{N} pending={pending}", flush=True)
        time.sleep(poll_s)
        rows, done, pending = snapshot()

    if pending:
        print(f"[wait] TIMEOUT after {int(time.time()-t0)}s; filling {len(pending)} missing.")
        for i in pending:
            item = manifest["individuals"][i]
            rows.append({
                "idx": i, 
                "individual_id": item["individual_id"], 
                "objectives": [-1e9, -1e9],
                "genes": item["genes"], 
                "path": str(gen_dir / f"ind_{i}" / "fitness.json"),
                "status": "timeout",
            })

    rows.sort(key=lambda r: r["objectives"][0], reverse=True)

    summary_filename = f"fitness_summary_{population_type}.json"
    (gen_dir / summary_filename).write_text(json.dumps(rows, indent=2))
    
    print("[top3 by obj[0]]")
    for r in rows[:3]:
        print(f"  idx={r['idx']:02d} id={r['individual_id']} obj={r['objectives']} status={r['status']}")
    return rows

def generate_offspring_manifest(run_tag: str, gen_idx: int, parent_rows, gene_ranges, nsga2: NSGAII):
    """
    从父代生成子代 manifest（未评估）
    
    Args:
        parent_rows: 父代评估结果
        gene_ranges: 基因范围
    
    Returns:
        子代 manifest 路径
    """
    # 1. 还原父代种群
    parent_pop = restore_population_from_rows(parent_rows, gene_ranges, gen_idx)

    # 2. ✅ 进行非支配排序和拥挤度分配
    fronts = nsga2._non_dominated_sort(parent_pop.individuals)
    for front in fronts:
        nsga2._assign_crowding_distance(front)
        
    # 3. 生成子代种群（未评估）
    offspring_pop = nsga2.generate_offspring_population(parent_pop)
    
    # 4. 转换为 manifest 格式
    individuals = []
    for i, ind in enumerate(offspring_pop.individuals):
        new_id = f"gen{gen_idx:03d}_offspring_{i:03d}"  # ✅ 同代，标识为子代
        individuals.append({
            "individual_id": new_id,
            "genes": ind.genes,
            "objectives": None
        })
    
    # 保存到 offspring 子目录
    return write_manifest(run_tag, gen_idx, individuals, is_parent=False)
    

def build_next_manifest(run_tag: str, next_gen_idx: int, parent_rows, offspring_rows, gene_ranges, nsga2: NSGAII):
    """
    ✅ 合并父代和子代，选择最优 N 个作为下一代父代（已评估，不需要重新评估）
    
    Args:
        next_gen_idx: 下一代的代数
        parent_rows: 当前代父代评估结果
        offspring_rows: 当前代子代评估结果
        gene_ranges: 基因范围
        nsga2: NSGA-II 算法实例
    
    Returns:
        下一代父代 manifest 路径
    """
    current_gen = next_gen_idx - 1
    
    # 1. 还原父代种群
    parent_pop = restore_population_from_rows(parent_rows, gene_ranges, current_gen)

    # 2. 还原子代种群
    offspring_pop = restore_population_from_rows(offspring_rows, gene_ranges, current_gen)

    # 3. 合并父代和子代，选择最优 N 个
    next_generation = nsga2.create_next_generation(parent_pop, offspring_pop)
    
    # ✅ 4. 创建来源映射（追溯到最原始的评估位置）
    source_info_map = {}
    
    # 父代：如果本身有 source，继承它（追溯）；否则用自己
    for r in parent_rows:
        original_source = r.get("source")
        original_path = r.get("path")
        
        if original_source and original_path:
            # ✅ 父代已经有 source，说明是从更早代继承来的，直接传递
            source_info_map[r["individual_id"]] = {
                "source": original_source,    # ✅ 保持指向最原始的个体
                "path": original_path,         # ✅ 保持指向最原始的评估路径
                "status": r.get("status", "ok")
            }
        else:
            # 父代没有 source，说明是第一次评估，用自己
            source_info_map[r["individual_id"]] = {
                "source": r["individual_id"],
                "path": r.get("path"),
                "status": r.get("status", "ok")
            }
    
    # 子代：总是指向自己（因为是刚评估的）
    for r in offspring_rows:
        source_info_map[r["individual_id"]] = {
            "source": r["individual_id"],      # ✅ 子代的原始 ID
            "path": r.get("path"),             # ✅ 子代的评估路径
            "status": r.get("status", "ok")
        }

    # 5. 重新命名为下一代父代（保留原始来源信息）
    individuals = []
    fitness_summary_rows = []  # ✅ 同时构造 fitness_summary 数据
    for i, ind in enumerate(next_generation.individuals):
        old_id = ind.individual_id
        new_id = f"gen{next_gen_idx:03d}_parent_{i:03d}"
        
        source_info = source_info_map.get(old_id, {})
        
        individuals.append({
            "individual_id": new_id,
            "genes": ind.genes,
        })
    
        # ✅ fitness_summary 数据（与 wait_and_collect 格式一致）
        fitness_summary_rows.append({
            "idx": i,
            "individual_id": new_id,
            "genes": ind.genes,
            "objectives": ind.objectives,
            "source": source_info.get("source", old_id),  # ✅ 添加 source
            "path": source_info.get("path"),
            "status": source_info.get("status", "ok")
        })

    # 6. 写入 manifest（只有配置）
    manifest_path = write_manifest(run_tag, next_gen_idx, individuals, is_parent=True)

    # 7. 写入 fitness_summary（完整信息）
    fitness_summary_rows.sort(key=lambda r: r["objectives"][0], reverse=True)
    gen_dir = manifest_path.parent
    summary_path = gen_dir / "fitness_summary_parent.json"
    summary_path.write_text(json.dumps(fitness_summary_rows, indent=2))
    
    return manifest_path

def restore_population_from_rows(rows: list, gene_ranges: dict, generation: int) -> Population:
    """
    ✅ 辅助函数：从评估结果还原种群
    
    Args:
        rows: 评估结果列表 (从 wait_and_collect 返回)
        gene_ranges: 基因范围
        generation: 代数
    
    Returns:
        还原的种群对象
    """
    pop = Population(size=len(rows), gene_ranges=gene_ranges, generation=generation)
    pop.individuals = []
    
    for r in rows:
        ind = Individual(
            individual_id=r["individual_id"],
            genes=r["genes"],
            generation=generation
        )
        ind.objectives = r["objectives"]
        ind.training_completed = True
        pop.individuals.append(ind)
    
    return pop

if __name__ == "__main__":
    sys.exit(main())
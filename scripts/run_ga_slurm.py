#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, json, time, subprocess, argparse, sys
from pathlib import Path

# === 你项目内的模块 ===
from tensegrity_playground.envs.evolution.individual import create_random_individual
from tensegrity_playground.envs.evolution.genetic_algorithm import GeneticAlgorithm
from tensegrity_playground.envs.evolution.population import Population
from tensegrity_playground.envs.evolution.gene_config import get_gene_ranges

def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)

def write_manifest(run_tag: str, gen_idx: int, individuals):
    gen_dir = Path(f"/scratch/izar/{os.environ['USER']}/ga_runs/{run_tag}/gen_{gen_idx:03d}")
    ensure_dir(gen_dir)
    manifest = {
        "run_tag": run_tag,
        "generation": gen_idx,
        "population_size": len(individuals),
        "individuals": [{"individual_id": ind["individual_id"], "genes": ind["genes"]} for ind in individuals],
    }
    (gen_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return gen_dir / "manifest.json"

def make_initial_population(pop_size: int, gene_ranges):
    individuals = []
    for i in range(pop_size):
        ind_id = f"gen000_idx{i:03d}"
        ind = create_random_individual(individual_id=ind_id, generation=0, gene_ranges=gene_ranges)
        individuals.append({"individual_id": ind.individual_id, "genes": ind.genes})
    return individuals

def submit_array(manifest_path: Path, run_tag: str, partition: str, concurrency: int):
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
    t0 = time.time()
    gen_dir = manifest_path.parent
    manifest = json.loads(manifest_path.read_text())
    N = len(manifest["individuals"])

    def snapshot():
        rows, done, pending = [], [], []
        for i, item in enumerate(manifest["individuals"]):
            f = gen_dir / f"ind_{i}" / "fitness.json"
            if f.exists():
                try:
                    d = json.loads(f.read_text())
                    rows.append({
                        "idx": i,
                        "individual_id": item["individual_id"],
                        "fitness": float(d.get("fitness")),
                        "genes": item["genes"],
                        "path": str(f),
                        "status": "ok",
                    })
                    done.append(i)
                except Exception:
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
                "idx": i, "individual_id": item["individual_id"], "fitness": -1e9,
                "genes": item["genes"], "path": str(gen_dir / f"ind_{i}" / "fitness.json"),
                "status": "timeout",
            })

    rows.sort(key=lambda r: r["fitness"], reverse=True)
    (gen_dir / "fitness_summary.json").write_text(json.dumps(rows, indent=2))
    print("[top3]")
    for r in rows[:3]:
        print(f"  idx={r['idx']:02d} id={r['individual_id']} fit={r['fitness']:.3f} status={r['status']}")
    return rows

def build_next_manifest(run_tag: str, gen_idx: int, summary_rows):
    # 还原 Population 并喂给 GA
    pop = Population(size=len(summary_rows), gene_ranges=None, generation=gen_idx)
    pop.individuals = []
    for r in summary_rows:
        pop.individuals.append(
            # Individual 的导入/构造由 Population/GA 内部使用；这里只需 genes/fitness
            type("Tmp", (), {
                "individual_id": r["individual_id"],
                "genes": r["genes"],
                "fitness": float(r["fitness"]),
                "training_completed": True,
                "generation": gen_idx
            })()
        )

    gene_ranges = get_gene_ranges(mode="base")
    ga = GeneticAlgorithm(
        population_size=len(summary_rows),
        gene_ranges=gene_ranges,
        mutation_rate=0.30,
        crossover_rate=0.80,
        elitism_ratio=0.10,
    )
    next_pop = ga.create_next_generation(pop)

    individuals = [{"individual_id": ind.individual_id, "genes": ind.genes}
                   for ind in next_pop.individuals]
    return write_manifest(run_tag, gen_idx + 1, individuals)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_tag", type=str, required=True)
    ap.add_argument("--pop_size", type=int, default=8)
    ap.add_argument("--generations", type=int, default=3)
    ap.add_argument("--partition", type=str, default="gpu")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--timeout_s", type=int, default=21600)
    ap.add_argument("--poll_s", type=float, default=15.0)
    args = ap.parse_args()

    ensure_dir(Path("logs"))

    # gen_000: 若不存在则创建；存在则复用（支持断点续跑）
    gen0_dir = Path(f"/scratch/izar/{os.environ['USER']}/ga_runs/{args.run_tag}/gen_000")
    man = gen0_dir / "manifest.json"
    if not man.exists():
        gene_ranges = get_gene_ranges(mode="base")
        individuals = make_initial_population(args.pop_size, gene_ranges)
        man = write_manifest(args.run_tag, 0, individuals)
        print("[init] wrote", man)
    else:
        print("[init] reuse", man)

    # 逐代循环（同步代际）
    for g in range(args.generations):
        print(f"\n===== Generation {g:03d} =====")
        jobid, N = submit_array(man, args.run_tag, args.partition, args.concurrency)
        rows = wait_and_collect(man, timeout_s=args.timeout_s, poll_s=args.poll_s)
        # 产出下一代 manifest
        if g < args.generations - 1:
            man = build_next_manifest(args.run_tag, g, rows)
            print("[next] wrote", man)

    print("\n[done] GA run complete.")

if __name__ == "__main__":
    sys.exit(main())

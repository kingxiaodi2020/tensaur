#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH -J worker
#SBATCH -p gpu              # 如分区名不同，稍后再改
#SBATCH --gpus=1
#SBATCH -t 48:00:00
#SBATCH -o logs/workers/%x-%A_%a.out

set -euo pipefail

# === 任务入参（后续用 job array 自动注入）===
INDIV_IDX="${INDIV_IDX:-${SLURM_ARRAY_TASK_ID:-0}}"
MANIFEST_PATH="${MANIFEST_PATH:-manifest.json}"
RUN_TAG="${RUN_TAG:-debug}"
export INDIV_IDX MANIFEST_PATH RUN_TAG 

echo "[worker] INDIV_IDX=$INDIV_IDX"
echo "[worker] MANIFEST_PATH=$MANIFEST_PATH"
echo "[worker] RUN_TAG=$RUN_TAG"

# === 环境（按你集群改）===
module purge
module load gcc openmpi
module load gcc cuda/12.1.1
module load python/3.10.4
unset LD_LIBRARY_PATH
source "$HOME/venvs/tensaur/bin/activate"

export MUJOCO_GL=egl
export DISPLAY=                           # 置空，避免 GUI 尝试
export XLA_PYTHON_CLIENT_PREALLOCATE=false 

# 回到提交目录
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# === 源码快照（避免并发互相影响，可省略）===
SNAP_DIR="/scratch/izar/$USER/tensaur_snap/${RUN_TAG}-${SLURM_JOB_ID}"
mkdir -p "$SNAP_DIR"
rsync -a --delete --exclude ".git" --exclude "wandb" --exclude "logs" --exclude "__pycache__" \
  "$SLURM_SUBMIT_DIR"/ "$SNAP_DIR"/
cd "$SNAP_DIR"
export PYTHONPATH="$SNAP_DIR:${PYTHONPATH:-}"

# === ensure ffmpeg for mediapy ===
FFMPEG_BIN=$(python - <<'PY'
try:
    import imageio_ffmpeg as m
    print(m.get_ffmpeg_exe())
except Exception:
    print("")
PY
)
if [ -n "$FFMPEG_BIN" ] && [ -x "$FFMPEG_BIN" ]; then
  export IMAGEIO_FFMPEG_EXE="$FFMPEG_BIN"        # mediapy/imageio 会优先读这个
  mkdir -p "$SNAP_DIR/bin"
  ln -sf "$FFMPEG_BIN" "$SNAP_DIR/bin/ffmpeg"    # 保底：提供一个名为 ffmpeg 的可执行
  export PATH="$SNAP_DIR/bin:$(dirname "$FFMPEG_BIN"):$PATH"
  echo "[worker] ffmpeg set to: $FFMPEG_BIN"
  "$FFMPEG_BIN" -version | head -1 || true
else
  echo "[worker] WARNING: imageio-ffmpeg not found; videos may fail"
fi

# === 该个体的独立输出目录 ===
# ✅ 直接用 manifest 所在目录作为本代目录
MANIFEST_DIR="$(dirname "$MANIFEST_PATH")"
CKPT_DIR="${MANIFEST_DIR}/ind_${INDIV_IDX}"
mkdir -p "$CKPT_DIR"
export CKPT_DIR

python -V
nvidia-smi || true

# === 运行单个体训练 ===
# 在最后添加训练调用
python - << 'PY'
import json, os, sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tensegrity_playground.envs.evolution.single_trainer import SingleIndividualTrainer
from tensegrity_playground.envs.evolution.individual import Individual

INDIV_IDX = int(os.environ["INDIV_IDX"])
MANIFEST_PATH = Path(os.environ["MANIFEST_PATH"]).resolve()
CKPT_DIR = Path(os.environ["CKPT_DIR"]).resolve()

# 读取 manifest
data = json.loads(MANIFEST_PATH.read_text())
entry = data["individuals"][INDIV_IDX]

# 创建个体
ind = Individual(
    individual_id=entry["individual_id"], 
    genes=entry["genes"],
    generation=data.get("generation", 0)
)

# ✅ 训练（多目标模式，每个任务运行 3 个 seed）
trainer = SingleIndividualTrainer()
results = trainer.train_individual(ind, mode="multi_objective")

# ✅ 保存结果（包含详细的每个 seed 的结果）
summary = {
    "individual_id": ind.individual_id,
    "objectives": results["objectives"],  # 平均后的 [avg_r_deviation, avg_max_x_final]
    "training_completed": ind.training_completed,
    "ckpt_dir": str(CKPT_DIR),
    # ✅ 保存详细结果（用于调试和分析稳定性）
    "avg_r_deviation": results.get("avg_r_deviation"),
    "avg_max_x_final": results.get("avg_max_x_final"),
    "radius_results": [r.get("r_deviation") for r in results.get("radius_results_list", [])],
    "walk_results": [w.get("max_x_final") for w in results.get("walk_results_list", [])],
}

output_file = CKPT_DIR / "fitness.json"
output_file.write_text(json.dumps(summary, indent=2))
print(f"[worker] Results saved to {output_file}")
print(f"[worker] Objectives (avg of 3 seeds): {results['objectives']}")
print(f"[worker]   - Radius results: {summary['radius_results']}")
print(f"[worker]   - Walk results: {summary['walk_results']}")
PY

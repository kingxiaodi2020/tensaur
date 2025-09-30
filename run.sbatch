#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH -J tens4_8000_10_rk_Arep10
#SBATCH -p gpu
#SBATCH --gpus=2
#SBATCH -t 24:00:00
#SBATCH -o logs/%x-%j.out

set -euo pipefail

echo "===== SLURM ENV ====="
echo "USER=$USER JOB_NAME=$SLURM_JOB_NAME JOB_ID=$SLURM_JOB_ID"
echo "SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR"
echo "PWD(before)=$(pwd)"

module purge
module load gcc openmpi
module load gcc cuda/12.1.1
module load python/3.10.4
unset LD_LIBRARY_PATH

source "$HOME/venvs/tensaur/bin/activate"

# 关键：回到提交目录，避免落到 /tmp
cd "$SLURM_SUBMIT_DIR"
echo "PWD(after cd)=$(pwd)"

# 产物目录
mkdir -p logs
CKPT_DIR="/scratch/izar/$USER/tensaur_temp/${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
mkdir -p "$CKPT_DIR"
echo "CKPT_DIR=$CKPT_DIR"

python -V
nvidia-smi || true

# 推荐：源码快照，保证并发与不受后续同步影响
SNAP_DIR="/scratch/izar/$USER/tensaur_snap/${SLURM_JOB_NAME}-${SLURM_JOB_ID}"
mkdir -p "$SNAP_DIR"
rsync -a --delete \
  --exclude ".git" \
  --exclude "wandb" \
  --exclude "logs" \
  --exclude "__pycache__" \
  "$SLURM_SUBMIT_DIR"/ "$SNAP_DIR"/

cd "$SNAP_DIR"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
echo "Using SNAP_DIR=$SNAP_DIR"
ls -al scripts | sed -n '1,80p' || true

# 用相对路径运行快照内脚本（或直接用绝对路径）
python scripts/train_ppo.py checkpoint_directory="$CKPT_DIR"
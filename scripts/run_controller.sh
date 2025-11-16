#!/bin/bash
#SBATCH -p gpu
#SBATCH -J ga_ctrl_go1_sync_full
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH -t 48:00:00
#SBATCH -o logs/controller/%x-%j.out

set -euo pipefail
module purge
module load python/3.10.4
source ~/venvs/tensaur/bin/activate
cd "$SLURM_SUBMIT_DIR"

mkdir -p logs/controller logs/workers

python scripts/run_ga_slurm.py \
  --run_tag test1_base \
  --pop_size 8 \
  --generations 1 \
  --partition gpu \
  --concurrency 8 \
  --timeout_s 36000 \
  --poll_s 15



# 0) 到项目根目录并激活环境
cd ~/projects/tensaur-main
source ~/venvs/tensaur/bin/activate
export PYTHONPATH="$PWD"

# 2) 后台启动 controller（不占GPU）
RUN_TAG=test6v_3gens
LOG="logs/controller/ga_${RUN_TAG}_$(date +%y%m%d_%H%M).out"

nohup stdbuf -oL -eL \
  python scripts/run_ga_slurm.py \
    --run_tag "$RUN_TAG" \
    --pop_size 6 \
    --generations 3 \
    --partition gpu \
    --concurrency 6 \
    --timeout_s 32400 \
    --poll_s 300 \
  > "$LOG" 2>&1 < /dev/null &

echo $! > "logs/controller/ga_${RUN_TAG}.pid"
echo "Controller started. Log: $LOG  PID: $(cat logs/controller/ga_${RUN_TAG}.pid)"

# 3) 实时看日志
tail -f "$LOG"

# 4) 如果要停止 controller
pgrep -fl "python .*scripts/run_ga_slurm.py"  
kill $(cat "logs/controller/ga_${RUN_TAG}.pid")
kill 3116500 2>/dev/null
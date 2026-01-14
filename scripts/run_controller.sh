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
RUN_TAG=run_geometry_0.5T
LOG="logs/controller/ga_${RUN_TAG}_$(date +%y%m%d_%H%M).out"

nohup stdbuf -oL -eL \
  python scripts/run_ga_slurm.py \
    --run_tag "$RUN_TAG" \
    --pop_size 16 \
    --generations 10 \
    --partition gpu \
    --concurrency 16 \
    --timeout_s 36000 \
    --poll_s 10 \
  > "$LOG" 2>&1 < /dev/null &

echo $! > "logs/controller/ga_${RUN_TAG}.pid"
echo "Controller started. Log: $LOG  PID: $(cat logs/controller/ga_${RUN_TAG}.pid)"

# 3) 实时看日志
tail -f "$LOG"

# 4) 如果要停止 controller
pgrep -fl "python .*scripts/run_ga_slurm.py"  
kill $(cat "logs/controller/ga_${RUN_TAG}.pid")
kill 3116500 2>/dev/null


# 2) 后台启动 controller（不占GPU）
RUN_TAG=nsga2_3seeds
LOG="logs/controller/ga_${RUN_TAG}_$(date +%y%m%d_%H%M).out"

nohup stdbuf -oL -eL \
  python scripts/run_nsga2_slurm.py \
    --run_tag "$RUN_TAG" \
    --pop_size 16 \
    --generations 11 \
    --partition gpu \
    --concurrency 16 \
    --timeout_s 172800 \
    --poll_s 60 \
    --mutation_rate 0.9 \
    --crossover_rate 0 \
    --gene_mode medium \
  > "$LOG" 2>&1 < /dev/null &

echo $! > "logs/controller/ga_${RUN_TAG}.pid"
echo "Controller started. Log: $LOG  PID: $(cat logs/controller/ga_${RUN_TAG}.pid)"

pgrep -fl "python .*scripts/run_nsga2_slurm.py"
kill 3116500 2>/dev/null
#查看完整名字
squeue -u $USER -o "%.18i %.30j %.10T %.20P"



mkdir -p logs/controller

cd ~/projects/tensaur-main
source ~/venvs/tensaur/bin/activate

RUN_TAG=10k_uneven
LOG="logs/controller/multi_seed_${RUN_TAG}_$(date +%y%m%d_%H%M).out"

nohup stdbuf -oL -eL \
  python scripts/run_multi_seed.py \
    --num-jobs 10 \
    --start-seed 1 \
    --job-name "$RUN_TAG" \
    --playground TensegrityQuadrupedWalk \
    --training-script scripts/train_walk.py \
  > "$LOG" 2>&1 < /dev/null &

echo $! > "logs/controller/multi_seed_${RUN_TAG}.pid"
echo "Started: PID=$(cat logs/controller/multi_seed_${RUN_TAG}.pid), Log: $LOG"

# 实时看日志
tail -f "$LOG"
#!/usr/bin/env python3
"""
Submits 16 parallel training jobs with different PPO seeds.
Each job runs on a separate node with a unique seed value.
"""

import subprocess
import argparse
from pathlib import Path
import sys

def submit_training_jobs(
    num_jobs: int = 16,
    job_name: str = "tensaur_multi_seed",
    cpus_per_task: int = 32,
    mem: str = "64G",
    gpus: int = 1,
    time_limit: str = "24:00:00",
    partition: str = "gpu",
    training_script: str = "scripts/train_run.py",
    playground: str = "TensegrityQuadrupedRun",
    start_seed: int = 1,
):
    """
    Submit multiple training jobs with different seeds.
    
    Args:
        num_jobs: Number of parallel jobs to submit
        job_name: Base name for SLURM jobs
        cpus_per_task: CPUs per task
        mem: Memory per node
        gpus: GPUs per node
        time_limit: Wall time limit
        partition: SLURM partition
        training_script: Path to training script
        playground: Playground configuration
        start_seed: Starting seed value (incremented for each job)
    """
    
    # Get project root directory (parent of scripts/)
    project_dir = Path(__file__).parent.parent.resolve()
    print(f"Project directory: {project_dir}")
    
    job_ids = []
    
    for i in range(num_jobs):
        seed = start_seed + i
        job_label = f"{job_name}_s{seed:02d}"
        
        # Create sbatch command
        sbatch_cmd = f"""#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem={mem}
#SBATCH -J {job_label}
#SBATCH -p {partition}
#SBATCH --gpus={gpus}
#SBATCH -t {time_limit}
#SBATCH -o logs/%x-%j.out

set -euo pipefail

echo "===== SLURM ENV ====="
echo "USER=$USER JOB_NAME=$SLURM_JOB_NAME JOB_ID=$SLURM_JOB_ID SEED={seed}"
echo "SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR"

module purge
module load gcc openmpi
module load gcc cuda/12.1.1
module load python/3.10.4
unset LD_LIBRARY_PATH
source "$HOME/venvs/tensaur/bin/activate"

export MUJOCO_GL=egl
export DISPLAY=""
export XLA_PYTHON_CLIENT_PREALLOCATE=false

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# Create snapshot directory for this job
SNAP_DIR="/scratch/izar/$USER/tensaur_snap/${{SLURM_JOB_NAME}}-${{SLURM_JOB_ID}}"
mkdir -p "$SNAP_DIR"
rsync -a --delete \\
  --exclude ".git" \\
  --exclude "wandb" \\
  --exclude "logs" \\
  --exclude "__pycache__" \\
  "$SLURM_SUBMIT_DIR"/ "$SNAP_DIR"/

cd "$SNAP_DIR"
export PYTHONPATH="$PWD:${{PYTHONPATH:-}}"

python -V
nvidia-smi || true

# Set checkpoint directory for this seed
CKPT_DIR="/scratch/izar/$USER/runfast_0.5T/${{SLURM_JOB_NAME}}-${{SLURM_JOB_ID}}"
mkdir -p "$CKPT_DIR"

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
  export IMAGEIO_FFMPEG_EXE="$FFMPEG_BIN"
  mkdir -p "$SNAP_DIR/bin"
  ln -sf "$FFMPEG_BIN" "$SNAP_DIR/bin/ffmpeg"
  export PATH="$SNAP_DIR/bin:$(dirname "$FFMPEG_BIN"):$PATH"
  echo "[worker] ffmpeg set to: $FFMPEG_BIN"
  "$FFMPEG_BIN" -version | head -1 || true
else
  echo "[worker] WARNING: imageio-ffmpeg not found; videos may fail"
fi

# Run training with seed override
python {training_script} \\
  playground={playground} \\
  agent.seed={seed} \\
  checkpoint_directory="$CKPT_DIR" \\
  hydra.run.dir="$CKPT_DIR"
"""
        
        # Submit the job
        try:
            result = subprocess.run(
                ["sbatch"],
                input=sbatch_cmd,
                capture_output=True,
                text=True,
                cwd=str(project_dir),
            )
            
            if result.returncode == 0:
                # Extract job ID from output
                output_line = result.stdout.strip()
                job_id = output_line.split()[-1]
                job_ids.append(job_id)
                print(f"✓ Job {i+1}/{num_jobs} submitted: {job_label} (Job ID: {job_id}, Seed: {seed})")
            else:
                print(f"✗ Job {i+1}/{num_jobs} failed: {job_label}")
                print(f"  Error: {result.stderr}")
                
        except Exception as e:
            print(f"✗ Error submitting job {i+1}/{num_jobs}: {e}")
    
    print(f"\n{'='*60}")
    print(f"Summary: {len(job_ids)} out of {num_jobs} jobs submitted successfully")
    print(f"{'='*60}")
    
    if job_ids:
        print(f"\nJob IDs: {', '.join(job_ids)}")
        print(f"\nMonitor jobs with:")
        print(f"  squeue -u $USER -j {','.join(job_ids)}")
        print(f"\nCancel all jobs with:")
        print(f"  scancel {','.join(job_ids)}")
    
    return len(job_ids) == num_jobs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Submit multiple training jobs with different PPO seeds"
    )
    parser.add_argument(
        "--num-jobs",
        type=int,
        default=16,
        help="Number of parallel jobs (default: 16)"
    )
    parser.add_argument(
        "--job-name",
        type=str,
        default="tensaur_multi_seed",
        help="Base name for jobs (default: tensaur_multi_seed)"
    )
    parser.add_argument(
        "--start-seed",
        type=int,
        default=1,
        help="Starting seed value (default: 1)"
    )
    parser.add_argument(
        "--playground",
        type=str,
        default="TensegrityQuadrupedWalk",
        help="Playground config (default: TensegrityQuadrupedWalk)"
    )
    parser.add_argument(
        "--training-script",
        type=str,
        default="scripts/train_walk.py",
        help="Training script path (default: scripts/train_walk.py)"
    )
    parser.add_argument(
        "--cpus-per-task",
        type=int,
        default=32,
        help="CPUs per task (default: 32)"
    )
    parser.add_argument(
        "--mem",
        type=str,
        default="64G",
        help="Memory per node (default: 64G)"
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="GPUs per node (default: 1)"
    )
    parser.add_argument(
        "--time-limit",
        type=str,
        default="24:00:00",
        help="Wall time limit (default: 24:00:00)"
    )
    parser.add_argument(
        "--partition",
        type=str,
        default="gpu",
        help="SLURM partition (default: gpu)"
    )
    
    args = parser.parse_args()
    
    success = submit_training_jobs(
        num_jobs=args.num_jobs,
        job_name=args.job_name,
        cpus_per_task=args.cpus_per_task,
        mem=args.mem,
        gpus=args.gpus,
        time_limit=args.time_limit,
        partition=args.partition,
        training_script=args.training_script,
        playground=args.playground,
        start_seed=args.start_seed,
    )
    
    sys.exit(0 if success else 1)
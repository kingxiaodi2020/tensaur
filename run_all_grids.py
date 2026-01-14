#!/usr/bin/env python3
"""
Run all three grid searches sequentially: Tx, dy, dz
"""

import subprocess
import sys

def run_grid_search(load_type, magnitude, direction=None, settle_load_steps=4000):
    """Run a single grid search"""
    cmd = [
        "python", "full_grid.py",
        "--load_type", load_type,
        "--magnitude", str(magnitude),
        "--settle_load_steps", str(settle_load_steps),
    ]
    
    if direction:
        cmd.extend(["--direction", direction])
    
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    result = subprocess.run(cmd)
    return result.returncode == 0


def main():
    # 按顺序运行三个 grid search (指定各自的 settle_load_steps)
    tasks = [
        ("torque", 5.0, None, 4000),        # Tx = 5 Nm, steps = 4000
        ("force", 10.0, "y", 4000),         # Fy = 10 N, steps = 4000
        ("force", 40.0, "z", 10000),        # Fz = 40 N, steps = 10000
    ]
    
    results = []
    
    for i, (load_type, magnitude, direction, steps) in enumerate(tasks, 1):
        print(f"\n[{i}/3] Starting grid search (settle_load_steps={steps})...")
        success = run_grid_search(load_type, magnitude, direction, steps)
        results.append((load_type, magnitude, direction, steps, success))
        
        if not success:
            print(f"[ERROR] Grid search failed: {load_type} {magnitude}")
            sys.exit(1)
    
    # 总结结果
    print(f"\n{'='*60}")
    print("All grid searches completed!")
    print(f"{'='*60}")
    for load_type, magnitude, direction, steps, success in results:
        status = "✓ Success" if success else "✗ Failed"
        if direction:
            print(f"  {status}: {load_type} F{direction}{int(magnitude)} (steps={steps})")
        else:
            print(f"  {status}: {load_type} Tx{int(magnitude)} (steps={steps})")


if __name__ == "__main__":
    main()
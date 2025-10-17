"""
异步任务结果报告脚本
"""
import argparse
import json
import time
from pathlib import Path
import sys

def extract_fitness_from_logs(config_path: str) -> float:
    """从训练日志中提取适应度"""
    # 这里需要根据你的训练输出格式来解析
    # 临时返回模拟值
    import random
    return random.uniform(800, 1200)

def report_result():
    parser = argparse.ArgumentParser(description='报告异步训练结果')
    parser.add_argument('--worker-id', required=True, help='Worker ID')
    parser.add_argument('--task-id', required=True, help='任务ID')
    parser.add_argument('--individual-id', required=True, help='个体ID')
    parser.add_argument('--success', type=int, required=True, help='训练是否成功 (0=成功, 非0=失败)')
    parser.add_argument('--config-path', required=True, help='配置文件路径')
    args = parser.parse_args()
    
    # 提取适应度
    if args.success == 0:
        try:
            fitness = extract_fitness_from_logs(args.config_path)
            success = True
            error_message = None
        except Exception as e:
            fitness = 0.0
            success = False
            error_message = f"Failed to extract fitness: {e}"
    else:
        fitness = 0.0
        success = False
        error_message = f"Training failed with exit code {args.success}"
    
    # 创建结果报告
    result_data = {
        'task_id': args.task_id,
        'individual_id': args.individual_id,
        'worker_id': args.worker_id,
        'fitness': fitness,
        'success': success,
        'error_message': error_message,
        'completion_time': time.time(),
        'training_time': 0.0  # 可以从日志中计算
    }
    
    # 保存结果文件
    result_file = Path(f"worker_reports/{args.worker_id}_{args.task_id}.json")
    result_file.parent.mkdir(exist_ok=True)
    
    with open(result_file, 'w') as f:
        json.dump(result_data, f, indent=2)
    
    print(f"✅ 结果已报告: Worker {args.worker_id}, 任务 {args.task_id}, 个体 {args.individual_id}")
    print(f"   成功: {success}, 适应度: {fitness:.2f}")
    print(f"   结果文件: {result_file}")

if __name__ == "__main__":
    report_result()
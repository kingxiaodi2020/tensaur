import sys
import os
import argparse
from pathlib import Path

# 动态查找并切换到正确的工作目录
def ensure_correct_working_directory():
    """确保在正确的工作目录下运行"""
    current = Path.cwd()
    
    # 如果当前已经在 tensaur-main 目录，直接返回
    if (current / "config").exists() and (current / "scripts").exists():
        return current
    
    # 查找 tensaur-main 目录
    tensaur_dirs = []
    
    # 在当前目录及其子目录中查找
    for path in [current] + list(current.rglob("tensaur-main")):
        if path.is_dir() and (path / "config").exists() and (path / "scripts").exists():
            tensaur_dirs.append(path)
    
    # 在父目录中查找
    for parent in current.parents:
        tensaur_path = parent / "tensaur-main"
        if tensaur_path.exists() and (tensaur_path / "config").exists():
            tensaur_dirs.append(tensaur_path)
    
    if tensaur_dirs:
        target_dir = tensaur_dirs[0]
        print(f"🔄 切换工作目录: {current} -> {target_dir}")
        os.chdir(target_dir)
        return target_dir
    else:
        print(f"⚠️  未找到 tensaur-main 目录，继续在当前目录运行")
        return current

# 确保在正确的工作目录
working_dir = ensure_correct_working_directory()
sys.path.append('src')

from tensegrity_playground.envs.evolution.individual import Individual
from tensegrity_playground.envs.evolution.single_trainer import SingleIndividualTrainer

def test_training(run_training=False, cleanup=False, interactive=False):
    """快速训练测试"""
    print("🚀 快速训练测试")
    print(f"当前工作目录: {os.getcwd()}")
    
    # 使用路径管理器检查目录结构（而不是硬编码路径）
    print(f"\n📁 检查目录结构:")
    
    # 创建临时的路径管理器来检查路径
    try:
        from tensegrity_playground.envs.evolution.path_manager import get_path_manager
        pm = get_path_manager()
        
        print(f"   项目根目录: {pm.project_root}")
        print(f"   config/ 存在: {pm.config_dir.exists()}")
        print(f"   config/agent/ppo.yaml 存在: {pm.get_config_path('agent/ppo.yaml').exists()}")
        print(f"   config/playground/TensegrityQuadrupedWalk.yaml 存在: {pm.get_config_path('playground/TensegrityQuadrupedWalk.yaml').exists()}")
        print(f"   config/playground_brax.yaml 存在: {pm.get_config_path('playground_brax.yaml').exists()}")
        print(f"   scripts/train_ppo.py 存在: {pm.get_script_path('train_ppo.py').exists()}")
        
    except Exception as e:
        print(f"   ⚠️ 路径管理器检查失败: {e}")
        # 回退到简单检查
        config_dir = Path("config")
        print(f"   config/ 存在: {config_dir.exists()}")
        print(f"   config/agent/ppo.yaml 存在: {(config_dir / 'agent' / 'ppo.yaml').exists()}")
        print(f"   config/playground/TensegrityQuadrupedWalk.yaml 存在: {(config_dir / 'playground' / 'TensegrityQuadrupedWalk.yaml').exists()}")
        print(f"   config/playground_brax.yaml 存在: {(config_dir / 'playground_brax.yaml').exists()}")
    
    # 创建测试个体
    test_genes = {
        'num_segments': 3,
        'segment_spacing': 0.05,
    }
    
    individual = Individual("quick_test", test_genes, generation=0)
    trainer = SingleIndividualTrainer()
    
    try:
        # 只测试配置生成，不实际训练
        xml_path = trainer.prepare_individual_xml(individual)
        config_path = trainer.create_training_config(individual, xml_path)
        
        print(f"\n✅ 成功生成:")
        print(f"   XML: {xml_path}")
        print(f"   配置: {config_path}")
        
        # 显示生成的配置内容
        with open(config_path, 'r') as f:
            config_content = f.read()
        print(f"\n📋 生成的配置:")
        print("-" * 40)
        print(config_content)
        print("-" * 40)
        
        # 测试命令构建
        config_name = f"temp_individual_configs/{Path(config_path).stem}"
        cmd = [
            sys.executable, "scripts/train_ppo.py",
            "--config-path", "config",
            "--config-name", config_name
        ]
        
        print(f"\n🔧 将要执行的命令:")
        print(f"   {' '.join(cmd)}")
        
        # 询问是否实际运行
        response = input(f"\n❓ 是否运行实际训练？(y/N): ")
        if response.lower() == 'y':
            print(f"🚀 开始实际训练...")
            fitness = trainer.train_individual(individual)
            print(f"🏁 训练结果: {fitness}")
        else:
            print(f"⏭️  跳过实际训练")
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 询问是否清理
        cleanup_response = input(f"\n🧹 是否清理临时文件？(y/N): ")
        if cleanup_response.lower() == 'y':
            trainer.cleanup()

if __name__ == "__main__":
    test_training()
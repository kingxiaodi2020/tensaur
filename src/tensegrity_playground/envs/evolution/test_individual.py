# import sys
# sys.path.append('src')

# from tensegrity_playground.envs.evolution.individual import Individual, create_random_individual

# # 测试1：手动创建个体
# print("=== 测试1：手动创建个体 ===")
# test_genes = {
#     'num_segments': 3,
#     'stiffness': 12000,
#     'damping': 12
# }

# try:
#     individual = Individual("test_001", test_genes, generation=0)
#     print("✓ 个体创建成功")
#     print(f"  ID: {individual.individual_id}")
#     print(f"  基因: {individual.genes}")
#     print(f"  适应度: {individual.fitness}")
# except Exception as e:
#     print(f"✗ 个体创建失败: {e}")

# # 测试2：随机生成个体
# print("\n=== 测试2：随机生成个体 ===")
# try:
#     random_individual = create_random_individual("random_001", generation=1)
#     print("✓ 随机个体创建成功")
#     print(f"  ID: {random_individual.individual_id}")
#     print(f"  基因: {random_individual.genes}")
#     print(f"  代数: {random_individual.generation}")
# except Exception as e:
#     print(f"✗ 随机个体创建失败: {e}")

# # 测试3：保存和加载
# print("\n=== 测试3：保存和加载 ===")
# try:
#     # 保存
#     random_individual.fitness = 85.5
#     random_individual.save_to_file("test_individual.json")
#     print("✓ 个体保存成功")
    
#     # 加载
#     loaded_individual = Individual.load_from_file("test_individual.json")
#     print("✓ 个体加载成功")
#     print(f"  加载的适应度: {loaded_individual.fitness}")
    
#     # 清理
#     import os
#     os.remove("test_individual.json")
#     print("✓ 测试文件清理完成")
    
# except Exception as e:
#     print(f"✗ 保存/加载测试失败: {e}")

"""
测试路径管理器
"""
import sys
sys.path.append('src')

from tensegrity_playground.envs.evolution.path_manager import get_path_manager

def test_path_manager():
    """测试路径管理器"""
    print("🧪 测试路径管理器")
    
    pm = get_path_manager()
    
    print(f"\n📁 路径信息:")
    print(f"  项目根目录: {pm.project_root}")
    print(f"  配置目录: {pm.config_dir}")
    print(f"  脚本目录: {pm.scripts_dir}")
    print(f"  临时目录: {pm.temp_dir}")
    print(f"  日志目录: {pm.logs_dir}")
    print(f"  XML目录: {pm.xmls_dir}")
    
    print(f"\n🔍 文件存在性检查:")
    print(f"  playground_brax.yaml: {pm.get_config_path('playground_brax.yaml').exists()}")
    print(f"  ppo.yaml: {pm.get_config_path('agent/ppo.yaml').exists()}")
    print(f"  train_ppo.py: {pm.get_script_path('train_ppo.py').exists()}")
    
    print(f"\n✅ 路径管理器测试完成!")

if __name__ == "__main__":
    test_path_manager()
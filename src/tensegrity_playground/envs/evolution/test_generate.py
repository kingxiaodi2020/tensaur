import sys
sys.path.append('src')

from tensegrity_playground.envs.tensaur.generate_go1 import build_config_from_genes, generate_quadruped_from_config

# 测试基因
test_genes = {
    'num_segments': 4,
    'segment_spacing': 0.065,
    'lateral_stiffness': 1000,
    'alpha': 0.8,
    'beta': 0.9
}

# # 生成配置和XML
config = build_config_from_genes(test_genes, "test_001")

print(f"  num_segments: {config['spine']['num_segments']}")
print(f"  segment_spacing: {config['spine']['segment_spacing']}")
print(f"  lateral_stiffness: {config['defaults']['lateral_tendon']['tendon']['stiffness']}")
print(f"  alpha: {config['spine']['alpha']}")

# 生成XML文件
try:
    generate_quadruped_from_config(config)
    print(f"✓ XML file generated successfully at: {config['output_path']}")
    
    # 检查文件是否存在
    import os
    if os.path.exists(config['output_path']):
        print("✓ XML file exists and is accessible")
    else:
        print("✗ XML file was not created")
        
except Exception as e:
    print(f"✗ Error generating XML: {e}")
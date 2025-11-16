import sys
from pathlib import Path
import numpy as np
import os

# 添加当前目录到路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir.parent))

from tensegrity_playground.envs.evolution.individual import create_random_individual, Individual
from tensegrity_playground.envs.evolution.gene_config import BASE_GENE_RANGES, ADVANCED_GENE_RANGES
from tensegrity_playground.envs.tensaur.generate_go1_tendon import build_config_from_genes, generate_quadruped_from_config

def print_separator(title: str = "", length: int = 80):
    """打印分隔线"""
    if title:
        print(f"\n{'=' * length}")
        print(f"  {title}")
        print(f"{'=' * length}\n")
    else:
        print(f"{'=' * length}\n")


def print_individual_genes(individual: Individual):
    """打印个体的基因信息"""
    print("📋 基因信息 (Genes):")
    print("-" * 60)
    for gene_name, gene_value in sorted(individual.genes.items()):
        if isinstance(gene_value, float):
            print(f"  {gene_name:.<40} {gene_value:.6f}")
        else:
            print(f"  {gene_name:.<40} {gene_value}")
    print()


def print_spine_geometry(config: dict):
    """打印脊椎几何参数"""
    spine = config["spine"]
    print("🦴 脊椎几何参数 (Spine Geometry):")
    print("-" * 60)
    print(f"  {'num_segments':<40} {spine['num_segments']}")
    print(f"  {'segment_spacing':<40} {spine['segment_spacing']:.6f} m")
    print(f"  {'alpha (α)':<40} {spine['alpha']:.6f} rad ({np.degrees(spine['alpha']):.2f}°)")
    print(f"  {'beta (β)':<40} {spine['beta']:.6f} rad ({np.degrees(spine['beta']):.2f}°)")
    print(f"  {'alpha_length':<40} {spine['alpha_length']:.6f} m")
    print(f"  {'beta_length':<40} {spine['beta_length']:.6f} m")
    print(f"  {'initial_z':<40} {spine['initial_z']:.6f} m")
    print(f"  {'fixed_length':<40} {spine['fixed_length']:.6f} m")
    print()


def print_tendon_parameters(config: dict):
    """打印所有8个拉索的刚度和阻尼"""
    defaults = config["defaults"]
    
    print("🎯 拉索参数 (Tendon Parameters):")
    print("-" * 60)
    print(f"{'拉索名称':<25} {'刚度 (Stiffness)':<20} {'阻尼 (Damping)':<20}")
    print("-" * 60)
    
    # 4个横向拉索
    lateral_tendons = ["lateral_top", "lateral_bottom", "lateral_left", "lateral_right"]
    for tendon_name in lateral_tendons:
        if tendon_name in defaults:
            stiff = defaults[tendon_name]["tendon"]["stiffness"]
            damp = defaults[tendon_name]["tendon"]["damping"]
            print(f"{tendon_name:<25} {stiff:<20.2f} {damp:<20.2f}")
    
    print()
    
    # 4个对角线拉索
    diagonal_tendons = ["diag_a1b1", "diag_a2b2", "diag_a1b2", "diag_a2b1"]
    for tendon_name in diagonal_tendons:
        if tendon_name in defaults:
            stiff = defaults[tendon_name]["tendon"]["stiffness"]
            damp = defaults[tendon_name]["tendon"]["damping"]
            print(f"{tendon_name:<25} {stiff:<20.2f} {damp:<20.2f}")
    
    print()


def print_summary_table(config: dict):
    """打印汇总表格"""
    defaults = config["defaults"]
    spine = config["spine"]
    
    print("\n📊 汇总表 (Summary):")
    print("-" * 80)
    
    # 脊椎参数
    print(f"\n脊椎参数 (Spine):")
    print(f"  • 节段数: {spine['num_segments']}")
    print(f"  • 节段间距: {spine['segment_spacing']:.4f} m")
    print(f"  • 形态参数 (α, β): ({spine['alpha']:.4f}, {spine['beta']:.4f}) rad")
    print(f"  • 长度参数: α_len={spine['alpha_length']:.4f}m, β_len={spine['beta_length']:.4f}m")
    
    # 拉索参数汇总
    all_tendons = ["lateral_top", "lateral_bottom", "lateral_left", "lateral_right",
                   "diag_a1b1", "diag_a2b2", "diag_a1b2", "diag_a2b1"]
    
    stiffnesses = []
    dampings = []
    for tendon_name in all_tendons:
        if tendon_name in defaults:
            stiffnesses.append(defaults[tendon_name]["tendon"]["stiffness"])
            dampings.append(defaults[tendon_name]["tendon"]["damping"])
    
    print(f"\n拉索参数 (Tendons - 8个):")
    print(f"  • 刚度范围: {min(stiffnesses):.2f} - {max(stiffnesses):.2f} N/m")
    print(f"  • 刚度平均值: {np.mean(stiffnesses):.2f} N/m")
    print(f"  • 阻尼范围: {min(dampings):.2f} - {max(dampings):.2f} Ns/m")
    print(f"  • 阻尼平均值: {np.mean(dampings):.2f} Ns/m")
    
    print()


def generate_xml_for_individual(config: dict, individual_id: int, output_dir: Path) -> str:
    """为个体生成 XML 文件"""
    # 设置输出路径
    xml_filename = f"individual_{individual_id}.xml"
    config['output_path'] = output_dir / xml_filename
    
    # 调用生成函数
    try:
        generate_quadruped_from_config(config, vis=True)
        xml_path = str(config['output_path'])
        print(f"✓ XML 文件已生成: {xml_path}")
        return xml_path
    except Exception as e:
        print(f"✗ 生成 XML 失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    print_separator("随机个体生成与参数展示")
    
    # ============ 第1部分：显示基因配置 ============
    print_separator("第1部分：基因配置信息")
    
    print("📖 可用的基因配置模式:")
    print("-" * 60)
    print("  • 'all': 包含所有高级基因")
    print("  • 'base': 基础基因（用于简单测试）")
    print("  • 'advanced': 高级基因（推荐）")
    print()
    
    print("✓ 使用 'advanced' 模式的基因范围:")
    print("-" * 60)
    for gene_name, (min_val, max_val) in sorted(BASE_GENE_RANGES.items()):
        if isinstance(min_val, float) or isinstance(max_val, float):
            print(f"  {gene_name:<30} [{min_val:.6f}, {max_val:.6f}]")
        else:
            print(f"  {gene_name:<30} [{min_val}, {max_val}]")
    print()
    
    # ============ 第2部分：创建随机个体 ============
    print_separator("第2部分：创建随机个体")
    
    # 创建3个随机个体
    num_individuals = 3
    individuals = []
    
    for i in range(num_individuals):
        ind = create_random_individual(
            individual_id=f"random_ind_{i}",
            generation=0,
            gene_ranges=BASE_GENE_RANGES
        )
        individuals.append(ind)
        print(f"✓ 创建个体 {i+1}/{num_individuals}: {ind.individual_id}")
    
    print()
    
    # ============ 第3部分：打印每个个体的详细参数并生成 XML ============
    output_dir = current_dir / "generated_xmls"
    os.makedirs(output_dir, exist_ok=True)  # 创建保存 XML 的目录
    
    for idx, individual in enumerate(individuals):
        print_separator(f"个体 {idx+1} 详细参数")
        
        # 打印基因
        print_individual_genes(individual)
        
        # 从基因构建配置
        config = build_config_from_genes(individual.genes, individual_id=idx)
        
        # 打印脊椎几何
        print_spine_geometry(config)
        
        # 打印拉索参数
        print_tendon_parameters(config)
        
        # 打印汇总
        print_summary_table(config)
        
        # 生成 XML 文件
        xml_path = generate_xml_for_individual(config, idx, output_dir)
        individual.xml_path = xml_path
    
    # ============ 第4部分：对比分析 ============
    print_separator("第4部分：个体间参数对比")
    
    print("📊 所有个体的关键参数对比:")
    print("-" * 120)
    print(f"{'个体ID':<20} {'num_seg':<12} {'spacing':<12} {'lat_stiff':<15} {'lat_damp':<12} {'diag_stiff':<15} {'diag_damp':<12}")
    print("-" * 120)
    
    for idx, individual in enumerate(individuals):
        config = build_config_from_genes(individual.genes, individual_id=idx)
        num_seg = config["spine"]["num_segments"]
        spacing = config["spine"]["segment_spacing"]
        lat_stiff = config["defaults"]["lateral_top"]["tendon"]["stiffness"]
        lat_damp = config["defaults"]["lateral_top"]["tendon"]["damping"]
        diag_stiff = config["defaults"]["diag_a1b1"]["tendon"]["stiffness"]
        diag_damp = config["defaults"]["diag_a1b1"]["tendon"]["damping"]
        
        print(f"{individual.individual_id:<20} {num_seg:<12} {spacing:<12.4f} {lat_stiff:<15.2f} {lat_damp:<12.2f} {diag_stiff:<15.2f} {diag_damp:<12.2f}")
    
    print()
    print_separator()
    print(f"✓ 所有 XML 文件已保存到: {output_dir}")
    print()


if __name__ == "__main__":
    main()
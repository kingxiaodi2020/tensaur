"""
统一的基因配置定义
"""
import numpy as np
# 标准基因范围定义
ALL_GENE_RANGES = {
    # base genes
    'num_segments': (2, 4),        # 节段数量
    'segment_spacing': (0.05, 0.10),  # 节段间距

    # tendon properties
    'lateral_stiffness': (1000, 10000),
    'lateral_damping': (5, 25),
    'diagonal_stiffness': (1000, 10000),
    'diagonal_damping': (5, 25),

    # advanced genes
    'alpha': (np.pi/4, np.pi/3),              # 形态参数α
    'beta': (np.pi/4, np.pi/3),               # 形态参数β
    'alpha_length': (0.06, 0.12),      # 形态参数α的长度
    'beta_length': (0.06, 0.12),       # 形态参数β的长度
}

# 测试用简化基因范围
BASE_GENE_RANGES = {
    'num_segments': (2, 3),
    # 'segment_spacing': (0.05, 0.10),
    'stiffness': (3000, 10000),
    'damping': (5, 25),
}

TENDON_GENE_RANGES = {
    # base genes
    'num_segments': (2, 4),        # 节段数量
    'segment_spacing': (0.05, 0.10),  # 节段间距

    # tendon properties
    'lateral_stiffness': (1000, 10000),
    'lateral_damping': (5, 25),
    'diagonal_stiffness': (1000, 10000),
    'diagonal_damping': (5, 25),
}

def get_gene_ranges(mode: str = 'all'):
    """获取基因范围配置"""
    if mode == 'all':
        return ALL_GENE_RANGES.copy()
    elif mode == 'base':
        return BASE_GENE_RANGES.copy()
    elif mode == 'tendon':
        return TENDON_GENE_RANGES.copy()
    else:
        raise ValueError(f"Unknown mode: {mode}")
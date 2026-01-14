"""
统一的基因配置定义
"""
import numpy as np

BASE_GENE_RANGES = {
    # base geometry
    'lateral_verti_stiffness': (500, 10000),
    'segment_spacing': (0.05, 0.12),
    
    # lever arm & resistance
    'alpha_length': (0.05, 0.12),      
    # 'beta_length': (0.06, 0.12),       
    # 'alpha': (np.pi/4, np.pi/3),            
    # 'beta': (np.pi/4, np.pi/3),              

    # # tendon properties
    # 'lateral_stiffness': (1000, 10000),
    
    # 'diagonal_stiffness': (1000, 10000),
}


MEDIUM_GENE_RANGES = {
    # # base geometry
    # 'num_segments': (2, 4),        
    # 'segment_spacing': (0.05, 0.10),  

    # # lever arm & resistance
    # 'alpha': (np.pi/4, np.pi/3),            
    # # 'beta': (np.pi/4, np.pi/3),              
    # 'alpha_length': (0.06, 0.12),     
    # 'beta_length': (0.06, 0.12),       

    # tendon properties
    # 'lateral_top_stiffness': (500, 15000),
    # 'lateral_bottom_stiffness': (500, 15000),
    # 'lateral_left_stiffness': (500, 15000),
    # 'lateral_right_stiffness': (500, 15000),
    'lateral_hori_stiffness': (500, 10000),
    'lateral_verti_stiffness': (500, 10000),
    'diagonal_stiffness': (500, 10000),
    
}

ADVANCED_GENE_RANGES = {
    # base geometry
    'num_segments': (2, 4),        
    'segment_spacing': (0.05, 0.10),

    # lever arm & resistance
    'alpha': (np.pi/4, np.pi/3),           
    'beta': (np.pi/4, np.pi/3),           
    'alpha_length': (0.06, 0.12),     
    'beta_length': (0.06, 0.12),    

    # tendon properties
    'lateral_top_stiffness': (1000, 10000),
    'lateral_bottom_stiffness': (1000, 10000),
    'lateral_left_stiffness': (1000, 10000),
    'lateral_right_stiffness': (1000, 10000),

    'diagonal_stiffness': (1000, 10000),
}

def get_gene_ranges(mode: str = 'all'):
    """select gene ranges based on mode"""
    if mode == 'medium':
        return MEDIUM_GENE_RANGES.copy()
    elif mode == 'base':
        return BASE_GENE_RANGES.copy()
    elif mode == 'advanced':
        return ADVANCED_GENE_RANGES.copy()
    else:
        raise ValueError(f"Unknown mode: {mode}")
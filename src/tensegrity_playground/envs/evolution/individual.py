import copy
import json
from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
from .gene_config import BASE_GENE_RANGES

@dataclass
class Individual:
    individual_id: str
    genes: Dict
    fitness: Optional[float] = None
    generation: int = 0
    training_completed: bool = False
    xml_path: Optional[str] = None
    
    def __post_init__(self):
        """验证基因参数"""
        required_genes = ['num_segments']
        for gene in required_genes:
            if gene not in self.genes:
                raise ValueError(f"Missing necessary gene: {gene}")
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            'individual_id': self.individual_id,
            'genes': self.genes,
            'fitness': self.fitness,
            'generation': self.generation,
            'training_completed': self.training_completed,
            'xml_path': self.xml_path
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Individual':
        """从字典创建个体"""
        return cls(**data)
    
    def save_to_file(self, filepath: str):
        """保存到文件"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'Individual':
        """从文件加载"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

def create_random_individual(individual_id: str, generation: int = 0, gene_ranges: Dict = None) -> Individual:
    """创建随机个体"""
    import random

    # 如果没有指定基因范围，使用默认值
    if gene_ranges is None:
        gene_ranges = BASE_GENE_RANGES
    
    genes = {}
    for gene_name, (min_val, max_val) in gene_ranges.items():
        if gene_name == 'num_segments':
            genes[gene_name] = random.randint(int(min_val), int(max_val))
        else:
            genes[gene_name] = random.uniform(min_val, max_val)
    
    return Individual(
        individual_id=individual_id,
        genes=genes,
        generation=generation
    )
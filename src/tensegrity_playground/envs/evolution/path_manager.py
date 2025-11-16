"""
路径管理器 - 统一管理项目中的所有路径，确保跨环境兼容
"""
from email.mime import base
from pathlib import Path
from typing import Optional
import os

class PathManager:
    """项目路径管理器 - 自动检测项目根目录并提供统一的路径接口"""
    
    def __init__(self):
        self._project_root = self._find_project_root()
        
    def _find_project_root(self) -> Path:
        """自动查找项目根目录"""
        # 从当前文件开始向上查找，直到找到包含特定标识文件的目录
        current = Path(__file__).resolve()
        
        # 项目根目录的标识文件/目录
        root_indicators = [
            'pyproject.toml',
            'config',
            'scripts',
            'src/tensegrity_playground'
        ]
        
        for parent in [current] + list(current.parents):
            # 检查是否包含所有标识
            if all((parent / indicator).exists() for indicator in root_indicators):
                return parent
        
        # 如果找不到，返回当前文件的父目录的父目录的父目录的父目录
        # src/tensegrity_playground/envs/evolution -> 项目根目录
        fallback = current.parent.parent.parent.parent
        print(f"⚠️ 未找到项目根目录标识，使用回退路径: {fallback}")
        return fallback
    
    @property
    def project_root(self) -> Path:
        """项目根目录"""
        return self._project_root
    
    @property
    def config_dir(self) -> Path:
        """配置文件目录"""
        return self._project_root / "config"
    
    @property
    def scripts_dir(self) -> Path:
        """脚本目录"""
        return self._project_root / "scripts"
    
    @property
    def temp_dir(self) -> Path:
        """临时文件目录"""
        temp_dir = self._project_root / "temp_training"
        temp_dir.mkdir(parents=True, exist_ok=True)
        return temp_dir
    
    @property
    def logs_dir(self) -> Path:
        """日志目录"""
        logs_dir = self._project_root / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        return logs_dir
    
    @property
    def xmls_dir(self) -> Path:
        """XML文件目录"""
        xmls_dir = self._project_root / "src/tensegrity_playground/envs/tensaur/xmls"
        xmls_dir.mkdir(parents=True, exist_ok=True)
        return xmls_dir
    
    def get_config_path(self, config_name: str) -> Path:
        """获取配置文件路径"""
        return self.config_dir / config_name
    
    def get_checkpoint_dir(self, individual_id: str) -> Path:
        """获取个体的checkpoint目录"""
        checkpoint_dir = self.temp_dir / f"checkpoints_{individual_id}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return checkpoint_dir
    
    def get_individual_config_path(self, individual_id: str) -> Path:
        """获取个体配置文件路径"""
        return self.config_dir / f"individual_{individual_id}.yaml"
    
    def get_script_path(self, script_name: str) -> Path:
        """获取脚本路径"""
        return self.scripts_dir / script_name
    
    def get_metrics_file(self, individual_id: str) -> Path:
        """获取metrics文件路径"""
        checkpoint_dir = self.get_checkpoint_dir(individual_id)
        # 查找metrics_final.json文件
        metrics_files = list(checkpoint_dir.glob("*/metrics_final.json"))
        if metrics_files:
            return metrics_files[0]
        else:
            # 如果没找到，返回预期路径
            return checkpoint_dir / "latest" / "metrics_final.json"
        
    def get_default_logging_dir(self, individual_id: str) -> Path:
        """
        默认的日志根目录：<project_root>/logs/ind_<id>/
        """
        base = self.logs_dir / f"ind_{individual_id}"
        base.mkdir(parents=True, exist_ok=True)
        return base


# 全局路径管理器实例
path_manager = PathManager()

def get_path_manager() -> PathManager:
    """获取全局路径管理器实例"""
    return path_manager
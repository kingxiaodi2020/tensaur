import subprocess
import tempfile
import yaml
import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional

sys.path.append('src')
from .individual import Individual
from .path_manager import get_path_manager
from tensegrity_playground.envs.tensaur.generate_go1 import build_config_from_genes, generate_quadruped_from_config

class SingleIndividualTrainer:
    """训练单个个体的训练器"""
    
    def __init__(self, base_config_path: Optional[str] = None):
        self.path_manager = get_path_manager()
        
        # 使用路径管理器获取配置路径
        if base_config_path is None:
            self.base_config_path = self.path_manager.get_config_path("playground_brax.yaml")
        else:
            self.base_config_path = Path(base_config_path)
        
        print(f"初始化训练器")
        print(f"  项目根目录: {self.path_manager.project_root}")
        print(f"  配置文件: {self.base_config_path}")
    
        self.ckpt_dir_env = os.environ.get("CKPT_DIR")
        self.logging_dir = None  # 将在 run 前确定

    def prepare_individual_xml(self, individual: Individual) -> str:
        """为个体准备XML文件"""
        # 生成XML配置
        config = build_config_from_genes(individual.genes, individual.individual_id)
        
        # 生成XML文件
        generate_quadruped_from_config(config)
        
        xml_path = str(config['output_path'])
        individual.xml_path = xml_path
        
        return xml_path

    def _make_logging_dir(self, individual_id: str) -> Path:
        if self.ckpt_dir_env:                      # 优先集中归档
            logdir = Path(self.ckpt_dir_env).resolve()
        else:                                      # 回退到原先默认目录（你已有的 PathManager 逻辑）
            logdir = self.path_manager.get_default_logging_dir(individual_id)
        logdir.mkdir(parents=True, exist_ok=True)
        self.logging_dir = logdir
        return logdir
    
    def train_individual(self, individual: Individual, mode: str = "single_objective") -> Dict:
        """
        训练单个个体并返回训练结果
        
        Args:
            individual: 待训练个体
            mode: "single_objective" 或 "multi_objective"
        
        Returns:
            包含训练结果的字典
        """
        print(f"start training individual {individual.individual_id} (mode={mode})")
        
        if mode == "multi_objective":
            # ✅ 多目标：顺序执行 radius → walk
            return self._train_multi_objective(individual)
        else:
            # 单目标：保持原有逻辑
            return self._train_single_objective(individual)


    def _train_multi_objective(self, individual: Individual, num_seeds: int = 3) -> Dict:
        """✅ 多目标训练：radius + walk，每个任务运行多个 seed 并取平均"""
        
        # 1. 准备 XML（只需一次）
        xml_path = self.prepare_individual_xml(individual)
        print(f"  XML文件已生成: {xml_path}")
        
        # 2. 任务 1: TurnRadius (plane terrain) - 运行多个 seed
        print(f"\n[Task 1/2] Training TurnRadius with {num_seeds} seeds...")
        radius_results_list = []
        for seed_idx in range(num_seeds):
            seed = 1 + seed_idx  # seeds: 1, 2, 3
            print(f"  Seed {seed}/{num_seeds}...")
            results = self._train_task(
                individual=individual,
                xml_path=xml_path,
                environment="TensegrityQuadrupedTurnRadius",
                terrain_mode="plane",
                task_suffix=f"radius_seed{seed}",
                seed=seed
            )
            radius_results_list.append(results)
        
        # 3. 任务 2: Walk (uneven terrain) - 运行多个 seed
        print(f"\n[Task 2/2] Training Walk with {num_seeds} seeds...")
        walk_results_list = []
        for seed_idx in range(num_seeds):
            seed = 1 + seed_idx  # seeds: 1, 2, 3
            print(f"  Seed {seed}/{num_seeds}...")
            results = self._train_task(
                individual=individual,
                xml_path=xml_path,
                environment="TensegrityQuadrupedWalk",
                terrain_mode="uneven",
                task_suffix=f"walk_seed{seed}",
                seed=seed
            )
            walk_results_list.append(results)
        
        # 4. ✅ 计算平均值
        avg_r_deviation = sum(r.get("r_deviation", -1000.0) for r in radius_results_list) / num_seeds
        avg_max_x_final = sum(w.get("max_x_final", -1000.0) for w in walk_results_list) / num_seeds
        
        objectives = [float(avg_r_deviation), float(avg_max_x_final)]
        
        individual.objectives = objectives
        individual.training_completed = (
            all(r.get("training_completed", False) for r in radius_results_list) and 
            all(w.get("training_completed", False) for w in walk_results_list)
        )
        
        print(f"\n✅ 多目标训练完成 (平均 {num_seeds} seeds):")
        print(f"  Objective 1 (avg r_deviation): {objectives[0]:.3f}")
        print(f"  Objective 2 (avg max_x_final): {objectives[1]:.3f}")
        print(f"  Radius results: {[r.get('r_deviation', -1000.0) for r in radius_results_list]}")
        print(f"  Walk results: {[w.get('max_x_final', -1000.0) for w in walk_results_list]}")
        
        return {
            "objectives": objectives,
            "radius_results_list": radius_results_list,
            "walk_results_list": walk_results_list,
            "avg_r_deviation": avg_r_deviation,
            "avg_max_x_final": avg_max_x_final
        }

    def _train_task(
        self, 
        individual: Individual, 
        xml_path: str, 
        environment: str, 
        terrain_mode: str,
        task_suffix: str,
        seed: int = 1
    ) -> Dict:
        """✅ 训练单个任务"""
        
        try:
            # 1. 动态修改 XML 的 terrain mode
            xml_with_terrain = self._modify_xml_terrain(xml_path, terrain_mode, individual)
            
            # 2. 创建日志目录
            self.logging_dir = self._make_logging_dir(f"{individual.individual_id}_{task_suffix}")
            
            # 3. 备份 XML
            import shutil
            xml_dest = self.logging_dir / f"{individual.individual_id}_{task_suffix}.xml"
            shutil.copy2(xml_with_terrain, xml_dest)
            print(f"  ✓ XML备份: {xml_dest}")
            
            # 4. 创建配置文件
            config_path = self._create_task_config(
                individual=individual,
                xml_path=xml_with_terrain,
                environment=environment,
                task_suffix=task_suffix,
                seed=seed
            )
            print(f"  配置文件已创建: {config_path}")
            
            # 5. 选择训练脚本
            if environment == "TensegrityQuadrupedTurnRadius":
                script_name = "train_radius.py"
            else:
                script_name = "train_walk.py"
            
            # 6. ✅ 运行训练（只覆盖 XML 路径，不覆盖环境）
            config_name = Path(config_path).stem
            script_path = self.path_manager.get_script_path(script_name)
            config_dir = self.path_manager.config_dir
            
            cmd = [
                sys.executable, str(script_path),
                "--config-path", str(config_dir),
                "--config-name", config_name,
                # ❌ 删除这行：f"playground={environment}",
            ]
            
            print(f"  执行命令: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                # capture_output=True,  # ✅ 捕获输出
                text=True,
                cwd=str(self.path_manager.project_root)
            )
            
            # ✅ 打印错误信息
            if result.returncode != 0:
                print(f"  ❌ 训练失败 (返回码: {result.returncode})")
                print(f"  stderr: {result.stderr}")
                return self._get_default_task_results(environment)
            
            # 7. 读取结果
            metrics_file = self._find_metrics_in_logdir()
            if metrics_file and metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                return self._extract_task_results(metrics, environment)
            else:
                print(f"  ⚠️ 未找到 metrics 文件")
                return self._get_default_task_results(environment)
        
        except Exception as e:
            print(f"  ✗ 训练出错: {e}")
            import traceback
            traceback.print_exc()
            return self._get_default_task_results(environment)

    def _modify_xml_terrain(self, xml_path: str, terrain_mode: str, individual: Individual) -> str:
        """✅ 动态修改 XML 的 terrain mode"""
        from tensegrity_playground.envs.tensaur.generate_go1 import (
            build_config_from_genes, 
            generate_quadruped_from_config,
            DEFAULT_CONFIG
        )
            
        # 重新生成 XML，覆盖 terrain mode
        config = build_config_from_genes(
            genes=individual.genes,  # 保持默认 genes
            individual_id=individual.individual_id,
            output_path=xml_path
        )
        config["terrain"]["mode"] = terrain_mode  # ✅ 关键修改
        
        # 生成新的 XML
        generate_quadruped_from_config(config, vis=True)
        
        return str(config["output_path"])

    def _create_task_config(
        self, 
        individual: Individual, 
        xml_path: str, 
        environment: str,
        task_suffix: str,
        seed: int = 1
    ) -> str:
        """✅ 创建任务专用配置文件"""
        
        base_config_path = self.base_config_path
        ppo_config_path = self.path_manager.get_config_path("agent/ppo.yaml")
        
        # 根据环境选择 playground 配置
        if environment == "TensegrityQuadrupedTurnRadius":
            playground_config_path = self.path_manager.get_config_path("playground/TensegrityQuadrupedTurnRadius.yaml")
        else:
            playground_config_path = self.path_manager.get_config_path("playground/TensegrityQuadrupedWalk.yaml")
        
        # 读取配置
        with open(base_config_path, 'r') as f:
            base_config = yaml.safe_load(f)
        
        with open(ppo_config_path, 'r') as f:
            ppo_config = yaml.safe_load(f)
        
        with open(playground_config_path, 'r') as f:
            playground_config = yaml.safe_load(f) or {}
        
        # ✅ 修改 defaults 中的环境配置
        defaults = base_config.get('defaults', [])
        
        # 查找并替换 playground 配置
        for i, default in enumerate(defaults):
            if isinstance(default, dict) and 'playground' in default:
                defaults[i] = {'playground': environment}  # ✅ 替换环境名
                break
        else:
            # 如果没找到，添加新的
            defaults.append({'playground': environment})
        
        # 创建配置
        config = {
            'defaults': defaults,  # ✅ 使用修改后的 defaults
            'hydra': base_config.get('hydra', {}),
            'checkpointing': base_config.get('checkpointing', True),
            'checkpoint_directory': str(self.logging_dir),
            'export_video': base_config.get('export_video', False),
            'render_every': base_config.get('render_every', 1),
            'render_camera': base_config.get('render_camera', -1),
            'wandb': base_config.get('wandb', False),
            'wandb_project': base_config.get('wandb_project', 'learning'),
            'wandb_entity': base_config.get('wandb_entity', 'tensegrity'),
            'tensorboard': base_config.get('tensorboard', True),
            'progress_bar': base_config.get('progress_bar', False),
            'verbose': base_config.get('verbose', True),
            'agent': {
                **ppo_config,
                'seed': seed  # ✅ 设置自定义 seed
            },
            'playground': {
                **playground_config,
                'xml': xml_path,
                # ❌ 删除 '_target_': environment
            }
        }
        
        print(f"  ✅ 配置 defaults: {config['defaults']}")
        print(f"  ✅ playground.xml: {xml_path}")
        
        # 保存配置
        config_path = self.path_manager.get_individual_config_path(f"{individual.individual_id}_{task_suffix}")
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        return str(config_path)

    def _extract_task_results(self, metrics: Dict, environment: str) -> Dict:
        """✅ 从 metrics 提取任务结果"""
        if environment == "TensegrityQuadrupedTurnRadius":
            # Radius: 提取 r_deviation
            r_deviation = -metrics.get('eval/episode_r_deviation', -10000.0)
            return {
                "r_deviation": float(r_deviation),
                "training_completed": True
            }
        else:
            # Walk: 提取 max_x_final
            max_x_final = metrics.get('eval/episode_max_x_final', -10000.0)
            return {
                "max_x_final": float(max_x_final),
                "training_completed": True
            }

    def _get_default_task_results(self, environment: str) -> Dict:
        """✅ 获取默认惩罚值"""
        if environment == "TensegrityQuadrupedTurnRadius":
            return {"r_deviation": -9999.0, "training_completed": False}
        else:
            return {"max_x_final": -9999.0, "training_completed": False}

#  Original single-objective training method

    def create_training_config(self, individual: Individual, xml_path: str) -> str:
        """创建训练配置文件"""
        
        # 使用路径管理器获取配置文件路径
        base_config_path = self.base_config_path
        ppo_config_path = self.path_manager.get_config_path("agent/ppo.yaml")
        playground_config_path = self.path_manager.get_config_path("playground/TensegrityQuadrupedTurnRadius.yaml")

        # 读取配置文件
        with open(base_config_path, 'r') as f:
            base_config = yaml.safe_load(f)
        
        with open(ppo_config_path, 'r') as f:
            ppo_config = yaml.safe_load(f)

        with open(playground_config_path, 'r') as f:
            playground_config = yaml.safe_load(f) or {}

        # # 使用路径管理器获取checkpoint目录
        # checkpoint_dir = self.path_manager.get_checkpoint_dir(individual.individual_id)
        
        # 创建完整配置
        config = {
            'defaults': base_config.get('defaults', []),
            'hydra': base_config.get('hydra', {}),
            'checkpointing': base_config.get('checkpointing', True),
            'checkpoint_directory': str(self.logging_dir),
            'export_video': base_config.get('export_video', False),
            'render_every': base_config.get('render_every', 1),
            'render_camera': base_config.get('render_camera', -1),
            'wandb': base_config.get('wandb', False),
            'wandb_project': base_config.get('wandb_project', 'learning'),
            'wandb_entity': base_config.get('wandb_entity', 'tensegrity'),
            'tensorboard': base_config.get('tensorboard', True),
            'progress_bar': base_config.get('progress_bar', False),
            'verbose': base_config.get('verbose', True),
            'agent': {
                **ppo_config,
            },
            'playground': {
                **playground_config,
                'xml': xml_path
            }
        }
        
        # 保存配置文件
        config_path = self.path_manager.get_individual_config_path(individual.individual_id)
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        return str(config_path)

    def _train_single_objective(self, individual: Individual, mode: str = "single_objective") -> Dict:
        """
        训练单个个体并返回训练结果
        
        Args:
            individual: 待训练个体
            mode: "single_objective" 或 "multi_objective"
        
        Returns:
            包含训练结果的字典
        """
        print(f"start training individual {individual.individual_id} (mode={mode})")
        
        try:
            # 1. 准备XML文件
            xml_path = self.prepare_individual_xml(individual)
            print(f"  XML文件已生成: {xml_path}")

            self.logging_dir = self._make_logging_dir(individual.individual_id)

            import shutil   # ✅ 复制 XML 到日志目录（在训练前备份）
            xml_dest = self.logging_dir / f"{individual.individual_id}.xml"
            shutil.copy2(xml_path, xml_dest)
            print(f"  ✓ XML备份: {xml_dest}")

            # 2. 创建训练配置
            config_path = self.create_training_config(individual, xml_path)
            print(f"  配置文件已创建: {config_path}")
            
            # 3. 运行训练
            config_name = Path(config_path).stem
            script_path = self.path_manager.get_script_path("train_run.py")
            # script_path = self.path_manager.get_script_path("train_radius.py")
            # script_path = self.path_manager.get_script_path("train_walk.py")
            config_dir = self.path_manager.config_dir
            
            cmd = [
                sys.executable, str(script_path),
                "--config-path", str(config_dir),
                "--config-name", config_name,
                "playground=TensegrityQuadrupedRun",  # ✅ Hydra 覆盖
                # "playground=TensegrityQuadrupedTurnRadius",  # ✅ Hydra 覆盖
                # "playground=TensegrityQuadrupedWalk",  # ✅ Hydra 覆盖
            ]

            print(f"  执行命令: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                text=True,
                cwd=str(self.path_manager.project_root)
            )
            
            # 4. 获取并解析 metrics
            metrics_file = self._find_metrics_in_logdir()
            if not metrics_file:
                metrics_file = self.path_manager.get_metrics_file(individual.individual_id)

            # 5. 根据模式提取不同的结果
            if result.returncode == 0 and metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                results = self._extract_results(metrics, mode)
            else:
                if result.stdout:
                    print("  未找到 metrics_final.json 或训练返回码非0，回退解析 stdout")
                    # results = self._extract_results_from_output(result.stdout, mode)
                else:
                    print("  无可用数据，使用默认惩罚值")
                    results = self._get_default_results(mode)

            # 6. 更新 individual 状态
            individual.training_completed = (result.returncode == 0)
            
            if mode == "single_objective":
                individual.fitness = results["fitness"]
                print(f"  ✓ 训练完成，fitness: {results['fitness']:.2f}")
            else:  # multi_objective
                individual.objectives = results["objectives"]
                print(f"  ✓ 训练完成，objectives: {results['objectives']}")

            return results

        except Exception as e:
            print(f"  ✗ 训练出错: {e}")
            import traceback
            traceback.print_exc()
            
            individual.training_completed = False
            results = self._get_default_results(mode)
            
            if mode == "single_objective":
                individual.fitness = results["fitness"]
            else:
                individual.objectives = results["objectives"]
            
            return results

    def _extract_results(self, metrics: Dict, mode: str) -> Dict:
        """从 metrics 中提取结果"""
        if mode == "single_objective":
            # 单目标：返回综合适应度
            # fitness = - metrics.get('eval/episode_r_deviation', -1000.0)
            fitness = metrics.get('eval/episode_max_x_final', -1000.0)
            return {"fitness": float(fitness)}
        
        else:  # multi_objective
            # 多目标：提取多个独立目标
            vel_x = metrics.get('eval/episode_actual_vel_x', -1e9)
            vel_y = metrics.get('eval/episode_actual_vel_y', -1e9)
            
            # 目标3：能量效率（可选）
            # energy_efficiency = -metrics.get('eval/energy_consumption', 1e9)
            
        return {
            "objectives": [float(vel_x), float(vel_y)],
            "vel_x": float(vel_x),  # 额外保存便于调试
            "vel_y": float(vel_y)   # 额外保存便于调试
        }
        
    def _get_default_results(self, mode: str) -> Dict:
        """获取默认惩罚值"""
        if mode == "single_objective":
            return {"fitness": 0.0}
        else:
            return {"objectives": [0.0, 0.0]}  # 根据目标数量调整
        
    def _find_metrics_in_logdir(self) -> Path | None:
        root = Path(self.logging_dir)
        # 1) 先看根目录
        p = root / "metrics_final.json"
        if p.exists():
            return p
        # 2) 再在一层子目录里找（train_ppo 用时间戳建子目录）
        candidates = sorted(root.glob("*/metrics_final.json"), key=lambda x: x.stat().st_mtime, reverse=True)
        return candidates[0] if candidates else None
            
    def cleanup(self):
        """清理临时文件"""
        # 清理生成的个体配置文件
        config_dir = self.path_manager.config_dir
        for pattern in ["individual_*.yaml"]:
            for config_file in config_dir.glob(pattern):
                config_file.unlink()
                print(f"✓ 清理文件: {config_file}")
        
        # 清理训练目录
        temp_dir = self.path_manager.temp_dir
        if temp_dir.exists():
            import shutil
            shutil.rmtree(temp_dir)
            print(f"✓ 清理训练目录: {temp_dir}")
    

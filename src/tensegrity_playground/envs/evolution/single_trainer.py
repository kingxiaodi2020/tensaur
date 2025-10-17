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

    def create_training_config(self, individual: Individual, xml_path: str) -> str:
        """创建训练配置文件"""
        
        # 使用路径管理器获取配置文件路径
        base_config_path = self.base_config_path
        ppo_config_path = self.path_manager.get_config_path("agent/ppo.yaml")
        playground_config_path = self.path_manager.get_config_path("playground/TensegrityQuadrupedWalk.yaml")

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

    def _make_logging_dir(self, individual_id: str) -> Path:
        if self.ckpt_dir_env:                      # 优先集中归档
            logdir = Path(self.ckpt_dir_env).resolve()
        else:                                      # 回退到原先默认目录（你已有的 PathManager 逻辑）
            logdir = self.path_manager.get_default_logging_dir(individual_id)
        logdir.mkdir(parents=True, exist_ok=True)
        self.logging_dir = logdir
        return logdir
    
    def train_individual(self, individual: Individual) -> float:
        """训练单个个体并返回适应度"""
        print(f"start training individual {individual.individual_id}")
        
        try:
            # 1. 准备XML文件
            xml_path = self.prepare_individual_xml(individual)
            print(f"  XML文件已生成: {xml_path}")

            self.logging_dir = self._make_logging_dir(individual.individual_id)

            # 2. 创建训练配置
            config_path = self.create_training_config(individual, xml_path)
            print(f"  配置文件已创建: {config_path}")
            
            # 3. 运行训练
            config_name = Path(config_path).stem
            script_path = self.path_manager.get_script_path("train_ppo.py")
            config_dir = self.path_manager.config_dir
            
            cmd = [
                sys.executable, str(script_path),
                "--config-path", str(config_dir),
                "--config-name", config_name,
            ]

            print(f"  执行命令: {' '.join(cmd)}")
            
            # 运行训练（从项目根目录运行）
            result = subprocess.run(
                cmd,
                text=True,
                cwd=str(self.path_manager.project_root)  # 确保从项目根目录运行
            )
            
            # 获取metrics文件
            metrics_file = self._find_metrics_in_logdir()
            if not metrics_file:
                metrics_file = self.path_manager.get_metrics_file(individual.individual_id)

            fitness = 0.0
            if result.returncode == 0 and metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                fitness = metrics.get('eval/episode_reward', 0.0)
            else:
                if result.stdout:
                    print("  未找到 metrics_final.json 或训练返回码非0，回退解析 stdout")
                    fitness = self.extract_fitness_from_output(result.stdout)
                else:
                    print("  无可用 stdout，适应度置 0.0")
                    fitness = 0.0

            individual.fitness = fitness
            individual.training_completed = True if result.returncode == 0 else False
                
            if result.returncode == 0:
                print(f"  ✓ 训练完成，适应度: {fitness:.2f}")
            else:
                print(f"  ✗ 训练失败，返回码: {result.returncode}，适应度: {fitness:.2f}")

            return fitness

        except Exception as e:
            print(f"  ✗ 训练出错: {e}")
            import traceback
            traceback.print_exc()
            individual.fitness = 0.0
            individual.training_completed = False
            return 0.0

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
    
    def extract_fitness_from_output(self, output: str) -> float:
        """从训练输出中提取适应度"""
        lines = output.split('\n')
        
        # 存储关键指标的最新值
        metrics = {
            'eval_episode_reward': [],
            'eval_episode_actual_vel_x': [],
            'eval_sps': [],
            'episode_reward': [],  # 备用指标
        }
        
        for line in lines:
            if 'Step' in line and ':' in line:
                # 解析类似 "Step 12345: {'eval/episode_reward': 1234.56, ...}" 的行
                try:
                    # 提取字典部分
                    if '{' in line and '}' in line:
                        dict_str = line[line.find('{'):line.rfind('}')+1]
                        # 安全地评估字典
                        import ast
                        step_metrics = ast.literal_eval(dict_str)
                        
                        # 提取关键指标
                        for key, value in step_metrics.items():
                            if 'eval/episode_reward' in key:
                                metrics['eval_episode_reward'].append(float(value))
                            elif 'eval/episode_actual_vel_x' in key:
                                metrics['eval_episode_actual_vel_x'].append(float(value))
                            elif 'eval_sps' in key:
                                metrics['eval_sps'].append(float(value))
                            elif 'episode_reward' in key and 'eval' not in key:
                                metrics['episode_reward'].append(float(value))
                                
                except (ValueError, SyntaxError, KeyError):
                    # 如果解析失败，尝试正则表达式
                    import re
                    
                    # 寻找 eval/episode_reward
                    reward_match = re.search(r'eval/episode_reward[\'"]?\s*:\s*([0-9.-]+)', line)
                    if reward_match:
                        metrics['eval_episode_reward'].append(float(reward_match.group(1)))
                    
                    # 寻找 eval/episode_actual_vel_x
                    vel_match = re.search(r'eval/episode_actual_vel_x[\'"]?\s*:\s*([0-9.-]+)', line)
                    if vel_match:
                        metrics['eval_episode_actual_vel_x'].append(float(vel_match.group(1)))
                    
                    # 寻找 eval_sps
                    sps_match = re.search(r'eval_sps[\'"]?\s*:\s*([0-9.-]+)', line)
                    if sps_match:
                        metrics['eval_sps'].append(float(sps_match.group(1)))
        
        # 计算综合适应度分数
        fitness = self._calculate_composite_fitness(metrics)
        
        # print(f"    提取的指标:")
        # if metrics['eval_episode_reward']:
        #     print(f"      eval/episode_reward: {metrics['eval_episode_reward'][-3:]} (最后3个)")
        # if metrics['eval_episode_actual_vel_x']:
        #     print(f"      eval/episode_actual_vel_x: {metrics['eval_episode_actual_vel_x'][-3:]} (最后3个)")
        # if metrics['eval_sps']:
        #     print(f"      eval_sps: {metrics['eval_sps'][-3:]} (最后3个)")
        # print(f"      综合适应度: {fitness}")
        
        return fitness

    def _calculate_composite_fitness(self, metrics: dict) -> float:
        """计算综合适应度分数"""
        
        # 基础适应度：优先使用 eval_episode_reward (范围: 500~1500)
        base_fitness = 0.0
        if metrics['eval_episode_reward']:
            # 使用最后几次评估的平均值，更稳定
            recent_rewards = metrics['eval_episode_reward'][-5:]  # 最后5次评估
            base_fitness = sum(recent_rewards) / len(recent_rewards)
            print(f"      使用 eval_episode_reward，最后5次平均: {base_fitness:.2f}")
        else:
            # 只有当没有 eval 数据时才使用训练数据作为备用
            if metrics['episode_reward']:
                recent_rewards = metrics['episode_reward'][-10:]
                base_fitness = sum(recent_rewards) / len(recent_rewards)
                print(f"      使用 episode_reward 作为备用，最后10次平均: {base_fitness:.2f}")
            else:
                print(f"      警告：没有找到任何 reward 数据！")

        # 速度奖励：eval_episode_actual_vel_x (范围: ~1000，表示累计1000个环境的速度)
        vel_bonus = 0.0
        if metrics['eval_episode_actual_vel_x']:
            recent_vels = metrics['eval_episode_actual_vel_x'][-5:]
            avg_vel = sum(recent_vels) / len(recent_vels)
            
            # 将累计速度转换为单个环境的平均速度
            avg_vel_per_env = avg_vel / 1000  # 假设是1000个环境的累计
            target_vel_per_env = 1.0  # 目标：单个环境1 m/s
            
            # 速度越接近目标，奖励越高，但权重适中
            vel_error = abs(avg_vel_per_env - target_vel_per_env)
            # 调整权重：最大给200分的速度奖励，与reward同等重要
            vel_bonus = max(0, 200 * (1 - vel_error))
            
            print(f"      累计速度: {avg_vel:.1f}, 平均速度: {avg_vel_per_env:.3f} m/s, 目标: {target_vel_per_env} m/s, 速度奖励: {vel_bonus:.2f}")
        else:
            print(f"      警告：没有找到速度数据！")
        
        # 效率奖励：eval_sps (范围: 2k~3k)
        efficiency_bonus = 0.0
        if metrics['eval_sps']:
            recent_sps = metrics['eval_sps'][-3:]
            avg_sps = sum(recent_sps) / len(recent_sps)
            
            # SPS越高，效率奖励越高，但权重较小
            # 2000 SPS = 20分, 3000 SPS = 30分
            efficiency_bonus = min(50, avg_sps / 100)  # 最大50分的效率奖励
            print(f"      平均SPS: {avg_sps:.0f}, 效率奖励: {efficiency_bonus:.2f}")
        
        # 综合适应度权重分配：
        # - base_fitness: 500~1500 (主要，占大头)
        # - vel_bonus: 0~200 (重要，确保速度正确)  
        # - efficiency_bonus: 0~50 (次要，训练效率)
        total_fitness = base_fitness + vel_bonus + efficiency_bonus
        
        print(f"      适应度组成: base={base_fitness:.2f} + vel={vel_bonus:.2f} + eff={efficiency_bonus:.2f} = {total_fitness:.2f}")
        
        return max(0.0, total_fitness)  # 确保非负
    
    def _find_metrics_in_logdir(self) -> Path | None:
        root = Path(self.logging_dir)
        # 1) 先看根目录
        p = root / "metrics_final.json"
        if p.exists():
            return p
        # 2) 再在一层子目录里找（train_ppo 用时间戳建子目录）
        candidates = sorted(root.glob("*/metrics_final.json"), key=lambda x: x.stat().st_mtime, reverse=True)
        return candidates[0] if candidates else None


    # import subprocess
# import tempfile
# import yaml
# import json
# import os
# import sys
# from pathlib import Path
# from typing import Dict, Optional

# sys.path.append('src')
# from .individual import Individual
# from tensegrity_playground.envs.tensaur.generate_go1 import build_config_from_genes, generate_quadruped_from_config

# class SingleIndividualTrainer:
#     """训练单个个体的训练器"""
    
#     def __init__(self, base_config_path: str = "config/playground_brax.yaml"):
#         self.base_config_path = Path(base_config_path).resolve() 
#         self.temp_dir = Path("config").resolve() # config目录
#         # print(f"初始化训练器，临时目录: {self.temp_dir.absolute()}")  # 添加调试信息
    
#     def prepare_individual_xml(self, individual: Individual) -> str:
#         """为个体准备XML文件"""
#         # 生成XML配置
#         config = build_config_from_genes(individual.genes, individual.individual_id)
        
#         # 生成XML文件
#         generate_quadruped_from_config(config)
        
#         xml_path = str(config['output_path'])
#         individual.xml_path = xml_path
        
#         return xml_path

#     def create_training_config(self, individual: Individual, xml_path: str) -> str:
#         """创建训练配置文件"""
        
#         # 读取基础配置
#         with open(self.base_config_path, 'r') as f:
#             import yaml
#             base_config = yaml.safe_load(f)
        
#         # 读取 PPO 配置
#         ppo_config_path = "config/agent/ppo.yaml"
#         with open(ppo_config_path, 'r') as f:
#             ppo_config = yaml.safe_load(f)

#         # 读取 TensegrityQuadrupedWalk 配置
#         playground_config_path = "config/playground/TensegrityQuadrupedWalk.yaml"
#         with open(playground_config_path, 'r') as f:
#             playground_config = yaml.safe_load(f) or {}

#         checkpoint_base = Path("./temp_training").resolve()  # 绝对路径
#         checkpoint_dir = checkpoint_base / f"checkpoints_{individual.individual_id}"
        
#         # 创建完整配置 - 不包含 playground.xml，通过命令行传递
#         config = {
#             'defaults': base_config.get('defaults', []),
#             'hydra': base_config.get('hydra', {}),
#             'checkpointing': base_config.get('checkpointing', True),
#             'checkpoint_directory': str(checkpoint_dir),
#             'export_video': base_config.get('export_video', False),
#             'render_every': base_config.get('render_every', 1),
#             'render_camera': base_config.get('render_camera', -1),
#             'wandb': base_config.get('wandb', False),
#             'wandb_project': base_config.get('wandb_project', 'learning'),
#             'wandb_entity': base_config.get('wandb_entity', 'tensegrity'),
#             'tensorboard': base_config.get('tensorboard', True),
#             'progress_bar': base_config.get('progress_bar', False),
#             'verbose': base_config.get('verbose', True),
#             'agent': {
#                 **ppo_config,
#                 # 'seed': 42,
#                 # 'num_timesteps': 500000,
#             },
#             'playground': {
#                 **playground_config,
#                 'xml': xml_path
#             }
#         }
        
#         # 保存配置文件，同时保存 XML 路径信息供后续使用
#         config_path = self.temp_dir / f"individual_{individual.individual_id}.yaml"
        
#         with open(config_path, 'w') as f:
#             yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
#         return str(config_path)

#     def train_individual(self, individual: Individual) -> float:
#         """训练单个个体并返回适应度"""
#         print(f"start training individual {individual.individual_id}")
        
#         try:
#             # 1. 准备XML文件
#             xml_path = self.prepare_individual_xml(individual)
#             print(f"  XML文件已生成: {xml_path}")
            
#             # 2. 创建训练配置
#             config_path = self.create_training_config(individual, xml_path)
#             print(f"  配置文件已创建: {config_path}")
            
#             # 3. 运行训练
#             config_name = Path(config_path).stem
#             script_path = "scripts/train_ppo.py"
#             config_dir_abs = os.path.abspath("config")
            
#             cmd = [
#                 sys.executable, script_path,
#                 "--config-path", config_dir_abs,
#                 "--config-name", config_name,
#             ]

#             print(f"  执行命令: {' '.join(cmd)}")
#             # 运行训练
#             result = subprocess.run(
#                 cmd,
#                 # capture_output=True,
#                 text=True,
#                 cwd=os.getcwd()
#             )
            
#             # 定位 metrics_final.json（与 create_training_config 中一致）
#             checkpoint_base = Path("./temp_training").resolve()
#             checkpoint_dir = checkpoint_base / f"checkpoints_{individual.individual_id}"
#             metrics_file = list(checkpoint_dir.glob("*/metrics_final.json"))

#             fitness = 0.0
#             if result.returncode == 0 and metrics_file:
#                 metrics_file = metrics_file[0]
#                 with open(metrics_file, 'r') as f:
#                     metrics = json.load(f)
#                 fitness = metrics.get('eval/episode_reward', 0.0)

#             else:
#                 if result.stdout:
#                     print("  未找到 metrics_final.json 或训练返回码非0，回退解析 stdout")
#                     fitness = self.extract_fitness_from_output(result.stdout)
#                 else:
#                     print("  无可用 stdout，适应度置 0.0")
#                     fitness = 0.0

#             individual.fitness = fitness
#             individual.training_completed = True if result.returncode == 0 else False
                
#             if result.returncode == 0:
#                 print(f"  ✓ 训练完成，适应度: {fitness:.2f}")
#             else:
#                 print(f"  ✗ 训练失败，返回码: {result.returncode}，适应度: {fitness:.2f}")

#         except Exception as e:
#             print(f"  ✗ 训练出错: {e}")
#             import traceback
#             traceback.print_exc()
#             return 0.0

#     def cleanup(self):
#         """清理临时文件"""
#         # 清理生成的个体配置文件
#         for pattern in ["individual_*.yaml", "individual_*_xml.txt"]:
#             for config_file in self.temp_dir.glob(pattern):
#                 config_file.unlink()
#                 print(f"✓ 清理文件: {config_file}")
        
#         # 清理 checkpoint 目录
#         temp_training_dir = Path("./temp_training")
#         if temp_training_dir.exists():
#             import shutil
#             shutil.rmtree(temp_training_dir)
#             print(f"✓ 清理训练目录: {temp_training_dir}")

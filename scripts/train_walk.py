# **Disclaimer**: This script heavily builds on the fantastic work of the Brax and
# MuJoCo Playground authors. This script simplifies the workflow by integrating
# hydra and wandb for hyperparameter optimization and experiment tracking.
# We encourage to checkout the original code from the authors:
# - Brax: https://github.com/google/brax
# - MuJoCo Playground: https://github.com/google-deepmind/mujoco_playground/

# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import csv
import time
import hydra
from omegaconf import DictConfig
from datetime import datetime
import functools
import os
import pathlib
import warnings

from brax.training.agents.ppo import networks as ppo_networks
# from brax.training.agents.ppo import train as ppo
from scripts import train as ppo
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
from flax.training import orbax_utils
import jax
import mediapy as media
import mujoco
from orbax import checkpoint as ocp
from tensorboardX import SummaryWriter
import wandb
from tqdm import tqdm
import copy
# load crawler_playground to register environments
import tensegrity_playground  # noqa: F401 # pylint:disable=unused-import
from mujoco_playground import registry
from mujoco_playground import wrapper
import json
import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning, module="jax")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="jax")
warnings.filterwarnings("ignore", category=UserWarning, module="absl")

# Set GPU parameters
xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MUJOCO_GL"] = "egl"
jax.config.update("jax_default_matmul_precision", "highest")


def huzzah(cfg):
    print()
    print("888888b.   8888888b.         d8888 Y88b   d88P ")
    print("888  '88b  888   Y88b       d88888  Y88b d88P  ")
    print("888  .88P  888    888      d88P888   Y88o88P   ")
    print("8888888K.  888   d88P     d88P 888    Y888P    ")
    print("888  'Y88b 8888888P'     d88P  888    d888b    ")
    print("888    888 888 T88b     d88P   888   d88888b   ")
    print("888   d88P 888  T88b   d8888888888  d88P Y88b  ")
    print("8888888P'  888   T88b d88P     888 d88P   Y88b ")
    print("ooooooooooooooooooooooooooooooooooooooooooooooo")
    print()
    print(f"Environment: \t\t{cfg.environment_id}")
    print(f"Algorithm: \t\t{cfg.agent_id}")
    print(f"Random Seed: \t\t{cfg.agent.seed}")
    print(f"# envs: \t\t{cfg.agent.num_envs}")
    print(f"# timesteps: \t\t{cfg.agent.num_timesteps}")
    print(f"Logging directory: \t{cfg.logging_dir}")
    print()


@hydra.main(
    config_path="../config", config_name="playground_brax", version_base="1.3"
)
def main(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)
    hydra_cfg = HydraConfig.get()
    # import ipdb
    # ipdb.set_trace()
    cfg["agent_id"] = hydra_cfg.runtime.choices["agent"]
    cfg["environment_id"] = hydra_cfg.runtime.choices["playground"]
    # print("Hydra env choice =", HydraConfig.get().runtime.choices.get("playground"))
    # print("cfg.environment_id =", cfg["environment_id"])
    env_cfg = registry.get_default_config(cfg.environment_id)

    if hasattr(cfg, "playground"):
        env_cfg.update(cfg.playground)

    # ===== 创建训练环境（启用随机化）=====
    train_env_cfg = copy.deepcopy(env_cfg)
    train_env_cfg.enable_reset_randomization = True  # 训练时启用随机化
    env = registry.load(cfg.environment_id, config_overrides=train_env_cfg)

    # ===== 创建评估环境（关闭随机化）=====
    eval_env_cfg = copy.deepcopy(env_cfg)
    eval_env_cfg.enable_reset_randomization = False  # 评估时关闭随机化
    eval_env = registry.load(cfg.environment_id, config_overrides=eval_env_cfg)

    # env = registry.load(cfg.environment_id, config_overrides=env_cfg)

    time_stamp = datetime.now().strftime("%y%m%d%H%M%S")
    run_id = (
        f"{time_stamp}"
        f"-{cfg.environment_id.lower()}"
        f"-{cfg.agent_id.lower()}"
        f"-{cfg.agent.seed}"
    )

    logdir = pathlib.Path(cfg.checkpoint_directory) / run_id
    logdir.mkdir(parents=True, exist_ok=True)
    cfg["logging_dir"] = logdir

    huzzah(cfg)

    if cfg.wandb:
        wandb.init(
            project=f"{cfg.environment_id.lower()}_{cfg.wandb_project}",
            name=run_id,
            entity=cfg.wandb_entity,
            config={
                "agent": cfg.agent,
                "environment": cfg.playground,
            },
        )

    if cfg.tensorboard:
        writer = SummaryWriter(logdir)

    if cfg.progress_bar:
        progress_bar = tqdm(
            total=cfg.agent.num_timesteps,
            ascii=True,
            desc="Time steps",
        )

    if cfg.checkpointing:
        ckpt_path = logdir / "checkpoints"
        ckpt_path.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, logdir / "metadata.yaml")

    # Checkpointing function
    def policy_params_fn(current_step, make_policy, params):
        if cfg.checkpointing:
            path = ckpt_path / f"ckpt_{current_step}"
            orbax_checkpointer = ocp.PyTreeCheckpointer()
            save_args = orbax_utils.save_args_from_target(params)
            orbax_checkpointer.save(
                path, params, force=True, save_args=save_args
            )

    # 保存最后收到的指标
    last_metrics = {}

    best_reward = float("-inf")
    best_step = None

    # 修改进度回调函数
    def progress_fn(num_steps, metrics):
        if cfg.progress_bar:
            if not hasattr(progress_fn, 'last_step'):
                progress_fn.last_step = 0
            progress_bar.update(num_steps - progress_fn.last_step)
            progress_fn.last_step = num_steps

        # 记录所有类型的metrics
        log_dict = {}
        for key, value in metrics.items():
            try:
                if hasattr(value, 'item'):
                    log_dict[key] = float(value.item())
                elif isinstance(value, (int, float)):
                    log_dict[key] = float(value)
                elif hasattr(value, '__float__'):
                    log_dict[key] = float(value)
            except (TypeError, ValueError):
                continue  # 跳过无法转换的metrics
        
        # # ✅ 计算 max_x / avg_episode_length 作为新的评估指标
        # max_x = log_dict.get("eval/episode_max_x", 0.0)
        # avg_length = log_dict.get("eval/avg_episode_length", 1.0)
        # normalized_progress = max_x / avg_length if avg_length >= 800 else 0.0
        # log_dict["eval/normalized_progress"] = float(normalized_progress)

        # 保存最新的指标(每次都更新)
        last_metrics.update(log_dict)
        nonlocal best_reward, best_step
        cur = log_dict.get("eval/episode_max_x_final", None)
        # cur = normalized_progress #log_dict.get("eval/episode_reward", None)
        if cur is not None:
            step = int(num_steps)
            if (cur > best_reward) or (cur == best_reward and (best_step is None or step > best_step)):
                best_reward, best_step = float(cur), step
                out = pathlib.Path(cfg.logging_dir) / "metrics_final.json"
                best_metrics = dict(log_dict)
                best_metrics["best_step"] = best_step
                # best_metrics["best_normalized_progress"] = float(cur)  # 额外保存
                # 立刻覆盖成“当前最优”
                with open(out, "w") as f:
                    json.dump(best_metrics, f, indent=2)

    # # Experiment logging function
    # def progress_fn(num_steps, metrics):
    #     # if cfg.progress_bar:
    #     #     progress_bar.update(num_steps)

    #     # if cfg.wandb:
    #     #     wandb.log(metrics, step=num_steps)

    #     # if cfg.tensorboard:
    #     #     for key, value in metrics.items():
    #     #         writer.add_scalar(key, value, num_steps)
    #     #     writer.flush()

    #     # if cfg.verbose:
    #     #     print(
    #     #         f"Step {num_steps}: reward={metrics['eval/episode_reward']:.3f}"
    #     #     )

    #     if cfg.progress_bar:
    #         if not hasattr(progress_fn, 'last_step'):
    #             progress_fn.last_step = 0
    #         progress_bar.update(num_steps - progress_fn.last_step)
    #         progress_fn.last_step = num_steps

    #     # 记录所有类型的metrics
    #     log_dict = {}
    #     for key, value in metrics.items():
    #         try:
    #             if hasattr(value, 'item'):
    #                 log_dict[key] = float(value.item())
    #             elif isinstance(value, (int, float)):
    #                 log_dict[key] = float(value)
    #             elif hasattr(value, '__float__'):
    #                 log_dict[key] = float(value)
    #         except (TypeError, ValueError):
    #             continue  # 跳过无法转换的metrics

        if cfg.wandb:
            wandb.log(log_dict, step=num_steps)

        if cfg.tensorboard:
            for key, value in log_dict.items():
                writer.add_scalar(key, value, num_steps)
            writer.flush()

        if cfg.verbose:
            print(f"Step {num_steps}: {log_dict}")


    # Main training routine
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks, **cfg.agent.network_factory
    )
    if "network_factory" in cfg.agent:
        del cfg.agent.network_factory

    # import ipdb
    # ipdb.set_trace()
    make_inference_fn, params, _ = ppo.train(
        environment=env,              # 训练环境（带随机化）
        eval_env=eval_env,            # 评估环境（无随机化，固定起点）
        progress_fn=progress_fn,
        network_factory=network_factory,
        policy_params_fn=policy_params_fn,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        **cfg.agent,
    )

    # # 在保存前，添加训练相关的元数据
    # last_metrics.update({
    #     "training_completed": True,
    #     # "timestamp": time.time(),
    #     # "total_steps": cfg.agent.num_timesteps,
    #     # "environment_id": cfg.environment_id,
    #     "individual_id": cfg.playground.get("individual_id", "unknown")
    # })

    # metrics_file = pathlib.Path(cfg.logging_dir) / "metrics_final.json"
    # with open(metrics_file, 'w') as f:
    #     json.dump(last_metrics, f, indent=2)
    # print(f"Final metrics saved at '{metrics_file}'.")
    mf = pathlib.Path(cfg.logging_dir) / "metrics_final.json"
    if mf.exists():
        data = json.loads(mf.read_text())
        data["training_completed"] = True
        mf.write_text(json.dumps(data, indent=2))
    else:
        # 万一评估没跑到，就用最后一次兜底
        last_metrics["training_completed"] = True
        last_metrics["last_metrics"] = "last step"
        mf.write_text(json.dumps(last_metrics, indent=2))

    if cfg.wandb:
        wandb.finish()

    if cfg.tensorboard:
        writer.close()

    if cfg.progress_bar:
        progress_bar.close()

    if cfg.export_video:
            if cfg.checkpointing and best_step is not None:
                best_ckpt_path = ckpt_path / f"ckpt_{best_step}"
                if best_ckpt_path.exists():
                    print(f"Loading best checkpoint from step {best_step}...")
                    orbax_checkpointer = ocp.PyTreeCheckpointer()
                    restore_args = ocp.checkpoint_utils.construct_restore_args(params)
                    params = orbax_checkpointer.restore(best_ckpt_path, item=params, restore_args=restore_args)
                    print(f"Best checkpoint loaded successfully (reward={best_reward:.3f})")
                else:
                    print(f"Warning: Best checkpoint at step {best_step} not found, using final params")
            else:
                print("Using final training params for video export")
            
            inference_fn = make_inference_fn(params, deterministic=True)

            jit_inference_fn = jax.jit(inference_fn)
            jit_reset_fn = jax.jit(eval_env.reset)
            jit_step_fn = jax.jit(eval_env.step)

            # Helper function to get sensor data    
            def get_sensor_data_mujoco(
                model: mujoco.MjModel, data: mujoco.MjData, sensor_name: str
            ):
                """Gets sensor data from regular MuJoCo data (not JAX)."""
                try:
                    sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
                    sensor_adr = model.sensor_adr[sensor_id]
                    sensor_dim = model.sensor_dim[sensor_id]
                    return data.sensordata[sensor_adr : sensor_adr + sensor_dim]
                except:
                    print(f"Sensor '{sensor_name}' not found in model")
                    return None
            
            # Add foot_names for contact sensors
            foot_names = ["fr", "fl", "rr", "rl"]
            contact_threshold = 1e-3  # Contact force threshold (N)
            trajectory_data = []    # save trajectory data if needed

            rng = jax.random.key(cfg.agent.seed)
            state = jit_reset_fn(rng)
            rollout = [state]
            for step in range(cfg.agent.eval_episode_length):
                rng, act_rng = jax.random.split(rng)
                ctrl, _ = jit_inference_fn(state.obs, act_rng)
                state = jit_step_fn(state, ctrl)

                # Get global position (x, y, z)
                pos_data = get_sensor_data_mujoco(eval_env.mj_model, state.data, "position")
                if pos_data is not None:
                    pos_x, pos_y, pos_z = pos_data[0], pos_data[1], pos_data[2]
                else:
                    pos_x, pos_y, pos_z = None, None, None
                            
                # Get foot contact data
                foot_contacts = []
                foot_forces = []
                for foot in foot_names:
                    contact_sensor_name = f"{foot}_foot_touch"
                    contact_force = get_sensor_data_mujoco(eval_env.mj_model, state.data, contact_sensor_name)

                    if contact_force is not None:
                        # Calculate the magnitude of the contact force
                        force_magnitude = float(np.linalg.norm(contact_force))
                        foot_forces.append(force_magnitude)

                        # Determine if the foot is in contact with the ground
                        is_contact = 1 if force_magnitude > contact_threshold else 0
                        foot_contacts.append(is_contact)
                    else:
                        foot_contacts.append(0)
                        foot_forces.append(0.0)

                # Append trajectory data
                trajectory_data.append({
                    "step": step,
                    "pos_x": float(pos_x) if pos_x is not None else None,
                    "pos_y": float(pos_y) if pos_y is not None else None,
                    "pos_z": float(pos_z) if pos_z is not None else None,
                    "FR_contact": foot_contacts[0],
                    "FL_contact": foot_contacts[1],
                    "RR_contact": foot_contacts[2],
                    "RL_contact": foot_contacts[3],
                    "FR_force": foot_forces[0],
                    "FL_force": foot_forces[1],
                    "RR_force": foot_forces[2],
                    "RL_force": foot_forces[3],
                })

                rollout.append(state)
                if state.done:
                    break
            
            # Optionally save trajectory data to a csv file
            trajectory_file = f"{logdir}/{run_id}_trajectory.csv"
            with open(trajectory_file, mode="w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=[
                    "step", "pos_x", "pos_y", "pos_z",
                    "FR_contact", "FL_contact", "RR_contact", "RL_contact",
                    "FR_force", "FL_force", "RR_force", "RL_force"
                ])
                writer.writeheader()
                writer.writerows(trajectory_data)
            print(f"Trajectory data saved at '{trajectory_file}'.")

            fps = int(1.0 / eval_env.dt / cfg.render_every)
            traj = rollout[:: cfg.render_every]

            scene_option = mujoco.MjvOption()
            scene_option.geomgroup[2] = True
            scene_option.geomgroup[3] = False
            scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = False
            scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
            scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False

            camera_names = [cfg.render_camera, "overview"]

            for camera in camera_names:
                frames = eval_env.render(
                    traj,
                    height=600,
                    width=800,
                    scene_option=scene_option,
                    camera=camera,
                )
                media.write_video(f"{logdir}/{run_id}_{camera}.mp4", frames, fps=fps)
                print(f"Rollout video saved at '{logdir}/{run_id}_{camera}.mp4'.")


if __name__ == "__main__":
    main()

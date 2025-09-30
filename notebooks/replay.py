import os
import jax
import mujoco
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.acme import running_statistics
from omegaconf import OmegaConf
from pathlib import Path
from jax.scipy.spatial.transform import Rotation
import jax.numpy as jnp
from orbax import checkpoint as ocp
from typing import Any, Dict, Optional
import mediapy as media
import warnings
import numpy as np
import matplotlib.pyplot as plt
import tensegrity_playground
from mujoco_playground import registry
# import scienceplots

# plt.style.use(["science", "grid"])

warnings.filterwarnings(
    "ignore",
    message="Couldn't find sharding info under RestoreArgs.*",
    category=UserWarning,
)

#test
def get_ppo_inference_fn(
    obs_size,
    act_size,
    normalize_obs: bool,
    network_factory_kwargs,
    variables,  # (normalizer_params, policy_variables)
):
    normalizer_params, policy_variables = variables

    def make_inference_fn(
        observation_size: int,
        action_size: int,
        normalize_observations: bool = True,
        network_factory_kwargs: dict | None = None,
    ):
        normalize = (lambda x, y: x)
        if normalize_observations:
            normalize = running_statistics.normalize

        ppo_net = ppo_networks.make_ppo_networks(
            observation_size,
            action_size,
            preprocess_observations_fn=normalize,
            **(network_factory_kwargs or {}),
        )
        make_policy = ppo_networks.make_inference_fn(ppo_net)
        return make_policy

    make_policy = make_inference_fn(
        obs_size,
        act_size,
        normalize_obs,
        network_factory_kwargs,
    )

    # 包一层，把 key 作为输入参数之一
    def policy_apply(obs, key):
        apply_fn = make_policy((normalizer_params, policy_variables), deterministic=True)
        return apply_fn(obs, key)  # 关键：传入 key_sample

    # JIT 并暴露需要的签名
    jit_inference_fn = jax.jit(policy_apply)
    return jit_inference_fn

def check_termination_reason(env, state):
    """检查终止原因"""
    data = state.data
    
    # 获取当前状态信息
    upvector_z = env.get_upvector(data)[-1]
    height_z = env.get_position(data)[-1]
    
    # 检查终止条件
    fall_termination = upvector_z < 0
    height_termination = height_z < 0.3
    
    reasons = []
    if fall_termination:
        reasons.append(f"Fall termination: upvector_z={upvector_z:.3f} ")
    if height_termination:
        reasons.append(f"Height termination: height_z={height_z:.3f} ")

    return reasons, upvector_z, height_z

# ========= 加载 =========
path = "/media/di/4441-E469/cluster_tmp/tensaurvel1-2699235/250925003453-tensegrityquadrupedwalk-ppo-1"
checkpoint = "ckpt_3112960"
path = Path(path)
cfg_path = path / "metadata.yaml"
ckpt_path = path / "checkpoints" / checkpoint

config = OmegaConf.load(cfg_path)
cfg_overrides = registry.get_default_config(config.environment_id)
env = registry.load(config.environment_id, config_overrides=cfg_overrides)

jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)

orbax_checkpointer = ocp.PyTreeCheckpointer()
normalizer_raw, policy_variables, value_variables = orbax_checkpointer.restore(ckpt_path, item=None)

print("Successfully loaded checkpoint with 3 components")

# 将 normalizer 参数恢复为 RunningStatisticsState（如果 normalize() 需要该类型）
normalizer_state = running_statistics.RunningStatisticsState(**normalizer_raw)

# 构建推理函数
jit_inference_fn = get_ppo_inference_fn(
    env.observation_size,
    env.action_size,
    normalize_obs=True,
    network_factory_kwargs=config.agent.network_factory,
    variables=(normalizer_state, policy_variables),
)

rng = jax.random.key(1)
rng, rng_init = jax.random.split(rng)
state = jit_reset(rng_init)
rollout = [state]
ctrl_hist = []

termination_stats = {
    "fall_count": 0,
    "height_count": 0,
    "total_episodes": 0,
    "episode_lengths": []
}

import ipdb
ipdb.set_trace()

episode_length = 0
for i in range(1000):
    rng, act_rng = jax.random.split(rng)
    ctrl, _ = jit_inference_fn(state.obs, act_rng)
    state = jit_step(state, ctrl)
    episode_length += 1

    if state.done:
        reasons, upvector_z, height_z = check_termination_reason(env, state)

        if reasons:
            for reason in reasons:
                print(f"Termination reason: {reason}")
                if "Fall termination" in reason:
                    termination_stats["fall_count"] += 1
                if "Height termination" in reason:
                    termination_stats["height_count"] += 1
        else:
            print("Termination reason: Unknown")
        termination_stats["total_episodes"] += 1
        termination_stats["episode_lengths"].append(episode_length)
        
        state = jit_reset(rng_init)
        episode_length = 0
        
    rollout.append(state)
    ctrl_hist.append(ctrl)

    render_every = 1
fps = 1.0 / env.dt / render_every

scene_option = mujoco.MjvOption()
scene_option.geomgroup[2] = True
scene_option.geomgroup[3] = False
scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = False
scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = False

camera_names = [
    mujoco.mj_id2name(env.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, i)
    for i in range(env.mj_model.ncam)
]
camera_names.append(0)

for camera in camera_names:
    frames = env.render(
        rollout[::render_every],
        height=480,
        width=640,
        camera=camera,
        scene_option=scene_option,
    )
    media.show_video(frames, fps=fps)
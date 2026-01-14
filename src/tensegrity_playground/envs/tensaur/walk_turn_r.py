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
"""Base classes for Go1."""

from typing import Any, Dict, Optional, Union

from etils import epath
import jax
import jax.numpy as jp
import jax.scipy.spatial.transform as jsp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
import numpy as np
from mujoco_playground._src import mjx_env

ROOT_PATH = epath.Path(__file__).parent
FLAT_TERRAIN_XML = ROOT_PATH / "xmls" / "walk_turn_radius.xml"

ACCELEROMETER_SENSOR = "accelerometer"
GLOBAL_LINVEL_SENSOR = "global_linvel"
GLOBAL_ANGVEL_SENSOR = "global_angvel"
GYRO_SENSOR = "gyro"
LOCAL_LINVEL_SENSOR = "local_linvel"
ORIENTATION_SENSOR = "orientation"
UPVECTOR_SENSOR = "upvector"
POSITION_SENSOR = "position"


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.002,
        episode_length=1000,
        Kp=35.0,
        Kd=0.5,
        action_repeat=1,
        action_scale=0.5,
        history_len=1,
        soft_joint_pos_limit_factor=0.95,
        xml=None,  # 支持从 config 传入 xml_path
        noise_config=config_dict.create(
            level=0.0,  # Set to 0.0 to disable noise.
            scales=config_dict.create(
                joint_pos=0.03,
                joint_vel=1.5,
                gyro=0.2,
                gravity=0.05,
                linvel=0.1,
            ),
        ),
        reward_config=config_dict.create(
            cmd=[0.5, 0.0, 0.0],  # [x, y, z] m/s
            center=[0.0, 2.0],
            radius=1.0,
            scales=config_dict.create(
                # Tracking
                tracking_lin_vel=1.0,
                # tracking_ang_vel=0.5,
                stay_radius=-1.0,
                # Other
                dof_pos_limits=-1.0,
                # pose=0.5,
                # Other.
                termination=-10.0,
                # stand_still=-1.0,
                # Regularization.
                torques=-0.0002,
                action_rate=-0.01,
                energy=-0.001,
                # Feet.
                # feet_clearance=-2.0,
                # feet_height=-0.2,
                # feet_slip=-0.1,
                # feet_air_time=0.1,
            ),
            tracking_sigma=0.5,
            max_foot_height=0.1,
        ),
    )


class Walk(mjx_env.MjxEnv):
    """Base class for Go1 environments."""

    def __init__(
        self,
        task: str,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[
            Dict[str, Union[str, int, list[Any]]]
        ] = None,
    ) -> None:
        super().__init__(config, config_overrides)

        task_to_xml = {
            "flat_terrain": FLAT_TERRAIN_XML,
        }
        
        # 支持从 config 传入 xml_path
        cfg_xml = self._config.get("xml", None)
        if cfg_xml is not None:
            if hasattr(cfg_xml, "as_posix"):
                xml_path = epath.Path(cfg_xml).as_posix()
            else:
                xml_path = str(cfg_xml)
        else:
            default_path = task_to_xml[task]
            xml_path = default_path.as_posix() if hasattr(default_path, "as_posix") else str(default_path)

        # xml_path = task_to_xml[task].as_posix()
        
        # Read XML and replace relative paths with absolute paths
        xml_content = epath.Path(xml_path).read_text()
        xml_dir = epath.Path(xml_path).parent
        
        # 替换地形文件路径为绝对路径
        # Replace all PNG file references with absolute paths
        import re
        def replace_png_path(match):
            filename = match.group(1)
            full_path = xml_dir / filename
            if full_path.exists():
                return f'file="{full_path.as_posix()}"'
            return match.group(0)  # Keep original if file doesn't exist

        xml_content = re.sub(r'file="([^"]+\.png)"', replace_png_path, xml_content)

        self._mj_model = mujoco.MjModel.from_xml_string(xml_content)

        # self._mj_model = mujoco.MjModel.from_xml_string(
        #     epath.Path(xml_path).read_text(),
        # )
        self._mj_model.opt.timestep = self._config.sim_dt

        # Modify PD gains.
        self._mj_model.dof_damping[6:] = config.Kd
        self._mj_model.actuator_gainprm[:, 0] = config.Kp
        self._mj_model.actuator_biasprm[:, 1] = -config.Kp

        # Increase offscreen framebuffer size to render at higher resolutions.
        self._mj_model.vis.global_.offwidth = 1920
        self._mj_model.vis.global_.offheight = 1080

        self._mjx_model = mjx.put_model(self._mj_model)
        self._xml_path = xml_path
        self._imu_site_id = self._mj_model.site("com_vertebrae_1").id

        self._post_init()

    def _post_init(self) -> None:
        self._init_q = jp.array(self._mj_model.keyframe("stable_pose").qpos)
        self._init_ctrl = jp.array(self._mj_model.keyframe("stable_pose").ctrl)

        self._default_pose = jp.array(
            [-0.0, 0.9, -1.55, 0.0, 0.9, -1.55, -0.0, 0.9, -1.55, 0.0, 0.9, -1.55]
        )
        
        # self._torso_body_id = self._mj_model.body(consts.ROOT_BODY).id
        # self._torso_mass = self._mj_model.body_subtreemass[self._torso_body_id]
        self._torso_body_id = self._mj_model.body("vertebrae_0").id        

        # look for correct leg joint indices
        self._leg_joint_names = [
            "fr_hip_roll", "fr_hip_pitch", "fr_knee",
            "fl_hip_roll", "fl_hip_pitch", "fl_knee",
            "rr_hip_roll", "rr_hip_pitch", "rr_knee",
            "rl_hip_roll", "rl_hip_pitch", "rl_knee",
        ]

        legs_qpos_idx = []
        legs_qvel_idx = []
        leg_joint_ids = []  # 添加这个来存储关节ID

        for jname in self._leg_joint_names:
            j_id = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if j_id == -1:
                raise ValueError(f"Joint '{jname}' not found in the model.")
            legs_qpos_idx.append(self._mj_model.jnt_qposadr[j_id])
            legs_qvel_idx.append(self._mj_model.jnt_dofadr[j_id])
            leg_joint_ids.append(j_id)  # 添加关节ID

        self._legs_qpos_idx = np.array(legs_qpos_idx, dtype=int)
        self._legs_qvel_idx = np.array(legs_qvel_idx, dtype=int)

        leg_joint_ranges = np.array([self._mj_model.jnt_range[jid] for jid in leg_joint_ids])
        
        # Note: First joint is freejoint.
        # self._lowers, self._uppers = self.mj_model.jnt_range[1:].T
        self._lowers, self._uppers = leg_joint_ranges.T
        self._soft_lowers = (
            self._lowers * self._config.soft_joint_pos_limit_factor
        )
        self._soft_uppers = (
            self._uppers * self._config.soft_joint_pos_limit_factor
        )
        
        # 获取足部site id，用于计算摆动高度
        self._feet_site_id = np.array(
            [self._mj_model.site(f"{leg}_foot_site").id for leg in ["fr", "fl", "rr", "rl"]]
            )

        # 获取足部触觉传感器
        foot_touch_sensor_adr = []
        for leg in ["fr", "fl", "rr", "rl"]:
            sensor_id = self._mj_model.sensor(f"{leg}_foot_touch").id
            sensor_adr = self._mj_model.sensor_adr[sensor_id]
            sensor_dim = self._mj_model.sensor_dim[sensor_id]
            foot_touch_sensor_adr.append(
                list(range(sensor_adr, sensor_adr + sensor_dim))
            )
        self._foot_touch_sensor_adr = jp.array(foot_touch_sensor_adr)
        
        # 获取足部线性速度传感器地址    
        foot_linvel_sensor_adr = []
        for leg in ["fr", "fl", "rr", "rl"]:
            sensor_id = self._mj_model.sensor(f"{leg}_foot_global_linvel").id
            sensor_adr = self._mj_model.sensor_adr[sensor_id]
            sensor_dim = self._mj_model.sensor_dim[sensor_id]
            foot_linvel_sensor_adr.append(
                list(range(sensor_adr, sensor_adr + sensor_dim))
            )
        self._foot_linvel_sensor_adr = jp.array(foot_linvel_sensor_adr)

        # 目标位置参数
        self._target_center = jp.array(self._config.reward_config.center)
        self._target_radius = self._config.reward_config.radius

            
    def reset(self, rng: jax.Array) -> mjx_env.State:
        qpos = self._init_q
        qvel = jp.zeros(self.mjx_model.nv)

        # x=+U(-0.5, 0.5), y=+U(-0.5, 0.5), yaw=U(-3.14, 3.14).
        # rng, key = jax.random.split(rng)
        # dx = jax.random.uniform(key, minval=-5.0, maxval=5.0)
        # qpos = qpos.at[0].set(qpos[0] + dx)

        # rng, key = jax.random.split(rng)
        # dy = jax.random.uniform(key, minval=-0.2, maxval=0.2)
        # qpos = qpos.at[1].set(qpos[1] + dy)

        # rng, key = jax.random.split(rng)
        # yaw = jax.random.uniform(key, (1,), minval=-3.14 / 6, maxval=3.14 / 6)
        # quat = math.axis_angle_to_quat(jp.array([0, 0, 1]), yaw)
        # new_quat = math.quat_mul(qpos[3:7], quat)
        # qpos = qpos.at[3:7].set(new_quat)

        # d(xyzrpy)=U(-0.5, 0.5)
        # rng, key = jax.random.split(rng)
        # qvel = qvel.at[0:6].set(
        #     jax.random.uniform(key, (6,), minval=-0.2, maxval=0.2)
        # )

        data = mjx_env.init(
            self.mjx_model,  qpos=qpos, qvel=qvel, ctrl=self._init_ctrl
        )

        # Target velocity commands.
        cmd = jp.array(self._config.reward_config.cmd)

        # Adaptation experiments.``
        info = {
            "rng": rng,
            "command": cmd,
            "local_vel": jp.zeros(3),  # 实际速度 (初始化为零向量)
            "last_act": jp.zeros(self.mjx_model.nu),
            "last_last_act": jp.zeros(self.mjx_model.nu),
            "feet_air_time": jp.zeros(4),
            "last_contact": jp.zeros(4, dtype=bool),
            "swing_peak": jp.zeros(4),
            "position": jp.zeros(3),
            "r_deviation": jp.zeros(()),
        }

        metrics = {}
        for k in self._config.reward_config.scales.keys():
            metrics[f"reward/{k}"] = jp.zeros(())
        metrics["swing_peak"] = jp.zeros(())
        metrics["local_vel_x"] = jp.zeros(())
        metrics["local_vel_y"] = jp.zeros(())
        metrics["global_yaw"] = jp.zeros(())
        metrics["r_deviation"] = jp.zeros(())   

        obs = self._get_obs(data, info)
        reward, done = jp.zeros(2)
        return mjx_env.State(data, obs, reward, done, metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        # <---------------- Simulator step ---------------->
        motor_targets = self._default_pose + action * self._config.action_scale
        data = mjx_env.step(
            self.mjx_model, state.data, motor_targets, self.n_substeps
        )

        contact = self.get_foot_contacts(data)

        contact_filt = contact | state.info["last_contact"]
        first_contact = (state.info["feet_air_time"] > 0.0) * contact_filt
        state.info["feet_air_time"] += self.dt

        p_f = data.site_xpos[self._feet_site_id]
        p_fz = p_f[..., -1]
        state.info["swing_peak"] = jp.maximum(state.info["swing_peak"], p_fz)

        state.info["position"] = self.get_position(data)
        state.info["r_deviation"] = self.r_deviation(data)
        state.info["local_vel"] = self.get_local_linvel(data)  #data.cvel[self._torso_body_id][3:]

        # <---------------- Evaluate change ---------------->
        obs = self._get_obs(data, state.info)
        done = self._get_termination(data)

        rewards = self._get_reward(
            data,
            action,
            state.info,
            state.metrics,
            done,
            first_contact,
            contact,
        )
        rewards = {
            k: v * self._config.reward_config.scales[k] 
            for k, v in rewards.items()
        }
        reward = sum(rewards.values()) * self.dt

        state.info["last_last_act"] = state.info["last_act"] 
        state.info["last_act"] = action
        state.info["feet_air_time"] *= ~contact
        state.info["last_contact"] = contact
        state.info["swing_peak"] *= ~contact

        for k, v in rewards.items():
            state.metrics[f"reward/{k}"] = v
        state.metrics["swing_peak"] = jp.mean(state.info["swing_peak"])
        state.metrics["local_vel_x"] = state.info["local_vel"][0]
        state.metrics["local_vel_y"] = state.info["local_vel"][1]
        state.metrics["global_yaw"] = self.get_global_angvel(data)[2]
        state.metrics["r_deviation"] = state.info["r_deviation"]


        done = done.astype(reward.dtype)
        state = state.replace(data=data, obs=obs, reward=reward, done=done)

        return state

    def _get_termination(self, data: mjx.Data) -> jax.Array:
        fall_termination = self.get_upvector(data)[-1] < 0
        return fall_termination 

    def _get_obs(
        self, data: mjx.Data, info: dict[str, Any]
    ) -> Dict[str, jax.Array]:
        gyro = self.get_gyro(data)
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_gyro = (
            gyro
            + (2 * jax.random.uniform(noise_rng, shape=gyro.shape) - 1)
            * self._config.noise_config.level
            * self._config.noise_config.scales.gyro
        )

        gravity = self.get_gravity(data)
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_gravity = (
            gravity
            + (2 * jax.random.uniform(noise_rng, shape=gravity.shape) - 1)
            * self._config.noise_config.level
            * self._config.noise_config.scales.gravity
        )

        joint_angles = data.sensordata[:12]
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_joint_angles = (
            joint_angles
            + (2 * jax.random.uniform(noise_rng, shape=joint_angles.shape) - 1)
            * self._config.noise_config.level
            * self._config.noise_config.scales.joint_pos
        )

        joint_vel = data.sensordata[12:24]
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_joint_vel = (
            joint_vel
            + (2 * jax.random.uniform(noise_rng, shape=joint_vel.shape) - 1)
            * self._config.noise_config.level
            * self._config.noise_config.scales.joint_vel
        )

        linvel = self.get_global_linvel(data)
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_linvel = (
            linvel
            + (2 * jax.random.uniform(noise_rng, shape=linvel.shape) - 1)
            * self._config.noise_config.level
            * self._config.noise_config.scales.linvel
        )

        forward_vector = mjx_env.get_sensor_data(self.mj_model, data, "forwardvector")
        state = jp.hstack(
            [
                noisy_linvel,  # 3
                noisy_gyro,  # 3
                noisy_gravity,  # 3
                noisy_joint_angles - self._default_pose,  # 12
                noisy_joint_vel,  # 12
                info["last_act"],  # 12
                info["command"],  # 3
                info["position"],  # 3
                forward_vector,  # 3
                info["r_deviation"]  # 2 xinjia
            ]
        )

        accelerometer = self.get_accelerometer(data)
        angvel = self.get_global_angvel(data)
        feet_vel = data.sensordata[self._foot_linvel_sensor_adr].ravel()
        
        privileged_state = jp.hstack([
            state,
            gyro,   # 3
            accelerometer,  # 3
            gravity,  # 3
            linvel, # 3
            angvel,  # 3
            joint_angles - self._default_pose,  # 12
            joint_vel,  # 12
            data.actuator_force,  # 12
            info["last_contact"],  # 4
            feet_vel,
            info["feet_air_time"],  # 4
        ])

        return {
            "state": state,
            "privileged_state": privileged_state,
        }

    def _get_reward(
        self,
        data: mjx.Data,
        action: jax.Array,
        info: dict[str, Any],
        metrics: dict[str, Any],
        done: jax.Array,
        first_contact: jax.Array,
        contact: jax.Array,
    ) -> dict[str, jax.Array]:
        del metrics  # Unused.
        return {
            "tracking_lin_vel": self._reward_tracking_lin_vel(
                info["command"], self.get_local_linvel(data)
            ),
            # "tracking_ang_vel": self._reward_tracking_ang_vel(
            #     info["command"], self.get_gyro(data)
            # ),

            # "orientation": self._cost_orientation(data),
            # "forward_progress": self._reward_forward_progress(
            #     self.get_local_linvel(data)
            # ),
            # "lin_vel_z": self._cost_lin_vel_z(self.get_global_linvel(data)),
            # "ang_vel_xy": self._cost_ang_vel_xy(self.get_global_angvel(data)),
            # "orientation": self._cost_orientation(self.get_upvector(data)),
            "stay_radius": self._cost_r_deviation(info["r_deviation"]),
            "termination": self._cost_termination(done),
            # "pose": self._reward_pose(data.qpos[self._legs_qpos_idx]),
            "torques": self._cost_torques(data.actuator_force),
            "action_rate": self._cost_action_rate(
                action, info["last_act"], info["last_last_act"]
            ),
            "energy": self._cost_energy(data.qvel[self._legs_qvel_idx], data.actuator_force),
            # "feet_slip": self._cost_feet_slip(data, contact, info),
            # "feet_clearance": self._cost_feet_clearance(data),
            # "feet_height": self._cost_feet_height(
            #     info["swing_peak"], first_contact, info
            # ),
            # "feet_air_time": self._reward_feet_air_time(
            #     info["feet_air_time"], first_contact, info["command"]
            #     #self.get_local_linvel(data)
            #     #info["command"]
            # ),
            "dof_pos_limits": self._cost_joint_pos_limits(data.qpos[self._legs_qpos_idx])
        }

    # Tracking rewards.

    def _reward_tracking_lin_vel(
        self,
        commands: jax.Array,
        local_vel: jax.Array,
    ) -> jax.Array:
        # Tracking of linear velocity commands (xy axes).
        lin_vel_error = jp.sum(jp.square(commands[:2] - local_vel[:2]))
        return jp.exp(-lin_vel_error / self._config.reward_config.tracking_sigma)

    def _reward_tracking_ang_vel(
        self,
        commands: jax.Array,
        ang_vel: jax.Array,
    ) -> jax.Array:
        # Tracking of angular velocity commands (yaw).
        ang_vel_error = jp.square(commands[2] - ang_vel[2])
        return jp.exp(-ang_vel_error / self._config.reward_config.tracking_sigma)

    def _cost_r_deviation(self, r_deviation: jax.Array) -> jax.Array:
        r_deviation = jp.square(r_deviation - self._target_radius)
        return r_deviation

    # Energy related rewards.

    def _cost_torques(self, torques: jax.Array) -> jax.Array:
        # Penalize torques.
        return jp.sqrt(jp.sum(jp.square(torques))) + jp.sum(jp.abs(torques))

    def _cost_energy(
        self, qvel: jax.Array, qfrc_actuator: jax.Array
    ) -> jax.Array:
        # Penalize energy consumption.
        return jp.sum(jp.abs(qvel) * jp.abs(qfrc_actuator))

    def _cost_action_rate(
        self, act: jax.Array, last_act: jax.Array, last_last_act: jax.Array
    ) -> jax.Array:
        del last_last_act  # Unused.
        return jp.sum(jp.square(act - last_act))
    
      # Other rewards.

    def _cost_termination(self, done: jax.Array) -> jax.Array:
    # Penalize early termination.
        return done

    def _cost_joint_pos_limits(self, qpos: jax.Array) -> jax.Array:
        # Penalize joints if they cross soft limits.
        out_of_limits = -jp.clip(qpos - self._soft_lowers, None, 0.0)
        out_of_limits += jp.clip(qpos - self._soft_uppers, 0.0, None)
        return jp.sum(out_of_limits)

    # Feet related rewards.
    def _cost_feet_slip(
        self, data: mjx.Data, contact: jax.Array, info: dict[str, Any]
    ) -> jax.Array:
        cmd_norm = jp.linalg.norm(info["command"])
        feet_vel = data.sensordata[self._foot_linvel_sensor_adr]
        vel_xy = feet_vel[..., :2]
        vel_xy_norm_sq = jp.sum(jp.square(vel_xy), axis=-1)
        return jp.sum(vel_xy_norm_sq * contact) * (cmd_norm > 0.01)

    def _cost_feet_clearance(self, data: mjx.Data) -> jax.Array:
        feet_vel = data.sensordata[self._foot_linvel_sensor_adr]
        vel_xy = feet_vel[..., :2]
        vel_norm = jp.sqrt(jp.linalg.norm(vel_xy, axis=-1))
        foot_pos = data.site_xpos[self._feet_site_id]
        foot_z = foot_pos[..., -1]
        delta = jp.abs(foot_z - self._config.reward_config.max_foot_height)
        return jp.sum(delta * vel_norm)

    def _cost_feet_height(
        self,
        swing_peak: jax.Array,
        first_contact: jax.Array,
        info: dict[str, Any],
    ) -> jax.Array:
        cmd_norm = jp.linalg.norm(info["command"])
        error = swing_peak / self._config.reward_config.max_foot_height - 1.0
        return jp.sum(jp.square(error) * first_contact) * (cmd_norm > 0.01)

    def _reward_feet_air_time(
        self, air_time: jax.Array, first_contact: jax.Array, commands: jax.Array
    ) -> jax.Array:
        # Reward air time.
        cmd_norm = jp.linalg.norm(commands)
        rew_air_time = jp.sum((air_time - 0.1) * first_contact)
        rew_air_time *= cmd_norm > 0.01  # No reward for zero commands.
        return rew_air_time

  # Sensor readings.

    def get_upvector(self, data: mjx.Data) -> jax.Array:
        return mjx_env.get_sensor_data(self.mj_model, data, UPVECTOR_SENSOR)

    def get_gravity(self, data: mjx.Data) -> jax.Array:
        return data.site_xmat[self._imu_site_id].T @ jp.array([0, 0, -1])

    def get_global_linvel(self, data: mjx.Data) -> jax.Array:
        return mjx_env.get_sensor_data(
            self.mj_model, data, GLOBAL_LINVEL_SENSOR
        )

    def get_global_angvel(self, data: mjx.Data) -> jax.Array:
        return mjx_env.get_sensor_data(
            self.mj_model, data, GLOBAL_ANGVEL_SENSOR
        )

    def get_local_linvel(self, data: mjx.Data) -> jax.Array:
        return mjx_env.get_sensor_data(
            self.mj_model, data, LOCAL_LINVEL_SENSOR
        )

    def get_accelerometer(self, data: mjx.Data) -> jax.Array:
        return mjx_env.get_sensor_data(
            self.mj_model, data, ACCELEROMETER_SENSOR
        )

    def get_gyro(self, data: mjx.Data) -> jax.Array:
        return mjx_env.get_sensor_data(self.mj_model, data, GYRO_SENSOR)

    def get_orientation(self, data: mjx.Data) -> jax.Array:
        return mjx_env.get_sensor_data(self.mj_model, data, ORIENTATION_SENSOR)

    def get_position(self, data: mjx.Data) -> jax.Array:
        return mjx_env.get_sensor_data(self.mj_model, data, POSITION_SENSOR)
    
    def get_foot_contacts(self, data: mjx.Data) -> jax.Array:
        """获取四只脚与地面的接触状态"""
        # 通过传感器数据获取足部接触状态
        vals= jp.squeeze(data.sensordata[self._foot_touch_sensor_adr], axis=-1)
        return vals > 1e-3
    
    def r_deviation(self, data: mjx.Data) -> jax.Array:
        # 1) 原始的 site 世界坐标（当前你在用的“position”传感器）
        site_world = self.get_position(data)  # shape: (3,)

        # 2) 机体局部 +x 轴在世界系下的方向（已有 forwardvector 传感器）
        x_axis_world = mjx_env.get_sensor_data(self.mj_model, data, "forwardvector")  # shape: (3,)

        # 3) 读取该 site 的局部 x 偏移（不硬编码 0.1881；模型变了也能适配）
        imu_x = self._mj_model.site_pos[self._imu_site_id][0]

        # 4) 把 site 的世界坐标修正回“真正圆心”的世界坐标
        center_world = site_world - imu_x * x_axis_world

        # 5) 用修正后的圆心做距离与速度惩罚
        x_offset = center_world[0] - self._target_center[0]
        y_offset = center_world[1] - self._target_center[1]

        dist_to_center = jp.linalg.norm(jp.array([x_offset, y_offset]))
        return dist_to_center   

    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def action_size(self) -> int:
        return self._mjx_model.nu

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model

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

from mujoco_playground._src import mjx_env

ROOT_PATH = epath.Path(__file__).parent
FLAT_TERRAIN_XML = ROOT_PATH / "xmls" / "scene_tensegrity_quadruped.xml"

ACCELEROMETER_SENSOR = "accelerometer"
GLOBAL_LINVEL_SENSOR = "global_linvel"
GLOBAL_ANGVEL_SENSOR = "global_angvel"
GYRO_SENSOR = "gyro"
LOCAL_LINVEL_SENSOR = "local_linvel"
ORIENTATION_SENSOR = "orientation"
UPVECTOR_SENSOR = "upvector"


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.001,
        episode_length=1000,
        Kp=35.0,
        Kd=0.5,
        action_repeat=1,
        action_scale=0.5,
        history_len=1,
        soft_joint_pos_limit_factor=0.95,
        noise_config=config_dict.create(
            level=1.0,  # Set to 0.0 to disable noise.
            scales=config_dict.create(
                joint_pos=0.03,
                joint_vel=1.5,
                gyro=0.2,
                gravity=0.05,
                linvel=0.1,
            ),
        ),
        reward_config=config_dict.create(
            target_x_vel=2.0,
            scales=config_dict.create(
                global_vel_x=1.0,
                local_yaw=0.2,
                # Regularization.
                torques=-0.0002,
                action_rate=-0.01,
                energy=-0.001,
            ),
        ),
    )


class Walk(mjx_env.MjxEnv):
    """Base class for Go1 environments."""

    def __init__(
        self,
        task: str,
        config: config_dict.ConfigDict,
        config_overrides: Optional[
            Dict[str, Union[str, int, list[Any]]]
        ] = None,
    ) -> None:
        super().__init__(config, config_overrides)

        task_to_xml = {
            "flat_terrain": FLAT_TERRAIN_XML,
        }
        xml_path = task_to_xml[task].as_posix()

        self._mj_model = mujoco.MjModel.from_xml_string(
            epath.Path(xml_path).read_text(),
        )
        self._mj_model.opt.timestep = self._config.sim_dt

        # Modify PD gains.
        self._mj_model.dof_damping[6:] = config.Kd
        self._mj_model.actuator_gainprm[:, 0] = config.Kp
        self._mj_model.actuator_biasprm[:, 1] = -config.Kp

        # Increase offscreen framebuffer size to render at higher resolutions.
        self._mj_model.vis.global_.offwidth = 3840
        self._mj_model.vis.global_.offheight = 2160

        self._mjx_model = mjx.put_model(self._mj_model)
        self._xml_path = xml_path
        self._imu_site_id = self._mj_model.site("com_vertebrae_1").id

        self._post_init()

    def _post_init(self) -> None:
        self._init_q = jp.array(self._mj_model.keyframe("stable_pose").qpos)
        self._init_ctrl = jp.array(self._mj_model.keyframe("stable_pose").ctrl)

        self._default_pose = jp.array(
            [-0.3, 0.8, -1.2, 0.3, 0.8, -1.2, -0.3, 0.8, -1.2, 0.3, 0.8, -1.2]
        )

        # Note: First joint is freejoint.
        self._lowers, self._uppers = self.mj_model.jnt_range.T
        self._soft_lowers = (
            self._lowers * self._config.soft_joint_pos_limit_factor
        )
        self._soft_uppers = (
            self._uppers * self._config.soft_joint_pos_limit_factor
        )

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
            self.mjx_model,  # qpos=qpos, qvel=qvel, ctrl=self._init_ctrl
        )

        # Target velocity commands.
        target_vel = self._config.reward_config.target_x_vel

        # Adaptation experiments.
        info = {
            "rng": rng,
            "target_vel": target_vel,
            "last_act": jp.zeros(self.mjx_model.nu),
            "last_last_act": jp.zeros(self.mjx_model.nu),
        }

        metrics = {}
        for k in self._config.reward_config.scales.keys():
            metrics[f"reward/{k}"] = jp.zeros(())

        obs = self._get_obs(data, info)
        reward, done = jp.zeros(2)
        return mjx_env.State(data, obs, reward, done, metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        # <---------------- Simulator step ---------------->
        motor_targets = self._default_pose + action * self._config.action_scale
        data = mjx_env.step(
            self.mjx_model, state.data, motor_targets, self.n_substeps
        )

        # <---------------- Evaluate change ---------------->
        obs = self._get_obs(data, state.info)
        done = self._get_termination(data)

        rewards = self._get_reward(
            data,
            action,
            state.info,
            state.metrics,
            done,
        )
        rewards = {
            k: v * self._config.reward_config.scales[k]
            for k, v in rewards.items()
        }
        reward = sum(rewards.values()) * self.dt

        for k, v in rewards.items():
            state.metrics[f"reward/{k}"] = v

        done = done.astype(reward.dtype)
        state = state.replace(data=data, obs=obs, reward=reward, done=done)

        return state

    def _get_termination(self, data: mjx.Data) -> jax.Array:
        fall_termination = self.get_upvector(data)[-1] < 0.0
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

        obs = jp.hstack(
            [
                noisy_linvel,  # 3
                noisy_gyro,  # 3
                noisy_gravity,  # 3
                noisy_joint_angles - self._default_pose,  # 12
                noisy_joint_vel,  # 12
            ]
        )

        return obs

    def _get_reward(
        self,
        data: mjx.Data,
        action: jax.Array,
        info: dict[str, Any],
        metrics: dict[str, Any],
        done: jax.Array,
    ) -> dict[str, jax.Array]:
        del metrics  # Unused.
        return {
            "global_vel_x": self._reward_global_vel_x(
                info["target_vel"], self.get_global_linvel(data)
            ),
            "local_yaw": self._reward_local_yaw(self._get_yaw(data)),
            # "torques": self._cost_torques(data.actuator_force),
            # "action_rate": self._cost_action_rate(
            #     action, info["last_act"], info["last_last_act"]
            # ),
            # "energy": self._cost_energy(data.qvel[6:], data.actuator_force),
        }

    # Custom rewards.

    def _reward_global_vel_x(
        self, target_vel: jax.Array, global_vel: jax.Array
    ) -> jax.Array:
        lin_vel_error = jp.square(global_vel[0] - target_vel)
        # scales gaussian so that R(0) = 0 and R(v>0) > 0
        scaling = (target_vel**2) / (-2 * jp.log(5e-2))
        return jp.exp(-lin_vel_error / scaling)

    def _cost_local_yaw(self, ang_vel) -> jax.Array:
        # Penalize yaw velocity.
        # Tracking of angular velocity commands (yaw).
        return jp.square(ang_vel[2])

    def _reward_local_yaw(
        self,
        yaw: jax.Array,
    ) -> jax.Array:
        # Tracking of angular velocity commands (yaw).
        ang_vel_error = jp.square(yaw)
        return jp.exp(-ang_vel_error / 2.0)

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

    def _get_yaw(self, data: mjx.Data) -> jax.Array:
        quat = self.get_orientation(data)
        # JAX SciPy expects [x, y, z, w] quaternion ordering
        quat_xyzw = quat[jp.array([1, 2, 3, 0])]
        euler_xyz = jsp.Rotation.from_quat(quat_xyzw).as_euler("xyz")
        yaw = euler_xyz[2]
        return yaw

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

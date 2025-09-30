from etils import epath
import numpy as np
from dm_control import mjcf

ROOT_PATH = epath.Path(__file__).parent

# 以 go1 的髋-髋间距作为“身长”
GO1_HIP_TO_HIP_LENGTH = 2 * 0.1881  # = 0.3762 m

DEFAULT_CONFIG = {
    "output_path": ROOT_PATH / "xmls" / "scene_tensegrity_quadruped.xml",
    "sim": {
        "timestep": 0.001,
        "integrator": "RK4",
        "iterations": 100,
        "ls_iterations": 50,
    },
    "defaults": {
        "tq1": {
            "geom": {"condim": 3, "contype": 0, "conaffinity": 0},
            "joint": {
                "type": "hinge",
                "limited": True,
                "damping": 0.1,
                "armature": 0.005,
                "frictionloss": 0.001,
            },
            "position": {
                "kp": 50,
                "ctrlrange": [-2.356, 2.356],
                "forcerange": [-50, 50],
            },
        },
        "vertebra_geom": {
            "geom": {
                "type": "capsule",
                "size": [0.01],
                "density": 2000,
                "rgba": [0.8, 0.6, 0.4, 1],
                "group": 0,
            },
        },
        "leg_geom": {
            "geom": {
                "type": "capsule",
                "size": [0.02],  # 胶囊半径更接近 go1 可视化尺度
                "density": 2000.0,  
                "rgba": [0.8, 0.6, 0.4, 1],
                "group": 1,
                "conaffinity": 0,
            },
        },
        # 仍保留拉索默认配置占位（本模型不使用）
        "lateral_tendon": {
            "tendon": {
                "stiffness": 3000,
                "damping": 1.0,
                "frictionloss": 0.05,
                "width": 0.002,
                "rgba": [1.0, 0.0, 0.0, 0.5],
            },
        },
        "diagonal_tendon": {
            "tendon": {
                "stiffness": 3000,
                "damping": 1.0,
                "frictionloss": 0.05,
                "width": 0.002,
                "rgba": [0.0, 0.0, 1.0, 0.5],
            },
        },
    },
    "leg_dimensions": {
        "hip_roll_length": 0.05,   # 外展短连杆长度（水平）
        "hip_pitch_length": 0.213,  # 大腿段长度
        "shin_length": 0.213,       # 小腿段长度
        "foot_radius": 0.023,       # 足端球半径
    },
    "spine": {
        # 主身体以“单根棍子”表示，长度默认等于 GO1 髋-髋间距
        "initial_z": 0.38,  # 腿变长后抬高初始质心高度以避免穿地 0.474 站直
        "enforce_fixed_length": True,
        "fixed_length": GO1_HIP_TO_HIP_LENGTH,
    },
    "actuation": {
        # 参考 go1 的关节范围与力矩限制（位置型执行器的 forcerange）
        "hip_roll": {"ctrlrange": [-0.863, 0.863], "forcerange": [-35.55, 35.55]},
        "hip_pitch": {"ctrlrange": [-0.686, 4.501], "forcerange": [-23.7, 23.7]},
        "knee": {"ctrlrange": [-2.818, -0.888], "forcerange": [-35.55, 35.55]},
    },
    "legs": {"z_offset": -0.025},
    "keyframe": {"leg_qpos": [-0.06, 0.9, -1.55, 0.06, 0.9, -1.55]},
}


def generate_root(config):
    model = mjcf.RootElement()
    model.compiler.angle = "radian"
    model.option.timestep = config["sim"]["timestep"]
    model.option.integrator = config["sim"]["integrator"]
    model.option.iterations = config["sim"]["iterations"]
    model.option.ls_iterations = config["sim"]["ls_iterations"]

    main_class = model.default.add("default", dclass="tq1")
    subclasses = config["defaults"]
    for name, entry in subclasses.items():
        if name == "tq1":
            target = main_class
        else:
            target = main_class.add("default", dclass=name)

        for elem_type in ("geom", "joint", "tendon", "position"):
            attrs = entry.get(elem_type)
            if attrs and hasattr(target, elem_type):
                element = getattr(target, elem_type)
                for k, v in attrs.items():
                    setattr(element, k, v)

    return model


def add_terrain(model, config):
    model.asset.add(
        "texture",
        name="skybox",
        type="skybox",
        builtin="gradient",
        rgb1=[0.4, 0.6, 0.8],
        rgb2=[0, 0, 0],
        width=800,
        height=800,
        mark="random",
        markrgb=[1, 1, 1],
    )
    model.asset.add(
        "texture",
        name="grid",
        type="2d",
        builtin="checker",
        rgb1=[0.1, 0.2, 0.3],
        rgb2=[0.2, 0.3, 0.4],
        width=300,
        height=300,
        mark="edge",
        markrgb=[0.2, 0.3, 0.4],
    )
    model.asset.add(
        "material",
        name="grid",
        texture="grid",
        texrepeat=[5, 5],
        texuniform=True,
        reflectance=0.2,
    )
    model.worldbody.add(
        "geom",
        name="floor",
        type="plane",
        size=[20, 20, 0.1],
        pos=[0, 0, 0],
        material="grid",
    )
    return model


def add_single_stick_body(model, config):
    """构建单根棍子的主身体，并在中点放置 IMU site；腿安装在两端。"""
    initial_z = config["spine"]["initial_z"]
    enforce = config["spine"].get("enforce_fixed_length", True)
    stick_len = config["spine"].get("fixed_length", GO1_HIP_TO_HIP_LENGTH) if enforce else GO1_HIP_TO_HIP_LENGTH

    name = "vertebrae_0"  # 保持与原命名风格兼容
    body = model.worldbody.add("body", name=name, pos=[0.0, 0.0, initial_z], childclass="tq1")
    body.add("joint", name=f"joint_{name}", type="free")

    # 一根沿 x 轴的 capsule，关于 x=0 对称（-L/2 到 +L/2）
    half = float(stick_len) / 2.0
    body.add(
        "geom",
        name=f"{name}_stick",
        type="capsule",
        fromto=[-half, 0.0, 0.0, half, 0.0, 0.0],
        dclass="vertebra_geom",
    )

    # 中点 IMU site（名称与原 add_sensors 兼容）
    body.add("site", name="com_vertebrae_1", pos=[half, 0.0, 0.0], size=[0.016])

    # 为可视调试，附加几台摄像机（保持你原来的放法在第一个 body 上）
    body.add("light", name="tracking", mode="trackcom", pos=[0, 0, 3], diffuse=[0.6] * 3)
    body.add(
        "camera",
        name="track",
        pos=[0.846, -1.3, 0.316],
        xyaxes=[0.866, 0.500, 0.000, -0.171, 0.296, 0.940],
        mode="trackcom",
    )
    body.add(
        "camera",
        name="top",
        pos=[-1, 0, 1],
        xyaxes=[0, -1, 0, 0.7, 0, 0.7],
        mode="trackcom",
    )
    body.add(
        "camera",
        name="side",
        pos=[0, -1, 0.3],
        xyaxes=[1, 0, 0, 0, 1, 2],
        mode="trackcom",
    )
    body.add(
        "camera",
        name="back",
        pos=[-1, 0, 0.3],
        xyaxes=[0, -1, 0, 1, 0, 2],
        mode="trackcom",
    )

    # 在棍子两端安装肩部几何，并挂两条腿（前端 fr/fl，后端 rr/rl）
    z = config["legs"]["z_offset"]
    # 前端（+x）
    x_offset_front = half
    body.add(
        "geom",
        name="front_shoulder_0",
        fromto=[x_offset_front, -0.12675, z, x_offset_front, 0.12675, z],
        dclass="leg_geom",
    )
    body.add(
        "geom",
        name="front_shoulder_attach_0",
        fromto=[x_offset_front, 0, 0, x_offset_front, 0, z],
        dclass="leg_geom",
        size=[0.005],
    )
    add_leg(body, "fr", [x_offset_front, -0.12675, z])
    add_leg(body, "fl", [x_offset_front, 0.12675, z])

    # 后端（-x）
    x_offset_rear = -half
    body.add(
        "geom",
        name="hind_shoulder_0",
        fromto=[x_offset_rear, -0.12675, z, x_offset_rear, 0.12675, z],
        dclass="leg_geom",
    )
    body.add(
        "geom",
        name="hind_shoulder_attach_0",
        fromto=[x_offset_rear, 0, 0, x_offset_rear, 0, z],
        dclass="leg_geom",
        size=[0.005],
    )
    add_leg(body, "rr", [x_offset_rear, -0.12675, z])
    add_leg(body, "rl", [x_offset_rear, 0.12675, z])

    return model


def add_leg(parent_body, prefix, base_pos):
    # 采用 DEFAULT_CONFIG 的腿部参数
    roll_len = DEFAULT_CONFIG["leg_dimensions"]["hip_roll_length"]
    pitch_len = DEFAULT_CONFIG["leg_dimensions"]["hip_pitch_length"]
    shin_len = DEFAULT_CONFIG["leg_dimensions"]["shin_length"]
    foot_r = DEFAULT_CONFIG["leg_dimensions"]["foot_radius"]

    act_cfg = DEFAULT_CONFIG["actuation"]
    hip_roll_range = act_cfg["hip_roll"]["ctrlrange"]
    hip_pitch_range = act_cfg["hip_pitch"]["ctrlrange"]
    knee_range = act_cfg["knee"]["ctrlrange"]

    # 髋外展段（沿 x 轴）
    root = parent_body.add("body", name=f"{prefix}_leg", pos=base_pos)
    root.add(
        "joint",
        name=f"{prefix}_hip_roll",
        axis=[1, 0, 0],
        range=hip_roll_range,
    )

    root.add(
        "geom",
        name=f"{prefix}_hip_roll_geom",
        fromto=[0, 0, 0, -roll_len, 0, 0],
        dclass="leg_geom",
    )

    # 髋俯仰段（大腿，沿 -z）
    hip = root.add("body", name=f"{prefix}_hip_pitch", pos=[-roll_len, 0, 0])
    hip.add(
        "joint",
        name=f"{prefix}_hip_pitch",
        axis=[0, 1, 0],
        range=hip_pitch_range,
    )

    hip.add(
        "geom",
        name=f"{prefix}_hip_pitch_geom",
        fromto=[0, 0, 0, 0, 0, -pitch_len],
        dclass="leg_geom",
    )

    # 膝关节段（小腿，沿 -z）
    knee = hip.add("body", name=f"{prefix}_knee", pos=[0, 0, -pitch_len])
    knee.add(
        "joint",
        name=f"{prefix}_knee",
        axis=[0, 1, 0],
        range=knee_range,
    )

    knee.add(
        "geom",
        name=f"{prefix}_shin_geom",
        fromto=[0, 0, 0, 0, 0, -shin_len],
        dclass="leg_geom",
        conaffinity=1,
    )

    # 足端
    foot = knee.add("body", name=f"{prefix}_foot", pos=[0, 0, -shin_len])
    foot.add(
        "geom",
        name=f"{prefix}_foot_geom",
        type="sphere",
        size=[foot_r],
        conaffinity=1,
        dclass="leg_geom",
        friction=[0.8, 0.02, 0.01],
    )
    foot.add("site", name=f"{prefix}_foot_site", pos=[0, 0, 0], size=[foot_r])


def add_actuation(model, config):
    act_cfg = config["actuation"]

    for leg in ["fr", "fl", "rr", "rl"]:
        for joint_name in ["hip_roll", "hip_pitch", "knee"]:
            cfg = act_cfg[joint_name]
            model.actuator.add(
                "position",
                name=f"{leg}_{joint_name}",
                joint=f"{leg}_{joint_name}",
                ctrlrange=cfg["ctrlrange"],
                forcerange=cfg.get("forcerange", None),
            )


def add_sensors(model, config):
    # Add sensors
    sensor = model.sensor

    # Add joint position and velocity sensors for all leg joints
    for leg in ["fr", "fl", "rr", "rl"]:
        for joint_name in ["hip_roll", "hip_pitch", "knee"]:
            full_joint_name = f"{leg}_{joint_name}"
            sensor.add(
                "jointpos",
                name=f"{full_joint_name}_pos",
                joint=full_joint_name,
            )

    # duplication to simplify reading out sensor data.
    for leg in ["fr", "fl", "rr", "rl"]:
        for joint_name in ["hip_roll", "hip_pitch", "knee"]:
            full_joint_name = f"{leg}_{joint_name}"
            sensor.add(
                "jointvel",
                name=f"{full_joint_name}_vel",
                joint=full_joint_name,
            )

    # IMU-based sensors
    sensor.add("gyro", site="com_vertebrae_1", name="gyro")
    sensor.add("velocimeter", site="com_vertebrae_1", name="local_linvel")
    sensor.add("accelerometer", site="com_vertebrae_1", name="accelerometer")
    sensor.add(
        "framepos", objtype="site", objname="com_vertebrae_1", name="position"
    )
    sensor.add(
        "framezaxis",
        objtype="site",
        objname="com_vertebrae_1",
        name="upvector",
    )
    sensor.add(
        "framexaxis",
        objtype="site",
        objname="com_vertebrae_1",
        name="forwardvector",
    )
    sensor.add(
        "framelinvel",
        objtype="site",
        objname="com_vertebrae_1",
        name="global_linvel",
    )
    sensor.add(
        "frameangvel",
        objtype="site",
        objname="com_vertebrae_1",
        name="global_angvel",
    )
    sensor.add(
        "framequat",
        objtype="site",
        objname="com_vertebrae_1",
        name="orientation",
    )

    # 为每只脚添加全局线性速度传感器
    for leg in ["fr", "fl", "rr", "rl"]:
        sensor.add(
            "framelinvel",
            name=f"{leg}_foot_global_linvel",
            objtype="site",
            objname=f"{leg}_foot_site",
        )

    # 每个脚和地面之间的触觉传感器
    for leg in ["fr", "fl", "rr", "rl"]:
        sensor.add(
            "touch",
            name=f"{leg}_foot_touch",
            site=f"{leg}_foot_site",
        )


def generate_keyframe(model, config):
    # 单根棍子模式：只有一个自由体 vertebrae_0
    z = config["spine"]["initial_z"]
    leg_q = config["keyframe"]["leg_qpos"]

    qpos = []
    qpos.extend([0, 0, z, 1, 0, 0, 0])  # vertebrae_0 的自由体位姿
    # 前、后各两条腿的关节初值
    qpos.extend(leg_q)  # 前右、前左
    qpos.extend(leg_q)  # 后右、后左

    ctrl = np.array(leg_q * 2)
    model.keyframe.add("key", name="stable_pose", qpos=np.array(qpos), ctrl=ctrl)


def generate_quadruped_from_config(config: dict):
    model = generate_root(config)
    add_terrain(model, config)
    add_single_stick_body(model, config)  # 用单根棍子的主身体
    add_actuation(model, config)
    add_sensors(model, config)
    generate_keyframe(model, config)

    path = config.get(
        "output_path", ROOT_PATH / "xmls" / "scene_tensegrity_quadruped.xml"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as file:
        file.write(model.to_xml_string())


if __name__ == "__main__":
    generate_quadruped_from_config(DEFAULT_CONFIG)


# from etils import epath
# import numpy as np
# from dm_control import mjcf

# ROOT_PATH = epath.Path(__file__).parent

# # 以 go1 的髋-髋间距作为“身长”
# GO1_HIP_TO_HIP_LENGTH = 2 * 0.1881  # = 0.3762 m

# DEFAULT_CONFIG = {
#     "output_path": ROOT_PATH / "xmls" / "scene_tensegrity_quadruped.xml",
#     "sim": {
#         "timestep": 0.01,
#         "integrator": "RK4",
#         "iterations": 100,
#         "ls_iterations": 50,
#     },
#     "defaults": {
#         "tq1": {
#             "geom": {"condim": 3, "contype": 0, "conaffinity": 0},
#             "joint": {
#                 "type": "hinge",
#                 "limited": True,
#                 "damping": 0.1,
#                 "armature": 0.005,
#                 "frictionloss": 0.001,
#             },
#             "position": {
#                 "kp": 50,
#                 "ctrlrange": [-2.356, 2.356],
#                 "forcerange": [-50, 50],
#             },
#         },
#         "vertebra_geom": {
#             "geom": {
#                 "type": "capsule",
#                 "size": [0.01],
#                 "density": 1750,
#                 "rgba": [0.8, 0.6, 0.4, 1],
#                 "group": 0,
#             },
#         },
#         "leg_geom": {
#             "geom": {
#                 "type": "capsule",
#                 "size": [0.02],
#                 "density": 1750.0,
#                 "rgba": [0.8, 0.6, 0.4, 1],
#                 "group": 1,
#                 "conaffinity": 0,
#             },
#         },
#         "lateral_tendon": {
#             "tendon": {
#                 "stiffness": 5000,
#                 "damping": 10.0,
#                 "frictionloss": 0.05,
#                 "width": 0.002,
#                 "rgba": [1.0, 0.0, 0.0, 0.5],
#             },
#         },
#         "diagonal_tendon": {
#             "tendon": {
#                 "stiffness": 5000,
#                 "damping": 10.0,
#                 "frictionloss": 0.05,
#                 "width": 0.002,
#                 "rgba": [0.0, 0.0, 1.0, 0.5],
#             },
#         },
#     },
#     "leg_dimensions": {
#         "hip_roll_length": 0.05,   # 外展短连杆长度（水平）
#         "hip_pitch_length": 0.213,  # 大腿段长度
#         "shin_length": 0.213,       # 小腿段长度
#         "foot_radius": 0.023,       # 足端球半径
#     },
#     "spine": {
#         "num_segments": 2,
#         "segment_spacing": 0.06,
#         "initial_z": 0.38,
#         "alpha": np.pi / 4,
#         "alpha_length": 0.08,
#         "beta": np.pi / 4,
#         "beta_length": 0.06,
#         "lateral_pretension": 0.98,
#         "diagonal_pretension": 0.98,
#         # 固定身长
#         "enforce_fixed_length": True,
#         "fixed_length": GO1_HIP_TO_HIP_LENGTH,
#         "length_tolerance": 1e-6,
#         # 新增：刚性脊椎开关（True=刚性链接；False=索链接）
#         "rigid_spine": True,
#     },
#     "actuation": {
#         "hip_roll": {"ctrlrange": [-0.863, 0.863], "forcerange": [-35.55, 35.55]},
#         "hip_pitch": {"ctrlrange": [-0.686, 4.501], "forcerange": [-23.7, 23.7]},
#         "knee": {"ctrlrange": [-2.818, -0.888], "forcerange": [-35.55, 35.55]},
#     },
#     "legs": {"z_offset": -0.025},
#     "keyframe": {"leg_qpos": [-0.06, 0.9, -1.55, 0.06, 0.9, -1.55]},
# }


# def generate_root(config):
#     model = mjcf.RootElement()
#     model.compiler.angle = "radian"
#     model.option.timestep = config["sim"]["timestep"]
#     model.option.integrator = config["sim"]["integrator"]
#     model.option.iterations = config["sim"]["iterations"]
#     model.option.ls_iterations = config["sim"]["ls_iterations"]

#     main_class = model.default.add("default", dclass="tq1")
#     subclasses = config["defaults"]
#     for name, entry in subclasses.items():
#         if name == "tq1":
#             target = main_class
#         else:
#             target = main_class.add("default", dclass=name)
#         for elem_type in ("geom", "joint", "tendon", "position"):
#             attrs = entry.get(elem_type)
#             if attrs and hasattr(target, elem_type):
#                 element = getattr(target, elem_type)
#                 for k, v in attrs.items():
#                     setattr(element, k, v)
#     return model


# def add_terrain(model, config):
#     model.asset.add(
#         "texture",
#         name="skybox",
#         type="skybox",
#         builtin="gradient",
#         rgb1=[0.4, 0.6, 0.8],
#         rgb2=[0, 0, 0],
#         width=800,
#         height=800,
#         mark="random",
#         markrgb=[1, 1, 1],
#     )
#     model.asset.add(
#         "texture",
#         name="grid",
#         type="2d",
#         builtin="checker",
#         rgb1=[0.1, 0.2, 0.3],
#         rgb2=[0.2, 0.3, 0.4],
#         width=300,
#         height=300,
#         mark="edge",
#         markrgb=[0.2, 0.3, 0.4],
#     )
#     model.asset.add(
#         "material",
#         name="grid",
#         texture="grid",
#         texrepeat=[5, 5],
#         texuniform=True,
#         reflectance=0.2,
#     )
#     model.worldbody.add(
#         "geom",
#         name="floor",
#         type="plane",
#         size=[20, 20, 0.1],
#         pos=[0, 0, 0],
#         material="grid",
#     )
#     return model


# def add_tetrahedral_spine(model, config):
#     from math import sin, cos

#     num_segments = config["spine"]["num_segments"]
#     segment_spacing = config["spine"]["segment_spacing"]
#     lateral_pretension = config["spine"]["lateral_pretension"]
#     rigid_spine = config["spine"].get("rigid_spine", False)

#     alpha = config["spine"]["alpha"]
#     beta = config["spine"]["beta"]
#     a1, a2 = (np.cos(alpha) * config["spine"]["alpha_length"],
#               np.sin(alpha) * config["spine"]["alpha_length"])
#     b1, b2 = (np.cos(beta) * config["spine"]["beta_length"],
#               np.sin(beta) * config["spine"]["beta_length"])

#     endpoints = {
#         "a1": [-a1, 0.0, a2],
#         "a2": [-a1, 0.0, -a2],
#         "b1": [b1, b2, 0.0],
#         "b2": [b1, -b2, 0.0],
#     }

#     initial_z = config["spine"]["initial_z"]

#     # 固定身长逻辑
#     enforce = config["spine"].get("enforce_fixed_length", False)
#     fixed_len = config["spine"].get("fixed_length", GO1_HIP_TO_HIP_LENGTH)
#     tol = config["spine"].get("length_tolerance", 1e-6)

#     chain_len = max(0.0, (num_segments - 1) * segment_spacing)
#     extra_total = 0.0
#     front_extra = 0.0
#     rear_extra = 0.0
#     if enforce:
#         if chain_len - fixed_len > tol:
#             raise ValueError(
#                 f"Spine length ({chain_len:.4f} m) with num_segments={num_segments}, "
#                 f"segment_spacing={segment_spacing:.4f} exceeds fixed_length={fixed_len:.4f} m. "
#                 "Reduce num_segments or segment_spacing."
#             )
#         extra_total = max(0.0, fixed_len - chain_len)
#         front_extra = extra_total / 2.0
#         rear_extra = extra_total / 2.0

#     imu_x = front_extra if (enforce and extra_total > tol) else 0.0
#     x_positions = [chain_len / 2.0 - i * segment_spacing for i in range(num_segments)]

#     if rigid_spine:
#         # 刚性版本：仅 vertebrae_0 为 free，其余以无关节子体形式刚性连接
#         root = model.worldbody.add(
#             "body", name="vertebrae_0", pos=[x_positions[0], 0, initial_z], childclass="tq1"
#         )
#         root.add("joint", name="joint_vertebrae_0", type="free")

#         # 可视与 IMU
#         root.add("light", name="tracking", mode="trackcom", pos=[0, 0, 3], diffuse=[0.6, 0.6, 0.6])
#         root.add(
#             "camera", name="track", pos=[0.846, -1.3, 0.316],
#             xyaxes=[0.866, 0.5, 0.0, -0.171, 0.296, 0.94], mode="trackcom",
#         )
#         root.add("camera", name="top", pos=[-1, 0, 1], xyaxes=[0, -1, 0, 0.7, 0, 0.7], mode="trackcom")
#         root.add("camera", name="side", pos=[0, -1, 0.3], xyaxes=[1, 0, 0, 0, 1, 2], mode="trackcom")
#         root.add("camera", name="back", pos=[-1, 0, 0.3], xyaxes=[0, -1, 0, 1, 0, 2], mode="trackcom")
#         root.add("site", name="com_vertebrae_1", pos=[imu_x, 0, 0], size=[0.006])

#         # 段端点几何
#         for key, endpoint in endpoints.items():
#             root.add("geom", name=f"vertebrae_0_{key}_geom",
#                      fromto=[0, 0, 0, *endpoint], dclass="vertebra_geom")
#             root.add("site", name=f"vertebrae_0_{key}", pos=endpoint, size=[0.005])

#         parent = root
#         for i in range(1, num_segments):
#             body = parent.add(
#                 "body", name=f"vertebrae_{i}", pos=[-segment_spacing, 0, 0], childclass="tq1"
#             )
#             # 可视连接杆（父原点 -> 子原点）
#             parent.add(
#                 "geom", name=f"spine_link_{i-1}_{i}", type="capsule",
#                 fromto=[0, 0, 0, -segment_spacing, 0, 0], dclass="vertebra_geom"
#             )
#             for key, endpoint in endpoints.items():
#                 body.add("geom", name=f"vertebrae_{i}_{key}_geom",
#                          fromto=[0, 0, 0, *endpoint], dclass="vertebra_geom")
#                 body.add("site", name=f"vertebrae_{i}_{key}", pos=endpoint, size=[0.005])
#             parent = body

#         # 延长段（可选）
#         if enforce and extra_total > tol:
#             root.add("geom", name="front_spacer",
#                      type="capsule", fromto=[0, 0, 0, front_extra, 0, 0], dclass="vertebra_geom")
#             parent.add("geom", name="rear_spacer",
#                        type="capsule", fromto=[0, 0, 0, -rear_extra, 0, 0], dclass="vertebra_geom")

#     else:
#         # 索版本：每段为 free，并通过 spatial tendons 相连
#         bodies = []
#         for i in range(num_segments):
#             name = f"vertebrae_{i}"
#             pos = [x_positions[i], 0, initial_z]
#             body = model.worldbody.add("body", name=name, pos=pos, childclass="tq1")
#             if i == 0:
#                 body.add("light", name="tracking", mode="trackcom", pos=[0, 0, 3], diffuse=[0.6] * 3)
#                 body.add(
#                     "camera", name="track", pos=[0.846, -1.3, 0.316],
#                     xyaxes=[0.866, 0.5, 0.0, -0.171, 0.296, 0.94], mode="trackcom"
#                 )
#                 body.add("camera", name="top", pos=[-1, 0, 1],
#                          xyaxes=[0, -1, 0, 0.7, 0, 0.7], mode="trackcom")
#                 body.add("camera", name="side", pos=[0, -1, 0.3],
#                          xyaxes=[1, 0, 0, 0, 1, 2], mode="trackcom")
#                 body.add("camera", name="back", pos=[-1, 0, 0.3],
#                          xyaxes=[0, -1, 0, 1, 0, 2], mode="trackcom")
#             body.add("joint", name=f"joint_{name}", type="free")
#             if i == 0:
#                 body.add("site", name="com_vertebrae_1", pos=[imu_x, 0, 0], size=[0.006])
#             for key, endpoint in endpoints.items():
#                 body.add("geom", name=f"{name}_{key}_geom",
#                          fromto=[0, 0, 0, *endpoint], dclass="vertebra_geom")
#                 body.add("site", name=f"{name}_{key}", pos=endpoint, size=[0.005])
#             bodies.append(body)

#         for i in range(num_segments - 1):
#             for site in ["a1", "a2", "b1", "b2"]:
#                 tendon = model.tendon.add(
#                     "spatial",
#                     name=f"lat_{site}_{i}",
#                     dclass="lateral_tendon",
#                     springlength=[segment_spacing * lateral_pretension],
#                 )
#                 tendon.add("site", site=f"vertebrae_{i}_{site}")
#                 tendon.add("site", site=f"vertebrae_{i + 1}_{site}")

#             for a, b in [("a1", "b1"), ("a2", "b2"), ("a1", "b2"), ("a2", "b1")]:
#                 tendon = model.tendon.add("spatial", name=f"diag_{a}{b}_{i}", dclass="diagonal_tendon")
#                 tendon.add("site", site=f"vertebrae_{i}_{a}")
#                 tendon.add("site", site=f"vertebrae_{i + 1}_{b}")

#     # 腿安装偏移（若启用固定身长）
#     front_offset = front_extra if (enforce and extra_total > tol) else 0.0
#     rear_offset = -rear_extra if (enforce and extra_total > tol) else 0.0
#     add_legs(model, config, front_offset=front_offset, rear_offset=rear_offset)
#     return model


# def add_legs(model, config, front_offset=0.0, rear_offset=0.0):
#     z = config["legs"]["z_offset"]
#     leg_segment = config["spine"]["num_segments"] - 1

#     for idx, segment in [(0, "front"), (leg_segment, "hind")]:
#         name = f"vertebrae_{idx}"
#         body = model.find("body", name)
#         x_offset = front_offset if idx == 0 else rear_offset

#         body.add("geom", name=f"{segment}_shoulder",
#                  fromto=[x_offset, -0.12675, z, x_offset, 0.12675, z], dclass="leg_geom")
#         body.add("geom", name=f"{segment}_shoulder_attach",
#                  fromto=[x_offset, 0, 0, x_offset, 0, z], dclass="leg_geom", size=[0.005])

#         for side in ["fr", "fl"] if idx == 0 else ["rr", "rl"]:
#             base = [x_offset, -0.12675, z] if side.endswith("r") else [x_offset, 0.12675, z]
#             add_leg(body, side, base)


# def add_leg(parent_body, prefix, base_pos):
#     roll_len = DEFAULT_CONFIG["leg_dimensions"]["hip_roll_length"]
#     pitch_len = DEFAULT_CONFIG["leg_dimensions"]["hip_pitch_length"]
#     shin_len = DEFAULT_CONFIG["leg_dimensions"]["shin_length"]
#     foot_r = DEFAULT_CONFIG["leg_dimensions"]["foot_radius"]

#     act_cfg = DEFAULT_CONFIG["actuation"]
#     hip_roll_range = act_cfg["hip_roll"]["ctrlrange"]
#     hip_pitch_range = act_cfg["hip_pitch"]["ctrlrange"]
#     knee_range = act_cfg["knee"]["ctrlrange"]

#     root = parent_body.add("body", name=f"{prefix}_leg", pos=base_pos)
#     root.add("joint", name=f"{prefix}_hip_roll", axis=[1, 0, 0], range=hip_roll_range)
#     root.add("geom", name=f"{prefix}_hip_roll_geom", fromto=[0, 0, 0, -roll_len, 0, 0], dclass="leg_geom")

#     hip = root.add("body", name=f"{prefix}_hip_pitch", pos=[-roll_len, 0, 0])
#     hip.add("joint", name=f"{prefix}_hip_pitch", axis=[0, 1, 0], range=hip_pitch_range)
#     hip.add("geom", name=f"{prefix}_hip_pitch_geom", fromto=[0, 0, 0, 0, 0, -pitch_len], dclass="leg_geom")

#     knee = hip.add("body", name=f"{prefix}_knee", pos=[0, 0, -pitch_len])
#     knee.add("joint", name=f"{prefix}_knee", axis=[0, 1, 0], range=knee_range)
#     knee.add("geom", name=f"{prefix}_shin_geom", fromto=[0, 0, 0, 0, 0, -shin_len],
#              dclass="leg_geom", conaffinity=1)

#     foot = knee.add("body", name=f"{prefix}_foot", pos=[0, 0, -shin_len])
#     foot.add("geom", name=f"{prefix}_foot_geom", type="sphere", size=[foot_r],
#              conaffinity=1, dclass="leg_geom", friction=[0.8, 0.02, 0.01])
#     foot.add("site", name=f"{prefix}_foot_site", pos=[0, 0, 0], size=[foot_r])


# def add_actuation(model, config):
#     act_cfg = config["actuation"]
#     for leg in ["fr", "fl", "rr", "rl"]:
#         for joint_name in ["hip_roll", "hip_pitch", "knee"]:
#             cfg = act_cfg[joint_name]
#             model.actuator.add(
#                 "position",
#                 name=f"{leg}_{joint_name}",
#                 joint=f"{leg}_{joint_name}",
#                 ctrlrange=cfg["ctrlrange"],
#                 forcerange=cfg.get("forcerange", None),
#             )


# def add_sensors(model, config):
#     sensor = model.sensor
#     for leg in ["fr", "fl", "rr", "rl"]:
#         for joint_name in ["hip_roll", "hip_pitch", "knee"]:
#             full = f"{leg}_{joint_name}"
#             sensor.add("jointpos", name=f"{full}_pos", joint=full)
#     for leg in ["fr", "fl", "rr", "rl"]:
#         for joint_name in ["hip_roll", "hip_pitch", "knee"]:
#             full = f"{leg}_{joint_name}"
#             sensor.add("jointvel", name=f"{full}_vel", joint=full)

#     sensor.add("gyro", site="com_vertebrae_1", name="gyro")
#     sensor.add("velocimeter", site="com_vertebrae_1", name="local_linvel")
#     sensor.add("accelerometer", site="com_vertebrae_1", name="accelerometer")
#     sensor.add("framepos", objtype="site", objname="com_vertebrae_1", name="position")
#     sensor.add("framezaxis", objtype="site", objname="com_vertebrae_1", name="upvector")
#     sensor.add("framexaxis", objtype="site", objname="com_vertebrae_1", name="forwardvector")
#     sensor.add("framelinvel", objtype="site", objname="com_vertebrae_1", name="global_linvel")
#     sensor.add("frameangvel", objtype="site", objname="com_vertebrae_1", name="global_angvel")
#     sensor.add("framequat", objtype="site", objname="com_vertebrae_1", name="orientation")

#     for leg in ["fr", "fl", "rr", "rl"]:
#         sensor.add("framelinvel", name=f"{leg}_foot_global_linvel",
#                    objtype="site", objname=f"{leg}_foot_site")

#     for leg in ["fr", "fl", "rr", "rl"]:
#         sensor.add("touch", name=f"{leg}_foot_touch", site=f"{leg}_foot_site")


# def generate_keyframe(model, config):
#     num_segments = config["spine"]["num_segments"]
#     z = config["spine"]["initial_z"]
#     leg_q = config["keyframe"]["leg_qpos"]
#     rigid_spine = config["spine"].get("rigid_spine", False)

#     qpos = []

#     if rigid_spine:
#         # 只有 vertebrae_0 是 free
#         qpos.extend([0, 0, z, 1, 0, 0, 0])
#     else:
#         # 每段都是 free
#         for _ in range(num_segments):
#             qpos.extend([0, 0, z, 1, 0, 0, 0])

#     # 腿关节位置（四条腿=12个 DOF）
#     # leg_q 定义为 6 个 [hr, hp, kn, hr, hp, kn]，前端两条腿
#     # 这里重复两次，覆盖后端两条腿
#     leg_q_all = leg_q * 2
#     qpos.extend(leg_q_all)

#     ctrl = np.array(leg_q_all)

#     # 可选安全检查（生成 XML 前不会有 model.nq，这里跳过 runtime 检查）
#     model.keyframe.add("key", name="stable_pose", qpos=np.array(qpos), ctrl=ctrl)


# def generate_quadruped_from_config(config: dict):
#     model = generate_root(config)
#     add_terrain(model, config)
#     add_tetrahedral_spine(model, config)
#     add_actuation(model, config)
#     add_sensors(model, config)
#     generate_keyframe(model, config)

#     path = config.get("output_path", ROOT_PATH / "xmls" / "scene_tensegrity_quadruped.xml")
#     path.parent.mkdir(parents=True, exist_ok=True)
#     with open(path, "w") as file:
#         file.write(model.to_xml_string())


# if __name__ == "__main__":
#     generate_quadruped_from_config(DEFAULT_CONFIG)
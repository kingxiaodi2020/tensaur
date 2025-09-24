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
        "integrator": "Euler",
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
                "kp": 35.0,
                "ctrlrange": [-2.356, 2.356],
                "forcerange": [-50, 50],
            },
        },
        "vertebra_geom": {
            "geom": {
                "type": "capsule",
                "size": [0.01],
                "density": 800,  # 370.0,
                "rgba": [0.8, 0.6, 0.4, 1],
                "group": 0,
            },
        },
        "leg_geom": {
            "geom": {
                "type": "capsule",
                "size": [0.02],  # 胶囊半径更接近 go1 可视化尺度
                "density": 1370.0,  # 实际质量由 inertial 指定
                "rgba": [0.8, 0.6, 0.4, 1],
                "group": 1,
                "conaffinity": 1,
            },
        },
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
        "hip_roll_length": 0.045,   # 外展短连杆长度（水平）
        "hip_pitch_length": 0.213,  # 大腿段长度
        "shin_length": 0.213,       # 小腿段长度
        "foot_radius": 0.023,       # 足端球半径
    },
    "spine": {
        "num_segments": 3,
        "segment_spacing": 0.06,
        "initial_z": 0.38,  # 腿变长后抬高初始质心高度以避免穿地 0.474 站直
        "alpha": np.pi / 4,
        "alpha_length": 0.08,
        "beta": np.pi / 4,
        "beta_length": 0.06,
        "lateral_pretension": 0.98,
        "diagonal_pretension": 0.98,

        # 固定身长配置（默认对齐 go1 髋-髋间距）
        "enforce_fixed_length": True,
        "fixed_length": GO1_HIP_TO_HIP_LENGTH,
        "length_tolerance": 1e-6,  # 容差
    },
    "actuation": {
        # 参考 go1 的关节范围与力矩限制（位置型执行器的 forcerange）
        "hip_roll": {"ctrlrange": [-0.863, 0.863], "forcerange": [-23.7, 23.7]},
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


def add_tetrahedral_spine(model, config):
    from math import sin, cos

    num_segments = config["spine"]["num_segments"]
    segment_spacing = config["spine"]["segment_spacing"]
    diagonal_pretension = config["spine"]["diagonal_pretension"]
    lateral_pretension = config["spine"]["lateral_pretension"]

    alpha = config["spine"]["alpha"]
    beta = config["spine"]["beta"]
    a1, a2 = (
        cos(alpha) * config["spine"]["alpha_length"],
        sin(alpha) * config["spine"]["alpha_length"],
    )
    b1, b2 = (
        cos(beta) * config["spine"]["beta_length"],
        sin(beta) * config["spine"]["beta_length"],
    )

    # 端点相对局部坐标
    endpoints = {
        "a1": [-a1, 0.0, a2],
        "a2": [-a1, 0.0, -a2],
        "b1": [b1, b2, 0.0],
        "b2": [b1, -b2, 0.0],
    }

    initial_z = config["spine"]["initial_z"]

    # 固定身长逻辑
    enforce = config["spine"].get("enforce_fixed_length", False)
    fixed_len = config["spine"].get("fixed_length", GO1_HIP_TO_HIP_LENGTH)
    tol = config["spine"].get("length_tolerance", 1e-6)

    chain_len = max(0.0, (num_segments - 1) * segment_spacing)
    extra_total = 0.0
    front_extra = 0.0
    rear_extra = 0.0

    if enforce:
        if chain_len - fixed_len > tol:
            raise ValueError(
                f"Spine length ({chain_len:.4f} m) with num_segments={num_segments}, "
                f"segment_spacing={segment_spacing:.4f} exceeds fixed_length={fixed_len:.4f} m. "
                "Reduce num_segments or segment_spacing."
            )
        extra_total = max(0.0, fixed_len - chain_len)
        # 前后对称延长
        front_extra = extra_total / 2.0
        rear_extra = extra_total / 2.0

    # 为了整体关于 x=0 对称，链段中心放在 x=0
    # 链起点、终点（不含延长段）分别为 +chain_len/2 与 -chain_len/2
    # 前/后延长段分别从这两端继续向外延伸 front_extra 和 rear_extra
    x_positions = [chain_len / 2.0 - i * segment_spacing for i in range(num_segments)]

    # 创建各个“椎体” body
    for i in range(num_segments):
        name = f"vertebrae_{i}"
        pos = [x_positions[i], 0, initial_z]
        body = model.worldbody.add("body", name=name, pos=pos, childclass="tq1")
        if i == 0:
            body.add(
                "light", 
                name="tracking", 
                mode="trackcom", 
                pos=[0, 0, 3], 
                diffuse=[0.6] * 3
            )
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
        body.add("joint", name=f"joint_{name}", type="free")
        body.add("site", name=f"com_vertebrae_{i}", pos=[0, 0, 0], size=[0.005])

        for key, endpoint in endpoints.items():
            body.add(
                "geom",
                name=f"{name}_{key}_geom",
                fromto=[0, 0, 0, *endpoint],
                dclass="vertebra_geom",
            )
            body.add("site", name=f"{name}_{key}", pos=endpoint, size=[0.005])

    # 椎体之间的“横向”和“对角”拉索
    for i in range(num_segments - 1):
        for site in ["a1", "a2", "b1", "b2"]:
            tendon = model.tendon.add(
                "spatial",
                name=f"lat_{site}_{i}",
                dclass="lateral_tendon",
                springlength=[segment_spacing * config["spine"]["lateral_pretension"]],
            )
            tendon.add("site", site=f"vertebrae_{i}_{site}")
            tendon.add("site", site=f"vertebrae_{i + 1}_{site}")

        for a, b in [("a1", "b1"), ("a2", "b2"), ("a1", "b2"), ("a2", "b1")]:
            tendon = model.tendon.add(
                "spatial",
                name=f"diag_{a}{b}_{i}",
                dclass="diagonal_tendon",
            )
            tendon.add("site", site=f"vertebrae_{i}_{a}")
            tendon.add("site", site=f"vertebrae_{i + 1}_{b}")

    # 若需要补长，则在最前/最后椎体上添加“延长段”几何（沿 x 轴）
    if enforce and extra_total > tol:
        front_body = model.find("body", "vertebrae_0")
        back_body = model.find("body", f"vertebrae_{num_segments - 1}")

        front_body.add(
            "geom",
            name="front_spacer",
            type="capsule",
            fromto=[0, 0, 0, front_extra, 0, 0],
            dclass="vertebra_geom",
        )
        back_body.add(
            "geom",
            name="rear_spacer",
            type="capsule",
            fromto=[0, 0, 0, -rear_extra, 0, 0],
            dclass="vertebra_geom",
        )

    # 添加四条腿（把腿基座沿 x 方向偏到延长段末端）
    front_offset = front_extra if (enforce and extra_total > tol) else 0.0   # +x
    rear_offset = -rear_extra if (enforce and extra_total > tol) else 0.0    # -x
    add_legs(model, config, front_offset=front_offset, rear_offset=rear_offset)

    return model


def add_legs(model, config, front_offset=0.0, rear_offset=0.0):
    z = config["legs"]["z_offset"]
    leg_segment = config["spine"]["num_segments"] - 1

    for idx, segment in [(0, "front"), (leg_segment, "hind")]:
        name = f"vertebrae_{idx}"
        body = model.find("body", name)

        # 根据前/后分别采用不同的 x 偏移（把肩部几何移动到髋安装点位置）
        x_offset = front_offset if idx == 0 else rear_offset

        body.add(
            "geom",
            name=f"{segment}_shoulder",
            fromto=[x_offset, -0.12675, z, x_offset, 0.12675, z],
            dclass="leg_geom",
        )
        body.add(
            "geom",
            name=f"{segment}_shoulder_attach",
            fromto=[x_offset, 0, 0, x_offset, 0, z],
            dclass="leg_geom",
            size=[0.005],
        )

        # 右腿 y<0, 左腿 y>0
        for side in ["fr", "fl"] if idx == 0 else ["rr", "rl"]:
            base = [x_offset, -0.12675, z] if side.endswith("r") else [x_offset, 0.12675, z]
            add_leg(body, side, base)


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
        damping=0.5,
        frictionloss=0.3,
        armature=0.005,
    )
    # 近似赋予髋外展部件的质量与惯量（参考 go1）
    root.add(
        "inertial",
        pos=[-roll_len / 2, 0, 0],
        mass=0.68,
        diaginertia=[0.000734, 0.000468, 0.000399],
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
        damping=0.5,
        frictionloss=0.3,
        armature=0.005,
    )
    # 大腿质量与惯量（参考 go1）
    hip.add(
        "inertial",
        pos=[0, 0, -pitch_len / 2],
        mass=1.009,
        diaginertia=[0.004787, 0.004609, 0.000709],
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
        damping=0.5,
        frictionloss=1.0,  # 膝关节摩擦更大
        armature=0.005,
    )
    # 小腿质量与惯量（参考 go1）
    knee.add(
        "inertial",
        pos=[0, 0, -shin_len / 2],
        mass=0.196,
        diaginertia=[0.001498, 0.001485, 3.6e-05],
    )
    knee.add(
        "geom",
        name=f"{prefix}_shin_geom",
        fromto=[0, 0, 0, 0, 0, -shin_len],
        dclass="leg_geom",
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
        condim=3,
        solimp=[0.9, 0.95, 0.023],
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
    sensor = model.sensor

    # 关节位置/速度
    for leg in ["fr", "fl", "rr", "rl"]:
        for joint_name in ["hip_roll", "hip_pitch", "knee"]:
            full_joint_name = f"{leg}_{joint_name}"
            sensor.add("jointpos", name=f"{full_joint_name}_pos", joint=full_joint_name)
            sensor.add("jointvel", name=f"{full_joint_name}_vel", joint=full_joint_name)

    # IMU-based 传感器（挂在中段）
    sensor.add("gyro", site="com_vertebrae_1", name="gyro")
    sensor.add("velocimeter", site="com_vertebrae_1", name="local_linvel")
    sensor.add("accelerometer", site="com_vertebrae_1", name="accelerometer")
    sensor.add("framepos", objtype="site", objname="com_vertebrae_1", name="position")
    sensor.add("framezaxis", objtype="site", objname="com_vertebrae_1", name="upvector")
    sensor.add("framexaxis", objtype="site", objname="com_vertebrae_1", name="forwardvector")
    sensor.add("framelinvel", objtype="site", objname="com_vertebrae_1", name="global_linvel")
    sensor.add("frameangvel", objtype="site", objname="com_vertebrae_1", name="global_angvel")
    sensor.add("framequat", objtype="site", objname="com_vertebrae_1", name="orientation")

    # 每只脚的全局速度与触觉
    for leg in ["fr", "fl", "rr", "rl"]:
        sensor.add(
            "framelinvel",
            name=f"{leg}_foot_global_linvel",
            objtype="site",
            objname=f"{leg}_foot_site",
        )
        sensor.add("touch", name=f"{leg}_foot_touch", site=f"{leg}_foot_site")


def generate_keyframe(model, config):
    num_segments = config["spine"]["num_segments"]
    spacing = config["spine"]["segment_spacing"]
    z = config["spine"]["initial_z"]
    leg_q = config["keyframe"]["leg_qpos"]

    qpos, ctrl = [], []
    for i in range(num_segments):
        qpos.extend([0, 0, z, 1, 0, 0, 0])  # 自由体位姿：位置先放 0（由仿真稳定后决定）
        if i == 0 or i == num_segments - 1:
            qpos.extend(leg_q)
    ctrl = np.array(leg_q * 2)
    model.keyframe.add("key", name="stable_pose", qpos=np.array(qpos), ctrl=ctrl)


def generate_quadruped_from_config(config: dict):
    model = generate_root(config)
    add_terrain(model, config)
    add_tetrahedral_spine(model, config)
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
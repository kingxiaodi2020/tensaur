from etils import epath
import numpy as np
from dm_control import mjcf
import copy
from pathlib import Path

ROOT_PATH = epath.Path(__file__).parent

# 以 go1 的髋-髋间距作为“身长”
GO1_HIP_TO_HIP_LENGTH = 2 * 0.1881  # = 0.3762 m
visual = True # for visual debug use
mode = "plane"  # "plane" | "uneven"
fixed_shoulder = True  # if lock hip_roll joints

DEFAULT_CONFIG = {
    "output_path": ROOT_PATH / "xmls" / "scene_tensegrity_quadruped.xml",
    "sim": {
        "timestep": 0.002,
        "integrator": "implicitfast",  # 改为 "rk4" 或 "euler"
        "iterations": 100,
        "ls_iterations": 50,
    },
    "defaults": {
        "tq1": {
            "geom": {"condim": 3, "contype": 0, "conaffinity": 0},
            "joint": {
                "type": "hinge",
                "limited": True,
                "damping": 0.5,
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
                "density": 1750,
                "rgba": [0.8, 0.6, 0.4, 1],
                "group": 1,
                "conaffinity": 1,
                "contype": 1,
            },
        },
        "leg_geom": {
            "geom": {
                "type": "capsule",
                "size": [0.02],  # 胶囊半径更接近 go1 可视化尺度
                "density": 1750.0,  
                "rgba": [0.8, 0.6, 0.4, 1],
                "group": 1,
                "conaffinity": 0,
            },
        },
        "lateral_tendon": {
            "tendon": {
                "stiffness": 5000,
                "damping": 	5,
                "frictionloss": 0.02,
                "width": 0.002,
                "rgba": [1.0, 0.0, 0.0, 0.5],
            },
        },
        "diagonal_tendon": {
            "tendon": {
                "stiffness": 3000,
                "damping": 5,
                "frictionloss": 0.02,
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
        "num_segments": 3,
        "segment_spacing": 0.07,
        "initial_z": 0.6,  # 腿变长后抬高初始质心高度以避免穿地 0.474 站直 0.38
        "alpha": np.pi / 4,
        "alpha_length": 0.10,
        "beta": np.pi / 4,
        "beta_length": 0.08,
        "lateral_pretension": 0.90,
        "diagonal_pretension": 0.90,

        # 固定身长配置（默认对齐 go1 髋-髋间距）
        "enforce_fixed_length": True,
        "fixed_length": GO1_HIP_TO_HIP_LENGTH,
        "length_tolerance": 1e-6,  # 容差
        "fix_base": True,
    },
    "actuation": {
        # 参考 go1 的关节范围与力矩限制（位置型执行器的 forcerange）
        "hip_roll": {"ctrlrange": [-0.863, 0.863], "forcerange": [-23.7*0.2, 23.7*0.2]},    #-0.863, 0.863
        "hip_pitch": {"ctrlrange": [-0.686, 4.501], "forcerange": [-23.7*0.2, 23.7*0.2]},
        "knee": {"ctrlrange": [-2.818, -0.888], "forcerange": [-35.55*0.2, 35.55*0.2]},
    },
    "legs": {"z_offset": -0.025},
    "keyframe": {"leg_qpos": [-0, 0.9, -1.55, 0, 0.9, -1.55]},    #[-0.06, 0.9, -1.55, 0.06, 0.9, -1.55]

    "terrain": {
        "mode": mode,   # "plane" | "uneven"
        # mode="uneven"：
        "hfield_png": "terrain_go1_11m_gradual2.png",
        "rx": 7.5,         # half length （m）  -> (-1,10)
        "ry": 1.5,         # half width（m）  -> 总宽 3 m
        "hz": 0.40,        # vertical height scale（m）
        "base": 0.001,     # base height offset（m）
        "center_x": 5.8,   # center x position（m）
        "friction": [1.0, 0.1, 0.01],
        "use_checker_material": True,
    },
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
        # builtin="gradient",
        # rgb1=[0.4, 0.6, 0.8],
        # rgb2=[0, 0, 0],
        builtin="flat",  # 使用 flat 而不是 gradient
        rgb1=[1.0, 1.0, 1.0],  # 设置为纯白色
        rgb2=[1.0, 1.0, 1.0],  # 设置为纯白色
        width=800,
        height=800,
        mark="random",
        markrgb=[1, 1, 1],
    )
    # model.asset.add(
    #     "texture",
    #     name="grid",
    #     type="2d",
    #     builtin="checker",
    #     rgb1=[0.1, 0.2, 0.3],
    #     rgb2=[0.2, 0.3, 0.4],
    #     width=300,
    #     height=300,
    #     mark="edge",
    #     markrgb=[0.2, 0.3, 0.4],
    # )
    # model.asset.add(
    #     "material",
    #     name="grid",
    #     texture="grid",
    #     texrepeat=[5, 5],
    #     texuniform=True,
    #     reflectance=0.2,
    # )
    # model.worldbody.add(
    #     "geom",
    #     name="floor",
    #     type="plane",
    #     size=[20, 20, 0.1],
    #     pos=[0, 0, 0],
    #     # material="grid",
    #     rgba=[1.0, 1.0, 1.0, 1.0],   
    # )
    
    return model

def add_uneven_terrain(model, config):
    """Adds uneven terrain using a heightfield PNG."""
    tcfg = config.get("terrain", {})
    png_path = tcfg.get("hfield_png")
    if not png_path:
        raise ValueError("Heightfield PNG is required for uneven terrain.")

    # Resolve the full path of the PNG file
    p = epath.Path(png_path)
    if not p.is_absolute():
        alt = ROOT_PATH / "xmls" / png_path
        alt2 = ROOT_PATH / png_path
        p = alt if alt.exists() else alt2 if alt2.exists() else None
        if not p:
            raise FileNotFoundError(f"Heightfield PNG not found: {png_path}")

    rx, ry, hz, base = float(tcfg.get("rx", 5.5)), float(tcfg.get("ry", 1.5)), float(tcfg.get("hz", 0.30)), float(tcfg.get("base", 0.005))
    cx, fric = float(tcfg.get("center_x", 4.8)), tcfg.get("friction", [1.0, 0.1, 0.01])
    use_mat = bool(tcfg.get("use_checker_material", True))

    # Add skybox texture
    model.asset.add("texture", name="skybox", type="skybox", builtin="gradient", rgb1=[0.4, 0.6, 0.8], rgb2=[0, 0, 0], width=800, height=800, mark="random", markrgb=[1, 1, 1])

    # Add ground material if enabled
    if use_mat:
        model.asset.add("texture", name="tex_ground", type="2d", builtin="checker", width=256, height=256, rgb1=[0.75, 0.75, 0.75], rgb2=[0.65, 0.65, 0.65])
        model.asset.add("material", name="mat_ground", texture="tex_ground", texrepeat=[30, 6], rgba=[1, 1, 1, 1])

    # Add heightfield asset
    model.asset.add("hfield", name="uneven_terrain", file=str(p), size=[rx, ry, hz, base])

    # Add terrain geometry
    kwargs = {"material": "mat_ground"} if use_mat else {}
    model.worldbody.add("geom", name="terrain", type="hfield", hfield="uneven_terrain", size=[rx, ry, hz], pos=[cx, 0, 0], friction=fric, **kwargs)

def add_tetrahedral_spine(model, config, with_legs=True):
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

    fix_base = config["spine"].get("fix_base", False)

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

    imu_x = 0.5 * (fixed_len - chain_len) if enforce else 0.0
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

            body.add(
                "camera", 
                name="overview",
                pos=[-1, 0, 2],  # 居中高位
                xyaxes=[0, -1, 0, 0.5, 0, 0.5],  # 向下俯视
                mode="trackcom",
                fovy=70,
            )

        if not (fix_base and i == 0):
            body.add("joint", name=f"joint_{name}", type="free")
        if i == 0:
            body.add("site", name="com_vertebrae_1", pos=[imu_x, 0, 0], size=[0.005])

        for key, endpoint in endpoints.items():
            body.add(
                "geom",
                name=f"{name}_{key}_geom",
                fromto=[0, 0, 0, *endpoint],
                dclass="vertebra_geom",
            )
            body.add("site", name=f"{name}_{key}", pos=endpoint, size=[0.005])

    def world_site_pos(body_name: str, site_suffix: str):
        body = model.find("body", body_name)
        site = model.find("site", f"{body_name}_{site_suffix}")
        return np.array(body.pos, dtype=float) + np.array(site.pos, dtype=float)
    
    # lateral/diagonal 的 pretension 仍来自 config["spine"]，不在 genes 里改  :contentReference[oaicite:2]{index=2}
    lateral_map = {"a1": "lateral_top", "a2": "lateral_bottom", "b1": "lateral_left", "b2": "lateral_right"}
    diag_map = {("a1","b1"): "diag_a1b1", ("a2","b2"): "diag_a2b2",
                ("a1","b2"): "diag_a1b2", ("a2","b1"): "diag_a2b1"}

    for i in range(num_segments - 1):
        # 4× lateral（同名点相连）
        for site in ["a1", "a2", "b1", "b2"]:
            pA = world_site_pos(f"vertebrae_{i}", site)
            pB = world_site_pos(f"vertebrae_{i+1}", site)
            L0 = float(np.linalg.norm(pB - pA))
            cls = lateral_map.get(site, "lateral_tendon")
            tendon = model.tendon.add(
                "spatial", name=f"lat_{site}_{i}", dclass=cls,
                springlength=[float(L0 * lateral_pretension)]
            )
            tendon.add("site", site=f"vertebrae_{i}_{site}")
            tendon.add("site", site=f"vertebrae_{i+1}_{site}")

        # 4× diagonal（四种配对）
        for a, b in [("a1","b1"),("a2","b2"),("a1","b2"),("a2","b1")]:
            pA = world_site_pos(f"vertebrae_{i}", a)
            pB = world_site_pos(f"vertebrae_{i+1}", b)
            L0 = float(np.linalg.norm(pB - pA))
            cls = diag_map.get((a,b), "diagonal_tendon")
            tendon = model.tendon.add(
                "spatial", name=f"diag_{a}{b}_{i}", dclass=cls,
                springlength=[float(L0 * diagonal_pretension)]
            )
            tendon.add("site", site=f"vertebrae_{i}_{a}")
            tendon.add("site", site=f"vertebrae_{i+1}_{b}")


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

        # 新增：脊柱后端 tip site，正好在延长段末端
        back_body.add(
            "site",
            name="spine_tip",
            pos=[-rear_extra, 0, 0],
            size=[0.005],
        )

    if with_legs:
        # 添加四条腿（把腿基座沿 x 方向偏到延长段末端）
        front_offset = front_extra if (enforce and extra_total > tol) else 0.0   # +x
        rear_offset = -rear_extra if (enforce and extra_total > tol) else 0.0    # -x
        add_legs(model, config, front_offset=front_offset, rear_offset=rear_offset)

    return model


def add_legs(model, config, front_offset=0.0, rear_offset=0.0):
    z = config["legs"]["z_offset"]
    leg_segment = config["spine"]["num_segments"] - 1

    for idx, segment in [(leg_segment, "front"), (0, "hind")]:
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

    # lock hip_roll joint if specified
    if fixed_shoulder:
        hip_roll_range = [0.0, 1e-20]

    # 髋外展段（沿 x 轴）
    root = parent_body.add("body", name=f"{prefix}_leg", pos=base_pos)
    root.add(
        "joint",
        name=f"{prefix}_hip_roll",
        axis=[1, 0, 0],
        range=hip_roll_range,
        frictionloss=0.3,  # 外展关节摩擦
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
        frictionloss=0.3,  # 髋俯仰关节摩擦
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
        frictionloss=1.0,  # 膝关节摩擦最大
    )

    knee.add(
        "geom",
        name=f"{prefix}_shin_geom",
        fromto=[0, 0, 0, 0, 0, -shin_len],
        dclass="leg_geom",
        conaffinity=0,
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
    lock_hip_roll = fixed_shoulder

    for leg in ["fr", "fl", "rr", "rl"]:
        for joint_name in ["hip_roll", "hip_pitch", "knee"]:
            cfg = act_cfg[joint_name]

            if joint_name == "hip_roll" and lock_hip_roll:
                model.actuator.add(
                    "position",
                    name=f"{leg}_{joint_name}",
                    joint=f"{leg}_{joint_name}",
                    ctrlrange=[0.0, 1e-20],
                    forcerange=[0.0, 0.0],
                )
            else:
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
    n = config["spine"]["num_segments"]
    spacing = config["spine"]["segment_spacing"]
    z = config["spine"]["initial_z"]
    leg_q = config["keyframe"]["leg_qpos"]

    chain_len = max(0.0, (n - 1) * spacing)

    qpos = []
    for i in range(n):
        x_i = chain_len/2.0 - i*spacing         # ★ 用这个，而不是 -i*spacing
        qpos.extend([x_i, 0, z, 1, 0, 0, 0])
        if i == 0 or i == n - 1:
            qpos.extend(leg_q)

    ctrl = np.array(leg_q * 2)
    model.keyframe.add("key", name="stable_pose", qpos=np.array(qpos), ctrl=ctrl)


# 在 generate_rod.py 中，生成XML时就固定了目标位置
def add_target_visualization(model, config, target_distance_bl=15.0, body_length=GO1_HIP_TO_HIP_LENGTH, tolearance=0.1):
    target_x = target_distance_bl * body_length  # 固定在 15*0.3762 = 5.643m 处
    # model.worldbody.add(
    #     "site",
    #     name="target_marker0",
    #     type="sphere",
    #     size=[tolearance],
    #     pos=[-0.05, 0.0, 0.6],  # 位置写死在XML中
    #     rgba=[1, 0, 0, 0.7],
    # )

    # model.worldbody.add(
    #     "site",
    #     name="target_marker1",
    #     type="sphere",
    #     size=[tolearance],
    #     pos=[-GO1_HIP_TO_HIP_LENGTH/2, 0.0, 0.6],  # 位置写死在XML中
    #     rgba=[1, 0, 0, 0.7],
    # )

    # model.worldbody.add(
    #     "site",
    #     name="target_marker2",
    #     type="sphere",
    #     size=[tolearance],
    #     pos=[0, 0.1, 0.6],  # 位置写死在XML中
    #     rgba=[1, 0, 0, 0.7],
    # )    

def generate_quadruped_from_config(config: dict, vis=visual):
    model = generate_root(config)
    
    terrain_mode = config.get("terrain", {}).get("mode", "plane")
    if terrain_mode == "uneven":
        add_uneven_terrain(model, config)
    else:
        add_terrain(model, config)
        
    add_tetrahedral_spine(model, config)
    add_actuation(model, config)
    add_sensors(model, config)
    generate_keyframe(model, config)

    if vis:
        add_target_visualization(model, config, target_distance_bl=15.0, body_length=GO1_HIP_TO_HIP_LENGTH, tolearance=0.02)
    
    path = config.get(
        "output_path", ROOT_PATH / "xmls" / "scene_tensegrity_quadruped.xml"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # 写入临时文件
    xml_string = model.to_xml_string()
    
    # 如果使用不平地形，处理文件引用
    if terrain_mode == "uneven":
        import re
        # 将所有带哈希的 PNG 文件名替换为简单文件名
        xml_string = re.sub(r'file="([^"]+)-[a-f0-9]{40}\.png"', r'file="\1.png"', xml_string)
    
    with open(path, "w") as file:
        file.write(xml_string)

def build_config_from_genes(genes: dict, individual_id: int, output_path: str | None=None) -> dict:
    cfg = copy.deepcopy(DEFAULT_CONFIG)

    # -------- spine 几何（你已有这些键） --------
    cfg["spine"]["num_segments"] = int(genes.get("num_segments", cfg["spine"]["num_segments"]))
    for k in ("segment_spacing","alpha","alpha_length","beta","beta_length"):
        if k in genes: cfg["spine"][k] = float(genes[k])

    # -------- helper：按优先级回退取值 --------
    def pick(*names, default=None):
        for n in names:
            if n in genes and genes[n] is not None:
                return float(genes[n])
        return default

    # == 全局（2 路） ==
    base_stiff = pick("stiffness", default=cfg["defaults"]["lateral_tendon"]["tendon"]["stiffness"])
    base_damp  = pick("damping",  default=cfg["defaults"]["lateral_tendon"]["tendon"]["damping"])

    lat_all_stiff  = pick("lateral_stiffness","lateral", default=base_stiff)
    lat_all_damp   = pick("lateral_damping","lateral_damp","lateral_d", default=base_damp)
    diag_all_stiff = pick("diagonal_stiffness","diagonal", default=base_stiff)
    diag_all_damp  = pick("diagonal_damping","diagonal_damp","diagonal_d", default=base_damp)

    # == 中层（4 路） ==
    # lateral：水平/垂直；diagonal：same(=a1b1&a2b2)/cross(=a1b2&a2b1)
    lat_hori_stiff = pick("lateral_hori_stiffness","lateral_hori", default=lat_all_stiff)
    lat_hori_damp  = pick("lateral_hori_damping","lateral_hori_damp", default=lat_all_damp)
    lat_vert_stiff = pick("lateral_verti_stiffness","lateral_verti","lateral_vert", default=lat_all_stiff)
    lat_vert_damp  = pick("lateral_verti_damping","lateral_vert_damping","lateral_vert_damp", default=lat_all_damp)

    diag_same_stiff  = pick("diag_same_stiffness","diag_same", default=diag_all_stiff)
    diag_same_damp   = pick("diag_same_damping","diag_same_damp", default=diag_all_damp)
    diag_cross_stiff = pick("diag_cross_stiffness","diag_cross", default=diag_all_stiff)
    diag_cross_damp  = pick("diag_cross_damping","diag_cross_damp", default=diag_all_damp)

    # == 最细（8 路） ==
    lat_top_stiff   = pick("lateral_top_stiffness","lateral_top", default=lat_vert_stiff)
    lat_top_damp    = pick("lateral_top_damping","lateral_top_damp", default=lat_vert_damp)
    lat_bottom_stiff= pick("lateral_bottom_stiffness","lateral_bottom","lateral_bot", default=lat_vert_stiff)
    lat_bottom_damp = pick("lateral_bottom_damping","lateral_bottom_damp","lateral_bot_damp", default=lat_vert_damp)
    lat_left_stiff  = pick("lateral_left_stiffness","lateral_left", default=lat_hori_stiff)
    lat_left_damp   = pick("lateral_left_damping","lateral_left_damp", default=lat_hori_damp)
    lat_right_stiff = pick("lateral_right_stiffness","lateral_right", default=lat_hori_stiff)
    lat_right_damp  = pick("lateral_right_damping","lateral_right_damp", default=lat_hori_damp)

    diag_a1b1_stiff = pick("diag_a1b1_stiffness","diag_a1b1", default=diag_same_stiff)
    diag_a1b1_damp  = pick("diag_a1b1_damping","diag_a1b1_damp", default=diag_same_damp)
    diag_a2b2_stiff = pick("diag_a2b2_stiffness","diag_a2b2", default=diag_same_stiff)
    diag_a2b2_damp  = pick("diag_a2b2_damping","diag_a2b2_damp", default=diag_same_damp)
    diag_a1b2_stiff = pick("diag_a1b2_stiffness","diag_a1b2", default=diag_cross_stiff)
    diag_a1b2_damp  = pick("diag_a1b2_damping","diag_a1b2_damp", default=diag_cross_damp)
    diag_a2b1_stiff = pick("diag_a2b1_stiffness","diag_a2b1", default=diag_cross_stiff)
    diag_a2b1_damp  = pick("diag_a2b1_damping","diag_a2b1_damp", default=diag_cross_damp)

    # -------- 写 defaults --------
    d = cfg["defaults"]

    common = {
        "frictionloss": d["lateral_tendon"]["tendon"]["frictionloss"],
        "width": d["lateral_tendon"]["tendon"]["width"],
    }

    # 保留原总类（与现有 XML 兼容）:contentReference[oaicite:0]{index=0}
    d["lateral_tendon"]["tendon"]["stiffness"]  = lat_all_stiff
    d["lateral_tendon"]["tendon"]["damping"]    = lat_all_damp
    d["diagonal_tendon"]["tendon"]["stiffness"] = diag_all_stiff
    d["diagonal_tendon"]["tendon"]["damping"]   = diag_all_damp

    # 4× lateral
    d["lateral_top"]    = {"tendon": {"stiffness": lat_top_stiff,    "damping": lat_top_damp, **common}}
    d["lateral_bottom"] = {"tendon": {"stiffness": lat_bottom_stiff, "damping": lat_bottom_damp, **common}}
    d["lateral_left"]   = {"tendon": {"stiffness": lat_left_stiff,   "damping": lat_left_damp, **common}}
    d["lateral_right"]  = {"tendon": {"stiffness": lat_right_stiff,  "damping": lat_right_damp, **common}}

    # 4× diagonal（用配对名防混淆）
    d["diag_a1b1"] = {"tendon": {"stiffness": diag_a1b1_stiff, "damping": diag_a1b1_damp, **common}}
    d["diag_a2b2"] = {"tendon": {"stiffness": diag_a2b2_stiff, "damping": diag_a2b2_damp, **common}}
    d["diag_a1b2"] = {"tendon": {"stiffness": diag_a1b2_stiff, "damping": diag_a1b2_damp, **common}}
    d["diag_a2b1"] = {"tendon": {"stiffness": diag_a2b1_stiff, "damping": diag_a2b1_damp, **common}}

    # 输出路径
    cfg["output_path"] = (ROOT_PATH / "xmls" / f"spine_only.xml") if output_path is None else Path(output_path)
    return cfg


def add_spine_only_sensors(model, config):
    sensor = model.sensor
    # 新增：末端 tip 的位置 / 姿态，用于算弯曲角度
    sensor.add(
        "framepos",
        objtype="site",
        objname="spine_tip",
        name="tip_position",
    )
    sensor.add(
        "framequat",
        objtype="site",
        objname="spine_tip",
        name="tip_orientation",
    )

        # ==== 新增：为每个 vertebra 添加 framepos 传感器 ====
    num_segments = config["spine"]["num_segments"]
    for i in range(num_segments):
        body_name = f"vertebrae_{i}"
        sensor.add(
            "framepos",
            name=f"vertebrae_{i}_pos",
            objtype="body",
            objname=body_name,
        )



def generate_spine_only_from_config(config: dict):
    cfg = copy.deepcopy(config)
    model = generate_root(cfg)
    add_terrain(model, cfg)          # 要不要地面随意
    add_tetrahedral_spine(model, cfg, with_legs=False)  # 下面我们加个 flag
    add_spine_only_sensors(model, cfg)
    add_target_visualization(model, cfg, target_distance_bl=15.0, body_length=GO1_HIP_TO_HIP_LENGTH, tolearance=0.02)
    # 写 xml
    path = cfg.get("output_path", ROOT_PATH / "spine_only.xml")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(model.to_xml_string())



if __name__ == "__main__":
    example_genes = {
        "lateral_hori_stiffness": 15000,
        "lateral_verti_stiffness": 15000,
        "diagonal_stiffness": 15000,
    }

    print("Building config...")
    config = build_config_from_genes(example_genes, individual_id=0)
    print(f"Output path: {config['output_path']}")
    
    print("Generating spine-only model...")
    saved_path = generate_spine_only_from_config(config)
    print(f"✓ File saved successfully at: {saved_path}")

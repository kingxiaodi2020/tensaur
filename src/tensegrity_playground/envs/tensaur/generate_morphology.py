from etils import epath
import numpy as np
from dm_control import mjcf

ROOT_PATH = epath.Path(__file__).parent
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
                "size": [0.005],
                "density": 800, #370.0,
                "rgba": [0.8, 0.6, 0.4, 1],
                "group": 0,
            },
        },
        "leg_geom": {
            "geom": {
                "type": "capsule",
                "size": [0.015],
                "density": 1370.0,
                "rgba": [0.8, 0.6, 0.4, 1],
                "group": 1,
                "conaffinity": 1,
            },
        },
        "lateral_tendon": {
            "tendon": {
                "stiffness": 1200,
                "damping": 1.0,
                "frictionloss": 0.05,
                "width": 0.002,
                "rgba": [1.0, 0.0, 0.0, 0.5],
            },
        },
        "diagonal_tendon": {
            "tendon": {
                "stiffness": 1200,
                "damping": 1.0,
                "frictionloss": 0.05,
                "width": 0.002,
                "rgba": [0.0, 0.0, 1.0, 0.5],
            },
        },
    },
    "leg_dimensions": {
        "hip_roll_length": 0.04,
        "hip_pitch_length": 0.08,
        "shin_length": 0.12,
        "foot_radius": 0.02,
    },
    "spine": {
        "num_segments": 8,
        "segment_spacing": 0.06,
        "initial_z": 0.25,
        "alpha": np.pi / 4,
        "alpha_length": 0.08,
        "beta": np.pi / 4,
        "beta_length": 0.06,
        "lateral_pretension": 0.98,
        "diagonal_pretension": 0.98,
    },
    "actuation": {
        "hip_roll": {"ctrlrange": [-0.785398, 0.785398]},
        "hip_pitch": {"ctrlrange": [-0.785398, 1.5708]},
        "knee": {"ctrlrange": [-1.57, 0.0]},
    },
    "legs": {"z_offset": -0.025},
    "keyframe": {"leg_qpos": [-0.3, 0.8, -1.2, 0.3, 0.8, -1.2]},
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
    from math import sin, cos, pi

    num_segments = config["spine"]["num_segments"]
    segment_spacing = config["spine"]["segment_spacing"]
    diagonal_pretension = config["spine"]["diagonal_pretension"]
    lateral_pretension = config["spine"]["lateral_pretension"]
    leg_z = config["legs"]["z_offset"]

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

    endpoints = {
        "a1": [-a1, 0.0, a2],
        "a2": [-a1, 0.0, -a2],
        "b1": [b1, b2, 0.0],
        "b2": [b1, -b2, 0.0],
    }

    for i in range(num_segments):
        name = f"vertebrae_{i}"
        pos = [-i * segment_spacing, 0, config["spine"]["initial_z"]]
        body = model.worldbody.add(
            "body", name=name, pos=pos, childclass="tq1"
        )
        if i == 0:
            body.add(
                "light",
                name="tracking",
                mode="trackcom",
                pos=[0, 0, 3],
                diffuse=[0.6] * 3,
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
        body.add(
            "site", name=f"com_vertebrae_{i}", pos=[0, 0, 0], size=[0.005]
        )
        for key, endpoint in endpoints.items():
            body.add(
                "geom",
                name=f"{name}_{key}_geom",
                fromto=[0, 0, 0, *endpoint],
                dclass="vertebra_geom",
            )
            body.add("site", name=f"{name}_{key}", pos=endpoint, size=[0.005])

    for i in range(num_segments - 1):
        for site in ["a1", "a2", "b1", "b2"]:
            tendon = model.tendon.add(
                "spatial",
                name=f"lat_{site}_{i}",
                dclass="lateral_tendon",
                springlength=[segment_spacing * lateral_pretension],
            )
            tendon.add("site", site=f"vertebrae_{i}_{site}")
            tendon.add("site", site=f"vertebrae_{i + 1}_{site}")

        for a, b in [("a1", "b1"), ("a2", "b2"), ("a1", "b2"), ("a2", "b1")]:
            site_1_pos = model.find("site", f"vertebrae_{i}_{a}").pos
            site_2_pos = model.find("site", f"vertebrae_{i + 1}_{b}").pos
            tendon_length = np.linalg.norm(site_1_pos - site_2_pos)

            tendon = model.tendon.add(
                "spatial",
                name=f"diag_{a}{b}_{i}",
                dclass="diagonal_tendon",
                # springlength=[tendon_length * diagonal_pretension],
            )
            tendon.add("site", site=f"vertebrae_{i}_{a}")
            tendon.add("site", site=f"vertebrae_{i + 1}_{b}")

    add_legs(model, config)

    return model


def add_legs(model, config):
    z = config["legs"]["z_offset"]
    leg_segment = config["spine"]["num_segments"] - 1
    for idx, segment in [(0, "front"), (leg_segment, "hind")]:
        name = f"vertebrae_{idx}"
        body = model.find("body", name)
        body.add(
            "geom",
            name=f"{segment}_shoulder",
            fromto=[0, -0.075, z, 0, 0.075, z],
            dclass="leg_geom",
        )
        body.add(
            "geom",
            name=f"{segment}_shoulder_attach",
            fromto=[0, 0, 0, 0, 0, z],
            dclass="leg_geom",
            size=[0.005],
        )
        #if idx == 0:
        #    body.add(
        #        "geom",
        #        name="head",
        #        type="ellipsoid",
        #        size=[0.08, 0.04, 0.02],
        #        pos=[0.10, 0, 0.0],
        #        density=1,
        #        rgba=[0.8, 0.6, 0.4, 1],
        #        group=2,
        #    )
        for side in ["fr", "fl"] if idx == 0 else ["rr", "rl"]:
            base = [0, -0.075, z] if side.endswith("r") else [0, 0.075, z]
            add_leg(body, side, base)


def add_leg(parent_body, prefix, base_pos):
    roll_len = DEFAULT_CONFIG["leg_dimensions"]["hip_roll_length"]
    pitch_len = DEFAULT_CONFIG["leg_dimensions"]["hip_pitch_length"]
    shin_len = DEFAULT_CONFIG["leg_dimensions"]["shin_length"]
    foot_r = DEFAULT_CONFIG["leg_dimensions"]["foot_radius"]

    act_cfg = DEFAULT_CONFIG["actuation"]
    hip_roll_range = act_cfg["hip_roll"]["ctrlrange"]
    hip_pitch_range = act_cfg["hip_pitch"]["ctrlrange"]
    knee_range = act_cfg["knee"]["ctrlrange"]

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
    )
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


def add_actuation(model, config):
    act_cfg = config["actuation"]

    for leg in ["fr", "fl", "rr", "rl"]:
        for joint_name in ["hip_roll", "hip_pitch", "knee"]:
            ctrlrange = act_cfg[joint_name]["ctrlrange"]

            model.actuator.add(
                "position",
                name=f"{leg}_{joint_name}",
                joint=f"{leg}_{joint_name}",
                ctrlrange=ctrlrange,
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


def generate_keyframe(model, config):
    num_segments = config["spine"]["num_segments"]
    spacing = config["spine"]["segment_spacing"]
    z = config["spine"]["initial_z"]
    leg_q = config["keyframe"]["leg_qpos"]

    qpos, ctrl = [], []
    for i in range(num_segments):
        qpos.extend([-i * spacing, 0, z, 1, 0, 0, 0])
        if i == 0 or i == num_segments - 1:
            qpos.extend(leg_q)
    ctrl = np.array(leg_q * 2)
    model.keyframe.add(
        "key", name="stable_pose", qpos=np.array(qpos), ctrl=ctrl
    )


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

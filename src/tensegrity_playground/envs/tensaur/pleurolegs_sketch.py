from etils import epath
import copy
import os
import re
from pathlib import Path

import numpy as np
from dm_control import mjcf

ROOT_PATH = epath.Path(__file__).parent

GO1_HIP_TO_HIP_LENGTH = 2 * 0.1881
VISUAL_DEBUG = True
TERRAIN_MODE = "plane"  # "plane" | "uneven"

LEG_CHAIN_SPECS = {
    "link_leg_1_L": {
        "foot_site": ("fr_foot_site", [0.0, -0.12, 0.0]),
        "segments": [
            {
                "inertial": {
                    "pos": [0.000385507, -0.0144502, -0.0248889],
                    "quat": [0.841913, 0.539448, 0.00388984, -0.0127757],
                    "mass": 0.281394,
                    "diaginertia": [0.000150974, 0.00014135, 0.000126484],
                },
                "joint": {"axis": [0.0, 0.0, -1.0], "range": [-1.5708, 1.5708], "damping": 0.1},
            },
            {
                "pos": [0.00025, -2.81e-05, -0.0235],
                "inertial": {
                    "pos": [0.000252758, -0.0790585, 0.0353242],
                    "quat": [0.998628, 0.0520833, 0.00501968, -0.00192733],
                    "mass": 0.352126,
                    "diaginertia": [0.000430099, 0.00034742, 0.000201667],
                },
                "joint": {"axis": [1.0, 0.0, 0.0], "range": [-0.785398, 1.5708], "damping": 0.1},
            },
            {
                "pos": [0.00025, -0.1082, 0.03375],
                "inertial": {
                    "pos": [4.3226e-05, -0.0184038, 0.0151445],
                    "quat": [0.708454, -0.0101092, -0.00124543, 0.705684],
                    "mass": 0.2193,
                    "diaginertia": [0.000107539, 9.65534e-05, 7.58652e-05],
                },
                "joint": {"axis": [0.0, 1.0, 0.0], "range": [-1.5708, 1.5708], "damping": 0.1},
            },
            {
                "pos": [-0.00035, -0.0235, 0.0],
                "inertial": {
                    "pos": [-1.69255e-05, -0.0734151, -9.70094e-08],
                    "quat": [0.499843, 0.500157, -0.499843, 0.500157],
                    "mass": 0.0555341,
                    "diaginertia": [0.000128424, 0.000120602, 1.16465e-05],
                },
                "joint": {"axis": [1.0, 0.0, 0.0], "range": [0.0, 2.0944], "damping": 0.1},
            },
        ],
    },
    "link_leg_1_R": {
        "foot_site": ("fl_foot_site", [0.0, 0.12, 0.0]),
        "segments": [
            {
                "inertial": {
                    "pos": [0.000385507, 0.0144502, -0.0248889],
                    "quat": [0.539448, 0.841913, 0.0127757, -0.00388984],
                    "mass": 0.281394,
                    "diaginertia": [0.000150974, 0.00014135, 0.000126484],
                },
                "joint": {"axis": [0.0, 0.0, 1.0], "range": [-1.5708, 1.5708], "damping": 0.1},
            },
            {
                "pos": [0.0, 2.81e-05, -0.0235],
                "inertial": {
                    "pos": [4.17974e-05, 0.0790384, 0.035368],
                    "quat": [0.998732, -0.050286, 0.00223043, 0.00109271],
                    "mass": 0.352297,
                    "diaginertia": [0.000431778, 0.000347272, 0.000200856],
                },
                "joint": {"axis": [-1.0, 0.0, 0.0], "range": [-0.785398, 1.5708], "damping": 0.1},
            },
            {
                "pos": [1e-06, 0.1082, 0.03375],
                "inertial": {
                    "pos": [0.000494663, 0.0184038, 0.0151445],
                    "quat": [0.689325, -0.00170409, -0.0104819, 0.724375],
                    "mass": 0.2193,
                    "diaginertia": [0.000107584, 9.65269e-05, 7.58821e-05],
                },
                "joint": {"axis": [0.0, 1.0, 0.0], "range": [-1.5708, 1.5708], "damping": 0.1},
            },
            {
                "pos": [-0.000251, 0.0235, 0.0],
                "inertial": {
                    "pos": [-1.69255e-05, 0.0734151, 9.70094e-08],
                    "quat": [0.500157, 0.499843, -0.500157, 0.499843],
                    "mass": 0.0555341,
                    "diaginertia": [0.000128424, 0.000120602, 1.16465e-05],
                },
                "joint": {"axis": [-1.0, 0.0, 0.0], "range": [-0.523599, 2.0944], "damping": 0.1},
            },
        ],
    },
    "link_leg_0_L": {
        "foot_site": ("rr_foot_site", [0.0, -0.12, 0.0]),
        "segments": [
            {
                "inertial": {
                    "pos": [-0.000385507, 0.0144271, -0.0248889],
                    "quat": [0.538889, 0.84227, -0.0128255, 0.00391209],
                    "mass": 0.281394,
                    "diaginertia": [0.000150977, 0.000141377, 0.000126461],
                },
                "joint": {"axis": [0.0, 0.0, -1.0], "range": [-1.5708, 1.5708], "damping": 0.1},
            },
            {
                "pos": [0.00025, 0.0, -0.0235],
                "inertial": {
                    "pos": [-7.80075e-05, -0.0950586, 0.035246],
                    "quat": [0.99536, 0.0961703, 0.00106114, -0.00288401],
                    "mass": 0.352182,
                    "diaginertia": [0.000457556, 0.000346982, 0.000228726],
                },
                "joint": {"axis": [1.0, 0.0, 0.0], "range": [-1.5708, 0.785398], "damping": 0.1},
            },
            {
                "pos": [-0.00025, -0.12475, 0.03375],
                "inertial": {
                    "pos": [-0.000502145, -0.0185309, 0.0151607],
                    "quat": [0.68881, 0.00221149, 0.0107882, 0.724858],
                    "mass": 0.216032,
                    "diaginertia": [0.000104563, 9.40795e-05, 7.48375e-05],
                },
                "joint": {"axis": [0.0, 1.0, 0.0], "range": [-1.5708, 1.5708], "damping": 0.1},
            },
            {
                "pos": [0.0, -0.0235, 2.81e-05],
                "inertial": {
                    "pos": [-2.37331e-05, -0.0706142, 0.00743435],
                    "quat": [0.49826, 0.501031, -0.501234, 0.499469],
                    "mass": 0.0535637,
                    "diaginertia": [0.00011299, 0.000106806, 1.29115e-05],
                },
                "joint": {"axis": [1.0, 0.0, 0.0], "range": [0.0, 2.0944], "damping": 0.1},
            },
        ],
    },
    "link_leg_0_R": {
        "foot_site": ("rl_foot_site", [0.0, 0.12, 0.0]),
        "segments": [
            {
                "inertial": {
                    "pos": [-0.000385507, -0.0144271, -0.0248889],
                    "quat": [0.84227, 0.538889, -0.00391209, 0.0128255],
                    "mass": 0.281394,
                    "diaginertia": [0.000150977, 0.000141377, 0.000126461],
                },
                "joint": {"axis": [0.0, 0.0, 1.0], "range": [-1.5708, 1.5708], "damping": 0.1},
            },
            {
                "pos": [0.00025, 0.0, -0.0235],
                "inertial": {
                    "pos": [4.17909e-05, 0.0950387, 0.0353633],
                    "quat": [0.995892, -0.0905152, 0.00220897, 0.00102721],
                    "mass": 0.352354,
                    "diaginertia": [0.000462018, 0.00034932, 0.000228194],
                },
                "joint": {"axis": [-1.0, 0.0, 0.0], "range": [-0.785398, 1.5708], "damping": 0.1},
            },
            {
                "pos": [0.0, 0.12475, 0.03375],
                "inertial": {
                    "pos": [-0.000502145, 0.0185309, 0.0151607],
                    "quat": [0.724858, 0.0107882, 0.00221149, 0.68881],
                    "mass": 0.216032,
                    "diaginertia": [0.000104563, 9.40795e-05, 7.48375e-05],
                },
                "joint": {"axis": [0.0, 1.0, 0.0], "range": [-1.5708, 1.5708], "damping": 0.1},
            },
            {
                "pos": [0.0, 0.0235, 2.81e-05],
                "inertial": {
                    "pos": [-2.37566e-05, 0.0706141, 0.00743449],
                    "quat": [0.501031, 0.49826, -0.49947, 0.501233],
                    "mass": 0.0535638,
                    "diaginertia": [0.00011299, 0.000106806, 1.29116e-05],
                },
                "joint": {"axis": [-1.0, 0.0, 0.0], "range": [0.0, 2.0944], "damping": 0.1},
            },
        ],
    },
}

DEFAULT_CONFIG = {
    "output_path": ROOT_PATH / "xmls" / "pleurolegs.xml",
    "sim": {
        "timestep": 0.002,
        "integrator": "implicitfast",
        "iterations": 100,
        "ls_iterations": 50,
    },
    "defaults": {
        "tq1": {
            "geom": {"condim": 3, "contype": 1, "conaffinity": 0},
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
                "group": 0,
            }
        },
        "leg_geom": {
            "geom": {
                "type": "capsule",
                "size": [0.02],
                "density": 1750.0,
                "rgba": [0.8, 0.6, 0.4, 1],
                "group": 1,
                "conaffinity": 0,
            }
        },
        "lateral_tendon": {
            "tendon": {
                "stiffness": 7000,
                "damping": 5,
                "frictionloss": 0.02,
                "width": 0.002,
                "rgba": [1.0, 0.0, 0.0, 0.5],
            }
        },
        "diagonal_tendon": {
            "tendon": {
                "stiffness": 7500,
                "damping": 5,
                "frictionloss": 0.02,
                "width": 0.002,
                "rgba": [0.0, 0.0, 1.0, 0.5],
            }
        },
    },
    "spine": {
        "num_segments": 3,
        "segment_spacing": 0.05,
        "initial_z": 0.6,
        "alpha": np.pi / 4,
        "alpha_length": 0.05,
        "beta": np.pi / 4,
        "beta_length": 0.08,
        "lateral_pretension": 0.90,
        "diagonal_pretension": 0.90,
        "enforce_fixed_length": True,
        "fixed_length": GO1_HIP_TO_HIP_LENGTH,
        "length_tolerance": 1e-6,
    },
    "legs": {
        "z_offset": -0.025,
        "mesh_dir": None,
        "mesh_rel_dir": "../../../../../PleurobotII",
        "export_relative_mesh_paths": True,
        "mesh_scale": [0.001, 0.001, 0.001],
        "collision": {
            "conaffinity": 0,
            "group": 2,
            "friction": [0.7, 0.0, 0.0],
            "solref": [-3300, -45],
            "fluidcoef": [0, 0, 0, 0, 0],
        },
        "visual": {"contype": 0, "conaffinity": 0, "group": 1},
    },
    "actuation": {"forcerange": [-5, 5]},
    "keyframe": {
        "leg_qpos": [0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.8],
    },
    "terrain": {
        "mode": TERRAIN_MODE,
        "hfield_png": "terrain_go1_11m_gradual2.png",
        "rx": 7.5,
        "ry": 4.0,
        "hz": 0.40,
        "base": 0.001,
        "center_x": 5.8,
        "friction": [1.0, 0.1, 0.01],
        "use_checker_material": True,
    },
}


def _iter_joint_names():
    for chain_name, chain_spec in LEG_CHAIN_SPECS.items():
        for idx, segment in enumerate(chain_spec["segments"]):
            yield f"joint_{chain_name}_{idx}", segment["joint"]["range"]


def generate_root(config):
    model = mjcf.RootElement()
    model.compiler.angle = "radian"
    model.option.timestep = config["sim"]["timestep"]
    model.option.integrator = config["sim"]["integrator"]
    model.option.iterations = config["sim"]["iterations"]
    model.option.ls_iterations = config["sim"]["ls_iterations"]

    main_class = model.default.add("default", dclass="tq1")
    for name, entry in config["defaults"].items():
        target = main_class if name == "tq1" else main_class.add("default", dclass=name)
        for elem_type in ("geom", "joint", "tendon", "position"):
            attrs = entry.get(elem_type)
            if attrs and hasattr(target, elem_type):
                element = getattr(target, elem_type)
                for key, value in attrs.items():
                    setattr(element, key, value)

    return model


def add_terrain(model, _config):
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
    model.worldbody.add("geom", name="floor", type="plane", size=[50, 50, 0.1], pos=[0, 0, 0], material="grid")


def add_uneven_terrain(model, config):
    tcfg = config.get("terrain", {})
    png_path = tcfg.get("hfield_png")
    if not png_path:
        raise ValueError("Heightfield PNG is required for uneven terrain.")

    p = epath.Path(png_path)
    if not p.is_absolute():
        alt = ROOT_PATH / "xmls" / png_path
        alt2 = ROOT_PATH / png_path
        p = alt if alt.exists() else alt2 if alt2.exists() else None
        if not p:
            raise FileNotFoundError(f"Heightfield PNG not found: {png_path}")

    rx = float(tcfg.get("rx", 7.5))
    ry = float(tcfg.get("ry", 1.5))
    hz = float(tcfg.get("hz", 0.30))
    base = float(tcfg.get("base", 0.005))
    center_x = float(tcfg.get("center_x", 4.8))
    friction = tcfg.get("friction", [1.0, 0.1, 0.01])
    use_mat = bool(tcfg.get("use_checker_material", True))

    model.asset.add("texture", name="skybox", type="skybox", builtin="gradient", rgb1=[0.4, 0.6, 0.8], rgb2=[0, 0, 0], width=800, height=800, mark="random", markrgb=[1, 1, 1])

    if use_mat:
        model.asset.add("texture", name="tex_ground", type="2d", builtin="checker", width=256, height=256, rgb1=[0.75, 0.75, 0.75], rgb2=[0.65, 0.65, 0.65])
        model.asset.add("material", name="mat_ground", texture="tex_ground", texrepeat=[30, 6], rgba=[1, 1, 1, 1])

    model.asset.add("hfield", name="uneven_terrain", file=str(p), size=[rx, ry, hz, base])

    kwargs = {"material": "mat_ground"} if use_mat else {}
    model.worldbody.add("geom", name="terrain", type="hfield", hfield="uneven_terrain", size=[rx, ry, hz], pos=[center_x, 0, 0], friction=friction, **kwargs)


def _resolve_mesh_root(config, output_path):
    legs_cfg = config.get("legs", {})
    output_dir = Path(output_path).resolve().parent

    candidates = []

    explicit = legs_cfg.get("mesh_dir")
    if explicit:
        explicit_path = Path(explicit).expanduser()
        if explicit_path.is_absolute():
            candidates.append(explicit_path)
        else:
            candidates.append((output_dir / explicit_path).resolve())
            candidates.append((ROOT_PATH / explicit_path).resolve())
            candidates.append((Path.cwd() / explicit_path).resolve())

    legacy_rel = legs_cfg.get("mesh_rel_dir")
    if legacy_rel:
        rel_path = Path(legacy_rel)
        candidates.append((output_dir / rel_path).resolve())
        candidates.append((ROOT_PATH / rel_path).resolve())
        candidates.append((Path.cwd() / rel_path).resolve())

    for root in (ROOT_PATH.resolve(), output_dir, Path.cwd().resolve()):
        for parent in (root, *root.parents):
            candidates.append(parent / "PleurobotII")

    seen = set()
    unique_candidates = []
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        unique_candidates.append(candidate)

    required_file = "link_leg_1_L_0_collision.stl"
    for candidate in unique_candidates:
        if candidate.is_dir() and (candidate / required_file).exists():
            return candidate

    checked = "\n".join(f"- {p}" for p in unique_candidates)
    raise FileNotFoundError(
        "Could not locate Pleurobot leg meshes. Checked:\n"
        f"{checked}\n"
        "Set config['legs']['mesh_dir'] to your PleurobotII folder."
    )


def _to_output_relative(path, output_dir):
    try:
        return Path(os.path.relpath(path, output_dir)).as_posix()
    except ValueError:
        return Path(path).as_posix()


def add_pleuro_leg_mesh_assets(model, config, output_path):
    mesh_root = _resolve_mesh_root(config, output_path)
    output_dir = Path(output_path).resolve().parent
    scale = config["legs"]["mesh_scale"]
    mesh_file_rewrites = {}

    for chain_name in LEG_CHAIN_SPECS:
        for idx in range(4):
            stem = f"{chain_name}_{idx}"
            collision_abs = (mesh_root / f"{stem}_collision.stl").resolve()
            visual_abs = (mesh_root / f"{stem}.stl").resolve()

            if not collision_abs.exists():
                raise FileNotFoundError(f"Missing mesh file: {collision_abs}")
            if not visual_abs.exists():
                raise FileNotFoundError(f"Missing mesh file: {visual_abs}")

            collision_file = collision_abs.as_posix()
            visual_file = visual_abs.as_posix()
            collision_name = f"mesh_{stem}_collision"
            visual_name = f"mesh_{stem}_visual"

            model.asset.add("mesh", name=collision_name, file=collision_file, scale=scale)
            model.asset.add("mesh", name=visual_name, file=visual_file, scale=scale)

            mesh_file_rewrites[collision_name] = _to_output_relative(collision_abs, output_dir)
            mesh_file_rewrites[visual_name] = _to_output_relative(visual_abs, output_dir)

    return mesh_file_rewrites


def _add_leg_chain(parent_body, base_pos, chain_name, chain_spec, config):
    collision_cfg = config["legs"]["collision"]
    visual_cfg = config["legs"]["visual"]

    current_body = parent_body.add("body", name=f"{chain_name}_0", pos=base_pos)

    for idx, segment in enumerate(chain_spec["segments"]):
        inertial = segment["inertial"]
        current_body.add(
            "inertial",
            pos=inertial["pos"],
            quat=inertial["quat"],
            mass=inertial["mass"],
            diaginertia=inertial["diaginertia"],
        )

        joint = segment["joint"]
        current_body.add(
            "joint",
            name=f"joint_{chain_name}_{idx}",
            pos=[0, 0, 0],
            axis=joint["axis"],
            limited=True,
            range=joint["range"],
            damping=joint["damping"],
        )

        current_body.add(
            "geom",
            name=f"{chain_name}_{idx}_collision",
            type="mesh",
            conaffinity=collision_cfg["conaffinity"],
            group=collision_cfg["group"],
            friction=collision_cfg["friction"],
            solref=collision_cfg["solref"],
            fluidcoef=collision_cfg["fluidcoef"],
            mesh=f"mesh_{chain_name}_{idx}_collision",
        )
        current_body.add(
            "geom",
            name=f"{chain_name}_{idx}_visual",
            type="mesh",
            contype=visual_cfg["contype"],
            conaffinity=visual_cfg["conaffinity"],
            group=visual_cfg["group"],
            mesh=f"mesh_{chain_name}_{idx}_visual",
        )

        if idx < len(chain_spec["segments"]) - 1:
            child_pos = chain_spec["segments"][idx + 1]["pos"]
            current_body = current_body.add("body", name=f"{chain_name}_{idx + 1}", pos=child_pos)

    site_name, site_pos = chain_spec["foot_site"]
    current_body.add("site", name=site_name, size=[0.01], pos=site_pos)


def add_pleuro_legs(model, config, front_offset=0.0, rear_offset=0.0):
    z = config["legs"]["z_offset"]
    front_idx = config["spine"]["num_segments"] - 1

    hind_body = model.find("body", "vertebrae_0")
    hind_x = front_offset
    hind_body.add("geom", name="hind_shoulder", fromto=[hind_x, -0.12675, z, hind_x, 0.12675, z], dclass="leg_geom")
    hind_body.add("geom", name="hind_shoulder_attach", fromto=[hind_x, 0, 0, hind_x, 0, z], dclass="leg_geom", size=[0.005])
    _add_leg_chain(hind_body, [hind_x, -0.12675, z], "link_leg_1_L", LEG_CHAIN_SPECS["link_leg_1_L"], config)
    _add_leg_chain(hind_body, [hind_x, 0.12675, z], "link_leg_1_R", LEG_CHAIN_SPECS["link_leg_1_R"], config)

    front_body = model.find("body", f"vertebrae_{front_idx}")
    front_x = rear_offset
    front_body.add("geom", name="front_shoulder", fromto=[front_x, -0.12675, z, front_x, 0.12675, z], dclass="leg_geom")
    front_body.add("geom", name="front_shoulder_attach", fromto=[front_x, 0, 0, front_x, 0, z], dclass="leg_geom", size=[0.005])
    _add_leg_chain(front_body, [front_x, -0.12675, z], "link_leg_0_L", LEG_CHAIN_SPECS["link_leg_0_L"], config)
    _add_leg_chain(front_body, [front_x, 0.12675, z], "link_leg_0_R", LEG_CHAIN_SPECS["link_leg_0_R"], config)


def add_tetrahedral_spine(model, config):
    from math import cos, sin

    num_segments = config["spine"]["num_segments"]
    if num_segments < 2:
        raise ValueError("num_segments must be >= 2 when using front and hind leg modules.")

    segment_spacing = config["spine"]["segment_spacing"]
    diagonal_pretension = config["spine"]["diagonal_pretension"]
    lateral_pretension = config["spine"]["lateral_pretension"]

    alpha = config["spine"]["alpha"]
    beta = config["spine"]["beta"]
    a1, a2 = cos(alpha) * config["spine"]["alpha_length"], sin(alpha) * config["spine"]["alpha_length"]
    b1, b2 = cos(beta) * config["spine"]["beta_length"], sin(beta) * config["spine"]["beta_length"]

    endpoints = {
        "a1": [-a1, 0.0, a2],
        "a2": [-a1, 0.0, -a2],
        "b1": [b1, b2, 0.0],
        "b2": [b1, -b2, 0.0],
    }

    initial_z = config["spine"]["initial_z"]
    enforce = config["spine"].get("enforce_fixed_length", False)
    fixed_len = config["spine"].get("fixed_length", GO1_HIP_TO_HIP_LENGTH)
    tol = config["spine"].get("length_tolerance", 1e-6)

    chain_len = max(0.0, (num_segments - 1) * segment_spacing)
    front_extra = 0.0
    rear_extra = 0.0

    if enforce:
        if chain_len - fixed_len > tol:
            raise ValueError(
                f"Spine length ({chain_len:.4f} m) with num_segments={num_segments}, "
                f"segment_spacing={segment_spacing:.4f} exceeds fixed_length={fixed_len:.4f} m."
            )
        extra_total = max(0.0, fixed_len - chain_len)
        front_extra = extra_total / 2.0
        rear_extra = extra_total / 2.0

    imu_x = 0.5 * (fixed_len - chain_len) if enforce else 0.0
    x_positions = [chain_len / 2.0 - i * segment_spacing for i in range(num_segments)]

    for i in range(num_segments):
        name = f"vertebrae_{i}"
        body = model.worldbody.add("body", name=name, pos=[x_positions[i], 0, initial_z], childclass="tq1")

        if i == 0:
            body.add("light", name="tracking", mode="trackcom", pos=[0, 0, 3], diffuse=[0.6, 0.6, 0.6])
            body.add("camera", name="track", pos=[0.846, -1.3, 0.316], xyaxes=[0.866, 0.500, 0.0, -0.171, 0.296, 0.940], mode="trackcom")
            body.add("camera", name="top", pos=[-1, 0, 1], xyaxes=[0, -1, 0, 0.7, 0, 0.7], mode="trackcom")
            body.add("camera", name="side", pos=[0, -1, 0.3], xyaxes=[1, 0, 0, 0, 1, 2], mode="trackcom")
            body.add("camera", name="back", pos=[-1, 0, 0.3], xyaxes=[0, -1, 0, 1, 0, 2], mode="trackcom")
            body.add("camera", name="overview", pos=[-1, 0, 2], xyaxes=[0, -1, 0, 0.5, 0, 0.5], mode="trackcom", fovy=70)

        body.add("joint", name=f"joint_{name}", type="free")
        if i == 0:
            body.add("site", name="com_vertebrae_1", pos=[imu_x, 0, 0], size=[0.02])

        for key, endpoint in endpoints.items():
            body.add("geom", name=f"{name}_{key}_geom", fromto=[0, 0, 0, *endpoint], dclass="vertebra_geom")
            body.add("site", name=f"{name}_{key}", pos=endpoint, size=[0.005])

    def world_site_pos(body_name, site_suffix):
        body = model.find("body", body_name)
        site = model.find("site", f"{body_name}_{site_suffix}")
        return np.array(body.pos, dtype=float) + np.array(site.pos, dtype=float)

    lateral_map = {"a1": "lateral_top", "a2": "lateral_bottom", "b1": "lateral_left", "b2": "lateral_right"}
    diag_map = {
        ("a1", "b1"): "diag_a1b1",
        ("a2", "b2"): "diag_a2b2",
        ("a1", "b2"): "diag_a1b2",
        ("a2", "b1"): "diag_a2b1",
    }

    for i in range(num_segments - 1):
        for site_name in ("a1", "a2", "b1", "b2"):
            p_a = world_site_pos(f"vertebrae_{i}", site_name)
            p_b = world_site_pos(f"vertebrae_{i + 1}", site_name)
            spring_len = float(np.linalg.norm(p_b - p_a) * lateral_pretension)
            tendon = model.tendon.add("spatial", name=f"lat_{site_name}_{i}", dclass=lateral_map[site_name], springlength=[spring_len])
            tendon.add("site", site=f"vertebrae_{i}_{site_name}")
            tendon.add("site", site=f"vertebrae_{i + 1}_{site_name}")

        for a_name, b_name in (("a1", "b1"), ("a2", "b2"), ("a1", "b2"), ("a2", "b1")):
            p_a = world_site_pos(f"vertebrae_{i}", a_name)
            p_b = world_site_pos(f"vertebrae_{i + 1}", b_name)
            spring_len = float(np.linalg.norm(p_b - p_a) * diagonal_pretension)
            tendon = model.tendon.add("spatial", name=f"diag_{a_name}{b_name}_{i}", dclass=diag_map[(a_name, b_name)], springlength=[spring_len])
            tendon.add("site", site=f"vertebrae_{i}_{a_name}")
            tendon.add("site", site=f"vertebrae_{i + 1}_{b_name}")

    if enforce and (front_extra + rear_extra) > tol:
        front_body = model.find("body", "vertebrae_0")
        rear_body = model.find("body", f"vertebrae_{num_segments - 1}")
        front_body.add("geom", name="front_spacer", type="capsule", fromto=[0, 0, 0, front_extra, 0, 0], dclass="vertebra_geom")
        rear_body.add("geom", name="rear_spacer", type="capsule", fromto=[0, 0, 0, -rear_extra, 0, 0], dclass="vertebra_geom")

    front_offset = front_extra if enforce and (front_extra + rear_extra) > tol else 0.0
    rear_offset = -rear_extra if enforce and (front_extra + rear_extra) > tol else 0.0
    add_pleuro_legs(model, config, front_offset=front_offset, rear_offset=rear_offset)


def add_actuation(model, config):
    forcerange = config["actuation"].get("forcerange", [-5, 5])
    for joint_name, ctrlrange in _iter_joint_names():
        model.actuator.add(
            "position",
            name=f"actuator_position_{joint_name}",
            joint=joint_name,
            ctrlrange=ctrlrange,
            forcerange=forcerange,
        )


def add_sensors(model, _config):
    sensor = model.sensor

    for joint_name, _ in _iter_joint_names():
        sensor.add("jointpos", name=f"jointpos_{joint_name}", joint=joint_name)
        sensor.add("jointvel", name=f"jointvel_{joint_name}", joint=joint_name)
        sensor.add("jointlimitfrc", name=f"jointlimitfrc_{joint_name}", joint=joint_name)

    sensor.add("gyro", name="gyro", site="com_vertebrae_1")
    sensor.add("velocimeter", name="local_linvel", site="com_vertebrae_1")
    sensor.add("accelerometer", name="accelerometer", site="com_vertebrae_1")
    sensor.add("framepos", name="position", objtype="site", objname="com_vertebrae_1")
    sensor.add("framezaxis", name="upvector", objtype="site", objname="com_vertebrae_1")
    sensor.add("framexaxis", name="forwardvector", objtype="site", objname="com_vertebrae_1")
    sensor.add("framelinvel", name="global_linvel", objtype="site", objname="com_vertebrae_1")
    sensor.add("frameangvel", name="global_angvel", objtype="site", objname="com_vertebrae_1")
    sensor.add("framequat", name="orientation", objtype="site", objname="com_vertebrae_1")

    for leg in ("fr", "fl", "rr", "rl"):
        sensor.add("framelinvel", name=f"{leg}_foot_global_linvel", objtype="site", objname=f"{leg}_foot_site")
        sensor.add("touch", name=f"{leg}_foot_touch", site=f"{leg}_foot_site")
        sensor.add("force", name=f"{leg}_foot_force", site=f"{leg}_foot_site")


def generate_keyframe(model, config):
    n = config["spine"]["num_segments"]
    spacing = config["spine"]["segment_spacing"]
    z = config["spine"]["initial_z"]
    leg_q = config["keyframe"]["leg_qpos"]

    if len(leg_q) != 8:
        raise ValueError("keyframe.leg_qpos must have 8 values (one end module: 2 legs x 4 joints).")

    chain_len = max(0.0, (n - 1) * spacing)

    qpos = []
    for i in range(n):
        x_i = chain_len / 2.0 - i * spacing
        qpos.extend([x_i, 0, z, 1, 0, 0, 0])
        if i == 0 or i == n - 1:
            qpos.extend(leg_q)

    ctrl = np.array(leg_q * 2)
    model.keyframe.add("key", name="stable_pose", qpos=np.array(qpos), ctrl=ctrl)


def add_target_visualization(model, config, tolerance=0.02):
    z = config["spine"]["initial_z"]
    for i, x in enumerate([0.0, 5, 10, 15, 20, 25, 30, 35, 40]):
        tag = f"target_marker{i}"
        model.worldbody.add("site", name=tag, type="sphere", size=[tolerance], pos=[x, 0.0, z - 0.02 if i == 0 else 0.3], rgba=[1, 0, 0, 0.7])


def generate_quadruped_from_config(config, vis=VISUAL_DEBUG):
    path = Path(config.get("output_path", ROOT_PATH / "xmls" / "scene_tensegrity_quadruped_pleurolegs.xml"))
    if not path.is_absolute():
        path = (ROOT_PATH / path).resolve()

    model = generate_root(config)
    mesh_file_rewrites = add_pleuro_leg_mesh_assets(model, config, output_path=path)

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
        add_target_visualization(model, config, tolerance=0.02)

    path.parent.mkdir(parents=True, exist_ok=True)

    xml_string = model.to_xml_string()
    if config.get("legs", {}).get("export_relative_mesh_paths", True):
        for mesh_name, rel_path in mesh_file_rewrites.items():
            pattern = rf'(<mesh[^>]*name="{re.escape(mesh_name)}"[^>]*file=")[^"]+("[^>]*/>)'
            xml_string = re.sub(pattern, rf'\1{rel_path}\2', xml_string, count=1)

    if terrain_mode == "uneven":
        xml_string = re.sub(r'file="([^"]+)-[a-f0-9]{40}\.png"', r'file="\1.png"', xml_string)

    with open(path, "w", encoding="utf-8") as file:
        file.write(xml_string)


def build_config_from_genes(genes, individual_id, output_path=None):
    cfg = copy.deepcopy(DEFAULT_CONFIG)

    cfg["spine"]["num_segments"] = int(genes.get("num_segments", cfg["spine"]["num_segments"]))
    for key in ("segment_spacing", "alpha", "alpha_length", "beta", "beta_length"):
        if key in genes:
            cfg["spine"][key] = float(genes[key])

    def pick(*names, default=None):
        for name in names:
            if name in genes and genes[name] is not None:
                return float(genes[name])
        return default

    base_stiff = pick("stiffness", default=cfg["defaults"]["lateral_tendon"]["tendon"]["stiffness"])
    base_damp = pick("damping", default=cfg["defaults"]["lateral_tendon"]["tendon"]["damping"])

    lat_all_stiff = pick("lateral_stiffness", "lateral", default=base_stiff)
    lat_all_damp = pick("lateral_damping", "lateral_damp", "lateral_d", default=base_damp)
    diag_all_stiff = pick("diagonal_stiffness", "diagonal", default=base_stiff)
    diag_all_damp = pick("diagonal_damping", "diagonal_damp", "diagonal_d", default=base_damp)

    lat_hori_stiff = pick("lateral_hori_stiffness", "lateral_hori", default=lat_all_stiff)
    lat_hori_damp = pick("lateral_hori_damping", "lateral_hori_damp", default=lat_all_damp)
    lat_vert_stiff = pick("lateral_verti_stiffness", "lateral_verti", "lateral_vert", default=lat_all_stiff)
    lat_vert_damp = pick("lateral_verti_damping", "lateral_vert_damping", "lateral_vert_damp", default=lat_all_damp)

    diag_same_stiff = pick("diag_same_stiffness", "diag_same", default=diag_all_stiff)
    diag_same_damp = pick("diag_same_damping", "diag_same_damp", default=diag_all_damp)
    diag_cross_stiff = pick("diag_cross_stiffness", "diag_cross", default=diag_all_stiff)
    diag_cross_damp = pick("diag_cross_damping", "diag_cross_damp", default=diag_all_damp)

    lat_top_stiff = pick("lateral_top_stiffness", "lateral_top", default=lat_vert_stiff)
    lat_top_damp = pick("lateral_top_damping", "lateral_top_damp", default=lat_vert_damp)
    lat_bottom_stiff = pick("lateral_bottom_stiffness", "lateral_bottom", "lateral_bot", default=lat_vert_stiff)
    lat_bottom_damp = pick("lateral_bottom_damping", "lateral_bottom_damp", "lateral_bot_damp", default=lat_vert_damp)
    lat_left_stiff = pick("lateral_left_stiffness", "lateral_left", default=lat_hori_stiff)
    lat_left_damp = pick("lateral_left_damping", "lateral_left_damp", default=lat_hori_damp)
    lat_right_stiff = pick("lateral_right_stiffness", "lateral_right", default=lat_hori_stiff)
    lat_right_damp = pick("lateral_right_damping", "lateral_right_damp", default=lat_hori_damp)

    diag_a1b1_stiff = pick("diag_a1b1_stiffness", "diag_a1b1", default=diag_same_stiff)
    diag_a1b1_damp = pick("diag_a1b1_damping", "diag_a1b1_damp", default=diag_same_damp)
    diag_a2b2_stiff = pick("diag_a2b2_stiffness", "diag_a2b2", default=diag_same_stiff)
    diag_a2b2_damp = pick("diag_a2b2_damping", "diag_a2b2_damp", default=diag_same_damp)
    diag_a1b2_stiff = pick("diag_a1b2_stiffness", "diag_a1b2", default=diag_cross_stiff)
    diag_a1b2_damp = pick("diag_a1b2_damping", "diag_a1b2_damp", default=diag_cross_damp)
    diag_a2b1_stiff = pick("diag_a2b1_stiffness", "diag_a2b1", default=diag_cross_stiff)
    diag_a2b1_damp = pick("diag_a2b1_damping", "diag_a2b1_damp", default=diag_cross_damp)

    defaults = cfg["defaults"]
    common = {
        "frictionloss": defaults["lateral_tendon"]["tendon"]["frictionloss"],
        "width": defaults["lateral_tendon"]["tendon"]["width"],
    }

    defaults["lateral_tendon"]["tendon"]["stiffness"] = lat_all_stiff
    defaults["lateral_tendon"]["tendon"]["damping"] = lat_all_damp
    defaults["diagonal_tendon"]["tendon"]["stiffness"] = diag_all_stiff
    defaults["diagonal_tendon"]["tendon"]["damping"] = diag_all_damp

    defaults["lateral_top"] = {"tendon": {"stiffness": lat_top_stiff, "damping": lat_top_damp, **common}}
    defaults["lateral_bottom"] = {"tendon": {"stiffness": lat_bottom_stiff, "damping": lat_bottom_damp, **common}}
    defaults["lateral_left"] = {"tendon": {"stiffness": lat_left_stiff, "damping": lat_left_damp, **common}}
    defaults["lateral_right"] = {"tendon": {"stiffness": lat_right_stiff, "damping": lat_right_damp, **common}}

    defaults["diag_a1b1"] = {"tendon": {"stiffness": diag_a1b1_stiff, "damping": diag_a1b1_damp, **common}}
    defaults["diag_a2b2"] = {"tendon": {"stiffness": diag_a2b2_stiff, "damping": diag_a2b2_damp, **common}}
    defaults["diag_a1b2"] = {"tendon": {"stiffness": diag_a1b2_stiff, "damping": diag_a1b2_damp, **common}}
    defaults["diag_a2b1"] = {"tendon": {"stiffness": diag_a2b1_stiff, "damping": diag_a2b1_damp, **common}}

    if "terrain_mode" in genes:
        cfg["terrain"]["mode"] = genes["terrain_mode"]

    cfg["output_path"] = ROOT_PATH / "xmls" / f"tensegrity_pleurolegs_{individual_id}.xml" if output_path is None else Path(output_path)
    print(f"[Info] Generated XML path: {cfg['output_path']}")
    return cfg


if __name__ == "__main__":
    example_genes = {
        "lateral_verti_stiffness": 3000,
        "lateral_hori_stiffness": 10000,
        "diagonal_stiffness": 10000,
    }

    config = build_config_from_genes(example_genes, individual_id=0)
    generate_quadruped_from_config(config)

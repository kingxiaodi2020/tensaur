from etils import epath
import numpy as np
from dm_control import mjcf
import copy
from pathlib import Path

ROOT_PATH = epath.Path(__file__).parent

# Reference body length based on go1 hip-to-hip distance
GO1_HIP_TO_HIP_LENGTH = 2 * 0.1881  # = 0.3762 m
visual = True  # for visual debug use
mode = "plane"  # "plane" | "uneven"

DEFAULT_CONFIG = {
    "output_path": ROOT_PATH / "xmls" / "scene_tensegrity_quadruped.xml",
    "sim": {
        "timestep": 0.002,
        "integrator": "implicitfast",
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
                "group": 0,
            },
        },
        "leg_geom": {
            "geom": {
                "type": "capsule",
                "size": [0.02],
                "density": 1750.0,
                "rgba": [0.8, 0.6, 0.4, 1],
                "group": 1,
                "conaffinity": 0,
            },
        },
        "lateral_tendon": {
            "tendon": {
                "stiffness": 5000,
                "damping": 5,
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
        "hip_roll_length": 0.05,
        "hip_pitch_length": 0.213,
        "shin_length": 0.213,
        "foot_radius": 0.023,
    },
    "spine": {
        "num_segments": 3,
        "segment_spacing": 0.1,
        "initial_z": 0.6,
        "alpha": np.pi / 4,
        "alpha_length": 0.08,
        "beta": np.pi / 4,
        "beta_length": 0.08,
        "lateral_pretension": 0.90,
        "diagonal_pretension": 0.90,
        "enforce_fixed_length": True,
        "fixed_length": GO1_HIP_TO_HIP_LENGTH,
        "length_tolerance": 1e-6,
    },
    "actuation": {
        "hip_roll": {"ctrlrange": [-0.863, 0.863], "forcerange": [-23.7*0.2, 23.7*0.2]},
        "hip_pitch": {"ctrlrange": [-0.686, 4.501], "forcerange": [-23.7*0.2, 23.7*0.2]},
        "knee": {"ctrlrange": [-2.818, -0.888], "forcerange": [-35.55*0.2, 35.55*0.2]},
    },
    "legs": {"z_offset": -0.025},
    "keyframe": {"leg_qpos": [-0, 0.9, -1.55, 0, 0.9, -1.55]},
    "terrain": {
        "mode": mode,
        "hfield_png": "terrain_go1_11m_gradual2.png",
        "rx": 7.5,
        "ry": 1.5,
        "hz": 0.40,
        "base": 0.001,
        "center_x": 5.8,
        "friction": [1.0, 0.1, 0.01],
        "use_checker_material": True,
    },
}


def build_config_with_tendon_params(genes: dict, individual_id: int = 0, output_path: str | None = None) -> dict:
    """
    Build configuration from genes with hierarchical tendon parameter support.
    
    Supports three levels of specificity:
    1. Global: stiffness, damping
    2. Category: lateral_stiffness, lateral_damping, diagonal_stiffness, diagonal_damping
    3. Individual: specific tendon names (e.g., lateral_top_stiffness, diag_a1b1_damping)
    
    Args:
        genes: Dictionary containing parameter values
        individual_id: Identifier for output file naming
        output_path: Optional custom output path
        
    Returns:
        Complete configuration dictionary
    """
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    
    # Helper function to pick value with priority fallback
    def pick(*names, default=None):
        for n in names:
            if n in genes and genes[n] is not None:
                return float(genes[n])
        return default
    
    # Update spine geometric parameters
    cfg["spine"]["num_segments"] = int(genes.get("num_segments", cfg["spine"]["num_segments"]))
    for k in ("segment_spacing", "alpha", "alpha_length", "beta", "beta_length"):
        if k in genes:
            cfg["spine"][k] = float(genes[k])
    
    # Global baseline values
    base_stiff = pick("stiffness", default=cfg["defaults"]["lateral_tendon"]["tendon"]["stiffness"])
    base_damp = pick("damping", default=cfg["defaults"]["lateral_tendon"]["tendon"]["damping"])
    
    # Category-level values
    lat_all_stiff = pick("lateral_stiffness", "lateral", default=base_stiff)
    lat_all_damp = pick("lateral_damping", "lateral_damp", "lateral_d", default=base_damp)
    diag_all_stiff = pick("diagonal_stiffness", "diagonal", default=base_stiff)
    diag_all_damp = pick("diagonal_damping", "diagonal_damp", "diagonal_d", default=base_damp)
    
    # Mid-level categorization
    lat_hori_stiff = pick("lateral_hori_stiffness", "lateral_hori", default=lat_all_stiff)
    lat_hori_damp = pick("lateral_hori_damping", "lateral_hori_damp", default=lat_all_damp)
    lat_vert_stiff = pick("lateral_verti_stiffness", "lateral_verti", "lateral_vert", default=lat_all_stiff)
    lat_vert_damp = pick("lateral_verti_damping", "lateral_vert_damping", "lateral_vert_damp", default=lat_all_damp)
    
    diag_same_stiff = pick("diag_same_stiffness", "diag_same", default=diag_all_stiff)
    diag_same_damp = pick("diag_same_damping", "diag_same_damp", default=diag_all_damp)
    diag_cross_stiff = pick("diag_cross_stiffness", "diag_cross", default=diag_all_stiff)
    diag_cross_damp = pick("diag_cross_damping", "diag_cross_damp", default=diag_all_damp)
    
    # Individual tendon level
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
    
    # Update defaults in config
    d = cfg["defaults"]
    
    # Base categories
    d["lateral_tendon"]["tendon"]["stiffness"] = lat_all_stiff
    d["lateral_tendon"]["tendon"]["damping"] = lat_all_damp
    d["diagonal_tendon"]["tendon"]["stiffness"] = diag_all_stiff
    d["diagonal_tendon"]["tendon"]["damping"] = diag_all_damp
    
    # Individual lateral tendons
    d["lateral_top"] = {"tendon": {"stiffness": lat_top_stiff, "damping": lat_top_damp}}
    d["lateral_bottom"] = {"tendon": {"stiffness": lat_bottom_stiff, "damping": lat_bottom_damp}}
    d["lateral_left"] = {"tendon": {"stiffness": lat_left_stiff, "damping": lat_left_damp}}
    d["lateral_right"] = {"tendon": {"stiffness": lat_right_stiff, "damping": lat_right_damp}}
    
    # Individual diagonal tendons
    d["diag_a1b1"] = {"tendon": {"stiffness": diag_a1b1_stiff, "damping": diag_a1b1_damp}}
    d["diag_a2b2"] = {"tendon": {"stiffness": diag_a2b2_stiff, "damping": diag_a2b2_damp}}
    d["diag_a1b2"] = {"tendon": {"stiffness": diag_a1b2_stiff, "damping": diag_a1b2_damp}}
    d["diag_a2b1"] = {"tendon": {"stiffness": diag_a2b1_stiff, "damping": diag_a2b1_damp}}
    
    # Set output path
    if output_path is not None:
        cfg["output_path"] = Path(output_path)
    else:
        xml_filename = f"tensegrity_{individual_id}.xml"
        cfg["output_path"] = ROOT_PATH / "xmls" / xml_filename
    
    return cfg


if __name__ == "__main__":
    # Example usage
    example_genes = {
        "num_segments": 3,
        "segment_spacing": 0.1,
        "lateral_stiffness": 5000,
        "lateral_damping": 5,
        "diagonal_stiffness": 3000,
        "diagonal_damping": 5,
    }
    
    config = build_config_with_tendon_params(example_genes, individual_id=1)
    print(f"Configuration generated for output: {config['output_path']}")
"""Generate Pleurobot tensegrity-spine XML from config / gene dict.

Architecture mirrors generate_go1_tendon.py:
  - DEFAULT_CONFIG         → baseline parameters
  - build_config_from_genes()  → override any subset via flat dict (GA / sweep)
  - generate_pleurobot_from_config()   → produce the XML

Spine is procedurally generated (free-body vertebrae + spatial tendons).
Legs / body meshes come from a template XML (PleurobotII/pleurobot_tensegrity_spine.xml).
"""
from __future__ import annotations

import copy
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT_PATH = Path(__file__).resolve().parent

# Pleurobot hip-to-hip fixed length
# front attach = body_01 root_x(0.2) + leg_offset(0.0635) = 0.2635
# rear  attach = body_06 root_x(0.71222) + leg_offset(0.06815) = 0.78037
PLEUROBOT_HIP_TO_HIP_LENGTH = 0.78037 - 0.2635  # = 0.51687 m

DEFAULT_CONFIG = {
    "template_xml": ROOT_PATH / "xmls" / "PleurobotII" / "pleurobot_tensegrity_spine.xml",
    "output_path": ROOT_PATH / "xmls" / "PleurobotII" / "pleurobot_tensegrity_spine.generated.xml",
    "sim": {
        "timestep": 0.001,
        "iterations": 1000,
    },
    "defaults": {
        "tq1": {
            "joint": {
                "type": "hinge",
                "limited": True,
                "damping": 0.5,
                "armature": 0.005,
                "frictionloss": 0.001,
            },
            "geom": {"condim": 3, "contype": 1, "conaffinity": 0},
        },
        "vertebra_geom": {
            "geom": {
                "type": "capsule",
                "size": [0.012],
                "density": 1200,
                "rgba": [0.8, 0.6, 0.4, 1],
                "group": 0,
            },
        },
        "lateral_tendon": {
            "tendon": {
                "stiffness": 1000,
                "damping": 5,
                "frictionloss": 0.02,
                "width": 0.002,
                "rgba": [1.0, 0.0, 0.0, 0.6],
            },
        },
        "diagonal_tendon": {
            "tendon": {
                "stiffness": 1000,
                "damping": 5,
                "frictionloss": 0.02,
                "width": 0.002,
                "rgba": [0.0, 0.0, 1.0, 0.6],
            },
        },
    },
    "spine": {
        "num_segments": 3,
        "segment_spacing": 0.1,
        "alpha": np.pi / 4,
        "alpha_length": 0.1,
        "beta": np.pi / 4,
        "beta_length": 0.08,
        "lateral_pretension": 0.90,
        "diagonal_pretension": 0.90,
        "enforce_fixed_length": True,
        "fixed_length": PLEUROBOT_HIP_TO_HIP_LENGTH,
        "length_tolerance": 1e-6,
    },
    "pleurobot_frame": {
        "body01_root_x": 0.2,
        "body06_root_x": 0.71222,
        "initial_height_z": 0.3,
        "front_attach_root_x": 0.2635,
        "rear_attach_root_x": 0.78037,
        "front_attach_body01_x": 0.0635,
        "rear_attach_body06_x": 0.06815,
        "com_site_body01_pos": [0.0635, 0.0, 0.0],
    },
    "actuation": {
        # Simplified position-only actuators (replaces template's pos/vel/tor triple)
        "gainprm": 10,
        "biasprm": [0, -10],
        "forcerange": [-5, 5],
        # Joint definitions: (short_name, joint_name, ctrlrange)
        # NOTE: ctrlrange is also synced to corresponding <joint range="..."> below.
        "joints": [
            ("pos_leg0_L0", "joint_leg_0_L_0", [-1.5708, 0.7854]),
            ("pos_leg0_L1", "joint_leg_0_L_1", [-0.5236, 1.5708]),
            ("pos_leg0_L2", "joint_leg_0_L_2", [-2.35619, 2.35619]),
            ("pos_leg0_L3", "joint_leg_0_L_3", [0.0, 2.26893]),
            ("pos_leg0_R0", "joint_leg_0_R_0", [-1.5708, 0.7854]),
            ("pos_leg0_R1", "joint_leg_0_R_1", [-0.5236, 1.5708]),
            ("pos_leg0_R2", "joint_leg_0_R_2", [-2.35619, 2.35619]),
            ("pos_leg0_R3", "joint_leg_0_R_3", [0.0, 2.26893]),
            ("pos_leg1_L0", "joint_leg_1_L_0", [-1.39626, 1.39626]),
            ("pos_leg1_L1", "joint_leg_1_L_1", [-0.5236, 1.5708]),
            ("pos_leg1_L2", "joint_leg_1_L_2", [-2.35619, 2.35619]),
            ("pos_leg1_L3", "joint_leg_1_L_3", [-0.2618, 1.91986]),
            ("pos_leg1_R0", "joint_leg_1_R_0", [-1.39626, 1.39626]),
            ("pos_leg1_R1", "joint_leg_1_R_1", [-0.5236, 1.5708]),
            ("pos_leg1_R2", "joint_leg_1_R_2", [-2.35619, 2.35619]),
            ("pos_leg1_R3", "joint_leg_1_R_3", [-0.2618, 1.91986]),
        ],
    },
    "collision": {
        # Temple-style 1-bit collision masks:
        #   tq1 default        -> contype=1, conaffinity=0
        #   vertebra_geom      -> contype=1, conaffinity=1  (vertebra-vertebra + vertebra-floor)
        #   floor              -> defaults (contype=1, conaffinity=1)
        #   foot (link_*_3)    -> conaffinity=0  (foot-floor only, no foot-foot)
        # This drastically cuts MJX broad-phase pairs vs. the previous
        # multi-bit scheme (contype=2/4, conaffinity=3/5/6).
        "enable_foot_foot_collision": False,
    },
    "keyframe": {
        "name": "stable_pose",
        "joint_overrides": {
            "joint_leg_0_L_1": 0.92,
            "joint_leg_0_R_1": 0.92,
            "joint_leg_1_L_1": 0.92,
            "joint_leg_1_R_1": 0.92,
            "joint_leg_0_L_3": 1.07,
            "joint_leg_0_R_3": 1.07,
            "joint_leg_1_L_3": 1.07,
            "joint_leg_1_R_3": 1.07,
        },
        "ctrl_vector": [
            0, 1, 0, 1,   # leg0_L: joints 0-3
            0, 1, 0, 1,   # leg0_R: joints 0-3
            0, 1, 0, 1,   # leg1_L: joints 0-3
            0, 1, 0, 1,   # leg1_R: joints 0-3
        ],
    },
}


# ---------------------------------------------------------------------------
# XML helpers
# ---------------------------------------------------------------------------

def _fmt(v: float) -> str:
    return f"{v:.5f}".rstrip("0").rstrip(".") if abs(v) >= 1e-12 else "0"


def _vec3(vals) -> str:
    return " ".join(_fmt(v) for v in vals)


def _child_by_name(parent: ET.Element, tag: str, name: str) -> ET.Element:
    node = parent.find(f"{tag}[@name='{name}']")
    if node is None:
        raise ValueError(f"Cannot find <{tag} name='{name}'>")
    return node


def _remove_children_by_pred(parent: ET.Element, pred) -> None:
    for child in list(parent):
        if pred(child):
            parent.remove(child)


def _remove_direct_sites_by_name(parent: ET.Element, site_name: str) -> None:
    for child in list(parent):
        if child.tag == "site" and child.attrib.get("name") == site_name:
            parent.remove(child)


def _distance(p, q) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p, q)))


# ---------------------------------------------------------------------------
# Vertebra body builder
# ---------------------------------------------------------------------------

def _add_vertebra_body(
    parent: ET.Element,
    name: str,
    pos_xyz,
    endpoints: dict[str, tuple[float, float, float]],
    spacer_name: str | None = None,
    spacer_dx: float | None = None,
) -> ET.Element:
    body = ET.SubElement(parent, "body", {
        "name": name,
        "childclass": "tq1",
        "pos": _vec3(pos_xyz),
        "quat": "0 0 0 1",
    })
    ET.SubElement(body, "joint", {"name": f"joint_{name}", "type": "free"})

    if spacer_name is not None and spacer_dx is not None:
        ET.SubElement(body, "geom", {
            "name": spacer_name,
            "class": "vertebra_geom",
            "fromto": f"0 0 0  {_fmt(spacer_dx)} 0 0",
        })

    for key in ("a1", "a2", "b1", "b2"):
        x, y, z = endpoints[key]
        ET.SubElement(body, "geom", {
            "name": f"{name}_{key}_geom",
            "class": "vertebra_geom",
            "fromto": f"0 0 0  {_vec3((x, y, z))}",
        })
        ET.SubElement(body, "site", {
            "name": f"{name}_{key}",
            "size": "0.005",
            "pos": _vec3((x, y, z)),
        })

    return body


# ---------------------------------------------------------------------------
# Tendon builder
# ---------------------------------------------------------------------------

def _append_spatial_tendon(
    tendon_root: ET.Element,
    name: str,
    klass: str,
    spring_length: float,
    site1: str,
    site2: str,
    stiffness: float,
    damping: float,
    rgba: tuple[float, ...] | None = None,
) -> None:
    attrs = {
        "name": name,
        "class": klass,
        "springlength": _fmt(spring_length),
        "stiffness": _fmt(stiffness),
        "damping": _fmt(damping),
    }
    if rgba is not None:
        attrs["rgba"] = " ".join(_fmt(v) for v in rgba)
    spatial = ET.SubElement(tendon_root, "spatial", attrs)
    ET.SubElement(spatial, "site", {"site": site1})
    ET.SubElement(spatial, "site", {"site": site2})


# ---------------------------------------------------------------------------
# Actuator name shortener (same as original)
# ---------------------------------------------------------------------------

def _short_actuator_name(long_name: str) -> str:
    m = re.fullmatch(
        r"actuator_(position|velocity|torque)_joint_leg_(\d+)_([LR])_(\d+)",
        long_name,
    )
    if m:
        mode_short = {"position": "pos", "velocity": "vel", "torque": "tor"}[m.group(1)]
        return f"{mode_short}_leg{m.group(2)}_{m.group(3)}{m.group(4)}"

    m2 = re.fullmatch(r"actuator_(position|velocity|torque)_joint_body_(\d+)", long_name)
    if m2:
        mode_short = {"position": "pos", "velocity": "vel", "torque": "tor"}[m2.group(1)]
        return f"{mode_short}_body{m2.group(2)}"

    return long_name


# ---------------------------------------------------------------------------
# Keyframe qpos collector
# ---------------------------------------------------------------------------

def _collect_default_qpos(worldbody: ET.Element, joint_overrides: dict[str, float] | None = None) -> list[float]:
    qpos: list[float] = []

    def dfs(body: ET.Element) -> None:
        for child in body:
            if child.tag == "joint":
                jname = child.attrib.get("name")
                jtype = child.attrib.get("type", "hinge")
                override_val = None if joint_overrides is None else joint_overrides.get(jname)
                if jtype == "free":
                    pos = [float(v) for v in body.attrib.get("pos", "0 0 0").split()]
                    quat = [float(v) for v in body.attrib.get("quat", "1 0 0 0").split()]
                    if len(pos) != 3:
                        pos = [0.0, 0.0, 0.0]
                    if len(quat) != 4:
                        quat = [1.0, 0.0, 0.0, 0.0]
                    qpos.extend(pos + quat)
                elif jtype == "ball":
                    quat = [float(v) for v in child.attrib.get("quat", "1 0 0 0").split()]
                    if len(quat) != 4:
                        quat = [1.0, 0.0, 0.0, 0.0]
                    qpos.extend(quat)
                else:
                    qpos.append(float(override_val) if override_val is not None else 0.0)
            if child.tag == "body":
                dfs(child)

    dfs(worldbody)
    return qpos


# ---------------------------------------------------------------------------
# Main generation pipeline
# ---------------------------------------------------------------------------

def generate_pleurobot_from_config(config: dict) -> None:
    spine = config["spine"]
    frame = config["pleurobot_frame"]
    defaults = config["defaults"]

    num_segments = spine["num_segments"]
    segment_spacing = spine["segment_spacing"]
    if num_segments < 2:
        raise ValueError("num_segments must be >= 2")
    if segment_spacing <= 0:
        raise ValueError("segment_spacing must be > 0")

    # Compute fixed-length layout
    fixed_length = frame["rear_attach_root_x"] - frame["front_attach_root_x"]
    chain_len = (num_segments - 1) * segment_spacing
    tol = spine.get("length_tolerance", 1e-6)

    enforce = spine.get("enforce_fixed_length", True)
    if enforce:
        if chain_len > fixed_length + tol:
            raise ValueError(
                f"Spine chain ({chain_len:.4f} m) exceeds hip-to-hip ({fixed_length:.4f} m). "
                "Reduce num_segments or segment_spacing."
            )
        front_extra = (fixed_length - chain_len) / 2.0
        rear_extra = front_extra
    else:
        front_extra = 0.0
        rear_extra = 0.0

    center_x = 0.5 * (frame["front_attach_root_x"] + frame["rear_attach_root_x"])
    x_positions = [center_x - chain_len / 2.0 + i * segment_spacing for i in range(num_segments)]

    # Endpoint geometry
    alpha = spine["alpha"]
    beta = spine["beta"]
    a_dx = math.cos(alpha) * spine["alpha_length"]
    a_dz = math.sin(alpha) * spine["alpha_length"]
    b_dx = -math.cos(beta) * spine["beta_length"]
    b_dy = math.sin(beta) * spine["beta_length"]

    endpoints = {
        "a1": (a_dx, 0.0, +a_dz),
        "a2": (a_dx, 0.0, -a_dz),
        "b1": (b_dx, +b_dy, 0.0),
        "b2": (b_dx, -b_dy, 0.0),
    }

    # Parse template
    template_xml = Path(config["template_xml"])
    tree = ET.parse(template_xml)
    root = tree.getroot()

   # Collision bitmask scheme (2-bit, disjoint layers):
    #   bit0 -> "link-floor" layer   (link meshes emit; floor responds)
    #   bit1 -> "vertebra"   layer   (vertebrae emit+respond; floor responds)
    # Resulting enabled pairs:
    #   vertebra <-> vertebra
    #   vertebra <-> floor
    #   link (incl. foot) <-> floor
    # Suppressed pairs:
    #   vertebra <-> link mesh   (different bits, no overlap)
    #   link <-> link            (link conaffinity=0, so no mutual response)
    #   foot <-> foot            (same as above)
    # ------------------------------------------------------------------
    tq1_default_geom = root.find("./default/default/default[@class='tq1']/geom")
    if tq1_default_geom is not None:
        tq1_default_geom.set("contype", "1")
        tq1_default_geom.set("conaffinity", "0")

    vg_default_geom = root.find(
        "./default/default/default[@class='tq1']/default[@class='vertebra_geom']/geom"
    )
    if vg_default_geom is not None:
        vg_default_geom.set("contype", "2")
        vg_default_geom.set("conaffinity", "2")

    foot_geom_pattern = re.compile(r"^link_leg_\d+_[LR]_3_collision$")
    enable_foot_foot = config.get("collision", {}).get("enable_foot_foot_collision", False)
    for g in root.iter("geom"):
        gname = g.attrib.get("name", "")
        if gname == "floor_collision":
            g.set("contype", "0")
            g.set("conaffinity", "3")
        elif foot_geom_pattern.fullmatch(gname):
            if enable_foot_foot:
                # Opt-in foot-foot: promote foot to its own bit (bit2).
                g.set("contype", "5")   # bit0 (floor) | bit2 (foot-foot)
                g.set("conaffinity", "4")  # only bit2
            else:
                # Foot-floor only: clear any multi-bit overrides.
                if "contype" in g.attrib:
                    del g.attrib["contype"]  # inherit tq1 contype=1
                g.set("conaffinity", "0")
                
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("No <worldbody> found in template XML")

    # Locate Pleurobot body hierarchy
    pleuro_root = _child_by_name(worldbody, "body", "Pleurobot3_v41")
    link_body_00 = _child_by_name(pleuro_root, "body", "link_body_00")
    link_body_01 = _child_by_name(link_body_00, "body", "link_body_01")
    link_body_06 = _child_by_name(pleuro_root, "body", "link_body_06")

    # Clean old vertebrae from all possible parents
    for parent in (link_body_01, link_body_06, worldbody):
        _remove_children_by_pred(parent, lambda e: e.tag == "body" and e.attrib.get("name", "").startswith("vertebrae_"))

    # Remove locked hinge joint from body_01
    _remove_children_by_pred(link_body_01, lambda e: e.tag == "joint")

    # Remove actuators/sensors referencing removed joint_body_00
    actuator_root = root.find("actuator")
    if actuator_root is not None:
        _remove_children_by_pred(actuator_root, lambda e: e.attrib.get("joint") == "joint_body_00")

    sensor_root = root.find("sensor")
    removed_body00_actuators = {
        "actuator_position_joint_body_00",
        "actuator_velocity_joint_body_00",
        "actuator_torque_joint_body_00",
    }
    if sensor_root is not None:
        _remove_children_by_pred(sensor_root, lambda e: (
            e.attrib.get("joint") == "joint_body_00"
            or e.attrib.get("objname") == "Pleurobot3_v41"
            or e.attrib.get("actuator") in removed_body00_actuators
        ))

    # Extract cameras/lights from Pleurobot root
    cameras_lights = [e for e in pleuro_root if e.tag in ("camera", "light")]
    for elem in cameras_lights:
        pleuro_root.remove(elem)

    # Remove old IMU site
    _remove_direct_sites_by_name(pleuro_root, "com_vertebrae_1")

    # Detach body subtrees
    pleuro_root.remove(link_body_00)
    pleuro_root.remove(link_body_06)
    worldbody.remove(pleuro_root)

    # ----------------------------------------------------------------
    # Update defaults in template
    # ----------------------------------------------------------------
    vg_default = root.find("./default/default/default[@class='tq1']/default[@class='vertebra_geom']/geom")
    if vg_default is not None:
        vg_cfg = defaults["vertebra_geom"]["geom"]
        vg_default.set("density", _fmt(vg_cfg["density"]))
        size_val = vg_cfg["size"]
        vg_default.set("size", _fmt(size_val[0] if isinstance(size_val, (list, tuple)) else size_val))

    lat_default = root.find("./default/default/default[@class='tq1']/default[@class='lateral_tendon']/tendon")
    diag_default = root.find("./default/default/default[@class='tq1']/default[@class='diagonal_tendon']/tendon")
    if lat_default is not None:
        lt = defaults["lateral_tendon"]["tendon"]
        lat_default.set("stiffness", _fmt(lt["stiffness"]))
        lat_default.set("damping", _fmt(lt["damping"]))
    if diag_default is not None:
        dt = defaults["diagonal_tendon"]["tendon"]
        diag_default.set("stiffness", _fmt(dt["stiffness"]))
        diag_default.set("damping", _fmt(dt["damping"]))

    # ----------------------------------------------------------------
    # Build vertebrae as free bodies in worldbody
    # ----------------------------------------------------------------
    x0_local = x_positions[0] - frame["body01_root_x"]
    front_spacer_dx = frame["front_attach_body01_x"] - x0_local

    xN_local = x_positions[-1] - frame["body06_root_x"]
    rear_spacer_dx = frame["rear_attach_body06_x"] - xN_local
    last_idx = num_segments - 1

    vertebra_bodies: dict[int, ET.Element] = {}
    for i in range(num_segments):
        spacer_name, spacer_dx = None, None
        if i == 0:
            spacer_name, spacer_dx = "front_spacer", front_spacer_dx
        elif i == last_idx:
            spacer_name, spacer_dx = "rear_spacer", rear_spacer_dx

        vertebra_bodies[i] = _add_vertebra_body(
            parent=worldbody,
            name=f"vertebrae_{i}",
            pos_xyz=(-x_positions[i], 0.0, frame["initial_height_z"]),
            endpoints=endpoints,
            spacer_name=spacer_name,
            spacer_dx=spacer_dx,
        )

    # Reparent front body to vertebrae_0
    link_body_00.set("pos", _vec3((-x_positions[0], 0.0, -0.00025)))
    vertebra_bodies[0].append(link_body_00)

    # Reparent rear body to vertebrae_N-1
    offset_rear_x = frame["body06_root_x"] - x_positions[-1]
    link_body_06.set("pos", _vec3((offset_rear_x, 0.0, -0.00025)))
    vertebra_bodies[last_idx].append(link_body_06)

    # Move cameras/lights to vertebrae_0
    for elem in cameras_lights:
        vertebra_bodies[0].append(elem)

    # Attach IMU site to link_body_01
    _remove_direct_sites_by_name(link_body_01, "com_vertebrae_1")
    _remove_direct_sites_by_name(link_body_01, "com_body_01")
    ET.SubElement(link_body_01, "site", {
        "name": "com_body_01",
        "size": "0.005",
        "pos": _vec3(frame["com_site_body01_pos"]),
        # MuJoCo quaternions are ordered as (w, x, y, z).
        # This is a 180 deg rotation around z, which flips x/y but keeps z up.
        "quat": "0 0 0 1",
    })

    # ----------------------------------------------------------------
    # Tendons: 8-way hierarchical stiffness from defaults
    # ----------------------------------------------------------------
    # Per-tendon-type stiffness/damping are read from extended defaults
    # (populated by build_config_from_genes).
    def _get_tendon_params(tendon_class: str) -> tuple[float, float]:
        """Return (stiffness, damping) for a tendon class."""
        entry = defaults.get(tendon_class, {}).get("tendon", {})
        fallback_class = "lateral_tendon" if "lat" in tendon_class else "diagonal_tendon"
        fb = defaults[fallback_class]["tendon"]
        return (
            entry.get("stiffness", fb["stiffness"]),
            entry.get("damping", fb["damping"]),
        )

    # Mapping from site name to tendon class
    lateral_class_map = {
        "a1": "lateral_top",
        "a2": "lateral_bottom",
        "b1": "lateral_left",
        "b2": "lateral_right",
    }
    diag_class_map = {
        ("a1", "b1"): "diag_a1b1",
        ("a2", "b2"): "diag_a2b2",
        ("a1", "b2"): "diag_a1b2",
        ("a2", "b1"): "diag_a2b1",
    }

    tendon_root = root.find("tendon")
    if tendon_root is None:
        tendon_root = ET.SubElement(root, "tendon")
    else:
        for child in list(tendon_root):
            tendon_root.remove(child)

    lateral_pretension = spine["lateral_pretension"]
    diagonal_pretension = spine["diagonal_pretension"]

    for i in range(num_segments - 1):
        dx = x_positions[i + 1] - x_positions[i]

        # 4x lateral tendons
        for site in ("a1", "a2", "b1", "b2"):
            p0 = endpoints[site]
            p1 = (endpoints[site][0] + dx, endpoints[site][1], endpoints[site][2])
            rest = _distance(p0, p1)
            cls_name = lateral_class_map[site]
            stiff, damp = _get_tendon_params(cls_name)
            lat_rgba = (0.0, 1.0, 0.0, 0.7) if site in ("b1", "b2") else None
            _append_spatial_tendon(
                tendon_root,
                name=f"lat_{site}_{i}",
                klass="lateral_tendon",
                spring_length=rest * lateral_pretension,
                site1=f"vertebrae_{i}_{site}",
                site2=f"vertebrae_{i+1}_{site}",
                stiffness=stiff,
                damping=damp,
                rgba=lat_rgba,
            )

        # 4x diagonal tendons
        for a, b in (("a1", "b1"), ("a2", "b2"), ("a1", "b2"), ("a2", "b1")):
            p0 = endpoints[a]
            p1 = (endpoints[b][0] + dx, endpoints[b][1], endpoints[b][2])
            rest = _distance(p0, p1)
            cls_name = diag_class_map[(a, b)]
            stiff, damp = _get_tendon_params(cls_name)
            _append_spatial_tendon(
                tendon_root,
                name=f"diag_{a}{b}_{i}",
                klass="diagonal_tendon",
                spring_length=rest * diagonal_pretension,
                site1=f"vertebrae_{i}_{a}",
                site2=f"vertebrae_{i+1}_{b}",
                stiffness=stiff,
                damping=damp,
            )

    # ----------------------------------------------------------------
    # Update sensor references to middle vertebra
    # ----------------------------------------------------------------
    mid_idx = num_segments // 2
    if sensor_root is not None:
        for s_node in sensor_root:
            if s_node.attrib.get("site") == "com_vertebrae_1":
                s_node.set("site", "com_body_01")
            if s_node.attrib.get("objtype") == "site" and s_node.attrib.get("objname") == "com_vertebrae_1":
                s_node.set("objname", "com_body_01")

        mid_name = f"vertebrae_{mid_idx}"
        # Remove vertebra body-velocity sensors (unused in current env pipeline).
        _remove_children_by_pred(
            sensor_root,
            lambda e: (
                e.tag in ("framelinvel", "frameangvel")
                and e.attrib.get("objtype") == "body"
                and e.attrib.get("objname", "").startswith("vertebrae_")
            ),
        )

        for s_node in sensor_root:
            if s_node.attrib.get("objname", "").startswith("vertebrae_"):
                s_node.set("objname", mid_name)

        # Rebuild per-vertebra pose sensors (one framepos + one framequat per vertebra).
        _remove_children_by_pred(
            sensor_root,
            lambda e: (
                e.tag in ("framepos", "framequat")
                and e.attrib.get("objtype") == "body"
                and e.attrib.get("objname", "").startswith("vertebrae_")
            ),
        )
        for i in range(num_segments):
            vname = f"vertebrae_{i}"
            ET.SubElement(sensor_root, "framepos", {
                "name": f"framepos_{vname}",
                "objtype": "body",
                "objname": vname,
            })
            ET.SubElement(sensor_root, "framequat", {
                "name": f"framequat_{vname}",
                "objtype": "body",
                "objname": vname,
            })

    # ----------------------------------------------------------------
    # Sync template joint ranges + rebuild simplified position actuators
    # ----------------------------------------------------------------
    act_cfg = config["actuation"]

    # Keep <joint range> consistent with configured actuator ctrlrange.
    joint_nodes_by_name = {
        j.attrib.get("name", ""): j
        for j in root.findall(".//joint")
        if j.attrib.get("name")
    }
    for _, joint_name, ctrlrange in act_cfg["joints"]:
        jnode = joint_nodes_by_name.get(joint_name)
        if jnode is None:
            raise ValueError(f"Joint not found in template XML: {joint_name}")
        jnode.set("limited", "true")
        jnode.set("range", f"{_fmt(ctrlrange[0])} {_fmt(ctrlrange[1])}")

    if actuator_root is not None:
        # Remove all existing actuators
        for child in list(actuator_root):
            actuator_root.remove(child)

        # Add simplified position-only actuators
        for act_name, joint_name, ctrlrange in act_cfg["joints"]:
            ET.SubElement(actuator_root, "general", {
                "name": act_name,
                "joint": joint_name,
                "ctrllimited": "true",
                "ctrlrange": f"{_fmt(ctrlrange[0])} {_fmt(ctrlrange[1])}",
                "forcelimited": "true",
                "forcerange": f"{_fmt(act_cfg['forcerange'][0])} {_fmt(act_cfg['forcerange'][1])}",
                "biastype": "affine",
                "gainprm": _fmt(act_cfg["gainprm"]),
                "biasprm": f"0 {_fmt(-act_cfg['gainprm'])}",
            })

    # Rebuild actuatorfrc sensors to match new actuator names
    if sensor_root is not None:
        # Remove old actuatorfrc sensors
        _remove_children_by_pred(sensor_root, lambda e: e.tag == "actuatorfrc")
        # Add new ones matching simplified actuator names
        for act_name, joint_name, _ in act_cfg["joints"]:
            ET.SubElement(sensor_root, "actuatorfrc", {
                "actuator": act_name,
                "name": f"actuatorfrc_position_{joint_name}",
            })

    # ----------------------------------------------------------------
    # Keyframe
    # ----------------------------------------------------------------
    kf_cfg = config["keyframe"]
    if actuator_root is not None:
        num_actuators = sum(1 for a in actuator_root if a.tag == "general")

        ctrl_values = kf_cfg.get("ctrl_vector")
        if ctrl_values is not None:
            if len(ctrl_values) != num_actuators:
                raise ValueError(
                    f"ctrl_vector length {len(ctrl_values)} != actuator count {num_actuators}"
                )
            ctrl_values = [float(v) for v in ctrl_values]
        else:
            ctrl_values = [0.0] * num_actuators

        keyframe_root = root.find("keyframe")
        if keyframe_root is None:
            keyframe_root = ET.SubElement(root, "keyframe")

        kf_name = kf_cfg.get("name", "stable_pose")
        for key_node in list(keyframe_root.findall("key")):
            if key_node.attrib.get("name") == kf_name:
                keyframe_root.remove(key_node)

        qpos_defaults = _collect_default_qpos(worldbody, kf_cfg.get("joint_overrides"))

        ET.SubElement(keyframe_root, "key", {
            "name": kf_name,
            "qpos": " ".join(_fmt(v) for v in qpos_defaults),
            "ctrl": " ".join(_fmt(v) for v in ctrl_values),
        })

    # ----------------------------------------------------------------
    # Write output
    # ----------------------------------------------------------------
    try:
        ET.indent(tree, space="  ")
    except Exception:
        pass

    output_path = Path(config["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_path, encoding="utf-8", xml_declaration=False)

    print(f"[OK] Generated: {output_path}")
    print(f"[INFO] fixed_length: {fixed_length:.5f} m, chain_len: {chain_len:.5f} m")
    print(f"[INFO] front_extra/rear_extra: {front_extra:.5f}/{rear_extra:.5f} m")
    print(f"[INFO] num_segments: {num_segments}, segment_spacing: {segment_spacing}")


# ---------------------------------------------------------------------------
# Gene-based config builder (mirrors generate_go1_tendon.py)
# ---------------------------------------------------------------------------

def build_config_from_genes(genes: dict, individual_id: int, output_path: str | None = None) -> dict:
    cfg = copy.deepcopy(DEFAULT_CONFIG)

    # -------- spine geometry --------
    cfg["spine"]["num_segments"] = int(genes.get("num_segments", cfg["spine"]["num_segments"]))
    for k in ("segment_spacing", "alpha", "alpha_length", "beta", "beta_length"):
        if k in genes:
            cfg["spine"][k] = float(genes[k])

    # Pretension
    for k in ("lateral_pretension", "diagonal_pretension"):
        if k in genes:
            cfg["spine"][k] = float(genes[k])

    # -------- helper: pick with fallback --------
    def pick(*names, default=None):
        for n in names:
            if n in genes and genes[n] is not None:
                return float(genes[n])
        return default

    # == Global (2-way) ==
    base_stiff = pick("stiffness", default=cfg["defaults"]["lateral_tendon"]["tendon"]["stiffness"])
    base_damp = pick("damping", default=cfg["defaults"]["lateral_tendon"]["tendon"]["damping"])

    lat_all_stiff = pick("lateral_stiffness", "lateral", default=base_stiff)
    lat_all_damp = pick("lateral_damping", "lateral_damp", "lateral_d", default=base_damp)
    diag_all_stiff = pick("diagonal_stiffness", "diagonal", default=base_stiff)
    diag_all_damp = pick("diagonal_damping", "diagonal_damp", "diagonal_d", default=base_damp)

    # == Mid-level (4-way) ==
    lat_hori_stiff = pick("lateral_hori_stiffness", "lateral_hori", default=lat_all_stiff)
    lat_hori_damp = pick("lateral_hori_damping", "lateral_hori_damp", default=lat_all_damp)
    lat_vert_stiff = pick("lateral_verti_stiffness", "lateral_verti", "lateral_vert", default=lat_all_stiff)
    lat_vert_damp = pick("lateral_verti_damping", "lateral_vert_damping", "lateral_vert_damp", default=lat_all_damp)

    diag_same_stiff = pick("diag_same_stiffness", "diag_same", default=diag_all_stiff)
    diag_same_damp = pick("diag_same_damping", "diag_same_damp", default=diag_all_damp)
    diag_cross_stiff = pick("diag_cross_stiffness", "diag_cross", default=diag_all_stiff)
    diag_cross_damp = pick("diag_cross_damping", "diag_cross_damp", default=diag_all_damp)

    # == Finest (8-way) ==
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

    # -------- Write defaults --------
    d = cfg["defaults"]

    common = {
        "frictionloss": d["lateral_tendon"]["tendon"]["frictionloss"],
        "width": d["lateral_tendon"]["tendon"]["width"],
    }

    # Update top-level tendon defaults
    d["lateral_tendon"]["tendon"]["stiffness"] = lat_all_stiff
    d["lateral_tendon"]["tendon"]["damping"] = lat_all_damp
    d["diagonal_tendon"]["tendon"]["stiffness"] = diag_all_stiff
    d["diagonal_tendon"]["tendon"]["damping"] = diag_all_damp

    # 4x lateral (per direction)
    d["lateral_top"] = {"tendon": {"stiffness": lat_top_stiff, "damping": lat_top_damp, **common}}
    d["lateral_bottom"] = {"tendon": {"stiffness": lat_bottom_stiff, "damping": lat_bottom_damp, **common}}
    d["lateral_left"] = {"tendon": {"stiffness": lat_left_stiff, "damping": lat_left_damp, **common}}
    d["lateral_right"] = {"tendon": {"stiffness": lat_right_stiff, "damping": lat_right_damp, **common}}

    # 4x diagonal (per pair)
    d["diag_a1b1"] = {"tendon": {"stiffness": diag_a1b1_stiff, "damping": diag_a1b1_damp, **common}}
    d["diag_a2b2"] = {"tendon": {"stiffness": diag_a2b2_stiff, "damping": diag_a2b2_damp, **common}}
    d["diag_a1b2"] = {"tendon": {"stiffness": diag_a1b2_stiff, "damping": diag_a1b2_damp, **common}}
    d["diag_a2b1"] = {"tendon": {"stiffness": diag_a2b1_stiff, "damping": diag_a2b1_damp, **common}}

    # Vertebra geom overrides
    if "vertebra_density" in genes:
        d["vertebra_geom"]["geom"]["density"] = float(genes["vertebra_density"])
    if "vertebra_size" in genes:
        d["vertebra_geom"]["geom"]["size"] = [float(genes["vertebra_size"])]

    # Output path
    if output_path is not None:
        cfg["output_path"] = Path(output_path)
    else:
        cfg["output_path"] = ROOT_PATH / "xmls" / "PleurobotII" / f"pleurobot_tensegrity_{individual_id}.xml"

    print(f"[Info] Generated XML path: {cfg['output_path']}")
    return cfg


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Generate Pleurobot tensegrity-spine XML")
    p.add_argument("--template", type=str, default=None, help="Template XML path")
    p.add_argument("--output", type=str, default=None, help="Output XML path")
    p.add_argument("--num-segments", type=int, default=None)
    p.add_argument("--segment-spacing", type=float, default=None)
    p.add_argument("--alpha-length", type=float, default=None)
    p.add_argument("--beta-length", type=float, default=None)
    p.add_argument("--lateral-verti-stiffness", type=float, default=None)
    p.add_argument("--lateral-hori-stiffness", type=float, default=None)
    p.add_argument("--diagonal-stiffness", type=float, default=None)
    p.add_argument("--tendon-damping", type=float, default=None)
    p.add_argument("--lateral-pretension", type=float, default=None)
    p.add_argument("--diagonal-pretension", type=float, default=None)

    args = p.parse_args()

    # Build genes dict from CLI args
    genes = {}
    if args.num_segments is not None:
        genes["num_segments"] = args.num_segments
    if args.segment_spacing is not None:
        genes["segment_spacing"] = args.segment_spacing
    if args.alpha_length is not None:
        genes["alpha_length"] = args.alpha_length
    if args.beta_length is not None:
        genes["beta_length"] = args.beta_length
    if args.lateral_verti_stiffness is not None:
        genes["lateral_verti_stiffness"] = args.lateral_verti_stiffness
    if args.lateral_hori_stiffness is not None:
        genes["lateral_hori_stiffness"] = args.lateral_hori_stiffness
    if args.diagonal_stiffness is not None:
        genes["diagonal_stiffness"] = args.diagonal_stiffness
    if args.tendon_damping is not None:
        genes["damping"] = args.tendon_damping
    if args.lateral_pretension is not None:
        genes["lateral_pretension"] = args.lateral_pretension
    if args.diagonal_pretension is not None:
        genes["diagonal_pretension"] = args.diagonal_pretension

    config = build_config_from_genes(genes, individual_id=0, output_path=args.output)

    if args.template is not None:
        config["template_xml"] = Path(args.template)
    if args.output is not None:
        config["output_path"] = Path(args.output)

    generate_pleurobot_from_config(config)

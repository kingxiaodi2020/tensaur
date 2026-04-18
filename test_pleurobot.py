"""Generate Pleurobot tensegrity-spine XML with customizable spine parameters.

This script uses an existing Pleurobot+spine XML as template and rewrites:
- vertebra bodies (welded ends + free middle bodies)
- spacer rods at both ends
- spatial tendons between adjacent vertebrae
- key spine site/sensor references
- tendon stiffness/damping defaults

Usage (example):
    python src/tensegrity_playground/envs/tensaur/generate_pleurobot_tensegrity_spine.py \
      --template PleurobotII/pleurobot_tensegrity_spine.xml \
      --output PleurobotII/pleurobot_tensegrity_spine.generated.xml \
      --num-segments 3 \
      --segment-spacing 0.07 \
      --alpha-length 0.10 \
      --beta-length 0.08 \
    --lateral-verti-stiffness 3000 \
    --lateral-hori-stiffness 10000 \
      --diagonal-stiffness 5000
"""

from __future__ import annotations

import argparse
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SpineParams:
    num_segments: int = 4
    segment_spacing: float = 0.1

    alpha_length: float = 0.05
    beta_length: float = 0.08

    lateral_verti_stiffness: float = 100.0
    lateral_hori_stiffness: float = 100.0
    diagonal_stiffness: float = 100.0
    
    alpha: float = math.pi / 4
    beta: float = math.pi / 4
    
    lateral_pretension: float = 0.90
    diagonal_pretension: float = 0.90

    tendon_damping: float = 5.0


@dataclass
class PleurobotFrame:
    body01_root_x: float = 0.2
    body06_root_x: float = 0.71222
    # 统一初始高度（非命令行参数）：同时用于 Pleurobot 根节点与自由脊柱段
    initial_height_z: float = 0.3
    # 脊柱几何密度（非命令行参数）：控制 vertebra_geom 默认密度
    vertebra_density: float = 1200.0
    # 脊柱胶囊半径（非命令行参数）：控制 <default class="vertebra_geom"><geom size="...">
    vertebra_capsule_size: float = 0.012
    # 关键帧控制向量（非命令行参数）：长度需等于 actuator 数量。
    # 若为 None，则 keyframe ctrl 全部为 0。
    keyframe_ctrl_vector: list[float] | None = field(
        default_factory=lambda: [
            0, 0, 0,  # leg0_L0: pos vel tor
            1, 0, 0,  # leg0_L1
            0, 0, 0,  # leg0_L2
            1, 0, 0,  # leg0_L3

            0, 0, 0,  # leg0_R0
            1, 0, 0,  # leg0_R1
            0, 0, 0,  # leg0_R2
            1, 0, 0,  # leg0_R3

            0, 0, 0,  # leg1_L0
            1, 0, 0,  # leg1_L1
            0, 0, 0,  # leg1_L2
            1, 0, 0,  # leg1_L3

            0, 0, 0,  # leg1_R0
            1, 0, 0,  # leg1_R1
            0, 0, 0,  # leg1_R2
            1, 0, 0,  # leg1_R3
        ]
    )
    # 关键帧名称（非命令行参数）
    keyframe_name: str = "stable_pose"
    # 关键帧关节覆盖值（非命令行参数）：按关节 name 指定初始 qpos
    keyframe_joint_overrides: dict[str, float] = field(
        default_factory=lambda: {
            # 四条腿对称弯曲站姿：hip_pitch 约 0.92 rad，knee 约 1.07 rad
            "joint_leg_0_L_1": 0.92,
            "joint_leg_0_R_1": 0.92,
            "joint_leg_1_L_1": 0.92,
            "joint_leg_1_R_1": 0.92,

            "joint_leg_0_L_3": 1.07,
            "joint_leg_0_R_3": 1.07,
            "joint_leg_1_L_3": 1.07,
            "joint_leg_1_R_3": 1.07,
        }
    )

    front_attach_root_x: float = 0.2635
    rear_attach_root_x: float = 0.78037

    front_attach_body01_x: float = 0.0635
    rear_attach_body06_x: float = 0.06815
    # IMU/观测参考点在 link_body_01 局部坐标系下的位置
    com_site_body01_pos: tuple[float, float, float] = (0.0635, 0.0, 0.0)


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


def _fmt(v: float) -> str:
    return f"{v:.5f}".rstrip("0").rstrip(".") if abs(v) >= 1e-12 else "0"


def _vec3(vals) -> str:
    return " ".join(_fmt(v) for v in vals)


def _add_vertebra_body(
    parent: ET.Element,
    name: str,
    pos_xyz,
    endpoints: dict[str, tuple[float, float, float]],
    free_joint: bool,
    spacer_name: str | None = None,
    spacer_dx: float | None = None,
) -> ET.Element:
    body_attrs = {
        "name": name,
        "childclass": "tq1",
        "pos": _vec3(pos_xyz),
    }
    if free_joint:
        # Keep orientation explicit in XML for readability/consistency.
        body_attrs["quat"] = "0 0 0 1"

    body = ET.SubElement(parent, "body", body_attrs)

    if free_joint:
        ET.SubElement(body, "joint", {"name": f"joint_{name}", "type": "free"})

    if spacer_name is not None and spacer_dx is not None:
        ET.SubElement(
            body,
            "geom",
            {
                "name": spacer_name,
                "class": "vertebra_geom",
                "fromto": f"0 0 0  {_fmt(spacer_dx)} 0 0",
            },
        )

    for key in ("a1", "a2", "b1", "b2"):
        x, y, z = endpoints[key]
        ET.SubElement(
            body,
            "geom",
            {
                "name": f"{name}_{key}_geom",
                "class": "vertebra_geom",
                "fromto": f"0 0 0  {_vec3((x, y, z))}",
            },
        )
        ET.SubElement(
            body,
            "site",
            {
                "name": f"{name}_{key}",
                "size": "0.005",
                "pos": _vec3((x, y, z)),
            },
        )

    return body


def _distance(p, q) -> float:
    return math.sqrt((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 + (p[2] - q[2]) ** 2)


def _append_spatial_tendon_with_physics(
    tendon_root: ET.Element,
    name: str,
    klass: str,
    spring_length: float,
    site1: str,
    site2: str,
    stiffness: float,
    damping: float,
    rgba: tuple[float, float, float, float] | None = None,
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

    spatial = ET.SubElement(
        tendon_root,
        "spatial",
        attrs,
    )
    ET.SubElement(spatial, "site", {"site": site1})
    ET.SubElement(spatial, "site", {"site": site2})


def _short_actuator_name(long_name: str) -> str:
    leg_match = re.fullmatch(
        r"actuator_(position|velocity|torque)_joint_leg_(\d+)_([LR])_(\d+)",
        long_name,
    )
    if leg_match:
        mode, leg_idx, side, joint_idx = leg_match.groups()
        mode_short = {"position": "pos", "velocity": "vel", "torque": "tor"}[mode]
        return f"{mode_short}_leg{leg_idx}_{side}{joint_idx}"

    body_match = re.fullmatch(r"actuator_(position|velocity|torque)_joint_body_(\d+)", long_name)
    if body_match:
        mode, body_idx = body_match.groups()
        mode_short = {"position": "pos", "velocity": "vel", "torque": "tor"}[mode]
        return f"{mode_short}_body{body_idx}"

    return long_name


def _position_mode_from_actuator_name(name: str) -> str | None:
    short_match = re.fullmatch(r"(pos|vel|tor)_.*", name)
    if short_match:
        mode = short_match.group(1)
        return {"pos": "position", "vel": "velocity", "tor": "torque"}[mode]

    long_match = re.fullmatch(r"actuator_(position|velocity|torque)_.*", name)
    if long_match:
        return long_match.group(1)

    return None


def _build_default_keyframe_ctrl_values(actuator_root: ET.Element) -> list[float]:
    ctrl_values: list[float] = []
    for a_node in actuator_root:
        if a_node.tag != "general":
            continue

        ctrl_values.append(0.0)

    return ctrl_values


def _collect_default_qpos(worldbody: ET.Element, joint_overrides: dict[str, float] | None = None) -> list[float]:
    """Collect MuJoCo default qpos in depth-first body order to match joint ordering.

    For hinge/slide joints we use 0. For free joints we copy the parent body's
    pose (pos + quat). Ball joints use identity quaternion.
    """

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
                else:  # hinge / slide
                    qpos.append(float(override_val) if override_val is not None else 0.0)

            # Continue traversal
            if child.tag == "body":
                dfs(child)

    dfs(worldbody)
    return qpos


def generate(template_xml: Path, output_xml: Path, params: SpineParams, frame: PleurobotFrame) -> None:
    if params.num_segments < 2:
        raise ValueError("num_segments must be >= 2")
    if params.segment_spacing <= 0:
        raise ValueError("segment_spacing must be > 0")

    fixed_length = frame.rear_attach_root_x - frame.front_attach_root_x
    chain_len = (params.num_segments - 1) * params.segment_spacing
    if chain_len > fixed_length + 1e-9:
        raise ValueError(
            f"Spine chain length {chain_len:.6f} exceeds fixed hip-to-hip length {fixed_length:.6f}."
        )

    front_extra = (fixed_length - chain_len) / 2.0
    rear_extra = front_extra

    center_x = 0.5 * (frame.front_attach_root_x + frame.rear_attach_root_x)
    x_positions = [center_x - chain_len / 2.0 + i * params.segment_spacing for i in range(params.num_segments)]

    # Endpoint geometry
    a_dx = math.cos(params.alpha) * params.alpha_length
    a_dz = math.sin(params.alpha) * params.alpha_length
    b_dx = -math.cos(params.beta) * params.beta_length
    b_dy = math.sin(params.beta) * params.beta_length

    endpoints = {
        "a1": (a_dx, 0.0, +a_dz),
        "a2": (a_dx, 0.0, -a_dz),
        "b1": (b_dx, +b_dy, 0.0),
        "b2": (b_dx, -b_dy, 0.0),
    }

    tree = ET.parse(template_xml)
    root = tree.getroot()

    worldbody = _child_by_name(root, "worldbody", None) if root.tag == "worldbody" else root.find("worldbody")
    if worldbody is None:
        raise ValueError("No <worldbody> found in template XML")

    pleuro_root = _child_by_name(worldbody, "body", "Pleurobot3_v41")
    link_body_00 = _child_by_name(pleuro_root, "body", "link_body_00")
    link_body_01 = _child_by_name(link_body_00, "body", "link_body_01")
    link_body_06 = _child_by_name(pleuro_root, "body", "link_body_06")

    # Remove old vertebra bodies from previous generation
    _remove_children_by_pred(link_body_01, lambda e: e.tag == "body" and e.attrib.get("name", "").startswith("vertebrae_"))
    _remove_children_by_pred(link_body_06, lambda e: e.tag == "body" and e.attrib.get("name", "").startswith("vertebrae_"))
    _remove_children_by_pred(worldbody, lambda e: e.tag == "body" and e.attrib.get("name", "").startswith("vertebrae_"))

    # Remove locked hinge joint from body_01 (joint_body_00, range ±1e-6)
    _remove_children_by_pred(link_body_01, lambda e: e.tag == "joint")

    # Remove actuators/sensors referencing the removed joint_body_00
    actuator_root = root.find("actuator")
    if actuator_root is not None:
        _remove_children_by_pred(
            actuator_root,
            lambda e: e.attrib.get("joint") == "joint_body_00",
        )
    sensor_root = root.find("sensor")
    removed_body00_actuators = {
        "actuator_position_joint_body_00",
        "actuator_velocity_joint_body_00",
        "actuator_torque_joint_body_00",
    }
    if sensor_root is not None:
        _remove_children_by_pred(
            sensor_root,
            lambda e: (
                e.attrib.get("joint") == "joint_body_00"
                or e.attrib.get("objname") == "Pleurobot3_v41"
                or e.attrib.get("actuator") in removed_body00_actuators
            ),
        )

    # Extract cameras, lights from Pleurobot root (will re-attach to vertebrae_0)
    cameras_lights = [e for e in pleuro_root if e.tag in ("camera", "light")]
    for elem in cameras_lights:
        pleuro_root.remove(elem)

    # Remove IMU site from Pleurobot root
    _remove_direct_sites_by_name(pleuro_root, "com_vertebrae_1")

    # Detach body subtrees from Pleurobot root before removing it
    pleuro_root.remove(link_body_00)   # contains body_01 + front legs
    pleuro_root.remove(link_body_06)   # contains rear legs

    # Remove the entire Pleurobot root body from worldbody
    # (all vertebrae will be free bodies in worldbody instead)
    worldbody.remove(pleuro_root)

    # Build ALL vertebrae as free bodies in worldbody
    x0_local = x_positions[0] - frame.body01_root_x
    front_spacer_dx = frame.front_attach_body01_x - x0_local

    xN_local = x_positions[-1] - frame.body06_root_x
    rear_spacer_dx = frame.rear_attach_body06_x - xN_local
    last_idx = params.num_segments - 1

    vertebra_bodies: dict[int, ET.Element] = {}
    for i in range(params.num_segments):
        spacer_name = None
        spacer_dx = None
        if i == 0:
            spacer_name = "front_spacer"
            spacer_dx = front_spacer_dx
        elif i == last_idx:
            spacer_name = "rear_spacer"
            spacer_dx = rear_spacer_dx

        vertebra_bodies[i] = _add_vertebra_body(
            parent=worldbody,
            name=f"vertebrae_{i}",
            pos_xyz=(-x_positions[i], 0.0, frame.initial_height_z),
            endpoints=endpoints,
            free_joint=True,
            spacer_name=spacer_name,
            spacer_dx=spacer_dx,
        )

    # Reparent front Pleurobot body (body_00 → body_01 → front legs) to vertebrae_0.
    # body_00 was at (0, 0, -0.00025) in Pleurobot root frame;
    # vertebrae_0 is at (x_positions[0], 0, 0) in that same frame.
    # Offset in vertebra local frame = (-x_positions[0], 0, -0.00025).
    link_body_00.set("pos", _vec3((-x_positions[0], 0.0, -0.00025)))
    vertebra_bodies[0].append(link_body_00)

    # Reparent rear Pleurobot body (body_06 → rear legs) to vertebrae_N-1.
    # body_06 was at (body06_root_x, 0, -0.00025) in Pleurobot root frame;
    # vertebrae_N-1 is at (x_positions[-1], 0, 0) in that frame.
    offset_rear_x = frame.body06_root_x - x_positions[-1]
    link_body_06.set("pos", _vec3((offset_rear_x, 0.0, -0.00025)))
    vertebra_bodies[last_idx].append(link_body_06)

    # Move cameras/lights to vertebrae_0
    for elem in cameras_lights:
        vertebra_bodies[0].append(elem)

    # Attach IMU site to link_body_01 (fixed location in body_01 local frame)
    _remove_direct_sites_by_name(link_body_01, "com_vertebrae_1")
    ET.SubElement(
        link_body_01,
        "site",
        {
            "name": "com_vertebrae_1",
            "size": "0.005",
            "pos": _vec3(frame.com_site_body01_pos),
        },
    )

    # Middle vertebra index is still used for body-frame sensors below.
    mid_idx = params.num_segments // 2

    # Tendon defaults stiffness/damping
    vertebra_geom_default = root.find("./default/default/default[@class='tq1']/default[@class='vertebra_geom']/geom")
    if vertebra_geom_default is not None:
        vertebra_geom_default.set("density", _fmt(frame.vertebra_density))
        vertebra_geom_default.set("size", _fmt(frame.vertebra_capsule_size))

    lateral_default = root.find("./default/default/default[@class='tq1']/default[@class='lateral_tendon']/tendon")
    diagonal_default = root.find("./default/default/default[@class='tq1']/default[@class='diagonal_tendon']/tendon")
    if lateral_default is not None:
        # Keep class-level default as average baseline; per-spatial values override it.
        lateral_avg = 0.5 * (params.lateral_verti_stiffness + params.lateral_hori_stiffness)
        lateral_default.set("stiffness", _fmt(lateral_avg))
        lateral_default.set("damping", _fmt(params.tendon_damping))
    if diagonal_default is not None:
        diagonal_default.set("stiffness", _fmt(params.diagonal_stiffness))
        diagonal_default.set("damping", _fmt(params.tendon_damping))

    tendon_root = root.find("tendon")
    if tendon_root is None:
        tendon_root = ET.SubElement(root, "tendon")
    else:
        for child in list(tendon_root):
            tendon_root.remove(child)

    # Add tendons for each adjacent vertebra pair
    local = endpoints
    for i in range(params.num_segments - 1):
        dx = x_positions[i + 1] - x_positions[i]

        # lateral tendons (same endpoints)
        for site in ("a1", "a2", "b1", "b2"):
            p0 = local[site]
            p1 = (local[site][0] + dx, local[site][1], local[site][2])
            rest = _distance(p0, p1)
            lat_stiffness = (
                params.lateral_verti_stiffness if site in ("a1", "a2") else params.lateral_hori_stiffness
            )
            # b-b tendons (b1/b2) are explicitly colored green for easier visualization.
            lat_rgba = (0.0, 1.0, 0.0, 0.7) if site in ("b1", "b2") else None
            _append_spatial_tendon_with_physics(
                tendon_root,
                name=f"lat_{site}_{i}",
                klass="lateral_tendon",
                spring_length=rest * params.lateral_pretension,
                site1=f"vertebrae_{i}_{site}",
                site2=f"vertebrae_{i+1}_{site}",
                stiffness=lat_stiffness,
                damping=params.tendon_damping,
                rgba=lat_rgba,
            )

        # diagonal tendons
        diag_pairs = (("a1", "b1"), ("a2", "b2"), ("a1", "b2"), ("a2", "b1"))
        for a, b in diag_pairs:
            p0 = local[a]
            p1 = (local[b][0] + dx, local[b][1], local[b][2])
            rest = _distance(p0, p1)
            _append_spatial_tendon_with_physics(
                tendon_root,
                name=f"diag_{a}{b}_{i}",
                klass="diagonal_tendon",
                spring_length=rest * params.diagonal_pretension,
                site1=f"vertebrae_{i}_{a}",
                site2=f"vertebrae_{i+1}_{b}",
                stiffness=params.diagonal_stiffness,
                damping=params.tendon_damping,
            )

    # Update sensor references to middle vertebra body
    sensor_root = root.find("sensor")
    if sensor_root is not None:
        mid_name = f"vertebrae_{mid_idx}"
        for s_node in sensor_root:
            if s_node.attrib.get("objname", "").startswith("vertebrae_"):
                s_node.set("objname", mid_name)

    # Shorten actuator names for clearer MuJoCo Control UI labels
    actuator_root = root.find("actuator")
    actuator_name_map: dict[str, str] = {}
    if actuator_root is not None:
        used_names = {
            a.attrib.get("name")
            for a in actuator_root
            if a.tag == "general" and a.attrib.get("name")
        }
        for a in actuator_root:
            if a.tag != "general":
                continue
            old_name = a.attrib.get("name")
            if not old_name:
                continue
            new_name = _short_actuator_name(old_name)
            if new_name == old_name:
                continue
            if new_name in used_names and new_name != old_name:
                raise ValueError(f"Actuator short name collision: {old_name} -> {new_name}")
            used_names.discard(old_name)
            used_names.add(new_name)
            a.set("name", new_name)
            actuator_name_map[old_name] = new_name

    # Keep sensor actuatorfrc references/names in sync with actuator renaming
    if sensor_root is not None and actuator_name_map:
        for s_node in sensor_root.findall("actuatorfrc"):
            act_name = s_node.attrib.get("actuator")
            if act_name in actuator_name_map:
                s_node.set("actuator", actuator_name_map[act_name])

            sensor_name = s_node.attrib.get("name", "")
            if sensor_name.startswith("actuatorfrc_"):
                old_suffix = sensor_name[len("actuatorfrc_") :]
                if old_suffix in actuator_name_map:
                    s_node.set("name", f"actuatorfrc_{actuator_name_map[old_suffix]}")

    # Add keyframe ctrl: use internal vector when provided; otherwise all zeros.
    if actuator_root is not None:
        num_general_actuators = sum(1 for a_node in actuator_root if a_node.tag == "general")

        if frame.keyframe_ctrl_vector is not None:
            if len(frame.keyframe_ctrl_vector) != num_general_actuators:
                raise ValueError(
                    f"keyframe_ctrl_vector length {len(frame.keyframe_ctrl_vector)} "
                    f"!= actuator count {num_general_actuators}"
                )
            ctrl_values = [float(v) for v in frame.keyframe_ctrl_vector]
        else:
            ctrl_values = _build_default_keyframe_ctrl_values(actuator_root)

        keyframe_root = root.find("keyframe")
        if keyframe_root is None:
            keyframe_root = ET.SubElement(root, "keyframe")

        for key_node in list(keyframe_root.findall("key")):
            if key_node.attrib.get("name") == frame.keyframe_name:
                keyframe_root.remove(key_node)

        # Default symmetric qpos derived from body poses (free joints) and hinge overrides for stance.
        qpos_defaults = _collect_default_qpos(worldbody, frame.keyframe_joint_overrides)

        ET.SubElement(
            keyframe_root,
            "key",
            {
                "name": frame.keyframe_name,
                "qpos": " ".join(_fmt(v) for v in qpos_defaults),
                "ctrl": " ".join(_fmt(v) for v in ctrl_values),
            },
        )

    # Keep first lines readable in generated file
    try:
        ET.indent(tree, space="  ")
    except Exception:
        pass

    output_xml.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_xml, encoding="utf-8", xml_declaration=False)

    print("[OK] Generated:", output_xml)
    print("[INFO] fixed_length:", f"{fixed_length:.5f} m")
    print("[INFO] chain_len:", f"{chain_len:.5f} m")
    print("[INFO] front_extra/rear_extra:", f"{front_extra:.5f}/{rear_extra:.5f} m")


def parse_args() -> argparse.Namespace:
    defaults = SpineParams()
    p = argparse.ArgumentParser(description="Generate Pleurobot tensegrity-spine XML")
    p.add_argument(
        "--template",
        type=Path,
        default=Path(__file__).resolve().parent / "xmls" / "PleurobotII" / "pleurobot_tensegrity_spine.xml",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "xmls" / "PleurobotII" / "pleurobot_tensegrity_spine.generated.xml",
    )

    p.add_argument("--num-segments", type=int, default=defaults.num_segments)
    p.add_argument("--segment-spacing", type=float, default=defaults.segment_spacing)
    p.add_argument("--alpha", type=float, default=defaults.alpha)
    p.add_argument("--alpha-length", type=float, default=defaults.alpha_length)
    p.add_argument("--beta", type=float, default=defaults.beta)
    p.add_argument("--beta-length", type=float, default=defaults.beta_length)
    p.add_argument("--lateral-pretension", type=float, default=defaults.lateral_pretension)
    p.add_argument("--diagonal-pretension", type=float, default=defaults.diagonal_pretension)
    p.add_argument("--lateral-verti-stiffness", type=float, default=defaults.lateral_verti_stiffness)
    p.add_argument("--lateral-hori-stiffness", type=float, default=defaults.lateral_hori_stiffness)
    p.add_argument("--diagonal-stiffness", type=float, default=defaults.diagonal_stiffness)
    p.add_argument("--tendon-damping", type=float, default=defaults.tendon_damping)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    params = SpineParams(
        num_segments=args.num_segments,
        segment_spacing=args.segment_spacing,
        alpha=args.alpha,
        alpha_length=args.alpha_length,
        beta=args.beta,
        beta_length=args.beta_length,
        lateral_pretension=args.lateral_pretension,
        diagonal_pretension=args.diagonal_pretension,
        lateral_verti_stiffness=args.lateral_verti_stiffness,
        lateral_hori_stiffness=args.lateral_hori_stiffness,
        diagonal_stiffness=args.diagonal_stiffness,
        tendon_damping=args.tendon_damping,
    )
    frame = PleurobotFrame()
    generate(args.template, args.output, params, frame)


if __name__ == "__main__":
    main()

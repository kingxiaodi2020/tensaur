#!/usr/bin/env python3
"""
Viewer version of spine deformation test (point-force model, consistent with tstats).

- Automatically detects all bodies named 'vertebrae_*'.
- base_body_COM = COM of the first vertebra (e.g. vertebrae_0)
- tip_body_COM  = COM of the last  vertebra (e.g. vertebrae_{N-1})

- tip_point = true load point at the endpoint of the given geom
              (e.g. rear_spacer fromto's p1).

Force model:
- At each simulation step, a point force is applied at tip_point
  (geom endpoint bound to its body).
- We recompute tip_point in world frame each step, compute r = tip_point - COM,
  torque = r x F, and set xfrc_applied accordingly.
- This matches the "point force at moving endpoint" behavior you see visually.

When you close the viewer (ESC), the script prints:
- bending_angle_deg (base_body_COM -> tip_body_COM)
- displacement of each vertebra COM (dx, dy, dz, |disp|)
- displacement of tip_point (dx, dy, dz, |disp|)

Usage:
    python tview_spine.py --xml tensegrity_0.xml --geom rear_spacer --force 30
"""

import argparse
import math
import xml.etree.ElementTree as ET

import mujoco as mj
import mujoco.viewer
import numpy as np


# ------------------------------
# Utility: draw red spheres
# ------------------------------
def draw_force_markers(viewer, start, force, scale=0.03, n_points=8):
    """
    Draw a line of small red spheres from 'start' along 'force' direction.
    Here 'start' will be the *current* tip_point, so the red dots move with the spine.
    """
    scn = viewer.user_scn
    scn.ngeom = 0

    norm_f = np.linalg.norm(force)
    if norm_f < 1e-8:
        return

    direction = force / norm_f
    total_len = norm_f * scale

    for i in range(n_points):
        t = (i + 1) / (n_points + 1)
        pos = start + direction * total_len * t

        mj.mjv_initGeom(
            scn.geoms[scn.ngeom],
            type=mj.mjtGeom.mjGEOM_SPHERE,
            size=[0.02, 0, 0],
            pos=pos,
            mat=np.eye(3).flatten(),
            rgba=np.array([1.0, 0.2, 0.2, 1.0]),
        )
        scn.ngeom += 1


# ------------------------------
# Parse fromto (6 numbers) from XML
# ------------------------------
def get_geom_fromto_from_xml(xml_path: str, geom_name: str) -> np.ndarray:
    """
    Read the fromto (6D vector: [p0, p1]) of a geom from the XML file.
    p1 is used as the local tip_point.
    """
    root = ET.parse(xml_path).getroot()
    for geom in root.findall(".//geom"):
        if geom.get("name") == geom_name:
            s = geom.get("fromto")
            if s is None:
                raise ValueError(f"geom '{geom_name}' has no fromto=")
            return np.fromstring(s, sep=" ")
    raise ValueError(f"Cannot find geom '{geom_name}' in XML: {xml_path}")


# ------------------------------
# Find all vertebrae_* body ids
# ------------------------------
def find_vertebra_ids(model) -> list:
    """
    Find all bodies named 'vertebrae_*' and return their body ids
    sorted by index (vertebrae_0, vertebrae_1, ..., vertebrae_N-1).
    """
    vertebra = []
    for bid in range(model.nbody):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, bid)
        if name is None:
            continue
        if name.startswith("vertebrae_"):
            try:
                idx = int(name.split("vertebrae_")[1])
            except ValueError:
                continue
            vertebra.append((idx, bid))

    if not vertebra:
        raise RuntimeError("No bodies named 'vertebrae_*' found in the model.")

    vertebra.sort(key=lambda x: x[0])
    return [bid for (_, bid) in vertebra]


# ------------------------------
# base-tip bending angle
# ------------------------------
def compute_bending_angle(base0, tip0, base, tip) -> float:
    """
    3D bending angle between base->tip COM vectors before and after:
        theta = arccos( (v0·v) / (|v0||v|) )
    """
    v0 = tip0 - base0
    v = tip - base
    n0 = np.linalg.norm(v0)
    n1 = np.linalg.norm(v)

    if n0 < 1e-8 or n1 < 1e-8:
        return 0.0

    cos_th = float(np.dot(v0, v) / (n0 * n1))
    cos_th = max(-1.0, min(1.0, cos_th))
    return math.degrees(math.acos(cos_th))


# ------------------------------
# Main
# ------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", type=str, default="spine_only.xml")
    parser.add_argument("--geom", type=str, default="rear_spacer",
                        help="Geom whose endpoint is the tip_point (e.g. rear_spacer).")
    parser.add_argument("--force", type=float, default=10.0,
                        help="Force magnitude in Newtons (applied along +y).")
    parser.add_argument("--force_dir", type=str, default="y",
                        help="Force direction: 'x', 'y', or 'z'.")
    args = parser.parse_args()

    xml_path = args.xml
    geom_name = args.geom

    # Validate force direction
    if args.force_dir not in {"x", "y", "z"}:
        raise ValueError("Invalid force_dir. Choose from 'x', 'y', or 'z'.")
    
    # 1) Load fromto (to get local tip_point)
    fromto = get_geom_fromto_from_xml(xml_path, geom_name)
    p1_local = fromto[3:6]        # endpoint local coordinates
    print(f"[info] geom '{geom_name}' endpoint local (tip_point local) =", p1_local)

    # 2) Load model & data
    model = mj.MjModel.from_xml_path(xml_path)
    data = mj.MjData(model)

    # 3) Detect vertebrae_* bodies
    vertebra_ids = find_vertebra_ids(model)
    vertebra_names = [
        mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, bid) for bid in vertebra_ids
    ]
    print("[info] detected vertebra bodies:", vertebra_names)

    base_id = vertebra_ids[0]
    tip_body_id = vertebra_ids[-1]  # for bending angle tip_body_COM

    # 4) Geom body (tip_point 所属 body)
    geom_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, geom_name)
    tip_geom_body_id = model.geom_bodyid[geom_id]

    if tip_geom_body_id != tip_body_id:
        print(
            f"[warn] geom '{geom_name}' is attached to body "
            f"{mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, tip_geom_body_id)}, "
            f"but tip_body_COM is "
            f"{mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, tip_body_id)}."
        )

    # 5) Define force vector based on direction
    force_vec = np.zeros(3)
    if args.force_dir == "x":
        force_vec[0] = args.force
    elif args.force_dir == "y":
        force_vec[1] = args.force
    elif args.force_dir == "z":
        force_vec[2] = args.force

    # ---- Let the system settle under gravity first ----
    for _ in range(2000):
        mj.mj_step(model, data)

    # Initial COM positions of all vertebrae
    vertebra0 = {bid: np.array(data.xipos[bid]) for bid in vertebra_ids}
    base0 = vertebra0[base_id]
    tip0 = vertebra0[tip_body_id]

    print("[info] initial base_body_COM  =", base0)
    print("[info] initial tip_body_COM   =", tip0)

    # Initial tip_point world position
    R0 = data.xmat[tip_geom_body_id].reshape(3, 3)
    body_pos0 = data.xpos[tip_geom_body_id]
    tip_point0 = body_pos0 + R0 @ p1_local
    print("[info] initial tip_point (world) =", tip_point0)

    # ------------------------------
    # Viewer loop (point-force model)
    # ------------------------------
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("\nViewer opened. Point force at tip_point. Press ESC to exit.")

        while viewer.is_running():
            with viewer.lock():
                # 1) 当前 tip_point in world
                R = data.xmat[tip_geom_body_id].reshape(3, 3)
                body_pos = data.xpos[tip_geom_body_id]
                tip_point = body_pos + R @ p1_local

                # 2) 点力 -> 合力 + 合力矩
                com = data.xipos[tip_geom_body_id]
                r = tip_point - com
                torque = np.cross(r, force_vec)

                data.xfrc_applied[tip_geom_body_id, :3] = force_vec
                data.xfrc_applied[tip_geom_body_id, 3:] = torque

                # physics step
                mj.mj_step(model, data)

                # 可视化：用当前 tip_point 作为起点（红球跟着末端走）
                draw_force_markers(viewer, tip_point, force_vec)

                viewer.sync()

        # Clear external force
        data.xfrc_applied[tip_geom_body_id, :] = 0.0

    # ------------------------------
    # After viewer exits → compute deformation
    # ------------------------------
    vertebra = {bid: np.array(data.xipos[bid]) for bid in vertebra_ids}
    base = vertebra[base_id]
    tip = vertebra[tip_body_id]

    # Current tip_point world pos (for displacement)
    R = data.xmat[tip_geom_body_id].reshape(3, 3)
    body_pos = data.xpos[tip_geom_body_id]
    tip_point = body_pos + R @ p1_local

    # Displacements
    vertebra_disps = {
        bid: vertebra[bid] - vertebra0[bid] for bid in vertebra_ids
    }
    tip_point_disp = tip_point - tip_point0
    tdx, tdy, tdz = tip_point_disp
    tmag = np.linalg.norm(tip_point_disp)

    bend_ang = compute_bending_angle(base0, tip0, base, tip)

    # ------------------------------
    # Print results
    # ------------------------------
    print("\n===== RESULTS =====")
    print("Force:", force_vec)
    print(f"Bending angle (base_body_COM -> tip_body_COM) = {bend_ang:.3f} deg")

    print("\n--- Vertebra COM displacements ---")
    for bid in vertebra_ids:
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, bid)
        disp = vertebra_disps[bid]
        dx, dy, dz = disp
        mag = np.linalg.norm(disp)
        print(
            f"  {name}: "
            f"dx={dx:.6f}, dy={dy:.6f}, dz={dz:.6f}, |disp|={mag:.6f}"
        )

    print("\n--- Tip point (geom endpoint) displacement ---")
    print(
        f"  tip_point_disp: "
        f"dx={tdx:.6f}, dy={tdy:.6f}, dz={tdz:.6f}, |disp|={tmag:.6f}"
    )


if __name__ == "__main__":
    main()


















# #!/usr/bin/env python3
# """
# Combined:
# - Visual force arrow (via viewer.user_scn)
# - True endpoint loading on rear_spacer / front_spacer
# - Static stiffness measurement (disp, stiffness, bending angle)

# Usage:
#     python test_spine_full.py --xml tensegrity_0.xml --geom rear_spacer --force 10
# """

# import argparse
# import numpy as np
# import xml.etree.ElementTree as ET
# import mujoco as mj
# import mujoco.viewer
# import math


# # ------------------------------
# # Utility: draw red spheres
# # ------------------------------
# def draw_force_markers(viewer, start, force, scale=0.03, n_points=8):
#     scn = viewer.user_scn
#     scn.ngeom = 0

#     norm_f = np.linalg.norm(force)
#     if norm_f < 1e-8:
#         return

#     direction = force / norm_f
#     total_len = norm_f * scale

#     for i in range(n_points):
#         t = (i + 1) / (n_points + 1)
#         pos = start + direction * total_len * t

#         mj.mjv_initGeom(
#             scn.geoms[scn.ngeom],
#             type=mj.mjtGeom.mjGEOM_SPHERE,
#             size=[0.02, 0, 0],
#             pos=pos,
#             mat=np.eye(3).flatten(),
#             rgba=np.array([1.0, 0.2, 0.2, 1.0])
#         )
#         scn.ngeom += 1


# # ------------------------------
# # Parse fromto from XML
# # ------------------------------
# def get_geom_fromto_from_xml(xml_path, geom_name):
#     tree = ET.parse(xml_path)
#     root = tree.getroot()
#     for geom in root.findall(".//geom"):
#         if geom.get("name") == geom_name:
#             fromto_str = geom.get("fromto")
#             if fromto_str is None:
#                 raise ValueError(f"geom '{geom_name}' has no fromto=")
#             return np.fromstring(fromto_str, sep=" ")
#     raise ValueError(f"Cannot find geom '{geom_name}' in XML")


# # ------------------------------
# # Compute bending angle using base-tip vectors
# # ------------------------------
# def compute_bending_angle(base0, tip0, base, tip):
#     v0 = tip0 - base0
#     v = tip - base
#     n0 = np.linalg.norm(v0)
#     n1 = np.linalg.norm(v)
#     if n0 < 1e-8 or n1 < 1e-8:
#         return 0.0

#     cos_th = np.dot(v0, v) / (n0 * n1)
#     cos_th = np.clip(cos_th, -1.0, 1.0)
#     return math.degrees(math.acos(cos_th))


# # ------------------------------
# # Main
# # ------------------------------
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--xml", type=str, default="spine_only.xml")
#     parser.add_argument("--geom", type=str, default="rear_spacer")
#     parser.add_argument("--force", type=float, default=10.0)
#     args = parser.parse_args()

#     xml_path = args.xml
#     geom_name = args.geom

#     # 1) Load fromto from XML
#     fromto = get_geom_fromto_from_xml(xml_path, geom_name)
#     p1_local = fromto[3:6]  # endpoint of the extension rod
#     print(f"[info] geom '{geom_name}' endpoint local =", p1_local)

#     # 2) Load MuJoCo model
#     model = mj.MjModel.from_xml_path(xml_path)
#     data = mj.MjData(model)

#     # Body that owns the geom
#     gid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, geom_name)
#     body_id = model.geom_bodyid[gid]

#     # define force 目前测试Turn on the spot
#     force_vec = np.array([0.0, args.force, 0.0])

#     # ---- Let the system settle under gravity first ----
#     for _ in range(2000):
#         mj.mj_step(model, data)

#     # Record base/tip initial positions
#     base_body = "vertebrae_0"
#     tip_body = "vertebrae_1"
#     base_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, base_body)
#     tip_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, tip_body)

#     base0 = np.array(data.xipos[base_id])
#     tip0 = np.array(data.xipos[tip_id])

#     print("[info] initial vertebrae_0 =", base0)
#     print("[info] initial tip  =", tip0)

#     # ------------------------------
#     # Viewer + force + stiffness
#     # ------------------------------
#     with mujoco.viewer.launch_passive(model, data) as viewer:
#         print("\nViewer opened. Apply force at tip. Press ESC to exit.")

#         # We will keep applying force until user exits
#         while viewer.is_running():
#             with viewer.lock():
#                 # Transform extension endpoint to world frame
#                 R = data.xmat[body_id].reshape(3, 3)
#                 body_pos = data.xpos[body_id]
#                 load_pos = body_pos + R @ p1_local

#                 # Equivalent force+torque at COM
#                 com = data.xipos[body_id]
#                 r = load_pos - com
#                 torque = np.cross(r, force_vec)

#                 data.xfrc_applied[body_id, :3] = force_vec
#                 data.xfrc_applied[body_id, 3:] = torque

#                 # physics step
#                 mj.mj_step(model, data)

#                 # Draw force visualization
#                 draw_force_markers(viewer, load_pos, force_vec)

#                 viewer.sync()

#         # -------------- After viewer exits, compute final stiffness --------------
#         base = np.array(data.xipos[base_id])
#         tip = np.array(data.xipos[tip_id])

#         disp = tip - tip0
#         disp_norm = np.linalg.norm(disp)
#         F_norm = np.linalg.norm(force_vec)

#         k_eff = F_norm / disp_norm if disp_norm > 1e-8 else float("inf")
#         bend_ang = compute_bending_angle(base0, tip0, base, tip)

#         print("\n===== RESULTS =====")
#         print("Force:", force_vec)
#         print("tip0: ", tip0)
#         print("tip : ", tip)
#         print("disp:", disp, "  |disp| =", disp_norm)
#         print("k_eff:", k_eff, "N/m")
#         print("bend_angle:", bend_ang, "deg")


# if __name__ == "__main__":
#     main()

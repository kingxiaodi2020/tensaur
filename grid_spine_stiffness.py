#!/usr/bin/env python3
"""
Grid search for spine response at the tip_point (rear_spacer endpoint).

- Vary one of:
    * lateral_hori_stiffness
    * lateral_verti_stiffness
    * diagonal_stiffness
  while keeping the other two fixed at baseline values.

- For each configuration:
    * generate a spine-only XML
    * apply 30 N in +y direction at rear_spacer endpoint
      (point-force model: each step we recompute tip_point and torque)
    * measure tip_point displacement dy (in world coordinates)

- For each scanned parameter, save data points as CSV:
    results/grid_<param_name>_Fy<force>.csv
  with columns: stiffness, tip_point_dy

Usage:
    python grid_spine_stiffness.py
"""

import copy
import numpy as np
import matplotlib.pyplot as plt
import mujoco as mj
import csv

from tendon_spine import (
    build_config_from_genes,
    generate_spine_only_from_config,
    ROOT_PATH,
)
from tstats_spine import (
    get_geom_fromto_from_xml,
    settle,
)


# ---------- single test (fixed config, single y-force) ----------

def run_single_y_test(
    xml_path: str,
    force_mag: float = 10.0,
    load_geom_name: str = "rear_spacer",
    settle_gravity_steps: int = 3000,
    settle_load_steps: int = 4000,
) -> np.ndarray:
    """
    Load given XML, settle under gravity, then apply +Fy at rear_spacer endpoint
    using the point-force model (force at moving endpoint).

    Returns:
        tip_point_disp: np.ndarray, 3D displacement of the tip_point
                        (geom endpoint) in world frame.
    """
    model = mj.MjModel.from_xml_path(xml_path)
    data = mj.MjData(model)

    # 1) settle under gravity
    settle(model, data, settle_gravity_steps)

    # save baseline state
    qpos0 = data.qpos.copy()
    qvel0 = data.qvel.copy()

    # reset to baseline
    data.qpos[:] = qpos0
    data.qvel[:] = qvel0
    mj.mj_forward(model, data)

    # 2) build force vector (only y direction)
    force_vec = np.array([0.0, force_mag, 0.0])

    # 3) read extension endpoint in local frame (from XML)
    fromto = get_geom_fromto_from_xml(xml_path, load_geom_name)
    p1_local = fromto[3:6]

    # 4) which body owns this geom?
    gid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, load_geom_name)
    load_body_id = model.geom_bodyid[gid]

    # 5) initial tip_point position (world)
    R0 = data.xmat[load_body_id].reshape(3, 3)
    body_pos0 = data.xpos[load_body_id]
    tip_point0 = body_pos0 + R0 @ p1_local

    # 6) apply point force for settle_load_steps
    for _ in range(settle_load_steps):
        # current tip_point (world)
        R = data.xmat[load_body_id].reshape(3, 3)
        body_pos = data.xpos[load_body_id]
        tip_point = body_pos + R @ p1_local

        # point force -> COM force + torque
        com = data.xipos[load_body_id]
        r = tip_point - com
        torque = np.cross(r, force_vec)

        data.xfrc_applied[load_body_id, :3] = force_vec
        data.xfrc_applied[load_body_id, 3:] = torque

        mj.mj_step(model, data)

    # clear force
    data.xfrc_applied[load_body_id, :] = 0.0

    # 7) final tip_point position
    R = data.xmat[load_body_id].reshape(3, 3)
    body_pos = data.xpos[load_body_id]
    tip_point = body_pos + R @ p1_local

    tip_point_disp = tip_point - tip_point0
    return tip_point_disp


# ---------- grid search ----------

def main():
    # grid: 500, 1000, 2000, 3000, 4000, ..., 15000
    grid_values = [500] + list(range(1000, 15001, 1000))

    # baseline genes
    baseline_genes = {
        "lateral_hori_stiffness": 5000.0,
        "lateral_verti_stiffness": 5000.0,
        "diagonal_stiffness": 5000.0,
    }

    # which parameters to scan
    scan_params = [
        "lateral_hori_stiffness",
        "lateral_verti_stiffness",
        "diagonal_stiffness",
    ]

    force_mag = 10.0  # 30 N in +y

    results_dir = ROOT_PATH / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    for param_name in scan_params:
        print(f"\n==== Scanning {param_name} with Fy = {force_mag} N ====")

        x_vals = []
        dy_vals = []

        for val in grid_values:
            genes = copy.deepcopy(baseline_genes)
            genes[param_name] = float(val)

            # build config & generate spine-only XML
            xml_dir = ROOT_PATH / "xmls"
            xml_dir.mkdir(parents=True, exist_ok=True)
            xml_path = xml_dir / f"spine_only_{param_name}_{int(val)}.xml"

            cfg = build_config_from_genes(genes, individual_id=0, output_path=xml_path)
            generate_spine_only_from_config(cfg)

            # run single y-direction test (point-force model)
            tip_disp = run_single_y_test(str(xml_path), force_mag=force_mag)
            dy = tip_disp[1]

            print(f"{param_name}={val:6.1f} -> tip_point_dy={dy: .6f} m")

            x_vals.append(val)
            dy_vals.append(dy)

        # ---- save data to CSV ----
        csv_path = results_dir / f"grid_{param_name}_Fy{int(force_mag)}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["stiffness", "tip_point_dy"])
            for s, dy in zip(x_vals, dy_vals):
                writer.writerow([s, dy])
        print(f"[info] saved data to {csv_path}")

        # ---- plotting (for immediate inspection) ----
        x_vals_arr = np.array(x_vals, dtype=float)
        dy_vals_arr = np.array(dy_vals, dtype=float)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.set_xlabel(f"{param_name} (stiffness)")
        ax.set_ylabel("tip_point dy (m)")
        ax.plot(x_vals_arr, dy_vals_arr, marker="o", linestyle="-")
        ax.grid(True, linestyle=":")
        fig.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()

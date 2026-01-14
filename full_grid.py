#!/usr/bin/env python3
"""
Full 3D grid search for spine response at the tip_point (rear_spacer endpoint).

支持两种加载模式：
1. 点力模式（force）：在 rear_spacer 端点施加力（x/y/z方向）
2. 扭矩模式（torque）：在最后一个椎体施加扭矩（x轴）

Parameters (all combinations):
    - lateral_hori_stiffness  in {500, 1000, 2000, ..., 15000}
    - lateral_verti_stiffness in {500, 1000, 2000, ..., 15000}
    - diagonal_stiffness      in {500, 1000, 2000, ..., 15000}

For each (lh, lv, dg):
    * build spine-only config & XML
    * apply load (force or torque)
    * measure response (displacement or torsion)

All data are saved into a CSV:
    results/spine_grid_<load_type><magnitude>.csv

Columns (force mode):
    lateral_hori_stiffness,
    lateral_verti_stiffness,
    diagonal_stiffness,
    tip_dx,
    tip_dy,
    tip_dz,
    tip_disp_norm

Columns (torque mode):
    lateral_hori_stiffness,
    lateral_verti_stiffness,
    diagonal_stiffness,
    total_twist_deg,
    vertebra_0_1_twist_deg,
    vertebra_1_2_twist_deg,
    ...
"""

import csv
import itertools
import numpy as np
import mujoco as mj
import math

from tendon_spine import (
    build_config_from_genes,
    generate_spine_only_from_config,
    ROOT_PATH,
)
from tstats_spine import (
    get_geom_fromto_from_xml,
    settle,
)


# ---------- utilities ----------

def find_vertebra_ids(model) -> list:
    """找到所有 vertebrae_* 并按编号排序"""
    vertebra = []
    for bid in range(model.nbody):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, bid)
        if name and name.startswith("vertebrae_"):
            try:
                idx = int(name.split("vertebrae_")[1])
                vertebra.append((idx, bid))
            except:
                continue
    vertebra.sort(key=lambda x: x[0])
    return [bid for _, bid in vertebra]


def compute_relative_twist(R_base, R_tip):
    """计算两个刚体之间的相对扭转角（沿 x 轴）"""
    R_rel = R_tip @ R_base.T
    # 提取绕 x 轴的旋转分量
    twist_x = math.atan2(R_rel[2, 1] - R_rel[1, 2], 
                          R_rel[1, 1] + R_rel[2, 2])
    return twist_x


# ---------- single force test ----------

def run_single_force_test(
    xml_path: str,
    force_mag: float = 30.0,
    force_dir: str = "y",
    load_geom_name: str = "rear_spacer",
    settle_gravity_steps: int = 2000,
    settle_load_steps: int = 4000,
) -> np.ndarray:
    """
    施加点力（force），返回 tip_point 位移
    
    Returns:
        tip_point_disp: np.ndarray, 3D displacement
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

    # 2) build force vector
    force_vec = np.zeros(3)
    if force_dir == "x":
        force_vec[0] = force_mag
    elif force_dir == "y":
        force_vec[1] = force_mag
    elif force_dir == "z":
        force_vec[2] = force_mag
    else:
        raise ValueError("Invalid force_dir. Choose from 'x', 'y', or 'z'.")

    # 3) read extension endpoint in local frame
    fromto = get_geom_fromto_from_xml(xml_path, load_geom_name)
    p1_local = fromto[3:6]

    # 4) which body owns this geom?
    gid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, load_geom_name)
    load_body_id = model.geom_bodyid[gid]

    # 5) initial tip_point position
    R0 = data.xmat[load_body_id].reshape(3, 3)
    body_pos0 = data.xpos[load_body_id]
    tip_point0 = body_pos0 + R0 @ p1_local

    # 6) apply point force
    for _ in range(settle_load_steps):
        R = data.xmat[load_body_id].reshape(3, 3)
        body_pos = data.xpos[load_body_id]
        tip_point = body_pos + R @ p1_local

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


# ---------- single torque test ----------

def run_single_torque_test(
    xml_path: str,
    torque_mag: float = 1.0,
    settle_gravity_steps: int = 2000,
    settle_load_steps: int = 4000,
) -> dict:
    """
    施加扭矩（torque_x），返回椎体间相对扭转角度
    
    Returns:
        result: dict with keys:
            - total_twist_deg: 总扭转角度（度）
            - segment_twists_deg: list of (vertebra_i, vertebra_j, twist_deg)
    """
    model = mj.MjModel.from_xml_path(xml_path)
    data = mj.MjData(model)

    # 找到所有椎体
    vertebra_ids = find_vertebra_ids(model)
    last_id = vertebra_ids[-1]

    # 1) settle under gravity
    settle(model, data, settle_gravity_steps)

    # save baseline state
    qpos0 = data.qpos.copy()
    qvel0 = data.qvel.copy()

    # 保存初始姿态（每个椎体）
    R0_all = {bid: data.xmat[bid].reshape(3, 3).copy() 
              for bid in vertebra_ids}

    # reset to baseline
    data.qpos[:] = qpos0
    data.qvel[:] = qvel0
    mj.mj_forward(model, data)

    # 2) apply torque
    tau = np.array([torque_mag, 0.0, 0.0])
    
    for _ in range(settle_load_steps):
        data.xfrc_applied[last_id, :3] = 0.0
        data.xfrc_applied[last_id, 3:] = tau
        mj.mj_step(model, data)

    # clear torque
    data.xfrc_applied[last_id, :] = 0.0

    # 3) 计算椎体间相对扭转
    total_twist = 0.0
    segment_twists = []
    
    for i in range(len(vertebra_ids) - 1):
        bid_i = vertebra_ids[i]
        bid_j = vertebra_ids[i + 1]
        
        R_i_0 = R0_all[bid_i]
        R_j_0 = R0_all[bid_j]
        R_i = data.xmat[bid_i].reshape(3, 3)
        R_j = data.xmat[bid_j].reshape(3, 3)
        
        # 初始相对姿态
        twist_0 = compute_relative_twist(R_i_0, R_j_0)
        # 最终相对姿态
        twist = compute_relative_twist(R_i, R_j)
        
        # 扭转变化量
        delta_twist = twist - twist_0
        total_twist += delta_twist
        
        segment_twists.append((i, i+1, math.degrees(delta_twist)))
    
    return {
        "total_twist_deg": math.degrees(total_twist),
        "segment_twists_deg": segment_twists,
    }


# ---------- full 3D grid search ----------

def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_type", type=str, default="force",
                        choices=["force", "torque"],
                        help="Loading type: 'force' or 'torque'")
    parser.add_argument("--magnitude", type=float, default=40.0,
                        help="Force magnitude (N) or torque magnitude (Nm)")
    parser.add_argument("--direction", type=str, default="z",
                        choices=["x", "y", "z"],
                        help="Force direction (only for force mode)")
    parser.add_argument("--settle_load_steps", type=int, default=4000,
                        help="Number of steps to settle under load")
    args = parser.parse_args()

    # grid: 500, 1000, 2000, 3000, ..., 15000
    grid_values = np.unique(np.round(np.logspace(
        np.log10(500), np.log10(15000), num=30  # num可调
    )).astype(int))

    xml_dir = ROOT_PATH / "xmls"
    xml_dir.mkdir(parents=True, exist_ok=True)
    xml_path = xml_dir / "spine_grid_tmp.xml"

    results_dir = ROOT_PATH / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成文件名
    if args.load_type == "force":
        csv_filename = f"log_500_F{args.direction}{int(args.magnitude)}.csv"
    else:
        csv_filename = f"log_500_Tx{int(args.magnitude)}.csv"
    
    csv_path = results_dir / csv_filename

    # 准备 CSV
    f_csv = open(csv_path, "w", newline="")
    writer = csv.writer(f_csv)
    
    if args.load_type == "force":
        writer.writerow([
            "lateral_hori_stiffness",
            "lateral_verti_stiffness",
            "diagonal_stiffness",
            "tip_dx",
            "tip_dy",
            "tip_dz",
            "tip_disp_norm",
        ])
    else:  # torque
        # 需要先生成一次 XML 来知道有多少个椎体
        genes = {
            "lateral_hori_stiffness": 1000.0,
            "lateral_verti_stiffness": 1000.0,
            "diagonal_stiffness": 1000.0,
        }
        cfg = build_config_from_genes(genes, individual_id=0, output_path=xml_path)
        generate_spine_only_from_config(cfg)
        
        model_tmp = mj.MjModel.from_xml_path(str(xml_path))
        data_tmp = mj.MjData(model_tmp)
        vertebra_ids = find_vertebra_ids(model_tmp)
        n_segments = len(vertebra_ids) - 1
        
        header = [
            "lateral_hori_stiffness",
            "lateral_verti_stiffness",
            "diagonal_stiffness",
            "total_twist_deg",
        ]
        for i in range(n_segments):
            header.append(f"twist_{i}_{i+1}_deg")
        writer.writerow(header)

    total_combos = len(grid_values) ** 3
    combo_idx = 0

    for lh, lv, dg in itertools.product(grid_values, grid_values, grid_values):
        combo_idx += 1
        print(
            f"[{combo_idx:4d}/{total_combos}] "
            f"lh={lh:5.0f}, lv={lv:5.0f}, dg={dg:5.0f}"
        )

        genes = {
            "lateral_hori_stiffness": float(lh),
            "lateral_verti_stiffness": float(lv),
            "diagonal_stiffness": float(dg),
        }

        # build config & generate XML
        cfg = build_config_from_genes(genes, individual_id=0, output_path=xml_path)
        generate_spine_only_from_config(cfg)

        if args.load_type == "force":
            # run force test
            tip_disp = run_single_force_test(
                str(xml_path), 
                force_mag=args.magnitude, 
                force_dir=args.direction,
                settle_load_steps=args.settle_load_steps,
            )
            dx, dy, dz = tip_disp
            disp_norm = float(np.linalg.norm(tip_disp))
            writer.writerow([lh, lv, dg, dx, dy, dz, disp_norm])
        
        else:  # torque
            # run torque test
            result = run_single_torque_test(
                str(xml_path),
                torque_mag=args.magnitude,
                settle_load_steps=args.settle_load_steps,
            )
            row = [lh, lv, dg, result["total_twist_deg"]]
            for _, _, twist_deg in result["segment_twists_deg"]:
                row.append(twist_deg)
            writer.writerow(row)

    f_csv.close()
    print(f"[done] saved full grid data to {csv_path}")


if __name__ == "__main__":
    main()
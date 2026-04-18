#!/usr/bin/env python3
import argparse
import numpy as np
import mujoco as mj
import mujoco.viewer
import math


def find_vertebra_ids(model):
    """找到所有 vertebrae_* 并排序"""
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
    # 提取绕 x 轴的旋转分量（Rodrigues公式的简化）
    # atan2 给出 (-π, π) 范围内的角度
    twist_x = math.atan2(R_rel[2, 1] - R_rel[1, 2], 
                          R_rel[1, 1] + R_rel[2, 2])
    return twist_x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", type=str, default="spine_only.xml")
    parser.add_argument("--torque", type=float, default=5.0)
    args = parser.parse_args()

    model = mj.MjModel.from_xml_path(args.xml)
    data = mj.MjData(model)

    # 找到所有椎体
    vertebra_ids = find_vertebra_ids(model)
    vertebra_names = [mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, bid) 
                      for bid in vertebra_ids]
    
    print(f"[info] Found {len(vertebra_ids)} vertebrae: {vertebra_names}")
    
    # 施加扭矩的目标（最后一个椎体）
    last_id = vertebra_ids[-1]
    last_name = vertebra_names[-1]
    
    print(f"[info] Applying torque on: {last_name}")

    # 重力稳定
    for _ in range(2000):
        mj.mj_step(model, data)

    # 保存初始姿态（每个椎体）
    R0_all = {bid: data.xmat[bid].reshape(3, 3).copy() 
              for bid in vertebra_ids}

    # Viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("\nViewer opened. Applying torque continuously. ESC to exit.\n")
        
        tau = np.array([args.torque, 0.0, 0.0])

        while viewer.is_running():
            with viewer.lock():
                data.xfrc_applied[last_id, :3] = 0.0
                data.xfrc_applied[last_id, 3:] = tau
                mj.mj_step(model, data)
            viewer.sync()

    # 计算椎体间相对扭转
    print("\n===== RESULTS =====")
    print(f"Applied torque: {tau} Nm")
    print("\n--- Relative Torsion Between Adjacent Vertebrae ---")
    
    total_twist = 0.0
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
        
        print(f"  {vertebra_names[i]} → {vertebra_names[i+1]}: "
              f"{math.degrees(delta_twist):.4f}°")
    
    print(f"\n--- Total Cumulative Torsion: {math.degrees(total_twist):.4f}° ---")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Spine deformation test (multi-vertebra, point-force model).

We distinguish:

- vertebrae_0, vertebrae_1, ..., vertebrae_{N-1}:
    all vertebra bodies, detected automatically by name prefix "vertebrae_".
    For each vertebra we report its COM displacement (dx, dy, dz, |disp|).

- base_body_COM:  COM of the first vertebra (e.g. vertebrae_0)
- tip_body_COM:   COM of the last  vertebra (e.g. vertebrae_{N-1})

- tip_point:      the true load point at the extension geom endpoint
                  (e.g., rear_spacer fromto's p1). We report its displacement.

Force model in this script:
- A point force is applied at tip_point (geom endpoint) in each step.
- For each step we recompute r = tip_point - COM and torque = r x F.
  This means the force always acts at the current end of the extension rod
  in world coordinates (point-force model, not fixed moment).

For each load direction (+x, +y, +z), we report:

- bending_angle_deg:  3D angle between base_body_COM -> tip_body_COM vectors
                      (before vs after load)
- COM displacement of each vertebra (dx, dy, dz, |disp|)
- tip_point displacement (dx, dy, dz, |disp|)

No stiffness (k_eff) is reported.
"""

import argparse
import math
from dataclasses import dataclass
import xml.etree.ElementTree as ET

import mujoco as mj
import numpy as np


# ----------------------------
# Data structures
# ----------------------------

@dataclass
class ForceResult:
    """Force mode 结果"""
    direction: str                 # "x", "y", "z"
    force_mag: float               # |F|
    vertebra_disps: dict           # body_id -> disp(np.ndarray[3])
    tip_point_disp: np.ndarray     # displacement of geom endpoint
    bend_angle_deg: float          # base->tip COM bending angle


@dataclass
class TorqueResult:
    """Torque mode 结果"""
    torque_mag: float              # |T|
    total_twist_deg: float         # 总扭转角（度）
    segment_twists: list           # [(i, j, twist_deg), ...]


# ----------------------------
# Utilities
# ----------------------------

def settle(model, data, steps: int = 4000) -> None:
    """Let the system settle for a given number of steps."""
    for _ in range(steps):
        mj.mj_step(model, data)


def get_body_pos(model, data, body_id: int) -> np.ndarray:
    """World position of body COM by body id."""
    return np.array(data.xipos[body_id])


def get_geom_fromto_from_xml(xml_path: str, geom_name: str) -> np.ndarray:
    """Read the fromto (6D vector: [p0, p1]) of a geom from the XML."""
    root = ET.parse(xml_path).getroot()
    for geom in root.findall(".//geom"):
        if geom.get("name") == geom_name:
            s = geom.get("fromto")
            if s is None:
                raise ValueError(f"Geom '{geom_name}' has no fromto=")
            return np.fromstring(s, sep=" ")
    raise ValueError(f"Geom '{geom_name}' not found in {xml_path}")


def compute_bending_angle(
    base0: np.ndarray,
    tip0: np.ndarray,
    base: np.ndarray,
    tip: np.ndarray,
) -> float:
    """
    3D bending angle between base->tip COM vectors before and after:
        theta = arccos( (v0·v) / (|v0||v|) ).
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


def compute_relative_twist(R_base, R_tip):
    """计算两个刚体之间的相对扭转角（沿 x 轴）"""
    R_rel = R_tip @ R_base.T
    # 提取绕 x 轴的旋转分量
    twist_x = math.atan2(R_rel[2, 1] - R_rel[1, 2], 
                          R_rel[1, 1] + R_rel[2, 2])
    return twist_x


def find_vertebra_ids(model) -> list:
    """
    自动寻找所有名称以 'vertebrae_' 开头的 body，
    并按编号排序（vertebrae_0, vertebrae_1, ..., vertebrae_N-1）。
    返回 body_id 列表（按顺序）。
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


# ----------------------------
# Force mode test (point-force model)
# ----------------------------

def run_force_test(
    model,
    data,
    vertebra_ids: list,
    load_geom_name: str,
    xml_path: str,
    force_vec: np.ndarray,
    settle_steps: int = 20000,
) -> ForceResult:
    """
    Apply force_vec at the tip_point (geom endpoint) and measure:

    - COM displacement of each vertebra in vertebra_ids
    - tip_point displacement
    - bending angle between base_body_COM -> tip_body_COM (first & last vertebra).

    Force model:
    - At each step, compute current tip_point in world coordinates,
      then r = tip_point - COM, torque = r x F, and set xfrc_applied.
    """

    # 1) tip_point (geom endpoint) in local coordinates
    fromto = get_geom_fromto_from_xml(xml_path, load_geom_name)
    p1_local = fromto[3:6]

    # 2) tip_point 所属 body
    geom_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, load_geom_name)
    tip_geom_body_id = model.geom_bodyid[geom_id]

    # 3) 初始 vertebra COM 位置
    vertebra0 = {bid: get_body_pos(model, data, bid) for bid in vertebra_ids}

    base_id = vertebra_ids[0]
    tip_body_id = vertebra_ids[-1]
    base0 = vertebra0[base_id]
    tip0 = vertebra0[tip_body_id]

    # tip_point 初始位置
    R0 = data.xmat[tip_geom_body_id].reshape(3, 3)
    body_pos0 = data.xpos[tip_geom_body_id]
    tip_point0 = body_pos0 + R0 @ p1_local

    # 4) 在 settle_steps 内，每步用当前姿态更新 tip_point / torque
    for _ in range(settle_steps):
        # 当前 tip_point
        R = data.xmat[tip_geom_body_id].reshape(3, 3)
        body_pos = data.xpos[tip_geom_body_id]
        tip_point = body_pos + R @ p1_local

        # 质心与力臂
        com = data.xipos[tip_geom_body_id]
        r = tip_point - com
        torque = np.cross(r, force_vec)

        data.xfrc_applied[tip_geom_body_id, :3] = force_vec
        data.xfrc_applied[tip_geom_body_id, 3:] = torque

        mj.mj_step(model, data)

    # 清除外力
    data.xfrc_applied[tip_geom_body_id, :] = 0.0

    # 5) 最终 vertebra COM & tip_point 位置
    vertebra = {bid: get_body_pos(model, data, bid) for bid in vertebra_ids}
    base = vertebra[base_id]
    tip = vertebra[tip_body_id]

    R = data.xmat[tip_geom_body_id].reshape(3, 3)
    body_pos = data.xpos[tip_geom_body_id]
    tip_point = body_pos + R @ p1_local

    # 位移
    vertebra_disps = {
        bid: vertebra[bid] - vertebra0[bid] for bid in vertebra_ids
    }
    tip_point_disp = tip_point - tip_point0

    bend_angle_deg = compute_bending_angle(base0, tip0, base, tip)

    # 方向标签
    direction = ""
    if abs(force_vec[0]) > 0:
        direction += "x"
    if abs(force_vec[1]) > 0:
        direction += "y"
    if abs(force_vec[2]) > 0:
        direction += "z"

    return ForceResult(
        direction=direction,
        force_mag=float(np.linalg.norm(force_vec)),
        vertebra_disps=vertebra_disps,
        tip_point_disp=tip_point_disp,
        bend_angle_deg=bend_angle_deg,
    )


# ----------------------------
# Torque mode test
# ----------------------------

def run_torque_test(
    model,
    data,
    vertebra_ids: list,
    torque_mag: float,
    settle_steps: int = 4000,
    convergence_threshold: float = 1e-4,
    check_interval: int = 500,
) -> TorqueResult:
    """
    在最后一个椎体上施加 x 轴扭矩，测量椎体间相对扭转
    
    Args:
        convergence_threshold: 角速度的绝对值小于此值认为收敛
        check_interval: 每隔多少步检查一次收敛
    
    Returns:
        TorqueResult with total twist and segment twists
    """
    last_id = vertebra_ids[-1]

    # 保存初始姿态（每个椎体）
    R0_all = {bid: data.xmat[bid].reshape(3, 3).copy() 
              for bid in vertebra_ids}

    # 施加扭矩
    tau = np.array([torque_mag, 0.0, 0.0])
    
    # ===== 新增：收敛检测 =====
    convergence_count = 0  # 连续收敛的检查次数
    required_convergence_checks = 3  # 需要连续3次检查都收敛
    is_converged = False
    
    for step in range(settle_steps):
        data.xfrc_applied[last_id, :3] = 0.0
        data.xfrc_applied[last_id, 3:] = tau
        mj.mj_step(model, data)

        # 每 check_interval 步检查一次收敛
        if (step + 1) % check_interval == 0:
            # 获取最后一个椎体的角速度
            last_body_qvel_idx = model.body_dofadr[last_id]
            if last_body_qvel_idx >= 0:
                # 获取该 body 的角速度（qvel 后3个分量通常是角速度）
                angvel = data.qvel[last_body_qvel_idx + 3:last_body_qvel_idx + 6]
                angvel_mag = np.linalg.norm(angvel)
                
                if angvel_mag < convergence_threshold:
                    convergence_count += 1
                    if convergence_count >= required_convergence_checks:
                        is_converged = True
                        print(f"  [Converged at step {step + 1}] angvel_mag = {angvel_mag:.2e}")
                        break
                else:
                    convergence_count = 0  # 重置计数器
    
    # 清除外力
    data.xfrc_applied[last_id, :] = 0.0
    
    # ===== 新增：如果未收敛，标记为异常 =====
    if not is_converged:
        print(f"  [WARNING] Not converged after {settle_steps} steps!")
        return TorqueResult(
            torque_mag=torque_mag,
            total_twist_deg=float('nan'),  # 返回 NaN 表示无效
            segment_twists=[(i, i+1, float('nan')) 
                           for i in range(len(vertebra_ids) - 1)],
        )

    # 计算椎体间相对扭转
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
    
    return TorqueResult(
        torque_mag=torque_mag,
        total_twist_deg=math.degrees(total_twist),
        segment_twists=segment_twists,
    )

# ----------------------------
# Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Spine deformation test with force or torque loading"
    )
    parser.add_argument("--xml", type=str, default="spine_only.xml",
                        help="Path to MuJoCo XML file of the spine.")
    
    # 加载模式选择
    parser.add_argument("--load_type", type=str, default="force",
                        choices=["force", "torque"],
                        help="Loading type: 'force' or 'torque'")
    
    # Force mode 参数
    parser.add_argument("--geom", type=str, default="rear_spacer",
                        help="Geom name whose endpoint is the tip_point "
                             "(e.g. rear_spacer). Only for force mode.")
    parser.add_argument("--force", type=float, default=10.0,
                        help="Force magnitude in Newtons. Only for force mode.")
    parser.add_argument("--force_dir", type=str, default="all",
                        choices=["x", "y", "z", "all"],
                        help="Force direction (x/y/z/all). Only for force mode.")
    
    # Torque mode 参数
    parser.add_argument("--torque", type=float, default=5.0,
                        help="Torque magnitude in Nm. Only for torque mode.")
    
    args = parser.parse_args()

    print(f"[Loading] {args.xml}")
    model = mj.MjModel.from_xml_path(args.xml)
    data = mj.MjData(model)

    # 1) 找到所有 vertebrae_* 的 body id
    vertebra_ids = find_vertebra_ids(model)
    vertebra_names = [
        mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, bid) for bid in vertebra_ids
    ]
    print("[info] detected vertebra bodies:", vertebra_names)

    # 2) 重力收敛，得到 baseline 姿态
    print("[Settling under gravity]")
    settle(model, data, 2000)

    # 保存 baseline 状态
    qpos0 = data.qpos.copy()
    qvel0 = data.qvel.copy()

    # 3) 根据加载模式执行测试
    if args.load_type == "force":
        # ========== Force Mode ==========
        
        # 确定要测试的方向
        if args.force_dir == "all":
            forces = [
                np.array([args.force, 0.0, 0.0]),
                np.array([0.0, args.force, 0.0]),
                np.array([0.0, 0.0, args.force]),
            ]
        elif args.force_dir == "x":
            forces = [np.array([args.force, 0.0, 0.0])]
        elif args.force_dir == "y":
            forces = [np.array([0.0, args.force, 0.0])]
        elif args.force_dir == "z":
            forces = [np.array([0.0, 0.0, args.force])]

        all_results = []

        for f in forces:
            # reset 到 baseline
            data.qpos[:] = qpos0
            data.qvel[:] = qvel0
            mj.mj_forward(model, data)

            print(f"\n=== Testing load {f} N ===")
            res = run_force_test(
                model=model,
                data=data,
                vertebra_ids=vertebra_ids,
                load_geom_name=args.geom,
                xml_path=args.xml,
                force_vec=f,
                settle_steps=4000,
            )
            all_results.append(res)

            print(f"Direction: {res.direction}")
            print(f"  |F| = {res.force_mag:.3f} N")
            print(f"  Bending angle (base->tip) = {res.bend_angle_deg:.3f} deg")

            print("  --- Vertebra COM displacements ---")
            for bid in vertebra_ids:
                name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, bid)
                disp = res.vertebra_disps[bid]
                dx, dy, dz = disp
                mag = np.linalg.norm(disp)
                print(
                    f"    {name}: "
                    f"dx={dx:.6f}, dy={dy:.6f}, dz={dz:.6f}, |disp|={mag:.6f}"
                )

            tip_disp = res.tip_point_disp
            tdx, tdy, tdz = tip_disp
            tmag = np.linalg.norm(tip_disp)
            print("  --- Tip point (geom endpoint) displacement ---")
            print(
                f"    tip_point_disp: "
                f"dx={tdx:.6f}, dy={tdy:.6f}, dz={tdz:.6f}, |disp|={tmag:.6f}"
            )

        # Summary
        print("\n====== Summary ======")
        for res in all_results:
            tip_disp = res.tip_point_disp
            tmag = np.linalg.norm(tip_disp)
            print(
                f"{res.direction:>2}: bend = {res.bend_angle_deg:7.3f} deg, "
                f"|tip_point_disp| = {tmag:.6f} m"
            )

    else:
        # ========== Torque Mode ==========
        
        # reset 到 baseline
        data.qpos[:] = qpos0
        data.qvel[:] = qvel0
        mj.mj_forward(model, data)

        print(f"\n=== Testing torque {args.torque} Nm (x-axis) ===")
        res = run_torque_test(
            model=model,
            data=data,
            vertebra_ids=vertebra_ids,
            torque_mag=args.torque,
            settle_steps=4000,
            convergence_threshold=0.1,  # 可调参数
            check_interval=200,
        )

        print(f"Torque magnitude: {res.torque_mag:.3f} Nm")
        
        # ===== 新增：检查是否有效 =====
        if np.isnan(res.total_twist_deg):
            print("[ERROR] System did not converge - results are invalid!")
            print("Consider:")
            print("  - Reducing torque magnitude")
            print("  - Increasing stiffness")
            print("  - Increasing settle_steps")
            return
        
        print(f"\n--- Relative Torsion Between Adjacent Vertebrae ---")
        
        for i, j, twist_deg in res.segment_twists:
            name_i = vertebra_names[i]
            name_j = vertebra_names[j]
            print(f"  {name_i} → {name_j}: {twist_deg:.4f}°")
        
        print(f"\n--- Total Cumulative Torsion: {res.total_twist_deg:.4f}° ---")


if __name__ == "__main__":
    main()
























# #!/usr/bin/env python3
# """
# Stiffness test for tensegrity spine.
# Loads force at rear_spacer (or front_spacer) endpoint instead of vertebrae_1 COM.

# Usage:
#     python test_spine_stats.py --xml spine_only.xml --geom rear_spacer --force 10
# """

# import argparse
# import math
# from dataclasses import dataclass
# import xml.etree.ElementTree as ET

# import mujoco as mj
# import numpy as np


# @dataclass
# class TestResult:
#     direction: str
#     force: float
#     disp: np.ndarray        # 3D displacement of tip
#     k_eff: float            # effective stiffness |F| / |disp|
#     bend_angle_deg: float   # bending angle in degrees


# # ----------------------------
# # Utilities
# # ----------------------------

# def settle(model, data, steps=4000):
#     """Let the system settle for a given number of steps."""
#     for _ in range(steps):
#         mj.mj_step(model, data)


# def get_body_pos(model, data, body_name: str) -> np.ndarray:
#     """World position of body CoM."""
#     bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name)
#     return np.array(data.xipos[bid])


# def compute_bending_angle(base0, tip0, base, tip) -> float:
#     """
#     3D bending angle between base-tip vectors before and after:
#         theta = arccos( (v0·v) / (|v0||v|) )
#     """
#     v0 = tip0 - base0
#     v = tip - base

#     n0 = np.linalg.norm(v0)
#     n1 = np.linalg.norm(v)
#     if n0 < 1e-8 or n1 < 1e-8:
#         return 0.0

#     cos_th = float(np.dot(v0, v) / (n0 * n1))
#     cos_th = max(-1.0, min(1.0, cos_th))
#     return math.degrees(math.acos(cos_th))


# def get_geom_fromto_from_xml(xml_path: str, geom_name: str) -> np.ndarray:
#     """从 XML 文件中读取指定 geom 的 fromto（6D 向量: [p0, p1]）。"""
#     root = ET.parse(xml_path).getroot()
#     for geom in root.findall(".//geom"):
#         if geom.get("name") == geom_name:
#             s = geom.get("fromto")
#             if s is None:
#                 raise ValueError(f"Geom '{geom_name}' has no fromto=")
#             return np.fromstring(s, sep=" ")
#     raise ValueError(f"Geom '{geom_name}' not found in {xml_path}")


# # ----------------------------
# # Single direction test
# # ----------------------------

# def run_direction_test(model,
#                        data,
#                        base_body: str,
#                        tip_body: str,
#                        load_geom_name: str,
#                        xml_path: str,
#                        force_vec: np.ndarray,
#                        settle_steps: int = 4000) -> TestResult:
#     """
#     在延长段末端施加力 force_vec，测量 tip_body 的位移和弯曲角度。
#     注意：调用该函数前调用者要把 data.qpos/qvel 设到想要的初始状态。
#     """

#     # 1) 读取几何体的 fromto，拿到局部终点
#     fromto = get_geom_fromto_from_xml(xml_path, load_geom_name)
#     p1_local = fromto[3:6]

#     # 2) 找到这个 geom 所属的 body
#     gid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM, load_geom_name)
#     load_body_id = model.geom_bodyid[gid]

#     # 3) 初始 base / tip 位置（重力收敛后的基准）
#     base0 = get_body_pos(model, data, base_body)
#     tip0 = get_body_pos(model, data, tip_body)

#     # 4) 计算加载点世界坐标
#     R = data.xmat[load_body_id].reshape(3, 3)
#     body_pos = data.xpos[load_body_id]
#     load_pos = body_pos + R @ p1_local

#     # 5) 点力 -> 作用在质心的合力 + 合力矩
#     com = data.xipos[load_body_id]
#     r = load_pos - com
#     torque = np.cross(r, force_vec)

#     data.xfrc_applied[load_body_id, :3] = force_vec
#     data.xfrc_applied[load_body_id, 3:] = torque

#     # 6) 在该力下收敛
#     settle(model, data, settle_steps)

#     # 7) 最终 base / tip
#     base = get_body_pos(model, data, base_body)
#     tip = get_body_pos(model, data, tip_body)

#     # 8) 清除外力，为下一次实验准备
#     data.xfrc_applied[load_body_id, :] = 0.0

#     # 9) 计算刚度和弯曲角
#     disp = tip - tip0
#     disp_norm = np.linalg.norm(disp)
#     F_norm = np.linalg.norm(force_vec)
#     k_eff = F_norm / disp_norm if disp_norm > 1e-8 else float("inf")
#     bend_angle = compute_bending_angle(base0, tip0, base, tip)

#     # 方向标签
#     direction = ""
#     if abs(force_vec[0]) > 0:
#         direction += "x"
#     if abs(force_vec[1]) > 0:
#         direction += "y"
#     if abs(force_vec[2]) > 0:
#         direction += "z"

#     return TestResult(direction, F_norm, disp, k_eff, bend_angle)


# # ----------------------------
# # Main
# # ----------------------------

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--xml", type=str, default="spine_only.xml",
#                         help="Path to MuJoCo XML file of the spine.")
#     parser.add_argument("--base_body", type=str, default="vertebrae_0",
#                         help="Name of fixed/base body.")
#     parser.add_argument("--tip_body", type=str, default="vertebrae_1",
#                         help="Name of free/tip body (tip segment).")
#     parser.add_argument("--geom", type=str, default="rear_spacer",
#                         help="Name of extension geom to apply load "
#                              "(e.g., rear_spacer or front_spacer).")
#     parser.add_argument("--force", type=float, default=10.0,
#                         help="Force magnitude in Newtons.")
#     args = parser.parse_args()

#     print(f"[Loading] {args.xml}")
#     model = mj.MjModel.from_xml_path(args.xml)
#     data = mj.MjData(model)

#     # 先在重力下收敛一次，得到基准姿态
#     print("[Settling under gravity]")
#     settle(model, data, 3000)

#     # 保存基准的 qpos / qvel，后面每个方向都从同一个状态出发
#     qpos0 = data.qpos.copy()
#     qvel0 = data.qvel.copy()

#     # 测试 +x, +y, +z 三个方向
#     forces = [
#         np.array([args.force, 0.0, 0.0]),
#         np.array([0.0, args.force, 0.0]),
#         np.array([0.0, 0.0, args.force]),
#     ]

#     results = []
#     for f in forces:
#         # 每个方向之前都把状态 reset 回“重力收敛基准姿态”
#         data.qpos[:] = qpos0
#         data.qvel[:] = qvel0
#         mj.mj_forward(model, data)  # 让 MuJoCo 更新派生量（xpos/xmat/...）

#         print(f"\n=== Testing load {f} N ===")
#         res = run_direction_test(
#             model,
#             data,
#             base_body=args.base_body,
#             tip_body=args.tip_body,
#             load_geom_name=args.geom,
#             xml_path=args.xml,
#             force_vec=f,
#             settle_steps=4000,
#         )
#         results.append(res)

#         print(f"Direction: {res.direction}")
#         print(f"  |F|      = {res.force:.3f} N")
#         print(f"  disp     = {res.disp}")
#         print(f"  |disp|   = {np.linalg.norm(res.disp):.6f} m")
#         print(f"  k_eff    = {res.k_eff:.2f} N/m")
#         print(f"  bend_ang = {res.bend_angle_deg:.3f} deg")

#     print("\n====== Summary ======")
#     for r in results:
#         print(
#             f"{r.direction:>2}: k_eff = {r.k_eff:8.2f} N/m, "
#             f"bend = {r.bend_angle_deg:7.3f} deg, "
#             f"|disp| = {np.linalg.norm(r.disp):.6f} m"
#         )


# if __name__ == "__main__":
#     main()

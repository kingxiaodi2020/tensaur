# from dm_control import mujoco
# import numpy as np

# # 加载模型
# model_path = "/media/di/4441-E469/tensaur-main/src/tensegrity_playground/envs/tensaur/xmls/scene_tensegrity_quadruped.xml"
# physics = mujoco.Physics.from_xml_path(model_path)

# legs = ["fr", "fl", "rr", "rl"]
# sensor_names = [f"{leg}_foot_touch" for leg in legs]

# # 预取每个 touch 传感器的 (adr, dim)
# touch_spec = {}
# for name in sensor_names:
#     sid = physics.model.sensor(name).id               # 传感器ID（不是数据索引）
#     adr = int(physics.model.sensor_adr[sid])          # 在 sensordata 里的起始地址
#     dim = int(physics.model.sensor_dim[sid])          # 维度（touch=1）
#     touch_spec[name] = (adr, dim)

# # 运行模拟
# physics.reset()

# for name in ["joint_vertebrae_0", "joint_vertebrae_1", "joint_vertebrae_2"]:
#     adr = int(physics.named.model.jnt_qposadr[name])  # 该关节在 qpos 的起始索引
#     physics.data.qpos[adr + 2] -= 0.01  # z -= 1cm
# physics.forward()


# # for _ in range(10):  # 模拟200步
# #     physics.step()
# #     print("ncon =", int(physics.data.ncon))
# #     for i in range(int(physics.data.ncon)):
# #         c = physics.data.contact[i]
# #     print("  contact:", physics.model.id2name(c.geom1, 'geom'),
# #           "<->", physics.model.id2name(c.geom2, 'geom'))

# for _ in range(10000):  # 模拟100步
#     # 随机生成控制输入
#     random_control = np.random.uniform(-1, 1, size=physics.model.nu)  # nu 是控制输入的维度
#     physics.data.ctrl[:] = random_control  # 应用随机控制输入
#     physics.step()
    
#     # 正确读取 touch 数值
#     for name in sensor_names:
#         adr, dim = touch_spec[name]
#         val = physics.data.sensordata[adr:adr+dim]
#         print(f"{name}: {float(val[0])}")

#     print("---")

import mujoco
import numpy as np

# 加载模型
model = mujoco.MjModel.from_xml_path(r"/media/di/4441-E469/tensaur-main/PleurobotII/pleurobot_tensegrity_spine.xml")
data = mujoco.MjData(model)
total_mass = 0

# 遍历所有部件，打印质量
for i in range(model.nbody):
    # 从名称字节数组中获取字符串
    name_adr = model.name_bodyadr[i]
    
    # 从字节流中提取名称直到遇到 null 字符
    end_idx = model.names.find(b'\0', name_adr)
    if end_idx == -1:  # 没找到 null 终止符，取到末尾
        body_name = model.names[name_adr:].decode('utf-8')
    else:
        body_name = model.names[name_adr:end_idx].decode('utf-8')
    
    body_mass = model.body_mass[i]

    total_mass += body_mass

    print(f"Body: {body_name}, Mass: {body_mass}")

print(f"Total Mass: {total_mass}")

# import mujoco as mj
# import numpy as np

# xml = "/media/di/4441-E469/tensaur-main/src/tensegrity_playground/envs/tensaur/xmls/scene_tensegrity_quadruped.xml"

# m = mj.MjModel.from_xml_path(xml)
# d = mj.MjData(m)

# # 如果要读 keyframe 姿态，先应用 keyframe，再 forward
# # kid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_KEY, "stable_pose")
# # if kid >= 0: mj.mj_resetDataKeyframe(m, d, kid)

# mj.mj_forward(m, d)

# # 找到 sensor 的索引与切片区间
# sid = mj.mj_name2id(m, mj.mjtObj.mjOBJ_SENSOR, "position")
# adr = m.sensor_adr[sid]            # 起始索引
# dim = m.sensor_dim[sid]            # = 3 for framepos

# pos_world = np.array(d.sensordata[adr:adr+dim], copy=True)
# print("framepos(position) =", pos_world)

# # 可做一致性校验：和 site_xpos 对比应一致
# site_id = mj.mj_name2id(m, mj.mjtObj.mjOBJ_SITE, "com_vertebrae_1")
# print("site_xpos =", d.site_xpos[site_id])

# import mujoco
# import jax.numpy as jp

# # 加载模型
# model_path = "/media/di/4441-E469/tensaur-main/src/tensegrity_playground/envs/tensaur/xmls/scene_tensegrity_quadruped.xml"
# mj_model = mujoco.MjModel.from_xml_path(model_path)
# mj_data = mujoco.MjData(mj_model)

# # 提取关节的 qpos 索引
# pos_idx = []
# for jname in [
#     "fr_hip_roll", "fr_hip_pitch", "fr_knee",
#     "fl_hip_roll", "fl_hip_pitch", "fl_knee",
#     "rr_hip_roll", "rr_hip_pitch", "rr_knee",
#     "rl_hip_roll", "rl_hip_pitch", "rl_knee"
# ]:
#     # 获取关节的 ID
#     j_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, jname)
#     if j_id == -1:
#         raise ValueError(f"Joint '{jname}' not found in the model.")
    
#     # 获取关节在 qpos 中的起始索引
#     pos_idx.append(mj_model.jnt_qposadr[j_id])

# # 转换为 JAX 数组
# legs_qpos_idx = jp.array(pos_idx, dtype=int)

# # 打印结果
# print("Legs qpos indices:", legs_qpos_idx)

# import mujoco
# import numpy as np

# # 加载模型
# model_path = "/media/di/4441-E469/tensaur-main/src/tensegrity_playground/envs/tensaur/xmls/scene_tensegrity_quadruped.xml"
# m = mujoco.MjModel.from_xml_path(model_path)
# d = mujoco.MjData(m)

# # 关节列表（四条腿的 12 个关节）
# leg_joints = [
#     "fr_hip_roll", "fr_hip_pitch", "fr_knee",
#     "fl_hip_roll", "fl_hip_pitch", "fl_knee",
#     "rr_hip_roll", "rr_hip_pitch", "rr_knee",
#     "rl_hip_roll", "rl_hip_pitch", "rl_knee",
# ]

# # 统计以 vertebrae_ 命名的 free joints 数量，以及它们在 qvel 中的占用区间
# free_joint_names = []
# free_qvel_spans = []  # [(start, len=6)]
# for i in range(m.njnt):
#     jname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)
#     if jname is None:
#         continue
#     # 通过 joint type 判断是否 free
#     if m.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
#         # 进一步用 body 名字是否以 vertebrae_ 开头来过滤
#         b = m.jnt_bodyid[i]
#         bname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)
#         if bname and bname.startswith("vertebrae_"):
#             free_joint_names.append(jname)
#             qvel_start = m.jnt_dofadr[i]  # free joint 在 nv 的起始 dof 下标
#             free_qvel_spans.append((qvel_start, 6))

# print(f"Total nv = {m.nv}")
# print("Free joints (vertebrae_*) in qvel (start, len=6):")
# for n, (s, l) in zip(free_joint_names, free_qvel_spans):
#     print(f"  {n:>20s}: start={s}, span={l}, indices=[{s}..{s+l-1}]")

# # 获取腿关节在 qvel 中的起始索引
# legs_qvel_idx = []
# for jname in leg_joints:
#     j_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jname)
#     if j_id == -1:
#         raise ValueError(f"Joint '{jname}' not found.")
#     qvel_start = m.jnt_dofadr[j_id]  # 1-DoF 关节在 qvel 中占 1 个元素
#     legs_qvel_idx.append(qvel_start)

# legs_qvel_idx = np.array(legs_qvel_idx, dtype=int)
# print("\nLeg joints qvel indices (order matches leg_joints):")
# print(legs_qvel_idx)

# # 打印对应的 qvel 值（当前 data，初始可能为 0）
# print("\nCurrent qvel values for leg joints:")
# print(d.qvel[legs_qvel_idx])

# # 可选：让模型前进若干步，产生非零速度后再打印一次
# for _ in range(10):
#     mujoco.mj_step(m, d)

# print("\nqvel values after 10 simulation steps:")
# print(d.qvel[legs_qvel_idx])

# import mujoco
# import numpy as np

# # 加载模型
# model_path = "/media/di/4441-E469/tensaur-main/src/tensegrity_playground/envs/tensaur/xmls/scene_tensegrity_quadruped.xml"
# mj_model = mujoco.MjModel.from_xml_path(model_path)
# mj_data = mujoco.MjData(mj_model)

# # 定义四足机器人的关节名称
# leg_joint_names = [
#     "fr_hip_roll", "fr_hip_pitch", "fr_knee",
#     "fl_hip_roll", "fl_hip_pitch", "fl_knee",
#     "rr_hip_roll", "rr_hip_pitch", "rr_knee",
#     "rl_hip_roll", "rl_hip_pitch", "rl_knee",
# ]

# # 初始化索引列表
# legs_qpos_idx = []  # qpos 索引（每个 1-DoF 关节占 1 个 qpos）
# legs_qvel_idx = []  # qvel 索引（每个 1-DoF 关节占 1 个 qvel）
# actuator_idx = []   # actuator 索引（用于对齐力矩/力）
# leg_joint_ids = []  # 关节 ID 列表

# # 遍历关节名称，提取索引
# for jname in leg_joint_names:
#     # 获取关节的 ID
#     j_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, jname)
#     if j_id == -1:
#         raise ValueError(f"Joint '{jname}' not found in the model.")
    
#     # 获取关节在 qpos 和 qvel 中的起始索引
#     legs_qpos_idx.append(mj_model.jnt_qposadr[j_id])
#     legs_qvel_idx.append(mj_model.jnt_dofadr[j_id])
#     leg_joint_ids.append(j_id)  # 关节 ID 列表

#     # 找到驱动该关节的 actuator 索引
#     found = False
#     for ia in range(mj_model.nu):  # 遍历所有 actuator
#         # trnid gives (joint_id, dof_id); when targeting a joint, trnid[ia, 0] == j_id
#         if mj_model.actuator_trnid[ia, 0] == j_id:
#             actuator_idx.append(ia)
#             found = True
#             break
#     if not found:
#         raise ValueError(f"No actuator found for joint '{jname}'")

# # 提取关节范围
# leg_joint_ranges = np.array([mj_model.jnt_range[jid] for jid in leg_joint_ids])
# self_lowers, self_uppers = mj_model.jnt_range[1:].T  # 提取下界和上界

# # 转换为 NumPy 数组
# legs_qpos_idx = np.array(legs_qpos_idx, dtype=int)
# legs_qvel_idx = np.array(legs_qvel_idx, dtype=int)
# actuator_idx = np.array(actuator_idx, dtype=int)

# # 打印结果
# print("Legs qpos indices:", legs_qpos_idx)
# print("Legs qvel indices:", legs_qvel_idx)
# print("Actuator indices:", actuator_idx)
# print("Leg joint ranges:", leg_joint_ranges)
# # print("Leg joint lower bounds:", self_lowers)
# print("Leg joint upper bounds:", self_uppers)
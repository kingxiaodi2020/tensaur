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
model = mujoco.MjModel.from_xml_path("/media/di/4441-E469/tensaur-main/src/tensegrity_playground/envs/tensaur/xmls/scene_tensegrity_quadruped.xml")
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
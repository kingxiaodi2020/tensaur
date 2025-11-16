import numpy as np
from dm_control import mjcf, mujoco
from dm_control.suite import base
import os

class DistanceMeasurement:
    """测量机器人质心到目标标记距离的工具类"""
    
    def __init__(self, xml_path):
        """
        初始化距离测量器
        
        Args:
            xml_path: MuJoCo XML文件路径
        """
        # 加载模型
        if isinstance(xml_path, str):
            with open(xml_path, 'r') as f:
                xml_string = f.read()
        else:
            xml_string = xml_path
            
        self.model = mujoco.MjModel.from_xml_string(xml_string)
        self.data = mujoco.MjData(self.model)
        
        # 获取传感器和目标位置
        self._setup_sensors_and_target()
    
    def _setup_sensors_and_target(self):
        """设置传感器索引和目标位置"""
        # 查找位置传感器
        try:
            self.position_sensor_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SENSOR, 'position'
            )
        except:
            raise ValueError("找不到名为 'position' 的传感器")
        
        # 从XML中提取目标位置（target_marker的位置）
        # 根据generate_rod.py中的设置，目标在15倍身长处
        GO1_HIP_TO_HIP_LENGTH = 2 * 0.1881  # = 0.3762 m
        target_distance_bl = 15.0
        target_x = target_distance_bl * GO1_HIP_TO_HIP_LENGTH
        
        self.target_position = np.array([target_x, 0.0, 0.3])
        print(f"目标位置设定为: {self.target_position}")
    
    def get_robot_position(self):
        """
        获取机器人当前位置（通过position传感器）
        
        Returns:
            numpy.ndarray: 机器人质心的3D位置 [x, y, z]
        """
        # 执行一步仿真以更新传感器数据
        mujoco.mj_step(self.model, self.data)
        
        # 读取位置传感器数据
        sensor_start = self.model.sensor_adr[self.position_sensor_id]
        sensor_data = self.data.sensordata[sensor_start:sensor_start + 3]
        
        return sensor_data.copy()
    
    def calculate_distance_to_target(self):
        """
        计算机器人质心到目标标记的距离
        
        Returns:
            tuple: (距离, 机器人位置, 目标位置)
        """
        robot_pos = self.get_robot_position()
        distance = np.linalg.norm(robot_pos - self.target_position)
        
        return distance, robot_pos, self.target_position
    
    def calculate_2d_distance_to_target(self):
        """
        计算机器人质心到目标标记的2D距离（忽略z轴）
        
        Returns:
            tuple: (2D距离, x方向距离, y方向距离)
        """
        robot_pos = self.get_robot_position()
        
        # 只考虑x, y方向
        robot_pos_2d = robot_pos[:2]
        target_pos_2d = self.target_position[:2]
        
        distance_2d = np.linalg.norm(robot_pos_2d - target_pos_2d)
        x_distance = self.target_position[0] - robot_pos[0]
        y_distance = self.target_position[1] - robot_pos[1]
        
        return distance_2d, x_distance, y_distance
    
    def print_distance_info(self):
        """打印详细的距离信息"""
        distance_3d, robot_pos, target_pos = self.calculate_distance_to_target()
        distance_2d, x_dist, y_dist = self.calculate_2d_distance_to_target()
        
        print(f"机器人位置: [{robot_pos[0]:.3f}, {robot_pos[1]:.3f}, {robot_pos[2]:.3f}]")
        print(f"目标位置: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")
        print(f"3D距离: {distance_3d:.3f} m")
        print(f"2D距离 (水平): {distance_2d:.3f} m")
        print(f"X方向距离: {x_dist:.3f} m ({'前进' if x_dist > 0 else '后退'})")
        print(f"Y方向距离: {y_dist:.3f} m ({'左转' if y_dist > 0 else '右转'})")
        print("-" * 50)
        
        return {
            'distance_3d': distance_3d,
            'distance_2d': distance_2d,
            'robot_position': robot_pos,
            'target_position': target_pos,
            'x_distance': x_dist,
            'y_distance': y_dist
        }

def main():
    """示例用法"""
    # XML文件路径
    xml_path = "/media/di/4441-E469/tensaur-main/src/tensegrity_playground/envs/tensaur/xmls/scene_tensegrity_quadruped.xml"
    
    if not os.path.exists(xml_path):
        print(f"XML文件不存在: {xml_path}")
        print("请检查路径或先运行 generate_rod.py 生成XML文件")
        return
    
    try:
        # 创建距离测量器
        distance_meter = DistanceMeasurement(xml_path)
        
        print("=" * 60)
        print("机器人距离目标测量")
        print("=" * 60)
        
        # 测量初始距离
        print("初始状态:")
        distance_meter.print_distance_info()
        
        # 模拟一些运动（这里只是示例，实际使用时可以结合控制器）
        print("模拟向前移动...")
        for i in range(5):
            # 手动修改位置进行测试（实际应用中这会由控制器驱动）
            distance_meter.data.qpos[0] += 0.5  # x方向移动0.5米
            print(f"步骤 {i+1}:")
            info = distance_meter.print_distance_info()
            
            # 检查是否接近目标
            if info['distance_2d'] < 0.5:
                print("🎯 已接近目标！")
                break
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
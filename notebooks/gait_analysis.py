import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取接触数据
contact_file = r"/media/di/4441-E469/tensaur-main/notebooks/gen000_ind1_contacts.csv"
data = pd.read_csv(contact_file)

# 创建步态图
fig, axes = plt.subplots(5, 1, figsize=(14, 10), sharex=True)

# 定义腿部名称和颜色
legs = {
        'FL': {'label': 'FL', 'color': '#FFC000', 'contact_col': 'FL_contact'},
        'FR': {'label': 'FR', 'color': '#70AD47', 'contact_col': 'FR_contact'},
        'RL': {'label': 'RL', 'color': '#ED7D31', 'contact_col': 'RL_contact'},
        'RR': {'label': 'RR', 'color': '#4472C4', 'contact_col': 'RR_contact'},
        'RL': {'label': 'RL', 'color': '#ED7D31', 'contact_col': 'RL_contact'},
}

# 绘制每只脚的接触状态
for idx, (leg_key, leg_info) in enumerate(legs.items()):
    ax = axes[idx]
    contact_col = leg_info['contact_col']
    
    # 绘制接触状态 (灰色填充表示着地,白色表示摆动)
    for i in range(len(data) - 1):
        if data[contact_col].iloc[i] == 1:
            ax.axvspan(
                data['Time(s)'].iloc[i], 
                data['Time(s)'].iloc[i+1],
                facecolor='gray', 
                alpha=0.7,
                edgecolor='none'
            )
    
    # 设置y轴
    ax.set_ylim(-0.1, 1.1)
    ax.set_yticks([])
    ax.set_ylabel(leg_info['label'], fontsize=11, fontweight='bold', rotation=0, 
                  ha='right', va='center')
    
    # 添加网格
    ax.grid(True, axis='x', alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # 在左侧添加颜色条
    ax.axvspan(-0.5, 0, facecolor=leg_info['color'], alpha=0.8, clip_on=False)

# 绘制总接触足数
ax_total = axes[4]
total_contacts = (
    data['FR_contact'] + data['FL_contact'] + 
    data['RR_contact'] + data['RL_contact']
)
ax_total.plot(data['Time(s)'], total_contacts, 'k-', linewidth=2)
ax_total.fill_between(data['Time(s)'], 0, total_contacts, alpha=0.3, color='gray')
ax_total.set_ylabel('Total\nContacts', fontsize=11, fontweight='bold', 
                    rotation=0, ha='right', va='center')
ax_total.set_ylim(-0.5, 4.5)
ax_total.set_yticks([0, 1, 2, 3, 4])
ax_total.grid(True, alpha=0.3, linestyle='--')
ax_total.spines['top'].set_visible(False)
ax_total.spines['right'].set_visible(False)

# 设置x轴
ax_total.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
ax_total.set_xlim(data['Time(s)'].min(), data['Time(s)'].max())

# 添加时间箭头
arrow_y = -1.2
ax_total.annotate('', xy=(data['Time(s)'].max(), arrow_y), 
                  xytext=(data['Time(s)'].min(), arrow_y),
                  arrowprops=dict(arrowstyle='->', lw=2, color='red'))
ax_total.text(data['Time(s)'].max() * 0.98, arrow_y - 0.3, 'time', 
             fontsize=10, color='red', ha='right', va='top', style='italic')

# 添加标题
fig.suptitle('Gait Pattern Analysis', fontsize=16, fontweight='bold', y=0.995)

plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig('gait_pattern.png', dpi=300, bbox_inches='tight')
print("✅ 步态图已保存为 gait_pattern.png")
plt.show()

# ========== 步态统计分析 ==========
print("\n" + "="*60)
print("步态统计分析")
print("="*60)

dt = 0.02  # 时间步长

for leg_key, leg_info in legs.items():
    contact_col = leg_info['contact_col']
    
    # 占空比 (Duty Factor)
    duty_factor = data[contact_col].mean()
    
    # 计算步态周期
    contacts = data[contact_col].values
    touchdowns = np.where(np.diff(contacts, prepend=0) > 0)[0]
    num_steps = len(touchdowns)
    
    if num_steps > 1:
        total_time = data['Time(s)'].iloc[-1]
        step_frequency = num_steps / total_time
        stride_time = 1.0 / step_frequency if step_frequency > 0 else 0
    else:
        step_frequency = 0
        stride_time = 0
    
    # 平均接触力
    force_col = f"{leg_key}_force(N)"
    avg_force = data[force_col].mean()
    max_force = data[force_col].max()
    
    print(f"\n{leg_info['label']} ({leg_key}):")
    print(f"  占空比 (Duty Factor): {duty_factor:.1%}")
    print(f"  步数: {num_steps}")
    print(f"  步频 (Hz): {step_frequency:.2f}")
    print(f"  步长时间 (s): {stride_time:.3f}")
    print(f"  平均接触力: {avg_force:.1f} N")
    print(f"  最大接触力: {max_force:.1f} N")

# 步态对称性分析
fr_duty = data['FR_contact'].mean()
fl_duty = data['FL_contact'].mean()
rr_duty = data['RR_contact'].mean()
rl_duty = data['RL_contact'].mean()

left_right_symmetry = abs((fl_duty + rl_duty) - (fr_duty + rr_duty)) / 2
front_rear_symmetry = abs((fr_duty + fl_duty) - (rr_duty + rl_duty)) / 2

print(f"\n步态对称性:")
print(f"  左右对称性偏差: {left_right_symmetry:.1%}")
print(f"  前后对称性偏差: {front_rear_symmetry:.1%}")

# 判断步态类型
avg_contacts = total_contacts.mean()
print(f"\n平均同时接触足数: {avg_contacts:.2f}")

if avg_contacts > 3:
    gait_type = "慢走 (Walk)"
elif avg_contacts > 2:
    gait_type = "快走/对角步态 (Trot)"
elif avg_contacts > 1:
    gait_type = "跑步 (Pace/Gallop)"
else:
    gait_type = "跳跃 (Bound/Pronk)"

print(f"推测步态类型: {gait_type}")

# ========== 绘制接触力时序图 ==========
fig2, axes2 = plt.subplots(4, 1, figsize=(14, 8), sharex=True)

for idx, (leg_key, leg_info) in enumerate(legs.items()):
    ax = axes2[idx]
    force_col = f"{leg_key}_force(N)"
    
    ax.plot(data['Time(s)'], data[force_col], color=leg_info['color'], linewidth=1.5)
    ax.fill_between(data['Time(s)'], 0, data[force_col], 
                     alpha=0.3, color=leg_info['color'])
    ax.set_ylabel(f"{leg_info['label']}\nForce (N)", fontsize=10, 
                  fontweight='bold', rotation=0, ha='right', va='center')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 标注最大力
    max_force_idx = data[force_col].idxmax()
    max_force = data[force_col].iloc[max_force_idx]
    max_time = data['Time(s)'].iloc[max_force_idx]
    if max_force > 0:
        ax.plot(max_time, max_force, 'r*', markersize=10)
        ax.text(max_time, max_force * 1.1, f'{max_force:.1f}N', 
               ha='center', fontsize=8, color='red')

axes2[3].set_xlabel('Time (s)', fontsize=12, fontweight='bold')
axes2[3].set_xlim(data['Time(s)'].min(), data['Time(s)'].max())

fig2.suptitle('Foot Contact Forces', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig('contact_forces.png', dpi=300, bbox_inches='tight')
print("✅ 接触力图已保存为 contact_forces.png")
plt.show()

print("\n" + "="*60)
print("分析完成!")
print("="*60)
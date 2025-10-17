import re
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# 事件文件
event_file = "/media/di/4441-E469/base_5gens/gen_004/ind_5/251014084638-tensegrityquadrupedwalk-ppo-1/events.out.tfevents.1760424398.i04"

# 权重表（正>0 归奖励，负<0 归惩罚）
reward_weights = {
    "tracking_lin_vel": 1.0,
    "tracking_ang_vel": 0.5,
    "lin_vel_z": -0.5,
    "ang_vel_xy": -0.05,
    "orientation": -5.0,
    "dof_pos_limits": -1.0,
    "pose": 0.5,
    "termination": -1.0,
    "torques": -0.0002,
    "action_rate": -0.01,
    "energy": -0.001,
    "feet_clearance": -2.0,
    "feet_height": -0.2,
    "feet_slip": -0.1,
    "feet_air_time": 0.1,
}

# 平滑
use_smoothing = False
smoothing = "ema"
ema_alpha = 0.15
ma_window = 15

# 标记更密，带白色描边
marker_every = 12
marker_size = 5.5
marker_edge_color = "white"
marker_edge_width = 0.9

line_width = 2.0
line_alpha = 0.95

# 强区分度颜色方案（手选 hex），避免同色系明暗不易区分
# 奖励（暖/中性色相混合，色相差大）：
REWARD_COLORS = [
    "#D81B60",  # raspberry
    "#E65100",  # deep orange
    "#FDD835",  # vivid yellow
    "#8E24AA",  # purple (偏暖)
    "#F4511E",  # orange red
    "#6D4C41",  # brown
    "#FF7043",  # orange
    "#EC407A",  # pink
    "#FBC02D",  # amber
    "#AD1457",  # dark pink
]
# 惩罚（冷色系但跨色相，彼此差异大）：
PENALTY_COLORS = [
    "#1E88E5",  # blue
    "#00897B",  # teal
    "#5E35B1",  # deep purple
    "#00ACC1",  # cyan
    "#3949AB",  # indigo
    "#26A69A",  # teal light
    "#1976D2",  # blue darker
    "#7E57C2",  # purple alt
    "#039BE5",  # blue sky
    "#2E7D32",  # green (冷感)
]

# 可选：色盲友好色板（Okabe–Ito），用于奖励/惩罚各自循环
OKABE_ITO = ["#E69F00","#56B4E9","#009E73","#F0E442","#0072B2","#D55E00","#CC79A7","#000000"]
# 你可以把 REWARD_COLORS = OKABE_ITO[::2]，PENALTY_COLORS = OKABE_ITO[1::2] 试试

# sum 高亮
sum_color = "#000000"
sum_linestyle = "-."
sum_linewidth = 3.4
sum_alpha = 1.0
sum_zorder = 6
sum_glow_color = "#FFD166"
sum_glow_linewidth = 7.0
sum_glow_alpha = 0.45
sum_marker = "o"
sum_marker_size = 8.0

reward_linestyle = "-"
penalty_linestyle = "--"

reward_markers = ['o', 's', '^', 'D', 'P', '*', 'h', 'X', 'v', 'p', '8', 'd']
penalty_markers = ['x', '+', '1', '2', '3', '4', '>', '<', '|', '_', 'H', '^']

def smooth(values):
    if not use_smoothing or smoothing is None:
        return values
    values = np.asarray(values, dtype=float)
    if smoothing == "ema":
        out = []
        for v in values:
            out.append(v if not out else ema_alpha * v + (1 - ema_alpha) * out[-1])
        return np.array(out)
    elif smoothing == "ma":
        if len(values) < ma_window or ma_window <= 1:
            return values
        kernel = np.ones(ma_window) / ma_window
        return np.convolve(values, kernel, mode="valid")
    return values

def load_ea_and_tags(event_file):
    ea = EventAccumulator(event_file)
    ea.Reload()
    return ea, ea.Tags().get("scalars", [])

def get_series(ea, tag):
    events = ea.Scalars(tag)
    if not events:
        return None, None, None
    steps = np.array([e.step for e in events])
    vals = np.array([e.value for e in events], dtype=float)
    wall = np.array([e.wall_time for e in events], dtype=float)
    return steps, vals, wall

def classify_key(key, vals=None):
    if key in reward_weights:
        return "reward" if reward_weights[key] > 0 else "penalty"
    if vals is not None and len(vals):
        m = float(np.nanmean(vals))
        if m > 0: return "reward"
        if m < 0: return "penalty"
    return "unknown"

def style_picker(index, kind):
    if kind == "reward":
        color = REWARD_COLORS[index % len(REWARD_COLORS)]
        marker = reward_markers[index % len(reward_markers)]
        ls = reward_linestyle
    elif kind == "penalty":
        color = PENALTY_COLORS[index % len(PENALTY_COLORS)]
        marker = penalty_markers[index % len(penalty_markers)]
        ls = penalty_linestyle
    else:
        color = "#7f7f7f"; marker = None; ls = ":"
    return color, ls, marker

def plot_episode_reward(ea, tags):
    pattern = re.compile(r"^episode/reward/([^/]+)$")
    reward_tags = []
    for t in tags:
        m = pattern.match(t)
        if m and not t.endswith("_std"):
            reward_tags.append((t, m.group(1)))
    has_sum = "episode/sum_reward" in tags

    plt.figure(figsize=(12, 6))
    ridx = 0; pidx = 0

    # 子项
    for full_tag, key in sorted(reward_tags, key=lambda x: x[1]):
        steps, vals, _ = get_series(ea, full_tag)
        if steps is None: continue
        y = smooth(vals)
        x = steps if len(y) == len(steps) else steps[:len(y)]
        kind = classify_key(key, vals)
        color, ls, mk = style_picker((ridx if kind=="reward" else pidx), kind)
        if kind == "reward": ridx += 1
        elif kind == "penalty": pidx += 1
        plt.plot(
            x, y,
            label=f"episode/reward/{key}",
            color=color, linestyle=ls, linewidth=line_width, alpha=line_alpha,
            marker=mk, markevery=marker_every, markersize=marker_size,
            markeredgecolor=marker_edge_color, markeredgewidth=marker_edge_width,
            zorder=3
        )

    # sum：背光 + 主体
    if has_sum:
        steps, vals, _ = get_series(ea, "episode/sum_reward")
        if steps is not None:
            y = smooth(vals)
            x = steps if len(y) == len(steps) else steps[:len(y)]
            plt.plot(x, y, color=sum_glow_color, linestyle=sum_linestyle,
                     linewidth=sum_glow_linewidth, alpha=sum_glow_alpha, zorder=sum_zorder-1)
            plt.plot(x, y, label="episode/sum_reward",
                     color=sum_color, linestyle=sum_linestyle,
                     linewidth=sum_linewidth, alpha=sum_alpha, zorder=sum_zorder,
                     marker=sum_marker, markevery=max(1, marker_every//2),
                     markersize=sum_marker_size, markeredgecolor="white", markeredgewidth=1.1)

    plt.title("Episode Rewards")
    plt.xlabel("step"); plt.ylabel("value")
    plt.grid(True, alpha=0.25)
    plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0., fontsize=9)
    plt.tight_layout()
    plt.show()

def plot_eval_episode_reward(ea, tags, include_children=True):
    pattern = re.compile(r"^eval/episode_reward(?:/([^/]+))?$") if include_children else re.compile(r"^eval/episode_reward$")
    eval_tags = []
    for t in tags:
        m = pattern.match(t)
        if m and not t.endswith("_std"):
            key = m.group(1)
            eval_tags.append((t, key))
    if not eval_tags:
        print("[warn] 未找到 eval/episode_reward（非 std）"); return

    plt.figure(figsize=(12, 6))
    ridx = 0; pidx = 0
    for full_tag, key in sorted(eval_tags, key=lambda x: ("" if x[1] is None else x[1])):
        steps, vals, _ = get_series(ea, full_tag)
        if steps is None: continue
        y = smooth(vals)
        x = steps if len(y) == len(steps) else steps[:len(y)]

        if key is None:
            plt.plot(x, y, color=sum_glow_color, linestyle=sum_linestyle,
                     linewidth=sum_glow_linewidth, alpha=sum_glow_alpha, zorder=sum_zorder-1)
            plt.plot(x, y, label="eval/episode_reward",
                     color=sum_color, linestyle=sum_linestyle,
                     linewidth=sum_linewidth, alpha=sum_alpha, zorder=sum_zorder,
                     marker=sum_marker, markevery=max(1, marker_every//2),
                     markersize=sum_marker_size, markeredgecolor="white", markeredgewidth=1.1)
            continue

        kind = classify_key(key, vals)
        color, ls, mk = style_picker((ridx if kind=="reward" else pidx), kind)
        if kind == "reward": ridx += 1
        elif kind == "penalty": pidx += 1

        plt.plot(
            x, y,
            label=f"eval/episode_reward/{key}",
            color=color, linestyle=ls, linewidth=line_width, alpha=line_alpha,
            marker=mk, markevery=marker_every, markersize=marker_size,
            markeredgecolor=marker_edge_color, markeredgewidth=marker_edge_width,
            zorder=3
        )

    plt.title("Eval Episode Rewards")
    plt.xlabel("step"); plt.ylabel("value")
    plt.yticks(np.arange(-1300, 3001, 100)) 
    plt.grid(True, alpha=0.25)
    plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0., fontsize=9)
    plt.tight_layout()
    plt.show()

def plot_eval_metrics(ea, tags):
    """Plot eval/episode_actual_vel_x and eval/avg_episode_length"""
    target_metrics = ["eval/episode_actual_vel_x", "eval/avg_episode_length"]
    found_metrics = [metric for metric in target_metrics if metric in tags]
    
    if not found_metrics:
        print("[warn] 未找到目标评估指标"); return
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    if len(found_metrics) == 1:
        axes = [axes] if len(found_metrics) == 1 and hasattr(axes, 'plot') else axes
    
    for i, metric in enumerate(found_metrics):
        ax = axes[i] if len(found_metrics) > 1 else axes[0]
        
        steps, vals, _ = get_series(ea, metric)
        if steps is None:
            continue
            
        y = smooth(vals)
        x = steps if len(y) == len(steps) else steps[:len(y)]
        
        # 根据指标类型选择颜色和样式
        if "vel_x" in metric:
            color = "#1E88E5"  # blue
            marker = "o"
            title = "Actual Velocity X"
            ylabel = "Velocity (m/s)"
        else:  # avg_episode_length
            color = "#E65100"  # orange
            marker = "s"
            title = "Episode Length"
            ylabel = "Steps"
        
        # 添加背景光晕效果
        ax.plot(x, y, color=color, alpha=0.3, linewidth=sum_glow_linewidth, zorder=1)
        
        # 主线条
        ax.plot(x, y, 
                color=color, linewidth=line_width, alpha=line_alpha,
                marker=marker, markevery=marker_every, markersize=marker_size,
                markeredgecolor=marker_edge_color, markeredgewidth=marker_edge_width,
                label=metric, zorder=3)
        
        ax.set_title(title)
        ax.set_xlabel("Step")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend()
    
    # 如果只有一个指标，隐藏第二个子图
    if len(found_metrics) == 1 and len(axes) == 2:
        fig.delaxes(axes[1])
    
    plt.tight_layout()
    plt.show()

def main():
    ea, tags = load_ea_and_tags(event_file)
    print(f"发现 {len(tags)} 个 scalar tags")
    plot_episode_reward(ea, tags)
    plot_eval_episode_reward(ea, tags, include_children=True)
    plot_eval_metrics(ea, tags)

if __name__ == "__main__":
    main()


# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np

# # 手动录入（可改为从 CSV 读取）
# rows = [
#     # Sim steps, Integrator, Num Segs, Stiffness, Damping, Alpha,beta, Alpha,beta length, Pretension, Success, Num steps
#     (0.005, "rk4", 2, 3000, 10.0,  "pi/3", 0.08, 0.8, "N", "80M"),
#     (0.005, "rk4", 2, 5000, 10.0,  "pi/3", 0.08, 0.8, "Y", "80M"),
#     (0.005, "rk4", 2, 5000, 12.5,  "pi/3", 0.08, 0.8, "Y", "80M"),
#     (0.005, "rk4", 3, 3000, 10.0,  "pi/4", 0.06, 0.8, "N", "80M"),
#     (0.005, "rk4", 3, 4000, 10.0,  "pi/3", 0.08, 0.8, "N", "80M"),
#     (0.005, "rk4", 3, 5000, 12.5,  "pi/3", 0.08, 0.8, "N", "80M"),
#     (0.005, "rk4", 4, 3000, 10.0,  "pi/4", 0.08, 0.8, "N", "80M"),
# ]
# cols = ["Sim steps","Integrator","Num Segs","Stiffness","Damping","Alpha,beta","Alpha, beta length","Pretension","Success","Num steps"]
# df = pd.DataFrame(rows, columns=cols)

# def plot_stiffness_vs_damping(df, x_col='Damping', y_col='Stiffness',
#                               seg_col='Num Segs', succ_col='Success',
#                               title='Stiffness vs Damping'):
#     d = df.copy()
#     # 类型转换
#     for c in [x_col, y_col, seg_col]:
#         d[c] = pd.to_numeric(d[c], errors='coerce')
#     d[succ_col] = d[succ_col].astype(str).str.strip().str.upper()
#     d = d.dropna(subset=[x_col, y_col, seg_col, succ_col])

#     # 符号映射
#     seg_marker = {2: 'o', 3: '^', 4: 's'}  # 圆点/三角/方块
#     succ_color = {'Y': 'green', 'N': 'red'}

#     plt.figure(figsize=(7,5), dpi=140)
#     for seg, marker in seg_marker.items():
#         for succ, color in succ_color.items():
#             m = (d[seg_col] == seg) & (d[succ_col] == succ)
#             if m.any():
#                 sub = d[m]
#                 plt.scatter(
#                     sub[x_col], sub[y_col],
#                     marker=marker, c=color, s=80,
#                     edgecolors='k', linewidths=0.5,
#                     label=f'Num Segs={seg}, Success={succ}'
#                 )

#     plt.xlabel('Damping')
#     plt.ylabel('Stiffness')
#     plt.title(title)
#     plt.grid(True, ls='--', alpha=0.3)
#     # 去重图例条目
#     handles, labels = plt.gca().get_legend_handles_labels()
#     uniq = dict(zip(labels, handles))
#     plt.legend(uniq.values(), uniq.keys(), ncol=2)
#     plt.tight_layout()
#     plt.show()

# plot_stiffness_vs_damping(df, title='Simsteps=0.005, RK4')
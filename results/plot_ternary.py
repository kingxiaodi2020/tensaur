#!/usr/bin/env python3
"""
3D scatter plot for spine stiffness grid search results with NaN visualization.

Usage:
    python plot_ternary.py --type dy    # lateral displacement (Fy)
    python plot_ternary.py --type dz    # vertical displacement (Fz)
    python plot_ternary.py --type Tx    # torsion (Torque x)
"""

import argparse
import pathlib
import csv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go

import scienceplots
plt.style.use(['science','ieee'])
plt.rcParams['font.family'] = 'Times New Roman Bold'
ROOT = pathlib.Path(__file__).resolve().parent.parent


def load_data(data_type: str):
    """Load data based on type selection."""
    if data_type == "dy":
        csv_path = ROOT / "results" / "log_500_Fy10.csv"
        response_key = "tip_dy"
        label = "Y-direction Displacement dy (m)"
        label_html = "Lateral Displacement d<sub>y</sub> (m)"
    elif data_type == "dz":
        csv_path = ROOT / "results" / "log_500_Fz40.csv"
        response_key = "tip_dz"
        label = "Z-direction Displacement dz (m)"
        label_html = "Sagittal Displacement d<sub>z</sub> (m)"
    elif data_type == "Tx":
        csv_path = ROOT / "results" / "log_500_Tx5.csv"
        response_key = "total_twist_deg"
        label = "Rotational Displacement dθ (deg)"
        label_html = "Rotational Displacement dθ (deg)"
    else:
        raise ValueError(f"Unknown data type: {data_type}. Choose 'dy', 'dz', or 'Tx'.")
    
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    
    if not rows:
        raise ValueError(f"No data found in {csv_path}")
    
    # Extract data，保留NaN
    lh = np.array([float(r["lateral_hori_stiffness"]) for r in rows])
    lv = np.array([float(r["lateral_verti_stiffness"]) for r in rows])
    dg = np.array([float(r["diagonal_stiffness"]) for r in rows])
    
    response = []
    for r in rows:
        try:
            val = float(r[response_key])
            response.append(val)
        except (ValueError, TypeError):
            response.append(np.nan)
    response = np.array(response)
    
    return lh, lv, dg, response, label, label_html, csv_path.name

def add_slice_plane_and_border(fig, slice_axis, slice_value,
                               x_min, x_max, y_min, y_max, z_min, z_max,
                               expand=0.05, color="#8B0000"):
    """
    在 3D 图里加一个切片：
    - 灰色半透明平面
    - 稍微比点云外扩一点的红色边框
    expand: 外扩比例（对 log 轴用乘法更自然）
    """
    # ✅ 添加标签映射
    axis_label_map = {
        'lh': 'Left/Right',
        'lv': 'Top/Bottom',
        'dg': 'Diagonal'
    }
    label_text = axis_label_map.get(slice_axis, slice_axis)

    if slice_axis == 'lv':
        # plane: y = slice_value
        xs_plane = [x_min, x_max, x_max, x_min]
        zs_plane = [z_min, z_min, z_max, z_max]
        fig.add_trace(go.Mesh3d(
            x=xs_plane + xs_plane,
            y=[slice_value] * 8,
            z=zs_plane + zs_plane,
            color='rgba(200,200,200,0.2)',
            opacity=0.2,
            showscale=False
        ))

        # 红色边框：在 x,z 方向外扩一点
        x1 = x_min / (1 + expand)
        x2 = x_max * (1 + expand)
        z1 = z_min / (1 + expand)
        z2 = z_max * (1 + expand)

        xs_border = [x1, x2, x2, x1]
        zs_border = [z1, z1, z2, z2]
        ys_border = [slice_value] * 4

        fig.add_trace(go.Scatter3d(
            x=xs_border + [xs_border[0]],
            y=ys_border + [ys_border[0]],
            z=zs_border + [zs_border[0]],
            mode='lines',
            line=dict(color=color, width=10, dash='solid'),
            showlegend=False,
            name=f"{label_text} = {slice_value}",
        ))

    elif slice_axis == 'lh':
        # plane: x = slice_value
        ys_plane = [y_min, y_max, y_max, y_min]
        zs_plane = [z_min, z_min, z_max, z_max]
        fig.add_trace(go.Mesh3d(
            x=[slice_value] * 8,
            y=ys_plane + ys_plane,
            z=zs_plane + zs_plane,
            color='rgba(200,200,200,0.2)',
            opacity=0.2,
            showscale=False
        ))

        # 红色边框：在 y,z 方向外扩
        y1 = y_min / (1 + expand)
        y2 = y_max * (1 + expand)
        z1 = z_min / (1 + expand)
        z2 = z_max * (1 + expand)

        ys_border = [y1, y2, y2, y1]
        zs_border = [z1, z1, z2, z2]
        xs_border = [slice_value] * 4

        fig.add_trace(go.Scatter3d(
            x=xs_border + [xs_border[0]],
            y=ys_border + [ys_border[0]],
            z=zs_border + [zs_border[0]],
            mode='lines',
            line=dict(color=color, width=10, dash='solid'),
            showlegend=False,
            name=f"{label_text} = {slice_value}",
        ))

    elif slice_axis == 'dg':
        # plane: z = slice_value
        xs_plane = [x_min, x_max, x_max, x_min]
        ys_plane = [y_min, y_min, y_max, y_max]
        fig.add_trace(go.Mesh3d(
            x=xs_plane + xs_plane,
            y=ys_plane + ys_plane,
            z=[slice_value] * 8,
            color='rgba(200,200,200,0.2)',
            opacity=0.2,
            showscale=False
        ))

        # 红色边框：在 x,y 方向外扩
        x1 = x_min / (1 + expand)
        x2 = x_max * (1 + expand)
        y1 = y_min / (1 + expand)
        y2 = y_max * (1 + expand)

        xs_border = [x1, x2, x2, x1]
        ys_border = [y1, y1, y2, y2]
        zs_border = [slice_value] * 4

        fig.add_trace(go.Scatter3d(
            x=xs_border + [xs_border[0]],
            y=ys_border + [ys_border[0]],
            z=zs_border + [zs_border[0]],
            mode='lines',
            line=dict(color=color, width=10, dash='solid'),
            showlegend=False,
            name=f"{label_text} = {slice_value}",
        ))

def plot_3d_scatter_with_nan(lh, lv, dg, response, label, label_html, title,
                             vmin=None, vmax=None,
                             slice_specs=None,
                             slice_axis=None, slice_value=None):
    """Create 3D scatter plot with NaN visualization using Plotly.

    slice_specs: list[("lh"/"lv"/"dg", value), ...]
                 例如 [("lh",1000), ("lv",2000), ("dg",5000)]
    旧参数 slice_axis/slice_value 仍然支持（会自动转成单个 slice_specs）
    """
    # 字体大小参数
    TITLE_FONT_SIZE = 20
    AXIS_FONT_SIZE = 1  
    TICK_FONT_SIZE = 20
    COLORBAR_TITLE_SIZE = 30
    FONT_FAMILY = 'Times New Roman, serif'
    
    color_values = np.abs(response)

    fig = go.Figure(data=[go.Scatter3d(
        x=lh,
        y=lv,
        z=dg,
        mode='markers',
        showlegend=False,
        marker=dict(
            size=6,
            color=color_values,
            cmin=vmin,
            cmax=vmax,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(
                title=dict(
                    text=label_html,
                    font=dict(size=COLORBAR_TITLE_SIZE, family=FONT_FAMILY, color='black')
                ),
                tickfont=dict(size=TICK_FONT_SIZE, family=FONT_FAMILY, color='black'),
                tickmode='array',
                tickvals=np.linspace(vmin, vmax, 6),  # 6个刻度：最小值 + 4个中间值 + 最大值
                ticktext=[f'{v:.2f}' for v in np.linspace(vmin, vmax, 6)],
                thickness=20,
                len=0.7
            ),
            opacity=1.0,
            line=dict(color='white', width=0.5)
        ),
        text=[f"Left/Right Stiffness: {x:.2f}<br>"
              f"Top/Bottom Stiffness: {y:.2f}<br>"
              f"Diagonal Stiffness: {z:.2f}<br>"
              f"Response: {c:.4f}"
              for x, y, z, c in zip(lh, lv, dg, color_values)],
        hoverinfo='text'
    )])

    slice_colors = [
        "#FF0303",  # 亮蓝 (Material Light Blue)
        "#000000",  # 亮蓝 (Material Light Blue)
        "#FFE23B",  # 亮蓝 (Material Light Blue)
    ]

    # 统一数据范围
    x_min, x_max = float(np.min(lh)), float(np.max(lh))
    y_min, y_max = float(np.min(lv)), float(np.max(lv))
    z_min, z_max = float(np.min(dg)), float(np.max(dg))

    # 兼容老接口：如果没传 slice_specs，就用单个 slice_axis/slice_value
    if slice_specs is None and slice_axis is not None and slice_value is not None:
        slice_specs = [(slice_axis, slice_value)]

    if slice_specs is not None:
        for i, (ax_name, val) in enumerate(slice_specs):
            color = slice_colors[i % len(slice_colors)]
            add_slice_plane_and_border(
                fig, ax_name, float(val),
                x_min, x_max, y_min, y_max, z_min, z_max,
                expand=0.15,
                color=color
            )


    # 后面的 tick / layout 不动……
    tick_vals = [500, 1000, 2000, 5000, 10000, 15000]
    tick_text = ['0.5k', '1k', '2k', '5k', '10k', '15k']

    fig.update_layout(
        title=dict(
            text=f"{title}",
            font=dict(size=TITLE_FONT_SIZE, family=FONT_FAMILY, color='black'),
            x=0.5, 
            xanchor='center'
        ),
        scene=dict(
            xaxis=dict(
                # title=dict(
                #     text="<br><br><br>Left/Right Stiffness (N/m)<br>k<sub>left/right</sub>",
                #     font=dict(size=AXIS_FONT_SIZE, family=FONT_FAMILY, color='black')
                # ),
                type="log",
                tickvals=tick_vals,
                ticktext=tick_text,
                tickfont=dict(size=TICK_FONT_SIZE, family=FONT_FAMILY, color='black'),
                backgroundcolor="rgb(240, 240, 240)",
                gridcolor="white",
                showbackground=True,
            ),
            yaxis=dict(
                # title=dict(
                #     text="<br><br><br>Top/Bottom Stiffness (N/m)<br>k<sub>top/bottom</sub>",
                #     font=dict(size=AXIS_FONT_SIZE, family=FONT_FAMILY, color='black')
                # ),
                type="log",
                tickvals=tick_vals,
                ticktext=tick_text,
                tickfont=dict(size=TICK_FONT_SIZE, family=FONT_FAMILY, color='black'),
                backgroundcolor="rgb(240, 240, 240)",
                gridcolor="white",
                showbackground=True,
            ),
            zaxis=dict(
                # title=dict(
                #     text="<br><br><br>Diagonal Stiffness (N/m)<br>k<sub>diagonal</sub>",
                #     font=dict(size=AXIS_FONT_SIZE, family=FONT_FAMILY, color='black')
                # ),
                type="log",
                tickvals=tick_vals,
                ticktext=tick_text,
                tickfont=dict(size=TICK_FONT_SIZE, family=FONT_FAMILY, color='black'),
                backgroundcolor="rgb(240, 240, 240)",
                gridcolor="white",
                showbackground=True,
            ),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.3)
            )
        ),
        width=1800,
        height=1000,
        hovermode='closest',
        showlegend=False,
        paper_bgcolor='white',
        plot_bgcolor='white'
    )

    return fig


def plot_slice_2d(lh, lv, dg, response, label, slice_axis: str, slice_value: float, 
                  tol: float = 1e-6, vmin=None, vmax=None, data_type: str = "Tx", save_path=None):
    """Plot a 2D cross-section with smooth heatmap background."""
    # 字体设置
    plt.rcParams['font.family'] = 'Times New Roman'
    
    # 字体大小参数
    TITLE_FONT_SIZE = 16
    LABEL_FONT_SIZE = 14
    TICK_FONT_SIZE = 12
    
    from scipy.interpolate import griddata
    
    response_abs = np.abs(response)

    axis_map = {
        'lh': lh,
        'lv': lv,
        'dg': dg,
    }

    if slice_axis not in axis_map:
        raise ValueError("slice_axis must be one of 'lh', 'lv', 'dg'")

    fixed = axis_map[slice_axis]

    # mask where fixed axis equals slice_value (within tol)
    mask_fixed = np.abs(fixed - slice_value) <= tol
    if not np.any(mask_fixed):
        idx_closest = np.argmin(np.abs(fixed - slice_value))
        closest_val = float(fixed[idx_closest])
        long_axis = {'lh': 'Left/Right Stiffness', 'lv': 'Top/Bottom Stiffness', 'dg': 'Diagonal Stiffness'}[slice_axis]
        print(f"[Warning] No exact match for {long_axis}={slice_value:.6g}. Using closest value {closest_val:.6g} instead.")
        mask_fixed = np.abs(fixed - closest_val) <= tol

    rem_axes = [a for a in ('lh', 'lv', 'dg') if a != slice_axis]
    a1_name, a2_name = rem_axes
    a1 = axis_map[a1_name][mask_fixed]
    a2 = axis_map[a2_name][mask_fixed]
    resp = response_abs[mask_fixed]

    if a1.size == 0 or a2.size == 0:
        raise ValueError(f"No data for slice {slice_axis}={slice_value}")

    # ✅ 打印调试信息
    print(f"\n[Debug] 2D Slice at {slice_axis}={slice_value}")
    print(f"  Global vmin={vmin:.6f}, vmax={vmax:.6f}")
    print(f"  Slice data range: [{np.nanmin(resp):.6f}, {np.nanmax(resp):.6f}]")

    title_label_map = {
        'lh': r'$k_{\mathrm{left/right}}$',
        'lv': r'$k_{\mathrm{top/bottom}}$',
        'dg': r'$k_{\mathrm{diagonal}}$'
    }

    axis_label_map = {
        'lh': r'$k_{\mathrm{left/right}}$ (N/m)',
        'lv': r'$k_{\mathrm{top/bottom}}$ (N/m)',
        'dg': r'$k_{\mathrm{diagonal}}$ (N/m)'
    }

    # ✅ 创建高分辨率网格用于插值背景
    a1_log = np.log10(a1)
    a2_log = np.log10(a2)
    
    a1_grid = np.logspace(np.log10(a1.min()), np.log10(a1.max()), 300)
    a2_grid = np.logspace(np.log10(a2.min()), np.log10(a2.max()), 300)
    A1, A2 = np.meshgrid(a1_grid, a2_grid)
    
    # 用 cubic 插值填充背景
    points = np.column_stack([a1_log, a2_log])
    values = resp
    grid_points = np.column_stack([np.log10(A1.ravel()), np.log10(A2.ravel())])
    Z = griddata(points, values, grid_points, method='cubic')
    Z = Z.reshape(A1.shape)
    
    # ✅ 关键修复：裁剪插值结果到全局范围
    Z = np.clip(Z, vmin, vmax)
    
    print(f"  Interpolated Z range (after clipping): [{np.nanmin(Z):.6f}, {np.nanmax(Z):.6f}]")
    
    # 绘图
    fig, ax = plt.subplots(1, 1)
    
    # ✅ 明确指定 levels 范围，确保不超出 vmin/vmax
    contour_levels = np.linspace(vmin, vmax, 26)  # 25个间隔 = 26个边界
    
    cf = ax.contourf(
        A1, A2, Z, 
        levels=contour_levels,
        cmap='viridis',
        vmin=vmin,
        vmax=vmax,
        alpha=0.85,
        extend='neither'
    )
    
    # # 2) 可选：加轮廓线显示 transition
    # contour_lines = ax.contour(
    #     A1, A2, Z,
    #     levels=8,
    #     colors='white',
    #     alpha=0.3,
    #     linewidths=0.5
    # )
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(axis_label_map[a1_name], fontsize=LABEL_FONT_SIZE, fontweight=900)
    ax.set_ylabel(axis_label_map[a2_name], fontsize=LABEL_FONT_SIZE, fontweight=900)
    ax.set_title(
        f"{title_label_map[a1_name]} vs {title_label_map[a2_name]}\n"
        f"when {title_label_map[slice_axis]} = {slice_value:.0f} N/m",
        fontsize=TITLE_FONT_SIZE, fontweight=900
    )

    ax.tick_params(axis='both', which='major', width=2, length=8, labelsize=TICK_FONT_SIZE)
    ax.tick_params(axis='both', which='minor', width=1.5, length=5)

    # 同时加粗刻度标签文字
    for label in ax.get_xticklabels():
        label.set_fontweight(900)
    for label in ax.get_yticklabels():
        label.set_fontweight(900)
    
    # # ✅ colorbar 刻度改成与 3D 图一致
    # cbar = fig.colorbar(cf, ax=ax, label=f"|{label}|", extend='neither')

    # # 定义各数据类型的期望刻度
    # tick_map = {
    #     'Tx': np.arange(10, 110, 10),      # 10, 20, 30, ..., 100
    #     'dy': np.array([0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14]),  # Fy
    #     'dz': np.array([0.05, 0.10, 0.15, 0.20]),  # Fz
    # }

    # desired_ticks = tick_map.get(data_type, np.linspace(vmin, vmax, 8))

    # # 只保留在 [vmin, vmax] 范围内的刻度
    # valid_ticks = desired_ticks[(desired_ticks >= vmin) & (desired_ticks <= vmax)]
    
    # # 如果没有有效刻度，回退到自动生成
    # if len(valid_ticks) == 0:
    #     valid_ticks = np.linspace(vmin, vmax, 8)
    
    # cbar.set_ticks(valid_ticks)

    # # 格式化标签
    # if data_type == 'Tx':
    #     cbar.set_ticklabels([f'{int(v)}' for v in valid_ticks])
    # else:
    #     cbar.set_ticklabels([f'{v:.2f}' for v in valid_ticks])
    
    # # 强制设置 colorbar 的数值范围
    # cbar.mappable.set_clim(vmin, vmax)
    
    ax.grid(True, alpha=0.2, linestyle='--')
    plt.tight_layout()
    
    # ✅ 自动保存为PDF和PNG
    if save_path:
        # 保存PDF
        fig.savefig(save_path, format='pdf', bbox_inches='tight', dpi=300)
        print(f"[Saved] {save_path}")
        
        # 保存PNG（替换扩展名）
        png_path = str(save_path).replace('.pdf', '.png')
        fig.savefig(png_path, format='png', bbox_inches='tight', dpi=300)
        print(f"[Saved] {png_path}")
    
    return fig

def plot_combined_2d_slices(lh, lv, dg, response, label, slice_specs, 
                            vmin=None, vmax=None, data_type: str = "Tx", save_path=None):
    """Plot multiple 2D cross-sections side by side in one figure."""
    from scipy.interpolate import griddata
    
    # 字体设置
    plt.rcParams['font.family'] = 'Times New Roman'
    
    # 定义边框颜色，与3D图切片颜色一致
    slice_colors = [
        "#000000",  # 黑色
        "#FF0303",  # 红色
        "#FFE23B",  # 黄色
    ]
    
    n_slices = len(slice_specs)
    fig, axes = plt.subplots(1, n_slices, figsize=(4, 4/2))
    
    # 确保 axes 是数组（即使只有一个子图）
    if n_slices == 1:
        axes = [axes]
    
    response_abs = np.abs(response)
    
    axis_map = {
        'lh': lh,
        'lv': lv,
        'dg': dg,
    }
    
    title_label_map = {
        'lh': r'$k_{\mathrm{left/right}}$',
        'lv': r'$k_{\mathrm{top/bottom}}$',
        'dg': r'$k_{\mathrm{diagonal}}$'
    }
    
    axis_label_map = {
        'lh': r'$k_{\mathrm{left/right}}$ (N/m)',
        'lv': r'$k_{\mathrm{top/bottom}}$ (N/m)',
        'dg': r'$k_{\mathrm{diagonal}}$ (N/m)'
    }
    
    for idx, (slice_axis, slice_value) in enumerate(slice_specs):
        ax = axes[idx]
        
        if slice_axis not in axis_map:
            raise ValueError("slice_axis must be one of 'lh', 'lv', 'dg'")
        
        fixed = axis_map[slice_axis]
        tol = 1e-6
        
        # mask where fixed axis equals slice_value (within tol)
        mask_fixed = np.abs(fixed - slice_value) <= tol
        if not np.any(mask_fixed):
            idx_closest = np.argmin(np.abs(fixed - slice_value))
            closest_val = float(fixed[idx_closest])
            long_axis = {'lh': 'Left/Right Stiffness', 'lv': 'Top/Bottom Stiffness', 'dg': 'Diagonal Stiffness'}[slice_axis]
            print(f"[Warning] No exact match for {long_axis}={slice_value:.6g}. Using closest value {closest_val:.6g} instead.")
            mask_fixed = np.abs(fixed - closest_val) <= tol
        
        rem_axes = [a for a in ('lh', 'lv', 'dg') if a != slice_axis]
        a1_name, a2_name = rem_axes
        a1 = axis_map[a1_name][mask_fixed]
        a2 = axis_map[a2_name][mask_fixed]
        resp = response_abs[mask_fixed]
        
        if a1.size == 0 or a2.size == 0:
            raise ValueError(f"No data for slice {slice_axis}={slice_value}")
        
        # 创建高分辨率网格用于插值背景
        a1_log = np.log10(a1)
        a2_log = np.log10(a2)
        
        a1_grid = np.logspace(np.log10(a1.min()), np.log10(a1.max()), 300)
        a2_grid = np.logspace(np.log10(a2.min()), np.log10(a2.max()), 300)
        A1, A2 = np.meshgrid(a1_grid, a2_grid)
        
        # 用 cubic 插值填充背景
        points = np.column_stack([a1_log, a2_log])
        values = resp
        grid_points = np.column_stack([np.log10(A1.ravel()), np.log10(A2.ravel())])
        Z = griddata(points, values, grid_points, method='cubic')
        Z = Z.reshape(A1.shape)
        
        # 裁剪插值结果到全局范围
        Z = np.clip(Z, vmin, vmax)
        
        # 明确指定 levels 范围，确保不超出 vmin/vmax
        contour_levels = np.linspace(vmin, vmax, 26)
        
        cf = ax.contourf(
            A1, A2, Z, 
            levels=contour_levels,
            cmap='viridis',
            vmin=vmin,
            vmax=vmax,
            alpha=0.85,
            extend='neither'
        )
        
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(axis_label_map[a1_name], fontweight=900)
        ax.set_ylabel(axis_label_map[a2_name], fontweight=900)
        ax.set_title(
            f"{title_label_map[a1_name]} vs {title_label_map[a2_name]}\n"
            f"when {title_label_map[slice_axis]} = {slice_value:.0f} N/m",
            fontweight=900
        )
        
        ax.tick_params(axis='both', which='major', width=2, length=8)
        ax.tick_params(axis='both', which='minor', width=1.5, length=5)
        
        # 加粗刻度标签文字
        for label in ax.get_xticklabels():
            label.set_fontweight(500)
        for label in ax.get_yticklabels():
            label.set_fontweight(500)
        
        ax.grid(True, alpha=0.2, linestyle='--')
        
        # ✅ 为每个子图添加彩色边框
        border_color = slice_colors[idx % len(slice_colors)]
        for spine in ax.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(3)
    
    # ✅ 添加共享的colorbar，放在最右边
    fig.subplots_adjust(right=0.8)  # 为colorbar留出空间
    cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(cf, cax=cbar_ax)
    
    # 设置colorbar标题，根据data_type使用合适的格式
    if data_type == 'Tx':
        cbar_label = r'Twist Angle $|d\theta|$ (deg)'
    elif data_type == 'dy':
        cbar_label = r'Lateral Deflection $|d_y|$ (m)'
    elif data_type == 'dz':
        cbar_label = r'Sagittal Deflection $|d_z|$ (m)'
    else:
        cbar_label = label
    
    cbar.set_label(cbar_label, rotation=-90, labelpad=10, fontweight=900)
    
    # 6个刻度：最小值 + 4个中间值 + 最大值
    tick_vals = np.linspace(vmin, vmax, 6)
    cbar.set_ticks(tick_vals)
    cbar.set_ticklabels([f'{v:.2f}' for v in tick_vals])
    
    # 加粗colorbar刻度标签
    for label_tick in cbar.ax.get_yticklabels():
        label_tick.set_fontweight(900)
    
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # 调整子图区域，为colorbar留空间
    
    # 自动保存为PDF和PNG
    if save_path:
        # 保存PDF
        fig.savefig(save_path, format='pdf', bbox_inches='tight', dpi=300)
        print(f"[Saved] Combined figure: {save_path}")
        
        # 保存PNG（替换扩展名）
        png_path = str(save_path).replace('.pdf', '.png')
        fig.savefig(png_path, format='png', bbox_inches='tight', dpi=300)
        print(f"[Saved] Combined figure: {png_path}")
    
    return fig

def main():
    parser = argparse.ArgumentParser(
        description="Visualize spine stiffness grid search results"
    )
    parser.add_argument(
        "--type",
        type=str,
        default="Tx",
        choices=["dy", "dz", "Tx"],
        help="Data type to plot: dy (lateral), dz (vertical), or Tx (torsion)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["3d", "2d", "both"],
        help="Plot mode: 3d scatter, 2d heatmaps, or both"
    )
    parser.add_argument(
        "--slice-axis",
        type=str,
        default=None,
        choices=['lh', 'lv', 'dg'],
        help="Fix one axis for a 2D slice: 'lh', 'lv', or 'dg'"
    )
    parser.add_argument(
        "--slice-value",
        type=float,
        default=None,
        help="Value of the fixed axis to slice at (e.g. 1000). If omitted, nearest available value is used."
    )

    parser.add_argument(
        "--slices",
        type=str,
        nargs='*',
        help="多个切片，例如: lh:1000 lv:2000 dg:3000 dg:8000"
    )

    args = parser.parse_args()
    
    # Load data
    print(f"[Loading] {args.type} data...")
    lh, lv, dg, response, label, label_html, csv_name = load_data(args.type)
    # Compute global color range from absolute responses (ignore NaN)
    response_abs_all = np.abs(response)
    if np.all(np.isnan(response_abs_all)):
        global_vmin, global_vmax = 0.0, 1.0
    else:
        global_vmin = float(np.nanmin(response_abs_all))
        global_vmax = float(np.nanmax(response_abs_all))
    
    valid_count = np.sum(~np.isnan(response))
    nan_count = np.sum(np.isnan(response))
    
    print(f"[Info] Loaded {len(lh)} data points from {csv_name}")
    print(f"[Info] Stable points: {valid_count}, Unstable (NaN) points: {nan_count}")
    if valid_count > 0:
        print(f"[Info] Response range: [{response[~np.isnan(response)].min():.6f}, "
              f"{response[~np.isnan(response)].max():.6f}]")
    
    title_map = {
        "dy": "Lateral Displacement (under Fy = 10N)",
        "dz": "Sagittal Displacement (under Fz = 40N)",
        "Tx": "Torsional Response (under Tx = 5Nm)"
    }
    title = title_map[args.type]

    # 解析切片配置：优先使用 --slices，多段；否则用老的 --slice-axis/--slice-value
    slice_specs = None
    if args.slices:
        slice_specs = []
        for token in args.slices:
            try:
                axis, val_str = token.split(":")
                axis = axis.strip()
                val = float(val_str)
                if axis not in ("lh", "lv", "dg"):
                    raise ValueError
                slice_specs.append((axis, val))
            except Exception:
                print(f"[Warning] 无法解析切片 '{token}'，格式应为 axis:value 例如 lh:1000")
    elif args.slice_axis is not None:
        slice_specs = [(args.slice_axis, args.slice_value)]
    
    # Plot based on mode
    if args.mode in ["3d", "both"]:
        print("[Plotting] 3D scatter...")
        fig_3d = plot_3d_scatter_with_nan(
            lh, lv, dg, response, label, label_html, title,
            vmin=global_vmin, vmax=global_vmax,
            slice_specs=slice_specs,  # ✅ 新参数
            slice_axis=None, slice_value=None  # 兼容位，不再使用
        )
        fig_3d.show()

    
    if args.mode in ["2d", "both"]:
        # 优先使用 --slices：对每个切片各画一张 2D 图
        if slice_specs:
            figs_slice = []  # ✅ 先存储所有图表
            
            for axis_2d, value_2d in slice_specs:
                print(f"[Plotting] 2D slice at {axis_2d}={value_2d}...")
                # 生成保存路径
                save_filename = f"slice_2d_{args.type}_{axis_2d}_{int(value_2d)}.pdf"
                save_path = ROOT / "results" / save_filename
                
                fig_slice = plot_slice_2d(
                    lh, lv, dg, response, label,
                    slice_axis=axis_2d,
                    slice_value=float(value_2d),
                    vmin=global_vmin,
                    vmax=global_vmax,
                    data_type=args.type,
                    save_path=save_path
                )
                figs_slice.append(fig_slice)
            
            # ✅ 生成合并的2D图（如果有多个切片）
            if len(slice_specs) >= 2:
                print(f"[Plotting] Combined 2D figure with {len(slice_specs)} slices...")
                combined_filename = f"slice_2d_{args.type}_combined.pdf"
                combined_save_path = ROOT / "results" / combined_filename
                
                fig_combined = plot_combined_2d_slices(
                    lh, lv, dg, response, label,
                    slice_specs=slice_specs,
                    vmin=global_vmin,
                    vmax=global_vmax,
                    data_type=args.type,
                    save_path=combined_save_path
                )
                figs_slice.append(fig_combined)
            
            # ✅ 所有图都创建好后，一次性显示
            for fig in figs_slice:
                fig.show()
            plt.show()

        # 没有 --slices 的情况下，退回老的单切片接口
        elif args.slice_axis is not None:
            if args.slice_value is None:
                axis_map = {'lh': lh, 'lv': lv, 'dg': dg}
                candidate = np.unique(axis_map[args.slice_axis])
                slice_val = float(np.median(candidate))
                print(f"[Info] No --slice-value provided; using median {args.slice_axis}={slice_val}")
            else:
                slice_val = args.slice_value
            print(f"[Plotting] 2D slice at {args.slice_axis}={slice_val}...")
            # 生成保存路径
            save_filename = f"slice_2d_{args.type}_{args.slice_axis}_{int(slice_val)}.pdf"
            save_path = ROOT / "results" / save_filename
            
            fig_slice = plot_slice_2d(
                lh, lv, dg, response, label,
                slice_axis=args.slice_axis,
                slice_value=slice_val,
                vmin=global_vmin,
                vmax=global_vmax,
                data_type=args.type,
                save_path=save_path
            )
            fig_slice.show()
            plt.show()

    
    print("[Done]")


if __name__ == "__main__":
    main()
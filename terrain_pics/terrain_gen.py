#!/usr/bin/env python3
"""
生成“顺着前进方向拉长的随机山脊”高度图 PNG，
大致形状类似你截图里的那种一条条隆起/沟壑。

用法:
    python generate_ridge_terrain_png.py
会生成:
    ridge_terrain.png
"""

import numpy as np
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    raise ImportError("需要先: pip install pillow")


def smooth_1d_along_axis(h, axis, iters=1, alpha=0.5):
    """只沿某一轴做邻域平滑（axis=0: x 方向, axis=1: y 方向）"""
    out = h.copy()
    for _ in range(iters):
        nbr = (np.roll(out, 1, axis=axis) + np.roll(out, -1, axis=axis)) / 2.0
        out = (1 - alpha) * out + alpha * nbr
    return out


def generate_ridge_heightmap(cfg: dict, output_png: Path):
    """
    cfg 结构示例见 main。
    关键参数：
      length, width     - 地形尺寸 (m)
      cell_size         - 网格分辨率 (m)
      amp               - 高度幅值 (m)
      roughness         - 噪声强度 (0~1)
      smooth_x_iters    - 沿 x 平滑次数 (越大越“顺着前进方向拉长”)
      smooth_y_iters    - 沿 y 平滑次数 (控制左右方向的圆滑程度)
      quant_step        - 高度量化步长 (m)，>0 时会产生台阶感
    """
    terr = cfg["terrain"]
    length = float(terr["length"])
    width = float(terr["width"])
    cell = float(terr["cell_size"])
    amp = float(terr["amp"])
    roughness = float(terr["roughness"])
    smooth_x_iters = int(terr["smooth_x_iters"])
    smooth_y_iters = int(terr["smooth_y_iters"])
    quant_step = float(terr["quant_step"])

    # 网格大小
    nx = int(np.ceil(length / cell))
    ny = int(np.ceil(width / cell))
    print(f"[RIDGE] grid = {nx} x {ny}, cell={cell} m")

    # 基础噪声：标准正态，缩放到 [-1,1]
    base = np.random.normal(size=(nx, ny))
    base /= np.max(np.abs(base)) + 1e-8

    # 按 roughness 缩放并乘以振幅
    heights = base * roughness * amp

    # 先沿 x 抹平很多遍 → 形成顺着 x 的长条特征
    if smooth_x_iters > 0:
        heights = smooth_1d_along_axis(heights, axis=0,
                                       iters=smooth_x_iters, alpha=0.6)

    # 再沿 y 稍微平滑一点，让 cross-section 不那么锯齿
    if smooth_y_iters > 0:
        heights = smooth_1d_along_axis(heights, axis=1,
                                       iters=smooth_y_iters, alpha=0.4)

    # 可选：高度量化，产生一层层“台阶”感
    if quant_step > 0.0:
        heights = np.round(heights / quant_step) * quant_step

    # 再限幅一下，避免极端值
    h_max = amp
    heights = np.clip(heights, -h_max, h_max)

    # 映射到 [0,255] 灰度
    h_norm = (heights + h_max) / (2 * h_max)   # 0~1
    h_u8 = np.clip(h_norm * 255.0, 0, 255).astype(np.uint8)

    # PIL 需要 (height, width) = (ny, nx)，所以转置
    img = Image.fromarray(h_u8.T, mode="L")
    output_png.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_png)
    print(f"[RIDGE] saved PNG to: {output_png}")


if __name__ == "__main__":
    # 这里是你可以调的参数
    cfg = {
        "terrain": {
            "length": 8.0,       # x 方向长度 (m)
            "width":  4.0,       # y 方向宽度 (m)
            "cell_size": 0.05,   # 网格分辨率 (m)

            "amp": 0.08,         # 高度幅值 (m)  → 最大起伏 ~ ±8cm
            "roughness": 1.0,    # 噪声强度 (0~1)

            # 关键：沿 x 平滑多一点，沿 y 平滑少一点
            "smooth_x_iters": 15,  # 沿前进方向抹很多次 → 长条 ridge
            "smooth_y_iters": 3,   # 沿左右方向稍微平滑

            # 台阶感：0.0 = 连续光滑; 0.01~0.02 = 有明显层级
            "quant_step": 0.015,
        }
    }

    out = Path("ridge_terrain.png")
    generate_ridge_heightmap(cfg, out)
    print("Done.")





# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# 生成带有横向（y方向）随机凸块的 MuJoCo 高度图 PNG
# - 左右脚接触地面高低不同（横向凸块）
# - 沿 x 轴方向凸块的高度和密度逐渐增大
# - 支持按段控制难度（0..1），难度越高凸块越高、越密集
# - 输出为 8-bit 灰度 PNG（mode='L'）

# 使用示例：
#   python terrain_gen.py --width 8 --density 2.0 \
#     --segments flat:1 ramp:10:0.0:0.8 --out terrain_8m_dense.png
# """

# import argparse
# import numpy as np
# from PIL import Image


# def smoothstep(z):
#     """平滑插值函数"""
#     return z * z * (3 - 2 * z)


# def generate_lateral_bumps(nx, ny, L, W, density_scale=1.0, height_scale=1.0, seed=4141):
#     """
#     生成横向（y方向）的随机凸块
    
#     参数:
#         nx, ny: 分辨率
#         L, W: 地形长度和宽度（米）
#         density_scale: 凸块密度系数（越大越密集）
#         height_scale: 凸块高度系数（越大越高）
#         seed: 随机种子
    
#     返回:
#         Z: [ny, nx] 高度图，范围 [0, 1]
#     """
#     rng = np.random.default_rng(seed)
#     Z = np.zeros((ny, nx), dtype=np.float32)
    
#     # ★ 根据地形面积动态调整凸块数量
#     # 基础密度：每平方米约 100 个凸块（增加密度）
#     area = L * W
#     num_bumps = int(100 * area * density_scale)
    
#     # 随机生成凸块位置（归一化坐标）
#     bump_x = rng.random(num_bumps)  # [0, 1]
#     bump_y = rng.random(num_bumps)  # [0, 1]
    
#     # ★ 凸块半径：0.08~0.20 米（增大半径，让凸块更圆润）
#     radius_min = 0.08 / min(L, W)
#     radius_max = 0.20 / min(L, W)
#     bump_radius = rng.uniform(radius_min, radius_max, num_bumps)
#     bump_height = rng.uniform(0.3, 1.0, num_bumps) * height_scale
    
#     # 创建网格坐标（归一化到 [0, 1]）
#     X = np.linspace(0, 1, nx)[None, :]  # [1, nx]
#     Y = np.linspace(0, 1, ny)[:, None]  # [ny, 1]
    
#     # 叠加所有凸块
#     for i in range(num_bumps):
#         # 计算到凸块中心的距离
#         dx = (X - bump_x[i]) * (L / W)  # 考虑长宽比
#         dy = Y - bump_y[i]
#         dist = np.sqrt(dx**2 + dy**2)
        
#         # ★ 使用更平滑的高斯曲线（增大分母，让凸块更圆润）
#         bump = bump_height[i] * np.exp(-dist**2 / (3 * bump_radius[i]**2))
#         Z += bump
    
#     return Z


# def add_perlin_like_noise(Z, nx, ny, scale=0.05, seed=0):
#     """添加细节噪声，让凸块边缘更自然"""
#     rng = np.random.default_rng(seed)
#     noise = rng.random((ny, nx)).astype(np.float32)
    
#     try:
#         from scipy.ndimage import gaussian_filter
#         noise = gaussian_filter(noise, sigma=3.0)
#     except ImportError:
#         pass
    
#     Z += scale * (noise - 0.5)
#     return Z


# def build_difficulty_envelope(x, segments):
#     """
#     生成难度包络曲线
    
#     segments: 列表，格式 [(kind, length, params...), ...]
#         - ("flat", L): 平地段，难度=0
#         - ("const", L, val): 常量难度
#         - ("ramp", L, v0, v1): 线性渐变难度
    
#     返回:
#         env: [nx] 数组，难度值 ∈ [0, 1]
#         L_total: 总长度
#     """
#     xs = np.asarray(x)
#     env = np.zeros_like(xs, dtype=np.float32)
    
#     cur = 0.0
#     for seg in segments:
#         kind = seg[0]
#         length = float(seg[1])
#         x0, x1 = cur, cur + length
        
#         if length <= 0:
#             cur = x1
#             continue
        
#         mask = (xs >= x0) & (xs < x1)
#         if not np.any(mask):
#             cur = x1
#             continue
        
#         t = (xs[mask] - x0) / max(1e-12, length)
        
#         if kind == "flat":
#             env[mask] = 0.0
#         elif kind == "const":
#             val = float(seg[2])
#             env[mask] = val
#         elif kind == "ramp":
#             v0, v1 = float(seg[2]), float(seg[3])
#             # 使用 smoothstep 使过渡平滑
#             env[mask] = v0 + (v1 - v0) * smoothstep(t)
#         else:
#             raise ValueError(f"Unknown segment kind: {kind}")
        
#         cur = x1
    
#     L_total = cur
#     env[xs >= L_total] = env[xs < L_total][-1] if np.any(xs < L_total) else 0.0
    
#     return env, L_total


# def compose_bumpy_terrain(L, W, nx, ny, segments, 
#                           base_density=1.0, base_height=1.0, 
#                           add_noise=True, seed=4141):
#     """
#     合成带横向凸块的地形
#     - 沿 x 轴：每个横截面的平均高度保持不变
#     - 沿 y 轴：左右不平整程度（凸块高度差）逐渐增大
#     """
#     # 1. 构建难度包络（逐列）
#     x = np.linspace(0.0, L, nx, endpoint=False)
#     env, L_total = build_difficulty_envelope(x, segments)
#     env = np.clip(env, 0.0, 1.0)  # [nx]
    
#     # 2. ★ 记录平地段位置
#     flat_masks = []
#     cur = 0.0
#     for seg in segments:
#         kind = seg[0]
#         length = float(seg[1])
        
#         if kind == "flat" and length > 0:
#             j0 = int(round(cur / L * nx))
#             j1 = int(round((cur + length) / L * nx))
#             flat_masks.append((j0, j1))
        
#         cur += length
    
#     # 3. ★ 生成凸块（只控制 y 方向的变化）
#     Z = np.zeros((ny, nx), dtype=np.float32)
    
#     # 计算总凸块数量
#     avg_difficulty = np.mean(env)
#     area = L * W
#     total_density = base_density * (0.5 + 2.5 * (avg_difficulty ** 1.5))
#     num_bumps = int(100 * area * total_density)
    
#     rng = np.random.default_rng(seed)
    
#     # 随机生成凸块位置
#     bump_x_norm = rng.random(num_bumps)
#     bump_y_norm = rng.random(num_bumps)
    
#     # 过滤掉落在平地段的凸块
#     bump_x_idx = (bump_x_norm * nx).astype(int)
#     bump_x_idx = np.clip(bump_x_idx, 0, nx - 1)
    
#     valid_bumps = np.ones(num_bumps, dtype=bool)
#     for j0, j1 in flat_masks:
#         in_flat = (bump_x_idx >= j0) & (bump_x_idx < j1)
#         valid_bumps &= ~in_flat
    
#     bump_x_norm = bump_x_norm[valid_bumps]
#     bump_y_norm = bump_y_norm[valid_bumps]
#     bump_x_idx = bump_x_idx[valid_bumps]
    
#     bump_difficulty = env[bump_x_idx]
    
#     # ★ 凸块高度：根据难度变化，但后面会减去每列均值
#     # 高度范围：0.5 -> 3.0（前后都有起伏）
#     num_valid = len(bump_x_norm)
#     bump_height = rng.uniform(0.5, 1.5, num_valid) * base_height * (0.5 + 2.5 * (bump_difficulty ** 1.5))
    
#     # ★ 半径：沿 x 方向拉长，沿 y 方向缩短（让凸块在 y 方向更陡峭）
#     radius_x = rng.uniform(0.15, 0.30, num_valid) / L  # x 方向：宽
#     radius_y = rng.uniform(0.08, 0.15, num_valid) / W  # y 方向：窄
    
#     # 创建网格坐标
#     X = np.linspace(0, 1, nx)[None, :]
#     Y = np.linspace(0, 1, ny)[:, None]
    
#     # 逐个叠加凸块（使用各向异性高斯）
#     for i in range(num_valid):
#         dx = (X - bump_x_norm[i]) / radius_x[i]  # x 方向归一化
#         dy = (Y - bump_y_norm[i]) / radius_y[i]  # y 方向归一化
#         dist_sq = dx**2 + dy**2
        
#         bump = bump_height[i] * np.exp(-dist_sq / 2)
#         Z += bump
    
#     # 4. ★ 关键：让每个 x 位置（每列）的平均高度保持不变
#     # 计算每列的均值
#     col_means = Z.mean(axis=0)  # [nx]
    
#     # 减去每列的均值，使每列平均高度为 0
#     Z -= col_means[None, :]
    
#     # 5. 添加细节噪声（只在 y 方向变化）
#     if add_noise:
#         rng_noise = np.random.default_rng(seed + 1000)
#         # 生成 y 方向的 1D 噪声，然后在 x 方向复制
#         noise_1d = rng_noise.random(ny).astype(np.float32)
        
#         try:
#             from scipy.ndimage import gaussian_filter1d
#             noise_1d = gaussian_filter1d(noise_1d, sigma=3.0)
#         except ImportError:
#             pass
        
#         # ★ 噪声强度根据 x 位置的难度变化
#         noise_scale = 0.05 * (0.2 + 1.8 * (env ** 1.5))  # [nx]
#         noise_2d = noise_1d[:, None] * noise_scale[None, :]  # [ny, nx]
        
#         Z += noise_2d - noise_2d.mean(axis=0)[None, :]  # 保持每列均值为 0
    
#     # 6. ★ 确保平地段完全为 0
#     for j0, j1 in flat_masks:
#         Z[:, j0:j1] = 0.0
    
#     # 7. ★ 增强对比度（保持每列均值为 0）
#     mask_2d = np.ones_like(Z, dtype=bool)
#     for j0, j1 in flat_masks:
#         mask_2d[:, j0:j1] = False
    
#     if np.any(mask_2d):
#         # 对非平地段的每列独立处理
#         for j in range(nx):
#             if mask_2d[:, j].any():
#                 col = Z[:, j]
#                 col_mean = col.mean()
#                 # 增强对比度
#                 col_centered = col - col_mean
#                 col_centered = np.sign(col_centered) * (np.abs(col_centered) ** 0.8)
#                 Z[:, j] = col_centered + col_mean
        
#         # 再次确保每列均值为 0（减去残差）
#         for j in range(nx):
#             if mask_2d[:, j].any():
#                 Z[:, j] -= Z[:, j].mean()
    
#     # 8. ★ 归一化：只缩放幅度，不改变均值
#     # 找到最大偏离（正负最大值）
#     max_deviation = max(abs(Z.min()), abs(Z.max()))
#     if max_deviation > 1e-9:
#         Z /= max_deviation
    
#     # 9. 最后确保平地段为 0
#     for j0, j1 in flat_masks:
#         Z[:, j0:j1] = 0.0
    
#     return Z


# def save_gray_png(Z, path):
#     """保存为 8-bit 灰度 PNG"""
#     img = (Z * 255).astype(np.uint8)
#     Image.fromarray(img, mode="L").save(path, format="PNG", optimize=False)


# def parse_segments(arg_segments):
#     """解析段配置参数"""
#     out = []
#     for token in arg_segments:
#         parts = token.split(":")
#         kind = parts[0].strip().lower()
        
#         if kind == "flat":
#             length = float(parts[1])
#             out.append(("flat", length))
#         elif kind == "const":
#             length = float(parts[1])
#             val = float(parts[2])
#             out.append(("const", length, val))
#         elif kind == "ramp":
#             length = float(parts[1])
#             v0 = float(parts[2])
#             v1 = float(parts[3])
#             out.append(("ramp", length, v0, v1))
#         else:
#             raise ValueError(f"Unknown segment type: {token}")
    
#     return out


# def main():
#     ap = argparse.ArgumentParser(description="生成横向凸块地形")
#     ap.add_argument("--length", type=float, default=15.0, help="地形总长度（米）")
#     ap.add_argument("--width", type=float, default=8.0, help="地形宽度（米）")
#     ap.add_argument("--nx", type=int, default=1536, help="x方向分辨率")
#     ap.add_argument("--ny", type=int, default=128, help="y方向分辨率")
#     ap.add_argument("--segments", nargs="+", 
#                     default=["flat:1", "ramp:14:0.0:0.8"],
#                     help="段配置: flat:L | const:L:val | ramp:L:v0:v1")
#     ap.add_argument("--density", type=float, default=2.0, 
#                     help="凸块基础密度系数")
#     ap.add_argument("--height", type=float, default=1.0,
#                     help="凸块基础高度系数")
#     ap.add_argument("--seed", type=int, default=4141, help="随机种子")
#     ap.add_argument("--out", type=str, default="terrain_staged.png",
#                     help="输出 PNG 文件路径")
#     args = ap.parse_args()
    
#     # 解析段配置
#     segments = parse_segments(args.segments)
#     L = sum(seg[1] for seg in segments)
    
#     # 生成地形
#     Z = compose_bumpy_terrain(
#         L=L, W=args.width, nx=args.nx, ny=args.ny,
#         segments=segments,
#         base_density=args.density,
#         base_height=args.height,
#         add_noise=True,
#         seed=args.seed
#     )
    
#     # 保存
#     save_gray_png(Z, args.out)
    
#     print(f"[OK] 已保存 -> {args.out}")
#     print(f"  地形长度: {L:.3f} m, 宽度: {args.width:.3f} m")
#     print(f"  分辨率: {args.nx} x {args.ny}")
#     print(f"  凸块数量: ~{int(80 * L * args.width * args.density)} 个")
#     print("\nMuJoCo hfield 配置:")
#     print(f'  "rx": {L/2:.1f},  # half length')
#     print(f'  "ry": {args.width/2:.1f},  # half width')


# if __name__ == "__main__":
#     main()

# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# 生成 MuJoCo 高度图 PNG（gym_custom_terrain 风格）并可按段控制难度长度。
# - 支持任意段数：每段指定长度（米）与目标难度（0..1）
# - “平地段” => 难度 0；也可指定一段连续 ramp（从 a 到 b）
# - 输出为 8-bit 灰度（mode='L'）

# 示例：
#   # 复现：前 2 m 平地，后 9 m 难度渐进到 1.0
#   python make_terrain.py --width 1 --nx 1536 --ny 128 \
#     --segments flat:2 ramp:9:0.15:1.0 \
#     --out terrain_go1_11m_gymcustom_gradual_flat2m.png

#   # 3 段难度：平地2m，easy 3m(0.3)，medium 3m(0.6)，hard 3m(1.0)
#   python make_terrain.py --segments flat:2 const:3:0.3 const:3:0.6 const:3:1.0 \
#     --out terrain_4segments.png
# """

# import argparse
# import numpy as np
# from PIL import Image

# def smoothstep(z):
#     return z*z*(3 - 2*z)

# def make_value_noise_facets(nx, ny, gx=90, gy=10, seed=4141):
#     """生成块状 value-noise，再做非线性增强，得到碎片/台地感。"""
#     rng = np.random.default_rng(seed)
#     g = rng.random((gy+1, gx+1)).astype(np.float32)
#     xs = np.linspace(0, gx, nx, endpoint=False)
#     ys = np.linspace(0, gy, ny, endpoint=False)
#     xi = np.floor(xs).astype(int)
#     yi = np.floor(ys).astype(int)
#     tx = xs - xi
#     ty = ys - yi

#     txs = smoothstep(tx)
#     tys = smoothstep(ty)

#     Z = np.zeros((ny, nx), dtype=np.float32)
#     for j in range(ny):
#         y0 = yi[j]; y1 = y0 + 1
#         wy0 = 1 - tys[j]; wy1 = tys[j]
#         v00 = g[y0, xi]; v10 = g[y0, xi+1]
#         v01 = g[y1, xi]; v11 = g[y1, xi+1]
#         zx0 = (1 - txs) * v00 + txs * v10
#         zx1 = (1 - txs) * v01 + txs * v11
#         Z[j, :] = wy0 * zx0 + wy1 * zx1

#     # 非线性对比，做“块/碎”感
#     Z = (Z - 0.5)
#     Z = np.sign(Z) * (np.abs(Z) ** 0.35)
#     Z = (Z + 1.0) * 0.5
#     return Z

# def add_oriented_ridges(Z, L, W, strength=0.03, seed=0):
#     """叠加轻微斜向脊线，避免完全规则的瓦片感。"""
#     ny, nx = Z.shape
#     X = np.linspace(0.0, L, nx, endpoint=False)[None, :]
#     Y = np.linspace(-W/2, W/2, ny, endpoint=False)[:, None]
#     ridges = strength * np.sin(10.5 * (X*np.cos(0.25) + Y*np.sin(0.25))) \
#            + 0.66*strength * np.sin(7.5  * (X*np.cos(-0.35) + Y*np.sin(-0.35)))
#     Z += ridges.astype(np.float32)
#     return Z

# def build_envelope_over_segments(x, segments):
#     """
#     生成 x 上的难度包络 env(x) ∈ [0,1]。
#     segments: 列表，每个元素为元组 (length, kind, params...)
#       kind = "flat"  => params: (0,)         # 难度=0
#       kind = "const" => params: (val,)       # 常量难度
#       kind = "ramp"  => params: (v0, v1)     # 线性 ramp
#     """
#     L = sum(s[0] for s in segments)
#     xs = np.asarray(x)
#     env = np.zeros_like(xs, dtype=np.float32)
#     cur = 0.0
#     for length, kind, *pars in segments:
#         x0, x1 = cur, cur + length
#         mask = (xs >= x0) & (xs < x1) if length > 0 else (xs == x0)
#         if not np.any(mask):
#             cur = x1; continue
#         t = (xs[mask] - x0) / max(1e-12, length)
#         if kind == "flat":
#             val = 0.0
#             env[mask] = val
#         elif kind == "const":
#             val = float(pars[0])
#             env[mask] = val
#         elif kind == "ramp":
#             v0, v1 = float(pars[0]), float(pars[1])
#             env[mask] = v0 + (v1 - v0) * smoothstep(smoothstep(t))  # 更柔滑
#         else:
#             raise ValueError(f"unknown segment kind: {kind}")
#         cur = x1

#     # 包络最后一个点
#     env[xs >= L] = env[xs < L][-1] if np.any(xs < L) else 0.0
#     return env, L

# def compose_heightfield(L, W, nx, ny, segments, base_amp=1.0,
#                         ridge_strength=0.03, seed=4141):
#     """
#     合成高度图（未归一化），返回 Z ∈ R^{ny×nx} 和总长度 L。
#     base_amp 调整体强度（env 还会再缩放一次）。
#     """
#     Z = make_value_noise_facets(nx, ny, gx=90, gy=10, seed=seed)
#     Z = add_oriented_ridges(Z, L, W, strength=ridge_strength, seed=seed+1)

#     # 转到 [-1,1] 并准备包络
#     Z = 2.0*(Z - 0.5)
#     x = np.linspace(0.0, L, nx, endpoint=False)
#     env, Lsum = build_envelope_over_segments(x, segments)

#     # 最终幅度：低端留一点起伏（0.1*base_amp），高端→ base_amp
#     amp = (0.10 + 0.90*env) * base_amp
#     Z = (Z * amp[None, :])

#     # 明确“平地段”置为最小值（完全平）
#     cur = 0.0
#     for length, kind, *pars in segments:
#         if kind == "flat" and length > 0:
#             # 把这段对应的列范围找出来
#             j0 = int(round(cur / L * nx))
#             j1 = int(round((cur + length) / L * nx))

#             # 取平地两侧一圈的平均高度，保证过渡比较自然
#             border_vals = []
#             if j0 > 0:
#                 border_vals.append(Z[:, j0-1])
#             if j1 < nx:
#                 border_vals.append(Z[:, j1])
#             if border_vals:
#                 flat_h = float(np.mean(border_vals))
#             else:
#                 flat_h = float(np.mean(Z))

#             # ★ 关键：用一个「标量」填整个区域，所有 x、y 同一个高度
#             Z[:, j0:j1] = flat_h

#         cur += length


#     # 规范化到 [0,1]
#     Z -= Z.min()
#     Z /= max(1e-9, Z.max())
#     return Z

# def save_gray_png(Z, path):
#     img = (Z * 255).astype(np.uint8)
#     Image.fromarray(img, mode="L").save(path, format="PNG", optimize=False)

# def parse_segments(arg_segments):
#     """
#     解析形如：
#       flat:2
#       const:3:0.3
#       ramp:9:0.15:1.0
#     的片段列表。
#     返回 [(length, kind, params...), ...]
#     """
#     out = []
#     for token in arg_segments:
#         parts = token.split(":")
#         kind = parts[0].strip().lower()
#         if kind == "flat":
#             length = float(parts[1])
#             out.append((length, "flat", 0.0))
#         elif kind == "const":
#             length = float(parts[1]); val = float(parts[2])
#             out.append((length, "const", val))
#         elif kind == "ramp":
#             length = float(parts[1]); v0 = float(parts[2]); v1 = float(parts[3])
#             out.append((length, "ramp", v0, v1))
#         else:
#             raise ValueError(f"bad segment spec: {token}")
#     return out

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--width", type=float, default=1.0, help="地形宽度（米），仅影响生成纹理方向感")
#     ap.add_argument("--nx", type=int, default=1536, help="采样分辨率（x）")
#     ap.add_argument("--ny", type=int, default=128, help="采样分辨率（y）")
#     ap.add_argument("--segments", nargs="+", required=True,
#                     help="段定义：flat:L | const:L:val | ramp:L:v0:v1（长度单位 m，难度∈[0,1]）")
#     ap.add_argument("--amp", type=float, default=1.0, help="整体幅度系数（配合 hfield hz 使用）")
#     ap.add_argument("--ridge", type=float, default=0.03, help="斜向脊线强度（0..~0.06）")
#     ap.add_argument("--seed", type=int, default=4141, help="随机种子")
#     ap.add_argument("--out", type=str, required=True, help="输出 PNG 路径")
#     args = ap.parse_args()

#     segs = parse_segments(args.segments)
#     L = sum(s[0] for s in segs)
#     Z = compose_heightfield(
#         L=L, W=args.width, nx=args.nx, ny=args.ny,
#         segments=segs, base_amp=args.amp, ridge_strength=args.ridge, seed=args.seed
#     )
#     save_gray_png(Z, args.out)
#     print(f"[OK] saved -> {args.out} (L={L:.3f} m, width={args.width:.3f} m, nx={args.nx}, ny={args.ny})")
#     print("建议 hfield： size=\"{rx} {ry} {hz} {base}\" 其中：")
#     print("  rx=L/2, ry=width/2, hz=垂直尺度(米)，base=基座高度(>0)")
#     print("  例如 L=11, width=1  ⇒ size=\"5.5 0.5 0.30 0.005\"，geom pos 取 x=4.5 覆盖 [-1,10]")

# if __name__ == "__main__":
#     main()



# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# 生成 MuJoCo 高度图 PNG（gym_custom_terrain 风格），
# 支持：
#   - 按段控制长度 & 难度（平地 / 常量难度 / 渐进 ramp）
#   - 强烈左右错台（横坡），考机器狗的 torsion 能力
# 输出为 8-bit 灰度 PNG（mode='L'），可直接用于 <hfield file="...">。

# 使用：
#   - 直接运行本文件，会生成示例：
#       前 1 m 平地 + 后 10 m 渐进粗糙 + 渐进变大的左右错台
#   - 想改平地长度 / 难度段，只要改 MAIN CONFIG 里的 segments 列表即可
# """

# import numpy as np
# from PIL import Image


# # ---------- 通用小函数 ----------

# def smoothstep(z):
#     return z * z * (3 - 2 * z)


# # ---------- 纵向“块状粗糙” + 波纹 ----------

# def make_value_noise_facets(nx, ny, gx=90, gy=10, seed=4141):
#     """
#     生成块状 value-noise，再做非线性增强，得到碎片/台地感。
#     - nx, ny: 采样分辨率
#     - gx, gy: 噪声网格尺寸（越小越块状）
#     """
#     rng = np.random.default_rng(seed)
#     g = rng.random((gy + 1, gx + 1)).astype(np.float32)

#     xs = np.linspace(0, gx, nx, endpoint=False)
#     ys = np.linspace(0, gy, ny, endpoint=False)
#     xi = np.floor(xs).astype(int)
#     yi = np.floor(ys).astype(int)
#     tx = xs - xi
#     ty = ys - yi

#     txs = smoothstep(tx)
#     tys = smoothstep(ty)

#     Z = np.zeros((ny, nx), dtype=np.float32)
#     for j in range(ny):
#         y0 = yi[j]
#         y1 = y0 + 1
#         wy0 = 1 - tys[j]
#         wy1 = tys[j]
#         v00 = g[y0, xi]
#         v10 = g[y0, xi + 1]
#         v01 = g[y1, xi]
#         v11 = g[y1, xi + 1]
#         zx0 = (1 - txs) * v00 + txs * v10
#         zx1 = (1 - txs) * v01 + txs * v11
#         Z[j, :] = wy0 * zx0 + wy1 * zx1

#     # 非线性对比：更“块状”
#     Z = (Z - 0.5)
#     Z = np.sign(Z) * (np.abs(Z) ** 0.35)
#     Z = (Z + 1.0) * 0.5
#     return Z


# def add_oriented_ridges(Z, L, W, strength=0.03, seed=0):
#     """
#     叠加轻微斜向脊线，避免完全规则的瓦片感。
#     """
#     ny, nx = Z.shape
#     X = np.linspace(0.0, L, nx, endpoint=False)[None, :]
#     Y = np.linspace(-W / 2, W / 2, ny, endpoint=False)[:, None]
#     ridges = strength * np.sin(10.5 * (X * np.cos(0.25) + Y * np.sin(0.25))) \
#            + 0.66 * strength * np.sin(7.5 * (X * np.cos(-0.35) + Y * np.sin(-0.35)))
#     Z += ridges.astype(np.float32)
#     return Z


# # ---------- 段式难度包络 ----------

# def build_envelope_over_segments(x, segments):
#     """
#     生成 x 上的难度包络 env(x) ∈ [0,1]。
#     segments: 列表，每项为 (kind, length, params...)
#       kind = "flat"  => ( "flat",  L )
#       kind = "const" => ( "const", L, val )
#       kind = "ramp"  => ( "ramp",  L, v0, v1 )
#     """
#     xs = np.asarray(x)
#     env = np.zeros_like(xs, dtype=np.float32)

#     cur = 0.0
#     for seg in segments:
#         kind = seg[0]
#         length = float(seg[1])
#         x0, x1 = cur, cur + length
#         if length <= 0:
#             cur = x1
#             continue

#         mask = (xs >= x0) & (xs < x1)
#         if not np.any(mask):
#             cur = x1
#             continue

#         t = (xs[mask] - x0) / max(1e-12, length)

#         if kind == "flat":
#             env[mask] = 0.0
#         elif kind == "const":
#             val = float(seg[2])
#             env[mask] = val
#         elif kind == "ramp":
#             v0, v1 = float(seg[2]), float(seg[3])
#             env[mask] = v0 + (v1 - v0) * smoothstep(smoothstep(t))
#         else:
#             raise ValueError(f"unknown segment kind: {kind}")

#         cur = x1

#     L_total = cur
#     # 最后一个点，补成尾部值
#     env[xs >= L_total] = env[xs < L_total][-1] if np.any(xs < L_total) else 0.0
#     return env, L_total


# # ---------- 强左右 torsion 的“错台/横坡” ----------

# def add_strong_lateral_bands(Z, L, W, env, band_width=1.0, strength=0.30):
#     """
#     给地形叠加明显的左右错台/横坡：
#       - 每 band_width 米换一次“左高右低 / 右高左低”
#       - strength 控制左右高度差的强度
#       - env(x) 决定前后渐进：前面几乎没有，后面非常明显
#     """
#     ny, nx = Z.shape
#     X = np.linspace(0.0, L, nx, endpoint=False)[None, :]      # [1, nx]
#     Y = np.linspace(-W/2, W/2, ny, endpoint=False)[:, None]   # [ny, 1]

#     # 每 band_width 米一个 band
#     bands = np.floor(X / band_width).astype(int)             # 0,1,2,...
#     # 偶数 band：左高右低(+1)，奇数：右高左低(-1)
#     sign = (bands % 2) * -2 + 1                              # 0->+1, 1->-1,...

#     # 左右方向 [-1,1]：左=-1，右=+1
#     slope = Y / (W / 2.0)

#     # env: 1D -> 2D
#     if env.ndim == 1:
#         env2 = env[None, :]     # [1, nx]
#     else:
#         env2 = env
#     env_full = np.repeat(env2, ny, axis=0)   # [ny, nx]

#     Z += strength * env_full * sign * slope
#     return Z


# # ---------- 主合成函数 ----------

# def compose_heightfield(L, W, nx, ny, segments,
#                         base_amp=1.0,
#                         ridge_strength=0.03,
#                         lateral_strength=0.30,
#                         band_width=1.0,
#                         seed=4141):
#     """
#     合成高度图 Z ∈ [0,1]^{ny×nx}：
#       - 纵向块状粗糙 + 波纹
#       - 段式难度包络（前平后难）
#       - 强左右错台/横坡（随 x 渐进）
#       - flat 段是真·平面（所有 x,y 同一高度）
#     """
#     # 1) 基础块状噪声 + 脊线
#     Z = make_value_noise_facets(nx, ny, gx=90, gy=10, seed=seed)
#     Z = add_oriented_ridges(Z, L, W, strength=ridge_strength, seed=seed+1)

#     # 2) 转到 [-1,1]
#     Z = 2.0 * (Z - 0.5)

#     # 3) 段式包络
#     x = np.linspace(0.0, L, nx, endpoint=False)
#     env, L_total = build_envelope_over_segments(x, segments)

#     # 纵向起伏幅度
#     amp = (0.10 + 0.90 * env) * base_amp
#     Z = Z * amp[None, :]

#     # 4) 强左右错台（torsion）
#     Z = add_strong_lateral_bands(Z, L, W, env, band_width=band_width, strength=lateral_strength)

#     # 5) flat 段：完全平面，用两侧平均高度填充
#     cur = 0.0
#     for seg in segments:
#         kind = seg[0]
#         length = float(seg[1])
#         if kind == "flat" and length > 0:
#             j0 = int(round(cur / L * nx))
#             j1 = int(round((cur + length) / L * nx))

#             border_vals = []
#             if j0 > 0:
#                 border_vals.append(Z[:, j0-1])
#             if j1 < nx:
#                 border_vals.append(Z[:, j1])
#             flat_h = float(np.mean(border_vals)) if border_vals else float(np.mean(Z))

#             # ★ 整块平面：所有 x,y 同一高度
#             Z[:, j0:j1] = flat_h

#         cur += length

#     # 6) 归一化到 [0,1]
#     Z -= Z.min()
#     Z /= max(1e-9, Z.max())
#     return Z


# def save_gray_png(Z, path):
#     img = (Z * 255).astype(np.uint8)
#     Image.fromarray(img, mode="L").save(path, format="PNG", optimize=False)


# # ---------- MAIN CONFIG（你主要改这里） ----------

# if __name__ == "__main__":
#     # 地形宽度（米） & 采样分辨率
#     WIDTH = 1.0
#     NX = 1536
#     NY = 128

#     # 段配置：
#     # 例：前 1 m 平地，后 10 m 难度从 0.15 → 1.0 渐进
#     segments = [
#         ("flat", 1.0),           # 1 m 完全平
#         ("ramp", 10.0, 0.15, 1.0)
#     ]
#     L = sum(seg[1] for seg in segments)   # 总长度（米）

#     # 噪声强度参数
#     base_amp = 1.0          # 纵向起伏基础幅度（配合 hfield hz 再放大）
#     ridge_strength = 0.03   # 斜向脊线强度
#     lateral_strength = 0.50 # 左右错台强度（想更狠就调大到 0.4）
#     band_width = 1.0        # 每多少米切换一次“左高右低 / 右高左低”

#     out_path = "terrain2.png"

#     Z = compose_heightfield(
#         L=L, W=WIDTH, nx=NX, ny=NY,
#         segments=segments,
#         base_amp=base_amp,
#         ridge_strength=ridge_strength,
#         lateral_strength=lateral_strength,
#         band_width=band_width,
#         seed=4141,
#     )
#     save_gray_png(Z, out_path)

#     print(f"[OK] saved -> {out_path}")
#     print(f"  total length L = {L:.3f} m, width = {WIDTH:.3f} m")
#     print("MuJoCo hfield 建议：")
#     print(f'  <hfield name="go1_terrain" file="{out_path}" '
#           f'size="{L/2:.3f} {WIDTH/2:.3f} 0.30 0.005"/>')
#     print("  <geom type=\"hfield\" hfield=\"go1_terrain\" "
#           f'size="{L/2:.3f} {WIDTH/2:.3f} 0.30" pos="{L/2-0.5:.3f} 0 0"/>')
#     print("  （上面 pos 约覆盖 x∈[-0.5, L-0.5]，你可以按需要微调）")

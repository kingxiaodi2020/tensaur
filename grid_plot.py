# #!/usr/bin/env python3
# import pathlib
# import csv
# import matplotlib.pyplot as plt

# ROOT = pathlib.Path(__file__).resolve().parent
# results_dir = ROOT / "results"

# def load_curve(param_name, force_mag=30):
#     # csv_path = results_dir / f"grid_{param_name}_Fy{int(force_mag)}.csv"
#     csv_path = results_dir / f"spine_grid_Fy{int(force_mag)}.csv"
#     xs, ys = [], []
#     with open(csv_path, "r") as f:
#         reader = csv.DictReader(f)
#         for row in reader:
#             xs.append(float(row["stiffness"]))
#             ys.append(float(row["tip_point_dy"]))
#     return xs, ys

# def main():
#     force_mag = 30.0

#     param_list = [
#         ("lateral_hori_stiffness", "lateral left/right components", "o"),
#         ("lateral_verti_stiffness", "lateral top/bottom components", "s"),
#         ("diagonal_stiffness", "diagonal components", "^"),
#     ]

#     plt.figure(figsize=(12, 8))

#     for param_name, label, marker in param_list:
#         xs, ys = load_curve(param_name, force_mag=force_mag)
#         plt.plot(xs, ys, marker=marker, linestyle="-", label=label)

#     plt.xlabel("Stiffness", fontsize=18)
#     plt.ylabel("Tip point deformation in Y direction (m)", fontsize=18)
#     plt.title(f"Tip point deformation vs Stiffness (Fy = {force_mag} N)", fontsize=24)

#     # Add explanatory text below the title
#     plt.text(
#         0.5, 0.1,  # x and y position in figure coordinates
#         "When varying one parameter, the other two are fixed at 5000.",
#         fontsize=14,
#         ha="center",
#         transform=plt.gcf().transFigure,
#     )

#     plt.grid(True, linestyle=":")
#     plt.legend()

#     plt.xticks([500, 1000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000], fontsize=12)
#     plt.tight_layout()
#     plt.savefig(results_dir / "Fy_bending_test.png", dpi=300)  # Save the plot
#     plt.show()
# if __name__ == "__main__":
#     main()

import plotly.graph_objs as go

# ---------- Batch 1: run_0.5T ----------
batch1 = [
    (4723.743026827156,1759.8706636804106,5934.220234450515),
    (5818.453530182165,7574.686237732304,6586.47871578632),
    (9799.474403437898,8982.132504563864,3889.985781321305),
    (5403.728037723931,9615.888052817876,5310.116497089842),
    (8621.211057548802,695.3915342805188,5695.973530095308),
    (6705.832064759157,9133.500530693544,4968.958755492026),
    (9506.52070859927,9943.345820188359,6261.482606594145),
    (6099.865760296637,1628.7572972806884,3923.805015135408),
    (7463.5284476660445,2371.5454882422086,2726.937278763107),
    (2549.3007143657173,9481.316391235549,3527.2733683506613),
    (5421.951577068067,510.0017338543368,1613.8729493302635),
    (4962.669398331149,2422.7481698754827,879.4393513682214),
    (3597.2241739813253,7004.8247666676625,2818.6043389682304),
    (829.5396691141839,9385.281012666497,3644.5227785542525),
]

# ---------- Batch 2: run_0.5T_LHS（第一次 LHS） ----------
batch2 = [
    (7051.479277283427,9771.287707075611,6672.3193338525225),
    (1233.8661044380638,7437.695539735532,5270.172184407486),
    (3369.7348802314705,858.8854551909778,9683.298709843466),
    (6244.361864602906,3689.611273531502,4040.011204025378),
    (1130.9739027650712,2801.8100551281605,1418.8384525695021),
    (1942.4906503438976,5790.077008757232,573.7909326778953),
    (5409.597668849458,4119.484592060172,6337.20186603715),
    (7531.521953428956,8914.11625663599,3474.3419476502004),
    (2836.927590539316,2476.2760205043764,7905.307362064009),
    (8472.401890953943,8453.087700327866,2795.9216622609197),
    (5078.044617972708,1252.1646721386085,8672.661913855834),
    (8657.898774874973,6239.776192845816,8335.140639445399),
    (4411.0759893818795,4688.906123585023,2513.9703461629406),
    (9796.76875798658,6664.182770575727,4815.416347588926),
]

# ---------- Batch 3: run_0.5T_LHS（优化 LHS） ----------
batch3 = [
    (3947.6657469394963,2039.1095634938729,2021.4318671919286),
    (4589.676414173051,854.7491066513287,9575.615324073491),
    (5712.709336077524,3751.3388590397776,8330.984270855548),
    (611.8289106309036,9282.922439105563,3269.983231398008),
    (6213.075437400268,4408.420435459498,5470.656346642358),
    (7618.700510800264,8321.767403853599,710.077210869621),
    (2203.5452810441147,1326.8833885071049,6874.884068963992),
    (8327.917708891368,5670.819700013541,7547.099529414071),
    (7008.90312490603,6337.923899581986,4709.999573780956),
    (1351.0335025233055,6947.241348108504,1844.3952627023666),
    (3232.526640954658,4871.200104322552,2697.871304415586),
    (9994.095052032633,7764.012285758081,4396.42967549013),
    (2685.612497431681,9597.85778998207,9273.07916325382),
    (8936.844041522634,2907.448410542354,6409.6652726503735),
]

def make_3d_fig(points, title):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    zs = [p[2] for p in points]

    fig = go.Figure(data=[go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode='markers',
        marker=dict(size=6),
        text=[f"({x:.1f}, {y:.1f}, {z:.1f})" for x, y, z in points],
        hovertemplate="x=%{x:.1f}<br>y=%{y:.1f}<br>z=%{z:.1f}<extra></extra>"
    )])

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="lateral_hori_stiffness",
            yaxis_title="lateral_verti_stiffness",
            zaxis_title="diagonal_stiffness",
            xaxis=dict(range=[500, 10000]),
            yaxis=dict(range=[500, 10000]),
            zaxis=dict(range=[500, 10000]),
            aspectmode="cube",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )
    return fig

fig1 = make_3d_fig(batch1, "Batch 1: run_0.5T (Original)")
fig2 = make_3d_fig(batch2, "Batch 2: run_0.5T_LHS (LHS)")
fig3 = make_3d_fig(batch3, "Batch 3: run_0.5T_LHS (Optimized LHS)")

fig1.show()
fig2.show()
fig3.show()

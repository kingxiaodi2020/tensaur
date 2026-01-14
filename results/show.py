#!/usr/bin/env python3
import pathlib
import csv
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent
results_dir = ROOT / "results"

def load_curve(param_name, force_mag=30):
    # csv_path = results_dir / f"grid_{param_name}_Fy{int(force_mag)}.csv"
    csv_path = results_dir / f"spine_grid_Fy{int(force_mag)}.csv"
    xs, ys = [], []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            xs.append(float(row["stiffness"]))
            ys.append(float(row["tip_point_dy"]))
    return xs, ys

def main():
    force_mag = 30.0

    param_list = [
        ("lateral_hori_stiffness", "lateral left/right components", "o"),
        ("lateral_verti_stiffness", "lateral top/bottom components", "s"),
        ("diagonal_stiffness", "diagonal components", "^"),
    ]

    plt.figure(figsize=(12, 8))

    for param_name, label, marker in param_list:
        xs, ys = load_curve(param_name, force_mag=force_mag)
        plt.plot(xs, ys, marker=marker, linestyle="-", label=label)

    plt.xlabel("Stiffness", fontsize=18)
    plt.ylabel("Tip point deformation in Y direction (m)", fontsize=18)
    plt.title(f"Tip point deformation vs Stiffness (Fy = {force_mag} N)", fontsize=24)

    # Add explanatory text below the title
    plt.text(
        0.5, 0.1,  # x and y position in figure coordinates
        "When varying one parameter, the other two are fixed at 5000.",
        fontsize=14,
        ha="center",
        transform=plt.gcf().transFigure,
    )

    plt.grid(True, linestyle=":")
    plt.legend()

    plt.xticks([500, 1000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000], fontsize=12)
    plt.tight_layout()
    plt.savefig(results_dir / "Fy_bending_test.png", dpi=300)  # Save the plot
    plt.show()
if __name__ == "__main__":
    main()

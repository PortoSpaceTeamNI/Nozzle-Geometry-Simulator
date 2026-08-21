"""Create vector plots for the final publication report from retained CSV data."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "report_final_publication" / "Images"


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def geometry_plot() -> None:
    figure, axis = plt.subplots(figsize=(8.2, 4.4), constrained_layout=True)
    specifications = (
        ("initial_geometry_final.csv", "Initial geometry", "#222222", "-"),
        ("quasi_1d_geometry_final.csv", "Quasi-1D optimum", "#777777", "--"),
        ("moc_optimized_geometry_final.csv", "MOC-assisted optimum", "#a3262a", "-"),
    )
    maximum_radius = 0.0
    maximum_x = 0.0
    for filename, label, color, linestyle in specifications:
        rows = _read(IMAGES / filename)
        x = np.array([float(row["x_mm"]) for row in rows])
        radius = np.array([float(row["radius_mm"]) for row in rows])
        maximum_radius = max(maximum_radius, float(np.max(radius)))
        maximum_x = max(maximum_x, float(np.max(x)))
        axis.plot(x, radius, color=color, linestyle=linestyle, linewidth=2.0, label=label)
        axis.plot(x, -radius, color=color, linestyle=linestyle, linewidth=2.0)
    axis.axhline(0.0, color="#999999", linewidth=0.7)
    axis.set(
        xlabel="Axial position [mm]",
        ylabel="Radius [mm]",
        xlim=(0.0, 1.03 * maximum_x),
        ylim=(-1.08 * maximum_radius, 1.08 * maximum_radius),
    )
    axis.grid(alpha=0.22)
    axis.legend(loc="upper right", frameon=True)
    figure.savefig(IMAGES / "geometry_model_comparison.pdf")
    figure.savefig(IMAGES / "geometry_model_comparison.png", dpi=300)
    plt.close(figure)


def convergence_plot() -> None:
    rows = _read(IMAGES / "moc_mesh_convergence.csv")
    rows = [row for row in rows if int(row["axial_stations"]) >= 600]
    nx = np.array([int(row["axial_stations"]) for row in rows])
    cf = np.array([float(row["friction_corrected_cf"]) for row in rows])
    step = np.array([float(row["step_cf_change_percent"]) for row in rows])
    residual = np.array([float(row["mass_flow_residual_percent"]) for row in rows])
    inlet_error = np.array([float(row["initial_mass_flow_error_percent"]) for row in rows])

    figure, axes = plt.subplots(2, 1, figsize=(7.6, 6.5), sharex=True, constrained_layout=True)
    axes[0].plot(nx, cf, marker="o", color="#1f4e79", linewidth=1.8)
    axes[0].set(ylabel=r"Friction-corrected $C_F$")
    axes[0].grid(alpha=0.25)
    axes[0].annotate(
        "accepted precise mesh",
        xy=(nx[-1], cf[-1]),
        xytext=(-110, 22),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "linewidth": 0.9},
    )

    axes[1].plot(nx, step, marker="o", label=r"successive $C_F$ change", color="#a3262a")
    axes[1].plot(nx, residual, marker="s", label="mass-flow residual", color="#2d6a4f")
    axes[1].plot(nx, inlet_error, marker="^", label="inlet--CEA mass-flow error", color="#6c4c8a")
    axes[1].axhline(0.1, color="#a3262a", linestyle=":", linewidth=1.0)
    axes[1].axhline(0.5, color="#555555", linestyle="--", linewidth=1.0)
    axes[1].set(xlabel=r"Axial stations, $N_x$", ylabel="Relative measure [%]")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="upper right", fontsize=8)
    figure.savefig(IMAGES / "moc_mesh_convergence.pdf")
    figure.savefig(IMAGES / "moc_mesh_convergence.png", dpi=300)
    plt.close(figure)


def optimization_plot() -> None:
    rows = _read(IMAGES / "publication_geometry_comparison.csv")
    labels = ("Initial", "Quasi-1D optimum", "MOC-assisted optimum")
    blimp = np.array([float(row["quasi_1d_blimp_cf"]) for row in rows])
    moc = np.array([float(row["precise_moc_cf"]) for row in rows])
    positions = np.arange(len(rows))
    width = 0.34
    figure, axis = plt.subplots(figsize=(8.0, 4.6), constrained_layout=True)
    axis.bar(positions - width / 2, blimp, width, label="Quasi-1D + BLIMP", color="#7895b2")
    bars = axis.bar(positions + width / 2, moc, width, label="MOC + BLIMP", color="#b55457")
    bars[1].set_hatch("///")
    bars[1].set_edgecolor("#4d1517")
    axis.text(
        positions[1] + width / 2,
        moc[1] + 0.0012,
        "rejected",
        ha="center",
        va="bottom",
        fontsize=8,
    )
    axis.set_xticks(positions, labels)
    axis.set_ylabel(r"Effective thrust coefficient, $C_F$")
    axis.set_ylim(min(np.min(blimp), np.min(moc)) - 0.008, max(np.max(blimp), np.max(moc)) + 0.009)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="upper left")
    figure.savefig(IMAGES / "optimization_model_comparison.pdf")
    figure.savefig(IMAGES / "optimization_model_comparison.png", dpi=300)
    plt.close(figure)


def main() -> None:
    geometry_plot()
    convergence_plot()
    optimization_plot()
    print(f"Saved publication plots in {IMAGES}")


if __name__ == "__main__":
    main()

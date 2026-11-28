# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Draw the report figures from the files in ``data/``; no simulation is run.

The measured plate-intrusion slopes of Li, Zhang & Goldman (2013) are read from the
processed table of the newton-calibration MPM soil case (``--li``), which
``examples/mpm_soil/data/prepare.py`` regenerates from the article's supplementary data.

    python scripts/make_figures.py --li <newton-calibration>/examples/mpm_soil/data/li2013_rft.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import patches  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FixedLocator, NullFormatter, NullLocator  # noqa: E402

REPORT = Path(__file__).resolve().parents[1]
DATA = REPORT / "data"
FIGURES = REPORT / "figures"

INK = "#192333"
MUTED = "#536174"
LINE = "#dce1e7"
WASH = "#f4f6f8"
BLUE = "#2563eb"  # literature, laboratory or baseline parameters
AMBER = "#d97706"  # calibrated parameters
SOIL = "#e9dcc3"
CLAY = "#cfd6de"
WATER = "#e8f1fb"
STEEL = "#8793a3"
OBJECTIVE = LinearSegmentedColormap.from_list("objective", ["#192333", "#536174", "#aab4c2", "#e3e8ee"])

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 7.5,
        "axes.titlesize": 7.5,
        "axes.labelsize": 7.5,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK,
        "ytick.labelcolor": INK,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "legend.fontsize": 6.5,
        "legend.frameon": False,
        "text.color": INK,
        "svg.fonttype": "path",
        "pdf.fonttype": 42,
        "savefig.dpi": 220,
        "svg.hashsalt": "mpm-soil-report",
    }
)

# No creation dates, so that rerunning the script reproduces the published files byte for byte.
METADATA = {"svg": {"Date": None}, "pdf": {"CreationDate": None}, "png": {}}

CAPTIONS: dict[str, str] = {}


def load(name: str) -> dict:
    return json.loads((DATA / name).read_text())


def grid(ax):
    ax.grid(True, color=LINE, lw=0.5, zorder=0)
    ax.set_axisbelow(True)


def label(ax, text):
    ax.text(-0.02, 1.04, text, transform=ax.transAxes, fontsize=8.5, fontweight="bold", ha="right", va="bottom")


def objective_bar(figure, ax, mappable, ticks, rect=(0.9, 0.08, 0.035, 0.42)):
    """Small vertical colour bar inside ``ax`` for the training objective."""
    cax = ax.inset_axes(list(rect))
    bar = figure.colorbar(mappable, cax=cax)
    bar.outline.set_visible(False)
    bar.ax.yaxis.set_minor_locator(NullLocator())
    bar.set_ticks(ticks)
    bar.set_ticklabels([f"{t:g}" for t in ticks])
    bar.ax.tick_params(labelsize=6, length=2)
    cax.set_title("J", fontsize=6.5, pad=3)
    return bar


def marker_legend(figure, entries, ncol):
    handles = [
        Line2D([], [], ls="none", marker=m, markersize=sz, markerfacecolor=fc, markeredgecolor=ec, label=text)
        for text, m, sz, fc, ec in entries
    ]
    figure.legend(handles=handles, loc="outside lower center", ncol=ncol, fontsize=6.3, handletextpad=0.3,
                  columnspacing=1.2)


def save(figure, name: str, caption: str):
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("svg", "pdf", "png"):
        figure.savefig(FIGURES / f"{name}.{suffix}", bbox_inches="tight", pad_inches=0.03, metadata=METADATA[suffix])
    plt.close(figure)
    CAPTIONS[name] = caption
    print(f"wrote figures/{name}.{{svg,pdf,png}}")


# ----------------------------------------------------------------------------- setups


def _arrow(ax, start, end, color=INK, lw=0.9, style="-|>", **kw):
    ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": style, "color": color, "lw": lw, **kw})


def _panel_title(ax, letter, title, subtitle):
    ax.annotate(f"{letter}   {title}", xy=(0.0, 1.0), xycoords="axes fraction", xytext=(0, 14),
                textcoords="offset points", ha="left", va="bottom", fontsize=7.5, fontweight="bold")
    ax.annotate(subtitle, xy=(0.0, 1.0), xycoords="axes fraction", xytext=(12, 3), textcoords="offset points",
                ha="left", va="bottom", fontsize=6.3, color=MUTED)


def figure_setups():
    figure, axes = plt.subplots(1, 3, figsize=(7.4, 2.5), gridspec_kw={"width_ratios": [1.0, 1.0, 1.2]})
    for ax in axes:
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_anchor("N")

    # (a) Li et al. plate intrusion: side view in the plane of motion.
    ax = axes[0]
    _panel_title(ax, "a", "Plate intrusion, five media", r"Li et al. (2013); $\sigma = \alpha(\beta,\gamma)\,|z|$")
    ax.add_patch(patches.Rectangle((-12, -15), 24, 15, facecolor=SOIL, edgecolor="none"))
    ax.plot([-12, -12, 12, 12], [1.5, -15, -15, 1.5], color=MUTED, lw=1.2)
    ax.plot([-12, 12], [0, 0], color=MUTED, lw=0.5)
    centre = np.array([1.5, -7.0])
    beta, gamma = math.radians(30.0), math.radians(45.0)
    half = 0.5 * 3.81
    along = np.array([-math.cos(beta), math.sin(beta)])  # plate direction, beta up from -x
    normal = np.array([-along[1], along[0]])
    corners = [centre + a * half * along + b * 0.32 * normal for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    ax.add_patch(patches.Polygon(corners, closed=True, facecolor=STEEL, edgecolor=INK, lw=0.7, zorder=3))
    ax.plot([centre[0] - 5.0, centre[0] + 0.6], [centre[1], centre[1]], color=MUTED, lw=0.5, ls=(0, (3, 2)))
    ax.add_patch(patches.Arc(centre, 3.0, 3.0, theta1=180 - 30, theta2=180, color=INK, lw=0.7))
    ax.text(centre[0] - 2.0, centre[1] + 0.35, r"$\beta$", fontsize=8, ha="right")
    velocity = np.array([-math.cos(gamma), -math.sin(gamma)])
    _arrow(ax, centre, centre + 5.0 * velocity, color=AMBER, lw=1.1)
    ax.add_patch(patches.Arc(centre, 7.6, 7.6, theta1=180, theta2=180 + 45, color=AMBER, lw=0.6))
    ax.text(centre[0] - 4.2, centre[1] - 2.5, r"$\gamma$", fontsize=8, color=AMBER, ha="right")
    ax.text(centre[0] - 2.0, centre[1] - 4.9, r"$\mathbf{v}$", fontsize=8, color=AMBER)
    _arrow(ax, (9.0, 0.0), (9.0, centre[1]), color=MUTED, lw=0.6, style="<|-|>")
    ax.text(9.5, 0.5 * centre[1], r"$|z|$", fontsize=7.5, color=MUTED, va="center")
    ax.text(-11.2, -14.0, "poppy seeds or glass beads", fontsize=6.2, color=MUTED)
    _arrow(ax, (-12, -16.4), (12, -16.4), color=MUTED, lw=0.5, style="<|-|>")
    ax.text(0, -18.0, "24 cm wide, 15 cm deep", ha="center", fontsize=6.2, color=MUTED, va="top")
    ax.set_xlim(-13, 13)
    ax.set_ylim(-19.5, 2.0)

    # (b) Zwanenburg et al. centrifuge strip plate: modelled half domain.
    ax = axes[1]
    _panel_title(ax, "b", "Strip plate in clay at 100 g", "Zwanenburg et al. (2024); B = 100 or 20 mm")
    clay_h, water_h, width = 9.5, 4.0, 15.0
    ax.add_patch(patches.Rectangle((0, 0), width, clay_h, facecolor=CLAY, edgecolor="none"))
    ax.add_patch(patches.Rectangle((0, clay_h), width, water_h, facecolor=WATER, edgecolor="none"))
    ax.plot([0, width], [clay_h + water_h, clay_h + water_h], color=MUTED, lw=0.4, ls=(0, (2, 2)))
    ax.plot([0, width, width], [0, 0, clay_h + water_h + 0.5], color=INK, lw=1.2)
    ax.plot([0, 0], [-0.6, clay_h + water_h + 1.5], color=INK, lw=0.7, ls=(0, (5, 2, 1, 2)))
    ax.text(-0.5, 0.5 * clay_h, "symmetry plane", fontsize=6, color=MUTED, rotation=90, ha="right", va="center")
    plate_half, plate_t, shaft_half = 3.3, 1.0, 1.0
    y0 = clay_h - 3.0
    ax.add_patch(patches.Rectangle((0, y0), plate_half, plate_t, facecolor=STEEL, edgecolor=INK, lw=0.7, zorder=3))
    shaft_top = clay_h + water_h + 1.5
    ax.add_patch(patches.Rectangle((0, y0 + plate_t), shaft_half, shaft_top - y0 - plate_t,
                                   facecolor=STEEL, edgecolor=INK, lw=0.7, zorder=3))
    _arrow(ax, (0.5 * shaft_half, shaft_top + 2.2), (0.5 * shaft_half, shaft_top + 0.2), color=AMBER, lw=1.1)
    ax.text(shaft_half + 0.5, shaft_top + 1.2, "5 mm/min (model: 0.1 m/s)", fontsize=6, color=AMBER, va="center")
    ax.text(width - 0.3, clay_h + 0.5 * water_h, "water\n(not modelled)", fontsize=6, color=MUTED, ha="right",
            va="center")
    _arrow(ax, (0, y0 - 0.6), (plate_half, y0 - 0.6), color=MUTED, lw=0.5, style="<|-|>")
    ax.text(0.5 * plate_half, y0 - 1.0, "B/2", ha="center", va="top", fontsize=6.2, color=MUTED)
    ax.text(0.62 * width, 0.35 * clay_h, "kaolin clay\n" + r"$s_u(z)$: SHANSEP", ha="center", fontsize=6.2)
    _arrow(ax, (width + 0.8, 0), (width + 0.8, clay_h), color=MUTED, lw=0.5, style="<|-|>")
    ax.text(width + 1.2, 0.5 * clay_h, "91–97\nmm", fontsize=6, color=MUTED, va="center")
    ax.text(0.5 * width, -1.0, "rigid base, smooth or rough", ha="center", fontsize=6.2, color=MUTED, va="top")
    ax.set_xlim(-1.6, width + 3.3)
    ax.set_ylim(-2.6, clay_h + water_h + 4.0)

    # (c) Franka rake stroke: tool-tip path in the bin frame.
    ax = axes[2]
    _panel_title(ax, "c", "Franka FR3 rake stroke", "80 mm blade; logged joints replayed in Newton")
    soil_h, wall_h = 6.0, 8.0
    ax.add_patch(patches.Rectangle((-15, -soil_h), 30, soil_h, facecolor=SOIL, edgecolor="none"))
    ax.plot([-15, -15, 15, 15], [-soil_h + wall_h, -soil_h, -soil_h, -soil_h + wall_h], color=MUTED, lw=1.2)
    ax.plot([-15, 15], [0, 0], color=MUTED, lw=0.5)
    start, length, depth_c, clear = -8.0, 12.0, 3.0, 1.5
    path_x = [start, start, start + length, start + length]
    path_y = [clear, -depth_c, -depth_c, clear]
    ax.plot(path_x, path_y, color=AMBER, lw=1.2, zorder=4)
    for (x0, y0_), (x1, y1) in zip(zip(path_x[:-1], path_y[:-1]), zip(path_x[1:], path_y[1:]), strict=True):
        mid = (0.5 * (x0 + x1), 0.5 * (y0_ + y1))
        _arrow(ax, (mid[0] - 0.01 * (x1 - x0), mid[1] - 0.01 * (y1 - y0_)), mid, color=AMBER, lw=1.0)
    tilt = math.radians(20.0)
    tip = np.array([start + 0.78 * length, -depth_c])
    up = np.array([-math.sin(tilt), math.cos(tilt)])  # positive pitch: blade bottom forward (+x)
    side = np.array([up[1], -up[0]])
    blade = [tip - 0.4 * side, tip + 0.4 * side, tip + 0.4 * side + 4.0 * up, tip - 0.4 * side + 4.0 * up]
    ax.add_patch(patches.Polygon(blade, closed=True, facecolor=STEEL, edgecolor=INK, lw=0.7, zorder=5))
    top = tip + 4.0 * up
    shaft = [top - 0.8 * side, top + 0.8 * side, top + 0.8 * side + 3.6 * up, top - 0.8 * side + 3.6 * up]
    ax.add_patch(patches.Polygon(shaft, closed=True, facecolor=STEEL, edgecolor=INK, lw=0.7, zorder=5))
    ax.text(start - 0.5, -1.6, "plunge", fontsize=6, color=MUTED, ha="right")
    ax.text(start + 0.6, -depth_c - 0.9, "drag (scored)", fontsize=6, color=MUTED, va="top")
    ax.text(start + length + 0.6, -2.2, "lift", fontsize=6, color=MUTED)
    ax.text(tip[0] + 1.2, 1.2, r"pitch $\pm 20°$", fontsize=6, color=MUTED)
    _arrow(ax, (7.2, 0), (7.2, -depth_c), color=MUTED, lw=0.5, style="<|-|>")
    ax.text(7.7, -0.5 * depth_c, "12–45 mm", fontsize=6, color=MUTED, va="center")
    ax.text(0, -soil_h - 0.6, "bin 300 × 160 mm, 60 mm of soil", ha="center", fontsize=6.2, color=MUTED, va="top")
    ax.set_xlim(-16.5, 16.5)
    ax.set_ylim(-soil_h - 2.2, 8.2)

    figure.subplots_adjust(left=0.01, right=0.99, top=0.80, bottom=0.02, wspace=0.12)
    save(
        figure,
        "setups",
        "Experiments replicated in Newton. (a) Li, Zhang & Goldman (2013) moved a thin aluminium plate at attack "
        "angle β along a straight path at intrusion angle γ through a 24 × 22 × 15 cm bed of poppy seeds and reported "
        "the stress per unit depth α(β, γ). (b) Zwanenburg et al. (2024) pushed rigid strip plates into soft kaolin clay "
        "at 100 g; the simulation models the half domain next to the plate's symmetry plane as a plane-strain slab. "
        "(c) A Franka FR3 drags a rake blade through a soil bin along a hold–plunge–drag–lift stroke; only the steady "
        "drag is scored. Drawings are schematic and not to scale.",
    )


# ----------------------------------------------------------------------------- plate intrusion


def _rft_arrays(li_path: Path):
    points = load("rft_points.json")
    li = json.loads(Path(li_path).read_text())
    medium = li["media"][points["medium"]]
    betas = np.asarray(medium["beta_rad"])
    gammas = np.asarray(medium["gamma_rad"])
    keep = betas > -1.5
    betas = betas[keep]
    measured = np.stack(
        [np.asarray(medium["alpha_z_N_per_cm3"])[keep], np.asarray(medium["alpha_x_N_per_cm3"])[keep]], axis=-1
    )
    order = np.argsort(betas)
    betas, measured = betas[order], measured[order]
    literature = np.full_like(measured, np.nan)
    calibrated = np.full_like(measured, np.nan)
    role = np.full(measured.shape[:2], "", dtype=object)
    for entry in points["points"]:
        i = int(np.argmin(np.abs(betas - entry["beta_rad"])))
        j = int(np.argmin(np.abs(gammas - entry["gamma_rad"])))
        literature[i, j] = entry["literature_alpha_z_x"]
        calibrated[i, j] = entry["calibrated_alpha_z_x"]
        role[i, j] = entry["role"]
    return betas, gammas, measured, literature, calibrated, role


def _edges(centres_deg):
    centres = np.asarray(centres_deg)
    middle = 0.5 * (centres[1:] + centres[:-1])
    return np.concatenate([[2 * centres[0] - middle[0]], middle, [2 * centres[-1] - middle[-1]]])


def figure_rft_maps(li_path: Path):
    betas, gammas, measured, literature, calibrated, role = _rft_arrays(li_path)
    beta_deg, gamma_deg = np.degrees(betas), np.degrees(gammas)
    figure, axes = plt.subplots(2, 3, figsize=(7.4, 4.2), layout="constrained", sharex=True, sharey=True)
    columns = (("Measured (Li et al. 2013)", measured), ("MPM, literature parameters", literature),
               ("MPM, calibrated", calibrated))
    train = np.argwhere(role == "train")
    for row, (name, limit) in enumerate(((r"$\alpha_z$", 0.40), (r"$\alpha_x$", 0.15))):
        for col, (title, values) in enumerate(columns):
            ax = axes[row, col]
            image = ax.pcolormesh(_edges(gamma_deg), _edges(beta_deg), values[:, :, row], cmap="RdBu_r",
                                  vmin=-limit, vmax=limit, shading="flat", edgecolors="white", linewidth=0.4)
            ax.set_title(f"{title}: {name}", loc="left")
            if col == 2:
                ax.scatter(gamma_deg[train[:, 1]], beta_deg[train[:, 0]], s=26, marker="s", facecolor="none",
                           edgecolor=INK, linewidth=0.9, label="training points")
            if row == 1:
                ax.set_xlabel(r"intrusion angle $\gamma$ [deg]")
            if col == 0:
                ax.set_ylabel(r"attack angle $\beta$ [deg]")
            ax.set_xticks([-90, -45, 0, 45, 90])
            ax.set_yticks([-60, -30, 0, 30, 60, 90])
            ax.tick_params(length=2)
            for spine in ax.spines.values():
                spine.set_visible(False)
        bar = figure.colorbar(image, ax=axes[row, :].tolist(), shrink=0.92, pad=0.01, aspect=18)
        bar.set_label(f"{name} [N/cm³]")
        bar.outline.set_visible(False)
        bar.ax.tick_params(labelsize=6.5, length=2)
    axes[0, 2].legend(loc="lower right", handletextpad=0.3, borderpad=0.2, fontsize=6)
    save(
        figure,
        "rft-maps",
        "Stress per unit depth on the plate for loosely packed poppy seeds over the 78 distinct (β, γ) pairs. "
        "Left: measured by Li et al. (2013, table S5). Centre: Newton MPM with literature parameters only (no fitted "
        "parameter). Right: after calibration on the five training points marked by squares; the other 73 points are "
        "held out. Positive α_z is lift, positive α_x points along +x. Colour limits are ±0.40 and ±0.15 N/cm³.",
    )


def figure_rft_parity(li_path: Path):
    _betas, _gammas, measured, literature, calibrated, role = _rft_arrays(li_path)
    figure, axes = plt.subplots(1, 2, figsize=(6.6, 3.0), layout="constrained")
    valid = ~np.isnan(literature[:, :, 0])
    held = valid & (role == "validation")
    train = valid & (role == "train")
    for ax, channel, name in zip(axes, (0, 1), (r"$\alpha_z$", r"$\alpha_x$"), strict=True):
        grid(ax)
        obs = measured[:, :, channel]
        low = min(obs[valid].min(), literature[valid][:, channel].min(), calibrated[valid][:, channel].min())
        high = max(obs[valid].max(), literature[valid][:, channel].max(), calibrated[valid][:, channel].max())
        pad = 0.06 * (high - low)
        ax.plot([low - pad, high + pad], [low - pad, high + pad], color=MUTED, lw=0.7, zorder=1)
        for values, color, marker, text in ((literature, BLUE, "o", "literature"), (calibrated, AMBER, "o", "calibrated")):
            sim = values[:, :, channel]
            rmse = float(np.sqrt(np.mean((sim[held] - obs[held]) ** 2)))
            ax.scatter(obs[held], sim[held], s=13, color=color, marker=marker, edgecolor="white", linewidth=0.4,
                       zorder=3, label=f"{text}: held-out RMSE {rmse:.3f}")
        ax.scatter(obs[train], calibrated[:, :, channel][train], s=34, marker="s", facecolor="none", edgecolor=INK,
                   linewidth=0.9, zorder=4, label="calibrated, training points")
        ax.set_xlim(low - pad, high + pad)
        ax.set_ylim(low - pad, high + pad)
        ax.set_aspect("equal")
        ax.set_xlabel(f"measured {name} [N/cm³]")
        ax.set_ylabel(f"Newton MPM {name} [N/cm³]")
        ax.legend(loc="upper left" if channel == 0 else "lower right", handletextpad=0.2, borderaxespad=0.2)
    label(axes[0], "a")
    label(axes[1], "b")
    save(
        figure,
        "rft-parity",
        "Simulated against measured stress per unit depth for the 73 held-out points (circles) with literature "
        "parameters (blue) and calibrated parameters (amber); squares are the five training points of the calibrated "
        "model. The line marks equality. RMSE values are in N/cm³ over held-out points only.",
    )


def _generations(ax, search, scatter_color=STEEL):
    gens = sorted({c["generation"] for c in search["candidates"] if c["purpose"] == "train"})
    current = next(c["objective"] for c in search["candidates"] if c["purpose"] == "baseline")
    best_so_far = []
    for g in gens:
        values = [c["objective"] for c in search["candidates"] if c["generation"] == g and c["objective"] is not None]
        ax.scatter([g] * len(values), values, s=13, color=scatter_color, edgecolor="white", linewidth=0.4, zorder=3)
        current = min(current, min(values))
        best_so_far.append(current)
    ax.step(gens, best_so_far, where="mid", color=AMBER, lw=1.6, zorder=4, label="best training objective so far")
    ax.set_xticks(gens)
    return gens


def figure_rft_search():
    search = load("rft_search.json")
    cal = load("rft_calibration.json")
    figure, axes = plt.subplots(1, 2, figsize=(6.8, 3.0), layout="constrained", gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    grid(ax)
    kinds = ("analysis", "train", "baseline", "qualification")
    pts = [c for c in search["candidates"] if c["objective"] is not None and c["purpose"] in kinds]
    x = np.array([c["parameters"]["friction"] for c in pts])
    y = np.array([c["parameters"]["plate_friction"] for c in pts])
    j = np.array([c["objective"] for c in pts])
    is_train = np.array([c["purpose"] == "train" for c in pts])
    norm = LogNorm(vmin=0.5, vmax=8.0)
    sc = ax.scatter(x[is_train], y[is_train], c=j[is_train], cmap=OBJECTIVE, norm=norm, s=30, edgecolor="white",
                    linewidth=0.5, zorder=3)
    ax.scatter(x[~is_train], y[~is_train], c=j[~is_train], cmap=OBJECTIVE, norm=norm, s=22, marker="^",
               edgecolor="white", linewidth=0.5, zorder=3)
    base = cal["baseline_parameters"]
    best = cal["parameters"]
    ax.scatter([base["friction"]], [base["plate_friction"]], s=70, marker="D", color=BLUE, edgecolor="white",
               linewidth=0.8, zorder=5)
    ax.scatter([best["friction"]], [best["plate_friction"]], s=140, marker="*", color=AMBER, edgecolor="white",
               linewidth=0.6, zorder=6)
    bounds = {d["name"]: d["bounds"] for d in cal["parameter_definitions"]}
    ax.set_xlim(*bounds["friction"])
    ax.set_ylim(*bounds["plate_friction"])
    secondary = ax.secondary_xaxis(
        "top", functions=(lambda m: np.degrees(np.arcsin(np.clip(m, -1, 1))), lambda p: np.sin(np.radians(p)))
    )
    secondary.set_xlabel(r"Mohr–Coulomb angle $\phi = \arcsin\mu$ [deg]", fontsize=6.5)
    secondary.tick_params(labelsize=6)
    ax.set_xlabel(r"soil friction $\mu$")
    ax.set_ylabel(r"plate friction $\mu_p$")
    objective_bar(figure, ax, sc, [0.5, 1, 2, 4, 8], rect=(0.93, 0.05, 0.03, 0.36))
    label(ax, "a")

    ax = axes[1]
    grid(ax)
    _generations(ax, search)
    baseline = next(c["objective"] for c in search["candidates"] if c["purpose"] == "baseline")
    held = cal["validation"]["experiments"][0]["objective"]
    ax.axhline(baseline, color=BLUE, lw=1.0, ls=(0, (4, 2)), label=f"training objective, literature start ({baseline:.2f})")
    ax.axhline(held, color=INK, lw=0.9, ls=(0, (1, 1.5)), label=f"held-out objective, accepted ({held:.2f})")
    ax.set_xlabel("CMA-ES generation (6 candidates each)")
    ax.set_ylabel("objective J")
    ax.set_ylim(0, None)
    ax.legend(loc="upper right", fontsize=6)
    label(ax, "b")
    marker_legend(
        figure,
        [
            ("CMA-ES candidate", "o", 5, STEEL, "white"),
            ("qualification or sensitivity probe", "^", 5, STEEL, "white"),
            ("literature start", "D", 5.5, BLUE, "white"),
            ("accepted", "*", 9, AMBER, "white"),
        ],
        ncol=4,
    )
    save(
        figure,
        "rft-search",
        "Calibration of the plate-intrusion replica. (a) Every simulated candidate in the parameter box, coloured by "
        "the training objective J over five points (darker is better); triangles are the numerical-qualification and "
        "sensitivity probes that precede the search. (b) Training objective of each CMA-ES candidate by generation and "
        "the best value so far; the search stopped after its budget of five generations. The dotted line is the "
        "held-out objective of the accepted candidate over 73 points (acceptance limit 1.0).",
    )


MEDIA_LABELS = {
    "poppy_seeds_LP": "Poppy seeds, LP",
    "poppy_seeds_CP": "Poppy seeds, CP",
    "glass_0p3mm_LP": "0.3 mm glass, LP",
    "glass_0p3mm_CP": "0.3 mm glass, CP",
    "glass_3mm_CP": "3 mm glass, CP",
}


def _outcome(entry) -> str:
    """Short calibration outcome for a panel title."""
    names = {"accepted": "accepted", "rejected_validation": "rejected in validation", "rejected_numerics": "time-step check failed"}
    parts = []
    for key, label in (("dt_1ms", "1 ms"), ("dt_0.5ms", "0.5 ms")):
        record = entry["calibrations"].get(key)
        if record:
            parts.append(f"{label}: {names.get(record['acceptance'], record['acceptance'])}")
    return "\n".join(parts) if parts else "not calibrated"


def figure_rft_media(li_path: Path):
    path = DATA / "rft_media.json"
    if not path.is_file():
        print("skip rft-media (no data)")
        return
    record = json.loads(path.read_text())
    li = json.loads(Path(li_path).read_text())
    train = np.asarray(record["training_points_rad"])
    media = list(record["media"])
    figure, axes = plt.subplots(2, len(media), figsize=(7.8, 3.75), layout="constrained")
    for col, medium in enumerate(media):
        entry = record["media"][medium]
        data = li["media"][medium]
        betas, gammas = np.asarray(data["beta_rad"]), np.asarray(data["gamma_rad"])
        points = np.asarray(entry["points_rad"])
        measured = np.array(
            [
                [
                    data["alpha_z_N_per_cm3"][int(np.argmin(np.abs(betas - b)))][int(np.argmin(np.abs(gammas - g)))],
                    data["alpha_x_N_per_cm3"][int(np.argmin(np.abs(betas - b)))][int(np.argmin(np.abs(gammas - g)))],
                ]
                for b, g in points
            ]
        )
        held = ~np.array([np.isclose(train, p, atol=1e-3).all(axis=1).any() for p in points])
        literature = np.asarray(entry["literature"])
        calibrated = np.asarray(entry["calibrated"]) if "calibrated" in entry else None
        for row, name in enumerate(("z", "x")):
            ax = axes[row, col]
            grid(ax)
            obs = measured[:, row]
            series = [literature[:, row]] + ([calibrated[:, row]] if calibrated is not None else [])
            low = min(obs.min(), *(v.min() for v in series))
            high = max(obs.max(), *(v.max() for v in series))
            pad = 0.06 * (high - low)
            ax.plot([low - pad, high + pad], [low - pad, high + pad], color=MUTED, lw=0.6, zorder=1)
            ax.scatter(obs[held], literature[held, row], s=7, color=BLUE, edgecolor="none", zorder=3)
            text = f"lit {np.sqrt(np.mean((literature[held, row] - obs[held]) ** 2)):.3f}"
            if calibrated is not None:
                ax.scatter(obs[held], calibrated[held, row], s=7, color=AMBER, edgecolor="none", zorder=4)
                text += f"\ncal {np.sqrt(np.mean((calibrated[held, row] - obs[held]) ** 2)):.3f}"
            ax.text(0.04, 0.96, text, transform=ax.transAxes, fontsize=5.6, va="top", ha="left", color=INK)
            ax.set_xlim(low - pad, high + pad)
            ax.set_ylim(low - pad, high + pad)
            ax.tick_params(labelsize=5.5, length=2)
            if row == 0:
                ax.set_title(
                    f"{MEDIA_LABELS[medium]}\n{entry['angle_deg']:g}°, {entry['bulk_density']:g} kg/m³\n{_outcome(entry)}",
                    fontsize=6.2,
                )
            if col == 0:
                ax.set_ylabel(f"MPM α$_{name}$ [N/cm³]", fontsize=6.5)
            ax.set_xlabel(f"measured α$_{name}$ [N/cm³]", fontsize=6.5)
    marker_legend(
        figure,
        [
            ("literature parameters", "o", 4.5, BLUE, BLUE),
            ("calibrated", "o", 4.5, AMBER, AMBER),
        ],
        ncol=2,
    )
    save(
        figure,
        "rft-media",
        "Simulated against measured stress per unit depth for the five media of Li et al. (2013), 73 held-out points "
        "each: with literature parameters (blue; friction from the table S1 angle, plate friction 0.40) and after "
        "calibration where one was accepted (amber). Numbers are held-out RMSE in N/cm³. Axes differ between media.",
    )


# ----------------------------------------------------------------------------- clay


def figure_clay():
    record = load("clay_laboratory.json")
    order = ("test1", "test8", "test3", "test6")
    figure, axes = plt.subplots(
        1, 5, figsize=(7.6, 2.55), layout="constrained", gridspec_kw={"width_ratios": [1, 1, 1, 1, 0.78]}
    )
    for ax, name in zip(axes[:4], order, strict=True):
        test = record["tests"][name]
        grid(ax)
        curve = test["measured_curve"]
        top = test["clay_height_m"] * 1000.0
        ax.axvspan(test["depth_mm"][0], test["depth_mm"][-1], color=WASH, zorder=0, lw=0)
        ax.plot(curve["penetration_mm"], curve["net_pressure_kPa"], color=INK, lw=1.5, label="centrifuge", zorder=3)
        ax.plot(test["depth_mm"], test["predicted_kPa"], color=BLUE, lw=1.3, marker="o", ms=2.2,
                label=f"MPM, laboratory $s_u$", zorder=4)
        ax.set_title(
            f"{name.replace('test', 'Test ')} · B = {test['plate_width_m'] * 1000:.0f} mm\n"
            f"{test['boundary']} · RMSE {test['rmse_kPa']:.1f} kPa", loc="left", fontsize=6.8,
        )
        ax.set_xlabel("penetration [mm]")
        ax.set_xlim(0, top - 2)
        ax.set_ylim(0, 260)
        ax.tick_params(length=2)
    axes[0].set_ylabel("net pressure [kPa]")
    axes[0].legend(loc="upper left", fontsize=6)
    ax = axes[4]
    grid(ax)
    setup = record["setup"]
    params = record["parameters"]
    g = setup["g_level"] * setup["gravity"]
    gamma_b = (setup["saturated_unit_weight_1g"] / setup["gravity"] - setup["water_density"]) * g
    z = np.linspace(0, 0.097, 200)
    sv = gamma_b * z
    ocr = np.maximum(params["preconsolidation"] / np.maximum(sv, 1e-9), 1.0)
    su = params["su_ratio"] * ocr ** params["shansep_exponent"] * sv
    ax.plot(su / 1000, z * 1000, color=BLUE, lw=1.4, label="SHANSEP,\nlaboratory")
    depth_nc = params["preconsolidation"] / gamma_b * 1000
    ax.axhline(depth_nc, color=MUTED, lw=0.6, ls=(0, (3, 2)))
    ax.text(0.4, depth_nc - 2.5, r"$\sigma'_v = \sigma'_p$", fontsize=6, color=MUTED)
    ax.invert_yaxis()
    ax.set_xlabel(r"$s_u$ [kPa]")
    ax.set_ylabel("depth below clay surface [mm]")
    ax.set_xlim(0, 14)
    ax.set_title("strength profile", loc="left")
    ax.legend(loc="lower left", fontsize=6)
    for a, text in zip(axes, "abcde", strict=True):
        label(a, text)
    save(
        figure,
        "clay-curves",
        "Load–penetration curves of the four centrifuge tests (black) and the Newton MPM prediction with the laboratory "
        "SHANSEP strength (blue), without any fitted parameter. Shaded: the scored range, from 5 mm of penetration to "
        "where 12 mm of clay remain below the plate. (e) The laboratory undrained strength profile at 100 g; below the "
        "dashed line the clay is normally consolidated.",
    )


def figure_force_series():
    clay = json.loads((DATA / "clay_force_series_test1.json").read_text())
    grain = json.loads((DATA / "rft_force_series.json").read_text())
    figure, axes = plt.subplots(2, 1, figsize=(6.8, 4.6), layout="constrained")

    ax = axes[0]
    grid(ax)
    pen = np.asarray(clay["penetration_mm"])
    p = np.asarray(clay["pressure_kPa"])
    lo, hi = 3.0, 21.0
    for layer in clay["particle_layers_mm"]:
        if lo <= layer <= hi:
            ax.axvline(layer, color=MUTED, lw=0.5, ls=(0, (1, 2)), zorder=1)
    sel = (pen >= lo) & (pen <= hi)
    ax.plot(pen[sel], p[sel], color=STEEL, lw=0.8, label="every step", zorder=2)
    ax.plot(clay["depths_mm"], clay["window_mean_kPa"], "o", color=BLUE, ms=4.2, zorder=4,
            label="1 mm window mean (scored)")
    ax.plot(clay["depths_mm"], clay["window_median_kPa"], "s", mfc="white", mec=INK, ms=4.5, zorder=5,
            label="1 mm window median")
    ax.set_xlabel("penetration [mm]")
    ax.set_ylabel("net pressure [kPa]")
    ax.set_xlim(lo, hi)
    ax.set_ylim(0, 64)
    ax.set_title("Clay, centrifuge test 1: regular particle lattice, 2.5 mm voxels, 0.5 ms steps", loc="left", fontsize=7)
    ax.legend(loc="lower right", fontsize=6, ncol=3, framealpha=0.9, frameon=True, edgecolor="none")
    label(ax, "a")

    ax = axes[1]
    grid(ax)
    depth = np.asarray(grain["depth_m"]) * 1000.0
    force = np.asarray(grain["force_z_N"])
    window = np.asarray(grain["fit_window_m"]) * 1000.0
    slope = np.sum(force[(depth >= window[0]) & (depth <= window[1])] * depth[(depth >= window[0]) & (depth <= window[1])]) / np.sum(depth[(depth >= window[0]) & (depth <= window[1])] ** 2)
    sel = (depth >= window[0]) & (depth <= 60.0)
    ratio = force / (slope * np.maximum(depth, 1e-9))
    ax.axhline(1.0, color=INK, lw=0.7, zorder=2)
    ax.set_xticks(np.arange(18.0, 61.0, 6.0))
    ax.plot(depth[sel], ratio[sel], color=STEEL, lw=0.8, zorder=3, label="every step")
    profile = grain["voxel_phase_profile"]["values"]
    ax.set_xlabel("plate depth [mm]")
    ax.set_ylabel("force / fitted trend")
    ax.set_xlim(window[0], 60.0)
    ax.set_ylim(0.4, 1.6)
    ax.set_title(
        f"Granular, broadside intrusion: 25 % particle jitter, 6 mm voxels, 1 ms steps; phase average "
        f"{min(profile):.2f}–{max(profile):.2f}",
        loc="left", fontsize=7,
    )
    label(ax, "b")
    save(
        figure,
        "force-series",
        "Plate force at every step for one path of each validation case. (a) Centrifuge test 1: dotted lines mark the "
        "penetrations at which the plate underside reaches a layer of particle centres (1.25 mm lattice, two layers per "
        "voxel). About 0.1 mm before each layer the contact force collapses for two to three steps, and the scored 1 mm "
        "window means read 9–14 % below the window medians. (b) Broadside intrusion into poppy seeds, force divided by "
        "the fitted linear trend: the force alternates with the 6 mm voxel period (ticks) between about 0.7 and 1.3 "
        "times the trend. The depth fit averages over these oscillations; excluding their low phase would raise the "
        "fitted slope by 2.7 %.",
    )


# ----------------------------------------------------------------------------- rake


def figure_rake_forces():
    record = load("rake_traces.json")
    order = ("train_shallow", "train_deep", "train_pitched", "holdout_fast_offset", "holdout_pitched_back")
    titles = {
        "train_shallow": "Training\n12 mm, 0°, 80 mm/s",
        "train_deep": "Training\n45 mm, 0°, 80 mm/s",
        "train_pitched": "Training\n30 mm, +20°, 80 mm/s",
        "holdout_fast_offset": "Held out\n25 mm, 0°, 100 mm/s, offset",
        "holdout_pitched_back": "Held out\n35 mm, −20°, 60 mm/s",
    }
    figure, axes = plt.subplots(2, 5, figsize=(7.6, 3.5), layout="constrained", sharey="row")
    for col, name in enumerate(order):
        stroke = record["strokes"][name]
        t = np.asarray(stroke["time_s"])
        logged = np.asarray(stroke["logged_N"])
        calibrated = np.asarray(stroke["calibrated_N"])
        baseline = None if stroke["baseline_N"] is None else np.asarray(stroke["baseline_N"])
        scored = np.asarray(stroke["scored"])
        for row, (axis, sign, ylabel) in enumerate(((0, -1.0, "drag force −$f_x$ [N]"), (2, 1.0, "vertical force $f_z$ [N]"))):
            ax = axes[row, col]
            grid(ax)
            if scored.any():
                ax.axvspan(t[scored].min(), t[scored].max(), color=WASH, lw=0, zorder=0)
            ax.plot(t, sign * logged[:, axis], color=INK, lw=1.3, label="logged (synthetic)", zorder=3)
            if baseline is not None:
                ax.plot(t, sign * baseline[:, axis], color=BLUE, lw=1.1, ls=(0, (4, 2)), label="baseline", zorder=3)
            ax.plot(t, sign * calibrated[:, axis], color=AMBER, lw=1.3, label="calibrated", zorder=4)
            if row == 0:
                ax.set_title(titles[name], loc="left", fontsize=6.8)
            else:
                ax.set_xlabel("time [s]")
            if col == 0:
                ax.set_ylabel(ylabel)
            ax.tick_params(length=2)
    axes[0, 0].legend(loc="upper left", fontsize=5.8, handlelength=1.6)
    save(
        figure,
        "rake-forces",
        "Soil force on the rake for the five strokes of the demonstration, in the robot base frame: the synthetic log "
        "(black), the simulation with the baseline guess (blue, training strokes only) and with the calibrated "
        "parameters (amber). Shaded: the scored steady drag. The calibrated soil matches the training strokes and the "
        "two held-out strokes with 1.0–2.0 N RMSE per stroke.",
    )


def figure_rake_search():
    search = load("rake_search.json")
    result = load("rake_result.json")
    hidden, base, best = (result["parameters"][k] for k in ("hidden", "baseline", "calibrated"))
    kinds = ("analysis", "train", "baseline", "qualification")
    pts = [c for c in search["candidates"] if c["objective"] is not None and c["purpose"] in kinds]
    f = np.array([c["parameters"]["friction"] for c in pts])
    co = np.array([c["parameters"]["cohesion"] for c in pts])
    tf = np.array([c["parameters"]["tool_friction"] for c in pts])
    j = np.array([c["objective"] for c in pts])
    is_train = np.array([c["purpose"] == "train" for c in pts])
    norm = LogNorm(vmin=0.3, vmax=30.0)
    figure, axes = plt.subplots(1, 3, figsize=(7.6, 2.9), layout="constrained")
    for ax, (yv, yl, yk) in zip(
        axes[:2],
        ((f, r"soil friction $\mu$", "friction"), (tf, r"tool friction $\mu_t$", "tool_friction")),
        strict=True,
    ):
        grid(ax)
        sc = ax.scatter(co[is_train], yv[is_train], c=j[is_train], cmap=OBJECTIVE, norm=norm, s=22, edgecolor="white",
                        linewidth=0.4, zorder=3)
        ax.scatter(co[~is_train], yv[~is_train], c=j[~is_train], cmap=OBJECTIVE, norm=norm, s=18, marker="^",
                   edgecolor="white", linewidth=0.4, zorder=3)
        ax.scatter([base["cohesion"]], [base[yk]], s=60, marker="D", color=BLUE, edgecolor="white", lw=0.8, zorder=5)
        ax.scatter([hidden["cohesion"]], [hidden[yk]], s=110, marker="P", color=INK, edgecolor="white", lw=0.6,
                   zorder=6)
        ax.scatter([best["cohesion"]], [best[yk]], s=140, marker="*", color=AMBER, edgecolor="white", lw=0.6, zorder=7)
        ax.set_xscale("log")
        ax.set_xlim(20, 600)
        ax.xaxis.set_major_locator(FixedLocator([20, 50, 100, 200, 500]))
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _p: f"{v:g}"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("cohesion [Pa]")
        ax.set_ylabel(yl)
    axes[0].set_ylim(0.35, 0.80)
    axes[1].set_ylim(0.05, 0.65)
    objective_bar(figure, axes[1], sc, [0.3, 1, 3, 10, 30], rect=(0.91, 0.05, 0.035, 0.40))
    ax = axes[2]
    grid(ax)
    _generations(ax, search)
    validation = result["objective"]["validation"]["experiments"][0]["objective"]
    ax.axhline(validation, color=INK, lw=0.9, ls=(0, (1, 1.5)), label=f"held-out objective, accepted ({validation:.2f})")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _p: f"{v:g}"))
    ax.set_xlabel("CMA-ES generation (8 candidates each)")
    ax.set_ylabel("objective J")
    ax.legend(loc="upper right", fontsize=6)
    for a, text in zip(axes, "abc", strict=True):
        label(a, text)
    marker_legend(
        figure,
        [
            ("CMA-ES candidate", "o", 5, STEEL, "white"),
            ("qualification or sensitivity probe", "^", 5, STEEL, "white"),
            ("baseline guess", "D", 5.5, BLUE, "white"),
            ("hidden parameters (generated the logs)", "P", 7, INK, "white"),
            ("accepted", "*", 9, AMBER, "white"),
        ],
        ncol=5,
    )
    save(
        figure,
        "rake-search",
        "Calibration of the rake demonstration. (a, b) Every simulated candidate coloured by the training objective J "
        "(darker is better), with the baseline guess, the hidden parameters that generated the synthetic logs and the "
        "accepted candidate. Low objectives spread along a valley in which cohesion trades against friction and tool "
        "friction. (c) Training objective of each candidate by CMA-ES generation and the best value so far; the search "
        "stopped after four generations without improvement. The dotted line is the held-out objective of the accepted "
        "candidate.",
    )


def _captions_from_paper() -> dict:
    """Figure numbers and captions as printed in index.html, keyed by figure file name."""
    import re  # noqa: PLC0415

    path = REPORT / "index.html"
    if not path.is_file():
        return {}
    html = path.read_text()
    out = {}
    for match in re.finditer(
        r'<img src="figures/([\w-]+)\.svg".*?<figcaption><strong>Figure (A?\d+)\.</strong>(.*?)</figcaption>', html, re.S
    ):
        text = re.sub(r"<[^>]+>", "", match.group(3))
        number = match.group(2)
        out[match.group(1)] = {"number": int(number) if number.isdigit() else number, "caption": " ".join(text.split())}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--li", type=Path, required=True, help="processed Li et al. (2013) table li2013_rft.json")
    args = parser.parse_args()
    figure_setups()
    figure_rft_maps(args.li)
    figure_rft_parity(args.li)
    figure_rft_search()
    figure_rft_media(args.li)
    figure_clay()
    figure_force_series()
    figure_rake_forces()
    figure_rake_search()
    # figures_rev3 imports this module by name; register the running script under that name so that its
    # figures record their captions in the CAPTIONS used for figures.json.
    sys.modules.setdefault("make_figures", sys.modules[__name__])
    import figures_rev3  # noqa: PLC0415 (revision 3 figures; imports this module)

    figure_rev3 = (figures_rev3.figure_setups_sand, figures_rev3.figure_discretization, figures_rev3.figure_rod_packing,
                   figures_rev3.figure_plate_sinkage, figures_rev3.figure_pullout, figures_rev3.figure_probe)
    for function in figure_rev3:
        function()
    paper = _captions_from_paper()
    manifest = {
        "captions": "taken from index.html when the paper contains the figure",
        "figures": [
            {
                "id": k,
                "number": paper.get(k, {}).get("number"),
                "caption": paper.get(k, {}).get("caption", v),
                "svg": f"{k}.svg",
                "pdf": f"{k}.pdf",
                "png": f"{k}.png",
            }
            for k, v in CAPTIONS.items()
        ],
    }
    (FIGURES / "figures.json").write_text(json.dumps(manifest, indent=1) + "\n")


if __name__ == "__main__":
    main()

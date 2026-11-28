# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Figures added in revision 3: the four sand and soil experiments and the discretization study.

Imported by ``make_figures.py``; reads the report data files only.
"""

from __future__ import annotations

import json
import math

import numpy as np
from matplotlib import patches
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from make_figures import (
    AMBER,
    BLUE,
    DATA,
    INK,
    LINE,
    MUTED,
    SOIL,
    STEEL,
    _arrow,
    _panel_title,
    grid,
    label,
    marker_legend,
    save,
)

SAND = "#ecdcb9"
MOIST = "#d9c49b"
LOAM = "#cdb48d"


def _load(name):
    return json.loads((DATA / name).read_text())


def figure_setups_sand():
    figure, axes = plt.subplots(2, 2, figsize=(7.0, 5.0))
    axes = axes.ravel()
    figure.subplots_adjust(left=0.02, right=0.98, top=0.9, bottom=0.02, wspace=0.12, hspace=0.32)
    for ax in axes:
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_anchor("N")

    # (a) Rod intrusion at controlled packing.
    ax = axes[0]
    _panel_title(ax, "a", "Rod intrusion at set packing", r"Ruck et al. (2025): slope $K = dF/dz$")
    ax.add_patch(patches.Rectangle((-10.8, -16), 21.6, 16, facecolor=SAND, edgecolor="none"))
    ax.plot([-10.8, -10.8, 10.8, 10.8], [2.0, -17.2, -17.2, 2.0], color=MUTED, lw=1.2)
    ax.add_patch(patches.Rectangle((-10.8, -17.2), 21.6, 1.2, facecolor="white", edgecolor=MUTED, lw=0.5, hatch="....."))
    for x in (-6, 0, 6):
        _arrow(ax, (x, -20.0), (x, -17.6), color=MUTED, lw=0.6)
    ax.text(0, -21.0, "air (fluidize, pulse, tap)", ha="center", va="top", fontsize=6.0, color=MUTED)
    ax.add_patch(patches.Rectangle((-0.635, -7.5), 1.27, 11.0, facecolor=STEEL, edgecolor=INK, lw=0.6, zorder=3))
    _arrow(ax, (2.6, 2.5), (2.6, -1.0), color=AMBER, lw=1.1)
    ax.text(3.1, 0.8, "2 cm/s", fontsize=6.2, color=AMBER)
    _arrow(ax, (-3.0, 0.0), (-3.0, -7.5), color=MUTED, lw=0.6, style="<|-|>")
    ax.text(-3.4, -3.8, "10 cm", fontsize=6.0, color=MUTED, ha="right", va="center")
    ax.text(0, -12.0, "sand, glass beads or LHS-1\n" + r"at $\phi$ = 0.49–0.625", ha="center", va="center",
            fontsize=6.0, color=MUTED)
    ax.text(0, -23.4, "21.6 cm chamber, 12.7 mm rod", ha="center", va="top", fontsize=6.0, color=MUTED)
    ax.set_xlim(-12, 12)
    ax.set_ylim(-25, 4)

    # (b) Rod pull-out.
    ax = axes[1]
    _panel_title(ax, "b", "Rod pull-out", "Douglas et al. (2026): peak force")
    ax.add_patch(patches.Rectangle((-11, -26), 22, 26, facecolor=SAND, edgecolor="none"))
    ax.plot([-11, 11], [0, 0], color=MUTED, lw=0.5)
    ax.add_patch(patches.Rectangle((-0.635, -15.0), 1.27, 10.0, facecolor="#c9a46a", edgecolor=INK, lw=0.6, zorder=3))
    ax.plot([0, 0], [-5.0, 4.5], color=INK, lw=0.6)
    _arrow(ax, (2.6, 0.5), (2.6, 4.0), color=AMBER, lw=1.1)
    ax.text(3.1, 1.6, "pull", fontsize=6.2, color=AMBER)
    _arrow(ax, (-3.0, 0.0), (-3.0, -5.0), color=MUTED, lw=0.6, style="<|-|>")
    ax.text(-3.4, -2.5, "d", fontsize=6.5, color=MUTED, ha="right", va="center")
    _arrow(ax, (-3.0, -5.0), (-3.0, -15.0), color=MUTED, lw=0.6, style="<|-|>")
    ax.text(-3.4, -10.0, "L", fontsize=6.5, color=MUTED, ha="right", va="center")
    ax.text(0, -21.5, "medium sand, 1630 kg/m³", ha="center", fontsize=6.0, color=MUTED)
    ax.text(0, -27.0, "rods 4.8–12.7 mm wide,\nL = 5–25 cm, d = 0–20 cm", ha="center", va="top", fontsize=6.0, color=MUTED)
    ax.set_xlim(-12, 12)
    ax.set_ylim(-31, 6)

    # (c) Probe penetration with five tip shapes.
    ax = axes[2]
    _panel_title(ax, "c", "Probe penetration, dry loam", "Mishra et al. (2018): force against depth")
    ax.add_patch(patches.Rectangle((-13, -20), 26, 20, facecolor=LOAM, edgecolor="none"))
    ax.plot([-13, -13, 13, 13], [1.5, -20, -20, 1.5], color=MUTED, lw=1.2)
    xs = (-9.6, -4.8, 0.0, 4.8, 9.6)
    names = ("flat", "ellipt.", "parab.", "cone", "root")
    r = 1.1
    for x, name in zip(xs, names):
        s = np.linspace(0, 1, 30)
        length = 2.2
        if name == "flat":
            prof = np.ones_like(s)
            length = 0.0
        elif name == "ellipt.":
            prof = np.sqrt(1 - (1 - s) ** 2)
        elif name == "parab.":
            prof = np.sqrt(s)
        elif name == "cone":
            prof = s
        else:
            prof = s**0.85
        tip_y = -6.0
        y = tip_y + s * length
        xr = r * prof
        poly = np.concatenate([np.stack([x + xr, y], 1), np.array([[x + r, 6.0], [x - r, 6.0]]),
                               np.stack([x - xr[::-1], y[::-1]], 1)])
        if name == "flat":
            poly = np.array([[x - r, tip_y], [x + r, tip_y], [x + r, 6.0], [x - r, 6.0]])
        ax.add_patch(patches.Polygon(poly, closed=True, facecolor=STEEL, edgecolor=INK, lw=0.5, zorder=3))
        ax.text(x, -7.2, name, ha="center", va="top", fontsize=5.4, color=INK)
    ax.text(0, -13.5, "sandy loam, oven-dried,\n1560 kg/m³", ha="center", va="center", fontsize=6.0, color=MUTED)
    ax.text(0, -21.5, "22 × 16 × 20 cm; D = 3–11 mm;\n10 mm/min to 100 mm", ha="center", va="top", fontsize=6.0,
            color=MUTED)
    ax.set_xlim(-14, 14)
    ax.set_ylim(-26, 7)

    # (d) Plate sinkage in a soil bin.
    ax = axes[3]
    _panel_title(ax, "d", "Plate sinkage, moist field sand", "Kruger et al. (2023): pressure against sinkage")
    ax.add_patch(patches.Rectangle((-16, -14), 32, 14, facecolor=MOIST, edgecolor="none"))
    ax.plot([-16, -16, 16, 16], [1.5, -14, -14, 1.5], color=MUTED, lw=1.2)
    ax.add_patch(patches.Rectangle((-3.75, -1.6), 7.5, 1.2, facecolor=STEEL, edgecolor=INK, lw=0.6, zorder=3))
    ax.plot([0, 0], [-0.4, 5.0], color=INK, lw=0.9)
    _arrow(ax, (5.2, 4.5), (5.2, 1.0), color=AMBER, lw=1.1)
    ax.text(5.8, 3.0, "16 mm/s", fontsize=6.2, color=AMBER)
    ax.text(0, -6.0, "Cullinan sand, 6.5 % water", ha="center", fontsize=6.0, color=MUTED)
    ax.text(0, -8.8, "five preparations, dry 1390–1538 kg/m³", ha="center", fontsize=6.0, color=MUTED)
    ax.text(0, -15.5, "150 mm plate to 80 mm; 2 m wide bin", ha="center", va="top", fontsize=6.0, color=MUTED)
    ax.set_xlim(-17, 17)
    ax.set_ylim(-18.5, 6)

    save(figure, "setups-sand", "Sand and soil experiments added in revision 3.")


def figure_discretization():
    study = _load("resolution_study.json")
    media = ["poppy_seeds_LP", "poppy_seeds_CP", "glass_0p3mm_LP", "glass_0p3mm_CP", "glass_3mm_CP"]
    labels = {"poppy_seeds_LP": "poppy LP", "poppy_seeds_CP": "poppy CP", "glass_0p3mm_LP": "0.3 mm glass LP",
              "glass_0p3mm_CP": "0.3 mm glass CP", "glass_3mm_CP": "3 mm glass CP"}
    variants = [("v6-seed14", "other arrangement"), ("v4", "4 mm voxels"), ("v3", "3 mm voxels"),
                ("v6-ppc3", "27 particles/voxel"), ("v6-pic27", "pic27 contact"), ("v6-p1d", "P1d strain")]
    figure, (ax, bx) = plt.subplots(1, 2, figsize=(7.4, 3.1), gridspec_kw={"width_ratios": [1.45, 1.0]},
                                    layout="constrained")
    colors = plt.cm.viridis(np.linspace(0.05, 0.85, len(media)))
    for j, (key, name) in enumerate(variants):
        for i, medium in enumerate(media):
            entry = study["granular"].get(medium, {}).get("variants", {}).get(key)
            if not entry or "sets" not in entry:
                continue
            for s, marker in (("literature", "o"), ("calibrated", "s")):
                value = entry["sets"].get(s, {}).get("max_change_vs_v6_scales")
                if value is None:
                    continue
                ax.scatter(max(value, 0.03), j + (i - 2) * 0.11, s=12, marker=marker, color=colors[i],
                           edgecolor="none", zorder=3)
    ax.axvline(3.0, color=AMBER, lw=0.8, ls=(0, (3, 2)))
    ax.text(3.3, -0.35, "acceptance limit", color=AMBER, fontsize=6.0, va="center")
    ax.set_xscale("log")
    ax.set_xlim(0.03, 40)
    ax.set_yticks(range(len(variants)), [n for _, n in variants])
    ax.invert_yaxis()
    ax.set_xlabel("largest change at the 11 diagnostic motions [scales]")
    grid(ax)
    label(ax, "a")
    handles = [Line2D([], [], ls="none", marker="o", color=c, markersize=4, label=labels[m]) for m, c in zip(media, colors)]
    handles += [Line2D([], [], ls="none", marker="o", color=MUTED, markersize=4, label="literature"),
                Line2D([], [], ls="none", marker="s", color=MUTED, markersize=4, label="calibrated")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.22), fontsize=5.8, ncol=4,
              handletextpad=0.2, columnspacing=0.8)

    clay = {test: dict(values) for test, values in study["clay"].items() if test.startswith("test")}
    earlier = _load("clay_laboratory_dt025.json")["tests"]
    for test in ("test1", "test3"):
        clay.setdefault(test, {})["v25-dt025"] = {"rmse_kPa": earlier[test]["rmse_kPa"]}
    order = [("v25-dt05", "2.5 mm, 0.5 ms"), ("v25-dt025", "2.5 mm, 0.25 ms"), ("v125-dt05", "1.25 mm, 0.5 ms"),
             ("v125-dt025", "1.25 mm, 0.25 ms"), ("v25-pic27-dt05", "pic27, 0.5 ms"), ("v25-pic27-dt025", "pic27, 0.25 ms")]
    for k, test in enumerate(("test1", "test3")):
        values = [clay.get(test, {}).get(v, {}).get("rmse_kPa", np.nan) for v, _ in order]
        bx.barh(np.arange(len(order)) + (k - 0.5) * 0.38, values, height=0.36, color=(BLUE if k == 0 else STEEL),
                label=("test 1, 100 mm smooth" if k == 0 else "test 3, 20 mm smooth"))
    bx.set_yticks(range(len(order)), [n for _, n in order])
    bx.invert_yaxis()
    bx.set_xlabel("clay RMSE against measurement [kPa]")
    grid(bx)
    label(bx, "b")
    bx.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), fontsize=6.0, ncol=1)
    save(figure, "discretization", "Discretization study.")


MEDIA_ROD = (("sand", "Silica sand"), ("glass_beads", "Glass beads"), ("lhs1", "LHS-1 simulant"))


def _law(phi, anchors, values):
    return np.interp(phi, anchors, values)


def _calibrated_law(entry):
    """Packing-law parameters of a rod calibration (released or best candidate), or None."""
    cal = entry.get("calibration") or {}
    values = cal.get("parameters") or (cal.get("best_candidate") or {}).get("parameters")
    if values and all(k in values for k in ("mu_loose", "mu_c", "mu_dense")):
        return values
    return None


def figure_rod_packing():
    data = _load("rod_intrusion.json")
    calibrated = {m: _calibrated_law(data.get(m, {})) for m, _ in MEDIA_ROD}
    with_law = [m for m, v in calibrated.items() if v]
    if with_law:
        figure, axes = plt.subplots(1, 4, figsize=(7.4, 2.35), layout="constrained",
                                    gridspec_kw={"width_ratios": [1, 1, 1, 1.05]})
    else:
        figure, axes = plt.subplots(1, 3, figsize=(7.0, 2.35), layout="constrained")
    for ax, (medium, title) in zip(axes[:3], MEDIA_ROD):
        entry = data.get(medium, {})
        lit = entry.get("literature")
        if lit is None:
            continue
        phi = np.asarray(lit["packings"])
        ax.errorbar(phi, lit["measured"], yerr=[s or 0 for s in lit["measured_std"]], fmt="o", ms=3, color=INK,
                    elinewidth=0.6, capsize=1.5, zorder=3)
        ax.plot(phi, lit["predicted"], color=BLUE, lw=1.0, marker="o", ms=2.2, zorder=4)
        cal = entry.get("calibration") or {}
        train, held = cal.get("train_calibrated"), cal.get("holdout_calibrated")
        best = entry.get("best_law_sweep")
        if best:
            bp = np.asarray(best["packings"])
            bk = np.asarray(best["predicted"])
            in_train = np.isin(np.round(bp, 4), np.round(best["train_packings"], 4))
            ax.plot(bp, bk, color=AMBER, lw=1.0, ls=(0, (4, 2)), zorder=4)
            ax.scatter(bp[in_train], bk[in_train], marker="s", s=13, facecolor="white", edgecolor=AMBER, lw=0.9, zorder=5)
            ax.scatter(bp[~in_train], bk[~in_train], marker="o", s=11, facecolor="white", edgecolor=AMBER, lw=0.9, zorder=5)
        elif calibrated[medium] and train and held and np.isfinite(train).all() and np.isfinite(held).all():
            tp, hp = np.asarray(cal["train_packings"]), np.asarray(cal["holdout_packings"])
            order = np.argsort(np.concatenate([tp, hp]))
            both = np.concatenate([train, held])[order]
            ax.plot(np.concatenate([tp, hp])[order], both, color=AMBER, lw=1.0, zorder=4)
            ax.scatter(tp, train, marker="s", s=13, color=AMBER, zorder=5)
            ax.scatter(hp, held, marker="o", s=9, color=AMBER, zorder=5)
        ax.axvline(entry["critical_fraction"], color=MUTED, lw=0.6, ls=(0, (3, 2)))
        ax.set_title(title, fontsize=7.0)
        ax.set_xlabel(r"packing fraction $\phi$")
        grid(ax)
        if medium == "sand":
            ax.set_ylabel("slope K [N/m]")
    if with_law:
        law_ax = axes[3]
        for medium, title in MEDIA_ROD:
            values = calibrated[medium]
            if not values:
                continue
            entry = data[medium]
            phi = np.asarray(entry["literature"]["packings"])
            phi_c = entry["critical_fraction"]
            x = np.linspace(phi.min(), phi.max(), 50)
            law = _law(x, (phi.min(), phi_c, phi.max()), (values["mu_loose"], values["mu_c"], values["mu_dense"]))
            law_ax.plot(x - phi_c, law, color=AMBER, lw=1.1)
            mu = math.sin(math.radians(entry["friction_deg"]))
            law_ax.plot([phi.min() - phi_c, phi.max() - phi_c], [mu, mu], color=BLUE, lw=0.9, ls=(0, (2, 2)))
            if len(with_law) > 1:
                law_ax.text(phi.max() - phi_c, law[-1], " " + title, fontsize=5.6, color=MUTED, va="center")
        law_ax.set_title("friction used" + ("" if len(with_law) > 1 else f" ({dict(MEDIA_ROD)[with_law[0]].lower()})"),
                         fontsize=7.0)
        law_ax.set_xlabel(r"$\phi - \phi_c$")
        law_ax.set_ylabel(r"MPM friction $\mu$")
        law_ax.axvline(0.0, color=MUTED, lw=0.6, ls=(0, (3, 2)))
        grid(law_ax)
    for ax, letter in zip(axes, "abcd"):
        label(ax, letter)
    entries = [("measured mean ± SD", "o", 4, INK, INK), ("constant friction from the tabulated angle", "o", 3, BLUE, BLUE)]
    if any(data.get(m, {}).get("best_law_sweep") for m, _ in MEDIA_ROD):
        entries += [("best packing law, rejected in training: training", "s", 4, "white", AMBER),
                    ("held out", "o", 3.5, "white", AMBER)]
    elif with_law:
        entries += [("calibrated packing law: training", "s", 4, AMBER, AMBER), ("held out", "o", 3.5, AMBER, AMBER)]
    marker_legend(figure, entries, ncol=len(entries))
    save(figure, "rod-packing", "Rod intrusion against packing.")


def figure_plate_sinkage():
    data = _load("plate_sinkage.json")
    sink = np.asarray(data["measured_curves_sinkage_mm"])
    figure, axes = plt.subplots(1, 3, figsize=(7.4, 2.45), layout="constrained", sharey=True)
    report = np.asarray([5, 10, 20, 40, 60, 80])
    colors = {"3": "#6b7a8f", "4": "#2f3d52", "5": INK}
    for ax, m in zip(axes, ("3", "4", "5")):
        for other in ("1", "2"):
            c = data["measured_curves"][other]
            ax.plot(sink, c["pressure_mean_kPa"], color=LINE, lw=1.0, zorder=1)
        c = data["measured_curves"][m]
        mean, std = np.asarray(c["pressure_mean_kPa"]), np.asarray(c["pressure_std_kPa"])
        ax.plot(sink, mean, color=INK, lw=1.2, zorder=3)
        lab = data.get("lab", {}).get(m)
        if lab:
            ax.plot(report, lab["predicted"], color=BLUE, marker="o", ms=3, lw=0.9, zorder=4)
        best = data.get("grid", {}).get(m, {}).get("best")
        if best:
            ax.plot(report, best[0]["predicted"], color=AMBER, marker="D", ms=2.8, lw=0.9, ls=(0, (3, 1.5)), zorder=5)
            ax.text(0.03, 0.97, f"best grid point: μ {best[0]['mu']:.2f}, c {best[0]['cohesion_kPa']:.0f} kPa",
                    transform=ax.transAxes, va="top", fontsize=6.0, color=AMBER)
        dry = c["dry_density"]
        ax.set_title(f"method {m}, dry {dry:.0f} kg/m³", fontsize=7.0)
        ax.set_xlabel("sinkage [mm]")
        grid(ax)
    axes[0].set_ylabel("plate pressure [kPa]")
    for ax, letter in zip(axes, "abc"):
        label(ax, letter)
    marker_legend(figure, [
        ("measured", "_", 8, INK, INK),
        ("methods 1 and 2 (density not measured)", "_", 8, LINE, LINE),
        ("lab parameters: triaxial friction, no cohesion", "o", 3.5, BLUE, BLUE),
        ("best point of a friction-cohesion grid, per method (diagnostic)", "D", 3.5, AMBER, AMBER),
    ], ncol=2)
    save(figure, "plate-sinkage", "Plate sinkage in moist Cullinan sand.")


PULLOUT_SERIES = (
    ("Dowel_10cmLong", "dowel\ndiameter", "o"),
    ("3dPrint_10cmLong", "print\ndiameter", "D"),
    ("cmLong_1-2diam", "dowel\nlength", "s"),
    ("Dowel_5cmLong_", "burial\ndepth", "^"),
)


def _pullout_series(name):
    for key, label_, marker in PULLOUT_SERIES:
        if key in name:
            return label_, marker
    return "other", "x"


def figure_pullout():
    data = _load("rod_pullout.json")
    lit = data.get("literature")
    if lit is None:
        return
    figure, (ax, bx) = plt.subplots(1, 2, figsize=(7.4, 2.7), layout="constrained", gridspec_kw={"width_ratios": [1, 1.25]})
    measured, std = np.asarray(lit["measured_N"]), np.asarray(lit["std_N"])
    predicted = np.asarray(lit["predicted_N"])
    calibrated = {}
    cal = data.get("calibration")
    if cal:
        for exp in cal["experiments"].values():
            if exp.get("calibrated") is None:
                continue
            for g, v in zip(exp["times"], np.asarray(exp["calibrated"])[:, 0]):
                calibrated[g] = v
    names = lit["geometries"]
    corner = {}
    if data.get("corner_check"):
        corner = dict(zip(data["corner_check"]["geometries"], data["corner_check"]["predicted_N"]))
    top = max(measured.max(), predicted.max(), max(calibrated.values(), default=0), max(corner.values(), default=0)) * 1.5
    ax.plot([0.1, top], [0.1, top], color=LINE, lw=0.8)
    for g, m, s, p in zip(names, measured, std, predicted):
        _, marker = _pullout_series(g)
        ax.errorbar(m, p, xerr=s, fmt=marker, ms=3.5, color=BLUE, elinewidth=0.5, capsize=1.2)
        if g in corner:
            ax.scatter(m, corner[g], marker=marker, s=12, color=MUTED, zorder=3)
        if g in calibrated:
            ax.scatter(m, calibrated[g], marker=marker, s=14, color=AMBER, zorder=4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.2, top)
    ax.set_ylim(0.2, top)
    ax.set_xlabel("measured peak force [N]")
    ax.set_ylabel("simulated peak force [N]")
    grid(ax)
    label(ax, "a")
    # (b) predicted / measured by series.
    for k, (key, name, marker) in enumerate(PULLOUT_SERIES):
        rows = [(g, m, p) for g, m, p in zip(names, measured, predicted) if key in g]
        if not rows:
            continue
        x = np.arange(len(rows)) + 5 * k
        bx.scatter(x, [p / m for _, m, p in rows], marker=marker, s=14, color=BLUE)
        corner_rows = [corner.get(g) for g, _, _ in rows]
        if any(v is not None for v in corner_rows):
            bx.scatter([xi for xi, v in zip(x, corner_rows) if v is not None],
                       [v / m for (_, m, _), v in zip(rows, corner_rows) if v is not None], marker=marker, s=12, color=MUTED)
        cal_rows = [calibrated.get(g) for g, _, _ in rows]
        if any(v is not None for v in cal_rows):
            bx.scatter([xi for xi, v in zip(x, cal_rows) if v is not None],
                       [v / m for (_, m, _), v in zip(rows, cal_rows) if v is not None], marker=marker, s=14, color=AMBER)
        bx.text(x.mean(), -0.06, name, ha="center", va="top", fontsize=5.6, color=MUTED, transform=bx.get_xaxis_transform())
    bx.axhline(1.0, color=MUTED, lw=0.6)
    bx.set_xticks([])
    bx.set_ylabel("simulated / measured")
    grid(bx)
    label(bx, "b")
    entries = [("literature sand and rod friction", "o", 3.5, BLUE, BLUE)]
    if corner:
        entries.append(("sand friction 0.95, rod friction 1.0", "o", 3.5, MUTED, MUTED))
    if calibrated:
        entries.append(("calibrated", "o", 3.5, AMBER, AMBER))
    marker_legend(figure, entries, ncol=len(entries))
    save(figure, "pullout", "Rod pull-out.")


PROBE_SHAPES = (("cylindrical", "flat"), ("elliptical", "elliptical"), ("parabolic", "parabolic"), ("conical", "conical"),
                ("root_like", "root-like"))


def figure_probe():
    data = _load("probe_penetration.json")
    gen = data.get("generic")
    if gen is None:
        return
    depths = gen["report_depths_mm"]
    keys = [k for k in ("11mm", "9mm", "7mm") if k in gen["probes"]]
    figure, axes = plt.subplots(1, len(keys), figsize=(7.4, 2.5), layout="constrained", sharey=False)
    axes = np.atleast_1d(axes)
    calibrated = {}
    cal = data.get("calibration")
    if cal:
        for exp in cal["experiments"].values():
            if exp.get("calibrated") is None:
                continue
            calibrated[exp.get("diameter", exp.get("times", [None])[0])] = exp
    colors = plt.cm.cividis(np.linspace(0.05, 0.85, len(PROBE_SHAPES)))
    for ax, key in zip(axes, keys):
        entry = gen["probes"][key]
        measured = np.asarray(entry["measured_N"], dtype=float)
        predicted = np.asarray([[np.nan if v is None else v for v in row] for row in entry["predicted_N"]])
        for i, shape in enumerate(entry["shapes"]):
            color = colors[[s for s, _ in PROBE_SHAPES].index(shape)]
            ax.plot(depths, measured[i], color=color, lw=1.2)
            ax.plot(depths, predicted[i], color=color, lw=0.8, ls=(0, (2, 2)))
        ax.set_title(f"{key[:-2]} mm probes", fontsize=7.0)
        ax.set_xlabel("depth [mm]")
        grid(ax)
    axes[0].set_ylabel("force [N]")
    for ax, letter in zip(axes, "abc"):
        label(ax, letter)
    handles = [Line2D([], [], color=c, lw=1.2, label=n) for (s, n), c in zip(PROBE_SHAPES, colors)]
    handles += [Line2D([], [], color=MUTED, lw=1.2, label="measured"),
                Line2D([], [], color=INK, lw=0.8, ls=(0, (2, 2)), label="simulated, generic soil")]
    figure.legend(handles=handles, loc="outside lower center", ncol=7, fontsize=6.0)
    save(figure, "probe", "Probe penetration in dry sandy loam.")

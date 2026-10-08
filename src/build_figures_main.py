"""Build the manuscript figures from the frozen numbers and the model frame.

The figures share one visual design: the palette, the panel-letter style, the grid
treatment and the annotation style. The three severity colours and the sequential teal
are fixed below.

Figure 2 is not built here. It is a hand-drawn draw.io diagram with icons and a DAG inset,
and it cannot be faithfully reproduced in matplotlib. It is drawn and exported in
draw.io.

What each figure shows:

  Figure 1  the severity composition and the rider-age profile; the figure file is
            used as it stands (see `figure1`)
  Figure 3  the corner plot and the speed-age surface; the cue axis reads
            "Narrative-derived cues"
  Figure 4  the discordance audit, led by the observed cross-tabulation; the
            demographic reporting shifters are not estimated (\\S6.3)
  Figure 5  average marginal effects on all three outcomes at 95%, with no claim
            about which signals are strongest

Run with:  python src/build_figures_main.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_tables import pfmt, rnd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "results" / "figures"

SEV3 = ("O", "BC", "KA")
SEV_NAME = {"O": "O (No injury)", "BC": "BC (Moderate)", "KA": "KA (Severe)"}
N_LEVELS = ("no cue", "O-consistent", "BC-consistent", "KA-consistent")

# The Okabe-Ito colourblind-safe set.
SEV = {"O": "#2F80B7", "BC": "#E69F00", "KA": "#D55E00"}
FILL = {"O": "#C5DBEA", "BC": "#F8E4B8", "KA": "#F3D2B8"}
INK, MUTED, GRID = "#1F1F1F", "#4A4A4A", "#E4E4E4"
TEAL = LinearSegmentedColormap.from_list("teal", ["#F7FBFA", "#DEEFED", "#9CC8C2",
                                                    "#66ABA2", "#4D9690"])

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 11,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "axes.labelsize": 12,
    "text.color": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 400,
    "savefig.dpi": 400,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
})


def panel_title(ax, letter: str, title: str, pad: float = 1.02, size: float = 14) -> None:
    """Panel heading: a bold letter, two spaces, then the title."""
    ax.set_title(f"{letter}  {title}", loc="left", fontsize=size, fontweight="bold",
                 color=INK, pad=12, y=pad)


def grid(ax, axis: str = "x") -> None:
    ax.grid(axis=axis, color=GRID, lw=0.9, zorder=0)
    ax.set_axisbelow(True)


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name)
    plt.close(fig)
    print(f"  wrote figures/{name}")


# ---------------------------------------------------------------- Figure 1

def figure1(frame: pd.DataFrame, base: pd.DataFrame) -> None:
    """Not regenerated. The figure file is used as it stands.

    Its two panels are the severity distribution and the rider-age profile, and every
    number in them agrees with the analysis frame: 77, 363 and 82 by severity class, and n = 56, 355 and 77
    with medians 24, 22 and 28 in the age panel. Re-plotting identical data would only
    introduce the chance of drift, so figures/figure_1.png is kept as it is;
    `_figure1_redraw` below holds the code that draws it.
    """
    print("  figures/figure_1.png kept as it is; data unchanged")
    return


def _figure1_redraw(frame: pd.DataFrame, base: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(13.0, 6.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.55], wspace=0.26)

    # Panel A: severity composition, horizontal bars with count and share.
    ax = fig.add_subplot(gs[0, 0])
    counts = [int((frame["sev3"] == s).sum()) for s in SEV3]
    total = sum(counts)
    y = np.arange(3)[::-1]
    for i, s in enumerate(SEV3):
        share = 100.0 * counts[i] / total
        ax.barh(y[i], share, color=SEV[s], height=0.62, zorder=3)
        inside = share > 35
        ax.text(share - 1.5 if inside else share + 1.2, y[i],
                f"{counts[i]} ({share:.1f}%)", va="center",
                ha="right" if inside else "left", fontsize=12, fontweight="bold",
                color="white" if inside else INK, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels([SEV_NAME[s].replace(" (", "\n(").replace(")", ")")
                        for s in SEV3], fontsize=11)
    ax.set_xlim(0, 80)
    ax.set_xticks(range(0, 80, 10))
    ax.set_xticklabels([f"{v}%" for v in range(0, 80, 10)])
    ax.set_xlabel("Share of crashes")
    grid(ax, "x")
    panel_title(ax, "A", "Severity composition")

    # Panel B: the age raincloud, on the analysis frame.
    ax = fig.add_subplot(gs[0, 1])
    ages = base.set_index("rider_row_id")["rider_age"] if "rider_row_id" in base else None
    merged = frame[["rider_row_id", "sev3"]].merge(
        base[["rider_row_id", "rider_age"]], on="rider_row_id", how="left")
    for i, s in enumerate(SEV3):
        a = merged.loc[merged["sev3"] == s, "rider_age"].dropna().to_numpy()
        base_y = 2 - i
        if len(a) > 1:
            xs = np.linspace(0, 100, 400)
            bw = 0.9 * a.std() * len(a) ** (-0.2) or 3.0
            dens = np.exp(-0.5 * ((xs[:, None] - a[None, :]) / bw) ** 2).sum(1)
            dens = dens / dens.max() * 0.40
            ax.fill_between(xs, base_y + 0.10, base_y + 0.10 + dens,
                            color=FILL[s], zorder=2)
            ax.plot(xs, base_y + 0.10 + dens, color=SEV[s], lw=2.0, zorder=3)
        rng = np.random.default_rng(11)
        ax.scatter(a, base_y - 0.18 + rng.normal(0, 0.045, len(a)), s=14,
                   color=SEV[s], alpha=0.30, linewidths=0, zorder=2)
        if len(a):
            q1, med, q3 = np.percentile(a, [25, 50, 75])
            ax.plot([q1, q3], [base_y - 0.30] * 2, color=INK, lw=3.0, zorder=4,
                    solid_capstyle="round")
            ax.plot([med], [base_y - 0.30], "o", ms=9, color=SEV[s],
                    markeredgecolor=INK, markeredgewidth=1.0, zorder=5)
            ax.plot([a.mean()], [base_y - 0.30], "o", ms=9, color="white",
                    markeredgecolor=INK, markeredgewidth=1.4, zorder=5)
            ax.text(99, base_y + 0.46,
                    f"n={len(a)}\nmedian {med:.0f}; IQR {q1:.0f}-{q3:.0f}",
                    fontsize=10, color=MUTED, va="top", ha="right")
    n_missing = int(merged["rider_age"].isna().sum())
    ax.axvline(25, color=MUTED, lw=1.0, ls=":", zorder=1)
    ax.text(25.8, 2.62, "age 25", fontsize=10, color=MUTED)
    ax.set_yticks([2, 1, 0])
    ax.set_yticklabels([SEV_NAME[s] for s in SEV3], fontsize=11)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, 2.8)
    ax.set_xlabel("Rider age (years)")
    ax.text(0.0, -0.13, f"filled dot = median; open dot = mean; bar = IQR. "
                        f"{n_missing} riders have no recorded age and are omitted here.",
            transform=ax.transAxes, fontsize=9.5, color=MUTED, ha="left", va="top")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    panel_title(ax, "B", "Age distributions by severity")

    # There is no Panel C. The severity composition within each covariate level is
    # reported in Table 1 for every level with exact counts.

    save(fig, "figure_1.png")


# ---------------------------------------------------------------- Figure 3

# The ten pre-crash cues. The five injury cues are a different measurement and are not
# counted here.
PRECRASH_CUES = ("wrong_way", "sidewalk_transition", "driveway_alley",
                 "failure_to_yield", "signal_violation", "swerve_loss_control",
                 "dooring", "distraction_impairment", "vehicle_turning_across",
                 "lane_positioning")

# The density ramp, pale to dark, from cream to orange.
DENSITY = LinearSegmentedColormap.from_list(
    "density", ["#F6F5EF", "#F0EBD6", "#EBDDA8", "#E4C86B", "#DDB13C",
                  "#D4941F", "#C77011", "#B85D09"])


def _kde2d(x, y, xs, ys):
    """Gaussian KDE on a grid, without a scipy dependency at import time."""
    from scipy.stats import gaussian_kde
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 8:
        return None
    k = gaussian_kde(np.vstack([x[ok], y[ok]]))
    gx, gy = np.meshgrid(xs, ys)
    return k(np.vstack([gx.ravel(), gy.ravel()])).reshape(gx.shape)


def figure3(frame: pd.DataFrame) -> None:
    base = pd.read_csv(DATA / "base_frame.csv")
    labels = pd.read_csv(DATA / "final_labels.csv")
    cues = [c for c in PRECRASH_CUES if c in labels.columns]
    labels = labels.assign(cue_count=labels[cues].fillna(0).astype(float).sum(axis=1))

    d = (frame[["Crash_ID", "rider_row_id", "sev3"]]
         .merge(base[["rider_row_id", "rider_age", "Crash_Speed_Limit"]],
                on="rider_row_id", how="left")
         .merge(labels[["Crash_ID", "cue_count"]], on="Crash_ID", how="left"))
    d["speed"] = d["Crash_Speed_Limit"].where(d["Crash_Speed_Limit"] > 0)

    VARS = [("rider_age", "Rider age\n(years)", (0, 100)),
            ("speed", "Posted speed\n(mph)", (0, 75)),
            ("cue_count", "Narrative-derived cues\n(count)", (-0.5, 5.5))]

    fig = plt.figure(figsize=(19.0, 8.9))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.06, 1.0], wspace=0.13)
    corner = outer[0, 0].subgridspec(3, 3, hspace=0.12, wspace=0.12)

    for i, (vi, li, limi) in enumerate(VARS):
        for j, (vj, lj, limj) in enumerate(VARS):
            if j > i:
                continue
            ax = fig.add_subplot(corner[i, j])
            if i == j:
                allv = d[vi].dropna()
                bins = (np.arange(-0.5, 6.5, 1) if vi == "cue_count"
                        else np.histogram_bin_edges(allv, bins=26, range=limi))
                ax.hist(allv, bins=bins, color="#C9C4BC", alpha=0.55, zorder=2)
                for s in SEV3:
                    v = d.loc[d["sev3"] == s, vi].dropna()
                    ax.hist(v, bins=bins, histtype="step", lw=2.0, color=SEV[s],
                            zorder=3)
                    if len(v):
                        ax.axvline(v.median(), color=SEV[s], lw=1.6, ls="--", zorder=4)
                q1, med, q3 = np.percentile(allv, [25, 50, 75])
                ax.set_title(f"{med:.1f} [{q1:.1f}-{q3:.1f}]", fontsize=12, color=INK,
                             pad=6)
                ax.set_yticks([])
                ax.set_xlim(*limi)
            else:
                for s in SEV3:
                    sub = d[d["sev3"] == s]
                    ax.scatter(sub[vj], sub[vi], s=10, color=SEV[s], alpha=0.16,
                               linewidths=0, zorder=2)
                xs = np.linspace(limj[0], limj[1], 90)
                ys = np.linspace(limi[0], limi[1], 90)
                for s in SEV3:
                    sub = d[d["sev3"] == s]
                    z = _kde2d(sub[vj].to_numpy(float), sub[vi].to_numpy(float), xs, ys)
                    if z is None:
                        continue
                    ax.contour(xs, ys, z, levels=4, colors=SEV[s], linewidths=1.3,
                               zorder=3)
                ax.set_xlim(*limj)
                ax.set_ylim(*limi)
            if i == 2:
                ax.set_xlabel(lj, fontsize=11)
            else:
                ax.set_xticklabels([])
            if j == 0 and i != 0:
                ax.set_ylabel(li, fontsize=11)
            elif j != 0:
                ax.set_yticklabels([])
            ax.tick_params(labelsize=9.5)

    handles = [plt.Line2D([], [], color=SEV[s], lw=3, label=SEV_NAME[s]) for s in SEV3]
    fig.legend(handles=handles, frameon=False, fontsize=12,
               loc="upper left", bbox_to_anchor=(0.325, 0.905))
    fig.text(0.055, 0.955, "A  Joint distributions of safety-relevant fields",
             fontsize=17, fontweight="bold", color=INK)

    # ---- Panel B: the speed-age support surface --------------------------
    ax = fig.add_subplot(outer[0, 1])
    xs = np.linspace(2, 75, 220)
    ys = np.linspace(2, 95, 220)
    z = _kde2d(d["speed"].to_numpy(float), d["rider_age"].to_numpy(float), xs, ys)
    z = 100.0 * z / z.max()
    ax.contourf(xs, ys, z, levels=np.linspace(0, 100, 11), cmap=DENSITY, zorder=1)
    cs = ax.contour(xs, ys, z, levels=[40, 70], colors=INK, linewidths=1.4, zorder=4)
    ax.contour(xs, ys, z, levels=[20], colors=INK, linewidths=1.2,
               linestyles="dashed", zorder=4)

    for s, marker in zip(SEV3, ("o", "o", "^")):
        sub = d[d["sev3"] == s]
        ax.scatter(sub["speed"], sub["rider_age"], s=26, marker=marker,
                   color=SEV[s], alpha=0.75, linewidths=0, zorder=5,
                   label=SEV_NAME[s])
    ax.axvline(30, color=MUTED, lw=1.0, ls=":", zorder=3)
    ax.axhline(25, color=MUTED, lw=1.0, ls=":", zorder=3)
    ax.annotate("Core speed-age\nsupport region", xy=(31.5, 34), xytext=(8, 82),
                fontsize=13, fontweight="bold", color=INK,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=1.2), zorder=6)
    leg = ax.legend(title="Crash severity", frameon=True, fontsize=11,
                    loc="upper right", framealpha=0.92, edgecolor="#CCCCCC")
    leg.get_title().set_fontsize(11)
    ax.set_xlabel("Posted speed limit (mph)")
    ax.set_ylabel("Rider age (years)")
    ax.set_xlim(2, 75)
    ax.set_ylim(2, 95)
    sm = plt.cm.ScalarMappable(cmap=DENSITY, norm=plt.Normalize(0, 100))
    cb = fig.colorbar(sm, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label("Relative record density (%)", fontsize=11, color=MUTED)
    cb.outline.set_visible(False)
    fig.text(0.565, 0.955, "B  Observed speed-age support surface",
             fontsize=17, fontweight="bold", color=INK)

    save(fig, "figure_3.png")


# ---------------------------------------------------------------- Figure 4

def _matrix_panel(ax, letter, title, matrix, rows, cols, note=None, title_size=14,
                  cell_size=12, tick_size=11):
    """One of the figure's right-hand matrices."""
    m = np.array(matrix)
    ax.imshow(m, cmap=TEAL, vmin=0, vmax=1, aspect="auto")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            strong = m[i, j] > 0.55
            ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=cell_size,
                    fontweight="bold" if strong else "normal",
                    color="white" if strong else INK)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, fontsize=tick_size - 0.5)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=tick_size)
    ax.set_xlabel("Narrative indicator", fontsize=tick_size)
    ax.set_ylabel("Latent severity class", fontsize=tick_size)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)
    panel_title(ax, letter, title, pad=1.03, size=title_size)


def figure4(numbers: dict) -> None:
    stage1 = json.loads((DATA / "audit_stage1.json")
                             .read_text(encoding="utf-8"))

    # The observed cross-tabulation is Table 3 of the manuscript, so the figure shows only
    # what the table cannot: P(coded severity | latent class) under each prior, with the
    # largest Pearson residual of each fit against the observed table.
    # Same width as Figure 3, so both print their panel titles at the same size.
    fig = plt.figure(figsize=(20.5, 8.0))
    gs = fig.add_gridspec(1, 2, wspace=0.22)
    for slot, key, letter, label in ((0, "moderate", "A", "Moderate prior"),
                                     (1, "weak", "B", "Weak prior")):
        block = stage1[key]
        km = block["class_alignment"]["kabco_measurement"]
        m = [[km[c][d] for d in SEV3] for c in SEV3]
        resid = block["fit_against_raw_table"]["max_abs_pearson_residual"]
        axm = fig.add_subplot(gs[0, slot])
        _matrix_panel(axm, letter,
                      f"{label}   (largest residual vs. Table 3: {resid:.2f})",
                      m, list(SEV3), list(SEV3), title_size=17, cell_size=16,
                      tick_size=13)
        axm.set_xlabel("Coded severity", fontsize=13)

    save(fig, "figure_4.png")


# ---------------------------------------------------------------- Figure 5

# The covariate grouping, with a fifth group for the reporting-completeness
# indicators. (heading, [(column, label)])
FOREST_GROUPS = (
    ("OPERATING CONTEXT", [
        ("speed_ge45", "Speed limit 45+ mph"),
        ("speed_35to40", "Speed limit 35 to 40 mph"),
        ("nighttime", "Nighttime crash"),
        ("light_dark_or_unknown", "Dark or not recorded"),
    ]),
    ("SPATIAL AND INFRASTRUCTURE CONTEXT", [
        ("road_not_city_street", "Not a city street"),
        ("ctrl_signal", "Traffic signal"),
        ("urban_area_200k", "Urban area of 200,000+"),
        ("bike_facility_nearest", "Cycle facility nearby"),
        ("bike_facility_100m", "Cycle facility within 100 m"),
        ("road_density_km_100m", "Road density within 100 m (per km)"),
        ("poverty_share", "Poverty share (0 to 1 scale)"),
    ]),
    ("RIDER CONTEXT", [
        ("gender_female", "Female rider"),
        ("eth_black", "Black rider"),
        ("eth_hispanic", "Hispanic rider"),
        ("eth_other_or_unknown", "Ethnicity other or not recorded"),
        ("age_under25", "Youth rider (<25)"),
        ("age_45plus", "Older rider (45+)"),
    ]),
    ("NARRATIVE-DERIVED PRE-CRASH CUES", [
        ("sidewalk_transition", "Sidewalk transition"),
        ("failure_to_yield", "Failure to yield"),
        ("signal_violation", "Signal violation"),
        ("distraction_impairment", "Distraction/impairment"),
        ("wrong_way", "Wrong-way riding"),
        ("swerve_loss_control", "Swerve/loss of control"),
        ("vehicle_turning_across", "Vehicle turning across"),
        ("lane_positioning", "Lane positioning"),
    ]),
    ("REPORTING COMPLETENESS", [
        ("age_unknown", "Rider age not recorded"),
        ("gender_unknown", "Rider gender not recorded"),
        ("speed_unknown", "Speed limit not recorded"),
        ("coord_missing", "Crash coordinates absent"),
    ]),
)

MARKER = {"O": "o", "BC": "s", "KA": "D"}


def _ladder_box(ax, x, w, title, colour, lines, y=0.0, h=0.92):
    """A Panel A box, returned with its texts so the layout can be checked."""
    box = plt.Rectangle((x, y), w, h, transform=ax.transAxes, facecolor="#FCFCFC",
                        edgecolor=colour, lw=2.0, joinstyle="round", zorder=2,
                        clip_on=False)
    ax.add_patch(box)
    texts = [ax.text(x + 0.018, y + h - 0.10, title, transform=ax.transAxes, fontsize=13,
                     fontweight="bold", color=colour, va="top", zorder=3)]
    for i, line in enumerate(lines):
        texts.append(ax.text(x + 0.018, y + h - 0.26 - 0.155 * i, line,
                             transform=ax.transAxes, fontsize=10.5,
                             color=INK if i == 0 else MUTED, va="top", zorder=3))
    return box, texts


def figure5(numbers: dict) -> None:
    frame = pd.read_csv(DATA / "model_frame.csv")
    ames = numbers["severity"]["ames_all"]
    rungs = numbers["severity"]["rungs"]
    shap = numbers["benchmark"].get("shap_top10", {})

    # A heading occupies its own slot, so it cannot land on a covariate's label.
    rows, headings = [], {}
    for heading, items in FOREST_GROUPS:
        present = [(c, l) for c, l in items if c in ames]
        if not present:
            continue
        headings[len(rows)] = heading
        rows.append((None, heading))
        rows.extend(present)
    rows = rows[::-1]
    headings = {len(rows) - 1 - k: v for k, v in headings.items()}

    # The specification ladder is Table 5 of the manuscript, so the figure shows only the
    # marginal effects, with the prevalence and SHAP rails.
    fig = plt.figure(figsize=(18.5, 12.5))
    gs = fig.add_gridspec(1, 1)

    # ---- Panel B: the grouped forest --------------------------------------
    gs2 = gs[0, 0].subgridspec(1, 3, width_ratios=[1.0, 0.145, 0.145], wspace=0.035)
    ax = fig.add_subplot(gs2[0, 0])

    ax.axvspan(-0.40, 0, color="#EDF3F8", zorder=0)
    ax.axvspan(0, 0.52, color="#FDF4EA", zorder=0)
    offsets = {"O": 0.26, "BC": 0.0, "KA": -0.26}
    for level in SEV3:
        for i, (col, _) in enumerate(rows):
            if col is None:
                continue
            e = ames[col][level]
            lo, hi = e["ci95"]
            yy = i + offsets[level]
            marked = e["excludes_zero"]
            ax.plot([lo, hi], [yy, yy], color=SEV[level], lw=2.4 if marked else 1.3,
                    alpha=1.0 if marked else 0.62, solid_capstyle="round", zorder=3)
            ax.plot([e["ame"]], [yy], MARKER[level], ms=8 if marked else 6,
                    color=SEV[level], markeredgecolor=INK if marked else "none",
                    markeredgewidth=0.9, zorder=4)
    ax.axvline(0, color=MUTED, lw=1.4, ls="--", zorder=2)

    ax.set_yticks([i for i, (c, _) in enumerate(rows) if c is not None])
    ax.set_yticklabels([l for c, l in rows if c is not None], fontsize=11.5)
    for i, (col, label) in enumerate(rows):
        if col is None:
            ax.text(-0.392, i, label, fontsize=11, fontweight="bold", color=MUTED,
                    va="center", zorder=5)
    ax.set_ylim(-1.0, len(rows) + 0.2)
    ax.set_xlim(-0.40, 0.52)
    ax.set_xlabel("Marginal probability shift, by outcome")
    grid(ax, "x")
    ax.text(0.11, 1.012, "Blue shade: lower probability", transform=ax.transAxes,
            fontsize=11, color=SEV["O"], fontweight="bold")
    ax.text(0.62, 1.012, "Warm shade: higher probability", transform=ax.transAxes,
            fontsize=11, color=SEV["KA"], fontweight="bold")
    handles = [plt.Line2D([], [], color=SEV[s], marker=MARKER[s], lw=2.4, ms=8,
                          label=f"P({s})") for s in SEV3]
    ax.legend(handles=handles, frameon=False, fontsize=12, ncol=3,
              loc="lower center", bbox_to_anchor=(0.5, 1.035))

    # ---- the two rails ----------------------------------------------------
    axe = fig.add_subplot(gs2[0, 1], sharey=ax)
    prev = []
    for col, _ in rows:
        if col is None:
            prev.append(np.nan)
        elif col in frame.columns and set(frame[col].dropna().unique()) <= {0, 1, 0.0, 1.0}:
            prev.append(100.0 * float(frame[col].astype(float).mean()))
        else:
            prev.append(np.nan)
    pmax = np.nanmax(prev) or 1.0
    for i, v in enumerate(prev):
        if np.isnan(v):
            if rows[i][0] is not None:
                axe.text(0.02, i, "continuous", fontsize=9.5, color=MUTED, va="center")
            continue
        axe.barh(i, v / pmax, color="#C9C4BC", height=0.5, zorder=3)
        axe.text(v / pmax + 0.04, i, f"{rnd(v, 0)}%", fontsize=10, color=MUTED,
                 va="center")
    axe.set_xlim(0, 1.5)
    axe.axis("off")
    axe.text(0.0, 1.012, "Covariate\nprevalence", transform=axe.transAxes, fontsize=12,
             fontweight="bold", color=INK, va="bottom")

    axs = fig.add_subplot(gs2[0, 2], sharey=ax)
    smax = max(shap.values()) if shap else 1.0
    for i, (col, _) in enumerate(rows):
        v = shap.get(col) if col else None
        if v is None:
            continue
        axs.barh(i, v / smax, color="#2E8B7A", height=0.5, zorder=3)
        axs.text(v / smax + 0.04, i, f"{rnd(v, 3)}", fontsize=10, color=MUTED,
                 va="center")
    axs.set_xlim(0, 1.6)
    axs.axis("off")
    axs.text(0.0, 1.012, "Predictive\nsalience (SHAP)", transform=axs.transAxes, fontsize=12,
             fontweight="bold", color=INK, va="bottom")

    # Every covariate among the ten with the largest SHAP values has a row.
    shown = [c for c, _ in rows if c is not None]
    missing = sorted(set(shap) - set(shown))
    if missing:
        raise SystemExit(f"Figure 5: SHAP covariates without a row: {missing}")
    record = {"generated": pd.Timestamp.now().isoformat(timespec="seconds"),
              "script": "src/build_figures_main.py figure5",
              "inputs": ["data/numbers.json", "data/model_frame.csv"],
              "covariates_shown": len(shown),
              "covariates_in_m1": len(rungs["M1"]["coefficients"]["slopes"]),
              "rows": shown}
    (DATA / "figure5_rows.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"  Figure 5 shows {len(shown)} of {record['covariates_in_m1']} covariates")
    save(fig, "figure_5.png")


def main() -> int:
    numbers = json.loads((DATA / "numbers.json").read_text(encoding="utf-8"))
    frame = pd.read_csv(DATA / "model_frame.csv")
    base = pd.read_csv(DATA / "base_frame.csv")
    if "rider_age" not in base.columns:
        raise SystemExit("base_frame.csv has no rider_age column")
    # One figure can be rebuilt alone, as in: python build_figures_main.py figure5
    only = set(sys.argv[1:])
    if only:
        for name in sorted(only):
            {"figure1": lambda: figure1(frame, base), "figure3": lambda: figure3(frame),
             "figure4": lambda: figure4(numbers), "figure5": lambda: figure5(numbers)}[name]()
        return 0
    figure1(frame, base)
    figure3(frame)
    figure4(numbers)
    figure5(numbers)
    print("\n  4 figure(s) processed.")
    print("  Figure 2 is a draw.io diagram and is handled separately.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

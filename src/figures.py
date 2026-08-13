"""Publication figures.

Design follows three rules. Series are distinguished by line style and marker as
well as by colour, so that the figures survive greyscale printing and are legible
to readers with colour vision deficiency. Panels that invite comparison share
axis limits. Every estimate that carries sampling uncertainty is drawn with its
confidence band rather than as a bare point.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from . import config as C
from . import regressions as rg

mpl.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": C.FIG_DPI,
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "legend.fontsize": 8,
})

STYLE = {
    "TARGET": dict(color="#1B3A6B", marker="o", ls="-"),
    "PATH": dict(color="#C1611F", marker="s", ls="--"),
    "QE": dict(color="#2E7D5B", marker="^", ls="-."),
}
REG_STYLE = {
    "QE": dict(color="#1B3A6B", marker="o", ls="-", label="QE regime (2015-2021)"),
    "NORMALISATION": dict(color="#C1611F", marker="s", ls="--",
                          label="Normalisation (2022-2025)"),
}


def _save(fig, name):
    path = C.OUT_FIG / f"{name}.pdf"
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(C.OUT_FIG / f"{name}.png", bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Figure 1: factor time series
# ---------------------------------------------------------------------------
def fig_factor_series(F: pd.DataFrame):
    fig, axes = plt.subplots(3, 1, figsize=(7.0, 6.0), sharex=True)
    split = pd.Timestamp(C.NORM_START)
    for ax, f in zip(axes, ["TARGET", "PATH", "QE"]):
        ax.axhline(0, color="black", lw=0.6)
        ax.axvspan(F["date"].min(), split, color="#1B3A6B", alpha=0.05)
        ax.axvline(split, color="#B03A2E", lw=1.0, ls=":")
        ax.bar(F["date"], F[f], width=28, color=STYLE[f]["color"], alpha=0.85)
        sd_qe = F.loc[F.regime == "QE", f].std()
        sd_no = F.loc[F.regime == "NORMALISATION", f].std()
        ax.set_ylabel(f"{f}\n(bp)")
        ax.text(0.015, 0.90, f"s.d.: {sd_qe:.2f} bp vs {sd_no:.2f} bp",
                transform=ax.transAxes, fontsize=7.5, va="top")
    y0 = axes[0].get_ylim()[1]
    axes[0].text(split, y0 * 0.98, " first rate hike", fontsize=7.5,
                 va="top", color="#B03A2E")
    axes[0].text(0.28, 1.06, "QE regime", transform=axes[0].transAxes,
                 fontsize=8.5, ha="center", color="#1B3A6B")
    axes[0].text(0.86, 1.06, "Normalisation", transform=axes[0].transAxes,
                 fontsize=8.5, ha="center", color="#B03A2E")
    axes[-1].set_xlabel("Governing Council meeting date")
    fig.align_ylabels(axes)
    return _save(fig, "fig1_factor_series")


# ---------------------------------------------------------------------------
# Figure 2: loading profiles
# ---------------------------------------------------------------------------
def fig_loadings(tgt, con):
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    x = [C.OIS_YEARS[m] for m in C.OIS_MATS]
    for name, L in [("TARGET", tgt.loadings), ("PATH", con.loadings),
                    ("QE", con.loadings)]:
        if name not in L.columns:
            continue
        ax.plot(x, L[name].values, label=name, lw=1.6, ms=4, **STYLE[name])
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xscale("log")
    ax.set_xticks([1 / 12, 0.5, 1, 2, 5, 10])
    ax.set_xticklabels(["1M", "6M", "1Y", "2Y", "5Y", "10Y"])
    ax.set_xlabel("OIS maturity")
    ax.set_ylabel("Loading (bp per unit of factor)")
    ax.legend(loc="upper left")
    return _save(fig, "fig2_loadings")


# ---------------------------------------------------------------------------
# Figure 3: curve response by regime
# ---------------------------------------------------------------------------
def fig_curve_response(D: pd.DataFrame):
    mats = C.DE_MATS
    x = [C.DE_YEARS[m] for m in mats]
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.2), sharey=True)
    for ax, f in zip(axes, ["TARGET", "PATH", "QE"]):
        for reg, st in REG_STYLE.items():
            b, lo, hi = [], [], []
            for m in mats:
                fit = rg.baseline(D[D.regime == reg], m)
                b.append(fit.params[f])
                ci = fit.conf_int(alpha=0.10).loc[f]
                lo.append(ci[0]); hi.append(ci[1])
            ax.plot(x, b, lw=1.6, ms=4, **st)
            ax.fill_between(x, lo, hi, color=st["color"], alpha=0.13, lw=0)
        ax.axhline(0, color="black", lw=0.6)
        ax.set_xscale("log")
        ax.set_xticks(x); ax.set_xticklabels([m.replace("DE", "") for m in mats])
        panel = {"TARGET": "(a)", "PATH": "(b)", "QE": "(c)"}[f]
        ax.set_title(f"{panel} {f} surprise", loc="left")
        ax.set_xlabel("Bund maturity")
    axes[0].set_ylabel("Response (bp per bp of surprise)")
    axes[0].legend(loc="lower left")
    return _save(fig, "fig3_curve_response")


# ---------------------------------------------------------------------------
# Figure 4: rolling coefficients
# ---------------------------------------------------------------------------
def fig_rolling(D_full: pd.DataFrame, deps=("DE10Y", "DE30Y")):
    fig, axes = plt.subplots(1, len(deps), figsize=(9.0, 3.3), sharex=True)
    axes = np.atleast_1d(axes)
    split = pd.Timestamp(C.NORM_START)
    for ax, dep in zip(axes, deps):
        roll = rg.rolling_betas(D_full, dep)
        f = "PATH"
        ax.plot(roll["date"], roll[f], color=STYLE[f]["color"], lw=1.5)
        ax.fill_between(roll["date"],
                        roll[f] - 1.645 * roll[f + "_se"],
                        roll[f] + 1.645 * roll[f + "_se"],
                        color=STYLE[f]["color"], alpha=0.16, lw=0)
        ax.axhline(0, color="black", lw=0.7)
        ax.axvline(split, color="#B03A2E", lw=1.0, ls=":")
        panel = "(a)" if dep == "DE10Y" else "(b)"
        ax.set_title(f"{panel} PATH response of {dep.replace('DE','')} Bund",
                     loc="left")
        ax.set_xlabel("End of rolling window")
    axes[0].set_ylabel(f"Coefficient ({C.ROLL_WINDOW}-meeting window)")
    fig.autofmt_xdate()
    return _save(fig, "fig4_rolling")


# ---------------------------------------------------------------------------
# Figure 5: information shocks
# ---------------------------------------------------------------------------
def fig_information(cls: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8), sharex=True, sharey=True)
    for ax, reg, ttl in zip(axes, ["QE", "NORMALISATION"],
                            ["QE regime (2015-2021)", "Normalisation (2022-2025)"]):
        d = cls[cls.regime == reg]
        pol, inf = d[d.shock_type == "POLICY"], d[d.shock_type == "INFORMATION"]
        # Shade the co-movement quadrants so the classification rule is legible
        # without reading the caption: NE and SW are information shocks.
        lim = 24, 4.6
        for sx, sy in [(1, 1), (-1, -1)]:
            ax.add_patch(mpl.patches.Rectangle(
                (0 if sx > 0 else -lim[0], 0 if sy > 0 else -lim[1]),
                lim[0], lim[1], facecolor="#C1611F", alpha=0.055, lw=0, zorder=0))
        ax.axhline(0, color="black", lw=0.7); ax.axvline(0, color="black", lw=0.7)
        ax.scatter(pol["rate_surprise"], pol["equity_response"], s=26,
                   facecolor="#1B3A6B", edgecolor="white", lw=0.5,
                   marker="o", label="Policy shock")
        ax.scatter(inf["rate_surprise"], inf["equity_response"], s=34,
                   facecolor="#C1611F", edgecolor="white", lw=0.5,
                   marker="D", label="Information shock")
        share = len(inf) / max(len(d), 1)
        ax.set_title(f"{ttl}\ninformation shocks: {share:.0%} of {len(d)} meetings")
        ax.set_xlabel("2Y OIS surprise (bp)")
    axes[0].set_ylabel("EURO STOXX 50 response (%)")
    axes[0].legend(loc="upper left")
    return _save(fig, "fig5_information")

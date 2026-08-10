#!/usr/bin/env python3
"""Reproduce every number and figure in the paper from the raw workbook.

Usage
-----
    python run_all.py

Requires ``data/raw/Dataset_EA-MPD.xlsx``, downloadable from
https://www.ecb.europa.eu/pub/pdf/annex/Dataset_EA-MPD.xlsx
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd

from src import config as C
from src import data_io as io
from src import factors as fa
from src import figures as fg
from src import outcomes as oc
from src import regressions as rg
from src import scenarios as sc
from src import shocks as sk

warnings.filterwarnings("ignore")
DEPS = ["DE2Y", "DE5Y", "DE10Y", "DE30Y", "LEVEL", "SLOPE", "CURV"]
RESULTS: dict = {}


PRETTY = {
    "TARGET": r"\textsc{target}", "PATH": r"\textsc{path}",
    "QE": r"\textsc{qe}", "TIMING": r"\textsc{timing}",
    "SLOPE": "Slope (10Y--2Y)", "CURV": "Curvature", "LEVEL": "Level",
    "SPREAD_IT10Y": "IT--DE 10Y", "SPREAD_ES10Y": "ES--DE 10Y",
    "SPREAD_FR10Y": "FR--DE 10Y", "R2": r"$R^2$",
    "NORMALISATION": "Normalisation", "share_information": "Info.\\ share",
    "periphery_extra": "Periphery extra", "periphery_se": "(s.e.)",
    "core_se": "(s.e.)", "shock_bp": "Shock size (bp)",
    "pnl_flattener_eur": "P\\&L flattener (EUR)",
    "d_slope_2s10s": "$\\Delta$ 2s10s (bp)",
}


def _label(x) -> str:
    """Convert a raw column or index label into LaTeX-safe display text."""
    s = str(x)
    if s in PRETTY:
        return PRETTY[s]
    s = s.replace("_x_QT", r" $\times$ QT").replace("_x_PER", r" $\times$ Periphery")
    s = s.replace("_bootp", " [boot $p$]")
    s = s.replace("dy_DE", r"$\Delta y$ ").replace("pnl_DE", "P\\&L ")
    s = s.replace("_eur", " (EUR)")
    for k, v in [("TARGET", r"\textsc{target}"), ("PATH", r"\textsc{path}"),
                 ("QE", r"\textsc{qe}"), ("TIMING", r"\textsc{timing}")]:
        s = s.replace(k, v)
    return s.replace("_", r"\_")


def _texify(obj):
    """Return a copy with display labels on every axis level."""
    o = obj.copy()
    if isinstance(o.index, pd.MultiIndex):
        o.index = pd.MultiIndex.from_tuples(
            [tuple(_label(v) for v in t) for t in o.index], names=o.index.names)
    else:
        o.index = [_label(v) for v in o.index]
    if isinstance(o.columns, pd.MultiIndex):
        o.columns = pd.MultiIndex.from_tuples(
            [tuple(_label(v) for v in t) for t in o.columns], names=o.columns.names)
    else:
        o.columns = [_label(v) for v in o.columns]
    return o


def w(name, obj, floatfmt="%.3f"):
    """Write a table to CSV (raw labels) and LaTeX (display labels)."""
    obj.to_csv(C.OUT_TAB / f"{name}.csv")
    with open(C.OUT_TAB / f"{name}.tex", "w") as fh:
        fh.write(_texify(obj).to_latex(escape=False,
                                       float_format=lambda x: floatfmt % x))
    return obj


def main():
    print("=" * 78)
    print("ECB policy surprises and the German yield curve")
    print("=" * 78)

    # -- 1. Data ------------------------------------------------------------
    audit = io.audit_dates()
    w("a1_date_audit", audit)
    print(f"\n[1] Date audit: {len(audit)} events mis-parsed by a naive reader")

    windows = io.load_all()
    sample = io.build_sample(windows)
    me = sample["ME"]
    n_qe = (me.regime == "QE").sum()
    n_no = (me.regime == "NORMALISATION").sum()
    print(f"    Sample: {len(me)} meetings, {me.date.min():%Y-%m-%d} to "
          f"{me.date.max():%Y-%m-%d} ({n_qe} QE, {n_no} normalisation)")
    RESULTS["n_total"], RESULTS["n_qe"], RESULTS["n_norm"] = len(me), int(n_qe), int(n_no)
    RESULTS["date_first"] = f"{me.date.min():%d %B %Y}"
    RESULTS["date_last"] = f"{me.date.max():%d %B %Y}"
    RESULTS["n_misparsed"] = len(audit)

    # -- 2. Factors ---------------------------------------------------------
    F, tgt, con = fa.build_factors(sample, k_pc=2)
    X_pc = sample["PC"][C.OIS_MATS].to_numpy(float)
    _, _, vs_pc = fa._pca(X_pc, 4)
    RESULTS["var_pr_pc1"] = float(tgt.var_share[0])
    RESULTS["var_pc_pc1"], RESULTS["var_pc_pc2"] = float(vs_pc[0]), float(vs_pc[1])
    print(f"\n[2] Variance shares: press release PC1 {tgt.var_share[0]:.1%}; "
          f"press conference PC1 {vs_pc[0]:.1%}, PC2 {vs_pc[1]:.1%}")

    loadings = pd.concat([tgt.loadings, con.loadings], axis=1)
    w("t1_loadings", loadings)
    w("a2_factor_corr", fa.orthogonality_check(F, rg.FACTORS))

    D = oc.build_outcomes(me).merge(F[["date"] + rg.FACTORS], on="date")
    desc = D.groupby("regime")[rg.FACTORS].agg(["std", "min", "max"]).T
    w("t2_descriptives", desc)
    for f in rg.FACTORS:
        RESULTS[f"sd_{f}_qe"] = float(D.loc[D.regime == "QE", f].std())
        RESULTS[f"sd_{f}_norm"] = float(D.loc[D.regime == "NORMALISATION", f].std())

    # -- 3. Baseline and regime interaction ---------------------------------
    print("\n[3] Baseline responses (pooled)")
    base = rg.collect_baseline(D, DEPS)
    w("t3_baseline", base)
    print(base.to_string())

    print("\n[4] Regime interaction")
    inter = rg.collect_interaction(D, DEPS)
    w("t4_interaction", inter)
    print(inter.to_string())

    for dep in DEPS:
        fit, d, cols = rg.regime_interaction(D, dep)
        for f in rg.FACTORS:
            k = f"{f}_x_QT"
            RESULTS[f"g_{f}_{dep}"] = float(fit.params[k])
            RESULTS[f"gse_{f}_{dep}"] = float(fit.bse[k])
            RESULTS[f"gp_{f}_{dep}"] = float(rg.wild_bootstrap_pvalue(d, dep, cols, k))
            RESULTS[f"b_{f}_{dep}"] = float(fit.params[f])

    for reg in ["QE", "NORMALISATION"]:
        sub = rg.collect_baseline(D[D.regime == reg], DEPS)
        w(f"a3_baseline_{reg.lower()}", sub)
        for dep in DEPS:
            fit = rg.baseline(D[D.regime == reg], dep)
            for f in rg.FACTORS:
                RESULTS[f"b_{f}_{dep}_{reg[:2].lower()}"] = float(fit.params[f])
                RESULTS[f"se_{f}_{dep}_{reg[:2].lower()}"] = float(fit.bse[f])

    # -- 3b. Joint stability tests ------------------------------------------
    print("\n[4b] Joint stability tests (Chow-type, bootstrap)")
    stab = pd.DataFrame({dep: rg.stability_test(D, dep) for dep in DEPS}).T
    stab = stab.astype({"df": int, "N": int}).round(3)
    w("t12_stability", stab)
    print(stab.to_string())
    for dep in DEPS:
        RESULTS[f"stab_W_{dep}"] = float(stab.loc[dep, "W"])
        RESULTS[f"stab_p_{dep}"] = float(stab.loc[dep, "p_boot"])

    # -- 4. Information shocks ---------------------------------------------
    cls = sk.classify(me)
    tab = sk.summarise(cls)
    w("t5_shock_types", tab)
    D = D.merge(cls[["date", "shock_type"]], on="date")
    share_qe = (cls[cls.regime == "QE"].shock_type == "INFORMATION").mean()
    share_no = (cls[cls.regime == "NORMALISATION"].shock_type == "INFORMATION").mean()
    RESULTS["info_share_qe"], RESULTS["info_share_norm"] = float(share_qe), float(share_no)
    print(f"\n[5] Information shocks: {share_qe:.1%} of QE-regime meetings, "
          f"{share_no:.1%} of normalisation meetings")

    pol = D[D.shock_type == "POLICY"]
    RESULTS["n_policy"] = int(len(pol))
    inter_pol = rg.collect_interaction(pol, DEPS)
    w("t6_interaction_policy_only", inter_pol)
    print(inter_pol.to_string())
    for dep in DEPS:
        fit, d, cols = rg.regime_interaction(pol, dep)
        for f in rg.FACTORS:
            k = f"{f}_x_QT"
            RESULTS[f"pol_g_{f}_{dep}"] = float(fit.params[k])
            RESULTS[f"pol_gp_{f}_{dep}"] = float(rg.wild_bootstrap_pvalue(d, dep, cols, k))

    # -- 5. Panel -----------------------------------------------------------
    panel = oc.build_panel(me, D)
    print("\n[6] Country panel, 10Y")
    rows = []
    for mat in C.PANEL_MATS:
        fit, _ = rg.panel_fragmentation(panel, mat)
        for f in rg.FACTORS:
            rows.append({"maturity": mat, "factor": f,
                         "core": round(fit.params[f], 3),
                         "core_se": round(fit.bse[f], 3),
                         "periphery_extra": round(fit.params[f + "_x_PER"], 3),
                         "periphery_se": round(fit.bse[f + "_x_PER"], 3),
                         "p": round(fit.pvalues[f + "_x_PER"], 4),
                         "N": int(fit.nobs)})
    panel_tab = pd.DataFrame(rows)
    w("t7_panel_fragmentation", panel_tab.set_index(["maturity", "factor"]))
    print(panel_tab.to_string(index=False))
    fit10, _ = rg.panel_fragmentation(panel, "10Y")
    for f in rg.FACTORS:
        RESULTS[f"pan_{f}_core"] = float(fit10.params[f])
        RESULTS[f"pan_{f}_per"] = float(fit10.params[f + "_x_PER"])
        RESULTS[f"pan_{f}_per_p"] = float(fit10.pvalues[f + "_x_PER"])
    RESULTS["n_panel"] = int(fit10.nobs)

    spreads = rg.collect_interaction(D, ["SPREAD_IT10Y", "SPREAD_ES10Y", "SPREAD_FR10Y"])
    w("t8_spreads", spreads)
    print("\n[7] Sovereign spreads")
    print(spreads.to_string())

    # -- 6. Robustness ------------------------------------------------------
    print("\n[8] Robustness")
    rob = {}

    F_alt, _, _ = fa.build_factors(sample, k_pc=3)
    D_alt = oc.build_outcomes(me).merge(
        F_alt[["date", "TARGET", "TIMING", "PATH", "QE"]], on="date")
    r = rg.collect_interaction(D_alt, ["DE10Y", "DE30Y", "SLOPE"],
                               factors=["TARGET", "TIMING", "PATH", "QE"])
    w("a4_rob_three_pc", r)
    rob["three_pc_path_10y"] = float(
        rg.regime_interaction(D_alt, "DE10Y",
                              ["TARGET", "TIMING", "PATH", "QE"])[0].params["PATH_x_QT"])

    smpl_incl = io.build_sample(windows, drop_transition=False)
    F_incl, _, _ = fa.build_factors(smpl_incl, k_pc=2)
    D_incl = oc.build_outcomes(smpl_incl["ME"]).merge(
        F_incl[["date"] + rg.FACTORS], on="date")
    D_incl["QT"] = (D_incl.date >= C.NORM_START).astype(float)
    r = rg.collect_interaction(D_incl, DEPS)
    w("a5_rob_with_transition", r)
    rob["with_transition_path_10y"] = float(
        rg.regime_interaction(D_incl, "DE10Y")[0].params["PATH_x_QT"])

    D_alt_break = D.copy()
    D_alt_break["QT"] = (D_alt_break.date >= C.ALT_BREAK).astype(float)
    r = rg.collect_interaction(D_alt_break, DEPS)
    w("a6_rob_alt_break", r)
    rob["alt_break_path_10y"] = float(
        rg.regime_interaction(D_alt_break, "DE10Y")[0].params["PATH_x_QT"])

    crisis = ["2020-03-12", "2020-03-18", "2022-03-10", "2022-06-15"]
    D_nc = D[~D.date.isin(pd.to_datetime(crisis))]
    r = rg.collect_interaction(D_nc, DEPS)
    w("a7_rob_no_crisis", r)
    rob["no_crisis_path_10y"] = float(
        rg.regime_interaction(D_nc, "DE10Y")[0].params["PATH_x_QT"])
    rob["n_no_crisis"] = int(len(D_nc))

    RESULTS["robustness"] = rob
    w("t9_robustness_summary", pd.DataFrame(
        {"PATH x QT on 10Y Bund": rob}).T)
    print(pd.DataFrame({"PATH x QT on 10Y Bund": rob}).T.to_string())

    # -- 7. Scenarios -------------------------------------------------------
    scen = sc.scenario_table(D)
    w("t10_scenarios", scen, floatfmt="%.0f")
    delta = sc.regime_delta(D)
    w("t11_regime_delta", delta)
    print("\n[9] Scenario calibration (10 bp surprise, EUR 100m notional)")
    print(scen[["regime", "factor", "dy_DE2Y", "dy_DE10Y",
                "d_slope_2s10s", "pnl_flattener_eur"]].to_string(index=False))

    # -- 8. Figures ---------------------------------------------------------
    print("\n[10] Figures")
    for p in [fg.fig_factor_series(F),
              fg.fig_loadings(tgt, con),
              fg.fig_curve_response(D),
              fg.fig_rolling(D_incl),
              fg.fig_information(cls)]:
        print("    ", p.name)

    D.to_csv(C.DATA_PROC / "analysis_sample.csv", index=False)
    with open(C.OUT_TAB / "results.json", "w") as fh:
        json.dump(RESULTS, fh, indent=1, default=float)
    print("\nDone. Tables in output/tables, figures in output/figures.")


if __name__ == "__main__":
    main()

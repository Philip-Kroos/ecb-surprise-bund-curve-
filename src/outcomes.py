"""Construction of the dependent variables.

All outcomes are intraday changes measured in basis points over the same event
window as the surprises, so that the exclusion restriction underlying the event
study applies to regressor and regressand alike.

Besides individual maturities we form the three standard empirical proxies for
the Nelson-Siegel factors following Diebold and Li (2006): the level as the
average across maturities, the slope as the ten-year minus two-year difference,
and the curvature as twice the five-year point less the two ends. Using these
proxies rather than estimating a parametric curve on each event day avoids
imposing a functional form on a three-point cross-section.
"""
from __future__ import annotations

import pandas as pd

from . import config as C


LEVEL_MATS = ["DE2Y", "DE5Y", "DE10Y"]


def build_outcomes(me: pd.DataFrame) -> pd.DataFrame:
    """Bund maturities, curve summary statistics and sovereign spreads."""
    out = me[["date", "regime", "QT"]].copy()

    for m in C.DE_MATS:
        out[m] = me[m]

    out["LEVEL"] = me[LEVEL_MATS].mean(axis=1)
    out["SLOPE"] = me["DE10Y"] - me["DE2Y"]
    out["CURV"] = 2 * me["DE5Y"] - me["DE2Y"] - me["DE10Y"]

    for c in ("IT", "ES", "FR"):
        out[f"SPREAD_{c}10Y"] = me[f"{c}10Y"] - me["DE10Y"]

    out["STOXX50"] = me["STOXX50"]
    out["SX7E"] = me["SX7E"]
    out["EURUSD"] = me["EURUSD"]
    return out


def build_panel(me: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    """Reshape to country-by-maturity long format for the fragmentation test."""
    rows = []
    for _, r in me.iterrows():
        for ctry in C.COUNTRIES:
            for mat in C.PANEL_MATS:
                col = f"{ctry}{mat}"
                if col not in me.columns:
                    continue
                rows.append({
                    "date": r["date"],
                    "country": ctry,
                    "maturity": mat,
                    "dy": r[col],
                })
    panel = pd.DataFrame(rows)
    panel["periphery"] = panel["country"].isin(C.PERIPHERY).astype(float)
    keep = ["date", "regime", "QT", "TARGET", "PATH", "QE"]
    return panel.merge(factors[keep], on="date", how="left").dropna(subset=["dy"])

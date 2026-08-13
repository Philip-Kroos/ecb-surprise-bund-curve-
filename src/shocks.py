"""Separation of policy shocks from central bank information shocks.

A restrictive surprise admits two readings. Either the Governing Council has
tightened relative to expectations, in which case the discount-rate channel
should push equity prices down, or it has revealed a more favourable assessment
of the outlook than the market held, in which case the cash-flow channel can
dominate and equities rise alongside yields. Jarocinski and Karadi (2020)
exploit exactly this to separate the two.

We implement the transparent sign-classification version rather than a full
sign-restricted vector autoregression. Within the monetary event window we
compare the sign of the interest rate surprise with the sign of the equity
response. Negative co-movement identifies a conventional policy shock, positive
co-movement an information shock. The advantage of this variant is that it
requires no auxiliary model and can be verified by inspection of a scatter plot;
the cost is that it classifies discretely rather than decomposing continuously,
so events near the axes are assigned on weak evidence. We therefore report a
version that discards events in a neighbourhood of the origin.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


RATE_PROXY = "OIS_2Y"
EQUITY = "STOXX50"


def classify(me: pd.DataFrame, rate_col: str = RATE_PROXY,
             equity_col: str = EQUITY, drop_quantile: float = 0.0) -> pd.DataFrame:
    """Label each event as a policy or an information shock.

    ``drop_quantile`` optionally removes events whose surprise magnitude falls
    in the lowest quantile of the joint distribution, where the sign of a
    near-zero move carries little information.
    """
    d = me[["date", "regime", "QT"]].copy()
    d["rate_surprise"] = me[rate_col]
    d["equity_response"] = me[equity_col]

    comovement = np.sign(d["rate_surprise"]) * np.sign(d["equity_response"])
    d["shock_type"] = np.where(comovement < 0, "POLICY",
                               np.where(comovement > 0, "INFORMATION", "UNDEFINED"))

    if drop_quantile > 0:
        mag = (d["rate_surprise"].abs().rank(pct=True)
               + d["equity_response"].abs().rank(pct=True)) / 2
        d.loc[mag < drop_quantile, "shock_type"] = "UNDEFINED"

    d["is_policy"] = (d["shock_type"] == "POLICY").astype(float)
    d["is_info"] = (d["shock_type"] == "INFORMATION").astype(float)
    return d


def summarise(cls: pd.DataFrame) -> pd.DataFrame:
    """Cross-tabulation of shock types by regime, reported in the results."""
    tab = pd.crosstab(cls["regime"], cls["shock_type"])
    tab["share_information"] = (tab.get("INFORMATION", 0) /
                                tab.sum(axis=1)).round(3)
    return tab

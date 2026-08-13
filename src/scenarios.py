"""Translation of the estimated responses into portfolio arithmetic.

We deliberately do not back-test a trading rule. With eighty-three events over
eleven years, and twenty-seven in the regime of interest, any Sharpe ratio
computed from event-window returns would be dominated by sampling noise, and
reporting one would convey precision the design cannot support.

What the estimates do support is a conditional sensitivity statement: given a
surprise of a stated size, how far does the curve move, and what does that imply
mechanically for a position of stated size. The numbers below are in-sample
calibrations of the response function, not forecasts of returns, and they
inherit the confidence intervals of the underlying coefficients.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from . import regressions as rg

SHOCK_BP = 10.0


def response_matrix(D: pd.DataFrame, regime: str,
                    mats=None, factors=None) -> pd.DataFrame:
    """Point estimates and standard errors of the response of each maturity."""
    mats = mats or C.DE_MATS
    factors = factors or rg.FACTORS
    d = D[D.regime == regime]
    b = pd.DataFrame(index=mats, columns=factors, dtype=float)
    se = b.copy()
    for m in mats:
        fit = rg.baseline(d, m)
        for f in factors:
            b.loc[m, f], se.loc[m, f] = fit.params[f], fit.bse[f]
    return b, se


def scenario_table(D: pd.DataFrame, notional_eur: float = 100e6,
                   shock_bp: float | None = None) -> pd.DataFrame:
    """Yield and P&L impact of a one-standard-deviation surprise, by regime.

    The scenario size is the within-regime standard deviation of the factor
    rather than a round number. A fixed ten basis point shock would correspond
    to roughly nine standard deviations of the target factor during the QE
    period, so the implied yield moves would be extrapolations far outside the
    support of the data. Sizing each scenario in units the regime actually
    produced keeps the calibration inside the sample.

    P&L uses the first-order approximation ``dP/P = -D_mod * dy``. Convexity is
    ignored, which is immaterial at these magnitudes and becomes material for
    the thirty-year point only beyond roughly fifty basis points.
    """
    rows = []
    for regime in ["QE", "NORMALISATION"]:
        b, se = response_matrix(D, regime)
        d = D[D.regime == regime]
        for f in rg.FACTORS:
            size = float(d[f].std()) if shock_bp is None else shock_bp
            dy = {m: b.loc[m, f] * size for m in C.DE_MATS}
            slope_fit = rg.baseline(d, "SLOPE")
            rec = {
                "regime": regime,
                "factor": f,
                "shock_bp": round(size, 2),
                **{f"dy_{m}": round(dy[m], 2) for m in C.DE_MATS},
                "d_slope_2s10s": round(slope_fit.params[f] * size, 2),
            }
            for m in C.DE_MATS:
                pnl = -C.MOD_DURATION[m] * (dy[m] / 1e4) * notional_eur
                rec[f"pnl_{m}_eur"] = round(pnl, 0)
            # Duration-neutral 2s10s flattener: long 10Y, short 2Y, matched DV01.
            dv01_10 = C.MOD_DURATION["DE10Y"] / 1e4 * notional_eur
            leg10 = -dv01_10 * dy["DE10Y"]
            leg2 = +dv01_10 * dy["DE2Y"]
            rec["pnl_flattener_eur"] = round(leg10 + leg2, 0)
            rows.append(rec)
    return pd.DataFrame(rows)


def regime_delta(D: pd.DataFrame, shock_bp: float = SHOCK_BP) -> pd.DataFrame:
    """How far the calibrated response moved between the two regimes."""
    b_qe, se_qe = response_matrix(D, "QE")
    b_no, se_no = response_matrix(D, "NORMALISATION")
    rows = []
    for m in C.DE_MATS:
        for f in rg.FACTORS:
            diff = (b_no.loc[m, f] - b_qe.loc[m, f]) * shock_bp
            sed = np.sqrt(se_qe.loc[m, f] ** 2 + se_no.loc[m, f] ** 2) * shock_bp
            rows.append({"maturity": m, "factor": f,
                         "qe_bp": round(b_qe.loc[m, f] * shock_bp, 2),
                         "norm_bp": round(b_no.loc[m, f] * shock_bp, 2),
                         "change_bp": round(diff, 2),
                         "se_change": round(sed, 2)})
    return pd.DataFrame(rows)

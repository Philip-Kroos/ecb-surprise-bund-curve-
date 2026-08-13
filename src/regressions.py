"""Estimation. Three inference layers, roughly in order of how much I trust them.

HC1 for the pooled baseline. Restricted wild bootstrap for anything involving
the regime interaction - with 27 post-liftoff meetings the asymptotics
over-reject badly, and this is the number a referee will go straight to.
Date-clustered SEs in the panel, since the same surprise hits every country on
a given day.

Where HC1 and the bootstrap disagree, report the bootstrap and say so.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from . import config as C

FACTORS = ["TARGET", "PATH", "QE"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _design(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return sm.add_constant(df[cols], has_constant="add")


def _fmt(coef, se, pval) -> str:
    stars = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.10 else ""
    return f"{coef:.3f}{stars} ({se:.3f})"


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------
def baseline(df: pd.DataFrame, dep: str, factors=None):
    """Pooled response regression with HC1 standard errors."""
    factors = factors or FACTORS
    d = df.dropna(subset=[dep] + factors)
    X, y = _design(d, factors), d[dep]
    return sm.OLS(y, X).fit(cov_type=C.COV_TYPE)


def regime_interaction(df: pd.DataFrame, dep: str, factors=None):
    """Response regression with a full set of regime interactions."""
    factors = factors or FACTORS
    d = df.dropna(subset=[dep] + factors + ["QT"]).copy()
    cols = list(factors)
    for f in factors:
        d[f"{f}_x_QT"] = d[f] * d["QT"]
        cols.append(f"{f}_x_QT")
    cols.append("QT")
    X, y = _design(d, cols), d[dep]
    return sm.OLS(y, X).fit(cov_type=C.COV_TYPE), d, cols


def subsample(df: pd.DataFrame, dep: str, regime: str, factors=None):
    """Separate regression within one regime, reported alongside interactions."""
    d = df[df["regime"] == regime]
    return baseline(d, dep, factors)


# ---------------------------------------------------------------------------
# Restricted wild bootstrap
# ---------------------------------------------------------------------------
def wild_bootstrap_pvalue(d: pd.DataFrame, dep: str, cols: list[str],
                          target: str, n_boot: int = None, seed: int = None) -> float:
    """Wild bootstrap p value for H0: coef on `target` = 0.

    Null-imposed, Rademacher weights. Drawing residuals from the restricted
    fit rather than the unrestricted one is the variant with decent size in
    small samples - see Cameron, Gelbach and Miller (2008).

    Written out in numpy rather than looping statsmodels: 21 coefficients x
    5000 replications was minutes, this is under a second. The HC1 variance of
    a single coefficient collapses to sum_i u_i^2 w_i^2 with w = X (X'X)^-1 c,
    so the whole thing vectorises over replications.
    """
    n_boot = n_boot or C.N_BOOT
    rng = np.random.default_rng(seed if seed is not None else C.BOOT_SEED)

    X_full = _design(d, cols)
    target_pos = list(X_full.columns).index(target)
    X = X_full.to_numpy(float)
    y = d[dep].to_numpy(float)
    n, k = X.shape

    A = np.linalg.inv(X.T @ X)
    P = A @ X.T                       # k x n, maps y to coefficients
    c = np.zeros(k); c[target_pos] = 1.0
    w = X @ (A @ c)                   # n, weights entering the HC1 sandwich
    dof = n / (n - k)

    def t_stat(Y):                    # Y is n x B
        B = P @ Y                     # k x B
        U = Y - X @ B                 # n x B
        var = dof * ((U ** 2) * (w ** 2)[:, None]).sum(axis=0)
        return B[target_pos] / np.sqrt(var)

    t_obs = float(t_stat(y[:, None])[0])

    # Model estimated under the null, i.e. with the target regressor excluded.
    keep = [i for i in range(k) if i != target_pos]
    Xr = X[:, keep]
    beta_r = np.linalg.lstsq(Xr, y, rcond=None)[0]
    yhat_r = Xr @ beta_r
    resid_r = y - yhat_r

    V = rng.choice(np.array([-1.0, 1.0]), size=(n, n_boot))
    Y_star = yhat_r[:, None] + resid_r[:, None] * V
    t_star = t_stat(Y_star)

    return float((np.abs(t_star) >= abs(t_obs)).sum() + 1) / (n_boot + 1)


# ---------------------------------------------------------------------------
# Joint stability (Chow-type) test with bootstrap
# ---------------------------------------------------------------------------
def stability_test(df: pd.DataFrame, dep: str, factors=None,
                   n_boot: int = 999, seed: int = None) -> dict:
    """Joint test that all three regime interactions are zero, per equation.

    Added after realising the paper was leaning on 21 separate t-tests. This
    is the omnibus version and it does NOT reject anywhere - which is in the
    paper, in the main text, because it should be.

    Wald with HC1 covariance, bootstrap distribution under the stable-slope
    null (regime intercept kept).
    """
    factors = factors or FACTORS
    rng = np.random.default_rng(seed if seed is not None else C.BOOT_SEED)
    fit, d, cols = regime_interaction(df, dep, factors)
    inter = [f"{f}_x_QT" for f in factors]

    def wald(f):
        return float(f.wald_test(" = 0, ".join(inter) + " = 0",
                                 scalar=True).statistic)

    W_obs = wald(fit)

    # Null model: common slopes, regime intercept retained.
    Xr = _design(d, factors + ["QT"])
    y = d[dep].to_numpy(float)
    fit_r = sm.OLS(y, Xr).fit()
    yhat, resid = fit_r.fittedvalues.to_numpy(), fit_r.resid.to_numpy()
    Xf = _design(d, cols)

    count = 0
    for _ in range(n_boot):
        v = rng.choice([-1.0, 1.0], size=len(y))
        f_b = sm.OLS(yhat + resid * v, Xf).fit(cov_type=C.COV_TYPE)
        if wald(f_b) >= W_obs:
            count += 1
    return {"W": W_obs, "df": len(inter),
            "p_boot": (count + 1) / (n_boot + 1), "N": int(fit.nobs)}


# ---------------------------------------------------------------------------
# Rolling estimation
# ---------------------------------------------------------------------------
def rolling_betas(df: pd.DataFrame, dep: str, factors=None,
                  window: int = None) -> pd.DataFrame:
    """Rolling-window coefficients, the continuous counterpart to the regime split.

    Estimated on the sample including the transition period so that the path of
    the coefficients is not interrupted by an artificial gap.
    """
    factors = factors or FACTORS
    window = window or C.ROLL_WINDOW
    d = df.dropna(subset=[dep] + factors).sort_values("date").reset_index(drop=True)
    rows = []
    for i in range(window, len(d) + 1):
        w = d.iloc[i - window:i]
        fit = sm.OLS(w[dep], _design(w, factors)).fit(cov_type=C.COV_TYPE)
        rec = {"date": w["date"].iloc[-1], "n": window}
        for f in factors:
            rec[f] = fit.params[f]
            rec[f + "_se"] = fit.bse[f]
        rows.append(rec)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------
def panel_fragmentation(panel: pd.DataFrame, maturity: str = "10Y",
                        interact_regime: bool = False):
    """Country panel with unit fixed effects and date-clustered standard errors.

    The periphery interaction tests whether balance-sheet surprises transmit
    differentially to Italian and Spanish yields, which is the empirical content
    of the fragmentation hypothesis.
    """
    d = panel[panel["maturity"] == maturity].copy()
    cols = []
    for f in FACTORS:
        d[f"{f}_x_PER"] = d[f] * d["periphery"]
        cols += [f, f"{f}_x_PER"]
    if interact_regime:
        for f in FACTORS:
            d[f"{f}_x_PER_x_QT"] = d[f] * d["periphery"] * d["QT"]
            d[f"{f}_x_QT"] = d[f] * d["QT"]
            cols += [f"{f}_x_QT", f"{f}_x_PER_x_QT"]
        cols.append("QT")
    dummies = pd.get_dummies(d["country"], prefix="c", drop_first=True, dtype=float)
    X = pd.concat([sm.add_constant(d[cols], has_constant="add"), dummies], axis=1)
    fit = sm.OLS(d["dy"], X).fit(cov_type="cluster",
                                 cov_kwds={"groups": d["date"].astype("category").cat.codes})
    return fit, d


# ---------------------------------------------------------------------------
# Collection helpers
# ---------------------------------------------------------------------------
def collect_baseline(df: pd.DataFrame, deps: list[str], factors=None) -> pd.DataFrame:
    """Formatted coefficient table across dependent variables."""
    factors = factors or FACTORS
    out = {}
    for dep in deps:
        fit = baseline(df, dep, factors)
        col = {f: _fmt(fit.params[f], fit.bse[f], fit.pvalues[f]) for f in factors}
        col["R2"] = f"{fit.rsquared:.3f}"
        col["N"] = int(fit.nobs)
        out[dep] = col
    return pd.DataFrame(out)


def collect_interaction(df: pd.DataFrame, deps: list[str], factors=None,
                        bootstrap: bool = True) -> pd.DataFrame:
    """Formatted regime-interaction table with bootstrap p values."""
    factors = factors or FACTORS
    out = {}
    for dep in deps:
        fit, d, cols = regime_interaction(df, dep, factors)
        col = {}
        for f in factors:
            col[f] = _fmt(fit.params[f], fit.bse[f], fit.pvalues[f])
        for f in factors:
            key = f"{f}_x_QT"
            col[key] = _fmt(fit.params[key], fit.bse[key], fit.pvalues[key])
            if bootstrap:
                col[key + "_bootp"] = f"[{wild_bootstrap_pvalue(d, dep, cols, key):.3f}]"
        col["R2"] = f"{fit.rsquared:.3f}"
        col["N"] = int(fit.nobs)
        out[dep] = col
    return pd.DataFrame(out)

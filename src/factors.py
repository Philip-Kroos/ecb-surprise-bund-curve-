"""Extraction and rotation of ECB policy surprise factors.

The identification proceeds in two blocks, exploiting the fact that the ECB
separates its rate decision (press release) from its communication about the
outlook and the balance sheet (press conference).

Block 1, press release window. A single principal component accounts for
nearly all of the covariation of OIS surprises. We label it TARGET and scale it
so that a one-unit realisation moves the one-month OIS by one basis point.

Block 2, press conference window. We extract ``k`` principal components and
rotate them by an orthonormal matrix chosen to satisfy exclusion restrictions.
In the baseline ``k = 2`` and the single restriction is that the balance-sheet
factor leaves the one-month OIS unchanged, which just identifies the rotation
angle. The resulting factors are labelled PATH and QE and are anchored on the
two-year and ten-year OIS respectively.

Because the rotation is orthogonal, the rotated factors span exactly the same
space as the unrotated components. Restrictions therefore relabel variation,
they do not create or destroy it. This is the sense in which the decomposition
is a measurement device rather than a structural model.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.optimize import minimize

from . import config as C


@dataclass
class FactorModel:
    """Container for one estimated and rotated factor block."""
    names: list[str]
    scores: pd.DataFrame          # n x k, in anchor basis points
    loadings: pd.DataFrame        # K x k, basis points per unit of factor
    var_share: np.ndarray         # share of total variance, unrotated PCs
    anchors: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Principal components
# ---------------------------------------------------------------------------
def _pca(X: np.ndarray, k: int):
    """Covariance-based PCA in the normalisation X = F @ Lambda'.

    Scores carry unit variance and orthonormal columns; the singular values are
    absorbed into the loadings. This normalisation is what makes the subsequent
    orthogonal rotation meaningful: rotating unit-variance orthonormal scores
    preserves their mutual orthogonality, whereas rotating variance-scaled
    scores does not, because R' diag(s^2) R is not diagonal unless the singular
    values coincide. Getting this wrong yields "orthogonal" factors that are in
    fact correlated.
    """
    n = X.shape[0]
    Xc = X - X.mean(axis=0, keepdims=True)
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    var_share = (s ** 2) / (s ** 2).sum()
    scores = U[:, :k] * np.sqrt(n)                    # n x k, unit variance
    loadings = (Vt[:k, :].T * s[:k]) / np.sqrt(n)     # K x k
    return scores, loadings, var_share[:k]


# ---------------------------------------------------------------------------
# Orthogonal rotation subject to zero restrictions
# ---------------------------------------------------------------------------
def _skew(params: np.ndarray, k: int) -> np.ndarray:
    A = np.zeros((k, k))
    idx = np.triu_indices(k, 1)
    A[idx] = params
    return A - A.T


def _rotate_to_restrictions(loadings: np.ndarray, k: int,
                            restriction_rows: list[int]) -> np.ndarray:
    """Find orthonormal R minimising squared restricted loadings of Lambda @ R.

    ``restriction_rows`` holds, for each factor position, the list of row
    indices whose rotated loading should be zero. The orthogonal group is
    parameterised as the matrix exponential of a skew-symmetric matrix, which
    guarantees orthonormality at every step of the optimisation and avoids
    constrained optimisation entirely.
    """
    n_par = k * (k - 1) // 2

    def objective(p):
        R = expm(_skew(p, k))
        L = loadings @ R
        pen = 0.0
        for j, rows in enumerate(restriction_rows):
            for r in rows:
                pen += L[r, j] ** 2
        return pen

    best, best_val = None, np.inf
    rng = np.random.default_rng(C.BOOT_SEED)
    for trial in range(60):
        p0 = np.zeros(n_par) if trial == 0 else rng.normal(0, 1.2, n_par)
        res = minimize(objective, p0, method="BFGS",
                       options={"maxiter": 3000, "gtol": 1e-12})
        if res.fun < best_val:
            best_val, best = res.fun, res.x
    if best_val > 1e-8:
        raise RuntimeError(
            f"rotation did not achieve the exclusion restrictions "
            f"(residual penalty {best_val:.3e}); the system is over-identified"
        )
    return expm(_skew(best, k))


# ---------------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------------
def _anchor_scale(scores, loadings, mats, names, anchors):
    """Rescale each factor so a unit move shifts its anchor by one basis point."""
    scores, loadings = scores.copy(), loadings.copy()
    for j, name in enumerate(names):
        row = mats.index(anchors[name])
        s = loadings[row, j]
        if abs(s) < 1e-10:
            raise RuntimeError(f"factor {name} has no loading on its anchor "
                               f"{anchors[name]}; choose a different anchor")
        scores[:, j] *= s
        loadings[:, j] /= s
    return scores, loadings


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def extract_target(pr: pd.DataFrame, mats=None) -> FactorModel:
    """TARGET factor from the press release window."""
    mats = mats or C.OIS_MATS
    X = pr[mats].to_numpy(float)
    scores, loadings, vs = _pca(X, 1)
    if loadings[mats.index(C.ANCHOR_TARGET), 0] < 0:      # sign normalisation
        scores, loadings = -scores, -loadings
    scores, loadings = _anchor_scale(scores, loadings, mats, ["TARGET"],
                                     {"TARGET": C.ANCHOR_TARGET})
    return FactorModel(
        names=["TARGET"],
        scores=pd.DataFrame(scores, columns=["TARGET"], index=pr.index),
        loadings=pd.DataFrame(loadings, index=mats, columns=["TARGET"]),
        var_share=vs,
        anchors={"TARGET": C.ANCHOR_TARGET},
    )


def extract_conference(pc: pd.DataFrame, k: int = 2, mats=None) -> FactorModel:
    """PATH and QE (and optionally TIMING) from the press conference window."""
    mats = mats or C.OIS_MATS
    X = pc[mats].to_numpy(float)
    scores, loadings, vs = _pca(X, k)

    if k == 2:
        names = ["PATH", "QE"]
        anchors = {"PATH": C.ANCHOR_PATH, "QE": C.ANCHOR_QE}
        restriction_rows = [[], [mats.index("OIS_1M")]]
    elif k == 3:
        names = ["TIMING", "PATH", "QE"]
        anchors = {"TIMING": "OIS_6M", "PATH": C.ANCHOR_PATH, "QE": C.ANCHOR_QE}
        restriction_rows = [[],
                            [mats.index("OIS_1M")],
                            [mats.index("OIS_1M"), mats.index("OIS_1Y")]]
    else:
        raise ValueError("k must be 2 or 3")

    R = _rotate_to_restrictions(loadings, k, restriction_rows)
    scores, loadings = scores @ R, loadings @ R

    for j, name in enumerate(names):                      # sign normalisation
        if loadings[mats.index(anchors[name]), j] < 0:
            scores[:, j] *= -1
            loadings[:, j] *= -1

    scores, loadings = _anchor_scale(scores, loadings, mats, names, anchors)
    return FactorModel(
        names=names,
        scores=pd.DataFrame(scores, columns=names, index=pc.index),
        loadings=pd.DataFrame(loadings, index=mats, columns=names),
        var_share=vs,
        anchors=anchors,
    )


def build_factors(sample: dict[str, pd.DataFrame], k_pc: int = 2):
    """Assemble the full factor panel used throughout the paper."""
    tgt = extract_target(sample["PR"])
    con = extract_conference(sample["PC"], k=k_pc)
    F = pd.concat([sample["ME"][["date", "regime", "QT"]],
                   tgt.scores, con.scores], axis=1)
    return F, tgt, con


def orthogonality_check(F: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """Correlation matrix of the estimated factors, reported in the appendix."""
    return F[names].corr().round(3)

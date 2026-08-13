"""Loading the EA-MPD workbook.

READ THIS BEFORE TOUCHING THE DATE COLUMN. The workbook mixes real Excel
dates (early rows) with dd/mm/yyyy *text* (later rows). pandas applies
month-first to the text and quietly swaps day and month whenever both are
under 13. Six dates in the current vintage, all of them 2024-25, i.e. all
inside the regime the paper is about. 11/09/2025 came out as 9 November, a
Sunday. Every meeting in the sample is a Wednesday or Thursday, which is the
check that caught it.

So: read raw cells, branch on type, and assert the weekday distribution.
audit_dates() prints the diff against the naive parse - it goes in the
appendix.
"""
from __future__ import annotations

import datetime as dt
import warnings

import numpy as np
import openpyxl
import pandas as pd

from . import config as C


# ---------------------------------------------------------------------------
# Date handling
# ---------------------------------------------------------------------------
def _parse_date_cell(value) -> pd.Timestamp | pd.NaT:
    """Parse one raw date cell, respecting day-first text conventions."""
    if value is None:
        return pd.NaT
    if isinstance(value, (dt.datetime, dt.date)):
        return pd.Timestamp(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        # Excel serial date, 1900 date system.
        return pd.Timestamp("1899-12-30") + pd.Timedelta(days=float(value))
    text = str(value).strip()
    if not text:
        return pd.NaT
    for fmt in ("%d/%m/%Y", "%d.%m.%Y", "%d-%m-%Y", "%Y-%m-%d"):
        try:
            return pd.Timestamp(dt.datetime.strptime(text, fmt))
        except ValueError:
            continue
    return pd.to_datetime(text, dayfirst=True, errors="coerce")


def _read_dates(path, sheet: str) -> pd.Series:
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb[sheet]
    raw = [row[0].value for row in ws.iter_rows(min_col=1, max_col=1)]
    wb.close()
    header, body = raw[0], raw[1:]
    if str(header).strip().lower() != "date":
        raise ValueError(f"unexpected first column header in {sheet!r}: {header!r}")
    # openpyxl reports trailing formatted-but-empty rows that pandas discards.
    while body and body[-1] is None:
        body.pop()
    return pd.Series([_parse_date_cell(v) for v in body], name="date")


def audit_dates(path=None, sheet: str = C.SHEET_ME) -> pd.DataFrame:
    """Compare the robust parse with a naive month-first parse.

    Returned frame lists every event where the two disagree. It is written to
    the appendix of the paper as documentation of the cleaning step.
    """
    path = path or C.EAMPD_FILE
    robust = _read_dates(path, sheet)
    naive_raw = pd.read_excel(path, sheet_name=sheet, usecols=[0])
    naive = pd.to_datetime(naive_raw.iloc[:, 0], errors="coerce")
    mask = robust.notna() & naive.notna() & (robust.values != naive.values)
    return pd.DataFrame({
        "robust": robust[mask].dt.strftime("%Y-%m-%d"),
        "naive": naive[mask].dt.strftime("%Y-%m-%d"),
        "weekday_robust": robust[mask].dt.day_name(),
        "weekday_naive": naive[mask].dt.day_name(),
    }).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Sheet loading
# ---------------------------------------------------------------------------
def load_window(sheet: str, path=None) -> pd.DataFrame:
    """Load one event window with correctly parsed, sorted, unique dates."""
    path = path or C.EAMPD_FILE
    df = pd.read_excel(path, sheet_name=sheet)
    df.columns = [str(c).strip() for c in df.columns]
    dates = _read_dates(path, sheet)
    if len(dates) != len(df):
        raise ValueError(
            f"{sheet}: raw date column has {len(dates)} rows but the parsed "
            f"sheet has {len(df)}. The workbook layout has changed; inspect it "
            "before proceeding rather than aligning silently."
        )
    df["date"] = dates.values
    df = df.dropna(subset=["date"])
    if df["date"].duplicated().any():
        dupes = df.loc[df["date"].duplicated(keep=False), "date"]
        warnings.warn(f"{sheet}: duplicate event dates {sorted(set(dupes))}")
        df = df.drop_duplicates(subset="date", keep="first")
    numeric = [c for c in df.columns if c != "date"]
    df[numeric] = df[numeric].apply(pd.to_numeric, errors="coerce")
    return df.sort_values("date").reset_index(drop=True)


def load_all(path=None) -> dict[str, pd.DataFrame]:
    """Load all three event windows, aligned on the common set of event dates."""
    windows = {
        "PR": load_window(C.SHEET_PR, path),
        "PC": load_window(C.SHEET_PC, path),
        "ME": load_window(C.SHEET_ME, path),
    }
    common = set(windows["PR"]["date"])
    for w in windows.values():
        common &= set(w["date"])
    common = pd.Index(sorted(common))
    for k, w in windows.items():
        windows[k] = (w[w["date"].isin(common)]
                      .sort_values("date").reset_index(drop=True))
    return windows


# ---------------------------------------------------------------------------
# Sample construction
# ---------------------------------------------------------------------------
def assign_regime(dates: pd.Series) -> pd.Series:
    """Map event dates to QE / TRANSITION / NORMALISATION."""
    out = pd.Series(index=dates.index, dtype=object)
    out[dates <= C.QE_END] = "QE"
    out[(dates >= C.TRANSITION_START) & (dates <= C.TRANSITION_END)] = "TRANSITION"
    out[dates >= C.NORM_START] = "NORMALISATION"
    return out


def build_sample(windows: dict[str, pd.DataFrame],
                 start: str = C.SAMPLE_START,
                 drop_transition: bool = True) -> dict[str, pd.DataFrame]:
    """Restrict all windows to the estimation sample and attach regime labels."""
    dates = windows["ME"]["date"]
    keep = dates >= pd.Timestamp(start)
    regime = assign_regime(dates)
    if drop_transition:
        keep &= regime.ne("TRANSITION")
    out = {}
    for k, w in windows.items():
        sub = w[keep.values].copy()
        sub["regime"] = assign_regime(sub["date"]).values
        sub["QT"] = (sub["regime"] == "NORMALISATION").astype(float)
        out[k] = sub.reset_index(drop=True)
    return out


def coverage_report(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Non-missing counts for a list of columns, used in the data appendix."""
    return pd.DataFrame({
        "n_obs": [df[c].notna().sum() for c in cols],
        "n_missing": [df[c].isna().sum() for c in cols],
        "sd_bp": [np.round(df[c].std(), 2) for c in cols],
    }, index=cols)

"""Central configuration for the ECB surprise / Bund curve project.

All sample choices, regime dates and identification anchors live here so that
every design decision is visible in one place and can be varied in robustness
checks without touching estimation code.
"""
from pathlib import Path

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
OUT_FIG = ROOT / "output" / "figures"
OUT_TAB = ROOT / "output" / "tables"

EAMPD_FILE = DATA_RAW / "Dataset_EA-MPD.xlsx"

SHEET_PR = "Press Release Window"
SHEET_PC = "Press Conference Window"
SHEET_ME = "Monetary Event Window"

# ----------------------------------------------------------------------------
# Sample
# ----------------------------------------------------------------------------
# Sample starts with the announcement of the expanded asset purchase programme
# (22 January 2015). We use 1 January 2015 as the cut so that the January 2015
# meeting itself is included.
SAMPLE_START = "2015-01-01"

# Regime split. The ECB raised policy rates for the first time in eleven years
# on 21 July 2022. We treat the first half of 2022 as a transition period and
# exclude it from the baseline, because the Governing Council had already
# signalled the end of net asset purchases but had not yet moved rates.
QE_END = "2021-12-31"
TRANSITION_START = "2022-01-01"
TRANSITION_END = "2022-06-30"
NORM_START = "2022-07-01"

# Robustness: alternative regime break at the announced end of net APP
# purchases (December 2021 decision, effective March 2022).
ALT_BREAK = "2022-03-01"

# ----------------------------------------------------------------------------
# Instruments
# ----------------------------------------------------------------------------
# OIS maturities used to extract the policy factors. We deliberately exclude
# OIS_SW (spot week) and the 15Y/20Y points: the former is dominated by
# liquidity management noise, the latter are thinly traded on many event days.
OIS_MATS = ["OIS_1M", "OIS_3M", "OIS_6M", "OIS_1Y", "OIS_2Y", "OIS_3Y",
            "OIS_4Y", "OIS_5Y", "OIS_7Y", "OIS_10Y"]

# Numeric maturity in years, for plotting loading profiles.
OIS_YEARS = {"OIS_1M": 1 / 12, "OIS_3M": 0.25, "OIS_6M": 0.5, "OIS_1Y": 1.0,
             "OIS_2Y": 2.0, "OIS_3Y": 3.0, "OIS_4Y": 4.0, "OIS_5Y": 5.0,
             "OIS_7Y": 7.0, "OIS_10Y": 10.0}

# Scaling anchors. Each factor is normalised so that a one-unit realisation
# moves its anchor instrument by exactly one basis point. Coefficients in the
# response regressions are therefore read as "basis points of yield change per
# basis point of surprise in the anchor instrument".
ANCHOR_TARGET = "OIS_1M"
ANCHOR_PATH = "OIS_2Y"
ANCHOR_QE = "OIS_10Y"

# Zero restrictions used to rotate the press-conference factors. See
# src/factors.py for the exact algebra and paper Section 4 for discussion.
ZERO_RESTRICTIONS = [("PATH", "OIS_1M"), ("QE", "OIS_1M"), ("QE", "OIS_1Y")]

# ----------------------------------------------------------------------------
# Dependent variables
# ----------------------------------------------------------------------------
DE_MATS = ["DE2Y", "DE5Y", "DE10Y", "DE30Y"]
DE_YEARS = {"DE2Y": 2.0, "DE5Y": 5.0, "DE10Y": 10.0, "DE30Y": 30.0}

COUNTRIES = ["DE", "FR", "IT", "ES"]
PANEL_MATS = ["2Y", "5Y", "10Y"]
PERIPHERY = ["IT", "ES"]

# Modified duration used to translate yield changes into P&L in the scenario
# table. Approximated for a par bond at a 2.5 percent yield level; the exact
# level assumption is stated in the paper and varied in the appendix.
MOD_DURATION = {"DE2Y": 1.93, "DE5Y": 4.65, "DE10Y": 8.72, "DE30Y": 20.93}

# ----------------------------------------------------------------------------
# Inference
# ----------------------------------------------------------------------------
COV_TYPE = "HC1"          # heteroskedasticity-robust standard errors
N_BOOT = 4999             # wild bootstrap replications
BOOT_SEED = 20260810
ROLL_WINDOW = 25          # meetings per rolling window

# ----------------------------------------------------------------------------
# Presentation
# ----------------------------------------------------------------------------
COL_QE = "#1B3A6B"
COL_NORM = "#B03A2E"
COL_NEUTRAL = "#5D6D7E"
COL_ACCENT = "#C99700"
FIG_DPI = 320

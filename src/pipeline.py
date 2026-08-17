"""
28-day retail demand forecasting pipeline.

Model form (frozen after the validation phase; see README):
  derive(i, T)   -> (train_start, ramp_flag)   from A[i, :T] ONLY
  fit_predict    -> level x day-of-week x event-lift

Data decisions deliberately EXCLUDED from the feature path:
  - market_signal.csv : derived from the target (10 x sales x lognormal noise,
                        zero-pattern identical in 114780/114780 cells) and does
                        not cover d_1914-1941. Never loaded. See assert_no_market_signal().
  - vendor_signal.csv : benchmark only (residual corr 0.0007 after level+dow).
  - sell_prices.csv   : 227 price changes in 16980 transitions; elasticity not
                        identifiable. Not used as a feature.
  - snap_MH/KA/TN     : on/off ratios 0.9959/0.9944/0.9956, inside null bands.
"""
from __future__ import annotations

import hashlib
import inspect
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"

READ_LOG: list[str] = []   # every data file this process opens, for the leakage guard


def _read(name: str) -> pd.DataFrame:
    READ_LOG.append(name)
    return pd.read_csv(DATA / name)

N_SERIES = 60
HISTORY_END = 1913          # last observed day, d_1913
HORIZON = 28                # d_1914 .. d_1941

# --- derive() thresholds, exactly as validated -------------------------------
MIN_SEG = 120               # min segment length for the changepoint scan
LAUNCH_JUMP = 5.0           # post > 5 x pre
LAUNCH_PRE_ZERO = 0.85      # pre-window > 85% zeros
RAMP_MIN_LEN = 180          # need this much post-start data to judge a ramp
RAMP_THRESHOLD = 0.15       # zero-rate(1st half) - zero-rate(2nd half)

# --- level / dow / event -----------------------------------------------------
LEVEL_SHORT = 28
LEVEL_LONG = 56
DOW_WINDOW = 364
DOW_CLIP = (0.4, 2.5)
EVENT_MIN_OCC = 3
EVENT_SHRINK_K = 4.0        # lift -> 1 + (raw-1) * n/(n+k)

# --- horizon event overrides (Phase: calendar audit) -------------------------
# These REPLACE the training-derived lift on these two days; they do not stack.
# d_1921 = 2023-04-10 Ram Navami : raw 1.183, CI [1.028,1.338], 2/5 occurrences
#          below 1.0, no trend (perm p=0.53) -> shrink hard.
# d_1928 = 2023-04-17 Eid al-Fitr: raw 1.501, CI [1.306,1.696], visible but
#          non-significant decline (perm p=0.21) -> recency-shaded.
HORIZON_EVENT_OVERRIDE = {1921: 1.09, 1928: 1.38}


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def load():
    """Return (sales_df, A, calendar_df, wday, event_name). Never touches market_signal."""
    sales = _read("sales_train.csv")
    dcols = [f"d_{i}" for i in range(1, HISTORY_END + 1)]
    assert list(sales.columns[-HISTORY_END:]) == dcols, "d columns not d_1..d_1913 in order"
    A = sales[dcols].to_numpy(dtype=float)
    assert A.shape == (N_SERIES, HISTORY_END), A.shape
    assert not np.isnan(A).any() and (A >= 0).all()

    cal = _read("calendar.csv")
    cal["dn"] = cal["d"].str.slice(2).astype(int)
    cal = cal.sort_values("dn").reset_index(drop=True)
    wday = cal["wday"].to_numpy()
    event = cal["event_name_1"].to_numpy()
    return sales, A, cal, wday, event


def week_order(cal: pd.DataFrame) -> dict[int, int]:
    """
    Chronological index for wm_yr_wk, built from the first `d` of each week.

    NEVER sort on wm_yr_wk itself: it is non-monotone (is_monotonic_increasing
    is False). Week 1901 spans 2019-01-01..2019-12-31 and week 2352 (2023-01-01)
    would sort AFTER week 2317 (2023-04-30). 8 of 284 weeks are not 7 days.
    """
    first = cal.sort_values("dn").drop_duplicates("wm_yr_wk")
    return {int(w): k for k, w in enumerate(first["wm_yr_wk"])}


# ---------------------------------------------------------------------------
# derive: all data decisions, from the training window only
# ---------------------------------------------------------------------------
def _maxt_break(x: np.ndarray):
    """Max |t| single-changepoint scan. Returns (t, k) or None if too short."""
    n = len(x)
    if n < 2 * MIN_SEG + 5:
        return None
    c = np.cumsum(x)
    c2 = np.cumsum(x * x)
    tot, tot2 = c[-1], c2[-1]
    k = np.arange(MIN_SEG, n - MIN_SEG)
    m1 = c[k - 1] / k
    m2 = (tot - c[k - 1]) / (n - k)
    v1 = np.maximum(c2[k - 1] / k - m1 ** 2, 1e-9)
    v2 = np.maximum((tot2 - c2[k - 1]) / (n - k) - m2 ** 2, 1e-9)
    t = np.abs(m1 - m2) / np.sqrt(v1 / k + v2 / (n - k))
    j = int(np.argmax(t))
    return float(t[j]), int(k[j])


def derive(A: np.ndarray, i: int, T: int) -> tuple[int, bool]:
    """
    Re-derive truncation point and ramp flag from A[i, :T] ONLY.

    Nothing here may reference data at or beyond T, or any constant obtained by
    inspecting the full history (e.g. a hardcoded d_1345 launch day).
    """
    x = A[i, :T]
    nz = np.flatnonzero(x > 0)
    if nz.size == 0:
        return 0, False
    start = int(nz[0])

    res = _maxt_break(x)
    if res is not None:
        _, k = res
        pre, post = x[:k], x[k:]
        launch_shaped = (
            post.mean() > LAUNCH_JUMP * max(pre.mean(), 1e-9)
            and (pre == 0).mean() > LAUNCH_PRE_ZERO
        )
        if launch_shaped:
            start = k

    seg = x[start:]
    ramp = False
    if len(seg) >= RAMP_MIN_LEN:
        h = len(seg) // 2
        ramp = ((seg[:h] == 0).mean() - (seg[h:] == 0).mean()) > RAMP_THRESHOLD
    return start, bool(ramp)


# ---------------------------------------------------------------------------
# fit / predict
# ---------------------------------------------------------------------------
def fit_predict(A, wday, event, i, T, horizon=HORIZON, overrides=None, hedge=True):
    """
    Forecast `horizon` days from origin T for series i.

    hedge=True  -> ramp-flagged series use mean(28d level, 56d level)
    hedge=False -> ramp-flagged series use the 28d level (original validated spec)
    """
    start, ramp = derive(A, i, T)
    x = A[i, start:T]
    wd = wday[start:T]
    if len(x) < LEVEL_SHORT or x.sum() == 0:
        return np.zeros(horizon), start, ramp

    if ramp:
        lvl_s = x[-LEVEL_SHORT:].mean()
        lvl_l = x[-LEVEL_LONG:].mean() if len(x) >= LEVEL_LONG else lvl_s
        level = 0.5 * (lvl_s + lvl_l) if hedge else lvl_s
    else:
        level = x[-LEVEL_LONG:].mean() if len(x) >= LEVEL_LONG else x.mean()

    win = min(len(x), DOW_WINDOW)
    xx, ww = x[-win:], wd[-win:]
    gm = xx.mean()
    dfac = np.ones(8)
    if gm > 0:
        for w in range(1, 8):
            m = ww == w
            if m.sum() >= 8:
                dfac[w] = float(np.clip(xx[m].mean() / gm, *DOW_CLIP))

    ev_tr = event[start:T]
    base_ev = np.array([dfac[w] for w in wd]) * gm
    lift = {}
    if gm > 0:
        for nm in pd.unique(ev_tr[pd.notna(ev_tr)]):
            m = ev_tr == nm
            n = int(m.sum())
            if n >= EVENT_MIN_OCC and base_ev[m].sum() > 0:
                raw = x[m].sum() / base_ev[m].sum()
                lift[nm] = 1.0 + (raw - 1.0) * n / (n + EVENT_SHRINK_K)

    overrides = overrides or {}
    out = np.zeros(horizon)
    for h in range(horizon):
        d = T + h
        day_no = d + 1                       # 1-indexed d_N
        v = level * dfac[wday[d]]
        if day_no in overrides:
            v *= overrides[day_no]           # REPLACES derived lift, no stacking
        else:
            nm = event[d]
            if pd.notna(nm) and nm in lift:
                v *= lift[nm]
        out[h] = v
    return np.maximum(out, 0.0), start, ramp


def rmsse_denom(A, i, T):
    """M5 scale, from data strictly prior to T, anchored at first non-zero."""
    x = A[i, :T]
    nz = np.flatnonzero(x > 0)
    if nz.size == 0:
        return np.nan
    y = x[nz[0]:]
    return float(np.mean(np.diff(y) ** 2)) if len(y) > 1 else np.nan


# ---------------------------------------------------------------------------
# leakage guard
# ---------------------------------------------------------------------------
def assert_no_market_signal():
    """
    Hard assert that market_signal is absent from the executable feature path.

    Parses each function to an AST and strips docstrings/comments, so the
    prose above (which names the file in order to explain its exclusion)
    cannot satisfy or trip the check. Only real code counts.
    """
    import ast
    import textwrap

    banned = ("market_signal", "mkt_signal")
    offenders = []
    for f in (load, _read, week_order, derive, _maxt_break, fit_predict, rmsse_denom):
        tree = ast.parse(textwrap.dedent(inspect.getsource(f)))
        # ast.walk yields the docstring Constant independently of its Expr
        # wrapper, so collect those node ids up front and skip them by identity.
        doc_ids = {
            id(n.value) for n in ast.walk(tree)
            if isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant)
            and isinstance(n.value.value, str)
        }
        for node in ast.walk(tree):
            if id(node) in doc_ids:
                continue
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if any(b in node.value for b in banned):
                    offenders.append((f.__name__, node.value))
            if isinstance(node, ast.Name) and any(b in node.id for b in banned):
                offenders.append((f.__name__, node.id))
            if isinstance(node, ast.Attribute) and any(b in node.attr for b in banned):
                offenders.append((f.__name__, node.attr))
    assert not offenders, f"market_signal referenced in feature path: {offenders}"

    # runtime check: which data files were actually opened this process
    assert READ_LOG, "assert_no_market_signal() called before load()"
    assert not any(any(b in p for b in banned) for p in READ_LOG), \
        f"market_signal was read at runtime: {READ_LOG}"
    assert set(READ_LOG) == {"sales_train.csv", "calendar.csv"}, \
        f"unexpected data files read: {sorted(READ_LOG)}"
    return True


def file_digest(name: str) -> str:
    return hashlib.md5((DATA / name).read_bytes()).hexdigest()[:12]

"""Single entry point: writes submission.csv and runs the 8-fold rolling-origin backtest."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline import (
    HISTORY_END,
    HORIZON,
    HORIZON_EVENT_OVERRIDE,
    N_SERIES,
    assert_no_market_signal,
    derive,
    file_digest,
    fit_predict,
    load,
    rmsse_denom,
    week_order,
)

ROOT = Path(__file__).resolve().parent.parent


def backtest(A, wday, event, sales, n_blocks=8, hedge=True):
    origins = [HISTORY_END - HORIZON * k for k in range(1, n_blocks + 1)][::-1]
    rows, detail = [], []
    for T in origins:
        rec = []
        for i in range(N_SERIES):
            yhat, start, ramp = fit_predict(A, wday, event, i, T, hedge=hedge)
            y = A[i, T:T + HORIZON]
            den = rmsse_denom(A, i, T)
            rm = np.sqrt(np.mean((y - yhat) ** 2) / den) if den and np.isfinite(den) and den > 0 else np.nan
            rec.append((i, sales["item_id"][i], rm, np.abs(y - yhat).sum(), y.sum(), start, ramp))
        P = pd.DataFrame(rec, columns=["i", "item", "rmsse", "abserr", "act", "start", "ramp"])
        detail.append(P.assign(T=T))
        ag = P[P["item"] == "HOMECARE_2_AGARBATTI"]
        ot = P[P["item"] != "HOMECARE_2_AGARBATTI"]
        rows.append((
            f"d_{T+1}-{T+HORIZON}",
            round(P["rmsse"].mean(), 4),
            round(P["abserr"].sum() / P["act"].sum(), 4),
            round(ag["rmsse"].mean(), 3),
            round(ot["rmsse"].mean(), 3),
            int(P["ramp"].sum()),
        ))
    return (
        pd.DataFrame(rows, columns=["block", "RMSSE", "WAPE", "AGAR", "other50", "n_ramp"]),
        pd.concat(detail, ignore_index=True),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-hedge", action="store_true", help="use original 28d level for ramp series")
    args = ap.parse_args()
    hedge = not args.no_hedge

    sales, A, cal, wday, event = load()
    assert_no_market_signal()
    print("[guard] market_signal absent from feature path (AST + runtime read-log): OK")
    print(f"[data] sales_train md5={file_digest('sales_train.csv')} shape={A.shape}")

    order = week_order(cal)
    hz_weeks = [2314, 2315, 2316, 2317]
    pos = [order[w] for w in hz_weeks]
    assert pos == list(range(pos[0], pos[0] + 4)), f"horizon weeks not contiguous: {pos}"
    assert not cal["wm_yr_wk"].is_monotonic_increasing, "expected non-monotone wm_yr_wk"
    print(f"[guard] week order from first-d; horizon weeks at positions {pos} (contiguous): OK")

    # ---- backtest -----------------------------------------------------------
    for label, h in (("hedge", True), ("no-hedge (validated spec)", False)):
        R, _ = backtest(A, wday, event, sales, hedge=h)
        print(f"\n=== 8-fold rolling-origin backtest [{label}] ===")
        print(R.to_string(index=False))
        print(f"mean RMSSE {R.RMSSE.mean():.4f}  sd {R.RMSSE.std():.4f}"
              f"  | WAPE {R.WAPE.mean():.4f}  | AGAR {R.AGAR.mean():.3f}"
              f"  other50 {R.other50.mean():.3f}")
        if h == hedge:
            chosen = R

    # ---- final forecast -----------------------------------------------------
    print(f"\n=== final fit T={HISTORY_END}, overrides {HORIZON_EVENT_OVERRIDE} ===")
    preds = np.zeros((N_SERIES, HORIZON))
    meta = []
    for i in range(N_SERIES):
        yhat, start, ramp = fit_predict(
            A, wday, event, i, HISTORY_END,
            overrides=HORIZON_EVENT_OVERRIDE, hedge=hedge,
        )
        preds[i] = yhat
        meta.append((sales["id"][i], start + 1, ramp))
    M = pd.DataFrame(meta, columns=["id", "train_start_d", "ramp"])
    print(f"truncated (start>d_1): {(M.train_start_d > 1).sum()}/60  |  ramp-flagged: {M.ramp.sum()}/60")
    print(M[M.ramp].assign(item=lambda d: d.id.str.rsplit('_', n=3).str[0])
          .groupby("item").agg(n=("id", "size"), start=("train_start_d", "median")).to_string())

    assert preds.shape == (N_SERIES, HORIZON)
    assert np.isfinite(preds).all() and (preds >= 0).all(), "non-finite or negative forecasts"

    sub = pd.DataFrame(preds, columns=[f"F{i}" for i in range(1, HORIZON + 1)])
    sub.insert(0, "id", sales["id"].to_numpy())
    out = ROOT / "submission.csv"
    sub.to_csv(out, index=False)
    print(f"\nwrote {out}  rows={len(sub)}  total_units={preds.sum():.1f}")
    print(sub.iloc[[0, 59], :6].to_string(index=False))


if __name__ == "__main__":
    main()

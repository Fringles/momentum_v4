#!/usr/bin/env python3
import argparse
import os
import sqlite3
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def read_cdi_series(cdi_path: str) -> pd.Series:
    assert os.path.exists(cdi_path), f"CDI file not found: {cdi_path}"
    cdi = pd.read_excel(cdi_path)
    cdi["Date"] = pd.to_datetime(cdi["Date"])
    cdi["Close"] = pd.to_numeric(cdi["Close"], errors="coerce")
    cdi = cdi.dropna().sort_values("Date").set_index("Date")["Close"].astype(float)
    return cdi


def load_bova11_from_db(db_path: str, ticker: str = "BOVA11") -> pd.Series:
    with sqlite3.connect(db_path) as con:
        q = """
        SELECT dp.date as date, dp.adjusted_close as adj_close
        FROM daily_prices dp
        JOIN securities s ON s.security_id = dp.security_id
        WHERE s.ticker = ?
        ORDER BY dp.date ASC
        """
        df = pd.read_sql_query(q, con, params=[ticker], parse_dates=["date"])  # type: ignore
    if df.empty:
        return pd.Series(dtype=float)
    return df.set_index("date")["adj_close"].astype(float).sort_index()


def load_pivots_from_db(db_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    with sqlite3.connect(db_path) as con:
        q = """
        SELECT dp.date as date, s.ticker as ticker,
               dp.open_price as open_price,
               dp.close_price as close_price,
               dp.adjusted_close as adjusted_close
        FROM daily_prices dp
        JOIN securities s ON s.security_id = dp.security_id
        WHERE s.security_type = 'Stock' AND s.country = 'BR'
        ORDER BY dp.date ASC
        """
        df = pd.read_sql_query(q, con, parse_dates=["date"])  # type: ignore
    # Adjusted close to compute return base
    pc = df.pivot(index="date", columns="ticker", values="adjusted_close").sort_index()
    # Use adjusted close daily returns for stability
    r = pc.pct_change()
    return pc, r


def main():
    p = argparse.ArgumentParser(description="Export daily price indices for strategy (LS, D10), CDI, and BOVA11")
    p.add_argument("--results-dir", default=os.path.join(os.getcwd(), "results"))
    p.add_argument("--out-csv", default=os.path.join(os.getcwd(), "results", "momentum_br_daily_prices.csv"))
    p.add_argument("--db-path", default=os.path.expanduser(
        "/Users/hemerson/Library/CloudStorage/GoogleDrive-hemersondv@gmail.com/My Drive/mel-database/financial_data.db"))
    p.add_argument("--cdi-path", default=os.path.expanduser(
        "/Users/hemerson/Library/CloudStorage/GoogleDrive-hemersondv@gmail.com/My Drive/mel-database/cdi_series.xlsx"))
    p.add_argument("--bova11-ticker", default="BOVA11")
    p.add_argument("--apply-overlay", action="store_true", help="Apply monthly hedge ratio to LS daily series, if available")
    args = p.parse_args()

    ts_path = os.path.join(args.results_dir, "momentum_br_timeseries.csv")
    assert os.path.exists(ts_path), f"Timeseries file not found: {ts_path}"
    pf = pd.read_csv(ts_path, parse_dates=["month_end", "trade_date", "next_trade_date"]).set_index("month_end").sort_index()

    # Load markets
    cdi_index = read_cdi_series(args.cdi_path)
    bm_price = load_bova11_from_db(args.db_path, args.bova11_ticker)
    bm_ret = bm_price.pct_change()
    pc, r = load_pivots_from_db(args.db_path)

    # Helpers
    def read_account_targets(period_label: str) -> pd.DataFrame:
        path = os.path.join(args.results_dir, f"account_targets_{period_label}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing account targets file: {path}")
        df = pd.read_csv(path, parse_dates=["trade_date"])  # type: ignore
        return df

    # Prepare containers for daily returns
    daily_ls_unhedged: Dict[pd.Timestamp, float] = {}
    daily_d10: Dict[pd.Timestamp, float] = {}
    daily_ls_hedged: Dict[pd.Timestamp, float] = {}

    # Iterate holding periods
    for me, row in pf.iterrows():
        trade_date = pd.Timestamp(row["trade_date"]).normalize()
        next_trade = pd.Timestamp(row["next_trade_date"]).normalize()
        period_label = me.strftime("%Y-%m")

        # Load composite targets (already combined across cohorts)
        df_t = read_account_targets(period_label)
        # LS weights (signed) and D10 weights (long-only)
        w_ls = (df_t[df_t["book"] == "LS"][["ticker", "target_weight"]]
                .set_index("ticker")["target_weight"].astype(float))
        w_d10 = (df_t[df_t["book"] == "D10"][["ticker", "target_weight"]]
                 .set_index("ticker")["target_weight"].astype(float))

        # Daily date range: use trading dates in return matrix between trade_date (exclusive) and next_trade (inclusive)
        # Use (trade_date, next_trade] to align with monthly open-to-open
        days = r.index[(r.index > trade_date) & (r.index <= next_trade)]
        if len(days) == 0:
            continue

        for d in days:
            ret_day = r.loc[d]
            # D10: long-only, renorm to sum 1 over available names
            names_d10 = w_d10.index.intersection(ret_day.index[ret_day.notna()])
            if len(names_d10) > 0:
                w_d = w_d10.loc[names_d10]
                s = float(w_d.sum())
                if s > 0:
                    wl = w_d / s
                    r_d10 = float((wl * ret_day.loc[names_d10].astype(float)).sum())
                else:
                    r_d10 = np.nan
            else:
                r_d10 = np.nan
            daily_d10[d] = r_d10

            # LS: preserve 0.5 per side on available names
            names_ls = w_ls.index.intersection(ret_day.index[ret_day.notna()])
            if len(names_ls) == 0:
                daily_ls_unhedged[d] = np.nan
                continue
            w_sub = w_ls.loc[names_ls]
            long = w_sub[w_sub > 0]
            short = w_sub[w_sub < 0]
            r_long = float(0.0)
            r_short = float(0.0)
            if len(long) > 0:
                sL = float(long.sum())
                if sL > 0:
                    wl = long * (0.5 / sL)
                    r_long = float((wl * ret_day.loc[long.index].astype(float)).sum())
            if len(short) > 0:
                sS = float(-short.sum())
                if sS > 0:
                    ws = short * (0.5 / sS)
                    r_short = float((ws * ret_day.loc[short.index].astype(float)).sum())
            daily_ls_unhedged[d] = r_long + r_short

            # Apply overlay if requested and available
            if args.apply_overlay and ("hedge_ratio" in pf.columns):
                h = row.get("hedge_ratio", np.nan)
                h = float(h) if pd.notna(h) else 0.0
                r_bm = float(bm_ret.loc[d]) if d in bm_ret.index and pd.notna(bm_ret.loc[d]) else 0.0
                daily_ls_hedged[d] = daily_ls_unhedged[d] + h * r_bm

    # Assemble daily price indices
    if len(daily_ls_unhedged) == 0 and len(daily_d10) == 0:
        raise RuntimeError("No daily returns computed; check results and database paths.")

    idx = sorted(set(daily_d10.keys()) | set(daily_ls_unhedged.keys()))
    df_ret = pd.DataFrame(index=pd.DatetimeIndex(idx).sort_values())
    if daily_d10:
        df_ret["D10_ret"] = pd.Series(daily_d10)
    if daily_ls_unhedged:
        df_ret["LS_pre_overlay_ret"] = pd.Series(daily_ls_unhedged)
    if daily_ls_hedged:
        df_ret["LS_ret"] = pd.Series(daily_ls_hedged)
    else:
        df_ret["LS_ret"] = df_ret["LS_pre_overlay_ret"]

    # Market benchmarks
    # Align CDI index and BOVA11 price to daily index range
    cdi_norm = (cdi_index / cdi_index.loc[cdi_index.index.min()]) if not cdi_index.empty else pd.Series(dtype=float)
    bm_norm = (bm_price / bm_price.iloc[0]) if not bm_price.empty else pd.Series(dtype=float)

    # Build price indices starting at 1.0
    def to_index(ret: pd.Series) -> pd.Series:
        return (1.0 + ret.fillna(0)).cumprod()

    out = pd.DataFrame(index=df_ret.index)
    out["LS_pre_overlay"] = to_index(df_ret["LS_pre_overlay_ret"]) if "LS_pre_overlay_ret" in df_ret else np.nan
    out["LS"] = to_index(df_ret["LS_ret"]) if "LS_ret" in df_ret else out["LS_pre_overlay"]
    out["D10"] = to_index(df_ret["D10_ret"]) if "D10_ret" in df_ret else np.nan
    # Rebase CDI and BOVA11 to 1 on first output date
    if not cdi_norm.empty:
        # reindex and forward fill to cover all output dates
        cdi_al = cdi_norm.reindex(out.index, method="ffill")
        out["CDI"] = cdi_al / float(cdi_al.iloc[0])
    if not bm_norm.empty:
        bm_al = bm_norm.reindex(out.index, method="ffill")
        out["BOVA11"] = bm_al / float(bm_al.iloc[0])

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    out.reset_index().rename(columns={"index": "date"}).to_csv(args.out_csv, index=False)
    print(f"Wrote daily price indices -> {args.out_csv}")


if __name__ == "__main__":
    main()


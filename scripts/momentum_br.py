#!/usr/bin/env python3
import argparse
import os
import sqlite3
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import json


@dataclass
class Config:
    db_path: str
    cdi_path: str
    liquidity_threshold: float = 2_000_000.0
    start_date: str = "2000-01-01"
    end_date: Optional[str] = None
    nw_lags: int = 6
    min_eligible: int = 50
    bova11_ticker: str = "BOVA11"
    cohorts: int = 1
    beta_overlay: bool = False
    overlay_lookback_months: int = 36
    overlay_halflife_days: int = 60
    overlay_shrink: float = 0.5
    overlay_band: float = 0.11293
    band_keep: float = 0.81393
    band_add: float = 0.90307
    gross_per_side: float = 0.50
    gross_tol: float = 0.02
    single_cap_pct: float = 0.025
    floor_factor: float = 0.25
    sector_tol: float = 0.09218
    effn_ratio: float = 0.70
    zero_frac: float = 0.20
    max_add_frac: float = 0.05
    exit_cap_frac: float = 0.07257
    adaptive_exit_cap: bool = True
    cap_min: float = 0.05
    cap_max: float = 0.12
    ls_turnover_budget: float = 0.32496
    use_turnover_budget: bool = True
    micro_add_frac: float = 0.01774
    kill_alpha_t_min: float = 2.0
    kill_net_sharpe_min: float = 0.0
    # Vol targeting
    apply_vol_target: bool = False
    vol_target_ann: float = 0.10
    vol_window_months: int = 36
    vol_min_months: int = 12
    live_capital: Optional[float] = None
    lot_size: int = 1
    write_state_snapshots: bool = False
    # Cold start options
    cold_start: bool = False
    ramp_frac: Optional[float] = None
    out_dir: str = "results"


def read_equity_data(cfg: Config) -> pd.DataFrame:
    """Read daily prices for Brazilian stocks from SQLite DB.

    Returns a tidy DataFrame: [date, ticker, adj_open, adj_close, volume]
    """
    assert os.path.exists(cfg.db_path), f"Database not found: {cfg.db_path}"
    con = sqlite3.connect(cfg.db_path)
    q = f'''
    SELECT
        dp.date as date,
        s.ticker as ticker,
        s.sector_id as sector_id,
        dp.open_price as open_price,
        dp.close_price as close_price,
        dp.adjusted_close as adjusted_close,
        dp.volume as volume
    FROM daily_prices dp
    JOIN securities s ON s.security_id = dp.security_id
    WHERE s.security_type = 'Stock'
      AND s.country = 'BR'
      AND dp.date >= ?
      {"AND dp.date <= ?" if cfg.end_date else ''}
    ORDER BY dp.date ASC
    '''
    params = [cfg.start_date]
    if cfg.end_date:
        params.append(cfg.end_date)
    df = pd.read_sql_query(q, con, params=params, parse_dates=["date"])  # type: ignore
    con.close()

    # Compute adjusted open using close adjustment factor
    # adj_factor = adjusted_close / close_price
    # adj_open = open_price * adj_factor
    with np.errstate(divide='ignore', invalid='ignore'):
        adj_factor = df["adjusted_close"] / df["close_price"]
        adj_open = df["open_price"] * adj_factor
    df["adj_open"] = adj_open.replace([np.inf, -np.inf], np.nan)
    df.rename(columns={"adjusted_close": "adj_close"}, inplace=True)
    df = df[["date", "ticker", "sector_id", "adj_open", "adj_close", "volume"]]
    return df


def read_sector_map(cfg: Config) -> Dict[int, str]:
    """Return mapping sector_id -> sector_name."""
    con = sqlite3.connect(cfg.db_path)
    q = "SELECT sector_id, sector_name FROM sectors"
    df = pd.read_sql_query(q, con)
    con.close()
    return {int(r["sector_id"]): str(r["sector_name"]) for _, r in df.iterrows()}


def read_cdi_series(cdi_path: str) -> pd.Series:
    """Read CDI 'Close' index series from Excel; coerce to numeric; return daily index as Series."""
    assert os.path.exists(cdi_path), f"CDI file not found: {cdi_path}"
    cdi = pd.read_excel(cdi_path)
    # Expect columns: Date, Close (index level). Coerce Close to numeric and drop invalid rows.
    cdi["Date"] = pd.to_datetime(cdi["Date"])
    cdi["Close"] = pd.to_numeric(cdi["Close"], errors="coerce")
    cdi = cdi.dropna().sort_values("Date")
    cdi = cdi.set_index("Date")["Close"].astype(float)
    return cdi


def build_pivots(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return pivots: adj_open, adj_close, volume with [date x ticker]."""
    # Ensure DatetimeIndex daily sorted
    df = df.dropna(subset=["date", "ticker"]).copy()
    df.sort_values(["date", "ticker"], inplace=True)
    po = df.pivot(index="date", columns="ticker", values="adj_open")
    pc = df.pivot(index="date", columns="ticker", values="adj_close")
    pv = df.pivot(index="date", columns="ticker", values="volume")
    return po, pc, pv


def compute_month_ends(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Return actual last trading day for each calendar month present in index."""
    s = pd.Series(index=index, data=1)
    month_ends = s.groupby(index.to_period("M")).apply(lambda x: x.index.max())
    month_ends = pd.DatetimeIndex(month_ends.values)
    month_ends = month_ends.sort_values()
    return month_ends


def last_weekday_of_month(d: pd.Timestamp) -> pd.Timestamp:
    """Return the last Monday–Friday calendar day for the month of d.
    Note: This ignores exchange-specific holidays but prevents treating early-month dates
    (e.g., the 1st/2nd) as a pseudo month-end in live mode when only partial data exists.
    """
    # Find last calendar day of month
    year = d.year
    month = d.month
    # First day of next month, then step back one day
    if month == 12:
        next_month = pd.Timestamp(year + 1, 1, 1)
    else:
        next_month = pd.Timestamp(year, month + 1, 1)
    last_cal = next_month - pd.Timedelta(days=1)
    # Move back to last weekday (Mon-Fri)
    while last_cal.weekday() >= 5:  # 5=Sat,6=Sun
        last_cal -= pd.Timedelta(days=1)
    return last_cal


def next_trading_day(index: pd.DatetimeIndex, ref_date: pd.Timestamp) -> Optional[pd.Timestamp]:
    arr = index.values
    pos = index.searchsorted(ref_date, side="right")
    if pos < len(index):
        return pd.Timestamp(arr[pos])
    return None


def compute_momentum_scores(monthly_close: pd.DataFrame) -> pd.DataFrame:
    """Compute 12-2 momentum score: ln(P[t-2]) - ln(P[t-12]). Index is calendar month-end."""
    logp = np.log(monthly_close)
    mom = logp.shift(2) - logp.shift(12)
    return mom


def decile_labels(n: int) -> List[int]:
    return list(range(1, n + 1))


def newey_west_tstat(x: pd.Series, lags: int = 6) -> Tuple[float, float]:
    """Return (mean, tstat) using Newey-West HAC standard errors with given lags."""
    import statsmodels.api as sm

    x = pd.Series(x).dropna()
    if len(x) < (lags + 2):
        return np.nan, np.nan
    # OLS of x on constant
    y = x.values
    X = np.ones((len(y), 1))
    model = sm.OLS(y, X)
    results = model.fit(cov_type='HAC', cov_kwds={'maxlags': lags})
    mean_est = results.params[0]
    se = results.bse[0]
    tstat = mean_est / se if se and not np.isnan(se) else np.nan
    return float(mean_est), float(tstat)


def ols_alpha_newey_west(y: pd.Series, x: pd.Series, lags: int = 6) -> Dict[str, float]:
    import statsmodels.api as sm

    df = pd.concat([y, x], axis=1, keys=["y", "x"]).dropna()
    if df.empty or len(df) < (lags + 3):
        return {"alpha": np.nan, "alpha_t": np.nan, "beta": np.nan}
    X = sm.add_constant(df["x"].values)
    model = sm.OLS(df["y"].values, X)
    results = model.fit(cov_type='HAC', cov_kwds={'maxlags': lags})
    alpha = results.params[0]
    beta = results.params[1]
    alpha_se = results.bse[0]
    alpha_t = alpha / alpha_se if alpha_se and not np.isnan(alpha_se) else np.nan
    return {"alpha": float(alpha), "alpha_t": float(alpha_t), "beta": float(beta)}


def capm_alpha_ir(y: pd.Series, x: pd.Series, lags: int = 6) -> Dict[str, float]:
    """Compute CAPM alpha (HAC t) and annualized information ratio vs benchmark.

    IR is defined as annualized alpha divided by annualized tracking error,
    where tracking error is the standard deviation of OLS residuals.
    """
    import statsmodels.api as sm

    df = pd.concat([y, x], axis=1, keys=["y", "x"]).dropna()
    if df.empty or len(df) < (lags + 3):
        return {"alpha": np.nan, "alpha_t": np.nan, "beta": np.nan, "ir": np.nan, "te_ann": np.nan}
    X = sm.add_constant(df["x"].values)
    model = sm.OLS(df["y"].values, X)
    results = model.fit(cov_type='HAC', cov_kwds={'maxlags': lags})
    alpha = float(results.params[0])
    beta = float(results.params[1])
    alpha_se = float(results.bse[0]) if results.bse is not None else np.nan
    alpha_t = alpha / alpha_se if alpha_se and not np.isnan(alpha_se) else np.nan
    resid = pd.Series(results.resid, index=df.index)
    te_m = float(resid.std(ddof=1)) if len(resid.dropna()) >= 3 else np.nan
    te_ann = te_m * np.sqrt(12.0) if te_m and np.isfinite(te_m) else np.nan
    ir = (np.sqrt(12.0) * alpha / te_m) if te_m and np.isfinite(te_m) and te_m > 0 else np.nan
    return {"alpha": alpha, "alpha_t": float(alpha_t), "beta": beta, "ir": float(ir), "te_ann": float(te_ann)}


def max_drawdown(returns: pd.Series) -> float:
    r = (1 + returns.fillna(0)).cumprod()
    peak = r.cummax()
    dd = r / peak - 1.0
    return float(dd.min())


def run_analysis(cfg: Config) -> None:
    out_dir = os.path.abspath(cfg.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    # 1) Load data
    print("Loading equity data ...")
    df = read_equity_data(cfg)
    print(f"Loaded {len(df):,} daily rows for Brazilian stocks")

    print("Building pivots ...")
    # Sector map
    sector_map = read_sector_map(cfg)
    ticker_sector: Dict[str, str] = {}
    # Use the most frequent sector assignment per ticker
    tmp = df[["ticker", "sector_id"]].dropna()
    if not tmp.empty:
        mode_sector = tmp.groupby("ticker")["sector_id"].agg(lambda x: x.value_counts().index[0])
        ticker_sector = {t: sector_map.get(int(sid), "Unknown") for t, sid in mode_sector.items()}

    po, pc, pv = build_pivots(df)
    all_dates = po.index
    month_ends = compute_month_ends(all_dates)
    # In live runs, if the latest available date falls within the current month (not true month-end),
    # do not treat it as a month-end. Only trade the last fully completed month.
    if len(month_ends) > 0:
        last_obs = all_dates[-1]
        last_me = month_ends[-1]
        if last_me.to_period('M') == last_obs.to_period('M'):
            lw = last_weekday_of_month(last_me)
            # If the observed month-end is earlier than the true last weekday of the month,
            # it indicates a partial month. Drop it from processing.
            if last_me.normalize() != lw.normalize():
                month_ends = month_ends[:-1]
    # map period -> last trading date
    period_to_mend: Dict[pd.Period, pd.Timestamp] = {
        pd.Period(d, freq='M'): d for d in month_ends
    }

    # 2) Momentum scores on monthly closes
    print("Computing monthly closes and momentum 12-2 ...")
    monthly_close = pc.resample('M').last()
    momentum = compute_momentum_scores(monthly_close)

    # 3) Liquidity filter: rolling 63-day median daily volume
    print("Computing rolling 63-day median volume ...")
    rolling_med_vol = pv.rolling(window=63, min_periods=20).median()

    # 4) Prepare trade calendar: trade at next day open (T+1)
    print("Preparing trading calendar ...")
    trade_dates: List[pd.Timestamp] = []
    next_trade_dates: List[pd.Timestamp] = []
    mend_labels: List[pd.Timestamp] = []
    for d in month_ends:
        # Special case: Live trading scenario where we're running on the actual last trading day
        # In this case, d_trade won't exist in our database yet, but we still need to generate targets
        if d == all_dates[-1]:
            print(f"Live trading detected: generating targets for month-end {d}")
            # Calculate next business day after month-end (even if not in database)
            d_trade = d + pd.Timedelta(days=1)
            # Skip weekends
            while d_trade.weekday() >= 5:  # 5=Saturday, 6=Sunday
                d_trade += pd.Timedelta(days=1)
            # For the holding period end, use a reasonable approximation (30 days later)
            # This won't affect target generation, only return calculations (which we can't do anyway in live mode)
            d_next_trade = d_trade + pd.Timedelta(days=30)
            print(f"Live trading: will trade on {d_trade}, holding period placeholder: {d_next_trade}")
            mend_labels.append(d)
            trade_dates.append(d_trade)
            next_trade_dates.append(d_next_trade)
            break  # This is the last month we can process
        
        d_trade = next_trading_day(all_dates, d)
        if d_trade is None:
            continue
        # find next month end then next trade date
        p = pd.Period(d, freq='M')
        p_next = p + 1
        d_next_mend = period_to_mend.get(p_next)
        if d_next_mend is None:
            # try deriving from index directly
            # find the max date with period p_next
            mask = all_dates.to_period('M') == p_next
            if mask.any():
                d_next_mend = all_dates[mask][-1]
            else:
                # If we can't find next month-end, check if this is the last complete month
                # For live trading: if we have data after the current month-end trade date,
                # we can still generate targets even without the full next month
                if d_trade in all_dates and len(all_dates[all_dates > d_trade]) > 0:
                    # Use the last available date as the end of holding period
                    d_next_trade = all_dates[-1]
                    mend_labels.append(d)
                    trade_dates.append(d_trade)
                    next_trade_dates.append(d_next_trade)
                break
        d_next_trade = next_trading_day(all_dates, d_next_mend)
        if d_next_trade is None:
            # Handle edge case: we're at the end of our data but can still trade
            # For live trading, if we have at least a few days of data after trade date,
            # we can generate targets even without knowing the exact end of holding period
            if d_trade in all_dates:
                days_after_trade = len(all_dates[all_dates > d_trade])
                if days_after_trade >= 1:  # At least 1 day of data after trade date
                    d_next_trade = all_dates[-1]
                    print(f"Using end-of-data {d_next_trade} as holding period end for month-end {d}")
                    mend_labels.append(d)
                    trade_dates.append(d_trade)
                    next_trade_dates.append(d_next_trade)
            continue
        mend_labels.append(d)
        trade_dates.append(d_trade)
        next_trade_dates.append(d_next_trade)

    if not trade_dates:
        raise RuntimeError("No trade periods detected.")

    # 5) Portfolio formation and returns
    print("Forming portfolios and computing returns ...")
    deciles = decile_labels(10)
    rows = []
    # Track portfolio target weights to compute turnover at each rebalance
    prev_w_d10: Dict[str, float] = {}
    prev_w_ls: Dict[str, float] = {}

    def compute_turnover(prev_w: Dict[str, float], cur_w: Dict[str, float]) -> float:
        keys = set(prev_w.keys()).union(cur_w.keys())
        return float(sum(abs(cur_w.get(k, 0.0) - prev_w.get(k, 0.0)) for k in keys))
    # Precompute daily returns for beta estimates (12m) and benchmark daily returns
    daily_ret = pc.pct_change()
    # Benchmark daily returns from adjusted close
    with sqlite3.connect(cfg.db_path) as con:
        q_bm = """
        SELECT dp.date as date, dp.adjusted_close as adj_close
        FROM daily_prices dp
        JOIN securities s ON s.security_id = dp.security_id
        WHERE s.ticker = ?
        ORDER BY dp.date ASC
        """
        bm_close = pd.read_sql_query(q_bm, con, params=[cfg.bova11_ticker], parse_dates=["date"])  # type: ignore
    bm_close = bm_close.set_index("date")["adj_close"].dropna().sort_index()
    bm_daily_ret = bm_close.pct_change()
    bm_index = (1.0 + bm_daily_ret.fillna(0.0)).cumprod()

    def betas_at_date(d: pd.Timestamp, names: List[str], lookback_days: int = 252) -> pd.Series:
        # window end inclusive d; select last lookback_days
        idx = daily_ret.index
        if d not in idx:
            # use previous available
            pos = idx.searchsorted(d, side="right") - 1
            if pos < 0:
                return pd.Series(index=names, dtype=float)
            d_eff = idx[pos]
        else:
            d_eff = d
        pos = idx.get_loc(d_eff)
        start = max(0, pos - lookback_days + 1)
        win_idx = idx[start:pos + 1]
        r_win = daily_ret.loc[win_idx, names]
        bm_win = bm_daily_ret.reindex(win_idx).dropna()
        if bm_win.empty or bm_win.var() == 0:
            return pd.Series(index=names, dtype=float)
        betas = {}
        var_m = float(bm_win.var(ddof=0))
        for n in names:
            ri = r_win[n].dropna()
            dfj = pd.concat([ri, bm_win], axis=1, keys=["ri", "rm"]).dropna()
            if len(dfj) < 60 or var_m == 0:
                betas[n] = np.nan
            else:
                cov = float(np.cov(dfj["ri"], dfj["rm"], ddof=0)[0, 1])
                betas[n] = cov / var_m if var_m != 0 else np.nan
        return pd.Series(betas)

    # Helper to compute scaled portfolio returns given weights and stock returns
    def ew_long_return(weights: Dict[str, float], stock_ret: pd.Series) -> Optional[float]:
        if not weights:
            return None
        names = [k for k in weights.keys() if k in stock_ret.index and pd.notna(stock_ret[k])]
        if not names:
            return None
        w_sel = {k: weights[k] for k in names}
        total = sum(max(0.0, w) for w in w_sel.values())
        if total <= 0:
            return None
        # normalize to sum 1 on the long side
        r = sum((max(0.0, w) / total) * float(stock_ret[k]) for k, w in w_sel.items())
        return float(r)

    def beta_neutral_ls_return(weights: Dict[str, float], stock_ret: pd.Series) -> Optional[float]:
        if not weights:
            return None
        names = [k for k in weights.keys() if k in stock_ret.index and pd.notna(stock_ret[k])]
        if not names:
            return None
        w_sel = {k: weights[k] for k in names}
        long_sum = sum(max(0.0, w) for w in w_sel.values())
        short_sum = sum(-min(0.0, w) for w in w_sel.values())
        if long_sum <= 0 or short_sum <= 0:
            return None
        # scale each side to maintain original gross 1 (0.5 per side)
        scale_L = 0.5 / long_sum
        scale_S = 0.5 / short_sum
        r = 0.0
        for k, w in w_sel.items():
            if w > 0:
                r += (w * scale_L) * float(stock_ret[k])
            elif w < 0:
                r += (w * scale_S) * float(stock_ret[k])
        return float(r)

    use_staggered = isinstance(cfg.cohorts, int) and cfg.cohorts and cfg.cohorts > 1
    # State for adaptive exit caps and severity tracking
    prev_perc_map: Dict[str, float] = {}
    D_hist: List[float] = []
    if use_staggered:
        # Maintain per-cohort weights; seed all cohorts on first valid month
        prev_w_d10_coh: List[Dict[str, float]] = [dict() for _ in range(cfg.cohorts)]
        prev_w_ls_coh: List[Dict[str, float]] = [dict() for _ in range(cfg.cohorts)]
        # Durations in months per cohort and side
        dur_long: List[Dict[str, int]] = [dict() for _ in range(cfg.cohorts)]
        dur_short: List[Dict[str, int]] = [dict() for _ in range(cfg.cohorts)]
        # Strike counters for marginal breaches
        strikes_long: List[Dict[str, int]] = [dict() for _ in range(cfg.cohorts)]
        strikes_short: List[Dict[str, int]] = [dict() for _ in range(cfg.cohorts)]
        seeded = False

    for idx_loop, (d_mend, d_trade, d_next_trade) in enumerate(zip(mend_labels, trade_dates, next_trade_dates)):
        # Check if this is live trading mode (trade date not in our price data)
        is_live_trading = d_trade not in po.index
        # eligible universe by liquidity at month-end
        vol_row = rolling_med_vol.loc[d_mend]
        eligible = vol_row.index[vol_row >= cfg.liquidity_threshold]
        if len(eligible) < cfg.min_eligible:
            # skip if too few names
            continue

        # momentum scores for this month (use calendar month-end label)
        period = pd.Period(d_mend, freq='M')
        mom_row = momentum.loc[period.to_timestamp('M')]
        mom_row = mom_row.dropna()
        mom_row = mom_row.loc[mom_row.index.intersection(eligible)]
        if mom_row.empty:
            continue

        # Sector-neutral z-score within sector, then global ranking by z-score
        if ticker_sector:
            sectors_series = pd.Series({t: ticker_sector.get(t, "Unknown") for t in mom_row.index})
            df_sc = pd.DataFrame({"mom": mom_row, "sector": sectors_series})
            def zscore(g: pd.Series) -> pd.Series:
                mu = g.mean()
                sd = g.std(ddof=0)
                return (g - mu) / sd if sd and np.isfinite(sd) and sd > 1e-12 else g * 0.0
            df_sc["z"] = df_sc.groupby("sector")["mom"].transform(zscore)
            signal_for_rank = df_sc["z"]
        else:
            signal_for_rank = mom_row
        if mom_row.empty:
            continue
        # assign deciles
        try:
            labels = pd.qcut(signal_for_rank, q=10, labels=deciles, duplicates='drop')
        except ValueError:
            # not enough unique values
            continue

        # stock period returns (open-to-open)
        if is_live_trading:
            # For live trading, we can't calculate returns since we don't have future prices
            # Use the last available prices for target generation
            start_open = po.loc[d_mend]  # Use month-end prices as reference
            stock_ret = pd.Series(index=eligible, dtype=float)  # Empty returns for live mode
        else:
            start_open = po.loc[d_trade]
            end_open = po.loc[d_next_trade].reindex(start_open.index)
            # Conservative handling: assume vanished names are full write-offs.
            if len(labels) > 0:
                missing_end = labels.index[end_open.reindex(labels.index).isna()]
                if len(missing_end) > 0:
                    end_open = end_open.copy()
                    end_open.loc[missing_end] = 0.0
            stock_ret = (end_open / start_open - 1.0).dropna()
        # intersect with names having returns
        valid_names = stock_ret.index.intersection(labels.index)
        if len(valid_names) < cfg.min_eligible:
            continue
        labels = labels.loc[valid_names]
        stock_ret = stock_ret.loc[valid_names]

        # compute decile returns (equal-weight) for reference only in non-staggered mode
        d10_members = labels[labels == 10].index.tolist()
        d1_members = labels[labels == 1].index.tolist()

        # Helper: rescale LS sides to 0.5 gross each
        def rescale_ls_sides(w: Dict[str, float]) -> Dict[str, float]:
            if not w:
                return {}
            long_sum = sum(max(0.0, x) for x in w.values())
            short_sum = sum(-min(0.0, x) for x in w.values())
            out: Dict[str, float] = {}
            for k, v in w.items():
                if v > 0 and long_sum > 0:
                    out[k] = v * (0.5 / long_sum)
                elif v < 0 and short_sum > 0:
                    out[k] = v * (0.5 / short_sum)
                else:
                    out[k] = 0.0
            return out

        # Build percentile ranks in [0,1], higher is better for long
        rnk = signal_for_rank.loc[valid_names].rank(method='first', ascending=True)
        n_names = float(len(rnk))
        if n_names > 1:
            perc = (rnk - 1.0) / (n_names - 1.0)
        else:
            perc = rnk * 0.0 + 0.5
        perc = perc.astype(float)

        # Function to build LS target weights from banded membership and beta-neutral sleeves
        def build_ls_target(prev_w: Dict[str, float]) -> Dict[str, float]:
            prev_long = {k for k, v in prev_w.items() if v > 0}
            prev_short = {k for k, v in prev_w.items() if v < 0}
            longs_keep = [t for t in prev_long if t in perc.index and perc.loc[t] >= cfg.band_keep]
            longs_add = perc[perc >= cfg.band_add].index.tolist()
            shorts_keep = [t for t in prev_short if t in perc.index and perc.loc[t] <= (1.0 - cfg.band_keep)]
            shorts_add = perc[perc <= (1.0 - cfg.band_add)].index.tolist()
            L = sorted(set(longs_keep).union(longs_add))
            S = sorted(set(shorts_keep).union(shorts_add))
            wL_base = {t: 0.5 / len(L) for t in L} if L else {}
            wS_base = {t: -0.5 / len(S) for t in S} if S else {}
            union = list(set(L) | set(S))
            if not union:
                return {}
            bet = betas_at_date(d_mend, union).reindex(union).fillna(1.0)
            B_L = float(sum(wL_base.get(t, 0.0) * bet.get(t, 1.0) for t in union))
            B_S = float(sum(wS_base.get(t, 0.0) * bet.get(t, 1.0) for t in union))
            denom = (B_L - B_S)
            if not np.isfinite(denom) or abs(denom) < 1e-9:
                cL = 1.0
                cS = 1.0
            else:
                cL = -2.0 * B_S / denom
                cS = 2.0 - cL
            w_t: Dict[str, float] = {}
            for t in L:
                w_t[t] = w_t.get(t, 0.0) + cL * wL_base[t]
            for t in S:
                w_t[t] = w_t.get(t, 0.0) + cS * wS_base[t]
            return rescale_ls_sides(w_t)

        # Function to build D10 long-only target from bands
        def build_d10_target(prev_w: Dict[str, float]) -> Dict[str, float]:
            prev_long = {k for k, v in prev_w.items() if v > 0}
            longs_keep = [t for t in prev_long if t in perc.index and perc.loc[t] >= cfg.band_keep]
            longs_add = perc[perc >= cfg.band_add].index.tolist()
            L = sorted(set(longs_keep).union(longs_add))
            if not L:
                return {}
            return {t: 1.0 / len(L) for t in L}

        if use_staggered:
            # Initialize all cohorts with first available weights for sensible startup using banded targets
            if not seeded:
                # Initial targets based on current percentiles without prior membership
                init_d10 = build_d10_target({})
                init_ls = build_ls_target({})
                for j in range(cfg.cohorts):
                    prev_w_d10_coh[j] = dict(init_d10)
                    prev_w_ls_coh[j] = dict(init_ls)
                seeded = True

            cohort_idx = idx_loop % cfg.cohorts
            # Track per-cohort LS target counts (long/short) to guide non-rebalance adds
            if 'ls_target_counts' not in locals():
                ls_target_counts_L = [0 for _ in range(cfg.cohorts)]
                ls_target_counts_S = [0 for _ in range(cfg.cohorts)]

            d10_rets = []
            ls_rets = []
            to_d10_list = []
            to_ls_list = []
            to_ls_reconst_list = []
            to_ls_reweight_list = []
            # Capture targets for manual execution/export (per month across cohorts)
            target_rows: List[Dict[str, object]] = []

            # Helper to get sector id
            def sector_of(t: str) -> str:
                return ticker_sector.get(t, "Unknown")
            # Monthly dispersion D and reversal regime R
            D_value = np.nan
            if len(d10_members) > 0 and len(d1_members) > 0:
                try:
                    med_top = float(signal_for_rank.loc[d10_members].median())
                    med_bot = float(signal_for_rank.loc[d1_members].median())
                    D_value = med_top - med_bot
                except Exception:
                    D_value = np.nan
            D_top = False
            if len(D_hist) >= 12 and np.isfinite(D_value):
                thr = float(np.nanpercentile(D_hist, 67))
                D_top = D_value >= thr
            if np.isfinite(D_value):
                D_hist.append(D_value)

            R_flag = 0
            try:
                p_cur = pd.Period(d_mend, freq='M')
                p_prev = p_cur - 1
                d_prev_mend = period_to_mend.get(p_prev)
                val_cur = float(bm_index.loc[:d_mend].iloc[-1]) if not bm_index.empty else np.nan
                val_prev = float(bm_index.loc[:d_prev_mend].iloc[-1]) if d_prev_mend is not None else np.nan
                r1m = (val_cur / val_prev - 1.0) if (np.isfinite(val_cur) and np.isfinite(val_prev) and val_prev != 0) else np.nan
                win = bm_index.loc[:d_mend].iloc[-126:] if len(bm_index.loc[:d_mend]) > 0 else pd.Series(dtype=float)
                if not win.empty:
                    peak = float(win.max())
                    cur = float(win.iloc[-1])
                    dd6 = cur / peak - 1.0 if peak else np.nan
                else:
                    dd6 = np.nan
                if np.isfinite(r1m) and np.isfinite(dd6) and (r1m > 0.08) and (dd6 <= -0.15):
                    R_flag = 1
            except Exception:
                R_flag = 0

            for j in range(cfg.cohorts):
                target_d10 = build_d10_target(prev_w_d10_coh[j])
                target_ls = build_ls_target(prev_w_ls_coh[j])

                if j == cohort_idx:
                    # Full rebalance to targets; update target counts for LS
                    new_d10 = target_d10
                    new_ls = target_ls
                    # Update LS target counts
                    nL = len([1 for k, v in new_ls.items() if v > 0])
                    nS = len([1 for k, v in new_ls.items() if v < 0])
                    ls_target_counts_L[j] = max(ls_target_counts_L[j], nL)
                    ls_target_counts_S[j] = max(ls_target_counts_S[j], nS)
                else:
                    # Non-rebalance month: banded minimal trading update
                    # D10 minimal: keep if >= keep band; add from >= add band to reach previous target size
                    # Freeze D10 on non-rebalance months to avoid unnecessary churn
                    new_d10 = dict(prev_w_d10_coh[j])

                    # LS minimal update per side
                    prev_ls = prev_w_ls_coh[j]
                    # Split sides
                    prev_long = {k: v for k, v in prev_ls.items() if v > 0 and k in perc.index}
                    prev_short = {k: -v for k, v in prev_ls.items() if v < 0 and k in perc.index}  # magnitudes

                    # Determine target counts from last rebalance or initialize
                    nL_tgt = ls_target_counts_L[j] if ls_target_counts_L[j] > 0 else max(1, len(prev_long))
                    nS_tgt = ls_target_counts_S[j] if ls_target_counts_S[j] > 0 else max(1, len(prev_short))

                    # Keep lists
                    keep_long = {k: v for k, v in prev_long.items() if perc.loc[k] >= cfg.band_keep}
                    keep_short = {k: v for k, v in prev_short.items() if perc.loc[k] <= (1.0 - cfg.band_keep)}
                    # Breachers and adaptive exit caps per side
                    breachers_long = [k for k in prev_long.keys() if k not in keep_long]
                    breachers_short = [k for k in prev_short.keys() if k not in keep_short]
                    nL_prev = max(1, len(prev_long))
                    nS_prev = max(1, len(prev_short))
                    if cfg.adaptive_exit_cap:
                        pL = len(breachers_long) / nL_prev if nL_prev else 0.0
                        pS = len(breachers_short) / nS_prev if nS_prev else 0.0
                        capL = np.clip(cfg.cap_min + 0.40 * pL + (0.05 if D_top else 0.0) + (0.05 if R_flag == 1 else 0.0), cfg.cap_min, cfg.cap_max)
                        capS = np.clip(cfg.cap_min + 0.40 * pS + (0.05 if D_top else 0.0) + (0.05 if R_flag == 1 else 0.0), cfg.cap_min, cfg.cap_max)
                    else:
                        capL = float(cfg.exit_cap_frac)
                        capS = float(cfg.exit_cap_frac)
                    max_exit_L = int(np.floor(capL * (ls_target_counts_L[j] if ls_target_counts_L[j] > 0 else nL_prev)))
                    max_exit_S = int(np.floor(capS * (ls_target_counts_S[j] if ls_target_counts_S[j] > 0 else nS_prev)))

                    # Severity-based prioritization and two-strike rule
                    def severity_long(ticker: str) -> float:
                        pc = float(perc.get(ticker, cfg.band_keep))
                        prev_pc = float(prev_perc_map.get(ticker, pc))
                        gap = max(0.0, (cfg.band_keep - pc) * 100.0)
                        speed = abs(pc - prev_pc) * 100.0
                        months = float(dur_long[j].get(ticker, 0))
                        stale = 1.0 + months / 8.0
                        return gap * speed * stale

                    def severity_short(ticker: str) -> float:
                        pc = float(perc.get(ticker, 1.0 - cfg.band_keep))
                        prev_pc = float(prev_perc_map.get(ticker, pc))
                        gap = max(0.0, (pc - (1.0 - cfg.band_keep)) * 100.0)
                        speed = abs(pc - prev_pc) * 100.0
                        months = float(dur_short[j].get(ticker, 0))
                        stale = 1.0 + months / 8.0
                        return gap * speed * stale

                    sevL = [(t, severity_long(t)) for t in breachers_long]
                    sevS = [(t, severity_short(t)) for t in breachers_short]
                    thrL = float(np.nanpercentile([s for _, s in sevL], 75)) if len(sevL) >= 4 else np.nan
                    thrS = float(np.nanpercentile([s for _, s in sevS], 75)) if len(sevS) >= 4 else np.nan
                    marg = 0.02
                    sevL.sort(key=lambda x: x[1], reverse=True)
                    exit_long = []
                    # Emergency exits for longs
                    for t, s in list(sevL):
                        pc = float(perc.get(t, cfg.band_keep))
                        prev_pc = float(prev_perc_map.get(t, pc))
                        hard = (pc < (cfg.band_keep - 0.10)) and (abs(pc - prev_pc) > 0.10) and (dur_long[j].get(t, 0) >= 6)
                        if hard:
                            exit_long.append(t)
                    # Budget per side (half of LS)
                    budget_L = cfg.ls_turnover_budget / 2.0 if cfg.use_turnover_budget else float('inf')
                    for t, s in sevL:
                        if t in exit_long:
                            continue
                        if len(exit_long) >= max_exit_L and not np.isfinite(budget_L):
                            break
                        pc = float(perc.get(t, cfg.band_keep))
                        marginal = (pc >= (cfg.band_keep - marg)) and (pc < cfg.band_keep)
                        take = False
                        if np.isfinite(thrL) and (s >= thrL):
                            take = True
                        else:
                            if marginal:
                                strikes_long[j][t] = strikes_long[j].get(t, 0) + 1
                                if strikes_long[j][t] >= 2:
                                    take = True
                            else:
                                take = True
                        if not take:
                            continue
                        if cfg.use_turnover_budget and np.isfinite(budget_L):
                            sim_keep = {k: v for k, v in keep_long.items() if (k not in exit_long) and (k != t)}
                            sim_add = []  # adds handled separately with micro budget
                            prev_side = prev_long
                            new_side = {k: v for k, v in build_side(sim_keep, sim_add, nL_tgt, sign=+1).items() if v > 0}
                            keys = set(prev_side.keys()) | set(new_side.keys())
                            to_side = sum(abs(new_side.get(k, 0.0) - prev_side.get(k, 0.0)) for k in keys)
                            if to_side <= budget_L:
                                exit_long.append(t)
                                budget_L -= to_side
                                continue
                            else:
                                continue
                        if len(exit_long) < max_exit_L:
                            exit_long.append(t)

                    sevS.sort(key=lambda x: x[1], reverse=True)
                    exit_short = []
                    # Emergency exits for shorts
                    for t, s in list(sevS):
                        pc = float(perc.get(t, 1.0 - cfg.band_keep))
                        prev_pc = float(prev_perc_map.get(t, pc))
                        hard = (pc > (1.0 - cfg.band_keep + 0.10)) and (abs(pc - prev_pc) > 0.10) and (dur_short[j].get(t, 0) >= 6)
                        if hard:
                            exit_short.append(t)
                    budget_S = cfg.ls_turnover_budget / 2.0 if cfg.use_turnover_budget else float('inf')
                    for t, s in sevS:
                        if t in exit_short:
                            continue
                        if len(exit_short) >= max_exit_S and not np.isfinite(budget_S):
                            break
                        pc = float(perc.get(t, 1.0 - cfg.band_keep))
                        marginal = (pc > (1.0 - cfg.band_keep)) and (pc <= (1.0 - cfg.band_keep + marg))
                        take = False
                        if np.isfinite(thrS) and (s >= thrS):
                            take = True
                        else:
                            if marginal:
                                strikes_short[j][t] = strikes_short[j].get(t, 0) + 1
                                if strikes_short[j][t] >= 2:
                                    take = True
                            else:
                                take = True
                        if not take:
                            continue
                        if cfg.use_turnover_budget and np.isfinite(budget_S):
                            sim_keep = {k: v for k, v in keep_short.items() if (k not in exit_short) and (k != t)}
                            sim_add = []
                            prev_side = prev_short
                            new_side_signed = build_side(sim_keep, sim_add, nS_tgt, sign=-1)
                            new_side = {k: -v for k, v in new_side_signed.items() if v < 0}
                            keys = set(prev_side.keys()) | set(new_side.keys())
                            to_side = sum(abs(new_side.get(k, 0.0) - prev_side.get(k, 0.0)) for k in keys)
                            if to_side <= budget_S:
                                exit_short.append(t)
                                budget_S -= to_side
                                continue
                            else:
                                continue
                        if len(exit_short) < max_exit_S:
                            exit_short.append(t)

                    # Apply exits; keep remaining breachers this month
                    keep_long.update({k: v for k, v in prev_long.items() if (k in breachers_long and k not in exit_long)})
                    keep_short.update({k: v for k, v in prev_short.items() if (k in breachers_short and k not in exit_short)})
                    # Micro add budget for super-signals under strong dispersion
                    add_long = []
                    add_short = []
                    if cfg.micro_add_frac > 0 and D_top:
                        nL_tgt_eff = nL_tgt if nL_tgt > 0 else max(1, len(prev_long))
                        nS_tgt_eff = nS_tgt if nS_tgt > 0 else max(1, len(prev_short))
                        micro_L = int(np.ceil(cfg.micro_add_frac * nL_tgt_eff))
                        micro_S = int(np.ceil(cfg.micro_add_frac * nS_tgt_eff))
                        if micro_L > 0:
                            super_long = [k for k in perc.index if (perc.loc[k] >= 0.98) and (k not in keep_long)]
                            super_long.sort(key=lambda t: (-perc.loc[t], t))
                            add_long = super_long[:micro_L]
                        if micro_S > 0:
                            super_short = [k for k in perc.index if (perc.loc[k] <= 0.02) and (k not in keep_short)]
                            super_short.sort(key=lambda t: (perc.loc[t], t))
                            add_short = super_short[:micro_S]

                    # Helper to build side weights with minimal trading, caps/floors; sector nudge optional
                    def build_side(prev_keep: Dict[str, float], add_list: List[str], n_target: int, sign: int, allow_sector_nudge: bool = False) -> Dict[str, float]:
                        # prev_keep values are magnitudes
                        gross = cfg.gross_per_side
                        # initial per-name target around equal-weight
                        eq = gross / max(n_target, 1)
                        cap = min(2.0 * eq, cfg.single_cap_pct)
                        floor = 0.25 * eq
                        w = dict(prev_keep)
                        # Limit adds per month to reduce churn
                        if n_target > 0 and len(add_list) > 0:
                            max_add = int(np.ceil(cfg.max_add_frac * n_target))
                            if max_add >= 0:
                                add_list = add_list[:max_add]
                        # Assign initial weights to adds at floor to minimize churn
                        for t in add_list:
                            w[t] = min(cap, floor)
                        # Pro-rata shave keepers to fund adds roughly
                        sum_keep = sum(prev_keep.values())
                        sum_add = sum(w.get(t, 0.0) for t in add_list)
                        if sum_keep > 0 and sum_add > 0:
                            shave = min(1.0, sum_add / sum_keep)
                            for k in prev_keep.keys():
                                w[k] = max(0.0, prev_keep[k] * (1.0 - shave))
                        # Only renormalize if outside tolerance band
                        s = sum(w.values())
                        if s > 0 and (abs(s - gross) > cfg.gross_tol):
                            scale = gross / s
                            for k in list(w.keys()):
                                w[k] *= scale
                        # Enforce caps/floors with simple redistribution (two passes)
                        for _ in range(2):
                            # Cap pass
                            excess = 0.0
                            under = []
                            for k, v in w.items():
                                if v > cap:
                                    excess += v - cap
                                    w[k] = cap
                                elif v < floor:
                                    under.append(k)
                            if excess > 0 and under:
                                alloc = sum(max(0.0, cap - w[k]) for k in under)
                                if alloc > 0:
                                    for k in under:
                                        room = max(0.0, cap - w[k])
                                        w[k] += excess * (room / alloc)
                            # Renorm to gross
                            s = sum(w.values())
                            if s > 0:
                                sc = gross / s
                                for k in w:
                                    w[k] *= sc
                        # Sector drift nudge (light) only if allowed
                        if allow_sector_nudge:
                            sectors = {}
                            for k, v in w.items():
                                sec = sector_of(k)
                                sectors[sec] = sectors.get(sec, 0.0) + v
                            if sectors:
                                m = len(sectors)
                                tgt_sec = {sec: gross / m for sec in sectors.keys()}
                                # Apply small nudge (25% of deviation beyond tol)
                                over = {sec: max(0.0, (sectors[sec] - tgt_sec[sec]) - cfg.sector_tol) for sec in sectors}
                                underw = {sec: max(0.0, (tgt_sec[sec] - sectors[sec]) - cfg.sector_tol) for sec in sectors}
                                tot_over = sum(over.values())
                                tot_under = sum(underw.values())
                                if tot_over > 0 and tot_under > 0:
                                    adj = 0.25 * min(tot_over, tot_under)
                                    # Reduce overweight names proportionally, add to underweight
                                    for k, v in list(w.items()):
                                        sec = sector_of(k)
                                        if over.get(sec, 0.0) > 0 and sectors[sec] > 0:
                                            w[k] -= adj * (v / sectors[sec])
                                    # Recompute sectors after reduction
                                    sectors2 = {}
                                    for k, v in w.items():
                                        sec = sector_of(k)
                                        sectors2[sec] = sectors2.get(sec, 0.0) + v
                                    for k, v in list(w.items()):
                                        sec = sector_of(k)
                                        if underw.get(sec, 0.0) > 0 and sectors2[sec] > 0:
                                            w[k] += adj * (v / sectors2[sec])
                                    # Renorm
                                    s = sum(max(0.0, vv) for vv in w.values())
                                    if s > 0:
                                        sc = gross / s
                                        for kk in w:
                                            w[kk] = max(0.0, w[kk] * sc)
                        # Emergency re-equalize if concentration too high
                        shares = [v / gross for v in w.values() if v > 0]
                        effN = 1.0 / sum((x * x for x in shares)) if shares else 0.0
                        if (effN < cfg.effn_ratio * max(1, n_target)) or (len([1 for v in w.values() if v <= floor]) > cfg.zero_frac * max(1, n_target)):
                            cnt = len(w)
                            if cnt > 0:
                                eqw = gross / cnt
                                w = {k: eqw for k in w.keys()}
                        # Apply sign for LS
                        if sign < 0:
                            return {k: -v for k, v in w.items() if v > 0}
                        else:
                            return {k: v for k, v in w.items() if v > 0}

                    # Sector nudge only on rebalance months; here it's non-rebalance
                    new_long = build_side(keep_long, add_long, nL_tgt, sign=+1, allow_sector_nudge=False)
                    new_short = build_side(keep_short, add_short, nS_tgt, sign=-1, allow_sector_nudge=False)
                    # Merge sides
                    new_ls = dict(new_long)
                    for k, v in new_short.items():
                        new_ls[k] = v

                # Capture target changes for export (LS and D10) BEFORE state update
                try:
                    prev_ls_w = dict(prev_w_ls_coh[j])
                    prev_d10_w = dict(prev_w_d10_coh[j])
                    def side_of(w: float) -> Optional[str]:
                        if w > 0:
                            return 'L'
                        if w < 0:
                            return 'S'
                        return None
                    px_open = start_open  # Series at d_trade
                    # LS book rows (union of prev and new names)
                    ls_names = set(prev_ls_w.keys()) | set(new_ls.keys())
                    for t in sorted(ls_names):
                        pw = float(prev_ls_w.get(t, 0.0))
                        nw = float(new_ls.get(t, 0.0))
                        prev_side = side_of(pw)
                        new_side = side_of(nw)
                        if new_side is None and prev_side is None:
                            continue
                        if prev_side is None and new_side is not None:
                            rationale = 'add'
                        elif prev_side is not None and new_side is None:
                            rationale = 'exit'
                        elif prev_side == new_side:
                            rationale = 'keep' if abs(nw - pw) <= 1e-9 else 'reweight'
                        else:
                            rationale = 'reweight'
                        dur = None
                        strikes = None
                        if new_side == 'L':
                            dur = int(dur_long[j].get(t, 0)) + (1 if nw != 0 else 0)
                            strikes = int(strikes_long[j].get(t, 0))
                        elif new_side == 'S':
                            dur = int(dur_short[j].get(t, 0)) + (1 if nw != 0 else 0)
                            strikes = int(strikes_short[j].get(t, 0))
                        else:
                            if prev_side == 'L':
                                dur = int(dur_long[j].get(t, 0))
                                strikes = int(strikes_long[j].get(t, 0))
                            elif prev_side == 'S':
                                dur = int(dur_short[j].get(t, 0))
                                strikes = int(strikes_short[j].get(t, 0))
                        target_rows.append({
                            'month_end': d_mend,
                            'trade_date': d_trade,
                            'cohort_id': j,
                            'book': 'LS',
                            'ticker': t,
                            'side': new_side if new_side is not None else (prev_side if prev_side is not None else ''),
                            'prev_weight': pw,
                            'target_weight': nw,
                            'delta_weight': nw - pw,
                            'rationale': rationale,
                            'sector': sector_of(t),
                            'duration_months': dur if dur is not None else 0,
                            'strikes': strikes if strikes is not None else 0,
                            'px_ref_open': float(px_open.get(t, np.nan)) if t in px_open.index else np.nan,
                        })
                    # D10 book rows (long-only)
                    d10_names = set(prev_d10_w.keys()) | set(new_d10.keys())
                    for t in sorted(d10_names):
                        pw = float(prev_d10_w.get(t, 0.0))
                        nw = float(new_d10.get(t, 0.0))
                        prev_in = pw > 0
                        new_in = nw > 0
                        if (not prev_in) and (not new_in):
                            continue
                        if (not prev_in) and new_in:
                            rationale = 'add'
                        elif prev_in and (not new_in):
                            rationale = 'exit'
                        else:
                            rationale = 'keep' if abs(nw - pw) <= 1e-9 else 'reweight'
                        dur = int(dur_long[j].get(t, 0)) + (1 if new_in else 0)
                        target_rows.append({
                            'month_end': d_mend,
                            'trade_date': d_trade,
                            'cohort_id': j,
                            'book': 'D10',
                            'ticker': t,
                            'side': 'L',
                            'prev_weight': pw,
                            'target_weight': nw,
                            'delta_weight': nw - pw,
                            'rationale': rationale,
                            'sector': sector_of(t),
                            'duration_months': dur,
                            'strikes': int(strikes_long[j].get(t, 0)),
                            'px_ref_open': float(px_open.get(t, np.nan)) if t in px_open.index else np.nan,
                        })
                except Exception:
                    # Targets export must not affect backtest; swallow errors
                    pass

                # Per-cohort turnover
                to_d10 = compute_turnover(prev_w_d10_coh[j], new_d10)
                # LS turnover decomposition
                prev_ls = prev_w_ls_coh[j]
                to_ls_total = compute_turnover(prev_ls, new_ls)
                # Reconstitution component
                prev_names = set(prev_ls.keys())
                new_names = set(new_ls.keys())
                exited = prev_names - new_names
                added = new_names - prev_names
                to_reconst = sum(abs(prev_ls[k]) for k in exited) + sum(abs(new_ls[k]) for k in added)
                to_reweight = to_ls_total - to_reconst
                to_d10_list.append(to_d10)
                to_ls_list.append(to_ls_total)
                to_ls_reconst_list.append(to_reconst)
                to_ls_reweight_list.append(to_reweight)

                # Update state
                prev_w_d10_coh[j] = new_d10
                prev_w_ls_coh[j] = new_ls
                # Update durations and strike counters per side
                names_long = {k for k, v in new_ls.items() if v > 0}
                names_short = {k for k, v in new_ls.items() if v < 0}
                # Increment for those present; reset for exited
                dur_long[j] = {k: dur_long[j].get(k, 0) + 1 for k in names_long}
                dur_short[j] = {k: dur_short[j].get(k, 0) + 1 for k in names_short}
                for k in list(dur_long[j].keys()):
                    if k not in names_long:
                        dur_long[j].pop(k, None)
                        strikes_long[j].pop(k, None)
                for k in list(dur_short[j].keys()):
                    if k not in names_short:
                        dur_short[j].pop(k, None)
                        strikes_short[j].pop(k, None)


                # Returns for this cohort using current weights
                r_d10 = ew_long_return(new_d10, stock_ret)
                r_ls = None
                # Use direct weighted sum; weights already scaled per side
                if new_ls:
                    names = [k for k in new_ls if k in stock_ret.index and pd.notna(stock_ret[k])]
                    r_ls = float(sum(new_ls[k] * float(stock_ret[k]) for k in names)) if names else None
                if r_d10 is not None:
                    d10_rets.append(r_d10)
                if r_ls is not None:
                    ls_rets.append(r_ls)

            d10_comp = float(np.mean(d10_rets)) if d10_rets else np.nan
            ls_comp = float(np.mean(ls_rets)) if ls_rets else np.nan
            # Composite turnover = average of cohort turnovers
            rows.append({
                "month_end": d_mend,
                "trade_date": d_trade,
                "next_trade_date": d_next_trade,
                "D10": d10_comp,
                "LS": ls_comp,
                "turnover_D10": float(np.mean(to_d10_list)) if to_d10_list else np.nan,
                "turnover_LS": float(np.mean(to_ls_list)) if to_ls_list else np.nan,
                "turnover_LS_reconst": float(np.mean(to_ls_reconst_list)) if to_ls_reconst_list else np.nan,
                "turnover_LS_reweight": float(np.mean(to_ls_reweight_list)) if to_ls_reweight_list else np.nan,
            })
            # Write targets export (per month) and optional orders export
            try:
                if target_rows:
                    period_label = pd.Period(d_mend, freq='M').strftime('%Y-%m')
                    df_targets = pd.DataFrame(target_rows)
                    # Apply vol-target scale to LS weights for live executables (orders/targets)
                    # Only when vol targeting is enabled via CLI/config
                    # Estimate VT using rolling vol of historical LS composite returns up to t-1
                    try:
                        if getattr(cfg, 'apply_vol_target', False):
                            from math import sqrt
                            # Build historical LS returns from previously computed rows (exclude current period)
                            ls_hist_all = [float(r.get("LS", np.nan)) for r in rows if (r.get("LS", None) is not None)]
                            ls_hist_curr = ls_hist_all[:-1] if len(ls_hist_all) >= 1 else []
                            ls_hist_prev = ls_hist_all[:-2] if len(ls_hist_all) >= 2 else []
                            def _vt_from(hist_list: List[float]) -> float:
                                if not hist_list:
                                    return 1.0
                                s = pd.Series(hist_list, dtype=float).dropna()
                                if s.empty:
                                    return 1.0
                                win = int(getattr(cfg, 'vol_window_months', 36) or 36)
                                minp = int(getattr(cfg, 'vol_min_months', 12) or 12)
                                s_win = s.iloc[-win:] if len(s) > win else s
                                if len(s_win) < max(2, minp):
                                    return 1.0
                                sigma_ann = float(s_win.std(ddof=1) * sqrt(12.0))
                                if not np.isfinite(sigma_ann) or sigma_ann <= 0:
                                    return 1.0
                                vt = float(getattr(cfg, 'vol_target_ann', 0.10) or 0.10) / sigma_ann
                                return vt if np.isfinite(vt) and vt > 0 else 1.0
                            vt_curr = _vt_from(ls_hist_curr)
                            vt_prev = _vt_from(ls_hist_prev)
                            # Apply to LS rows only
                            if 'book' in df_targets.columns:
                                mask_ls = df_targets['book'] == 'LS'
                                if mask_ls.any():
                                    pw = df_targets.loc[mask_ls, 'prev_weight'].astype(float)
                                    tw = df_targets.loc[mask_ls, 'target_weight'].astype(float)
                                    df_targets.loc[mask_ls, 'prev_weight'] = pw * vt_prev
                                    df_targets.loc[mask_ls, 'target_weight'] = tw * vt_curr
                                    df_targets.loc[mask_ls, 'delta_weight'] = df_targets.loc[mask_ls, 'target_weight'] - df_targets.loc[mask_ls, 'prev_weight']
                                    # Annotate for audit
                                    df_targets.loc[mask_ls, 'vt_prev'] = vt_prev
                                    df_targets.loc[mask_ls, 'vt_curr'] = vt_curr
                    except Exception:
                        pass
                    df_targets.sort_values(["book", "cohort_id", "side", "ticker"], inplace=True)
                    df_targets.to_csv(os.path.join(out_dir, f"targets_{period_label}.csv"), index=False)
                    # Optional orders export if live capital is provided
                    if getattr(cfg, 'live_capital', None) is not None:
                        live_cap = float(cfg.live_capital)  # type: ignore
                        lot = int(getattr(cfg, 'lot_size', 1))
                        ord_rows = []
                        for _, r in df_targets.iterrows():
                            dw = float(r.get('delta_weight', 0.0))
                            px = float(r.get('px_ref_open', float('nan')))
                            if not np.isfinite(px) or dw == 0.0:
                                continue
                            shares = dw * live_cap / px
                            # Round to lot size
                            if lot and lot > 1:
                                shares = np.sign(shares) * (abs(shares) // lot) * lot
                            else:
                                shares = round(shares)
                            if shares == 0:
                                continue
                            action = 'BUY' if shares > 0 else 'SELL'
                            # OPEN/CLOSE labels
                            pw = float(r.get('prev_weight', 0.0))
                            nw = float(r.get('target_weight', 0.0))
                            openclose = 'OPEN' if (pw == 0.0 and nw != 0.0) else ('CLOSE' if (pw != 0.0 and nw == 0.0) else '')
                            ord_rows.append({
                                'month_end': r['month_end'],
                                'trade_date': r['trade_date'],
                                'cohort_id': int(r['cohort_id']),
                                'book': r['book'],
                                'ticker': r['ticker'],
                                'side': r['side'],
                                'action': action if not openclose else openclose,
                                'w_delta': dw,
                                'px_ref_open': px,
                                'shares': int(shares),
                                'est_notional': float(px * shares),
                                'rationale': r.get('rationale', ''),
                            })
                        if ord_rows:
                            pd.DataFrame(ord_rows).sort_values(["book", "cohort_id", "ticker"]).to_csv(
                                os.path.join(out_dir, f"orders_{period_label}.csv"), index=False
                            )

                            # Aggregate to account-level targets (net across cohorts)
                            try:
                                # Sum weights per (book, ticker) then scale by number of cohorts
                                agg = (df_targets
                                       .groupby(["book", "ticker"], as_index=False)
                                       .agg(prev_weight=("prev_weight", "sum"),
                                            target_weight=("target_weight", "sum"),
                                            delta_weight=("delta_weight", "sum"),
                                            px_ref_open=("px_ref_open", "first"),
                                            sector=("sector", "first")))
                                n_coh = int(getattr(cfg, 'cohorts', 1) or 1)
                                if n_coh > 1:
                                    agg['prev_weight'] = agg['prev_weight'] / n_coh
                                    agg['target_weight'] = agg['target_weight'] / n_coh
                                    agg['delta_weight'] = agg['delta_weight'] / n_coh

                                # Derive side from aggregated target weight (D10 is always long)
                                def _agg_side(row):
                                    if row["book"] == "D10":
                                        return "L"
                                    tw = float(row["target_weight"])
                                    if tw > 0:
                                        return "L"
                                    if tw < 0:
                                        return "S"
                                    return ""

                                agg["side"] = agg.apply(_agg_side, axis=1)
                                agg.insert(0, "trade_date", d_trade)
                                agg.insert(0, "month_end", d_mend)
                                # Order columns for readability
                                cols = [
                                    "month_end", "trade_date", "book", "ticker", "side",
                                    "prev_weight", "target_weight", "delta_weight",
                                    "px_ref_open", "sector"
                                ]
                                df_account_targets = agg.loc[:, cols].sort_values(["book", "ticker"])  # type: ignore
                                df_account_targets.to_csv(os.path.join(out_dir, f"account_targets_{period_label}.csv"), index=False)

                                # Generate netted live orders (one per symbol/book) directly from account-level delta weights
                                ord_agg_rows = []
                                for _, r in df_account_targets.iterrows():
                                    dw = float(r.get('delta_weight', 0.0))
                                    px = float(r.get('px_ref_open', float('nan')))
                                    if not np.isfinite(px) or dw == 0.0:
                                        continue
                                    shares = dw * live_cap / px
                                    # Round to lot size
                                    if lot and lot > 1:
                                        shares = np.sign(shares) * (abs(shares) // lot) * lot
                                    else:
                                        shares = round(shares)
                                    if shares == 0:
                                        continue
                                    action = 'BUY' if shares > 0 else 'SELL'
                                    ord_agg_rows.append({
                                        'month_end': r['month_end'],
                                        'trade_date': r['trade_date'],
                                        'book': r['book'],
                                        'ticker': r['ticker'],
                                        'side': r['side'],
                                        'action': action,
                                        'w_delta': dw,
                                        'px_ref_open': px,
                                        'shares': int(shares),
                                        'est_notional': float(px * shares),
                                    })
                                if ord_agg_rows:
                                    df_orders_live = pd.DataFrame(ord_agg_rows).sort_values(["book", "ticker"])  # type: ignore
                                    df_orders_live.to_csv(os.path.join(out_dir, f"orders_live_{period_label}.csv"), index=False)

                                # Optional: Cold-start exports for bootstrapping new accounts
                                if getattr(cfg, 'cold_start', False):
                                    try:
                                        # Determine ramp fraction default = 1 / cohorts if not provided
                                        rf = getattr(cfg, 'ramp_frac', None)
                                        if rf is None:
                                            rf = 1.0 / float(max(1, int(getattr(cfg, 'cohorts', 1) or 1)))
                                        # Clamp to [0, 1]
                                        try:
                                            rf = float(rf)
                                        except Exception:
                                            rf = 1.0 / float(max(1, int(getattr(cfg, 'cohorts', 1) or 1)))
                                        if not np.isfinite(rf):
                                            rf = 1.0 / float(max(1, int(getattr(cfg, 'cohorts', 1) or 1)))
                                        rf = max(0.0, min(1.0, rf))

                                        cold = agg.copy()
                                        # In cold-start, treat prev=0 and delta = ramp_frac * target
                                        cold['prev_weight'] = 0.0
                                        cold['delta_weight'] = rf * cold['target_weight']
                                        # Derive side as above
                                        cold['side'] = cold.apply(_agg_side, axis=1)
                                        if 'trade_date' not in cold.columns:
                                            cold.insert(0, 'trade_date', d_trade)
                                        if 'month_end' not in cold.columns:
                                            cold.insert(0, 'month_end', d_mend)
                                        df_account_targets_cs = cold.loc[:, cols].sort_values(["book", "ticker"])  # type: ignore
                                        df_account_targets_cs.to_csv(os.path.join(out_dir, f"account_targets_coldstart_{period_label}.csv"), index=False)

                                        # Build cold-start live orders from cold-start delta weights
                                        ord_cs_rows = []
                                        for _, r in df_account_targets_cs.iterrows():
                                            dw = float(r.get('delta_weight', 0.0))
                                            px = float(r.get('px_ref_open', float('nan')))
                                            if not np.isfinite(px) or dw == 0.0:
                                                continue
                                            shares = dw * live_cap / px
                                            if lot and lot > 1:
                                                shares = np.sign(shares) * (abs(shares) // lot) * lot
                                            else:
                                                shares = round(shares)
                                            if shares == 0:
                                                continue
                                            action = 'BUY' if shares > 0 else 'SELL'
                                            ord_cs_rows.append({
                                                'month_end': r['month_end'],
                                                'trade_date': r['trade_date'],
                                                'book': r['book'],
                                                'ticker': r['ticker'],
                                                'side': r['side'],
                                                'action': action,
                                                'w_delta': dw,
                                                'px_ref_open': px,
                                                'shares': int(shares),
                                                'est_notional': float(px * shares),
                                            })
                                        if ord_cs_rows:
                                            df_orders_live_cs = pd.DataFrame(ord_cs_rows).sort_values(["book", "ticker"])  # type: ignore
                                            df_orders_live_cs.to_csv(os.path.join(out_dir, f"orders_live_coldstart_{period_label}.csv"), index=False)
                                    except Exception:
                                        # Cold-start is best-effort; do not interrupt run
                                        pass

                                # Create allocations plan mapping parent orders back to cohorts
                                # Parent quantities per (book, ticker)
                                parent_qty = {}
                                for r in ord_agg_rows:
                                    key = (r['book'], r['ticker'])
                                    parent_qty[key] = int(r['shares'])

                                # Child (cohort) desired quantities per (book, ticker, cohort)
                                child_qty = {}
                                for r in ord_rows:
                                    key = (r['book'], r['ticker'], int(r['cohort_id']))
                                    child_qty[key] = int(r['shares'])

                                # Build allocation ratios for cohorts aligned with parent side
                                alloc_rows = []
                                # Group cohort orders by (book, ticker)
                                from collections import defaultdict
                                by_bt = defaultdict(list)
                                for (book, ticker, coh), qty in child_qty.items():
                                    by_bt[(book, ticker)].append((coh, qty))

                                for (book, ticker), items in by_bt.items():
                                    pqty = int(parent_qty.get((book, ticker), 0))
                                    if pqty == 0:
                                        # No parent order -> ratios are 0; still record desired quantities
                                        for coh, qty in items:
                                            alloc_rows.append({
                                                'month_end': d_mend,
                                                'trade_date': d_trade,
                                                'book': book,
                                                'ticker': ticker,
                                                'parent_qty': 0,
                                                'cohort_id': int(coh),
                                                'cohort_desired_qty': int(qty),
                                                'alloc_ratio': 0.0,
                                            })
                                        continue
                                    sign = np.sign(pqty)
                                    same_sign_abs_sum = sum(abs(q) for _, q in items if np.sign(q) == sign)
                                    for coh, qty in items:
                                        if same_sign_abs_sum > 0 and np.sign(qty) == sign:
                                            ratio = abs(qty) / same_sign_abs_sum
                                        else:
                                            ratio = 0.0
                                        alloc_rows.append({
                                            'month_end': d_mend,
                                            'trade_date': d_trade,
                                            'book': book,
                                            'ticker': ticker,
                                            'parent_qty': int(pqty),
                                            'cohort_id': int(coh),
                                            'cohort_desired_qty': int(qty),
                                            'alloc_ratio': float(ratio),
                                        })
                                if alloc_rows:
                                    df_alloc = pd.DataFrame(alloc_rows).sort_values(["book", "ticker", "cohort_id"])  # type: ignore
                                    df_alloc.to_csv(os.path.join(out_dir, f"allocations_{period_label}.csv"), index=False)
                            except Exception:
                                # Keep aggregation/allocation best-effort; do not interrupt run
                                pass
                # Minimal state snapshots (optional)
                if getattr(cfg, 'write_state_snapshots', False):
                    state_dir = os.path.join(out_dir, "state")
                    os.makedirs(state_dir, exist_ok=True)
                    period_label = pd.Period(d_mend, freq='M').strftime('%Y-%m')
                    for j in range(cfg.cohorts):
                        snap = {
                            "month_end": str(pd.Timestamp(d_mend).date()),
                            "trade_date": str(pd.Timestamp(d_trade).date()),
                            "cohort_id": j,
                            "weights_ls": prev_w_ls_coh[j],
                            "weights_d10": prev_w_d10_coh[j],
                            "dur_long": dur_long[j],
                            "dur_short": dur_short[j],
                            "strikes_long": strikes_long[j],
                            "strikes_short": strikes_short[j],
                            "ls_target_counts": {"long": int(ls_target_counts_L[j]), "short": int(ls_target_counts_S[j])},
                        }
                        with open(os.path.join(state_dir, f"cohort_{j}_{period_label}.json"), "w", encoding="utf-8") as fh:
                            json.dump(snap, fh, ensure_ascii=False, indent=2)
            except Exception:
                # Never interrupt backtest on export failure
                pass
            # Store current percentiles for next month's speed calculation
            for tkr, val in perc.items():
                prev_perc_map[tkr] = float(val)
        else:
            # Non-staggered: original behavior
            # Decile equal-weight returns for reporting
            d_rets: Dict[int, float] = {}
            for k in deciles:
                members = labels[labels == k].index
                if len(members) == 0:
                    d_rets[k] = np.nan
                else:
                    d_rets[k] = float(stock_ret.loc[members].mean())

            # LS return via weighted sum of stock returns (all rebalanced now)
            ls_ret = float(sum(w_ls_new.get(t, 0.0) * stock_ret.get(t, np.nan) for t in w_ls_new.keys()))

            # Turnover using current target weights
            to_d10 = compute_turnover(prev_w_d10, w_d10_new)
            to_ls = compute_turnover(prev_w_ls, w_ls_new)
            prev_w_d10 = w_d10_new
            prev_w_ls = w_ls_new

            rows.append({
                "month_end": d_mend,
                "trade_date": d_trade,
                "next_trade_date": d_next_trade,
                **{f"D{k}": d_rets.get(k, np.nan) for k in deciles},
                "LS": ls_ret,
                "turnover_D10": to_d10,
                "turnover_LS": to_ls,
            })

    if not rows:
        raise RuntimeError("No portfolio months constructed (check data coverage).")

    pf = pd.DataFrame(rows).set_index("month_end").sort_index()

    # 6) CDI and benchmark (BOVA11) returns over the same periods
    print("Computing CDI and benchmark returns ...")
    cdi_index = read_cdi_series(cfg.cdi_path)
    # BOVA11 returns open-to-open across same trade periods
    # Try to locate BOVA11 from the same tables
    with sqlite3.connect(cfg.db_path) as con:
        q_bova = """
        SELECT dp.date as date, dp.open_price as open_price, dp.close_price as close_price, dp.adjusted_close as adj_close
        FROM daily_prices dp
        JOIN securities s ON s.security_id = dp.security_id
        WHERE s.ticker = ?
        ORDER BY dp.date ASC
        """
        df_bm = pd.read_sql_query(q_bova, con, params=[cfg.bova11_ticker], parse_dates=["date"])  # type: ignore
    if not df_bm.empty:
        bm_adj_open = df_bm["open_price"] * (df_bm["adj_close"] / df_bm["close_price"]).replace([np.inf, -np.inf], np.nan)
        bm = df_bm.assign(adj_open=bm_adj_open)[["date", "adj_open"]].dropna()
        bm = bm.set_index("date")["adj_open"].sort_index()
    else:
        bm = pd.Series(dtype=float)

    # Map period CDI and benchmark returns
    cdi_rets = []
    bm_rets = []
    for d_trade, d_next_trade in pf[["trade_date", "next_trade_date"]].itertuples(index=False):
        # CDI as index ratio over the holding period
        if d_trade in cdi_index.index and d_next_trade in cdi_index.index:
            cdi_r = float(cdi_index.loc[d_next_trade] / cdi_index.loc[d_trade] - 1.0)
        else:
            # try nearest previous available
            try:
                cdi_start = cdi_index.loc[:d_trade].iloc[-1]
                cdi_end = cdi_index.loc[:d_next_trade].iloc[-1]
                cdi_r = float(cdi_end / cdi_start - 1.0)
            except Exception:
                cdi_r = np.nan
        cdi_rets.append(cdi_r)

        # Benchmark
        if not bm.empty:
            try:
                b_r = float(bm.loc[d_next_trade] / bm.loc[d_trade] - 1.0)
            except Exception:
                # nearest previous available
                try:
                    b_start = bm.loc[:d_trade].iloc[-1]
                    b_end = bm.loc[:d_next_trade].iloc[-1]
                    b_r = float(b_end / b_start - 1.0)
                except Exception:
                    b_r = np.nan
        else:
            b_r = np.nan
        bm_rets.append(b_r)

    pf["CDI"] = cdi_rets
    pf["BOVA11"] = bm_rets

    # Composite-level beta overlay vs BOVA11 (EWMA up to t-1, shrinkage, band; then VT uses hedged series)
    if cfg.beta_overlay:
        ls_raw = pf["LS"].astype(float)
        bm_m = pf["BOVA11"].astype(float)
        # Approximate 60d halflife in months (~21 trading days/month)
        hl_months = max(1, int(round(cfg.overlay_halflife_days / 21.0)))
        look = max(6, int(cfg.overlay_lookback_months))
        betas = []
        h_list = []
        h_prev = 0.0
        idx = pf.index
        for i in range(len(idx)):
            start = max(0, i - look)
            y = ls_raw.iloc[start:i]
            x = bm_m.iloc[start:i]
            df = pd.concat([y, x], axis=1).dropna()
            if len(df) < 6:
                betas.append(0.0)
                h_list.append(h_prev)
                continue
            # EWMA weights on monthly history as a proxy for daily 60d HL
            n = len(df)
            ages = np.arange(n)[::-1]  # 0 is most recent
            decay = np.log(2.0) / float(hl_months)
            w_np = np.exp(-decay * ages)
            w = pd.Series(w_np, index=df.index)
            w = w / w.sum()
            x_c = df.iloc[:, 1] - float((w * df.iloc[:, 1]).sum())
            y_c = df.iloc[:, 0] - float((w * df.iloc[:, 0]).sum())
            varx = float((w * (x_c ** 2)).sum())
            cov = float((w * (x_c * y_c)).sum())
            beta = cov / varx if varx > 0 else 0.0
            # Shrink toward 0
            beta_hat = (1.0 - cfg.overlay_shrink) * beta
            # Hedge ratio with band; carry previous if small
            if abs(beta_hat) > cfg.overlay_band:
                h_t = -beta_hat
            else:
                h_t = h_prev
            betas.append(beta_hat)
            h_list.append(h_t)
            h_prev = h_t
        beta_series = pd.Series(betas, index=idx)
        hedge_ratio = pd.Series(h_list, index=idx)
        pf["LS_pre_overlay"] = ls_raw
        pf["beta_est"] = beta_series
        pf["hedge_ratio"] = hedge_ratio
        # Apply hedge contemporaneously: r'_t = r_LS + h_t * r_IBOV
        pf["LS"] = ls_raw + hedge_ratio * bm_m

    # Vol targeting on hedged series (if requested). Uses rolling vol of hedged LS.
    # We also compute and store an LS_vt comparison series even if not applied to LS.
    ls_for_vt = pf["LS"].astype(float).copy() if "LS" in pf.columns else pd.Series(dtype=float)
    if not ls_for_vt.empty:
        # Rolling annualized realized vol of LS (months -> ann)
        vt_win = max(6, int(getattr(cfg, 'vol_window_months', 36) or 36))
        vt_min = max(6, int(getattr(cfg, 'vol_min_months', 12) or 12))
        ls_roll_vol_ann = ls_for_vt.rolling(window=vt_win, min_periods=vt_min).std() * np.sqrt(12.0)
        tgt = float(getattr(cfg, 'vol_target_ann', 0.10) or 0.10)
        scale_vt = tgt / ls_roll_vol_ann
        scale_vt = scale_vt.replace([np.inf, -np.inf], np.nan).fillna(1.0)
        pf["vt_scale"] = scale_vt
        pf["LS_vt"] = ls_for_vt * scale_vt
        if getattr(cfg, 'apply_vol_target', False):
            pf["LS_pre_vt"] = ls_for_vt
            pf["LS"] = pf["LS_vt"].astype(float)

    # 7) Performance and inference
    print("Computing performance statistics ...")
    results = {}
    # LS: raw returns + CAPM alpha/IR vs IBOV (BOVA11)
    if "LS" in pf.columns:
        r_ls = pf["LS"].astype(float)
        hit_rate = float((r_ls > 0).mean())
        mdd = max_drawdown(r_ls)
        mu, tstat = newey_west_tstat(r_ls, lags=cfg.nw_lags)
        capm = capm_alpha_ir(r_ls, pf["BOVA11"], lags=cfg.nw_lags)
        results["LS"] = {
            "mean_monthly": mu,
            "tstat_monthly": tstat,
            "hit_rate": hit_rate,
            "max_drawdown": mdd,
            "capm_alpha": capm["alpha"],
            "capm_alpha_t": capm["alpha_t"],
            "capm_beta": capm["beta"],
            "capm_ir_ann": capm["ir"],
            "capm_te_ann": capm["te_ann"],
        }

    # LS_vt: comparison metrics for vol-targeted series (not net of costs here)
    if "LS_vt" in pf.columns:
        r_lsvt = pf["LS_vt"].astype(float)
        hit_rate = float((r_lsvt > 0).mean())
        mdd = max_drawdown(r_lsvt)
        mu, tstat = newey_west_tstat(r_lsvt, lags=cfg.nw_lags)
        capm = capm_alpha_ir(r_lsvt, pf["BOVA11"], lags=cfg.nw_lags)
        results["LS_vt"] = {
            "mean_monthly": mu,
            "tstat_monthly": tstat,
            "hit_rate": hit_rate,
            "max_drawdown": mdd,
            "capm_alpha": capm["alpha"],
            "capm_alpha_t": capm["alpha_t"],
            "capm_beta": capm["beta"],
            "capm_ir_ann": capm["ir"],
            "capm_te_ann": capm["te_ann"],
        }
    # LS_pre_vt: baseline hedged series when VT applied (for comparison)
    if "LS_pre_vt" in pf.columns:
        r_lspre = pf["LS_pre_vt"].astype(float)
        hit_rate = float((r_lspre > 0).mean())
        mdd = max_drawdown(r_lspre)
        mu, tstat = newey_west_tstat(r_lspre, lags=cfg.nw_lags)
        capm = capm_alpha_ir(r_lspre, pf["BOVA11"], lags=cfg.nw_lags)
        results["LS_pre_vt"] = {
            "mean_monthly": mu,
            "tstat_monthly": tstat,
            "hit_rate": hit_rate,
            "max_drawdown": mdd,
            "capm_alpha": capm["alpha"],
            "capm_alpha_t": capm["alpha_t"],
            "capm_beta": capm["beta"],
            "capm_ir_ann": capm["ir"],
            "capm_te_ann": capm["te_ann"],
        }

    # D10/D1: keep existing excess-to-CDI Sharpe for reference
    for label in ["D10", "D1"]:
        if label not in pf.columns:
            continue
        r = pf[label].astype(float)
        excess = r - pf["CDI"].astype(float)
        ann_sharpe = float(np.sqrt(12.0) * excess.mean() / excess.std(ddof=1)) if excess.std(ddof=1) > 0 else np.nan
        hit_rate = float((r > 0).mean())
        mdd = max_drawdown(r)
        mu, tstat = newey_west_tstat(r, lags=cfg.nw_lags)
        mu_ex, tstat_ex = newey_west_tstat(excess, lags=cfg.nw_lags)
        results[label] = {
            "mean_monthly": mu,
            "tstat_monthly": tstat,
            "mean_monthly_excess": float(excess.mean()),
            "tstat_monthly_excess": tstat_ex,
            "ann_sharpe_excess": ann_sharpe,
            "hit_rate": hit_rate,
            "max_drawdown": mdd,
        }

    # Alpha/IR for LS vs benchmark as primary yardstick
    alpha_stats = capm_alpha_ir(pf["LS"], pf["BOVA11"], lags=cfg.nw_lags) if "LS" in pf.columns else {"alpha": np.nan, "alpha_t": np.nan, "beta": np.nan, "ir": np.nan}
    results["LS_alpha_vs_BOVA11"] = {
        "alpha_bps_per_month": alpha_stats.get("alpha", np.nan) * 1e4,  # bps
        "alpha_tstat": alpha_stats.get("alpha_t", np.nan),
        "beta": alpha_stats.get("beta", np.nan),
        "ir_ann": alpha_stats.get("ir", np.nan),
    }
    # Also report for LS_vt if present
    if "LS_vt" in pf.columns:
        alpha_stats_vt = capm_alpha_ir(pf["LS_vt"], pf["BOVA11"], lags=cfg.nw_lags)
        results["LS_vt_alpha_vs_BOVA11"] = {
            "alpha_bps_per_month": alpha_stats_vt.get("alpha", np.nan) * 1e4,
            "alpha_tstat": alpha_stats_vt.get("alpha_t", np.nan),
            "beta": alpha_stats_vt.get("beta", np.nan),
            "ir_ann": alpha_stats_vt.get("ir", np.nan),
        }
    if "LS_pre_vt" in pf.columns:
        alpha_stats_pre = capm_alpha_ir(pf["LS_pre_vt"], pf["BOVA11"], lags=cfg.nw_lags)
        results["LS_pre_vt_alpha_vs_BOVA11"] = {
            "alpha_bps_per_month": alpha_stats_pre.get("alpha", np.nan) * 1e4,
            "alpha_tstat": alpha_stats_pre.get("alpha_t", np.nan),
            "beta": alpha_stats_pre.get("beta", np.nan),
            "ir_ann": alpha_stats_pre.get("ir", np.nan),
        }

    # Stability by subperiods
    def subperiod_mask(start: str, end: str) -> pd.Series:
        return (pf.index >= pd.to_datetime(start)) & (pf.index <= pd.to_datetime(end))

    subperiods = {
        "2003-01-01_to_2010-12-31": ("2003-01-01", "2010-12-31"),
        "2011-01-01_to_2018-12-31": ("2011-01-01", "2018-12-31"),
        "2019-01-01_to_2025-12-31": ("2019-01-01", "2025-12-31"),
    }
    stability = {}
    for name, (a, b) in subperiods.items():
        m = subperiod_mask(a, b)
        sub = pf.loc[m]
        if sub.empty:
            stability[name] = {"LS_mean": np.nan, "LS_t": np.nan, "D10_mean": np.nan, "D10_t": np.nan}
        else:
            mu_ls, t_ls = newey_west_tstat(sub["LS"], lags=cfg.nw_lags)
            mu_d10, t_d10 = newey_west_tstat(sub["D10"], lags=cfg.nw_lags)
            stability[name] = {"LS_mean": mu_ls, "LS_t": t_ls, "D10_mean": mu_d10, "D10_t": t_d10}

    # 8) Cost grid: apply 20/50/100 bps per rebalance (half per side); report Sharpe, alpha, turnover
    print("Applying transaction cost grid and computing net metrics ...")
    cost_bps_grid = [20, 50, 100]
    grid_results = []
    # Average turnover per month
    avg_to_d10 = float(pf["turnover_D10"].mean()) if "turnover_D10" in pf else np.nan
    avg_to_ls = float(pf["turnover_LS"].mean()) if "turnover_LS" in pf else np.nan

    # For vol targeting: compute rolling 36-month realized vol of gross LS
    ls_gross = pf["LS"].astype(float)
    ls_roll_vol_ann = ls_gross.rolling(window=36, min_periods=12).std() * np.sqrt(12.0)
    target_ann_vol = 0.10

    for bps in cost_bps_grid:
        per_side_rate = (bps / 10000.0) / 2.0  # half per side
        # net returns = gross - cost_rate * turnover
        d10_net = pf["D10"].astype(float) - per_side_rate * pf["turnover_D10"].astype(float)
        ls_net = pf["LS"].astype(float) - per_side_rate * pf["turnover_LS"].astype(float)
        # Sharpe vs CDI (excess)
        d10_ex = d10_net - pf["CDI"].astype(float)
        ls_ex = ls_net - pf["CDI"].astype(float)
        d10_sharpe = float(np.sqrt(12.0) * d10_ex.mean() / d10_ex.std(ddof=1)) if d10_ex.std(ddof=1) > 0 else np.nan
        # HAC t-stats on mean and on alpha/IR vs BOVA11
        d10_mu, d10_t = newey_west_tstat(d10_net, lags=cfg.nw_lags)
        ls_mu, ls_t = newey_west_tstat(ls_net, lags=cfg.nw_lags)
        d10_alpha = ols_alpha_newey_west(d10_net, pf["BOVA11"], lags=cfg.nw_lags)
        ls_alpha = capm_alpha_ir(ls_net, pf["BOVA11"], lags=cfg.nw_lags)

        # Vol-target LS at 10% using 36-month rolling vol of gross LS
        scale = target_ann_vol / ls_roll_vol_ann
        scale = scale.replace([np.inf, -np.inf], np.nan).fillna(1.0)
        ls_net_vt = ls_net * scale
        ls_ex_vt = ls_net_vt - pf["CDI"].astype(float)
        ls_mu_vt, ls_t_vt = newey_west_tstat(ls_net_vt, lags=cfg.nw_lags)
        ls_alpha_vt = capm_alpha_ir(ls_net_vt, pf["BOVA11"], lags=cfg.nw_lags)
        ls_mdd_vt = max_drawdown(ls_net_vt)
        ls_hit_vt = float((ls_net_vt > 0).mean())

        grid_results.append({
            "bps": bps,
            "D10_sharpe": d10_sharpe,
            "D10_mean": d10_mu,
            "D10_t": d10_t,
            "D10_alpha_bps": d10_alpha["alpha"] * 1e4,
            "D10_alpha_t": d10_alpha["alpha_t"],
            "LS_mean": ls_mu,
            "LS_t": ls_t,
            "LS_alpha_bps": ls_alpha["alpha"] * 1e4,
            "LS_alpha_t": ls_alpha["alpha_t"],
            "LS_ir": ls_alpha["ir"],
            "LS_mean_vt": ls_mu_vt,
            "LS_t_vt": ls_t_vt,
            "LS_alpha_bps_vt": ls_alpha_vt["alpha"] * 1e4,
            "LS_alpha_t_vt": ls_alpha_vt["alpha_t"],
            "LS_ir_vt": ls_alpha_vt["ir"],
            "LS_maxDD_vt": ls_mdd_vt,
            "LS_hit_vt": ls_hit_vt,
            "avg_turnover_D10": avg_to_d10,
            "avg_turnover_LS": avg_to_ls,
        })

    # Kill rule (early): if LS alpha IR <= 0 after 50 bps, or alpha insignificant (|t|<2) in any long subperiod
    kill_triggered = False
    ls_50 = next((g for g in grid_results if g["bps"] == 50), None)
    if ls_50 is not None and (ls_50.get("LS_ir") is not None):
        if (not np.isfinite(ls_50["LS_ir"])) or (ls_50["LS_ir"] <= cfg.kill_net_sharpe_min):
            kill_triggered = True

    # Subperiod alpha significance for LS at 50 bps
    if not kill_triggered and ls_50 is not None:
        per_side_rate = (50 / 10000.0) / 2.0
        ls_net = pf["LS"].astype(float) - per_side_rate * pf["turnover_LS"].astype(float)
        subperiods = {
            "2011-01-01_to_2018-12-31": ("2011-01-01", "2018-12-31"),
            "2019-01-01_to_2025-12-31": ("2019-01-01", "2025-12-31"),
        }
        for name, (a, b) in subperiods.items():
            m = (pf.index >= pd.to_datetime(a)) & (pf.index <= pd.to_datetime(b))
            sub_ls = ls_net.loc[m]
            sub_bm = pf["BOVA11"].loc[m]
            al = ols_alpha_newey_west(sub_ls, sub_bm, lags=cfg.nw_lags)
            alpha_t = al.get("alpha_t", np.nan)
            if (not np.isfinite(alpha_t)) or (abs(alpha_t) < cfg.kill_alpha_t_min):
                kill_triggered = True
                break

    # Output
    print("\n=== Summary (equal-weight) ===")
    # LS block (raw + CAPM yardstick)
    if "LS" in results:
        v = results["LS"]
        print("\nLS:")
        print(f"  Mean monthly (raw): {v['mean_monthly']:.5f} (t={v['tstat_monthly']:.2f})")
        print(f"  CAPM alpha vs IBOV: {v['capm_alpha']*1e4:.2f} bps/mo (t={v['capm_alpha_t']:.2f}), Beta={v['capm_beta']:.3f}")
        print(f"  Alpha IR (annualized): {v['capm_ir_ann']:.2f} (TE={v['capm_te_ann']:.2%} ann)")
        print(f"  Hit rate: {v['hit_rate']:.2%}")
        print(f"  Max DD: {v['max_drawdown']:.2%}")

    # LS_vt block (comparison)
    if "LS_vt" in results:
        v = results["LS_vt"]
        print("\nLS (Vol-Targeted):")
        print(f"  Mean monthly (raw): {v['mean_monthly']:.5f} (t={v['tstat_monthly']:.2f})")
        print(f"  CAPM alpha vs IBOV: {v['capm_alpha']*1e4:.2f} bps/mo (t={v['capm_alpha_t']:.2f}), Beta={v['capm_beta']:.3f}")
        print(f"  Alpha IR (annualized): {v['capm_ir_ann']:.2f} (TE={v['capm_te_ann']:.2%} ann)")
        print(f"  Hit rate: {v['hit_rate']:.2%}")
        print(f"  Max DD: {v['max_drawdown']:.2%}")
    if "LS_pre_vt" in results:
        v = results["LS_pre_vt"]
        print("\nLS (Baseline, Pre-VT):")
        print(f"  Mean monthly (raw): {v['mean_monthly']:.5f} (t={v['tstat_monthly']:.2f})")
        print(f"  CAPM alpha vs IBOV: {v['capm_alpha']*1e4:.2f} bps/mo (t={v['capm_alpha_t']:.2f}), Beta={v['capm_beta']:.3f}")
        print(f"  Alpha IR (annualized): {v['capm_ir_ann']:.2f} (TE={v['capm_te_ann']:.2%} ann)")
        print(f"  Hit rate: {v['hit_rate']:.2%}")
        print(f"  Max DD: {v['max_drawdown']:.2%}")

    # D10/D1 blocks (keep CDI-excess context for reference)
    for k in ["D10", "D1"]:
        if k not in results:
            continue
        v = results[k]
        print(f"\n{k}:")
        print(f"  Mean monthly: {v['mean_monthly']:.5f} (t={v['tstat_monthly']:.2f})")
        print(f"  Mean excess (over CDI): {v['mean_monthly_excess']:.5f} (t={v['tstat_monthly_excess']:.2f})")
        print(f"  Ann. Sharpe (excess): {v['ann_sharpe_excess']:.2f}")
        print(f"  Hit rate: {v['hit_rate']:.2%}")
        print(f"  Max DD: {v['max_drawdown']:.2%}")

    a = results.get("LS_alpha_vs_BOVA11", {})
    if a:
        print("\nPrimary Yardstick — CAPM alpha vs IBOV (HAC lags={}):".format(cfg.nw_lags))
        print(f"  Alpha: {a['alpha_bps_per_month']:.2f} bps/month (t={a['alpha_tstat']:.2f}), Beta={a['beta']:.3f}, IR(ann)={a.get('ir_ann', float('nan')):.2f}")
    av = results.get("LS_vt_alpha_vs_BOVA11", {})
    if av:
        print("  LS (Vol-Targeted) Alpha: {} bps/month (t={:.2f}), Beta={:.3f}, IR(ann)={:.2f}".format(
            av.get('alpha_bps_per_month', float('nan')),
            av.get('alpha_tstat', float('nan')),
            av.get('beta', float('nan')),
            av.get('ir_ann', float('nan')),
        ))
    ap = results.get("LS_pre_vt_alpha_vs_BOVA11", {})
    if ap:
        print("  LS (Baseline) Alpha: {} bps/month (t={:.2f}), Beta={:.3f}, IR(ann)={:.2f}".format(
            ap.get('alpha_bps_per_month', float('nan')),
            ap.get('alpha_tstat', float('nan')),
            ap.get('beta', float('nan')),
            ap.get('ir_ann', float('nan')),
        ))

    print("\n=== Stability by subperiods ===")
    for name, v in stability.items():
        print(f"{name} -> LS mean {v['LS_mean']:.5f} (t={v['LS_t']:.2f}), D10 mean {v['D10_mean']:.5f} (t={v['D10_t']:.2f})")

    # Timing hygiene confirmation
    print("\nTiming hygiene: Momentum spec uses monthly adj closes with 12-2 window; portfolios formed at month-end using data up to t, traded at next day's open (T+1), and held to next T+1 open. All reported t-stats use Newey–West (lags={}).".format(cfg.nw_lags))

    # Cost grid summary
    print("\n=== Cost Grid (half per side) ===")
    print(f"Average turnover: D10={avg_to_d10:.2%}/mo, LS={avg_to_ls:.2%}/mo")
    for gr in grid_results:
        print(
            f"bps={gr['bps']:>3} -> D10 Sharpe={gr['D10_sharpe']:.2f}, alpha={gr['D10_alpha_bps']:.1f} bps (t={gr['D10_alpha_t']:.2f}) | "
            f"LS alpha={gr['LS_alpha_bps']:.1f} bps (t={gr['LS_alpha_t']:.2f}), IR(ann)={gr['LS_ir']:.2f}; "
            f"LS vol-target 10%: alpha={gr['LS_alpha_bps_vt']:.1f} bps (t={gr['LS_alpha_t_vt']:.2f}), IR(ann)={gr['LS_ir_vt']:.2f}, "
            f"maxDD={gr['LS_maxDD_vt']:.2%}, hit={gr['LS_hit_vt']:.2%}"
        )

    # Helper report: turnover decomposition and cost efficiency
    avg_to_ls_reconst = float(pf["turnover_LS_reconst"].mean()) if "turnover_LS_reconst" in pf else np.nan
    avg_to_ls_reweight = float(pf["turnover_LS_reweight"].mean()) if "turnover_LS_reweight" in pf else np.nan
    share_reconst = (avg_to_ls_reconst / avg_to_ls) if (avg_to_ls and np.isfinite(avg_to_ls) and avg_to_ls > 0) else np.nan
    share_reweight = (avg_to_ls_reweight / avg_to_ls) if (avg_to_ls and np.isfinite(avg_to_ls) and avg_to_ls > 0) else np.nan
    ce_alpha_bps = results.get("LS_alpha_vs_BOVA11", {}).get("alpha_bps_per_month", np.nan)
    cost_eff = (ce_alpha_bps / (avg_to_ls * 100)) if (avg_to_ls and np.isfinite(avg_to_ls) and avg_to_ls > 0) else np.nan
    print("\n=== Turnover Decomposition (LS) ===")
    print(f"Avg turnover: total={avg_to_ls:.2%}, reconst={avg_to_ls_reconst:.2%}, reweight={avg_to_ls_reweight:.2%}")
    print(f"Shares: reconst={share_reconst:.0%}, reweight={share_reweight:.0%}")
    print("\n=== Cost Efficiency (LS) ===")
    print(f"Alpha per 1x turnover: {cost_eff:.1f} bps per 100% turnover (alpha={ce_alpha_bps:.1f} bps/mo)")

    if kill_triggered:
        print("\nKill rule triggered: LS Sharpe <= 0 after 50 bps, or LS alpha insignificant (|t|<2) in a long subperiod.")
        # Save and exit early
        pf.to_csv(os.path.join(out_dir, "momentum_br_timeseries.csv"))
        pd.Series(results).to_json(os.path.join(out_dir, "momentum_br_summary.json"))
        pd.DataFrame(grid_results).to_csv(os.path.join(out_dir, "momentum_br_cost_grid.csv"), index=False)
        return

    # Save timeseries
    pf.to_csv(os.path.join(out_dir, "momentum_br_timeseries.csv"))
    pd.Series(results).to_json(os.path.join(out_dir, "momentum_br_summary.json"))
    pd.DataFrame(grid_results).to_csv(os.path.join(out_dir, "momentum_br_cost_grid.csv"), index=False)


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Brazil Momentum Factor Backtest (12-2, monthly, EW)")
    p.add_argument("--db-path", default=os.path.expanduser(
        "/Users/hemerson/Library/CloudStorage/GoogleDrive-hemersondv@gmail.com/My Drive/mel-database/financial_data.db"),
                   help="Path to SQLite database with daily prices")
    p.add_argument("--cdi-path", default=os.path.expanduser(
        "/Users/hemerson/Library/CloudStorage/GoogleDrive-hemersondv@gmail.com/My Drive/mel-database/cdi_series.xlsx"),
                   help="Path to CDI Excel file with Date, Close columns (index level)")
    p.add_argument("--liquidity-threshold", type=float, default=2_000_000.0,
                   help="Median daily traded value (BRL) over past ~3 months")
    p.add_argument("--start-date", default="2000-01-01")
    p.add_argument("--end-date", default=None)
    p.add_argument("--nw-lags", type=int, default=6)
    p.add_argument("--min-eligible", type=int, default=50)
    p.add_argument("--bova11-ticker", default="BOVA11")
    p.add_argument("--cohorts", type=int, default=1,
                   help="Number of staggered cohorts (1 = monthly all, 3 = rotate monthly for quarterly per cohort)")
    p.add_argument("--beta-overlay", action="store_true",
                   help="Apply composite-level beta overlay vs IBOV using rolling monthly beta")
    p.add_argument("--overlay-lookback-months", type=int, default=36,
                   help="Lookback window (months) to estimate composite beta for overlay")
    p.add_argument("--overlay-halflife-days", type=int, default=60,
                   help="EWMA halflife in trading days for beta estimate (approx converted to months)")
    p.add_argument("--overlay-shrink", type=float, default=0.5,
                   help="Shrinkage toward 0 for beta estimate (0..1), e.g., 0.5")
    p.add_argument("--overlay-band", type=float, default=0.11293,
                   help="Do not update hedge if |beta_hat| <= band; carry previous hedge")
    # Vol targeting options
    try:
        from argparse import BooleanOptionalAction  # py39+
        p.add_argument("--apply-vol-target", action=BooleanOptionalAction, default=False,
                       help="Apply 10% annualized vol targeting to hedged LS using rolling window")
    except Exception:
        p.add_argument("--apply-vol-target", action="store_true", default=False,
                       help="Apply 10% annualized vol targeting to hedged LS using rolling window")
    p.add_argument("--vol-target-ann", type=float, default=0.10,
                   help="Annualized volatility target (e.g., 0.10 for 10%)")
    p.add_argument("--vol-window-months", type=int, default=36,
                   help="Rolling window in months for realized vol estimate (e.g., 36)")
    p.add_argument("--vol-min-months", type=int, default=12,
                   help="Minimum months required for realized vol estimate (default 12)")
    p.add_argument("--band-keep", type=float, default=0.81393,
                   help="Keep band percentile threshold (e.g., 0.80)")
    p.add_argument("--band-add", type=float, default=0.90307,
                   help="Add band percentile threshold (e.g., 0.90)")
    p.add_argument("--sector-tol", type=float, default=0.09218,
                   help="Sector share tolerance per side (e.g., 0.10 = 10pp)")
    p.add_argument("--gross-tol", type=float, default=0.02,
                   help="Side gross renorm tolerance around 0.50 (e.g., 0.02 = ±2pp)")
    p.add_argument("--exit-cap-frac", type=float, default=0.07257,
                   help="Max fraction of target names per side allowed to exit on non-rebal months")
    p.add_argument("--ls-turnover-budget", type=float, default=0.32496,
                   help="Target LS monthly turnover budget (fraction, e.g., 0.30). Applied on non-rebalance months if enabled.")
    try:
        from argparse import BooleanOptionalAction  # py39+
        p.add_argument("--use-turnover-budget", action=BooleanOptionalAction, default=True,
                       help="Enable top-down LS turnover budget on non-rebalance months (disable with --no-use-turnover-budget)")
    except Exception:
        p.add_argument("--use-turnover-budget", action="store_true", default=True,
                       help="Enable top-down LS turnover budget on non-rebalance months")
    p.add_argument("--micro-add-frac", type=float, default=0.01774,
                   help="Fraction of target names eligible for micro adds in non-rebalance months (default: %(default)s)")
    p.add_argument("--kill-alpha-t-min", type=float, default=2.0,
                   help="Kill rule threshold for |alpha t-stat| after costs (default: %(default)s)")
    p.add_argument("--kill-net-sharpe-min", type=float, default=0.0,
                   help="Kill rule threshold for LS net Sharpe after costs (default: %(default)s)")
    try:
        from argparse import BooleanOptionalAction  # py39+
        p.add_argument("--adaptive-exit-cap", action=BooleanOptionalAction, default=True,
                       help="Enable adaptive exit-cap per side (turn off to use fixed --exit-cap-frac)")
    except Exception:
        # Fallback: provide explicit enable flag only
        p.add_argument("--adaptive-exit-cap", action="store_true", default=True,
                       help="Enable adaptive exit-cap per side (turn off by passing --no-adaptive-exit-cap if available)")
    p.add_argument("--live-capital", type=float, default=None,
                   help="Live capital amount for orders generation (enables orders_YYYY-MM.csv export)")
    p.add_argument("--lot-size", type=int, default=1,
                   help="Minimum lot size for share rounding in orders generation")
    p.add_argument("--write-state-snapshots", action="store_true", default=False,
                   help="Write cohort state snapshots to state/ directory for operational tracking")
    # Cold start and ramping
    try:
        from argparse import BooleanOptionalAction  # py39+
        p.add_argument("--cold-start", action=BooleanOptionalAction, default=False,
                       help="If enabled, also export cold-start files that ignore previous weights (bootstrap new accounts)")
    except Exception:
        p.add_argument("--cold-start", action="store_true", default=False,
                       help="If enabled, also export cold-start files that ignore previous weights (bootstrap new accounts)")
    p.add_argument("--ramp-frac", type=float, default=None,
                   help="Ramp fraction (0..1) for cold start sizing; default = 1 / cohorts if not set")
    p.add_argument("--out-dir", default="results",
                   help="Directory to write results outputs (default: %(default)s)")
    return p


def config_from_args(args: argparse.Namespace, *, start_date: Optional[str] = None,
                     end_date: Optional[str] = None, out_dir: Optional[str] = None) -> Config:
    return Config(
        db_path=args.db_path,
        cdi_path=args.cdi_path,
        liquidity_threshold=args.liquidity_threshold,
        start_date=start_date if start_date is not None else args.start_date,
        end_date=end_date if end_date is not None else args.end_date,
        nw_lags=args.nw_lags,
        min_eligible=args.min_eligible,
        bova11_ticker=args.bova11_ticker,
        cohorts=args.cohorts,
        beta_overlay=args.beta_overlay,
        overlay_lookback_months=args.overlay_lookback_months,
        overlay_halflife_days=args.overlay_halflife_days,
        overlay_shrink=args.overlay_shrink,
        overlay_band=args.overlay_band,
        apply_vol_target=getattr(args, "apply_vol_target", False),
        vol_target_ann=getattr(args, "vol_target_ann", 0.10),
        vol_window_months=getattr(args, "vol_window_months", 36),
        vol_min_months=getattr(args, "vol_min_months", 12),
        band_keep=args.band_keep,
        band_add=args.band_add,
        sector_tol=args.sector_tol,
        gross_tol=args.gross_tol,
        exit_cap_frac=args.exit_cap_frac,
        ls_turnover_budget=args.ls_turnover_budget,
        use_turnover_budget=getattr(args, "use_turnover_budget", False),
        adaptive_exit_cap=getattr(args, "adaptive_exit_cap", True),
        live_capital=args.live_capital,
        lot_size=args.lot_size,
        write_state_snapshots=args.write_state_snapshots,
        cold_start=getattr(args, "cold_start", False),
        ramp_frac=args.ramp_frac,
        out_dir=out_dir if out_dir is not None else getattr(args, "out_dir", "results"),
        kill_alpha_t_min=args.kill_alpha_t_min,
        kill_net_sharpe_min=args.kill_net_sharpe_min,
    )


if __name__ == "__main__":
    args = make_parser().parse_args()
    cfg = config_from_args(args)
    run_analysis(cfg)

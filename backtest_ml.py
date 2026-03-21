"""
backtest_ml.py — Walk-forward ML backtest for the ensemble predictor.

Methodology
-----------
  1. For each ticker, fetch 5 years of daily OHLCV (IBKR → yfinance fallback).
  2. Prime the fundamental cache once (EDGAR, ~0.5 s/ticker).
  3. Walk forward with an EXPANDING training window:
       • Re-train the ensemble every RETRAIN_EVERY trading days (default: 21).
       • First prediction starts only after MIN_TRAIN_BARS of warm-up.
       • Each prediction uses the 1M (21-day) horizon only — the sweet spot
         for swing trading.
  4. At each prediction bar, if AUROC (in-fold CV) ≥ threshold:
       • UP   → go long for HOLD_DAYS days, exit at the open of day HOLD_DAYS+1.
       • DOWN → stay flat (no shorting — mirrors the live system).
  5. Aggregate per-ticker and portfolio equity curves.
  6. Benchmark against SPY buy-and-hold over the same period.

Metrics reported
----------------
  Per-ticker : AUROC (mean over all folds), Win%, Avg hold return,
               Sharpe (annualised), Max drawdown, # trades.
  Portfolio  : Equal-weight long-only, daily rebalanced by signal.
               Sharpe, CAGR, Max drawdown vs SPY.
  Signal quality : Precision@top-decile, KS statistic mean.

Usage
-----
  python backtest_ml.py                         # full watchlist, 5y, 1M horizon
  python backtest_ml.py --tickers AAPL MSFT     # specific tickers
  python backtest_ml.py --horizon 5             # 1W swing
  python backtest_ml.py --years 3               # shorter window
  python backtest_ml.py --no-xs                 # skip cross-sectional features (speed)
"""

from __future__ import annotations

import argparse
import datetime
import logging
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# ── ensure project root on path ──────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import ml_predictor as ml
from edgar import get_price_history, get_fundamentals

logging.basicConfig(level=logging.WARNING)

# ── colour helpers ────────────────────────────────────────────────────────────
try:
    from colorama import Fore, Style, init as _ci
    _ci(autoreset=True)
    G = Fore.GREEN; R = Fore.RED; Y = Fore.YELLOW; B = Fore.CYAN; RESET = Style.RESET_ALL
except ImportError:
    G = R = Y = B = RESET = ""

SEP  = "─" * 116
SEP2 = "═" * 116

# ── backtest parameters (overridable via CLI) ─────────────────────────────────
DEFAULT_YEARS       = 5          # years of price history
DEFAULT_HORIZON     = 21         # 1M prediction horizon (trading days)
DEFAULT_HOLD_DAYS   = 21         # days to hold a long position
DEFAULT_RETRAIN     = 21         # re-train ensemble every N bars
MIN_TRAIN_BARS      = 300        # bars before first prediction
RESULTS_DIR         = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Core walk-forward engine
# ─────────────────────────────────────────────────────────────────────────────

def _sector_for(symbol: str) -> str | None:
    """Best-effort GICS sector lookup from EDGAR SIC."""
    try:
        from edgar import _get_cik, _get_sic
        cik = _get_cik(symbol)
        sic = _get_sic(cik) if cik else None
        sic2 = str(sic)[:2] if sic else ""
        return ml._SIC_TO_SECTOR.get(sic2)
    except Exception:
        return None


def walkforward_ml(
    symbol:       str,
    history:      pd.DataFrame,
    horizon:      int   = DEFAULT_HORIZON,
    hold_days:    int   = DEFAULT_HOLD_DAYS,
    retrain_every: int  = DEFAULT_RETRAIN,
    min_train:    int   = MIN_TRAIN_BARS,
    sector:       str | None = None,
    auroc_threshold: float = 0.51,
    use_xs:       bool = True,
) -> dict:
    """
    Walk-forward ML backtest for one ticker.

    Returns a results dict with per-trade log and equity curve.
    """
    if history is None or history.empty or len(history) < min_train + horizon + hold_days:
        return {"symbol": symbol, "error": "insufficient data", "trades": [], "equity": pd.Series(dtype=float)}

    closes = history["Close"].dropna()
    opens  = history.get("Open", closes)
    n      = len(closes)

    # Pre-build the full feature matrix once (expensive; ~3–6 s per ticker)
    try:
        feats_full = ml.build_all_features(symbol, history, sector=sector)
        # 95%-NaN imputation (mirrors predict())
        frac_nan = feats_full.isna().mean()
        mostly_nan = frac_nan[frac_nan > 0.95].index.tolist()
        if mostly_nan:
            feats_full[mostly_nan] = feats_full[mostly_nan].fillna(0.0)
        feats_full = feats_full.ffill().bfill()
    except Exception as exc:
        return {"symbol": symbol, "error": f"feature build failed: {exc}", "trades": [], "equity": pd.Series(dtype=float)}

    # Build alpha-vs-SPY labels (same as predict())
    future_ret  = closes.shift(-horizon) / closes - 1
    spy_ret_h   = ml._get_spy_horizon_return(history.index, horizon)
    if spy_ret_h is not None:
        labels = (future_ret > spy_ret_h).astype(int)
    else:
        labels = (future_ret > 0).astype(int)

    # Align features and labels
    combined = pd.concat([feats_full, labels.rename("_label")], axis=1).dropna()
    if len(combined) > horizon:
        combined = combined.iloc[:-horizon]   # drop last horizon rows (no valid label)

    if len(combined) < min_train:
        return {"symbol": symbol, "error": "not enough clean samples", "trades": [], "equity": pd.Series(dtype=float)}

    feat_names = [c for c in combined.columns if c != "_label"]
    X_all = combined[feat_names].values
    y_all = combined["_label"].values
    idx_all = combined.index   # DatetimeIndex aligned to price bars

    # ── walk-forward simulation ───────────────────────────────────────────────
    trades: list[dict] = []
    aurocs: list[float] = []
    precisions_top10: list[float] = []
    ks_stats: list[float] = []

    last_retrain_i  = -999
    lgbm_ = xgbm_ = logreg_ = scaler_ = None   # current fitted ensemble
    auroc_last = 0.0                            # AUROC of current ensemble

    position_open_until: int | None = None  # bar index when current long exits
    equity_curve: list[float] = [1.0]       # starting NAV = 1.0
    position_returns: list[float] = []

    for i in range(min_train, len(X_all)):
        date_i = idx_all[i]

        # ── Re-train if needed ────────────────────────────────────────────────
        if (i - last_retrain_i) >= retrain_every:
            X_tr = X_all[:i]
            y_tr = y_all[:i]

            # Quick 3-fold walk-forward AUROC on training window
            fold_auroc, fold_ks = ml._walkforward_auroc(
                X_tr, y_tr,
                clf_factory=lambda: ml._build_lgbm(X_tr.shape[1]),
                n_splits=3,
                gap=horizon,
                min_train=100,
            )
            auroc_last = fold_auroc
            aurocs.append(fold_auroc)
            ks_stats.append(fold_ks)

            if fold_auroc >= auroc_threshold:
                try:
                    lgbm_, xgbm_, logreg_, scaler_ = ml._train_ensemble(X_tr, y_tr, feat_names)
                except Exception:
                    lgbm_ = None

            last_retrain_i = i

        # ── Skip if no valid model or AUROC below gate ────────────────────────
        if lgbm_ is None or auroc_last < auroc_threshold:
            equity_curve.append(equity_curve[-1])
            continue

        # ── Predict on current bar ────────────────────────────────────────────
        x_now = X_all[i : i + 1]
        try:
            p_up = ml._ensemble_predict_proba(x_now, lgbm_, xgbm_, logreg_, scaler_)
        except Exception:
            equity_curve.append(equity_curve[-1])
            continue

        direction = "UP" if p_up >= 0.50 else "DOWN"

        # ── Enter long if predicted UP and no open position ───────────────────
        # Find the actual price bar aligned to this feature row
        if date_i not in closes.index:
            equity_curve.append(equity_curve[-1])
            continue

        price_loc = closes.index.get_loc(date_i)
        # Entry at next bar's open (realistic: signal on close, enter next open)
        entry_bar = price_loc + 1
        if entry_bar >= len(closes):
            equity_curve.append(equity_curve[-1])
            continue

        if direction == "UP" and (position_open_until is None or i >= position_open_until):
            exit_bar = min(entry_bar + hold_days, len(closes) - 1)
            entry_px = float(opens.iloc[entry_bar])

            # ── Simulate trailing stop + partial exits bar-by-bar ─────────
            # ATR estimate: use 14-bar average true range at entry
            _hi = history["High"].iloc[max(0, entry_bar-14):entry_bar]
            _lo = history["Low"].iloc[max(0, entry_bar-14):entry_bar]
            _pc = closes.iloc[max(0, entry_bar-14):entry_bar]
            if len(_hi) >= 2 and len(_lo) >= 2:
                _tr = np.maximum(_hi.values[1:] - _lo.values[1:],
                        np.maximum(np.abs(_hi.values[1:] - _pc.values[:-1]),
                                   np.abs(_lo.values[1:] - _pc.values[:-1])))
                atr_abs = float(np.mean(_tr)) if len(_tr) > 0 else entry_px * 0.02
            else:
                atr_abs = entry_px * 0.02  # fallback

            from risk_manager import TrailingStopState, evaluate_partial_exits
            stop_px = entry_px - config.ATR_STOP_MULTIPLIER * atr_abs
            stop_px = max(stop_px, entry_px * (1 - config.ATR_MAX_STOP_PCT))
            ts = TrailingStopState(entry_price=entry_px, atr_14_abs=atr_abs,
                                  initial_stop=stop_px)

            exit_px = None
            exit_reason = "hold_expiry"
            partials_taken: set[int] = set()
            partial_pnl = 0.0  # cumulative P&L from partial exits
            remaining_frac = 1.0  # fraction of position still held

            # Time stop: limit max hold in calendar days (approx 1 bar ≈ 1 trading day)
            time_stop_bar = exit_bar
            if config.TIME_STOP_ENABLED:
                time_stop_bar = min(entry_bar + config.TIME_STOP_DAYS, exit_bar)

            for bar_j in range(entry_bar, time_stop_bar + 1):
                bar_price = float(closes.iloc[bar_j])

                # Update trailing stop
                ts.update(bar_price)

                # Check trailing stop hit
                if bar_price <= ts.current_stop:
                    exit_px = ts.current_stop
                    exit_reason = f"trailing_{ts.stage_label}"
                    break

                # Check partial exits
                pexits = evaluate_partial_exits(
                    current_price=bar_price, entry_price=entry_px,
                    atr_14_abs=atr_abs, exits_already_taken=partials_taken,
                )
                for pe in pexits:
                    sell_frac = pe.fraction * remaining_frac
                    partial_ret = (bar_price - entry_px) / entry_px
                    partial_pnl += sell_frac * partial_ret
                    remaining_frac -= sell_frac
                    partials_taken.add(pe.exit_id)

            # Time stop: if we reached the time limit without trailing stop exit
            if exit_px is None and config.TIME_STOP_ENABLED and time_stop_bar < exit_bar:
                exit_px = float(closes.iloc[time_stop_bar])
                exit_reason = "time_stop"

            # If no trailing/time stop was hit, exit at the hold-expiry bar
            if exit_px is None:
                exit_px = float(opens.iloc[exit_bar])

            # Blended return: partial exits + remainder exit
            remainder_ret = (exit_px - entry_px) / entry_px if entry_px > 0 else 0.0
            trade_ret = partial_pnl + remaining_frac * remainder_ret

            # Record trade
            trades.append({
                "symbol":       symbol,
                "entry_date":   str(closes.index[entry_bar].date()),
                "exit_date":    str(closes.index[exit_bar].date()),
                "entry_price":  round(entry_px, 4),
                "exit_price":   round(exit_px, 4),
                "pct_return":   round(trade_ret * 100, 3),
                "p_up":         round(p_up, 4),
                "auroc_at_entry": round(auroc_last, 4),
                "direction":    direction,
                "hold_bars":    exit_bar - entry_bar,
                "exit_reason":  exit_reason,
                "partials_taken": len(partials_taken),
                "trail_stage":  ts.stage_label,
            })
            position_returns.append(trade_ret)
            position_open_until = i + hold_days  # no new trades until this expires

            # Advance equity curve by trade return
            equity_curve.append(equity_curve[-1] * (1 + trade_ret))
        else:
            equity_curve.append(equity_curve[-1])

        # ── Precision @ top-decile: collect for final scoring ─────────────────
        # (We use the realised label as ground truth)
        if i < len(y_all):
            actual_label = int(y_all[i])
            # Record p_up with actual outcome for later precision@10 calc
            precisions_top10.append((p_up, actual_label))

    # ── Post-processing ───────────────────────────────────────────────────────
    equity = pd.Series(equity_curve, dtype=float)

    mean_auroc = float(np.mean(aurocs)) if aurocs else 0.0
    mean_ks    = float(np.mean(ks_stats)) if ks_stats else 0.0

    # Precision @ top decile: among the highest-confidence UP predictions,
    # what fraction actually beat SPY?
    prec_at_10 = float("nan")
    if len(precisions_top10) >= 10:
        p10_df     = pd.DataFrame(precisions_top10, columns=["p_up", "actual"])
        threshold  = p10_df["p_up"].quantile(0.90)
        top10_rows = p10_df[p10_df["p_up"] >= threshold]
        if len(top10_rows) > 0:
            prec_at_10 = float(top10_rows["actual"].mean())

    # Annualised Sharpe from trade returns (no risk-free rate adjustment)
    trade_rets = np.array(position_returns)
    sharpe = float("nan")
    if len(trade_rets) >= 2:
        mean_r = float(np.mean(trade_rets))
        std_r  = float(np.std(trade_rets, ddof=1))
        if std_r > 0:
            # Rough annualisation: assume hold_days days per trade
            trades_per_year = 252 / hold_days
            sharpe = (mean_r / std_r) * np.sqrt(trades_per_year)

    # Max drawdown from equity curve
    eq_arr = equity.values
    peak   = np.maximum.accumulate(eq_arr)
    dd     = (eq_arr - peak) / np.where(peak > 0, peak, 1)
    max_dd = float(np.min(dd))

    closed_trades  = [t for t in trades if t["pct_return"] is not None]
    winning_trades = [t for t in closed_trades if t["pct_return"] > 0]
    win_rate       = len(winning_trades) / len(closed_trades) if closed_trades else float("nan")
    avg_ret        = float(np.mean([t["pct_return"] for t in closed_trades])) if closed_trades else float("nan")
    cum_return     = float(equity.iloc[-1] - 1.0)

    return {
        "symbol":          symbol,
        "error":           None,
        "trades":          trades,
        "equity":          equity,
        "mean_auroc":      mean_auroc,
        "mean_ks":         mean_ks,
        "prec_at_10":      prec_at_10,
        "n_trades":        len(closed_trades),
        "win_rate":        win_rate,
        "avg_ret_pct":     avg_ret,
        "sharpe":          sharpe,
        "max_drawdown":    max_dd,
        "cum_return":      cum_return,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SPY benchmark
# ─────────────────────────────────────────────────────────────────────────────

def _spy_benchmark(start_date: pd.Timestamp, end_date: pd.Timestamp) -> dict:
    """Buy-and-hold SPY from start_date to end_date."""
    try:
        import yfinance as yf
        df = yf.Ticker("SPY").history(start=str(start_date.date()),
                                      end=str(end_date.date()), auto_adjust=True)
        if df is None or df.empty:
            return {}
        c   = df["Close"].dropna()
        c.index = c.index.tz_localize(None) if c.index.tzinfo else c.index
        eq  = c / c.iloc[0]
        log = np.log(c / c.shift(1)).dropna()
        sp  = float(log.mean() / log.std() * np.sqrt(252)) if log.std() > 0 else float("nan")
        peak = np.maximum.accumulate(eq.values)
        dd   = (eq.values - peak) / np.where(peak > 0, peak, 1)
        return {
            "cum_return": float(eq.iloc[-1] - 1.0),
            "sharpe":     sp,
            "max_dd":     float(np.min(dd)),
            "equity":     eq,
        }
    except Exception:
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Printing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _c(val: float | None, fmt: str = "+.2f", good_positive: bool = True,
        suffix: str = "%") -> str:
    """Colour-formatted value string."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return f"{'N/A':>10}"
    s = f"{val:{fmt}}{suffix}"
    colour = (G if val > 0 else R) if good_positive else (R if val > 0 else G)
    return f"{colour}{s:>10}{RESET}"


def _print_trade_table(trades: list[dict], symbol: str) -> None:
    hdr = (
        f"  {'#':<4} {'ENTRY':>12} {'EXIT':>12} {'ENTRY $':>9} {'EXIT $':>9} "
        f"{'RET%':>8} {'P(UP)':>7} {'AUROC':>7} {'HOLD':>5}"
    )
    print(f"\n  {B}TRADE LOG — {symbol}{RESET}")
    print(f"  {SEP}")
    print(hdr)
    print(f"  {SEP}")
    for i, t in enumerate(trades, 1):
        ret = t["pct_return"]
        col = G if ret > 0 else R
        print(
            f"  {i:<4} {t['entry_date']:>12} {t['exit_date']:>12} "
            f"{t['entry_price']:>9.2f} {t['exit_price']:>9.2f} "
            f"{col}{ret:>+7.2f}%{RESET}  {t['p_up']:>5.3f}  {t['auroc_at_entry']:>5.3f}  {t['hold_bars']:>4}d"
        )
    print(f"  {SEP}")


# ─────────────────────────────────────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────────────────────────────────────

def run_ml_backtest(
    watchlist:    list[str],
    years:        int  = DEFAULT_YEARS,
    horizon:      int  = DEFAULT_HORIZON,
    hold_days:    int  = DEFAULT_HOLD_DAYS,
    retrain_every: int = DEFAULT_RETRAIN,
    min_train:    int  = MIN_TRAIN_BARS,
    auroc_threshold: float = 0.51,
    use_xs:       bool = True,
    show_trades:  bool = False,
) -> None:
    date_str = datetime.date.today().isoformat()

    print(f"\n{SEP2}")
    print(f"  {B}BUY-THE-SAUCE  ML ENSEMBLE BACKTEST  ·  {date_str}{RESET}")
    print(SEP2)
    print(f"  Tickers         : {len(watchlist)}  ({', '.join(watchlist[:10])}{'…' if len(watchlist)>10 else ''})")
    print(f"  Price history   : {years} years")
    print(f"  Horizon         : {horizon}d  (1{'W' if horizon==5 else 'M' if horizon==21 else 'Y'})")
    print(f"  Hold period     : {hold_days} days")
    print(f"  Retrain every   : {retrain_every} bars")
    print(f"  Min train bars  : {min_train}")
    print(f"  AUROC gate      : ≥ {auroc_threshold:.2f}")
    print(f"  Cross-sect. feats: {'ON' if use_xs else 'OFF'}")
    print(SEP2)

    # ── Optionally disable cross-sectional features (faster) ─────────────────
    # We monkey-patch build_all_features to skip xs if use_xs=False
    _orig_build = ml.build_all_features
    if not use_xs:
        def _no_xs(symbol, history, sector=None):
            import pandas as _pd
            tech     = ml.build_technical_features(history)
            calendar = ml.build_calendar_features(history)
            patterns = ml.build_price_pattern_features(history)
            macro    = ml.build_macro_features(history, sector=sector)
            hmm      = ml.build_hmm_features(history)
            mc       = ml.build_mc_feature(symbol, history).to_frame()
            fund     = ml.build_fundamental_features(symbol, history)
            return _pd.concat([tech, calendar, patterns, macro, hmm, mc, fund], axis=1)
        ml.build_all_features = _no_xs

    results = []
    all_trades_flat: list[dict] = []
    price_start: pd.Timestamp | None = None
    price_end:   pd.Timestamp | None = None

    for sym in watchlist:
        print(f"\n  [{sym}] fetching price & fundamentals …", end="", flush=True)
        hist, src = get_price_history(sym, period_years=years)
        if hist is None or hist.empty:
            print(f"  {R}no price data{RESET}")
            results.append({"symbol": sym, "error": "no price data"})
            continue

        # Prime fundamental cache so ML doesn't re-fetch EDGAR
        try:
            info = get_fundamentals(sym)
            ml.prime_fundamental_cache(sym, info)
        except Exception:
            pass

        sector = _sector_for(sym)
        print(f" {len(hist)} bars ({src}, sector={sector or 'N/A'}) … training …", end="", flush=True)

        if price_start is None:
            price_start = hist.index[0]
        price_end = hist.index[-1]

        res = walkforward_ml(
            symbol         = sym,
            history        = hist,
            horizon        = horizon,
            hold_days      = hold_days,
            retrain_every  = retrain_every,
            min_train      = min_train,
            sector         = sector,
            auroc_threshold = auroc_threshold,
            use_xs         = use_xs,
        )

        if res.get("error"):
            print(f"  {R}{res['error']}{RESET}")
            results.append(res)
            continue

        all_trades_flat.extend(res["trades"])
        results.append(res)

        n = res["n_trades"]
        wr = res["win_rate"]
        ar = res["avg_ret_pct"]
        sh = res["sharpe"]
        dd = res["max_drawdown"]
        au = res["mean_auroc"]
        print(
            f" done.  "
            f"AUROC={au:.3f}  trades={n}  "
            f"win={wr:.0%}  avgRet={ar:+.2f}%  "
            f"sharpe={sh:+.2f}  maxDD={dd:.1%}"
        )

        if show_trades and res["trades"]:
            _print_trade_table(res["trades"], sym)

    # Restore original build_all_features
    ml.build_all_features = _orig_build

    # ── Per-ticker summary table ──────────────────────────────────────────────
    valid = [r for r in results if not r.get("error")]
    if not valid:
        print(f"\n  {R}No valid results — check price data availability.{RESET}")
        return

    print(f"\n\n{SEP2}")
    print(f"  {B}PER-TICKER SUMMARY{RESET}")
    print(SEP2)
    hdr = (
        f"  {'TICKER':<7} {'AUROC':>7} {'KS':>6} {'P@10%':>7} "
        f"{'TRADES':>7} {'WIN%':>7} {'AVG RET':>9} {'SHARPE':>8} {'MAXDD':>8} {'CUM RET':>9}"
    )
    print(hdr)
    print(f"  {SEP}")

    all_trade_rets   = []
    portfolio_wins   = 0
    portfolio_total  = 0

    for r in sorted(valid, key=lambda x: x.get("mean_auroc", 0), reverse=True):
        sym  = r["symbol"]
        au   = r.get("mean_auroc", float("nan"))
        ks   = r.get("mean_ks", float("nan"))
        p10  = r.get("prec_at_10", float("nan"))
        nt   = r.get("n_trades", 0)
        wr   = r.get("win_rate", float("nan"))
        ar   = r.get("avg_ret_pct", float("nan"))
        sh   = r.get("sharpe", float("nan"))
        dd   = r.get("max_drawdown", float("nan"))
        cr   = r.get("cum_return", float("nan"))

        au_col = G if (not np.isnan(au) and au >= 0.54) else (Y if (not np.isnan(au) and au >= 0.51) else R)
        wr_col = G if (not np.isnan(wr) and wr >= 0.55) else (Y if (not np.isnan(wr) and wr >= 0.50) else R)

        def _f(v, fmt="+.2f", suf="%", good_pos=True):
            if np.isnan(v): return f"{'N/A':>9}"
            col = (G if v > 0 else R) if good_pos else (R if v > 0 else G)
            return f"{col}{v:{fmt}}{suf}{RESET}"

        print(
            f"  {sym:<7} "
            f"{au_col}{au:>6.3f}{RESET}  "
            f"{ks:>5.3f}  "
            f"{'' if np.isnan(p10) else f'{p10:.2f}':>6}  "
            f"{nt:>7}  "
            f"{wr_col}{'' if np.isnan(wr) else f'{wr:.0%}':>6}{RESET}  "
            f"{_f(ar):>9}  "
            f"{_f(sh, fmt='+.2f', suf='', good_pos=True):>8}  "
            f"{_f(dd*100, fmt='.1f', suf='%', good_pos=False):>8}  "
            f"{_f(cr*100, fmt='+.1f', suf='%', good_pos=True):>9}"
        )

        if r["trades"]:
            t_rets = [t["pct_return"] / 100 for t in r["trades"]]
            all_trade_rets.extend(t_rets)
            portfolio_wins  += sum(1 for t in t_rets if t > 0)
            portfolio_total += len(t_rets)

    print(f"  {SEP}")

    # ── Portfolio-level aggregate stats ───────────────────────────────────────
    print(f"\n{SEP2}")
    print(f"  {B}PORTFOLIO AGGREGATE  (equal-weight across all tickers){RESET}")
    print(SEP2)

    if all_trade_rets:
        arr = np.array(all_trade_rets)
        port_mean = float(arr.mean())
        port_std  = float(arr.std(ddof=1)) if len(arr) > 1 else float("nan")
        trades_per_year = 252 / hold_days * len(valid)
        port_sharpe = (port_mean / port_std * np.sqrt(trades_per_year)) if port_std > 0 else float("nan")
        port_win    = portfolio_wins / portfolio_total if portfolio_total else float("nan")
        port_cr     = float(np.prod(1 + arr) - 1)
        port_expect = float(port_win * arr[arr > 0].mean() + (1 - port_win) * arr[arr < 0].mean()) \
                      if len(arr[arr > 0]) > 0 and len(arr[arr < 0]) > 0 else float("nan")

        print(f"  {'Total trades (all tickers)':<32}: {len(arr)}")
        print(f"  {'Portfolio win rate':<32}: {G if port_win>=0.55 else Y}{port_win:.1%}{RESET}")
        print(f"  {'Avg return per trade':<32}: {G if port_mean>0 else R}{port_mean:+.3%}{RESET}")
        print(f"  {'Expectancy per trade':<32}: {G if port_expect>0 else R}{port_expect:+.3%}{RESET}")
        print(f"  {'Annualised Sharpe (portfolio)':<32}: {G if port_sharpe>0.5 else Y}{port_sharpe:+.2f}{RESET}")
        print(f"  {'Cumulative return (sequential)':<32}: {G if port_cr>0 else R}{port_cr:+.1%}{RESET}")
    else:
        print(f"  {R}No trades generated across any ticker.{RESET}")

    # ── SPY benchmark comparison ──────────────────────────────────────────────
    if price_start and price_end:
        spy = _spy_benchmark(price_start, price_end)
        if spy:
            print(f"\n{SEP2}")
            print(f"  {B}SPY BUY-AND-HOLD BENCHMARK  ({price_start.date()} → {price_end.date()}){RESET}")
            print(SEP2)
            spy_cr  = spy.get("cum_return", float("nan"))
            spy_sh  = spy.get("sharpe", float("nan"))
            spy_dd  = spy.get("max_dd", float("nan"))
            print(f"  {'Cumulative return':<32}: {G if spy_cr>0 else R}{spy_cr:+.1%}{RESET}")
            print(f"  {'Annualised Sharpe':<32}: {G if spy_sh>0.5 else Y}{spy_sh:+.2f}{RESET}")
            print(f"  {'Max drawdown':<32}: {R}{spy_dd:.1%}{RESET}")
            print()
            if all_trade_rets and not np.isnan(port_sharpe):
                delta_sharpe = port_sharpe - spy_sh
                delta_ret    = port_cr     - spy_cr
                print(f"  {'ML vs SPY — Δ Cum Return':<32}: {G if delta_ret>0 else R}{delta_ret:+.1%}{RESET}")
                print(f"  {'ML vs SPY — Δ Sharpe':<32}: {G if delta_sharpe>0 else R}{delta_sharpe:+.2f}{RESET}")

    # ── Signal quality summary ────────────────────────────────────────────────
    print(f"\n{SEP2}")
    print(f"  {B}SIGNAL QUALITY SUMMARY{RESET}")
    print(SEP2)
    valid_au = [r["mean_auroc"] for r in valid if not np.isnan(r.get("mean_auroc", float("nan")))]
    valid_ks = [r["mean_ks"]    for r in valid if not np.isnan(r.get("mean_ks",    float("nan")))]
    valid_p10 = [r["prec_at_10"] for r in valid
                 if not np.isnan(r.get("prec_at_10", float("nan")))]
    if valid_au:
        print(f"  Mean walk-forward AUROC  : {np.mean(valid_au):.3f}  "
              f"(range {np.min(valid_au):.3f}–{np.max(valid_au):.3f})")
    if valid_ks:
        print(f"  Mean KS statistic        : {np.mean(valid_ks):.3f}")
    if valid_p10:
        print(f"  Mean Precision@top-decile: {np.mean(valid_p10):.3f}  "
              f"(random = 0.50; above 0.55 is useful)")
    n_above = sum(1 for a in valid_au if a >= auroc_threshold)
    print(f"  Tickers above AUROC gate : {n_above}/{len(valid_au)}  (gate = {auroc_threshold})")
    print()
    print(f"  INTERPRETATION:")
    print(f"    AUROC 0.51–0.54  → marginal edge (academic upper bound for large-caps)")
    print(f"    AUROC ≥ 0.55     → statistically robust — use with conviction")
    print(f"    Precision@10 ≥ 0.55 → top-decile predictions are profitable")
    print(f"    Sharpe > 0.5     → risk-adjusted return above passive investing")
    print(SEP2)

    # ── Save CSV of all trades ────────────────────────────────────────────────
    if all_trades_flat:
        csv_path = os.path.join(RESULTS_DIR, f"backtest_ml_{date_str}.csv")
        pd.DataFrame(all_trades_flat).to_csv(csv_path, index=False)
        print(f"\n  Trades saved → {csv_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    from watchlist import WATCHLIST
    p = argparse.ArgumentParser(description="Walk-forward ML ensemble backtest.")
    p.add_argument("--tickers",  nargs="+", default=None)
    p.add_argument("--years",    type=int,   default=DEFAULT_YEARS)
    p.add_argument("--horizon",  type=int,   default=DEFAULT_HORIZON,
                   help="Prediction horizon in trading days (5=1W, 21=1M, 252=1Y)")
    p.add_argument("--hold",     type=int,   default=DEFAULT_HOLD_DAYS, dest="hold_days")
    p.add_argument("--retrain",  type=int,   default=DEFAULT_RETRAIN)
    p.add_argument("--min-train",type=int,   default=MIN_TRAIN_BARS, dest="min_train")
    p.add_argument("--auroc",    type=float, default=getattr(config, "ML_MIN_AUROC_THRESHOLD", 0.51))
    p.add_argument("--no-xs",    action="store_true", dest="no_xs",
                   help="Disable cross-sectional peer features (faster)")
    p.add_argument("--trades",   action="store_true", dest="show_trades",
                   help="Print per-ticker trade log")
    args = p.parse_args()
    args.watchlist = args.tickers if args.tickers else WATCHLIST
    return args


if __name__ == "__main__":
    args = _parse_args()
    run_ml_backtest(
        watchlist      = args.watchlist,
        years          = args.years,
        horizon        = args.horizon,
        hold_days      = args.hold_days,
        retrain_every  = args.retrain,
        min_train      = args.min_train,
        auroc_threshold= args.auroc,
        use_xs         = not args.no_xs,
        show_trades    = args.show_trades,
    )

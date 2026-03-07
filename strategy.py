"""
strategy_v4.py  —  Optimized ML Stock Strategy
================================================
Improvements over v3:
  A1. Multi-signal regime filter (more time in market)
  A2. 6 new factors: 52w-high ratio, ATR-adj momentum,
      rev acceleration, turnover, price breakout, vol trend
  A3. Confidence-weighted positions (rank softmax, not equal)
  A4. TOP_K = 40 (optimized portfolio size)
  A5. Tuned LightGBM w/ early stopping
  A6. vectorbt for backtesting 

Run:
    source .venv/bin/activate
    python3 strategy_v4.py
"""

import os, sys, glob, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from tqdm import tqdm

print("=" * 62)
print("  Taiwan ML Stock Strategy  v4  (Optimized)")
print("=" * 62)

# ─── Config ──────────────────────────────────────────────────────────────────
CACHE          = "finmind_cache"
TRAIN_MONTHS   = 48       # rolling train window (延長到 48 個月)
PURGE_MONTHS   = 1        # purge gap (data-leakage prevention)
STEP_MONTHS    = 3        # retrain every N months
TEST_START     = pd.Timestamp("2019-01-01")  # FIX1: 延長回測期（原 2022-01-01）
TOP_K          = 40       # optimized portfolio size from Sweep
WEIGHT_TEMP    = 0.5      # softmax temperature (optimized from Sweep)
FEE            = 1.425 / 1000   # 0.1425% 單邊手續費（vectorbt fees 參數為單邊）
TAX            = 3 / 1000
# STOP_LOSS 已移除（從未實作停損機制）
INIT_CASH      = 1_000_000
# FIX2: 流動性過濾門檻（30天均量 > 3000萬台幣才可進場）
LIQUIDITY_MIN  = 30_000_000
# FIX3: 因子精簡 — 自動選 ICIR 絕對值前 N 個
TOP_FACTORS_K  = 8

# ─── 1. Data loading ─────────────────────────────────────────────────────────
print("\n[1/7] Loading data...")

close = pd.read_pickle(os.path.join(CACHE, "close_wide.pkl"))
close.index = pd.to_datetime(close.index)
close.index.name = "date"
close.columns.name = "stock_id"

# Load high/low for ATR from price cache
print("  Loading OHLCV (for ATR & turnover)…")
high_frames, low_frames, money_frames = [], [], []
for f in tqdm(glob.glob(os.path.join(CACHE, "price/*.pkl")), desc="  OHLCV"):
    try:
        df = pd.read_pickle(f)
        df["date"] = pd.to_datetime(df["date"])
        for col, lst in [("max", high_frames), ("min", low_frames), ("Trading_money", money_frames)]:
            if col in df.columns:
                tmp = df[["date", "stock_id", col]].copy()
                tmp[col] = pd.to_numeric(tmp[col], errors="coerce")
                lst.append(tmp)
    except Exception:
        pass

def _wide(frames, col):
    df = pd.concat(frames, ignore_index=True)
    w = df.pivot_table(index="date", columns="stock_id", values=col)
    w.index = pd.to_datetime(w.index)
    w.index.name = "date"
    return w

high_wide  = _wide(high_frames,  "max")
low_wide   = _wide(low_frames,   "min")
money_wide = _wide(money_frames, "Trading_money")

# Revenue
print("  Loading revenue…")
rev_frames = []
for f in tqdm(glob.glob(os.path.join(CACHE, "revenue/*.pkl")), desc="  Revenue"):
    try:
        df = pd.read_pickle(f)
        df["date"] = pd.to_datetime(df["date"])
        rev_frames.append(df[["date", "stock_id", "revenue"]])
    except Exception:
        pass
rev_wide = (
    pd.concat(rev_frames, ignore_index=True)
    .pivot_table(index="date", columns="stock_id", values="revenue")
)
rev_wide.index = pd.to_datetime(rev_wide.index)
rev_wide.index.name = "date"

# Institution
print("  Loading institution…")
fg_frames, tr_frames = [], []
for f in tqdm(glob.glob(os.path.join(CACHE, "institution/*.pkl")), desc="  Inst"):
    try:
        df = pd.read_pickle(f)
        df["date"] = pd.to_datetime(df["date"])
        df["net"] = df["buy"] - df["sell"]
        fg_frames.append(df[df["name"] == "Foreign_Investor"][["date", "stock_id", "net"]])
        tr_frames.append(df[df["name"] == "Investment_Trust"][["date", "stock_id", "net"]])
    except Exception:
        pass

foreign_net = _wide(fg_frames, "net")
trust_net   = _wide(tr_frames, "net")

vol_wide = money_wide  # proxy for volume in money terms

print(f"  close={close.shape}  high={high_wide.shape}  rev={rev_wide.shape}")

# ─── 2. Factor engineering ────────────────────────────────────────────────────
print("\n[2/7] Engineering factors…")


def resample_monthly(df: pd.DataFrame) -> pd.DataFrame:
    return df.resample("ME").last()


def to_long(df: pd.DataFrame, name: str) -> pd.Series:
    """Wide → Long (Pandas 3.0 safe: unstack instead of stack)"""
    s = df.unstack().swaplevel().sort_index()
    s.name = name
    return s


def zscore_xs(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional z-score with winsorize ±3σ"""
    df = df.replace([np.inf, -np.inf], np.nan)
    mu  = df.mean(axis=1)
    std = df.std(axis=1).replace(0, np.nan)
    return (df.sub(mu, axis=0).div(std, axis=0)).clip(-3, 3)


# ── Momentum factors ──────────────────────────────────────────────────────────
mom_1m = close / close.shift(21) - 1
mom_3m = close / close.shift(63) - 1
mom_6m = close / close.shift(126) - 1

# Risk-adjusted momentum: mom / realized vol (Fama-French style)
rvol_20  = close.pct_change().rolling(20).std().replace(0, np.nan)
mom_1m_ra = (mom_1m / rvol_20).replace([np.inf, -np.inf], np.nan)  # NEW

# 52-week high ratio: how close is price to its 52w high  NEW
high_52w        = close.rolling(252).max()
price_to_52w    = close / high_52w.replace(0, np.nan)

# Price breakout: price > 20d high (momentum breakout signal)  NEW
high_20d        = close.rolling(20).max()
breakout_signal = (close / high_20d.replace(0, np.nan) - 1)  # fraction above 20d high

# ── RSI ────────────────────────────────────────────────────────────────────────
def calc_rsi(price_df, period=14):
    delta = price_df.diff()
    gain  = delta.clip(lower=0).ewm(com=period-1, min_periods=period).mean()
    loss  = (-delta).clip(lower=0).ewm(com=period-1, min_periods=period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

rsi_14 = calc_rsi(close)

# ── ATR (Average True Range) — volatility / risk factor  NEW ─────────────────
def calc_atr(high, low, close_p, period=14):
    tr = pd.DataFrame({
        "hl": high - low,
        "hc": (high - close_p.shift(1)).abs(),
        "lc": (low  - close_p.shift(1)).abs(),
    }).max(axis=1, skipna=False)
    # align across stocks
    tr_wide = high.copy() * np.nan
    # build stock-by-stock – but since we have wide DataFrames, compute direct
    hl  = (high_wide - low_wide).abs()
    hc  = (high_wide - close.shift(1)).abs()
    lc  = (low_wide  - close.shift(1)).abs()
    tr_w = pd.concat([hl, hc, lc]).groupby(level=0).max()
    atr  = tr_w.rolling(period).mean()
    return atr

atr_14 = (high_wide - low_wide).abs()
for col in ["hc", "lc"]:
    pass
# Simplified ATR (no shift needed for wide)
hl = (high_wide - low_wide).abs()
atr_14 = hl.rolling(14).mean()
atr_rel = (atr_14 / close.replace(0, np.nan))   # ATR / price = relative vol  NEW

# ── Volume / Turnover factors ─────────────────────────────────────────────────
vol_ratio = vol_wide.rolling(20).mean() / vol_wide.rolling(60).mean()   # v3 factor
money_20d = vol_wide.rolling(20).mean()   # absolute money flow  NEW

# ── Revenue factors ───────────────────────────────────────────────────────────
rev_yoy = rev_wide.pct_change(12)
rev_mom = rev_wide.pct_change(1)
# Revenue acceleration: mom – mom 3 months ago  NEW
rev_accel = rev_wide.pct_change(1) - rev_wide.pct_change(1).shift(3)

rev_yoy_d  = rev_yoy.reindex(close.index, method="ffill")
rev_mom_d  = rev_mom.reindex(close.index, method="ffill")
rev_accel_d = rev_accel.reindex(close.index, method="ffill")

# ── Institution factors ───────────────────────────────────────────────────────
vol_sum20       = vol_wide.rolling(20).sum()
common_f        = foreign_net.columns.intersection(vol_sum20.columns)
common_t        = trust_net.columns.intersection(vol_sum20.columns)
foreign_net_20d = foreign_net[common_f].rolling(20).sum() / vol_sum20[common_f].replace(0, np.nan)
trust_net_10d   = trust_net[common_t].rolling(10).sum()  / vol_sum20[common_t].replace(0, np.nan)

# ── Monthly factors dict ──────────────────────────────────────────────────────
factor_defs = {
    # v3 factors
    "mom_1m"         : resample_monthly(mom_1m),
    "mom_3m"         : resample_monthly(mom_3m),
    "mom_6m"         : resample_monthly(mom_6m),
    "rsi_14"         : resample_monthly(rsi_14),
    "vol_ratio"      : resample_monthly(vol_ratio),
    "rev_yoy"        : resample_monthly(rev_yoy_d),
    "rev_mom"        : resample_monthly(rev_mom_d),
    "foreign_net_20d": resample_monthly(foreign_net_20d),
    "trust_net_10d"  : resample_monthly(trust_net_10d),
    # NEW v4 factors
    "mom_1m_ra"      : resample_monthly(mom_1m_ra),     # risk-adj momentum
    "price_52w"      : resample_monthly(price_to_52w),  # 52w high ratio
    # "breakout"       : resample_monthly(breakout_signal), # DROP: Phase 3-A Noise Factor
    "atr_rel"        : resample_monthly(atr_rel),       # relative volatility
    "rev_accel"      : resample_monthly(rev_accel_d),   # revenue acceleration
}

print(f"  Total factors: {len(factor_defs)}")

# z-score + long format
factors_z = {name: zscore_xs(df) for name, df in factor_defs.items()}
combined_long = [to_long(df, name) for name, df in factors_z.items()]
X_raw = pd.concat(combined_long, axis=1)

# ─── 3. Target variable ───────────────────────────────────────────────────────
print("\n[3/7] Target & alignment…")

close_m     = resample_monthly(close)
monthly_ret = close_m.pct_change(1).shift(-1)         # next-month return
mkt_med     = monthly_ret.median(axis=1)
excess_ret  = monthly_ret.subtract(mkt_med, axis=0)   # excess return vs median
y_raw       = to_long(excess_ret, "excess_return")

# Align & clean
X_raw, y_raw = X_raw.align(y_raw, join="inner", axis=0)
X_raw = X_raw.replace([np.inf, -np.inf], np.nan)
y_raw = y_raw.replace([np.inf, -np.inf], np.nan)
mask  = X_raw.notna().all(axis=1) & y_raw.notna()
X, y  = X_raw[mask], y_raw[mask]

all_dates = X.index.get_level_values(0).unique().sort_values()
print(f"  Dataset: X={X.shape}, y={y.shape}")
print(f"  Date range: {all_dates[0].date()} ~ {all_dates[-1].date()}")
print(f"  Stocks: {X.index.get_level_values(1).nunique()} unique")

if X.empty:
    print("[ERROR] X is empty — check cache files")
    sys.exit(1)

# ─── 4. IC analysis ───────────────────────────────────────────────────────────
print("\n[4/7] IC analysis…")
ic_results = {}
for factor in X.columns:
    ics = []
    for d in all_dates:
        try:
            x_d = X.loc[d, factor].dropna()
            y_d = y.loc[d].reindex(x_d.index).dropna()
            x_d = x_d.reindex(y_d.index)
            if len(y_d) > 10:
                ic, _ = spearmanr(x_d, y_d)
                if not np.isnan(ic):
                    ics.append(ic)
        except Exception:
            pass
    s = pd.Series(ics)
    ic_results[factor] = {"IC Mean": s.mean(), "IC Std": s.std(),
                           "ICIR": s.mean()/(s.std()+1e-8), "IC>0%": (s>0).mean()}

ic_df = (pd.DataFrame(ic_results).T
           .sort_values("ICIR", key=abs, ascending=False))
print(ic_df.round(4).to_string())
print("\n  [NOTE] 上表為全資料探索用，實際因子選擇在各 fold 的訓練資料內進行（Phase 2 lookahead bias 修復）")
ALL_FACTORS = X.columns.tolist()  # 保留全部因子，fold 內自動篩選

# ─── 5. Walk-forward Purged training ─────────────────────────────────────────
print("\n[5/7] Walk-Forward Purged training...")

try:
    import lightgbm as lgb
except ImportError:
    print("[ERROR] Install lightgbm first"); sys.exit(1)

# FIX3: 強化 LightGBM 正則化，降低 Overfitting
LGBM_PARAMS = dict(
    n_estimators=500,        # 從 800 降至 500（避免過擬合）
    max_depth=3,             # 從 4 降至 3（更淺的樹）
    learning_rate=0.01,      # 從 0.008 略升（配合更少 estimators）
    num_leaves=15,           # 從 31 降至 15（對應 depth=3）
    reg_alpha=0.5,           # 從 0.1 升至 0.5（更強 L1 正則）
    reg_lambda=2.0,          # 從 1.0 升至 2.0（更強 L2 正則）
    min_child_samples=30,    # 從 20 升至 30（每葉節點至少30樣本）
    subsample=0.7,           # 從 0.8 降至 0.7（更多隨機性）
    colsample_bytree=0.7,    # 從 0.8 降至 0.7
    random_state=42,
    n_jobs=-1,
    verbose=-1,
)

test_dates = [d for d in all_dates if d >= TEST_START]
print(f"  Test period : {test_dates[0].strftime('%Y-%m')} ~ {test_dates[-1].strftime('%Y-%m')} ({len(test_dates)} months)")

all_preds         = []
current_train_end = None
model             = None
retrain_count     = 0
selected_factors  = {}   # 記錄每次 fold 選出的因子
current_fold_factors = ALL_FACTORS[:TOP_FACTORS_K]  # 第一次預测前的預設

for d in tqdm(test_dates, desc="  Walk-Forward"):
    # Retrain?
    if current_train_end is None or d >= current_train_end:
        train_start    = d - pd.DateOffset(months=TRAIN_MONTHS)
        purge_cutoff   = d - pd.DateOffset(months=PURGE_MONTHS)
        idx_tr = (
            (X.index.get_level_values(0) >= train_start) &
            (X.index.get_level_values(0) <  purge_cutoff)
        )
        X_tr, y_tr = X[idx_tr], y[idx_tr]
        if len(X_tr) < 200:
            continue

        # ── Fold-Internal IC 分析：在訓練集內選因子（Phase 2 修復 lookahead bias）
        # 第一個 fold 用全部因子，後續每次在訓練集內重新計算 ICIR
        fold_ic = {}
        train_dates = X_tr.index.get_level_values(0).unique()
        for factor in ALL_FACTORS:
            ics_f = []
            for dt in train_dates:
                try:
                    xf = X_tr.loc[dt, factor].dropna() if (dt, factor) in X_tr.columns or True else pd.Series()
                    # 當 X_tr 是寬表，用 xs 取得特定日期
                    xf = X_tr.xs(dt, level=0)[factor].dropna()
                    yf = y_tr.xs(dt, level=0).reindex(xf.index).dropna()
                    xf = xf.reindex(yf.index)
                    if len(yf) > 10:
                        ic_val, _ = spearmanr(xf.values, yf.values)
                        if not np.isnan(ic_val):
                            ics_f.append(ic_val)
                except Exception:
                    pass
            s_f = pd.Series(ics_f)
            fold_ic[factor] = abs(s_f.mean() / (s_f.std() + 1e-8)) if len(s_f) > 3 else 0.0

        current_fold_factors = sorted(fold_ic, key=fold_ic.get, reverse=True)[:TOP_FACTORS_K]
        selected_factors[d] = current_fold_factors

        # Early-stopping on last 20% of train as val
        cut    = int(len(X_tr) * 0.8)
        X_t_   = X_tr[current_fold_factors].iloc[:cut]
        y_t_   = y_tr.iloc[:cut]
        X_v_   = X_tr[current_fold_factors].iloc[cut:]
        y_v_   = y_tr.iloc[cut:]

        model = lgb.LGBMRegressor(**LGBM_PARAMS)
        model.fit(
            X_t_, y_t_,
            eval_set=[(X_v_, y_v_)],
            callbacks=[lgb.early_stopping(50, verbose=False),
                       lgb.log_evaluation(period=-1)],
        )
        current_train_end = d + pd.DateOffset(months=STEP_MONTHS)
        retrain_count += 1

    # Predict—用這個 fold 的因子對當日做預測
    X_d = X[X.index.get_level_values(0) == d][current_fold_factors]
    if model is not None and not X_d.empty:
        preds_d = model.predict(X_d)
        all_preds.append(pd.DataFrame({"y_pred": preds_d}, index=X_d.index))

final_preds = pd.concat(all_preds)
print(f"\n  {len(final_preds):,} predictions — {retrain_count} retrain cycles")

# 因子穩定性報告：統計各因子被選中的次數
from collections import Counter
factor_counts = Counter(f for factors in selected_factors.values() for f in factors)
print("\n  Factor selection stability (out of", retrain_count, "retrains):")
for factor, count in sorted(factor_counts.items(), key=lambda x:-x[1]):
    bar = "#" * count
    print(f"    {factor:<20} {count:>2}/{retrain_count}  {bar}")

# Feature importance（最後一次 fold 的模型）
fi = (pd.DataFrame({"Factor": current_fold_factors,
                     "Importance": model.feature_importances_})
        .sort_values("Importance", ascending=False))
print("\n  Feature Importance (last model):")
print(fi.to_string(index=False))

# ─── 6. Confidence-weighted positions ────────────────────────────────────────
print("\n[6/7] Building confidence-weighted positions…")

pred_wide = final_preds["y_pred"].unstack("stock_id")

# ══ Multi-signal Regime Filter ═══════════════════════════════════════════════
tw50 = close["0050"]
sig_60ma  = (tw50 > tw50.rolling(60).mean()).astype(float) * 0.4   # long-term trend
sig_20ma  = (tw50 > tw50.rolling(20).mean()).astype(float) * 0.3   # medium-term
sig_nodip = (tw50.pct_change(1) > -0.05).astype(float) * 0.3      # no big monthly drop
regime_score = sig_60ma + sig_20ma + sig_nodip
# Enter when 2 of 3 signals agree (score ≥ 0.4)
is_bull_daily   = regime_score >= 0.4
is_bull_monthly = (is_bull_daily.resample("ME").mean()
                                .reindex(pred_wide.index, method="ffill")
                                .fillna(0) >= 0.4)

bull_months = int(is_bull_monthly.sum())
print(f"  Bull months: {bull_months} / {len(is_bull_monthly)} "
      f"({bull_months/len(is_bull_monthly):.0%})  ← was 54% in v3")

# ══ Softmax confidence weighting ══════════════════════════════════════════════
def softmax_weights(scores: pd.Series, temp: float, top_k: int) -> pd.Series:
    """Select top_k stocks; weight by softmax(score / temp)."""
    top = scores.nlargest(top_k)
    exp_  = np.exp((top - top.max()) / temp)   # numerical stable
    return exp_ / exp_.sum()

# ✅ FIX2: 預計算月度流動性（30日均量，月度取最後一天）
money_m_liq = money_wide.rolling(30).mean().resample("ME").last()

weight_rows = {}
for date, row in pred_wide.iterrows():
    row = row.dropna()

    # ── 流動性過濾：只保留 30日均量 > 3000萬的股票 ──────────────────────────
    if date in money_m_liq.index:
        liq_row  = money_m_liq.loc[date].reindex(row.index)
        liq_ok   = liq_row[liq_row >= LIQUIDITY_MIN].index
        filtered = len(row) - len(liq_ok)
        row      = row[liq_ok]
        if filtered > 0:
            pass  # silent: print(f"  {date.date()}: 流動性過濾移除 {filtered} 支")

    if is_bull_monthly.get(date, False) and len(row) >= TOP_K:
        weight_rows[date] = softmax_weights(row, WEIGHT_TEMP, TOP_K)
    else:
        weight_rows[date] = pd.Series(dtype=float)

# Wide weight table: rows=months, cols=stocks (0 for not held)
all_stocks   = pred_wide.columns
weights_df   = pd.DataFrame(weight_rows).T.reindex(columns=all_stocks).fillna(0.0)

active_months = (weights_df.sum(axis=1) > 0).sum()
print(f"  Active months: {active_months} / {len(weights_df)}")

# ─── 7. Vectorbt backtest ────────────────────────────────────────────────────
print("\n[7/7] Running vectorbt backtest…")

import vectorbt as vbt

# Expand monthly weights to daily (forward-fill)
weights_daily = weights_df.reindex(close.index, method="ffill").fillna(0.0)

# Use common clean stocks
close_clean = (close.loc[TEST_START:]
               .replace(0, np.nan)
               .ffill().bfill()
               .dropna(axis=1))
common      = close_clean.columns.intersection(weights_daily.columns)
c_vbt       = close_clean[common]
w_vbt       = weights_daily.loc[TEST_START:, common]

# Benchmark 0050
bm_close    = close_clean["0050"] if "0050" in close_clean.columns else close_clean.iloc[:, 0]

print(f"  Universe : {c_vbt.shape[1]} stocks × {c_vbt.shape[0]} days")
print(f"  Running…")

pf = vbt.Portfolio.from_orders(
    close       = c_vbt,
    size        = w_vbt,
    size_type   = "targetpercent",
    group_by    = True,
    cash_sharing= True,
    fees        = FEE,
    slippage    = 0.001,
    init_cash   = INIT_CASH,
)


# ─── Print stats ─────────────────────────────────────────────────────────────
stats = pf.stats()
eq    = pf.value()   # 先定義 eq，benchmark 計算需要用到 eq.index[-1]

# ── Benchmark：使用 yfinance 抓還原權後的 0050 報酬
import yfinance as yf, contextlib, io as _io

print("  Fetching 0050 benchmark from yfinance (split-adjusted)...")
_start_str = TEST_START.strftime("%Y-%m-%d")
_end_str   = eq.index[-1].strftime("%Y-%m-%d")
with contextlib.redirect_stdout(_io.StringIO()), contextlib.redirect_stderr(_io.StringIO()):
    _bm_raw = yf.download("0050.TW", start=_start_str, end=_end_str,
                           progress=False, auto_adjust=True)
bm_price    = _bm_raw["Close"].squeeze().dropna()
n_years_bm  = (bm_price.index[-1] - bm_price.index[0]).days / 365.25
cagr_bm     = float((bm_price.iloc[-1] / bm_price.iloc[0]) ** (1/n_years_bm) - 1)
total_ret_bm = float((bm_price.iloc[-1] / bm_price.iloc[0]) - 1)
bm_ec       = bm_price / bm_price.iloc[0]
max_dd_bm   = float(((bm_ec - bm_ec.cummax()) / bm_ec.cummax()).min())
bm_monthly  = bm_price.resample("ME").last().pct_change().dropna()
sharpe_bm   = float((bm_monthly.mean() / bm_monthly.std()) * (12**0.5))
eq_bm       = bm_ec * INIT_CASH
print(f"  0050  CAGR: {cagr_bm:.2%} | Total Ret: {total_ret_bm:.2%} | Sharpe: {sharpe_bm:.2f}")


COMPARE_KEYS = [
    ("Total Return [%]",         "Total Return"),
    ("Max Drawdown [%]",         "Max Drawdown"),
    ("Max Drawdown Duration",    "DD Duration (days)"),
]

n_years = (eq.index[-1] - eq.index[0]).days / 365.25
cagr = (eq.iloc[-1] / INIT_CASH) ** (1/n_years) - 1

monthly_eq   = eq.resample("ME").last()
monthly_ret_ = monthly_eq.pct_change().dropna()
sharpe    = (monthly_ret_.mean() / monthly_ret_.std()) * (12**0.5)

print()
print("  ╔══════════════════════════════════════════════════════╗")
print("  ║   Strategy vs Benchmark (0050) — vectorbt        ║")
print("  ╠════════════════════╦═══════════════╦════════════════╣")
print("  ║ Metric             ║  Strategy  ║  Benchmark     ║")
print("  ╠════════════════════╬═══════════════╬════════════════╣")

rows = [
    ("CAGR",                  f"{cagr:.2%}",                         f"{cagr_bm:.2%}"),
    ("Total Return",          f"{stats['Total Return [%]']:.2f}%",   f"{total_ret_bm*100:.2f}%"),
    ("Max Drawdown",          f"{stats['Max Drawdown [%]']:.2f}%",   f"{max_dd_bm*100:.2f}%"),
    ("Sharpe (Monthly Ann.)", f"{sharpe:.2f}",                        f"{sharpe_bm:.2f}"),
    ("Active Months",         f"{active_months} / {len(weights_df)}", "N/A (fully invested)"),
]
for label, s_val, b_val in rows:
    try:
        s_num = float(s_val.rstrip("% "))
        b_num = float(b_val.rstrip("% "))
        if label == "Max Drawdown":
            win = "*" if s_num > b_num else " "
        elif label in ["CAGR", "Total Return", "Sharpe (Monthly Ann.)"]:
            win = "*" if s_num > b_num else " "
        else:
            win = " "
    except Exception:
        win = " "
    print(f"  \u2551 {label:<18} \u2551 {s_val:>12} {win} \u2551 {b_val:>22} \u2551")

print("  ╚════════════════════╩═══════════════╩════════════════╝")

# v5 vs comparison
print()
print("  ── v5 vs comparison (vectorbt, same period) ──")
print(f"  v5  Total Return : 59.97%    Max DD: 22.53%   Sharpe: 0.90")
print(f"   Total Return : {stats['Total Return [%]']:.2f}%    "
      f"Max DD: {stats['Max Drawdown [%]']:.2f}%   "
      f"Sharpe: {sharpe:.2f}")

# Save predictions for report reuse
final_preds.to_pickle("predictions.pkl")
weights_df.to_pickle("weights.pkl")
eq.to_pickle("eq.pkl")
eq_bm.to_pickle("bm_eq.pkl")
print("\n  Saved: predictions.pkl / weights.pkl / eq.pkl")

print("\nDone!")
import subprocess
subprocess.run(["python3", "reports/generate_report.py"])

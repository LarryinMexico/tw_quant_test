#!/usr/bin/env python3
"""
live_trade.py
=============
虛擬基金（紙上交易）核心引擎。
讀取 portfolio.json 與 weights.pkl 產出的訊號，進行淨值結算與選股推播。

費用模型:
  買進: 0.1425% 手續費 (實際打折後約 0.05~0.1%, 這裡用 0.1425%)
  賣出: 0.1425% 手續費 + 0.3% 交易稅
  滑價: 0.1% (市場衝擊)

風險控制:
  最多投入 90% 資金 (保留 10% 現金緩衝)
"""

import os
import json
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

CACHE_DIR      = "finmind_cache"
PORTFOLIO_FILE = "portfolio.json"

# 費用參數
FEE_RATE          = 0.001425   # 0.1425% 手續費
TAX_RATE          = 0.003      # 0.3% 交易稅（賣出）
SLIPPAGE          = 0.001      # 0.1% 滑價（雙向）
# 風險控制
MAX_INVEST_RATIO  = 0.90       # 最多投入 90% 資金（保留 10% 現金緩衝）
MAX_SINGLE_WEIGHT = 0.08       # 單支股票最多 8%
# GitHub Pages
GITHUB_PAGES_URL = "https://larryinmexico.github.io/tw_quant_test/"

LINE_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_USER  = os.getenv("LINE_USER_ID")

def send_line_message(text):
    if not LINE_TOKEN or not LINE_USER:
        print("未設定 LINE Token，跳過推播")
        return
    url     = "https://api.line.me/v2/bot/message/push"
    headers = {
        "Authorization": f"Bearer {LINE_TOKEN}",
        "Content-Type":  "application/json"
    }
    payload = {
        "to":       LINE_USER,
        "messages": [{"type": "text", "text": text}]
    }
    try:
        res = requests.post(url, headers=headers, json=payload, timeout=10)
        if res.status_code == 200:
            print("LINE 推播成功")
        else:
            print(f"LINE 推播失敗 {res.status_code} {res.text}")
    except Exception as e:
        print(f"網路錯誤: {e}")


# ─── 1. 取得股票中文名稱 ────────────────────────────────────────────────────────
def fetch_stock_names(stock_ids):
    name_map = {}
    try:
        resp = requests.get(
            "https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockInfo",
            timeout=10
        )
        df   = pd.DataFrame(resp.json()["data"])
        ids  = [str(s) for s in stock_ids]
        df   = df[df["stock_id"].isin(ids)].drop_duplicates("stock_id")
        name_map = dict(zip(df["stock_id"], df["stock_name"]))
    except Exception as e:
        print(f"[WARN] 無法取得股票名稱: {e}")
    return name_map


# ─── 2. 讀取策略訊號 ────────────────────────────────────────────────────────────
print("1 讀取資料與策略訊號")
weights_df     = pd.read_pickle("weights.pkl")
target_weights = weights_df.iloc[-1]
target_weights = target_weights[target_weights > 0]

# ─── 3. 讀取虛擬存摺 ────────────────────────────────────────────────────────────
if not os.path.exists(PORTFOLIO_FILE):
    pf = {
        "cash":             1_000_000.0,
        "last_trade_date":  None,
        "positions":        {},
        "history":          [],
        "trade_log_history": [],
    }
else:
    with open(PORTFOLIO_FILE, "r") as f:
        pf = json.load(f)
    if "trade_log_history" not in pf:
        pf["trade_log_history"] = []


# ─── 4. 獲取最新股價 (yfinance) ─────────────────────────────────────────────────
def get_latest_prices(stock_ids):
    import yfinance as yf
    try:
        url  = "https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockInfo"
        resp = requests.get(url, timeout=10)
        info = pd.DataFrame(resp.json()["data"])
        twse = set(info[info["type"] == "twse"]["stock_id"])
        tpex = set(info[info["type"] == "tpex"]["stock_id"])
    except:
        twse = set()
        tpex = set()

    tickers = []
    mapping = {}
    for s in stock_ids:
        if str(s) in twse:
            t = f"{s}.TW"
        elif str(s) in tpex:
            t = f"{s}.TWO"
        else:
            t = f"{s}.TW"
        tickers.append(t)
        mapping[t] = str(s)

    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        data = yf.download(tickers, period="5d", progress=False)

    if "Close" not in data.columns:
        if len(tickers) == 1:
            close_df = pd.DataFrame({tickers[0]: data["Close"]})
        else:
            raise ValueError("Failed to fetch prices")
    else:
        close_df = data["Close"]

    close_df    = close_df.dropna(axis=1, how="all")
    latest_date = close_df.index[-1]
    last_row    = close_df.iloc[-1]
    last_row.index = [mapping.get(x, x) for x in last_row.index]
    return latest_date, last_row.dropna()


stock_ids              = list(set(target_weights.index) | set(pf["positions"].keys()))
latest_date, latest_prices = get_latest_prices(stock_ids)

# 取得中文名稱
name_map = fetch_stock_names(stock_ids)

# ─── 5. 結算淨值 ─────────────────────────────────────────────────────────────────
print(f"2 結算淨值，最新收盤日: {latest_date.date()}")

equity_value = 0.0
for stock, shares in pf["positions"].items():
    if stock in latest_prices:
        equity_value += latest_prices[stock] * shares
    else:
        print(f"  [WARN] 找不到 {stock} 報價，資產可能低估")

current_nav = pf["cash"] + equity_value

# ─── 6. 大盤空頭濾網 ─────────────────────────────────────────────────────────────
is_bull_market = True
try:
    import yfinance as yf
    tw50_data = yf.download("0050.TW", period="100d", progress=False)
    if not tw50_data.empty and "Close" in tw50_data.columns:
        tw50_close = tw50_data["Close"]
        if isinstance(tw50_close, pd.DataFrame):
            tw50_close = tw50_close.iloc[:, 0]
        tw50_ma60  = tw50_close.rolling(60).mean().iloc[-1]
        tw50_latest = tw50_close.iloc[-1]
        is_bull_market = bool(tw50_latest > tw50_ma60)
except Exception as e:
    print(f"  空頭濾網檢測失敗，維持現狀: {e}")

if not is_bull_market:
    print("觸發大盤空頭濾網 (0050 跌破季線)，啟動緊急清倉避險！")
    target_weights = pd.Series(dtype=float)

# ─── 7. 換股邏輯 ─────────────────────────────────────────────────────────────────
# 8% 單支持股上限：weight cap 後重新標準化
if not target_weights.empty:
    target_weights = target_weights.clip(upper=MAX_SINGLE_WEIGHT)
    target_weights = target_weights / target_weights.sum()  # 重新正規化

current_stocks   = set(pf["positions"].keys())
target_stocks    = set(target_weights.index[target_weights > 0]) if not target_weights.empty else set()

# force_rebalance 旗標：在 portfolio.json 中設 "force_rebalance": true 可強制換倉
# 用途：更換模型、調整 weight cap 等參數後手動觸發（執行後自動清除）
force_rebalance  = pf.get("force_rebalance", False)
if force_rebalance:
    print("  [force_rebalance=true] 強制換倉模式，換倉後將自動清除旗標")
    pf["force_rebalance"] = False  # 執行後自動清除，避免每天都換

is_rebalance_day = force_rebalance or \
                   (current_stocks != target_stocks) or \
                   (len(pf["positions"]) == 0 and not target_weights.empty)

trade_logs = []
session_pnl = None  # 本次換倉的損益

if is_rebalance_day:
    print(f"  觸發部位調整 (目前現金: {pf['cash']:,.0f})")
    sell_revenue = 0.0
    sell_cost_basis = 0.0  # 用於估算損益（買入成本以 portfolio 記錄為準）

    # ── 全部平倉（加入手續費 + 稅 + 滑價）
    for stock, shares in list(pf["positions"].items()):
        if stock in latest_prices:
            raw_price   = latest_prices[stock]
            slip_price  = raw_price * (1 - SLIPPAGE)  # 賣出滑價（略低於市價）
            revenue     = shares * slip_price
            fee_tax     = revenue * (FEE_RATE + TAX_RATE)
            net_revenue = revenue - fee_tax
            pf["cash"] += net_revenue
            sell_revenue += net_revenue
            cn_name = name_map.get(str(stock), stock)
            trade_logs.append(f"賣出 {stock} {cn_name} {shares}股 @{slip_price:.1f} 淨收 {net_revenue:,.0f}")
    pf["positions"] = {}
    current_nav = pf["cash"]  # 全回現金後更新基準

    # ── 再建倉（風控：最多投入 MAX_INVEST_RATIO = 90%）
    if not target_weights.empty:
        investable_cash = current_nav * MAX_INVEST_RATIO
        buy_total_cost  = 0.0
        for stock, w in target_weights.items():
            if stock not in latest_prices:
                continue
            budget    = investable_cash * w
            raw_price = latest_prices[stock]
            slip_price = raw_price * (1 + SLIPPAGE)  # 買入滑價（略高）
            shares    = int(budget // slip_price)
            cost      = shares * slip_price
            fee       = cost * FEE_RATE

            if shares > 0 and (pf["cash"] - cost - fee) >= 0:
                pf["positions"][stock] = shares
                pf["cash"] -= (cost + fee)
                buy_total_cost += (cost + fee)
                cn_name = name_map.get(str(stock), stock)
                trade_logs.append(f"買進 {stock} {cn_name} {shares}股 @{slip_price:.1f} 成本 {cost+fee:,.0f}")

        # 估算損益（本次賣出淨收 - 本次買進成本）
        session_pnl = sell_revenue - buy_total_cost if sell_revenue > 0 else None

    pf["last_trade_date"] = str(latest_date.date())

    # 記錄到 trade_log_history
    pf["trade_log_history"].append({
        "date": str(latest_date.date()),
        "logs": trade_logs[:20],
        "pnl":  round(session_pnl, 2) if session_pnl is not None else None,
    })

# ─── 8. 記錄歷史 NAV ──────────────────────────────────────────────────────────────
# 重新計算真實 NAV（含新倉位）
equity_value = sum(
    latest_prices.get(stock, 0) * shares
    for stock, shares in pf["positions"].items()
)
current_nav = pf["cash"] + equity_value

if pf["history"] and pf["history"][-1]["date"] == str(latest_date.date()):
    pf["history"][-1]["nav"] = current_nav
else:
    pf["history"].append({"date": str(latest_date.date()), "nav": current_nav})

pf["latest_date"]   = str(latest_date.date())
pf["latest_prices"] = latest_prices.to_dict()

with open(PORTFOLIO_FILE, "w") as f:
    json.dump(pf, f, indent=2, ensure_ascii=False)

# ─── 9. LINE 推播 ─────────────────────────────────────────────────────────────────
print("3 發送通知")

nav_change_str = ""
if len(pf["history"]) >= 2:
    prev_nav   = pf["history"][-2]["nav"]
    nav_change = (current_nav / prev_nav) - 1
    nav_change_str = f"日報酬 {nav_change:+.2%}\n"

# 現金比例警示
cash_ratio = pf["cash"] / current_nav
cash_warn  = "" if cash_ratio >= 0.05 else "現金不足5%，流動性偏緊\n"

# 持股前5大（含中文名稱）
top_holdings = sorted(
    pf["positions"].items(),
    key=lambda x: x[1] * latest_prices.get(x[0], 0),
    reverse=True
)[:5]

top5_lines = ""
for stock, shares in top_holdings:
    val      = shares * latest_prices.get(stock, 0)
    cn_name  = name_map.get(str(stock), "")
    top5_lines += f"  {stock} {cn_name} {shares}股 {val:,.0f}\n"

pnl_line = ""
if session_pnl is not None:
    sign = "+" if session_pnl >= 0 else ""
    pnl_line = f"換倉損益估算 {sign}{session_pnl:,.0f}\n"

msg = f"""模擬交易 
日期 {latest_date.date()}
總淨值 {current_nav:,.0f}
剩餘現金 {pf['cash']:,.0f} ({cash_ratio:.1%})
{nav_change_str}{pnl_line}{cash_warn}
目前持股前5大:
{top5_lines}
Dashboard: {GITHUB_PAGES_URL}"""

if trade_logs:
    msg += f"\n今日交易紀錄 ({len(trade_logs)}筆)\n"
    msg += "\n".join(trade_logs[:5])
    if len(trade_logs) > 5:
        msg += f"\n...等{len(trade_logs)}筆"

send_line_message(msg)
print("\n今日實盤模擬完成")
print(f"   NAV: {current_nav:,.0f}  現金: {pf['cash']:,.0f} ({cash_ratio:.1%})")
if session_pnl is not None:
    print(f"   換倉損益估算: {session_pnl:+,.0f}")

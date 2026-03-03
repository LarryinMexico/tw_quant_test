#!/usr/bin/env python3
"""
live_trade.py
=============
虛擬基金（紙上交易）核心引擎。
讀取 portfolio.json 與 strategy_v6.py 產出的訊號，進行淨值結算與選股推播。
"""

import os
import json
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

CACHE_DIR = "finmind_cache"
PORTFOLIO_FILE = "portfolio.json"

# 1. Load Telegram/LINE logic
LINE_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
LINE_USER  = os.getenv("LINE_USER_ID")

def send_line_message(text):
    if not LINE_TOKEN or not LINE_USER:
        print("未設定LINEToken跳過推播")
        return
    url = "https://api.line.me/v2/bot/message/push"
    headers = {
        "Authorization": f"Bearer {LINE_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "to": LINE_USER,
        "messages": [{"type": "text", "text": text}]
    }
    try:
        res = requests.post(url, headers=headers, json=payload, timeout=10)
        if res.status_code == 200:
            print("LINE推播成功")
        else:
            print(f"LINE推播失敗 {res.status_code} {res.text}")
    except Exception as e:
        print(f"網路錯誤 {e}")

print("1 讀取資料與策略訊號")
close_wide = pd.read_pickle(os.path.join(CACHE_DIR, "close_wide.pkl"))
latest_date = close_wide.index[-1]
latest_prices = close_wide.loc[latest_date].dropna()

weights_df = pd.read_pickle("weights_v6.pkl")
# 取出策略「最新一天」的目標權重（通常是月底產生的下個月權重）
target_weights = weights_df.iloc[-1]
target_weights = target_weights[target_weights > 0]

# 3. 讀取虛擬存摺
if not os.path.exists(PORTFOLIO_FILE):
    # 如果完全沒有檔，給予 100 萬起始本金
    pf = {"cash": 1000000.0, "last_trade_date": None, "positions": {}, "history": []}
else:
    with open(PORTFOLIO_FILE, "r") as f:
        pf = json.load(f)

print(f"2 結算淨值 最新收盤日 {latest_date.date()}")

equity_value = 0.0
for stock, shares in pf["positions"].items():
    if stock in latest_prices:
        equity_value += latest_prices[stock] * shares
    else:
        print(f"找不到 {stock} 的今日報價資產可能低估")

current_nav = pf["cash"] + equity_value

# 判斷是否需要「換股」(建倉)
# 如果部位是空的，或是換月了，我們就執行交易（簡化版：目前如果是空手就無腦買）
is_rebalance_day = len(pf["positions"]) == 0

trade_logs = []
if is_rebalance_day:
    print(f"  觸發建倉換股邏輯 分配 {current_nav:,.0f} 資金")
    pf["cash"] = current_nav
    pf["positions"] = {}
    
    investable_cash = current_nav * 0.98
    
    for stock, w in target_weights.items():
        if stock not in latest_prices:
            continue
        budget = investable_cash * w
        price = latest_prices[stock]
        shares = int(budget // price)
        cost = shares * price
        
        if shares > 0:
            pf["positions"][stock] = shares
            pf["cash"] -= cost
            fee = cost * 0.001425
            pf["cash"] -= fee
            trade_logs.append(f"買進 {stock} {shares}股 {price:.1f}")
            
    pf["last_trade_date"] = str(latest_date.date())

# 紀錄歷史淨值
pf["history"].append({"date": str(latest_date.date()), "nav": current_nav})

with open(PORTFOLIO_FILE, "w") as f:
    json.dump(pf, f, indent=2, ensure_ascii=False)

print("3 發送通知")

nav_change_str = ""
if len(pf["history"]) >= 2:
    prev_nav = pf["history"][-2]["nav"]
    nav_change = (current_nav / prev_nav) - 1
    nav_change_str = f"📈 日報酬 {nav_change:+.2%}\n"

msg = f"""📊 台股ML虛擬基金 v6
📅 日期 {latest_date.date()}
💰 總淨值 {current_nav:,.0f}
💵 剩餘現金 {pf['cash']:,.0f}
{nav_change_str}
📦 目前持股 前5大
"""

top_holdings = sorted(pf["positions"].items(), key=lambda x: x[1]*latest_prices.get(x[0], 0), reverse=True)[:5]
for stock, shares in top_holdings:
    val = shares * latest_prices.get(stock, 0)
    msg += f"- {stock} {shares}股 {val:,.0f}\n"

if trade_logs:
    msg += f"\n🚨 今日交易紀錄\n" + "\n".join(trade_logs[:5]) + ("\n等" if len(trade_logs)>5 else "")

send_line_message(msg)
print("\n今日實盤模擬完成")

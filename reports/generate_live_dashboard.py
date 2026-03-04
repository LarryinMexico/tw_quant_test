"""
generate_live_dashboard.py
==========================
生成 TradingView 風格 Live Dashboard（純深色主題）。
功能:
  - 持股中文名稱顯示
  - 圓餅圖：Bloomberg/TradingView 藍色系專業配色
  - 圓餅圖與持股清單不重疊
  - 頁面底部：每日 NAV 歷史 & 交易損益明細
  - 全面 RWD 手機版面適應
"""

import json
import pandas as pd
import plotly.graph_objects as go
import os
import requests

# ─── 讀取資料 ─────────────────────────────────────────────────────────────────
with open("portfolio.json", "r") as f:
    pf = json.load(f)

latest_date       = pf.get("latest_date", "N/A")
latest_prices     = pf.get("latest_prices", {})
history           = pf.get("history", [])
dates             = [row["date"] for row in history]
navs              = [row["nav"]  for row in history]
trade_log_history = pf.get("trade_log_history", [])

# ─── 統計計算 ──────────────────────────────────────────────────────────────────
current_nav = navs[-1] if navs else pf.get("cash", 1_000_000.0)
prev_nav    = navs[-2] if len(navs) >= 2 else 1_000_000.0
init_nav    = 1_000_000.0
daily_ret   = (current_nav / prev_nav) - 1
total_ret   = (current_nav / init_nav) - 1
cash        = pf.get("cash", 0)
positions   = pf.get("positions", {})

GITHUB_PAGES_URL = "https://larryinmexico.github.io/tw_quant_test/"

# ─── 取得股票中文名稱 ─────────────────────────────────────────────────────────
def fetch_stock_names(stock_ids):
    name_map = {}
    try:
        resp = requests.get(
            "https://api.finmindtrade.com/api/v4/data?dataset=TaiwanStockInfo",
            timeout=10
        )
        df = pd.DataFrame(resp.json()["data"])
        df = df[df["stock_id"].isin([str(s) for s in stock_ids])].drop_duplicates("stock_id")
        name_map = dict(zip(df["stock_id"], df["stock_name"]))
    except Exception as e:
        print(f"[WARN] 無法取得股票名稱: {e}")
    return name_map

name_map = fetch_stock_names(list(positions.keys()))

# ─── 持股資料整理 ─────────────────────────────────────────────────────────────
stock_data   = []
total_equity = 0
for stock, shares in positions.items():
    price = latest_prices.get(stock, 0)
    val   = shares * price
    total_equity += val
    stock_data.append({
        "stock":  stock,
        "name":   name_map.get(stock, stock),
        "shares": shares,
        "price":  price,
        "value":  val,
    })

stock_data.sort(key=lambda x: x["value"], reverse=True)
for item in stock_data:
    item["weight"] = item["value"] / current_nav if current_nav > 0 else 0

# ─── 顏色常數 ─────────────────────────────────────────────────────────────────
TV_BG    = "#131722"
TV_PANEL = "#1E222D"
TV_GRID  = "#2B3139"
TV_TEXT  = "#D1D4DC"
TV_MUTED = "#8D94A6"
TV_GREEN = "#22AB94"
TV_RED   = "#F05350"
TV_BLUE  = "#2962FF"

# Bloomberg / TradingView 風格配色 — 藍色系 + 少量互補色
CHART_COLORS = [
    "#2962FF",  # 主藍
    "#1565C0",  # 深藍
    "#42A5F5",  # 淺藍
    "#00BCD4",  # 青藍
    "#26C6DA",  # 淺青
    "#00ACC1",  # 深青
    "#29B6F6",  # 天藍
    "#0288D1",  # 標準藍
    "#4CAF50",  # 綠（互補）
    "#00897B",  # 深青綠
    "#7E57C2",  # 紫（互補）
    "#9575CD",  # 淺紫
    "#FFA726",  # 金（警示）
    "#FF7043",  # 橙紅（警示）
    "#546E7A",  # 灰藍
    "#78909C",  # 淺灰藍
    "#5C6BC0",  # 靛藍
    "#3949AB",  # 深靛藍
    "#1E88E5",  # 亮藍
    "#039BE5",  # 鮮藍
]

# ─── Plotly 圖表 ──────────────────────────────────────────────────────────────

# NAV 折線圖
fig_line = go.Figure()
fig_line.add_trace(go.Scatter(
    x=dates, y=navs, mode="lines",
    line=dict(color=TV_BLUE, width=2),
    fill="tozeroy", fillcolor="rgba(41, 98, 255, 0.15)",
    hoverinfo="x+y"
))
fig_line.update_layout(
    margin=dict(l=50, r=20, t=20, b=40),
    plot_bgcolor=TV_PANEL, paper_bgcolor=TV_PANEL,
    font=dict(color=TV_TEXT),
    xaxis=dict(showgrid=True, gridcolor=TV_GRID, tickcolor=TV_MUTED),
    yaxis=dict(showgrid=True, gridcolor=TV_GRID, tickformat=",.0f",
               side="right", tickcolor=TV_MUTED),
    hovermode="x unified"
)
html_line = fig_line.to_html(
    include_plotlyjs="cdn", full_html=False,
    config={"displayModeBar": False}
)

# 圓餅圖（donut）— 專業配色
labels = ["Cash"] + [s["stock"] for s in stock_data]
values = [cash]   + [s["value"] for s in stock_data]
pie_colors = [TV_MUTED] + [CHART_COLORS[i % len(CHART_COLORS)] for i in range(len(stock_data))]

fig_pie = go.Figure(data=[go.Pie(
    labels=labels, values=values, hole=.65,
    marker=dict(colors=pie_colors, line=dict(color="#131722", width=1.5)),
    textinfo="none", hoverinfo="label+percent"
)])
fig_pie.update_layout(
    showlegend=False,
    margin=dict(l=8, r=8, t=8, b=8),
    plot_bgcolor=TV_PANEL, paper_bgcolor=TV_PANEL,
    font=dict(color=TV_TEXT),
    height=185,
)
html_pie = fig_pie.to_html(
    include_plotlyjs=False, full_html=False,
    config={"displayModeBar": False}
)

# ─── Holdings Table HTML ──────────────────────────────────────────────────────
holdings_html = ""
for item in stock_data:
    color = CHART_COLORS[stock_data.index(item) % len(CHART_COLORS)]
    holdings_html += f"""
    <div class="table-row">
        <div class="col-ticker">
            <span class="stock-id">{item['stock']}</span>
            <span class="stock-name">{item['name']}</span>
        </div>
        <div class="col-right">{item['price']:.2f}</div>
        <div class="col-right">{item['shares']}</div>
        <div class="col-right">${item['value']:,.0f}</div>
        <div class="col-right" style="color:{color}; font-weight:600;">{item['weight'] * 100:.1f}%</div>
    </div>"""

# ─── 每日 NAV 歷史 ────────────────────────────────────────────────────────────
history_table_rows = ""
display_history = history[-30:]
for i, row in enumerate(reversed(display_history)):
    date_str = row["date"]
    nav_val  = row["nav"]
    # 找前一天 NAV
    global_idx = history.index(row)
    prev_val   = history[global_idx - 1]["nav"] if global_idx > 0 else init_nav
    day_ret    = (nav_val / prev_val - 1) * 100
    color      = TV_GREEN if day_ret >= 0 else TV_RED
    sign       = "+" if day_ret >= 0 else ""
    history_table_rows += f"""
    <tr>
        <td>{date_str}</td>
        <td>${nav_val:,.0f}</td>
        <td style="color:{color};">{sign}{day_ret:.2f}%</td>
    </tr>"""

# ─── 交易損益明細 ─────────────────────────────────────────────────────────────
trade_detail_html = ""
if trade_log_history:
    for entry in reversed(trade_log_history[-10:]):
        d        = entry.get("date", "")
        pnl      = entry.get("pnl", None)
        logs     = entry.get("logs", [])
        pnl_color = TV_GREEN if pnl and pnl >= 0 else TV_RED
        pnl_str  = f'　損益: <span style="color:{pnl_color};">{"+" if pnl and pnl>=0 else ""}{pnl:,.0f}</span>' if pnl is not None else ""
        logs_li  = "".join(f"<li>{l}</li>" for l in logs)
        trade_detail_html += f"""
        <div class="trade-entry">
            <div class="trade-date">{d}{pnl_str}</div>
            <ul class="trade-list">{logs_li}</ul>
        </div>"""
else:
    trade_detail_html = '<p style="color:#8D94A6; padding:16px; font-size:12px;">尚無交易紀錄</p>'

# ─── 格式化指標 ────────────────────────────────────────────────────────────────
daily_color = TV_GREEN if daily_ret >= 0 else TV_RED
total_color = TV_GREEN if total_ret >= 0 else TV_RED
daily_sign  = "+" if daily_ret >= 0 else ""
total_sign  = "+" if total_ret >= 0 else ""

# ─── 組合完整 HTML ─────────────────────────────────────────────────────────────
html_content = f"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0">
    <title>台股ML虛擬基金 Dashboard</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0;
             font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }}
        body {{ background: #131722; color: #D1D4DC; padding: 16px;
                font-size: 14px; min-height: 100vh; }}

        /* ── Layout ── */
        .grid-main {{
            display: grid;
            grid-template-columns: 3fr 1fr;
            gap: 16px;
            max-width: 1600px;
            margin: 0 auto 16px;
        }}
        .header-bar {{
            grid-column: 1 / -1;
        }}

        /* ── Bottom section ── */
        .grid-bottom {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
            max-width: 1600px;
            margin: 0 auto;
        }}

        /* ── Tablet (≤1024px) ── */
        @media (max-width: 1024px) {{
            .grid-main {{ grid-template-columns: 1fr; }}
            .grid-bottom {{ grid-template-columns: 1fr; }}
        }}

        /* ── Mobile (≤640px) ── */
        @media (max-width: 640px) {{
            body {{ padding: 8px; font-size: 13px; }}
            .metric-val {{ font-size: 18px !important; }}
            .header-right {{ flex-direction: column; gap: 10px !important; }}
            .col-right:nth-child(3) {{ display: none; }}   /* hide "Shares" on mobile */
        }}

        /* ── Panel ── */
        .panel {{
            background: #1E222D;
            border-radius: 6px;
            border: 1px solid #2A2E39;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }}
        .panel-header {{
            padding: 10px 16px;
            font-size: 11px;
            font-weight: 600;
            color: #8D94A6;
            border-bottom: 1px solid #2A2E39;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        ::-webkit-scrollbar {{ width: 4px; height: 4px; }}
        ::-webkit-scrollbar-track {{ background: transparent; }}
        ::-webkit-scrollbar-thumb {{ background: #363C4E; border-radius: 3px; }}

        /* ── Top Bar ── */
        .top-bar-content {{
            padding: 14px 24px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            width: 100%;
        }}
        .metric-group {{ display: flex; flex-direction: column; }}
        .metric-title {{ color: #8D94A6; font-size: 11px; margin-bottom: 3px;
                          text-transform: uppercase; letter-spacing: 0.5px; }}
        .metric-val {{ font-size: 22px; font-weight: 700; }}
        .metric-sub {{ font-size: 13px; font-weight: 600; margin-left: 10px; }}
        .dot {{ height: 8px; width: 8px; background: {TV_GREEN}; border-radius: 50%;
                display: inline-block; margin-right: 6px;
                box-shadow: 0 0 8px {TV_GREEN};
                animation: blink 2s infinite; }}
        @keyframes blink {{ 0%,100%{{opacity:1}} 50%{{opacity:.45}} }}
        .header-right {{ display:flex; gap:40px; text-align:right; }}

        /* ── Chart panels ── */
        .chart-fill {{ flex: 1; min-height: 320px; }}

        /* ── Right sidebar ── */
        .right-sidebar {{
            display: flex;
            flex-direction: column;
            max-height: 680px;
        }}

        /* ── Donut ── */
        .donut-wrapper {{
            padding: 12px 16px 4px;
            border-bottom: 1px solid #2A2E39;
            flex-shrink: 0;
        }}
        .donut-container {{
            position: relative;
            width: 185px;
            height: 185px;
            margin: 0 auto 4px;
        }}
        .donut-label {{
            position: absolute; top: 50%; left: 50%;
            transform: translate(-50%, -50%);
            text-align: center; pointer-events: none;
        }}
        .donut-val {{ font-size: 22px; font-weight: 700; color: #D1D4DC; }}
        .donut-sub {{ font-size: 10px; color: #8D94A6; text-transform: uppercase;
                      margin-top: 2px; letter-spacing: 0.5px; }}

        /* ── Holdings Table ── */
        .tbl-header {{
            display: flex;
            padding: 7px 14px;
            color: #8D94A6;
            font-size: 10px;
            border-bottom: 1px solid #2A2E39;
            text-transform: uppercase;
            letter-spacing: 0.3px;
            flex-shrink: 0;
        }}
        .table-row {{
            display: flex; padding: 9px 14px;
            border-bottom: 1px solid #2A2E39;
            font-size: 12px; align-items: center;
            transition: background 0.15s;
        }}
        .table-row:hover {{ background: rgba(255,255,255,0.03); cursor: default; }}
        .col-ticker {{ flex: 1.5; font-weight: 600; display: flex;
                        flex-direction: column; gap: 1px; }}
        .stock-id   {{ font-size: 13px; }}
        .stock-name {{ font-size: 10px; color: #8D94A6; font-weight: 400; }}
        .col-right  {{ flex: 1; text-align: right; }}
        .holdings-scroll {{ flex: 1; overflow-y: auto; }}

        /* ── Bottom tables ── */
        .history-table {{
            width: 100%; border-collapse: collapse; font-size: 12px;
        }}
        .history-table th {{
            padding: 7px 14px; text-align: left;
            background: #161d2b; color: #8D94A6;
            font-size: 10px; text-transform: uppercase;
            letter-spacing: 0.3px; border-bottom: 1px solid #2A2E39;
        }}
        .history-table td {{
            padding: 8px 14px; border-bottom: 1px solid #2A2E39;
        }}
        .history-table tr:last-child td {{ border-bottom: none; }}
        .history-table tr:hover td {{ background: rgba(255,255,255,0.02); }}
        .scroll-body {{ overflow-y: auto; max-height: 300px; }}

        /* ── Trade log ── */
        .trade-entry {{ padding: 10px 14px; border-bottom: 1px solid #2A2E39; }}
        .trade-date  {{ font-size: 12px; font-weight: 600; margin-bottom: 5px; }}
        .trade-list  {{ list-style: none; padding: 0; }}
        .trade-list li {{
            font-size: 11px; color: #8D94A6; padding: 2px 0;
        }}
        .trade-list li::before {{ content: "→ "; color: {TV_BLUE}; }}
    </style>
</head>
<body>

<div class="grid-main">
    <!-- Top Bar -->
    <div class="header-bar panel">
        <div class="top-bar-content">
            <div class="metric-group">
                <span class="metric-title"><span class="dot"></span> Live Model Portfolio</span>
                <div>
                    <span class="metric-val">${current_nav:,.0f}</span>
                    <span class="metric-sub" style="color:{daily_color};">{daily_sign}{daily_ret*100:.2f}% (1D)</span>
                    <span class="metric-sub" style="color:{total_color}; margin-left:12px;">{total_sign}{total_ret*100:.2f}% (All)</span>
                </div>
            </div>
            <div class="header-right">
                <div class="metric-group">
                    <span class="metric-title">Equity</span>
                    <span style="font-size:15px;font-weight:600;">${total_equity:,.0f}</span>
                </div>
                <div class="metric-group">
                    <span class="metric-title">Cash</span>
                    <span style="font-size:15px;font-weight:600;">${cash:,.0f}</span>
                </div>
                <div class="metric-group" style="text-align:right;">
                    <span class="metric-title">Market Date</span>
                    <span style="font-size:15px;font-weight:600;">{latest_date}</span>
                </div>
            </div>
        </div>
    </div>

    <!-- Main Chart -->
    <div class="panel chart-fill">
        <div class="panel-header">Performance Curve</div>
        <div style="flex:1; padding:0; min-height:300px;">
            {html_line}
        </div>
    </div>

    <!-- Right Sidebar -->
    <div class="panel right-sidebar">
        <div class="panel-header">Portfolio Allocation</div>

        <div class="donut-wrapper">
            <div class="donut-container">
                {html_pie}
                <div class="donut-label">
                    <div class="donut-val">{len(stock_data)}</div>
                    <div class="donut-sub">Positions</div>
                </div>
            </div>
        </div>

        <div class="panel-header" style="border-top:none;">Current Holdings</div>
        <div class="tbl-header">
            <div class="col-ticker">Symbol / 名稱</div>
            <div class="col-right">Price</div>
            <div class="col-right">Shares</div>
            <div class="col-right">Value</div>
            <div class="col-right">Weight</div>
        </div>
        <div class="holdings-scroll">
            {holdings_html}
        </div>
    </div>
</div>

<!-- Bottom: NAV History + Trade Log -->
<div class="grid-bottom">
    <div class="panel">
        <div class="panel-header">📅 每日淨值歷史（最近30天）</div>
        <div class="scroll-body">
            <table class="history-table">
                <thead><tr><th>日期</th><th>淨值 (NAV)</th><th>當日報酬</th></tr></thead>
                <tbody>{history_table_rows}</tbody>
            </table>
        </div>
    </div>

    <div class="panel">
        <div class="panel-header">💼 交易紀錄 &amp; 損益明細</div>
        <div class="scroll-body">
            {trade_detail_html}
        </div>
    </div>
</div>

</body>
</html>
"""

os.makedirs("frontend", exist_ok=True)
with open("frontend/index.html", "w", encoding="utf-8") as f:
    f.write(html_content)

print("✅ Dashboard 生成完成 → frontend/index.html")
print(f"   持股: {len(stock_data)} 支，歷史: {len(history)} 天")

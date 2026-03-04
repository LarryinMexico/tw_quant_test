import json
import pandas as pd
import plotly.graph_objects as go
import os

with open("portfolio.json", "r") as f:
    pf = json.load(f)

latest_date = pf.get("latest_date", "N/A")
latest_prices = pf.get("latest_prices", {})

history = pf.get("history", [])
dates = [row["date"] for row in history]
navs = [row["nav"] for row in history]

# Stats calculation
current_nav = navs[-1] if navs else pf.get("cash", 1000000.0)
prev_nav = navs[-2] if len(navs) >= 2 else 1000000.0
init_nav = 1000000.0

daily_ret = (current_nav / prev_nav) - 1
total_ret = (current_nav / init_nav) - 1

cash = pf.get("cash", 0)
positions = pf.get("positions", {})

stock_data = []
total_equity = 0
for stock, shares in positions.items():
    price = latest_prices.get(stock, 0)
    val = shares * price
    total_equity += val
    stock_data.append({"stock": stock, "shares": shares, "price": price, "value": val})

stock_data.sort(key=lambda x: x["value"], reverse=True)

for item in stock_data:
    item["weight"] = item["value"] / current_nav if current_nav > 0 else 0

# Colors for UI
TV_BG = "#131722"
TV_PANEL = "#1E222D"
TV_GRID = "#2B3139"
TV_TEXT = "#D1D4DC"
TV_MUTED = "#8D94A6"
TV_GREEN = "#22AB94"
TV_RED = "#F05350"
TV_BLUE = "#2962FF"

# 1. NAV Line Chart
fig_line = go.Figure()
fig_line.add_trace(go.Scatter(
    x=dates, y=navs, mode="lines",
    line=dict(color=TV_BLUE, width=2),
    fill='tozeroy', fillcolor="rgba(41, 98, 255, 0.15)",
    hoverinfo="x+y"
))
fig_line.update_layout(
    margin=dict(l=50, r=20, t=20, b=40),
    plot_bgcolor=TV_PANEL, paper_bgcolor=TV_PANEL,
    font=dict(color=TV_TEXT),
    xaxis=dict(showgrid=True, gridcolor=TV_GRID, tickcolor=TV_MUTED),
    yaxis=dict(showgrid=True, gridcolor=TV_GRID, tickformat=",.0f", side="right", tickcolor=TV_MUTED),
    hovermode="x unified"
)
html_line = fig_line.to_html(include_plotlyjs="cdn", full_html=False, config={"displayModeBar": False})

# 2. Allocation Pie Chart
labels = ["Cash"] + [s["stock"] for s in stock_data]
values = [cash] + [s["value"] for s in stock_data]
colors = [TV_MUTED] + [f"hsl({(i*45)%360}, 65%, 60%)" for i in range(len(stock_data))]

fig_pie = go.Figure(data=[go.Pie(
    labels=labels, values=values, hole=.65,
    marker=dict(colors=colors, line=dict(color=TV_PANEL, width=2)),
    textinfo='none', hoverinfo='label+percent'
)])
fig_pie.update_layout(
    showlegend=False,
    margin=dict(l=10, r=10, t=10, b=10),
    plot_bgcolor=TV_PANEL, paper_bgcolor=TV_PANEL,
    font=dict(color=TV_TEXT)
)
html_pie = fig_pie.to_html(include_plotlyjs=False, full_html=False, config={"displayModeBar": False})

# Build Holdings Table
holdings_html = ""
for item in stock_data:
    holdings_html += f"""
    <div class="table-row">
        <div class="col-ticker">{item['stock']}</div>
        <div class="col-right">{item['price']:.2f}</div>
        <div class="col-right">{item['shares']}</div>
        <div class="col-right">${item['value']:,.0f}</div>
        <div class="col-right" style="color: {TV_BLUE};">{item['weight'] * 100:.1f}%</div>
    </div>
    """

# Metrics formats
daily_color = TV_GREEN if daily_ret >= 0 else TV_RED
total_color = TV_GREEN if total_ret >= 0 else TV_RED
daily_sign = "+" if daily_ret >= 0 else ""
total_sign = "+" if total_ret >= 0 else ""

html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TradingView Style Dashboard</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Trebuchet MS', Roboto, Ubuntu, sans-serif; }}
        body {{ background-color: {TV_BG}; color: {TV_TEXT}; padding: 16px; font-size: 14px; }}
        
        .grid-container {{
            display: grid;
            grid-template-columns: 3fr 1fr;
            grid-template-rows: auto 1fr;
            gap: 16px;
            max-width: 1600px;
            margin: 0 auto;
            height: calc(100vh - 32px);
        }}
        
        @media (max-width: 1024px) {{ .grid-container {{ grid-template-columns: 1fr; height: auto; }} }}

        .panel {{ background-color: {TV_PANEL}; border-radius: 4px; border: 1px solid #2A2E39; display: flex; flex-direction: column; overflow: hidden; }}
        .panel-header {{ padding: 12px 16px; font-size: 14px; font-weight: 600; border-bottom: 1px solid #2A2E39; display: flex; justify-content: space-between; align-items: center; }}
        .panel-content {{ padding: 16px; flex: 1; overflow-y: auto; }}
        
        ::-webkit-scrollbar {{ width: 6px; }}
        ::-webkit-scrollbar-track {{ background: {TV_PANEL}; }}
        ::-webkit-scrollbar-thumb {{ background: #363C4E; border-radius: 3px; }}

        .header-top {{ grid-column: 1 / -1; display: flex; gap: 16px; margin-bottom: 8px; flex-wrap: wrap; }}
        .ticker-info {{ display: flex; align-items: center; gap: 24px; }}
        
        .metric-group {{ display: flex; flex-direction: column; }}
        .metric-title {{ color: {TV_MUTED}; font-size: 12px; margin-bottom: 4px; text-transform: uppercase; letter-spacing: 0.5px; }}
        .metric-val {{ font-size: 20px; font-weight: 600; }}
        .metric-sub {{ font-size: 14px; font-weight: 500; margin-left: 8px; display: inline-block; }}
        
        .table-header {{ display: flex; padding: 8px 16px; color: {TV_MUTED}; font-size: 12px; border-bottom: 1px solid #2A2E39; }}
        .table-row {{ display: flex; padding: 12px 16px; border-bottom: 1px solid #2A2E39; font-size: 13px; align-items: center; transition: background 0.15s; }}
        .table-row:hover {{ background-color: rgba(255,255,255,0.03); cursor: default; }}
        
        .col-ticker {{ flex: 1; font-weight: 600; }}
        .col-right {{ flex: 1; text-align: right; }}
        
        .dot {{ height: 8px; width: 8px; background-color: {TV_GREEN}; border-radius: 50%; display: inline-block; margin-right: 6px; box-shadow: 0 0 8px {TV_GREEN}; }}
        
        .donut-container {{ position: relative; width: 200px; height: 200px; margin: 0 auto 16px; }}
        .donut-label {{ position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); text-align: center; }}
        .donut-val {{ font-size: 20px; font-weight: 600; color: {TV_TEXT}; }}
        .donut-sub {{ font-size: 11px; color: {TV_MUTED}; text-transform: uppercase; margin-top: 2px; }}
    </style>
</head>
<body>

<div class="grid-container">
    
    <!-- Top Bar Metrics -->
    <div class="header-top panel">
        <div class="panel-content" style="padding: 16px 24px; display: flex; justify-content: space-between; align-items: center; width: 100%;">
            <div class="ticker-info">
                <div class="metric-group">
                    <span class="metric-title"><span class="dot"></span> Live Model Portfolio</span>
                    <div>
                        <span class="metric-val">${current_nav:,.0f}</span>
                        <span class="metric-sub" style="color: {daily_color};">{daily_sign}{daily_ret * 100:.2f}% (1D)</span>
                        <span class="metric-sub" style="color: {total_color}; margin-left: 12px;">{total_sign}{total_ret * 100:.2f}% (All)</span>
                    </div>
                </div>
            </div>
            <div style="display: flex; gap: 40px; text-align: right;">
                <div class="metric-group">
                    <span class="metric-title">Equity</span>
                    <span style="font-size: 15px;">${total_equity:,.0f}</span>
                </div>
                <div class="metric-group">
                    <span class="metric-title">Cash</span>
                    <span style="font-size: 15px;">${cash:,.0f}</span>
                </div>
                <div class="metric-group" style="text-align: right;">
                    <span class="metric-title">Market Date</span>
                    <span style="font-size: 15px;">{latest_date}</span>
                </div>
            </div>
        </div>
    </div>

    <!-- Main Chart -->
    <div class="panel">
        <div class="panel-header">
            <span>Performance Curve</span>
        </div>
        <div class="panel-content" style="padding: 0;">
            {html_line}
        </div>
    </div>

    <!-- Right Sidebar (Allocation & Holdings) -->
    <div class="panel" style="display: flex; flex-direction: column;">
        <div class="panel-header">
            <span>Portfolio Allocation</span>
        </div>
        <div style="padding: 24px 16px 8px; border-bottom: 1px solid #2A2E39;">
            <div class="donut-container">
                {html_pie}
                <div class="donut-label">
                    <div class="donut-val">{len(stock_data)}</div>
                    <div class="donut-sub">Positions</div>
                </div>
            </div>
        </div>
        
        <div class="panel-header" style="border-top: none;">
            <span>Current Holdings</span>
        </div>
        <div class="table-header">
            <div class="col-ticker">Symbol</div>
            <div class="col-right">Price</div>
            <div class="col-right">Shares</div>
            <div class="col-right">Value</div>
            <div class="col-right">Weight</div>
        </div>
        <div style="flex: 1; overflow-y: auto;">
            {holdings_html}
        </div>
    </div>
</div>

</body>
</html>
"""

os.makedirs("frontend", exist_ok=True)
with open("frontend/index.html", "w") as f:
    f.write(html_content)

print("首頁 (TradingView 風格) 已生成 frontend/index.html")

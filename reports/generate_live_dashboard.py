import json
import pandas as pd
import plotly.graph_objects as go
import os

with open("portfolio.json", "r") as f:
    pf = json.load(f)

latest_date = pf.get("latest_date", "N/A")
latest_prices = pf.get("latest_prices", {})

fig = go.Figure()

dates = [row["date"] for row in pf["history"]]
navs = [row["nav"] for row in pf["history"]]

fig.add_trace(go.Scatter(
    x=dates, y=navs, mode="lines+markers",
    name="NAV", line=dict(color="#00FFAA", width=3)
))

fig.update_layout(
    title="台股ML虛擬基金 績效追蹤",
    plot_bgcolor="#111111",
    paper_bgcolor="#111111",
    font=dict(color="#ECF0F1"),
    xaxis=dict(showgrid=False),
    yaxis=dict(showgrid=True, gridcolor="#333333")
)

html_chart = fig.to_html(include_plotlyjs="cdn", full_html=False)

holdings_html = ""
for stock, shares in pf["positions"].items():
    price = latest_prices.get(stock, 0)
    val = shares * price
    holdings_html += f"""
    <div style="display:flex; justify-content:space-between; padding:10px; border-bottom:1px solid #333;">
        <span style="font-weight:bold; color:#00FFAA;">{stock}</span>
        <span>{shares} 股</span>
        <span>${val:,.0f}</span>
    </div>
    """

full_page = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Live Dashboard</title>
    <style>
        body {{
            background-color: #0b0c10;
            color: #c5c6c7;
            font-family: -apple-system, sans-serif;
            margin: 0; padding: 20px;
        }}
        .container {{ max-width: 900px; margin: 0 auto; }}
        h1 {{ color: #66fcf1; text-align: center; }}
        .nav-box {{
            background: #1f2833; padding: 20px;
            border-radius: 10px; margin: 20px 0;
            text-align: center; font-size: 24px;
        }}
        .chart-box {{
            background: #1f2833; padding: 20px;
            border-radius: 10px; margin: 20px 0;
        }}
        .holdings-box {{
            background: #1f2833; padding: 20px;
            border-radius: 10px; margin: 20px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>量化實盤追蹤看板</h1>
        <div class="nav-box">
            <div>今日總淨值 $ {pf['cash'] + sum(shares * latest_prices.get(stock, 0) for stock, shares in pf['positions'].items()):,.0f}</div>
            <div style="font-size: 16px; color: #aaa; margin-top: 10px;">現金部位 $ {pf['cash']:,.0f}</div>
            <div style="font-size: 14px; color: #888; margin-top: 5px;">最後更新日期 {latest_date}</div>
        </div>
        
        <div class="chart-box">
            {html_chart}
        </div>
        
        <div class="holdings-box">
            <h2 style="color:#66fcf1;">現有持股 Top 20</h2>
            {holdings_html}
        </div>
    </div>
</body>
</html>
"""

with open("frontend/index.html", "w") as f:
    f.write(full_page)

print("首頁已生成 frontend/index.html")

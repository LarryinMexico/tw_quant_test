"""
generate_report.py
==================
Comprehensive backtest HTML tearsheet using Plotly.
All labels in English to avoid font rendering issues.
"""

import os, warnings, textwrap
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio

# ─── 1. Load data ───────────────────────────────────────────────────────────
print("[1/5] Loading data...")

close = pd.read_pickle("finmind_cache/close_wide.pkl")
close.index = pd.to_datetime(close.index)
close.index.name = "date"
close.columns.name = "stock_id"
close_m = close.resample("ME").last()

preds   = pd.read_pickle("predictions.pkl")
X, y = None, None

stock_list = pd.read_pickle("finmind_cache/stock_list.pkl")
# Keep latest entry per stock_id
stock_list = stock_list.sort_values("date").drop_duplicates("stock_id", keep="last")
industry_map = stock_list.set_index("stock_id")["industry_category"].to_dict()

preds.index = preds.index.set_names(["date", "stock_id"])
pred_wide = preds["y_pred"].unstack("stock_id")

# ─── 2. Regime filter & positions ───────────────────────────────────────────
print("[2/5] Building positions...")

TOP_K = 30
tw50 = close["0050"]
is_bull = tw50 > tw50.rolling(60).mean()
is_bull_m = is_bull.resample("ME").last().reindex(pred_wide.index, method="ffill").fillna(False)

FEE = 1.425 / 1000 / 3
TAX = 3 / 1000
STOP_LOSS = 0.10

pos_rows = {}
for date, row in pred_wide.iterrows():
    top_stocks = row.nlargest(TOP_K).index.tolist()
    if is_bull_m.get(date, False):
        r = pd.Series(False, index=pred_wide.columns)
        r[top_stocks] = True
        pos_rows[date] = r
    else:
        pos_rows[date] = pd.Series(False, index=pred_wide.columns)

position_df = pd.DataFrame(pos_rows).T

# ─── 3. Performance computation (FROM EXACT VECTORBT) ───────────────────────
print("[3/5] Loading Vectorbt exact equity...")

eq      = pd.read_pickle("eq.pkl")
eq_bm      = pd.read_pickle("bm_eq.pkl")
weights_df = pd.read_pickle("weights.pkl")

# We build position matrix and holdings based off weights_df
active_stocks = (weights_df > 0)
exposure = active_stocks.sum(axis=1) / TOP_K

holdings_over_time = {}
for date, row in active_stocks.iterrows():
    held = row[row]
    holdings_over_time[date] = held.index.tolist()

strategy_ret = eq.resample("ME").last().pct_change().dropna()
bm_monthly   = eq_bm.resample("ME").last().pct_change().dropna()
excess_ret   = strategy_ret - bm_monthly

equity    = (1 + strategy_ret).cumprod()
bm_equity = (1 + bm_monthly).cumprod()

INIT_CASH = 1000000
# total ret based on exact end value to make it match strategy engine exactly
total_ret = (eq.iloc[-1] / INIT_CASH) - 1
n_years   = (eq.index[-1] - eq.index[0]).days / 365.25
cagr      = (eq.iloc[-1] / INIT_CASH) ** (1/n_years) - 1

vol_m     = strategy_ret.std()
vol_ann   = vol_m * (12 ** 0.5)
sharpe_ann = (strategy_ret.mean() / vol_m) * (12 ** 0.5) if vol_m > 0 else 0

# drawdown exactly from daily eq
cummax_daily = eq.cummax()
drawdown_daily = (eq - cummax_daily) / cummax_daily
max_dd = drawdown_daily.min()

drawdown = (equity - equity.cummax()) / equity.cummax()
win_ratio = (strategy_ret > 0).mean()

bm_cagr     = (eq_bm.iloc[-1] / INIT_CASH) ** (1/n_years) - 1
bm_vol      = bm_monthly.std() * (12 ** 0.5)
bm_sharpe   = (bm_monthly.mean() / bm_monthly.std()) * (12 ** 0.5)

cummax_bm_daily = eq_bm.cummax()
bm_drawdown = (eq_bm - cummax_bm_daily) / cummax_bm_daily

# Rolling 6M Sharpe
rolling6_sharpe = strategy_ret.rolling(6).apply(
    lambda x: (x.mean() / x.std()) * (12 ** 0.5) if x.std() > 0 else 0, raw=True
)

# Annual returns
annual_ret    = strategy_ret.groupby(strategy_ret.index.year).apply(lambda x: (1+x).prod() - 1)
annual_bm     = bm_monthly.groupby(bm_monthly.index.year).apply(lambda x: (1+x).prod() - 1)

# Top 5 drawdown periods
in_dd = drawdown < 0
dd_periods = []
start = None
for date, val in drawdown.items():
    if val < 0 and start is None:
        start = date
    elif val >= 0 and start is not None:
        dd_chunk = drawdown[start:date]
        dd_periods.append({
            "start": start, "end": date,
            "trough": dd_chunk.idxmin(),
            "max_dd": dd_chunk.min(),
            "duration": (date - start).days
        })
        start = None
if start is not None:
    dd_chunk = drawdown[start:]
    dd_periods.append({"start": start, "end": dd_chunk.index[-1],
                        "trough": dd_chunk.idxmin(), "max_dd": dd_chunk.min(),
                        "duration": (dd_chunk.index[-1] - start).days})

dd_periods = sorted(dd_periods, key=lambda x: x["max_dd"])[:5]

# Monthly returns heatmap (year x month)
df_hm = strategy_ret.copy().to_frame("ret")
df_hm["year"]  = df_hm.index.year
df_hm["month"] = df_hm.index.month
hm_pivot = df_hm.pivot_table(index="year", columns="month", values="ret")
month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
hm_pivot.columns = [month_names[m-1] for m in hm_pivot.columns]

# Factor quintile analysis (for each factor, split into 5 buckets by factor value)
print("  Skipping factor quintile analysis (X/y not available)...")
factor_quintile_stats = {}

# Industry frequency in top holdings
industry_counts = {}
for date, stocks in holdings_over_time.items():
    for s in stocks:
        ind = industry_map.get(s, "Unknown")
        industry_counts[ind] = industry_counts.get(ind, 0) + 1
industry_series = pd.Series(industry_counts).sort_values(ascending=False)

# Exposure over time we calculated via active_stocks

print("[4/5] Building Plotly HTML report...")

# ─── 4. Build Plotly figures ─────────────────────────────────────────────────
DARK_BG   = "#0d1117"
PANEL_BG  = "#161b22"
GRID_CLR  = "#30363d"
TEXT_CLR  = "#e6edf3"
RED       = "#f85149"
GREEN     = "#3fb950"
BLUE      = "#58a6ff"
GOLD      = "#ffa657"
GRAY      = "#8b949e"
ACCENT    = "#bc8cff"

def base_layout(title="", height=400):
    return dict(
        title=dict(text=title, font=dict(color=TEXT_CLR, size=14), x=0.01),
        paper_bgcolor=DARK_BG,
        plot_bgcolor=PANEL_BG,
        font=dict(color=TEXT_CLR, family="Inter, system-ui, sans-serif"),
        height=height,
        margin=dict(l=60, r=20, t=50, b=40),
        xaxis=dict(gridcolor=GRID_CLR, showgrid=True, zeroline=False),
        yaxis=dict(gridcolor=GRID_CLR, showgrid=True, zeroline=False),
        legend=dict(bgcolor="rgba(0,0,0,0.4)", bordercolor=GRID_CLR, borderwidth=1),
    )

figs_html = []  # list of (section_title, html_string)

# ── Panel 1: Summary stats ───────────────────────────────────────────────────
summary_data = {
    "Metric": [
        "Period", "CAGR", "Total Return", "Max Drawdown",
        "Ann. Volatility", "Sharpe Ratio (Ann.)", "Monthly Win Rate",
        "Best Month", "Worst Month", "Avg Monthly Return",
        "Benchmark CAGR (0050)", "Benchmark Sharpe", "Alpha (Monthly Avg)",
    ],
    "Strategy": [
        f"{equity.index[0].strftime('%Y-%m')} ~ {equity.index[-1].strftime('%Y-%m')}",
        f"{cagr:.2%}", f"{total_ret:.2%}", f"{max_dd:.2%}",
        f"{vol_ann:.2%}", f"{sharpe_ann:.2f}", f"{win_ratio:.2%}",
        f"{strategy_ret.max():.2%}", f"{strategy_ret.min():.2%}",
        f"{strategy_ret.mean():.2%}",
        f"{bm_cagr:.2%}", f"{bm_sharpe:.2f}",
        f"{excess_ret.mean():.2%}",
    ],
}
df_summary = pd.DataFrame(summary_data)
fig_summary = go.Figure(go.Table(
    header=dict(values=["Metric", "Value"],
                fill_color="#21262d", font=dict(color=TEXT_CLR, size=12),
                align="left", line_color=GRID_CLR),
    cells=dict(values=[df_summary["Metric"], df_summary["Strategy"]],
               fill_color=[PANEL_BG, PANEL_BG],
               font=dict(color=[GRAY, GREEN], size=12),
               align="left", line_color=GRID_CLR),
))
fig_summary.update_layout(**base_layout("Performance Summary", height=420))
figs_html.append(("Performance Summary", fig_summary.to_html(full_html=False, include_plotlyjs=False)))


# ── Panel 2: Cumulative Returns ──────────────────────────────────────────────
fig_equity = go.Figure()
fig_equity.add_trace(go.Scatter(x=eq.index, y=eq.values/INIT_CASH, name="Strategy",
                                line=dict(color=RED, width=2)))
fig_equity.add_trace(go.Scatter(x=eq_bm.index, y=eq_bm.values/INIT_CASH, name="Benchmark (0050)",
                                line=dict(color=GRAY, width=1.5, dash="dot")))
fig_equity.update_layout(**base_layout("Cumulative Returns (Start=1.0)", height=350))
fig_equity.update_yaxes(tickformat=".2f")
figs_html.append(("Cumulative Returns", fig_equity.to_html(full_html=False, include_plotlyjs=False)))


# ── Panel 3: Underwater Plot ─────────────────────────────────────────────────
fig_dd = go.Figure()
fig_dd.add_trace(go.Scatter(x=drawdown_daily.index, y=drawdown_daily.values * 100,
                             fill="tozeroy", name="Strategy DD",
                             line=dict(color="coral", width=1),
                             fillcolor="rgba(255,100,80,0.3)"))
fig_dd.add_trace(go.Scatter(x=bm_drawdown.index, y=bm_drawdown.values * 100,
                             fill="tozeroy", name="Benchmark DD",
                             line=dict(color=GRAY, width=1, dash="dot"),
                             fillcolor="rgba(140,140,140,0.15)"))
fig_dd.update_layout(**base_layout("Underwater / Drawdown Plot (%)", height=300))
fig_dd.update_yaxes(ticksuffix="%")
figs_html.append(("Underwater Plot", fig_dd.to_html(full_html=False, include_plotlyjs=False)))


# ── Panel 4: Top 5 Drawdown Periods table ───────────────────────────────────
dd_records = []
for i, p in enumerate(dd_periods, 1):
    dd_records.append({
        "Rank": i,
        "Start": p["start"].strftime("%Y-%m"),
        "Trough": p["trough"].strftime("%Y-%m"),
        "End": p["end"].strftime("%Y-%m"),
        "Max Drawdown": f"{p['max_dd']:.2%}",
        "Duration (days)": p["duration"],
    })
df_dd_table = pd.DataFrame(dd_records)
fig_dd_table = go.Figure(go.Table(
    header=dict(values=list(df_dd_table.columns),
                fill_color="#21262d", font=dict(color=TEXT_CLR, size=12), align="center"),
    cells=dict(values=[df_dd_table[c] for c in df_dd_table.columns],
               fill_color=PANEL_BG, font=dict(color=[TEXT_CLR]*len(df_dd_table.columns), size=11),
               align="center", line_color=GRID_CLR),
))
fig_dd_table.update_layout(**base_layout("Top 5 Drawdown Periods", height=250))
figs_html.append(("Top 5 Drawdown Periods", fig_dd_table.to_html(full_html=False, include_plotlyjs=False)))


# ── Panel 5: Rolling 6-Month Sharpe ─────────────────────────────────────────
fig_rolling = go.Figure()
fig_rolling.add_trace(go.Scatter(x=rolling6_sharpe.index, y=rolling6_sharpe.values,
                                  name="Rolling 6M Sharpe",
                                  line=dict(color=BLUE, width=2)))
fig_rolling.add_hline(y=0, line=dict(color=GRAY, dash="dash", width=1))
fig_rolling.add_hline(y=1, line=dict(color=GREEN, dash="dot", width=1),
                       annotation_text="Sharpe=1", annotation_position="top right")
fig_rolling.update_layout(**base_layout("Rolling 6-Month Sharpe Ratio (Annualized)", height=300))
figs_html.append(("Rolling Sharpe", fig_rolling.to_html(full_html=False, include_plotlyjs=False)))


# ── Panel 6: Monthly Returns Heatmap ────────────────────────────────────────
z_vals = hm_pivot.fillna(0).values * 100
text_vals = []
for i in range(hm_pivot.shape[0]):
    row_texts = []
    for j in range(hm_pivot.shape[1]):
        val = hm_pivot.iloc[i, j]
        row_texts.append(f"{val*100:.1f}%" if not np.isnan(val) else "")
    text_vals.append(row_texts)

fig_hm = go.Figure(go.Heatmap(
    z=z_vals,
    x=hm_pivot.columns.tolist(),
    y=hm_pivot.index.tolist(),
    text=text_vals,
    texttemplate="%{text}",
    colorscale=[[0,"#c0392b"],[0.5,"#1a1a2e"],[1,"#27ae60"]],
    zmid=0,
    colorbar=dict(title="Ret %", ticksuffix="%"),
))
fig_hm.update_layout(**base_layout("Monthly Returns Heatmap (%)", height=300))
figs_html.append(("Monthly Returns Heatmap", fig_hm.to_html(full_html=False, include_plotlyjs=False)))


# ── Panel 7: Annual Returns ──────────────────────────────────────────────────
common_years = annual_ret.index.intersection(annual_bm.index)
fig_annual = go.Figure()
fig_annual.add_trace(go.Bar(x=common_years, y=annual_ret.loc[common_years].values * 100,
                             name="Strategy", marker_color=BLUE, opacity=0.85))
fig_annual.add_trace(go.Bar(x=common_years, y=annual_bm.loc[common_years].values * 100,
                             name="Benchmark (0050)", marker_color=GRAY, opacity=0.7))
fig_annual.update_layout(**base_layout("Annual Returns (%)", height=320), barmode="group")
fig_annual.update_yaxes(ticksuffix="%")
figs_html.append(("Annual Returns", fig_annual.to_html(full_html=False, include_plotlyjs=False)))


# ── Panel 8: Distribution of Monthly Returns ─────────────────────────────────
fig_dist = go.Figure()
fig_dist.add_trace(go.Histogram(x=strategy_ret.values * 100, nbinsx=30,
                                 name="Strategy", marker_color=BLUE, opacity=0.75))
fig_dist.add_trace(go.Histogram(x=bm_monthly.values * 100, nbinsx=30,
                                 name="Benchmark", marker_color=GRAY, opacity=0.5))
fig_dist.add_vline(x=0, line=dict(color=TEXT_CLR, dash="dash"))
fig_dist.add_vline(x=strategy_ret.mean()*100, line=dict(color=GREEN, dash="dot"),
                    annotation_text=f"Mean: {strategy_ret.mean():.2%}", annotation_position="top right")
fig_dist.update_layout(**base_layout("Distribution of Monthly Returns (%)", height=320), barmode="overlay")
fig_dist.update_xaxes(ticksuffix="%")
figs_html.append(("Return Distribution", fig_dist.to_html(full_html=False, include_plotlyjs=False)))


# ── Panel 9: Return Quantiles (strat vs bm) ──────────────────────────────────
quantiles = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
q_labels  = ["5%", "10%", "25%", "50% (Median)", "75%", "90%", "95%"]
q_strat   = [strategy_ret.quantile(q) * 100 for q in quantiles]
q_bm      = [bm_monthly.quantile(q) * 100 for q in quantiles]
fig_quantile = go.Figure()
fig_quantile.add_trace(go.Bar(x=q_labels, y=q_strat, name="Strategy",
                               marker_color=[GREEN if v > 0 else RED for v in q_strat], opacity=0.85))
fig_quantile.add_trace(go.Bar(x=q_labels, y=q_bm, name="Benchmark",
                               marker_color=GRAY, opacity=0.7))
fig_quantile.update_layout(**base_layout("Monthly Return Quantiles (%)", height=320), barmode="group")
fig_quantile.update_yaxes(ticksuffix="%")
figs_html.append(("Return Quantiles", fig_quantile.to_html(full_html=False, include_plotlyjs=False)))


# ── Panel 10: Portfolio Exposure over Time ───────────────────────────────────
fig_exp = go.Figure()
fig_exp.add_trace(go.Bar(x=exposure.index, y=exposure.values * 100,
                          name="Exposure (%)", marker_color=BLUE, opacity=0.8))
fig_exp.add_hline(y=100, line=dict(color=GREEN, dash="dot"),
                   annotation_text="Full", annotation_position="top right")
fig_exp.update_layout(**base_layout("Portfolio Exposure Over Time (%)", height=280))
fig_exp.update_yaxes(range=[0, 110], ticksuffix="%")
figs_html.append(("Exposure", fig_exp.to_html(full_html=False, include_plotlyjs=False)))


# ── Panel 11: Total Holdings over Time ───────────────────────────────────────
n_holdings = position_df.sum(axis=1)
fig_holdings = go.Figure()
fig_holdings.add_trace(go.Scatter(x=n_holdings.index, y=n_holdings.values,
                                   name="# Holdings", fill="tozeroy",
                                   line=dict(color=GOLD, width=2),
                                   fillcolor="rgba(255,166,87,0.15)"))
fig_holdings.add_hline(y=TOP_K, line=dict(color=GRAY, dash="dot"),
                        annotation_text=f"Target={TOP_K}", annotation_position="top right")
fig_holdings.update_layout(**base_layout("Total Holdings Over Time", height=280))
figs_html.append(("Holdings Count", fig_holdings.to_html(full_html=False, include_plotlyjs=False)))


# ── Panel 12: Top 10 Holdings frequency ──────────────────────────────────────
all_held = {}
for stocks in holdings_over_time.values():
    for s in stocks:
        all_held[s] = all_held.get(s, 0) + 1
top10 = pd.Series(all_held).sort_values(ascending=False).head(10)

fig_top10 = go.Figure(go.Bar(
    x=top10.index.tolist(),
    y=top10.values,
    marker_color=BLUE, opacity=0.85,
    text=[f"{v}m" for v in top10.values],
    textposition="outside",
))
fig_top10.update_layout(**base_layout("Top 10 Most-Held Stocks (Months in Portfolio)", height=320))
figs_html.append(("Top 10 Holdings", fig_top10.to_html(full_html=False, include_plotlyjs=False)))


# ── Panel 13: Industry Selection Frequency ────────────────────────────────────
top_industries = industry_series.head(15)
fig_industry = go.Figure(go.Bar(
    y=top_industries.index.tolist()[::-1],
    x=top_industries.values[::-1],
    orientation="h",
    marker_color=ACCENT, opacity=0.85,
))
fig_industry.update_layout(**base_layout("Industry Selection Frequency (Total Appearances)", height=420))
figs_html.append(("Industry Frequency", fig_industry.to_html(full_html=False, include_plotlyjs=False)))


# ── Panel 14: Factor Quintile Performance ────────────────────────────────────
for factor_name, qs_df in factor_quintile_stats.items():
    if qs_df.empty:
        continue
    fig_fq = make_subplots(
        rows=1, cols=3,
        subplot_titles=["Ann. Return (12x)", "Sharpe (Ann.)", "Max Drawdown"],
    )
    q_labels_5 = ["Q1 (low)", "Q2", "Q3", "Q4", "Q5 (high)"]
    labels_used = q_labels_5[:len(qs_df)]

    ann_ret = qs_df["mean_ret"] * 12 * 100
    sharpes = qs_df["sharpe"]
    mdd     = qs_df["max_dd"] * 100

    col_ann = [GREEN if v > 0 else RED for v in ann_ret]
    col_sh  = [GREEN if v > 0 else RED for v in sharpes]

    fig_fq.add_trace(go.Bar(x=labels_used, y=ann_ret.values, marker_color=col_ann, name="Ann.Ret"), row=1, col=1)
    fig_fq.add_trace(go.Bar(x=labels_used, y=sharpes.values, marker_color=col_sh, name="Sharpe"), row=1, col=2)
    fig_fq.add_trace(go.Bar(x=labels_used, y=mdd.values, marker_color="coral", name="Max DD"), row=1, col=3)

    fig_fq.update_layout(
        title=dict(text=f"Factor Quintile Analysis — {factor_name}", font=dict(color=TEXT_CLR, size=13)),
        paper_bgcolor=DARK_BG, plot_bgcolor=PANEL_BG,
        font=dict(color=TEXT_CLR), height=320, showlegend=False,
        margin=dict(l=60, r=20, t=60, b=40),
    )
    for ax in ["xaxis","xaxis2","xaxis3","yaxis","yaxis2","yaxis3"]:
        fig_fq.update_layout(**{ax: dict(gridcolor=GRID_CLR)})

    figs_html.append((f"Factor Quintile: {factor_name}",
                       fig_fq.to_html(full_html=False, include_plotlyjs=False)))


# ── Panel 15: Covid/Bear period: 2022-01 ~ 2023-06 ─────────────────────────
period_label = "Bear Market Period (2022-01 ~ 2023-06)"
strat_period  = strategy_ret["2022":"2023-06"]
bm_period     = bm_monthly["2022":"2023-06"]
eq_p  = (1 + strat_period).cumprod()
bm_p  = (1 + bm_period).cumprod()
fig_period = go.Figure()
fig_period.add_trace(go.Scatter(x=eq_p.index, y=eq_p.values, name="Strategy", line=dict(color=RED, width=2)))
fig_period.add_trace(go.Scatter(x=bm_p.index, y=bm_p.values, name="Benchmark(0050)", line=dict(color=GRAY, width=1.5, dash="dot")))
fig_period.update_layout(**base_layout(period_label, height=300))
figs_html.append((period_label, fig_period.to_html(full_html=False, include_plotlyjs=False)))


# ─── 5. Assemble full HTML ───────────────────────────────────────────────────
print("[5/5] Assembling HTML report...")

NAV_ITEMS = "".join([
    f'<li><a href="#{title.lower().replace(" ","_").replace("/","").replace("-","_")}">{title}</a></li>'
    for title, _ in figs_html
])

SECTIONS = ""
for title, fig_html in figs_html:
    sec_id = title.lower().replace(" ", "_").replace("/","").replace("-","_")
    SECTIONS += f"""
    <section id="{sec_id}">
      <h2>{title}</h2>
      <div class="chart-wrapper">
        {fig_html}
      </div>
    </section>
    """

STATS_ROW = ""
for metric, val in zip(
    ["CAGR", "Total Return", "Max Drawdown", "Sharpe (Ann.)", "Win Rate", "Alpha/Month"],
    [f"{cagr:.2%}", f"{total_ret:.2%}", f"{max_dd:.2%}", f"{sharpe_ann:.2f}", f"{win_ratio:.2%}", f"{excess_ret.mean():.2%}"]
):
    color = "var(--green)" if ("%" not in val or float(val.rstrip("%")) > 0) else "var(--red)"
    if metric in ["Max Drawdown"]: color = "var(--red)" if float(val.rstrip("%")) < 0 else "var(--green)"
    STATS_ROW += f"""
    <div class="kpi-card">
      <div class="kpi-label">{metric}</div>
      <div class="kpi-value" style="color:{color};">{val}</div>
    </div>"""

HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>ML Stock Strategy — Backtest Report</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  :root {{
    --bg: #0d1117; --panel: #161b22; --border: #30363d;
    --text: #e6edf3; --muted: #8b949e;
    --green: #3fb950; --red: #f85149; --blue: #58a6ff;
    --gold: #ffa657; --purple: #bc8cff;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text);
          font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, sans-serif;
          display: flex; min-height: 100vh; }}
  nav {{ position: fixed; width: 220px; height: 100vh; top: 0; left: 0;
          background: var(--panel); border-right: 1px solid var(--border);
          overflow-y: auto; padding: 20px 0; z-index: 100; }}
  nav h1 {{ font-size: 12px; font-weight: 700; color: var(--muted);
              text-transform: uppercase; letter-spacing: 1px;
              padding: 8px 16px 16px; }}
  nav ul {{ list-style: none; }}
  nav ul li a {{ display: block; padding: 7px 16px; font-size: 12px;
                  color: var(--muted); text-decoration: none;
                  border-left: 2px solid transparent;
                  transition: all 0.15s; }}
  nav ul li a:hover {{ color: var(--text); border-left-color: var(--blue);
                         background: rgba(255,255,255,0.03); }}
  main {{ margin-left: 220px; padding: 32px; flex: 1; max-width: 1400px; }}
  header {{ margin-bottom: 24px; }}
  header h1 {{ font-size: 24px; font-weight: 700; color: var(--text); }}
  header p {{ color: var(--muted); font-size: 13px; margin-top: 4px; }}
  .kpi-row {{ display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 28px; }}
  .kpi-card {{ background: var(--panel); border: 1px solid var(--border);
                border-radius: 8px; padding: 16px 20px; flex: 1; min-width: 120px; }}
  .kpi-label {{ font-size: 11px; color: var(--muted); text-transform: uppercase;
                 letter-spacing: 0.5px; margin-bottom: 6px; }}
  .kpi-value {{ font-size: 22px; font-weight: 700; }}
  section {{ margin-bottom: 28px; }}
  section h2 {{ font-size: 14px; font-weight: 600; color: var(--muted);
                 text-transform: uppercase; letter-spacing: 0.5px;
                 margin-bottom: 10px; padding-bottom: 6px;
                 border-bottom: 1px solid var(--border); }}
  .chart-wrapper {{ background: var(--panel); border: 1px solid var(--border);
                      border-radius: 8px; overflow: hidden; padding: 4px; }}
  ::-webkit-scrollbar {{ width: 6px; }} 
  ::-webkit-scrollbar-track {{ background: var(--bg); }}
  ::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 3px; }}
</style>
</head>
<body>
<nav>
  <h1>ML Strategy</h1>
  <ul>{NAV_ITEMS}</ul>
</nav>
<main>
  <header>
    <h1>ML Stock Strategy — Backtest Report (Vectorbt Accuracy)</h1>
    <p>Walk-Forward Purged CV &nbsp;|&nbsp; LightGBM &nbsp;|&nbsp; Top-30 Monthly Rebalance &nbsp;|&nbsp;
       Period: {equity.index[0].strftime('%Y-%m')} ~ {equity.index[-1].strftime('%Y-%m')} &nbsp;|&nbsp;
       Generated: 2026-02-28</p>
  </header>
  <div class="kpi-row">{STATS_ROW}</div>
  {SECTIONS}
</main>
</body>
</html>"""

output_path = "backtest_report.html"
with open(output_path, "w", encoding="utf-8") as f:
    f.write(HTML)

print(f"\n✅ Report saved: {output_path}")
print(f"   Sections: {len(figs_html)}")
print(f"   Open with: open {output_path}")

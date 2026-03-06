# AGENTS.md — ML 選股策略專案知識庫

## 專案概述

台股 ML 因子選股回測專案。原先使用本地伺服器，因免費版 Token 資料過期斷層（2018年後無資料），已全面改寫為 **FinMind 抓取資料 + vectorbt 回測 + LightGBM 預測** 的架構。

### 技術棧

- **語言**：Python 3.13（via uv）
- **套件管理**：uv
- **資料來源**：FinMind API（資料存在 `finmind_cache/`）
- **ML 模型**：LightGBM（處理 NaN 非常強，支援 Non-linear）
- **回測框架**：vectorbt（快速向量化回測）
- **視覺化報表**：Plotly（`reports/generate_report.py` 生成 HTML）
- **自動化**：GitHub Actions（每日 UTC 07:30 = 台灣 15:30 執行）
- **前端部署**：GitHub Pages → https://larryinmexico.github.io/tw_quant_test/

---

## 核心執行指令

### 1. 更新資料庫 (FinMind Cache)
嚴守 FinMind 免費版每小時 600 次限制，腳本具備斷點續傳功能。
```bash
# 抓取價量、營收、三大法人 (~4 小時)
source .venv/bin/activate && python3 data_loaders/01_fetch_finmind_data.py

# 抓取基本面 (PE/PB/殖利率/財務比率) (~4~12 小時)
source .venv/bin/activate && python3 data_loaders/02_fetch_fundamental_data.py --dataset all
```

### 2. 執行策略回測
```bash
source .venv/bin/activate && python3 strategy.py
```

### 3. 輸出 HTML 績效報表
```bash
source .venv/bin/activate && python3 reports/generate_report.py
open frontend/backtest_report.html
```

### 4. 更新 Live Dashboard
```bash
source .venv/bin/activate && python3 reports/generate_live_dashboard.py
open frontend/index.html
```

### 5. 執行 Unit Tests
```bash
source .venv/bin/activate && python3 -m pytest tests/ -v
# 預期結果：25 passed, 0 failed（約 1 秒）
# 不需要 API Token 或真實資料，全部使用 mock data
```

---

## GitHub Actions 自動化

**排程**：`cron: '30 7 * * 1-5'` = **UTC 07:30 (台灣 15:30)，週一到週五**

**延遲說明**：
- Actions 觸發後需等 GitHub runner 排隊，官方說 0~30 分鐘正常，高峰期更長
- 實際 LINE 收到通知時間 ≈ 15:30 + 5~25 分鐘

**執行流程**：
1. 安裝 Python 套件
2. `python3 live_trade.py` → 結算淨值、更新 portfolio.json、發 LINE
3. `python3 reports/generate_live_dashboard.py` → 生成 frontend/index.html
4. git push portfolio.json + frontend/index.html
5. Deploy 到 GitHub Pages

---

## Live Trading 引擎

### `live_trade.py` — 核心參數
```python
FEE_RATE          = 0.001425   # 手續費 0.1425%（單邊）
TAX_RATE          = 0.003      # 交易稅 0.3%（賣出）
SLIPPAGE          = 0.001      # 滑價 0.1%（雙向）
MAX_INVEST_RATIO  = 0.90       # 最多投入 90%（保留 10% 現金）
MAX_SINGLE_WEIGHT = 0.08       # 單支股票最多 8%（迭代收斂 cap）
GITHUB_PAGES_URL  = "https://larryinmexico.github.io/tw_quant_test/"
```

**費用模型**（每次換倉一圈約 0.54%）：
| 方向 | 手續費 | 交易稅 | 滑價 |
|------|--------|--------|------|
| 買進 | 0.1425% | — | +0.1% |
| 賣出 | 0.1425% | 0.3% | -0.1% |

**空頭濾網**：0050 即時股價 < 60日均線 → 清倉避險

**中文名稱**：呼叫 FinMind `TaiwanStockInfo` API 取得，顯示在 LINE 與前端

**cost_basis 成本追蹤**：
- 每次買入時以「成交後均價（含 0.1% 滑價）」寫入 `portfolio.json`
- 換倉賣出時計算 `realized_pnl = 賣出淨收入 - 成本`
- LINE 通知顯示「換倉實現損益」，Dashboard 顯示每筆交易明細

**force_rebalance 機制**：
- 在 `portfolio.json` 中設定 `"force_rebalance": true` 可強制觸發完整換倉
- 執行一次後自動清除（reset to `false`）
- 適用場景：模型更新後、調整參數後希望立即生效

### `reports/generate_live_dashboard.py` — Dashboard 生成器
- 讀取 `portfolio.json`，生成 `frontend/index.html`
- 圓餅圖：Bloomberg/TradingView 藍色系配色（20色調色盤）
- RWD 適應：≤1024px 垂直排列，≤640px 手機版（隱藏部分欄位）
- 底部：每日 NAV 歷史（30天）+ 換倉交易損益明細（含 realized_pnl）

---

## Unit Tests

位置：`tests/` 目錄

| 檔案 | 測試項目 | 測試數 |
|------|---------|--------|
| `test_live_trade.py` | 買賣費用計算、滑價方向、weight cap、現金保留 ≥10%、NAV 計算 | 13 |
| `test_strategy_utils.py` | softmax 加總=1、高/低溫度行為、zscore 形狀、winsorize、weight cap 迭代收斂 | 12 |

**執行**：`python3 -m pytest tests/ -v`

**重要**：`apply_weight_cap` 使用**迭代收斂算法**（最多20次）。一次 clip 後 renorm 可能讓其他股票重新超過8%上限，需迭代直到所有股票 ≤ 8%。此算法僅在持股數量 ≥ `ceil(1/cap)` = 13 支時才有數學上可行的解。

---

## 策略版本進化史

| 版本 | 特色與架構 | 測試期 | 結果（嚴格方法論） |
|---|---|---|---|
| **v3** | Pandas 3.0 相容修復、自建簡單回測模擬器 | 2022~2024 | CAGR +34.9%（只算進場月，費用低估）|
| **v4** | vectorbt + 14個因子 + Regime + Softmax | 2022~2024 | CAGR +21.5% / Sharpe 1.11 |
| **v5** | 加入基本面因子（Earnings Yield, PB, 殖利率）| 2022~2024 | CAGR +17.0% / Max DD -22% |
| **(P1 修正前)** | 延長至 2020~2026 + 流動性過濾 + Top-8 ICIR（全資料）| 2020~2026 | CAGR +17.22%（**費用低估 3x + Benchmark 失真**）|
| **(current / P1+P2)** | Phase 1（手續費修正 + Benchmark 改用 yfinance） + Phase 2（Fold-Internal IC，移除 factor selection lookahead bias） | 2020~2026（含 COVID）| CAGR **+4.59%** / Total +37.9% / Sharpe 0.30 / Max DD -39.55% |

> **重要洞見**：0050 (Benchmark) 在 2020~2026 的真實 CAGR 為 **+26.59%**（yfinance 還原 2020年11月 3:1 分割）。目前策略 CAGR 4.59% 跑輸 0050 甚多。原因：方法論修正後消除了 lookahead bias（全資料選因子）與手續費低估，導致數字大幅下修。這是可信的基準，策略有很大的改進空間。

---

## 模型與演算法細節（current）

1. **資料對齊 (Wide Format)**：
   全數轉為 `(date, stock_id)` 的寬表再 Unstack，解決 Pandas 3.0 `stack(dropna=True)` 改版造成的因子錯位問題。
2. **Rolling Walk-Forward CV (Purged)**：
   訓練 48 個月，Purge 1 個月避免 Lookahead Bias，測試 3 個月。步進式迴圈訓練 LightGBM，共約 24 次 retrain。
3. **Fold-Internal IC 分析（Phase 2 已修復）**：
   每次 retrain 時，在當次的訓練集內計算 ICIR，動態選出 Top-8 絕對 ICIR 因子，完全消除因子選擇的 lookahead bias。
   **最穩定因子**（24折中被選到的次數）：
   ```
   vol_ratio        24/24  最穩定
   mom_1m_ra        23/24
   trust_net_10d    22/24
   mom_1m           21/24
   rsi_14           19/24
   atr_rel          19/24
   mom_6m           16/24  中等穩定
   price_52w         8/24  不穩定（只在特定市場環境有效）
   foreign_net_20d   3/24  幾乎無效
   ```
4. **特徵處理**：
   因子進行 Cross-Sectional Z-Score（截面標準化），避免受大盤絕對數值影響。
5. **目標值 (Y)**：
   下個月的「超額報酬」（Next Month Return - Median Market Return）。
6. **Softmax 選股權重**：
   取出預測前 20 名，用 `softmax(score / temp)` 分配權重。溫度 `WEIGHT_TEMP=5.0`（接近等權）。
7. **流動性過濾**：
   建倉前過濾掉 30 日均量 < 3000萬台幣的股票，避免買進難以實際成交的小市值股。
8. **LightGBM 強化正則化（防 Overfitting）**：
   `max_depth=3, num_leaves=15, reg_alpha=0.5, reg_lambda=2.0, min_child_samples=30`
9. **Benchmark 計算（修正後）**：
   使用 yfinance `auto_adjust=True` 抓取 0050.TW，自動還原 2020年11月 3:1 股票分割。
   FinMind 快取的原始價格未處理分割，若直接使用會造成 CAGR 嚴重失真（-22% vs 真實 +26.59%）。
10. **手續費（修正後）**：
    `FEE = 0.001425`（0.1425% 單邊），vectorbt 的 `fees` 參數為單邊費率。
    舊版誤用 `1.425/1000/3 = 0.0475%`，低估交易成本約 3倍。

---

## 已知問題與限制

| 問題 | 狀態 | 說明 |
|------|------|------|
| 策略跑輸 0050 | 待改進 | Phase 1+2 修正後真實 CAGR 4.59% vs 0050的 26.59% |
| `price_52w` 因子不穩定 | 待研究 | 24個 fold 中只有 8 次被選到，是 Phase 3 主要研究標的 |
| 高嚴重度問題（流動性、回測期）| 已修正 | 見 high_severity_plan.md |

---

## 檔案結構說明

### 核心腳本（目前使用中）
- `data_loaders/01_fetch_finmind_data.py` - 下載台股 OHLCV, 營收, 法人
- `data_loaders/02_fetch_fundamental_data.py` - 下載台股其他基本面 (PE, PB)
- `strategy.py` - 終極動能策略，含 Fold-Internal IC、流動性過濾、yfinance Benchmark
- `live_trade.py` - 虛擬基金核心引擎（每日結算 + LINE推播 + 換倉 + cost_basis）
- `reports/generate_report.py` - 利用 vectorbt 精確變數生成 14 張圖表的 HTML 報告
- `reports/generate_live_dashboard.py` - Live Dashboard 生成（TradingView 風格）
- `tests/` - Unit Tests（25個，不需 API Token）
- `finmind_cache/` - 所有原始 `.pkl` 快取檔與寬表
- `frontend/index.html` - Live Dashboard（GitHub Pages 前端）
- `portfolio.json` - 虛擬倉位 + NAV 歷史 + 交易紀錄 + cost_basis

### Research Notebooks（探索用，按序執行）
- `Research/01_Data_Pipeline.ipynb` - 資料載入 + 流動性快速檢查
- `Research/02_Feature_Engineering.ipynb` - 14 因子計算 + IC 分析 + Top-8 ICIR 選因子
- `Research/03_Model_Training.ipynb` - Walk-Forward Purged CV + LightGBM + Regime Filter
- `Research/04_Backtester.ipynb` - vectorbt 回測 + 月度熱力圖 + 儲存結果

---

## Phase 3：策略優化研究路線圖

> **前提**：Phase 1（手續費、Benchmark 修正）和 Phase 2（Fold-Internal IC）已完成。
> 以下研究需在修正後的正確環境下進行，避免在有偏差的基礎上優化。

### P3-A：因子穩定性深度分析（最優先）

**目標**：找出在所有市場環境下都穩定有效的因子組合。

執行 `Research/06_Factor_Stability.ipynb`：
```python
# 按年分解每個因子的 ICIR
# 目標：找到 2020/2021/2022/.../2025 每一年 ICIR 符號相同的因子
# 如果一個因子某年 ICIR=+0.4、另一年=-0.3，則不應使用

for year in [2020, 2021, 2022, 2023, 2024, 2025]:
    year_data = X[X.index.get_level_values(0).year == year]
    # 計算該年份的 ICIR...
    # 期望結果：vol_ratio 和 mom_1m_ra 應該是最穩定的（Fold 分析已顯示 24/24）
```

**預期洞察**：
- `vol_ratio`（24/24 被選）應該每年 IC 方向一致
- `price_52w`（只有 8/24 被選）應該在某些年份 IC 符號相反，代表只在特定市場有效

---

### P3-B：Benchmark 對齊問題（第二優先）

**問題**：yfinance 的 0050.TW `auto_adjust=True` 包含了股息再投入調整，但你的策略**不**包含股息再投入（每月換股，持股期間沒有股息調整）。這可能造成被比較的基準略高於公平值。

**解決方案**：建立公平的 Benchmark
```python
# 方法 1：使用不含股息再投入的 0050 報酬（更公平）
bm_no_div = yf.download("0050.TW", auto_adjust=False)["Adj Close"]  # 只含拆股還原

# 方法 2：加入策略的股息估算（更複雜）
# 假設持股期間平均殖利率 3%，+補正到策略報酬
adj_cagr = strategy_cagr + 0.03  # 粗估
```

---

### P3-C：持股數量優化

**目標**：找到 risk-adjusted return 最佳的持股數量。

執行 `Research/07_Portfolio_Size_Sweep.ipynb`：
```python
results = {}
for top_k in [5, 10, 15, 20, 30, 50]:
    # 修改 TOP_K 參數，重跑 walk-forward + vectorbt
    # 記錄 CAGR / Max DD / Sharpe / Turnover
    results[top_k] = {"cagr": ..., "sharpe": ..., "max_dd": ..., "turnover": ...}

# 預期：持股越少 CAGR 越高但 Max DD 也越高，需找到膝點
# 目前 TOP_K=20；持股 5~10 可能 Sharpe 更高
```

---

### P3-D：Softmax 溫度優化

**目標**：目前 `WEIGHT_TEMP=5.0` 幾乎是等權分配，softmax 幾乎沒有效果。

**理論**：降低溫度可讓高信心股票佔更多比重，但 Max DD 也會提高。

執行 `Research/08_Temperature_Sweep.ipynb`：
```python
for temp in [1.0, 2.0, 3.0, 5.0, 10.0]:
    # TEMP=1.0：非常集中，前幾名佔壓倒性比重
    # TEMP=5.0：接近等權（現狀）
    # TEMP=10.0：完全等權
    # 尋找 Sharpe 最優的 TEMP 值
```

---

### P3-E：換倉成本（Turnover）分析

**目標**：估算每月換倉的實際成本影響，測試換倉約束。

執行 `Research/09_Turnover_Cost_Analysis.ipynb`：
```python
# 計算每月換倉比例（turnover rate）
# 計算真實年化交易成本
# 測試：allow max 40% / 60% turnover per month
# 觀察：換倉越多成本越高，可能不如減少換倉

monthly_turnover = []
for t in range(1, len(weights_df)):
    prev = weights_df.iloc[t-1]
    curr = weights_df.iloc[t]
    turnover = (prev - curr).abs().sum() / 2
    monthly_turnover.append(turnover)

avg_turnover = pd.Series(monthly_turnover).mean()
annual_cost  = avg_turnover * 12 * (FEE*2 + TAX + SLIPPAGE*2)
print(f"年化換倉成本：{annual_cost:.2%}")
```

---

### 研究優先序和預期效益

| 研究 | 優先序 | 預期效益 | 難度 |
|------|--------|---------|------|
| P3-A 因子穩定性 | 1 | **高**（找出真正穩定的因子，是後續一切優化的基礎）| 中 |
| P3-C 持股數量 | 2 | **中**（可能從 CAGR 4.59% 提升至 7~10%）| 低 |
| P3-D Softmax 溫度 | 3 | **低~中** | 低 |
| P3-E 換倉成本 | 4 | **中**（降低磨損後估計可改善約 1~2% CAGR）| 低 |
| P3-B Benchmark 對齊 | 5 | **低**（只是公平性調整，不影響策略本身）| 低 |

---

## 高嚴重度問題（已修正）

| 問題 | 修正狀態 | Commit |
|------|---------|--------|
| 手續費計算錯誤（`FEE / 3`）| **已修正（Phase 1-A）** | `99d491f` |
| 0050 Benchmark 失真（未還原分割）| **已修正（Phase 1-B，改用 yfinance）** | `99d491f` |
| `STOP_LOSS` 死變數 | **已清除（Phase 1-C）** | `99d491f` |
| 因子選擇 Lookahead Bias | **已修正（Phase 2）** | `c39e9fd` |
| 回測期太短 | **已修正（延長至 2020~2026）** | 先前 commit |
| 流動性過濾缺失 | **已修正（3000萬日均量）** | 先前 commit |

# AGENTS.md — ML 選股策略專案知識庫 

## 專案概述

台股 ML 因子選股回測專案。原先使用 本地伺服器，因免費版 Token 資料過期斷層（2018年後無資料），已全面改寫為 **FinMind 抓取資料 + vectorbt 回測 + LightGBM 預測** 的架構。

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
# 建議先只跑 per: python3 data_loaders/02_fetch_fundamental_data.py --dataset per
source .venv/bin/activate && python3 data_loaders/02_fetch_fundamental_data.py --dataset all
```

### 2. 執行策略回測 (最新 版)
```bash
source .venv/bin/activate && python3 strategy.py
```

### 3. 輸出 HTML 績效報表
```bash
source .venv/bin/activate && python3 reports/generate_report.py
open backtest_report.html
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
FEE_RATE          = 0.001425   # 手續費 0.1425%
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

### `reports/generate_live_dashboard.py` — Dashboard 生成器
- 讀取 `portfolio.json`，生成 `frontend/index.html`
- 圓餅圖：Bloomberg/TradingView 藍色系配色（20色調色盤）
- RWD 適應：≤1024px 垂直排列，≤640px 手機版（隱藏部分欄位）
- 底部：每日 NAV 歷史（30天）+ 換倉交易損益明細

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

| 版本 | 特色與架構 | 結果 (2022-2024) |
|---|---|---|
| **v3** | Pandas 3.0 相容修復、自建簡單回測模擬器 | CAGR +34.9% (只算有進場月份) |
| **v4** | 全面改用 vectorbt。14 個技術面/籌碼面因子。Multi-signal Regime (0050 60MA+20MA)。Softmax 權重。 | CAGR **+21.5%** / 勝率 56% / Max DD -21% |
| **v5** | 加入 4 個基本面因子 (Earnings Yield, PB, 殖利率, PE Momentum)。修改 LightGBM NaN 處理門檻 (保留 60% 因子覆蓋即可) | CAGR **+17.0%** / Max DD -22% |
| **(current)** | 回歸 v4 純動量與籌碼配置 (棄用會拉低近年績效的估值因子)。完全串接 vectorbt 每日精確淨值至 Plotly，並修復了下載真實 EPS 的腳本 (data_loaders/03_fix_financial.py)。 | CAGR **+21.52%** / Total Return **+79.18%** / Sharpe 1.11 |

> **洞見**：v5 績效低於 v4，原因在於 2022-2024 是 AI 成長股領漲的行情，價值因子反而會錯失飆股。直接打包了 v4 的強大動能邏輯與完美的 vectorbt 精確報表，為目前最強穩定版本。

---

## 檔案結構說明

### 核心腳本 (目前使用中)
- `data_loaders/01_fetch_finmind_data.py` - 下載台股 OHLCV, 營收, 法人
- `data_loaders/02_fetch_fundamental_data.py` - 下載台股其他基本面 (PE, PB)
- `data_loaders/03_fix_financial.py` - 修正後的財報下載器 (抓取 EPS, 現金流量等)
- `strategy.py` - 終極動能策略 (最佳版本，精確輸出 vectorbt 狀態)
- `live_trade.py` - 虛擬基金核心引擎（每日結算 + LINE推播 + 換倉）
- `reports/generate_report.py` - 利用 vectorbt 精確變數生成 14 張圖表的終極 HTML 報告
- `reports/generate_live_dashboard.py` - Live Dashboard 生成（TradingView 風格）
- `tests/` - Unit Tests（25個，不需 API Token）
- `finmind_cache/` - 所有原始 `.pkl` 快取檔與寬表 (Wide DataFrame)
- `frontend/index.html` - Live Dashboard（GitHub Pages 前端）
- `portfolio.json` - 虛擬倉位 + NAV 歷史 + 交易紀錄

### 已廢棄 / 可刪除的舊檔案
以往基於 Jupyter 和舊版策略的腳本，已全數被清除或取代。
- 舊版策略: `strategy_v3.py`, `strategy_v4.py`, `strategy_v5.py`
- 舊版報表: `reports/generate_report.py`, `generate_report_v5.py`

---

## 模型與演算法細節 (v5)

1. **資料對齊 (Wide Format)**：
   全數轉為 `(date, stock_id)` 的寬表再 Unstack，解決 Pandas 3.0 `stack(dropna=True)` 改版造成的因子錯位問題。
2. **Rollling Walk-Forward CV (Purged)**：
   訓練 36 個月，Purge 1 個月避免 Lookahead Bias，測試 3 個月。步進式迴圈訓練 LightGBM。
3. **特徵處理**：
   因子進行 Cross-Sectional Z-Score (截面標準化)，避免受大盤絕對數值影響。
4. **目標值 (Y)**：
   下個月的「超額報酬」 (Next Month Return - Median Market Return)。
5. **Softmax 選股權重**：
   不採用 Equal Weight，而是取出預測前 20 名，用 `softmax(score / temp)` 分配權重，提高模型信心部位的佔比。溫度 `WEIGHT_TEMP=5.0` → 趨近等權，可調低至 1.5~2.0 使高分股佔更多比例。
6. **NaN 處理**：
   決策樹原生支援 Missing Values，v5 允許單檔股票缺失達 40% 的因子仍參與訓練，避免基本面資料缺失導致樣本被丟棄。

## 高嚴重度問題待解決

見 brain 目錄下的 `high_severity_plan.md`，包含：
1. **回測期太短**（延長至 2019+，需重跑 FinMind 下載）
2. **流動性過濾缺失**（在 strategy.py 加入 3000萬日均量門檻）
3. **Overfitting 風險**（因子精簡至Top-8 ICIR + LightGBM 正則化加強）

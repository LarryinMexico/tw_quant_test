# CLAUDE.md — ML 選股策略專案知識庫 (v6)

## 專案概述

台股 ML 因子選股回測專案。原先使用 FinLab，因免費版 Token 資料過期斷層（2018年後無資料），已全面改寫為 **FinMind 抓取資料 + vectorbt 回測 + LightGBM 預測** 的架構。

### 技術棧

- **語言**：Python 3.13（via uv）
- **套件管理**：uv
- **資料來源**：FinMind API（資料存在 `finmind_cache/`）
- **ML 模型**：LightGBM（處理 NaN 非常強，支援 Non-linear）
- **回測框架**：vectorbt（快速向量化回測）
- **視覺化報表**：Plotly（`generate_report.py` 生成 HTML）

---

## 核心執行指令

### 1. 更新資料庫 (FinMind Cache)
嚴守 FinMind 免費版每小時 600 次限制，腳本具備斷點續傳功能。
```bash
# 抓取價量、營收、三大法人 (~4 小時)
source .venv/bin/activate && python3 01_fetch_finmind_data.py

# 抓取基本面 (PE/PB/殖利率/財務比率) (~4~12 小時)
# 建議先只跑 per: python3 02_fetch_fundamental_data.py --dataset per
source .venv/bin/activate && python3 02_fetch_fundamental_data.py --dataset all
```

### 2. 執行策略回測 (最新 v6 版)
```bash
source .venv/bin/activate && python3 strategy_v6.py
```

### 3. 輸出 HTML 績效報表
```bash
source .venv/bin/activate && python3 generate_report_v6.py
open backtest_report_v6.html
```

---

## 策略版本進化史

| 版本 | 特色與架構 | 結果 (2022-2024) |
|---|---|---|
| **v3** | Pandas 3.0 相容修復、自建簡單回測模擬器 | CAGR +34.9% (只算有進場月份) |
| **v4** | 全面改用 vectorbt。14 個技術面/籌碼面因子。Multi-signal Regime (0050 60MA+20MA)。Softmax 權重。 | CAGR **+21.5%** / 勝率 56% / Max DD -21% |
| **v5** | 加入 4 個基本面因子 (Earnings Yield, PB, 殖利率, PE Momentum)。修改 LightGBM NaN 處理門檻 (保留 60% 因子覆蓋即可) | CAGR **+17.0%** / Max DD -22% |
| **v6** | 回歸 v4 純動量與籌碼配置 (棄用會拉低近年績效的估值因子)。完全串接 vectorbt 每日精確淨值至 Plotly，並修復了下載真實 EPS 的腳本 (03_fix_financial.py)。 | CAGR **+21.52%** / Total Return **+79.18%** / Sharpe 1.11 |

> **洞見**：v5 績效低於 v4，原因在於 2022-2024 是 AI 成長股領漲的行情，價值因子反而會錯失飆股。v6 直接打包了 v4 的強大動能邏輯與完美的 vectorbt 精確報表，為目前最強穩定版本。

---

## 檔案結構說明

### 核心腳本 (目前使用中)
- `01_fetch_finmind_data.py` - 下載台股 OHLCV, 營收, 法人
- `02_fetch_fundamental_data.py` - 下載台股其他基本面 (PE, PB)
- `03_fix_financial.py` - 修正後的財報下載器 (抓取 EPS, 現金流量等)
- `strategy_v6.py` - v6 終極動能策略 (最佳版本，精確輸出 vectorbt 狀態)
- `generate_report_v6.py` - 利用 vectorbt 精確變數生成 14 張圖表的終極 HTML 報告
- `finmind_cache/` - 所有原始 `.pkl` 快取檔與寬表 (Wide DataFrame)

### 已廢棄 / 可刪除的舊檔案
以往基於 Jupyter 和舊版策略的腳本，已全數被清除或取代。
- 舊版策略: `strategy_v3.py`, `strategy_v4.py`, `strategy_v5.py`
- 舊版報表: `generate_report.py`, `generate_report_v5.py`

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
   不採用 Equal Weight，而是取出預測前 20 名，用 `softmax(score / temp)` 分配權重，提高模型信心部位的佔比。
6. **NaN 處理**：
   決策樹原生支援 Missing Values，v5 允許單檔股票缺失達 40% 的因子仍參與訓練，避免基本面資料缺失導致樣本被丟棄。

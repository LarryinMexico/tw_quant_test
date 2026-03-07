# 台股 ML 量化選股系統

基於 Python 與 Machine Learning 的台股量化選股與自動回測紙上交易系統。
採用 FinMind 作為免費每日資料來源，以 vectorbt 進行精確回測，LightGBM 進行選股預測。

## 系統架構

整個專案由三個核心階段構成：

### 1 資料層
- `data_loaders/01_fetch_finmind_data.py` 抓取價量、月營收、三大法人數據
- `data_loaders/02_fetch_fundamental_data.py` 抓取財報比率（PE/PB/殖利率）

### 2 回測與策略層
- `strategy.py` 終極 ML 選股模型
  - Walk-Forward Purged CV（48個月訓練視窗，Purge 1個月）
  - Fold-Internal IC 分析（在每個 fold 的訓練資料內動態選出 Top-8 ICIR 因子）
  - Multi-signal Regime Filter（0050 均線濾網，只在多頭進場）
  - Softmax 信心加權（前 20 名持股）
  - 流動性過濾（30日均量 > 3000萬才可進場）
- `reports/generate_report.py` 生成 14 張圖表的 Plotly Dashboard

### 3 實盤紙上交易
- `live_trade.py` 每日盤後（台灣時間約 15:30）自動：
  - 從 yfinance + FinMind 抓取最新收盤價，計算未實現損益
  - 月底自動換倉，計算並記錄真實**實現損益**（基於 cost_basis 成本基礎）
  - 推播 LINE 通知 + 更新 GitHub Pages Dashboard
- `portfolio.json` 虛擬存摺（含 cost_basis 每股成本）

## 最新績效數字（Phase 1+2 嚴格方法論修正後）

> 此為截至 2026-03 的回測結果（2020~2026，含 COVID 崩盤）

| 指標 | 策略 (Phase 3 最終版) | 0050 Benchmark |
|------|------|---------------|
| CAGR | +9.25% | +26.59% |
| Total Return | +88.58% | +441.5% |
| Sharpe | 0.54 | 1.23 |
| Max DD | -28.30% | -33.83% |

> 0050 Benchmark 使用 yfinance `auto_adjust=True` 計算，已正確還原 2020年11月的 3:1 股票分割。

**重要說明**：過去版本 CAGR 顯示 +17.22% 是因為手續費低估（`FEE/3` 的計算錯誤）+ Benchmark 未還原分割（顯示 0.29% 假值）+ 因子選擇 Lookahead Bias。Phase 1~3 修正與優化後，採用 `TOP_K=40`、`keep_top_k=80` (Turnover Inertia)、`WEIGHT_TEMP=5.0` (Equal-Weighting) 大幅降低換倉摩擦成本與集中風險，使 Max DD 降至 -28.30%，CAGR 回升至 9.25%。真實反映了扣除高昂手續費與滑價後的實盤預期數字。

## Live Dashboard

https://larryinmexico.github.io/tw_quant_test/

每日盤後（台灣時間 15:30 後約 5～30 分鐘）自動更新

## 如何在本地端手動更新策略

```bash
# 1. 更新資料（約 4 小時）
source .venv/bin/activate
python3 data_loaders/01_fetch_finmind_data.py

# 2. 重新訓練模型（約 5~15 分鐘）
python3 strategy.py

# 3. 生成回測報告
python3 reports/generate_report.py
```

## 雲端全自動化設計

透過 GitHub Actions（UTC 07:30 = 台灣 15:30，週一到週五），每日盤後自動：
1. 執行 `live_trade.py` 計算損益、更新資料
2. 生成最新 Dashboard HTML
3. Push 更新至 GitHub，觸發 GitHub Pages 部署
4. 發送 LINE 通知（含 Dashboard 連結與當日損益）

## 關鍵設計決策

### 為何使用 Fold-Internal IC 分析（Phase 2）
原先在全部 2020~2026 資料上計算 ICIR 再選因子，等同讓因子選擇「看到了未來」（Lookahead Bias）。
修正後，每次 Walk-Forward retrain 時只在該 fold 的訓練資料內計算 ICIR，以真正的 OOS 方式選因子。

### 為何放棄基本面因子
2022~2024 是 AI 成長股領漲的牛市，價值投資因子反而會讓模型錯失飆漲暴發股。
目前使用技術面 + 籌碼面 14 個因子，Fold-Internal IC 自動選出最穩定的 Top-8。

### 費用模型
| 方向 | 手續費 | 交易稅 | 滑價 | 合計 |
|------|--------|--------|------|------|
| 買進 | 0.1425% | — | +0.1% | 0.2425% |
| 賣出 | 0.1425% | 0.3% | -0.1% | 0.5425% |
| **一圈** | | | | **約 0.79%** |

## Unit Tests

測試涵蓋核心交易邏輯，不需要 API Token，全部使用 mock data

```bash
source .venv/bin/activate
python3 -m pytest tests/ -v
```

預期結果：25 passed

| 測試檔案 | 涵蓋項目 |
|---------|---------|
| `tests/test_live_trade.py` | 買入手續費、賣出稅費、滑價方向、Weight Cap 8% 上限、現金保留 ≥10%、NAV 計算 |
| `tests/test_strategy_utils.py` | Softmax 權重加總=1、高/低溫度行為、Z-score 截面正規化、Winsorize ±3σ、迭代 Weight Cap 收斂 |

## force_rebalance 手動換倉

在 `portfolio.json` 加一行，下一次 Actions 執行時就會強制換倉（執行後自動清除）：

```json
{
  "force_rebalance": true,
  ...
}
```

適用於：更換模型、調整參數、重大市場事件後希望立即更新倉位。

## Research Notebooks（探索與優化用）

`Research/` 目錄下依序執行，共享記憶體變數：

| Notebook | 用途 |
|----------|------|
| `01_Data_Pipeline.ipynb` | 資料載入 + 流動性快速檢查 |
| `02_Feature_Engineering.ipynb` | 14 因子計算 + IC 分析 + Top-8 ICIR 篩選 |
| `03_Model_Training.ipynb` | Walk-Forward + LightGBM + Regime Filter |
| `04_Backtester.ipynb` | vectorbt 回測 + 月度報酬熱力圖 |

詳細的 Phase 3 優化研究路線圖見 `AGENTS.md`。

# 台股 ML 量化選股系統

基於 Python 與 Machine Learning 的台股量化選股與自動回測紙上交易系統。
採用 FinMind 作為免費每日資料來源，以 vectorbt 進行精確回測，LightGBM 進行選股預測。

## 📖 策略白話文解釋 (我的 AI 到底在幹嘛？)

如果你沒有財金或程式背景，可以把這套系統想像成一個**「每個月只上一次班的超級基金經理人」**。它運作的邏輯非常清晰防呆：

### 1. 🌍 大盤安全濾網 (0050 紅綠燈)
 AI 不管多會選股，遇到股災（像 COVID 或 2022 狂跌）也是會死。
所以它每天都會看一眼代表大盤的 `0050 (台灣 50)`。如果 0050 的股價**跌破季線 (60日均線)**，AI 就會認定市場進入「空頭暴風雨」。
- **應對：強迫出清所有股票，100% 抱著現金避險。**

### 2. 🤖 AI 狗仔隊評分 (14個財富密碼因子)
如果大盤安全，AI 基金經理人每個月月底會把台股所有的 1,800 間公司拉出來打分數。打分數的線索（因子）包含：
- **短線急單**：投信連買十天、短線上漲動能高。
- **主力狂熱**：RSI 超買區、爆量的股票。
- **基本面輔助**：PE 便宜、營收成長的公司。
AI (LightGBM模型) 透過學習過去 48 個月的歷史，會預測出這 1800 支股票「下個月最可能暴漲」的機率，並列出一份 **全球風雲排行榜**。

### 3. 🎯 資金平權與留校察看 (抗滑價避險網)
雖然 AI 列出了排行榜，但它不會傻傻地把錢全梭哈在第 1 名。它內建了業界的避險機制：
- **買 40 支股票 (分散風險)**：它會挑出排行榜的前 40 名，並且「平均」把錢切成 40 份去買，避免其中一家公司突然倒閉。
- **不要太常換股 (Inertia, 幫你省手續費)**：如果一檔股票原本在名單內，下個月它稍微退步掉到了第 50 名，AI **「依然會留著它不賣」**（留校察看 80 名內都安全）。只有當它退步到 80 名以外徹底沒救了，AI 才會付手續費把它賣掉。這每年幫你省下了將近 10% 的瘋狂換倉成本！

### 4. 🧮 紙上虛擬收銀機 (Live Trade)
系統有一份 `portfolio.json`，就像你的證券 APP 存摺：
- 它會記錄你「每天的帳戶總餘額」。
- 每次月底買賣，它會**老實扣掉 0.1425%手續費、0.3%交易稅，以及怕買不到扣的滑差**。
- 這代表你最後在儀表板看到的 9.25% 年化報酬率，是你真實可以放進口袋裡的錢！

---

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

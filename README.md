這是一套基於Python與MachineLearning的台股量化選股與自動回測紙上交易系統
本系統採用FinMind作為免費每日資料來源
並以vectorbt進行回測

## 系統架構與流程

整個專案由三個核心階段所構成

### 1 資料層
負責向API爬取資料並寫入本地快取
- data_loaders/01_fetch_finmind_data 抓取價量營收三大法人數據
- data_loaders/02_fetch_fundamental_data 抓取財報比率
- data_loaders/03_fix_financial 針對EPS稅後淨利做專屬抓取與修正

### 2 回測與策略層
- strategy 終極模型
  利用過去三年動能與籌碼建立特徵
  經過LightGBM特徵篩選
  並套用趨勢濾網產出每月推薦的Top20股票機率分佈並交給vectorbt回測
- reports/generate_report 生成14張圖表的PlotlyDashboard
  具備極高精準度的淨值回撤風險等分析數據

### 3 實盤與紙上交易
- live_trade 每日盤後從雲端自動啟動
  它會重新抓一次今天最新的價格計算目前虛擬基金部位的未實現損益
  然後推播LINE報告給你（含 Dashboard 網頁連結）
  並在月底自動進行虛擬換股
- portfolio 你的虛擬存摺
  紀錄目前可用的現金餘額買進的股票清單以及過往績效歷史

## 為什麼排除基本面
經過演化測試
由於2022至2024年主要是AI成長股的牛市
價值投資因子反而會導致模型錯失飆漲的暴發股
因此在此時空背景下排除基本面的籌碼與技術動能模型反而是最強解

## 如何在本地端手動更新策略

未來想自己重新跑一次三年歷史回測
只要依序執行下列動作

```bash
source .venv/bin/activate
python3 data_loaders/01_fetch_finmind_data.py
./run.sh
```

## 雲端全自動化設計

我們將程式碼推播至私有的GitHubRepo
透過LINE變數發佈通知
使用GitHubActions保留FinMind的資料庫
讓它每日只需花極短的時間接續抓取增量資訊
每日盤後自動推算出虛擬基金資產與未來部位
發佈到前端網頁與LINE
徹底解放人工盯盤

## GitHub Pages Dashboard

https://larryinmexico.github.io/tw_quant_test/

每日盤後（台灣時間 15:30 後約 5～30 分鐘）自動更新

## Unit Tests

測試涵蓋核心交易邏輯，不需要 API Token，全部使用 mock data

```bash
# 安裝完依賴後執行
source .venv/bin/activate
python3 -m pytest tests/ -v
```

預期結果：25 passed  
測試項目：

| 測試檔案 | 涵蓋項目 |
|---------|---------|
| tests/test_live_trade.py | 買入手續費、賣出稅費、滑價方向、Weight Cap 8% 上限、現金保留 ≥10%、NAV 計算 |
| tests/test_strategy_utils.py | Softmax 權重加總=1、高/低溫度行為、Z-score 截面正規化、Winsorize ±3σ、迭代 Weight Cap 收斂 |

測試設計原則：
- 費用模型：買入 0.1425% + 0.1% 滑價；賣出 0.1425% + 0.3% 稅 + 0.1% 滑價
- Weight cap 使用迭代收斂（最多20次），確保重新正規化後仍無超過8%的股票
- 需要至少13支股票才能在8%上限下分配完整（ceil(1/0.08)=13）

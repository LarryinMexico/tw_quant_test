#!/usr/bin/env python3
"""
02_fetch_fundamental_data.py
============================
下載台股基本面資料（PE、PB、EPS、ROE、毛利率、負債比）
支援斷點續傳，嚴守 FinMind 免費每小時 600 次限制。

資料集：
  TaiwanStockPER              → PE 比、PB 比、殖利率（日頻，per股，直接對應月底選股）
  TaiwanFinancialStatements   → EPS、毛利率、業外收入比（季報，需 forward-fill）
  TaiwanStockBalanceSheet     → 總資產、總負債、股東權益（季報 → ROE、D/E 計算）

執行方式：
  source .venv/bin/activate
  python3 02_fetch_fundamental_data.py

約需要時間（免費 token）：
  TaiwanStockPER    2500 股 × 6.1s = ~4.2 小時  （最重要，先跑這個）
  FinancialStmt     2500 股 × 6.1s = ~4.2 小時
  BalanceSheet      2500 股 × 6.1s = ~4.2 小時

建議：只要 TaiwanStockPER 就夠了（PE + PB 最有用），
      一次跑一個 dataset 中斷後再繼續。
"""

import os, time, requests, argparse
from typing import Optional
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv("FINMIND_API_TOKEN")
if not TOKEN:
    raise ValueError("找不到 FINMIND_API_TOKEN，請確認 .env 有設定此欄位")

# ───── 設定 ──────────────────────────────────────────────────────────────────
START_DATE = "2019-01-01"
END_DATE   = "2024-12-31"
CACHE_DIR  = "finmind_cache"
SLEEP_SEC  = 6.2          # 每次請求後等待（600次/1hr = 1次/6s，留 0.2s 緩衝）
API_URL    = "https://api.finmindtrade.com/api/v4/data"

DATASETS = {
    "per":           ("TaiwanStockPER",           "per"),           # PE, PB, div yield
    "financial":     ("TaiwanFinancialStatements", "financial"),     # EPS, gross profit...
    "balance_sheet": ("TaiwanStockBalanceSheet",  "balance_sheet"), # assets, liabilities
}

# ───── 解析參數 ───────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="FinMind 基本面資料下載器")
parser.add_argument(
    "--dataset", "-d",
    choices=list(DATASETS.keys()) + ["all"],
    default="per",
    help="要下載哪個 dataset（預設：per）"
)
args = parser.parse_args()
dataset_keys = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]

# ───── 建立快取目錄 ───────────────────────────────────────────────────────────
for key in dataset_keys:
    os.makedirs(os.path.join(CACHE_DIR, DATASETS[key][1]), exist_ok=True)

# ───── API 共用函式 ───────────────────────────────────────────────────────────
def api_get(dataset_name: str, stock_id: str) -> Optional[pd.DataFrame]:
    """呼叫 FinMind API，回傳 DataFrame；失敗或無資料回傳 None。"""
    try:
        resp = requests.get(API_URL, params={
            "dataset":    dataset_name,
            "data_id":    stock_id,
            "start_date": START_DATE,
            "end_date":   END_DATE,
            "token":      TOKEN,
        }, timeout=30)
        result = resp.json()
        if result.get("status") == 200:
            data = result.get("data", [])
            return pd.DataFrame(data) if data else pd.DataFrame()
        else:
            msg = result.get("msg", "unknown error")
            if "rate limit" in msg.lower():
                print(f"\n  ⚠️  Rate limit hit! 等待 60 秒…")
                time.sleep(60)
                return api_get(dataset_name, stock_id)   # retry once
            print(f"  ⚠️  {stock_id}/{dataset_name}: {msg}")
            return None
    except Exception as e:
        print(f"  ❌  {stock_id}/{dataset_name} 錯誤: {e}")
        return None


def fetch_dataset(stock_ids: list, api_dataset: str, subdir: str):
    """斷點續傳地下載整個 dataset。"""
    save_dir = os.path.join(CACHE_DIR, subdir)
    downloaded, skipped, empty_count = 0, 0, 0

    print(f"\n📥  下載 {api_dataset}（共 {len(stock_ids)} 支，儲存至 {save_dir}/）")
    print(f"     預計耗時：{len(stock_ids) * SLEEP_SEC / 3600:.1f} 小時（免費 token）")

    for sid in tqdm(stock_ids, desc=f"  {subdir}"):
        save_path = os.path.join(save_dir, f"{sid}.pkl")

        # 斷點續傳：已存在就跳過
        if os.path.exists(save_path):
            skipped += 1
            continue

        df = api_get(api_dataset, sid)

        if df is None:
            # 紀錄失敗但繼續
            time.sleep(SLEEP_SEC)
            continue

        if df.empty:
            # 標記此股無資料（避免重複嘗試）
            df.to_pickle(save_path)
            empty_count += 1
        else:
            df.to_pickle(save_path)
            downloaded += 1

        time.sleep(SLEEP_SEC)

    print(f"\n  ✅  完成：下載 {downloaded} 支，空資料 {empty_count} 支，跳過 {skipped} 支")


# ───── 讀取股票清單 ───────────────────────────────────────────────────────────
stock_list_path = os.path.join(CACHE_DIR, "stock_list.pkl")
if not os.path.exists(stock_list_path):
    print("  下載股票清單…")
    resp = requests.get("https://api.finmindtrade.com/api/v4/data", params={
        "dataset": "TaiwanStockInfo",
        "token":   TOKEN,
    }, timeout=30)
    stock_df = pd.DataFrame(resp.json()["data"])
    # 只保留上市/上櫃普通股（排除 ETF、受益憑證等）
    stock_df = stock_df[stock_df["type"].isin(["tse", "otc"])]
    stock_df.to_pickle(stock_list_path)
    print(f"  股票清單：{len(stock_df)} 支")
else:
    stock_df = pd.read_pickle(stock_list_path)

# 只用 4 位數股票代號（普通股）
stock_ids = [
    sid for sid in stock_df["stock_id"].unique()
    if str(sid).isdigit() and len(str(sid)) == 4
]
print(f"\n  股票數量：{len(stock_ids)} 支（普通股）")

# ───── 執行下載 ───────────────────────────────────────────────────────────────
for key in dataset_keys:
    api_name, subdir = DATASETS[key]
    fetch_dataset(stock_ids, api_name, subdir)

# ───── 產出寬表（合併成 wide pkl for fast loading）────────────────────────────
print("\n\n📦  後處理：合併為寬表…")

for key in dataset_keys:
    api_name, subdir = DATASETS[key]
    save_dir = os.path.join(CACHE_DIR, subdir)
    pkl_files = [f for f in os.listdir(save_dir) if f.endswith(".pkl")]

    if not pkl_files:
        print(f"  ⚠️  {subdir}/ 沒有任何 pkl 檔，跳過合併")
        continue

    # Detect available columns from a sample
    sample_files = [f for f in pkl_files[:10] if os.path.getsize(os.path.join(save_dir, f)) > 1000]
    if not sample_files:
        print(f"  ⚠️  {subdir}/ 無有效資料，跳過")
        continue

    sample_df = pd.read_pickle(os.path.join(save_dir, sample_files[0]))
    print(f"\n  {subdir} 欄位: {sample_df.columns.tolist()}")

    # Determine which columns to pivot based on dataset
    if key == "per":
        value_cols = ["PER", "PBR", "dividend_yield"]
    elif key == "financial":
        # Will vary — use auto detect
        value_cols = [c for c in sample_df.columns if c not in ["date", "stock_id", "type"]][:5]
    elif key == "balance_sheet":
        value_cols = [c for c in sample_df.columns if c not in ["date", "stock_id", "type"]][:5]
    else:
        continue

    print(f"  合併欄位: {value_cols}")

    frames = []
    for f in tqdm(pkl_files, desc=f"  合併 {subdir}"):
        path = os.path.join(save_dir, f)
        if os.path.getsize(path) < 100:
            continue
        try:
            df = pd.read_pickle(path)
            if df.empty or "date" not in df.columns:
                continue
            df["date"] = pd.to_datetime(df["date"])
            frames.append(df)
        except Exception:
            pass

    if not frames:
        print(f"  ⚠️  {subdir} 無有效資料，跳過")
        continue

    all_df = pd.concat(frames, ignore_index=True)

    for col in value_cols:
        if col not in all_df.columns:
            continue
        try:
            wide = all_df.pivot_table(index="date", columns="stock_id", values=col)
            out_path = os.path.join(CACHE_DIR, f"{subdir}_{col.lower()}_wide.pkl")
            wide.to_pickle(out_path)
            print(f"  ✅  {out_path}  shape={wide.shape}")
        except Exception as e:
            print(f"  ⚠️  {col} 合併失敗: {e}")

print("\n✅  所有任務完成！")
print("\n使用方式（在 strategy_v5.py 中）：")
print("  per_wide = pd.read_pickle('finmind_cache/per_per_wide.pkl')    # PE 比")
print("  pbr_wide = pd.read_pickle('finmind_cache/per_pbr_wide.pkl')    # PB 比")

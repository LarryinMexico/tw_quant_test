"""
01_fetch_finmind_data.py
========================
FinMind 台股資料下載器
- 支援斷點續傳（已下載的股票自動跳過）
- 自動限速（每次請求後等待 6 秒，嚴守每小時 600 次免費限制）
- 抓取三種資料集：股價、月營收、三大法人
- 每支股票一次就抓齊 START_DATE ~ END_DATE 全部資料（最省 API）

完成後預計產出：
  finmind_cache/price/       每支股票的 pkl 檔
  finmind_cache/revenue/     每支股票的月營收 pkl
  finmind_cache/institution/ 每支股票法人買賣超 pkl

執行方式：
  .venv/bin/python 01_fetch_finmind_data.py

中斷後再執行會自動從上次停止的地方繼續。
"""
import os
import time
import requests
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv("FINMIND_API_TOKEN")
if not TOKEN:
    raise ValueError("找不到 FINMIND_API_TOKEN，請確認 .env 檔案中有此設定")

# ===== 設定 =====
START_DATE = "2019-01-01"
END_DATE   = "2024-12-31"
CACHE_DIR  = "finmind_cache"
SLEEP_SEC  = 6.1   # 每次請求後等待秒數（600次/小時 = 1次/6秒）
API_URL    = "https://api.finmindtrade.com/api/v4/data"

# 建立快取目錄
for subdir in ["price", "revenue", "institution"]:
    os.makedirs(os.path.join(CACHE_DIR, subdir), exist_ok=True)

# Original api_get function removed as logic integrated into fetch_dataset

def get_stock_list():
    """取得台股上市+上櫃股票清單"""
    cache_file = os.path.join(CACHE_DIR, "stock_list.pkl")
    if os.path.exists(cache_file):
        return pd.read_pickle(cache_file)
    
    resp = requests.get(API_URL, params={
        "dataset": "TaiwanStockInfo",
        "token": TOKEN
    }, timeout=30)
    df = pd.DataFrame(resp.json()["data"])
    # 只保留一般股票：上市(twse) + 上櫃(tpex)，排除興櫃(emerging)
    df = df[df["type"].isin(["twse", "tpex"])].copy()
    # 只保留各股票納入加設日期最新的一筆（同一股票可能有多筆）
    df = df.sort_values("date").drop_duplicates(subset=["stock_id"], keep="last")
    df.to_pickle(cache_file)
    print(f"股票清單：上市 {len(df[df['type']=='twse'])} 支 + 上櫃 {len(df[df['type']=='tpex'])} 支 = 共 {len(df)} 支")
    return df

def fetch_dataset(stock_ids, dataset_name, api_dataset, subdir):
    """下載某個資料集的所有股票，有快取則跳過"""
    print(f"\n📥 開始下載 {dataset_name}（共 {len(stock_ids)} 支）...")
    skipped = 0
    downloaded = 0
    
    for sid in tqdm(stock_ids, desc=dataset_name):
        cache_file = os.path.join(CACHE_DIR, subdir, f"{sid}.pkl")
        empty_marker = os.path.join(CACHE_DIR, subdir, f"{sid}.empty")
        
        req_start = START_DATE
        existing_df = None
        
        if os.path.exists(empty_marker):
            skipped += 1
            continue
            
        if os.path.exists(cache_file):
            try:
                existing_df = pd.read_pickle(cache_file)
                if not existing_df.empty:
                    max_date = pd.to_datetime(existing_df["date"]).max()
                    req_start = (max_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                    
                    if pd.to_datetime(req_start) > pd.to_datetime(END_DATE):
                        # 已經是最新的，不需再抓
                        skipped += 1
                        continue
            except Exception:
                pass # 如果損壞，就從頭開始抓
        
        try:
            resp = requests.get(API_URL, params={
                "dataset": api_dataset,
                "data_id": sid,
                "start_date": req_start,     # <-- 動態起點
                "end_date": END_DATE,
                "token": TOKEN
            }, timeout=30)
            result = resp.json()
            
            if result.get("status") == 200:
                new_data = result.get("data", [])
                new_df = pd.DataFrame(new_data)
                
                if not new_df.empty:
                    if existing_df is not None:
                        df = pd.concat([existing_df, new_df], ignore_index=True)
                    else:
                        df = new_df
                    df.to_pickle(cache_file)
                    downloaded += 1
                else:
                    if existing_df is None:
                        # 從來就沒有資料，真的空殼
                        with open(empty_marker, 'w') as f:
                            f.write('')
                    skipped += 1
            else:
                print(f"  ⚠️  {sid} / {api_dataset} 狀態碼非 200: {result.get('msg')}")
        except Exception as e:
            print(f"  ⚠️  {sid} / {api_dataset} 錯誤: {e}")
        
        time.sleep(SLEEP_SEC)
    
    print(f"  ✅ {dataset_name} 完成：下載 {downloaded} 支，跳過 {skipped} 支（已快取）")

def load_and_pivot(subdir, value_col, date_col="date", id_col="stock_id"):
    """把快取目錄中的所有 pkl 合併成一個 Wide-form DataFrame"""
    frames = []
    for f in os.listdir(os.path.join(CACHE_DIR, subdir)):
        if not f.endswith(".pkl"):
            continue
        df = pd.read_pickle(os.path.join(CACHE_DIR, subdir, f))
        if value_col in df.columns:
            frames.append(df[[id_col, date_col, value_col]])
    
    if not frames:
        return None
    
    all_df = pd.concat(frames, ignore_index=True)
    all_df[date_col] = pd.to_datetime(all_df[date_col])
    wide = all_df.pivot_table(index=date_col, columns=id_col, values=value_col)
    return wide

# ===== 主程式 =====
if __name__ == "__main__":
    print("=" * 50)
    print("  FinMind 台股資料下載器（斷點續傳版）")
    print(f"  期間：{START_DATE} ~ {END_DATE}")
    print(f"  限速：每 {SLEEP_SEC} 秒 1 次（確保不超每小時 600 次）")
    print("=" * 50)
    
    # Step 1: 取得股票清單
    print("\n📋 取得股票清單...")
    stock_df = get_stock_list()
    stock_ids = stock_df["stock_id"].tolist()
    print(f"  共 {len(stock_ids)} 支股票")
    
    # Step 2: 下載股價
    fetch_dataset(stock_ids, "股價(TaiwanStockPrice)", "TaiwanStockPrice", "price")
    
    # Step 3: 下載月營收
    fetch_dataset(stock_ids, "月營收(TaiwanStockMonthRevenue)", "TaiwanStockMonthRevenue", "revenue")
    
    # Step 4: 下載三大法人
    fetch_dataset(stock_ids, "法人買賣超(TaiwanStockInstitutionalInvestorsBuySell)",
                  "TaiwanStockInstitutionalInvestorsBuySell", "institution")
    
    # Step 5: 合併成 Wide-form
    print("\n🔧 合併成  Wide-form...")
    close_wide = load_and_pivot("price", "close")
    if close_wide is not None:
        close_wide.to_pickle(os.path.join(CACHE_DIR, "close_wide.pkl"))
        print(f"  close_wide: {close_wide.shape} → 已儲存至 {CACHE_DIR}/close_wide.pkl")
    
    print("\n✅ 全部完成！快取放在 finmind_cache/ 目錄。")

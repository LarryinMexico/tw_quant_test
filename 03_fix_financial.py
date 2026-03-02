# 03_fix_financial.py
import os, requests, time
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv("FINMIND_API_TOKEN")

CACHE_DIR = "finmind_cache"
os.makedirs(os.path.join(CACHE_DIR, "financial"), exist_ok=True)

# correct api dataset name
DATASET_NAME = "TaiwanStockFinancialStatements"

stock_df = pd.read_pickle("finmind_cache/stock_list.pkl")
stock_ids = [sid for sid in stock_df["stock_id"].unique() if str(sid).isdigit() and len(str(sid)) == 4]
print(f"Total stocks to download: {len(stock_ids)}")

for sid in tqdm(stock_ids, desc="Downloading FinancialStatements"):
    path = os.path.join(CACHE_DIR, "financial", f"{sid}.pkl")
    if os.path.exists(path):
        continue
        
    try:
        resp = requests.get("https://api.finmindtrade.com/api/v4/data", params={
            "dataset": DATASET_NAME,
            "data_id": sid,
            "start_date": "2019-01-01",
            "end_date": "2024-12-31",
            "token": TOKEN
        }, timeout=15).json()
        
        if resp.get("status") == 200:
            df = pd.DataFrame(resp.get("data", []))
            df.to_pickle(path)
        else:
            if "rate limit" in resp.get("msg", "").lower():
                print("Rate limit! Sleeping 60s")
                time.sleep(60)
    except Exception as e:
        pass
    
    time.sleep(6.2) # stay under 600 req/hr

print("Download complete. To merge, run python3 02_fetch_fundamental_data.py to process wide tables again.")

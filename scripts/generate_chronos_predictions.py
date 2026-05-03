from __future__ import annotations

import sys
from pathlib import Path
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.lora_experiment.chronos_lora_v2 import ChronosLoRARankerV2
from models.ramt.dataset import ALL_FEATURE_COLS, build_ticker_universe

"""
User Story: As a researcher, I need a standalone script to generate Chronos-T5 predictions 
for all tickers in the test set so that the dashboard and backtester can access them.
"""

DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
SEQ_LEN = 30
TEST_START = "2012-01-01"
TEST_END = "2015-12-31"

def generate_predictions(start_date: str = TEST_START, end_date: str = TEST_END, suffix: str = "2012_2015"):
    print("🚀 Generating Chronos-T5 LoRA V2 Predictions...")
    
    # Load Model
    model_path = ROOT / "models" / "lora_experiment" / "chronos_v2_adapter.pt"
    model = ChronosLoRARankerV2(input_dim=len(ALL_FEATURE_COLS))
    checkpoint = torch.load(model_path, map_location="cpu")
    model.encoder.load_state_dict(checkpoint['encoder_lora_state_dict'])
    model.input_projection.load_state_dict(checkpoint['input_projection_state_dict'])
    model.ranking_head.load_state_dict(checkpoint['ranking_head_state_dict'])
    model.to(DEVICE)
    model.eval()
    
    processed_dir = ROOT / "data" / "processed"
    tickers = build_ticker_universe(str(processed_dir))
    
    all_preds = []
    
    for ticker in tqdm(tickers, desc="Tickers"):
        p = processed_dir / f"{ticker}_features.parquet"
        if not p.exists(): continue
        
        df = pd.read_parquet(p)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date")
        
        # Filter for the specific test window
        test_df = df[(df["Date"] >= pd.to_datetime(start_date)) & (df["Date"] <= pd.to_datetime(end_date))].copy()
        if test_df.empty: continue
        
        # Get start index in original df
        start_idx = df.index.get_loc(test_df.index[0])
        
        feats = df[list(ALL_FEATURE_COLS)].to_numpy(dtype=np.float32)
        
        for i in range(start_idx, len(df)):
            if i < SEQ_LEN: continue
            
            # Stop if we exceed end_date
            if df.iloc[i]["Date"] > pd.to_datetime(end_date):
                break

            x = torch.tensor(feats[i-SEQ_LEN:i]).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                pred = model(x).item()
                
            all_preds.append({
                "Date": df.iloc[i]["Date"],
                "Ticker": ticker,
                "predicted": pred,
                "actual": df.iloc[i]["Sector_Alpha"] if "Sector_Alpha" in df.columns else np.nan
            })
            
    out_df = pd.DataFrame(all_preds)
    out_path = ROOT / "results" / "lora" / f"lora_v2_predictions_{suffix}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"✅ Predictions saved to {out_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=str, default="2012-01-01")
    parser.add_argument("--end", type=str, default="2015-12-31")
    parser.add_argument("--suffix", type=str, default="2012_2015")
    args = parser.parse_args()
    
    generate_predictions(start_date=args.start, end_date=args.end, suffix=args.suffix)

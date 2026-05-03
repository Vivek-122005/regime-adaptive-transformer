"""
Train HMM model on 2011 NIFTY data for blind 2012-2015 backtest.
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

# Add project root to sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_nifty_2011_data():
    """Load NIFTY index data for 2011."""
    nifty_path = ROOT / "data" / "raw" / "historical_2012" / "_NSEI.parquet"
    
    if not nifty_path.exists():
        raise FileNotFoundError(f"NIFTY data not found at {nifty_path}")
    
    df = pd.read_parquet(nifty_path)
    df["Date"] = pd.to_datetime(df["Date"])
    
    # Filter to 2011 data only
    df_2011 = df[(df["Date"] >= "2011-01-01") & (df["Date"] <= "2011-12-31")].copy()
    
    print(f"Loaded NIFTY data: {len(df)} total rows, {len(df_2011)} rows in 2011")
    print(f"Date range: {df_2011['Date'].min()} -> {df_2011['Date'].max()}")
    
    return df_2011


def compute_features(df):
    """Compute returns and volatility for HMM training."""
    # Calculate daily returns using Adj Close
    df = df.sort_values("Date").copy()
    df["Ret_1d"] = np.log(df["Adj Close"] / df["Adj Close"].shift(1))
    
    # Calculate 20-day realized volatility
    df["Realized_Vol_20"] = df["Ret_1d"].rolling(20).std()
    
    # Drop rows with NaN values
    df_clean = df.dropna(subset=["Ret_1d", "Realized_Vol_20"])
    
    print(f"After feature computation: {len(df_clean)} valid rows")
    return df_clean


def train_hmm(returns, volatility, n_components=3):
    """Train Gaussian HMM on returns and volatility."""
    # Prepare features
    features = np.column_stack([returns, volatility])
    
    # Standardize features
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std = np.where(std == 0, 1, std)  # Avoid division by zero
    features_std = (features - mean) / std
    
    print(f"Training HMM on {len(features_std)} samples")
    print(f"Features shape: {features_std.shape}")
    
    # Create and train HMM
    hmm = GaussianHMM(
        n_components=n_components,
        covariance_type="diag",
        n_iter=300,
        tol=1e-4,
        random_state=42,
        verbose=False
    )
    
    hmm.fit(features_std)
    
    # Get regime predictions
    regimes = hmm.predict(features_std)
    
    # Map to semantic regimes (bull=1, bear=2, high_vol=0)
    regime_means = {}
    for i in range(n_components):
        mask = regimes == i
        if np.any(mask):
            regime_means[i] = float(np.mean(returns[mask]))
    
    # Sort by mean return to identify bull/bear
    sorted_regimes = sorted(regime_means.items(), key=lambda x: x[1])
    
    semantic_mapping = {}
    if len(sorted_regimes) == 3:
        # Lowest mean = bear (2), highest = bull (1), middle = high_vol (0)
        semantic_mapping[sorted_regimes[0][0]] = 2  # Bear
        semantic_mapping[sorted_regimes[1][0]] = 0  # High vol
        semantic_mapping[sorted_regimes[2][0]] = 1  # Bull
    elif len(sorted_regimes) == 2:
        semantic_mapping[sorted_regimes[0][0]] = 2  # Bear
        semantic_mapping[sorted_regimes[1][0]] = 1  # Bull
    else:
        semantic_mapping[sorted_regimes[0][0]] = 0  # Default to high vol
    
    # Convert to semantic regimes
    semantic_regimes = np.array([semantic_mapping[r] for r in regimes])
    
    print("HMM training completed")
    print("Regime mapping:", semantic_mapping)
    print("Regime distribution:", {i: int(np.sum(semantic_regimes == i)) for i in range(3)})
    
    return hmm, semantic_mapping, (mean, std)


def main():
    """Train HMM on 2011 NIFTY data and save model."""
    print("Training HMM model for 2012-2015 blind backtest")
    print("=" * 60)
    
    # Load and prepare data
    df = load_nifty_2011_data()
    df_features = compute_features(df)
    
    if len(df_features) < 60:
        raise ValueError("Insufficient data for HMM training (need at least 60 rows)")
    
    # Train HMM
    returns = df_features["Ret_1d"].values
    volatility = df_features["Realized_Vol_20"].values
    
    hmm, semantic_mapping, (feature_mean, feature_std) = train_hmm(returns, volatility)
    
    # Create output directory
    output_dir = ROOT / "results" / "historical_2012"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model artifacts
    model_data = {
        "hmm": hmm,
        "semantic_mapping": semantic_mapping,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "training_period": "2011-01-01 to 2011-12-31",
        "n_components": 3,
        "n_training_samples": len(df_features)
    }
    
    model_path = output_dir / "hmm_pre2012.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)
    
    print(f"\nHMM model saved to {model_path}")
    print(f"Model contains: {list(model_data.keys())}")
    
    # Save some basic stats
    stats = {
        "training_samples": len(df_features),
        "training_start": str(df_features["Date"].min().date()),
        "training_end": str(df_features["Date"].max().date()),
        "regime_mapping": semantic_mapping,
        "mean_return": float(np.mean(returns)),
        "mean_volatility": float(np.mean(volatility))
    }
    
    stats_path = output_dir / "hmm_training_stats_2011.json"
    import json
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    
    print(f"Training stats saved to {stats_path}")


if __name__ == "__main__":
    main()

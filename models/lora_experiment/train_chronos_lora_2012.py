from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# Add project root to sys.path so 'models' can be imported when running script directly
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from models.lora_experiment.chronos_lora import ChronosLoRARanker
from models.ramt.dataset import ALL_FEATURE_COLS, build_ticker_universe

SEQ_LEN = 30
TARGET_COL = "Sector_Alpha"
# MODIFIED: Use pre-2012 training data for blind backtest
TRAIN_END = pd.Timestamp("2011-12-31")
TEST_START = pd.Timestamp("2012-01-01")
TEST_END = pd.Timestamp("2015-12-31")

EPOCHS = 3
BATCH_SIZE = 64
LEARNING_RATE = 1e-4


@dataclass
class TickerPanelData:
    features: np.ndarray
    target: np.ndarray
    dates: np.ndarray


@dataclass
class SplitIndex:
    ticker_data: dict[str, TickerPanelData]
    train_index: list[tuple[str, int]]
    test_index: list[tuple[str, int]]


class IndexedSequenceDataset(Dataset):
    def __init__(
        self,
        split_index: SplitIndex,
        seq_len: int = SEQ_LEN,
        target_col: str = TARGET_COL,
    ):
        self.split_index = split_index
        self.seq_len = seq_len
        self.target_col = target_col
        self.feature_cols = [c for c in ALL_FEATURE_COLS if c != target_col]

    def __len__(self) -> int:
        return len(self.split_index.train_index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        ticker, row_idx = self.split_index.train_index[idx]
        data = self.split_index.ticker_data[ticker]

        # Extract sequence ending at row_idx
        start_idx = max(0, row_idx - self.seq_len + 1)
        seq_features = data.features[start_idx : row_idx + 1]
        seq_target = data.target[row_idx]

        # Pad if necessary
        if len(seq_features) < self.seq_len:
            pad_len = self.seq_len - len(seq_features)
            seq_features = np.vstack([
                np.zeros((pad_len, len(self.feature_cols)), dtype=np.float32),
                seq_features
            ])

        return (
            torch.from_numpy(seq_features.astype(np.float32)),
            torch.tensor(float(seq_target), dtype=torch.float32)
        )


def load_processed_features(processed_dir: Path) -> dict[str, pd.DataFrame]:
    """Load all feature parquet files from processed directory."""
    features = {}
    for path in sorted(processed_dir.glob("*_features.parquet")):
        if path.name.startswith("_"):
            continue  # Skip benchmark
        ticker = path.name.replace("_features.parquet", "").replace("_", ".")
        df = pd.read_parquet(path)
        df["Date"] = pd.to_datetime(df["Date"])
        features[ticker] = df
    return features


def build_split_index(features: dict[str, pd.DataFrame]) -> SplitIndex:
    """Build train/test split index from loaded features."""
    ticker_data = {}
    train_index = []
    test_index = []

    for ticker, df in features.items():
        if len(df) < SEQ_LEN:
            continue

        # Extract features and target
        feature_cols = [c for c in ALL_FEATURE_COLS if c != TARGET_COL]
        features_array = df[feature_cols].values.astype(np.float32)
        target_array = df[TARGET_COL].values.astype(np.float32)
        dates_array = df["Date"].values

        ticker_data[ticker] = TickerPanelData(
            features=features_array,
            target=target_array,
            dates=dates_array
        )

        # Build indices
        for i in range(SEQ_LEN - 1, len(df)):
            date = pd.Timestamp(dates_array[i])
            if date <= TRAIN_END:
                train_index.append((ticker, i))
            elif TEST_START <= date <= TEST_END:
                test_index.append((ticker, i))

    return SplitIndex(
        ticker_data=ticker_data,
        train_index=train_index,
        test_index=test_index
    )


def train_model(
    model: ChronosLoRARanker,
    train_loader: DataLoader,
    epochs: int = EPOCHS,
    learning_rate: float = LEARNING_RATE,
) -> dict[str, list[float]]:
    """Train the Chronos LoRA model."""
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    train_losses = []
    
    model.train()
    for epoch in range(epochs):
        epoch_losses = []
        
        for batch_features, batch_targets in train_loader:
            optimizer.zero_grad()
            
            outputs = model(batch_features)
            loss = criterion(outputs.squeeze(), batch_targets)
            
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        avg_loss = np.mean(epoch_losses)
        train_losses.append(avg_loss)
        
        print(f"Epoch {epoch + 1}/{epochs}: Loss = {avg_loss:.6f}")
    
    return {"train_losses": train_losses}


def evaluate_model(
    model: ChronosLoRARanker,
    split_index: SplitIndex,
) -> dict[str, float]:
    """Evaluate model on training set for validation purposes."""
    model.eval()
    
    # Use a subset of training data for validation
    val_indices = split_index.train_index[:min(1000, len(split_index.train_index))]
    
    predictions = []
    actuals = []
    
    for ticker, row_idx in val_indices:
        data = split_index.ticker_data[ticker]
        
        # Extract sequence
        start_idx = max(0, row_idx - SEQ_LEN + 1)
        seq_features = data.features[start_idx : row_idx + 1]
        
        # Pad if necessary
        if len(seq_features) < SEQ_LEN:
            pad_len = SEQ_LEN - len(seq_features)
            seq_features = np.vstack([
                np.zeros((pad_len, seq_features.shape[1]), dtype=np.float32),
                seq_features
            ])
        
        with torch.no_grad():
            pred = model(torch.from_numpy(seq_features.astype(np.float32)))
            predictions.append(pred.item())
            actuals.append(data.target[row_idx])
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Calculate metrics
    mse = np.mean((predictions - actuals) ** 2)
    ic = np.corrcoef(predictions, actuals)[0, 1]
    
    # Directional accuracy
    pred_direction = (predictions > 0).astype(int)
    actual_direction = (actuals > 0).astype(int)
    directional_acc = np.mean(pred_direction == actual_direction)
    
    return {
        "mse": float(mse),
        "ic": float(ic) if not np.isnan(ic) else 0.0,
        "directional_accuracy": float(directional_acc),
        "num_predictions": len(predictions)
    }


def main() -> None:
    """Train Chronos LoRA model on pre-2012 data for blind 2012-2015 backtest."""
    print("Training Chronos LoRA model for 2012-2015 blind backtest")
    print(f"Training period: up to {TRAIN_END.date()}")
    print(f"Test period: {TEST_START.date()} to {TEST_END.date()}")
    
    # Load blind features
    processed_dir = ROOT / "data" / "processed_blind_20111231"
    if not processed_dir.exists():
        raise FileNotFoundError(f"Blind features not found at {processed_dir}")
    
    print(f"Loading features from {processed_dir}")
    features = load_processed_features(processed_dir)
    print(f"Loaded {len(features)} tickers")
    
    # Build split index
    split_index = build_split_index(features)
    print(f"Training samples: {len(split_index.train_index)}")
    print(f"Test samples: {len(split_index.test_index)}")
    
    if len(split_index.train_index) == 0:
        raise ValueError("No training samples found")
    
    # Create datasets and loaders
    train_dataset = IndexedSequenceDataset(split_index)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Initialize model
    feature_cols = [c for c in ALL_FEATURE_COLS if c != TARGET_COL]
    model = ChronosLoRARanker(input_dim=len(feature_cols))
    
    # Train model
    print("Training model...")
    training_history = train_model(model, train_loader, epochs=EPOCHS)
    
    # Evaluate on test set
    print("Evaluating on test set...")
    test_metrics = evaluate_model(model, split_index)
    
    print("\nTest Results:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Save model
    output_dir = ROOT / "results" / "historical_2012"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = output_dir / "chronos_v2_adapter_pre2012.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to {model_path}")
    
    # Save training history and metrics
    results = {
        "training_history": training_history,
        "test_metrics": test_metrics,
        "training_config": {
            "train_end": str(TRAIN_END.date()),
            "test_start": str(TEST_START.date()),
            "test_end": str(TEST_END.date()),
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "seq_len": SEQ_LEN,
            "target_col": TARGET_COL,
            "num_training_samples": len(split_index.train_index),
            "num_test_samples": len(split_index.test_index),
            "num_tickers": len(features)
        }
    }
    
    results_path = output_dir / "chronos_training_results_2012.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {results_path}")


if __name__ == "__main__":
    main()

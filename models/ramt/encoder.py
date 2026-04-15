import torch
import torch.nn as nn

from models.ramt.dataset import (
    ALL_FEATURE_COLS,
    CROSS_ASSET_COLS,
    MOMENTUM_COLS,
    PRICE_COLS,
    REGIME_COLS,
    TECH_COLS,
    TICKER_LIST,
    VOL_COLS,
    VOLUME_COLS,
)


class FeedForwardEncoder(nn.Module):
    """
    Generic feedforward encoder for continuous feature groups.
    Projects input_dim features to embed_dim.

    Used for: price, volatility, technical, momentum,
              volume, cross-asset feature groups.

    Architecture:
        Linear(input_dim → embed_dim)
        LayerNorm(embed_dim)
        ReLU
        Dropout(dropout)
        Linear(embed_dim → embed_dim)
        LayerNorm(embed_dim)

    Input:  (batch, seq_len, input_dim)
    Output: (batch, seq_len, embed_dim)
    """

    def __init__(self, input_dim, embed_dim=32, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        return self.net(x)
        # output: (batch, seq_len, embed_dim)


class RegimeEncoder(nn.Module):
    """
    Categorical encoder for HMM_Regime feature.
    Uses nn.Embedding because regime is categorical (0/1/2)
    not continuous. Each regime gets its own learned vector.

    Why Embedding and not Linear:
        Regime integers 0, 1, 2 are categories not quantities.
        Treating them as numbers implies regime 2 = twice regime 1.
        This is wrong — they are distinct market states.
        Embedding gives each regime its own learned representation.

    Input:  (batch, seq_len) — integer tensor of regime labels
    Output: (batch, seq_len, embed_dim)
    """

    def __init__(self, num_regimes=3, embed_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(num_regimes, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: (batch, seq_len) integers
        # squeeze if shape is (batch, seq_len, 1)
        if x.dim() == 3:
            x = x.squeeze(-1)
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)
        return self.norm(embedded)


class TickerEncoder(nn.Module):
    """
    Learns unique embedding per stock.

    Why needed:
    Model trains on 50 stocks simultaneously.
    It needs to know WHICH stock it is predicting.
    TCS behaves differently from RELIANCE.

    Each stock gets a learned 32-dim vector.
    Model learns:
      TCS = IT, USD-sensitive, large-cap growth
      RELIANCE = energy, crude-sensitive, conglomerate

    This is exactly like word embeddings in NLP.
    Each stock is a "word" with its own meaning.
    """

    def __init__(self, num_tickers=50, embed_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(num_tickers, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, ticker_ids):
        # ticker_ids: (batch,) integer tensor
        if ticker_ids.dim() == 2 and ticker_ids.shape[-1] == 1:
            ticker_ids = ticker_ids.squeeze(-1)
        embedded = self.embedding(ticker_ids)
        return self.norm(embedded)


class MultimodalEncoder(nn.Module):
    """
    Combines all feature group encoders into one module.

    Feature groups and their encoders:
        Price (6 features)      → FeedForwardEncoder
        Volatility (5 features) → FeedForwardEncoder
        Technical (8 features)  → FeedForwardEncoder
        Momentum (4 features)   → FeedForwardEncoder
        Volume (2 features)     → FeedForwardEncoder
        Regime (1 feature)      → RegimeEncoder (categorical)
        CrossAsset (1 feature)  → FeedForwardEncoder

    Fusion:
        Concatenate all 7 embeddings: (batch, seq_len, 7×32=224)
        Project to embed_dim: Linear(224 → embed_dim)
        LayerNorm + Dropout

    Input:  x (batch, seq_len, 27) — full feature tensor
            (HMM_Regime column is recovered as integer for embedding)

    Output: (batch, seq_len, embed_dim)
    """

    def __init__(self, embed_dim=64, group_dim=32, dropout=0.1):
        super().__init__()

        self.embed_dim = embed_dim
        self.group_dim = group_dim

        # One encoder per feature group
        self.price_encoder = FeedForwardEncoder(
            len(PRICE_COLS), group_dim, dropout
        )
        self.vol_encoder = FeedForwardEncoder(len(VOL_COLS), group_dim, dropout)
        self.tech_encoder = FeedForwardEncoder(len(TECH_COLS), group_dim, dropout)
        self.momentum_encoder = FeedForwardEncoder(
            len(MOMENTUM_COLS), group_dim, dropout
        )
        self.volume_encoder = FeedForwardEncoder(
            len(VOLUME_COLS), group_dim, dropout
        )
        self.regime_encoder = RegimeEncoder(num_regimes=3, embed_dim=group_dim)
        self.cross_asset_encoder = FeedForwardEncoder(
            len(CROSS_ASSET_COLS), group_dim, dropout
        )
        self.ticker_encoder = TickerEncoder(
            num_tickers=max(1, len(TICKER_LIST)), embed_dim=group_dim
        )

        # Compute column indices for each group
        # These must match ALL_FEATURE_COLS order exactly
        self._build_indices()

        # Fusion layer: 7 groups (+ optional ticker) × group_dim → embed_dim
        fusion_input_dim = 8 * group_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
        )

    def _build_indices(self):
        """
        Build column index slices for each feature group.
        Based on ALL_FEATURE_COLS order from dataset.py.
        """
        all_cols = ALL_FEATURE_COLS

        def get_indices(cols):
            return [all_cols.index(c) for c in cols]

        self.price_idx = get_indices(PRICE_COLS)
        self.vol_idx = get_indices(VOL_COLS)
        self.tech_idx = get_indices(TECH_COLS)
        self.momentum_idx = get_indices(MOMENTUM_COLS)
        self.volume_idx = get_indices(VOLUME_COLS)
        self.regime_idx = get_indices(REGIME_COLS)
        self.cross_asset_idx = get_indices(CROSS_ASSET_COLS)

    def forward(self, x, ticker_id=None):
        """
        Args:
            x: (batch, seq_len, num_features) — full scaled feature tensor
               Note: HMM_Regime column is float after scaling
               We convert it back to integer for embedding lookup

        Returns:
            fused: (batch, seq_len, embed_dim)
        """
        # Extract each feature group
        price_x = x[:, :, self.price_idx]
        vol_x = x[:, :, self.vol_idx]
        tech_x = x[:, :, self.tech_idx]
        momentum_x = x[:, :, self.momentum_idx]
        volume_x = x[:, :, self.volume_idx]
        regime_x = x[:, :, self.regime_idx]
        cross_x = x[:, :, self.cross_asset_idx]

        # Encode each group
        price_emb = self.price_encoder(price_x)
        vol_emb = self.vol_encoder(vol_x)
        tech_emb = self.tech_encoder(tech_x)
        momentum_emb = self.momentum_encoder(momentum_x)
        volume_emb = self.volume_encoder(volume_x)
        cross_emb = self.cross_asset_encoder(cross_x)

        # Regime needs integer input for embedding
        # After StandardScaler, HMM_Regime is float
        # Recover original integer by rounding and clamping
        regime_int = regime_x.squeeze(-1).round().long()
        regime_int = regime_int.clamp(0, 2)
        regime_emb = self.regime_encoder(regime_int)

        embeddings = [
            price_emb,
            vol_emb,
            tech_emb,
            momentum_emb,
            volume_emb,
            regime_emb,
            cross_emb,
        ]

        # Add ticker embedding if provided
        if ticker_id is not None and hasattr(self, "ticker_encoder"):
            ticker_emb = self.ticker_encoder(ticker_id)
            ticker_emb = ticker_emb.unsqueeze(1).expand(-1, x.shape[1], -1)
            embeddings.append(ticker_emb)
        else:
            # If not provided, append zeros to keep fusion input dimension stable
            embeddings.append(torch.zeros_like(price_emb))

        fused = torch.cat(embeddings, dim=-1)

        # Project to embed_dim
        return self.fusion(fused)
        # output: (batch, seq_len, embed_dim=64)


if __name__ == "__main__":
    print("Testing MultimodalEncoder...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create encoder
    encoder = MultimodalEncoder(embed_dim=64, group_dim=32, dropout=0.1).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"Encoder parameters: {total_params:,}")

    # Test forward pass
    batch_size = 8
    seq_len = 30
    num_features = len(ALL_FEATURE_COLS)

    # Simulate scaled input (StandardScaler output)
    x = torch.randn(batch_size, seq_len, num_features).to(device)

    # Forward pass
    ticker_id = torch.zeros(batch_size, dtype=torch.long).to(device)
    out = encoder(x, ticker_id=ticker_id)

    print(f"\nInput shape:  {x.shape}")
    print(f"Output shape: {out.shape}")

    # Verify output shape
    assert out.shape == (batch_size, seq_len, 64), (
        f"Expected (8, 30, 64) got {out.shape}"
    )

    # Check no NaNs
    assert not torch.isnan(out).any(), "NaN in output!"

    # Check output range (should be reasonable after LayerNorm)
    print(f"Output min: {out.min().item():.4f}")
    print(f"Output max: {out.max().item():.4f}")
    print(f"Output mean: {out.mean().item():.4f}")

    # Test with real data from DataModule
    print("\nTesting with real data...")
    from models.ramt.dataset import RAMTDataModule

    dm = RAMTDataModule("TCS_NS", seq_len=30, batch_size=8)
    folds = dm.get_walk_forward_indices()
    train_idx, test_idx = folds[0]
    train_loader, val_loader, test_loader, dates = dm.get_fold_loaders(
        train_idx, test_idx
    )

    batch = next(iter(train_loader))
    X_batch, y_batch, r_batch, t_batch = batch
    X_batch = X_batch.to(device)

    out_real = encoder(X_batch, ticker_id=t_batch.to(device).squeeze(-1))
    print(f"Real data input shape:  {X_batch.shape}")
    print(f"Real data output shape: {out_real.shape}")
    assert not torch.isnan(out_real).any(), "NaN in real data output!"

    print("\nAll encoder checks passed.")

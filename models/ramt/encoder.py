import torch
import torch.nn as nn

from models.ramt.dataset import (
    ALL_FEATURE_COLS,
    CROSS_ASSET_COLS,
    MOMENTUM_COLS,
    PRICE_COLS,
    REGIME_COLS,
    TECH_COLS,
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

        # Compute column indices for each group
        # These must match ALL_FEATURE_COLS order exactly
        self._build_indices()

        # Fusion layer: 7 groups × group_dim → embed_dim
        fusion_input_dim = 7 * group_dim
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

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, 27) — full scaled feature tensor
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

        # Concatenate all embeddings
        # Each: (batch, seq_len, group_dim=32)
        fused = torch.cat(
            [
                price_emb,
                vol_emb,
                tech_emb,
                momentum_emb,
                volume_emb,
                regime_emb,
                cross_emb,
            ],
            dim=-1,
        )
        # fused: (batch, seq_len, 7×32=224)

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
    num_features = 27

    # Simulate scaled input (StandardScaler output)
    x = torch.randn(batch_size, seq_len, num_features).to(device)

    # Forward pass
    out = encoder(x)

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

    dm = RAMTDataModule("JPM", seq_len=30, batch_size=8)
    folds = dm.get_walk_forward_indices()
    train_idx, test_idx = folds[0]
    train_loader, val_loader, test_loader, dates = dm.get_fold_loaders(
        train_idx, test_idx
    )

    X_batch, y_batch, r_batch = next(iter(train_loader))
    X_batch = X_batch.to(device)

    out_real = encoder(X_batch)
    print(f"Real data input shape:  {X_batch.shape}")
    print(f"Real data output shape: {out_real.shape}")
    assert not torch.isnan(out_real).any(), "NaN in real data output!"

    print("\nAll encoder checks passed.")

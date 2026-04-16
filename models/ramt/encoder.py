import torch
import torch.nn as nn

from models.ramt.dataset import (
    ALL_FEATURE_COLS,
    MACRO_COLS,
    PRICE_COLS,
    TECH_COLS,
    TICKER_LIST,
    VOLUME_COLS,
)


class FeedForwardEncoder(nn.Module):
    """Projects input_dim features per timestep to embed_dim."""

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
        return self.net(x)


class RegimeEncoder(nn.Module):
    """Categorical embedding for HMM regime 0/1/2 (passed separately from feature matrix)."""

    def __init__(self, num_regimes=3, embed_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(num_regimes, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        if x.dim() == 3:
            x = x.squeeze(-1)
        embedded = self.embedding(x)
        return self.norm(embedded)


class TickerEncoder(nn.Module):
    def __init__(self, num_tickers=50, embed_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(num_tickers, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, ticker_ids):
        if ticker_ids.dim() == 2 and ticker_ids.shape[-1] == 1:
            ticker_ids = ticker_ids.squeeze(-1)
        embedded = self.embedding(ticker_ids)
        return self.norm(embedded)


class MultimodalEncoder(nn.Module):
    """
    Encodes lean Parquet features (no HMM inside X).

    Groups:
      - Price: Ret_1d, Ret_5d, Ret_21d
      - Technical: RSI_14, BB_Dist
      - Volume: Volume_Surge
      - Macro: 4 lagged macro returns
    Regime is provided as integer labels (batch,) and expanded over seq_len.
    """

    def __init__(self, embed_dim=64, group_dim=32, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.group_dim = group_dim

        self.price_encoder = FeedForwardEncoder(len(PRICE_COLS), group_dim, dropout)
        self.tech_encoder = FeedForwardEncoder(len(TECH_COLS), group_dim, dropout)
        self.volume_encoder = FeedForwardEncoder(len(VOLUME_COLS), group_dim, dropout)
        self.macro_encoder = FeedForwardEncoder(len(MACRO_COLS), group_dim, dropout)
        self.regime_encoder = RegimeEncoder(num_regimes=3, embed_dim=group_dim)
        self.ticker_encoder = TickerEncoder(
            num_tickers=max(1, len(TICKER_LIST)), embed_dim=group_dim
        )

        self._build_indices()

        # 5 groups (price, tech, vol, macro, regime) + ticker = 6 * group_dim
        fusion_input_dim = 6 * group_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
        )

    def _build_indices(self):
        all_cols = ALL_FEATURE_COLS

        def get_indices(cols):
            return [all_cols.index(c) for c in cols]

        self.price_idx = get_indices(PRICE_COLS)
        self.tech_idx = get_indices(TECH_COLS)
        self.volume_idx = get_indices(VOLUME_COLS)
        self.macro_idx = get_indices(MACRO_COLS)

    def forward(self, x, regime, ticker_id=None):
        """
        Args:
            x: (batch, seq_len, len(ALL_FEATURE_COLS)) — scaled features only
            regime: (batch,) int64 labels 0/1/2 (same regime broadcast across time)
        """
        price_x = x[:, :, self.price_idx]
        tech_x = x[:, :, self.tech_idx]
        volume_x = x[:, :, self.volume_idx]
        macro_x = x[:, :, self.macro_idx]

        price_emb = self.price_encoder(price_x)
        tech_emb = self.tech_encoder(tech_x)
        volume_emb = self.volume_encoder(volume_x)
        macro_emb = self.macro_encoder(macro_x)

        regime_seq = regime.long().unsqueeze(1).expand(-1, x.shape[1])
        regime_emb = self.regime_encoder(regime_seq)

        embeddings = [price_emb, tech_emb, volume_emb, macro_emb, regime_emb]

        if ticker_id is not None:
            ticker_emb = self.ticker_encoder(ticker_id)
            ticker_emb = ticker_emb.unsqueeze(1).expand(-1, x.shape[1], -1)
            embeddings.append(ticker_emb)
        else:
            embeddings.append(torch.zeros_like(price_emb))

        fused = torch.cat(embeddings, dim=-1)
        return self.fusion(fused)


if __name__ == "__main__":
    print("Testing MultimodalEncoder...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = MultimodalEncoder(embed_dim=64, group_dim=32, dropout=0.1).to(device)
    batch_size = 8
    seq_len = 30
    nf = len(ALL_FEATURE_COLS)
    x = torch.randn(batch_size, seq_len, nf).to(device)
    regime = torch.randint(0, 3, (batch_size,), device=device)
    ticker_id = torch.zeros(batch_size, dtype=torch.long, device=device)
    out = encoder(x, regime, ticker_id=ticker_id)
    assert out.shape == (batch_size, seq_len, 64)
    print("OK", out.shape)

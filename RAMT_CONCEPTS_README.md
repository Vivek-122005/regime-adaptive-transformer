# RAMT — concepts & notes (living document)

This file collects explanations for ideas, code, and methods used in this repo.  
Ask in chat to add a new topic; it gets appended here in plain language.

---

## Why we use encoders (`MultimodalEncoder`)

Encoders exist so downstream parts of the model (fusion, attention, prediction head) receive **consistent, learned representations** instead of raw feature columns in one flat vector.

1. **One space for many input types**  
   The 27 features are grouped (returns, volatility, technicals, momentum, volume, regime, cross-asset). Scales and meaning differ. Each group encoder maps its slice to the **same width** (`group_dim`, e.g. 32) so we can **concatenate** and **fuse** with one linear layer into `embed_dim` for the transformer.

2. **Nonlinear preprocessing per group**  
   A small feedforward stack lets the model learn **group-specific** transforms (e.g. how volatility features interact within the vol group) before mixing groups. Global nonlinearity-only-later can be harder to train on heterogeneous inputs.

3. **Regime is categorical**  
   `HMM_Regime` is **categorical** (states 0, 1, 2), not a continuous magnitude. **`nn.Embedding`** gives each state its own learned vector instead of treating “2” as twice “1”. Continuous groups use linear layers on real values; regime uses embedding after integer labels are recovered from the scaled column.

4. **Fixed width for the transformer**  
   Attention blocks expect a fixed **token size** (`embed_dim`). Encoders adapt `(batch, seq, 27)` → `(batch, seq, embed_dim)`.

**Short summary:** specialize per modality, align dimensions, treat regime as categories, then fuse.

---

## What the RAMT dataset module does (`dataset.py`)

The RAMT data path is built around **one ticker**, **processed CSV features**, **sequences for a transformer**, and **walk-forward evaluation** without scaler leakage.

### Column constants

`PRICE_COLS`, `VOL_COLS`, `TECH_COLS`, `MOMENTUM_COLS`, `VOLUME_COLS`, `REGIME_COLS`, `CROSS_ASSET_COLS` are concatenated into **`ALL_FEATURE_COLS`** (27 features in a fixed order). The encoder and any model code should use this order so column indices stay consistent.

### Loading (`RAMTDataModule._load_data`)

- Reads `data/processed/{ticker}_features.csv`.
- **Target:** next trading day’s return: `target = Log_Return.shift(-1)`, then rows with missing target are dropped.
- Checks that every column in `ALL_FEATURE_COLS` exists.
- Keeps **raw** feature matrix `features_raw`, **targets** `targets`, and per-row **integer** `HMM_Regime` in `regimes` (used as labels; the same regime column also appears inside scaled `X` for the multimodal encoder).

### Scaling (per fold, in `get_fold_loaders`)

- **Train** indices are split into **actual train** and **validation** (last `val_fraction` of the train window, at least one row).
- **`StandardScaler` is fit only on actual-train rows**, then applied to val and test. Val/test never influence mean/variance — **no leakage** from future or held-out segments into normalization.

### Sequences (`SequenceDataset`)

- For each valid time index `i ≥ seq_len`, one sample is:
  - **`X`:** `seq_len` consecutive rows of scaled features ending **before** the target day → shape `(seq_len, 27)`.
  - **`y`:** scalar next-day target at index `i`.
  - **`regime`:** `HMM_Regime` at that same index `i` (integer), returned as a length-1 tensor for batching.

So the model sees a **history window** and predicts **one step ahead**, with an explicit regime label aligned to the prediction time.

### Walk-forward folds (`get_walk_forward_indices`)

- Starts with an initial training fraction of the series (default 60%), then repeatedly extends the train end and takes a fixed-length **test** block (default 63 days), sliding forward until the end of the data.
- Each fold is `(train_idx, test_idx)`; you call **`get_fold_loaders(train_idx, test_idx)`** to get `train_loader`, `val_loader`, `test_loader`, plus **`test_dates`** for the test segment (aligned to sequence-valid test positions).

### What you get out

**PyTorch `DataLoader`s** yielding batches of `(X, y, regime)` for training RAMT with **expanding training history**, **held-out forward chunks**, and **honest scaling** per fold.

---

<!-- New sections go below this line -->

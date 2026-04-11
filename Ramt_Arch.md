# RAMT Architecture — Complete Documentation
## Why We Are Building What We Are Building

---

## The Big Picture

### What We Are Trying To Do
Predict whether a stock will go up or down tomorrow.
More specifically: predict the exact log return for tomorrow.

### Why This Is Hard
Three fundamental problems:

Problem 1 — Signal to Noise Ratio
Stock returns are mostly random. On any given day,
random factors (news, sentiment, big orders) dominate.
The actual predictable signal is tiny — maybe 1-2%
of the variance. Everything else is noise.

Problem 2 — Non-Stationarity
The patterns that worked in 2017 (calm bull market)
do not work in 2020 (COVID crisis). The statistical
properties of the data keep changing. This breaks
most standard ML models which assume stable patterns.

Problem 3 — Regime Shifts
Markets switch between fundamentally different states.
In a bull market, momentum strategies work.
In a bear market, momentum strategies fail badly.
A single model cannot handle both simultaneously.

### Our Solution — RAMT
Build a model that:
1. Detects which regime the market is currently in (HMM)
2. Uses specialized experts per regime (MoE)
3. Captures temporal patterns across 30 days (Transformer)
4. Combines multiple types of features (Multimodal)
5. Learns across US and Indian markets (Transfer Learning)

---

## Phase 1 — What We Already Built

### XGBoost Baseline
A gradient boosting model that uses 35 engineered features
to predict next-day returns. Walk-forward validated.

Result: DA% = 52.13%, Sharpe = 0.27

Key innovation: Regime-stratified training.
Three separate XGBoost models — one per HMM regime.
TCS Sharpe improved from 0.04 to 0.82.

### LSTM Baseline
A recurrent neural network that processes 30-day sequences.
Same walk-forward structure as XGBoost.

Result: DA% = 51.02%, Sharpe = 0.28

Finding: LSTM is WORSE than XGBoost on 4 of 5 tickers.
This is expected. LSTM has no regime awareness.
It tries to learn one set of rules for all market conditions.
This directly motivates RAMT.

### What Phase 1 Proved
Regime awareness matters more than model complexity.
A simple XGBoost with regime stratification
beats a more complex LSTM without regime awareness.

RAMT Phase 2 goal:
Keep the regime awareness + add Transformer complexity.
Best of both worlds.

---

## Phase 2 — What We Are Building

### The RAMT Pipeline
Raw Data (30 days × 27 features)
↓
Step 1: Split into feature groups
↓
Step 2: Encode each group separately (encoder.py)
↓
Step 3: Fuse all encodings (encoder.py)
↓
Step 4: Transformer temporal attention (model.py)
↓
Step 5: Regime-conditioned MoE routing (moe.py)
↓
Step 6: Final prediction (next-day return)

---

## Component 1 — Dataset Module
### File: models/ramt/dataset.py

### What It Does
Loads processed CSVs and creates PyTorch sequences.
Every sample = 30 consecutive trading days of features
+ the return on day 31 (target).

### Why We Need This
Raw data is a flat table — one row per day.
Neural networks need sequences — windows of days.
The dataset module handles this conversion.

### Why seq_len = 30
30 trading days = 6 weeks.
Our ACF/PACF analysis showed autocorrelation
dies out at approximately 20-30 lags.
30 days captures one monthly market cycle.
Long enough for momentum patterns.
Short enough to avoid stale information.

### The Three Outputs Per Sample

Output 1 — X (features):
Shape: (30, 27)
30 days of history, 27 features per day.
This is the model's "view of the past".

Output 2 — y (target):
Shape: (1,)
The log return on day 31.
This is what the model must predict.

Output 3 — regime (integer):
Shape: (1,)
The HMM regime at the last day (day 30).
Integer: 0=HighVol, 1=Bull, 2=Bear.
This tells the MoE which expert to activate.

### Why StandardScaler Here (Not Later)
Scaling must happen INSIDE each walk-forward fold.
Scaler is fit ONLY on training data.
Then applied to validation and test.

If we scaled the whole dataset first:
test data statistics would leak into training.
This is data leakage — results would be
artificially good and useless in real trading.

### Walk-Forward Index Generation
Same structure as XGBoost:
Initial train: 60% of data
Step: 63 days (1 quarter)
Test: 63 days per fold

This ensures fair comparison between
XGBoost, LSTM, and RAMT.

---

## Component 2 — Multimodal Encoders
### File: models/ramt/encoder.py

### What It Does
Takes the 27 features and encodes each group
into a unified 32-dimensional embedding.
Then fuses all 7 embeddings into one 64-dim vector.

### Why Separate Encoders Per Feature Group

Feature groups have completely different statistical properties:

Price features (Return_Lag_1...20):
  Values range from -0.15 to +0.15 (log returns)
  Approximately zero-mean
  Need to capture momentum and reversal patterns

Volatility features (Realized_Vol, Garman_Klass):
  Values range from 0.005 to 0.09
  Always positive
  Need to capture risk level and regime transitions

Technical features (RSI, MACD, Bollinger):
  RSI: 0 to 100
  MACD: small signed values
  BB_Position: 0 to 1
  Mixed scales — hard for one layer to handle

Regime feature (HMM_Regime):
  Values: 0, 1, or 2 (categorical integers)
  Cannot be treated as continuous numbers
  Needs an Embedding layer (like word embeddings in NLP)

If we fed all 27 features into one linear layer:
  The layer would struggle with mixed scales
  Regime integer 2 would be treated as "twice" regime 1
  (wrong — these are categories, not quantities)

Separate encoders solve this:
  Each group normalizes in its own space
  Regime gets proper categorical embedding
  All outputs are in the same 32-dim space
  Fusion layer combines them on equal footing

### The Encoder Architectures

FeedForwardEncoder (for continuous features):
Input (batch, seq_len, input_dim)
↓
Linear(input_dim → 32)
↓
LayerNorm
↓
ReLU activation
↓
Linear(32 → 32)
Output (batch, seq_len, 32)

Why LayerNorm:
  Normalizes across feature dimension
  Stabilizes training
  Standard in transformer architectures

Why ReLU:
  Adds nonlinearity
  Model can learn non-linear feature relationships

RegimeEncoder (for categorical HMM_Regime):
Input (batch, seq_len) — integers 0/1/2
↓
Embedding(num_regimes=3, embed_dim=32)
Output (batch, seq_len, 32)


Why Embedding:
  Regime 0, 1, 2 are categories not numbers
  Embedding learns a unique 32-dim vector per regime
  Like word embeddings in NLP — each word gets its own vector
  Bull regime gets its own learned representation
  Bear regime gets its own learned representation
  These representations are learned from data

MultimodalFusion:
7 embeddings each (batch, seq_len, 32)
↓
Concatenate along feature dim
↓
(batch, seq_len, 224)  [7 × 32 = 224]
↓
Linear(224 → 64)
↓
LayerNorm
Output (batch, seq_len, 64)

Why concatenate then project:
  Concatenation preserves all information from all groups
  Linear projection learns how to combine them optimally
  64-dim output is the "unified language" all components speak

---

## Component 3 — Transformer Backbone
### Part of: models/ramt/model.py

### What It Does
Takes the fused embedding sequence (batch, 30, 64)
and applies self-attention to find relationships
between different time steps.

### Why Transformer Instead of LSTM

LSTM processes sequentially:
  Day 1 → Day 2 → Day 3 → ... → Day 30
  Information from Day 1 must travel through 29 steps
  to influence Day 30's prediction.
  Long-range information gets diluted (vanishing gradient).

Transformer uses attention:
  Day 30 can directly attend to Day 1, Day 15, Day 28
  No sequential bottleneck
  "How relevant was Day 1 to today's prediction?"
  Learns these relevance weights from data.

### Self-Attention Formula
Attention(Q, K, V) = softmax(QK^T / sqrt(dk)) × V

Q = Query  — "What am I looking for?"
K = Keys   — "What does each day offer?"
V = Values — "What is the actual information?"

For each day (query), compute similarity with all days (keys).
Use similarities as weights to combine values.
High similarity = this past day is very relevant today.

### Multi-Head Attention (4 heads)
Instead of one attention computation, we run 4 parallel ones.
Each head learns to attend to different aspects:
  Head 1 might focus on momentum patterns
  Head 2 might focus on volatility spikes
  Head 3 might focus on regime transitions
  Head 4 might focus on recent price action

Results from all heads are concatenated and projected.

### Architecture Parameters
d_model = 64        (embedding dimension)
nhead = 4           (4 attention heads)
dim_feedforward = 128  (feedforward layer size)
num_layers = 2      (2 transformer layers stacked)
dropout = 0.1       (regularization)

Why 2 layers:
  Layer 1: captures local patterns (recent days)
  Layer 2: captures global patterns (interactions across 30 days)
  More layers would overfit on our ~3900 row dataset

### Positional Encoding
Transformer has no built-in sense of order.
Day 1 and Day 30 look the same without position info.
Positional encoding adds a unique signal to each position:
  Day 1 gets encoding vector e1
  Day 2 gets encoding vector e2
  ...
  Day 30 gets encoding vector e30

We use learnable positional encodings:
  nn.Embedding(seq_len=30, embed_dim=64)
  These are learned from data unlike fixed sinusoidal encodings

---

## Component 4 — Mixture of Experts
### File: models/ramt/moe.py

### What It Does
Uses the HMM regime to route predictions through
specialized expert Transformers.
Each expert specializes in one market regime.

### Why MoE Instead of Single Model

Single model approach (what LSTM does):
  One model trained on all market conditions
  Must learn bull AND bear AND crisis patterns
  Patterns conflict — bull momentum vs bear reversal
  Model compromises — suboptimal for all regimes

MoE approach (what RAMT does):
  Bull Expert: trained to understand bull market patterns
  Bear Expert: trained to understand bear market patterns
  Crisis Expert: trained to understand high-vol patterns
  Each expert focuses on one regime — no compromise needed

XGBoost Phase 1 used hard routing:
  If regime == Bear: use ONLY Bear model
  Binary switch — either/or

RAMT uses soft gating:
  Gate weights: Bull=10%, Bear=75%, HighVol=15%
  Prediction = 0.10×Bull + 0.75×Bear + 0.15×HighVol
  Weighted blend — smooth transitions between regimes
  More realistic — markets transition gradually

### The Three Experts

Each ExpertTransformer:
Input (batch, seq_len, 64)
↓
TransformerEncoderLayer (d_model=64, nhead=4)
↓
TransformerEncoderLayer (d_model=64, nhead=4)
↓
Pool last timestep → (batch, 64)
↓
Linear(64 → 1)
Output (batch, 1) — return prediction

All 3 experts have identical architecture.
But they have SEPARATE weights.
They learn different patterns from different regime data.

### The Gating Network
Input 1: fused_embedding last timestep (batch, 64)
Input 2: regime one-hot encoding (batch, 3)
[1,0,0] = HighVol
[0,1,0] = Bull
[0,0,1] = Bear
↓
Concatenate → (batch, 67)
↓
Linear(67 → 32) → ReLU
↓
Linear(32 → 3) → Softmax
Output (batch, 3) — weights over 3 experts

Why concatenate regime one-hot with embedding:
  Regime label gives explicit routing signal
  Embedding gives context about current market state
  Together: "We know it's bear regime AND
             we can see the current market features"
  More informed routing than either alone

Why Softmax:
  Outputs sum to 1.0
  Interpretable as probabilities
  Gate weights: "75% Bear expert, 25% others"

### Final MoE Outputexpert_outputs = [bull_pred, bear_pred, crisis_pred]
= [(batch,1), (batch,1), (batch,1)]
↓
Stack → (batch, 3)
↓
gate_weights → (batch, 3)
↓
weighted_sum = sum(gate_weights × expert_outputs, dim=1)
Output (batch, 1) — final blended prediction

---

## Component 5 — Combined Loss Function
### File: models/ramt/losses.py

### What It Does
Combines two loss signals:
1. MSE — minimize prediction error magnitude
2. Directional — penalize wrong-direction predictions

### Why Not Just MSE

MSE alone:
  Penalizes large magnitude errors equally regardless of direction
  A prediction of +0.1% when actual is +2% has MSE = (2-0.1)² = 3.61
  A prediction of -0.1% when actual is +2% has MSE = (2+0.1)² = 4.41
  MSE difference: 4.41 - 3.61 = 0.80 (small penalty for wrong direction)

In trading, direction is everything:
  +0.1% prediction, +2% actual → YOU PROFIT (correct direction)
  -0.1% prediction, +2% actual → YOU LOSE (wrong direction)
  MSE barely distinguishes these. Trading outcome is completely different.

### The Combined Lossmse_loss = mean((y_pred - y_true)²)directional_loss = mean(ReLU(-(y_true × y_pred)))Explanation of directional_loss:
y_true × y_pred:
Both positive (both up): product > 0 (correct)
Both negative (both down): product > 0 (correct)
Opposite signs: product < 0 (wrong direction)ReLU(-(product)):
Correct direction: ReLU(negative) = 0 (no penalty)
Wrong direction: ReLU(positive) > 0 (penalty!)mean: average over all samplestotal_loss = mse_loss + 0.3 × directional_loss

Lambda = 0.3:
  70% weight on magnitude accuracy (MSE)
  30% weight on directional accuracy
  Balance: we want accurate magnitudes AND correct directions

---

## Component 6 — Training Loop
### File: models/train.py

### What It Does
Trains RAMT on all tickers with proper optimization.

### Optimizer — AdamW
Adam optimizer + weight decay decoupled.

Why AdamW over plain Adam:
  Adam with weight decay has a bug — weight decay
  is applied to the adaptive gradient estimate,
  not the actual parameters.
  AdamW fixes this — proper L2 regularization.
  Standard choice for all modern transformer training.

Parameters:
  lr = 0.001 (learning rate)
  weight_decay = 1e-4 (regularization strength)

### Learning Rate Scheduler — Cosine Annealing Warm RestartsLearning Rate
↑
0.001│\        /\        /
│  \    /    \    /    
0.0001│    /        /        /
└────────────────────────→ Epochs
0    10   20   30   40   50
↑         ↑         ↑
restart   restart   restart

Why cosine annealing:
  Smooth decay — not sudden drops
  Warm restarts — escape local minima
  After each restart, LR goes back up briefly
  Helps model explore different loss landscape regions

Why warm restarts for financial data:
  Financial data has non-stationary patterns
  Different regimes have different loss landscape shapes
  Restarts help model not get stuck in one regime's minimum

T_0 = 10: restart every 10 epochs initially
T_mult = 2: double restart period each time
  Restart 1: every 10 epochs
  Restart 2: every 20 epochs
  Restart 3: every 40 epochs

### Early Stopping
Monitor: validation loss
Patience: 15 epochs
If validation loss does not improve for 15 consecutive epochs:
  Stop training
  Load best model weights (from checkpoint)

Why early stopping:
  Prevents overfitting
  Financial data is noisy — model can memorize noise
  Best model is usually not the last epoch

### Training Loop Per FoldFor each epoch:

Set model to train mode
For each batch:
a. Forward pass → predictions
b. Compute combined loss
c. Backward pass → gradients
d. Gradient clipping (max_norm=1.0)
e. Optimizer step
f. Scheduler step
Set model to eval mode
Compute validation loss
Check early stopping
Save if best validation loss


Why gradient clipping (max_norm=1.0):
  Transformers can have exploding gradients
  Clipping prevents single large gradient update
  from destroying learned weights
  Standard practice in transformer training

---

## Component 7 — Walk-Forward for RAMT
### File: models/walk_forward_dl.py

### What It Does
Applies the same walk-forward structure as XGBoost
but trains a fresh RAMT model for each fold.

### Why Fresh Model Each Fold
We do not fine-tune the previous fold's model.
Each fold trains from scratch on expanding data.

Why:
  Financial data distribution shifts over time
  A model trained on 2010-2018 may have learned
  patterns that are harmful for 2019 prediction
  if 2018 was a bear year and 2019 turns bull
  Fresh model avoids this negative transfer

### The Walk-Forward ProcessFold 1:
Train: 2010 ─────────────── 2018
Test:  2018 Q1
Fresh model, fresh scalerFold 2:
Train: 2010 ──────────────────── 2018 Q1
Test:  2018 Q2
Fresh model, fresh scaler...Fold 25:
Train: 2010 ────────────────────────── 2025 Q3
Test:  2025 Q4
Fresh model, fresh scaler

### Additional Metrics in Phase 2

Max Drawdown:
  Largest peak-to-trough decline in cumulative returns
  Measures worst-case loss if you used this strategy
  Formula: min((cumulative - rolling_max) / rolling_max)
  Lower (less negative) is better

Profit Factor:
  Total gains / Total losses
  Above 1.0 = strategy makes more than it loses
  Formula: sum(positive returns) / sum(abs(negative returns))
  Higher is better

Calmar Ratio:
  Annual return / Max Drawdown
  Risk-adjusted return accounting for worst drawdown
  Formula: (mean_return × 252) / abs(max_drawdown)
  Higher is better

---

## Why This Architecture Will Beat XGBoost

### XGBoost Limitations RAMT Fixes

Limitation 1 — No temporal structure:
  XGBoost sees one row at a time
  It uses lagged features as a workaround
  RAMT processes 30-day sequences with attention
  Can learn complex temporal patterns XGBoost misses

Limitation 2 — Hard regime routing:
  XGBoost uses hard switch (regime=Bear → Bear model)
  RAMT uses soft gating (blend experts continuously)
  Market transitions are gradual — soft is more realistic

Limitation 3 — No cross-feature temporal interaction:
  XGBoost: RSI on Day 15 cannot interact with
           Volatility on Day 28 (different rows)
  RAMT: attention mechanism captures any day-to-day
        interaction across the 30-day window

Limitation 4 — No shared learning across tickers:
  XGBoost trains separately per ticker
  RAMT architecture can share weights across tickers
  JPM patterns can inform RELIANCE prediction

---

## Model Parameter Count (Approximate)

LSTM Baseline: 36,385 parameters

RAMT:
  Encoders:          7 × ~1,500 = ~10,500
  Fusion layer:      224×64 + 64 = ~14,400
  Positional enc:    30×64 = 1,920
  Transformer (2L):  ~2 × 33,000 = ~66,000
  MoE (3 experts):   3 × ~16,000 = ~48,000
  Gating network:    67×32 + 32×3 = ~2,200
  Total: ~143,000 parameters

RAMT is ~4x larger than LSTM.
But with proper regularization and walk-forward
validation, this should not cause overfitting
on our ~3900 row training sets.

---

## Expected Results

### What We Expect RAMT To Achieve
DA% > 54% average (vs XGBoost 52.13%)
Sharpe > 1.0 average (vs XGBoost 0.27)
RMSE < 0.0173 average (vs XGBoost 0.0187)

### Why We Expect Improvement
1. Attention captures 30-day temporal patterns
   that XGBoost's lagged features miss
2. Soft MoE routing handles regime transitions
   better than hard XGBoost routing
3. Multimodal encoders handle feature group
   heterogeneity better than flat XGBoost input
4. Combined loss directly optimizes direction
   not just magnitude

### Honest Assessment
Daily stock return prediction is extremely hard.
Even a 1% improvement in DA% is significant.
We may not hit Sharpe > 2.0 (proposal target)
on CPU training with limited data.
But we should clearly beat LSTM and approach
or exceed XGBoost — that proves the concept.

---
models/
├── baseline_lstm.py        ✅ Done — LSTM walk-forward
├── baseline_xgboost.py     ✅ Done — XGBoost walk-forward
└── ramt/
├── init.py         ✅ Done — package init
├── dataset.py          ✅ Done — data pipeline
├── encoder.py          ⏳ Day 4-5 — multimodal encoders
├── moe.py              ⏳ Day 6-7 — mixture of experts
├── losses.py           ⏳ Day 8 — combined loss
├── model.py            ⏳ Day 8 — full RAMT model
├── train.py            ⏳ Day 9-10 — training loop
└── walk_forward_dl.py  ⏳ Day 11-12 — walk-forward
results/
├── xgboost_predictions.csv ✅ Done
├── lstm_predictions.csv    ✅ Done
├── ramt_predictions.csv    ⏳ Day 12
└── full_comparison.csv     ⏳ Day 13

---

## Key Numbers To Remember
Dataset:
Tickers: JPM, RELIANCE, TCS, HDFCBANK, EPIGRAL
Rows: ~3900 per ticker (2010-2026)
Features: 27 numeric inputs to RAMT
Sequence length: 30 days
Target: next-day log return
Architecture:
Embed dim: 64
Attention heads: 4
Transformer layers: 2
MoE experts: 3
Walk-forward step: 63 days
Baselines to beat:
XGBoost: DA=52.13%, Sharpe=0.27
LSTM:    DA=51.02%, Sharpe=0.28
Target:  DA>54%,    Sharpe>1.0

---

*RAMT Phase 2 Documentation*
*Shivansh Gupta (230054) + Vivek Vishnoi (230119)*
*Rishihood University*
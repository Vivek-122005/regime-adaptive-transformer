## Attention-based explainability (RAMT, 30-day window)

This project uses a **Transformer (RAMT)** on **30-day sequences**. Transformers can learn to **ignore irrelevant days** and focus on “trigger” days. In finance, this is also your best practical tool for **sanity checking** that the model is not just memorizing noise.

This doc explains:
- what attention inspection is,
- how to run it in this repo,
- what patterns to look for,
- and the next “production” upgrades (ranking loss, friction, risk-adjusted target, liquidity gating).

---

## 1) What we already have (current repo status)

- **Attention extraction implemented**:
  - `models/ramt/moe.py`: `ExplainableTransformerEncoderLayer` stores per-head attention weights.
  - `models/ramt/model.py`: `build_ramt(..., explainable_attn=True)` enables attention capture.
  - `models/inspect_attention.py`: runs a single forward pass for a `ticker/date` and saves attention CSVs.

- **Ranking-aware training already present**:
  - `models/ramt/train_ranking.py` includes a **pairwise logistic ranking loss** (`_pairwise_rank_loss`) added to the regression loss.
  - Updated: `models/ramt/train_ranking.py` now uses a **LambdaRank-style** (NDCG-weighted) objective per rebalance-date group for better top-of-list accuracy.

---

## 2) Why attention inspection matters (“the Why filter”)

### The trap (overfitting by “cheating”)
The model can appear strong in backtests while actually relying on:
- rare outlier days,
- a single extreme return event,
- or unstable patterns that do not repeat.

### The fix
Inspect attention maps to check **consistency**:
- If the model consistently focuses on **recent days (28–30)** plus a meaningful lag (e.g., ~Day 21), it likely learned a stable pattern (momentum / reversion mix).
- If attention is scattered randomly across days and varies wildly across dates, it’s a red flag for weak signal.

Important limitation:
- Attention is **not perfect causality**, but it’s a strong **debugging lens**.

---

## 3) How to generate attention maps in this repo

### Step A — Train once and save artifacts
Running the final runner will train and also save:
- `results/ramt/ramt_model_state.pt`
- `results/ramt/ramt_scaler.joblib`

Command:

```bash
.venv/bin/python models/run_final_2024_2026.py
```

### Step B — Inspect attention for a specific rebalance date
Pick a `Date` that exists in `results/final_strategy/ranking_predictions.csv`, then run:

```bash
.venv/bin/python models/inspect_attention.py --ticker TCS_NS --date 2024-10-09
```

Outputs:
- `results/ramt/attention/<prefix>_map_mean.csv` (legacy docs referred to `attention_map_mean.csv` at repo root; new runs use a prefix under `results/ramt/attention/`)
  - mean attention matrix (30×30), averaged across experts/layers/heads
- `results/ramt/attention/<prefix>_last_token.csv`
  - a single vector of length 30: how much the **last timestep** attends to each day

New (recommended) outputs (prefix is configurable with `--out-prefix`):
- outputs are written under `results/ramt/attention/` as `<prefix>_map_mean.csv`, `<prefix>_last_token.csv`
- plus `<prefix>_heatmap.html` (and optionally `<prefix>_heatmap.png` if `kaleido` is installed)

Example:

```bash
.venv/bin/python models/inspect_attention.py --ticker TCS_NS --date 2024-10-09 --out-prefix tcs_2024_10_09
```

### Consistency report across multiple rebalance dates

```bash
.venv/bin/python models/attention_consistency_report.py --ticker TCS_NS --n 12
```

Output:
- `results/ramt/attention/attention_consistency.csv` (top attended days + “mass on last 5/10 days” + entropy)

Index meaning:
- `day_index = 0` is the **oldest** day in the 30-day block
- `day_index = 29` is the **most recent** day

---

## 4) What to look for in the results

### A) “Last token attention” (most practical)
Open `results/ramt/attention/<prefix>_last_token.csv` and check:
- Is most weight near day 29 (recent days)?
- Are there stable secondary peaks (e.g., day ~21, ~10)?
- Does the shape remain similar across multiple rebalance dates?

### B) Full attention map
In `results/ramt/attention/<prefix>_map_mean.csv`:
- A strong diagonal-ish pattern often indicates “local recency” preference.
- Off-diagonal structure indicates “the model uses older context as triggers”.

---

## 5) Production upgrades to focus on next

### 5.1 Shift further from regression → ranking (top-5 accuracy)
Current training uses MSE + directional loss + pairwise ranking loss.
To optimize “make money”, you want to optimize “top of list is correct”:
- consider stronger listwise objectives (ListNet/ListMLE) or LambdaRank-style objectives
- validate with ranking metrics (IC, NDCG@5, top-5 hit rate)

### 5.2 Model friction (India market reality)
Backtests that ignore friction often fail live because rebalancing costs compound:
- STT + brokerage + slippage + spread

Two practical improvements:
- **Backtest costs**: subtract a cost per turnover event (e.g., 15–20 bps each rebalance)
- **Turnover penalty** in training or portfolio construction:
  - don’t swap holdings unless the “edge” is large enough to pay costs

### 5.3 Refine target: risk-adjusted alpha
`Monthly_Alpha` is good, but it can prefer very volatile stocks.
Upgrade targets to prefer “smooth” outperformers:
- volatility-normalized alpha (e.g., alpha / rolling vol)
- or multi-task (predict alpha + risk) and rank by expected risk-adjusted score

### 5.4 Liquidity gate (avoid untradable picks)
Before selecting the top-5:
- filter out stocks below a minimum average traded value / volume
- this reduces slippage and improves live execution reliability

---

## 6) Recommended checklist (in order)

- **Critical**: run attention inspection across ~10 rebalance dates and confirm consistent focus patterns
- **High**: validate ranking metrics (IC, NDCG@5, long-short spread) on 2024–2026
- **High**: add transaction cost + slippage model in backtest
- **High**: add a turnover limiter (portfolio-level)
- **Medium**: liquidity gate
- **Medium**: risk-adjusted target


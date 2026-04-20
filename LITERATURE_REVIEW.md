# Literature Review

This literature review explains the research context behind Phase 2 of the
project: the Regime-Adaptive Multimodal Transformer (RAMT) for monthly
cross-sectional equity selection on the NIFTY 200 universe.

The two core reference papers for the Phase 2 implementation were:

1. Vaswani et al. (2017), "Attention Is All You Need".
2. Lim et al. (2021), "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting".

The first paper provides the core Transformer mechanism: scaled dot-product
attention, multi-head attention, positional encoding, and feed-forward
Transformer blocks. The second paper shows how attention-based architectures
can be adapted to structured time-series forecasting with heterogeneous inputs,
gating, static covariates, temporal context, and interpretability. RAMT takes
inspiration from both, but changes the problem from sequence-to-sequence
forecasting into cross-sectional stock ranking.

Before Phase 2, the project also considered two more ambitious research
directions: multimodal news-price reinforcement learning and graph-based equity
correlation forecasting. These ideas shaped the early project thinking, even
though the final implementation used a more tractable subset of the ideas.

The final empirical result of the project is also part of the literature review:
although RAMT was architecturally motivated, a simpler 21-day momentum strategy
with HMM regime sizing performed better under the available data scale and
backtest assumptions. This does not invalidate the deep learning design; it
shows why classical baselines and realistic validation are essential in
financial machine learning.

---

## 1. Phase 1 Research Inspiration: News-Price Alignment and Graph Correlation Forecasting

In Phase 1, the project was initially influenced by two advanced research
directions. The first was a multimodal news-price trading system that combines
large language models, Transformers, and reinforcement learning. The second was
a hybrid Transformer Graph Neural Network for forecasting future equity
correlations. These papers were not implemented directly in full, but they
helped define what a more advanced financial deep learning system could look
like.

### 1.1 Cross-Modal News-Price Transformer Reinforcement Learning

The paper "Aligning News and Prices: A Cross-Modal LLM-Enhanced Transformer DRL
Framework for Volatility-Adaptive Stock Trading" proposes a trading system that
uses both market prices and financial news. Its central argument is that prices
alone are often incomplete because market-moving information may first appear in
text: headlines, earnings news, macro announcements, analyst commentary, or
company-specific events.

The paper combines several components:

- A news encoder based on large language models such as BERT/GPT-style models.
- A price encoder for historical price and volume sequences.
- A reprogramming layer that maps numerical price features into a semantic
  representation compatible with language-model embeddings.
- Cross-attention between news and price representations.
- Transformer encoders for temporal feature extraction.
- A Soft Actor-Critic reinforcement learning module for trading decisions.

The important idea is not simply "use news". The more specific idea is that
news and price are different modalities. Price is numerical, regular, and
available at fixed intervals. News is textual, irregular, event-driven, and
often forward-looking. A cross-modal model tries to align these two information
sources so that the trading policy can learn relationships such as:

```text
negative earnings news + falling price momentum + high volatility
    -> reduce exposure or avoid the stock
```

or:

```text
positive sector news + stable price trend + improving volume
    -> increase allocation
```

In the original Phase 1 plan, this paper suggested that a stronger version of
the project could include financial news sentiment and event information. For
example, for each stock-date pair, the system could have collected recent news
headlines, encoded them using a financial language model, fused them with the
stock's 30-day price window, and then used the combined representation for
ranking or portfolio decisions.

The final project did not implement this full design. There is no news dataset,
LLM news encoder, sentiment extraction module, price-to-language reprogramming
layer, or SAC reinforcement learning policy in the codebase. Instead, the
implemented project kept the structured market-data side:

- price returns,
- technical indicators,
- volume surge,
- macro variables,
- HMM market regime,
- Transformer-based sequence modeling,
- portfolio backtesting.

This was a practical design decision. For Indian equities, reliable historical
stock-specific news aligned to every stock and every trading day is difficult to
collect cleanly. Adding news, LLM embeddings, and SAC would also make the system
much harder to validate. Because Phase 1 already showed that daily return
prediction was very noisy, the project moved toward a more controlled Phase 2
problem: monthly cross-sectional ranking using structured features.

Therefore, this paper is best described as an **initial inspiration and future
extension**, not as an implemented baseline. It influenced the idea of combining
multiple market information sources and using attention-based fusion, but the
final project does not claim to reproduce its multimodal LLM-RL architecture.

### 1.2 Hybrid Transformer Graph Neural Network for Equity Correlations

The second Phase 1 reference was "Forecasting Equity Correlations with Hybrid
Transformer Graph Neural Network" by Fanshawe, Masih, and Cameron. This paper
studies a different but highly relevant problem: predicting future correlations
between stocks.

In portfolio construction, correlation matters because the risk of a portfolio
does not depend only on the expected return of each stock. It also depends on
how the stocks move together. Two stocks with strong individual signals may
still create a fragile portfolio if they are highly correlated and crash
together during stress periods.

The paper models this using a hybrid architecture:

- A Transformer encoder processes each stock's recent time-series features.
- A graph represents stocks as nodes and stock-pair relationships as edges.
- Edge attributes include historical correlation strength, sign, and sector or
  industry similarity.
- A Graph Attention Network passes information across related stocks.
- The target is future stock-pair correlation, transformed using Fisher-z.
- The model predicts residual correlation changes over a rolling historical
  baseline.

This is important because it separates two levels of market structure:

```text
Stock-level temporal behavior:
    What has this individual stock been doing recently?

Cross-stock relational behavior:
    How does this stock move relative to other stocks?
```

The paper also addresses a common financial modeling problem: prediction
collapse. If a model predicts correlations too close to the historical average,
it may achieve acceptable error but fail to separate useful high-correlation and
low-correlation pairs. The paper therefore adds distribution-aware objectives,
including histogram matching, to preserve the shape of predicted correlations.

This paper influenced the project in three main ways.

First, it supported the idea of using a Transformer over recent stock history.
RAMT uses rolling windows of market features and a Transformer-style encoder to
represent each stock's recent behavior.

Second, it reinforced the value of market context. The final project includes
macro variables, sector information, and HMM regimes because stock behavior is
not independent of the broader market environment.

Third, it showed why cross-sectional structure matters. The final strategy does
not build a graph, but it does rank stocks across the same rebalance date and
uses a sector cap to avoid selecting an overly concentrated group of names.

However, the graph-correlation paper was not implemented directly. The project
does not contain:

- stock-stock graph construction,
- Graph Neural Network layers,
- Graph Attention Network message passing,
- pairwise edge features,
- Fisher-z correlation targets,
- residual correlation forecasting,
- SPONGE clustering,
- basket-trading based on predicted correlations.

The reason is that the final project objective changed. The paper predicts
future stock-pair correlations, while this project predicts or ranks future
stock alpha. In other words, the paper asks:

```text
Which stocks will move together?
```

This project asks:

```text
Which stocks are likely to outperform over the next 21 trading days?
```

Those are related but different problems. Implementing the full graph approach
would require building a pairwise stock dataset, defining edge labels for every
stock pair, training a graph model, and then designing a portfolio construction
method based on predicted correlation clusters. That would be a separate major
project. For Phase 2, the more realistic path was to keep the model focused on
stock-level ranking and use sector caps plus HMM sizing for risk control.

Thus, this paper should also be presented as an **early research influence and
future work direction**. It explains why a future version of RAMT could include
cross-stock attention or graph-based correlation awareness, but the current
implementation remains a stock-level ranking model rather than a graph
correlation forecasting model.

---

## 2. Problem Context: Stock Selection as Noisy Time-Series Ranking

Stock prediction is difficult because financial returns are noisy,
non-stationary, weakly predictable, and sensitive to regime changes. In many
supervised learning tasks, the model sees a strong relationship between input
and label. Equity returns are different. A stock can have strong technical
signals, favorable macro context, and high recent momentum, but a single
earnings surprise, policy event, liquidity shock, or market-wide correction can
dominate the next month's return.

This project therefore does not frame the problem as raw price prediction. It
frames the problem as monthly relative stock selection:

- For each stock and date, compute the next 21-trading-day return.
- Subtract the NIFTY benchmark return over the same forward window.
- Rank stocks within the same rebalance date.
- Select only the strongest few names for the portfolio.

The target used in Phase 2 is `Sector_Alpha`, which is derived from
`Monthly_Alpha`. `Monthly_Alpha` is:

```text
stock forward 21-day log return - NIFTY forward 21-day log return
```

`Sector_Alpha` then removes the median alpha of the stock's sector on the same
date. This matters because a model should not receive high marks just for
learning that one sector was strong in a particular month. The real question is:
within the sector and within the date, which stock is relatively better?

This transforms the learning problem from:

```text
Predict the exact return of one stock.
```

into:

```text
Given approximately 200 stocks on the same date, rank them so the top names
are more likely to outperform over the next 21 trading days.
```

That distinction drives the entire Phase 2 design. It explains why the model
uses sequences, why the loss includes ranking terms, why cross-sectional metrics
such as Information Coefficient are reported, and why the final strategy can be
evaluated using top-k portfolio returns instead of only prediction error.

---

## 3. Classical Momentum and Factor Literature

Momentum is one of the most established empirical findings in asset pricing.
Jegadeesh and Titman (1993) showed that stocks with strong past returns tend to
continue outperforming over intermediate horizons. Later factor research,
including Asness, Moskowitz, and Pedersen (2013), documented that momentum is
not limited to one market; it appears across asset classes and countries.

The direct connection to this project is the feature `Ret_21d`, a 21-trading-day
log return. In the final strategy, this feature becomes the ranking signal:

```text
predicted_alpha = Ret_21d
```

This may look too simple, but the literature supports momentum as a serious
baseline rather than a naive rule. In a noisy equity universe, a short-horizon
momentum sort can be more robust than a high-capacity neural model, especially
when the available data is limited and the model is asked to separate only the
top few stocks.

The project's results support this view. The RAMT model produced collapsed
cross-sectional predictions and negative/weak correlation with realized alpha.
By contrast, pure `Ret_21d` momentum produced stronger top-5 behavior in the
test window. This means the final pivot was not an abandonment of research
rigor; it was consistent with the factor literature's warning that simple
momentum is a difficult benchmark to beat.

Why this matters for the marking scheme:

- It justifies why the final signal is not "just a shortcut".
- It shows that the baseline is literature-backed.
- It prevents the deep learning model from being evaluated in isolation.

---

## 4. Regime Detection Literature

Financial markets do not behave the same way in every period. A model that works
in a calm bull market may fail during a liquidity shock or high-volatility
drawdown. Regime-switching models address this by assuming that the observed
time series is generated under different hidden states.

Hamilton (1989) introduced Markov regime-switching models in economics. The
core idea is that the data may move between hidden states, and each state has
different statistical properties. In financial applications, Hidden Markov
Models (HMMs) are commonly used to detect market regimes such as expansion,
stress, crash, or high-volatility conditions. Guidolin and Timmermann (2007)
applied regime-switching ideas to asset allocation, showing that regime
awareness can change portfolio decisions.

In this project, the HMM is not mainly used to predict returns directly. It is
used to control portfolio risk. The feature pipeline fits a 3-state Gaussian HMM
using:

```text
Ret_1d
Realized_Vol_20
```

The raw HMM states are then mapped to semantic regimes:

```text
highest mean return state -> Bull
lowest mean return state  -> Bear
middle state              -> High-vol
```

In the final backtest, the NIFTY HMM regime controls exposure:

```text
Bull     -> deploy 100 percent capital
High-vol -> deploy 50 percent capital
Bear     -> deploy 20 percent capital
```

This design is important because it separates two tasks:

1. Ranking: which stocks look strongest?
2. Sizing: how much risk should the portfolio take in the current market state?

RAMT originally tried to put regime awareness inside the neural architecture
through regime embeddings and regime cross-attention. After the transformer
failed to improve the signal, the final strategy moved regime awareness into
the portfolio construction layer. This is a cleaner use of the HMM: instead of
forcing the model to learn a complex regime-conditioned ranking function, the
regime model acts as a risk sleeve.

The multi-window HMM ablations in `RESULTS.md` show why this matters. HMM sizing
can reduce drawdowns in difficult periods, although it can also suppress upside
in benign markets. This is consistent with the interpretation of HMM sizing as
tail-risk insurance rather than alpha generation.

---

## 5. Deep Learning for Financial Time Series

Deep learning is a natural candidate for financial time series because market
data is sequential. Price, volume, volatility, macro variables, and technical
indicators evolve through time, and the order of observations matters. Earlier
deep learning approaches often used recurrent models such as RNNs, GRUs, and
LSTMs because they process sequences step by step and maintain hidden state.

This project used Phase 1 baselines such as XGBoost and LSTM before moving to
RAMT. That progression was important. The first target, daily return prediction,
was too noisy. Daily returns at the individual-stock level contain very low
signal-to-noise ratio, so even if a model appears to learn patterns on the
training data, those patterns can disappear out of sample.

The Phase 2 pivot changed the target from daily return prediction to 21-day
forward alpha. This is more aligned with equity selection because:

- Monthly horizons reduce some daily noise.
- Cross-sectional ranking is more stable than exact scalar prediction.
- Portfolio construction only needs the top few names, not perfect forecasts
  for every stock.

However, deep learning still faces several problems in this domain:

- The dataset is small compared with NLP or vision tasks.
- The return label is noisy even at monthly horizons.
- The equity universe is static, which introduces survivorship bias.
- Regime shifts change the input-output relationship.
- Backtest metrics can look good if friction, turnover, or date alignment are
  handled incorrectly.

The RAMT experiments demonstrate these limitations. The model was technically
valid and trained on meaningful features, but its predictions collapsed toward a
narrow range. The lesson is not that deep learning is useless in finance. The
lesson is that deep learning must be validated against simple, hard-to-beat
baselines and realistic trading assumptions.

---

## 6. Transformer Models for Time Series

### 6.1 Attention Is All You Need

Vaswani et al. (2017) introduced the Transformer architecture. Its central idea
is self-attention: each token in a sequence learns how much attention to pay to
every other token. The scaled dot-product attention mechanism is:

```text
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
```

where:

- `Q` is the query matrix.
- `K` is the key matrix.
- `V` is the value matrix.
- `d_k` is the key dimension used for scaling.

The Transformer replaced recurrence with attention and feed-forward blocks,
allowing sequence elements to interact directly. Multi-head attention repeats
this operation in multiple learned subspaces, so different heads can learn
different relationships.

Connection to RAMT:

- RAMT uses a Transformer encoder over 30-day stock feature sequences.
- Learnable positional embeddings tell the model which day in the 30-day window
  each feature vector belongs to.
- Multi-head attention is used to let the model learn relationships across
  days, such as whether recent momentum, older reversal, or macro shocks matter.
- The model keeps the last timestep representation as the summary used for
  monthly and daily prediction heads.

Why the vanilla paper is not enough:

The original Transformer was built for machine translation, not stock ranking.
Language tokens carry dense semantic information. Daily market feature vectors
are much noisier and have weaker local meaning. This is why RAMT does not simply
copy the original architecture. It adds financial inductive bias through
multimodal feature groups, regime embeddings, ticker embeddings, and ranking
loss.

### 6.2 Temporal Fusion Transformer

Lim et al. (2021) introduced the Temporal Fusion Transformer (TFT), an
attention-based architecture for interpretable multi-horizon time-series
forecasting. TFT is relevant because it handles a realistic forecasting setting
with different kinds of inputs:

- static covariates,
- known future inputs,
- observed historical inputs,
- gating mechanisms,
- recurrent local processing,
- interpretable attention over time.

The main idea taken from TFT is not a direct code copy. The influence is
architectural: real time-series forecasting often needs heterogeneous inputs and
interpretability, not only a plain attention block.

Connection to RAMT:

- RAMT separates features into modalities: price, technical, volume, and macro.
- RAMT includes ticker embeddings, similar in spirit to static covariates.
- RAMT includes regime embeddings, which condition the model on market state.
- RAMT includes attention inspection tools to check whether the model focuses on
  meaningful days.
- RAMT uses dual heads: a monthly ranking head and a daily auxiliary head.

Difference from TFT:

TFT is designed for multi-horizon forecasting of time-series targets. RAMT is
designed for cross-sectional stock ranking. TFT predicts future values for a
series; RAMT scores many stocks on the same rebalance date and only needs the
relative order. Therefore, RAMT replaces pure forecasting evaluation with
ranking diagnostics and portfolio backtesting.

### 6.3 Informer

Informer (Zhou et al., 2021) was designed for long-sequence time-series
forecasting. The paper addresses the computational cost of self-attention by
using ProbSparse attention, self-attention distilling, and a generative-style
decoder. Its goal is efficient forecasting when sequences are very long.

Connection to this project:

RAMT uses only 30-day input windows, so Informer's efficiency improvements are
not necessary. However, Informer is important literature because it shows that
time-series Transformers often require domain-specific changes. A direct
vanilla Transformer is not automatically optimal for forecasting.

Why not implement Informer here?

The bottleneck in this project was not extremely long input length. The harder
problem was low signal-to-noise ratio and cross-sectional ranking collapse.
Using Informer's sparse attention would reduce computation for long sequences,
but it would not directly solve the ranking-loss failure or the limited data
scale.

### 6.4 Autoformer

Autoformer (Wu et al., 2021) argues that long-term time-series forecasting
benefits from decomposition and auto-correlation rather than only pointwise
self-attention. It decomposes series into trend and seasonal components and
uses an auto-correlation mechanism to capture repeated patterns.

Connection to this project:

Financial prices and returns can contain short-term momentum, reversal, and
volatility regimes. Autoformer's decomposition idea is relevant because it
suggests that raw self-attention may be too generic for time-series structure.

Why not implement Autoformer here?

RAMT operates on short 30-day windows and only 10 features. The project goal was
not long-horizon forecasting of a continuous series, but top-k stock selection.
Autoformer's decomposition blocks could be a future improvement, especially for
separating trend and reversal components, but they were not the first priority
for this Phase 2 implementation.

### 6.5 PatchTST

PatchTST (Nie et al., 2023) treats time-series segments as patches, similar to
how Vision Transformers process image patches. Instead of treating every single
time step as an independent token, PatchTST groups local windows into patch
tokens. This can reduce noise, increase effective context length, and improve
forecasting performance.

Connection to this project:

RAMT currently treats each day in the 30-day window as one timestep token. In
financial data, a single day can be noisy and may not carry stable meaning.
Patch-style tokenization could be useful because a 5-day or 10-day patch may
represent a more meaningful market pattern than one daily row.

Potential future improvement:

Replace the daily-token input:

```text
30 tokens x 10 features
```

with patch tokens such as:

```text
6 tokens x 5-day patches
```

This could help the model learn weekly momentum/reversal structures while
reducing sensitivity to one-day noise.

---

## 7. Learning-to-Rank Literature

Most regression models optimize scalar prediction error. For stock selection,
that is not always the right objective. A portfolio does not buy every stock
proportionally to its predicted return. It usually buys the top few names.
Therefore, the model should be judged by whether it ranks winners above losers
on the same rebalance date.

Learning-to-rank literature provides three broad approaches:

1. Pointwise ranking: predict a score for each item independently.
2. Pairwise ranking: compare pairs and penalize incorrect ordering.
3. Listwise ranking: optimize the quality of the entire ranked list.

RankNet (Burges et al., 2005) introduced a neural pairwise ranking approach.
LambdaRank and LambdaMART later improved ranking optimization by focusing
gradients on rank positions that matter for metrics such as NDCG. ListNet (Cao
et al., 2007) moved toward listwise ranking by modeling probability
distributions over permutations or top-ranked lists.

Connection to RAMT:

RAMT uses `TournamentRankingLoss`, a pairwise margin-ranking loss. For every
pair of stocks on the same date where the true alpha of stock `i` is greater
than the true alpha of stock `j`, the model is encouraged to assign a higher
score to `i`:

```text
loss = ReLU(margin - (pred_i - pred_j)) * abs(alpha_i - alpha_j)
```

The magnitude weighting makes larger alpha gaps more important. This aligns the
training objective with portfolio construction: the model should focus on
economically meaningful separations, not tiny differences near zero.

Why this was a good idea:

- The problem is cross-sectional.
- The portfolio trades top-k names.
- Ranking quality matters more than exact alpha calibration.
- Pairwise losses are easier to implement than full listwise objectives.

Why it still failed:

The project found a practical failure mode. If predictions collapse toward a
narrow range early in training, pairwise score differences remain small and the
model does not learn enough separation. The daily MSE auxiliary head was added
to stabilize training, but it did not fully solve this. This suggests that
future work should test stronger listwise losses such as ListMLE/ListNet,
LambdaRank-style objectives, or direct NDCG@k optimization.

---

## 8. Finance-Specific Transformer Papers

Recent finance-specific Transformer papers show that attention-based models can
be useful for markets, but they also show that architecture must match the
financial task.

### MASTER: Market-Guided Stock Transformer

MASTER is a market-guided Transformer for stock prediction. Its key idea is that
stock movement should not be modeled only from a stock's own history; the model
should also consider market-wide information and cross-stock relations.

Connection to this project:

RAMT uses a similar motivation. It includes macro features, NIFTY-relative
targets, HMM market regime, and ticker embeddings. However, RAMT does not
directly model a graph of stock-to-stock relations. Instead, cross-sectional
structure enters through sector-neutral targets and ranking loss.

Potential future improvement:

Add an explicit cross-stock attention block or sector-level attention layer so
the model can compare stocks within the same date before producing scores.
Currently, RAMT scores each stock sequence independently and uses ranking loss
to compare outputs. A market-guided cross-sectional block could make the
comparison explicit.

### StockFormer and trading-oriented Transformers

StockFormer-style papers explore Transformer branches for extracting temporal
dynamics, asset relations, and trading policy signals. Some combine predictive
modeling with reinforcement learning. These papers are relevant because they
move beyond simple one-stock forecasting and try to connect representation
learning to portfolio decisions.

Connection to this project:

RAMT does not use reinforcement learning. It separates prediction and portfolio
construction:

```text
RAMT or momentum -> ranking_predictions.csv -> backtest rules
```

This separation makes the project easier to audit. The downside is that the
model is not trained directly on portfolio-level utility such as Sharpe,
drawdown, turnover, or transaction costs.

Potential future improvement:

Train the model with a differentiable portfolio objective or add a turnover
penalty during selection. This would connect the learning objective more
directly to real trading performance.

### DeepLOB and high-frequency market models

DeepLOB uses deep learning for limit order book data and short-horizon price
movement prediction. It is not directly comparable to this project because this
project uses daily OHLCV and macro features, not intraday order book snapshots.
However, DeepLOB is useful as a contrast: when rich high-frequency data is
available, deep models can exploit microstructure patterns. In this project, the
data is lower frequency and less information-dense, so a smaller and more
carefully validated architecture is appropriate.

---

## 9. Research Gap

Existing time-series Transformer papers mainly optimize forecasting accuracy for
one or more time series. Existing financial prediction papers often focus on
price movement prediction, return regression, or trading policy learning. This
project studies a more specific and practical question:

```text
Can a regime-aware Transformer improve monthly cross-sectional stock selection
on Indian equities when evaluated through realistic portfolio backtesting?
```

The project combines several ideas that are often studied separately:

- time-series attention from Transformer literature,
- heterogeneous feature processing inspired by TFT,
- HMM-based market regime detection,
- cross-sectional equity ranking,
- sector-neutral targets,
- realistic friction-aware portfolio backtesting,
- comparison against simple factor baselines.

The gap is not merely architectural. It is methodological. Many stock prediction
projects stop at validation loss, RMSE, or accuracy. This project evaluates the
model at the level where the financial decision is actually made:

```text
Did the selected top stocks produce better risk-adjusted portfolio performance?
```

The answer was no for RAMT and yes for the simpler momentum + HMM strategy.
That negative result is important. It shows that in low signal-to-noise domains,
model complexity must earn its place against strong simple baselines.

---

## 10. How This Literature Influenced Our Design

| Literature Idea | Source | How We Used It |
|---|---|---|
| Self-attention | Vaswani et al. (2017) | Used Transformer encoder blocks over 30-day feature sequences. |
| Multi-head attention | Vaswani et al. (2017) | Allowed the model to learn different temporal relationships across recent and older days. |
| Positional encoding | Vaswani et al. (2017) | Added learnable position embeddings so day order inside the 30-day window is visible to the model. |
| News-price alignment | Aligning News and Prices | Considered during Phase 1 as a possible multimodal extension, but not implemented because the final system uses structured market data only. |
| LLM-based financial news features | Aligning News and Prices | Treated as future work; no news encoder or sentiment module is included in the current implementation. |
| Heterogeneous time-series inputs | Lim et al. (2021), TFT | Split inputs into price, technical, volume, macro, regime, and ticker groups before fusion. |
| Static/context covariates | Lim et al. (2021), TFT | Used ticker embeddings and regime embeddings as context variables. |
| Interpretability through attention | Lim et al. (2021), TFT | Added attention inspection tools to check whether RAMT attends to meaningful timesteps. |
| Regime switching | Hamilton (1989), Guidolin and Timmermann (2007) | Used a 3-state Gaussian HMM to label market regime and size portfolio exposure. |
| Cross-sectional momentum | Jegadeesh and Titman (1993), Asness et al. (2013) | Used `Ret_21d` as a benchmark and final signal after RAMT underperformed. |
| Learning to rank | RankNet, LambdaRank, ListNet | Implemented `TournamentRankingLoss` because top-k ordering matters more than exact return regression. |
| Efficient long-sequence attention | Informer | Reviewed but not implemented because RAMT uses short 30-day windows. |
| Time-series decomposition | Autoformer | Identified as a possible future improvement for separating trend/reversal components. |
| Patch tokenization | PatchTST | Identified as a possible future improvement to reduce daily noise by using multi-day patches. |
| Finance-specific Transformers | MASTER, StockFormer | Motivated regime-aware and market-aware design, while showing future scope for cross-stock attention. |
| Graph-based correlation forecasting | Fanshawe et al. | Considered as a future extension for cross-stock risk modeling; not implemented because the project target is stock alpha ranking, not pairwise correlation prediction. |
| Distribution-aware correlation loss | Fanshawe et al. | Inspired awareness of prediction collapse, but the implemented project uses ranking losses and diagnostics rather than Fisher-z residual and histogram losses. |

---

## 11. Critical Summary

The literature supports the original RAMT hypothesis: attention can model
sequential dependencies, TFT shows that heterogeneous time-series inputs can be
handled with interpretable attention, HMMs provide a reasonable market-regime
tool, and learning-to-rank losses match the top-k structure of portfolio
selection.

However, the empirical results show that a justified architecture is not the
same as a successful trading model. Financial time series have low
signal-to-noise ratio, and NIFTY 200 daily data from a limited historical window
is small compared with the datasets where Transformers usually dominate.

The RAMT implementation was a valid research attempt because it made several
task-specific modifications:

- multimodal feature encoders,
- regime embeddings,
- regime cross-attention,
- ticker embeddings,
- dual prediction heads,
- ranking-aware loss,
- walk-forward training,
- friction-aware backtesting.

The two Phase 1 papers clarify what the project did not implement. The final
system does not include LLM-based news sentiment, SAC reinforcement learning, or
graph neural networks for stock-pair correlation forecasting. These were
reasonable research directions, but they were outside the final implementation
scope and are better positioned as future extensions.

But the model still failed to beat momentum. The failure was informative:

- Predictions collapsed into a narrow range.
- Validation correlation became negative.
- A LightGBM diagnostic showed the features had some signal.
- A simple `Ret_21d` sort produced better top-5 behavior.

The final conclusion is therefore balanced:

```text
Transformers are powerful sequence models, but they are not automatically better
than simple factor strategies in noisy financial ranking tasks.
```

For this project, the strongest final system is not pure RAMT. It is:

```text
21-day momentum ranking + sector cap + HMM regime sizing + realistic friction-aware backtest.
```

This conclusion is consistent with both the literature and the experimental
evidence. The project contributes by showing not only how a regime-adaptive
Transformer can be designed, but also how it should be rejected when a simpler,
more robust baseline wins.

---

## References

Asness, C. S., Moskowitz, T. J., and Pedersen, L. H. (2013). Value and momentum
everywhere. Journal of Finance, 68(3), 929-985.

Burges, C., Shaked, T., Renshaw, E., Lazier, A., Deeds, M., Hamilton, N., and
Hullender, G. (2005). Learning to Rank using Gradient Descent. Proceedings of
the 22nd International Conference on Machine Learning.

Burges, C. J. C., Ragno, R., and Le, Q. V. (2006). Learning to Rank with
Nonsmooth Cost Functions. Advances in Neural Information Processing Systems.

Cao, Z., Qin, T., Liu, T.-Y., Tsai, M.-F., and Li, H. (2007). Learning to Rank:
From Pairwise Approach to Listwise Approach. Proceedings of the 24th
International Conference on Machine Learning.

Fanshawe, J., Masih, R., and Cameron, A. (2026). Forecasting Equity
Correlations with Hybrid Transformer Graph Neural Network. arXiv:2601.04602.

Gao, S., Wang, Y., and Yang, X. (2023). StockFormer: Learning Hybrid Trading
Machines with Predictive Coding. Proceedings of the Thirty-Second International
Joint Conference on Artificial Intelligence.

Guidolin, M., and Timmermann, A. (2007). Asset allocation under multivariate
regime switching. Journal of Economic Dynamics and Control, 31(11), 3503-3544.

Hamilton, J. D. (1989). A New Approach to the Economic Analysis of Nonstationary
Time Series and the Business Cycle. Econometrica, 57(2), 357-384.

Anonymous Authors. (2026). Aligning News and Prices: A Cross-Modal
LLM-Enhanced Transformer DRL Framework for Volatility-Adaptive Stock Trading.
Under review.

Jegadeesh, N., and Titman, S. (1993). Returns to Buying Winners and Selling
Losers: Implications for Stock Market Efficiency. Journal of Finance, 48(1),
65-91.

Lim, B., Arik, S. O., Loeff, N., and Pfister, T. (2021). Temporal Fusion
Transformers for Interpretable Multi-horizon Time Series Forecasting.
International Journal of Forecasting, 37(4), 1748-1764.

Nie, Y., Nguyen, N. H., Sinthong, P., and Kalagnanam, J. (2023). A Time Series
is Worth 64 Words: Long-term Forecasting with Transformers. International
Conference on Learning Representations.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N.,
Kaiser, L., and Polosukhin, I. (2017). Attention Is All You Need. Advances in
Neural Information Processing Systems.

Wu, H., Xu, J., Wang, J., and Long, M. (2021). Autoformer: Decomposition
Transformers with Auto-Correlation for Long-Term Series Forecasting. Advances
in Neural Information Processing Systems.

Zhou, H., Zhang, S., Peng, J., Zhang, S., Li, J., Xiong, H., and Zhang, W.
(2021). Informer: Beyond Efficient Transformer for Long Sequence Time-Series
Forecasting. Proceedings of the AAAI Conference on Artificial Intelligence.

Zhang, Z., Zohren, S., and Roberts, S. (2019). DeepLOB: Deep Convolutional
Neural Networks for Limit Order Books. IEEE Transactions on Signal Processing,
67(11), 3001-3012.

---

## Source Links

- Attention Is All You Need: https://papers.nips.cc/paper/7181-attention-is-all-you-need
- Aligning News and Prices: local reference PDF, `/Users/vivekvishnoi/Downloads/9001_Aligning_News_and_Prices_.pdf`
- Forecasting Equity Correlations with Hybrid Transformer Graph Neural Network: https://arxiv.org/abs/2601.04602
- Temporal Fusion Transformer: https://arxiv.org/abs/1912.09363
- Informer: https://arxiv.org/abs/2012.07436
- Autoformer: https://papers.nips.cc/paper/2021/hash/bcc0d400288793e8bdcd7c19a8ac0c2b-Abstract.html
- PatchTST: https://arxiv.org/abs/2211.14730
- StockFormer: https://www.ijcai.org/proceedings/2023/0530

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.append(str(ROOT))

# ─── Page Config ────────────────────────────────────────────
st.set_page_config(
    page_title="RAMT Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ─────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f172a; }
    .stApp { background-color: #0f172a; }
    
    .metric-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    
    .bull-badge {
        background: #166534;
        color: #bbf7d0;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 14px;
    }
    
    .bear-badge {
        background: #7f1d1d;
        color: #fecaca;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 14px;
    }
    
    .highvol-badge {
        background: #7c2d12;
        color: #fed7aa;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 14px;
    }
    
    .buy-signal {
        background: #14532d;
        border: 1px solid #16a34a;
        border-radius: 8px;
        padding: 12px;
        color: #86efac;
        font-size: 18px;
        font-weight: bold;
        text-align: center;
    }
    
    .sell-signal {
        background: #450a0a;
        border: 1px solid #dc2626;
        border-radius: 8px;
        padding: 12px;
        color: #fca5a5;
        font-size: 18px;
        font-weight: bold;
        text-align: center;
    }
    
    .hold-signal {
        background: #1c1917;
        border: 1px solid #78716c;
        border-radius: 8px;
        padding: 12px;
        color: #d6d3d1;
        font-size: 18px;
        font-weight: bold;
        text-align: center;
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: bold;
        color: #f1f5f9;
    }
    
    div[data-testid="stMetricLabel"] {
        color: #94a3b8;
        font-size: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ─── Portfolio View ───────────────────────────────────────────


def show_portfolio_signals(regime_info, rankings: pd.DataFrame):
    """
    Main dashboard view:

    CURRENT REGIME: BULL
    POSITION SIZE: 100%

    THIS MONTH BUY LIST:
    1. TCS      score: 0.87  momentum: +24%
    2. INFY     score: 0.82  momentum: +19%
    3. HCLTECH  score: 0.75  momentum: +16%
    4. WIPRO    score: 0.71  momentum: +14%
    5. TECHM    score: 0.68  momentum: +12%

    AVOID (negative score):
    TATASTEEL  score: -0.45
    ONGC       score: -0.32
    """
    st.markdown("### 📌 Monthly Portfolio Signals")

    if not regime_info:
        st.warning("Regime info not available. Run feature engineering first.")
        return

    regime_name = regime_info["name"]
    if regime_info["current"] == 2:
        position_size = 0.0
        top_n = 0
    elif regime_info["current"] == 0:
        position_size = 0.5
        top_n = 3
    else:
        position_size = 1.0
        top_n = 5

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Current Regime", regime_name)
    with col_b:
        st.metric("Position Size", f"{position_size*100:.0f}%")

    if position_size == 0.0:
        st.info("BEAR regime → stay in cash this month.")
        return

    if rankings is None or rankings.empty:
        st.info(
            "No monthly rankings found yet. Once `results/monthly_rankings.csv` exists, "
            "this view will show the top stocks for the current month."
        )
        return

    expected_cols = {"Date", "Ticker", "score", "momentum"}
    if not expected_cols.issubset(set(rankings.columns)):
        st.warning(
            f"Rankings file is missing expected columns: {sorted(expected_cols)}. "
            f"Found: {sorted(rankings.columns)}"
        )
        return

    rankings = rankings.copy()
    rankings["Date"] = pd.to_datetime(rankings["Date"])
    current_month = rankings["Date"].max()
    month_df = rankings[rankings["Date"] == current_month].sort_values("score", ascending=False)

    st.markdown(f"#### This month buy list ({current_month.date()})")
    buy_df = month_df.head(top_n).copy()
    buy_df["momentum"] = (buy_df["momentum"] * 100).round(2)
    st.dataframe(
        buy_df[["Ticker", "score", "momentum"]].rename(columns={"momentum": "momentum_%"}),
        use_container_width=True,
        hide_index=True,
    )

    avoid_df = month_df[month_df["score"] < 0].head(10).copy()
    if not avoid_df.empty:
        st.markdown("#### Avoid (negative score)")
        st.dataframe(
            avoid_df[["Ticker", "score"]],
            use_container_width=True,
            hide_index=True,
        )


# ─── Constants ───────────────────────────────────────────────
TICKERS = {
    "TCS": "TCS_NS",
    "RELIANCE": "RELIANCE_NS", 
    "HDFC Bank": "HDFCBANK_NS",
    "EPIGRAL": "EPIGRAL_NS",
    "JPM (US)": "JPM"
}

REGIME_COLORS = {
    0: "#f97316",  # orange - high vol
    1: "#22c55e",  # green - bull
    2: "#ef4444"   # red - bear
}

REGIME_NAMES = {
    0: "HIGH VOL",
    1: "BULL",
    2: "BEAR"
}

# ─── Data Loaders ────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_features(ticker_code):
    """Load processed features CSV for a ticker."""
    path = ROOT / f"data/processed/{ticker_code}_features.csv"
    if not path.exists():
        return None
    # Your processed CSVs have a Date column; make it the index for plotting.
    df = pd.read_csv(path, parse_dates=["Date"])
    if "Date" in df.columns:
        df = df.sort_values("Date").set_index("Date")
    return df

@st.cache_data(ttl=300)
def load_predictions(model_name):
    """Load prediction CSV for a model."""
    path = ROOT / f"results/{model_name}_predictions.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=['Date'])
    return df

@st.cache_data(ttl=300)
def load_all_predictions():
    """Load all available prediction files."""
    models = {}
    for name in ['xgboost', 'lstm', 'ramt']:
        df = load_predictions(name)
        if df is not None:
            models[name.upper()] = df
    return models


@st.cache_data(ttl=300)
def load_ranking_predictions():
    """Walk-forward / fixed-train RAMT outputs (rebalance dates)."""
    path = ROOT / "results/ranking_predictions.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, parse_dates=["Date"])


@st.cache_data(ttl=300)
def load_monthly_rankings():
    path = ROOT / "results/monthly_rankings.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data(ttl=300)
def load_backtest_results():
    path = ROOT / "results/backtest_results.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["date"])
    return df.sort_values("date")

def compute_metrics(y_true, y_pred):
    """Compute trading metrics from predictions."""
    if len(y_true) == 0:
        return {}
    
    rmse = float(np.sqrt(np.mean((y_true - y_pred)**2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    da = float(np.mean(np.sign(y_true) == np.sign(y_pred)) * 100)
    
    rolling_std = pd.Series(y_pred).rolling(20).std()
    rolling_std = rolling_std.fillna(float(np.std(y_pred)))
    position = np.clip(y_pred / (rolling_std.values + 1e-8), -2, 2)
    strategy_ret = y_true * position
    
    sharpe = float(
        np.mean(strategy_ret) / 
        (np.std(strategy_ret) + 1e-8) * np.sqrt(252)
    )
    
    cumulative = np.cumprod(1 + strategy_ret)
    rolling_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - rolling_max) / (rolling_max + 1e-8)
    max_dd = float(drawdown.min())
    
    return {
        'RMSE': round(rmse, 4),
        'MAE': round(mae, 4),
        'DA%': round(da, 2),
        'Sharpe': round(sharpe, 2),
        'MaxDD': round(max_dd, 4)
    }

def get_current_regime(df):
    """Get current regime info from features DataFrame."""
    if df is None or 'HMM_Regime' not in df.columns:
        return None
    
    last_regime = int(df['HMM_Regime'].iloc[-1])
    
    # Count days in current regime
    regimes = df['HMM_Regime'].values
    days_in_regime = 1
    for i in range(len(regimes)-2, -1, -1):
        if int(regimes[i]) == last_regime:
            days_in_regime += 1
        else:
            break
    
    # Compute regime distribution last 60 days
    recent = df['HMM_Regime'].tail(60)
    total = len(recent)
    bull_pct = float((recent == 1).sum() / total * 100)
    bear_pct = float((recent == 2).sum() / total * 100)
    hv_pct = float((recent == 0).sum() / total * 100)
    
    return {
        'current': last_regime,
        'name': REGIME_NAMES[last_regime],
        'color': REGIME_COLORS[last_regime],
        'days': days_in_regime,
        'bull_pct': bull_pct,
        'bear_pct': bear_pct,
        'hv_pct': hv_pct
    }

def get_latest_signal(predictions_df, ticker_code):
    """Get the most recent prediction signal for a ticker."""
    if predictions_df is None:
        return None
    
    ticker_preds = predictions_df[
        predictions_df['Ticker'] == ticker_code
    ].copy()
    
    if ticker_preds.empty:
        return None
    
    ticker_preds = ticker_preds.sort_values('Date')
    last_row = ticker_preds.iloc[-1]
    
    pred = float(last_row['y_pred'])
    actual = float(last_row['y_true'])
    
    if pred > 0.003:
        signal = "BUY"
        signal_class = "buy-signal"
    elif pred < -0.003:
        signal = "SELL / AVOID"
        signal_class = "sell-signal"
    else:
        signal = "HOLD / WEAK"
        signal_class = "hold-signal"
    
    return {
        'prediction': pred,
        'prediction_pct': round(pred * 100, 3),
        'signal': signal,
        'signal_class': signal_class,
        'last_date': str(last_row['Date'])[:10]
    }

# ─── Charts ──────────────────────────────────────────────────

def plot_actual_vs_predicted(predictions_df, ticker_code, 
                              days=120):
    """Plot actual vs predicted returns."""
    if predictions_df is None:
        return None
    
    df = predictions_df[
        predictions_df['Ticker'] == ticker_code
    ].copy().sort_values('Date').tail(days)
    
    if df.empty:
        return None
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            f'Daily Returns — Actual vs Predicted ({days} days)',
            'Prediction Accuracy (Correct Direction = Green)'
        ),
        vertical_spacing=0.12,
        row_heights=[0.6, 0.4]
    )
    
    # Chart 1 — Actual returns bars
    fig.add_trace(
        go.Bar(
            x=df['Date'],
            y=df['y_true'] * 100,
            name='Actual Return %',
            marker_color=[
                '#22c55e' if v > 0 else '#ef4444' 
                for v in df['y_true']
            ],
            opacity=0.7
        ),
        row=1, col=1
    )
    
    # Predicted line
    fig.add_trace(
        go.Scatter(
            x=df['Date'],
            y=df['y_pred'] * 100,
            name='Predicted Return %',
            line=dict(color='#f97316', width=2),
            mode='lines'
        ),
        row=1, col=1
    )
    
    # Zero line
    fig.add_hline(
        y=0, line_dash="dash", 
        line_color="#475569", 
        line_width=1,
        row=1, col=1
    )
    
    # Chart 2 — Direction accuracy
    correct = (
        np.sign(df['y_true'].values) == 
        np.sign(df['y_pred'].values)
    ).astype(int)
    
    fig.add_trace(
        go.Bar(
            x=df['Date'],
            y=correct,
            name='Direction Correct',
            marker_color=[
                '#22c55e' if c == 1 else '#ef4444' 
                for c in correct
            ],
            showlegend=False
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        paper_bgcolor='#0f172a',
        plot_bgcolor='#1e293b',
        font=dict(color='#f1f5f9', size=12),
        legend=dict(
            bgcolor='#1e293b',
            bordercolor='#334155'
        ),
        height=500,
        margin=dict(t=50, b=20)
    )
    
    fig.update_xaxes(
        gridcolor='#334155', 
        showgrid=True
    )
    fig.update_yaxes(
        gridcolor='#334155', 
        showgrid=True
    )
    
    return fig


def plot_cumulative_returns(predictions_df, ticker_code):
    """Plot strategy vs buy-and-hold cumulative returns."""
    if predictions_df is None:
        return None
    
    df = predictions_df[
        predictions_df['Ticker'] == ticker_code
    ].copy().sort_values('Date')
    
    if df.empty:
        return None
    
    y_true = df['y_true'].values
    y_pred = df['y_pred'].values
    
    # Buy and hold
    bah = np.cumprod(1 + y_true) - 1
    
    # Strategy: long when predicted positive, cash when negative
    position = np.sign(y_pred)
    position[position == 0] = 0
    strategy_daily = y_true * position
    strategy = np.cumprod(1 + strategy_daily) - 1
    
    fig = go.Figure()
    
    # Strategy line
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=strategy * 100,
        name='RAMT Strategy',
        line=dict(color='#22c55e', width=2.5),
        fill='tozeroy',
        fillcolor='rgba(34, 197, 94, 0.1)'
    ))
    
    # Buy and hold line
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=bah * 100,
        name='Buy & Hold',
        line=dict(color='#3b82f6', width=2, dash='dash'),
    ))
    
    # Zero line
    fig.add_hline(
        y=0, line_dash="dash",
        line_color="#475569", line_width=1
    )
    
    fig.update_layout(
        title='Cumulative Returns — Strategy vs Buy & Hold (%)',
        paper_bgcolor='#0f172a',
        plot_bgcolor='#1e293b',
        font=dict(color='#f1f5f9', size=12),
        legend=dict(bgcolor='#1e293b', bordercolor='#334155'),
        height=350,
        margin=dict(t=50, b=20),
        yaxis_title='Cumulative Return (%)',
        xaxis_title='Date'
    )
    
    fig.update_xaxes(gridcolor='#334155')
    fig.update_yaxes(gridcolor='#334155')
    
    return fig


def plot_regime_history(features_df, days=180):
    """Plot regime history with colored background."""
    if features_df is None:
        return None
    
    df = features_df.tail(days).copy()
    
    if 'HMM_Regime' not in df.columns:
        return None
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            'Price with Regime Background',
            'Regime Timeline'
        ),
        vertical_spacing=0.12,
        row_heights=[0.65, 0.35]
    )
    
    # Price line
    if 'Close' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['Close'],
                name='Close Price',
                line=dict(color='#f1f5f9', width=1.5)
            ),
            row=1, col=1
        )
    
    # Regime colored bars
    regime_colors_map = {
        0: 'rgba(249, 115, 22, 0.3)',
        1: 'rgba(34, 197, 94, 0.3)',
        2: 'rgba(239, 68, 68, 0.3)'
    }
    
    # Add regime shading
    prev_regime = None
    start_date = None
    
    for date, row in df.iterrows():
        regime = int(row['HMM_Regime'])
        if regime != prev_regime:
            if prev_regime is not None and start_date is not None:
                fig.add_vrect(
                    x0=start_date, x1=date,
                    fillcolor=regime_colors_map[prev_regime],
                    layer="below", line_width=0,
                    row=1, col=1
                )
            start_date = date
            prev_regime = regime
    
    # Regime bar chart
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=[1] * len(df),
            marker_color=[
                REGIME_COLORS[int(r)] 
                for r in df['HMM_Regime']
            ],
            name='Regime',
            showlegend=False
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        paper_bgcolor='#0f172a',
        plot_bgcolor='#1e293b',
        font=dict(color='#f1f5f9', size=12),
        height=480,
        margin=dict(t=50, b=20),
        legend=dict(bgcolor='#1e293b')
    )
    
    fig.update_xaxes(gridcolor='#334155')
    fig.update_yaxes(gridcolor='#334155')
    
    return fig


def _ranking_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """Scalar fit stats for ranking CSV (alpha as decimal returns)."""
    if len(actual) < 2:
        return {}
    rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
    mae = float(np.mean(np.abs(actual - predicted)))
    da = float(np.mean(np.sign(actual) == np.sign(predicted)) * 100)
    corr = float(np.corrcoef(actual, predicted)[0, 1])
    return {
        "RMSE": round(rmse, 5),
        "MAE": round(mae, 5),
        "DA%": round(da, 2),
        "ρ": round(corr, 3),
        "n": int(len(actual)),
    }


def plot_ramt_ranking_predicted_vs_actual(
    rank_df: pd.DataFrame, ticker_code: str
) -> tuple[go.Figure | None, dict]:
    """
    Professional validation view: time-aligned series + calibration scatter.
    Expects columns Date, Ticker, predicted_alpha, actual_alpha.
    """
    if rank_df is None or rank_df.empty:
        return None, {}
    need = {"Date", "predicted_alpha", "actual_alpha", "Ticker"}
    if not need.issubset(rank_df.columns):
        return None, {}

    d = rank_df[rank_df["Ticker"] == ticker_code].copy().sort_values("Date")
    if d.empty:
        return None, {}

    act = d["actual_alpha"].to_numpy(dtype=float)
    pred = d["predicted_alpha"].to_numpy(dtype=float)
    metrics = _ranking_metrics(act, pred)

    act_pct = act * 100.0
    pred_pct = pred * 100.0
    dates = d["Date"]

    # Institutional palette (distinct, print-safe)
    c_realized = "#1e40af"
    c_realized_mk = "#3b82f6"
    c_predicted = "#0f766e"
    c_predicted_mk = "#14b8a6"
    c_ref = "#78716c"
    c_scatter = "#0d9488"
    c_scatter_edge = "#ccfbf1"
    grid = "#334155"
    zero_ln = "#57534e"

    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.54, 0.46],
        vertical_spacing=0.20,
        subplot_titles=(
            f"<b>{ticker_code}</b> · Realized vs predicted α (rebalance dates)",
            "<b>Calibration</b> · predicted vs realized α",
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=act_pct,
            name="Realized α",
            mode="lines+markers",
            line=dict(color=c_realized, width=2.25),
            marker=dict(size=8, color=c_realized_mk, line=dict(width=0)),
            hovertemplate="%{x|%Y-%m-%d}<br>Realized: %{y:.3f}%<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=pred_pct,
            name="Predicted α",
            mode="lines+markers",
            line=dict(color=c_predicted, width=2.25),
            marker=dict(size=8, color=c_predicted_mk, line=dict(width=0)),
            hovertemplate="%{x|%Y-%m-%d}<br>Predicted: %{y:.3f}%<extra></extra>",
        ),
        row=1,
        col=1,
    )

    lo = float(min(act_pct.min(), pred_pct.min()))
    hi = float(max(act_pct.max(), pred_pct.max()))
    span = hi - lo
    pad = max(span * 0.12, 0.25)
    lim_lo, lim_hi = lo - pad, hi + pad

    fig.add_trace(
        go.Scatter(
            x=[lim_lo, lim_hi],
            y=[lim_lo, lim_hi],
            mode="lines",
            name="Perfect fit (y = x)",
            line=dict(color=c_ref, dash="dash", width=1.25),
            hoverinfo="skip",
        ),
        row=2,
        col=1,
    )
    date_str = dates.dt.strftime("%Y-%m-%d")
    fig.add_trace(
        go.Scatter(
            x=act_pct,
            y=pred_pct,
            mode="markers",
            name="Rebalance observations",
            marker=dict(
                size=9,
                color=c_scatter,
                line=dict(width=1.25, color=c_scatter_edge),
                opacity=0.92,
            ),
            text=date_str,
            hovertemplate=(
                "Realized: %{x:.3f}%<br>Predicted: %{y:.3f}%<br>%{text}<extra></extra>"
            ),
        ),
        row=2,
        col=1,
    )

    fig.update_xaxes(
        title_text="Date",
        title_standoff=18,
        row=1,
        col=1,
        gridcolor=grid,
        showgrid=True,
        zeroline=False,
        tickangle=-35,
        automargin=True,
        tickformat="%b %Y",
        nticks=min(18, max(6, len(dates))),
    )
    fig.update_yaxes(
        title_text="Alpha (%)",
        title_standoff=14,
        row=1,
        col=1,
        gridcolor=grid,
        showgrid=True,
        zeroline=True,
        zerolinecolor=zero_ln,
        automargin=True,
    )
    fig.update_xaxes(
        title_text="Realized α (%)",
        title_standoff=16,
        row=2,
        col=1,
        gridcolor=grid,
        range=[lim_lo, lim_hi],
        showgrid=True,
        zeroline=False,
        automargin=True,
    )
    fig.update_yaxes(
        title_text="Predicted α (%)",
        title_standoff=14,
        row=2,
        col=1,
        gridcolor=grid,
        range=[lim_lo, lim_hi],
        showgrid=True,
        zeroline=False,
        automargin=True,
    )

    fig.update_layout(
        height=760,
        paper_bgcolor="#0f172a",
        plot_bgcolor="#1e293b",
        font=dict(color="#e2e8f0", size=12, family="Inter, ui-sans-serif, system-ui, sans-serif"),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.06,
            x=0.5,
            xanchor="center",
            bgcolor="rgba(15,23,42,0.94)",
            bordercolor="#475569",
            borderwidth=1,
            font=dict(size=11, color="#e2e8f0"),
            tracegroupgap=24,
        ),
        margin=dict(l=64, r=40, t=96, b=112),
    )

    for ann in fig.layout.annotations:
        ann.font.size = 13
        ann.font.color = "#f1f5f9"
        ann.yshift = -8

    return fig, metrics


def plot_metrics_comparison(all_predictions, ticker_code):
    """Bar chart comparing models on key metrics."""
    if not all_predictions:
        return None
    
    models_data = []
    for model_name, pred_df in all_predictions.items():
        ticker_df = pred_df[
            pred_df['Ticker'] == ticker_code
        ]
        if ticker_df.empty:
            continue
        
        metrics = compute_metrics(
            ticker_df['y_true'].values,
            ticker_df['y_pred'].values
        )
        metrics['Model'] = model_name
        models_data.append(metrics)
    
    if not models_data:
        return None
    
    metrics_df = pd.DataFrame(models_data)
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            'Directional Accuracy (%)',
            'Sharpe Ratio'
        )
    )
    
    colors = ['#3b82f6', '#22c55e', '#f97316']
    
    fig.add_trace(
        go.Bar(
            x=metrics_df['Model'],
            y=metrics_df['DA%'],
            marker_color=colors[:len(metrics_df)],
            name='DA%',
            text=metrics_df['DA%'].round(2),
            textposition='outside',
            textfont=dict(color='#f1f5f9')
        ),
        row=1, col=1
    )
    
    # Reference line at 50%
    fig.add_hline(
        y=50, line_dash="dash",
        line_color="#ef4444", line_width=1,
        annotation_text="Random (50%)",
        annotation_font_color="#ef4444",
        row=1, col=1
    )
    
    fig.add_trace(
        go.Bar(
            x=metrics_df['Model'],
            y=metrics_df['Sharpe'],
            marker_color=colors[:len(metrics_df)],
            name='Sharpe',
            text=metrics_df['Sharpe'].round(2),
            textposition='outside',
            textfont=dict(color='#f1f5f9')
        ),
        row=1, col=2
    )
    
    fig.update_layout(
        paper_bgcolor='#0f172a',
        plot_bgcolor='#1e293b',
        font=dict(color='#f1f5f9'),
        height=300,
        showlegend=False,
        margin=dict(t=40, b=20)
    )
    
    fig.update_xaxes(gridcolor='#334155')
    fig.update_yaxes(gridcolor='#334155')
    
    return fig


# ─── Main App ────────────────────────────────────────────────

def main():
    
    # ── Sidebar ──────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 📈 RAMT Dashboard")
        st.markdown("---")
        
        # Ticker selector
        selected_display = st.selectbox(
            "Select Stock",
            list(TICKERS.keys()),
            index=0
        )
        ticker_code = TICKERS[selected_display]
        
        st.markdown("---")
        
        # Model selector
        available_models = []
        for name in ['xgboost', 'lstm', 'ramt']:
            path = ROOT / f"results/{name}_predictions.csv"
            if path.exists():
                available_models.append(name.upper())
        
        if available_models:
            selected_model = st.selectbox(
                "Primary Model",
                available_models,
                index=0
            )
        else:
            selected_model = "XGBOOST"
            st.warning("No prediction files found")
        
        st.markdown("---")
        
        # Days selector
        days = st.slider(
            "Chart History (days)",
            min_value=30,
            max_value=365,
            value=120,
            step=30
        )
        
        st.markdown("---")
        st.markdown("### Project Info")
        st.markdown("**RAMT** — Regime-Adaptive")
        st.markdown("Multimodal Transformer")
        st.markdown("Rishihood University")
        
        st.markdown("---")
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()
    
    # ── Load Data ─────────────────────────────────────────────
    features_df = load_features(ticker_code)
    regime_info = get_current_regime(features_df)
    rankings_df = load_monthly_rankings()
    bt_df = load_backtest_results()
    ranking_preds_df = load_ranking_predictions()
    all_predictions = load_all_predictions()
    primary_preds = load_predictions(selected_model.lower())

    # ── Header ────────────────────────────────────────────────
    st.markdown("# 📈 RAMT Portfolio Dashboard")
    st.markdown("---")

    show_portfolio_signals(regime_info, rankings_df)

    st.markdown("---")
    st.markdown("### 📈 Portfolio vs NIFTY (paper trading)")
    if bt_df is None or bt_df.empty:
        st.info("No backtest results yet. Run `python models/run_final_2024_2026.py` first.")
    else:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=bt_df["date"],
                y=bt_df.get("portfolio_value", (1 + bt_df["cumulative_return"]) * 100000),
                name="Portfolio value (₹)",
                line=dict(color="#22c55e", width=2.5),
            )
        )
        if "nifty_value" in bt_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=bt_df["date"],
                    y=bt_df["nifty_value"],
                    name="NIFTY value (₹)",
                    line=dict(color="#3b82f6", width=2, dash="dash"),
                )
            )
        fig.update_layout(
            paper_bgcolor="#0f172a",
            plot_bgcolor="#1e293b",
            font=dict(color="#f1f5f9"),
            height=380,
            margin=dict(t=30, b=20, l=10, r=10),
            yaxis_title="Value (₹)",
            xaxis_title="Month",
            legend=dict(bgcolor="#1e293b", bordercolor="#334155"),
        )
        fig.update_xaxes(gridcolor="#334155")
        fig.update_yaxes(gridcolor="#334155")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 🧾 Trade history")
        trade_cols = ["date", "regime", "portfolio_return", "stocks_held", "cash"]
        show_cols = [c for c in trade_cols if c in bt_df.columns]
        st.dataframe(bt_df[show_cols].tail(24), use_container_width=True, hide_index=True)
    
    st.markdown("---")
    st.markdown("### RAMT predicted vs actual (out-of-sample)")
    st.caption(
        "Uses `results/ranking_predictions.csv` from walk-forward or fixed-train RAMT. "
        "Alpha is shown as percent per rebalance date."
    )
    fig_ramt, ramt_m = plot_ramt_ranking_predicted_vs_actual(
        ranking_preds_df, ticker_code
    )
    if fig_ramt is None:
        st.info(
            "No ranking predictions for this ticker yet. Run training "
            "(`python models/run_final_2024_2026.py` or `models/ramt/train_ranking.py`) "
            "so `results/ranking_predictions.csv` exists and includes this symbol."
        )
    else:
        mcols = st.columns(5)
        mcols[0].metric("Direction accuracy", f"{ramt_m.get('DA%', 0):.2f}%")
        mcols[1].metric("RMSE", f"{ramt_m.get('RMSE', 0):.5f}")
        mcols[2].metric("MAE", f"{ramt_m.get('MAE', 0):.5f}")
        mcols[3].metric("Correlation ρ", f"{ramt_m.get('ρ', 0):.3f}")
        mcols[4].metric("Observations", f"{ramt_m.get('n', 0)}")
        st.plotly_chart(fig_ramt, use_container_width=True)

    st.markdown("---")
    
    # ── Row 6: Model Comparison + Live Predictor ──────────────
    col_metrics, col_predictor = st.columns([1, 1])
    
    with col_metrics:
        st.markdown("### 🏆 Model Comparison")
        
        fig_comp = plot_metrics_comparison(
            all_predictions, ticker_code
        )
        if fig_comp:
            st.plotly_chart(fig_comp, use_container_width=True)
        
        # Metrics table
        if all_predictions:
            rows = []
            for model_name, pred_df in all_predictions.items():
                t_df = pred_df[
                    pred_df['Ticker'] == ticker_code
                ]
                if t_df.empty:
                    continue
                m = compute_metrics(
                    t_df['y_true'].values,
                    t_df['y_pred'].values
                )
                m['Model'] = model_name
                rows.append(m)
            
            if rows:
                metrics_table = pd.DataFrame(rows)
                metrics_table = metrics_table.set_index('Model')
                
                st.dataframe(
                    metrics_table.style
                    .format({
                        'RMSE': '{:.4f}',
                        'MAE': '{:.4f}',
                        'DA%': '{:.2f}%',
                        'Sharpe': '{:.2f}',
                        'MaxDD': '{:.4f}'
                    })
                    .background_gradient(
                        subset=['DA%'],
                        cmap='RdYlGn',
                        vmin=49,
                        vmax=56
                    )
                    .background_gradient(
                        subset=['Sharpe'],
                        cmap='RdYlGn',
                        vmin=-1,
                        vmax=2
                    ),
                    use_container_width=True
                )
    
    with col_predictor:
        st.markdown("### 🔮 Live Predictor")
        st.markdown(
            "Input today's values to get tomorrow's prediction"
        )
        
        # Auto-fill button
        if st.button("📥 Auto-fill Latest Data"):
            if features_df is not None:
                last_row = features_df.iloc[-1]
                st.session_state['return_lag1'] = float(
                    last_row.get('Return_Lag_1', 0)
                )
                st.session_state['rsi'] = float(
                    last_row.get('RSI_14', 50)
                )
                st.session_state['macd'] = float(
                    last_row.get('MACD', 0)
                )
                st.session_state['vol_ratio'] = float(
                    last_row.get('Vol_Ratio', 1)
                )
                st.session_state['regime'] = int(
                    last_row.get('HMM_Regime', 1)
                )
                st.success(
                    f"Filled with data from "
                    f"{features_df.index[-1]}"
                )
        
        with st.form("prediction_form"):
            c1, c2 = st.columns(2)
            
            with c1:
                return_lag1 = st.number_input(
                    "Today's Return (%)",
                    value=st.session_state.get(
                        'return_lag1', 0.0
                    ) * 100,
                    step=0.1,
                    format="%.3f"
                ) / 100
                
                rsi = st.number_input(
                    "RSI (14-day)",
                    min_value=0.0,
                    max_value=100.0,
                    value=float(
                        st.session_state.get('rsi', 50.0)
                    ),
                    step=1.0
                )
                
                vol_ratio = st.number_input(
                    "Vol Ratio (5d/20d)",
                    min_value=0.0,
                    max_value=5.0,
                    value=float(
                        st.session_state.get('vol_ratio', 1.0)
                    ),
                    step=0.1
                )
            
            with c2:
                macd = st.number_input(
                    "MACD",
                    value=float(
                        st.session_state.get('macd', 0.0)
                    ),
                    step=0.01,
                    format="%.4f"
                )
                
                regime = st.selectbox(
                    "HMM Regime",
                    options=[0, 1, 2],
                    format_func=lambda x: REGIME_NAMES[x],
                    index=int(
                        st.session_state.get('regime', 1)
                    )
                )
                
                momentum_20 = st.number_input(
                    "20-day Momentum (%)",
                    value=0.0,
                    step=0.5,
                    format="%.2f"
                ) / 100
            
            submitted = st.form_submit_button(
                "🚀 Get Prediction",
                use_container_width=True
            )
        
        if submitted:
            # Simple heuristic prediction
            # Based on feature importance from XGBoost results
            score = (
                momentum_20 * 0.25 +
                return_lag1 * 0.20 +
                macd * 0.15 +
                (rsi - 50) / 100 * 0.10 +
                (1 - vol_ratio) * 0.10
            )
            
            # Regime adjustment
            if regime == 1:  # Bull
                score *= 1.2
            elif regime == 2:  # Bear
                score *= 0.6
            else:  # High vol
                score *= 0.4
            
            pred_pct = score * 100
            
            if pred_pct > 0.3:
                signal = "🟢 BUY"
                box_class = "buy-signal"
                action = f"Consider buying {selected_display}"
            elif pred_pct < -0.3:
                signal = "🔴 SELL / AVOID"
                box_class = "sell-signal"
                action = f"Avoid or reduce {selected_display}"
            else:
                signal = "🟡 HOLD / WEAK"
                box_class = "hold-signal"
                action = "Signal too weak to act"
            
            st.markdown(
                f"<div class='{box_class}'>"
                f"{signal}<br>"
                f"<span style='font-size:14px;font-weight:normal'>"
                f"Predicted return: {pred_pct:+.3f}%<br>"
                f"Regime: {REGIME_NAMES[regime]}<br>"
                f"Action: {action}<br>"
                f"Stop loss if buying: 7% below entry"
                f"</span></div>",
                unsafe_allow_html=True
            )
            
            # Warning
            st.warning(
                "⚠️ This prediction uses a heuristic model. "
                "Always do your own research before trading."
            )
    
    st.markdown("---")
    
    # ── Row 7: Raw Data Table ─────────────────────────────────
    with st.expander("📊 View Raw Prediction Data"):
        if primary_preds is not None and {"y_true", "y_pred"}.issubset(
            primary_preds.columns
        ):
            t_df = primary_preds[
                primary_preds['Ticker'] == ticker_code
            ].tail(50).sort_values('Date', ascending=False)
            
            t_df['Direction_Correct'] = (
                np.sign(t_df['y_true']) == 
                np.sign(t_df['y_pred'])
            )
            t_df['y_true_pct'] = (t_df['y_true'] * 100).round(3)
            t_df['y_pred_pct'] = (t_df['y_pred'] * 100).round(3)
            
            display_cols = [
                'Date', 'y_true_pct', 
                'y_pred_pct', 'Direction_Correct'
            ]
            
            st.dataframe(
                t_df[display_cols].rename(columns={
                    'y_true_pct': 'Actual (%)',
                    'y_pred_pct': 'Predicted (%)',
                    'Direction_Correct': 'Correct?'
                }),
                use_container_width=True,
                height=300
            )
        elif ranking_preds_df is not None and not ranking_preds_df.empty:
            t_df = ranking_preds_df[
                ranking_preds_df["Ticker"] == ticker_code
            ].tail(50).sort_values("Date", ascending=False)
            if t_df.empty:
                st.info("No ranking rows for this ticker.")
            else:
                t_df = t_df.copy()
                t_df["Direction_Correct"] = (
                    np.sign(t_df["actual_alpha"])
                    == np.sign(t_df["predicted_alpha"])
                )
                t_df["actual_pct"] = (t_df["actual_alpha"] * 100).round(3)
                t_df["pred_pct"] = (t_df["predicted_alpha"] * 100).round(3)
                st.dataframe(
                    t_df[
                        ["Date", "actual_pct", "pred_pct", "Direction_Correct"]
                    ].rename(
                        columns={
                            "actual_pct": "Actual (%)",
                            "pred_pct": "Predicted (%)",
                            "Direction_Correct": "Correct?",
                        }
                    ),
                    use_container_width=True,
                    height=300,
                )
        else:
            st.info(
                "No `results/*_predictions.csv` or ranking predictions file found."
            )
    
    # ── Footer ────────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#475569;'>"
        "RAMT — Regime-Adaptive Multimodal Transformer | "
        "Rishihood University | "
        "Shivansh Gupta & Vivek Vishnoi | "
        "⚠️ For research purposes only — not financial advice"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()


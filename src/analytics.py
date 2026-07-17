"""
analytics.py
Plotly figure builders for model evaluation and data-quality views.
Kept separate from app.py so the dashboard layer stays thin and the
chart logic is unit-testable and reusable.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Signal palette shared with the dashboard
C_DIST = "#38bdf8"   # signal blue   – distance
C_TIME = "#34d399"   # signal green  – time
C_BAL  = "#fbbf24"   # signal amber  – balanced
C_RED  = "#f87171"
GRID   = "rgba(148,163,184,0.15)"
MODEL_COLORS = {"XGBoost": C_DIST, "LightGBM": C_TIME, "CatBoost": C_BAL}

_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, system-ui, sans-serif", color="#cbd5e1", size=13),
    margin=dict(t=50, b=40, l=50, r=20),
)


def _apply(fig, height=360, title=None):
    fig.update_layout(**_LAYOUT, height=height)
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    if title:
        fig.update_layout(title=dict(text=title, font=dict(size=15)))
    return fig


def metric_bars(results_df, metric, lower_is_better=True):
    """Bar chart of one metric across models."""
    df = results_df.copy()
    order = df.sort_values(metric, ascending=lower_is_better)["model"].tolist()
    fig = px.bar(df, x="model", y=metric, color="model",
                 category_orders={"model": order},
                 color_discrete_map=MODEL_COLORS, text_auto=".3f")
    fig.update_traces(textposition="outside",
                      hovertemplate="%{x}<br>%{y:.4f}<extra></extra>")
    fig.update_layout(showlegend=False)
    direction = "lower is better" if lower_is_better else "higher is better"
    return _apply(fig, title=f"{metric} ({direction})")


def pred_vs_actual(y_true, y_pred, model_name):
    """Scatter of predictions vs actuals with the ideal y=x reference."""
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=y_true, y=y_pred, mode="markers",
        marker=dict(size=4, color=C_DIST, opacity=0.35),
        name="predictions",
        hovertemplate="actual %{x:.1f}<br>pred %{y:.1f}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi], mode="lines",
        line=dict(color=C_RED, dash="dash", width=2), name="ideal (y = x)"))
    fig.update_layout(xaxis_title="Actual circuit speed (km/h)",
                      yaxis_title="Predicted circuit speed (km/h)",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02))
    return _apply(fig, height=420, title=f"Predicted vs Actual — {model_name}")


def residual_scatter(y_true, y_pred, model_name):
    """Residuals (actual − predicted) against predicted value."""
    resid = y_true - y_pred
    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=y_pred, y=resid, mode="markers",
        marker=dict(size=4, color=C_TIME, opacity=0.35),
        hovertemplate="pred %{x:.1f}<br>resid %{y:.1f}<extra></extra>"))
    fig.add_hline(y=0, line=dict(color=C_RED, dash="dash", width=2))
    fig.update_layout(xaxis_title="Predicted circuit speed (km/h)",
                      yaxis_title="Residual (actual − predicted)")
    return _apply(fig, height=380, title=f"Residuals vs Predicted — {model_name}")


def residual_hist(y_true, y_pred, model_name):
    """Distribution of residuals — should be centred near zero."""
    resid = y_true - y_pred
    fig = px.histogram(x=resid, nbins=60, color_discrete_sequence=[C_BAL])
    fig.add_vline(x=0, line=dict(color=C_RED, dash="dash", width=2))
    fig.update_layout(xaxis_title="Residual (actual − predicted)",
                      yaxis_title="Count", showlegend=False)
    return _apply(fig, height=380, title=f"Residual Distribution — {model_name}")


def box_with_outliers(series, label):
    """Box plot exposing IQR fences and outliers for one column."""
    fig = go.Figure()
    fig.add_trace(go.Box(y=series, name=label, boxpoints="outliers",
                         marker=dict(color=C_DIST, outliercolor=C_RED),
                         line=dict(color=C_DIST)))
    fig.update_layout(showlegend=False, yaxis_title=label)
    return _apply(fig, height=360, title=f"Outlier Spread — {label}")


def outlier_pct_bar(report_df):
    """Share of IQR outliers per screened column."""
    fig = px.bar(report_df, x="column", y="pct_outliers",
                 color="pct_outliers", color_continuous_scale="Tealrose",
                 text_auto=".2f")
    fig.update_traces(textposition="outside",
                      hovertemplate="%{x}<br>%{y:.2f}%% outliers<extra></extra>")
    fig.update_layout(xaxis_title=None, yaxis_title="% outliers",
                      coloraxis_showscale=False)
    return _apply(fig, title="IQR Outliers by Column")


def hop_distance_bar(hop_df):
    """Per-hop distance, coloured by ML-predicted speed."""
    fig = px.bar(hop_df.head(30), x="From Code", y="KM",
                 color="ML Speed (km/h)", color_continuous_scale="Blues")
    return _apply(fig, height=340, title="Distance per Hop (km)")

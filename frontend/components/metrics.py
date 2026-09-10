# frontend/components/metrics.py
import streamlit as st
import numpy as np
import pandas as pd

PERCENTILE_SPECS = [
    ("0.5", "P50", "Median"),
    ("0.75", "P75", "75th percentile"),
    ("0.975", "P95", "95% PI upper"),
    ("0.995", "P99", "99% PI upper"),
]


def _series(data, key, n=28):
    values = np.array(data.get(key, [0] * n), dtype=float)
    if values.size < n:
        values = np.pad(values, (0, n - values.size))
    return values[:n]


def _fmt_units(value: float) -> str:
    if np.isnan(value):
        return "—"
    if abs(value) >= 100:
        return f"{value:,.0f}"
    return f"{value:,.1f}"


def render_kpi_cards(h_sales, bt_data, f_data):
    actuals_tail = h_sales[-28:]
    bt_median = _series(bt_data, "0.5")

    mae = np.mean(np.abs(actuals_tail - bt_median))

    pi_95_upper = _series(f_data, "0.975")
    pi_95_lower = _series(f_data, "0.025")
    uncertainty_width = np.mean(pi_95_upper - pi_95_lower)

    bt_95_upper = _series(bt_data, "0.975")
    bt_95_lower = _series(bt_data, "0.025")
    within_bounds = np.sum((actuals_tail >= bt_95_lower) & (actuals_tail <= bt_95_upper))
    capture_rate = (within_bounds / 28) * 100

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Capture Rate (95% PI)", f"{round(capture_rate, 1)}%", help="Calibration Check: Should be near 95%")
    m2.metric("Backtest MAE", f"{round(mae, 2)}", delta_color="inverse", help="Mean Absolute Error of the Median prediction")
    m3.metric("Uncertainty Width", f"{round(uncertainty_width, 1)}", help="Average width of 95% Forecast Ribbon")
    m4.metric("Avg History", f"{round(np.mean(h_sales), 2)} units")

    return actuals_tail, bt_median, pi_95_upper, pi_95_lower


def render_percentile_panel(f_data):
    """Inventory-oriented quantile summary: next day, week 1, and 28-day totals."""
    st.subheader("Forecast percentiles")
    st.caption(
        "Service-level views of the next 28 days. P95 / P99 use the upper edge of the 95% and 99% prediction intervals."
    )

    totals_28 = {}
    for key, short, _help in PERCENTILE_SPECS:
        totals_28[short] = float(np.sum(_series(f_data, key)))

    cards = st.columns(4)
    for col, (key, short, help_text) in zip(cards, PERCENTILE_SPECS):
        col.metric(short, _fmt_units(totals_28[short]), help=f"{help_text} · 28-day total units")

    p50 = totals_28["P50"]
    p95 = totals_28["P95"]
    safety_stock = max(p95 - p50, 0.0)
    st.caption(
        f"Safety stock at a 95% service level: **{_fmt_units(safety_stock)} units** over 28 days (P95 − P50)."
    )

    rows = []
    horizons = [("Next day", slice(0, 1)), ("7-day total", slice(0, 7)), ("28-day total", slice(0, 28))]
    for label, indexer in horizons:
        row = {"Horizon": label}
        for key, short, _help in PERCENTILE_SPECS:
            row[short] = round(float(np.sum(_series(f_data, key)[indexer])), 1)
        rows.append(row)

    table = pd.DataFrame(rows).set_index("Horizon")
    formatted = table.copy()
    for col in formatted.columns:
        formatted[col] = formatted[col].map(lambda x: f"{x:,.1f}")
    st.table(formatted)

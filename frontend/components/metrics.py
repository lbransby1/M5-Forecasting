# frontend/components/metrics.py
import streamlit as st
import numpy as np

PERCENTILE_SPECS = [
    ("0.5", "P50", "Median · 28-day total"),
    ("0.75", "P75", "75th percentile · 28-day total"),
    ("0.975", "P95", "95% PI upper · 28-day total"),
    ("0.995", "P99", "99% PI upper · 28-day total"),
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
    m1.metric("Capture Rate", f"{round(capture_rate, 1)}%", help="Share of backtest days inside the 95% PI")
    m2.metric("Backtest MAE", f"{round(mae, 2)}", help="Mean absolute error of the median prediction")
    m3.metric("95% PI Width", f"{round(uncertainty_width, 1)}", help="Average daily width of the 95% forecast band")
    m4.metric("Avg History", f"{round(float(np.mean(h_sales)), 2)}")

    return actuals_tail, bt_median, pi_95_upper, pi_95_lower


def render_percentile_panel(f_data):
    totals_28 = {short: float(np.sum(_series(f_data, key))) for key, short, _ in PERCENTILE_SPECS}
    cards = st.columns(4)
    for col, (key, short, help_text) in zip(cards, PERCENTILE_SPECS):
        col.metric(short, _fmt_units(totals_28[short]), help=help_text)

    p50, p95 = totals_28["P50"], totals_28["P95"]
    safety_stock = max(p95 - p50, 0.0)
    next_p50 = float(_series(f_data, "0.5")[0])
    next_p95 = float(_series(f_data, "0.975")[0])
    week_p50 = float(np.sum(_series(f_data, "0.5")[:7]))
    week_p95 = float(np.sum(_series(f_data, "0.975")[:7]))
    st.caption(
        f"Safety stock (P95 − P50): **{_fmt_units(safety_stock)}** over 28 days"
        f" · Next day P50/P95 **{_fmt_units(next_p50)} / {_fmt_units(next_p95)}**"
        f" · 7-day P50/P95 **{_fmt_units(week_p50)} / {_fmt_units(week_p95)}**"
    )

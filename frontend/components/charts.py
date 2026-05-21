# frontend/components/charts.py
import streamlit as st
import plotly.graph_objects as go
import numpy as np

def plot_probabilistic_forecast(h_sales, bt_data, f_data, is_cumulative=False):
    history_len = len(h_sales)
    h_days = list(range(-history_len + 1, 1))
    bt_days = list(range(-27, 1))
    f_days = list(range(1, 29))

    # --- DATA TRANSFORMATION ---
    plot_h = np.array(h_sales)
    plot_bt = {k: np.array(v) for k, v in bt_data.items()}
    plot_f = {k: np.array(v) for k, v in f_data.items()}
    
    yaxis_title = "Units Sold (Daily)"

    if is_cumulative:
        yaxis_title = "Cumulative Units Sold"
        plot_h = np.cumsum(plot_h)
        
        # Anchor the backtest to the cumulative total of history right before it starts (Day -28)
        bt_anchor = plot_h[-29]
        plot_bt = {k: bt_anchor + np.cumsum(v) for k, v in bt_data.items()}
        
        # Anchor the forecast to the final cumulative total of history (Day 0)
        f_anchor = plot_h[-1]
        plot_f = {k: f_anchor + np.cumsum(v) for k, v in f_data.items()}

    # --- PLOTTING ---
    fig = go.Figure()

    def add_fan(figure, d, days, color_base, name_prefix):
        intervals = [('0.005', '0.995', 0.1, '99% PI'), ('0.025', '0.975', 0.2, '95% PI'),
                     ('0.165', '0.835', 0.3, '67% PI'), ('0.25', '0.75', 0.4, '50% PI')]
        for low, high, alpha, label in intervals:
            if low in d and high in d:
                figure.add_trace(go.Scatter(
                    x=days + days[::-1], y=list(d[high]) + list(d[low])[::-1],
                    fill='toself', fillcolor=f'rgba({color_base}, {alpha})',
                    line=dict(color='rgba(255,255,255,0)'), name=f"{name_prefix} {label}", hoverinfo="skip"
                ))
        if '0.5' in d:
            figure.add_trace(go.Scatter(x=days, y=d['0.5'], name=f"{name_prefix} Median", 
                                      line=dict(color=f'rgb({color_base})', width=3)))

    # Use a solid line for cumulative history, but keep the bar chart for daily noise
    if is_cumulative:
        fig.add_trace(go.Scatter(x=h_days, y=plot_h, mode='lines', name="Actual Sales (Cumulative)", line=dict(color='black', width=2)))
    else:
        fig.add_trace(go.Bar(x=h_days, y=plot_h, name="Actual Sales", marker_color='rgba(150,150,150,0.3)'))

    add_fan(fig, plot_bt, bt_days, "220, 50, 50", "Backtest")
    add_fan(fig, plot_f, f_days, "0, 100, 255", "Future")

    fig.update_layout(
        height=600, template="plotly_white", hovermode="x unified",
        xaxis=dict(title="Timeline (Days)", range=[-history_len, 30]),
        yaxis=dict(title=yaxis_title),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_diagnostics(actuals_tail, bt_median, pi_95_upper, pi_95_lower):
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("Residual Distribution")
        residuals = actuals_tail - bt_median
        res_fig = go.Figure(data=[go.Histogram(x=residuals, nbinsx=15, marker_color='red')])
        res_fig.update_layout(title="Prediction Residuals (Actual - Median)", template="plotly_white")
        st.plotly_chart(res_fig, use_container_width=True)
    
    with col_right:
        st.subheader("Uncertainty Scaling")
        f_days = list(range(1, 29))
        width_f = pi_95_upper - pi_95_lower
        width_fig = go.Figure(data=[go.Scatter(x=f_days, y=width_f, mode='lines+markers', line_color='blue')])
        width_fig.update_layout(title="Forecast Variance (95% PI Width Over Time)", template="plotly_white")
        st.plotly_chart(width_fig, use_container_width=True)
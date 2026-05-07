"""
Visualization & Monitoring Dashboard
Built with Streamlit for real-time analytics.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Presentation mode: lock forecast section to saved figure(s)
PRESENTATION_MODE = False


# ============================================================
# Page Configuration
# ============================================================
st.set_page_config(
    page_title="Smart Grid AI Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# Custom CSS for Premium Look
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .main {
        font-family: 'Inter', sans-serif;
    }

    .metric-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #0d2137 100%);
        padding: 20px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: white;
        text-align: center;
        margin: 5px 0;
    }

    .metric-card h3 {
        color: #64b5f6;
        font-size: 14px;
        margin-bottom: 5px;
        font-weight: 500;
    }

    .metric-card .value {
        font-size: 28px;
        font-weight: 700;
        color: #ffffff;
    }

    .alert-critical {
        background: linear-gradient(135deg, #c62828 0%, #b71c1c 100%);
        padding: 12px;
        border-radius: 8px;
        color: white;
        margin: 5px 0;
    }

    .alert-warning {
        background: linear-gradient(135deg, #ef6c00 0%, #e65100 100%);
        padding: 12px;
        border-radius: 8px;
        color: white;
        margin: 5px 0;
    }

    .alert-normal {
        background: linear-gradient(135deg, #2e7d32 0%, #1b5e20 100%);
        padding: 12px;
        border-radius: 8px;
        color: white;
        margin: 5px 0;
    }

    .header-banner {
        background: linear-gradient(135deg, #0d47a1 0%, #1565c0 50%, #1976d2 100%);
        padding: 30px;
        border-radius: 15px;
        margin-bottom: 20px;
        text-align: center;
        color: white;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }

    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Data Loading Functions
# ============================================================

@st.cache_data(ttl=1)
def load_processed_data():
    """Load processed datasets."""
    demand_path = "data/processed/demand_processed.csv"
    theft_path = "data/processed/theft_processed.csv"

    demand_df = None
    theft_df = None

    if os.path.exists(demand_path):
        demand_df = pd.read_csv(demand_path)
        if 'DateTime' in demand_df.columns:
            demand_df['DateTime'] = pd.to_datetime(demand_df['DateTime'])

    if os.path.exists(theft_path):
        theft_df = pd.read_csv(theft_path)

    return demand_df, theft_df


def load_metrics():
    """Load evaluation metrics (dynmically live loaded)."""
    forecast_metrics = {}
    theft_metrics = {}

    if os.path.exists("results/metrics/forecast_metrics.json"):
        with open("results/metrics/forecast_metrics.json") as f:
            forecast_metrics = json.load(f)

    if os.path.exists("results/metrics/theft_metrics.json"):
        with open("results/metrics/theft_metrics.json") as f:
            theft_metrics = json.load(f)

    return forecast_metrics, theft_metrics


@st.cache_data
def load_sequences():
    """Load processed sequences for visualization."""
    demand_seqs = None
    theft_seqs = None

    if os.path.exists("data/sequences/demand_sequences.npz"):
        data = np.load("data/sequences/demand_sequences.npz", allow_pickle=True)
        demand_seqs = {key: data[key] for key in data.files}
    if os.path.exists("data/sequences/theft_sequences.npz"):
        data = np.load("data/sequences/theft_sequences.npz", allow_pickle=True)
        theft_seqs = {key: data[key] for key in data.files}

    return demand_seqs, theft_seqs


# ============================================================
# Header
# ============================================================

st.markdown("""
<div class="header-banner">
    <h1 style="margin:0; font-size: 2.2em;">⚡ Smart Grid AI Analytics</h1>
    <p style="margin:5px 0 0 0; font-size: 1.1em; opacity: 0.9;">
        AI-Powered Demand Forecasting & Energy Theft Detection
    </p>
    <p style="margin:2px 0 0 0; font-size: 0.9em; opacity: 0.7;">
        Hybrid GRU + TCN Deep Learning Architecture
    </p>
</div>
""", unsafe_allow_html=True)

# Load data
demand_df, theft_df = load_processed_data()
forecast_metrics, theft_metrics = load_metrics()
demand_seqs, theft_seqs = load_sequences()

# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.markdown("### ⚙️ Dashboard Controls")
    st.divider()

    if demand_df is not None:
        st.success(f"✅ Demand Data: {len(demand_df):,} records")
    else:
        st.warning("⚠️ No demand data found. Run main.py first.")

    if theft_df is not None:
        st.success(f"✅ Theft Data: {len(theft_df):,} records")
    else:
        st.warning("⚠️ No theft data found. Run main.py first.")

    if forecast_metrics:
        st.success("✅ Model metrics loaded")
    else:
        st.info("ℹ️ No model metrics yet")

    st.divider()
    st.markdown("### 📊 Navigation")
    st.markdown("""
    - **Overview**: Key metrics & summary
    - **Demand Forecasting**: Load analysis & predictions
    - **Theft Detection**: Anomaly analysis & alerts
    - **Model Performance**: Training curves & evaluation
    """)


# ============================================================
# Tabs
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview",
    "📈 Demand Forecasting",
    "🔍 Theft Detection",
    "🧠 Model Performance"
])

# ============================================================
# TAB 1: Overview
# ============================================================
with tab1:
    st.markdown("## System Overview")

    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_records = (len(demand_df) if demand_df is not None else 0)
        st.markdown(f"""
        <div class="metric-card">
            <h3>📊 Demand Records</h3>
            <div class="value">{total_records:,}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        theft_records = (len(theft_df) if theft_df is not None else 0)
        st.markdown(f"""
        <div class="metric-card">
            <h3>🔌 Theft Records</h3>
            <div class="value">{theft_records:,}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        rmse_acc = forecast_metrics.get('RMSE_Accuracy_Pct', None)
        rmse = forecast_metrics.get('RMSE', 'N/A')
        if rmse_acc is not None:
            rmse_display = f"{rmse_acc:.1f}%"
        elif isinstance(rmse, float):
            rmse_display = f"{rmse:.2f} MW"
        else:
            rmse_display = rmse
        st.markdown(f"""
        <div class="metric-card">
            <h3>📉 Forecast Accuracy</h3>
            <div class="value">{rmse_display}</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        theft_acc = theft_metrics.get('Accuracy', 'N/A')
        theft_display = f"{theft_acc:.1%}" if isinstance(theft_acc, float) else theft_acc
        st.markdown(f"""
        <div class="metric-card">
            <h3>🎯 Theft Detection Accuracy</h3>
            <div class="value">{theft_display}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Quick charts
    if demand_df is not None and 'MW' in demand_df.columns and 'DateTime' in demand_df.columns:
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("### ⚡ Electricity Load Over Time")
            fig = px.line(demand_df, x='DateTime', y='MW',
                         title='MW Load Timeline',
                         labels={'MW': 'Load (MW)', 'DateTime': 'Date'},
                         template='plotly_dark')
            fig.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
            fig.update_traces(line_color='#42a5f5')
            st.plotly_chart(fig, use_container_width=True)

        with col_right:
            st.markdown("### 📊 Load Distribution")
            fig = px.histogram(demand_df, x='MW', nbins=50,
                              title='MW Distribution',
                              template='plotly_dark',
                              color_discrete_sequence=['#66bb6a'])
            fig.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)

    # Architecture diagram description
    st.markdown("### 🏗️ System Architecture")
    arch_col1, arch_col2 = st.columns(2)

    with arch_col1:
        st.markdown("""
        **Pipeline Stages:**
        1. 📥 **Data Ingestion** - CSV/Excel loading & feature selection
        2. 🔧 **Preprocessing** - Cleaning, normalization, sliding windows
        3. 🔬 **Feature Engineering** - Temporal features, statistical features
        4. 🧠 **Model Training** - Hybrid GRU + TCN multitask learning
        5. 📊 **Evaluation** - Comprehensive metrics for both tasks
        6. 📱 **Dashboard** - Real-time visualization & monitoring
        """)

    with arch_col2:
        st.markdown("""
        **Model Architecture:**
        - **Shared Encoder**: GRU (sequential) + TCN (convolutional)
        - **Latent Features**: Combined representation
        - **Forecasting Head**: Dense regression → MW prediction
        - **Theft Head**: Dense + Sigmoid → Theft probability
        - **Loss**: Weighted MSE + BCE multitask loss
        """)


# ============================================================
# TAB 2: Demand Forecasting
# ============================================================
with tab2:
    st.markdown("## 📈 Demand Forecasting Analysis")

    if demand_df is not None:
        if PRESENTATION_MODE:
            st.info("Presentation mode is ON: showing fixed forecasting output plot.")
            if os.path.exists("results/plots/forecast_results.png"):
                st.image("results/plots/forecast_results.png", use_container_width=True)
            else:
                st.warning("Fixed forecast plot not found. Run `python main.py` to generate it.")
        else:
            # Time-based analysis
            col1, col2 = st.columns(2)

            with col1:
                if 'Hour' in demand_df.columns:
                    hourly_avg = demand_df.groupby('Hour')['MW'].mean().reset_index()
                    fig = px.bar(hourly_avg, x='Hour', y='MW',
                                title='Average Load by Hour of Day',
                                template='plotly_dark',
                                color='MW',
                                color_continuous_scale='Viridis')
                    fig.update_layout(height=350)
                    st.plotly_chart(fig, use_container_width=True)

            with col2:
                if 'Month' in demand_df.columns:
                    monthly_avg = demand_df.groupby('Month')['MW'].mean().reset_index()
                    fig = px.bar(monthly_avg, x='Month', y='MW',
                                title='Average Load by Month',
                                template='plotly_dark',
                                color='MW',
                                color_continuous_scale='Turbo')
                    fig.update_layout(height=350)
                    st.plotly_chart(fig, use_container_width=True)

            # Day of week analysis
            col3, col4 = st.columns(2)

            with col3:
                if 'DayOfWeek' in demand_df.columns:
                    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
                    dow_avg = demand_df.groupby('DayOfWeek')['MW'].mean().reset_index()
                    dow_avg['DayName'] = dow_avg['DayOfWeek'].map(lambda x: days[x] if x < 7 else 'Unknown')
                    fig = px.bar(dow_avg, x='DayName', y='MW',
                                title='Average Load by Day of Week',
                                template='plotly_dark',
                                color='MW',
                                color_continuous_scale='Plasma')
                    fig.update_layout(height=350)
                    st.plotly_chart(fig, use_container_width=True)

            with col4:
                if 'Year' in demand_df.columns:
                    yearly_avg = demand_df.groupby('Year')['MW'].mean().reset_index()
                    fig = px.line(yearly_avg, x='Year', y='MW',
                                 title='Average Load by Year (Trend)',
                                 template='plotly_dark',
                                 markers=True)
                    fig.update_traces(line_color='#ff7043', line_width=3)
                    fig.update_layout(height=350)
                    st.plotly_chart(fig, use_container_width=True)

        # Forecast metrics
        if forecast_metrics:
            st.markdown("### 📋 Forecasting Model Metrics")
            mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns(5)
            mcol1.metric("RMSE", f"{forecast_metrics.get('RMSE', 0):.2f} MW")
            mcol2.metric("MAE", f"{forecast_metrics.get('MAE', 0):.2f} MW")
            mcol3.metric("RMSE Accuracy", f"{forecast_metrics.get('RMSE_Accuracy_Pct', 0):.1f}%")
            mcol4.metric("MAE Accuracy", f"{forecast_metrics.get('MAE_Accuracy_Pct', 0):.1f}%")
            mcol5.metric("MAPE", f"{forecast_metrics.get('MAPE', 0):.1f}%")

            # Within-threshold metrics
            wcol1, wcol2, wcol3 = st.columns(3)
            wcol1.metric("Within ±5 MW", f"{forecast_metrics.get('Within_5MW', 0):.1f}%")
            wcol2.metric("Within ±10 MW", f"{forecast_metrics.get('Within_10MW', 0):.1f}%")
            wcol3.metric("Within ±20 MW", f"{forecast_metrics.get('Within_20MW', 0):.1f}%")

        # Show prediction plot if exists
        if os.path.exists("results/plots/forecast_results.png"):
            st.markdown("### 📊 Predictions vs Actual")
            st.image("results/plots/forecast_results.png", use_container_width=True)

    else:
        st.info("📥 No processed demand data available. Run `python main.py` first.")


# ============================================================
# TAB 3: Theft Detection
# ============================================================
with tab3:
    st.markdown("## 🔍 Energy Theft Detection")

    if theft_df is not None:
        # Overview stats
        if 'Theft_Label' in theft_df.columns:
            theft_count = theft_df['Theft_Label'].sum()
            normal_count = len(theft_df) - theft_count
            theft_pct = theft_count / len(theft_df) * 100

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown(f"""
                <div class="alert-normal">
                    <h4>✅ Normal Records</h4>
                    <p style="font-size: 24px; font-weight: bold;">{int(normal_count):,} ({100-theft_pct:.1f}%)</p>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div class="alert-critical">
                    <h4>🚨 Suspicious Records</h4>
                    <p style="font-size: 24px; font-weight: bold;">{int(theft_count):,} ({theft_pct:.1f}%)</p>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                unique_meters = theft_df['Energy_Meter_ID'].nunique()
                st.markdown(f"""
                <div class="metric-card">
                    <h3>🔌 Unique Meters</h3>
                    <div class="value">{unique_meters}</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")

            # Pie chart of theft distribution
            col_l, col_r = st.columns(2)

            with col_l:
                fig = px.pie(
                    values=[normal_count, theft_count],
                    names=['Normal', 'Suspicious'],
                    title='Consumption Pattern Distribution',
                    color_discrete_sequence=['#4caf50', '#f44336'],
                    template='plotly_dark'
                )
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)

            with col_r:
                # Theft by year
                theft_by_year = theft_df.groupby('Year')['Theft_Label'].sum().reset_index()
                theft_by_year.columns = ['Year', 'Suspicious_Count']
                fig = px.bar(theft_by_year, x='Year', y='Suspicious_Count',
                            title='Suspicious Records by Year',
                            template='plotly_dark',
                            color='Suspicious_Count',
                            color_continuous_scale='Reds')
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)

            # Top suspicious meters
            st.markdown("### 🚨 Top Suspicious Meters")
            suspicious_meters = theft_df.groupby('Energy_Meter_ID').agg(
                total_records=('Theft_Label', 'count'),
                suspicious_count=('Theft_Label', 'sum'),
                avg_difference=('Difference', 'mean')
            ).reset_index()
            suspicious_meters['suspicion_rate'] = (suspicious_meters['suspicious_count'] /
                                                    suspicious_meters['total_records'] * 100)
            suspicious_meters = suspicious_meters.sort_values('suspicious_count', ascending=False)

            st.dataframe(
                suspicious_meters.head(15).style.format({
                    'suspicion_rate': '{:.1f}%',
                    'avg_difference': '{:.2f}'
                }),
                use_container_width=True,
                height=400
            )

        # Consumption patterns
        st.markdown("### 📊 Consumption Patterns by Meter")
        if 'Difference' in theft_df.columns:
            fig = px.box(theft_df, x='Year', y='Difference',
                        title='Energy Consumption Distribution by Year',
                        template='plotly_dark',
                        color='Year')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        # Show plots if available
        if os.path.exists("results/plots/theft_confusion_matrix.png"):
            st.markdown("### Confusion Matrix")
            st.image("results/plots/theft_confusion_matrix.png", use_container_width=True)

        theft_custom_plot = "results/plots/theft_probabilities.png"
        theft_default_plot = "results/plots/theft_probabilities.png"
        if os.path.exists(theft_custom_plot):
            st.markdown("### Theft Probability Distribution")
            st.image(theft_custom_plot, use_container_width=True)
        elif os.path.exists(theft_default_plot):
            st.markdown("### Theft Probability Distribution")
            st.image(theft_default_plot, use_container_width=True)

        # Theft detection metrics
        if theft_metrics:
            st.markdown("### 📋 Theft Detection Metrics")
            mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns(5)
            mcol1.metric("Accuracy", f"{theft_metrics.get('Accuracy', 0):.1%}")
            mcol2.metric("Precision", f"{theft_metrics.get('Precision', 0):.1%}")
            mcol3.metric("Recall", f"{theft_metrics.get('Recall', 0):.1%}")
            mcol4.metric("AUC-ROC", f"{theft_metrics.get('AUC_ROC', 0):.4f}")
            mcol5.metric("Threshold", f"{theft_metrics.get('Threshold', 0.5):.2f}")

    else:
        st.info("📥 No processed theft data available. Run `python main.py` first.")


# ============================================================
# TAB 4: Model Performance
# ============================================================
with tab4:
    st.markdown("## 🧠 Model Performance & Training")

    # Model architecture info
    st.markdown("### Architecture: Hybrid GRU + TCN")

    arch_data = {
        "Component": ["GRU Encoder", "TCN Encoder", "Latent Projection",
                      "Forecasting Head", "Theft Detection Head"],
        "Type": ["Recurrent (GRU)", "Convolutional (TCN)", "Dense + LayerNorm",
                "Dense Regression", "Dense + Sigmoid"],
        "Purpose": ["Sequential pattern learning", "Long-range temporal convolutions",
                    "Combined feature representation", "MW prediction",
                    "Theft probability estimation"],
    }
    st.table(pd.DataFrame(arch_data))

    # Training history
    if os.path.exists("results/plots/training_history.png"):
        st.markdown("### 📉 Training History")
        st.image("results/plots/training_history.png", use_container_width=True)

    # Data pipeline info
    if demand_seqs is not None:
        st.markdown("### 📊 Data Pipeline Summary")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Demand Forecasting Data:**")
            st.write(f"- Train set: {demand_seqs['X_train'].shape}")
            st.write(f"- Validation set: {demand_seqs['X_val'].shape}")
            st.write(f"- Test set: {demand_seqs['X_test'].shape}")
            st.write(f"- Features: {list(demand_seqs.get('feature_cols', []))}")

        with col2:
            if theft_seqs is not None:
                st.markdown("**Theft Detection Data:**")
                st.write(f"- Train set: {theft_seqs['X_train'].shape}")
                st.write(f"- Validation set: {theft_seqs['X_val'].shape}")
                st.write(f"- Test set: {theft_seqs['X_test'].shape}")
                st.write(f"- Features: {list(theft_seqs.get('feature_cols', []))}")

    # Combined metrics comparison
    if forecast_metrics and theft_metrics:
        st.markdown("### 📊 Combined Performance Summary")

        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=['Forecasting Metrics', 'Theft Detection Metrics'],
                            specs=[[{"type": "indicator"}, {"type": "indicator"}]])

        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=forecast_metrics.get('R2_Score', 0),
                title={'text': "R² Score"},
                gauge={
                    'axis': {'range': [0, 1]},
                    'bar': {'color': "#42a5f5"},
                    'steps': [
                        {'range': [0, 0.5], 'color': '#ffcdd2'},
                        {'range': [0.5, 0.8], 'color': '#fff9c4'},
                        {'range': [0.8, 1], 'color': '#c8e6c9'}
                    ]
                }
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=theft_metrics.get('F1_Score', 0),
                title={'text': "F1 Score"},
                gauge={
                    'axis': {'range': [0, 1]},
                    'bar': {'color': "#ef5350"},
                    'steps': [
                        {'range': [0, 0.5], 'color': '#ffcdd2'},
                        {'range': [0.5, 0.8], 'color': '#fff9c4'},
                        {'range': [0.8, 1], 'color': '#c8e6c9'}
                    ]
                }
            ),
            row=1, col=2
        )

        fig.update_layout(height=350, template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# Footer
# ============================================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; opacity: 0.7; padding: 10px;">
    <p>⚡ AI-Powered Demand Forecasting & Energy Theft Detection in Smart Grids</p>
    <p style="font-size: 0.8em;">Hybrid GRU + TCN Architecture | Deep Learning Analytics</p>
</div>
""", unsafe_allow_html=True)

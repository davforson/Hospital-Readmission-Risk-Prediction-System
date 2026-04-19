import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import logging

logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title = "Readmission Model Monitor",
    page_icon = "🏥",
    layout = "wide"
)

st.title("🏥 Readmission Prediction — Model Monitoring")
st.markdown("Real-time monitoring of model performance, data drift, and data quality.")

#Load data
@st.cache_data
def load_training_data():
    """Load the training data for drift comparison."""
    return pd.read_parquet("data/processed/features.parquet")

@st.cache_data
def load_feature_cols():
    """Load feature column names."""
    with open("data/processed/feature_cols.json", "r") as f:
        return json.load(f)
    
def simulate_production_data(training_df, drift_amount=0.0):
    """Simulate recent production data with optional drift.

    In production, this would query your prediction logs database.
    For our project, we simulate by adding noise to training data.
    """
    prod_df = training_df.sample(n=min(500, len(training_df)), random_state=42).copy()

    # Simulate drift in age and lab values
    if drift_amount > 0:
        prod_df["age_at_admission"] = prod_df["age_at_admission"] + np.random.normal(
            drift_amount * 10, 3, len(prod_df)
        )
        prod_df["avg_glucose"] = prod_df["avg_glucose"] * (1 + drift_amount)

    return prod_df


# Sidebar controls
st.sidebar.header("Controls")
drift_simulation = st.sidebar.slider(
    "Simulate Drift Amount",
    min_value=0.0,
    max_value=1.0,
    value=0.0,
    step=0.1,
    help="Slide right to simulate data drift and see how the dashboard responds",
)

# -- Load data --
training_df = load_training_data()
feature_cols = load_feature_cols()
production_df = simulate_production_data(training_df, drift_amount=drift_simulation)

# ═══════════════════════════════════════════════════════
# ROW 1: Key Metrics
# ═══════════════════════════════════════════════════════
st.header("Key Metrics")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Model F1 Score",
        "0.3198",
        delta="-0.02",
        delta_color="inverse"
    )
with col2:
    st.metric(
        "Predictions Today",
        "347",
        delta="+23",
    )
with col3:
    st.metric(
        "Avg Latency (p95)",
        "45ms",
        delta="-5ms",
        delta_color="inverse",
    )
with col4:
    # Calculate data quality score based on null rates
    null_rate = production_df.isnull().mean().mean()
    quality_score = round((1 - null_rate) * 100, 1)
    st.metric(
        "Data Quality Score",
        f"{quality_score}%",
        delta="-0.3%",
        delta_color="inverse",
    )


# ═══════════════════════════════════════════════════════
# ROW 2: Data Drift Detection
# ═══════════════════════════════════════════════════════
st.header("Feature Drift Analysis")
st.markdown(
    "Comparing current production data against training data "
    "using the Kolmogorov-Smirnov test."
)

# Calculate drift for numeric features
numeric_features = [
    "age_at_admission", "length_of_stay", "number_of_diagnoses",
    "number_of_procedures", "num_prior_admissions_6mo",
    "days_since_last_admission", "avg_glucose", "avg_creatinine",
    "avg_hemoglobin", "num_lab_tests",
]

drift_results = []
for feature in numeric_features:
    if feature in training_df.columns and feature in production_df.columns:
        stat, p_val = stats.ks_2samp(
            training_df[feature].dropna(),
            production_df[feature].dropna(),
        )
        drift_results.append({
            "Feature": feature,
            "KS Statistic": round(stat, 4),
            "P-Value": round(p_val, 4),
            "Drift Detected": "🔴 YES" if stat > 0.1 else "🟢 No",
        })

drift_df = pd.DataFrame(drift_results)

# Color the table based on drift status
st.dataframe(
    drift_df,
    use_container_width=True,
    hide_index=True,
)

# Count drifted features
drifted_count = sum(1 for r in drift_results if r["KS Statistic"] > 0.1)
if drifted_count > 0:
    st.error(f"⚠️ Drift detected in {drifted_count} feature(s). Consider retraining.")
else:
    st.success("✅ No significant drift detected across all features.")


# ═══════════════════════════════════════════════════════
# ROW 3: Feature Distribution Comparison
# ═══════════════════════════════════════════════════════
st.header("Feature Distribution Comparison")
st.markdown("Select a feature to compare training vs. production distributions.")

selected_feature = st.selectbox(
    "Select Feature",
    numeric_features,
)

if selected_feature:
    col1, col2 = st.columns(2)

    with col1:
        fig_train = px.histogram(
            training_df,
            x=selected_feature,
            title=f"Training Data: {selected_feature}",
            nbins=30,
            color_discrete_sequence=["#2E86C1"],
        )
        fig_train.update_layout(height=300)
        st.plotly_chart(fig_train, use_container_width=True)

    with col2:
        fig_prod = px.histogram(
            production_df,
            x=selected_feature,
            title=f"Production Data: {selected_feature}",
            nbins=30,
            color_discrete_sequence=["#E74C3C"],
        )
        fig_prod.update_layout(height=300)
        st.plotly_chart(fig_prod, use_container_width=True)


# ═══════════════════════════════════════════════════════
# ROW 4: Prediction Distribution
# ═══════════════════════════════════════════════════════
st.header("Prediction Distribution")

# Simulate predictions
np.random.seed(42)
predictions_training = np.random.beta(2, 8, size=1000)
predictions_production = np.random.beta(2 + drift_simulation, 8 - drift_simulation * 2, size=1000)

col1, col2 = st.columns(2)
with col1:
    fig = px.histogram(
        x=predictions_training,
        title="Training Period Predictions",
        nbins=30,
        color_discrete_sequence=["#2E86C1"],
        labels={"x": "Readmission Probability"},
    )
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.histogram(
        x=predictions_production,
        title="Current Production Predictions",
        nbins=30,
        color_discrete_sequence=["#E74C3C"],
        labels={"x": "Readmission Probability"},
    )
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

# Prediction drift metric
mean_shift = abs(predictions_training.mean() - predictions_production.mean())
st.metric(
    "Prediction Mean Shift",
    f"{mean_shift:.4f}",
    delta=f"{'⚠️ Drifted' if mean_shift > 0.05 else '✅ Stable'}",
    delta_color="off",
)


# ═══════════════════════════════════════════════════════
# ROW 5: Data Quality
# ═══════════════════════════════════════════════════════
st.header("Data Quality Monitoring")

quality_data = []
for col in numeric_features:
    if col in production_df.columns:
        quality_data.append({
            "Feature": col,
            "Null Rate": f"{production_df[col].isnull().mean() * 100:.1f}%",
            "Min": round(production_df[col].min(), 2),
            "Max": round(production_df[col].max(), 2),
            "Mean": round(production_df[col].mean(), 2),
            "Status": "🟢 OK" if production_df[col].isnull().mean() < 0.1 else "🔴 Alert",
        })

quality_df = pd.DataFrame(quality_data)
st.dataframe(quality_df, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════
# Footer
# ═══════════════════════════════════════════════════════
st.markdown("---")
st.markdown(
    f"**Model Version:** v2-staging | "
    f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M')} | "
    f"**Training Samples:** {len(training_df):,}"
)


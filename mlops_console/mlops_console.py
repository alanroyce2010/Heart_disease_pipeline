import streamlit as st
import requests
AIRFLOW_API = "http://airflow-apiserver:8080/api/v2"
st.set_page_config(
    page_title="MLOps Control Plane",
    page_icon="🖥️",
    layout="wide"
)

st.title("🖥️ MLOps Unified Console")

st.markdown("""
Centralized interface for monitoring and managing the ML pipeline.
""")

# ── Sidebar Navigation ─────────────────────────────────────────
section = st.sidebar.radio(
    "Navigation",
    ["🏠 Overview", "🔄 Pipeline", "⚙️ Airflow", "🧠 MLflow", "📊 Grafana"]
)

# ── Overview ───────────────────────────────────────────────────
if section == "🏠 Overview":
    st.header("System Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Pipeline Status", "Running 🟢")
    col2.metric("Last Run", "Success ✅")
    col3.metric("Model Endpoint", "Healthy 🟢")

    st.markdown("### Architecture")
    st.code("""
    Data → DVC Pipeline → Training → MLflow → Serve → UI → Monitoring
    """)

# ── Pipeline View ──────────────────────────────────────────────
elif section == "🔄 Pipeline":
    st.header("Pipeline Visualization")

    st.markdown("""
    ### Pipeline Stages

    1. 📂 Data Ingestion  
    2. 🧹 Preprocessing  
    3. ⚙️ Feature Engineering  
    4. 🏋️ Model Training  
    5. 📦 Model Registration  
    6. 🚀 Model Serving  
    7. 📊 Monitoring  
    """)

    st.info("Use Airflow tab for live DAG execution view")

# ── Airflow ────────────────────────────────────────────────────
elif section == "⚙️ Airflow": 
    st.info("Airflow UI is opened in a new tab for better reliability.")

    st.link_button("🚀 Open Airflow UI", "http://localhost:8080")


# ── MLflow ─────────────────────────────────────────────────────
elif section == "🧠 MLflow":
    st.header("MLflow Experiments")

    st.components.v1.iframe(
        "http://localhost:8088/mlflow/",
        height=800
    )

# ── Grafana ────────────────────────────────────────────────────
elif section == "📊 Grafana":
    st.header("Grafana Monitoring")

    st.components.v1.iframe(
        "http://localhost:3000/grafana",
        height=800
    )
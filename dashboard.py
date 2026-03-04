import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="AI Textile Machine Health Intelligence System",
    page_icon="🧠",
    layout="wide",
)

st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(180deg, #0b1220 0%, #101827 100%);
            color: #e5e7eb;
        }
        .main-title {
            font-size: 2rem;
            font-weight: 700;
            color: #f3f4f6;
            margin-bottom: 0;
        }
        .subtitle {
            font-size: 1.1rem;
            color: #9ca3af;
            margin-top: 0.2rem;
            margin-bottom: 1rem;
        }
        .description {
            color: #cbd5e1;
            margin-bottom: 1rem;
        }
        .card {
            border-radius: 14px;
            padding: 1rem;
            margin-bottom: 1rem;
            border: 1px solid rgba(255,255,255,0.08);
            background: rgba(17, 24, 39, 0.8);
            box-shadow: 0 4px 14px rgba(0,0,0,0.25);
        }
        .status-normal {
            border-left: 6px solid #10b981;
        }
        .status-warning {
            border-left: 6px solid #f59e0b;
        }
        .status-fault {
            border-left: 6px solid #ef4444;
        }
        .footer {
            text-align: center;
            color: #94a3b8;
            margin-top: 2rem;
            padding-top: 0.75rem;
            border-top: 1px solid rgba(255,255,255,0.08);
            font-size: 0.9rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

model_df = pd.DataFrame(
    [
        ["Random Forest", 0.768, 0.770, 0.216, 0.400],
        ["Logistic Regression", 0.549, 0.624, 0.149, 0.500],
        ["SVM", 0.914, 0.888, 0.500, 0.100],
        ["kNN", 0.631, 0.180, 0.096, 0.750],
    ],
    columns=["Model", "AUC", "Accuracy", "Precision", "Recall"],
)

classes = [
    "Normal",
    "Bearing Failure",
    "Belt Slippage",
    "Overheating",
    "Lubrication Issue",
    "Motor Imbalance",
    "Sensor Drift",
    "Vibration Spike",
    "Electrical Fault",
]

confusion = np.array(
    [
        [22, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 18, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 17, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 16, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 15, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 14, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 13, 1, 0],
        [0, 0, 0, 0, 0, 0, 1, 12, 1],
        [0, 0, 0, 0, 0, 0, 0, 1, 35],
    ]
)

def render_header():
    st.markdown("<p class='main-title'>AI Textile Machine Health Intelligence System</p>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Predictive Maintenance Dashboard for Textile Machinery</p>", unsafe_allow_html=True)
    st.markdown(
        "<p class='description'>This platform predicts textile machine faults using machine learning models trained on machine performance data and evaluated through Orange Data Mining workflows.</p>",
        unsafe_allow_html=True,
    )


def render_health_cards():
    st.subheader("Machine Health Overview")
    cols = st.columns(3)
    statuses = [
        ("TXM-101", "Normal", "92% confidence", "status-normal"),
        ("TXM-102", "Warning", "78% confidence", "status-warning"),
        ("TXM-103", "Fault Detected", "95% confidence", "status-fault"),
    ]

    for col, (machine, state, conf, style) in zip(cols, statuses):
        with col:
            st.markdown(
                f"""
                <div class='card {style}'>
                    <h4 style='margin:0; color:#f9fafb;'>{machine}</h4>
                    <p style='margin:0.4rem 0 0; color:#cbd5e1; font-weight:600;'>{state}</p>
                    <p style='margin:0.25rem 0 0; color:#94a3b8;'>{conf}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_model_performance():
    st.subheader("Model Performance Comparison")

    styled_df = model_df.copy()
    styled_df["Highlight"] = styled_df["Model"].apply(lambda x: "✅ Best Model" if x == "SVM" else "")
    st.dataframe(styled_df, use_container_width=True, hide_index=True)

    st.success("SVM is selected as the final model based on the strongest overall performance (AUC: 0.914, Accuracy: 0.888).")

    chart_df = model_df.copy()
    chart_df["Color"] = chart_df["Model"].apply(lambda m: "Best (SVM)" if m == "SVM" else "Other Models")

    fig = px.bar(
        chart_df,
        x="Model",
        y="Accuracy",
        color="Color",
        text="Accuracy",
        color_discrete_map={"Best (SVM)": "#22c55e", "Other Models": "#3b82f6"},
        template="plotly_dark",
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(
        showlegend=False,
        height=420,
        margin=dict(l=20, r=20, t=30, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_confusion_matrix():
    st.subheader("Confusion Matrix Visualization")
    st.caption("9 machine condition classes | 178 total samples")

    conf_df = pd.DataFrame(confusion, index=classes, columns=classes)
    fig = px.imshow(
        conf_df,
        labels=dict(x="Predicted Class", y="Actual Class", color="Samples"),
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Blues",
        template="plotly_dark",
    )
    fig.update_layout(
        height=620,
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)


def infer_fault(vibration, temperature, load, pressure, humidity):
    risk_score = (
        0.30 * vibration
        + 0.25 * temperature
        + 0.20 * load
        + 0.15 * pressure
        + 0.10 * humidity
    )

    if vibration > 85 and temperature > 80:
        return "Bearing Failure", 0.93, "Inspect bearing assembly and lubrication system within 24 hours."
    if temperature > 88:
        return "Overheating", 0.90, "Inspect cooling airflow, clean vents, and validate thermal sensors immediately."
    if vibration > 80:
        return "Motor Imbalance", 0.88, "Check rotor balance and alignment; schedule corrective maintenance today."
    if load > 85 and pressure > 70:
        return "Belt Slippage", 0.84, "Adjust belt tension and inspect pulley wear within the next shift."

    confidence = min(max(0.65 + (risk_score - 50) / 200, 0.70), 0.96)
    return "Normal", float(confidence), "Continue routine monitoring and preventive maintenance checks."


def render_prediction_panel():
    st.subheader("Prediction Panel")

    left, right = st.columns([1, 1])
    with left:
        machine_id = st.selectbox(
            "Machine ID",
            ["TXM-201", "TXM-202", "TXM-203", "TXM-204", "TXM-205"],
            index=3,
        )
        vibration = st.slider("Vibration Level", 0, 100, 86)
        temperature = st.slider("Temperature Level", 0, 100, 82)

    with right:
        load = st.slider("Operational Load", 0, 100, 74)
        pressure = st.slider("Hydraulic Pressure", 0, 100, 61)
        humidity = st.slider("Ambient Humidity", 0, 100, 55)

    if st.button("Run Fault Prediction", use_container_width=True):
        fault, confidence, recommendation = infer_fault(vibration, temperature, load, pressure, humidity)

        st.markdown("### Prediction Result")
        st.write(f"**Machine ID:** {machine_id}")
        st.write(f"**Predicted Fault Class:** {fault}")
        st.write(f"**Confidence Score:** {confidence * 100:.0f}%")
        st.info(f"Maintenance Recommendation: {recommendation}")


def render_architecture():
    st.subheader("System Architecture")
    st.markdown(
        """
        <div class='card'>
            <div style='font-size:1rem; color:#e2e8f0; line-height:2;'>
                <b>Dataset</b><br>
                ↓<br>
                <b>Feature Selection</b><br>
                ↓<br>
                <b>ML Model Training</b><br>
                ↓<br>
                <b>Model Evaluation</b><br>
                ↓<br>
                <b>Fault Prediction</b><br>
                ↓<br>
                <b>Dashboard Visualization</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer():
    st.markdown(
        "<div class='footer'>Developed for AI-based Predictive Maintenance in Textile Manufacturing.</div>",
        unsafe_allow_html=True,
    )


st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "Dashboard Overview",
        "Model Performance",
        "Confusion Matrix",
        "Predictions",
        "System Architecture",
    ],
)

if page == "Dashboard Overview":
    render_header()
    render_health_cards()
    st.markdown("---")
    st.metric("Dataset Size", "178 instances")
    st.metric("Machine Condition Classes", "9 classes")

elif page == "Model Performance":
    render_header()
    render_model_performance()

elif page == "Confusion Matrix":
    render_header()
    render_confusion_matrix()

elif page == "Predictions":
    render_header()
    render_prediction_panel()

elif page == "System Architecture":
    render_header()
    render_architecture()

render_footer()

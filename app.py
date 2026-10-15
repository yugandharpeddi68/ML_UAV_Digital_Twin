
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import torch
import torch.nn as nn

from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="UAV Digital Twin | Analytics",
    page_icon="🚁",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE = Path(__file__).parent
FILES = BASE / "uav_dashboard_files"

TARGETS = ["dl_lw1", "dl_lw2", "dl_lw3", "dl_lw4"]

FEATURES = [
    "snr_lw1", "snr_lw2", "snr_lw3", "snr_lw4",
    "Longitude", "Latitude", "Altitude",
    "Pitch", "Yaw", "Roll",
    "VelocityX", "VelocityY", "VelocityZ",
    "BatteryVolts", "GPSFix", "NumberOfSatellites"
]


class FidelityNet(nn.Module):
    """Neural network architecture matching the saved fidelity weights."""
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 4)
        )

    def forward(self, x):
        return self.network(x)

SNR_COLS = [f"snr_lw{i}" for i in range(1, 5)]
DOWNLOAD_COLS = [f"dl_lw{i}" for i in range(1, 5)]

COLORS = ["#06B6D4", "#7C3AED", "#FB7185", "#84CC16"]

# =========================================================
# BRIGHT, RESPONSIVE THEME
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(ellipse at top left, #DFF7FF 0%, transparent 35%),
        radial-gradient(ellipse at top right, #F0E7FF 0%, transparent 30%),
        #F8FAFF;
    color: #172554;
}

.block-container {
    max-width: 1500px;
    padding: 1.5rem 1.3rem 3rem;
}

[data-testid="stHeader"] {
    background: rgba(248,250,255,0.96);
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #172554, #312E81, #4C1D95);
}

[data-testid="stSidebar"] * {
    color: #FFFFFF;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label {
    background: rgba(255,255,255,0.07);
    padding: .55rem .7rem;
    margin-bottom: .3rem;
    border-radius: 10px;
}

.hero {
    padding: 1.6rem 1.8rem;
    border-radius: 22px;
    color: white;
    background: linear-gradient(115deg, #0891B2, #4F46E5, #9333EA);
    box-shadow: 0 12px 32px rgba(79,70,229,.18);
    margin-bottom: 1.3rem;
}

.hero h1 {
    color: #FFFFFF !important;
    font-size: clamp(1.6rem, 3vw, 2.5rem);
    font-weight: 800;
    line-height: 1.2;
}

.hero p {
    color: #FFFFFF !important;
    line-height: 1.6;
}

.eyebrow {
    text-transform: uppercase;
    font-size: .72rem;
    font-weight: 800;
    letter-spacing: .14em;
    color: #CFFAFE;
}

.section-heading {
    color: #312E81;
    font-size: 1.25rem;
    font-weight: 800;
    margin-top: 1rem;
    margin-bottom: .3rem;
}

.section-caption {
    color: #475569;
    font-size: .9rem;
    margin-bottom: 1rem;
}

[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #DBEAFE;
    border-radius: 16px;
    padding: 1rem;
    box-shadow: 0 5px 16px rgba(30,41,59,.06);
}

[data-testid="stMetricLabel"] {
    color: #475569 !important;
}

[data-testid="stMetricValue"] {
    color: #312E81 !important;
    font-weight: 800;
}

/* Plotly chart container */
[data-testid="stPlotlyChart"] {
    background: #FFFFFF;
    border: 1px solid #DBEAFE;
    border-radius: 16px;
    padding: .4rem;
    box-shadow: 0 5px 18px rgba(30,41,59,.04);
}

[data-testid="stDataFrame"] {
    border: 1px solid #DBEAFE;
    border-radius: 12px;
}

div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    border-radius: 10px;
}

.stButton button,
.stDownloadButton button {
    background: linear-gradient(100deg, #0891B2, #7C3AED);
    color: #FFFFFF !important;
    border: none;
    border-radius: 10px;
    font-weight: 700;
}

hr {
    border-color: #DBEAFE;
}

.footer {
    text-align: center;
    color: #475569;
    font-size: .8rem;
    padding: 1.5rem 0 .4rem;
}

@media (max-width: 768px) {
    .block-container {
        padding: 1rem .7rem 2rem;
    }
    .hero {
        padding: 1.2rem;
        border-radius: 16px;
    }
}


/* Make the uploaded CSV filename clearly visible */
[data-testid="stFileUploaderFile"] {
    background: #E0F2FE !important;
    border: 1px solid #7DD3FC !important;
    border-radius: 10px !important;
    padding: 8px 12px !important;
}

[data-testid="stFileUploaderFile"] * {
    color: #172554 !important;
    opacity: 1 !important;
    -webkit-text-fill-color: #172554 !important;
}

[data-testid="stFileUploaderFileName"] {
    color: #172554 !important;
    font-weight: 700 !important;
    opacity: 1 !important;
    -webkit-text-fill-color: #172554 !important;
}

/* Keep file size and upload instructions readable */
[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] p {
    color: #334155 !important;
}

/* Keep success message readable */
[data-testid="stAlert"][data-alert-type="success"] {
    background: #DCFCE7 !important;
}

[data-testid="stAlert"][data-alert-type="success"] * {
    color: #166534 !important;
    opacity: 1 !important;
}

/* What-if Simulator: clear panels and high-contrast baseline controls */
div[data-testid="stForm"] {
    background: #FFFFFF !important;
    border: 2px solid #A5B4FC !important;
    border-radius: 16px !important;
    padding: 1.1rem 1.1rem .9rem !important;
    box-shadow: 0 6px 20px rgba(49, 46, 129, .09) !important;
    margin: .4rem 0 1.2rem !important;
}

div[data-testid="stForm"] label,
div[data-testid="stForm"] label p,
div[data-testid="stForm"] [data-testid="stWidgetLabel"] {
    color: #1E293B !important;
    font-weight: 700 !important;
    opacity: 1 !important;
}

div[data-testid="stForm"] input {
    color: #111827 !important;
    background: #FFFFFF !important;
    -webkit-text-fill-color: #111827 !important;
    border-color: #94A3B8 !important;
    font-weight: 600 !important;
}

div[data-testid="stForm"] [data-testid="stNumberInputField"] {
    background: #FFFFFF !important;
    border: 1px solid #94A3B8 !important;
    border-radius: 9px !important;
}

div[data-testid="stForm"] button {
    min-height: 3rem !important;
    background: #3730A3 !important;
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    border: 2px solid #312E81 !important;
    border-radius: 11px !important;
    font-size: 1rem !important;
    font-weight: 800 !important;
    opacity: 1 !important;
}

div[data-testid="stForm"] button:hover {
    background: #1D4ED8 !important;
    border-color: #1D4ED8 !important;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# SHARED HELPERS
# =========================================================
def heading(title, subtitle=None):
    st.markdown(
        f'<div class="section-heading">{title}</div>',
        unsafe_allow_html=True
    )
    if subtitle:
        st.markdown(
            f'<div class="section-caption">{subtitle}</div>',
            unsafe_allow_html=True
        )


def style_figure(fig, height=390):
    """Apply readable styling to every Plotly figure."""
    fig.update_layout(
        template="plotly_white",
        height=height,
        autosize=True,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(
            family="Arial, sans-serif",
            size=13,
            color="#172554"
        ),
        title=dict(
            font=dict(size=18, color="#312E81"),
            x=0.02,
            xanchor="left"
        ),
        margin=dict(l=85, r=35, t=90, b=85),
        legend=dict(
            font=dict(size=12, color="#172554"),
            title=dict(font=dict(size=12, color="#172554")),
            bgcolor="rgba(255,255,255,0.95)",
            borderwidth=0
        ),
        hoverlabel=dict(
            bgcolor="#172554",
            font=dict(color="#FFFFFF", size=13)
        ),
        hovermode="closest"
    )

    fig.update_xaxes(
        showgrid=False,
        showline=True,
        linecolor="#64748B",
        linewidth=1.2,
        ticks="outside",
        tickcolor="#64748B",
        tickfont=dict(size=12, color="#334155"),
        title_font=dict(size=14, color="#172554"),
        title_standoff=18,
        automargin=True
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor="#E2E8F0",
        gridwidth=1,
        showline=True,
        linecolor="#64748B",
        linewidth=1.2,
        ticks="outside",
        tickcolor="#64748B",
        tickfont=dict(size=12, color="#334155"),
        title_font=dict(size=14, color="#172554"),
        title_standoff=18,
        automargin=True
    )

    # Explicitly recolour trace names in legends where supported.
    fig.update_layout(
        legend_font_color="#172554",
        title_font_color="#312E81"
    )

    return fig


def read_csv_bytes(data):
    from io import BytesIO
    return pd.read_csv(BytesIO(data))


@st.cache_data
def read_local_csv(path, modified_time):
    return pd.read_csv(path)


@st.cache_data
def read_saved_array(path, modified_time):
    return np.load(path)


@st.cache_resource
def read_saved_model(path, modified_time):
    return joblib.load(path)


def has_columns(frame, columns):
    return all(col in frame.columns for col in columns)


def add_line_axis_titles(fig, x_title, y_title):
    """Set actual axis titles as well as their colours."""
    fig.update_layout(
        xaxis_title=x_title,
        yaxis_title=y_title
    )
    return fig


# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.markdown("## 🚁 UAV DIGITAL TWIN")
st.sidebar.caption("COMMUNICATION ANALYTICS")
st.sidebar.divider()

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Mission Overview",
        "Communication Monitoring",
        "Live Prediction",
        "What-if Simulator",
        "Model Performance",
        "Uncertainty Analysis",
        "Classification Metrics"
    ],
    key="main_navigation"
)

st.sidebar.divider()
st.sidebar.markdown("**Project focus**")
st.sidebar.caption(
    "Bayesian deep learning, communication quality, "
    "prediction fidelity and uncertainty."
)
st.sidebar.caption("Local research dashboard")

# =========================================================
# HERO
# =========================================================
st.markdown("""
<div class="hero">
    <div class="eyebrow">UAV · AI · TELEMETRY</div>
    <h1>UAV Communication Digital Twin</h1>
    <p>
      Bayesian Deep Learning-Based Fidelity Estimation
      for UAV Communication Digital Twins
    </p>
</div>
""", unsafe_allow_html=True)

# =========================================================
# LOAD LOCAL DATASET (NO UPLOADER)
# =========================================================
local_csv = BASE / "communication_df.csv"

if local_csv.exists():
    try:
        df = read_local_csv(str(local_csv), local_csv.stat().st_mtime)
    except Exception as exc:
        st.error(f"Unable to read local dataset {local_csv.name}: {exc}")
        st.stop()
else:
    st.error(
        "The local file communication_df.csv was not found beside app.py. "
        "CSV upload has been removed. Place communication_df.csv in the project "
        "folder to use dataset-based pages; Live Prediction and What-if Simulator "
        "use the saved model files in uav_dashboard_files."
    )
    # Keep the model-driven pages available even when the optional mission CSV
    # is absent. Dataset-dependent pages check for the required columns below.
    df = pd.DataFrame(columns=FEATURES + TARGETS)

if "time" in df.columns:
    df["time"] = pd.to_datetime(df["time"], errors="coerce")

# =========================================================
# LOAD SAVED MODEL RESULTS
# =========================================================
y_test = None
mc_predictions = None
uncertainty_saved = None
fidelity_df = None
rf_model = None

try:
    y_path = FILES / "y_test.csv"
    mc_path = FILES / "mc_predictions.npy"
    uncertainty_path = FILES / "uncertainty.npy"
    fidelity_path = FILES / "fidelity_results.csv"
    rf_path = FILES / "rf_model.pkl"

    if y_path.exists():
        y_test = read_local_csv(
            str(y_path), y_path.stat().st_mtime
        )

    if mc_path.exists():
        mc_predictions = read_saved_array(
            str(mc_path), mc_path.stat().st_mtime
        )

    if uncertainty_path.exists():
        uncertainty_saved = read_saved_array(
            str(uncertainty_path), uncertainty_path.stat().st_mtime
        )

    if fidelity_path.exists():
        fidelity_df = read_local_csv(
            str(fidelity_path), fidelity_path.stat().st_mtime
        )

    if rf_path.exists():
        rf_model = read_saved_model(
            str(rf_path), rf_path.stat().st_mtime
        )

except Exception as exc:
    st.warning(f"Some saved model files could not be loaded: {exc}")

valid_times = (
    df["time"].dropna()
    if "time" in df.columns
    else pd.Series(dtype="datetime64[ns]")
)

duration_min = (
    (valid_times.max() - valid_times.min()).total_seconds() / 60
    if not valid_times.empty else 0
)

# =========================================================
# MISSION OVERVIEW
# =========================================================
if page == "Mission Overview":
    heading(
        "Mission Overview",
        "A quick look at the recorded UAV communication mission."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mission Records", f"{len(df):,}")
    c2.metric("Dataset Features", f"{len(df.columns)}")
    c3.metric("Mission Duration", f"{duration_min:.1f} min")
    c4.metric("Missing Values", f"{int(df.isna().sum().sum()):,}")

    heading("Telemetry Snapshot", "First 10 rows from the mission.")
    st.dataframe(df.head(10), use_container_width=True)

    if "Altitude" in df.columns:
        x_col = "time" if "time" in df.columns else df.index
        fig = px.area(
            df,
            x=x_col,
            y="Altitude",
            title="UAV Altitude Profile",
            color_discrete_sequence=[COLORS[0]]
        )
        add_line_axis_titles(fig, "Time", "Altitude")
        st.plotly_chart(
            style_figure(fig),
            use_container_width=True
        )

    if has_columns(df, SNR_COLS):
        heading(
            "Signal Quality Snapshot",
            "Compare recorded signal strength across base stations."
        )

        snr_long = df.melt(
            id_vars=["time"] if "time" in df.columns else None,
            value_vars=SNR_COLS,
            var_name="Base Station",
            value_name="SNR (dB)"
        )

        fig = px.box(
            snr_long,
            x="Base Station",
            y="SNR (dB)",
            color="Base Station",
            title="SNR Distribution by Base Station",
            color_discrete_sequence=COLORS
        )
        add_line_axis_titles(fig, "Base Station", "SNR (dB)")
        st.plotly_chart(
            style_figure(fig),
            use_container_width=True
        )

# =========================================================
# COMMUNICATION MONITORING
# =========================================================
elif page == "Communication Monitoring":
    heading(
        "Communication Monitoring",
        "Explore recorded signal strength and download history."
    )

    if "time" not in df.columns:
        st.warning("The dataset does not contain a time column.")
        st.stop()

    view = df.dropna(subset=["time"]).sort_values("time").copy()

    if view.empty:
        st.warning("No valid timestamps were found.")
        st.stop()

    snr_cols = [c for c in SNR_COLS if c in view.columns]
    download_cols = [c for c in DOWNLOAD_COLS if c in view.columns]

    if snr_cols:
        heading("Signal-to-Noise Ratio")

        snr_long = view[["time"] + snr_cols].melt(
            id_vars="time",
            var_name="Base Station",
            value_name="SNR (dB)"
        )

        fig = px.line(
            snr_long,
            x="time",
            y="SNR (dB)",
            color="Base Station",
            title="Signal Quality Over Mission Time",
            color_discrete_sequence=COLORS
        )
        add_line_axis_titles(fig, "Time", "SNR (dB)")
        st.plotly_chart(
            style_figure(fig),
            use_container_width=True
        )
    else:
        st.info("No SNR columns were found.")

    if download_cols:
        heading("Cumulative Data Download")

        dl_long = view[["time"] + download_cols].melt(
            id_vars="time",
            var_name="Base Station",
            value_name="Download (Mbits)"
        )

        fig = px.line(
            dl_long,
            x="time",
            y="Download (Mbits)",
            color="Base Station",
            title="Cumulative Download by Base Station",
            color_discrete_sequence=COLORS
        )
        add_line_axis_titles(fig, "Time", "Download (Mbits)")
        st.plotly_chart(
            style_figure(fig),
            use_container_width=True
        )

        last_values = view[download_cols].iloc[-1]
        cards = st.columns(len(download_cols))

        for i, col in enumerate(download_cols):
            with cards[i]:
                value = pd.to_numeric(last_values[col], errors="coerce")
                st.metric(
                    col.upper(),
                    f"{value:.2f} Mbits"
                    if pd.notna(value) else "N/A"
                )
    else:
        st.info("No download columns were found.")

    st.caption(
        "These graphs show recorded CSV telemetry, not a live UAV feed."
    )

# =========================================================
# MODEL PERFORMANCE
# =========================================================
elif page == "Live Prediction":
    heading(
        "Live Prediction",
        "Estimate the four downlink outputs from UAV telemetry and communication conditions."
    )

    model_path = FILES / "fidelity_net_weights.pth"
    scaler_x_path = FILES / "scaler_X.pkl"
    scaler_y_path = FILES / "scaler_y.pkl"

    missing_files = [
        path.name for path in (model_path, scaler_x_path, scaler_y_path)
        if not path.exists()
    ]

    if missing_files:
        st.error(
            "Required model files are missing from uav_dashboard_files: "
            + ", ".join(missing_files)
        )
    else:
        missing_columns = [col for col in FEATURES if col not in df.columns]

        if missing_columns:
            st.error(
                "The loaded dataset is missing required input columns: "
                + ", ".join(missing_columns)
            )
        else:
            st.markdown(
                "Set the input values below, then select **Predict outputs**. "
                "Defaults are based on the loaded mission dataset."
            )

            numeric_features = df[FEATURES].apply(pd.to_numeric, errors="coerce")
            ranges = numeric_features.agg(["min", "max"])
            defaults = numeric_features.median()

            with st.form("live_prediction_form"):
                st.subheader("Communication conditions")
                values = {}
                left, right = st.columns(2)

                for i, feature in enumerate(FEATURES):
                    low = ranges.loc["min", feature]
                    high = ranges.loc["max", feature]
                    default = defaults[feature]
                    column = left if i % 2 == 0 else right

                    if not np.isfinite(default):
                        default = 0.0
                    if not np.isfinite(low):
                        low = default
                    if not np.isfinite(high):
                        high = default

                    with column:
                        if np.isclose(low, high):
                            values[feature] = st.number_input(
                                feature,
                                value=float(default),
                                format="%.5f",
                                key=f"live_{feature}"
                            )
                        else:
                            values[feature] = st.number_input(
                                feature,
                                min_value=float(low),
                                max_value=float(high),
                                value=float(np.clip(default, low, high)),
                                format="%.5f",
                                key=f"live_{feature}"
                            )

                submitted = st.form_submit_button(
                    "Predict outputs",
                    type="primary",
                    use_container_width=True
                )

            if submitted:
                try:
                    scaler_X = joblib.load(scaler_x_path)
                    scaler_y = joblib.load(scaler_y_path)

                    input_df = pd.DataFrame(
                        [[values[feature] for feature in FEATURES]],
                        columns=FEATURES
                    )
                    scaled_input = scaler_X.transform(input_df)

                    model = FidelityNet()
                    try:
                        state_dict = torch.load(
                            model_path,
                            map_location="cpu",
                            weights_only=True
                        )
                    except TypeError:
                        state_dict = torch.load(model_path, map_location="cpu")

                    model.load_state_dict(state_dict)
                    model.eval()

                    with torch.no_grad():
                        tensor = torch.tensor(
                            scaled_input,
                            dtype=torch.float32
                        )
                        scaled_prediction = model(tensor).cpu().numpy()

                    prediction = scaler_y.inverse_transform(
                        scaled_prediction
                    )[0]

                    st.success("Prediction completed.")
                    st.subheader("Predicted downlink outputs")

                    metric_columns = st.columns(4)
                    for i, target in enumerate(TARGETS):
                        metric_columns[i].metric(
                            target.upper(),
                            f"{prediction[i]:.3f}"
                        )

                    prediction_df = pd.DataFrame({
                        "Output": TARGETS,
                        "Predicted value": prediction
                    })

                    fig = px.bar(
                        prediction_df,
                        x="Output",
                        y="Predicted value",
                        color="Output",
                        text="Predicted value",
                        title="Predicted download value by communication link",
                        color_discrete_sequence=COLORS
                    )
                    fig.update_traces(
                        texttemplate="%{text:.3f}",
                        textposition="outside",
                        cliponaxis=False
                    )
                    add_line_axis_titles(
                        fig, "Communication Link", "Predicted Download Value"
                    )
                    st.plotly_chart(
                        style_figure(fig),
                        use_container_width=True
                    )

                    # Flag inputs outside the observed mission range if a future
                    # input mechanism permits them; the current form uses bounds.
                    st.download_button(
                        "Download prediction CSV",
                        data=prediction_df.to_csv(index=False).encode("utf-8"),
                        file_name="uav_live_prediction.csv",
                        mime="text/csv"
                    )

                    st.caption(
                        "This page performs manual-input model inference. "
                        "It is not connected to a live UAV feed. Predictions "
                        "are estimates and should be interpreted in light of "
                        "the model's training data and validation results."
                    )
                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")

elif page == "What-if Simulator":
    heading(
        "What-if Simulator",
        "Change one or more UAV conditions and compare the model's predictions with a baseline scenario."
    )

    model_path = FILES / "fidelity_net_weights.pth"
    scaler_x_path = FILES / "scaler_X.pkl"
    scaler_y_path = FILES / "scaler_y.pkl"
    missing_files = [p.name for p in (model_path, scaler_x_path, scaler_y_path) if not p.exists()]

    if missing_files:
        st.error("Required model files are missing from uav_dashboard_files: " + ", ".join(missing_files))
    else:
        # Use dataset medians when available; otherwise use scaler training means
        # so this page does not depend on uploading or loading a CSV.
        if not df.empty and all(col in df.columns for col in FEATURES):
            feature_frame = df[FEATURES].apply(pd.to_numeric, errors="coerce")
            defaults = feature_frame.median().fillna(0.0)
            ranges = feature_frame.agg(["min", "max"])
        else:
            scaler_X = joblib.load(scaler_x_path)
            centers = getattr(scaler_X, "mean_", np.zeros(len(FEATURES)))
            defaults = pd.Series(centers, index=FEATURES).fillna(0.0)
            ranges = pd.DataFrame({"min": defaults - 10, "max": defaults + 10})
            st.info("No local mission CSV found. Inputs use scaler-based defaults; adjust them to explore scenarios.")

        st.markdown("#### Baseline scenario")
        baseline = {}
        with st.form("whatif_baseline_form"):
            left, right = st.columns(2)
            for i, feature in enumerate(FEATURES):
                low = pd.to_numeric(ranges.loc["min", feature], errors="coerce")
                high = pd.to_numeric(ranges.loc["max", feature], errors="coerce")
                default = float(defaults[feature])
                if not np.isfinite(low): low = default - 10
                if not np.isfinite(high): high = default + 10
                if high <= low:
                    low, high = default - 1, default + 1
                with (left if i % 2 == 0 else right):
                    baseline[feature] = st.number_input(
                        feature, min_value=float(low), max_value=float(high),
                        value=float(np.clip(default, low, high)), format="%.5f",
                        key=f"whatif_base_{feature}"
                    )
            baseline_submitted = st.form_submit_button("Set baseline", use_container_width=True)

        # Store baseline values across reruns; use the form's submitted values
        # on first run and the saved session state thereafter.
        if baseline_submitted or "whatif_baseline_values" not in st.session_state:
            st.session_state["whatif_baseline_values"] = baseline.copy()
        baseline_values = st.session_state["whatif_baseline_values"]

        st.markdown("#### Scenario to compare")
        st.caption("Change any inputs below. Values are kept within the observed dataset range when that dataset is available.")
        scenario = {}
        with st.form("whatif_scenario_form"):
            left, right = st.columns(2)
            for i, feature in enumerate(FEATURES):
                low = pd.to_numeric(ranges.loc["min", feature], errors="coerce")
                high = pd.to_numeric(ranges.loc["max", feature], errors="coerce")
                default = float(baseline_values.get(feature, defaults[feature]))
                if not np.isfinite(low): low = default - 10
                if not np.isfinite(high): high = default + 10
                if high <= low: low, high = default - 1, default + 1
                with (left if i % 2 == 0 else right):
                    scenario[feature] = st.number_input(
                        feature, min_value=float(low), max_value=float(high),
                        value=float(np.clip(default, low, high)), format="%.5f",
                        key=f"whatif_scenario_{feature}"
                    )
            compare_submitted = st.form_submit_button("Compare scenarios", type="primary", use_container_width=True)

        if compare_submitted:
            try:
                scaler_X = joblib.load(scaler_x_path)
                scaler_y = joblib.load(scaler_y_path)
                model = FidelityNet()
                try:
                    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
                except TypeError:
                    state_dict = torch.load(model_path, map_location="cpu")
                model.load_state_dict(state_dict)
                model.eval()
                input_df = pd.DataFrame(
                    [[baseline_values[f] for f in FEATURES], [scenario[f] for f in FEATURES]],
                    columns=FEATURES
                )
                with torch.no_grad():
                    scaled_x = scaler_X.transform(input_df)
                    scaled_y = model(torch.tensor(scaled_x, dtype=torch.float32)).cpu().numpy()
                predictions = scaler_y.inverse_transform(scaled_y)
                comparison = pd.DataFrame({
                    "Output": TARGETS,
                    "Baseline": predictions[0],
                    "What-if scenario": predictions[1],
                })
                comparison["Change"] = comparison["What-if scenario"] - comparison["Baseline"]
                comparison["Change (%)"] = np.where(
                    np.abs(comparison["Baseline"]) > 1e-9,
                    comparison["Change"] / comparison["Baseline"] * 100,
                    np.nan
                )
                st.markdown("#### Predicted output comparison")
                cols = st.columns(4)
                for i, target in enumerate(TARGETS):
                    cols[i].metric(target.upper(), f"{predictions[1, i]:.3f}", f"{comparison.loc[i, 'Change']:+.3f} vs baseline")
                fig = go.Figure()
                fig.add_trace(go.Bar(name="Baseline", x=TARGETS, y=predictions[0]))
                fig.add_trace(go.Bar(name="What-if scenario", x=TARGETS, y=predictions[1]))
                fig.update_layout(barmode="group", title="Baseline vs what-if predicted downlink values", xaxis_title="Communication link", yaxis_title="Predicted download value")
                st.plotly_chart(style_figure(fig), use_container_width=True)
                st.dataframe(comparison.round(4), use_container_width=True, hide_index=True)
                st.download_button(
                    "Download scenario comparison",
                    data=comparison.to_csv(index=False).encode("utf-8"),
                    file_name="uav_what_if_comparison.csv", mime="text/csv"
                )
                st.caption("This is a model-based scenario comparison, not a live UAV control action. Predictions depend on the training data and may be unreliable for inputs outside the model's training distribution.")
            except Exception as exc:
                st.error(f"Scenario comparison failed: {exc}")

elif page == "Model Performance":
    heading(
        "Model Performance",
        "Review exported fidelity scores and regression metrics."
    )

    if fidelity_df is not None and not fidelity_df.empty:
        required = {"Link", "MAE", "Fidelity_Score"}

        if required.issubset(fidelity_df.columns):
            c1, c2 = st.columns(2)

            with c1:
                fig = px.bar(
                    fidelity_df,
                    x="Link",
                    y="Fidelity_Score",
                    color="Link",
                    text="Fidelity_Score",
                    title="Project-Specific Fidelity Score",
                    color_discrete_sequence=COLORS
                )
                fig.update_traces(
                    texttemplate="%{text:.2f}",
                    textposition="outside",
                    cliponaxis=False
                )
                add_line_axis_titles(
                    fig, "Communication Link", "Fidelity Score"
                )
                st.plotly_chart(
                    style_figure(fig),
                    use_container_width=True
                )

            with c2:
                fig = px.bar(
                    fidelity_df,
                    x="Link",
                    y="MAE",
                    color="Link",
                    text="MAE",
                    title="Mean Absolute Error",
                    color_discrete_sequence=COLORS
                )
                fig.update_traces(
                    texttemplate="%{text:.2f}",
                    textposition="outside",
                    cliponaxis=False
                )
                add_line_axis_titles(
                    fig, "Communication Link", "MAE"
                )
                st.plotly_chart(
                    style_figure(fig),
                    use_container_width=True
                )

            st.dataframe(fidelity_df, use_container_width=True)

            st.caption(
                "Fidelity_Score is the project-specific score saved from "
                "your notebook. It is not a classification accuracy."
            )
        else:
            st.warning("The fidelity CSV has unexpected columns.")
    else:
        st.info("fidelity_results.csv was not found.")

    st.divider()
    heading(
        "Monte Carlo Dropout Regression",
        "Calculated from the saved prediction array and test targets."
    )

    compatible = (
        mc_predictions is not None
        and y_test is not None
        and mc_predictions.ndim == 3
        and mc_predictions.shape[2] == 4
        and len(y_test) == mc_predictions.shape[1]
        and has_columns(y_test, TARGETS)
    )

    if compatible:
        mc_mean = mc_predictions.mean(axis=0)
        rows = []

        for i, target in enumerate(TARGETS):
            actual = y_test[target].to_numpy()
            pred = mc_mean[:, i]

            rows.append({
                "Target": target,
                "MAE": mean_absolute_error(actual, pred),
                "RMSE": np.sqrt(mean_squared_error(actual, pred)),
                "R²": r2_score(actual, pred)
                if len(actual) > 1 else np.nan
            })

        metric_df = pd.DataFrame(rows)

        st.dataframe(
            metric_df.style.format({
                "MAE": "{:.3f}",
                "RMSE": "{:.3f}",
                "R²": "{:.3f}"
            }),
            use_container_width=True
        )

        target = st.selectbox(
            "Select target",
            TARGETS,
            key="performance_target"
        )

        idx = TARGETS.index(target)
        actual = y_test[target].to_numpy()
        pred = mc_mean[:, idx]

        compare_df = pd.DataFrame({
            "Test Record": np.arange(len(actual)),
            "Actual": actual,
            "MC Mean Prediction": pred
        })

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=compare_df["Test Record"],
            y=compare_df["Actual"],
            name="Actual",
            mode="lines+markers",
            line=dict(color=COLORS[0], width=3)
        ))

        fig.add_trace(go.Scatter(
            x=compare_df["Test Record"],
            y=compare_df["MC Mean Prediction"],
            name="MC Mean Prediction",
            mode="lines",
            line=dict(color=COLORS[2], width=3, dash="dash")
        ))

        fig.update_layout(title=f"{target}: Actual vs Predicted")
        add_line_axis_titles(fig, "Test Record", "Download Value")
        st.plotly_chart(
            style_figure(fig),
            use_container_width=True
        )
    else:
        st.info(
            "Compatible MC predictions and y_test.csv are needed "
            "to calculate regression metrics."
        )

# =========================================================
# UNCERTAINTY ANALYSIS
# =========================================================
elif page == "Uncertainty Analysis":
    heading(
        "Uncertainty Analysis",
        "Explore Monte Carlo Dropout variation across test records."
    )

    compatible = (
        mc_predictions is not None
        and y_test is not None
        and mc_predictions.ndim == 3
        and mc_predictions.shape[2] == 4
        and len(y_test) == mc_predictions.shape[1]
        and has_columns(y_test, TARGETS)
    )

    if not compatible:
        st.warning(
            "Compatible MC predictions and test targets are required."
        )
        st.stop()

    mc_mean = mc_predictions.mean(axis=0)
    mc_std = mc_predictions.std(axis=0)

    target = st.selectbox(
        "Select communication link",
        TARGETS,
        key="uncertainty_target"
    )

    idx = TARGETS.index(target)
    actual = y_test[target].to_numpy()
    pred = mc_mean[:, idx]
    std = mc_std[:, idx]
    abs_error = np.abs(actual - pred)

    uncertainty_df = pd.DataFrame({
        "Test Record": np.arange(len(actual)),
        "Actual": actual,
        "MC Mean Prediction": pred,
        "MC Standard Deviation": std,
        "Absolute Error": abs_error
    })

    c1, c2, c3 = st.columns(3)

    c1.metric("Mean Absolute Error", f"{abs_error.mean():.3f}")
    c2.metric("Mean MC Standard Deviation", f"{std.mean():.3f}")

    correlation = (
        np.corrcoef(std, abs_error)[0, 1]
        if np.std(std) > 0 and np.std(abs_error) > 0
        else np.nan
    )

    c3.metric(
        "Error–Uncertainty Correlation",
        f"{correlation:.3f}" if np.isfinite(correlation) else "N/A"
    )

    heading("Actual vs Predicted")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=uncertainty_df["Test Record"],
        y=uncertainty_df["Actual"],
        name="Actual",
        mode="lines+markers",
        line=dict(color=COLORS[0], width=3)
    ))

    fig.add_trace(go.Scatter(
        x=uncertainty_df["Test Record"],
        y=uncertainty_df["MC Mean Prediction"],
        name="MC Mean",
        mode="lines",
        line=dict(color=COLORS[1], width=3, dash="dash")
    ))

    fig.update_layout(title=f"{target}: Actual vs MC Mean")
    add_line_axis_titles(fig, "Test Record", "Download Value")

    st.plotly_chart(
        style_figure(fig),
        use_container_width=True
    )

    c1, c2 = st.columns(2)

    with c1:
        fig = px.scatter(
            uncertainty_df,
            x="Test Record",
            y="MC Standard Deviation",
            color="Absolute Error",
            color_continuous_scale="Plasma",
            title="Prediction Uncertainty by Record",
            hover_data=["Actual", "MC Mean Prediction"]
        )
        add_line_axis_titles(
            fig, "Test Record", "MC Standard Deviation"
        )
        st.plotly_chart(
            style_figure(fig),
            use_container_width=True
        )

    with c2:
        fig = px.scatter(
            uncertainty_df,
            x="MC Standard Deviation",
            y="Absolute Error",
            color="Absolute Error",
            color_continuous_scale="Viridis",
            title="Uncertainty vs Absolute Error",
            hover_data=["Test Record"]
        )
        add_line_axis_titles(
            fig, "MC Standard Deviation", "Absolute Error"
        )
        st.plotly_chart(
            style_figure(fig),
            use_container_width=True
        )

    st.dataframe(
        uncertainty_df.head(15),
        use_container_width=True
    )

    st.caption(
        "MC standard deviation is an uncertainty estimate. "
        "It is not automatically a calibrated confidence interval."
    )

# =========================================================
# CLASSIFICATION METRICS
# =========================================================
elif page == "Classification Metrics":
    from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

    heading(
        "Classification Metrics",
        "Per-link threshold analysis alongside continuous regression evaluation."
    )

    st.info(
        "The model predicts continuous download values. This page converts those values into "
        "Below/Above classes for exploratory analysis. There is no universal cutoff of 600: "
        "each link has its own scale. The default for each link is its 25th percentile in the "
        "available actual test values, shown as a provisional visualization threshold only. "
        "For formal reporting, set thresholds from an engineering requirement or validation data, "
        "not the held-out test set."
    )

    compatible = (
        mc_predictions is not None
        and y_test is not None
        and np.asarray(mc_predictions).ndim == 3
        and np.asarray(mc_predictions).shape[2] == len(TARGETS)
        and len(y_test) == np.asarray(mc_predictions).shape[1]
        and has_columns(y_test, TARGETS)
    )

    if compatible:
        mc_mean = np.asarray(mc_predictions, dtype=float).mean(axis=0)
        actual_all = y_test[TARGETS].to_numpy(dtype=float)
        pred_all = mc_mean
        thresholds = {}
        per_link_rows = []

        # Diagnose target quality before interpreting classification metrics.
        # Near-constant or capped test targets can make scores misleading even
        # when the confusion-matrix code is correct.
        st.markdown("### Test-data quality check")
        diagnostic_rows = []
        suspicious_targets = []
        for i, target in enumerate(TARGETS):
            values = actual_all[:, i]
            values = values[np.isfinite(values)]
            if values.size == 0:
                diagnostic_rows.append({
                    "Link": target, "Samples": 0, "Unique values": 0,
                    "Minimum": np.nan, "Median": np.nan, "Maximum": np.nan,
                    "Std. dev.": np.nan, "Status": "No finite values"
                })
                suspicious_targets.append(target)
                continue
            unique_count = int(np.unique(np.round(values, 8)).size)
            value_range = float(np.max(values) - np.min(values))
            std_value = float(np.std(values))
            suspicious = unique_count <= 2 or np.isclose(value_range, 0.0, atol=1e-8)
            status = "Check: nearly constant/capped" if suspicious else "Variation present"
            if suspicious:
                suspicious_targets.append(target)
            diagnostic_rows.append({
                "Link": target,
                "Samples": int(values.size),
                "Unique values": unique_count,
                "Minimum": float(np.min(values)),
                "Median": float(np.median(values)),
                "Maximum": float(np.max(values)),
                "Std. dev.": std_value,
                "Status": status
            })
        diagnostic_df = pd.DataFrame(diagnostic_rows)
        st.dataframe(diagnostic_df.style.format({
            "Minimum": "{:.3f}", "Median": "{:.3f}",
            "Maximum": "{:.3f}", "Std. dev.": "{:.6f}"
        }), use_container_width=True, hide_index=True)
        if suspicious_targets:
            st.warning(
                "The actual test targets for " + ", ".join(suspicious_targets) +
                " have no or very little variation. Classification and regression metrics for "
                "these links may be misleading. Verify the target creation, clipping/capping, "
                "preprocessing, and train/calibration/test split in the pipeline. Do not change "
                "the predictions just to produce an Above class; regenerate the saved test files "
                "from the corrected pipeline if needed."
            )

        st.markdown("### Thresholds by link")
        threshold_cols = st.columns(4)
        for i, target in enumerate(TARGETS):
            actual = actual_all[:, i]
            predicted = pred_all[:, i]
            valid = np.isfinite(actual) & np.isfinite(predicted)
            if not valid.any():
                thresholds[target] = 0.0
                threshold_cols[i].warning(f"{target}: no finite values")
                continue

            actual_valid = actual[valid]
            pred_valid = predicted[valid]
            lo = float(min(np.min(actual_valid), np.min(pred_valid)))
            hi = float(max(np.max(actual_valid), np.max(pred_valid)))
            if np.isclose(lo, hi):
                hi = lo + 1.0
            default = float(np.quantile(actual_valid, 0.25))
            default = float(np.clip(default, lo, hi))
            step = max((hi - lo) / 200.0, 0.001)
            with threshold_cols[i]:
                thresholds[target] = st.slider(
                    f"{target} threshold",
                    min_value=lo,
                    max_value=hi,
                    value=default,
                    step=step,
                    format="%.3f",
                    key=f"threshold_{target}",
                    help="Provisional per-link threshold. Choose a validated threshold for formal evaluation."
                )

        selected = st.selectbox(
            "Evaluate",
            ["All links combined"] + TARGETS,
            key="classification_link_select"
        )

        for i, target in enumerate(TARGETS):
            actual = actual_all[:, i]
            predicted = pred_all[:, i]
            valid = np.isfinite(actual) & np.isfinite(predicted)
            if not valid.any():
                continue
            cutoff = thresholds[target]
            y_true_link = (actual[valid] >= cutoff).astype(int)
            y_pred_link = (predicted[valid] >= cutoff).astype(int)
            per_link_rows.append({
                "Link": target,
                "Threshold": cutoff,
                "Actual Above": int(y_true_link.sum()),
                "Predicted Above": int(y_pred_link.sum()),
                "Precision": precision_score(y_true_link, y_pred_link, zero_division=0),
                "Recall": recall_score(y_true_link, y_pred_link, zero_division=0),
                "F1-score": f1_score(y_true_link, y_pred_link, zero_division=0),
                "Accuracy": accuracy_score(y_true_link, y_pred_link),
                "Samples": int(valid.sum())
            })

        if selected == "All links combined":
            true_parts, pred_parts = [], []
            for i, target in enumerate(TARGETS):
                actual = actual_all[:, i]
                predicted = pred_all[:, i]
                valid = np.isfinite(actual) & np.isfinite(predicted)
                if valid.any():
                    cutoff = thresholds[target]
                    true_parts.extend((actual[valid] >= cutoff).astype(int).tolist())
                    pred_parts.extend((predicted[valid] >= cutoff).astype(int).tolist())
            y_true = np.asarray(true_parts, dtype=int)
            y_pred = np.asarray(pred_parts, dtype=int)
        else:
            i = TARGETS.index(selected)
            actual = actual_all[:, i]
            predicted = pred_all[:, i]
            valid = np.isfinite(actual) & np.isfinite(predicted)
            cutoff = thresholds[selected]
            y_true = (actual[valid] >= cutoff).astype(int)
            y_pred = (predicted[valid] >= cutoff).astype(int)

        if y_true.size:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Precision", f"{precision_score(y_true, y_pred, zero_division=0):.3f}")
            c2.metric("Recall", f"{recall_score(y_true, y_pred, zero_division=0):.3f}")
            c3.metric("F1-score", f"{f1_score(y_true, y_pred, zero_division=0):.3f}")
            c4.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.3f}")

            st.markdown("### Confusion matrix")
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm,
                x=["Predicted: Below", "Predicted: Above"],
                y=["Actual: Below", "Actual: Above"],
                text=cm,
                texttemplate="%{text}",
                colorscale="Blues",
                showscale=True
            ))
            fig_cm.update_layout(
                xaxis_title="Predicted class",
                yaxis_title="Actual class",
                margin=dict(l=10, r=10, t=20, b=10)
            )
            st.plotly_chart(fig_cm, use_container_width=True)

            st.markdown("### Metrics by link")
            link_df = pd.DataFrame(per_link_rows)
            st.dataframe(link_df.style.format({
                "Threshold": "{:.3f}",
                "Precision": "{:.3f}",
                "Recall": "{:.3f}",
                "F1-score": "{:.3f}",
                "Accuracy": "{:.3f}"
            }), use_container_width=True, hide_index=True)

            # Direct diagnostic: distinguishes a threshold issue from a model/data issue.
            if selected != "All links combined":
                row = link_df.loc[link_df["Link"] == selected]
                if not row.empty and int(row.iloc[0]["Predicted Above"]) == 0:
                    i_selected = TARGETS.index(selected)
                    finite_pred = pred_all[:, i_selected]
                    finite_pred = finite_pred[np.isfinite(finite_pred)]
                    pred_range = (
                        f" Saved prediction range: {finite_pred.min():.3f} to {finite_pred.max():.3f}."
                        if finite_pred.size else " No finite saved predictions were found."
                    )
                    st.warning(
                        f"The model predicts no {selected} test values at or above the selected "
                        f"threshold ({thresholds[selected]:.3f})." + pred_range +
                        " This is a real result from the saved predictions, not a plotting error. "
                        "Check the data-quality table above and verify target scaling and test-file "
                        "alignment. Do not force positive labels to change the confusion matrix."
                    )

            st.markdown("### Regression metrics (primary project evaluation)")
            regression_rows = []
            for i, target in enumerate(TARGETS):
                actual = actual_all[:, i]
                predicted = pred_all[:, i]
                valid = np.isfinite(actual) & np.isfinite(predicted)
                if valid.any():
                    regression_rows.append({
                        "Target": target,
                        "MAE": mean_absolute_error(actual[valid], predicted[valid]),
                        "RMSE": float(np.sqrt(mean_squared_error(actual[valid], predicted[valid]))),
                        "R²": r2_score(actual[valid], predicted[valid]) if valid.sum() > 1 else np.nan
                    })
            st.dataframe(pd.DataFrame(regression_rows).style.format({
                "MAE": "{:.3f}", "RMSE": "{:.3f}", "R²": "{:.3f}"
            }), use_container_width=True, hide_index=True)
            st.warning(
                "These classification metrics are exploratory because the classes are created from "
                "continuous regression outputs. The 25th-percentile defaults use actual test values "
                "for display only; use validation-set thresholds or documented engineering limits "
                "before presenting formal classification results."
            )
        else:
            st.warning("No finite actual or predicted values are available for this evaluation.")
    else:
        st.warning(
            "Compatible saved test targets and MC prediction results are needed. "
            "Check y_test.csv and mc_predictions.npy in uav_dashboard_files."
        )

# =========================================================
# FOOTER
# =========================================================
st.divider()

st.markdown("""
<div class="footer">
    UAV Communication Digital Twin · Research Analytics Dashboard
</div>
""", unsafe_allow_html=True)

import pandas as pd
import streamlit as st
from models.autoencoder import Autoencoder
import torch
import torch.nn as nn
import plotly.express as px
from streamlit_autorefresh import st_autorefresh

if "errors" not in st.session_state:
    st.session_state.errors = []

if "timestamps" not in st.session_state:
    st.session_state.timestamps = []

if "alerts" not in st.session_state:
    st.session_state.alerts = []

@st.cache_data
def load_data():
    return pd.read_csv("data/demo_test.csv")

df = load_data()

@st.cache_resource
def load_model():
    model = Autoencoder(input_dim=51, encoding_dim=5)
    model.load_state_dict(torch.load("src/models/autoencoder.pth", map_location=torch.device('cpu')))
    model.eval()
    loss_function = nn.MSELoss()
    return model, loss_function

st_autorefresh(interval=1000, key="monitor")

def monitor():
    if "idx" not in st.session_state:
        st.session_state.idx = 0
        
    row = df.iloc[st.session_state.idx]
    model, loss_function = load_model()
    row_features = row.drop(labels=["Timestamp", "Normal/Attack"])
    # convert row to series with float values
    row_features = row_features.astype(float)
    features = torch.tensor(row_features.values, dtype=torch.float32)
    features = features.unsqueeze(0)
    with torch.no_grad():
        reconstructed = model(features)
        loss = loss_function(reconstructed, features)

    st.session_state.errors.append(loss.item())
    st.session_state.timestamps.append(row["Timestamp"])

    WINDOW = 100

    st.session_state.errors = (st.session_state.errors[-WINDOW:])
    st.session_state.timestamps = (st.session_state.timestamps[-WINDOW:])

    #threshold should be 95th percentile of training errors
    training_errors = torch.load("src/models/training_errors.pt", map_location=torch.device('cpu'))
    THRESHOLD = torch.quantile(torch.tensor(training_errors), 0.95).item()

    # calculate error confidence interval
    error_tensor = torch.tensor(st.session_state.errors)
    CI = 1.96 * error_tensor.std() / torch.sqrt(torch.tensor(len(error_tensor)))

    if loss.item() > THRESHOLD:
        st.session_state.alerts.append({
            "timestamp": row["Timestamp"],
            "error": float(loss.item()),
            "CI": float(CI)
        })
        
    ci_map = {a["timestamp"]: a["CI"] for a in st.session_state.alerts}
    ci_list = [ci_map.get(t, None) for t in st.session_state.timestamps]

    chart_df = pd.DataFrame({
        "timestamp": st.session_state.timestamps,
        "error": st.session_state.errors,
        "CI": ci_list
    })

    chart_df["timestamp"] = pd.to_datetime(chart_df["timestamp"], errors="coerce")

    fig = px.line(
        chart_df,
        x="timestamp",
        y="error",
        title="Reconstruction Error"
        
        )
    fig.add_hline(y=THRESHOLD, line_dash="dash")

    anomalies = chart_df[chart_df["error"] > THRESHOLD]

    fig.add_scatter(
        x=anomalies["timestamp"],
        y=anomalies["error"],
        mode="markers",
        name="Anomalies"
        )

    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(pd.DataFrame(st.session_state.alerts).tail(20))

    st.session_state.idx += 1

monitor()
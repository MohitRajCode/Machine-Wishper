import streamlit as st
import sys
from pathlib import Path

# Make the src folder importable
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from predict import predict_failure

st.set_page_config(page_title="MachineWhisperer", page_icon="⚙️", layout="wide")

st.title(" MachineWhisperer")
st.subheader("Predictive Maintenance Dashboard")
st.write("Enter machine sensor readings to estimate the risk of machine failure.")

st.divider()

# Sensor input section
st.header("Machine Sensor Readings")

col1, col2 = st.columns(2)

with col1:
    machine_type = st.selectbox(
        "Machine Type",
        options=["L", "M", "H"],
        index=1,
        help="Select the machine product type.",
    )

    air_temperature = st.number_input(
        "Air Temperature (K)", min_value=250.0, max_value=350.0, value=300.0, step=0.1
    )

    process_temperature = st.number_input(
        "Process Temperature (K)",
        min_value=250.0,
        max_value=400.0,
        value=310.0,
        step=0.1,
    )

with col2:
    rotational_speed = st.number_input(
        "Rotational Speed (rpm)", min_value=0, max_value=5000, value=1500, step=10
    )

    torque = st.number_input(
        "Torque (Nm)", min_value=0.0, max_value=200.0, value=40.0, step=0.5
    )

    tool_wear = st.number_input(
        "Tool Wear (minutes)", min_value=0, max_value=500, value=100, step=1
    )

st.divider()

if st.button("Predict Machine Failure", type="primary"):
    result = predict_failure(
        machine_type=machine_type,
        air_temperature=air_temperature,
        process_temperature=process_temperature,
        rotational_speed=rotational_speed,
        torque=torque,
        tool_wear=tool_wear,
    )

    st.header("Prediction Result")

    probability_percent = result["failure_probability"] * 100

    col1, col2 = st.columns(2)

    with col1:
        if result["prediction"] == 1:
            st.error("⚠️ Potential Failure Detected")
        else:
            st.success("✅ No Failure Predicted")

    with col2:
        st.metric("Estimated Failure Score", f"{probability_percent:.2f}%")

    st.progress(min(max(float(result["failure_probability"]), 0.0), 1.0))

    st.caption(
        f"Decision threshold: {result['threshold']:.2f} "
        "(40% means the model flags readings with a score of "
        "40% or higher)."
    )

    st.warning(
        "This is a machine-learning estimate, not a guarantee "
        "of machine safety. Use validated sensor data and "
        "engineering checks before making maintenance decisions."
    )

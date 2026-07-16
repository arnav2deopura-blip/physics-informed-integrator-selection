import os
import sys
import math
from pathlib import Path
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import urllib.request

# 1. Setup paths so we scan both the root and 'src' folders safely
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# 2. Import using the full package name. 
# This tells Python that 'orbit_ml' is the parent, fixing the relative import error!
import orbit_ml.config as config
import orbit_ml.physics as physics

# Pull out the exact functions the dashboard code expects
MODEL_NAME = config.MODEL_NAME
PHYSICS_FEATURE_COLUMNS = config.PHYSICS_FEATURE_COLUMNS
GM = config.GM

add_orbit_features = physics.add_orbit_features
physics_run_sim = physics.run_sim_diagnostics
step_euler = physics.step_euler
step_rk4 = physics.step_rk4
step_leapfrog = physics.step_leapfrog
get_energy = physics.get_energy
get_angular_momentum = physics.get_angular_momentum

# Page Configurations (The Agency Visual Polish)
st.set_page_config(
    page_title="Physics-Informed ML Orbital Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache the ML Model so it doesn't reload on every slider move
@st.cache_resource
def load_ml_model():
    model_path = ROOT / "orbit_integrator_rf.joblib"
    
    # If the model file isn't in the GitHub repo, download it dynamically on the cloud server!
    if not model_path.exists():
        # REPLACE THIS URL WITH YOUR DIRECT DOWNLOAD LINK
        url = "https://drive.google.com/uc?export=download&id=1x3VQUxWwIEWFekePihidyMiJjMM-9iBS"
        with st.spinner("Downloading ML model weight files from cloud storage..."):
            urllib.request.urlretrieve(url, model_path)
            
    return joblib.load(model_path)

model = load_ml_model()


# --- DEFINE ST.FRAGMENT FOR THE MANUAL SLIDER (PREVENTS PAGE SCROLLING JUMPS) ---
@st.fragment
def render_manual_simulation_block(init_state, sim_time, perturb_eps, dt_rk4):
    # Give the user options to deliberately "break" the system
    st.markdown("---")
    st.subheader("Manual Overrides (Stress Testing)")
    st.markdown("Want to show a client or college admissions panel what happens when you don't use the AI model? Override the step sizes manually below:")
    
    # Ensure there is a persistent placeholder tracking your manual slider adjustment
    if "user_dt" not in st.session_state:
        st.session_state.user_dt = float(dt_rk4)

    # Every time the slider moves, it permanently updates the session state
    manual_dt = st.slider(
        "Force Manual Timestep Size (Δt)", 
        min_value=0.001, 
        max_value=0.5, 
        value=st.session_state.user_dt, 
        step=0.005,
        key="manual_dt_slider"
    )
    
    # Sync the selection to your variable
    st.session_state.user_dt = manual_dt
    
    # Run manual stress tests
    mx, my, m_energy, _, _, m_times = physics_run_sim(
        step_rk4, 
        init_state, 
        manual_dt, 
        sim_time, 
        perturb_eps, 
        track={"path_x", "path_y", "energy", "times"}
    )
    
    # Calculate Energy Drift Error
    e0 = get_energy(init_state, perturb_eps)
    energy_errors = [abs((e - e0) / (e0 if e0 != 0 else 1e-12)) for e in m_energy]

    m_col1, m_col2 = st.columns([2, 1])
    with m_col1:
        fig_manual = go.Figure()
        fig_manual.add_trace(go.Scatter(x=[0], y=[0], mode='markers', marker=dict(size=12, color='gold'), name='Central Mass'))
        fig_manual.add_trace(go.Scatter(x=mx, y=my, mode='lines', name='RK4 with Manual Δt', line=dict(color='fuchsia', width=2.5)))
        fig_manual.update_layout(title="Manual Step-Size Simulation Path", template="plotly_dark", yaxis=dict(scaleanchor="x", scaleratio=1))
        st.plotly_chart(fig_manual, use_container_width=True)
        
    with m_col2:
        fig_err = go.Figure()
        fig_err.add_trace(go.Scatter(x=m_times, y=energy_errors, mode='lines', line=dict(color='red')))
        fig_err.update_layout(
            title="Relative Energy Error Over Time",
            xaxis_title="Time (s)",
            yaxis_title="Relative Error",
            yaxis_type="log", # Error explodes exponentially, log scale captures it beautifully
            template="plotly_dark"
        )
        st.plotly_chart(fig_err, use_container_width=True)


# --- HEADER INTERFACE ---
st.title("AI-Driven Stability Selection in Orbital Mechanics")
st.markdown("""
This interactive dashboard bridges Machine Learning and Numerical Analysis. 
Adjust the initial conditions in the sidebar. A Physics-Informed Random Forest Regressor will instantly predict the maximum stable timestep (Δt) for three classic numerical integrators before running the live simulation to verify the results.
""")

# --- SIDEBAR CONTROL PANEL ---
st.sidebar.header("Initial Orbital Conditions")

# Dropdown or sliders for setting initial state
init_x = 1.0
init_y = 0.0

st.sidebar.subheader("Initial Velocities")
vx = st.sidebar.slider(r"Initial Velocity X ($v_x$)", min_value=-2.0, max_value=2.0, value=0.0, step=0.05)
vy = st.sidebar.slider(r"Initial Velocity Y ($v_y$)", min_value=0.1, max_value=2.5, value=1.0, step=0.05)

st.sidebar.subheader("Simulation Scope")
sim_time = st.sidebar.slider("Total Simulation Time", min_value=1.0, max_value=100.0, value=20.0, step=1.0)
perturb_eps = st.sidebar.slider(r"Gravitational Perturbation ($\epsilon$)", min_value=0.0, max_value=0.01, value=0.002, step=0.0005)

# --- CALCULATE INTERMEDIATE PHYSICS FEATURES ---
r_init = math.hypot(init_x, init_y)
# Wrap in a temporary dataframe to pass through your existing physics feature generator
temp_df = pd.DataFrame([{
    "vx": vx, "vy": vy, "r": r_init, "sim-time": sim_time, "perturb_eps": perturb_eps
}])
temp_df = add_orbit_features(temp_df) # Uses your physics.py logic to compute ecc, period, etc.

eccentricity = temp_df.iloc[0]["ecc"]
orbital_period = temp_df.iloc[0]["period"]
orbit_count = temp_df.iloc[0]["orbit_count"]

# Display calculated physical metrics in the sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("Derived Physical Properties")
st.sidebar.metric(label="Calculated Eccentricity", value=f"{eccentricity:.3f}")
st.sidebar.metric(label="Orbital Period ($T$)", value=f"{orbital_period:.2f} s")
st.sidebar.metric(label="Total Target Orbits", value=f"{orbit_count:.1f}")

# --- SECTION 1: AI PREDICTIONS ---
st.subheader("Machine Learning Timestep Recommendations")

if model is None:
    st.error(f"Could not find `{MODEL_NAME}`. Run `python run_study.py` first to train and save the model.")
else:
    # Prepare inputs exactly how your Random Forest expects them
    features = temp_df[PHYSICS_FEATURE_COLUMNS].values
    predictions = model.predict(features)[0] # Returns [dt_euler, dt_rk4, dt_leapfrog]
    
    dt_euler, dt_rk4, dt_leapfrog = predictions

    # Create beautiful columns for metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Forward Euler Max Stable Δt", value=f"{dt_euler:.5f}", delta="1st Order (Unstable)")
    with col2:
        st.metric(label="Classical RK4 Max Stable Δt", value=f"{dt_rk4:.5f}", delta="4th Order (High Accuracy)", delta_color="inverse")
    with col3:
        st.metric(label="Leapfrog Max Stable Δt", value=f"{dt_leapfrog:.5f}", delta="2nd Order (Symplectic)", delta_color="off")

# --- SECTION 2: LIVE SIMULATION VERIFICATION ---
st.subheader("Live Simulation Verification")

if model is not None:
    # Run all three simulations using your backend physics.py script
    init_state = [init_x, init_y, vx, vy]
    
    # Run simulations using the predicted step sizes
    ex, ey, _, _, _, _ = physics_run_sim(step_euler, init_state, dt_euler, sim_time, perturb_eps)
    rkx, rky, _, _, _, _ = physics_run_sim(step_rk4, init_state, dt_rk4, sim_time, perturb_eps)
    lx, ly, _, _, _, _ = physics_run_sim(step_leapfrog, init_state, dt_leapfrog, sim_time, perturb_eps)

    # Build interactive Plotly chart for the orbits
    fig = go.Figure()
    
    # Central Star / Mass
    fig.add_trace(go.Scatter(x=[0], y=[0], mode='markers', marker=dict(size=15, color='gold'), name='Central Mass (GM=1)'))
    
    # Trajectories
    fig.add_trace(go.Scatter(x=ex, y=ey, mode='lines', name=f'Forward Euler (Δt={dt_euler:.4f})', line=dict(dash='dot', color='red')))
    fig.add_trace(go.Scatter(x=rkx, y=rky, mode='lines', name=f'RK4 (Δt={dt_rk4:.4f})', line=dict(color='deepskyblue', width=2.5)))
    fig.add_trace(go.Scatter(x=lx, y=ly, mode='lines', name=f'Leapfrog (Δt={dt_leapfrog:.4f})', line=dict(color='limegreen', width=2)))

    fig.update_layout(
        title="Orbital Trajectories Comparison",
        xaxis_title="Position X",
        yaxis_title="Position Y",
        height=600,
        template="plotly_dark",
        yaxis=dict(scaleanchor="x", scaleratio=1) # Forces 1:1 aspect ratio so circles aren't squished!
    )
    
    st.plotly_chart(fig, use_container_width=True)

    # --- CALL THE FRAGMENT FUNCTION HERE ---
    # This safely executes the manual slider block without letting the whole page reset/jump
    render_manual_simulation_block(init_state, sim_time, perturb_eps, dt_rk4)

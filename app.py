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
import io
from fpdf import FPDF

# setup paths to scan both the root and 'src' folders safely
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import orbit_ml.config as config
import orbit_ml.physics as physics

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

# Page Configurations
st.set_page_config(
    page_title="OrbitML: AI-Assisted Numerical Integrator Selection",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_ml_model():
    model_path = ROOT / MODEL_NAME
    if not model_path.exists():
        file_id = "1x3VQUxWwIEWFekePihidyMiJjMM-9iBS"
        url = f"https://drive.google.com/uc?id={file_id}"
        
        with st.spinner("Downloading ML model files from cloud storage..."):
            import gdown
            gdown.download(url, str(model_path), quiet=False)
    
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

model = load_ml_model()

class OrbitReport(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(31, 41, 55)
        self.cell(0, 10, "OrbitML Verification Report", ln=True, align="C")
        self.set_font("Helvetica", "I", 10)
        self.cell(0, 5, "Automated Math Integrity & Machine Learning Evaluation", ln=True, align="C")
        self.ln(10)
        
    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

def generate_pdf_report(orbit_regime, eccentricity, e_dt, r_dt, l_dt, e_err, r_err, l_err):
    pdf = OrbitReport()
    pdf.add_page()

    clean_regime = orbit_regime
    for emoji in ["🟢", "🟡", "🟠", "🔴"]:
        clean_regime = clean_regime.replace(emoji, "").strip()
        
    # Section 1: Physical Parameters
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "1. Target Orbital Environment", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"  - Classification Regime: {clean_regime}", ln=True)
    pdf.cell(0, 6, f"  - Computed System Eccentricity: {eccentricity:.4f}", ln=True)
    pdf.ln(5)
    
    # Section 2: ML Decisions
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "2. AI Optimization Thresholds", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"  - Forward Euler Predicted Safe Step (dt): {e_dt:.5f}", ln=True)
    pdf.cell(0, 6, f"  - Classical RK4 Predicted Safe Step (dt): {r_dt:.5f}", ln=True)
    pdf.cell(0, 6, f"  - Symplectic Leapfrog Predicted Safe Step (dt): {l_dt:.5f}", ln=True)
    pdf.ln(5)
    
    # Section 3: Diagnostic Results
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "3. Numerical Stability Matrix", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"  - Forward Euler Final Energy Accumulation Error: {e_err:.2e}", ln=True)
    pdf.cell(0, 6, f"  - Classical RK4 Final Cumulative Truncation Error: {r_err:.2e}", ln=True)
    pdf.cell(0, 6, f"  - Symplectic Leapfrog Preservation Energy Drift: {l_err:.2e}", ln=True)

    pdf_buffer = io.BytesIO()
    pdf_string = pdf.output()
    pdf_buffer.write(pdf_string)
    pdf_buffer.seek(0)
    return pdf_buffer

@st.fragment
def render_manual_simulation_block(init_state, sim_time, perturb_eps, dt_rk4):
    st.markdown("---")
    st.subheader("Manual Overrides (Stress Testing)")
    st.markdown("Want to show a client or college admissions panel what happens when you don't use the AI model? Override the step sizes manually below:")
    
    if "user_dt" not in st.session_state:
        st.session_state.user_dt = float(dt_rk4)

    manual_dt = st.slider(
        "Force Manual Timestep Size (Δt)", 
        min_value=0.001, 
        max_value=0.5, 
        value=st.session_state.user_dt, 
        step=0.005,
        key="manual_dt_slider"
    )
    
    st.session_state.user_dt = manual_dt
    
    mx, my, m_energy, _, _, m_times = physics_run_sim(
        step_rk4, 
        init_state, 
        manual_dt, 
        sim_time, 
        perturb_eps, 
        track={"path_x", "path_y", "energy", "times"}
    )
    
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
            yaxis_type="log", 
            template="plotly_dark"
        )
        st.plotly_chart(fig_err, use_container_width=True)


# header interface
st.title("OrbitML: AI-Assisted Integrator Selection")
st.caption("Bridging Machine Learning and Numerical Analysis for Orbital Dynamics Verification")

# sidebar control panel
st.sidebar.header("Initial Orbital Conditions")

# Match paper's radius sampling limits (0.5 to 2.0)
init_x = st.sidebar.slider("Initial Position X (Radius)", min_value=0.5, max_value=2.0, value=1.0, step=0.1)
init_y = 0.0

st.sidebar.subheader("Initial Velocities")
# Match paper's velocity sampling limits
vx = st.sidebar.slider(r"Initial Velocity X ($v_x$)", min_value=-0.5, max_value=0.5, value=0.0, step=0.05)
vy = st.sidebar.slider(r"Initial Velocity Y ($v_y$)", min_value=0.3, max_value=1.2, value=1.0, step=0.05)

st.sidebar.subheader("Simulation Scope")
sim_time = st.sidebar.slider("Total Simulation Time", min_value=1.0, max_value=100.0, value=20.0, step=1.0)

# Enforce unperturbed pure two-body system physics as strictly defined in the methodology
perturb_eps = 0.0

# calculate intermediate physics features
r_init = math.hypot(init_x, init_y)
temp_df = pd.DataFrame([{
    "vx": vx, "vy": vy, "r": r_init, "sim-time": sim_time, "perturb_eps": perturb_eps
}])
temp_df = add_orbit_features(temp_df) 

eccentricity = temp_df.iloc[0]["ecc"]
orbital_period = temp_df.iloc[0]["period"]
orbit_count = temp_df.iloc[0]["orbit_count"]

# orbit regime classification
if eccentricity < 0.05:
    orbit_regime = "🟢 Nearly Circular"
elif eccentricity < 0.90:
    orbit_regime = "🟡 Elliptical"
elif eccentricity < 1.0:
    orbit_regime = "🟠 Highly Eccentric (Near-Parabolic)"
else:
    orbit_regime = "🔴 Hyperbolic (Escape Trajectory)"

# Display calculated physical metrics in the sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("Derived Physical Properties")
st.sidebar.markdown(f"**Classification:** {orbit_regime}")
st.sidebar.metric(label="Calculated Eccentricity", value=f"{eccentricity:.3f}")
st.sidebar.metric(label="Orbital Period ($T$)", value=f"{orbital_period:.2f} s")
st.sidebar.metric(label="Total Target Orbits", value=f"{orbit_count:.1f}")

st.sidebar.markdown("---")
st.sidebar.markdown("**Link(s):** [GitHub Repo](https://github.com/arnav2deopura-blip/physics-informed-integrator-selection)")


# navigation tabs
tab1, tab2, tab3 = st.tabs(["Overview & Science", "Interactive Predictor", "Model Analytics"])

with tab1:
    st.subheader("Project Background & Objectives")
    st.markdown(r"""
    ### The Core Challenge
    In astrodynamics, evaluating complex orbits over extended periods requires numerical integration engines. If an integration timestep ($\Delta t$) chosen is too large, the system accumulates artificial numerical energy, leading to orbits that mathematically explode or unphysically collapse. Conversely, picking a timestep that is needlessly small wastes critical processing power.
    
    ### How OrbitML Solves It
    This platform maps initial physical orbital metrics directly to algorithmic stability boundaries. A Physics-Informed Random Forest Regressor acts as an optimization layer, instantly calculating the maximum safe timestep ($\Delta t$) for three distinct families of numerical integrators before any mathematical steps are calculated:
    
    1. **Forward Euler (1st-Order):** A simple explicit scheme. Computationally cheap per loop, but mathematically unsuited for long-term bound orbits as it continuously introduces artificial energy.
    2. **Classical Runge-Kutta (RK4, 4th-Order):** A highly accurate multi-stage mathematical engine. It drops energy errors significantly but requires multiple function calculations per step.
    3. **Leapfrog Integrator (2nd-Order):** A symplectic integrator. While technically lower order than RK4, it preserves the geometric phases and energy configurations of the physical system indefinitely over long runtimes.
    
    *Switch over to the Interactive Predictor tab to test your own orbital conditions live!*
    """)

with tab2:
    st.subheader("Machine Learning Timestep Recommendations")

    if model is None:
        st.error(f"Could not find `{MODEL_NAME}`. Run your training script first to save the model.")
    else:
        features = temp_df[PHYSICS_FEATURE_COLUMNS].values
        predictions = model.predict(features)[0] 
        
        dt_euler, dt_rk4, dt_leapfrog = predictions

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="Forward Euler Max Stable Δt", value=f"{dt_euler:.5f}", delta="1st Order (Unstable)")
        with col2:
            st.metric(label="Classical RK4 Max Stable Δt", value=f"{dt_rk4:.5f}", delta="4th Order (High Accuracy)", delta_color="inverse")
        with col3:
            st.metric(label="Leapfrog Max Stable Δt", value=f"{dt_leapfrog:.5f}", delta="2nd Order (Symplectic)", delta_color="off")

        st.markdown("---")
        rec_col1, rec_col2 = st.columns(2)
        
        with rec_col1:
            # Compute Uncertainty / Prediction Variance across individual trees
            try:
                # Query every individual tree in the Random Forest to see its unique prediction
                tree_predictions = np.array([tree.predict(features)[0] for tree in model.estimators_])
                # Calculate the standard deviation (spread) of the trees' choices
                prediction_variance = np.var(tree_predictions, axis=0)
                avg_variance = np.mean(prediction_variance)
                
                # Turn the variance number into an interpretable category
                if avg_variance < 0.001:
                    confidence_badge = "🟢 High Confidence"
                    variance_text = "Low (Trees are in high agreement)"
                elif avg_variance < 0.01:
                    confidence_badge = "🟡 Moderate Confidence"
                    variance_text = "Medium (Acceptable tree divergence)"
                else:
                    confidence_badge = "🔴 Low Confidence"
                    variance_text = "High (Model is outside nominal training boundaries)"
            except AttributeError:
                # Fallback if the loaded model doesn't support estimators
                confidence_badge = "⚪ Not Available"
                variance_text = "Unknown"

            # Display the Model Uncertainty Analytics
            st.markdown(f"#### AI Prediction Insight")
            st.markdown(f"**Model Status:** {confidence_badge}")
            st.markdown(f"**Prediction Variance:** `{variance_text}`")

        with rec_col2:
            # Determine the most computationally efficient choice programmatically
            max_dt = max(dt_euler, dt_rk4, dt_leapfrog)
            
            if max_dt == dt_rk4:
                recommended_integrator = "Classical RK4"
                reasoning = "It yields the largest predicted stable timestep window, minimizing total iterative loops required for accurate results."
            elif max_dt == dt_leapfrog:
                recommended_integrator = "Leapfrog (Symplectic)"
                reasoning = "It yields the largest predicted stable timestep window while maintaining geometric structural phase space conservation."
            else:
                recommended_integrator = "Forward Euler"
                reasoning = "It yields the largest predicted math step size, though caution is advised due to its first-order nature."

            st.info(f"**Recommended Integrator:** **{recommended_integrator}** because it offers the **largest predicted stable timestep** ({max_dt:.4f}), providing optimal throughput for ~{orbit_count:.1f} orbital periods.")

        # Physics Constraint Alert: Triggered dynamically by high eccentricity
        if eccentricity > 0.50:
            st.warning(
                f"⚠️ **Physics Constraint Alert:** Your current orbit has a high eccentricity (`e = {eccentricity:.2f}`). "
                "Forward Euler is mathematically guaranteed to fail (diverge) here because explicit first-order methods cannot "
                "conserve energy during high-acceleration closest approaches (the periapsis). The **Leapfrog** or **RK4** trajectories below should be more accurate."
            )
        st.markdown("---")
        # =========================================================================

    st.subheader("Live Simulation Verification")

    if model is not None:
        init_state = [init_x, init_y, vx, vy]
        
        # Capture the energy lists
        ex, ey, e_energy, _, _, _ = physics_run_sim(step_euler, init_state, dt_euler, sim_time, perturb_eps)
        rkx, rky, rk_energy, _, _, _ = physics_run_sim(step_rk4, init_state, dt_rk4, sim_time, perturb_eps)
        lx, ly, l_energy, _, _, _ = physics_run_sim(step_leapfrog, init_state, dt_leapfrog, sim_time, perturb_eps)

        # Calculate local relative energy arrays for color mapping
        e0 = get_energy(init_state, perturb_eps)
        safe_e0 = e0 if e0 != 0 else 1e-12
        
        euler_local_err = [abs((e - e0) / safe_e0) for e in e_energy]
        rk4_local_err = [abs((e - e0) / safe_e0) for e in rk_energy]
        leapfrog_local_err = [abs((e - e0) / safe_e0) for e in l_energy]

        fig = go.Figure()
        
        # Central Gravity Source
        fig.add_trace(go.Scatter(x=[0], y=[0], mode='markers', marker=dict(size=15, color='gold'), name='Central Mass'))
        
        # Forward Euler
        fig.add_trace(go.Scatter(
            x=ex, y=ey, mode='markers+lines',
            name='Forward Euler',
            line=dict(width=1, color='rgba(200,50,50,0.3)'),
            marker=dict(
                size=4, color=euler_local_err, colorscale='Reds', 
                showscale=False, cmin=0, cmax=1e-1
            )
        ))
        
        # RK4
        fig.add_trace(go.Scatter(
            x=rkx, y=rky, mode='markers+lines',
            name='Classical RK4',
            line=dict(width=1.5, color='rgba(50,180,240,0.3)'),
            marker=dict(
                size=4, color=rk4_local_err, colorscale='Blues', 
                showscale=False, cmin=0, cmax=1e-5
            )
        ))
        
        # Leapfrog
        fig.add_trace(go.Scatter(
            x=lx, y=ly, mode='markers+lines',
            name='Leapfrog (Symplectic)',
            line=dict(width=1.5, color='rgba(50,220,100,0.3)'),
            marker=dict(
                size=4, color=leapfrog_local_err, colorscale='Greens', 
                showscale=False, cmin=0, cmax=1e-5
            )
        ))

        fig.update_layout(
            title="Orbital Trajectories (Color intensity indicates local energy drift)",
            xaxis_title="Position X",
            yaxis_title="Position Y",
            height=620,
            template="plotly_dark",
            yaxis=dict(scaleanchor="x", scaleratio=1) 
        )
        
        st.plotly_chart(fig, use_container_width=True)

        # benchmark comparison table
        st.markdown("### Algorithmic Performance Benchmark")
        
        # Get the exact starting energy of the system
        e0 = get_energy(init_state, perturb_eps)
        safe_e0 = e0 if e0 != 0 else 1e-12
        
        # Calculate relative error using the very last item [-1] in the energy arrays
        euler_err = abs((e_energy[-1] - e0) / safe_e0)
        rk4_err = abs((rk_energy[-1] - e0) / safe_e0)
        leapfrog_err = abs((l_energy[-1] - e0) / safe_e0)
        
        # Create the Pandas DataFrame
        benchmark_data = pd.DataFrame({
            "Integrator Method": ["Forward Euler (1st Order)", "Classical RK4 (4th Order)", "Leapfrog (Symplectic)"],
            "Predicted Stable Δt": [f"{dt_euler:.5f}", f"{dt_rk4:.5f}", f"{dt_leapfrog:.5f}"],
            "Final Energy Drift Error": [f"{euler_err:.2e}", f"{rk4_err:.2e}", f"{leapfrog_err:.2e}"]
        })
        
        # Render the table in Streamlit without the ugly index column
        st.dataframe(benchmark_data, use_container_width=True, hide_index=True)

        pdf_file = generate_pdf_report(
            orbit_regime=orbit_regime,
            eccentricity=eccentricity,
            e_dt=dt_euler, r_dt=dt_rk4, l_dt=dt_leapfrog,
            e_err=euler_err, r_err=rk4_err, l_err=leapfrog_err
        )
        
        st.download_button(
            label="Export Scientific Verification PDF",
            data=pdf_file,
            file_name="orbitml_verification_report.pdf",
            mime="application/pdf",
            use_container_width=True
        )

        # Call the standalone fragment manual section
        render_manual_simulation_block(init_state, sim_time, perturb_eps, dt_rk4)

with tab3:
    st.subheader("Explainable AI (XAI) & Feature Mechanics")
    st.markdown("""
    ### Interpretability Overview
    Rather than treating the Random Forest Regressor as a complete black box, the model maps physical conservation constants to algorithmic constraints. 
    
    * **Eccentricity Correlation:** As eccentricity approaches e → 1, velocities at the periapsis (closest approach) spike drastically. The model actively scales down predicted stable timesteps dynamically to capture these high-acceleration phases.
    * **Orbit Count Correlation:** Accumulated numerical error grows with the number of cycles rather than raw elapsed time. The Random Forest heavily weights how many orbital periods the solver must survive.
    """)
    
    st.markdown("---")
    st.markdown("### Random Forest Feature Importance")
    st.markdown("The chart below illustrates which physical parameters the Machine Learning model relies on most heavily when calculating the maximum stable timestep.")

    feature_names = [
        "Eccentricity", 
        "Orbital Period", 
        "Initial Velocity Y", 
        "Initial Velocity X", 
        "Simulation Time",
        "Radius", 
        "Orbit Count"
    ]
    
    importance_values = [0.08357984, 0.05824969, 0.11379373, 0.01776321, 0.08107479, 0.18621281, 0.45932593]

    # Build the horizontal bar chart
    fig_importance = go.Figure(go.Bar(
        x=importance_values,
        y=feature_names,
        orientation='h',
        marker=dict(
            color=importance_values,
            colorscale='Viridis', # Creates a nice color gradient based on value
            reversescale=True
        )
    ))

    fig_importance.update_layout(
        title="Relative Importance of Orbital Features",
        xaxis_title="Importance Weight",
        yaxis_title="Physical Feature",
        yaxis={'categoryorder': 'total ascending'}, # Automatically sorts the bars from smallest to largest
        template="plotly_dark",
        height=450,
        margin=dict(l=0, r=0, t=40, b=0)
    )

    st.plotly_chart(fig_importance, use_container_width=True)

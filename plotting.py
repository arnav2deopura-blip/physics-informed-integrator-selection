from __future__ import annotations

from turtle import lt, rt
from turtle import rt
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from scipy.ndimage import gaussian_filter

from .config import GM, PHYSICS_FEATURE_COLUMNS
from .physics import add_orbit_features
from .physics import run_sim as physics_run_sim
from .physics import run_sim_diagnostics, state_from_periapsis, step_euler, step_leapfrog, step_rk4
from .search import find_dt_for_integrator, max_energy_error

def _bin_means(benchmark_data: pd.DataFrame, column: str, ecc_bins: np.ndarray) -> list[float]:
    means = []
    for lo, hi in zip(ecc_bins[:-1], ecc_bins[1:]):
        subset = benchmark_data[(benchmark_data["ecc_target"] >= lo) & (benchmark_data["ecc_target"] < hi)]
        means.append(float(subset[column].mean()) if len(subset) else np.nan)
    return means

def create_all_figures(
    final_model,
    benchmark_data: pd.DataFrame,
    regime_context: dict,
    orbit_analysis: dict,
    total_time: float = 20.0,
) -> None:
    # --- DATA PREP FOR SAMPLE TRAJECTORIES ---
    vx = orbit_analysis["vx"]
    vy = orbit_analysis["vy"]
    means = orbit_analysis["means"]
    dt_euler, dt_rk4, dt_leapfrog = means

    init = [1.0, 0.0, vx, vy]
    ex, ey, ee, et = physics_run_sim(step_euler, init, dt_euler, total_time)
    rx, ry, re, rt = physics_run_sim(step_rk4, init, dt_rk4, total_time)
    lx, ly, le, lt = physics_run_sim(step_leapfrog, init, dt_leapfrog, total_time)
    e0_euler = max(abs(ee[0]), 1e-12)
    e0_rk4 = max(abs(re[0]), 1e-12)
    e0_leapfrog = max(abs(le[0]), 1e-12)
    ee_rel = [abs((value - ee[0]) / e0_euler) for value in ee]
    re_rel = [abs((value - re[0]) / e0_rk4) for value in re]
    le_rel = [abs((value - le[0]) / e0_leapfrog) for value in le]

    # 1. AI-OPTIMIZED TRAJECTORIES
    plt.figure(figsize=(8, 6))
    plt.plot(ex, ey, "r", label=f"Euler (dt={dt_euler:.2e})")
    plt.plot(rx, ry, "b", label=f"RK4 (dt={dt_rk4:.3g})")
    plt.plot(lx, ly, "g", label=f"Leapfrog (dt={dt_leapfrog:.3g})")
    plt.plot(0, 0, "yo", label="Sun")
    plt.axis("equal")
    plt.legend()
    plt.title("AI-Optimized Orbital Trajectories")
    plt.tight_layout()
    plt.savefig("fig_01_trajectories.png", dpi=300, bbox_inches='tight')

    # 2. ENERGY STABILITY CHECK
    plt.figure(figsize=(8, 6))
    plt.semilogy(et, np.array(ee_rel) + 1e-15, "r", label="Euler")
    plt.semilogy(rt, np.array(re_rel) + 1e-15, "b", label="RK4")
    plt.semilogy(lt, np.array(le_rel) + 1e-15, "g", label="Leapfrog")
    plt.xlabel("Time")
    plt.ylabel("Relative Energy Error")
    plt.title("Energy Stability Check (Sample Orbit)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("fig_02_energy_stability.png", dpi=300, bbox_inches='tight')

    # 3. STABILITY MAP (CONTOUR)
    plt.figure(figsize=(8, 6))
    vy_range = np.linspace(0.3, 1.2, 40)
    r_range = np.linspace(0.5, 2.0, 40)
    vy_mesh, r_mesh = np.meshgrid(vy_range, r_range)
    vx_slice = 0.0

    grid_inputs = pd.DataFrame({
        "vx": [vx_slice] * (40 * 40),
        "vy": vy_mesh.ravel(),
        "r": r_mesh.ravel(),
        "sim-time": [10.0] * (40 * 40),
    })
    grid_inputs = add_orbit_features(grid_inputs)
    grid_preds = final_model.predict(grid_inputs[PHYSICS_FEATURE_COLUMNS].values)
    dt_map = gaussian_filter(grid_preds[:, 1].reshape(40, 40), sigma=1.0)
    
    plt.contourf(vy_mesh, r_mesh, dt_map, levels=20, cmap="plasma")
    plt.colorbar(label="Predicted Stable dt (RK4)")
    plt.xlabel("Initial vy")
    plt.ylabel("Orbital Radius r")
    plt.title(f"Stability Map (RK4, vx={vx_slice:.2f})")
    plt.tight_layout()
    plt.savefig("fig_03_stability_map.png", dpi=300, bbox_inches='tight')

    # 4. SCALING CHECK (KEPLER'S LAW)
    plt.figure(figsize=(8, 6))
    r_vals = np.linspace(0.7, 2.0, 40)
    vy_circ = np.sqrt(GM / r_vals)
    orbital_periods = 2.0 * np.pi * np.sqrt(r_vals**3 / GM)

    rk4_scaling = np.array(
        [
            find_dt_for_integrator(
                "rk4",
                0.0,
                float(vy_val),
                float(r_val),
                float(5.0 * period),
            )
            for r_val, vy_val, period in zip(r_vals, vy_circ, orbital_periods)
        ]
    )

    x_scaling = r_vals**1.5
    valid = rk4_scaling > 0.0001
    slope, intercept = np.polyfit(x_scaling[valid], rk4_scaling[valid], 1)

    plt.scatter(x_scaling[valid], rk4_scaling[valid], s=20, label="Brute-force RK4 stable dt")
    plt.plot(
        x_scaling[valid],
        slope * x_scaling[valid] + intercept,
        "k--",
        label=r"Fit vs $r^{1.5}$",
    )
    plt.xlabel(r"$r^{1.5}$")
    plt.ylabel("Stable dt")
    plt.title(r"Scaling Check: RK4 stable dt vs Keplerian timescale")
    plt.legend()
    plt.tight_layout()
    plt.savefig("fig_04_keplerian_scaling.png", dpi=300, bbox_inches="tight")

    # 5. ENERGY ERROR CURVES (LOG-LOG)
    test_state = [1.0, 0.0, 0.15, 0.85]
    dt_values = np.logspace(-3, np.log10(0.3), 25)
    euler_errors = [max_energy_error(step_euler, test_state, dt, 10.0) for dt in dt_values]
    rk4_errors = [max_energy_error(step_rk4, test_state, dt, 10.0) for dt in dt_values]
    leapfrog_errors = [max_energy_error(step_leapfrog, test_state, dt, 10.0) for dt in dt_values]

    plt.figure(figsize=(8, 6))
    plt.loglog(dt_values, euler_errors, "r-o", ms=3, label="Euler")
    plt.loglog(dt_values, rk4_errors, "b-o", ms=3, label="RK4")
    plt.loglog(dt_values, leapfrog_errors, "g-o", ms=3, label="Leapfrog")
    plt.xlabel("dt")
    plt.ylabel("Max Relative Energy Error")
    plt.title("Step-Size Sensitivity (Eccentric Orbit)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("fig_05_step_sensitivity.png", dpi=300, bbox_inches='tight')

    # 6. UNCERTAINTY MAP
    plt.figure(figsize=(8, 6))
    tree_preds = np.array([tree.predict(grid_inputs[PHYSICS_FEATURE_COLUMNS].values)[:, 1] for tree in final_model.estimators_])
    std_map = gaussian_filter(tree_preds.std(axis=0).reshape(40, 40), sigma=1.0)
    plt.contourf(vy_mesh, r_mesh, std_map, levels=20)
    plt.colorbar(label="Prediction Uncertainty (Std Dev)")
    plt.xlabel("Initial vy")
    plt.ylabel("Orbital Radius r")
    plt.title("Model Uncertainty: High Variance Regimes")
    plt.tight_layout()
    plt.savefig("fig_06_uncertainty_map.png", dpi=300, bbox_inches='tight')

    # 7. PHYSICS-AWARE VS BASELINE BOXPLOT
    plt.figure(figsize=(8, 6))
    plt.yscale("log")
    rk4_err_basic = np.abs(regime_context["pred_regime_basic"][:, 1] - regime_context["y_test_regime"][:, 1])
    rk4_err_phys = np.abs(regime_context["pred_regime_phys"][:, 1] - regime_context["y_test_regime"][:, 1])
    plt.boxplot([rk4_err_basic, rk4_err_phys], labels=["Baseline (vx, vy, r, t)", "Physics-Aware (+e, period)"])
    plt.ylabel("Absolute Prediction Error")
    plt.title("Prediction Improvement via Physics-Informed Features")
    plt.tight_layout()
    plt.savefig("fig_07_feature_improvement.png", dpi=300, bbox_inches='tight')

    # 8. FEATURE IMPORTANCE
    plt.figure(figsize=(8, 6))
    labels = ["vx", "vy", "r", "t", "ecc", "period", "orbit_count"]
    plt.bar(labels, final_model.feature_importances_)
    plt.title("Relative Information Gain of Input Features")
    plt.ylabel("Importance Score")
    plt.tight_layout()
    plt.savefig("fig_08_feature_importance.png", dpi=300, bbox_inches='tight')

    # 9. PERFORMANCE SCORE VS ECCENTRICITY
    plt.figure(figsize=(8, 6))
    ecc_bins = np.linspace(0.0, 1.0, 11)
    ecc_centers = 0.5 * (ecc_bins[:-1] + ecc_bins[1:])
    for column, color, label in [("score_euler", "r", "Euler"), ("score_rk4", "b", "RK4"), ("score_leapfrog", "g", "Leapfrog")]:
        plt.plot(ecc_centers, _bin_means(benchmark_data, column, ecc_bins), "o-", color=color, label=label)
    plt.xlabel("Initial Eccentricity")
    plt.ylabel("Mean Performance Score")
    plt.title("Integrator Efficiency vs Orbital Eccentricity")
    plt.legend()
    plt.tight_layout()
    plt.savefig("fig_09_performance_by_ecc.png", dpi=300, bbox_inches='tight')

    # 10. ANGULAR MOMENTUM CONSERVATION
    rep_candidates = benchmark_data[(benchmark_data["dt_rk4"] > 0.0001) & (benchmark_data["dt_leapfrog"] > 0.0001)]
    if len(rep_candidates) > 0:
        rep_case = rep_candidates.sort_values(["near_collision", "ecc_target", "period_mult"], ascending=[False, False, False]).iloc[0]
        rep_state, rep_period = state_from_periapsis(rep_case["ecc_target"], rep_case["periapsis"])
        rep_total_time = rep_case["orbit_period"] * rep_case["period_mult"]

        plt.figure(figsize=(8, 6))
        for name, step_func, dt_col, color in [("Euler", step_euler, "dt_euler", "r"), ("RK4", step_rk4, "dt_rk4", "b"), ("Leapfrog", step_leapfrog, "dt_leapfrog", "g")]:
            _, _, _, angmom, _, times = run_sim_diagnostics(step_func, rep_state, rep_case[dt_col], rep_total_time, rep_case["perturb_eps"], max_steps=200000, track={"angmom", "times"})
            scale = max(abs(angmom[0]), 1e-12)
            rel_l = np.abs((np.array(angmom) - angmom[0]) / scale)
            plt.semilogy(np.array(times) / rep_period, rel_l + 1e-15, color=color, label=f"{name} (dt={rep_case[dt_col]:.4g})")
        plt.xlabel("Time / Orbital Period")
        plt.ylabel("Rel. Angular Momentum Error")
        plt.title("Angular Momentum Conservation (Long-Term)")
        plt.legend()
        plt.tight_layout()
        plt.savefig("fig_10_momentum_conservation.png", dpi=300, bbox_inches='tight')
    else:
        warnings.warn("No representative benchmark case found for momentum plot.")

    # 11. LONG-TERM ENERGY DRIFT
    long_candidates = benchmark_data[(benchmark_data["period_mult"] >= 100) & (benchmark_data["dt_euler"] > 0.0001) & (benchmark_data["dt_rk4"] > 0.0001) & (benchmark_data["dt_leapfrog"] > 0.0001)]
    if len(long_candidates) > 0:
        long_case = long_candidates.sort_values(["period_mult", "ecc_target"], ascending=[False, False]).iloc[0]
        long_state, long_period = state_from_periapsis(long_case["ecc_target"], long_case["periapsis"])
        period_counts = [1, 10, 30, 100]

        plt.figure(figsize=(8, 6))
        for name, step_func, dt_col, color in [("Euler", step_euler, "dt_euler", "r"), ("RK4", step_rk4, "dt_rk4", "b"), ("Leapfrog", step_leapfrog, "dt_leapfrog", "g")]:
            drifts = []
            for n_orbits in period_counts:
                _, _, energy, _, _, _ = run_sim_diagnostics(step_func, long_state, long_case[dt_col], n_orbits * long_period, long_case["perturb_eps"], max_steps=200000, track={"energy"})
                scale = max(abs(energy[0]), 1e-12)
                drifts.append(abs((energy[-1] - energy[0]) / scale))
            plt.loglog(period_counts, np.array(drifts) + 1e-15, "o-", color=color, label=name)
        plt.xlabel("Number of Orbital Periods")
        plt.ylabel("Final Relative Energy Drift")
        plt.title("Energy Drift vs Simulation Duration")
        plt.legend()
        plt.tight_layout()
        plt.savefig("fig_11_long_term_drift.png", dpi=300, bbox_inches='tight')

    # 12. ENERGY ERROR VS ECCENTRICITY
    plt.figure(figsize=(8, 6))
    for column, color, label in [("energy_err_euler", "r", "Euler"), ("energy_err_rk4", "b", "RK4"), ("energy_err_leapfrog", "g", "Leapfrog")]:
        plt.semilogy(ecc_centers, np.array(_bin_means(benchmark_data, column, ecc_bins)) + 1e-15, "o-", color=color, label=label)
    plt.xlabel("Initial Eccentricity")
    plt.ylabel("Mean Max Relative Energy Error")
    plt.title("Solver Accuracy vs Eccentricity (Semilog)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("fig_12_error_vs_ecc.png", dpi=300, bbox_inches='tight')

    # 13. WINNER HEATMAP
    plt.figure(figsize=(10, 6))
    regime_order = ["low_e", "mid_e", "high_e", "near_parabolic", "near_collision"]
    period_order = sorted(float(value) for value in benchmark_data["period_mult"].unique())
    period_labels = [f"{value:g}" for value in period_order]
    winner_grid = np.full((len(regime_order), len(period_order)), np.nan)
    name_to_num = {"euler": 0, "rk4": 1, "leapfrog": 2}
    num_to_name = ["Euler", "RK4", "Leapfrog"]

    for i, regime in enumerate(regime_order):
        for j, mult in enumerate(period_order):
            subset = benchmark_data[
                (benchmark_data["regime"] == regime)
                & np.isclose(benchmark_data["period_mult"], mult)
            ]
            if len(subset) == 0: continue
            mean_scores = subset[["score_euler", "score_rk4", "score_leapfrog"]].mean()
            winner = mean_scores.idxmax().replace("score_", "")
            winner_grid[i, j] = name_to_num[winner]

    plt.imshow(winner_grid, aspect="auto", cmap=ListedColormap(["red", "blue", "green"]), vmin=-0.5, vmax=2.5)
    for i in range(len(regime_order)):
        for j in range(len(period_order)):
            if not np.isnan(winner_grid[i, j]):
                plt.text(j, i, num_to_name[int(winner_grid[i, j])], ha="center", va="center", color="white", fontweight='bold')

    plt.xticks(range(len(period_order)), period_labels)
    plt.yticks(range(len(regime_order)), regime_order)
    plt.xlabel("Simulation Length (Orbital Periods)")
    plt.ylabel("Dynamical Regime")
    plt.title("Optimal Integrator Selection Grid")
    plt.tight_layout()
    plt.savefig("fig_13_winner_heatmap.png", dpi=300, bbox_inches='tight')

    print("\nSUCCESS 13 figures saved as high-resolution PNGs.")
    plt.show()

def plot_learning_curves(study_df: pd.DataFrame):
    # FIG 14: Model Error vs Dataset Size
    plt.figure(figsize=(8, 6))
    plt.plot(study_df["size"], study_df["mae"], 'o-', color='blue', linewidth=2)
    plt.xlabel("Number of Training Orbits")
    plt.ylabel("Mean Absolute Error (RK4 dt)")
    plt.title("Model Convergence: Error Reduction vs. Dataset Size")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("fig_14_error_vs_size.png", dpi=300)

    # FIG 15: Prediction Variance vs Dataset Size
    plt.figure(figsize=(8, 6))
    plt.plot(study_df["size"], study_df["variance"], 'o-', color='purple', linewidth=2)
    plt.xlabel("Number of Training Orbits")
    plt.ylabel("Mean Prediction Variance (Inter-tree Disagreement)")
    plt.title("Model Confidence: Uncertainty Reduction vs. Dataset Size")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("fig_15_variance_vs_size.png", dpi=300)
    
    plt.show()

from __future__ import annotations

import warnings

from .config import (
    BENCHMARK_CASES,
    BENCHMARK_DATASET_NAME,
    MODEL_NAME,
    N_WORKERS,
    NUM_RANDOM_ORBITS,
    TRAINING_DATASET_NAME,
)
from .data import generate_challenging_benchmark, generate_training_dataset
from .modeling import (
    analyze_single_orbit,
    fit_fixed_split_models,
    held_out_baseline_comparison,
    high_eccentricity_holdout,
    integrator_selection_stats,
    print_metric_report,
    run_feature_ablation,
    save_model,
    speed_comparison,
    train_final_model,
)
from .plotting import create_all_figures


def _print_feature_ablation(report: dict) -> None:
    print("--- FEATURE ABLATION REPORT (mean +/- std over 10 seeds) ---")
    print("Baseline features: vx, vy, r, sim-time")
    for i, integrator in enumerate(["Euler", "RK4", "Leapfrog"]):
        print(
            f"{integrator}: R^2 = {report['baseline_r2_mean'][i]:.4f} +/- {report['baseline_r2_std'][i]:.4f}   "
            f"MAE = {report['baseline_mae_mean'][i]:.5f} +/- {report['baseline_mae_std'][i]:.5f}"
        )

    print("Physics-aware features: vx, vy, r, sim-time, ecc, period")
    for i, integrator in enumerate(["Euler", "RK4", "Leapfrog"]):
        print(
            f"{integrator}: R^2 = {report['phys_r2_mean'][i]:.4f} +/- {report['phys_r2_std'][i]:.4f}   "
            f"MAE = {report['phys_mae_mean'][i]:.5f} +/- {report['phys_mae_std'][i]:.5f}"
        )


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

    print(f"Generating {NUM_RANDOM_ORBITS} training orbits using {N_WORKERS} workers...")
    data, failure_rates = generate_training_dataset()
    print("\nFailure rates:")
    for name, rate in failure_rates.items():
        print(f"{name}: {rate}")
    print(f"Data generated. Saved to {TRAINING_DATASET_NAME}")

    print("\nBuilding challenging-regime benchmark...")
    print(f"Building {BENCHMARK_CASES} benchmark cases using {N_WORKERS} workers...")
    benchmark_data = generate_challenging_benchmark()
    print(f"Saved challenging benchmark data to {BENCHMARK_DATASET_NAME}")

    print("\n--- CHALLENGING REGIME SUMMARY ---")
    print(benchmark_data.groupby("period_mult")[["score_euler", "score_rk4", "score_leapfrog"]].mean())

    print("\nVery high eccentricity only (e >= 0.9):")
    print(
        benchmark_data[benchmark_data["ecc_target"] >= 0.9][
            ["score_euler", "score_rk4", "score_leapfrog"]
        ].mean()
    )

    print("\nNear-collision only:")
    print(
        benchmark_data[benchmark_data["near_collision"] == 1][
            ["score_euler", "score_rk4", "score_leapfrog"]
        ].mean()
    )

    benchmark_true_best = benchmark_data[
        ["score_euler", "score_rk4", "score_leapfrog"]
    ].values.argmax(axis=1)
    print("\n--- BENCHMARK SCORE WINNERS ---")
    print(f"Euler wins by score: {100.0 * (benchmark_true_best == 0).mean():.1f}%")
    print(f"RK4 wins by score: {100.0 * (benchmark_true_best == 1).mean():.1f}%")
    print(f"Leapfrog wins by score: {100.0 * (benchmark_true_best == 2).mean():.1f}%")

    feature_report = run_feature_ablation(data)
    _print_feature_ablation(feature_report)

    split_context = fit_fixed_split_models(data)
    baseline_report = held_out_baseline_comparison(
        split_context["data_test"],
        split_context["y_test"],
        split_context["pred_phys"],
    )

    print("\n--- HELD-OUT BASELINE COMPARISON ---")
    print(f"Naive period rule: {baseline_report['naive_period_rule']:.5f}")
    print(f"Kepler scaling rule: {baseline_report['kepler_scaling_rule']:.5f}")
    print(f"ML model: {baseline_report['ml_model']:.5f}")
    print(f"Best improvement vs naive: {baseline_report['improvement_vs_naive']:.2f}x")
    print(f"Best improvement vs Kepler: {baseline_report['improvement_vs_kepler']:.2f}x")

    selection_report = integrator_selection_stats(
        split_context["y_test"],
        split_context["pred_phys"],
    )
    print("\n--- DT-BASED INTEGRATOR SELECTION ACCURACY ---")
    print(
        f"Model chooses best integrator correctly {100.0 * selection_report['accuracy']:.2f}% of the time"
    )
    print(f"Always pick RK4 accuracy: {100.0 * selection_report['rk4_accuracy']:.2f}%")
    print(
        f"Improvement over always-RK4: {100.0 * selection_report['improvement']:.2f} percentage points"
    )
    print(f"Fraction where Euler is best: {100.0 * selection_report['euler_fraction']:.1f}%")
    print(f"Fraction where RK4 is best: {100.0 * selection_report['rk4_fraction']:.1f}%")
    print(
        f"Fraction where Leapfrog is best: {100.0 * selection_report['leapfrog_fraction']:.1f}%"
    )

    regime_context = high_eccentricity_holdout(data)
    print("--- HIGH-ECCENTRICITY HOLDOUT REPORT ---")
    print(f"Held-out eccentricity threshold: e >= {regime_context['ecc_cut']:.3f}")
    print_metric_report(
        "Baseline model on hardest 20% of orbits",
        regime_context["y_test_regime"],
        regime_context["pred_regime_basic"],
    )
    print_metric_report(
        "Physics-aware model on hardest 20% of orbits",
        regime_context["y_test_regime"],
        regime_context["pred_regime_phys"],
    )

    print("Training final PHYSICS-AWARE model on full dataset...")
    final_model = train_final_model(data)
    save_model(final_model, MODEL_NAME)
    print(f"Final physics-aware model saved to {MODEL_NAME}")

    speed_report = speed_comparison(final_model)
    print("\n--- SPEED COMPARISON: ML vs BRUTE FORCE ---")
    print(f"ML Prediction Time (avg over 100 runs): {speed_report['ml_time_ms']:.3f} ms")
    print(
        f"Brute-force search time (5 orbits x 3 integrators): {speed_report['brute_time_ms']:.1f} ms"
    )
    print(f"ML is ~{speed_report['speedup']:.0f}x faster")

    orbit_analysis = analyze_single_orbit(final_model)
    means = orbit_analysis["means"]
    stds = orbit_analysis["stds"]
    print(f"--- AI INTEGRATOR ANALYSIS FOR vx={orbit_analysis['vx']}, vy={orbit_analysis['vy']} ---")
    print(f"Predicted stable dt for Euler:    {means[0]:.5f} +/- {stds[0]:.5f}")
    print(f"Predicted stable dt for RK4:      {means[1]:.5f} +/- {stds[1]:.5f}")
    print(f"Predicted stable dt for Leapfrog: {means[2]:.5f} +/- {stds[2]:.5f}")
    print("-" * 40)
    if means[0] < 0.005:
        print("WARNING: The timestep for Euler is very small.")
        print("It is recommended to switch to another integrator.")
    else:
        print("Euler is stable enough for this orbit.")
    print(f"Most Efficient Integrator: {orbit_analysis['best_name']} (allows dt={means.max():.4f})")
    print("-" * 40)

    create_all_figures(final_model, benchmark_data, regime_context, orbit_analysis)

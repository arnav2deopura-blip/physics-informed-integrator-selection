from __future__ import annotations

import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from .config import BASIC_FEATURE_COLUMNS, GM, PHYSICS_FEATURE_COLUMNS, SEED_COUNT, TARGET_COLUMNS
from .physics import add_orbit_features
from .search import find_dt_for_integrator


def print_metric_report(title, y_true, y_pred) -> None:
    scores = r2_score(y_true, y_pred, multioutput="raw_values")
    maes = mean_absolute_error(y_true, y_pred, multioutput="raw_values")
    print(title)
    print(f"Euler:    R^2 = {scores[0]:.4f}   MAE = {maes[0]:.5f}")
    print(f"RK4:      R^2 = {scores[1]:.4f}   MAE = {maes[1]:.5f}")
    print(f"Leapfrog: R^2 = {scores[2]:.4f}   MAE = {maes[2]:.5f}")
    print()


def _rf(n_estimators: int = 100, random_state: int = 42) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )


def run_feature_ablation(data: pd.DataFrame, seed_count: int = SEED_COUNT) -> dict:
    x_basic = data[BASIC_FEATURE_COLUMNS].values
    x_phys = data[PHYSICS_FEATURE_COLUMNS].values
    targets = data[TARGET_COLUMNS].values
    all_idx = np.arange(len(data))

    baseline_r2_list = []
    phys_r2_list = []
    baseline_mae_list = []
    phys_mae_list = []

    for seed in range(seed_count):
        train_idx, test_idx = train_test_split(all_idx, test_size=0.3, random_state=seed)
        x_basic_train = x_basic[train_idx]
        x_basic_test = x_basic[test_idx]
        x_phys_train = x_phys[train_idx]
        x_phys_test = x_phys[test_idx]
        y_train = targets[train_idx]
        y_test = targets[test_idx]

        model_basic = _rf(random_state=seed)
        model_basic.fit(x_basic_train, y_train)
        pred_basic = model_basic.predict(x_basic_test)

        model_phys = _rf(random_state=seed)
        model_phys.fit(x_phys_train, y_train)
        pred_phys = model_phys.predict(x_phys_test)

        baseline_r2_list.append(r2_score(y_test, pred_basic, multioutput="raw_values"))
        phys_r2_list.append(r2_score(y_test, pred_phys, multioutput="raw_values"))
        baseline_mae_list.append(mean_absolute_error(y_test, pred_basic, multioutput="raw_values"))
        phys_mae_list.append(mean_absolute_error(y_test, pred_phys, multioutput="raw_values"))

    return {
        "baseline_r2_mean": np.mean(baseline_r2_list, axis=0),
        "baseline_r2_std": np.std(baseline_r2_list, axis=0),
        "phys_r2_mean": np.mean(phys_r2_list, axis=0),
        "phys_r2_std": np.std(phys_r2_list, axis=0),
        "baseline_mae_mean": np.mean(baseline_mae_list, axis=0),
        "baseline_mae_std": np.std(baseline_mae_list, axis=0),
        "phys_mae_mean": np.mean(phys_mae_list, axis=0),
        "phys_mae_std": np.std(phys_mae_list, axis=0),
    }


def fit_fixed_split_models(data: pd.DataFrame, random_state: int = 42) -> dict:
    x_basic = data[BASIC_FEATURE_COLUMNS].values
    x_phys = data[PHYSICS_FEATURE_COLUMNS].values
    targets = data[TARGET_COLUMNS].values
    all_idx = np.arange(len(data))

    train_idx, test_idx = train_test_split(all_idx, test_size=0.3, random_state=random_state)
    data_test = data.iloc[test_idx].reset_index(drop=True)

    x_basic_train = x_basic[train_idx]
    x_basic_test = x_basic[test_idx]
    x_phys_train = x_phys[train_idx]
    x_phys_test = x_phys[test_idx]
    y_train = targets[train_idx]
    y_test = targets[test_idx]

    model_basic = _rf(random_state=random_state)
    model_basic.fit(x_basic_train, y_train)
    pred_basic = model_basic.predict(x_basic_test)

    model_phys = _rf(random_state=random_state)
    model_phys.fit(x_phys_train, y_train)
    pred_phys = model_phys.predict(x_phys_test)

    return {
        "train_idx": train_idx,
        "test_idx": test_idx,
        "data_test": data_test,
        "y_test": y_test,
        "pred_basic": pred_basic,
        "pred_phys": pred_phys,
        "model_basic": model_basic,
        "model_phys": model_phys,
    }


def held_out_baseline_comparison(data_test: pd.DataFrame, y_test: np.ndarray, pred_phys: np.ndarray) -> dict:
    baseline1 = 0.01 * data_test["period"].to_numpy()
    baseline2 = 0.05 * np.sqrt(data_test["r"].to_numpy() ** 3 / GM)

    err1 = float(np.mean(np.abs(baseline1 - y_test[:, 1])))
    err2 = float(np.mean(np.abs(baseline2 - y_test[:, 1])))
    model_err = float(np.mean(np.abs(pred_phys[:, 1] - y_test[:, 1])))

    return {
        "naive_period_rule": err1,
        "kepler_scaling_rule": err2,
        "ml_model": model_err,
        "improvement_vs_naive": err1 / model_err,
        "improvement_vs_kepler": err2 / model_err,
    }


def integrator_selection_stats(y_test: np.ndarray, pred_phys: np.ndarray) -> dict:
    true_best = np.argmax(y_test, axis=1)
    pred_best = np.argmax(pred_phys, axis=1)
    rk4_pred = np.full(len(true_best), 1)

    accuracy = float(np.mean(true_best == pred_best))
    rk4_accuracy = float(np.mean(true_best == rk4_pred))

    return {
        "accuracy": accuracy,
        "rk4_accuracy": rk4_accuracy,
        "improvement": accuracy - rk4_accuracy,
        "euler_fraction": float(np.mean(true_best == 0)),
        "rk4_fraction": float(np.mean(true_best == 1)),
        "leapfrog_fraction": float(np.mean(true_best == 2)),
    }


def high_eccentricity_holdout(data: pd.DataFrame, random_state: int = 42) -> dict:
    ecc_cut = float(data["ecc"].quantile(0.80))
    train_regime = data[data["ecc"] < ecc_cut]
    test_regime = data[data["ecc"] >= ecc_cut]

    x_train_basic = train_regime[BASIC_FEATURE_COLUMNS].values
    x_test_basic = test_regime[BASIC_FEATURE_COLUMNS].values
    x_train_phys = train_regime[PHYSICS_FEATURE_COLUMNS].values
    x_test_phys = test_regime[PHYSICS_FEATURE_COLUMNS].values

    y_train = train_regime[TARGET_COLUMNS].values
    y_test = test_regime[TARGET_COLUMNS].values

    model_basic = _rf(random_state=random_state)
    model_basic.fit(x_train_basic, y_train)
    pred_basic = model_basic.predict(x_test_basic)

    model_phys = _rf(random_state=random_state)
    model_phys.fit(x_train_phys, y_train)
    pred_phys = model_phys.predict(x_test_phys)

    return {
        "ecc_cut": ecc_cut,
        "y_test_regime": y_test,
        "pred_regime_basic": pred_basic,
        "pred_regime_phys": pred_phys,
    }


def train_final_model(data: pd.DataFrame, random_state: int = 42):
    model = _rf(n_estimators=200, random_state=random_state)
    model.fit(data[PHYSICS_FEATURE_COLUMNS].values, data[TARGET_COLUMNS].values)
    return model


def save_model(model, destination: str) -> None:
    joblib.dump(model, destination)


def speed_comparison(final_model) -> dict:
    sample_inputs = pd.DataFrame(
        {
            "vx": [-0.2, -0.1, 0.0, 0.1, 0.2],
            "vy": [0.4, 0.6, 0.8, 1.0, 1.2],
            "r": [1.0] * 5,
            "sim-time": [10.0] * 5,
        }
    )
    sample_inputs = add_orbit_features(sample_inputs)

    start = time.time()
    features = sample_inputs[PHYSICS_FEATURE_COLUMNS].values
    for _ in range(100):
        final_model.predict(features)
    ml_time = (time.time() - start) / 100.0

    start = time.time()
    for vx_val, vy_val in zip(sample_inputs["vx"], sample_inputs["vy"]):
        find_dt_for_integrator("euler", vx_val, vy_val, 1.0, 10.0)
        find_dt_for_integrator("rk4", vx_val, vy_val, 1.0, 10.0)
        find_dt_for_integrator("leapfrog", vx_val, vy_val, 1.0, 10.0)
    brute_time = time.time() - start

    return {
        "ml_time_ms": ml_time * 1000.0,
        "brute_time_ms": brute_time * 1000.0,
        "speedup": brute_time / max(ml_time, 1e-12),
    }


def analyze_single_orbit(
    final_model,
    vx: float = 0.15,
    vy: float = 0.7,
    r_input: float = 1.0,
    sim_input: float = 10.0,
) -> dict:
    input_df = pd.DataFrame(
        [[vx, vy, r_input, sim_input]],
        columns=BASIC_FEATURE_COLUMNS,
    )
    input_df = add_orbit_features(input_df)
    features = input_df[PHYSICS_FEATURE_COLUMNS].values

    all_tree_preds = np.array([tree.predict(features)[0] for tree in final_model.estimators_])
    means = all_tree_preds.mean(axis=0)
    stds = all_tree_preds.std(axis=0)

    dts = {"Euler": means[0], "RK4": means[1], "Leapfrog": means[2]}
    best_name = max(dts, key=dts.get)

    return {
        "vx": vx,
        "vy": vy,
        "means": means,
        "stds": stds,
        "best_name": best_name,
        "input_df": input_df,
    }

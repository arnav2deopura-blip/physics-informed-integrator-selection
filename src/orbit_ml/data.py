from __future__ import annotations

import random
from multiprocessing import Pool

import pandas as pd

from .config import (
    BENCHMARK_CASES,
    BENCHMARK_DATASET_NAME,
    BENCHMARK_PERIOD_MULTS,
    BENCHMARK_REPEATS_PER_CELL,
    CHUNK_SIZE,
    N_WORKERS,
    NUM_RANDOM_ORBITS,
    PERTURB_EPS,
    TRAINING_DATASET_NAME,
)
from .physics import add_orbit_features, get_energy, get_orbital_period, state_from_periapsis
from .search import find_dt_by_combined_metrics, find_dt_for_integrator


def sample_challenging_orbit_for_cell(regime: str, period_mult: float):
    near_collision = 0

    if regime == "low_e":
        eccentricity = random.uniform(0.05, 0.30)
        periapsis = random.uniform(0.6, 1.5)
    elif regime == "mid_e":
        eccentricity = random.uniform(0.30, 0.80)
        periapsis = random.uniform(0.2, 1.0)
    elif regime == "high_e":
        eccentricity = random.uniform(0.80, 0.95)
        periapsis = random.uniform(0.08, 0.50)
    elif regime == "near_parabolic":
        eccentricity = random.uniform(0.90, 0.999)
        periapsis = random.uniform(0.15, 0.60)
    elif regime == "near_collision":
        eccentricity = random.uniform(0.60, 0.95)
        periapsis = random.uniform(0.03, 0.08)
        near_collision = 1
    else:
        raise ValueError(f"Unknown regime: {regime}")

    perturb_eps = PERTURB_EPS if random.random() < 0.35 else 0.0
    state, orbit_period = state_from_periapsis(eccentricity, periapsis)
    return state, eccentricity, periapsis, orbit_period, period_mult, perturb_eps, near_collision, regime

def run_single_benchmark(task):
    regime, period_mult = task
    (
        start_state,
        ecc_target,
        periapsis,
        orbit_period,
        period_mult,
        perturb_eps,
        near_collision,
        regime,
    ) = sample_challenging_orbit_for_cell(regime, period_mult)

    dt_e, eerr_e, lerr_e, edrift_e, ldrift_e, score_e = find_dt_by_combined_metrics(
        "euler", start_state, periapsis, orbit_period, period_mult, perturb_eps
    )
    dt_r, eerr_r, lerr_r, edrift_r, ldrift_r, score_r = find_dt_by_combined_metrics(
        "rk4", start_state, periapsis, orbit_period, period_mult, perturb_eps
    )
    dt_l, eerr_l, lerr_l, edrift_l, ldrift_l, score_l = find_dt_by_combined_metrics(
        "leapfrog", start_state, periapsis, orbit_period, period_mult, perturb_eps
    )

    return [
        ecc_target,
        periapsis,
        orbit_period,
        period_mult,
        perturb_eps,
        near_collision,
        regime,
        dt_e,
        dt_r,
        dt_l,
        eerr_e,
        eerr_r,
        eerr_l,
        lerr_e,
        lerr_r,
        lerr_l,
        edrift_e,
        edrift_r,
        edrift_l,
        ldrift_e,
        ldrift_r,
        ldrift_l,
        score_e,
        score_r,
        score_l,
    ]

def generate_one_training_orbit(_):
    while True:
        vx_rand = round(random.uniform(-0.5, 0.5), 3)
        vy_rand = round(random.uniform(0.3, 1.2), 3)
        r_rand = round(random.uniform(0.5, 2.0), 3)

        if get_energy([r_rand, 0.0, vx_rand, vy_rand]) < 0:
            break

    orbit_period = get_orbital_period([r_rand, 0.0, vx_rand, vy_rand])

    roll = random.random()
    if roll < 0.25:
        period_mult = 10 ** random.uniform(-1.3, -0.3)   # about 0.05 to 0.5 orbits
    elif roll < 0.50:
        period_mult = 10 ** random.uniform(-0.3, 0.7)    # about 0.5 to 5 orbits
    elif roll < 0.75:
        period_mult = 10 ** random.uniform(0.7, 1.7)     # about 5 to 50 orbits
    else:
        period_mult = 10 ** random.uniform(1.7, 2.5)     # about 50 to 300 orbits

    sim_rand = round(period_mult * orbit_period, 3)

    dt_e = find_dt_for_integrator("euler", vx_rand, vy_rand, r_rand, sim_rand)
    dt_r = find_dt_for_integrator("rk4", vx_rand, vy_rand, r_rand, sim_rand)
    dt_l = find_dt_for_integrator("leapfrog", vx_rand, vy_rand, r_rand, sim_rand)

    euler_success = int(dt_e > 0.0001)
    rk4_success = int(dt_r > 0.0001)
    leapfrog_success = int(dt_l > 0.0001)

    return {
        "row": [
            vx_rand,
            vy_rand,
            r_rand,
            sim_rand,
            dt_e,
            dt_r,
            dt_l,
            euler_success,
            rk4_success,
            leapfrog_success,
        ],
        "euler_fail": 1 - euler_success,
        "rk4_fail": 1 - rk4_success,
        "leapfrog_fail": 1 - leapfrog_success,
        "usable": int(euler_success or rk4_success or leapfrog_success),
    }

def generate_training_dataset(
    num_random_orbits: int = NUM_RANDOM_ORBITS,
    n_workers: int = N_WORKERS,
    chunk_size: int = CHUNK_SIZE,
) -> tuple[pd.DataFrame, dict[str, float]]:
    rows = []
    euler_fail = 0
    rk4_fail = 0
    leapfrog_fail = 0

    with Pool(processes=n_workers) as pool:
        results = pool.imap_unordered(
            generate_one_training_orbit,
            range(num_random_orbits),
            chunksize=chunk_size,
        )

        for result in results:
            euler_fail += result["euler_fail"]
            rk4_fail += result["rk4_fail"]
            leapfrog_fail += result["leapfrog_fail"]
            if result["usable"]:
                rows.append(result["row"])

        dataframe = pd.DataFrame(
        rows,
        columns=[
            "vx",
            "vy",
            "r",
            "sim-time",
            "dt_euler",
            "dt_rk4",
            "dt_leapfrog",
            "euler_success",
            "rk4_success",
            "leapfrog_success",
        ],
    )
    dataframe = add_orbit_features(dataframe)
    dataframe.to_csv(TRAINING_DATASET_NAME, index=False)

    failure_rates = {
        "Euler": euler_fail / num_random_orbits,
        "RK4": rk4_fail / num_random_orbits,
        "Leapfrog": leapfrog_fail / num_random_orbits,
    }
    return dataframe, failure_rates


def generate_challenging_benchmark(
    benchmark_cases: int = BENCHMARK_CASES,
    n_workers: int = N_WORKERS,
    chunk_size: int = CHUNK_SIZE,
) -> pd.DataFrame:
    regime_order = ["low_e", "mid_e", "high_e", "near_parabolic", "near_collision"]

    tasks = []
    for regime in regime_order:
        for period_mult in BENCHMARK_PERIOD_MULTS:
            for _ in range(BENCHMARK_REPEATS_PER_CELL):
                tasks.append((regime, period_mult))

    with Pool(processes=n_workers) as pool:
        rows = list(
            pool.imap_unordered(
                run_single_benchmark,
                tasks,
                chunksize=chunk_size,
            )
        )

    dataframe = pd.DataFrame(
        rows,
        columns=[
            "ecc_target",
            "periapsis",
            "orbit_period",
            "period_mult",
            "perturb_eps",
            "near_collision",
            "regime",
            "dt_euler",
            "dt_rk4",
            "dt_leapfrog",
            "energy_err_euler",
            "energy_err_rk4",
            "energy_err_leapfrog",
            "angmom_err_euler",
            "angmom_err_rk4",
            "angmom_err_leapfrog",
            "energy_drift_euler",
            "energy_drift_rk4",
            "energy_drift_leapfrog",
            "angmom_drift_euler",
            "angmom_drift_rk4",
            "angmom_drift_leapfrog",
            "score_euler",
            "score_rk4",
            "score_leapfrog",
        ],
    )
    dataframe["best_score_integrator"] = (
        dataframe[["score_euler", "score_rk4", "score_leapfrog"]]
        .idxmax(axis=1)
        .str.replace("score_", "", regex=False)
    )
    dataframe.to_csv(BENCHMARK_DATASET_NAME, index=False)
    return dataframe

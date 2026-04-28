from __future__ import annotations

import numpy as np

from .config import GM
from .physics import INTEGRATORS, State, get_energy, run_sim_diagnostics


def _relative_error(reference: float, current: float) -> float:
    denominator = max(abs(reference), 1e-12)
    return abs(current - reference) / denominator


def max_energy_error(step_func, start_state: State, dt_val: float, limit_t: float) -> float:
    curr_time = 0.0
    state = start_state[:]
    start_energy = get_energy(state)
    max_err = 0.0

    while curr_time < limit_t:
        state = step_func(state, dt_val)
        max_err = max(max_err, _relative_error(start_energy, get_energy(state)))
        curr_time += dt_val

    return max_err


def find_dt_for_integrator(
    integrator_name: str,
    vx_start: float,
    vy_start: float,
    r_start: float = 1.0,
    sim_time: float = 5.0,
) -> float:
    dt_test = 0.3
    error_tolerance = 0.05 if integrator_name == "euler" else 0.01

    while dt_test >= 0.0005:
        max_error = 0.0
        state = [r_start, 0.0, vx_start, vy_start]
        start_energy = get_energy(state)
        curr_time = 0.0

        while curr_time < sim_time:
            state = INTEGRATORS[integrator_name](state, dt_test)
            max_error = max(max_error, _relative_error(start_energy, get_energy(state)))
            curr_time += dt_test

        if max_error < error_tolerance:
            best_dt = dt_test
            dt_try = dt_test * 1.5

            while dt_try < 0.3:
                state = [r_start, 0.0, vx_start, vy_start]
                start_energy = get_energy(state)
                curr_time = 0.0
                max_error = 0.0

                while curr_time < sim_time:
                    state = INTEGRATORS[integrator_name](state, dt_try)
                    max_error = max(max_error, _relative_error(start_energy, get_energy(state)))
                    curr_time += dt_try

                if max_error < error_tolerance:
                    best_dt = dt_try
                    dt_try *= 1.2
                else:
                    break

            return best_dt

        dt_test = round(dt_test / 2.0, 5)

    return 0.0001


def combined_performance_score(
    stable_dt: float,
    local_time: float,
    energy_error: float,
    angmom_error: float,
    energy_drift: float,
    angmom_drift: float,
    period_mult: int,
) -> float:
    dt_score = min(stable_dt / max(0.50 * local_time, 1e-12), 1.0)
    energy_score = float(np.exp(-energy_error / 0.01))
    angmom_score = float(np.exp(-angmom_error / 0.01))
    drift_score = float(np.exp(-(energy_drift + angmom_drift) / 0.005))

    if period_mult <= 5:
        return 100.0 * (
            0.25 * dt_score
            + 0.30 * energy_score
            + 0.30 * angmom_score
            + 0.15 * drift_score
        )

    return 100.0 * (
        0.10 * dt_score
        + 0.20 * energy_score
        + 0.20 * angmom_score
        + 0.50 * drift_score
    )


def find_dt_by_combined_metrics(
    integrator_name: str,
    start_state: State,
    periapsis: float,
    orbit_period: float,
    period_mult: int,
    perturb_eps: float = 0.0,
) -> tuple[float, float, float, float, float, float]:
    step_func = INTEGRATORS[integrator_name]
    limit_t = orbit_period * period_mult
    local_time = np.sqrt(periapsis**3 / GM)
    dt_candidates = local_time * np.array(
        [0.50, 0.35, 0.25, 0.18, 0.12, 0.08, 0.06, 0.04, 0.03, 0.02, 0.015, 0.01, 0.006, 0.004, 0.002]
    )

    best_result = None

    for dt in dt_candidates:
        _, _, energy, angmom, _, times = run_sim_diagnostics(
            step_func, start_state, float(dt), limit_t, perturb_eps, max_steps=500000
        )

        if len(times) < 2 or times[-1] < 0.95 * limit_t:
            continue

        energy0 = energy[0]
        angmom0 = angmom[0]
        energy_scale = max(abs(energy0), 1e-12)
        angmom_scale = max(abs(angmom0), 1e-12)

        energy_arr = np.asarray(energy, dtype=float)
        angmom_arr = np.asarray(angmom, dtype=float)

        energy_error = float(np.max(np.abs((energy_arr - energy0) / energy_scale)))
        angmom_error = float(np.max(np.abs((angmom_arr - angmom0) / angmom_scale)))
        energy_drift_short = float(abs((energy_arr[-1] - energy0) / energy_scale))
        angmom_drift_short = float(abs((angmom_arr[-1] - angmom0) / angmom_scale))

        stress_time = min(max(limit_t, 20.0 * orbit_period), 200.0 * orbit_period)
        _, _, energy_long, angmom_long, _, times_long = run_sim_diagnostics(
            step_func, start_state, float(dt), stress_time, perturb_eps, max_steps=500000
        )

        if len(times_long) < 2:
            continue

        energy_long0 = max(abs(energy_long[0]), 1e-12)
        angmom_long0 = max(abs(angmom_long[0]), 1e-12)
        energy_drift_long = float(abs((energy_long[-1] - energy_long[0]) / energy_long0))
        angmom_drift_long = float(abs((angmom_long[-1] - angmom_long[0]) / angmom_long0))

        energy_drift = energy_drift_short + energy_drift_long
        angmom_drift = angmom_drift_short + angmom_drift_long

        if (
            not np.isfinite(energy_error)
            or not np.isfinite(angmom_error)
            or not np.isfinite(energy_drift)
            or not np.isfinite(angmom_drift)
        ):
            continue

        if (
            energy_error > 0.50
            or angmom_error > 0.50
            or energy_drift > 0.50
            or angmom_drift > 0.50
        ):
            continue

        score = combined_performance_score(
            float(dt),
            float(local_time),
            energy_error,
            angmom_error,
            energy_drift,
            angmom_drift,
            period_mult,
        )

        candidate = (
            float(dt),
            energy_error,
            angmom_error,
            energy_drift,
            angmom_drift,
            score,
        )
        if best_result is None or candidate[-1] > best_result[-1]:
            best_result = candidate

    if best_result is None:
        return 0.0001, 1.0, 1.0, 1.0, 1.0, 0.0

    return best_result

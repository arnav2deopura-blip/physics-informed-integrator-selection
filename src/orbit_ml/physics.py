from __future__ import annotations

import math
from typing import Callable

import numpy as np
import pandas as pd

from .config import COLLISION_RADIUS, GM

State = list[float]
Integrator = Callable[[State, float, float], State]


def radius(x: float, y: float) -> float:
    return max(math.hypot(x, y), 1e-9)


def get_derivatives(state: State, perturb_eps: float = 0.0) -> State:
    x, y, vx, vy = state
    r = radius(x, y)
    radial_factor = 1.0 + perturb_eps / (r**2)
    ax = -GM * x / r**3 * radial_factor
    ay = -GM * y / r**3 * radial_factor
    return [vx, vy, ax, ay]


def get_energy(state: State, perturb_eps: float = 0.0) -> float:
    x, y, vx, vy = state
    r = radius(x, y)
    extra_potential = -GM * perturb_eps / (3.0 * r**3)
    return 0.5 * (vx**2 + vy**2) - GM / r + extra_potential


def get_angular_momentum(state: State) -> float:
    x, y, vx, vy = state
    return x * vy - y * vx


def get_eccentricity(state: State) -> float:
    energy = get_energy(state)
    angular_momentum = get_angular_momentum(state)
    e_sq = 1.0 + (2.0 * energy * angular_momentum**2) / (GM**2)
    return math.sqrt(max(e_sq, 0.0))


def get_orbital_period(state: State) -> float:
    energy = get_energy(state)
    if energy >= 0:
        return 1e10
    semi_major_axis = -GM / (2.0 * energy)
    return 2.0 * math.pi * math.sqrt(semi_major_axis**3 / GM)


def add_orbit_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe = dataframe.copy()

    vx = dataframe["vx"].to_numpy(dtype=float)
    vy = dataframe["vy"].to_numpy(dtype=float)
    r = dataframe["r"].to_numpy(dtype=float)

    energy = 0.5 * (vx**2 + vy**2) - GM / r
    angular_momentum = r * vy

    e_sq = 1.0 + (2.0 * energy * angular_momentum**2) / (GM**2)
    dataframe["ecc"] = np.sqrt(np.maximum(e_sq, 0.0))

    period = np.full(len(dataframe), 1e10, dtype=float)
    bound = energy < 0
    semi_major_axis = -GM / (2.0 * energy[bound])
    period[bound] = 2.0 * np.pi * np.sqrt(semi_major_axis**3 / GM)
    dataframe["period"] = period

    dataframe["orbit_count"] = dataframe["sim-time"] / np.maximum(dataframe["period"], 1e-12)
    return dataframe


def state_from_periapsis(eccentricity: float, periapsis: float) -> tuple[State, float]:
    semi_major_axis = periapsis / (1.0 - eccentricity)
    periapsis_speed = math.sqrt(GM * (1.0 + eccentricity) / periapsis)
    period = 2.0 * math.pi * math.sqrt(semi_major_axis**3 / GM)
    return [periapsis, 0.0, 0.0, periapsis_speed], period


def step_euler(state: State, dt: float, perturb_eps: float = 0.0) -> State:
    x, y, vx, vy = state
    dx, dy, ax, ay = get_derivatives(state, perturb_eps)
    return [x + dx * dt, y + dy * dt, vx + ax * dt, vy + ay * dt]


def step_rk4(state: State, dt: float, perturb_eps: float = 0.0) -> State:
    k1 = get_derivatives(state, perturb_eps)
    state_k2 = [state[i] + 0.5 * dt * k1[i] for i in range(4)]
    k2 = get_derivatives(state_k2, perturb_eps)
    state_k3 = [state[i] + 0.5 * dt * k2[i] for i in range(4)]
    k3 = get_derivatives(state_k3, perturb_eps)
    state_k4 = [state[i] + dt * k3[i] for i in range(4)]
    k4 = get_derivatives(state_k4, perturb_eps)
    return [
        state[i] + (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i])
        for i in range(4)
    ]


def step_leapfrog(state: State, dt: float, perturb_eps: float = 0.0) -> State:
    x, y, vx, vy = state
    _, _, ax, ay = get_derivatives(state, perturb_eps)

    x_new = x + vx * dt + 0.5 * ax * dt**2
    y_new = y + vy * dt + 0.5 * ay * dt**2
    _, _, ax_new, ay_new = get_derivatives([x_new, y_new, vx, vy], perturb_eps)
    vx_new = vx + 0.5 * (ax + ax_new) * dt
    vy_new = vy + 0.5 * (ay + ay_new) * dt
    return [x_new, y_new, vx_new, vy_new]


INTEGRATORS: dict[str, Integrator] = {
    "euler": step_euler,
    "rk4": step_rk4,
    "leapfrog": step_leapfrog,
}


def run_sim(step_func: Integrator, start_state: State, dt_val: float, limit_t: float) -> tuple[list[float], list[float], list[float], list[float]]:
    curr_time = 0.0
    state = start_state[:]
    path_x = [state[0]]
    path_y = [state[1]]
    energy = [get_energy(state)]
    times = [0.0]

    while curr_time < limit_t:
        state = step_func(state, dt_val)
        path_x.append(state[0])
        path_y.append(state[1])
        energy.append(get_energy(state))
        curr_time += dt_val
        times.append(curr_time)

    return path_x, path_y, energy, times


def run_sim_diagnostics(
    step_func: Integrator,
    start_state: State,
    dt_val: float,
    limit_t: float,
    perturb_eps: float = 0.0,
    max_steps: int = 1000000,
    track: set[str] | None = None,
    compute_stats_only: bool = False,
) -> tuple[list[float], list[float], list[float], list[float], list[float], list[float]]:
    if track is None:
        track = {"path_x", "path_y", "energy", "angmom", "radii", "times"}
    
    curr_time = 0.0
    state = start_state[:]
    path_x = [state[0]] if "path_x" in track else []
    path_y = [state[1]] if "path_y" in track else []
    radii = [radius(state[0], state[1])] if "radii" in track else []
    times = [0.0] if ("times" in track and not compute_stats_only) else []
    
    # For energy/angmom: either store full lists OR compute stats only
    if compute_stats_only and ("energy" in track or "angmom" in track):
        energy_0 = get_energy(state, perturb_eps)
        angmom_0 = get_angular_momentum(state)
        energy_max_dev = 0.0
        angmom_max_dev = 0.0
        energy = [energy_0]
        angmom = [angmom_0]
    else:
        energy = [get_energy(state, perturb_eps)] if "energy" in track else []
        angmom = [get_angular_momentum(state)] if "angmom" in track else []
        energy_0 = energy[0] if energy else 0.0
        angmom_0 = angmom[0] if angmom else 0.0
        energy_max_dev = 0.0
        angmom_max_dev = 0.0
    
    step_count = 0

    while curr_time < limit_t and step_count < max_steps:
        state = step_func(state, dt_val, perturb_eps)
        r_now = radius(state[0], state[1])

        if r_now < COLLISION_RADIUS or r_now > 50.0 or not np.all(np.isfinite(state)):
            break

        if "path_x" in track:
            path_x.append(state[0])
        if "path_y" in track:
            path_y.append(state[1])
        if "radii" in track:
            radii.append(r_now)
        
        if "energy" in track:
            e_val = get_energy(state, perturb_eps)
            if compute_stats_only:
                energy_scale = max(abs(energy_0), 1e-12)
                dev = abs((e_val - energy_0) / energy_scale)
                energy_max_dev = max(energy_max_dev, dev)
            else:
                energy.append(e_val)
        
        if "angmom" in track:
            l_val = get_angular_momentum(state)
            if compute_stats_only:
                angmom_scale = max(abs(angmom_0), 1e-12)
                dev = abs((l_val - angmom_0) / angmom_scale)
                angmom_max_dev = max(angmom_max_dev, dev)
            else:
                angmom.append(l_val)
        
        curr_time += dt_val
        if "times" in track and not compute_stats_only:
            times.append(curr_time)
        step_count += 1

    # If using stats-only mode, append the max deviations to return lists
    if compute_stats_only:
        if "energy" in track:
            energy.append(energy_max_dev)
        if "angmom" in track:
            angmom.append(angmom_max_dev)
        if "times" in track:
            times = [0.0, curr_time]

    return path_x, path_y, energy, angmom, radii, times

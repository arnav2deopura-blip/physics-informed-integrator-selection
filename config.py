from __future__ import annotations

from multiprocessing import cpu_count

GM = 1.0
PERTURB_EPS = 0.002
COLLISION_RADIUS = 0.01
BENCHMARK_PERIOD_MULTS = [0.05, 0.1, 0.25, 0.5, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0]
BENCHMARK_REPEATS_PER_CELL = 16
BENCHMARK_CASES = 5 * len(BENCHMARK_PERIOD_MULTS) * BENCHMARK_REPEATS_PER_CELL
NUM_RANDOM_ORBITS = 20000
N_WORKERS = max(1, cpu_count() // 2)
CHUNK_SIZE = 4
SEED_COUNT = 20

BASIC_FEATURE_COLUMNS = ["vx", "vy", "r", "sim-time"]
PHYSICS_FEATURE_COLUMNS = BASIC_FEATURE_COLUMNS + ["ecc", "period", "orbit_count"]
TARGET_COLUMNS = ["dt_euler", "dt_rk4", "dt_leapfrog"]

TRAINING_DATASET_NAME = "multi_integrator_data.csv"
BENCHMARK_DATASET_NAME = "challenging_integrator_data.csv"
MODEL_NAME = "integrator_recommender.pkl"

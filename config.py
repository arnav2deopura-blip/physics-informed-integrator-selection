from __future__ import annotations

from multiprocessing import cpu_count

GM = 1.0
PERTURB_EPS = 0.002
COLLISION_RADIUS = 0.01

NUM_RANDOM_ORBITS = 1000
BENCHMARK_CASES = 240
N_WORKERS = min(8, cpu_count())
CHUNK_SIZE = 4
SEED_COUNT = 10

BASIC_FEATURE_COLUMNS = ["vx", "vy", "r", "sim-time"]
PHYSICS_FEATURE_COLUMNS = BASIC_FEATURE_COLUMNS + ["ecc", "period"]
TARGET_COLUMNS = ["dt_euler", "dt_rk4", "dt_leapfrog"]

TRAINING_DATASET_NAME = "multi_integrator_data.csv"
BENCHMARK_DATASET_NAME = "challenging_integrator_data.csv"
MODEL_NAME = "integrator_recommender.pkl"

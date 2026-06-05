# A Physics-Informed Random Forest for Selecting Stable Timesteps in Two-Body Orbital Simulations

This repository contains the code and data for my numerical analysis research project. I studied how three integrators (Euler, RK4, and leapfrog) behave on planar two-body orbits, then trained a random forest to predict the largest stable timestep for each method without running a full brute-force search every time.

The full write-up is in `orbit_ml_paper.tex` and the PDF manuscript in the parent project folder. This README summarizes the science and explains how to reproduce the results.

## Why I did this

Orbital simulations depend heavily on two choices: which numerical integrator to use and how large the timestep can be. In practice, people often find those values by trial and error. That works for one orbit, but it becomes slow when many initial conditions need to be tested.

The physics of a simple two-body orbit is well understood, yet finite-step solvers still drift in energy and angular momentum if the timestep is too large. I wanted to see whether a machine learning model could learn that relationship from data and recommend a stable timestep before a long simulation starts.

## Research question

Can a physics-informed random forest predict the largest stable timestep for Euler, RK4, and leapfrog on a bound planar orbit, and can it identify which integrator allows the largest stable step for a given initial condition?

## Setup

I worked in normalized units with GM = 1. The central mass sits at the origin. The orbiting body is described by position and velocity (x, y, vx, vy).

For each orbit I tracked specific energy E and specific angular momentum L. A simulation was considered stable when both relative energy error and relative angular momentum error stayed below 1% for the full run.

I compared three integrators:

1. Forward Euler (first order)
2. Classical fourth-order Runge-Kutta, RK4
3. Leapfrog (second order, symplectic)

For each orbit and each integrator, I searched for the largest timestep that still met the 1% error threshold. If no timestep in the search range worked, I recorded a small fallback value of 1e-4.

## Data

**Training set.** I generated 20,000 random bound orbits by sampling vx, vy, and r over fixed ranges. Simulation length was chosen as a multiple of the orbital period, spanning roughly 0.05 to 300 orbits. After removing cases where every integrator failed, 19,380 usable examples remained.

**Benchmark set.** I built a separate set of 800 cases (5 dynamical regimes × 10 simulation durations × 16 repeats). The regimes were low eccentricity, mid eccentricity, high eccentricity, near-parabolic, and near-collision. These cases were sampled in eccentricity-periapsis space so that difficult orbits were included on purpose. About 35% of benchmark cases included a small perturbation to test slightly non-ideal conditions.

**Features.** The baseline model used vx, vy, r, and simulation time. The physics-informed model added eccentricity, orbital period, and orbit count (simulation time divided by period).

## Machine learning approach

I trained a multi-output random forest regressor to predict three timesteps at once: stable dt for Euler, RK4, and leapfrog. The recommended integrator was the one with the largest predicted stable timestep, since that usually means the most efficient choice at the fixed error threshold.

I compared the baseline feature set against the physics-informed set using a 70/30 train-test split and repeated the ablation over 20 random seeds. I also held out the highest-eccentricity 20% of orbits as a harder test set and compared the model against simple scaling rules based on orbital period and Keplerian time (r^1.5).

## Main results

### Integrator behavior

Euler failed to find any stable timestep in 72.9% of candidate orbits. RK4 failed 3.0% of the time and leapfrog failed 6.2% of the time.

In trajectory plots, Euler spirals away from the true orbit even at very small timesteps. RK4 and leapfrog stay close to the expected path at the timesteps selected by the model.

RK4 had the lowest short-term energy error in most cases. Leapfrog preserved angular momentum far better over long runs (relative error around 1e-14 in a representative case, compared with about 1e-6 for RK4 and 1e-4 for Euler).

Timestep sensitivity tests on an eccentric orbit (e ≈ 0.35) matched the expected convergence orders: first order for Euler, second for leapfrog, and fourth for RK4. The largest stable RK4 timestep scaled almost linearly with r^1.5, which matches Keplerian timing.

### Benchmark winners

On the 800-case benchmark, RK4 won 87.9% of cases under a combined score that balanced timestep size, peak conservation error, and long-term drift. Leapfrog won 7.9% and Euler won 4.2%. Leapfrog was competitive mainly in long, low-eccentricity runs where long-term conservation mattered more than short-term accuracy.

### Model performance

Adding physics-informed features improved RK4 prediction on the standard test split:

| Metric | Baseline features | Physics-informed features |
|--------|-------------------|---------------------------|
| R² | 0.9735 ± 0.0016 | 0.9857 ± 0.0015 |
| MAE | 0.02555 ± 0.00054 | 0.01836 ± 0.00059 |

On the hardest 20% of orbits by eccentricity, RK4 prediction improved from R² = 0.9267 and MAE = 0.04111 to R² = 0.9643 and MAE = 0.03000.

Simple tuned scaling rules on the training data had much higher error (MAE about 0.29 for a period rule and 0.24 for a Kepler scaling rule) than the random forest (MAE about 0.018).

When I converted the three predicted timesteps into a single integrator choice, the model matched the true best integrator 97.7% of the time on the held-out set (approximate 95% interval: 97.3% to 98.1%). Always choosing RK4 would have been correct 95.6% of the time, so the model gained about 2.1 percentage points by recovering the cases where leapfrog or Euler was actually better.

Orbit count was the most important feature (importance 0.456), followed by orbital period (0.197). Raw simulation time contributed very little (0.007) once those derived quantities were included.

ML prediction averaged about 38 ms per case, compared with about 1.33 s for brute-force search on the same sample, a speedup of roughly 35×.

Learning curves showed that error and model uncertainty both drop quickly through the first few thousand training orbits and then level off.

## Figures

Running the full pipeline saves 15 figures. Together they show:

1. Integrator trajectories at model-selected timesteps
2. Short-term energy conservation over time
3. RK4 stable timestep map over initial radius and tangential velocity
4. Stable RK4 timestep vs Keplerian scaling (r^1.5)
5. Timestep sensitivity and convergence order for an eccentric orbit
6. Random forest prediction uncertainty across state space
7. Feature-set comparison on high-eccentricity orbits
8. Random forest feature importances
9. Benchmark performance score vs eccentricity
10. Long-term angular momentum conservation
11. Long-term energy drift vs simulation length
12. Peak energy error vs eccentricity
13. Best integrator by dynamical regime and simulation length
14. Learning curve: prediction error vs training set size
15. Learning curve: model uncertainty vs training set size

## What this project shows

1. Euler is a poor default for this orbital problem; RK4 and leapfrog are far more reliable.
2. Stable timestep size follows physical scaling (especially with r^1.5), but a learned model captures more detail than a single scaling rule.
3. Physics-informed features (eccentricity, period, orbit count) improve prediction, especially on difficult orbits.
4. A random forest can act as a fast surrogate for brute-force timestep search in this restricted setting.

## What this project does not show

This is not a proof that one integrator is always best for every orbital system. The study is limited to planar bound two-body orbits with fixed GM = 1 and a fixed 1% error threshold. The model predicts labels from a specific search procedure; it does not replace careful numerical analysis for new physics, three-dimensional motion, N-body systems, or adaptive timestep methods.

The 97.7% integrator selection accuracy is measured against labels from my own dataset, not against an independent external benchmark.

## Limitations

The simulation is two-dimensional and uses no perturbations beyond the small term in the benchmark set. The random forest outputs piecewise-constant predictions, so it may not behave smoothly right at a stability boundary. In practice, using a small safety factor (such as 0.95 times the predicted timestep) would be a reasonable precaution.

## How to run

Requirements: Python 3.10 or newer, plus numpy, pandas, matplotlib, scipy, scikit-learn, and joblib.

```
pip install numpy pandas matplotlib scipy scikit-learn joblib
python run_study.py
```

The full pipeline generates training and benchmark CSV files, trains and evaluates models, prints metrics to the console and `output_log.txt`, saves the final model as `integrator_recommender.pkl`, and writes 15 figures to the `figures` folder.

Re-running from scratch takes a while because the timestep search runs over thousands of orbits in parallel. If the CSV files already exist, you can load them directly for faster model-only experiments.

## Repository layout

| File | Role |
|------|------|
| `run_study.py` | Entry point for the full study |
| `src/orbit_ml/config.py` | Constants, feature columns, dataset names |
| `src/orbit_ml/physics.py` | Integrators, orbital features, simulation helpers |
| `src/orbit_ml/search.py` | Stable timestep search and benchmark scoring |
| `src/orbit_ml/data.py` | Training and benchmark dataset generation |
| `src/orbit_ml/modeling.py` | Training, ablations, holdout tests, baselines |
| `src/orbit_ml/pipeline.py` | End-to-end workflow and printed reports |
| `src/orbit_ml/plotting.py` | All figure generation |
| `multi_integrator_data.csv` | Training data (generated) |
| `challenging_integrator_data.csv` | Benchmark data (generated) |
| `integrator_recommender.pkl` | Saved final model (generated) |
| `output_log.txt` | Console log from the last run |

## Related files

The LaTeX manuscript (`orbit_ml_paper.tex`) and PDF in the parent `Science 2026-2027 Project` folder contain the full introduction, methods detail, discussion, and references. I prepared a version of that paper for submission to the National High School Journal of Science (NHSJS).

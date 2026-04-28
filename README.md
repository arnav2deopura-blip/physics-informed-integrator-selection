# Physics-Informed Machine Learning for Orbital Integrator Optimization

## Overview

This project develops a physics-informed machine learning framework to predict stable timesteps and select optimal numerical integrators for orbital simulations. Traditional methods rely on computationally expensive brute-force testing, while this approach learns stability behavior directly from data.

## Research Question

Can a physics-informed machine learning model accurately predict stable timesteps and optimal integrators for orbital simulations, while outperforming traditional brute-force and heuristic methods?

## Methods

* Simulated 1000+ orbital systems across diverse regimes:

  * Low to high eccentricity
  * Near-parabolic trajectories
  * Near-collision scenarios
* Implemented three numerical integrators:

  * Euler
  * Runge-Kutta 4 (RK4)
  * Leapfrog
* Evaluated performance using:

  * Energy conservation error
  * Angular momentum error
  * Long-term drift
* Generated a dataset of stable timesteps for each integrator
* Trained a Random Forest model to predict stable timesteps
* Added physics-based features:

  * Orbital eccentricity
  * Orbital period

## Results

* RK4 outperformed other integrators in ~97% of tested cases
* Machine learning predictions:

  * ~83× improvement over naive timestep rules
  * ~20× improvement over Kepler-based scaling
* Physics-informed features significantly improved model accuracy
* Model achieved ~8× speedup compared to brute-force search

## Key Insight

While machine learning can accurately predict stability limits, the results show that RK4 is consistently the most robust integrator across the tested orbital regimes.

## Files

* `run_study.py` – starts the simulation
* `pipeline.py` - communicates to other files for when to do their work
* `physics.py` - contains logic for all three integrators, defines physical constants, and computes the orbits and related factors
* `search.py` - runs "brute-force" tests to find the exact time an orbit becomes unstable, which the ML tries to predict
* `data.py` - generates the orbits
* `config.py` - contains all the constants
* `modeling.py` - contains the ML logic
* `__init__.py` - tells Python that all these files belong together
* `plotting.py` - makes all the charts and graphs
* `multi_integrator_data.csv` – training dataset

## How to Run

1. Install dependencies:

```
pip install -r requirements.txt
```

2. Run the project:

```
python run_study.py
```

## Dependencies

* numpy
* pandas
* matplotlib
* scikit-learn
* scipy
* joblib

## Future Work

* Extend to N-body systems
* Test additional integrators (e.g., symplectic higher-order methods)
* Improve performance in extreme regimes
* Explore neural network models

## Author

Arnav Deopura

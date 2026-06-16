"""Machine Learning model for the 2026 Football World Cup.

Package with all the logic shared between the script (`run_pipeline.py`)
and the notebook (`World_Cup_2026.ipynb`):

- `data`       -> loading, cleaning, reconstruction of the 12 groups
- `elo`        -> chronological Elo ratings of the national teams
- `features`   -> feature engineering without information leakage
- `model`      -> Poisson model (Dixon-Coles) trained by maximum likelihood
- `tournament` -> Monte Carlo simulation of the tournament (groups + knockouts)
- `viz`        -> charts and HTML dashboard
"""

__version__ = "1.0.0"

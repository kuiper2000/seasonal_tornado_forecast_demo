# Seasonal Tornado Forecast — GitHub Demo

This folder contains a self-contained demonstration of the **seasonal tornado forecast model** described in the associated manuscript. The model uses **Sea Surface Temperature (SST) principal components (PCs)** as predictors to produce probabilistic forecasts of tornado activity over the continental United States (CONUS).

---

## Files

| File | Description |
|---|---|
| `Tornado_github.py` | Core model class (`tornado_git`) |
| `Tornado_github.ipynb` | End-to-end demonstration notebook |
| `github_demo.npz` | Tornado monthly count data (1992–2021, 1403 grid points) |
| `SST_EOFs_1995_2017_based.npz` | Pre-computed SST EOFs and PCs used as predictors |

---

## Input Data

### SST file — `SST_EOFs_1995_2017_based.npz`

| Key | Shape | Description |
|---|---|---|
| `pcs_obs_sst` | `(20, 15, 12, 30, 12)` | Observed SST PCs — 20 modes · 15 ensemble members · 12 init months · 30 years · 12 lead months |
| `EOF` | `(20, 360, 576)` | SST EOF spatial patterns (20 modes, latitude × longitude) |
| `variance` | `(20,)` | Explained variance fraction for each EOF mode |

> **Preprocessing**: ensemble members are averaged before use:
> ```python
> pcs_obs_sst = data_sst['pcs_obs_sst'].mean(axis=1)  # → (20, 12, 30, 12)
> ```
> Then reshaped to `(20, 12, 360)` for the model predictor array.

### Tornado file — `github_demo.npz`

| Key | Shape | Description |
|---|---|---|
| `tornado_month` | `(30, 1403)` | Monthly CONUS tornado counts (30 years × 1403 grid points) |
| `lat` | `(23, 61)` | Latitude grid |
| `lon` | `(23, 61)` | Longitude grid |

---

## Model Overview (`Tornado_github.py`)

The `tornado_git` class provides two main methods:

### 1. `_forecast(leave_one_out=True, normalize=True)`
Trains the regression model using a **leave-one-out cross-validation** scheme.

- Tornado counts at each grid point are converted to **ECDF percentiles**
- For each combination of SST mode, initialization month, and predictor month, a **ridge regression** is fitted between lagged SST PCs and tornado percentiles
- Returns:
  - `predict` — shape `(n_modes, 12, 12, n_years, n_tor_grid)`: cross-validated predictions
  - `coef` — shape `(12, 12, n_years, n_modes, n_tor_grid)`: regression coefficients

### 2. `predict_new(sst_pcs, init_month, pred_month, coef, n_modes=None)`
Applies pre-trained coefficients to forecast tornado activity for a **new, unseen year**.

| Argument | Type | Description |
|---|---|---|
| `sst_pcs` | `array (n_modes,)` | SST PC vector at the chosen predictor month |
| `init_month` | `int` (1–12) | Initialization month |
| `pred_month` | `int` (1–12) | Predictor month whose SST PCs are supplied |
| `coef` | `ndarray` | Pre-trained coefficients from `_forecast()` |
| `n_modes` | `int` or `None` | Number of SST modes to use (default: all) |

Returns `forecast (n_tor_grid,)` — predicted tornado percentile clipped to `[0, 1]`, and `coef_used (n_modes, n_tor_grid)` — mean coefficients used.

---

## High-Level Wrapper (`predict_conus_tornado`)

The notebook defines a convenient wrapper function that handles the full pipeline — from data preparation and cross-validation training through to forecasting and visualisation:

```python
result = predict_conus_tornado(
    target_year      = 2011,          # Year to forecast (within data range)
    pcs_obs_sst      = pcs_obs_sst,   # (20, 12, 30, 12) ensemble-averaged SST PCs
    EOFs             = EOFs,           # (20, 360, 576) EOF spatial patterns
    tornado          = tornado,        # (30, 1403) raw tornado counts
    tornado_lat_full = tornado_lat_full,
    tornado_lon_full = tornado_lon_full,
    init_month       = 12,            # Initialisation month (default: Dec)
    pred_month       = 12,            # SST predictor month  (default: Dec)
    n_modes          = 8,             # Number of SST modes  (default: 20)
    init_year        = 1992,          # First year in dataset
)
```

### Returns — `result` dict

| Key | Type | Description |
|---|---|---|
| `conus_total_forecast` | `float` | Predicted total CONUS tornado count (back-transformed from percentile space) |
| `conus_total_observed` | `float` | Observed total CONUS tornado count |
| `forecast_pct` | `ndarray (n_tor_grid,)` | Per-grid forecast in ECDF percentile space |
| `observed_pct` | `ndarray (n_tor_grid,)` | Per-grid observed percentile |
| `correlation` | `float` | Spatial Pearson r (forecast vs observed) |

### Output figure (6 panels)

| Panel | Content |
|---|---|
| Top-left | Forecast vs observed scatter plot (spatial correlation *r*) |
| Top-centre | Sorted grid-point comparison (observed vs forecast percentile) |
| Top-right | SST mode importance (mean absolute coefficient per mode) |
| Bottom-left | Forecast tornado percentile anomaly map (CONUS) |
| Bottom-centre | Observed tornado percentile anomaly map (CONUS) |
| Bottom-right | CONUS total tornado count: forecast vs observed in historical context |

---

## Notebook Walkthrough (`Tornado_github.ipynb`)

| Cell | Type | Content |
|---|---|---|
| **Cell 0** | Code | `git clone` the repository; install dependencies (`descartes`, `netCDF4`, `cartopy`) |
| **Cell 1** | Markdown | Section header — *Preparing User Defined Data* |
| **Cell 2** | Code | Load `SST_EOFs_1995_2017_based.npz` and `github_demo.npz`; average ensemble members; extract `pcs_obs_sst`, `EOFs`, `tornado`, `tornado_lat_full`, `tornado_lon_full`; instantiate `tornado_git` |
| **Cell 3** | Code | *(commented out)* Optional stand-alone `_forecast()` call for inspecting raw cross-validation output |
| **Cell 4** | Markdown | Section header — *Model Main Class* |
| **Cell 5** | Code | Definition of `predict_conus_tornado()` — the high-level wrapper |
| **Cell 6** | Markdown | Section header — *Run Model* |
| **Cell 7** | Code | **User input cell** — set `target_year`, `INIT_MONTH`, `PRED_MONTH`, `N_MODES`, then call `predict_conus_tornado()` |

---

## Quick Start (Google Colab)

```python
# ── Cell 0: setup ─────────────────────────────────────────────────────────────
!git clone https://github.com/kuiper2000/seasonal_tornado_forecast_demo.git
!pip install descartes netCDF4 cartopy

# ── Cell 2: load data ─────────────────────────────────────────────────────────
import numpy as np, sys
sys.path.insert(0, '/content/seasonal_tornado_forecast_demo/')
from Tornado_github import tornado_git

data_sst     = np.load('/content/seasonal_tornado_forecast_demo/SST_EOFs_1995_2017_based.npz')
data_tornado = np.load('/content/seasonal_tornado_forecast_demo/github_demo.npz')

pcs_obs_sst      = data_sst['pcs_obs_sst'].mean(axis=1)   # (20, 12, 30, 12)
EOFs             = data_sst['EOF']                          # (20, 360, 576)
tornado          = data_tornado['tornado_month']            # (30, 1403)
tornado_lat_full = data_tornado['lat']                      # (23, 61)
tornado_lon_full = data_tornado['lon']                      # (23, 61)

# ── Cell 7: forecast ──────────────────────────────────────────────────────────
target_year = 2011   # ← change to any year in [1992, 2021]
INIT_MONTH  = 12     # Initialisation month (1–12)
PRED_MONTH  = 12     # SST predictor month  (1–12)
N_MODES     = 8      # Number of SST modes

result = predict_conus_tornado(
    target_year      = target_year,
    pcs_obs_sst      = pcs_obs_sst,
    EOFs             = EOFs,
    tornado          = tornado,
    tornado_lat_full = tornado_lat_full,
    tornado_lon_full = tornado_lon_full,
    init_month       = INIT_MONTH,
    pred_month       = PRED_MONTH,
    n_modes          = N_MODES,
)

print(f"CONUS tornado forecast : {result['conus_total_forecast']:,.0f}")
print(f"CONUS tornado observed : {result['conus_total_observed']:,.0f}")
print(f"Spatial correlation r  : {result['correlation']:.3f}")
```

---

## Dependencies

```
numpy
pandas
scipy
statsmodels
scikit-learn
matplotlib
cartopy
netCDF4
descartes
```

---

## Notes

- **Predictor averaging**: `pcs_obs_sst` has a 15-member ensemble dimension (axis 1) that is averaged before use, yielding shape `(20, 12, 30, 12)`, then reshaped to `(20, 12, 360)`.
- **ECDF percentile space**: all forecasts and observations are expressed as grid-point-wise empirical CDF percentiles in `[0, 1]`. Subtract 0.5 to obtain anomalies relative to the climatological median.
- **CONUS total count**: `predict_conus_tornado()` back-transforms percentiles to raw tornado counts via inverse-ECDF (`np.percentile` over the training distribution) before summing.
- **`tornado_percentile_min`**: after `_forecast()` is called, the per-grid minimum percentile is stored on the model instance and automatically added back by `predict_new()` to restore the original ECDF scale.
- **N_MODES**: the default in `predict_conus_tornado()` is 20, but the demo notebook uses `N_MODES = 8`; users can tune this parameter for different skill–complexity trade-offs.

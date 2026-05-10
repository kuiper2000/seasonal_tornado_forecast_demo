# Seasonal CONUS Tornado Forecast — GitHub Demo

A self-contained demonstration of the **seasonal tornado forecast model** described in the associated manuscript. The model uses **Sea Surface Temperature (SST) principal components (PCs)** as predictors and applies **ridge / OLS regression with leave-one-out cross-validation (LOOCV)** to produce probabilistic forecasts of tornado activity over the continental United States (CONUS).

---

## Repository Contents

| File | Description |
|---|---|
| `Tornado_github.py` | Core model class `tornado_git` |
| `Tornado_github.ipynb` | End-to-end demonstration notebook |
| `SST_EOFs_1995_2017_based.npz` | Pre-computed SST EOFs and ensemble PCs (predictors) |
| `github_demo.npz` | Monthly CONUS tornado counts and grid (1992–2021) |

---

## Input Data

### `SST_EOFs_1995_2017_based.npz`

| Key | Shape | Description |
|---|---|---|
| `pcs_obs_sst` | `(20, 15, 12, 30, 12)` | Observed SST PCs — 20 modes · 15 ensemble members · 12 init months · 30 years · 12 lead months |
| `EOF` | `(20, 360, 576)` | SST EOF spatial patterns |
| `variance` | `(20,)` | Explained variance fraction per mode |

> **Ensemble averaging** is applied before use:
> ```python
> pcs_obs_sst = data_sst['pcs_obs_sst'].mean(axis=1)  # → (20, 12, 30, 12)
> ```

### `github_demo.npz`

| Key | Shape | Description |
|---|---|---|
| `tornado_month` | `(30, 1403)` | Annual CONUS tornado counts — 30 years × 1403 grid points |
| `lat` | `(23, 61)` | Latitude grid |
| `lon` | `(23, 61)` | Longitude grid |

---

## Core Model — `Tornado_github.py`

### Class `tornado_git`

```python
model = tornado_git(tornado_data, predictor, init_year)
```

| Argument | Shape | Description |
|---|---|---|
| `tornado_data` | `(n_years, n_grid)` | Raw tornado counts |
| `predictor` | `(n_modes, 12, n_years×12)` | Ensemble-averaged SST PCs |
| `init_year` | `int` | First year of the dataset (e.g. 1992) |

---

### `_forecast(leave_one_out=True, normalize=True)`

Trains the regression model over all combinations of SST mode, initialisation month, and predictor month using **leave-one-out cross-validation**.

**Steps:**
1. Convert tornado counts to **ECDF percentiles** at each grid point.
2. Subtract the per-grid minimum (`tor_pct_min`) so the training target starts at 0.
3. For each left-out year, fit `LinearRegression(fit_intercept=True)` on the remaining years using lagged SST PCs as predictors.
4. Store **both** regression slopes (`coef_`) **and** the intercept (`intercept_`) on the model instance.
5. Record the LOOCV prediction using `model_ols.predict()` (slopes + intercept).

**Returns:**
- `predict` — shape `(n_modes, 12, 12, n_years, n_grid)` — LOOCV predictions (min-removed space)
- `coef` — shape `(12, 12, n_years, n_modes, n_grid)` — regression slopes per fold

**Instance attributes set:**
- `self.tornado_percentile_min` — `(n_grid,)` per-grid ECDF minimum
- `self.intercept_` — `(12, 12, n_years, n_grid)` per-fold OLS intercepts

---

### `predict_new(sst_pcs, init_month, pred_month, coef, n_modes=None)`

Applies pre-trained coefficients to forecast tornado activity for any SST PC vector.

**Full reconstruction (min-removed → original ECDF scale):**
```
forecast = sst_pcs[:n_modes] @ mean(coef)      # slopes × SST signal
         + mean(intercept_)                    # ≈ climatological mean − tor_pct_min
         + tor_pct_min                         # restore original ECDF floor
forecast = clip(forecast, 0, 1)
```

> ⚠️ **Important:** The intercept term is essential. Without it, the forecast anchors at `tor_pct_min` (the minimum ever observed at each grid point) instead of the climatological mean, producing a systematic negative anomaly bias across all grid points.

| Argument | Type | Description |
|---|---|---|
| `sst_pcs` | `array (n_modes,)` | SST PC vector for the target period |
| `init_month` | `int` 1–12 | Initialisation month |
| `pred_month` | `int` 1–12 | Predictor month whose SST PCs are supplied |
| `coef` | `ndarray` | Pre-trained slopes from `_forecast()` |
| `n_modes` | `int` or `None` | Number of SST modes to use |

**Returns:** `forecast (n_grid,)`, `coef_used (n_modes, n_grid)`

---

## Notebook Walkthrough — `Tornado_github.ipynb`

The notebook is split into **two independent parts**.

### Cell 0 — Setup
Clone the repository and install dependencies (commented out for local use).

### Cell 1 — Markdown header
*"Preparing User Defined Data"*

### Cell 2 — Load data & initialise
Load `.npz` files, average ensemble members, extract the five input arrays:
```python
pcs_obs_sst      # (20, 12, 30, 12)  ensemble-averaged SST PCs
EOFs             # (20, 360, 576)    EOF spatial patterns
tornado          # (30, 1403)        raw tornado counts
tornado_lat_full # (23, 61)
tornado_lon_full # (23, 61)
```

### Cell 3 — Markdown header
*"Part 1 — Model Training"*

### Cell 4 — `train_tornado_model()` definition
Defines the training wrapper. Key outputs:

| Output key | Description |
|---|---|
| `model` | Trained `tornado_git` instance (intercept and `tor_pct_min` stored) |
| `coef` | `(12, 12, n_years, n_modes, n_grid)` regression slopes |
| `tor_arr` | `(n_years, n_grid)` reshaped tornado counts |
| `obs_full` | `(n_years, n_grid)` ECDF percentile observations |
| `loocv_r_years` | `(n_years,)` per-year LOOCV spatial Pearson *r* |
| `mean_loocv_r` | Scalar mean LOOCV skill |
| `mode_importance` | `(n_modes,)` mean \|coefficient\| per SST mode |

**Training figure (2 panels):**
- **Left** — LOOCV spatial *r* per year (blue = positive, red = negative), dashed mean line
- **Right** — SST mode importance bar chart (dominant mode highlighted in crimson)

### Cell 5 — Part 1 user inputs
```python
INIT_MONTH = 12   # Initialisation month (1–12)
PRED_MONTH = 12   # SST predictor month  (1–12)
N_MODES    = 8    # Number of SST modes

training = train_tornado_model(pcs_obs_sst, tornado,
                                tornado_lat_full, tornado_lon_full,
                                init_month=INIT_MONTH,
                                pred_month=PRED_MONTH,
                                n_modes=N_MODES)
```

### Cell 6 — Markdown header
*"Part 2 — Prediction"*

### Cell 7 — `predict_conus_tornado()` definition
Defines the prediction wrapper including `_ecdf_invert()` helper.

**Anomaly computation:**
```python
# clim_ref = per-grid climatological mean ECDF percentile
clim_ref  = obs_full.mean(axis=0)          # (n_grid,)
fore_anom = forecast_pct[active] - clim_ref[active]
```
> Using `clim_ref` instead of the fixed value `0.5` is critical. Sparse tornado regions can have 70–90 % zero-tornado years, giving `tor_pct_min ≈ 0.7–0.9` and `clim_ref ≈ 0.75`. Subtracting `0.5` from such grid points always yields a spurious positive anomaly.

**Back-transform (percentile → raw count):**
Uses `_ecdf_invert()` — the true step-function inverse of the ECDF — rather than `np.percentile()` (linear interpolation), which is inconsistent with the forward transform and systematically underestimates integer tornado counts.

**Prediction figure layout adapts to context:**

| Layout | Condition | Panels |
|---|---|---|
| 2 × 3 | `obs_tornado_year` provided | Scatter, sorted grid comparison, mode importance, forecast map, observed map, CONUS total bar |
| 1 × 3 | Forecast only | Mode importance, forecast map, CONUS total bar |

### Cell 8 — Part 2 user inputs
```python
target_year  = 2011   # ← any year in [1992, 2021]

# Option A — SST PCs from a training year (enables verification)
sst_pcs_user = training['model'].predictor[:, init_month-1, predictor_posi[year_idx]]

# Option B — provide your own vector for out-of-sample forecast
# sst_pcs_user = np.array([...])   # shape: (N_MODES,)

OBS_YEAR = target_year   # set to None for pure out-of-sample forecast

result = predict_conus_tornado(sst_pcs_user, training,
                                tornado_lat_full, tornado_lon_full,
                                obs_tornado_year=OBS_YEAR)
```

---

## Quick Start (Google Colab)

```python
# Cell 0
!git clone https://github.com/kuiper2000/seasonal_tornado_forecast_demo.git
%pip install descartes netCDF4 cartopy

# Cell 2 — load data
import numpy as np, sys
sys.path.insert(0, '/content/seasonal_tornado_forecast_demo/')
from Tornado_github import tornado_git

data_sst     = np.load('/content/seasonal_tornado_forecast_demo/SST_EOFs_1995_2017_based.npz')
data_tornado = np.load('/content/seasonal_tornado_forecast_demo/github_demo.npz')

pcs_obs_sst      = data_sst['pcs_obs_sst'].mean(axis=1)   # (20, 12, 30, 12)
EOFs             = data_sst['EOF']
tornado          = data_tornado['tornado_month']            # (30, 1403)
tornado_lat_full = data_tornado['lat']
tornado_lon_full = data_tornado['lon']

# Cell 5 — train
training = train_tornado_model(
    pcs_obs_sst=pcs_obs_sst, tornado=tornado,
    tornado_lat_full=tornado_lat_full, tornado_lon_full=tornado_lon_full,
    init_month=12, pred_month=12, n_modes=8,
)

# Cell 8 — predict (Option A: in-sample year for verification)
import pandas as pd
target_year    = 2011
year_idx       = target_year - training['init_year']
dates_1M       = pd.date_range(f"{training['init_year']}-{training['init_month']:02d}",
                                periods=training['tor_arr'].shape[0]*12, freq='1ME')
predictor_posi = np.where(dates_1M.month == training['pred_month'])[0]
sst_pcs_user   = training['model'].predictor[:, training['init_month']-1,
                                               predictor_posi[year_idx]]

result = predict_conus_tornado(
    sst_pcs_user=sst_pcs_user, training_result=training,
    tornado_lat_full=tornado_lat_full, tornado_lon_full=tornado_lon_full,
    obs_tornado_year=target_year,
)
print(f"Forecast: {result['conus_total_forecast']:,.0f}  |  "
      f"Observed: {result['conus_total_observed']:,.0f}  |  "
      f"r = {result['correlation']:.3f}")
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

## Known Implementation Notes

| Issue | Root cause | Fix applied |
|---|---|---|
| Persistent negative anomaly bias in `predict_new()` | OLS `intercept_` was not saved during training; forecast anchored at `tor_pct_min` (floor) instead of climatological mean | `intercept_` now stored per fold in `_forecast()` and applied in `predict_new()` |
| Spurious spatial anomaly patterns (some regions always red/blue) | Fixed `0.5` used as anomaly reference — invalid for non-uniform ECDF distributions | Anomaly now computed as `forecast_pct − clim_ref` where `clim_ref = obs_full.mean(axis=0)` |
| CONUS total count underestimation | `np.percentile()` (linear interpolation) used as ECDF inverse — inconsistent with step-function forward transform | Replaced with `_ecdf_invert()`: returns smallest observed count with ECDF ≥ forecast percentile |

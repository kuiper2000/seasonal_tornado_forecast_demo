# Seasonal Tornado Forecast — GitHub Demo

This folder contains a self-contained demonstration of the **seasonal tornado forecast model** described in the associated manuscript. The model uses **Sea Surface Temperature (SST) principal components (PCs)** as predictors to produce probabilistic forecasts of tornado activity over the continental United States.

---

## Files

| File | Description |
|---|---|
| `Tornado_github.py` | Core model class (`tornado_git`) |
| `Tornado_github.ipynb` | End-to-end demonstration notebook |
| `github_demo.npz` | Tornado monthly count data (1992–2021, 1403 grid points) |
| `SST_EOFs_1995_2017_based.npz` | Pre-computed SST EOFs and PCs used as predictors |

---

## Model Overview (`Tornado_github.py`)

The `tornado_git` class provides three main components:

### 1. `_forecast(leave_one_out=True, normalize=True)`
Trains the regression model using a **leave-one-out cross-validation** scheme.

- Tornado counts are converted to **ECDF percentiles** at each grid point
- For each combination of SST mode, initialization month, and predictor month, a **ridge regression** is fitted between lagged SST PCs and tornado percentiles
- Returns:
  - `predict` — array of shape `(n_modes, 12, 12, n_years, n_tor_grid)`: cross-validated predictions
  - `coef` — array of shape `(12, 12, n_years, n_modes, n_tor_grid)`: regression coefficients

### 2. `predict_new(sst_pcs, init_month, pred_month, coef, n_modes=None)`
Applies pre-trained coefficients to forecast tornado activity for a **new, unseen year**.

- `sst_pcs`: SST PC vector for the predictor month `(n_modes,)`
- `init_month` / `pred_month`: integers 1–12 specifying the initialization and predictor months
- `coef`: pre-trained coefficients from `_forecast()`
- Returns `forecast (n_tor_grid,)` — predicted tornado percentile clipped to `[0, 1]`, and `coef_used (n_modes, n_tor_grid)` — mean coefficients used

---

## Notebook Walkthrough (`Tornado_github.ipynb`)

| Cell | Content |
|---|---|
| **Cell 0** | Load SST EOF/PC data and tornado counts; instantiate `tornado_git` |
| **Cell 1** | Run `_forecast()` to train the model and retrieve `predict` and `coef` |
| **Cell 2** | `predict_new()` demonstration on a randomly selected year (1994); scatter and sorted grid-point comparison plots |
| **Cell 3** | Cartopy forecast map — side-by-side forecast vs observed tornado percentile anomaly (relative to 0.5 climatology), masked to active grid points over the continental US |

---

## Quick Start

```python
import numpy as np
from Tornado_github import tornado_git

# Load data
data_sst     = np.load('SST_EOFs_1995_2017_based.npz')
data_tornado = np.load('github_demo.npz')

pcs_obs_sst = data_sst['pcs_obs_sst'].mean(axis=1)   # (20, 12, 360)
tornado     = data_tornado['tornado_month']            # (30, 1403)

# Instantiate and train
model = tornado_git(tornado_data=tornado,
                    predictor=np.reshape(pcs_obs_sst, [20, 12, 30*12]),
                    init_year=1992)
predict, coef = model._forecast()

# Forecast a new year using June SST PCs with July initialization
sst_pcs_new = ...   # shape (20,) — SST PCs for the new year
forecast, coef_used = model.predict_new(
    sst_pcs    = sst_pcs_new,
    init_month = 7,    # July initialization
    pred_month = 6,    # June SST PCs
    coef       = coef,
    n_modes    = 20
)
# forecast: (1403,) — predicted tornado percentile at each grid point, clipped to [0, 1]
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
```

---

## Notes

- The best forecast skill (Pearson r ≈ 0.60) is found using **July initialization** with **June SST PCs** as predictors, targeting spring tornado activity
- The dominant SST predictor mode is **Mode 13**, based on mean absolute regression coefficient magnitude
- Forecast values are **ECDF percentiles** clipped to `[0, 1]`; subtract 0.5 for anomaly relative to climatological median
- The `predict_new()` method stores `tornado_percentile_min` on the model instance after `_forecast()` is called, and automatically adds it back to restore the original percentile scale

#!/usr/bin/env python
# coding: utf-8

import numpy as np
from   netCDF4 import Dataset as NetCDFFile
import geopandas
import descartes
import pandas as pd
from   statsmodels.distributions.empirical_distribution import ECDF
from   sklearn import linear_model
from   datetime import datetime
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
regr = linear_model.LinearRegression()


class tornado_git():
    def __init__(self, tornado_data=np.ndarray, predictor=np.ndarray, init_year=int):
        self.tornado   = tornado_data
        self.predictor = predictor    # shape: (n_modes, 12, n_years*12)
        self.init_year = init_year

        # tornado data has dimension of time X 1
        # sst data has dimension of time X lat X lon

    def _quantile_mapping(self, input_variable):
        ecdf       = ECDF(input_variable)
        percentile = ecdf(input_variable)
        return percentile

    def _min_max_scaling(self, input_variable, axis):
        input_min    = np.min(input_variable, axis=axis)
        scaled_input = input_variable - input_min
        return scaled_input

    def _find_next_target_month(self, dates, month):
        # the input should have a format of dd/mm/yyyy
        string_input_with_date = dates
        current_year           = datetime.strptime(string_input_with_date, "%d/%m/%Y")
        string_input_with_date = '' + str(current_year.day) + '/' + str(month).zfill(2) + '/' + str(current_year.year) + ''
        first_year             = datetime.strptime(string_input_with_date, "%d/%m/%Y")
        string_input_with_date = '' + str(current_year.day) + '/' + str(month).zfill(2) + '/' + str(current_year.year + 1) + ''
        second_year            = datetime.strptime(string_input_with_date, "%d/%m/%Y")

        if current_year <= first_year:
            return first_year.year
        else:
            return second_year.year

    def _forecast(self, leave_one_out=True, normalize=True):
        pcs_obs_sst        = self.predictor
        dim                = np.shape(self.tornado)
        dim_sst            = np.shape(self.predictor)
        self.tornado       = np.reshape(self.tornado, [dim[0], int(np.size(self.tornado) / dim[0])])
        tornado_percentile = np.zeros(np.shape(self.tornado))
        for i in range(int(np.size(self.tornado) / dim[0])):
            tornado_percentile[:, i] = self._quantile_mapping(self.tornado[:, i])
        tornado_dates               = pd.date_range(str(self.init_year) + '-03', periods=dim[0], freq='12ME')
        tornado_percentile_min      = np.min(tornado_percentile, axis=0)
        self.tornado_percentile_min = tornado_percentile_min   # stored for use in predict_new()
        tornado_percentile          = tornado_percentile - tornado_percentile_min

        coef    = np.zeros((12, 12, dim[0], dim_sst[0], dim[1]))
        predict = np.zeros((int(dim_sst[0]), 12, 12, dim[0], dim[1]))

        for mode in range(1, int(dim_sst[0]) + 1):
            print('working on mode = ' + str(mode))
            for init_month in range(1, 13):         # initialization month
                dates_1M = pd.date_range(
                    str(self.init_year) + '-' + str(init_month).zfill(2),
                    periods=dim[0] * 12, freq='1ME'
                )

                for month in range(1, 13):          # month used as predictor
                    predictor_posi   = np.squeeze(np.where(dates_1M.month == month))
                    dates_predictor  = dates_1M[predictor_posi]
                    target_init_year = self._find_next_target_month(
                        '15/' + str(dates_predictor[0].month) + '/' + str(dates_predictor[0].year), 3
                    )

                    target_posi = np.squeeze(np.where(tornado_dates.year >= target_init_year))
                    size_min    = np.min([np.size(target_posi), np.size(predictor_posi)])
                    series1     = tornado_percentile[target_posi[0:size_min], :]
                    series2     = self.predictor[0:mode, init_month - 1, predictor_posi[0:size_min]]

                    for year in range(target_init_year, self.init_year + dim[0]):
                        if leave_one_out:
                            series4 = np.delete(series2, year - target_init_year, axis=1)
                            series3 = np.delete(series1, year - target_init_year, axis=0)
                        else:
                            series4 = series2
                            series3 = series1

                        regularization = np.std(series4) ** 2
                        ans = np.linalg.inv(
                            series4.dot(series4.T) + regularization * 0.001 * np.eye(series4.shape[0])
                        ).dot(series4.dot(series3))

                        coef[init_month - 1, month - 1, year - self.init_year, 0:mode, :] = ans
                        X_val        = np.zeros((mode, 1))
                        Y_val        = np.zeros((1, 1))
                        X_val[:, 0]  = np.transpose(series2[0:mode, year - target_init_year])[:]
                        predict[mode - 1, init_month - 1, month - 1, year - self.init_year, :] = X_val.T.dot(ans)

                        if normalize:
                            Y_train   = series3
                            X_train   = series4.T
                            model_ols = linear_model.LinearRegression()
                            model_ols.fit(X_train, Y_train)
                            coef[init_month - 1, month - 1, year - self.init_year, 0:mode, :] = (model_ols.coef_).T
                            X_val        = np.zeros((mode, 1))
                            X_val[:, 0]  = np.transpose(series2[0:mode, year - target_init_year])[:]
                            predict[mode - 1, init_month - 1, month - 1, year - self.init_year, :] = \
                                model_ols.predict(np.transpose(X_val))[0]

        return predict, coef

    def predict_new(self, sst_pcs, init_month, pred_month, coef, n_modes=None):
        """
        Forecast tornado activity for a new (unseen) year using pre-trained
        regression coefficients returned by _forecast().

        Parameters
        ----------
        sst_pcs    : array-like, shape (n_modes,)
                     SST principal components at the chosen predictor month
                     for the year to be forecast.
        init_month : int (1-12)
                     Initialization month used during training.
        pred_month : int (1-12)
                     Predictor month whose SST PCs are supplied in sst_pcs.
        coef       : ndarray, shape (12, 12, n_years, n_modes, n_tor_months)
                     Pre-trained coefficients from _forecast().
                     Coefficients are averaged over the year dimension so
                     the full training-period mean is used for inference.
        n_modes    : int or None
                     Number of SST modes to use. Defaults to all modes
                     present in sst_pcs.

        Returns
        -------
        forecast   : ndarray, shape (n_tor_months,)
                     Predicted tornado percentile for every tornado grid point.
        coef_used  : ndarray, shape (n_modes, n_tor_months)
                     Mean regression coefficients used for this prediction.
        """
        sst_pcs = np.asarray(sst_pcs).ravel()       # (n_modes,)
        if n_modes is None:
            n_modes = sst_pcs.shape[0]

        X = sst_pcs[:n_modes]                        # (n_modes,)

        # Average pre-trained coef over all training years for the
        # chosen (init_month, pred_month) combination.
        # coef[init_month-1, pred_month-1] has shape (n_years, n_modes, n_tor_months)
        coef_used = coef[init_month - 1,
                         pred_month - 1,
                         :,
                         :n_modes, :].mean(axis=0)   # (n_modes, n_tor_months)

        forecast = X @ coef_used                     # (n_tor_months,)  — still in min-removed space

        # Add back the per-grid-point minimum so the forecast is in the same
        # units as the original ECDF percentile (fair comparison with observations)
        if hasattr(self, 'tornado_percentile_min'):
            forecast = forecast + self.tornado_percentile_min

        # Clip to [0, 1] — by definition a percentile cannot be outside this range
        forecast = np.clip(forecast, 0.0, 1.0)

        return forecast, coef_used

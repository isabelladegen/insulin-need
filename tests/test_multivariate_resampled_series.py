import os

import numpy as np
import pytest
from hamcrest import *

from src.configurations import Configuration
from src.continuous_series import Cols
from src.multivariate_resampled_series import MultivariateResampledSeries

from src.stats import DailyTimeseries, WeeklyTimeseries


@pytest.mark.skipif(not os.path.isdir(Configuration().perid_data_folder), reason="reads real data")
def test_returns_resampled_df_for_hours_of_day():
    zip_id = '14092221'
    sampling = DailyTimeseries()
    series = MultivariateResampledSeries(zip_id, Cols.Mean, sampling)

    df = series.get_multivariate_df()

    no_of_data_points = 6072  # that's 253 days of data with IOB, COB, BG data every 60 min
    assert_that(df.shape, is_((no_of_data_points, 3)))
    assert_that(len(df[sampling.iob_col]), is_(no_of_data_points))
    assert_that(len(df[sampling.cob_col]), is_(no_of_data_points))
    assert_that(len(df[sampling.bg_col]), is_(no_of_data_points))


@pytest.mark.skipif(not os.path.isdir(Configuration().perid_data_folder), reason="reads real data")
def test_returns_resampled_df_for_days_of_week():
    zip_id = '14092221'
    sampling = WeeklyTimeseries()
    series = MultivariateResampledSeries(zip_id, Cols.Mean, sampling)

    df = series.get_multivariate_df()

    no_of_data_points = 343  # 49 weeks of data
    assert_that(df.shape, is_((no_of_data_points, 3)))
    assert_that(len(df[sampling.iob_col]), is_(no_of_data_points))
    assert_that(len(df[sampling.cob_col]), is_(no_of_data_points))
    assert_that(len(df[sampling.bg_col]), is_(no_of_data_points))


@pytest.mark.skipif(not os.path.isdir(Configuration().perid_data_folder), reason="reads real data")
def test_calculates_resampled_3d_nparray_of_weekly_ts():
    zip_id = '14092221'
    sampling = WeeklyTimeseries()
    series = MultivariateResampledSeries(zip_id, Cols.Mean, sampling)

    result = series.get_multivariate_3d_numpy_array()

    assert_that(result.shape, is_((49, 7, 3)))


@pytest.mark.skipif(not os.path.isdir(Configuration().perid_data_folder), reason="reads real data")
def test_calculates_resampled_3d_nparray_of_daily_ts():
    zip_id = '14092221'
    sampling = DailyTimeseries()
    series = MultivariateResampledSeries(zip_id, Cols.Mean, sampling)

    result = series.get_multivariate_3d_numpy_array()

    assert_that(result.shape, is_((253, 24, 3)))


@pytest.mark.skipif(not os.path.isdir(Configuration().perid_data_folder), reason="reads real data")
def test_calculates_resampled_1d_nparray_of_weekly_ts():
    zip_id = '14092221'
    sampling = WeeklyTimeseries()
    series = MultivariateResampledSeries(zip_id, Cols.Mean, sampling)

    iob = series.get_1d_numpy_array(sampling.iob_col)
    cob = series.get_1d_numpy_array(sampling.cob_col)
    bg = series.get_1d_numpy_array(sampling.bg_col)

    assert_that(iob.shape, is_((49, 7, 1)))
    assert_that(list(series.get_multivariate_df()[sampling.iob_col]) == list(iob.reshape(7 * 49).astype(np.float32)))
    assert_that(cob.shape, is_((49, 7, 1)))
    assert_that(list(series.get_multivariate_df()[sampling.cob_col]) == list(cob.reshape(7 * 49).astype(np.float32)))
    assert_that(bg.shape, is_((49, 7, 1)))
    assert_that(list(series.get_multivariate_df()[sampling.bg_col]) == list(bg.reshape(7 * 49).astype(np.float32)))


@pytest.mark.skipif(not os.path.isdir(Configuration().perid_data_folder), reason="reads real data")
def test_calculates_resampled_1d_nparray_of_daily_ts():
    zip_id = '14092221'
    sampling = DailyTimeseries()
    series = MultivariateResampledSeries(zip_id, Cols.Mean, sampling)

    iob = series.get_1d_numpy_array(sampling.iob_col)
    cob = series.get_1d_numpy_array(sampling.cob_col)
    bg = series.get_1d_numpy_array(sampling.bg_col)

    assert_that(iob.shape, is_((253, 24, 1)))
    assert_that(list(series.get_multivariate_df()[sampling.iob_col]) == list(iob.reshape(24 * 253).astype(np.float32)))
    assert_that(cob.shape, is_((253, 24, 1)))
    assert_that(list(series.get_multivariate_df()[sampling.cob_col]) == list(cob.reshape(24 * 253).astype(np.float32)))
    assert_that(bg.shape, is_((253, 24, 1)))
    assert_that(list(series.get_multivariate_df()[sampling.bg_col]) == list(bg.reshape(24 * 253).astype(np.float32)))

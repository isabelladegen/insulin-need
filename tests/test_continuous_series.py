import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest
from hamcrest import *

from src.configurations import Configuration
from src.continuous_series import ContinuousSeries, Resolution
from src.helper import device_status_file_path_for
from src.preprocess import number_of_interval_in_days
from src.read import read_flat_device_status_df_from_file
from tests.helper.BgDfBuilder import create_time_stamps

# build fake data
max_interval = 180  # how frequent readings need per day, 60=every hour, 180=every two hours
min_days_of_data = 30  # how many days of consecutive readings with at least a reading every interval
sample_rule = '1D'
min_series_length = number_of_interval_in_days(min_days_of_data, max_interval)
start_date1 = datetime(year=2019, month=12, day=25, hour=12, minute=0, tzinfo=timezone.utc)
start_date2 = datetime(year=2021, month=1, day=3, hour=1, minute=0, tzinfo=timezone.utc)
times1 = create_time_stamps(start_date1, min_series_length, max_interval)
times2 = create_time_stamps(start_date2, 2 * min_series_length, max_interval)
values1 = list(np.random.uniform(low=0.1, high=14.6, size=min_series_length))
values2 = list(np.random.uniform(low=0.5, high=10.6, size=2 * min_series_length))
values3 = list(np.random.uniform(low=90, high=400, size=min_series_length + 2 * min_series_length))
time_col = 't'
value_col = 'v'
value_col2 = 'another col'
times = times1 + times2
values = values1 + values2
df = pd.DataFrame(data={time_col: times, value_col: values, value_col2: values3})


def test_plots_resampled_sub_series():
    series = ContinuousSeries(df, min_days_of_data, max_interval, time_col, value_col, sample_rule)
    # no asserts as it generates a plot
    series.plot_resampled_series()


def test_describes_series():
    series = ContinuousSeries(df, min_days_of_data, max_interval, time_col, value_col2, sample_rule)
    series.describe()


def test_returns_index_and_value_column_for_resampled_value():
    series = ContinuousSeries(df, min_days_of_data, max_interval, time_col, value_col2, sample_rule)

    index = -1
    col = 'mean'
    x, y = series.get_resampled_x_and_y_for(index, col)

    resampled_df = series.resampled_series[index]
    assert_that(len(x), is_(resampled_df.shape[0]))
    assert_that(len(y), is_(resampled_df.shape[0]))
    assert_that(x, is_(list(resampled_df.index)))
    assert_that(y.equals(resampled_df[value_col2][col].astype(np.float64)))


def test_plots_resampled_z_score_normalised_value():
    series = ContinuousSeries(df, min_days_of_data, max_interval, time_col, value_col, sample_rule)
    # no asserts as it generates a plot
    series.plot_z_score_normalised_resampled_series()


def test_plots_heatmap_of_min_of_values_for_days_of_weeks_months():
    series = ContinuousSeries(df, min_days_of_data, max_interval, time_col, value_col, sample_rule)

    # no asserts as it generates a plot
    series.plot_heatmap_resampled(resolution=Resolution.DaysMonths, aggfunc=np.mean,
                                  resample_col='min')


def test_plots_heatmap_of_mean_of_values_for_days_of_weeks_hours():
    series = ContinuousSeries(df, 1, max_interval, time_col, value_col, "1H")

    # no asserts as it generates a plot
    series.plot_heatmap_resampled(resolution=Resolution.DaysHours, aggfunc=np.mean,
                                  resample_col='mean')


def test_plots_clustered_heathmap_of_mean_values_for_days_of_weeks_and_months():
    series = ContinuousSeries(df, min_days_of_data, max_interval, time_col, value_col, sample_rule)

    # no asserts as it generates a plot
    series.plot_clustered_heatmap_resampled(resolution=Resolution.DaysMonths, aggfunc=np.mean,
                                            resample_col='mean')


def test_pivot_table_contains_all_columns_for_day_of_week_and_months_in_order():
    series = ContinuousSeries(df, min_days_of_data, max_interval, time_col, value_col, sample_rule)

    pivot = series.pivot_df_for_day_of_week_month('mean', np.mean)

    assert_that(pivot.shape, is_((7, 12)))
    assert_that(list(pivot.columns), contains_exactly(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), pivot.columns)
    assert_that(list(pivot.index), contains_exactly(0, 1, 2, 3, 4, 5, 6), pivot.index)


def test_pivot_table_contains_all_columns_for_day_of_week_and_hours_in_order():
    series = ContinuousSeries(df, min_days_of_data, max_interval, time_col, value_col, "1H")

    pivot = series.pivot_df_for_day_of_week_and_hours('mean', np.mean)

    assert_that(pivot.shape, is_((7, 24)))
    assert_that(list(pivot.columns),
                contains_exactly(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23),
                pivot.columns)
    assert_that(list(pivot.index), contains_exactly(0, 1, 2, 3, 4, 5, 6), pivot.index)


@pytest.mark.skipif(not os.path.isdir(Configuration().data_dir), reason="reads real data")
def test_continuous_series_real_data():
    zip_id = '14092221'
    max_interval = 180  # how frequent readings need per day, 60=every hour, 180=every three hours
    min_days_of_data = 1  # how many days of consecutive readings with at least a reading every max interval, 7 = a week
    sample_rule = '1D'  # the frequency of the regular time series after resampling
    time_col = 'openaps/enacted/timestamp'
    value_col = 'openaps/enacted/IOB'
    file = device_status_file_path_for('../data/perid', zip_id)
    full_df = read_flat_device_status_df_from_file(file, Configuration())
    series = ContinuousSeries(full_df, min_days_of_data, max_interval, time_col, value_col, sample_rule)

    assert_that(series.resampled_series)

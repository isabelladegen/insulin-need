# create fake timeseries
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from hamcrest import *

from src.continuous_series import ContinuousSeries
from src.preprocess import number_of_interval_in_days
from tests.helper.BgDfBuilder import create_time_stamps

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
time_col = 't'
value_col = 'v'
times = times1 + times2
values = values1 + values2
df = pd.DataFrame(data={time_col: times, value_col: values})


def test_plots_resampled_sub_series():
    series = ContinuousSeries(df, min_days_of_data, max_interval, time_col, value_col, sample_rule)
    # no asserts as it generates a plot
    series.plot_resampled_series()


def test_returns_index_and_value_column_for_resampled_value():
    series = ContinuousSeries(df, min_days_of_data, max_interval, time_col, value_col, sample_rule)

    index = -1
    col = 'mean'
    x, y = series.get_resampled_x_and_y_for(index, col)

    resampled_df = series.resampled_series[index]
    assert_that(len(x), is_(resampled_df.shape[0]))
    assert_that(len(y), is_(resampled_df.shape[0]))
    assert_that(x, is_(list(resampled_df.index)))
    assert_that(y.equals(resampled_df[value_col][col].astype(np.float64)))

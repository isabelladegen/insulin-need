import os

import pytest
from hamcrest import *

from src.configurations import Configuration
from src.continuous_series import Cols
from src.multiple_continuous_series import MultipleContinuousSeries

zip_ids = ['57176789', '13484299', '86025410']
max_interval = 180  # how frequent readings need per day, 60=every hour, 180=every three hours,
min_days_of_data = 1  # how many days of consecutive readings with at least a reading every max interval, 7 = a week
sample_rule = '1D'  # the frequency of the regular time series after resampling
time_col = 'openaps/enacted/timestamp'
value_columns = ['openaps/enacted/IOB', 'openaps/enacted/COB', 'openaps/enacted/bg']


@pytest.mark.skipif(not os.path.isdir(Configuration().perid_data_folder), reason="reads real data")
def test_creates_a_continuous_series_for_all_ids_and_value_columns():
    mcs = MultipleContinuousSeries(zip_ids, min_days_of_data, max_interval, time_col, value_columns, sample_rule)
    result = mcs.continuous_series

    # entry for each zip id
    assert_that(len(mcs.continuous_series), is_(len(zip_ids)))
    for zip_id in result:
        # continuous series for each value columns for each zip id
        assert_that(len(result[zip_id]), is_(len(value_columns)))
        first_resampled_df = result[zip_id][value_columns[0]].resampled_series[0]
        resampled_columns = list(first_resampled_df[value_columns[0]].columns)
        assert_that(resampled_columns, contains_exactly('min', 'max', 'mean', 'std', 'z-score'), resampled_columns)


@pytest.mark.skipif(not os.path.isdir(Configuration().perid_data_folder), reason="reads real data")
def test_plots_heathmap_for_each_zip_id_and_value_column():
    ids = ['57176789', '13484299']
    mcs = MultipleContinuousSeries(ids, min_days_of_data, max_interval, time_col, value_columns, sample_rule)
    # no asserts as just for plotting
    mcs.plot_heatmaps()


@pytest.mark.skipif(not os.path.isdir(Configuration().perid_data_folder), reason="reads real data")
def test_calculates_x_train_for_zipids():
    ids = ['14092221', '13484299']
    mcs = MultipleContinuousSeries(ids, 1, 60, time_col, value_columns, "1H")
    x_trains = mcs.as_dictionary_of_x_train(Cols.Mean)

    assert_that(x_trains[ids[0]].shape, is_((304, 24, 3)))
    assert_that(x_trains[ids[1]].shape, is_((15, 24, 3)))


@pytest.mark.skipif(not os.path.isdir(Configuration().perid_data_folder), reason="reads real data")
def test_plots_totals_heathmap_for_each_zip_id_and_value_column():
    ids = ['57176789', '13484299']
    mcs = MultipleContinuousSeries(ids, min_days_of_data, max_interval, time_col, value_columns, sample_rule)
    # no asserts as just for plotting
    mcs.plot_total_months_heatmaps()

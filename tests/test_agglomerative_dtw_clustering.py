import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from hamcrest import *
import pytest

from src.agglomerative_dtw_clustering import AgglomerativeTSClustering
from src.configurations import Configuration
from src.continuous_series import ContinuousSeries, Cols

# build fake data
from src.multivariate_resampled_series import MultivariateResampledSeries
from src.preprocess import number_of_interval_in_days
from src.stats import DailyTimeseries
from tests.helper.BgDfBuilder import create_time_stamps

max_interval = 60  # how frequent readings need per day, 60=every hour, 180=every two hours
min_days_of_data = 30  # how many days of consecutive readings with at least a reading every interval
sample_rule = '1H'
xtick = list(range(0, 24, 2))
min_series_length = number_of_interval_in_days(min_days_of_data, max_interval)
start_date1 = datetime(year=2019, month=12, day=25, hour=12, minute=0, tzinfo=timezone.utc)
start_date2 = datetime(year=2021, month=1, day=3, hour=1, minute=0, tzinfo=timezone.utc)
times1 = create_time_stamps(start_date1, min_series_length, max_interval)
times2 = create_time_stamps(start_date2, 2 * min_series_length, max_interval)
values1 = list(np.random.uniform(low=0.1, high=14.6, size=min_series_length))
values2 = list(np.random.uniform(low=0.5, high=10.6, size=2 * min_series_length))
values3 = list(np.random.uniform(low=90, high=400, size=min_series_length + 2 * min_series_length))
col_to_cluster = Cols.Mean
time_col = 't'
value_col = 'v'
value_col2 = 'another col'
times = times1 + times2
values = values1 + values2
df = pd.DataFrame(data={time_col: times, value_col: values, value_col2: values3})
dailySampling = DailyTimeseries()


def test_agglomerative_clustering():
    series = ContinuousSeries(df, min_days_of_data, max_interval, time_col, value_col, sample_rule)

    x_train = series.as_x_train(col_to_cluster)
    ac = AgglomerativeTSClustering(x_train=x_train, x_train_column_names=["v"], sampling=dailySampling)

    no_ts = x_train.shape[0]
    assert_that(ac.distance_matrix.shape, is_((no_ts, no_ts)))
    assert_that(len(ac.y_pred), is_(no_ts))
    n_cluster = ac.no_clusters
    assert_that(n_cluster, greater_than(2))
    assert_that(n_cluster, less_than(no_ts))


@pytest.mark.skipif(not os.path.isdir(Configuration().perid_data_folder), reason="reads real data")
def test_plots_clustered_ts_and_others_in_grid():
    mv = MultivariateResampledSeries('13484299', col_to_cluster, dailySampling)

    x_train = mv.get_1d_numpy_array(dailySampling.cob_col)
    no_ts = x_train.shape[0]

    ac = AgglomerativeTSClustering(x_train=x_train, x_train_column_names=["COB"], sampling=dailySampling)

    assert_that(ac.distance_matrix.shape, is_((no_ts, no_ts)))
    assert_that(len(ac.y_pred), is_(no_ts))
    n_cluster = ac.no_clusters
    assert_that(n_cluster, greater_than(2))
    assert_that(n_cluster, less_than(no_ts))

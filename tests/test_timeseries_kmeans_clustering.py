import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest
from hamcrest import *

from src.configurations import Configuration
from src.continuous_series import ContinuousSeries, Resolution, Cols
from src.helper import device_status_file_path_for
from src.multivariate_resampled_series import MultivariateResampledSeries
from src.preprocess import number_of_interval_in_days
from src.read import read_flat_device_status_df_from_file
from src.stats import DailyTimeseries
from src.timeseries_kmeans_clustering import TimeSeriesKMeansClustering
from tests.helper.BgDfBuilder import create_time_stamps

# build fake data
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


def test_plots_clusters_in_grid():
    series = ContinuousSeries(df, min_days_of_data, max_interval, time_col, value_col, sample_rule)

    x_train = series.as_x_train(col_to_cluster)
    km = TimeSeriesKMeansClustering(n_clusters=4, x_train=x_train, x_train_column_names=["IOB"], x_ticks=xtick)

    # no asserts as it generates a plot
    km.plot_clusters_in_grid(y_label_substr=col_to_cluster)


def test_plots_all_barrycenters_in_one_plot_for_one_column():
    series = ContinuousSeries(df, min_days_of_data, max_interval, time_col, value_col, sample_rule)

    x_train = series.as_x_train(col_to_cluster)
    km = TimeSeriesKMeansClustering(n_clusters=4, x_train=x_train, x_train_column_names=["IOB"], x_ticks=xtick)

    # no asserts as it generates a plot
    km.plot_barry_centers_in_one_plot(y_label_substr=col_to_cluster)


@pytest.mark.skipif(not os.path.isdir(Configuration().perid_data_folder), reason="reads real data")
def test_plots_all_barrycenters_in_one_plot_for_multiple_columns():
    sampling = DailyTimeseries()
    mv = MultivariateResampledSeries('13484299', col_to_cluster, sampling)

    x_train = mv.get_1d_numpy_array(sampling.cob_col)
    x_full = mv.get_multivariate_3d_numpy_array()
    km = TimeSeriesKMeansClustering(n_clusters=3, x_train=x_train, x_train_column_names=["COB"], x_ticks=xtick,
                                    x_full=x_full, x_full_column_names=["IOB", "COB", "BG"])

    # no asserts as it generates a plot
    km.plot_barry_centers_in_one_plot(y_label_substr=col_to_cluster)


@pytest.mark.skipif(not os.path.isdir(Configuration().perid_data_folder), reason="reads real data")
def test_plots_all_columns_barry_centers_in_one_plot_for_multiple_clusters():
    sampling = DailyTimeseries()
    mv = MultivariateResampledSeries('13484299', col_to_cluster, sampling)

    x_train = mv.get_1d_numpy_array(sampling.cob_col)
    x_full = mv.get_multivariate_3d_numpy_array()
    km = TimeSeriesKMeansClustering(n_clusters=3, x_train=x_train, x_train_column_names=["COB"], x_ticks=xtick,
                                    x_full=x_full, x_full_column_names=["IOB", "COB", "BG"])

    # no asserts as it generates a plot
    km.plot_barrycenters_of_different_cols_in_one_plot(y_label_substr=col_to_cluster)


@pytest.mark.skipif(not os.path.isdir(Configuration().perid_data_folder), reason="reads real data")
def test_plots_clustered_ts_and_others_in_grid():
    sampling = DailyTimeseries()
    mv = MultivariateResampledSeries('13484299', col_to_cluster, sampling)

    x_train = mv.get_1d_numpy_array(sampling.cob_col)
    x_full = mv.get_multivariate_3d_numpy_array()
    km = TimeSeriesKMeansClustering(n_clusters=3, x_train=x_train, x_train_column_names=["COB"], x_ticks=xtick,
                                    x_full=x_full, x_full_column_names=["IOB", "COB", "BG"])

    # no asserts as it generates a plot
    km.plot_clusters_in_grid(y_label_substr=col_to_cluster)


def test_plots_silhouette_blob_for_k():
    series = ContinuousSeries(df, min_days_of_data, max_interval, time_col, value_col, sample_rule)

    x_train = series.as_x_train(col_to_cluster)
    km = TimeSeriesKMeansClustering(n_clusters=4, x_train=x_train, x_train_column_names=["IOB"], x_ticks=xtick)

    km.plot_silhouette_blob_for_k(ks=[2, 3, 4, 5, 6, 7, 8, 20])


def test_plots_silhouette_blob_for_small_ks():
    series = ContinuousSeries(df, min_days_of_data, max_interval, time_col, value_col, sample_rule)

    x_train = series.as_x_train(col_to_cluster)
    km = TimeSeriesKMeansClustering(n_clusters=4, x_train=x_train, x_train_column_names=["IOB"], x_ticks=xtick)

    km.plot_silhouette_blob_for_k(ks=[2, 3, 4, 5, 6, 7, 8, 9])


def test_plots_silhouette_blob_for_single_row_ks():
    series = ContinuousSeries(df, min_days_of_data, max_interval, time_col, value_col, sample_rule)

    x_train = series.as_x_train(col_to_cluster)
    km = TimeSeriesKMeansClustering(n_clusters=4, x_train=x_train, x_train_column_names=["IOB"], x_ticks=xtick)

    km.plot_silhouette_blob_for_k(ks=[2, 7, 8, 28])


def test_plots_silhouette_score_for_k():
    series = ContinuousSeries(df, min_days_of_data, max_interval, time_col, value_col, sample_rule)

    x_train = series.as_x_train(col_to_cluster)
    km = TimeSeriesKMeansClustering(n_clusters=4, x_train=x_train, x_train_column_names=["IOB"], x_ticks=xtick)

    km.plot_mean_silhouette_score_for_k(range(2, 6))


def test_plots_elbow_method_for_k():
    series = ContinuousSeries(df, min_days_of_data, max_interval, time_col, value_col, sample_rule)

    x_train = series.as_x_train(col_to_cluster)
    km = TimeSeriesKMeansClustering(n_clusters=4, x_train=x_train, x_train_column_names=["IOB"], x_ticks=xtick)

    km.plot_sum_of_square_distances_for_k(range(2, 6))

import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest
from hamcrest import *

from src.configurations import Configuration, GeneralisedCols, Hourly
from src.preprocess import number_of_interval_in_days
from src.read_preprocessed_df import ReadPreprocessedDataFrame
from src.reshape_resampled_data_into_timeseries import ReshapeResampledDataIntoTimeseries
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
time_col = GeneralisedCols.datetime.value
value_col = GeneralisedCols.mean_iob.value
value_col2 = 'another col'
times = times1 + times2
values = values1 + values2
df = pd.DataFrame(data={time_col: times, value_col: values, value_col2: values3})
daily_ts = DailyTimeseries()
y_label = "mean"


def test_plots_clusters_in_grid():
    translate = ReshapeResampledDataIntoTimeseries(df, daily_ts, [GeneralisedCols.mean_iob])

    km = TimeSeriesKMeansClustering(n_clusters=4, x_train=translate.to_x_train(), x_train_column_names=["IOB"],
                                    timeseries_description=daily_ts)

    # no asserts as it generates a plot
    km.plot_clusters_in_grid(y_label_substr=y_label)


def test_uses_additional_parameters_for_distance_calculation():
    translate = ReshapeResampledDataIntoTimeseries(df, daily_ts, [GeneralisedCols.mean_iob])
    x_train = translate.to_x_train()

    distance_params = {"global_constraint": "sakoe_chiba",
                       "sakoe_chiba_radius": 1}
    ks = [4]
    km_sakoe = TimeSeriesKMeansClustering(n_clusters=4, x_train=x_train, x_train_column_names=["IOB"],
                                          timeseries_description=DailyTimeseries(), distance_metric="dtw",
                                          metric_prams=distance_params)
    km_standard = TimeSeriesKMeansClustering(n_clusters=4, x_train=x_train, x_train_column_names=["IOB"],
                                             timeseries_description=DailyTimeseries(), distance_metric="dtw",
                                             metric_prams=None)

    ss_sakoe = km_sakoe.calculate_mean_silhouette_score_for_ks(ks)[0]
    ss_standard = km_standard.calculate_mean_silhouette_score_for_ks(ks)[0]
    print("SS Sakoe: " + str(ss_sakoe))
    print("SS Standard: " + str(ss_standard))

    assert_that(ss_sakoe, less_than(ss_standard))


def test_returns_y_of_classes():
    translate = ReshapeResampledDataIntoTimeseries(df, daily_ts, [GeneralisedCols.mean_iob])
    no_clusters = 5
    x_train = translate.to_x_train()
    km = TimeSeriesKMeansClustering(n_clusters=no_clusters, x_train=x_train, x_train_column_names=["IOB"],
                                    timeseries_description=daily_ts)

    y = km.y_pred

    assert_that(len(y), is_(x_train.shape[0]))
    assert_that(len(set(y)), is_(no_clusters))


def test_plots_all_barrycenters_in_one_plot_for_one_column():
    translate = ReshapeResampledDataIntoTimeseries(df, daily_ts, [GeneralisedCols.mean_iob])
    km = TimeSeriesKMeansClustering(n_clusters=4, x_train=translate.to_x_train(), x_train_column_names=["IOB"],
                                    timeseries_description=daily_ts)

    # no asserts as it generates a plot
    km.plot_barry_centers_in_one_plot(y_label_substr=y_label)


@pytest.mark.skipif(not os.path.isdir(Configuration().perid_data_folder), reason="reads real data")
def test_plots_all_barrycenters_in_one_plot_for_multiple_columns():
    real_df = ReadPreprocessedDataFrame(sampling=Hourly(), zip_id='13484299').df
    translate = ReshapeResampledDataIntoTimeseries(real_df, daily_ts, Configuration.resampled_mean_columns())

    x_train = translate.to_x_train(cols=[GeneralisedCols.mean_cob.value])
    x_full = translate.to_x_train()
    km = TimeSeriesKMeansClustering(n_clusters=3, x_train=x_train, x_train_column_names=["COB"],
                                    timeseries_description=daily_ts,
                                    x_full=x_full, x_full_column_names=["IOB", "COB", "BG"])

    # no asserts as it generates a plot
    km.plot_barry_centers_in_one_plot(y_label_substr='mean')


@pytest.mark.skipif(not os.path.isdir(Configuration().perid_data_folder), reason="reads real data")
def test_plots_all_columns_barry_centers_in_one_plot_for_multiple_clusters():
    real_df = ReadPreprocessedDataFrame(sampling=Hourly(), zip_id='13484299').df
    translate = ReshapeResampledDataIntoTimeseries(real_df, daily_ts, Configuration.resampled_mean_columns())

    x_train = translate.to_x_train(cols=[GeneralisedCols.mean_cob.value])
    x_full = translate.to_x_train()
    km = TimeSeriesKMeansClustering(n_clusters=3, x_train=x_train, x_train_column_names=["COB"],
                                    timeseries_description=daily_ts,
                                    x_full=x_full, x_full_column_names=["IOB", "COB", "BG"])

    # no asserts as it generates a plot
    km.plot_barrycenters_of_different_cols_in_one_plot(y_label_substr=y_label)


@pytest.mark.skipif(not os.path.isdir(Configuration().perid_data_folder), reason="reads real data")
def test_plots_clustered_ts_and_others_in_grid():
    real_df = ReadPreprocessedDataFrame(sampling=Hourly(), zip_id='13484299').df
    translate = ReshapeResampledDataIntoTimeseries(real_df, daily_ts, Configuration.resampled_mean_columns())

    x_train = translate.to_x_train(cols=[GeneralisedCols.mean_cob.value])
    x_full = translate.to_x_train()
    km = TimeSeriesKMeansClustering(n_clusters=3, x_train=x_train, x_train_column_names=["COB"],
                                    timeseries_description=daily_ts,
                                    x_full=x_full, x_full_column_names=["IOB", "COB", "BG"])

    # no asserts as it generates a plot
    km.plot_clusters_in_grid(y_label_substr=y_label)


def test_plots_silhouette_blob_for_k():
    translate = ReshapeResampledDataIntoTimeseries(df, daily_ts, [GeneralisedCols.mean_iob])

    km = TimeSeriesKMeansClustering(n_clusters=4, x_train=translate.to_x_train(), x_train_column_names=["IOB"],
                                    timeseries_description=daily_ts)

    km.plot_silhouette_blob_for_k(ks=[2, 3, 4, 5, 6, 7, 8, 20])


def test_plots_silhouette_blob_for_small_ks():
    translate = ReshapeResampledDataIntoTimeseries(df, daily_ts, [GeneralisedCols.mean_iob])
    km = TimeSeriesKMeansClustering(n_clusters=4, x_train=translate.to_x_train(), x_train_column_names=["IOB"],
                                    timeseries_description=daily_ts)

    km.plot_silhouette_blob_for_k(ks=[2, 3, 4, 5, 6, 7, 8, 9])


def test_plots_silhouette_blob_for_single_row_ks():
    translate = ReshapeResampledDataIntoTimeseries(df, daily_ts, [GeneralisedCols.mean_iob])
    km = TimeSeriesKMeansClustering(n_clusters=4, x_train=translate.to_x_train(), x_train_column_names=["IOB"],
                                    timeseries_description=daily_ts)

    km.plot_silhouette_blob_for_k(ks=[2, 7, 8, 28])


def test_plots_silhouette_score_for_k():
    translate = ReshapeResampledDataIntoTimeseries(df, daily_ts, [GeneralisedCols.mean_iob])
    km = TimeSeriesKMeansClustering(n_clusters=4, x_train=translate.to_x_train(), x_train_column_names=["IOB"],
                                    timeseries_description=daily_ts)

    km.plot_mean_silhouette_score_for_k(range(2, 6))


def test_plots_elbow_method_for_k():
    translate = ReshapeResampledDataIntoTimeseries(df, daily_ts, [GeneralisedCols.mean_iob])
    km = TimeSeriesKMeansClustering(n_clusters=4, x_train=translate.to_x_train(), x_train_column_names=["IOB"],
                                    timeseries_description=daily_ts)

    km.plot_sum_of_square_distances_for_k(range(2, 6))

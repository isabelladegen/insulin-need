import os

from hamcrest import *
import pytest

from src.agglomerative_dtw_clustering import AgglomerativeTSClustering
from src.configurations import Configuration, Hourly, GeneralisedCols
from src.continuous_series import Cols

from src.read_preprocessed_df import ReadPreprocessedDataFrame
from src.reshape_resampled_data_into_timeseries import ReshapeResampledDataIntoTimeseries
from src.stats import DailyTimeseries

y_sub_label = Cols.Mean
daily_ts = DailyTimeseries()


@pytest.mark.skipif(not os.path.isdir(Configuration().perid_data_folder), reason="reads real data")
def test_plots_silhouette_analysis_for_clustering():
    real_df = ReadPreprocessedDataFrame(sampling=Hourly(), zip_id='13484299').df
    translate = ReshapeResampledDataIntoTimeseries(real_df, daily_ts, Configuration.resampled_mean_columns())

    x_train = translate.to_x_train(cols=[GeneralisedCols.mean_cob.value])
    ac = AgglomerativeTSClustering(x_train=x_train, x_train_column_names=["COB"], timeseries_description=daily_ts)

    ac.plot_silhouette_analysis()


@pytest.mark.skipif(not os.path.isdir(Configuration().perid_data_folder), reason="reads real data")
def test_plots_clustered_ts_and_others_in_grid():
    real_df = ReadPreprocessedDataFrame(sampling=Hourly(), zip_id='13484299').df
    translate = ReshapeResampledDataIntoTimeseries(real_df, daily_ts, Configuration.resampled_mean_columns())

    x_train = translate.to_x_train(cols=[GeneralisedCols.mean_cob.value])
    x_full = translate.to_x_train()

    no_ts = x_train.shape[0]
    ac = AgglomerativeTSClustering(x_train=x_train, x_train_column_names=["COB"], timeseries_description=daily_ts,
                                   x_full=x_full, x_full_column_names=["IOB", "COB", "BG"])

    assert_that(ac.distance_matrix.shape, is_((no_ts, no_ts)))
    assert_that(len(ac.y_pred), is_(no_ts))
    n_cluster = ac.no_clusters
    assert_that(n_cluster, greater_than(2))
    assert_that(n_cluster, less_than(no_ts))
    ac.plot_clusters_in_grid(y_label_substr=y_sub_label)


@pytest.mark.skipif(not os.path.isdir(Configuration().perid_data_folder), reason="reads real data")
def test_clusters_using_sakoe_chiba():
    real_df = ReadPreprocessedDataFrame(sampling=Hourly(), zip_id='13484299').df
    translate = ReshapeResampledDataIntoTimeseries(real_df, daily_ts, Configuration.resampled_mean_columns())

    x_train = translate.to_x_train(cols=[GeneralisedCols.mean_cob.value])
    x_full = translate.to_x_train()
    no_ts = x_train.shape[0]

    ac = AgglomerativeTSClustering(x_train=x_train, x_train_column_names=["COB"], timeseries_description=daily_ts,
                                   x_full=x_full, x_full_column_names=["IOB", "COB", "BG"],
                                   distance_constraint="sakoe_chiba",
                                   sakoe_chiba_radius=3)

    assert_that(ac.distance_matrix.shape, is_((no_ts, no_ts)))
    assert_that(len(ac.y_pred), is_(no_ts))
    n_cluster = ac.no_clusters
    assert_that(n_cluster, greater_than(2))
    assert_that(n_cluster, less_than(no_ts))
    ac.plot_clusters_in_grid(y_label_substr=y_sub_label)


@pytest.mark.skipif(not os.path.isdir(Configuration().perid_data_folder), reason="reads real data")
def test_returns_y_pred_as_binary_with_most_frequent_normal_and_other_classes_anomaly():
    real_df = ReadPreprocessedDataFrame(sampling=Hourly(), zip_id='13484299').df
    translate = ReshapeResampledDataIntoTimeseries(real_df, daily_ts, Configuration.resampled_mean_columns())

    x_train = translate.to_x_train(cols=[GeneralisedCols.mean_cob.value])
    ac = AgglomerativeTSClustering(x_train=x_train, x_train_column_names=["COB"], timeseries_description=daily_ts)

    result = ac.get_y_pred_as_binary()
    assert_that(len(set(result)), is_(2))  # normal and anomaly as class
    assert_that(len(result), is_(x_train.shape[0]))
    assert_that(result.count("normal"), less_than(result.count("anomaly")))


@pytest.mark.skipif(not os.path.isdir(Configuration().perid_data_folder), reason="reads real data")
def test_plots_dendrogram():
    real_df = ReadPreprocessedDataFrame(sampling=Hourly(), zip_id='13484299').df
    translate = ReshapeResampledDataIntoTimeseries(real_df, daily_ts, Configuration.resampled_mean_columns())

    x_train = translate.to_x_train(cols=[GeneralisedCols.mean_cob.value])
    x_full = translate.to_x_train()
    ac = AgglomerativeTSClustering(x_train=x_train, x_train_column_names=["COB"], timeseries_description=daily_ts,
                                   x_full=x_full, x_full_column_names=["IOB", "COB", "BG"])

    no_ts = x_train.shape[0]
    assert_that(ac.distance_matrix.shape, is_((no_ts, no_ts)))
    assert_that(len(ac.y_pred), is_(no_ts))
    n_cluster = ac.no_clusters
    assert_that(n_cluster, greater_than(2))
    assert_that(n_cluster, less_than(no_ts))
    ac.plot_dendrogram()

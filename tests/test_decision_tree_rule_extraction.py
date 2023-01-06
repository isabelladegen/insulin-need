import os
import pytest
from hamcrest import *

from src.configurations import Configuration, Hourly, GeneralisedCols
from src.decision_tree_rule_extraction import DecisionTreeRuleExtraction
from src.read_preprocessed_df import ReadPreprocessedDataFrame
from src.translate_into_timeseries import TranslateIntoTimeseries, DailyTimeseries
from src.timeseries_kmeans_clustering import TimeSeriesKMeansClustering


@pytest.mark.skipif(not os.path.isdir(Configuration().perid_data_folder), reason="reads real data")
def test_calculates_decision_trees_and_plots_describing_graphs():
    real_df = ReadPreprocessedDataFrame(sampling=Hourly(), zip_id='13484299').df
    daily_ts = DailyTimeseries()
    translate = TranslateIntoTimeseries(real_df, daily_ts, Configuration.resampled_mean_columns())

    col_to_analyse = GeneralisedCols.mean_cob
    x_train = translate.to_x_train(cols=[col_to_analyse])
    x_full = translate.to_x_train()

    km = TimeSeriesKMeansClustering(n_clusters=3, x_train=x_train, x_train_column_names=["COB"],
                                    timeseries_description=daily_ts,
                                    x_full=x_full, x_full_column_names=["IOB", "COB", "BG"])
    y = km.y_pred
    x_dt_train = translate.to_vectorised_df(col_to_analyse)

    dt = DecisionTreeRuleExtraction(x_dt_train, y)
    assert_that(len(dt.model.feature_importances_), is_(len(x_dt_train.columns)))
    dt.tree_rules()
    dt.plot_tree()
    dt.plot_feature_importance()

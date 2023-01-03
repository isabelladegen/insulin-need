import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest
from hamcrest import *

from src.configurations import Configuration
from src.continuous_series import Cols
from src.decision_tree_rule_extraction import DecisionTreeRuleExtraction
from src.multivariate_resampled_series import MultivariateResampledSeries
from src.stats import DailyTimeseries
from src.timeseries_kmeans_clustering import TimeSeriesKMeansClustering

col_to_cluster = Cols.Mean


@pytest.mark.skipif(not os.path.isdir(Configuration().perid_data_folder), reason="reads real data")
def test_calculates_decision_trees_and_plots_describing_graphs():
    sampling = DailyTimeseries()
    mv = MultivariateResampledSeries('13484299', col_to_cluster, sampling)

    x_train = mv.get_1d_numpy_array(sampling.cob_col)
    x_full = mv.get_multivariate_3d_numpy_array()
    km = TimeSeriesKMeansClustering(n_clusters=3, x_train=x_train, x_train_column_names=["COB"], timeseries_description=sampling,
                                    x_full=x_full, x_full_column_names=["IOB", "COB", "BG"])
    y = km.y_pred
    x_dt_train = mv.get_vectorised_df(sampling.cob_col)

    dt = DecisionTreeRuleExtraction(x_dt_train, y)
    assert_that(len(dt.model.feature_importances_), is_(len(x_dt_train.columns)))
    dt.tree_rules()
    dt.plot_tree()
    dt.plot_feature_importance()

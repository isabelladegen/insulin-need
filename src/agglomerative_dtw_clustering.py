import numpy as np
from sklearn.cluster import AgglomerativeClustering
from tslearn.metrics import dtw
from tslearn.preprocessing import TimeSeriesScalerMinMax

from src.stats import Sampling


class AgglomerativeTSClustering:
    """Collection of convenience function including visualisations for agglomerative dtw clustering with sklearn

    Attributes
    ----------
    model : AgglomerativeClustering
        sklearn model

    y_pred : np.array
        cluster number for each ts in x_train
    """

    def __init__(self, x_train: np.array, x_train_column_names: [str], sampling: Sampling,
                 scaler=TimeSeriesScalerMinMax(), x_full: np.array = None, x_full_column_names: [str] = None):
        """Collection of convenience function for tslearn k-means.

        Parameters
        ----------
        x_train : np.array
            timeseries to cluster as np.array of shape=(n_ts, sz, d), where n_ts is number of days, sz is 24, and
            d=1 for single variate and > for multivariate ts

        x_train_column_names: []
            list of ts names in x_train

        sampling : Sampling
            used to calibrate the x axis of the graphs

        scaler : TimeSeriesScalerMinMax or TimeSeriesScalerMeanVariance or None if no scaling
            Default is MinMax scaling

        x_full : np.array
            used to plot the other time series that were not used in clustering. e.g if clustered on IOB this would
            calculate barrycenters for COB and BG and plot them according to where they would fit based on the IOB
            clustering

        x_full_column_names : []
            columns for x full to be able to find the right TS in the np.array
            :param sampling:
        """
        self.__scaler = scaler
        if self.__scaler is None:
            self.__x_train = x_train
        else:
            # this is really important especially for multivariate TS
            self.__x_train = self.__scaler.fit_transform(x_train)  # normalise data
        self.__x_train_column_names = x_train_column_names
        self.__x_ticks = sampling.x_ticks
        self.__x_label = "X = " + sampling.description

        if x_full is not None:
            assert (x_full_column_names is not None)
            # scale x full in the same way
            self.__x_full = x_full if self.__scaler is None else self.__scaler.fit_transform(x_full)
            self.__cols_to_plot = x_full_column_names
        else:
            self.__x_full = None
            self.__cols_to_plot = self.__x_train_column_names

        # create clusters
        self.model = AgglomerativeClustering(n_clusters=None, affinity="precomputed", connectivity=None,
                                             compute_full_tree=True, linkage='single', distance_threshold=0.5)

        self.distance_matrix = self.__calculate_distance_matrix()
        self.y_pred = self.model.fit_predict(self.distance_matrix)
        self.no_clusters = max(self.y_pred)

    def __calculate_distance_matrix(self):
        # return symmetric matrix of dtw distances
        n_ts = self.__x_train.shape[0]
        distance_matrix = np.zeros((n_ts, n_ts))
        for row in range(n_ts):
            for column in range(n_ts):
                s1 = self.__x_train[row, :, 0]
                s2 = self.__x_train[column, :, 0]
                distance_matrix[row][column] = dtw(s1, s2, global_constraint=None, sakoe_chiba_radius=None,
                                                   itakura_max_slope=None)
        return distance_matrix
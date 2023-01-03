import numpy as np
from tslearn.preprocessing import TimeSeriesScalerMinMax

from src.stats import TimeSeriesDescription
from src.timeseries_kmeans_clustering import TimeSeriesKMeansClustering


class TimeSeriesKMeansClusteringCrossValidation:
    """Collection of convenience function for cross validating tslearn k-means.

    Attributes
    ----------
    n_fold_km : list with n_fold TimeSeriesKMeans k-means models each run on one of n_fold-1 of the ts
    """

    def __init__(self, n_fold: int, n_clusters: int, x_train: np.array, x_train_column_names: [str],
                 timeseries_description: TimeSeriesDescription,
                 scaler=TimeSeriesScalerMinMax(), x_full: np.array = None, x_full_column_names: [str] = None,
                 distance_metric="dtw", metric_prams: {} = None):
        """Collection of convenience function for tslearn k-means.

        Parameters
        ----------
        n_fold : Int
            Number of folds to calculate k-means for. This should be a even dividend of n_ts

        n_clusters : Int
            Number of clusters

        x_train : np.array
            timeseries to cluster as np.array of shape=(n_ts, sz, d), where n_ts is number of days, sz is 24, and
            d=1 for single variate and > for multivariate ts

        x_train_column_names: []
            list of ts names in x_train

        x_ticks : [int]
            x_ticks to be used

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
            :param timeseries_description:
        """
        self.label_font_size = 20
        self.__n_fold = n_fold
        self.__n_clusters = n_clusters
        self.__x_train = x_train
        self.__x_train_column_names = x_train_column_names
        self.__metric = distance_metric
        self.__metric_params = metric_prams
        self.__max_iter = 10
        self.__random_state = 66
        self.__x_ticks = timeseries_description.x_ticks
        self.__x_label = "X = " + timeseries_description.description

        # create x_train for each fold
        self.__x_train_folds = np.split(self.__x_train, self.__n_fold)  # splits data into n_fold chunks
        self.__x_trains = []
        for i in range(0, self.__n_fold):
            self.__x_trains.append(np.concatenate(self.__x_train_folds[:i] + self.__x_train_folds[i + 1:]))

        # TODO figure out how this works
        if x_full is not None:
            assert (x_full_column_names is not None)
            # scale x full in the same way
            # self.__x_full = x_full if self.__scaler is None else self.__scaler.fit_transform(x_full)
            self.__cols_to_plot = x_full_column_names
        else:
            self.__x_full = None
            self.__cols_to_plot = self.__x_train_column_names

        # create clusters for each x_train
        self.n_fold_km = []
        for x in self.__x_trains:
            km = TimeSeriesKMeansClustering(n_clusters=2, x_train=x, x_train_column_names=['IOB', 'COB', 'BG'],
                                            timeseries_description=timeseries_description)
            self.n_fold_km.append(km)

    def silhouette_scores(self):
        """Calculates avg silhouette score for each of the n_fold models.

        Returns
        -------
        silhouette_scores : [float]
            Mean Silhouette Coefficient for all n_fold models.

        """
        result = []
        for model in self.n_fold_km:
            result.append(model.avg_silhouette_score())
        return result

    def plot_barycenters_for_each_model(self):
        """Calculates avg silhouette score for each of the n_fold models.

        """
        for model in self.n_fold_km:
            model.plot_barrycenters_of_different_cols_in_one_plot("", show_title=False, show_legend=False,
                                                                  show_overall_labels=False, sort_clusters=True)

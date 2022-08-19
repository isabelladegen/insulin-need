import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from tslearn.barycenters import dtw_barycenter_averaging
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

    def plot_clusters_in_grid(self, y_label_substr: str):
        """Plots clusters in a grid of cols being dimensions and rows being clusters.

        Parameters
        ----------
        y_label_substr: str
            Part of the plot y label
        """
        no_clusters = self.no_clusters
        no_dimensions = len(self.__cols_to_plot)

        # setup figure
        plt.rcParams.update({'figure.facecolor': 'white', 'axes.facecolor': 'white', 'figure.dpi': 150})
        fig_size = (4 * no_dimensions, no_clusters * 2)  # allow for multicolumn grids and single column grids
        fig, axs = plt.subplots(nrows=no_clusters,
                                ncols=no_dimensions,
                                sharey=True,
                                sharex=True,
                                figsize=fig_size, squeeze=0)
        fig.suptitle(
            "Agglomerative Clustering. Distance threshold "
            + str(self.model.distance_threshold)
            + ". Linkage "
            + str(self.model.linkage)
            + ". Clustered by "
            + ', '.join(self.__x_train_column_names)
            + ". No of TS "
            + str(len(self.y_pred)))

        # clusters are on the rows
        for row_idx in range(no_clusters):
            is_in_cluster_yi = (self.y_pred == row_idx)

            # plot all the time series for cluster row_idx and all dimensions
            if self.__x_full is None:  # just plot x train
                series_in_cluster_yi = self.__x_train[is_in_cluster_yi]
            else:  # plot x_full
                series_in_cluster_yi = self.__x_full[is_in_cluster_yi]
            for xx in series_in_cluster_yi:
                for col_idx, col in enumerate(self.__cols_to_plot):
                    ts = xx[:, col_idx]
                    # plot the ts for each variate in columns
                    axs[row_idx, col_idx].plot(ts.ravel(), 'k-', alpha=.2)

            # plot the barrycenter line and title
            for col_idx, col in enumerate(self.__cols_to_plot):
                # calculate and plot barrycenters for the none clustered cols
                bc = dtw_barycenter_averaging(series_in_cluster_yi[:, :, col_idx])
                # plot barrycenters red if clustered by  that variate
                if col in self.__x_train_column_names:
                    color = "r-"
                else:
                    color = "b-"
                axs[row_idx, col_idx].plot(bc.ravel(), color)

                axs[row_idx, col_idx].set_xticks(self.__x_ticks)
                axs[row_idx, col_idx].grid(which='major', alpha=0.2, color='grey')

                # set title for first row
                if row_idx == 0:
                    axs[0, col_idx].set_title(col)

            # set y label for row with cluster information
            axs[row_idx, 0].set_ylabel('Cluster ' + str(row_idx + 1) + '\n No TS = ' + str(is_in_cluster_yi.sum()))

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        plt.subplots_adjust(top=.9)
        # add overall x, y text
        fig.add_subplot(111, frame_on=False)
        plt.tick_params(labelcolor="none", bottom=False, left=False)
        plt.ylabel("Y =" + y_label_substr + " values", labelpad=30)
        plt.xlabel(self.__x_label)
        plt.show()

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

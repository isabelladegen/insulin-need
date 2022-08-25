import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_samples
from tslearn.barycenters import dtw_barycenter_averaging
from tslearn.clustering import TimeSeriesKMeans, silhouette_score
from tslearn.metrics import cdist_soft_dtw_normalized, cdist_dtw
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, TimeSeriesScalerMinMax

from tslearn.utils import to_time_series_dataset, to_time_series

# Time series implementation of ts_silhouette_samples
from src.stats import DailyTimeseries, Sampling


def ts_silhouette_samples(X, labels, metric=None, metric_params=None, n_jobs=None, verbose=0, **kwds):
    sklearn_metric = None
    if metric_params is None:
        metric_params_ = {}
    else:
        metric_params_ = metric_params.copy()
    for k in kwds.keys():
        metric_params_[k] = kwds[k]
    if "n_jobs" in metric_params_.keys():
        del metric_params_["n_jobs"]
    if metric == "precomputed":
        sklearn_X = X
    elif metric == "dtw" or metric is None:
        sklearn_X = cdist_dtw(X, n_jobs=n_jobs, verbose=verbose,
                              **metric_params_)
    elif metric == "softdtw":
        sklearn_X = cdist_soft_dtw_normalized(X, **metric_params_)
    elif metric == "euclidean":
        X_ = to_time_series_dataset(X)
        X_ = X_.reshape((X.shape[0], -1))
        sklearn_X = cdist(X_, X_, metric="euclidean")
    else:
        X_ = to_time_series_dataset(X)
        n, sz, d = X_.shape
        sklearn_X = X_.reshape((n, -1))

        def sklearn_metric(x, y):
            return metric(to_time_series(x.reshape((sz, d)),
                                         remove_nans=True),
                          to_time_series(y.reshape((sz, d)),
                                         remove_nans=True))
    metric = "precomputed" if sklearn_metric is None else sklearn_metric

    return silhouette_samples(sklearn_X, labels, metric=metric, **kwds)


class TimeSeriesKMeansClustering:
    """Collection of convenience function for tslearn k-means.

    Attributes
    ----------
    model : TimeSeriesKMeans
        k means model

    y_pred : np.array
        cluster number for each ts in x_train
    """

    def __init__(self, n_clusters: int, x_train: np.array, x_train_column_names: [str], sampling: Sampling,
                 scaler=TimeSeriesScalerMinMax(), x_full: np.array = None, x_full_column_names: [str] = None,
                 distance_metric="dtw", metric_prams=None):
        """Collection of convenience function for tslearn k-means.

        Parameters
        ----------
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
            :param sampling:
        """
        self.__n_clusters = n_clusters
        self.__scaler = scaler
        if self.__scaler is None:
            self.__x_train = x_train
        else:
            # this is really important especially for multivariate TS
            self.__x_train = self.__scaler.fit_transform(x_train)  # normalise data
        self.__x_train_column_names = x_train_column_names
        self.__metric = distance_metric
        self.__metric_params = metric_prams
        self.__max_iter = 10
        self.__random_state = 66
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
        self.model = TimeSeriesKMeans(n_clusters=self.__n_clusters, metric=self.__metric,
                                      metric_params=self.__metric_params, max_iter=self.__max_iter,
                                      random_state=self.__random_state)
        self.y_pred = self.model.fit_predict(self.__x_train)

    def plot_clusters_in_grid(self, y_label_substr: str):
        """Plots clusters in a grid of cols being dimensions and rows being clusters.

        Parameters
        ----------
        y_label_substr: str
            Part of the plot y label
        """
        no_clusters = self.model.n_clusters
        no_dimensions = len(self.__cols_to_plot)

        # setup figure
        plt.rcParams.update({'figure.facecolor': 'white', 'axes.facecolor': 'white', 'figure.dpi': 150})
        fig_size = (4 * no_dimensions, no_clusters * 2)  # allow for multicolumn grids and single column grids
        fig, axs = plt.subplots(nrows=no_clusters,
                                ncols=no_dimensions,
                                sharey=True,
                                sharex=True,
                                figsize=fig_size, squeeze=0)
        fig.suptitle("DBA k-means. Clustered by " + ', '.join(self.__x_train_column_names) + ". No of TS "
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
                # a column for which the barrycenters has already been calculated for clustering
                if col in self.__x_train_column_names:
                    axs[row_idx, col_idx].plot(
                        self.model.cluster_centers_[row_idx][:, self.__x_train_column_names.index(col)], "r-")
                else:  # calculate barrycenters for the none clustered cols
                    bc = dtw_barycenter_averaging(series_in_cluster_yi[:, :, col_idx])
                    axs[row_idx, col_idx].plot(bc.ravel(), "b-")

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

    def plot_barry_centers_in_one_plot(self, y_label_substr):
        """Plots row for all barry centers in one plot, if x_full given does it for the other cols too. Useful to see
        how similar the barrycenters for the different clusters are

        Parameters
        ----------
        y_label_substr: str
            Part of the plot y label
        """
        no_clusters = self.model.n_clusters
        no_dimensions = len(self.__cols_to_plot)

        # setup figure
        plt.rcParams.update({'figure.facecolor': 'white', 'axes.facecolor': 'white', 'figure.dpi': 150})
        fig_size = (10, 5)
        fig, axs = plt.subplots(nrows=1,
                                ncols=no_dimensions,
                                sharey=True,
                                sharex=True,
                                figsize=fig_size, squeeze=0)
        fig.suptitle("DBA k-means barrycenters. Clustered by " + ', '.join(self.__x_train_column_names) + ". No of TS "
                     + str(len(self.y_pred)) + ". " + y_label_substr + " values")

        # clusters are on the rows
        for cluster_idx in range(no_clusters):
            is_in_cluster_yi = (self.y_pred == cluster_idx)

            # plot the barrycenters
            for col_idx, col in enumerate(self.__cols_to_plot):
                label = "Cluster " + str(cluster_idx + 1)
                # a column for which the barrycenters has already been calculated for clustering
                if col in self.__x_train_column_names:
                    axs[0, col_idx].plot(
                        self.model.cluster_centers_[cluster_idx][:, self.__x_train_column_names.index(col)],
                        label=label)
                else:  # calculate barrycenters for the none clustered cols
                    series_in_cluster_yi = self.__x_full[is_in_cluster_yi]
                    bc = dtw_barycenter_averaging(series_in_cluster_yi[:, :, col_idx])
                    axs[0, col_idx].plot(bc.ravel(), label=label)

                axs[0, col_idx].set_xticks(self.__x_ticks)
                axs[0, col_idx].grid(which='major', alpha=0.2, color='grey')
                axs[0, col_idx].set_title(col)

        handles, labels = axs[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right')
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        plt.subplots_adjust(top=.9)
        # add overall x, y text
        fig.add_subplot(111, frame_on=False)
        plt.tick_params(labelcolor="none", bottom=False, left=False)
        plt.xlabel(self.__x_label)
        plt.show()

    def plot_barrycenters_of_different_cols_in_one_plot(self, y_label_substr):
        """Plots barrycenters with clusters as rows and columns collapsed into one row. Useful to see how IOB, COB and BG
        behave in each cluster

        Parameters
        ----------
        y_label_substr: str
            Part of the plot y label
        """
        no_clusters = self.model.n_clusters
        no_dimensions = len(self.__cols_to_plot)

        # setup figure
        plt.rcParams.update({'figure.facecolor': 'white', 'axes.facecolor': 'white', 'figure.dpi': 150})
        fig_size = (10, no_clusters * 2)
        fig, axs = plt.subplots(nrows=no_clusters,
                                ncols=1,
                                sharey=True,
                                sharex=True,
                                figsize=fig_size, squeeze=0)
        fig.suptitle("DBA k-means barrycenters. Clustered by " + ', '.join(self.__x_train_column_names) + ". No of TS "
                     + str(len(self.y_pred)))

        # clusters are on the rows
        for row_idx in range(no_clusters):
            is_in_cluster_yi = (self.y_pred == row_idx)

            # plot the barrycenter line and title
            for col_idx, col in enumerate(self.__cols_to_plot):
                # a column for which the barrycenters has already been calculated for clustering
                if col in self.__x_train_column_names:
                    axs[row_idx, 0].plot(
                        self.model.cluster_centers_[row_idx][:, self.__x_train_column_names.index(col)], "-", label=col)
                else:  # calculate barrycenters for the none clustered cols
                    series_in_cluster_yi = self.__x_full[is_in_cluster_yi]
                    bc = dtw_barycenter_averaging(series_in_cluster_yi[:, :, col_idx])
                    axs[row_idx, 0].plot(bc.ravel(), "-", label=col)

                axs[row_idx, 0].set_xticks(self.__x_ticks)
                axs[row_idx, 0].grid(which='major', alpha=0.2, color='grey')

            # set y label for row with cluster information
            axs[row_idx, 0].set_ylabel('Cluster ' + str(row_idx + 1) + '\n No TS = ' + str(is_in_cluster_yi.sum()))

        handles, labels = axs[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right')
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        plt.subplots_adjust(top=.9)
        # add overall x, y text
        fig.add_subplot(111, frame_on=False)
        plt.tick_params(labelcolor="none", bottom=False, left=False)
        plt.ylabel("Y =" + y_label_substr + " values", labelpad=30)
        plt.xlabel(self.__x_label)
        plt.show()

    def plot_silhouette_blob_for_k(self, ks: [int]):
        """Plots silhouettes for all given ks

        Parameters
        ----------
        ks : [int]
            all cluster numbers to plot silhouette score for, min list size =4, looks better if len(ks) a
            multiple of 4
        """
        no_clusters = len(ks)
        no_rows = int(len(ks) / 4)
        no_cols = 4
        plt.rcParams.update({'figure.facecolor': 'white', 'axes.facecolor': 'white', 'figure.dpi': 150})
        max_k = max(ks)
        gap = 8
        fig_size = (10, 5)  # allow for multicolumn grids and single column grids

        fig, axs = plt.subplots(nrows=no_rows,
                                ncols=no_cols,
                                sharey=True,
                                sharex=True,
                                figsize=fig_size, squeeze=0)
        fig.suptitle("Silhouette Analysis. Clustered by " + ', '.join(self.__x_train_column_names) + ". No of TS "
                     + str(len(self.y_pred)))

        current_row_idx = 0
        current_col_idx = 0
        min_x = -0.2
        for plot_no in range(no_clusters):
            ax = axs[current_row_idx, current_col_idx]
            ax.set_xlim([min_x, 1])
            # The (n_clusters+1)*gap is for inserting blank space between silhouette
            # plots of individual clusters, to demarcate them clearly.
            ax.set_ylim([0, len(self.y_pred) + (max_k + 1) * gap])

            # Run Kmeans
            k = ks[plot_no]
            model = TimeSeriesKMeans(n_clusters=k, metric=self.__metric, max_iter=self.__max_iter,
                                     random_state=self.__random_state)
            y_pred = model.fit_predict(self.__x_train)

            # calculate silhouette score
            silhouette_avg = silhouette_score(self.__x_train, y_pred, metric=self.__metric)
            print(
                "For n_clusters =",
                k,
                "The average silhouette_score is :",
                silhouette_avg,
            )

            # Compute the silhouette scores for each sample
            sample_silhouette_values = ts_silhouette_samples(self.__x_train, y_pred)

            y_lower = gap
            for i in range(k):
                # Aggregate the silhouette scores for samples belonging to
                # cluster i, and sort them
                ith_cluster_silhouette_values = sample_silhouette_values[y_pred == i]
                ith_cluster_silhouette_values.sort()
                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i

                color = cm.nipy_spectral(float(i) / no_clusters)
                ax.fill_betweenx(
                    np.arange(y_lower, y_upper),
                    0,
                    ith_cluster_silhouette_values,
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.7,
                )

                # Label the silhouette plots with their cluster numbers at the middle
                ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                # Compute the new y_lower for next plot
                y_lower = y_upper + gap  # 10 for the 0 samples

            ax.set_title("k=" + str(k))

            # The vertical line for average silhouette score of all the values
            ax.axvline(x=silhouette_avg, color="red", linestyle="--")

            ax.set_yticks([])  # Clear the yaxis labels / ticks
            ax.set_xticks([min_x, 0, 0.2, 0.4, 0.6, 0.8, 1])

            # Update for next plot
            if current_col_idx == 3:
                current_col_idx = 0
                current_row_idx = current_row_idx + 1
            else:
                current_col_idx = current_col_idx + 1

        lines, labels = fig.axes[-1].get_legend_handles_labels()
        fig.legend(lines, labels)
        fig.supxlabel("Silhouette coefficient values")
        fig.supylabel("Clusters")
        fig.tight_layout(pad=1)
        plt.show()

    def plot_mean_silhouette_score_for_k(self, ks: [int]):
        """Plots mean silhouette score for all the given k

        Parameters
        ----------
        ks : [int]
            all cluster numbers to plot silhouette score for
        """
        silhouette_avg = self.__calculate_mean_silhouette_score_for_ks(ks)

        plt.rcParams.update({'figure.facecolor': 'white'})
        # plot silhouette score
        plt.figure(figsize=(8, 6), dpi=80, facecolor='white')
        plt.plot(ks, silhouette_avg, 'o-')
        plt.xticks(ks)
        plt.xlabel('Values of K')
        plt.ylabel('Mean silhouette score')
        plt.title('Silhouette score for k')
        plt.show()

    def plot_sum_of_square_distances_for_k(self, ks: [int]):
        """Plots sum of square distances for all the given k

        Parameters
        ----------
        ks : [int]
            all cluster numbers to plot silhouette score for
        """
        squared_distances = self.__calculate_sum_of_squared_distances(ks)

        # plot silhouette score
        plt.rcParams.update({'figure.facecolor': 'white'})
        plt.figure(figsize=(8, 6), dpi=80)
        plt.plot(ks, squared_distances, 'o-')
        plt.xticks(ks)
        plt.xlabel('Values of K')
        plt.ylabel('Sum of squared distances')
        plt.title('Elbow method for finding optimal k clusterd cols ' + ', '.join(self.__x_train_column_names))
        plt.show()

    def __calculate_mean_silhouette_score_for_ks(self, ks: [int]):
        silhouette_avg = []
        for num_clusters in ks:
            # initialise kmeans
            model = TimeSeriesKMeans(n_clusters=num_clusters, metric=self.__metric, max_iter=self.__max_iter,
                                     random_state=self.__random_state)
            y_pred = model.fit_predict(self.__x_train)

            # silhouette score
            silhouette_avg.append(silhouette_score(self.__x_train, y_pred, metric=self.__metric))
        return silhouette_avg

    def __calculate_sum_of_squared_distances(self, ks: [int]):
        sum_of_squared_distances = []
        for num_clusters in ks:
            # initialise kmeans
            model = TimeSeriesKMeans(n_clusters=num_clusters, metric=self.__metric, max_iter=self.__max_iter,
                                     random_state=self.__random_state)
            model.fit(self.__x_train)
            sum_of_squared_distances.append(model.inertia_)
        return sum_of_squared_distances

import numpy as np
from matplotlib import pyplot as plt
from tslearn.clustering import TimeSeriesKMeans, silhouette_score
from tslearn.preprocessing import TimeSeriesScalerMeanVariance


class TimeSeriesKMeansClustering:
    """Collection of convenience function for tslearn k-means.

    Attributes
    ----------
    model : TimeSeriesKMeans
        k means model

    y_pred : np.array
        cluster number for each ts in x_train
    """

    def __init__(self, n_clusters: int, x_train: np.array, normalise: bool, x_train_column_names: []):
        """Collection of convenience function for tslearn k-means.

        Parameters
        ----------
        n_clusters : Int
            Number of clusters

        x_train : np.array
            timeseries to cluster as np.array of shape=(n_ts, sz, d), where n_ts is number of days, sz is 24, and
            d=1 for single variate and > for multivariate ts

        normalise : bool
            whether to use a scaler on x_train or nt

        x_train_column_names: []
            list of ts names in x_train
        """
        self.__n_clusters = n_clusters
        self.__normalise = normalise
        if self.__normalise:
            self.__x_train = TimeSeriesScalerMeanVariance().fit_transform(x_train)  # normalise data
        else:
            self.__x_train = x_train
        self.__x_train_column_names = x_train_column_names
        self.__metric = "dtw"
        self.__max_iter = 10
        self.__random_state = 66

        # create clusters
        self.model = TimeSeriesKMeans(n_clusters=self.__n_clusters, metric=self.__metric, max_iter=self.__max_iter,
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
        no_dimensions = self.model.cluster_centers_.shape[2]
        plt.rcParams['figure.dpi'] = 150
        fig_size = (5*no_dimensions, no_clusters * 2)  # allow for multicolumn grids and single column grids

        fig, axs = plt.subplots(nrows=no_clusters,
                                ncols=no_dimensions,
                                sharey=True,
                                sharex=True,
                                figsize=fig_size, squeeze=0, facecolor='white')
        fig.suptitle("DBA k-means. No of TS " + str(len(self.y_pred)))
        # clusters are on the rows
        for row_idx in range(no_clusters):
            is_in_cluster_yi = (self.y_pred == row_idx)

            # plot all the time series for cluster row_idx and all dimensions
            for xx in self.__x_train[is_in_cluster_yi]:
                # plot the ts for each variate in columns
                for col_idx in range(no_dimensions):
                    axs[row_idx, col_idx].plot(xx[:, col_idx].ravel(), 'k-', alpha=.2)

            # plot the cluster line and title
            for col_idx in range(no_dimensions):
                axs[row_idx, col_idx].plot(self.model.cluster_centers_[row_idx][:, col_idx], "r-")
                if row_idx == 0:
                    axs[0, col_idx].set_title(self.__x_train_column_names[col_idx])

            # set y label for row with cluster information
            axs[row_idx, 0].set_ylabel('Cluster ' + str(row_idx + 1) + '\n No TS = ' + str(is_in_cluster_yi.sum()))

        plt.tight_layout()

        # add overall x, y text
        fig.add_subplot(111, frame_on=False)
        plt.tick_params(labelcolor="none", bottom=False, left=False)
        normalised_label_string = " normalised " if self.__normalise else " "
        plt.ylabel(
            "Y =" + normalised_label_string + y_label_substr + " values", labelpad=30)
        plt.xlabel("X = hours of day (UTC)")

    def plot_mean_silhouette_score_for_k(self, ks: [int]):
        """Plots mean silhouette score for all the given k

        Parameters
        ----------
        ks : [int]
            all cluster numbers to plot silhouette score for
        """
        silhouette_avg = self.__calculate_mean_silhouette_score_for_ks(ks)

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
        plt.figure(figsize=(8, 6), dpi=80, facecolor='white')
        plt.plot(ks, squared_distances, 'o-')
        plt.xticks(ks)
        plt.xlabel('Values of K')
        plt.ylabel('Sum of squared distances')
        plt.title('Elbow method for finding optimal k')
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

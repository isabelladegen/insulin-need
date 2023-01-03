import numpy as np
from matplotlib import pyplot as plt, cm
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
from tslearn.barycenters import dtw_barycenter_averaging
from tslearn.clustering import silhouette_score
from tslearn.metrics import dtw
from tslearn.preprocessing import TimeSeriesScalerMinMax

from src.stats import TimeSeriesDescription
from src.timeseries_kmeans_clustering import ts_silhouette_samples


class AgglomerativeTSClustering:
    """Collection of convenience function including visualisations for agglomerative dtw clustering with sklearn

    Attributes
    ----------
    model : AgglomerativeClustering
        sklearn model

    y_pred : np.array
        cluster number for each ts in x_train
    """

    def __init__(self, x_train: np.array, x_train_column_names: [str], sampling: TimeSeriesDescription,
                 scaler=TimeSeriesScalerMinMax(), x_full: np.array = None, x_full_column_names: [str] = None,
                 distance_threshold=0.5,
                 linkage="single",
                 distance_constraint=None,
                 sakoe_chiba_radius=None
                 ):
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
        self.distance_constraint = distance_constraint
        self.sakoe_chiba_radius = sakoe_chiba_radius
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
                                             compute_full_tree=True, linkage=linkage,
                                             distance_threshold=distance_threshold)

        self.distance_matrix = self.__calculate_distance_matrix()
        self.y_pred = self.model.fit_predict(self.distance_matrix)
        self.no_clusters = max(self.y_pred)+1
        self.__calculate_silhouette_values()

    def plot_clusters_in_grid(self, y_label_substr: str, only_display_multiple_ts_clusters=True):
        """Plots clusters in a grid of cols being dimensions and rows being clusters.

        Parameters
        ----------
        y_label_substr: str
            Part of the plot y label
        """
        no_dimensions = len(self.__cols_to_plot)
        if only_display_multiple_ts_clusters:
            cluster_indexes = np.unique(self.y_pred_for_non_single_clusters)
            x_train = self.x_train_for_non_single_clusters
            y_pred = self.y_pred_for_non_single_clusters
            x_full = self.x_full_for_non_single_clusters
        else:
            cluster_indexes = np.unique(self.y_pred)
            x_train = self.__x_train
            y_pred = self.y_pred
            x_full = self.__x_full
        no_clusters = len(cluster_indexes)

        # setup figure
        plt.rcParams.update({'figure.facecolor': 'white', 'axes.facecolor': 'white', 'figure.dpi': 150})
        fig_size = (4 * no_dimensions, no_clusters * 2)  # allow for multicolumn grids and single column grids
        fig, axs = plt.subplots(nrows=no_clusters,
                                ncols=no_dimensions,
                                sharey=True,
                                sharex=True,
                                figsize=fig_size, squeeze=0)
        title = "Agglomerative Clustering. Distance threshold " + str(
            self.model.distance_threshold) + ". Linkage " + str(self.model.linkage) + ". Clustered by " + ', '.join(
            self.__x_train_column_names) + ". No of TS " + str(len(self.y_pred))

        # clusters are on the rows

        for row_idx, cluster_idx in enumerate(cluster_indexes):
            is_in_cluster_yi = (y_pred == cluster_idx)

            # plot all the time series for cluster row_idx and all dimensions
            if x_full is None:  # just plot x train
                series_in_cluster_yi = x_train[is_in_cluster_yi]
            else:  # plot x_full
                series_in_cluster_yi = x_full[is_in_cluster_yi]
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
            axs[row_idx, 0].set_ylabel('Cluster ' + str(cluster_idx) + '\n No TS = ' + str(is_in_cluster_yi.sum()))

        fig.tight_layout()
        # add overall x, y text
        fig.add_subplot(111, frame_on=False)
        plt.tick_params(labelcolor="none", bottom=False, left=False)
        plt.ylabel("Y =" + y_label_substr + " values", labelpad=30)
        plt.xlabel(self.__x_label + "\n" + title)
        plt.show()

    def plot_silhouette_analysis(self):
        """Plots silhouette score for all clusters with more than one TS
        """
        cluster_index = np.unique(self.y_pred_for_non_single_clusters)
        no_non_single_clusters = len(cluster_index)

        gap = 25
        min_x = -0.2
        fig_size = (10, 5)

        plt.rcParams.update({'figure.facecolor': 'white', 'axes.facecolor': 'white', 'figure.dpi': 150})
        plt.show()
        fig, axs = plt.subplots(nrows=1,
                                ncols=1,
                                sharey=True,
                                sharex=True,
                                figsize=fig_size, squeeze=0)

        ax = axs[0, 0]
        title = "Silhouette Analysis. Distance threshold " + str(
            self.model.distance_threshold) + ". Linkage " + str(self.model.linkage) + "\n Clustered by " + ', '.join(
            self.__x_train_column_names) + ". Total no of TS " + str(
            len(self.y_pred)) + ". Avg Silhouette Score " + str(
            round(self.silhouette_avg, 3))
        ax.set_title(title)
        ax.set_xlim([min_x, 1])
        # The (n_clusters+1)*gap is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax.set_ylim([0, len(self.y_pred_for_non_single_clusters) + (no_non_single_clusters + 1) * gap])

        # plot all silhouettes for non single ts clusters
        y_lower = gap
        for i, cluster_index in enumerate(cluster_index):
            # Aggregate the silhouette scores for samples belonging to
            # cluster cluster_index, and sort them
            ith_cluster_silhouette_values = self.sample_silhouette_values[
                self.y_pred_for_non_single_clusters == cluster_index]
            ith_cluster_silhouette_values.sort()
            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / no_non_single_clusters)
            ax.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )

            # Label the silhouette plots with their cluster numbers at the middle
            ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(cluster_index) + ", no TS " + str(
                np.count_nonzero(self.y_pred_for_non_single_clusters == cluster_index)))

            # Compute the new y_lower for next plot
            y_lower = y_upper + gap  # 10 for the 0 samples

        # The vertical line for average silhouette score of all the values
        ax.axvline(x=self.silhouette_avg, color="red", linestyle="--")

        ax.set_yticks([])  # Clear the yaxis labels / ticks
        ax.set_xticks([min_x, 0, 0.2, 0.4, 0.6, 0.8, 1])
        lines, labels = fig.axes[-1].get_legend_handles_labels()
        fig.legend(lines, labels)
        ax.set_xlabel("Silhouette coefficient values")
        ax.set_ylabel(str(no_non_single_clusters) + " non single ts clusters. Total " + str(self.no_clusters))
        plt.show()

    def plot_dendrogram(self, truncated_mode: str = None, p: int = 30, show_contracted: bool = False,
                        show_leave_count=False, count_sort=False, distance_sort=False, no_labels=False):
        """Plots dendrogram of clusters
        """
        counts = np.zeros(self.model.children_.shape[0])
        n_samples = len(self.model.labels_)
        for i, merge in enumerate(self.model.children_):
            current_count = 0
            for child_idx in merge:
                if child_idx < n_samples:
                    current_count += 1  # leaf node
                else:
                    current_count += counts[child_idx - n_samples]
            counts[i] = current_count

        linkage_matrix = np.column_stack([self.model.children_, self.model.distances_, counts]).astype(float)

        if self.distance_constraint is not None:
            print("Additional distance metrics: " + self.distance_constraint + ' & ' + str(self.sakoe_chiba_radius))

        # Plot the corresponding dendrogram
        plt.rcParams.update({'figure.facecolor': 'white',
                             'axes.facecolor': 'white',
                             'figure.dpi': 150,
                             'figure.figsize': (20, 15)
                             })
        truncated_str = '(truncated)' if truncated_mode else ''
        plt.title('Hierarchical Clustering Dendrogram ' + truncated_str + '. No TS ' + str(
            len(self.y_pred)), fontsize=20)
        plt.xlabel('No clusters: ' + str(len(set(self.y_pred))) +
                   '. No ts in majority cluster: ' + str(list(self.y_pred).count(np.bincount(self.y_pred).argmax())),
                   fontsize=18)
        plt.ylabel('Distance. Linkage: ' + self.model.linkage + ', threshold: ' + str(self.model.distance_threshold),
                   fontsize=18)
        dendrogram(
            linkage_matrix,
            truncate_mode=truncated_mode,
            p=p,
            leaf_rotation=90.,
            show_contracted=show_contracted,
            show_leaf_counts=show_leave_count,
            count_sort=count_sort,
            color_threshold=self.model.distance_threshold,
            distance_sort=distance_sort,
            leaf_font_size=12,
            no_labels=no_labels
        )

    def get_y_pred_as_binary(self):
        """Returns y pred with only two classes: normal and anomaly. The normal class is the most frequent class
        """
        most_frequent_class = np.bincount(self.y_pred).argmax()
        result = ["normal" if x == most_frequent_class else "anomaly" for x in self.y_pred]
        print("Number of classes: " + str(len(set(self.y_pred))))
        print("Majority class: " + str(most_frequent_class))
        print("Frequency of majority class: " + str(list(self.y_pred).count(most_frequent_class)))
        print("Frequency of other classes: " + str(result.count("anomaly")))
        return result

    def __calculate_distance_matrix(self):
        # return symmetric matrix of dtw distances
        n_ts = self.__x_train.shape[0]
        distance_matrix = np.zeros((n_ts, n_ts))
        for row in range(n_ts):
            for column in range(n_ts):
                s1 = self.__x_train[row, :, 0]
                s2 = self.__x_train[column, :, 0]
                distance_matrix[row][column] = dtw(s1, s2, global_constraint=self.distance_constraint,
                                                   sakoe_chiba_radius=self.sakoe_chiba_radius,
                                                   itakura_max_slope=None)
        return distance_matrix

    def __calculate_silhouette_values(self):
        """ Calculates silhouette score for clusters with more than one ts and silhouette values for all clusters
        """

        # np.delete(a, 2, axis=1)
        cluster_dic = self.__get_dictionary_of_clusters_and_ts_in_cluster()
        actual_clusters = self.__get_multiple_ts_clusters(cluster_dic)

        x_train_for_actual_clusters = None
        y_pred_for_actual_clusters = None
        x_full_for_actual_clusters = None
        for cluster_idx in actual_clusters.keys():
            is_in_cluster_y = (self.y_pred == cluster_idx)
            series_in_cluster_yi = self.__x_train[is_in_cluster_y]
            y_in_cluster_yi = self.y_pred[is_in_cluster_y]
            if self.__x_full is not None:
                x_full_in_cluster_yi = self.__x_full[is_in_cluster_y]
            if x_train_for_actual_clusters is None:
                x_train_for_actual_clusters = series_in_cluster_yi
                y_pred_for_actual_clusters = y_in_cluster_yi
                if self.__x_full is not None:
                    x_full_for_actual_clusters = x_full_in_cluster_yi
            else:
                x_train_for_actual_clusters = np.concatenate((x_train_for_actual_clusters, series_in_cluster_yi),
                                                             axis=0)
                y_pred_for_actual_clusters = np.concatenate((y_pred_for_actual_clusters, y_in_cluster_yi))
                if self.__x_full is not None:
                    x_full_for_actual_clusters = np.concatenate((x_full_for_actual_clusters, x_full_in_cluster_yi),
                                                                axis=0)

        self.x_train_for_non_single_clusters = x_train_for_actual_clusters
        self.y_pred_for_non_single_clusters = y_pred_for_actual_clusters
        self.x_full_for_non_single_clusters = x_full_for_actual_clusters
        self.silhouette_avg = silhouette_score(x_train_for_actual_clusters, y_pred_for_actual_clusters, metric="dtw")
        self.sample_silhouette_values = ts_silhouette_samples(x_train_for_actual_clusters, y_pred_for_actual_clusters)

    def __get_dictionary_of_clusters_and_ts_in_cluster(self):
        """ Creates dictionary with cluster index as key and TS in that cluster

            :returns
            {key = cluster index  : value = ts in that cluster}
        """
        result = {}
        for cluster_index in range(self.no_clusters):
            is_in_cluster_yi = (self.y_pred == cluster_index)
            series_in_cluster_yi = self.__x_train[is_in_cluster_yi]
            result[cluster_index] = series_in_cluster_yi
        return result

    def __get_multiple_ts_clusters(self, clusters_dict):
        """ Return dictionary of all the clusters that have more than one ts

           :returns
           {key = cluster index with more than one ts  : value = ts in that cluster}
        """
        result = {}
        for cluster_index in clusters_dict:
            # cluster has more than one ts
            timeseries_in_cluster = clusters_dict[cluster_index]
            if timeseries_in_cluster.shape[0] > 1:
                result[cluster_index] = timeseries_in_cluster
        return result

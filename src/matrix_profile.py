from math import sqrt

import numpy as np
import pandas as pd
import stumpy
from matplotlib import pyplot as plt
from numpy import ndarray
from scipy.stats import stats


class MatrixProfile:
    label_font_size = 35

    def __init__(self, time_series: [], values_series: ndarray, motif_length_m: int):
        self.__index = time_series
        self.__values = values_series
        self.__motif_length_m = motif_length_m
        self.mp = stumpy.stump(T_A=self.__values, m=self.__motif_length_m)
        self.__x_mp = self.__index[:self.mp.shape[0]]  # the last m+1 dates are missing as |mp|=|T|-m+1
        self.__motif_index_sorted = np.argsort(self.mp[:, 0])  # create array of indexes sorted by lowest distance first
        self.__z_score_normalised_values = stats.zscore(values_series)

    def get_motif_and_nearest_neighbor_idx_for_xth_motif(self, x: int = 0):
        motif_idx = self.__motif_index_sorted[x]
        nearest_neighbor_idx = self.mp[motif_idx, 1]
        return motif_idx, nearest_neighbor_idx

    def get_distance_for_motif(self, motif_idx: int):
        return round(self.mp[motif_idx, 0], 3)

    def describe_time_series(self):
        ts_start_date = self.__index[0].strftime('%Y-%m-%d %a')
        ts_end_date = self.__index[-1].strftime('%Y-%m-%d %a')
        max_mp_distance = self.max_possible_distance()
        median_frequency = pd.Series(data=self.__index).diff().median()
        describe = pd.Series(data=self.__values).describe()
        print(f'max possible mp distance is {max_mp_distance}')
        print(f'Time Series start date is {ts_start_date}')
        print(f'Time Series end date is {ts_end_date}')
        print(f'Time Series median frequency {median_frequency}')
        print(f'Distribution of values is {describe}')

    # if x is zero it will be the motif with the lowest distance
    def describe_motif_x(self, x: int):
        motif_idx, nearest_neighbor_idx = self.get_motif_and_nearest_neighbor_idx_for_xth_motif(x)
        motif_start_date_str = self.__index[motif_idx].strftime('%Y-%m-%d %a')
        nearest_neighbor_start_date_str = self.__index[nearest_neighbor_idx].strftime('%Y-%m-%d %a')
        distance_to_nearest_neighbour = round(self.mp[motif_idx, 0], 1)
        nearest_neighours_distance_to_its_nearest_neighbour = round(self.mp[nearest_neighbor_idx, 0], 2)
        print(f'Motive Index is {motif_idx}')
        print(f'Nearest Neighbour Index is {nearest_neighbor_idx}')
        print(f'Motive Distance to nearest neighbour is {distance_to_nearest_neighbour}')
        print(f'Nearest neighbours\' is nearest to idx {self.mp[nearest_neighbor_idx, 1]} '
              f'with distance {nearest_neighours_distance_to_its_nearest_neighbour}')
        print(f'Motive start date is: {motif_start_date_str}')
        print(f'Nearest neighbour start date is: {nearest_neighbor_start_date_str}')

    # if x is zero it will be the motif with the lowest distance
    def plot_ts_motif_and_profile(self, x: int, ts_y_label: str, overall_x_label: str, show_z_score_normalised=False):
        plt.rcParams['figure.figsize'] = (20, 15)
        fig, axs = plt.subplots(3 if show_z_score_normalised else 2, sharex=True)
        plt.rcParams.update({'font.size': 25})
        mp_ax_idx = 2 if show_z_score_normalised else 1

        motif_idx, nearest_neighbor_idx = self.get_motif_and_nearest_neighbor_idx_for_xth_motif(x)

        plt.suptitle(
            'Motif ' + str(x) + ': '
            + self.__index[0].strftime('%Y-%m-%d') + ' - '
            + self.__index[-1].strftime('%Y-%m-%d')
            + ', length '
            + str(self.__motif_length_m)
            + ', distance '
            + str(self.get_distance_for_motif(motif_idx))
        )

        self.__plot_motif_on_series(self.__values, axs[0], motif_idx, nearest_neighbor_idx, ts_y_label)

        if show_z_score_normalised:
            self.__plot_motif_on_series(self.__z_score_normalised_values, axs[1], motif_idx, nearest_neighbor_idx,
                                        'z-score normalised')

        # show mp
        axs[mp_ax_idx].plot(self.__x_mp, self.mp[:, 0], linewidth=3, marker='o')
        axs[mp_ax_idx].set_ylabel('Matrix Profile', fontsize=self.label_font_size)
        axs[mp_ax_idx].axvline(x=self.__x_mp[motif_idx], linestyle="dashed")
        axs[mp_ax_idx].axvline(x=self.__x_mp[nearest_neighbor_idx], linestyle="dashed")
        axs[mp_ax_idx].tick_params(axis='x', labelsize=self.label_font_size)
        axs[mp_ax_idx].tick_params(axis='y', labelsize=self.label_font_size)
        plt.tight_layout()
        plt.xlabel(overall_x_label, fontsize=self.label_font_size)
        plt.show()

    def __plot_motif_on_series(self, values, axs, motif_idx, nearest_neighbor_idx, ts_y_label):
        axs.plot(self.__index, values, linewidth=3, marker='o')  # plot with time as index
        axs.set_ylabel(ts_y_label, fontsize=self.label_font_size)
        # highlight the motive with the lowest distance
        axs.plot(self.__index[motif_idx:motif_idx + self.__motif_length_m],
                 values[motif_idx:motif_idx + self.__motif_length_m],
                 linewidth=5, marker='o')
        axs.plot(self.__index[nearest_neighbor_idx:nearest_neighbor_idx + self.__motif_length_m],
                 values[nearest_neighbor_idx:nearest_neighbor_idx + self.__motif_length_m],
                 linewidth=5, marker='o')
        axs.axvline(x=self.__x_mp[motif_idx], linestyle="dashed")
        axs.axvline(x=self.__x_mp[nearest_neighbor_idx], linestyle="dashed")
        axs.tick_params(axis='x', labelsize=self.label_font_size)
        axs.tick_params(axis='y', labelsize=self.label_font_size)

    # returns index of the matrix profile for the motif that is the least similar to anywhere else - a discord
    def least_similar_x(self):
        return len(self.__motif_index_sorted) - 1

    # see paper: https://core.ac.uk/download/pdf/287941767.pdf
    def max_possible_distance(self):
        return round(2 * sqrt(self.__motif_length_m), 2)

    # see top motifs https://stumpy.readthedocs.io/en/latest/api.html#motifs
    # returns  motif_distances and motif_indices
    def top_motives(self, max_distance: float, min_neighbours: int = 1):
        return stumpy.motifs(self.__values, self.mp[:, 0], min_neighbors=min_neighbours, max_distance=max_distance)

    def plot_top_motives_for_max_distance_and_min_neighbours(self, y_label: str, x_label: str, max_distance: float,
                                                             min_neighbours: int = 1, show_mp=False):
        motif_distances, motif_indices = self.top_motives(max_distance, min_neighbours)
        plt.rcParams['figure.figsize'] = (20, 15) if show_mp else (20, 10)
        fig, axs = plt.subplots(2 if show_mp else 1, sharex=True, squeeze=0)
        plt.rcParams.update({'font.size': 25})
        plt.suptitle('Top motifs with distance less than ' + str(round(max_distance, 2))
                     + ', min neighbours ' + str(min_neighbours)
                     + ', m=' + str(self.__motif_length_m))

        axs[0, 0].plot(self.__index, self.__values, linewidth=3, marker='o')  # plot ts with time as index
        axs[0, 0].set_ylabel(y_label, fontsize=self.label_font_size)
        axs[0, 0].tick_params(axis='x', labelsize=self.label_font_size)
        axs[0, 0].tick_params(axis='y', labelsize=self.label_font_size)
        # highlight the motifs
        for idx in list(motif_indices[0]):
            axs[0, 0].plot(self.__index[idx:idx + self.__motif_length_m],
                           self.__values[idx:idx + self.__motif_length_m],
                           linewidth=5, marker='o')
            axs[0, 0].axvline(x=self.__index[idx], linestyle="dashed")

        # plot mp if set to true
        if show_mp:
            axs[1, 0].plot(self.__x_mp, self.mp[:, 0], linewidth=3, marker='o')
            axs[1, 0].set_ylabel('Matrix Profile', fontsize=self.label_font_size)
            axs[1, 0].tick_params(axis='x', labelsize=self.label_font_size)
            axs[1, 0].tick_params(axis='y', labelsize=self.label_font_size)

            # dashed lines
            for idx in list(motif_indices[0]):
                axs[1, 0].axvline(x=self.__index[idx], linestyle="dashed")

        plt.tight_layout()
        plt.xlabel(x_label, fontsize=self.label_font_size)
        plt.show()

    @classmethod
    def get_longest_series_values_times(cls, continuous_time_series_dfs: [pd.DataFrame],
                                        variate: str):
        """ Method to calculate and reshape into values and time the longest continuous time series in a list of
        dataframes

        Parameters
        ----------
        continuous_time_series_dfs : [pd.DataFrame]
            list of pandas dataframes of shape (time,variates) created by TranslateIntoTimeseries

        variate : str
            which variate to return the values for


        Returns
        -------
        values, times : ndarray
            shaped for the matrix profile
        """
        series_lengths = [df.shape[0] for df in continuous_time_series_dfs]
        index_longest_series = np.argmax(series_lengths)
        longest_series = continuous_time_series_dfs[index_longest_series]

        values = longest_series[variate].to_numpy()
        times = list(longest_series.index)
        return values, times

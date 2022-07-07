from math import sqrt

import numpy as np
import pandas as pd
import stumpy
from matplotlib import pyplot as plt


class MatrixProfile:
    def __init__(self, time_series, values_series: [float], motif_length_m: int):
        self.__index = time_series
        self.__values = values_series
        self.__motif_length_m = motif_length_m
        self.mp = stumpy.stump(self.__values, self.__motif_length_m)  # true matrix profile
        self.__x_mp = self.__index[:self.mp.shape[0]]  # the last m+1 dates are missing as |mp|=|T|-m+1
        self.__motif_index_sorted = np.argsort(self.mp[:, 0])  # create array of indexes sorted by lowest distance first

    def get_motif_and_nearest_neighbor_idx_for_xth_motif(self, x: int = 0):
        motif_idx = self.__motif_index_sorted[x]
        nearest_neighbor_idx = self.mp[motif_idx, 1]
        return motif_idx, nearest_neighbor_idx

    def get_distance_for_motif(self, motif_idx: int):
        return self.mp[motif_idx, 0]

    def describe_time_series(self):
        ts_start_date = self.__index[0].strftime('%Y-%m-%d %a')
        ts_end_date = self.__index[-1].strftime('%Y-%m-%d %a')
        max_mp_distance = round(2 * sqrt(self.__motif_length_m), 2)
        median_frequency = pd.Series(data=self.__index).diff().median()
        describe = pd.Series(data=self.__values).describe()
        print(f'Time Series start date is {ts_start_date}')
        print(f'Time Series end date is {ts_end_date}')
        print(f'Time Series median frequency {median_frequency}')
        print(f'Distribution of values is {describe}')
        print(f'max mp distance is {max_mp_distance}')

    # if x is zero it will be the motif with the lowest distance
    def describe_motif_x(self, x: int):
        motif_idx, nearest_neighbor_idx = self.get_motif_and_nearest_neighbor_idx_for_xth_motif(x)
        motif_start_date_str = self.__index[motif_idx].strftime('%Y-%m-%d %a')
        nearest_neighbor_start_date_str = self.__index[nearest_neighbor_idx].strftime('%Y-%m-%d %a')
        print(f'Motive Index is {motif_idx}')
        print(f'Nearest Neighbour Index is {nearest_neighbor_idx}')
        print(f'Motive Distance to nearest neighbour is {self.mp[motif_idx, 0]}')
        print(f'Nearest neighbours\' is nearest to idx {self.mp[nearest_neighbor_idx, 1]} '
              f'with distance {self.mp[nearest_neighbor_idx, 0]}')
        print(f'Motive start date is: {motif_start_date_str}')
        print(f'Nearest neighbour start date is: {nearest_neighbor_start_date_str}')

    # if x is zero it will be the motif with the lowest distance
    def plot_ts_motif_and_profile(self, x: int, ts_y_label: str, overall_x_label: str):
        plt.rcParams['figure.figsize'] = (20, 10)
        fig, axs = plt.subplots(2, sharex=True)
        plt.rcParams.update({'font.size': 20})

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

        axs[0].plot(self.__index, self.__values, marker='o')  # plot with time as index
        axs[0].set_ylabel(ts_y_label)
        # highlight the motive with the lowest distance
        axs[0].plot(self.__index[motif_idx:motif_idx + self.__motif_length_m],
                    self.__values[motif_idx:motif_idx + self.__motif_length_m],
                    linewidth=3, marker='o')
        axs[0].plot(self.__index[nearest_neighbor_idx:nearest_neighbor_idx + self.__motif_length_m],
                    self.__values[nearest_neighbor_idx:nearest_neighbor_idx + self.__motif_length_m],
                    linewidth=3, marker='o')
        axs[0].axvline(x=self.__x_mp[motif_idx], linestyle="dashed")
        axs[0].axvline(x=self.__x_mp[nearest_neighbor_idx], linestyle="dashed")
        axs[1].plot(self.__x_mp, self.mp[:, 0], marker='o')
        axs[1].set_ylabel('Matrix Profile')
        axs[1].axvline(x=self.__x_mp[motif_idx], linestyle="dashed")
        axs[1].axvline(x=self.__x_mp[nearest_neighbor_idx], linestyle="dashed")
        plt.tight_layout()
        plt.xlabel(overall_x_label)
        plt.show()

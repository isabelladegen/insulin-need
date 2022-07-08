import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from src.preprocess import number_of_interval_in_days, continuous_subseries
from src.resample import resample_df


class ContinuousSeries:
    def __init__(self, df: pd.DataFrame,
                 min_days_of_data: int,
                 max_interval_between_reading: int,
                 time_col: str,
                 value_col: str,
                 resample_rule: str):
        self.df = df
        # how frequent readings need to be per day, 60 = once per hour, 180 = every 2 hour, ...
        self.__max_interval_between_readings = max_interval_between_reading
        self.__time_column = time_col  # column that's used for resampling
        self.__value_column = value_col  # column that's used for timeseries
        # how many days of consecutive readings with at least a reading every interval
        self.__min_days_of_data = min_days_of_data
        self.__min_series_length = number_of_interval_in_days(self.__min_days_of_data,
                                                              self.__max_interval_between_readings)
        self.subseries = continuous_subseries(self.df, self.__min_series_length, self.__max_interval_between_readings,
                                              self.__time_column)
        self.__resample_rule = resample_rule
        self.resampled_series = self.__resample()

    def __resample(self):
        result = []
        for group in self.subseries:
            result.append(
                resample_df(group, self.__resample_rule, self.__time_column, self.__value_column))
        return result

    def describe(self):
        aggregators = self.resampled_series[0].columns
        print(f'Number of continuous series: {len(self.subseries)}')
        print(f'Resampled series aggregators: {aggregators}')
        print(f'Max gap between readings {self.__max_interval_between_readings} minutes')
        print(f'Minimum consecutive days of data is {self.__min_days_of_data}')
        print(f'Resampling rule: {self.__resample_rule}')
        print(f'Number data points in continuous series: {[s.shape[0] for s in self.subseries]}')
        print(f'Number data points in resampled series: {[s.shape[0] for s in self.resampled_series]}')

    def plot_resampled_series(self):
        width = 20
        height = 15
        number_of_plots = len(self.resampled_series)

        plt.rcParams.update({'font.size': 30})
        fig, axs = plt.subplots(number_of_plots, sharey=True, figsize=(width, height))

        title = 'Resampled Time Series. Resample rule: ' \
                + self.__resample_rule \
                + ' , min consecutive days of data: ' \
                + str(self.__min_days_of_data)
        fig.suptitle(title)

        for idx, df in enumerate(self.resampled_series):
            y = df[self.__value_column]
            axs[idx].plot(df.index, y, marker='o',
                          label=list(y.columns))  # plot with time as index
            axs[idx].set_xlabel('')
            axs[idx].legend().set_visible(False)

        lines, labels = fig.axes[-1].get_legend_handles_labels()
        fig.legend(lines, labels)
        fig.supxlabel(self.__time_column)
        fig.supylabel(self.__value_column)
        fig.tight_layout(pad=2)
        plt.show()

    def get_resampled_x_and_y_for(self, series_index: int, resampled_sub_col: str):
        df = self.resampled_series[series_index]
        x = list(df.index)
        y = df[self.__value_column][resampled_sub_col].astype(np.float64)
        return x, y

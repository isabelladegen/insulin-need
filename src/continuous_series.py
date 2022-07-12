from calendar import calendar
from math import ceil

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt, ticker

from src.preprocess import number_of_interval_in_days, continuous_subseries
from src.resample import resample_df, z_score_normalise


class ContinuousSeries:
    std_col_name = 'std'
    mean_col_name = 'mean'
    z_score_col_name = 'z-score'

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
                                              self.__time_column, self.__value_column)
        self.__resample_rule = resample_rule
        self.resampled_series = self.__resample()

    def __resample(self):
        result = []
        for group in self.subseries:
            ts = resample_df(group, self.__resample_rule, self.__time_column, self.__value_column)
            ts = z_score_normalise(ts, (self.__value_column, self.mean_col_name),
                                   (self.__value_column, self.z_score_col_name))
            result.append(ts)
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
        height = 20
        number_of_plots = len(self.resampled_series)

        plt.rcParams.update({'font.size': 15})
        fig, axs = plt.subplots(number_of_plots, sharey=True, figsize=(width, height))

        title = 'Resampled Time Series. Resample rule: ' \
                + self.__resample_rule \
                + ' , min consecutive days of data: ' \
                + str(self.__min_days_of_data)
        fig.suptitle(title)

        for idx, df in enumerate(self.resampled_series):
            # multi index column
            y = df[self.__value_column]
            std = y[self.std_col_name]
            mean = y[self.mean_col_name]
            columns_to_plot_as_lines = list(y.columns)
            columns_to_plot_as_lines.remove(self.std_col_name)
            columns_to_plot_as_lines.remove(self.mean_col_name)
            columns_to_plot_as_lines.remove(self.z_score_col_name)
            y = y[columns_to_plot_as_lines]
            axs[idx].plot(df.index, y, marker='o',
                          label=columns_to_plot_as_lines)  # plot with time as index
            axs[idx].errorbar(df.index, mean, yerr=list(std), fmt='-o', capsize=3, label=self.mean_col_name)
            axs[idx].set_xlabel('')
            axs[idx].yaxis.set_minor_locator(ticker.MultipleLocator(0.5))
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

    def plot_z_score_normalised_resampled_series(self):
        width = 20
        height = 20
        number_of_plots = len(self.resampled_series)

        plt.rcParams.update({'font.size': 15})
        fig, axs = plt.subplots(number_of_plots, sharey=True, figsize=(width, height))

        title = 'Resampled Time Series z-score of mean. Resample rule: ' \
                + self.__resample_rule \
                + ' , min consecutive days of data: ' \
                + str(self.__min_days_of_data)
        fig.suptitle(title)

        for idx, df in enumerate(self.resampled_series):
            # multi index column
            y = df[(self.__value_column, self.z_score_col_name)]
            axs[idx].plot(df.index, y, marker='o',
                          label=self.z_score_col_name)  # plot with time as index
            axs[idx].set_xlabel('')
            axs[idx].yaxis.set_minor_locator(ticker.MultipleLocator(0.5))
            axs[idx].legend().set_visible(False)

        lines, labels = fig.axes[-1].get_legend_handles_labels()
        fig.legend(lines, labels)
        fig.supxlabel(self.__time_column)
        fig.supylabel(self.__value_column)
        fig.tight_layout(pad=2)
        plt.show()

    def plot_heathmap_resampled(self):
        y_axis = 'Day of week'
        x_axis = 'Month'
        data = None
        # build one big dataframe of all the subseries
        for df in self.resampled_series:
            new_df = df.copy()
            new_df.columns = new_df.columns.droplevel()
            new_df[y_axis] = new_df.index.dayofweek
            # new_df[self.week_of_month] = pd.Series(new_df.index).apply(lambda d: self.calulate_week_of_month(d)).values
            new_df[x_axis] = new_df.index.month

            if data is None:
                data = new_df
            else:
                data = pd.concat([data, new_df])

        data = data[[y_axis, self.mean_col_name, x_axis]]
        data[self.mean_col_name] = data[self.mean_col_name].astype(np.float64)
        pivot = pd.pivot_table(data=data,
                               index=y_axis,
                               values=self.mean_col_name,
                               columns=x_axis,
                               aggfunc=np.mean)
        # aggfunc=lambda x: self.some_magic(x))

        ax = sns.heatmap(pivot, linewidth=0.5)
        ax.set(title="Heatmap of mean values")

        plt.show()


# def some_magic(self, x):
#     print(x)
#     return x.count()

# def calulate_week_of_month(self, dt):
#     first_day = dt.replace(day=1)
#     dom = dt.day
#     adjusted_dom = dom + (1 + first_day.weekday()) % 7
#
#     return int(ceil(adjusted_dom / 7.0))

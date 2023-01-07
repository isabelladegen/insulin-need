from dataclasses import dataclass

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from src.configurations import Resampling, GeneralisedCols, Aggregators
from src.translate_into_timeseries import TimeColumns, TimeSeriesDescription, TranslateIntoTimeseries


@dataclass
class ConfidenceIntervalCols:
    ci_96hi = "ci96_hi"
    ci_96lo = "ci96_lo"


class Stats:
    """Class for plotting statistics

    """

    def __init__(self, df: pd.DataFrame, sampling: Resampling, ts_description: TimeSeriesDescription,
                 times: [TimeColumns], value_columns: [GeneralisedCols]):
        """
            Parameters
            ----------
            df : DataFrame
                Data to use in stats

            sampling: Resampling
                How the data in df was resampled - Irregular, Daily, Hourly

            ts_description: TimeSeriesDescription
                How the data was shaped into a time series: IrregularTimeseries, DailyTimeseries, WeeklyTimeseries

            times : [TimeColumns]
                Which times to calculate statistics over. Hours only makes sense if the Df has times for different hours

            value_columns : [GeneralisedCols]
                Which value columns of the df to calculate statistics for
        """
        self.df_with_time_cols = TranslateIntoTimeseries.add_time_feature_columns(df)
        self.__sampling = sampling
        self.__ts_description = ts_description
        self.__time_columns = times
        self.__value_columns = value_columns
        # Dictionary of stats dataframes keyed by TimeColumns. Dfs are of format: rows=time (hour, weekdays,...),
        # columns=Multi index, first level aggregators mean, std, count, second level value_columns
        self.stats_per_time = self.__calculate_stats()
        self.font_size = 20
        self.tick_font_size = 15
        # creates a string of all the indexes and counts the characters of all the numbers in the x ticks
        # to decide on the width ratios for the plots
        self.no_x_values = [len(''.join(map(str, list(df.index)))) for df in
                            self.stats_per_time.values()]

    def plot_confidence_interval(self, show_title: bool = True):
        """Plots confidence interval for the given time and value columns
        """
        rows_values = self.__value_columns
        columns_times = self.__time_columns

        # style plot
        axes, fig = self.__reset_plt_for_row_value_column_times_plots(columns_times, rows_values)
        horizontal_line_width = 0.5
        line_width = 4
        line_color = "steelblue"

        # calculate max/min ci values per row accross all time columns
        max_ci = pd.concat(
            [df[ConfidenceIntervalCols.ci_96hi].max().to_frame().T for df in self.stats_per_time.values()]).max()
        min_ci = pd.concat(
            [df[ConfidenceIntervalCols.ci_96lo].min().to_frame().T for df in self.stats_per_time.values()]).min()

        # plot each box
        for cdx, column in enumerate(columns_times):
            stats_df = self.stats_per_time[column]
            x = list(stats_df.index)
            for rdx, row in enumerate(rows_values):
                # configure plot size and accuracy
                y_min = min_ci[row]
                y_max = max_ci[row]
                spacer = y_max / 10
                y_axis_scale = (y_min - spacer, y_max + spacer)

                left = [xs - horizontal_line_width / 2 for xs in x]
                right = [xs + horizontal_line_width / 2 for xs in x]

                # values to plot
                mean = list(stats_df[Aggregators.mean][row])
                ci_hi = list(stats_df[ConfidenceIntervalCols.ci_96hi][row])
                ci_low = list(stats_df[ConfidenceIntervalCols.ci_96lo][row])

                ax = axes[rdx][cdx]
                # plot vertical line
                ax.plot([x, x], [ci_hi, ci_low], linewidth=line_width, color=line_color, )
                # plot bottom and top horizontal lines
                ax.plot([left, right], [ci_hi, ci_hi], linewidth=line_width, color=line_color, )
                ax.plot([left, right], [ci_low, ci_low], color=line_color, linewidth=line_width)
                # plot mean dot
                ax.plot(x, mean, 'o', color='r', markersize=6)

                ax.set_xticks(x)
                # forcing all y_axis  in a row to be the same to ensure we can compare across different times
                ax.set(ylim=y_axis_scale)

                # more styling
                ax.tick_params(axis='x', labelsize=self.tick_font_size)
                ax.tick_params(axis='y', labelsize=self.tick_font_size)
                ax.grid(which='major', alpha=0.3, color='grey')

                # print labels
                if cdx == 0:  # first column print y labels
                    ax.set_ylabel(row, fontsize=self.font_size)

                if rdx == len(rows_values) - 1:  # last row print x labels
                    ax.set_xlabel(column, fontsize=self.font_size)

        self.__show_title_for_plot(fig, show_title)

    def plot_violin_plot(self, show_title: bool = True):
        """Plots violin plots for the given time and value columns
        """
        rows_values = self.__value_columns
        columns_values = self.__time_columns

        # style plot
        axes, fig = self.__reset_plt_for_row_value_column_times_plots(columns_values, rows_values)
        palette = 'Blues'
        sns.set(style="whitegrid")

        for rdx, row in enumerate(rows_values):
            for cdx, column in enumerate(columns_values):
                ax = sns.violinplot(x=column, y=row, data=self.df_with_time_cols, ax=axes[rdx, cdx], palette=palette)
                self.__style_shared_axis_labels(ax, cdx, column, rdx, row, len(rows_values))

                # print labels
                if cdx == 0:  # first column print y labels
                    ax.set_ylabel(row, fontsize=self.font_size)
                if rdx == len(rows_values) - 1:  # last row print x labels
                    ax.set_xlabel(column, fontsize=self.font_size)

        self.__show_title_for_plot(fig, show_title)

    def plot_box_plot(self, show_title: bool = True):
        """Plots box plots for the given time and value columns
        """
        rows_values = self.__value_columns
        columns_values = self.__time_columns

        # style plot
        axes, fig = self.__reset_plt_for_row_value_column_times_plots(columns_values, rows_values)
        palette = 'Blues'
        sns.set(style="whitegrid")

        for rdx, row in enumerate(rows_values):
            for cdx, column in enumerate(columns_values):
                ax = sns.boxplot(x=column, y=row, data=self.df_with_time_cols, ax=axes[rdx, cdx], palette=palette)
                self.__style_shared_axis_labels(ax, cdx, column, rdx, row, len(rows_values))

        self.__show_title_for_plot(fig, show_title)

    def __show_title_for_plot(self, fig, show_title):
        if show_title:
            title = "Resampling: " + self.__sampling.description + ", TS reshaping: " + \
                    self.__ts_description.name
            fig.suptitle(title, fontsize=self.font_size)
        fig.tight_layout()
        plt.show()

    def __style_shared_axis_labels(self, ax, cdx, column, rdx, row, number_of_rows):
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.tick_params(axis='x', labelsize=self.tick_font_size)
        ax.tick_params(axis='y', labelsize=self.tick_font_size)

        # print labels
        if cdx == 0:  # first column print y labels
            ax.set_ylabel(row, fontsize=self.font_size)
        if rdx == number_of_rows - 1:  # last row print x labels
            ax.set_xlabel(column, fontsize=self.font_size)

    def __reset_plt_for_row_value_column_times_plots(self, columns_times, rows_values):
        plt.rcParams.update(plt.rcParamsDefault)
        plt.rcParams.update({'figure.facecolor': 'white', 'axes.facecolor': 'white', 'figure.dpi': 150})
        fig, axes = plt.subplots(nrows=len(rows_values),
                                 ncols=len(columns_times),
                                 sharey='row',
                                 sharex='col',
                                 squeeze=0,
                                 figsize=(20, len(rows_values) * 5),
                                 gridspec_kw={'width_ratios': self.no_x_values})
        return axes, fig

    def __calculate_stats(self):
        # calculates stats for each time column
        times = {}
        overall_df = self.df_with_time_cols.copy()
        for time in self.__time_columns:
            columns = self.__value_columns.copy()
            columns.append(time)
            df = overall_df[columns]
            grouped_df = df.groupby(time)
            means = grouped_df.mean()
            stds = grouped_df.std()
            counts = grouped_df.count()
            df = pd.concat([means, stds, counts], axis=1, keys=[Aggregators.mean, Aggregators.std, Aggregators.count])
            df = self.__calculate_confidence_intervals(df)
            times[time] = df
        return times

    @staticmethod
    def __calculate_confidence_intervals(stats_df: pd.DataFrame):
        """Calculates lo_ and hi_ci

        :param stats_df:
            format: index = times; columns = multi-index, level one aggregators, level two values
        :return:
            stats_df with ci columns
        """
        means = stats_df[Aggregators.mean]
        stds = stats_df[Aggregators.std]
        counts = stats_df[Aggregators.count]

        hi_ci = means.apply(calculate_hi_ci, args=(stds, counts))
        low_ci = means.apply(calculate_lo_ci, args=(stds, counts))
        ci_df = pd.concat([hi_ci, low_ci], axis=1,
                          keys=[ConfidenceIntervalCols.ci_96hi, ConfidenceIntervalCols.ci_96lo])
        return pd.concat([stats_df, ci_df], axis=1)


def calculate_hi_ci(mean_series, std_df, count_df):
    column = mean_series.name
    return mean_series + 1.96 * std_df[column] / np.sqrt(count_df[column])


def calculate_lo_ci(mean_series, std_df, count_df):
    column = mean_series.name
    return mean_series - 1.96 * std_df[column] / np.sqrt(count_df[column])

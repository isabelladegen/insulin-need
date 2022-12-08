import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt, ticker
from enum import Enum

from src.preprocess import number_of_interval_in_days, continuous_subseries
from src.resample import resample_df, z_score_normalise


class Resolution(Enum):
    DaysMonths = 1
    DaysHours = 2
    Day = 3
    Week = 4


class Cols(str, Enum):
    Mean = 'mean'
    Std = 'std'
    Z_score = 'z-score'
    Min = 'min'
    Max = 'max'


class TimeFeatures(str, Enum):
    day_of_week = 'Day of week'
    hour_of_day = 'Hour'
    month_of_year = 'Month'


cs_std_col_name = 'std'
cs_mean_col_name = 'mean'
cs_z_score_col_name = 'z-score'
months = range(1, 13)
hours = range(0, 24)
days_of_week = range(0, 7)


class ContinuousSeries:

    def __init__(self, df: pd.DataFrame,
                 min_days_of_data: int,
                 max_interval_between_reading: int,
                 time_col: str,
                 value_col: str,
                 resample_rule: str):
        self.df = df[[time_col, value_col]]
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
        self.data_points_resampled = [s.shape[0] for s in self.resampled_series]

    def __resample(self):
        result = []
        for group in self.subseries:
            ts = resample_df(group, self.__resample_rule, self.__time_column, self.__value_column)
            ts = z_score_normalise(ts, (self.__value_column, cs_mean_col_name),
                                   (self.__value_column, cs_z_score_col_name))
            result.append(ts)
        return result

    def describe(self):
        aggregators = self.resampled_series[0].columns
        print(f'Number of continuous series: {len(self.subseries)}')
        print(f'Resampled series aggregators: {aggregators}')
        print(f'Max gap between readings {self.__max_interval_between_readings} minutes')
        print(f'Minimum consecutive days of data is {self.__min_days_of_data}')
        print(f'Resampling rule: {self.__resample_rule}')
        points_in_cont_series = [s.shape[0] for s in self.subseries]
        print(f'Number data points in continuous series: {points_in_cont_series}')
        print(f'Total number of data points in continuous series: {sum(points_in_cont_series)}')
        print(f'Number data points in resampled series: {self.data_points_resampled}')
        print(f'Total number of resampled data points: {sum(self.data_points_resampled)}')

    def plot_resampled_series(self):
        width = 20
        height = 20
        number_of_plots = len(self.resampled_series)

        plt.rcParams.update({'figure.facecolor': 'white', 'figure.dpi': 150, 'font.size': 15})
        fig, axs = plt.subplots(number_of_plots, sharey=True, figsize=(width, height), squeeze=0)

        title = 'Resampled Time Series. Resample rule: ' \
                + self.__resample_rule \
                + ' , min consecutive days of data: ' \
                + str(self.__min_days_of_data)
        fig.suptitle(title)

        for idx, df in enumerate(self.resampled_series):
            # multi index column
            y = df[self.__value_column]
            std = y[cs_std_col_name]
            mean = y[cs_mean_col_name]
            columns_to_plot_as_lines = list(y.columns)
            columns_to_plot_as_lines.remove(cs_std_col_name)
            columns_to_plot_as_lines.remove(cs_mean_col_name)
            columns_to_plot_as_lines.remove(cs_z_score_col_name)
            y = y[columns_to_plot_as_lines]
            axs[idx, 0].plot(df.index, y, marker='o',
                             label=columns_to_plot_as_lines)  # plot with time as index
            axs[idx, 0].errorbar(df.index, mean, yerr=list(std), fmt='-o', capsize=3, label=cs_mean_col_name)
            axs[idx, 0].set_xlabel('')
            axs[idx, 0].yaxis.set_minor_locator(ticker.MultipleLocator(0.5))
            axs[idx, 0].legend().set_visible(False)

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
        fig, axs = plt.subplots(number_of_plots, sharey=True, figsize=(width, height), squeeze=0)

        title = 'Resampled Time Series z-score of mean. Resample rule: ' \
                + self.__resample_rule \
                + ' , min consecutive days of data: ' \
                + str(self.__min_days_of_data)
        fig.suptitle(title)

        for idx, df in enumerate(self.resampled_series):
            # multi index column
            y = df[(self.__value_column, cs_z_score_col_name)]
            axs[idx, 0].plot(df.index, y, marker='o',
                             label=cs_z_score_col_name)  # plot with time as index
            axs[idx, 0].set_xlabel('')
            axs[idx, 0].yaxis.set_minor_locator(ticker.MultipleLocator(0.5))
            axs[idx, 0].legend().set_visible(False)

        lines, labels = fig.axes[-1].get_legend_handles_labels()
        fig.legend(lines, labels)
        fig.supxlabel(self.__time_column)
        fig.supylabel(self.__value_column)
        fig.tight_layout(pad=2)
        plt.show()

    # resample_col= min, max, mean, std, z-score
    def pivot_df_for_day_of_week_month(self, resample_col: str, aggfunc):
        y_axis = TimeFeatures.day_of_week.value
        x_axis = TimeFeatures.month_of_year.value
        data = self.__calculate_time_features()

        return self.__pivot_data(aggfunc, data, resample_col, x_axis, y_axis, months)

    def pivot_df_for_day_of_week_and_hours(self, resample_col: str, aggfunc):
        y_axis = TimeFeatures.day_of_week.value
        x_axis = TimeFeatures.hour_of_day.value
        data = self.__calculate_time_features()

        return self.__pivot_data(aggfunc, data, resample_col, x_axis, y_axis, hours)

    def plot_heatmap_resampled(self, resolution: Resolution, aggfunc=np.mean, resample_col=cs_mean_col_name):
        pivot_function = self.get_pivot_function(resolution)
        pivot = pivot_function(resample_col, aggfunc)
        self.__plot_heatmap(aggfunc, pivot, resample_col)

    def get_pivot_function(self, resolution: Resolution):
        if resolution is Resolution.DaysMonths:
            return getattr(self, 'pivot_df_for_day_of_week_month')
        if resolution is Resolution.DaysHours:
            return getattr(self, 'pivot_df_for_day_of_week_and_hours')

    # just used for tests now
    def as_x_train(self, column: str, length_of_ts=Resolution.Day):
        """Convert resampled ts into 3d ndarray of regular equal length TS.

        Parameters
        ----------
        column : Cols
            Which resample value to use

        length_of_ts : Resolution
            How long the regular TS will be

        Returns
        -------
        numpy array
            X_train of shape=(n_ts, sz, d), where n_ts is number of days or weeks (depending on length_of_ts), sz is 24
            or 7 (depending on length of ts), and d=1
        """
        # Combine all resampled ts into one df
        if length_of_ts == Resolution.Day:
            filtered_df = self.resampled_daily_series_df(column)
            result = filtered_df.to_numpy().reshape(len(np.unique(filtered_df.index.date)), 24, 1)
            return result
        if length_of_ts == Resolution.Week:
            filtered_df = self.resampled_weekly_series_df(column)
            number_of_weeks = len(filtered_df.groupby(by=[filtered_df.index.year, filtered_df.index.week]).count())
            result = filtered_df.to_numpy().reshape(number_of_weeks, 7, 1)
            return result

    def resampled_daily_series_df(self, column):
        """Converts resampled ts into combined df of daily series, only keeping days with a reading per hour

        Parameters
        ----------
        column : str
           Which resample value to use

        Returns
        -------
        pandas Dataframe
            filtered df of combined resampled ts with each date having 24 readings
        """
        df = pd.concat(self.resampled_series).droplevel(level=0, axis=1)
        df.sort_index(inplace=True)
        # Dates that have 24 readings for equal length time periods
        df_for_col = df[column]
        dates = df_for_col.groupby(by=df.index.date).count()
        dates = dates.where(dates == 24).dropna()
        # Drop dates for which we don't have 24 readings
        filtered_df = df_for_col[np.isin(df_for_col.index.date, dates.index)]
        return filtered_df

    def resampled_weekly_series_df(self, column: str):
        """Converts resampled ts into combined df of weekly series, only keeping the weeks with a reading per day

        Parameters
        ----------
        column : str
           Which resample value to use

        Returns
        -------
        pandas Dataframe
            filtered df of combined resampled ts with each date having a daily reading for each weekday
        """
        df = pd.concat(self.resampled_series).droplevel(level=0, axis=1)
        df.sort_index(inplace=True)
        df_for_col = df[column]
        # count how many days of data each week in each year has
        years_weeks = df_for_col.groupby(by=[df.index.year, df.index.isocalendar().week]).count()

        # Drop years_week for which we don't have 7 readings, one per day
        years_weeks = years_weeks.where(years_weeks == 7).dropna()

        # drop the rows where the year/week is not in the years_weeks index
        filtered_df = df_for_col[pd.MultiIndex.from_tuples(list(zip(df.index.year, df.index.isocalendar().week))).isin(
            list(years_weeks.index.to_flat_index()))]
        return filtered_df

    def as_tabular_x_train(self, x: str, y: str, z: str = None):
        """Converts resampled ts into combined df with x, y, z as columns

        Parameters
        ----------


        Returns
        -------
        pandas Dataframe
            df with columns x, y, z
        """
        data = self.__calculate_time_features()
        columns = [x, y]
        if z is not None:
            columns.append(z)
        return data[columns]

    def __calculate_time_features(self):
        data = None
        for df in self.resampled_series:
            new_df = df.copy()
            new_df.columns = new_df.columns.droplevel()
            new_df[TimeFeatures.month_of_year.value] = new_df.index.month
            new_df[TimeFeatures.day_of_week.value] = new_df.index.dayofweek
            new_df[TimeFeatures.hour_of_day.value] = new_df.index.hour

            if data is None:
                data = new_df
            else:
                data = pd.concat([data, new_df])
        return data

    def __plot_heatmap(self, aggfunc, pivot, resample_col):
        plt.rcParams['figure.dpi'] = 150
        ax = sns.heatmap(pivot, linewidth=0.5,
                         yticklabels=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                         square=False,
                         cmap="mako")
        sns.set(font_scale=1)
        title = resample_col + ' ' \
                + self.__value_column \
                + ' aggregated using ' \
                + aggfunc.__name__ \
                + '\n Resample rule: ' \
                + self.__resample_rule \
                + ', min consecutive days of data: ' \
                + str(self.__min_days_of_data)
        ax.set_title(title, pad=10)
        plt.show()

    def __pivot_data(self, aggfunc, data, resample_col, x_axis: str, y_axis: str, expected_columns):
        data = data[[y_axis, resample_col, x_axis]].copy()

        # change to float
        data[resample_col] = data[resample_col].astype(np.float64, copy=False)
        pivot = pd.pivot_table(data=data,
                               index=y_axis,
                               values=resample_col,
                               columns=x_axis,
                               aggfunc=aggfunc)

        # insert expected_columns that are missing in columns
        existing_columns = list(pivot.columns)
        missing_columns = list(set(expected_columns) - set(existing_columns))
        for missing_column in missing_columns:
            pivot[missing_column] = np.NAN

        # insert expected_rows that are missing in rows
        existing_rows = list(pivot.index)
        missing_rows = list(set(days_of_week) - set(existing_rows))
        for missing_row in missing_rows:
            pivot.loc[missing_row] = np.NAN

        # sort columns
        pivot = pivot.reindex(sorted(pivot.columns), axis=1)
        # sort rows
        pivot.sort_index(inplace=True)
        return pivot

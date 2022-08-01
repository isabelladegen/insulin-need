import operator

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from src.configurations import Configuration
from src.continuous_series import ContinuousSeries, Resolution
from src.helper import device_status_file_path_for
from src.read import read_flat_device_status_df_from_file


class MultipleContinuousSeries:
    x_axis_name_lookup = {Resolution.DaysMonths: 'Month',
                          Resolution.DaysHours: 'Hour'
                          }
    pivot_function = {Resolution.DaysMonths: 'pivot_df_for_day_of_week_month',
                      Resolution.DaysHours: 'pivot_df_for_day_of_week_and_hours'
                      }
    weekdays = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    vmins = {'openaps/enacted/IOB': 0.0, 'openaps/enacted/COB': 0.0, 'openaps/enacted/bg': 50.0}

    def __init__(self, zip_ids: [str],
                 min_days_of_data: int,
                 max_interval_between_reading: int,
                 time_col: str,
                 value_columns: [str],
                 resample_rule: str
                 ):
        self.__zip_ids = zip_ids  # ids of multiple people
        self.__time_column = time_col  # column that's used for resampling
        self.__value_columns = value_columns  # columns that will be resampled for each of the ids
        # how frequent readings need per day, 60=every hour, 180=every three hours
        self.__max_interval_between_readings = max_interval_between_reading
        # how many days of consecutive readings with at least a reading every max interval, 7 = a week, 1 = a day
        self.__min_days_of_data = min_days_of_data
        self.__resample_rule = resample_rule  # the frequency of the regular time series after resampling
        self.continuous_series = self.__created_continuous_series()

    def __created_continuous_series(self):
        result_zip = {}
        for zip_id in self.__zip_ids:
            # read file
            file = device_status_file_path_for('../data/perid', zip_id)
            full_df = read_flat_device_status_df_from_file(file, Configuration())

            # create resampled series for each value column
            result_continuous_series = {}
            for value_col in self.__value_columns:
                series = ContinuousSeries(full_df, self.__min_days_of_data, self.__max_interval_between_readings,
                                          self.__time_column, value_col, self.__resample_rule)
                result_continuous_series[value_col] = series

            result_zip[zip_id] = result_continuous_series

        return result_zip

    def plot_heatmaps(self, resolution=Resolution.DaysMonths):
        plt.rcParams['figure.dpi'] = 150
        # grid with rows being the value columns and columns being the different zip ids
        fig, axes = plt.subplots(nrows=len(self.__value_columns),
                                 ncols=len(self.continuous_series),
                                 sharey=True,
                                 sharex=True,squeeze=0,
                                 figsize=(10, 7))
        y_axis = 'Day of week'
        x_axis = MultipleContinuousSeries.x_axis_name_lookup[resolution]

        title = 'Mean values, heatmaps: x=' + x_axis + ' y=Day of Week' \
                + '\n Preproc: Resample rule: ' \
                + self.__resample_rule \
                + ' , min consecutive days of data: ' \
                + str(self.__min_days_of_data) \
                + ', at least on reading every ' \
                + str(self.__max_interval_between_readings) \
                + 'min'
        fig.suptitle(title)
        fig.supxlabel('Different People')
        fig.supylabel('Different Time Series')

        for column_idx, (zip_id, value_columns) in enumerate(self.continuous_series.items()):
            for row_idx, (col_name, series) in enumerate(value_columns.items()):
                pivot_table = getattr(series, MultipleContinuousSeries.pivot_function[resolution])('mean', np.mean)
                vmin = MultipleContinuousSeries.vmins[col_name] if col_name in MultipleContinuousSeries.vmins.keys() \
                    else 0.0
                ax = sns.heatmap(pivot_table,
                                 linewidth=0.5,
                                 yticklabels=MultipleContinuousSeries.weekdays,
                                 square=False,
                                 vmin=vmin,
                                 ax=axes[row_idx][column_idx],
                                 cmap="mako"
                                 )
                ax.set_title('Data points ' + str(sum(series.data_points_resampled)))
                ax.set(xlabel='', ylabel='')

        # column label
        for ax, zip_id in zip(axes[0], self.__zip_ids):
            dp = ax.get_title()
            ax.set_title(zip_id + '\n' + dp)

        # row labels
        for ax, row in zip(axes[:, 0], self.__value_columns):
            ax.set_ylabel(row)
        fig.tight_layout()
        plt.show()

    def plot_total_months_heatmaps(self):
        plt.rcParams['figure.dpi'] = 150
        # grid with rows being the value columns and columns being the different zip ids
        fig, axes = plt.subplots(nrows=len(self.__value_columns),
                                 ncols=len(self.continuous_series),
                                 sharey=True,
                                 sharex=True,
                                 figsize=(10, 7))

        weekdays = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        title = 'TS total mean values per Weekday accross all months' \
                + '\n Resample rule: ' \
                + self.__resample_rule \
                + ' , min consecutive days of data: ' \
                + str(self.__min_days_of_data) \
                + ', at least on reading every ' \
                + str(self.__max_interval_between_readings) \
                + 'min'
        fig.suptitle(title)

        vmins = {'openaps/enacted/IOB': 0.0, 'openaps/enacted/COB': 0.0, 'openaps/enacted/bg': 50.0}
        for column_idx, (zip_id, value_columns) in enumerate(self.continuous_series.items()):
            for row_idx, (col_name, series) in enumerate(value_columns.items()):
                pivot_table = series.pivot_df_for_day_of_week_month('mean', np.mean)
                # sum up all months
                pivot_table["total"] = pivot_table.sum(axis=1)
                pivot_table.loc['Total'] = pivot_table.sum()
                pivot_table.iloc[-1, -1] = np.NaN
                vmin = vmins[col_name] if col_name in vmins.keys() else 0.0
                ax = sns.heatmap(pivot_table,
                                 linewidth=0.5,
                                 yticklabels=weekdays,
                                 square=False,
                                 vmin=vmin,
                                 ax=axes[row_idx][column_idx],
                                 cmap="mako"
                                 )
                ax.set_title('Data points ' + str(sum(series.data_points_resampled)))
                ax.set(xlabel='', ylabel='')

        # column label
        for ax, zip_id in zip(axes[0], self.__zip_ids):
            dp = ax.get_title()
            ax.set_title(zip_id + '\n' + dp)

        # row labels
        for ax, row in zip(axes[:, 0], self.__value_columns):
            ax.set_ylabel(row)

        fig.tight_layout()
        plt.show()

    def as_dictionary_of_x_train(self, column: str):
        """Convert resampled IOB, COB and BG ts into 3d ndarray of regular, equal length TS for each zip id.
            Resulting X_train of shape=(n_ts, sz, d), where n_ts is number of days, sz is 24, and d=3

               Parameters
               ----------
               column : Cols
                   Which resample value to use

        """
        result = {}
        for zip_id, value_columns in self.continuous_series.items():
            # get all three series IOB, COB and BG
            combined_daily_ts_dfs = []
            for series in value_columns.values():
                combined_daily_ts_dfs.append(series.resampled_daily_series_df(column))

            # ensure all three ts have the same dates
            # find dates that are common across the different features
            dates = [set(x.index.date) for x in combined_daily_ts_dfs]
            common_dates = None
            for idx in range(len(dates)-1):
                if common_dates is None:
                    common_dates = dates[idx].intersection(dates[idx+1])
                else:
                    common_dates = common_dates.intersection(dates[idx+1])

            # filter each df to only contain data for the common dates
            filtered_dfs = [df[np.isin(df.index.date, list(common_dates))] for df in combined_daily_ts_dfs]

            # convert into 3d numpy array
            variates = [df.to_numpy().reshape(len(common_dates), 24, 1) for df in filtered_dfs]

            # stack 3d numpy array for adding each variate along third dimension
            result[zip_id] = np.dstack(tuple(variates))
            # reset for next iteration
            combined_daily_ts_dfs = []
        return result

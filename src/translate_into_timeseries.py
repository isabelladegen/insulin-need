from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from src.configurations import GeneralisedCols


@dataclass
class TimeSeriesDescription:
    """ Describes granularity of ts

    """
    length = 0  # max length if variable length allowed
    description = ''
    x_ticks = []
    name = 'Base'


@dataclass
class DailyTimeseries(TimeSeriesDescription):
    """ Class to describe daily time series
    - Df consists of hourly readings per day, max 24 readings
    - each day is a new time series
    """
    length = 24
    description = 'hours of day (UTC)'
    x_ticks = list(range(0, 24, 2))
    name = 'Days'


@dataclass
class WeeklyTimeseries(TimeSeriesDescription):
    """ Class to describe weekly time series
        - Df should have one reading per day for each day of the week
        - each iso calendar week is a new time series
    """
    length = 7
    description = 'Day of Week, 0=Monday'
    x_ticks = list(range(0, 7))
    name = "Weeks"


@dataclass
class IrregularTimeseries(TimeSeriesDescription):
    """ Class to describe irregular time series
        - Df has any number of readings per day or hours with gaps
        - this cannot be used by many methods
    """
    length = None
    description = 'irregular'
    x_ticks = None  # this cannot be used by many methods
    name = "None"


@dataclass
class TimeColumns:  # for additional time features
    day_of_year = "day of year"
    week_of_year = "week of year"
    month = 'months'
    year = 'years'
    week_day = 'weekdays'
    hour = 'hours'


class TranslateIntoTimeseries:
    """Class for turning resampled df into multidimensional of time series that can be used by different algorithms

    - reshapes data into time series Daily or Weekly, e.g. 1d or 3d numpy array
    - reshapes
    - does some further preprocessing for this:
        - drops rows that don't have a reading for each column
        - drops dates with insufficient samples
        - makes dates the index

    Methods
    -------

    """

    def __init__(self, df: pd.DataFrame, ts_description: TimeSeriesDescription, columns: [str]):
        """
            Parameters
            ----------
            df : DataFrame
                Dataframe resampled with sampling - use ReadPreprocessedDataFrame to get the right format of df

            ts_description : TimeSeriesDescription
                DailyTimeseries, WeeklyTimeseries

            columns : Columns to keep to translate into np arrays
        """
        raw_df = df.copy()
        self.ts_description = ts_description
        self.columns = columns
        self.processed_df = self.__preprocess(raw_df)

    """
           Shape of results 2D, eg IOB and COB
           2D(IOB, COB):

            [[[iob_day_1_hour1 cob_day_1_hour1],
              [iob_day_1_hour2 cob_day_1_hour2],
              ...
              [iob_day_1_hour23 cob_day_1_hour23]],
            [[iob_day_2_hour1 cob_day_2_hour1],
             [iob_day_2_hour2 cob_day_2_hour2],
             ...
             [iob_day_2_hour23 cob_day_2_hour23]],
            ...
            [[iob_day_x_hour1 cob_day_x_hour1],
             [iob_day_x_hour2 cob_day_x_hour2],
             ...
             [iob_day_x_hour23
            cob_day_x_hour23]]]
            
             Shape of result 1D:

            [[[day_1_hour1],
              [day_1_hour2],
              ...
              [day_1_hour23]],
             [[day_2_hour1],
              [day_2_hour2],
              [...
               [day_2_hour23]],
              ...
              [[day_x_hour1],
               [day_x_hour2],
               ...
               [day_x_hour23]]]

        """

    def to_x_train(self, cols: [str] = []):
        """Returns resampled regular ts as multidimensional ndarray of the columns provided.

        Parameters
        ----------
        cols : [str]
            Optional parameter if further filtering is required for example to do single variate kmeans.
            When empty all columns initially provided will be added to x_train

        Returns
        -------
        numpy array
            X_train of shape=(n_ts, sz, d), where n_ts is number of time series,
            sz is length of each time series, and d=number of columns
        """
        if isinstance(self.ts_description, IrregularTimeseries):  # no processing into time series
            raise NotImplementedError(
                "This translation does not make sense for time series description of type " + str(
                    type(self.ts_description)))
        if cols:
            columns = cols
        else:
            columns = self.columns
        df = self.processed_df[columns]

        length_of_ts = self.ts_description.length
        number_of_time_series = int(len(df) / length_of_ts)
        number_of_variates = len(columns)

        return df.to_numpy().reshape(
            number_of_time_series,
            length_of_ts,
            number_of_variates
        )

    def to_vectorised_df(self, value_column):
        """Returns df vectorised dataframe of shape X_train of shape=(n_ts, n_features) for variate series_name.
        The features are:
        Daily TS:
            - the value column for each hour of the day and additional time feature columns

        Weekly TS:
            - the value column for each day of the week and additional time feature columns


        Parameters
        ----------
        value_column : str
            which of the multivariate series to return

        Returns
        -------
        dataframe
            X_train of shape=(n_ts, n_features), where n_ts is number of days or weeks (depending on sampling resolution),
            and n_features is each sample in each ts as a feature, plus the special time columns
        """
        # get numpy 1D array that's already shaped with hours resp weekdays as columns and drop 3rd dimension
        array = self.to_x_train([value_column])
        shape = array.shape
        twoDArray = array.reshape(shape[0], shape[1])
        df = pd.DataFrame(twoDArray)

        series_short_name = value_column.split('/')[-1]
        if isinstance(self.ts_description, DailyTimeseries):
            df.columns = [series_short_name + " at " + str(x) for x in df.columns]  # create strings
            time_columns = [TimeColumns.day_of_year, TimeColumns.week_day, TimeColumns.week_of_year, TimeColumns.month,
                            TimeColumns.year]
            cols_for_uniques = [TimeColumns.day_of_year, TimeColumns.year]
        elif isinstance(self.ts_description, WeeklyTimeseries):
            df.columns = [series_short_name + " " + x for x in
                          ["Mon", "Tue", "Wen", "Thu", "Fri", "Sat", "Sun"]]  # create strings
            time_columns = [TimeColumns.week_of_year, TimeColumns.month, TimeColumns.year]
            cols_for_uniques = [TimeColumns.week_of_year, TimeColumns.year]
        else:
            raise NotImplementedError(
                "Don't know how to vectorised provide time series description of type " + str(
                    type(self.ts_description)))

        orig_df = self.to_df_with_time_features()[time_columns]
        reduced_df = orig_df.drop_duplicates(subset=cols_for_uniques, keep='first')
        for column in list(reduced_df.columns):
            df[column] = list(reduced_df[column])
        return df

    def to_continuous_time_series_dfs(self):
        """Returns list of dfs with uninterrupted series of:
        Daily TS:
            - days without a gap inbetween

        Weekly TS:
            - weeks without a gap inbetween

        Returns
        -------
        list of preprocessed dataframe split into continuous series

        """
        df = self.processed_df.copy()
        # calculate gap in minutes between dates
        deltas = self.processed_df.index.to_series().diff().astype('timedelta64[m]').astype('Int64')
        if isinstance(self.ts_description, DailyTimeseries):
            # needs a sample every 60min the series to continue
            # get list of integer indices for all rows where the deltas that are bigger than 60min
            split_at = [deltas.index.get_loc(t) for t in deltas.loc[deltas > 60].index]
        elif isinstance(self.ts_description, WeeklyTimeseries):  # needs a sample for each day of a week
            # needs a sample every 1440min (= every day) for the series to continue
            split_at = [deltas.index.get_loc(t) for t in deltas.loc[deltas > 1440].index]
        elif isinstance(self.ts_description, IrregularTimeseries):  # no processing into time series
            # enforce a sample every 1440min (= every day) for the series to continue
            split_at = [deltas.index.get_loc(t) for t in deltas.loc[deltas > 1440].index]
        else:
            raise NotImplementedError(
                "Don't know how to split df for time series description of type " + str(type(self.ts_description)))
        # split dataframe into those indexes
        return np.split(df, split_at)

    def __preprocess(self, raw_df):
        """ Filters and processes the raw resampled df ready to be translated

        - Sets index to datetime
        - Drops NAs across all supplied columns
        - Drops dates with too few samples

        """
        df = raw_df.set_index(GeneralisedCols.datetime)
        df = df[self.columns]

        # drop rows where any of the columns have a nan -> all variates must have all values for all cols
        df.dropna(inplace=True)

        # remove samples for which we don't have the full time series
        df.sort_index(inplace=True)
        if isinstance(self.ts_description, DailyTimeseries):  # needs a sample for each hour of the day
            # Find all dates that have 24 readings for equal length time periods
            dates = df.groupby(by=df.index.date).count()
            dates = dates.where(dates == self.ts_description.length).dropna()
            # Drop dates for which we don't have 24 readings
            df = df[np.isin(df.index.date, dates.index)]
        elif isinstance(self.ts_description, WeeklyTimeseries):  # needs a sample for each day of a week
            # count how many days of data each week in each year has
            years_weeks = df.groupby(by=[df.index.year, df.index.isocalendar().week]).count()
            # Drop years_week for which we don't have 7 readings, one per day
            years_weeks = years_weeks.where(years_weeks == self.ts_description.length).dropna()
            # drop the rows where the year/week is not in the years_weeks index
            df = df[pd.MultiIndex.from_tuples(list(zip(df.index.year, df.index.isocalendar().week))).isin(
                list(years_weeks.index.to_flat_index()))]
        elif isinstance(self.ts_description, IrregularTimeseries):  # no processing into time series
            df = df
        else:
            raise NotImplementedError(
                "Don't know how to process provide time series description of type " + str(type(self.ts_description)))

        return df

    def to_df_with_time_features(self):
        """Returns processed df with following time features as additional columns: month, day of week, time of day,
        year

        Returns
        -------
        pandas Dataframe
        """
        df = self.processed_df.copy()
        return self.add_time_feature_columns(df)

    @classmethod
    def add_time_feature_columns(cls, df: pd.DataFrame):
        df[TimeColumns.hour] = df.index.hour
        df[TimeColumns.month] = df.index.month
        df[TimeColumns.week_day] = df.index.weekday
        df[TimeColumns.year] = df.index.year
        df[TimeColumns.week_of_year] = df.index.isocalendar().week
        df[TimeColumns.day_of_year] = df.index.day_of_year
        return df

    @classmethod
    def get_longest_df(cls, continuous_time_series_dfs: [pd.DataFrame]):
        """ Method find the the longest continuous time series in a list of dataframes

        Parameters
        ----------
        continuous_time_series_dfs : [pd.DataFrame]
            list of pandas dataframes of shape (time,variates) created by TranslateIntoTimeseries


        Returns
        -------
        dataframe : pd.DataFrame
            longest DataFrame
        """
        dfs_lengths = [df.shape[0] for df in continuous_time_series_dfs]
        index_longest_df = np.argmax(dfs_lengths)
        return continuous_time_series_dfs[index_longest_df]

    @classmethod
    def get_df_including_date(cls, continuous_ts_dfs: [pd.DataFrame], date: datetime):
        """ Method to find result in a list of dfs that includes the date given

           Parameters
           ----------
           continuous_ts_dfs : [pd.DataFrame]
               list of pandas dataframes of shape (time,variates) created by TranslateIntoTimeseries

           date: datetime
               date that needs to be in df

           Returns
           -------
           dataframe : pd.DataFrame
        """
        for frame in continuous_ts_dfs:
            if date.date() in frame.index.date:
                return frame
        return None  # no frame found that includes that date

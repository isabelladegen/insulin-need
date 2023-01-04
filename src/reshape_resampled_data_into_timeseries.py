import numpy as np
import pandas as pd

from src.configurations import GeneralisedCols
from src.multivariate_resampled_series import TimeColumns
from src.stats import TimeSeriesDescription, DailyTimeseries, WeeklyTimeseries


class ReshapeResampledDataIntoTimeseries:
    """Class for turning resampled df into multidimensional numpy arrays of time series

    - reshapes dates into time series
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
            [[iob_day_x_hour1 cob_dayx1_hour1],
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

    def to_vectorised_df(self, series_name):
        """Returns df vectorised dataframe of shape X_train of shape=(n_ts, n_features) for variate series_name

        Parameters
        ----------
        series_name : str
            which of the multivariate series to return

        Returns
        -------
        dataframe
            X_train of shape=(n_ts, n_features), where n_ts is number of days or weeks (depending on sampling resolution),
            and n_features is each sample in each ts as a feature, plus the special time columns
        """
        # get numpy 1D array that's already shaped with hours resp weekdays as columns and drop 3rd dimension
        array = self.to_x_train([series_name])
        shape = array.shape
        twoDArray = array.reshape(shape[0], shape[1])
        df = pd.DataFrame(twoDArray)

        series_short_name = series_name.split('/')[-1]
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

    def __preprocess(self, raw_df):
        """ Filters the raw df ready to be translated into ndarrays

        - Sets index to datetime
        - Drops NAs
        - Drops dates with too few samples

        """
        df = raw_df.set_index(GeneralisedCols.datetime)
        df = df[self.columns]

        # drop rows where any of the columns have a nan -> all variates must have all values for all cols
        df.dropna(inplace=True)

        # remove dates that have fewer readings than required (currently not supporting different length ts
        df.sort_index(inplace=True)
        if isinstance(self.ts_description, DailyTimeseries):
            # Find all dates that have 24 readings for equal length time periods
            dates = df.groupby(by=df.index.date).count()
            dates = dates.where(dates == self.ts_description.length).dropna()
            # Drop dates for which we don't have 24 readings
            df = df[np.isin(df.index.date, dates.index)]
        elif isinstance(self.ts_description, WeeklyTimeseries):
            # count how many days of data each week in each year has
            years_weeks = df.groupby(by=[df.index.year, df.index.isocalendar().week]).count()
            # Drop years_week for which we don't have 7 readings, one per day
            years_weeks = years_weeks.where(years_weeks == self.ts_description.length).dropna()
            # drop the rows where the year/week is not in the years_weeks index
            df = df[pd.MultiIndex.from_tuples(list(zip(df.index.year, df.index.isocalendar().week))).isin(
                list(years_weeks.index.to_flat_index()))]
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
        df[TimeColumns.hour] = df.index.hour
        df[TimeColumns.month] = df.index.month
        df[TimeColumns.week_day] = df.index.weekday
        df[TimeColumns.year] = df.index.year
        df[TimeColumns.week_of_year] = df.index.isocalendar().week
        df[TimeColumns.day_of_year] = df.index.day_of_year
        return df

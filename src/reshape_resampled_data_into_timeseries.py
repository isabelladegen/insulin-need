import numpy as np
import pandas as pd

from src.configurations import GeneralisedCols
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

    def to_x_train(self):
        """Returns resampled regular ts as multidimensional ndarray of the columns provided.

        Returns
        -------
        numpy array
            X_train of shape=(n_ts, sz, d), where n_ts is number of time series,
            sz is length of each time series, and d=number of columns
        """
        length_of_ts = self.ts_description.length
        number_of_time_series = int(len(self.processed_df) / length_of_ts)
        number_of_variates = len(self.columns)
        return self.processed_df.to_numpy().reshape(
            number_of_time_series,
            length_of_ts,
            number_of_variates
        )

    def get_vectorised_df(self, series_name):
        """Returns df vectorised into rows being n_ts and columns being features n_ts, and special time columns

        Parameters
        ----------
        series_name : str
            which of the multivariate series to return, IOB, COB or BG (use sampling class for proper name)

        Returns
        -------
        dataframe
            X_train of shape=(n_ts, n_features), where n_ts is number of days or weeks (depending on sampling resolution),
            and n_features is each sample in each ts as a feature, plus the special time columns
        """

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

from dataclasses import dataclass

from src.configurations import Configuration
from src.continuous_series import ContinuousSeries, Resolution, Cols
from src.helper import device_status_file_path_for
from src.read import read_flat_device_status_df_from_file
from src.stats import Sampling


@dataclass
class TimeColumns:  # Daily TS
    month = 'month'
    year = 'year'
    week_day = 'weekday'
    hour = 'hours'


# Sadly this reads real data from file which probably is a mistake, but helpful for the notebooks
class MultivariateResampledSeries:
    __multivariate_df = None
    __multivariate_df_with_time_cols = None
    __multivariate_raw_df_with_time_cols = None
    __multivariate_nparray = None
    __iob_nparray = None
    __cob_nparray = None
    __bg_nparray = None

    def __init__(self, zip_id: str, resample_value: Cols, sampling: Sampling, keep_raw_df=False):
        self.zip_id = zip_id
        self.resample_value = resample_value
        self.sampling = sampling

        # read data for zip id into full_df
        file = device_status_file_path_for(Configuration().perid_data_folder, zip_id)
        full_df = read_flat_device_status_df_from_file(file, Configuration())

        if keep_raw_df:
            raw_df = full_df[[sampling.time_col, sampling.iob_col, sampling.cob_col, sampling.bg_col]]
            raw_df = raw_df.dropna()  # not interested in data where one of these columns is missing
            # raw data, not resampled, but enforced to have a reading for all columns
            raw_df = raw_df.drop_duplicates()
            self.raw_df = raw_df.set_index([sampling.time_col])

        # create all series
        self.iob_series = ContinuousSeries(full_df,
                                           self.sampling.min_days_of_data,
                                           self.sampling.max_interval,
                                           self.sampling.time_col,
                                           self.sampling.iob_col,
                                           self.sampling.sample_rule)
        self.cob_series = ContinuousSeries(full_df,
                                           self.sampling.min_days_of_data,
                                           self.sampling.max_interval,
                                           self.sampling.time_col,
                                           self.sampling.cob_col,
                                           self.sampling.sample_rule)
        self.bg_series = ContinuousSeries(full_df,
                                          self.sampling.min_days_of_data,
                                          self.sampling.max_interval,
                                          self.sampling.time_col,
                                          self.sampling.bg_col,
                                          self.sampling.sample_rule)

        # create combined dataframes for each series
        if sampling.resolution == Resolution.Day:  # every day has 24 hours
            self.iob_df = self.iob_series.resampled_daily_series_df(self.resample_value)
            self.cob_df = self.cob_series.resampled_daily_series_df(self.resample_value)
            self.bg_df = self.bg_series.resampled_daily_series_df(self.resample_value)
        if sampling.resolution == Resolution.Week:  # every week has 7 days
            self.iob_df = self.iob_series.resampled_weekly_series_df(self.resample_value)
            self.cob_df = self.cob_series.resampled_weekly_series_df(self.resample_value)
            self.bg_df = self.bg_series.resampled_weekly_series_df(self.resample_value)

    def get_multivariate_df(self):
        """Converts data into regular ts based on Sampling rule into combined df with IOB, COB and BG as columns, only
        keeping values that have IOB, COB and BG for that date

        Returns
        -------
        pandas Dataframe
           df with columns mean IOB, COB, BG indexed by time
        """
        if self.__multivariate_df is None:
            # combine df into one df only keeping data for time stamps with all three values
            df = self.iob_df.rename(self.sampling.iob_col).to_frame()
            df = df.merge(self.cob_df.rename(self.sampling.cob_col), left_index=True, right_index=True)
            df = df.merge(self.bg_df.rename(self.sampling.bg_col), left_index=True, right_index=True)
            self.__multivariate_df = df
        return self.__multivariate_df

    def get_multivariate_df_with_special_time_columns(self):
        """Returns df of IOB, COB and BG with month, day of week, time of day (only sensible with hourly sampling),
        year added

        Returns
        -------
        pandas Dataframe
           df with columns mean IOB, COB, BG, month, day of week, time of day, year added indexed by time
        """
        if self.__multivariate_df_with_time_cols is None:
            df = self.get_multivariate_df().copy()
            df[TimeColumns.hour] = df.index.hour
            df[TimeColumns.month] = df.index.month
            df[TimeColumns.week_day] = df.index.weekday
            df[TimeColumns.year] = df.index.year
            self.__multivariate_df_with_time_cols = df
        return self.__multivariate_df_with_time_cols

    def get_time_column_from_raw_data_for(self, time_format: TimeColumns):
        """Returns a one column raw df with time formatted to time_format. Raw df is unsampled with nan's removed
         and deduplicated for IOB, COB and BG

        Parameters
        ----------
        time_format: TimeColumns
            what time interpretation to calculate from time index

        Returns
        -------
        pandas Dataframe
           df with column time_format
        """
        if time_format == TimeColumns.month:
            return self.raw_df.index.month
        if time_format == TimeColumns.year:
            return self.raw_df.index.year
        if time_format == TimeColumns.week_day:
            return self.raw_df.index.weekday
        if time_format == TimeColumns.hour:
            return self.raw_df.index.hour
        return None

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

    """

    def get_multivariate_3d_numpy_array(self):
        """Returns resampled regular ts as 3d ndarray of IOB, COB and BG

        Returns
        -------
        numpy array
            X_train of shape=(n_ts, sz, d), where n_ts is number of days or weeks (depending on sampling resolution),
            sz is 24 or 7 (depending on sampling length of ts), and d=3 as IOB, COB and BG
        """
        if self.__multivariate_nparray is None:
            df = self.get_multivariate_df()
            self.__multivariate_nparray = df.to_numpy().reshape(int(len(df) / self.sampling.length),
                                                                self.sampling.length,
                                                                3)
        return self.__multivariate_nparray

    """
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

    def get_1d_numpy_array(self, series_name):
        """Returns resampled regular ts as 1d ndarray

        Parameters
        ----------
        series_name : str
            which of the multivariate series to return, IOB, COB or BG (use sampling class for proper name)

        Returns
        -------
        numpy array
            X_train of shape=(n_ts, sz, d), where n_ts is number of days or weeks (depending on sampling resolution),
            sz is 24 or 7 (depending on sampling length of ts), and d=1
        """
        df = self.get_multivariate_df()
        return df[series_name].to_numpy().reshape(int(len(df) / self.sampling.length), self.sampling.length, 1)

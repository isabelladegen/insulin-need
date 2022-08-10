from src.configurations import Configuration
from src.continuous_series import ContinuousSeries, Resolution, Cols
from src.helper import device_status_file_path_for
from src.read import read_flat_device_status_df_from_file
from src.stats import Sampling


class MultivariateResampledSeries:
    multivariate_df = None
    multivariate_nparray = None
    iob_nparray = None
    cob_nparray = None
    bg_nparray = None

    def __init__(self, zip_id: str, resample_value: Cols, sampling: Sampling):
        self.zip_id = zip_id
        self.resample_value = resample_value
        self.sampling = sampling

        # read data for zip id into full_df
        file = device_status_file_path_for(Configuration().perid_data_folder, zip_id)
        self.full_df = read_flat_device_status_df_from_file(file, Configuration())

        # create all series
        self.iob_series = ContinuousSeries(self.full_df,
                                           self.sampling.min_days_of_data,
                                           self.sampling.max_interval,
                                           self.sampling.time_col,
                                           self.sampling.iob_col,
                                           self.sampling.sample_rule)
        self.cob_series = ContinuousSeries(self.full_df,
                                           self.sampling.min_days_of_data,
                                           self.sampling.max_interval,
                                           self.sampling.time_col,
                                           self.sampling.cob_col,
                                           self.sampling.sample_rule)
        self.bg_series = ContinuousSeries(self.full_df,
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
        if self.multivariate_df is None:
            # combine df into one df only keeping data for time stamps with all three values
            df = self.iob_df.rename(self.sampling.iob_col).to_frame()
            df = df.merge(self.cob_df.rename(self.sampling.cob_col), left_index=True, right_index=True)
            df = df.merge(self.bg_df.rename(self.sampling.bg_col), left_index=True, right_index=True)
            self.multivariate_df = df
        return self.multivariate_df

    def get_multivariate_3d_numpy_array(self):
        """Returns resampled regular ts as 3d ndarray of IOB, COB and BG

        Returns
        -------
        numpy array
            X_train of shape=(n_ts, sz, d), where n_ts is number of days or weeks (depending on sampling resolution),
            sz is 24 or 7 (depending on sampling length of ts), and d=3 as IOB, COB and BG
        """
        if self.multivariate_nparray is None:
            df = self.get_multivariate_df()
            self.multivariate_nparray = df.to_numpy().reshape(int(len(df) / self.sampling.length), self.sampling.length,
                                                              3)
        return self.multivariate_nparray

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

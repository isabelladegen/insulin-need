from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal

import numpy as np
import pandas as pd

from src.configurations import Resampling, GeneralisedCols, Configuration


class ResampleDataFrame:
    """Class for resampling irregular dataframes into regular ones
        Methods
        -------
    """

    def __init__(self, irregular_df: pd.DataFrame):
        """
            Parameters
            ----------
            irregular_df : DataFrame
                Irregular sampled df
        """
        self.__df = irregular_df.sort_values(by=GeneralisedCols.datetime)
        self.__config = Configuration()  # this could become a parameter if config is required like that

    def resample_to(self, sampling: Resampling):
        """
            Parameters
            ----------
            sampling : Resampling
                What the time series needs to be resampled to
        """
        cols_to_resample = self.__config.value_columns_to_resample()
        columns = self.__config.info_columns() + self.__config.resampled_value_columns() + self.__config.resampling_count_columns()

        # resample by value column to avoid resampling over missing values in some of the value columns
        resulting_df = None
        for column in cols_to_resample:
            sub_columns = self.__config.info_columns() + [column]
            sub_df = self.__df[sub_columns].copy()  # df with only one value column
            sub_df = sub_df.dropna()  # to ensure we don't sample over missing values
            if sub_df.shape[0] == 0:
                continue
            # calculate minutes interval between non nan samples for each interval (day or hour) and only keep
            # days/hours where the interval is smaller than the max allowed gap
            if sampling.needs_max_gap_checking:
                if sampling.sample_rule == '1D':
                    if len(set(sub_df[GeneralisedCols.datetime].dt.date)) == 1:  # only one day of data
                        result = sub_df[GeneralisedCols.datetime].diff().astype('timedelta64[m]')
                    else:
                        result = sub_df.groupby(by=sub_df[GeneralisedCols.datetime].dt.date,
                                                   group_keys=True).apply(
                            lambda x: x[GeneralisedCols.datetime].diff().astype('timedelta64[m]'))
                    sub_df['diff'] = result.reset_index(level=0, drop=True)
                    # days with bigger gaps than max
                    bigger_gaps_dates = set(
                        sub_df.loc[sub_df['diff'] > sampling.max_gap_in_min][GeneralisedCols.datetime].dt.date)
                    df_right_max_gaps = sub_df[~(sub_df[GeneralisedCols.datetime].dt.date).isin(bigger_gaps_dates)]

                    # For each date left we need to calculate the gap between the last/first timestamp
                    # of the day/hour and the next/previous day/hour and drop that date if it is bigger than 180
                    last_datetimestamps = list(
                        df_right_max_gaps.groupby(df_right_max_gaps[GeneralisedCols.datetime].dt.date).last()[
                            GeneralisedCols.datetime])
                    first_datetimestamps = list(
                        df_right_max_gaps.groupby(df_right_max_gaps[GeneralisedCols.datetime].dt.date).first()[
                            GeneralisedCols.datetime])
                    latest_time_each_date = [t.replace(hour=23, minute=59, second=59) for t in last_datetimestamps]
                    earliest_time_each_date = [t.replace(hour=0, minute=0, second=0) for t in last_datetimestamps]
                    last_or_first_time_interval_too_big = []
                    for idx, last_available_t in enumerate(last_datetimestamps):
                        min_to_midnight = (latest_time_each_date[idx] - last_available_t).total_seconds() / 60.0
                        if min_to_midnight > sampling.max_gap_in_min:
                            last_or_first_time_interval_too_big.append(last_available_t.date())

                    for idx, first_available_t in enumerate(first_datetimestamps):
                        min_to_first_timestamp = (first_available_t - earliest_time_each_date[
                            idx]).total_seconds() / 60.0
                        if min_to_first_timestamp > sampling.max_gap_in_min:
                            last_or_first_time_interval_too_big.append(first_available_t.date())

                    # only keep dates that don't have a last time stamp that's more than max interval to midnight away
                    df_right_max_gaps = df_right_max_gaps[
                        ~(df_right_max_gaps[GeneralisedCols.datetime].dt.date).isin(
                            set(last_or_first_time_interval_too_big))]

                    sub_df = df_right_max_gaps.drop(['diff'], axis=1)
                else:
                    raise NotImplementedError

            # resample
            sub_df = sub_df.set_index([GeneralisedCols.datetime])
            agg_dict = dict(sampling.general_agg_cols_dictionary)
            agg_dict[column] = sampling.agg_cols
            resampled_df = sub_df.resample(sampling.sample_rule).agg(agg_dict)

            if resampled_df.shape[0] == 0:
                continue

            if resulting_df is None:
                resulting_df = resampled_df
            else:
                resulting_df = resulting_df.combine_first(resampled_df)

        if resulting_df is None:
            return pd.DataFrame(columns=columns)

        # ensure columns are as expected
        resulting_df.columns = resulting_df.columns.to_flat_index()
        resulting_df.columns = [' '.join(col) if col[1] != 'first' else col[0] for col in
                                resulting_df.columns.values]
        resulting_df.reset_index(inplace=True)

        # add na columns for columns that don't exist
        missing_columns = list(set(self.__config.resampled_value_columns()).difference(list(resulting_df.columns)))
        resulting_df[missing_columns] = np.NaN

        # drop entries that are just there for the counts
        resulting_df = resulting_df.dropna(subset=self.__config.resampled_value_columns(), how='all')

        # round numbers to 3 decimal places
        resulting_df[GeneralisedCols.mean_iob] = resulting_df[GeneralisedCols.mean_iob].apply(self.__round_numbers)
        resulting_df[GeneralisedCols.mean_cob] = resulting_df[GeneralisedCols.mean_cob].apply(self.__round_numbers)
        resulting_df[GeneralisedCols.mean_bg] = resulting_df[GeneralisedCols.mean_bg].apply(self.__round_numbers)

        resulting_df[GeneralisedCols.max_iob] = resulting_df[GeneralisedCols.max_iob].apply(self.__round_numbers)
        resulting_df[GeneralisedCols.max_cob] = resulting_df[GeneralisedCols.max_cob].apply(self.__round_numbers)
        resulting_df[GeneralisedCols.max_bg] = resulting_df[GeneralisedCols.max_bg].apply(self.__round_numbers)

        resulting_df[GeneralisedCols.min_iob] = resulting_df[GeneralisedCols.min_iob].apply(self.__round_numbers)
        resulting_df[GeneralisedCols.min_cob] = resulting_df[GeneralisedCols.min_cob].apply(self.__round_numbers)
        resulting_df[GeneralisedCols.min_bg] = resulting_df[GeneralisedCols.min_bg].apply(self.__round_numbers)

        resulting_df[GeneralisedCols.std_iob] = resulting_df[GeneralisedCols.std_iob].apply(self.__round_numbers)
        resulting_df[GeneralisedCols.std_cob] = resulting_df[GeneralisedCols.std_cob].apply(self.__round_numbers)
        resulting_df[GeneralisedCols.std_bg] = resulting_df[GeneralisedCols.std_bg].apply(self.__round_numbers)

        # add missing columns
        missing_columns = list(set(columns) - set(resulting_df.columns))
        resulting_df[missing_columns] = None

        # replace na with 0 for count columns
        count_columns = self.__config.resampling_count_columns()
        resulting_df[count_columns] = resulting_df[count_columns].fillna(0)

        # reorder columns
        return resulting_df.loc[:, columns]

    @staticmethod
    def __round_numbers(x):
        if np.isnan(x):
            return x
        return float(Decimal(str(x)).quantize(Decimal('.100'), rounding=ROUND_HALF_UP))

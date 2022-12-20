from decimal import ROUND_HALF_UP, Decimal

import numpy as np
import pandas as pd

from src.configurations import Resampling, GeneralisedCols
from src.preprocess import group_into_consecutive_intervals


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
        self.__all_value_columns = [GeneralisedCols.mean_iob.value,
                                    GeneralisedCols.mean_cob.value,
                                    GeneralisedCols.mean_bg.value,
                                    GeneralisedCols.min_iob.value,
                                    GeneralisedCols.min_cob.value,
                                    GeneralisedCols.min_bg.value,
                                    GeneralisedCols.max_iob.value,
                                    GeneralisedCols.max_cob.value,
                                    GeneralisedCols.max_bg.value,
                                    GeneralisedCols.std_iob.value,
                                    GeneralisedCols.std_cob.value,
                                    GeneralisedCols.std_bg.value,
                                    ]

    def resample_to(self, sampling: Resampling):
        """
            Parameters
            ----------
            sampling : Resampling
                What the time series needs to be resampled to
        """
        cols_to_resample = [GeneralisedCols.iob.value, GeneralisedCols.cob.value, GeneralisedCols.bg.value]

        # resample by value column to avoid resampling over missing values in some of the value columns
        resulting_df = None
        for column in cols_to_resample:
            df = self.__df[[GeneralisedCols.datetime, GeneralisedCols.id, column, GeneralisedCols.system]].copy()
            df = df.dropna()  # to ensure we don't sample over missing values
            # calculate minutes interval between samples for each interval (day or hour)
            result = self.__df.groupby(by=self.__df[GeneralisedCols.datetime].dt.date, group_keys=True).apply(
                lambda x: x[GeneralisedCols.datetime].diff().astype('timedelta64[m]'))
            df['diff'] = result.reset_index(level=0, drop=True)
            # days with bigger gaps than max
            bigger_gaps_dates = set(df.loc[df['diff'] > sampling.max_gap_in_min][GeneralisedCols.datetime].dt.date)

            # subframe with only dates with sufficient samples
            rows_with_sufficient_samples = df[~(df[GeneralisedCols.datetime].dt.date).isin(bigger_gaps_dates)]

            # need to also drop the dates that have less than 8 entries
            infrequent_dates = (rows_with_sufficient_samples[GeneralisedCols.datetime].dt.date.value_counts() < 8)
            dates_with_too_few_samples = set(infrequent_dates.where(infrequent_dates).dropna().index.values)

            rows_with_sufficient_samples = rows_with_sufficient_samples[
                ~(rows_with_sufficient_samples[GeneralisedCols.datetime].dt.date).isin(dates_with_too_few_samples)]

            df = rows_with_sufficient_samples.drop(['diff'], axis=1)
            df = df.set_index([GeneralisedCols.datetime.value])
            agg_dict = dict(sampling.general_agg_cols_dictionary)
            agg_dict[column] = sampling.agg_cols
            resampled_df = df.resample(sampling.sample_rule).agg(agg_dict)
            resampled_df = resampled_df.dropna(how='all')

            if resampled_df.shape[0] is 0:
                continue

            if resulting_df is None:
                resulting_df = resampled_df
            else:
                resulting_df = resulting_df.combine_first(resampled_df)

        resampled_df = resulting_df

        # ensure columns are as expected
        resampled_df.columns = resampled_df.columns.to_flat_index()
        resampled_df.columns = [' '.join(col) if col[1] != 'first' else col[0] for col in
                                resampled_df.columns.values]
        resampled_df.reset_index(inplace=True)

        # add na columns for columns that don't exist
        missing_columns = list(set(self.__all_value_columns).difference(list(resampled_df.columns)))
        resampled_df[missing_columns] = np.NaN

        # round numbers to 3 decimal places
        resampled_df[GeneralisedCols.mean_iob] = resampled_df[GeneralisedCols.mean_iob].apply(self.__round_numbers)
        resampled_df[GeneralisedCols.mean_cob] = resampled_df[GeneralisedCols.mean_cob].apply(self.__round_numbers)
        resampled_df[GeneralisedCols.mean_bg] = resampled_df[GeneralisedCols.mean_bg].apply(self.__round_numbers)

        resampled_df[GeneralisedCols.max_iob] = resampled_df[GeneralisedCols.max_iob].apply(self.__round_numbers)
        resampled_df[GeneralisedCols.max_cob] = resampled_df[GeneralisedCols.max_cob].apply(self.__round_numbers)
        resampled_df[GeneralisedCols.max_bg] = resampled_df[GeneralisedCols.max_bg].apply(self.__round_numbers)

        resampled_df[GeneralisedCols.min_iob] = resampled_df[GeneralisedCols.min_iob].apply(self.__round_numbers)
        resampled_df[GeneralisedCols.min_cob] = resampled_df[GeneralisedCols.min_cob].apply(self.__round_numbers)
        resampled_df[GeneralisedCols.min_bg] = resampled_df[GeneralisedCols.min_bg].apply(self.__round_numbers)

        resampled_df[GeneralisedCols.std_iob] = resampled_df[GeneralisedCols.std_iob].apply(self.__round_numbers)
        resampled_df[GeneralisedCols.std_cob] = resampled_df[GeneralisedCols.std_cob].apply(self.__round_numbers)
        resampled_df[GeneralisedCols.std_bg] = resampled_df[GeneralisedCols.std_bg].apply(self.__round_numbers)
        return resampled_df

    @staticmethod
    def __round_numbers(x):
        if np.isnan(x):
            return x
        return float(Decimal(str(x)).quantize(Decimal('.100'), rounding=ROUND_HALF_UP))

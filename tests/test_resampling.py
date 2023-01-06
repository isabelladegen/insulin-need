import os
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP

import numpy as np
import pandas as pd
import pytest
from hamcrest import *
from numpy import nan

from src.configurations import GeneralisedCols, Daily, Hourly, Configuration, Irregular
from src.read_preprocessed_df import ReadPreprocessedDataFrame
from src.resampling import ResampleDataFrame

# test data:
# day1 with a sample exactly every three hour
# day2 without a sample every three hour -> should not be included in mean
# day3 more than one sample every three our

# day 1
d10 = datetime(year=2019, month=1, day=10, hour=0, minute=5, tzinfo=timezone.utc)
d11 = datetime(year=2019, month=1, day=10, hour=3, minute=5, tzinfo=timezone.utc)
d12 = datetime(year=2019, month=1, day=10, hour=6, minute=5, tzinfo=timezone.utc)
d13 = datetime(year=2019, month=1, day=10, hour=9, minute=5, tzinfo=timezone.utc)
d14 = datetime(year=2019, month=1, day=10, hour=12, minute=5, tzinfo=timezone.utc)
d15 = datetime(year=2019, month=1, day=10, hour=15, minute=5, tzinfo=timezone.utc)
d16 = datetime(year=2019, month=1, day=10, hour=18, minute=5, tzinfo=timezone.utc)
d17 = datetime(year=2019, month=1, day=10, hour=21, minute=5, tzinfo=timezone.utc)

# day 2
d20 = datetime(year=2019, month=1, day=11, hour=0, minute=5, tzinfo=timezone.utc)
d21 = datetime(year=2019, month=1, day=11, hour=3, minute=6, tzinfo=timezone.utc)  # misses the every 3h by 1 min
d22 = datetime(year=2019, month=1, day=11, hour=6, minute=5, tzinfo=timezone.utc)
d23 = datetime(year=2019, month=1, day=11, hour=9, minute=5, tzinfo=timezone.utc)
d24 = datetime(year=2019, month=1, day=11, hour=12, minute=5, tzinfo=timezone.utc)
d25 = datetime(year=2019, month=1, day=11, hour=15, minute=5, tzinfo=timezone.utc)
d26 = datetime(year=2019, month=1, day=11, hour=18, minute=5, tzinfo=timezone.utc)
d27 = datetime(year=2019, month=1, day=11, hour=21, minute=5, tzinfo=timezone.utc)

# day 3
d30 = datetime(year=2019, month=1, day=12, hour=0, minute=5, tzinfo=timezone.utc)
d301 = datetime(year=2019, month=1, day=12, hour=1, minute=5, tzinfo=timezone.utc)
d31 = datetime(year=2019, month=1, day=12, hour=3, minute=5, tzinfo=timezone.utc)
d311 = datetime(year=2019, month=1, day=12, hour=4, minute=5, tzinfo=timezone.utc)
d32 = datetime(year=2019, month=1, day=12, hour=6, minute=5, tzinfo=timezone.utc)
d321 = datetime(year=2019, month=1, day=12, hour=6, minute=45, tzinfo=timezone.utc)
d33 = datetime(year=2019, month=1, day=12, hour=9, minute=5, tzinfo=timezone.utc)
d34 = datetime(year=2019, month=1, day=12, hour=12, minute=5, tzinfo=timezone.utc)
d35 = datetime(year=2019, month=1, day=12, hour=15, minute=5, tzinfo=timezone.utc)
d36 = datetime(year=2019, month=1, day=12, hour=18, minute=5, tzinfo=timezone.utc)
d37 = datetime(year=2019, month=1, day=12, hour=21, minute=5, tzinfo=timezone.utc)

zip_id1 = '555'
no_samples = 27
iob = 5
additional_iob = 2
additional_iob2 = 1
cob = 1
additional_cob1 = 3
additional_cob2 = 3.5
bg = 6.4
additional_bg1 = 15.6
additional_bg2 = 2.2
irregular_data = {GeneralisedCols.datetime: [d10, d11, d12, d13, d14, d15, d16, d17,
                                                   d20, d21, d22, d23, d24, d25, d26, d27,
                                                   d30, d301, d31, d311, d32, d321, d33, d34, d35, d36, d37],
                  GeneralisedCols.iob: [iob] * 16 + [iob, additional_iob, iob, additional_iob2, iob,
                                                           additional_iob] + [iob] * 5,
                  GeneralisedCols.cob: [cob] * 16 + [cob, additional_cob1, cob, additional_cob2, cob,
                                                           additional_cob1] + [cob] * 5,  # same value for each time
                  GeneralisedCols.bg: [bg] * 16 + [bg, additional_bg1, bg, additional_bg2, bg,
                                                         additional_bg1] + [bg] * 5,
                  GeneralisedCols.system: ['bla'] * no_samples,
                  GeneralisedCols.id: [zip_id1] * no_samples
                  }
irregular_df = pd.DataFrame(irregular_data)
irregular_df[GeneralisedCols.datetime] = pd.to_datetime(irregular_df[GeneralisedCols.datetime], utc=True,
                                                        errors="raise")
irregular_df[GeneralisedCols.id] = irregular_df[GeneralisedCols.id].astype(str)


def test_turns_irregular_df_into_daily_sampled_min_max_mean_std_df_for_downsample_to_daily_values():
    resampling = ResampleDataFrame(irregular_df)

    df = resampling.resample_to(Daily())
    assert_that(df[GeneralisedCols.datetime].dt.day, has_item(10))
    assert_that(df[GeneralisedCols.datetime].dt.day, has_item(12))
    assert_that(df.shape[0], is_(2))
    assert_that(df[GeneralisedCols.id].unique(), is_(zip_id1))
    assert_that(df[GeneralisedCols.system].unique(), is_('bla'))

    day1 = df.loc[df[GeneralisedCols.datetime].dt.day == 10]
    day3 = df.loc[df[GeneralisedCols.datetime].dt.day == 12]
    # check mean values
    assert_that(day1[GeneralisedCols.mean_iob].values[0], is_(iob))
    assert_that(day3[GeneralisedCols.mean_iob].values[0], is_(4.091))
    assert_that(day1[GeneralisedCols.mean_cob].values[0], is_(cob))
    assert_that(day3[GeneralisedCols.mean_cob].values[0], is_(1.591))
    assert_that(day1[GeneralisedCols.mean_bg].values[0], is_(bg))
    assert_that(day3[GeneralisedCols.mean_bg].values[0], is_(7.691))

    # check min values
    assert_that(day1[GeneralisedCols.min_iob].values[0], is_(iob))
    assert_that(day3[GeneralisedCols.min_iob].values[0], is_(additional_iob2))
    assert_that(day1[GeneralisedCols.min_cob].values[0], is_(cob))
    assert_that(day3[GeneralisedCols.min_cob].values[0], is_(cob))
    assert_that(day1[GeneralisedCols.min_bg].values[0], is_(bg))
    assert_that(day3[GeneralisedCols.min_bg].values[0], is_(additional_bg2))

    # check max values
    assert_that(day1[GeneralisedCols.max_iob].values[0], is_(iob))
    assert_that(day3[GeneralisedCols.max_iob].values[0], is_(iob))
    assert_that(day1[GeneralisedCols.max_cob].values[0], is_(cob))
    assert_that(day3[GeneralisedCols.max_cob].values[0], is_(additional_cob2))
    assert_that(day1[GeneralisedCols.max_bg].values[0], is_(bg))
    assert_that(day3[GeneralisedCols.max_bg].values[0], is_(additional_bg1))

    # check std values
    assert_that(day1[GeneralisedCols.std_iob].values[0], is_(0))
    assert_that(day3[GeneralisedCols.std_iob].values[0], is_(1.578))
    assert_that(day1[GeneralisedCols.std_cob].values[0], is_(0))
    assert_that(day3[GeneralisedCols.std_cob].values[0], is_(1.020))
    assert_that(day1[GeneralisedCols.std_bg].values[0], is_(0))
    assert_that(day3[GeneralisedCols.std_bg].values[0], is_(4.106))


def test_ensures_sampling_is_frequent_enough_for_each_variate_for_daily_resampling():
    just_iob = {GeneralisedCols.datetime: [d10, d11, d12, d13, d14, d15, d16, d17,
                                                 d20, d21, d22, d23, d24, d25, d26, d27
                                                 ],
                GeneralisedCols.iob: [iob] * 16,  # sufficient values
                GeneralisedCols.cob: [cob] * 3 + [nan] * 5 + [cob] * 8,  # not enough values for day 1 and 2
                GeneralisedCols.bg: [bg] * 8 + [nan] * 5 + [bg] * 3,  # not enough values for day 2
                GeneralisedCols.system: ['bla'] * 16,
                GeneralisedCols.id: [zip_id1] * 16
                }
    just_iob_df = pd.DataFrame(just_iob)
    just_iob_df[GeneralisedCols.datetime] = pd.to_datetime(just_iob_df[GeneralisedCols.datetime], utc=True,
                                                           errors="raise")
    resampling = ResampleDataFrame(just_iob_df)

    df = resampling.resample_to(Daily())
    assert_that(df.shape[0], is_(1))

    day1 = df.loc[df[GeneralisedCols.datetime].dt.day == 10]
    assert_that(day1[GeneralisedCols.mean_iob].values[0], is_(iob))
    assert_that(np.isnan(day1[GeneralisedCols.mean_cob].values[0]))
    assert_that(day1[GeneralisedCols.mean_bg].values[0], is_(bg))

    # check min values
    assert_that(day1[GeneralisedCols.min_iob].values[0], is_(iob))
    assert_that(np.isnan(day1[GeneralisedCols.min_cob].values[0]))
    assert_that(day1[GeneralisedCols.min_bg].values[0], is_(bg))

    # check max values
    assert_that(day1[GeneralisedCols.max_iob].values[0], is_(iob))
    assert_that(np.isnan(day1[GeneralisedCols.max_cob].values[0]))
    assert_that(day1[GeneralisedCols.max_bg].values[0], is_(bg))

    # check std values
    assert_that(day1[GeneralisedCols.std_iob].values[0], is_(0))
    assert_that(np.isnan(day1[GeneralisedCols.std_cob].values[0]))
    assert_that(day1[GeneralisedCols.std_bg].values[0], is_(0))


def test_turns_irregular_df_into_regular_hourly_sampled_df():
    # setup data
    no_reading_t = datetime(year=2019, month=1, day=10, hour=1, minute=5, tzinfo=timezone.utc)
    beginning_t = datetime(year=2019, month=1, day=11, hour=5, minute=0, tzinfo=timezone.utc)
    end_t = datetime(year=2019, month=1, day=12, hour=23, minute=59, second=59, tzinfo=timezone.utc)
    two_middle_t1 = datetime(year=2019, month=1, day=13, hour=12, minute=23, tzinfo=timezone.utc)
    two_middle_t2 = datetime(year=2019, month=1, day=13, hour=12, minute=45, tzinfo=timezone.utc)
    no_times = 5
    iob1 = 6.6
    iob2 = 2.4
    cob1 = 15
    cob2 = 40
    bg1 = 5.5
    bg2 = 16.4

    data = {GeneralisedCols.datetime: [no_reading_t, beginning_t, end_t, two_middle_t1, two_middle_t2],
            GeneralisedCols.iob: [np.NaN, iob1, iob2, iob1, iob2],
            GeneralisedCols.cob: [np.NaN, cob1, np.NaN, np.NaN, cob2],
            GeneralisedCols.bg: [np.NaN, np.NaN, bg2, bg1, bg2],
            GeneralisedCols.system: ['ttte'] * no_times,
            GeneralisedCols.id: [zip_id1] * no_times
            }
    df_irregular = pd.DataFrame(data)
    df_irregular[GeneralisedCols.datetime] = pd.to_datetime(df_irregular[GeneralisedCols.datetime], utc=True,
                                                            errors="raise")

    # resample
    df = ResampleDataFrame(df_irregular).resample_to(Hourly())

    # check general results
    times = df[GeneralisedCols.datetime].dt.hour
    assert_that(times, has_item(beginning_t.hour))
    assert_that(times, has_item(end_t.hour))
    assert_that(times, has_item(two_middle_t2.hour))
    assert_that(df.shape[0], is_(3))  # 3 hours with sufficient data
    assert_that(df[GeneralisedCols.id].unique(), is_(zip_id1))
    assert_that(df[GeneralisedCols.system].unique(), is_('ttte'))

    beginning_row = df.loc[times == beginning_t.hour]
    end_row = df.loc[times == end_t.hour]
    two_middle_row = df.loc[times == two_middle_t1.hour]

    # check counts
    assert_that(beginning_row[GeneralisedCols.count_iob].values[0], is_(1))
    assert_that(end_row[GeneralisedCols.count_iob].values[0], is_(1))
    assert_that(two_middle_row[GeneralisedCols.count_iob].values[0], is_(2))

    assert_that(beginning_row[GeneralisedCols.count_cob].values[0], is_(1))
    assert_that(end_row[GeneralisedCols.count_cob].values[0], is_(0))
    assert_that(two_middle_row[GeneralisedCols.count_cob].values[0], is_(1))

    assert_that(beginning_row[GeneralisedCols.count_bg].values[0], is_(0))
    assert_that(end_row[GeneralisedCols.count_bg].values[0], is_(1))
    assert_that(two_middle_row[GeneralisedCols.count_bg].values[0], is_(2))

    # check mean values
    assert_that(beginning_row[GeneralisedCols.mean_iob].values[0], is_(iob1))
    assert_that(end_row[GeneralisedCols.mean_iob].values[0], is_(iob2))
    assert_that(two_middle_row[GeneralisedCols.mean_iob].values[0], is_(4.500))
    assert_that(beginning_row[GeneralisedCols.mean_cob].values[0], is_(cob1))
    assert_that(np.isnan(end_row[GeneralisedCols.mean_cob].values[0]))
    assert_that(two_middle_row[GeneralisedCols.mean_cob].values[0], is_(cob2))
    assert_that(np.isnan(beginning_row[GeneralisedCols.mean_bg].values[0]))
    assert_that(end_row[GeneralisedCols.mean_bg].values[0], is_(bg2))
    assert_that(two_middle_row[GeneralisedCols.mean_bg].values[0], is_(10.95))

    # check min values
    assert_that(beginning_row[GeneralisedCols.min_iob].values[0], is_(iob1))
    assert_that(end_row[GeneralisedCols.min_iob].values[0], is_(iob2))
    assert_that(two_middle_row[GeneralisedCols.min_iob].values[0], is_(iob2))
    assert_that(beginning_row[GeneralisedCols.min_cob].values[0], is_(cob1))
    assert_that(np.isnan(end_row[GeneralisedCols.min_cob].values[0]))
    assert_that(two_middle_row[GeneralisedCols.min_cob].values[0], is_(cob2))
    assert_that(np.isnan(beginning_row[GeneralisedCols.min_bg].values[0]))
    assert_that(end_row[GeneralisedCols.min_bg].values[0], is_(bg2))
    assert_that(two_middle_row[GeneralisedCols.min_bg].values[0], is_(bg1))

    # check max values
    assert_that(beginning_row[GeneralisedCols.max_iob].values[0], is_(iob1))
    assert_that(end_row[GeneralisedCols.max_iob].values[0], is_(iob2))
    assert_that(two_middle_row[GeneralisedCols.max_iob].values[0], is_(iob1))
    assert_that(beginning_row[GeneralisedCols.max_cob].values[0], is_(cob1))
    assert_that(np.isnan(end_row[GeneralisedCols.max_cob].values[0]))
    assert_that(two_middle_row[GeneralisedCols.max_cob].values[0], is_(cob2))
    assert_that(np.isnan(beginning_row[GeneralisedCols.max_bg].values[0]))
    assert_that(end_row[GeneralisedCols.max_bg].values[0], is_(bg2))
    assert_that(two_middle_row[GeneralisedCols.max_bg].values[0], is_(bg2))

    # check std values
    assert_that(np.isnan(beginning_row[GeneralisedCols.std_iob].values[0]))
    assert_that(np.isnan(end_row[GeneralisedCols.std_iob].values[0]))  # std of a single value is undefined
    assert_that(two_middle_row[GeneralisedCols.std_iob].values[0], is_(2.970))
    assert_that(np.isnan(beginning_row[GeneralisedCols.std_cob].values[0]))
    assert_that(np.isnan(end_row[GeneralisedCols.std_cob].values[0]))
    assert_that(np.isnan(two_middle_row[GeneralisedCols.std_cob].values[0]))
    assert_that(np.isnan(beginning_row[GeneralisedCols.std_bg].values[0]))
    assert_that(np.isnan(end_row[GeneralisedCols.std_bg].values[0]))
    assert_that(two_middle_row[GeneralisedCols.std_bg].values[0], is_(7.707))


def test_can_deal_with_empty_dataframe():
    nan_list = [np.NaN] * 3
    t1 = datetime(year=2019, month=1, day=10, hour=1, minute=5, tzinfo=timezone.utc)
    t2 = datetime(year=2019, month=1, day=10, hour=2, minute=5, tzinfo=timezone.utc)
    t3 = datetime(year=2019, month=1, day=10, hour=3, minute=5, tzinfo=timezone.utc)
    empty_data = {GeneralisedCols.datetime: [t1, t2, t3],
                  GeneralisedCols.iob: nan_list,
                  GeneralisedCols.cob: nan_list,
                  GeneralisedCols.bg: nan_list,
                  GeneralisedCols.system: nan_list,
                  GeneralisedCols.id: nan_list
                  }
    df_irregular = pd.DataFrame(empty_data)
    df = ResampleDataFrame(df_irregular).resample_to(Daily())

    config = Configuration()
    expected_number_of_columns = len(config.info_columns()) + len(config.resampled_value_columns()) + len(
        config.resampling_count_columns())
    assert_that(df.shape, is_((0, expected_number_of_columns)))


def test_returns_none_value_columns_first_for_empty_df():
    nan_list = [np.NaN]
    t1 = datetime(year=2019, month=1, day=10, hour=1, minute=5, tzinfo=timezone.utc)
    empty_data = {GeneralisedCols.datetime: [t1],
                  GeneralisedCols.iob: nan_list,
                  GeneralisedCols.cob: nan_list,
                  GeneralisedCols.bg: nan_list,
                  GeneralisedCols.system: nan_list,
                  GeneralisedCols.id: nan_list
                  }
    df_irregular = pd.DataFrame(empty_data)
    df = ResampleDataFrame(df_irregular).resample_to(Daily())

    config = Configuration()
    columns = list(df.columns)
    expected_info_columns = config.info_columns()
    assert_that(columns[0], is_(expected_info_columns[0]))  # datetime first
    assert_that(columns[1], is_(expected_info_columns[1]))  # id second
    assert_that(columns[2], is_(expected_info_columns[2]))  # system third


def test_returns_none_value_columns_first():
    values = [10.0]
    t1 = datetime(year=2019, month=1, day=10, hour=1, minute=5, tzinfo=timezone.utc)
    empty_data = {GeneralisedCols.datetime: [t1],
                  GeneralisedCols.iob: values,
                  GeneralisedCols.cob: values,
                  GeneralisedCols.bg: values,
                  GeneralisedCols.system: values,
                  GeneralisedCols.id: values
                  }
    df_irregular = pd.DataFrame(empty_data)
    df = ResampleDataFrame(df_irregular).resample_to(Hourly())

    config = Configuration()
    columns = list(df.columns)
    expected_info_columns = config.info_columns()
    assert_that(columns[0], is_(expected_info_columns[0]))  # datetime first
    assert_that(columns[1], is_(expected_info_columns[1]))  # id second
    assert_that(columns[2], is_(expected_info_columns[2]))  # system third


def test_can_deal_with_single_day_of_data():
    values = [10.0]*8
    t1 = datetime(year=2019, month=1, day=10, hour=1, minute=5, tzinfo=timezone.utc)
    t2 = datetime(year=2019, month=1, day=10, hour=4, minute=5, tzinfo=timezone.utc)
    t3 = datetime(year=2019, month=1, day=10, hour=7, minute=5, tzinfo=timezone.utc)
    t4 = datetime(year=2019, month=1, day=10, hour=10, minute=5, tzinfo=timezone.utc)
    t5 = datetime(year=2019, month=1, day=10, hour=13, minute=5, tzinfo=timezone.utc)
    t6 = datetime(year=2019, month=1, day=10, hour=15, minute=5, tzinfo=timezone.utc)
    t7 = datetime(year=2019, month=1, day=10, hour=18, minute=5, tzinfo=timezone.utc)
    t8 = datetime(year=2019, month=1, day=10, hour=21, minute=5, tzinfo=timezone.utc)
    data = {GeneralisedCols.datetime: [t1, t2, t3, t4, t5, t6, t7, t8],
                  GeneralisedCols.iob: values,
                  GeneralisedCols.cob: values,
                  GeneralisedCols.bg: values,
                  GeneralisedCols.system: values,
                  GeneralisedCols.id: values
                  }
    df_irregular = pd.DataFrame(data)
    df = ResampleDataFrame(df_irregular).resample_to(Daily())

    config = Configuration()
    columns = config.info_columns() + config.resampled_value_columns() + config.resampling_count_columns()
    assert_that(df.shape[0], is_(1))
    assert_that(list(df.columns), is_(columns))


def test_counts_as_integer_types():
    values = [5.5, 10.9, np.NaN]
    t1 = datetime(year=2019, month=1, day=10, hour=1, minute=5, tzinfo=timezone.utc)
    t2 = datetime(year=2019, month=1, day=10, hour=1, minute=55, tzinfo=timezone.utc)
    t3 = datetime(year=2019, month=1, day=10, hour=2, minute=5, tzinfo=timezone.utc)
    empty_data = {GeneralisedCols.datetime: [t1, t2, t3],
                  GeneralisedCols.iob: values,
                  GeneralisedCols.cob: [np.NaN, np.NaN, np.NaN],
                  GeneralisedCols.bg: [np.NaN, np.NaN, np.NaN],
                  GeneralisedCols.system: values,
                  GeneralisedCols.id: values
                  }
    df_irregular = pd.DataFrame(empty_data)
    df = ResampleDataFrame(df_irregular).resample_to(Hourly())

    config = Configuration()
    types_of_count_columns = set(df[config.resampling_count_columns()].dtypes)
    assert_that(len(types_of_count_columns), is_(1))
    assert_that(str(list(types_of_count_columns)[0]), is_('int64'))


def test_some_rounding_values():
    numbers = [1.4556, 1.4, 0.55546, 0.5335]
    expect_rounded = [1.456, 1.400, 0.555, 0.534]
    rounded = [float(Decimal(str(x)).quantize(Decimal('.100'), rounding=ROUND_HALF_UP)) for x in numbers]

    assert_that(rounded[0], is_(expect_rounded[0]))
    assert_that(rounded[1], is_(expect_rounded[1]))
    assert_that(rounded[2], is_(expect_rounded[2]))
    assert_that(rounded[3], is_(expect_rounded[3]))


@pytest.mark.skipif(not os.path.isdir(Configuration().perid_data_folder), reason="reads real data")
def test_resampling_on_real_data():
    test_id = '49796612'
    df = ReadPreprocessedDataFrame(sampling=Irregular(), zip_id=test_id).df

    resampled_df = ResampleDataFrame(df).resample_to(Daily())

    assert_that(resampled_df, is_(not_none()))

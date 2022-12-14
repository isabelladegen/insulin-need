import os
from datetime import datetime, timezone, timedelta

import pandas as pd
import pytest
from hamcrest import *
from pandas import DatetimeTZDtype

from src.configurations import Configuration, Daily, Hourly, GeneralisedCols, Irregular
from src.translate_into_timeseries import TranslateIntoTimeseries, TimeColumns, DailyTimeseries, WeeklyTimeseries, \
    IrregularTimeseries
from src.read_preprocessed_df import ReadPreprocessedDataFrame

# test data:
# day1 sample every hour
# day2 on hour missing in the day
# day3 sample every hour

# times for day 1
d10 = datetime(year=2019, month=1, day=10, hour=0, minute=0, tzinfo=timezone.utc)
d11 = datetime(year=2019, month=1, day=10, hour=1, minute=0, tzinfo=timezone.utc)
d12 = datetime(year=2019, month=1, day=10, hour=2, minute=0, tzinfo=timezone.utc)
d13 = datetime(year=2019, month=1, day=10, hour=3, minute=0, tzinfo=timezone.utc)
d14 = datetime(year=2019, month=1, day=10, hour=4, minute=0, tzinfo=timezone.utc)
d15 = datetime(year=2019, month=1, day=10, hour=5, minute=0, tzinfo=timezone.utc)
d16 = datetime(year=2019, month=1, day=10, hour=6, minute=0, tzinfo=timezone.utc)
d17 = datetime(year=2019, month=1, day=10, hour=7, minute=0, tzinfo=timezone.utc)
d18 = datetime(year=2019, month=1, day=10, hour=8, minute=0, tzinfo=timezone.utc)
d19 = datetime(year=2019, month=1, day=10, hour=9, minute=0, tzinfo=timezone.utc)
d110 = datetime(year=2019, month=1, day=10, hour=10, minute=0, tzinfo=timezone.utc)
d111 = datetime(year=2019, month=1, day=10, hour=11, minute=0, tzinfo=timezone.utc)
d112 = datetime(year=2019, month=1, day=10, hour=12, minute=0, tzinfo=timezone.utc)
d113 = datetime(year=2019, month=1, day=10, hour=13, minute=0, tzinfo=timezone.utc)
d114 = datetime(year=2019, month=1, day=10, hour=14, minute=0, tzinfo=timezone.utc)
d115 = datetime(year=2019, month=1, day=10, hour=15, minute=0, tzinfo=timezone.utc)
d116 = datetime(year=2019, month=1, day=10, hour=16, minute=0, tzinfo=timezone.utc)
d117 = datetime(year=2019, month=1, day=10, hour=17, minute=0, tzinfo=timezone.utc)
d118 = datetime(year=2019, month=1, day=10, hour=18, minute=0, tzinfo=timezone.utc)
d119 = datetime(year=2019, month=1, day=10, hour=19, minute=0, tzinfo=timezone.utc)
d120 = datetime(year=2019, month=1, day=10, hour=20, minute=0, tzinfo=timezone.utc)
d121 = datetime(year=2019, month=1, day=10, hour=21, minute=0, tzinfo=timezone.utc)
d122 = datetime(year=2019, month=1, day=10, hour=22, minute=0, tzinfo=timezone.utc)
d123 = datetime(year=2019, month=1, day=10, hour=23, minute=0, tzinfo=timezone.utc)

# times for day 2 - misses 0
d21 = datetime(year=2019, month=1, day=11, hour=1, minute=0, tzinfo=timezone.utc)
d22 = datetime(year=2019, month=1, day=11, hour=2, minute=0, tzinfo=timezone.utc)
d23 = datetime(year=2019, month=1, day=11, hour=3, minute=0, tzinfo=timezone.utc)
d24 = datetime(year=2019, month=1, day=11, hour=4, minute=0, tzinfo=timezone.utc)
d25 = datetime(year=2019, month=1, day=11, hour=5, minute=0, tzinfo=timezone.utc)
d26 = datetime(year=2019, month=1, day=11, hour=6, minute=0, tzinfo=timezone.utc)
d27 = datetime(year=2019, month=1, day=11, hour=7, minute=0, tzinfo=timezone.utc)
d28 = datetime(year=2019, month=1, day=11, hour=8, minute=0, tzinfo=timezone.utc)
d29 = datetime(year=2019, month=1, day=11, hour=9, minute=0, tzinfo=timezone.utc)
d210 = datetime(year=2019, month=1, day=11, hour=10, minute=0, tzinfo=timezone.utc)
d211 = datetime(year=2019, month=1, day=11, hour=11, minute=0, tzinfo=timezone.utc)
d212 = datetime(year=2019, month=1, day=11, hour=12, minute=0, tzinfo=timezone.utc)
d213 = datetime(year=2019, month=1, day=11, hour=13, minute=0, tzinfo=timezone.utc)
d214 = datetime(year=2019, month=1, day=11, hour=14, minute=0, tzinfo=timezone.utc)
d215 = datetime(year=2019, month=1, day=11, hour=15, minute=0, tzinfo=timezone.utc)
d216 = datetime(year=2019, month=1, day=11, hour=16, minute=0, tzinfo=timezone.utc)
d217 = datetime(year=2019, month=1, day=11, hour=17, minute=0, tzinfo=timezone.utc)
d218 = datetime(year=2019, month=1, day=11, hour=18, minute=0, tzinfo=timezone.utc)
d219 = datetime(year=2019, month=1, day=11, hour=19, minute=0, tzinfo=timezone.utc)
d220 = datetime(year=2019, month=1, day=11, hour=20, minute=0, tzinfo=timezone.utc)
d221 = datetime(year=2019, month=1, day=11, hour=21, minute=0, tzinfo=timezone.utc)
d222 = datetime(year=2019, month=1, day=11, hour=22, minute=0, tzinfo=timezone.utc)
d223 = datetime(year=2019, month=1, day=11, hour=23, minute=0, tzinfo=timezone.utc)

# times for day 3 - has all
d30 = datetime(year=2021, month=2, day=12, hour=0, minute=0, tzinfo=timezone.utc)
d31 = datetime(year=2021, month=2, day=12, hour=1, minute=0, tzinfo=timezone.utc)
d32 = datetime(year=2021, month=2, day=12, hour=2, minute=0, tzinfo=timezone.utc)
d33 = datetime(year=2021, month=2, day=12, hour=3, minute=0, tzinfo=timezone.utc)
d34 = datetime(year=2021, month=2, day=12, hour=4, minute=0, tzinfo=timezone.utc)
d35 = datetime(year=2021, month=2, day=12, hour=5, minute=0, tzinfo=timezone.utc)
d36 = datetime(year=2021, month=2, day=12, hour=6, minute=0, tzinfo=timezone.utc)
d37 = datetime(year=2021, month=2, day=12, hour=7, minute=0, tzinfo=timezone.utc)
d38 = datetime(year=2021, month=2, day=12, hour=8, minute=0, tzinfo=timezone.utc)
d39 = datetime(year=2021, month=2, day=12, hour=9, minute=0, tzinfo=timezone.utc)
d310 = datetime(year=2021, month=2, day=12, hour=10, minute=0, tzinfo=timezone.utc)
d311 = datetime(year=2021, month=2, day=12, hour=11, minute=0, tzinfo=timezone.utc)
d312 = datetime(year=2021, month=2, day=12, hour=12, minute=0, tzinfo=timezone.utc)
d313 = datetime(year=2021, month=2, day=12, hour=13, minute=0, tzinfo=timezone.utc)
d314 = datetime(year=2021, month=2, day=12, hour=14, minute=0, tzinfo=timezone.utc)
d315 = datetime(year=2021, month=2, day=12, hour=15, minute=0, tzinfo=timezone.utc)
d316 = datetime(year=2021, month=2, day=12, hour=16, minute=0, tzinfo=timezone.utc)
d317 = datetime(year=2021, month=2, day=12, hour=17, minute=0, tzinfo=timezone.utc)
d318 = datetime(year=2021, month=2, day=12, hour=18, minute=0, tzinfo=timezone.utc)
d319 = datetime(year=2021, month=2, day=12, hour=19, minute=0, tzinfo=timezone.utc)
d320 = datetime(year=2021, month=2, day=12, hour=20, minute=0, tzinfo=timezone.utc)
d321 = datetime(year=2021, month=2, day=12, hour=21, minute=0, tzinfo=timezone.utc)
d322 = datetime(year=2021, month=2, day=12, hour=22, minute=0, tzinfo=timezone.utc)
d323 = datetime(year=2021, month=2, day=12, hour=23, minute=0, tzinfo=timezone.utc)

zip_id1 = '555'
iob = 2.4
iob2 = 7
iob3 = 8
cob = 15
cob2 = 20
cob3 = 5
bg = 5.8
bg2 = 12.3
bg3 = 4.2
no_samples = 24 + 23 + 24
day1_times = [d10, d11, d12, d13, d14, d15, d16, d17, d18, d19, d110, d111, d112, d113, d114, d115,
              d116, d117, d118, d119, d120, d121, d122, d123]
hourly_data = {GeneralisedCols.datetime: [d10, d11, d12, d13, d14, d15, d16, d17,
                                                d18, d19, d110, d111, d112, d113, d114, d115,
                                                d116, d117, d118, d119, d120, d121, d122, d123,
                                                d21, d22, d23, d24, d25, d26, d27, d28,
                                                d29, d210, d211, d212, d213, d214, d215, d216,
                                                d217, d218, d219, d220, d221, d222, d223, d30,
                                                d31, d32, d33, d34, d35, d36, d37, d38,
                                                d39, d310, d311, d312, d313, d314, d315, d316,
                                                d317, d318, d319, d320, d321, d322, d323,
                                                ],
               GeneralisedCols.mean_iob: [iob] * 24 + [iob2] * 23 + [iob3] * 24,
               GeneralisedCols.mean_cob: [cob] * 24 + [cob2] * 23 + [cob3] * 24,  # same value for each time
               GeneralisedCols.mean_bg: [bg] * 24 + [bg2] * 23 + [bg3] * 24,
               GeneralisedCols.system: ['bla'] * no_samples,
               GeneralisedCols.id: [zip_id1] * no_samples
               }
hourly_df = pd.DataFrame(hourly_data)
hourly_df[GeneralisedCols.datetime] = pd.to_datetime(hourly_df[GeneralisedCols.datetime], utc=True,
                                                     errors="raise")
hourly_df[GeneralisedCols.id] = hourly_df[GeneralisedCols.id].astype(str)

hourly = Hourly()
daily = Daily()
mean_cols = Configuration.resampled_mean_columns()
daily_ts = DailyTimeseries()
weekly_ts = WeeklyTimeseries()


def test_only_keeps_columns_provided():
    translate = TranslateIntoTimeseries(hourly_df, daily_ts, mean_cols)

    df = translate.processed_df

    assert_that(list(df.columns), only_contains(mean_cols[0], mean_cols[1], mean_cols[2]))


def test_drops_days_with_insufficient_samples_for_daily_time_series():
    translate = TranslateIntoTimeseries(hourly_df, daily_ts, mean_cols)

    df = translate.processed_df

    dates_in_df = list(set(df.index.date))
    dates_in_df.sort()
    assert_that(dates_in_df, contains_exactly(d10.date(), d30.date()))


def test_sets_datetime_col_as_dateindex_on_df():
    translate = TranslateIntoTimeseries(hourly_df, daily_ts, mean_cols)

    assert_that(translate.processed_df.index.dtype, is_(DatetimeTZDtype('ns', 'UTC')))


def test_drops_nan_rows_for_cols():
    # drop one bg value on day 3 to remove that day from the final df
    df_with_nan = hourly_df.copy()
    df_with_nan.at[df_with_nan.index[-1], GeneralisedCols.mean_bg] = None
    translate = TranslateIntoTimeseries(df_with_nan, daily_ts, mean_cols)

    df = translate.processed_df

    dates_in_df = list(set(df.index.date))
    dates_in_df.sort()
    assert_that(dates_in_df, contains_exactly(d10.date()))


def test_returns_df_with_additional_time_feature_columns():
    translate = TranslateIntoTimeseries(hourly_df, daily_ts, mean_cols)

    df = translate.to_df_with_time_features()

    actual_columns = list(df.columns)
    assert_that(actual_columns, has_items(TimeColumns.hour, TimeColumns.week_day, TimeColumns.month, TimeColumns.year,
                                          TimeColumns.week_of_year, TimeColumns.day_of_year))


def test_translates_df_into_3d_nparray_of_daily_ts_if_three_cols_provided():
    translate = TranslateIntoTimeseries(hourly_df, daily_ts, mean_cols)

    result = translate.to_x_train()

    assert_that(result.shape, is_((2, 24, 3)))  # no days, no samples per day, no variates


def test_translates_df_into_2d_nparray_of_daily_ts_if_two_cols_provided():
    translate = TranslateIntoTimeseries(hourly_df, daily_ts,
                                        [GeneralisedCols.mean_iob, GeneralisedCols.mean_bg])

    result = translate.to_x_train()

    assert_that(result.shape, is_((2, 24, 2)))


def test_translates_df_into_dataframe_with_time_features():
    translate = TranslateIntoTimeseries(hourly_df, daily_ts, mean_cols)

    result = translate.to_vectorised_df(GeneralisedCols.mean_iob)

    number_of_days = 2
    number_of_features = 24 + 5
    assert_that(result.shape, is_((number_of_days, number_of_features)))
    assert_that(result[TimeColumns.week_day][0], is_(d10.weekday()))
    assert_that(result[TimeColumns.week_day][1], is_(d30.weekday()))
    assert_that(result[TimeColumns.week_of_year][0], is_(d10.isocalendar().week))
    assert_that(result[TimeColumns.week_of_year][1], is_(d30.isocalendar().week))
    assert_that(result[TimeColumns.month][0], is_(d10.month))
    assert_that(result[TimeColumns.month][1], is_(d30.month))
    assert_that(result[TimeColumns.day_of_year][0], is_(d10.timetuple().tm_yday))
    assert_that(result[TimeColumns.day_of_year][1], is_(d30.timetuple().tm_yday))
    assert_that(result[TimeColumns.year][0], is_(d10.year))
    assert_that(result[TimeColumns.year][1], is_(d30.year))


def test_splits_preprocessed_df_into_dfs_of_continuous_days_of_daily_time_series():
    # build more test data
    day2_times = [d + timedelta(days=1) for d in day1_times]
    day3_times = [d + timedelta(days=1) for d in day2_times]
    day4_times = [d + timedelta(days=1) for d in day3_times]  # 4 continuous days
    day6_times = [d + timedelta(days=2) for d in day4_times]  # add a gap and just one day
    day8_times = [d + timedelta(days=2) for d in day6_times]  # two days
    day9_times = [d + timedelta(days=1) for d in day8_times]
    data = {
        GeneralisedCols.datetime: day1_times + day2_times + day3_times + day4_times + day6_times + day8_times + day9_times,
        GeneralisedCols.mean_iob: [iob] * 24 + [iob2] * 24 + [iob] * 24 + [iob2] * 24 + [iob3] * 24 + [
            iob] * 24 + [iob3] * 24,
        GeneralisedCols.mean_cob: [cob] * 24 + [cob2] * 24 + [cob] * 24 + [cob2] * 24 + [cob3] * 24 + [
            cob] * 24 + [cob3] * 24,
        GeneralisedCols.mean_bg: [bg] * 24 + [bg2] * 24 + [bg] * 24 + [bg2] * 24 + [bg3] * 24 + [
            bg] * 24 + [bg3] * 24,
        GeneralisedCols.system: ['bla'] * 7 * 24,
        GeneralisedCols.id: [zip_id1] * 7 * 24
    }
    data_df = pd.DataFrame(data)
    data_df[GeneralisedCols.datetime] = pd.to_datetime(data_df[GeneralisedCols.datetime], utc=True, errors="raise")
    data_df[GeneralisedCols.id] = data_df[GeneralisedCols.id].astype(str)

    translate = TranslateIntoTimeseries(data_df, daily_ts, mean_cols)
    results = translate.to_continuous_time_series_dfs()

    assert_that(len(results), is_(3))
    first_series = results[0]
    assert_that(first_series.shape[0], is_(4 * 24))  # four days with each 24 samples
    assert_that(list(set(first_series.index.date)),
                has_items(day1_times[0].date(), day2_times[0].date(), day3_times[0].date(), day4_times[0].date()))

    second_series = results[1]
    assert_that(second_series.shape[0], is_(1 * 24))  # one day with each 24 samples
    assert_that(list(set(second_series.index.date)), has_items(day6_times[0].date()))

    third_series = results[2]
    assert_that(third_series.shape[0], is_(2 * 24))  # two days with each 24 samples
    assert_that(list(set(third_series.index.date)), has_items(day8_times[0].date(), day9_times[0].date()))


@pytest.mark.skipif(not os.path.isdir(Configuration().perid_data_folder), reason="reads real data")
def test_calculates_resampled_3d_nparray_of_daily_time_series():
    # read already sampled data
    df = ReadPreprocessedDataFrame(sampling=Hourly(), zip_id='14092221').df

    translate = TranslateIntoTimeseries(df, daily_ts, mean_cols)
    result = translate.to_x_train()

    assert_that(result.shape, is_((376, 24, 3)))


@pytest.mark.skipif(not os.path.isdir(Configuration().perid_data_folder), reason="reads real data")
def test_calculates_resampled_3d_and_1d_nparray_of_daily_time_series():
    df = ReadPreprocessedDataFrame(sampling=Hourly(), zip_id='13484299').df

    translate = TranslateIntoTimeseries(df, daily_ts, mean_cols)

    x_3d = translate.to_x_train()
    x_1d = translate.to_x_train(cols=[GeneralisedCols.mean_cob])

    assert_that(x_3d.shape, is_((30, 24, 3)))
    assert_that(x_1d.shape, is_((30, 24, 1)))


@pytest.mark.skipif(not os.path.isdir(Configuration().perid_data_folder), reason="reads real data")
def test_calculates_resampled_3d_nparray_of_weekly_time_series():
    # read already sampled data
    df = ReadPreprocessedDataFrame(sampling=Daily(), zip_id='14092221').df

    translate = TranslateIntoTimeseries(df, weekly_ts, mean_cols)
    result = translate.to_x_train()

    assert_that(result.shape, is_((50, 7, 3)))


@pytest.mark.skipif(not os.path.isdir(Configuration().perid_data_folder), reason="reads real data")
def test_returns_vectorised_df_daily_sampling():
    raw_df = ReadPreprocessedDataFrame(sampling=Hourly(), zip_id='14092221').df
    translate = TranslateIntoTimeseries(raw_df, daily_ts, mean_cols)

    df = translate.to_vectorised_df(GeneralisedCols.mean_iob)

    number_of_days = 376
    number_of_features = 24 + 5
    assert_that(df.shape, is_((number_of_days, number_of_features)))


@pytest.mark.skipif(not os.path.isdir(Configuration().perid_data_folder), reason="reads real data")
def test_returns_vectorised_df_weekly_sampling():
    raw_df = ReadPreprocessedDataFrame(sampling=Daily(), zip_id='14092221').df
    translate = TranslateIntoTimeseries(raw_df, weekly_ts, mean_cols)

    df = translate.to_vectorised_df(GeneralisedCols.mean_iob)

    number_of_days = 50
    number_of_features = 7 + 3
    assert_that(df.shape, is_((number_of_days, number_of_features)))


@pytest.mark.skipif(not os.path.isdir(Configuration().perid_data_folder), reason="reads real data")
def test_returns_list_of_continuous_weekly_time_series_for_daily_sampling():
    raw_df = ReadPreprocessedDataFrame(sampling=Daily(), zip_id='14092221').df
    translate = TranslateIntoTimeseries(raw_df, WeeklyTimeseries(), mean_cols)

    results = translate.to_continuous_time_series_dfs()

    assert_that(len(results), is_(11))


@pytest.mark.skipif(not os.path.isdir(Configuration().perid_data_folder), reason="reads real data")
def test_returns_list_of_continuous_daily_time_series_for_hourly_sampling():
    raw_df = ReadPreprocessedDataFrame(sampling=Hourly(), zip_id='14092221').df
    translate = TranslateIntoTimeseries(raw_df, daily_ts, mean_cols)

    results = translate.to_continuous_time_series_dfs()

    assert_that(len(results), is_(83))


@pytest.mark.skipif(not os.path.isdir(Configuration().perid_data_folder), reason="reads real data")
def test_works_for_irregular_sampled_and_irregular_time_series():
    raw_df = ReadPreprocessedDataFrame(sampling=Irregular(), zip_id='14092221').df
    translate = TranslateIntoTimeseries(raw_df, IrregularTimeseries(),
                                        [GeneralisedCols.iob, GeneralisedCols.cob, GeneralisedCols.bg])

    processed_df = translate.processed_df
    assert_that(processed_df.shape[1], is_(3))

    continuous_dfs = translate.to_continuous_time_series_dfs()
    assert_that(len(continuous_dfs), is_(6))
    assert_that(TranslateIntoTimeseries.get_longest_df(continuous_dfs).shape[0], is_(30898))
    date = datetime(year=2018, month=9, day=29, tzinfo=timezone.utc)
    assert_that(TranslateIntoTimeseries.get_df_including_date(continuous_dfs, date).shape[0], is_(20064))

    time_features = translate.to_df_with_time_features()
    assert_that(time_features.shape[1], is_(9))

    try:
        translate.to_x_train()
        raise RuntimeError('Function didn\'t raise exception')
    except NotImplementedError:
        pass

    try:
        translate.to_vectorised_df(GeneralisedCols.iob)
        raise RuntimeError('Function didn\'t raise exception')
    except NotImplementedError:
        pass


@pytest.mark.skipif(not os.path.isdir(Configuration().perid_data_folder), reason="reads real data")
def test_works_for_daily_sampled_irregular_time_series():
    raw_df = ReadPreprocessedDataFrame(sampling=Daily(), zip_id='14092221').df
    translate = TranslateIntoTimeseries(raw_df, IrregularTimeseries(), mean_cols)

    processed_df = translate.processed_df
    assert_that(processed_df.shape[1], is_(3))

    continuous_dfs = translate.to_continuous_time_series_dfs()
    assert_that(len(continuous_dfs), is_(34))
    assert_that(TranslateIntoTimeseries.get_longest_df(continuous_dfs).shape[0], is_(87))
    date = datetime(year=2018, month=9, day=30, tzinfo=timezone.utc)
    assert_that(TranslateIntoTimeseries.get_df_including_date(continuous_dfs, date).shape[0], is_(71))

    time_features = translate.to_df_with_time_features()
    assert_that(time_features.shape[1], is_(9))

    try:
        translate.to_x_train()
        raise RuntimeError('Function didn\'t raise exception')
    except NotImplementedError:
        pass

    try:
        translate.to_vectorised_df(GeneralisedCols.iob)
        raise RuntimeError('Function didn\'t raise exception')
    except NotImplementedError:
        pass


@pytest.mark.skipif(not os.path.isdir(Configuration().perid_data_folder), reason="reads real data")
def test_works_for_hourly_sampled_irregular_time_series():
    raw_df = ReadPreprocessedDataFrame(sampling=Hourly(), zip_id='14092221').df
    translate = TranslateIntoTimeseries(raw_df, IrregularTimeseries(), mean_cols)

    processed_df = translate.processed_df
    assert_that(processed_df.shape[1], is_(3))

    continuous_dfs = translate.to_continuous_time_series_dfs()
    assert_that(len(continuous_dfs), is_(5))
    assert_that(TranslateIntoTimeseries.get_longest_df(continuous_dfs).shape[0], is_(7826))
    date = datetime(year=2018, month=9, day=29, tzinfo=timezone.utc)
    assert_that(TranslateIntoTimeseries.get_df_including_date(continuous_dfs, date).shape[0], is_(3205))

    time_features = translate.to_df_with_time_features()
    assert_that(time_features.shape[1], is_(9))

    try:
        translate.to_x_train()
        raise RuntimeError('Function didn\'t raise exception')
    except NotImplementedError:
        pass

    try:
        translate.to_vectorised_df(GeneralisedCols.iob)
        raise RuntimeError('Function didn\'t raise exception')
    except NotImplementedError:
        pass

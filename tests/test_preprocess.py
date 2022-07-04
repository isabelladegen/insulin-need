from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
from hamcrest import *

from src.configurations import Configuration
from src.preprocess import dedub_device_status_dataframes, group_into_consecutive_intervals, \
    number_of_groups_with_more_than_x_items, number_of_interval_in_days, continuous_subseries
from src.read import ReadRecord
from tests.helper.BgDfBuilder import BgDfBuilder, create_time_stamps
from tests.helper.DeviceStatusDfBuilder import DeviceStatusDfBuilder
from tests.helper.ReadRecordBuilder import ReadRecordBuilder


def test_removes_duplicated_device_status_rows_from_each_df_in_read_record():
    # build test ReadRecords
    zip_id1 = '1'
    dub_1 = 0
    df1 = DeviceStatusDfBuilder().with_duplicated_rows_indices(dub_1).build()
    test_record1 = ReadRecordBuilder().with_id(zip_id1).with_df(df1).build()

    zip_id2 = '2'
    dub_2 = 4
    df2 = DeviceStatusDfBuilder().with_duplicated_rows_indices(dub_2).build()
    test_record2 = ReadRecordBuilder().with_id(zip_id2).with_df(df2).build()
    assert_that(df1.shape[0], is_(10))  # ensure the builder doesn't dedub
    assert_that(df2.shape[0], is_(10))

    results = dedub_device_status_dataframes([test_record1, test_record2])

    # assert resulting dataframe has been dedubed
    number_of_cols = len(Configuration().device_status_col_type)
    assert_that(len(results), is_(2))
    assert_that(results[0].df.shape, is_((10 - dub_1, number_of_cols)))
    assert_that(results[1].df.shape, is_((10 - dub_2, number_of_cols)))


def test_can_deal_with_missing_columns():
    zip_id1 = '5'
    dub_1 = 2
    df = DeviceStatusDfBuilder().with_duplicated_rows_indices(dub_1).build()
    drop_cols = ['openaps/enacted/deliverAt', 'openaps/iob/bolusinsulin', 'openaps/iob/lastBolusTime',
                 'pump/status/suspended']
    df.drop(
        drop_cols,
        axis=1, inplace=True)
    test_record = ReadRecordBuilder().with_id(zip_id1).with_df(df).build()
    assert_that(df.shape[0], is_(10))

    result = dedub_device_status_dataframes([test_record])
    number_of_cols = len(Configuration().device_status_col_type)
    assert_that(result[0].df.shape, is_((10 - dub_1, number_of_cols - len(drop_cols))))


def test_can_deal_with_none_df():
    record = ReadRecord()
    result = dedub_device_status_dataframes([record])
    assert_that(result, is_not(None))


def test_groups_df_into_consecutive_sampling_time():
    # build df
    date1 = datetime(year=2018, month=12, day=25, hour=12, minute=0, tzinfo=timezone.utc)
    date2 = datetime(year=2020, month=8, day=14, hour=6, minute=11, tzinfo=timezone.utc)
    date3 = datetime(year=2020, month=8, day=14, hour=6, minute=17, tzinfo=timezone.utc)

    no1 = 10
    no2 = 1
    no3 = 5

    times1 = create_time_stamps(date1, no1)
    values1 = list(np.random.randint(50, 400, no1))
    times2 = [date2]
    values2 = list(np.random.randint(50, 400, no2))
    times3 = create_time_stamps(date3, no3)
    values3 = list(np.random.randint(50, 400, no3))

    df = pd.DataFrame(data={'time': times1 + times2 + times3, 'bg': values1 + values2 + values3})

    # create group columns
    result = group_into_consecutive_intervals(df, 5)

    assert_that(len(result['group'].unique()), is_(3))  # three groups of 5 min intervals
    group1 = result.loc[result['group'] == 0]
    group2 = result.loc[result['group'] == 1]
    group3 = result.loc[result['group'] == 2]
    assert_that(list(group1['bg']), is_(values1))
    assert_that(list(group1['time']), is_(times1))
    assert_that(list(group2['bg']), is_(values2))
    assert_that(list(group2['time']), is_(times2))
    assert_that(list(group3['bg']), is_(values3))
    assert_that(list(group3['time']), is_(times3))


def test_groups_df_into_consecutive_sampling_time_respecting_date():
    time_col = 'time'
    interval = 15  # sampling interval
    # only keep continuous 15 sampled series for 2 days
    min_length = number_of_interval_in_days(2, interval)

    # build time series
    date1 = datetime(year=2018, month=12, day=25, hour=12, minute=0, tzinfo=timezone.utc)
    date2 = datetime(year=2020, month=3, day=14, hour=6, minute=11, tzinfo=timezone.utc)
    date2_more = date2 + timedelta(minutes=10)
    date3 = datetime(year=2021, month=8, day=14, hour=6, minute=17, tzinfo=timezone.utc)

    # Two series sampled at interval i and of at least length min_length
    times1 = create_time_stamps(date1, min_length, interval)
    times3 = create_time_stamps(date3, min_length + 1, interval)  # this will be longer and be in
    # One series less than length min_length but with more sample
    times2 = create_time_stamps(date2, min_length, interval)  # this will not be long enough
    times2_more = create_time_stamps(date2_more, min_length, interval)
    times_too_short = times2 + times2_more

    df = pd.DataFrame(data={time_col: times_too_short + times1 + times3})  # wrong order to force sorting

    result = group_into_consecutive_intervals(df, interval, time_col)

    assert_that(len(result['group'].unique()), is_(3))


def test_grouping_can_deal_with_nan_values():
    df = BgDfBuilder().build(add_nan=2)

    result = group_into_consecutive_intervals(df, 5)
    assert_that(len(result['group'].unique()), is_(1))


def test_counts_how_many_groups_have_more_than_x_items():
    example_groups = [0, 0, 0, 1, 2, 2]
    cut_off = 3

    df = pd.DataFrame(data={'group': example_groups})

    result = number_of_groups_with_more_than_x_items(df, cut_off)
    assert_that(result, is_(1))


def test_counts_how_many_groups_have_more_than_x_items_can_deal_with_0():
    example_groups = [0, 0, 0, 1, 2, 2]
    cut_off = 4

    df = pd.DataFrame(data={'group': example_groups})

    result = number_of_groups_with_more_than_x_items(df, cut_off)
    assert_that(result, is_(0))


def test_returns_list_of_continuous_series_of_length_x_and_sampled_at_interval():
    time_col = 'time'
    interval = 15  # sampling interval
    # only keep continuous 15 sampled series for 2 days
    min_length = number_of_interval_in_days(2, interval)

    # build time series
    date1 = datetime(year=2018, month=12, day=25, hour=12, minute=0, tzinfo=timezone.utc)
    date2 = datetime(year=2020, month=3, day=14, hour=6, minute=11, tzinfo=timezone.utc)
    date2_more = date2 + timedelta(minutes=10)
    date3 = datetime(year=2021, month=8, day=14, hour=6, minute=17, tzinfo=timezone.utc)

    # Two series sampled at interval i and of at least length min_length
    times1 = create_time_stamps(date1, min_length, interval)
    times3 = create_time_stamps(date3, min_length + 1, interval)  # this will be longer and be in
    # One series less than length min_length but with more sample
    times2 = create_time_stamps(date2, min_length-2, interval)  # this will not be long enough
    times2_more = create_time_stamps(date2_more, min_length-2, interval)
    times_too_short = times2 + times2_more

    df = pd.DataFrame(data={time_col: times_too_short + times1 + times3})  # wrong order to force sorting

    result = continuous_subseries(df, min_length, interval, time_col)
    sub_df1 = result[0]
    sub_df2 = result[1]

    assert_that(len(result), is_(2), "There wasn't 2 continuous sampled sub series")
    assert_that(list(sub_df1[time_col]) == times1)
    assert_that(list(sub_df2[time_col]) == times3)

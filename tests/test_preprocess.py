from hamcrest import *

from src.configurations import Configuration
from src.format import as_flat_dataframe
from src.preprocess import dedub_device_status_dataframes
from src.read import ReadRecord
from tests.helper.BgDfBuilder import BgDfBuilder
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


def test_can_deal_none_df():
    record = ReadRecord()
    result = dedub_device_status_dataframes([record])
    assert_that(result, is_not(None))

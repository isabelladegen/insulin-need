from hamcrest import *

from src.format import as_flat_dataframe
from tests.helper.BgDfBuilder import BgDfBuilder
from tests.helper.ReadRecordBuilder import ReadRecordBuilder


def test_adds_zip_id_to_data_frame():
    # build test ReadRecords
    zip_id1 = '123466'
    df1 = BgDfBuilder().build()
    test_record1 = ReadRecordBuilder().with_id(zip_id1).with_df(df1).build()

    zip_id2 = 'ab'
    df2 = BgDfBuilder().build()
    test_record2 = ReadRecordBuilder().with_id(zip_id2).with_df(df2).build()

    # transform ReadRecords
    result_df = as_flat_dataframe([test_record1, test_record2])
    resulting_rows = df1.shape[0] + df2.shape[0]
    result_df1 = result_df.loc[result_df.id == zip_id1]
    result_df2 = result_df.loc[result_df.id == zip_id2]

    # assert resulting dataframe matches original ReadRecords
    assert_that(result_df.shape, is_((resulting_rows, 3)))
    assert_that(result_df.columns[0], is_('id'))
    assert_that(result_df.columns[1], is_('time'))
    assert_that(result_df.columns[2], is_('bg'))
    assert_that(result_df.index, has_item(19))  # this is testing reindexing after concatenation
    assert_that(result_df1[['time', 'bg']].equals(df1))
    assert_that(result_df2[['time', 'bg']].reset_index(drop=True).equals(df2))


def test_skips_records_with_no_bg_data():
    # build test ReadRecords
    none_df_record = ReadRecordBuilder().with_df(None).build()

    zip_id1 = '3456'
    df1 = BgDfBuilder().build()
    test_record1 = ReadRecordBuilder().with_id(zip_id1).with_df(df1).build()

    zip_id2 = '345'
    df2 = BgDfBuilder().build()
    test_record2 = ReadRecordBuilder().with_id(zip_id2).with_df(df2).build()

    # transform ReadRecords
    result_df = as_flat_dataframe([test_record1, none_df_record, test_record2])

    # assert resulting dataframe contains both df 1 and df2
    assert_that(result_df.shape, is_((df1.shape[0] + df2.shape[0], 3)))


def test_drops_rows_with_nan_value():
    # build test ReadRecords
    zip_id1 = '3456'
    nans = 2
    df1 = BgDfBuilder().build(add_nan=nans)
    test_record1 = ReadRecordBuilder().with_id(zip_id1).with_df(df1).build()

    # transform ReadRecords
    result_df = as_flat_dataframe([test_record1], True)

    # assert resulting dataframe contains both df 1 and df2
    assert_that(result_df.shape, is_((df1.shape[0] - nans, 3)))

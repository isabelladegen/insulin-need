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
    assert_that(result_df1.equals(df1))
    assert_that(result_df2.equals(df2))

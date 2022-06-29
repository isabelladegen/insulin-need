import pytest
import pandas as pd
from pathlib import Path
from hamcrest import *

from src.write import write_read_record
from tests.helper.BgDfBuilder import BgDfBuilder
from tests.helper.ReadRecordBuilder import ReadRecordBuilder

folder = 'data/'
flat_file_name = 'some_flat_file.csv'
flat_file_path = Path(folder, flat_file_name)


def test_writes_multiple_read_records_as_flat_dataframe():
    # build test ReadRecords
    id1 = '123466'
    df1 = BgDfBuilder().build()
    test_record1 = ReadRecordBuilder().with_id(id1).with_df(df1).build()

    id2 = 'ab'
    df2 = BgDfBuilder().build()
    test_record2 = ReadRecordBuilder().with_id(id2).with_df(df2).build()

    id3 = '456'
    df3 = BgDfBuilder().build()
    test_record3 = ReadRecordBuilder().with_id(id3).with_df(df3).build()
    records = [test_record1, test_record2, test_record3]

    # write as flat csv
    write_read_record(records, True, folder, flat_file_name)

    # read from csv
    df = pd.read_csv(flat_file_path, index_col=[0])
    # all three ids were written
    assert_that(df.shape, is_((30, 3)))
    assert_that(sub_df_for_id(df, id1).equals(df1))
    assert_that(sub_df_for_id(df, id2).equals(df2))
    assert_that(sub_df_for_id(df, id3).equals(df3))


def sub_df_for_id(df, id1):
    result = df.loc[df['id'] == id1].drop(columns=['id'])
    result['time'] = pd.to_datetime(result['time'])
    result = result.reset_index().drop(columns=['index'])
    return result


@pytest.fixture(autouse=True)
def run_before_and_after_tests(tmpdir):
    # Setup:

    yield  # this is where the testing happens
    flat_file_path.unlink(True)
    # Teardown:

import glob
import os
import pytest
import pandas as pd
from pathlib import Path
from hamcrest import *
from src.configurations import TestConfiguration, Configuration
from src.read import read_bg_from_zip, read_all_bg, is_a_bg_csv_file, convert_problem_timestamps, \
    read_all_android_aps_files, read_device_status_from_zip, is_a_device_status_csv_file, read_all_device_status, \
    read_flat_device_status_df_from_file, convert_left_over_time_cols

config = TestConfiguration()


def get_file_for_id(file_id='99908129'):
    test_data_dir = config.data_dir
    # get all zip files in folder
    filepaths = glob.glob(str(test_data_dir) + "/*.zip")
    test_file = [f for f in filepaths if f.endswith('%s.zip' % file_id)][0]
    return test_file


def assert_all_time_cols_have_time_dtype(time_cols, df):
    types = df.dtypes
    for time_col in time_cols:  # test that time columns have been converted
        if time_col in df.columns:  # not all files have all columns
            assert_that(str(types[time_col]).lower(), contains_string('datetime'),
                        reason=str(time_col) + ' is not of dtype datetime64')


@pytest.mark.skipif(not os.path.isdir(Configuration().data_dir), reason="reads real data")
def test_reads_bg_from_given_zip_file():
    test_data_dir = config.data_dir
    # get all zip files in folder
    filepaths = glob.glob(str(test_data_dir) + "/*.zip")
    path = filepaths[1]
    result = read_bg_from_zip(path, config)
    assert_that(result, is_not(empty()))
    assert_that(Path(path).stem, is_(result.zip_id))
    assert_that(result.df.shape[0], greater_than(10))


@pytest.mark.skip(reason="takes a long time")
def test_reads_all_peoples_files():
    result = read_all_bg(config)
    assert_that(len(result), is_(145))


@pytest.mark.skipif(not os.path.isdir(Configuration().data_dir), reason="reads real data")
def test_is_a_bg_csv_file():
    example_name = 'direct-sharing-31/84984656_entries.json_csv/84984656_entries.json.csv'
    assert_that(is_a_bg_csv_file(config, '84984656', example_name), is_(True))


@pytest.mark.skipif(not os.path.isdir(Configuration().data_dir), reason="reads real data")
def test_is_a_device_status_csv_file():
    example_name = "direct-sharing-31/99908129_devicestatus__to_2018-02-02_csv/99908129_devicestatus__to_2018-02-02_aa.csv"
    assert_that(is_a_device_status_csv_file(config, '99908129', example_name), is_(True))


def test_can_read_messed_up_time_stamps():
    data = ['10/09/2016 00:44:23 AM', '2015-12-04 00:57:16.704+0100', '07/01/2015 14:30:28 PM',
            '07/01/2015 02:30:28 PM', '2016-10-28T12:17:18.191+0000', 'Tue Jul 18 13:11:04 EDT 2017',
            'Fri Jan 29 12:55:11 EST 2016', 'Fri Jan 29 12:55:11 CDT 2016', '2016-10-28T12:17:18.191+0000',
            '10/09/2016 19:39:21 PM']
    df = pd.DataFrame(data)

    convert_problem_timestamps(df, df.columns[0])
    assert_that(str(df.iloc[0, 0]), is_('2016-10-08 23:44:23+00:00'))  # correctly changes date
    assert_that(str(df.iloc[1, 0]), is_('2015-12-03 23:57:16.704000+00:00'))  # correctly changes date for utc
    assert_that(str(df.iloc[2, 0]), is_('2015-07-01 14:30:28+00:00'))
    assert_that(str(df.iloc[3, 0]), is_('2015-07-01 14:30:28+00:00'))
    assert_that(str(df.iloc[4, 0]), is_('2016-10-28 12:17:18.191000+00:00'))  # just to get right grounding
    assert_that(str(df.iloc[5, 0]), is_('2017-07-18 17:11:04+00:00'))  # correctly changes EDT
    assert_that(str(df.iloc[6, 0]), is_('2016-01-29 17:55:11+00:00'))  # correctly reads EST
    assert_that(str(df.iloc[7, 0]), is_('2016-01-29 17:55:11+00:00'))  # correctly reads CDT
    assert_that(str(df.iloc[8, 0]), is_('2016-10-28 12:17:18.191000+00:00'))
    assert_that(str(df.iloc[9, 0]), is_('2016-10-09 19:39:21+00:00'))


@pytest.mark.skipif(not os.path.isdir(Configuration().data_dir), reason="reads real data")
def test_can_read_android_aps_uploads():
    result = read_all_android_aps_files(config)
    assert_that(len(result), is_(38))


@pytest.mark.skipif(not os.path.isdir(Configuration().data_dir), reason="reads real data")
def test_reads_device_status_from_given_zip_file():
    configuration = TestConfiguration()
    configuration.device_status_col_type = None  # test a config where all columns are read
    # get all zip files in folder
    test_file = get_file_for_id()

    result = read_device_status_from_zip(test_file, configuration)
    assert_that(result, is_not(empty()))
    assert_that(Path(test_file).stem, is_(result.zip_id))
    assert_that(result.df.shape[0], greater_than(100))
    assert_that(result.df.shape[1], equal_to(559))


@pytest.mark.skipif(not os.path.isdir(Configuration().data_dir), reason="reads real data")
def test_reads_only_columns_in_config_from_device_status_from_given_zip_file():
    test_file = get_file_for_id()
    result = read_device_status_from_zip(test_file, config)
    assert_that(result, is_not(empty()))
    assert_that(Path(test_file).stem, is_(result.zip_id))
    assert_that(result.df.shape[0], greater_than(100))
    assert_that(result.df.shape[1], equal_to(29))


@pytest.mark.skipif(not os.path.isdir(Configuration().data_dir), reason="reads real data")
def test_reads_original_device_status_data_writes_to_flat_df_reads_it_back():
    # get all zip files in folder
    test_file = get_file_for_id('14092221')
    result = read_device_status_from_zip(test_file, config)
    time_cols_in_file = [col for col in result.df.columns if col in config.time_cols()]
    assert_that(len(time_cols_in_file), greater_than(0), "There were no time columns")
    assert_all_time_cols_have_time_dtype(config.time_cols(), result.df)

    # safe flat file and re-read
    flat_file = Path('data/test_df.csv')
    result.df.to_csv(flat_file)

    # read flat file
    flat_df = read_flat_device_status_df_from_file(flat_file, config)
    assert_all_time_cols_have_time_dtype(config.time_cols(), flat_df)


def test_can_convert_mixed_time_stamp_and_epoch_time_cols():
    time_cols = ['time']
    difficult_data = ['2018-11-12T12:35:25.000Z', None, '2018-11-12T12:35:25.000Z', '1542026031970']
    df = pd.DataFrame({'time': difficult_data})
    convert_left_over_time_cols(df, time_cols)
    assert_all_time_cols_have_time_dtype(time_cols, df)


@pytest.mark.skip(reason="takes a real long time reading all data")
def test_reads_all_device_status_files():
    result = read_all_device_status(config)
    assert_that(len(result), is_(145))


@pytest.mark.skip(reason="takes a real long time reading all data")
def test_reads_flat_device_data_file():
    config = TestConfiguration()
    df = read_flat_device_status_df_from_file(Configuration().flat_device_status_116_file, config)
    assert_that(df.shape, is_((10480156, 32)))
    assert_all_time_cols_have_time_dtype(config.time_cols(), df)

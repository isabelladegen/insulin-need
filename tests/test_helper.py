import os
from src.configurations import Configuration
from src.helper import files_for_id, bg_file_path_for, device_status_file_path_for, preprocessed_file_for
import pytest
from hamcrest import *

from src.resampling import Irregular, Hourly, Daily

zip_id = '14092221'


@pytest.mark.skipif(not os.path.isdir(Configuration().perid_data_folder), reason="reads real data")
def test_gives_full_path_for_id_and_file_name():
    # write csv files
    files = files_for_id(Configuration().perid_data_folder, zip_id)

    assert_that(len(list(filter(lambda x: Configuration().bg_file in x, files))), is_(1))
    assert_that(len(list(filter(lambda x: Configuration().device_file in x, files))), is_(1))
    assert_that(len(list(filter(lambda x: Irregular.csv_file_name() in x, files))), is_(1))


@pytest.mark.skipif(not os.path.isdir(Configuration().perid_data_folder), reason="reads real data")
def test_returns_bg_file_path_for_id():
    file = bg_file_path_for(Configuration().perid_data_folder, zip_id)

    assert_that(file.name, is_(Configuration().bg_file))


@pytest.mark.skipif(not os.path.isdir(Configuration().perid_data_folder), reason="reads real data")
def test_returns_device_status_file_path_for_id():
    file = device_status_file_path_for(Configuration().perid_data_folder, zip_id)

    assert_that(file.name, is_(Configuration().device_file))


@pytest.mark.skipif(not os.path.isdir(Configuration().perid_data_folder), reason="reads real data")
def test_returns_irregular_hourly_or_daily_iob_cob_bg_file_for_id():
    folder = Configuration().perid_data_folder

    irregular_file_name = preprocessed_file_for(folder, zip_id, Irregular()).name
    # hourly_file_name = preprocessed_file_for(folder, zip_id, Hourly()).name
    # daily_file_name = preprocessed_file_for(folder, zip_id, Daily()).name

    assert_that(irregular_file_name, is_(Irregular.csv_file_name()))
    # TODO comment once files exist
    # assert_that(hourly_file_name, is_(Configuration().hourly_iob_cob_bg_file))
    # assert_that(daily_file_name, is_(Configuration().daily_iob_cob_bg_file))

from src.configurations import Configuration
from src.helper import files_for_id, bg_file_path_for, device_status_file_path_for
import pytest
import pandas as pd
from pathlib import Path
from hamcrest import *

zip_id = '14092221'


def test_gives_full_path_for_id_and_file_name():
    # write csv files
    files = files_for_id(Configuration().perid_data_folder, zip_id)

    assert_that(len(files), is_(2))
    assert_that(len(list(filter(lambda x: Configuration().bg_file in x, files))), is_(1))
    assert_that(len(list(filter(lambda x: Configuration().device_file in x, files))), is_(1))


def test_returns_bg_file_path_for_id():
    file = bg_file_path_for(Configuration().perid_data_folder, zip_id)

    assert_that(file.name, Configuration().bg_file)


def test_returns_device_status_file_path_for_id():
    file = device_status_file_path_for(Configuration().perid_data_folder, zip_id)

    assert_that(file.name, Configuration().device_file)

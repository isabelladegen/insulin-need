from dataclasses import dataclass
from typing import List

import pandas as pd
import yaml
from os import path
from enum import Enum

from src.resampling import Resampling

ROOT_DIR = path.realpath(path.join(path.dirname(__file__), '..'))


def load_private_yaml():
    private_file = path.join(ROOT_DIR, 'private.yaml')
    assert (path.exists(private_file))
    with open(private_file, "r") as f:
        config = yaml.safe_load(f)
    return config


# values for openAPS
class OpenAPSConfigs(str, Enum):
    iob = 'openaps/enacted/IOB'
    cob = 'openaps/enacted/COB'
    bg = 'openaps/enacted/bg'
    datetime = 'openaps/enacted/timestamp'
    system_name = 'OpenAPS'


# generalised configs across systems
class GeneralisedCols(str, Enum):
    iob = 'iob'
    cob = 'cob'
    bg = 'bg'
    id = 'id'
    datetime = 'datetime'
    system = 'system'


@dataclass
class Configuration:
    config = load_private_yaml()
    # READ CONFIGURATIONS
    data_dir: str = config['openAPS_data_path']
    as_flat_file = config['flat_file']
    data_folder = path.join(ROOT_DIR, 'data')
    perid_data_folder = path.join(data_folder, 'perid')

    # bg files
    bg_csv_file_extension = '.json.csv'
    bg_csv_file_start = '_entries'
    bg_csv_file_android = 'BgReadings.csv'
    android_upload_info = 'UploadInfo.csv'
    bg_file = 'bg_df.csv'
    device_file = 'device_status_dedubed.csv'

    # device status files
    device_status_csv_file_start = '_devicestatus'
    device_status_csv_file_extension = '.csv'
    device_status_col_type = {  # if None all columns are read, key is column name, value is dtype
        'id': str,
        'created_at': str,
        'device': str,
        'pump/clock': str,
        'pump/status/timestamp': str,
        'pump/status/suspended': str,
        'pump/status/status': str,
        'pump/status/bolusing': str,
        'pump/iob/timestamp': str,
        'pump/iob/iob': str,
        'openaps/enacted/deliverAt': str,
        'openaps/enacted/timestamp': str,
        'openaps/enacted/rate': pd.Float32Dtype(),
        'openaps/enacted/duration': pd.Float32Dtype(),
        'openaps/enacted/insulinReq': str,
        'openaps/enacted/COB': pd.Float32Dtype(),
        'openaps/enacted/IOB': pd.Float32Dtype(),
        'openaps/enacted/bg': pd.Float32Dtype(),
        'openaps/enacted/eventualBG': pd.Float32Dtype(),
        'openaps/enacted/minPredBG': pd.Float32Dtype(),
        'openaps/enacted/sensitivityRatio': pd.Float32Dtype(),
        'openaps/enacted/reason': str,
        'openaps/enacted/units': str,
        'openaps/iob/iob': pd.Float32Dtype(),
        'openaps/iob/bolusinsulin': pd.Float64Dtype(),
        'openaps/iob/microBolusInsulin': pd.Float32Dtype(),
        'openaps/iob/lastBolusTime': str,
        'openaps/iob/timestamp': str,
        'openaps/iob/lastTemp/rate': pd.Float32Dtype(),
        'openaps/iob/basaliob': str,
        'openaps/iob/netbasalinsulin': pd.Float32Dtype(),
        'openaps/iob/activity': str,
    }
    flat_device_status_116_file_name = 'device_status_116_df.csv'
    flat_device_status_116_file = data_folder + flat_device_status_116_file_name
    dedub_flat_device_status_116_file_name = 'device_status_116_df_dedubed.csv'
    dedub_flat_device_status_116_file = data_folder + dedub_flat_device_status_116_file_name

    # columns to keep
    # TODO use generalised cols instead
    keep_columns = [OpenAPSConfigs.datetime, OpenAPSConfigs.iob, OpenAPSConfigs.bg,
                    OpenAPSConfigs.cob]

    # Android APS has different format
    android_aps_zip = 'AndroidAPS Uploader.zip'

    def common_cols(self):
        return ['id', 'created_at', 'device']

    def enacted_cols(self):
        return [k for k in self.device_status_col_type.keys() if 'enacted' in k]

    def iob_cols(self):
        return [k for k in self.device_status_col_type.keys() if 'openaps/iob/' in k]

    def pump_cols(self):
        return [k for k in self.device_status_col_type.keys() if 'pump/' in k]

    def time_cols(self):
        return ['created_at', 'openaps/enacted/deliverAt', 'pump/clock'] \
               + [k for k in self.device_status_col_type.keys() if 'time' in str(k).lower()]

    def flat_preprocessed_file_for(self, sampling: Resampling):
        return path.join(self.data_folder, sampling.csv_file_name())


# Configuration to use for unit tests. This turns Wandb logging off.
@dataclass
class TestConfiguration(Configuration):
    wandb_mode = 'todo'

from dataclasses import dataclass

import pandas as pd
import yaml
from os import path
from enum import Enum

ROOT_DIR = path.realpath(path.join(path.dirname(__file__), '..'))


def load_private_yaml():
    private_file = path.join(ROOT_DIR, 'private.yaml')
    assert (path.exists(private_file))
    with open(private_file, "r") as f:
        config = yaml.safe_load(f)
    return config


class OpenAPSConfigs(str, Enum):
    # values for openAPS
    iob = 'openaps/enacted/IOB'
    cob = 'openaps/enacted/COB'
    bg = 'openaps/enacted/bg'
    datetime = 'openaps/enacted/timestamp'
    system_name = 'OpenAPS'


class ResamplingAggCols(str, Enum):
    # colum name and name of aggregation function
    min = 'min'
    max = 'max'
    mean = 'mean'
    std = 'std'


class GeneralisedCols(str, Enum):
    # generalised configs across systems
    iob = 'iob'
    cob = 'cob'
    bg = 'bg'
    id = 'id'
    mean_iob = iob + ' ' + ResamplingAggCols.mean.value
    mean_cob = cob + ' ' + ResamplingAggCols.mean.value
    mean_bg = bg + ' ' + ResamplingAggCols.mean.value
    min_iob = iob + ' ' + ResamplingAggCols.min.value
    min_cob = cob + ' ' + ResamplingAggCols.min.value
    min_bg = bg + ' ' + ResamplingAggCols.min.value
    max_iob = iob + ' ' + ResamplingAggCols.max.value
    max_cob = cob + ' ' + ResamplingAggCols.max.value
    max_bg = bg + ' ' + ResamplingAggCols.max.value
    std_iob = iob + ' ' + ResamplingAggCols.std.value
    std_cob = cob + ' ' + ResamplingAggCols.std.value
    std_bg = bg + ' ' + ResamplingAggCols.std.value
    datetime = 'datetime'
    system = 'system'


@dataclass
class Resampling:
    max_gap_in_min = None
    # how big the gap between two datetime stamps can be
    min_no_samples_per_interval = None
    # how many samples at max_gap_per_min intervals there have to be
    sample_rule = None
    # the frequency of the regular time series after resampling: 1H a reading every hour, 1D a reading every day
    description = 'Base'
    agg_cols = [ResamplingAggCols.min.value, ResamplingAggCols.max.value, ResamplingAggCols.mean.value,
                ResamplingAggCols.std.value]

    general_agg_cols_dictionary = {GeneralisedCols.id.value: 'first',
                                   GeneralisedCols.system.value: 'first',
                                   }

    @staticmethod
    def csv_file_name():
        return ''


@dataclass
class Irregular(Resampling):
    description = 'irregular'

    @staticmethod
    def csv_file_name():
        return 'irregular_iob_cob_bg.csv'


@dataclass
class Hourly(Resampling):
    max_gap_in_min = 60
    # there needs to be a reading at least every hour for the data points to be resampled for that hour
    sample_rule = '1H'
    description = 'hourly resampled'

    @staticmethod
    def csv_file_name():
        return 'hourly_iob_cob_bg.csv'


@dataclass
class Daily(Resampling):
    max_gap_in_min = 180
    # a reading every three hours for a daily resampling to be created
    sample_rule = '1D'
    description = 'daily resampled'

    @staticmethod
    def csv_file_name():
        return 'daily_iob_cob_bg.csv'


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

    @staticmethod
    def common_cols():
        # this should probably move to OpenAPS as it is OpenAPS specific
        return ['id', 'created_at', 'device']

    @staticmethod
    def info_columns():
        # returns the columns that have other info but not values to resample
        return [GeneralisedCols.datetime, GeneralisedCols.id, GeneralisedCols.system]

    @staticmethod
    def value_columns_to_resample():
        # returns all columns with values that need resampling
        return [GeneralisedCols.iob.value, GeneralisedCols.cob.value, GeneralisedCols.bg.value]

    @staticmethod
    def resampled_value_columns():
        # returns the columns that
        return [GeneralisedCols.mean_iob.value,
                GeneralisedCols.mean_cob.value,
                GeneralisedCols.mean_bg.value,
                GeneralisedCols.min_iob.value,
                GeneralisedCols.min_cob.value,
                GeneralisedCols.min_bg.value,
                GeneralisedCols.max_iob.value,
                GeneralisedCols.max_cob.value,
                GeneralisedCols.max_bg.value,
                GeneralisedCols.std_iob.value,
                GeneralisedCols.std_cob.value,
                GeneralisedCols.std_bg.value,
                ]

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

# builds device status dataframes for testing
import string
from datetime import datetime, timezone, timedelta
import random

import numpy as np
import pandas as pd

from src.configurations import Configuration


def create_time_stamps(start_dt, rows):
    result = []
    dt = start_dt
    values_interval = timedelta(minutes=5)
    for i in range(rows):
        result.append(dt)
        dt = dt + values_interval
    return result


def create_random_strings(string_length, number_of_random_strings):
    result = []
    for i in range(number_of_random_strings):
        result.append(create_random_string(string_length))
    return result


def create_random_string(length=5):
    chars = string.ascii_uppercase + string.digits
    return ''.join(random.choice(chars) for _ in range(length))


class DeviceStatusDfBuilder:
    def __init__(self):
        self.duplicated_rows = 0
        self.number_of_rows = 10

    def build(self):
        # build unique rows
        unique_rows = self.number_of_rows - self.duplicated_rows
        start_dt = datetime(year=2021, month=6, day=10, hour=14, minute=0, tzinfo=timezone.utc)
        created_at = create_time_stamps(start_dt, self.number_of_rows)
        devices = create_random_strings(5, self.number_of_rows)
        ids = ['123456'] * self.number_of_rows

        times = create_time_stamps(start_dt, unique_rows)
        times = times + [times[0]] * self.duplicated_rows

        config = Configuration()
        col_type = config.device_status_col_type
        columns = list(col_type.keys())
        data_dict = {'id': ids,
                     'created_at': created_at,
                     'device': devices, }

        for col in columns:
            if col in config.time_cols():
                data_dict[col] = times
                continue
            if col in config.iob_cols():  # we'll have already done the time cols
                self.random_column_values(col, col_type, data_dict, unique_rows)
                continue
            if col in config.enacted_cols():  # we'll have already done the time cols
                self.random_column_values(col, col_type, data_dict, unique_rows)
                continue
            if col in config.pump_cols():
                self.random_column_values(col, col_type, data_dict, unique_rows)
                continue

        df = pd.DataFrame(data_dict)
        return df

    def random_column_values(self, col, col_type, data_dict, unique_rows):
        if col_type[col] is pd.Float32Dtype():  # create a random int
            randint = np.random.randint(0, 200, unique_rows)
            randint = list(randint) + [randint[0]] * self.duplicated_rows
            data_dict[col] = randint
        else:
            randstrings = create_random_strings(10, unique_rows)
            randstrings = randstrings + [randstrings[0]] * self.duplicated_rows
            data_dict[col] = randstrings

    def with_duplicated_rows_indices(self, duplicated_rows: int):
        self.duplicated_rows = duplicated_rows
        return self

# builds bg dataframes for testing
from datetime import datetime, timezone, timedelta
import random

import numpy as np
import pandas as pd


def create_time_stamps(start_dt, rows, interval_in_min=5):
    result = []
    dt = start_dt
    values_interval = timedelta(minutes=interval_in_min)
    for i in range(rows):
        result.append(dt)
        dt = dt + values_interval
    return result


class BgDfBuilder:
    def build(self, add_nan=0):
        rows = 10
        start_dt = datetime(year=2021, month=6, day=10, hour=14, minute=0, tzinfo=timezone.utc)

        times = create_time_stamps(start_dt, rows)
        values = np.random.randint(50, 400, rows)
        df = pd.DataFrame(data={'time': times, 'bg': values})

        # insert nan
        ix = [(row, col) for row in range(df.shape[0]) for col in range(df.shape[1])]
        for row, col in random.sample(ix, add_nan):
            df.iat[row, col] = np.nan

        return df

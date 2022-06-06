# builds bg dataframes for testing
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd


def create_time_stamps(start_dt, rows):
    result = []
    dt = start_dt
    values_interval = timedelta(minutes=5)
    for i in range(rows):
        result.append(dt)
        dt = dt + values_interval
    return result


class BgDfBuilder:
    def build(self):
        rows = 10
        start_dt = datetime(year=2021, month=6, day=10, hour=14, minute=0, tzinfo=timezone.utc)

        times = create_time_stamps(start_dt, rows)
        values = np.random.randint(50, 400, rows)
        return pd.DataFrame(data={'time': times, 'bg': values})

from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
from hamcrest import *

from src.preprocess import number_of_interval_in_days
from src.resample import resample_df
from tests.helper.BgDfBuilder import create_time_stamps


def test_resamples_df():
    time_col = 'time'
    sample_col = 'values'
    interval = 60  # a value per hour
    # 3 days of hourly samples
    min_length = number_of_interval_in_days(3, interval)

    start_date = datetime(year=2018, month=12, day=25, hour=12, minute=0, tzinfo=timezone.utc)

    # build df of 3 days of data, sampled hourly
    times = create_time_stamps(start_date, min_length, interval)
    values = list(np.random.randint(50, 400, min_length))
    df = pd.DataFrame(data={time_col: times, sample_col: values})

    result = resample_df(df, '1D', time_col)
    resulting_times = list(result.index)

    assert_that(result.shape[0], is_(4), "There wasn't three days of data")
    assert_that(resulting_times[0], is_(start_date.date()))
    assert_that(resulting_times[1], is_(start_date.date() + timedelta(days=1)))
    assert_that(resulting_times[2], is_(start_date.date() + timedelta(days=2)))
    assert_that(resulting_times[3], is_(start_date.date() + timedelta(days=3)))

import os

import pytest
from hamcrest import *

from src.configurations import Configuration
from src.stats import Stats, DailyTimeseries

zip_id: str = '14092221'


@pytest.mark.skipif(not os.path.isdir(Configuration().perid_data_folder), reason="reads real data")
def test_describes_data_for_daily_ts():
    stats = Stats(zip_id)
    sampling = DailyTimeseries()

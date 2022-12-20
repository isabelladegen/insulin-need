from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP

import numpy as np
import pandas as pd
import pytest
from hamcrest import *
from numpy import nan

from src.configurations import GeneralisedCols, Daily
from src.resampling import ResampleDataFrame

# test data:
# day1 with a sample exactly every three hour
# day2 without a sample every three hour -> should not be included in mean
# day3 more than one sample every three our

# day 1
d10 = datetime(year=2019, month=1, day=10, hour=0, minute=5, tzinfo=timezone.utc)
d11 = datetime(year=2019, month=1, day=10, hour=3, minute=5, tzinfo=timezone.utc)
d12 = datetime(year=2019, month=1, day=10, hour=6, minute=5, tzinfo=timezone.utc)
d13 = datetime(year=2019, month=1, day=10, hour=9, minute=5, tzinfo=timezone.utc)
d14 = datetime(year=2019, month=1, day=10, hour=12, minute=5, tzinfo=timezone.utc)
d15 = datetime(year=2019, month=1, day=10, hour=15, minute=5, tzinfo=timezone.utc)
d16 = datetime(year=2019, month=1, day=10, hour=18, minute=5, tzinfo=timezone.utc)
d17 = datetime(year=2019, month=1, day=10, hour=21, minute=5, tzinfo=timezone.utc)

# day 2
d20 = datetime(year=2019, month=1, day=11, hour=0, minute=5, tzinfo=timezone.utc)
d21 = datetime(year=2019, month=1, day=11, hour=3, minute=6, tzinfo=timezone.utc)  # misses the every 3h by 1 min
d22 = datetime(year=2019, month=1, day=11, hour=6, minute=5, tzinfo=timezone.utc)
d23 = datetime(year=2019, month=1, day=11, hour=9, minute=5, tzinfo=timezone.utc)
d24 = datetime(year=2019, month=1, day=11, hour=12, minute=5, tzinfo=timezone.utc)
d25 = datetime(year=2019, month=1, day=11, hour=15, minute=5, tzinfo=timezone.utc)
d26 = datetime(year=2019, month=1, day=11, hour=18, minute=5, tzinfo=timezone.utc)
d27 = datetime(year=2019, month=1, day=11, hour=21, minute=5, tzinfo=timezone.utc)

# day 3
d30 = datetime(year=2019, month=1, day=12, hour=0, minute=5, tzinfo=timezone.utc)
d301 = datetime(year=2019, month=1, day=12, hour=1, minute=5, tzinfo=timezone.utc)
d31 = datetime(year=2019, month=1, day=12, hour=3, minute=5, tzinfo=timezone.utc)
d311 = datetime(year=2019, month=1, day=12, hour=4, minute=5, tzinfo=timezone.utc)
d32 = datetime(year=2019, month=1, day=12, hour=6, minute=5, tzinfo=timezone.utc)
d321 = datetime(year=2019, month=1, day=12, hour=6, minute=45, tzinfo=timezone.utc)
d33 = datetime(year=2019, month=1, day=12, hour=9, minute=5, tzinfo=timezone.utc)
d34 = datetime(year=2019, month=1, day=12, hour=12, minute=5, tzinfo=timezone.utc)
d35 = datetime(year=2019, month=1, day=12, hour=15, minute=5, tzinfo=timezone.utc)
d36 = datetime(year=2019, month=1, day=12, hour=18, minute=5, tzinfo=timezone.utc)
d37 = datetime(year=2019, month=1, day=12, hour=21, minute=5, tzinfo=timezone.utc)

zip_id1 = '555'
no_samples = 27
iob = 5
additional_iob = 2
additional_iob2 = 1
cob = 1
additional_cob1 = 3
additional_cob2 = 3.5
bg = 6.4
additional_bg1 = 15.6
additional_bg2 = 2.2
irregular_data = {GeneralisedCols.datetime.value: [d10, d11, d12, d13, d14, d15, d16, d17,
                                                   d20, d21, d22, d23, d24, d25, d26, d27,
                                                   d30, d301, d31, d311, d32, d321, d33, d34, d35, d36, d37],
                  GeneralisedCols.iob.value: [iob] * 16 + [iob, additional_iob, iob, additional_iob2, iob,
                                                           additional_iob] + [iob] * 5,
                  GeneralisedCols.cob.value: [cob] * 16 + [cob, additional_cob1, cob, additional_cob2, cob,
                                                           additional_cob1] + [cob] * 5,  # same value for each time
                  GeneralisedCols.bg.value: [bg] * 16 + [bg, additional_bg1, bg, additional_bg2, bg,
                                                         additional_bg1] + [bg] * 5,
                  GeneralisedCols.system.value: ['bla'] * no_samples,
                  GeneralisedCols.id.value: [zip_id1] * no_samples
                  }
irregular_df = pd.DataFrame(irregular_data)
irregular_df[GeneralisedCols.datetime] = pd.to_datetime(irregular_df[GeneralisedCols.datetime], utc=True,
                                                        errors="raise")
irregular_df[GeneralisedCols.id] = irregular_df[GeneralisedCols.id].astype(str)


def test_turns_irregular_df_into_daily_sampled_min_max_mean_std_df_for_downsample_to_daily_values():
    resampling = ResampleDataFrame(irregular_df)

    df = resampling.resample_to(Daily())
    assert_that(df[GeneralisedCols.datetime].dt.day, has_item(10))
    assert_that(df[GeneralisedCols.datetime].dt.day, has_item(12))
    assert_that(df.shape[0], is_(2))
    assert_that(df[GeneralisedCols.id].unique(), is_(zip_id1))
    assert_that(df[GeneralisedCols.system].unique(), is_('bla'))

    day1 = df.loc[df[GeneralisedCols.datetime].dt.day == 10]
    day3 = df.loc[df[GeneralisedCols.datetime].dt.day == 12]
    # check mean values
    assert_that(day1[GeneralisedCols.mean_iob].values[0], is_(iob))
    assert_that(day3[GeneralisedCols.mean_iob].values[0], is_(4.091))
    assert_that(day1[GeneralisedCols.mean_cob].values[0], is_(cob))
    assert_that(day3[GeneralisedCols.mean_cob].values[0], is_(1.591))
    assert_that(day1[GeneralisedCols.mean_bg].values[0], is_(bg))
    assert_that(day3[GeneralisedCols.mean_bg].values[0], is_(7.691))

    # check min values
    assert_that(day1[GeneralisedCols.min_iob].values[0], is_(iob))
    assert_that(day3[GeneralisedCols.min_iob].values[0], is_(additional_iob2))
    assert_that(day1[GeneralisedCols.min_cob].values[0], is_(cob))
    assert_that(day3[GeneralisedCols.min_cob].values[0], is_(cob))
    assert_that(day1[GeneralisedCols.min_bg].values[0], is_(bg))
    assert_that(day3[GeneralisedCols.min_bg].values[0], is_(additional_bg2))

    # check max values
    assert_that(day1[GeneralisedCols.max_iob].values[0], is_(iob))
    assert_that(day3[GeneralisedCols.max_iob].values[0], is_(iob))
    assert_that(day1[GeneralisedCols.max_cob].values[0], is_(cob))
    assert_that(day3[GeneralisedCols.max_cob].values[0], is_(additional_cob2))
    assert_that(day1[GeneralisedCols.max_bg].values[0], is_(bg))
    assert_that(day3[GeneralisedCols.max_bg].values[0], is_(additional_bg1))

    # check std values
    assert_that(day1[GeneralisedCols.std_iob].values[0], is_(0))
    assert_that(day3[GeneralisedCols.std_iob].values[0], is_(1.578))
    assert_that(day1[GeneralisedCols.std_cob].values[0], is_(0))
    assert_that(day3[GeneralisedCols.std_cob].values[0], is_(1.020))
    assert_that(day1[GeneralisedCols.std_bg].values[0], is_(0))
    assert_that(day3[GeneralisedCols.std_bg].values[0], is_(4.106))


def test_ensures_sampling_is_frequent_enough_for_all_variates():
    just_iob = {GeneralisedCols.datetime.value: [d10, d11, d12, d13, d14, d15, d16, d17,
                                                 d20, d21, d22, d23, d24, d25, d26, d27
                                                 ],
                GeneralisedCols.iob.value: [iob] * 16,  # sufficient values
                GeneralisedCols.cob.value: [cob] * 3 + [nan] * 5 + [cob] * 8,  # not enough values for day 1 and 2
                GeneralisedCols.bg.value: [bg] * 8 + [nan] * 5 + [bg] * 3,  # not enough values for day 2
                GeneralisedCols.system.value: ['bla'] * 16,
                GeneralisedCols.id.value: [zip_id1] * 16
                }
    just_iob_df = pd.DataFrame(just_iob)
    just_iob_df[GeneralisedCols.datetime] = pd.to_datetime(just_iob_df[GeneralisedCols.datetime], utc=True,
                                                           errors="raise")
    resampling = ResampleDataFrame(just_iob_df)

    df = resampling.resample_to(Daily())
    assert_that(df.shape[0], is_(1))

    day1 = df.loc[df[GeneralisedCols.datetime].dt.day == 10]
    assert_that(day1[GeneralisedCols.mean_iob].values[0], is_(iob))
    assert_that(np.isnan(day1[GeneralisedCols.mean_cob].values[0]))
    assert_that(day1[GeneralisedCols.mean_bg].values[0], is_(bg))

    # check min values
    assert_that(day1[GeneralisedCols.min_iob].values[0], is_(iob))
    assert_that(np.isnan(day1[GeneralisedCols.min_cob].values[0]))
    assert_that(day1[GeneralisedCols.min_bg].values[0], is_(bg))

    # check max values
    assert_that(day1[GeneralisedCols.max_iob].values[0], is_(iob))
    assert_that(np.isnan(day1[GeneralisedCols.max_cob].values[0]))
    assert_that(day1[GeneralisedCols.max_bg].values[0], is_(bg))

    # check std values
    assert_that(day1[GeneralisedCols.std_iob].values[0], is_(0))
    assert_that(np.isnan(day1[GeneralisedCols.std_cob].values[0]))
    assert_that(day1[GeneralisedCols.std_bg].values[0], is_(0))


# check rows don't get deleted if there is one nan in one variate but not the other

def test_some_rounding_values():
    numbers = [1.4556, 1.4, 0.55546, 0.5335]
    expect_rounded = [1.456, 1.400, 0.555, 0.534]
    rounded = [float(Decimal(str(x)).quantize(Decimal('.100'), rounding=ROUND_HALF_UP)) for x in numbers]

    assert_that(rounded[0], is_(expect_rounded[0]))
    assert_that(rounded[1], is_(expect_rounded[1]))
    assert_that(rounded[2], is_(expect_rounded[2]))
    assert_that(rounded[3], is_(expect_rounded[3]))

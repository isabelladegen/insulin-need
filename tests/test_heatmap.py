import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest
from hamcrest import *

from src.configurations import Configuration, GeneralisedCols, Irregular
from src.heatmap import Heatmap, Months, Weekdays, Hours
from src.read_preprocessed_df import ReadPreprocessedDataFrame

irregular_sampling = Irregular()
t1 = datetime(year=2021, month=1, day=10, hour=0, minute=0, tzinfo=timezone.utc)
t2 = datetime(year=2022, month=2, day=1, hour=2, minute=6, tzinfo=timezone.utc)
t3 = datetime(year=2021, month=3, day=30, hour=12, minute=20, tzinfo=timezone.utc)
t4 = datetime(year=2021, month=4, day=11, hour=8, minute=55, tzinfo=timezone.utc)
t5 = datetime(year=2021, month=5, day=10, hour=9, minute=23, tzinfo=timezone.utc)
t6 = datetime(year=2022, month=6, day=10, hour=22, minute=7, tzinfo=timezone.utc)
t7 = datetime(year=2021, month=7, day=15, hour=17, minute=31, tzinfo=timezone.utc)
t8 = datetime(year=2021, month=8, day=20, hour=16, minute=9, tzinfo=timezone.utc)
t9 = datetime(year=2022, month=9, day=10, hour=1, minute=11, tzinfo=timezone.utc)
t10 = datetime(year=2021, month=10, day=3, hour=5, minute=40, tzinfo=timezone.utc)
t11 = datetime(year=2021, month=11, day=5, hour=10, minute=59, tzinfo=timezone.utc)
t12 = datetime(year=2022, month=12, day=18, hour=5, minute=45, tzinfo=timezone.utc)

zip_id1 = '123'
zip_id2 = 'aaa'
zip_id3 = '555'
data = {GeneralisedCols.datetime: [t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12],
        GeneralisedCols.iob: np.arange(0, 6, 0.5),
        GeneralisedCols.cob: np.arange(0, 60, 5.0),
        GeneralisedCols.bg: np.arange(4, 11.2, 0.6),
        GeneralisedCols.system: ['bla'] * 12,
        GeneralisedCols.id: [zip_id1] * 12
        }
df1 = pd.DataFrame(data)

df2 = pd.DataFrame(data)
df2[GeneralisedCols.iob] = df2[GeneralisedCols.iob].multiply(-1)
df2[GeneralisedCols.id] = zip_id2

df3 = pd.DataFrame(data)
df3[GeneralisedCols.cob] = df3[GeneralisedCols.cob].multiply(5.5)
df3[GeneralisedCols.id] = zip_id3

all_dfs = pd.concat([df1, df2, df3], ignore_index=True)


def test_pivots_df_according_to_x_and_y_axis_and_plots_it_in_heatmap_months_weekdays():
    heatmap = Heatmap(df1, irregular_sampling)
    plotted_data = heatmap.plot_heatmap(plot_rows=[GeneralisedCols.iob, GeneralisedCols.cob, GeneralisedCols.bg],
                                        x_axis=Months(),
                                        y_axis=Weekdays())

    assert_that(len(plotted_data), is_(1))  # one zip id
    rows_dict = plotted_data[list(plotted_data.keys())[0]]
    assert_that(len(rows_dict), is_(3))  # iob, cob, bg
    assert_that(list(rows_dict[GeneralisedCols.iob].columns), is_(Months().ticks))
    assert_that(list(rows_dict[GeneralisedCols.iob].index), is_(Weekdays().ticks))


def test_pivots_df_according_to_x_and_y_axis_and_plots_it_in_heatmap_hour_weekdays():
    heatmap = Heatmap(df1, irregular_sampling)
    plotted_data = heatmap.plot_heatmap(plot_rows=[GeneralisedCols.iob, GeneralisedCols.cob, GeneralisedCols.bg],
                                        x_axis=Hours(),
                                        y_axis=Weekdays())

    assert_that(len(plotted_data), is_(1))  # one zip id
    rows_dict = plotted_data[list(plotted_data.keys())[0]]
    assert_that(len(rows_dict), is_(3))  # iob, cob, bg
    assert_that(list(rows_dict[GeneralisedCols.iob].columns), is_(Hours().ticks))
    assert_that(list(rows_dict[GeneralisedCols.iob].index), is_(Weekdays().ticks))


def test_pivots_df_according_to_x_and_y_axis_and_plots_it_in_heatmap_hour_months():
    heatmap = Heatmap(df1, irregular_sampling)
    plotted_data = heatmap.plot_heatmap(plot_rows=[GeneralisedCols.iob, GeneralisedCols.cob, GeneralisedCols.bg],
                                        x_axis=Hours(),
                                        y_axis=Months())

    assert_that(len(plotted_data), is_(1))  # one zip id
    rows_dict = plotted_data[list(plotted_data.keys())[0]]
    assert_that(len(rows_dict), is_(3))  # iob, cob, bg
    assert_that(list(rows_dict[GeneralisedCols.iob].columns), is_(Hours().ticks))
    assert_that(list(rows_dict[GeneralisedCols.iob].index), is_(Months().ticks))


def test_fills_missing_rows_and_columns_in_data_with_nan():
    heatmap = Heatmap(df1, irregular_sampling)
    plotted_data = heatmap.plot_heatmap(plot_rows=[GeneralisedCols.iob, GeneralisedCols.cob, GeneralisedCols.bg],
                                        x_axis=Months(),
                                        y_axis=Weekdays())

    assert_that(len(plotted_data), is_(1))  # one zip id
    rows_dict = plotted_data[list(plotted_data.keys())[0]]
    assert_that(len(rows_dict), is_(3))  # iob, cob, bg
    assert_that(list(rows_dict[GeneralisedCols.iob].columns), is_(Months().ticks))
    assert_that(list(rows_dict[GeneralisedCols.iob].index), is_(Weekdays().ticks))
    assert_that(list(rows_dict[GeneralisedCols.iob].index), is_(Weekdays().ticks))


def test_plots_all_zip_ids_as_columns_by_default():
    heatmap = Heatmap(all_dfs, irregular_sampling)
    plotted_data = heatmap.plot_heatmap(plot_rows=[GeneralisedCols.iob, GeneralisedCols.cob, GeneralisedCols.bg],
                                        x_axis=Months(),
                                        y_axis=Weekdays())

    assert_that(len(plotted_data), is_(3))
    rows_dict = plotted_data[list(plotted_data.keys())[0]]
    assert_that(len(rows_dict), is_(3))  # iob, cob, bg
    assert_that(list(rows_dict[GeneralisedCols.iob].columns), is_(Months().ticks))
    assert_that(list(rows_dict[GeneralisedCols.iob].index), is_(Weekdays().ticks))


def test_plots_only_specified_zip_id_if_provided():
    heatmap = Heatmap(all_dfs, irregular_sampling)
    plotted_data = heatmap.plot_heatmap(plot_rows=[GeneralisedCols.iob, GeneralisedCols.cob, GeneralisedCols.bg],
                                        zip_ids=[zip_id2, zip_id3],
                                        x_axis=Months(),
                                        y_axis=Weekdays())

    assert_that(len(plotted_data), is_(2))
    rows_dict = plotted_data[list(plotted_data.keys())[0]]
    assert_that(len(rows_dict), is_(3))  # iob, cob, bg
    assert_that(list(rows_dict[GeneralisedCols.iob].columns), is_(Months().ticks))
    assert_that(list(rows_dict[GeneralisedCols.iob].index), is_(Weekdays().ticks))


def test_doesnt_show_title_if_set_to_false():
    heatmap = Heatmap(df1, irregular_sampling)
    heatmap.plot_heatmap(plot_rows=[GeneralisedCols.iob, GeneralisedCols.cob, GeneralisedCols.bg],
                         x_axis=Hours(),
                         y_axis=Weekdays(),
                         show_title=False)


@pytest.mark.skipif(not os.path.isdir(Configuration().perid_data_folder), reason="reads real data")
def test_plots_multivariate_heatmap_for_irregular_sampled_file():
    zip_id = '14092221'
    df = ReadPreprocessedDataFrame(sampling=irregular_sampling, zip_id=zip_id).df

    heatmap = Heatmap(df, irregular_sampling)

    heatmap.plot_heatmap(plot_rows=[GeneralisedCols.iob, GeneralisedCols.cob, GeneralisedCols.bg],
                                        x_axis=Months(),
                                        y_axis=Weekdays())

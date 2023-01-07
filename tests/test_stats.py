import os

import pytest

from src.configurations import Configuration, GeneralisedCols, Irregular, Hourly, Daily
from src.read_preprocessed_df import ReadPreprocessedDataFrame
from src.stats import Stats
from src.translate_into_timeseries import TranslateIntoTimeseries, IrregularTimeseries, TimeColumns, DailyTimeseries

irregular_sampling = Irregular()
irregular_raw_df = ReadPreprocessedDataFrame(sampling=irregular_sampling, zip_id='14092221').df

hourly_sampling = Hourly()
hourly_raw_df = ReadPreprocessedDataFrame(sampling=hourly_sampling, zip_id='14092221').df

daily_sampling = Daily()
daily_raw_df = ReadPreprocessedDataFrame(sampling=daily_sampling, zip_id='14092221').df

all_time_columns = [TimeColumns.hour, TimeColumns.week_day, TimeColumns.month, TimeColumns.year]
mean_iob_cob_bg_cols = Configuration.resampled_mean_columns()


@pytest.mark.skipif(not os.path.isdir(Configuration().perid_data_folder), reason="reads real data")
def test_stats_ci_violin_box_plots_for_irregular_time_series():
    ts_description = IrregularTimeseries()
    value_columns = [GeneralisedCols.iob, GeneralisedCols.cob, GeneralisedCols.bg]
    translate = TranslateIntoTimeseries(irregular_raw_df, ts_description, value_columns)

    df = translate.processed_df
    stats = Stats(df, irregular_sampling, ts_description, all_time_columns, value_columns)
    stats.plot_confidence_interval()
    stats.plot_violin_plot()
    stats.plot_box_plot()


@pytest.mark.skipif(not os.path.isdir(Configuration().perid_data_folder), reason="reads real data")
def test_stats_ci_violin_box_plots_for_hourly_sampled_irregular_time_series():
    ts_description = IrregularTimeseries()
    translate = TranslateIntoTimeseries(hourly_raw_df, ts_description, mean_iob_cob_bg_cols)

    df = translate.processed_df
    stats = Stats(df, hourly_sampling, ts_description, all_time_columns, mean_iob_cob_bg_cols)
    stats.plot_confidence_interval()
    stats.plot_violin_plot()
    stats.plot_box_plot()


@pytest.mark.skipif(not os.path.isdir(Configuration().perid_data_folder), reason="reads real data")
def test_stats_ci_violin_box_plots_for_hourly_sampled_daily_time_series():
    ts_description = DailyTimeseries()  # this drops values
    translate = TranslateIntoTimeseries(hourly_raw_df, ts_description, mean_iob_cob_bg_cols)

    df = translate.processed_df
    stats = Stats(df, hourly_sampling, ts_description, all_time_columns, mean_iob_cob_bg_cols)
    stats.plot_confidence_interval()
    stats.plot_violin_plot()
    stats.plot_box_plot()


@pytest.mark.skipif(not os.path.isdir(Configuration().perid_data_folder), reason="reads real data")
def test_stats_ci_violin_box_plots_for_daily_sampled_irregular_time_series():
    ts_description = IrregularTimeseries()  # hours does not make sense
    translate = TranslateIntoTimeseries(daily_raw_df, ts_description, mean_iob_cob_bg_cols)

    df = translate.processed_df
    time_columns = [TimeColumns.week_day, TimeColumns.month, TimeColumns.year]
    stats = Stats(df, daily_sampling, ts_description, time_columns, mean_iob_cob_bg_cols)
    stats.plot_confidence_interval()
    stats.plot_violin_plot()
    stats.plot_box_plot()


@pytest.mark.skipif(not os.path.isdir(Configuration().perid_data_folder), reason="reads real data")
def test_stats_can_deal_with_one_time_column_only():
    ts_description = IrregularTimeseries()  # hours does not make sense
    translate = TranslateIntoTimeseries(hourly_raw_df, ts_description, mean_iob_cob_bg_cols)

    df = translate.processed_df
    time_columns = [TimeColumns.month]
    stats = Stats(df, hourly_sampling, ts_description, time_columns, mean_iob_cob_bg_cols)
    stats.plot_confidence_interval()
    stats.plot_violin_plot()
    stats.plot_box_plot()


@pytest.mark.skipif(not os.path.isdir(Configuration().perid_data_folder), reason="reads real data")
def test_stats_can_deal_with_one_value_row_only():
    value_columns = [GeneralisedCols.mean_iob]
    ts_description = IrregularTimeseries()  # hours does not make sense
    translate = TranslateIntoTimeseries(hourly_raw_df, ts_description, value_columns)

    df = translate.processed_df
    stats = Stats(df, hourly_sampling, ts_description, all_time_columns, value_columns)
    stats.plot_confidence_interval()
    stats.plot_violin_plot()
    stats.plot_box_plot()

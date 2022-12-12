from dataclasses import dataclass

from src.continuous_series import Resolution


@dataclass
class Sampling:
    time_col = 'openaps/enacted/timestamp'
    iob_col = 'openaps/enacted/IOB'
    cob_col = 'openaps/enacted/COB'
    bg_col = 'openaps/enacted/bg'
    max_interval = None  # how frequent readings need per day, 60=every hour, 180=every three hours
    min_days_of_data = None  # how many days of consecutive readings with at least a reading every max interval
    sample_rule = None  # the frequency of the regular time series after resampling
    resolution = None
    length = 0
    description = ''
    x_ticks = []


@dataclass
# Keep all days when there are 24 datapoints with at least a reading every 60min
class DailyTimeseries(Sampling):  # Daily TS
    # k means day, heatmap for days
    max_interval = 60  # how frequent readings need per day, 60=every hour
    min_days_of_data = 1  # how many days of consecutive readings with at least a reading every max interval, 7 = a week
    sample_rule = '1H'  # the frequency of the regular time series after resampling
    resolution = Resolution.Day
    length = 24
    description = 'hours of day (UTC)'
    x_ticks = list(range(0, 24, 2))


@dataclass
# Keep all weeks when there are 7 days of data with each day having at least a reading every 180min
class WeeklyTimeseries(Sampling):  # Weekly TS
    # K means week
    max_interval = 180  # how frequent readings need per day, 60=every hour, 180=every three hours
    min_days_of_data = 1  # how many days of consecutive readings with at least a reading every max interval, 7 = a week
    sample_rule = '1D'  # the frequency of the regular time series after resampling
    resolution = Resolution.Week
    length = 7
    description = 'Day of Week, 0=Monday'
    x_ticks = list(range(0, 7))


@dataclass
class Months(Sampling):
    # months heat map
    max_interval = 180  # how frequent readings need per day, 60=every hour, 180=every three hours
    min_days_of_data = 1  # how many days of consecutive readings with at least a reading every max interval, 7 = a week
    sample_rule = '1D'  # the frequency of the regular time series after resampling
    resolution = Resolution.DaysMonths

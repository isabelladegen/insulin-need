from dataclasses import dataclass


@dataclass
class TimeColumns:  # Daily TS
    day_of_year = "day of year"
    week_of_year = "week"
    month = 'month'
    year = 'year'
    week_day = 'weekday'
    hour = 'hours'

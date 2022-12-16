from dataclasses import dataclass


@dataclass
class Resampling:
    max_gap_in_min = None
    # how frequent readings need to be: one every 60=every hour; one reading 180=every three hours
    sample_rule = None
    # the frequency of the regular time series after resampling: 1H a reading every hour, 1D a reading every day
    description = 'Base'

    @staticmethod
    def csv_file_name():
        return ''


@dataclass
class Irregular(Resampling):
    description = 'irregular'

    @staticmethod
    def csv_file_name():
        return 'irregular_iob_cob_bg.csv'


@dataclass
class Hourly(Resampling):
    max_gap_in_min = 60
    # there needs to be a reading at least every hour for the data points to be resampled for that hour
    sample_rule = '1H'
    description = 'hourly resampled'

    @staticmethod
    def csv_file_name():
        return 'hourly_iob_cob_bg.csv'


@dataclass
class Daily(Resampling):
    max_gap_in_min = 180
    # a reading every three hours for a daily resampling to be created
    sample_rule = '1D'
    description = 'daily resampled'

    @staticmethod
    def csv_file_name():
        return 'daily_iob_cob_bg.csv'

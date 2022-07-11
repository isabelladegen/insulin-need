from datetime import timedelta

from src.configurations import Configuration
from src.read import ReadRecord


# Remove duplicates of rows when the only unique data from row to row is id, created_at and device
def dedub_device_status_dataframes(read_records: [ReadRecord]):
    # Remove duplicated entries and nan if the only unique or non nan data is id, created_at or device
    cols_not_all_nan = list(Configuration().device_status_col_type.keys())
    cols_not_all_nan.remove('id')
    cols_not_all_nan.remove('created_at')
    cols_not_all_nan.remove('device')

    results = []
    for record in read_records:
        rr = ReadRecord()
        if record.df is not None:
            cols_to_consider = [col for col in record.df.columns if col in cols_not_all_nan]
            rr.df = record.df.dropna(how='all', subset=cols_to_consider).drop_duplicates(subset=cols_to_consider)
        rr.zip_id = record.zip_id
        rr.has_no_files = record.has_no_files
        rr.number_of_entries_files = record.number_of_entries_files
        rr.is_android_upload = record.is_android_upload
        rr.calculate_stats('created_at')
        results.append(rr)
    return results


def group_into_consecutive_intervals(df, max_gap_in_min, time_col='time'):
    df.sort_values(by=time_col, inplace=True)
    return df.assign(diff_in_min=(diff := df[time_col].diff()), group=diff.gt(timedelta(minutes=max_gap_in_min)).cumsum())


def number_of_groups_with_more_than_x_items(df, x):
    return (df['group'].value_counts() >= x).sum()


# return how many samples are in the given number of days if sampled at the given minute_interval
# rounds down to closest int
def number_of_interval_in_days(days, minute_interval):
    return int(days * 24 * 60 / minute_interval)


# splits df into smaller dfs of items that are sampled at interval and there's at least min_length continuous intervals
# if value_col is provided than nan's in value col will be dropped to keep the frequency for time and value columns
def continuous_subseries(df, min_length, interval_in_min, time_col, value_col: str = None):
    # group original df into groups that are sampled at least more than the given interval
    if value_col is not None:
        df = df.dropna(subset=[value_col])
    grouped_df = group_into_consecutive_intervals(df, interval_in_min, time_col)

    # get list of group numbers where the value count is at least min_length
    # downsample each group to the interval required, then count the values

    result = []
    for group_idx in grouped_df['group'].unique():
        group_sub_df = grouped_df.loc[grouped_df['group'] == group_idx]
        # skip groups that don't have enough samples
        if group_sub_df.shape[0] < min_length:
            continue
        # resample the group to remove more frequent readings
        resampled = group_sub_df.resample(str(interval_in_min) + 'min', on=time_col).first()
        # skip groups only have enough samples due to being more frequent than interval
        if resampled.shape[0] < min_length:
            continue
        # add the original df for groups with the right number and frequency of samples
        result.append(group_sub_df[list(df.columns)])

    return result

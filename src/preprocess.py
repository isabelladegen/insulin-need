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

from hamcrest import *
import pandas as pd

from src.configurations import GeneralisedCols, OpenAPSConfigs
from tests.helper.ReadRecordBuilder import ReadRecordBuilder

zip_id = '2345678'
data = {'Bla': ['1', '2', '3', 't'],
        'T': [20, 21, 19, 18],
        'F': [10, 4.5, 7.6, 5.8]
        }
no_rows = len(data['F'])
no_columns = len(data)
data_df = pd.DataFrame(data)

data_with_open_aps_col_names = {OpenAPSConfigs.iob.value: ['1', '2', '3', 't', 'c'],
                                OpenAPSConfigs.cob.value: [20, 21, 18, 18, 19],
                                OpenAPSConfigs.bg.value: [10, 4.5, 5.8, 5.8, 7.6],
                                OpenAPSConfigs.datetime.value: ['d', 'p', 'f', 'g', 'l']
                                }


def test_returns_df_with_id_col():
    record = ReadRecordBuilder().with_id(zip_id).with_df(data_df).build()

    df = record.df_with_id()

    assert_that(df.columns, has_item('id'))
    assert_that(df.shape, is_((no_rows, no_columns + 1)))
    assert_that((df['id'] == zip_id).all())


def test_returns_df_only_with_keep_cols_including_id():
    record = ReadRecordBuilder().with_id(zip_id).with_df(data_df).build()

    column_to_keep = ['T', 'Bla']
    df = record.df_with_id(keep_cols=column_to_keep)

    assert_that(df.columns, has_item('id'))
    assert_that(df.columns, has_item(column_to_keep[0]))
    assert_that(df.columns, has_item(column_to_keep[1]))
    assert_that(df.shape, is_((no_rows, len(column_to_keep) + 1)))


def test_inserts_empty_col_if_not_in_the_data():
    record = ReadRecordBuilder().with_id(zip_id).with_df(data_df).build()

    column_to_keep = ['T', 'Does not exist']
    df = record.df_with_id(keep_cols=column_to_keep)

    assert_that(df.columns, has_item(column_to_keep[0]))
    assert_that(df.columns, has_item(column_to_keep[1]))


def test_drops_rows_that_are_all_nan_():
    data_with_nan = {'A': ['1', '2', '3', 't'],
                     'T': [None, None, 19, 18],
                     'F': [None, None, 7.6, 5.8]
                     }

    record = ReadRecordBuilder().with_id(zip_id).with_df(pd.DataFrame(data_with_nan)).build()
    record2 = ReadRecordBuilder().with_id(zip_id).with_df(
        pd.DataFrame({key: data_with_nan[key] for key in ['T', 'F']})).build()

    df_full = record.df_with_id()
    df_tf = record.df_with_id(keep_cols=['T', 'F'])
    df_af = record.df_with_id(keep_cols=['A', 'F'])

    assert_that(df_full.shape[0], is_(4))  # none dropped
    assert_that(df_tf.shape[0], is_(2))  # two rows dropped
    assert_that(df_af.shape[0], is_(4))  # none dropped
    assert_that(record2.df_with_id().shape[0], is_(2))  # two rows dropped


def test_returns_only_unique_rows_for_the_columns_kept():
    # df with duplicates in columns that are being kept
    data_with_duplicates = {'Bla': ['1', '2', '3', 't', 'c'],
                            'T': [20, 21, 18, 18, 19],
                            'F': [10, 4.5, 5.8, 5.8, 7.6]
                            }
    record = ReadRecordBuilder().with_id(zip_id).with_df(pd.DataFrame(data_with_duplicates)).build()

    df1 = record.df_with_id(keep_cols=['T', 'Bla'])  # no duplicates
    df2 = record.df_with_id(keep_cols=['T'])  # 1 duplicate
    df3 = record.df_with_id(keep_cols=['T', 'F'])  # both have a duplicated item same as 2

    assert_that(df1.shape[0], is_(len(data_with_duplicates['F'])))
    assert_that(df2.shape[0], is_(len(set(data_with_duplicates['T']))))
    assert_that(df3.shape[0], is_(len(set(data_with_duplicates['T']))))

    assert_that(df2.index, contains_exactly(0, 1, 2, 3))  # check index gets reset


def test_adds_system_column_to_df():
    record = ReadRecordBuilder().with_id(zip_id).with_df(data_df).build()
    record.system = OpenAPSConfigs.system_name.value

    column_to_keep = ['T']
    df = record.df_with_id(keep_cols=column_to_keep)

    assert_that(df.columns, has_item(GeneralisedCols.system.value))
    assert_that((df[GeneralisedCols.system.value] == OpenAPSConfigs.system_name.value).all())


def test_doesnt_add_system_column_when_not_provided():
    record = ReadRecordBuilder().with_id(zip_id).with_df(data_df).build()

    column_to_keep = ['T']
    df = record.df_with_id(keep_cols=column_to_keep)

    assert_that(len(df.columns), is_(len(column_to_keep) + 1))
    assert_that(df.columns, has_item(GeneralisedCols.id.value))


def test_renames_columns_into_standard_names_when_system_provided():
    record = ReadRecordBuilder().with_id(zip_id).with_df(pd.DataFrame(data_with_open_aps_col_names)).build()
    record.system = OpenAPSConfigs.system_name.value

    df = record.df_with_id()

    assert_that(df.columns, has_item(GeneralisedCols.iob.value))
    assert_that(df.columns, has_item(GeneralisedCols.cob.value))
    assert_that(df.columns, has_item(GeneralisedCols.bg.value))
    assert_that(df.columns, has_item(GeneralisedCols.datetime.value))


def test_doesnt_rename_columns_into_standard_names_when_system_not_provided():
    record = ReadRecordBuilder().with_id(zip_id).with_df(pd.DataFrame(data_with_open_aps_col_names)).build()

    df = record.df_with_id()

    assert_that(df.columns, has_item(OpenAPSConfigs.iob.value))
    assert_that(df.columns, has_item(OpenAPSConfigs.cob.value))
    assert_that(df.columns, has_item(OpenAPSConfigs.bg.value))
    assert_that(df.columns, has_item(OpenAPSConfigs.datetime.value))

# TODO what happens with dateindex?

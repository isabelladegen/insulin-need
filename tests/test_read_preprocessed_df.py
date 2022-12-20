import os

import pytest
from hamcrest import *
from pandas import DatetimeTZDtype

from src.configurations import GeneralisedCols, Configuration, Irregular
from src.read_preprocessed_df import ReadPreprocessedDataFrame


@pytest.mark.skipif(not os.path.isfile(Configuration().flat_preprocessed_file_for(Irregular())),
                    reason="reads real data")
def test_reads_flat_file_for_irregular_sampling():
    processed_df = ReadPreprocessedDataFrame(sampling=Irregular())
    df = processed_df.df

    assert_that(df.columns, has_item(GeneralisedCols.iob))
    assert_that(df.columns, has_item(GeneralisedCols.cob))
    assert_that(df.columns, has_item(GeneralisedCols.bg))
    assert_that(df.columns, has_item(GeneralisedCols.datetime))
    assert_that(df.columns, has_item(GeneralisedCols.system))
    assert_that(df.columns, has_item(GeneralisedCols.id))

    assert_that(len(df[GeneralisedCols.id].unique()), is_(116))

    assert_that(df[GeneralisedCols.iob].dtype, is_('float64'))
    assert_that(df[GeneralisedCols.cob].dtype, is_('float64'))
    assert_that(df[GeneralisedCols.bg].dtype, is_('float64'))
    assert_that(df[GeneralisedCols.system].dtype, is_(object))
    assert_that(df[GeneralisedCols.id].dtype, is_(object))
    assert_that(df[GeneralisedCols.datetime].dtype, is_(DatetimeTZDtype('ns', 'UTC')))


@pytest.mark.skipif(not os.path.isdir(Configuration().perid_data_folder), reason="reads real data")
def test_reads_persons_file_for_irregular_sampling():
    zip_id = '14092221'
    df = ReadPreprocessedDataFrame(sampling=Irregular(), zip_id=zip_id).df

    assert_that(df.columns, has_item(GeneralisedCols.iob))
    assert_that(df.columns, has_item(GeneralisedCols.cob))
    assert_that(df.columns, has_item(GeneralisedCols.bg))
    assert_that(df.columns, has_item(GeneralisedCols.datetime))
    assert_that(df.columns, has_item(GeneralisedCols.system))
    assert_that(df.columns, has_item(GeneralisedCols.id))

    assert_that(df[GeneralisedCols.id].unique()[0], is_(zip_id))

    assert_that(df[GeneralisedCols.iob].dtype, is_('float64'))
    assert_that(df[GeneralisedCols.cob].dtype, is_('float64'))
    assert_that(df[GeneralisedCols.bg].dtype, is_('float64'))
    assert_that(df[GeneralisedCols.system].dtype, is_(object))
    assert_that(df[GeneralisedCols.id].dtype, is_(object))
    assert_that(df[GeneralisedCols.datetime].dtype, is_(DatetimeTZDtype('ns', 'UTC')))

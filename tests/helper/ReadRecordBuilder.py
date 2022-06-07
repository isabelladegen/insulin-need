# builds read records object for unit testing
import pandas as pd

from src.read import ReadRecord
from tests.helper.BgDfBuilder import BgDfBuilder


class ReadRecordBuilder:
    def __init__(self):
        self.df = BgDfBuilder().build()
        self.id = '123456789'

    def build(self):
        record = ReadRecord()
        record.zip_id = self.id
        record.df = self.df
        return record

    def with_id(self, id: str):
        self.id = id
        return self

    def with_df(self, df: pd.DataFrame):
        self.df = df
        return self

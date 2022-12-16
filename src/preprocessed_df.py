import pandas as pd

from src.configurations import Configuration, GeneralisedCols
from src.helper import preprocessed_file_for
from src.resampling import Resampling


class PreprocessedDataFrame:
    """
        Class for reading preprocessed csv files into pandas df and configure sampling resolution for other classes to use

        Attributes
        ----------
            df: pd.DataFrame
                dataframe read

        Methods
        -------
    """

    def __init__(self, sampling: Resampling, zip_id: str = None):
        """
            Parameters
            ----------
            sampling : Resampling
                What the time series are resampled as this defines which file is being read

            zip_id : str
                Id for data to read; default None which reads the flat file version with all people
        """
        self.__sampling = sampling
        self.__zip_id = zip_id
        self.__config = Configuration()
        self.df = self.__read_df()

    def __read_df(self):
        if self.__zip_id:
            file = preprocessed_file_for(self.__config.perid_data_folder, self.__zip_id, self.__sampling)
        else:
            file = self.__config.flat_preprocessed_file_for(self.__sampling)
        df = pd.read_csv(file,
                         dtype={GeneralisedCols.id: str, GeneralisedCols.system: str},
                         date_parser=lambda c: pd.to_datetime(c, utc=True, errors="raise"),
                         parse_dates=[GeneralisedCols.datetime]
                         )
        return df

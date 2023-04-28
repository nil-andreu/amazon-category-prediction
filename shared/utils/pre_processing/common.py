import typing

import pandas as pd

from shared.utils.constants import INITIAL_COLUMNS_DROP, INITIAL_COLUMNS_LIST_TYPE


class CommonPreProcessing:
    def __init__(
        self,
        df,
        list_columns: typing.List[str] = INITIAL_COLUMNS_LIST_TYPE,
        drop_columns: typing.List[str] = INITIAL_COLUMNS_DROP,
    ):
        self._df: pd.DataFrame = df
        self._drop_columns: typing.List[str] = drop_columns
        self._list_columns: typing.List[str] = list_columns

    def pre_process(self):
        self.drop_columns()
        self.handle_nan_columns()
        self.convert_list_to_tuple()
        self.remove_duplicated_rows()
        return self._df

    def drop_columns(self):
        self._df = self._df.drop(self._drop_columns, axis=1)
        return self._df

    def handle_nan_columns(self):
        for column in self._df.columns.values:
            self._df[column] = self._df[column].apply(
                lambda x: self.replace_empty_value_with_none(x)
            )
        return self._df

    def convert_list_to_tuple(
        self,
    ):
        for list_column in self._list_columns:
            self._df[list_column] = self._df[list_column].apply(
                lambda x: self.convert_tuple(x)
            )

        return self._df

    def remove_duplicated_rows(self):
        self._df = self._df.drop_duplicates().reset_index(drop=True)
        return self._df

    @staticmethod
    def replace_empty_value_with_none(
        value: typing.Union[list, str]
    ) -> typing.Optional[typing.Union[list, str]]:

        return value or None

    @staticmethod
    def convert_tuple(value: typing.Optional[typing.List[str]]):
        return tuple(value) if value else value

    @property
    def df(self):
        return self._df

    @df.setter
    def df(self, df):
        self._df = df

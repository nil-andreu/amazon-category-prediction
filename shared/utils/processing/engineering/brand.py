import typing

import pandas as pd


class BrandProcessing:
    """
    In this brand processing, we are going to count the occurences of each brand
    and group non-popular brands into 'Other' group.
    """

    def __init__(self, df: pd.DataFrame):
        self._df: pd.DataFrame = df
        self._brand_counter: typing.Dict[str, int] = dict()
        self._brand_counter_processed: typing.Dict[str, int] = {"Other": 0}
        self._brand_other_names_list: typing.List[str] = []

    def get_brand_counter(self, min_observations: int = 20):
        self.create_brand_counter()
        self.classify_other_brand(min_observations)
        return self._brand_counter_processed, self._brand_other_names_list

    def create_brand_counter(self) -> dict:
        """Create a map of how many times each brand appears"""
        self._brand_counter = self._df.brand.value_counts().to_dict()
        return self._brand_counter

    def classify_other_brand(self, min_observations):
        """For those that have less than certain threshold of occurences, will be inside of group Other"""
        for idx, value in self._brand_counter.items():
            if value > min_observations:
                self._brand_counter_processed[idx] = value
            else:
                # Keep track of which are the brands inside of 'Other' class
                self._brand_other_names_list.append(idx)
                self._brand_counter_processed["Other"] += value

        return self._brand_counter_processed, self._brand_other_names_list

    def get_df_filtered(self, brand_name: str) -> pd.DataFrame:
        """Get the dataframe that only corresponds for that particular brand"""
        if brand_name == "Other":
            df_filtered = self._df[self._df.brand.isin(self._brand_other_names_list)]

        else:
            df_filtered = self.get_df_filtered_by_brand_name(brand_name)

        return df_filtered

    def get_df_filtered_by_brand_name(self, brand_name: str) -> pd.DataFrame:
        return self._df[self._df.brand == brand_name]

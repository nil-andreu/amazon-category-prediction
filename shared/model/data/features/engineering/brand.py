import typing
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

from shared.utils.processing.engineering.brand import BrandProcessing


class BrandMapCategoryProbabilities(BrandProcessing):
    def __init__(self, df: pd.DataFrame, folder_path: str = "../shared/data/brand_map"):
        super().__init__(df)

        self._df = df
        self._brand_df = None
        self._folder_path = folder_path

    def get_brand_df(self) -> pd.DataFrame:
        """Get matrix of brand probabilities for the main categories"""
        self.get_brand_counter()

        with tqdm(total=len(self._brand_counter_processed)) as pbar:
            for brand_name, brand_total in self._brand_counter_processed.items():
                # In the case it is other
                df_filtered = self.get_df_filtered(brand_name)

                probability_brand = self.get_probability_brand(df_filtered, brand_total)

                self._brand_df = pd.concat(
                    [
                        self._brand_df,
                        pd.DataFrame(probability_brand, index=[brand_name]),
                    ],
                    axis=0,
                )

                pbar.update(1)

        # The ones not found have NaN
        self._brand_df.fillna(0.0, inplace=True)

        return self._brand_df

    def save_brand_df(self):
        date_time = datetime.today().strftime("%Y-%m-%d_%H:%M:%S")
        self._brand_df.to_csv(
            f"{self._folder_path}/{date_time}.csv", index=True, header=True
        )

    @staticmethod
    def get_probability_brand(
        df_filtered: pd.DataFrame,
        brand_observations: int,
    ) -> typing.Dict[str, float]:
        probability_brand = {}
        for category, value in df_filtered.main_cat.value_counts().to_dict().items():
            probability_brand[category] = round((value / brand_observations), 2)

        return probability_brand


class Brand:
    def __init__(self, brand_df: pd.DataFrame):
        self._brand_df: pd.DataFrame = brand_df
        self._brand_probabilities: dict = dict()

    def get_feature(self, brand_name: str):
        try:
            self._brand_probabilities = self._brand_df.loc[brand_name].to_dict()
        except (IndexError, KeyError):
            self._brand_probabilities = self._brand_df.loc["Other"].to_dict()

        return np.array(list(self._brand_probabilities.values()))

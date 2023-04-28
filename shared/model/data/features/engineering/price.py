import typing
from datetime import datetime

import numpy as np
import pandas as pd

from shared.utils.common import min_max_scale
from shared.utils.constants import PRICE_MAXIMUM_VALUE, PRICE_MINIMUM_VALUE
from shared.utils.processing.engineering.brand import BrandProcessing


class BrandMapPriceMedian(BrandProcessing):
    def __init__(self, df: pd.DataFrame, folder_path: str = "../shared/data/price_map"):
        super().__init__(df)
        self._df = df
        self._brand_median_price_df = None
        self._folder_path = folder_path

    def get_price_df(self):
        self.create_brand_median_price_df()
        self.save_price_df()
        return self._brand_median_price_df

    def create_brand_median_price_df(self):
        _, other_list = self.get_brand_counter()

        self._brand_median_price_df = self._df.copy()
        self._brand_median_price_df.brand = self._brand_median_price_df.brand.apply(
            lambda x: "Other" if self._brand_counter_processed.get(x, False) else x
        )

        self.group_brand_median()
        self.create_df_group_brand_median()

        # If still nulls, median from all
        self._brand_median_price_df.fillna(
            self._brand_median_price_df.price_median.median(), inplace=True
        )

        return self._brand_median_price_df

    def group_brand_median(self):
        self._brand_median_price_df = self._brand_median_price_df.groupby("brand")[
            "price"
        ].median()

        return self._brand_median_price_df

    def create_df_group_brand_median(self):
        self._brand_median_price_df = pd.DataFrame(
            self._brand_median_price_df,
        )

        self._brand_median_price_df.columns = ["price_median"]

        return self._brand_median_price_df

    def save_price_df(self):
        date_time = datetime.today().strftime("%Y-%m-%d_%H:%M:%S")
        self._brand_median_price_df.to_csv(
            f"{self._folder_path}/{date_time}.csv", index=True, header=True
        )


class Price:
    def __init__(
        self,
        price_df: pd.DataFrame,
        minimum_log_price: float = PRICE_MINIMUM_VALUE,
        maximum_log_price: float = PRICE_MAXIMUM_VALUE,
    ):
        self._price_log: float = 0.0
        self._price_log_scaled: float = 0.0

        self._price_df: pd.DataFrame = price_df

        self._minimum_log_price = minimum_log_price
        self._maximum_log_price = maximum_log_price

    def get_feature(self, price, brand_name: typing.Optional[str]):
        if price is np.nan or not price:
            price = self.infer_price_nan(brand_name)

        price_log = self.apply_log_price(price)
        price_log_scaled = min_max_scale(
            price_log,
            max_value=self._minimum_log_price,
            min_value=self._maximum_log_price,
        )

        return price_log_scaled if str(price) != "nan" else 0.0

    def infer_price_nan(self, brand_name):
        try:
            price = self.get_price_df(brand_name)

        except KeyError:
            price = self.get_price_df("Other")

        return price

    def get_price_df(self, brand_name: str):
        return self._price_df.loc[brand_name, "price_median"]

    @staticmethod
    def apply_log_price(price):
        price_log = np.log(price + 1)
        return price_log

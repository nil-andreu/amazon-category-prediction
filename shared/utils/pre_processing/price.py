import pandas as pd

from shared.utils.re_utils import remove_dollar


class PricePreProcessing:
    def __init__(self, df: pd.DataFrame):
        self._df = df

    def pre_process(self):
        """
        Convert a price of '$200.50' to 200.5
        This PricePreProcessing goes after the CommonPreProcessing.
        """
        self._df.price = self._df.price.apply(lambda x: self.apply_dollar_removal(x))

        self._df.price = self._df.price.apply(
            lambda x: self.apply_convert_price_to_float(x)
        )

        return self._df

    def apply_dollar_removal(self, price):
        try:
            return self.remove_dollar_from_price(price)
        except ValueError:
            return None

    def apply_convert_price_to_float(self, price):
        try:
            return self.convert_price_to_float(price)
        except ValueError:
            return None

    @staticmethod
    def remove_dollar_from_price(price: str):
        return remove_dollar(price) if price else None

    @staticmethod
    def convert_price_to_float(price: str):
        return float(price) if price else None

import pandas as pd

from shared.utils.pre_processing.common import CommonPreProcessing
from shared.utils.pre_processing.price import PricePreProcessing


class TestCommonPreProcessing:
    @classmethod
    def setup_class(cls):
        cls.common_preprocessing = CommonPreProcessing(pd.DataFrame())

    """First test then general properties"""

    def test_df_property_get(self):
        assert self.common_preprocessing.df.empty

    def test_df_property_set(self):
        common_pre_processing = CommonPreProcessing(pd.DataFrame())
        details = {
            "Name": ["Ankit", "Aishwarya", "Shaurya", "Shivangi"],
            "Age": [23, 21, 22, 21],
            "University": ["BHU", "JNU", "DU", "BHU"],
        }
        common_pre_processing.df = pd.DataFrame(details)
        assert common_pre_processing.df.equals(pd.DataFrame(details))

    """Second we are going to test static methods, where we can use a general CPP"""

    def test_replace_empty_value_with_none(self):
        function = self.common_preprocessing.replace_empty_value_with_none
        assert function("") is None
        assert function([]) is None
        assert function("data") == "data"
        assert function(["data"]) == ["data"]

    def test_convert_tuple(self):
        assert self.common_preprocessing.convert_tuple(["dummy1"]) == ("dummy1",)
        assert self.common_preprocessing.convert_tuple(
            ["dummy1", "dummy2", "dummy3"]
        ) == ("dummy1", "dummy2", "dummy3")
        assert self.common_preprocessing.convert_tuple(None) is None

    """Finally, we test general methods"""

    def test_drop_columns(self):
        details_initial = {
            "Name": ["Ankit", "", "Shaurya", "Shivangi"],
            "Age": [23, None, 22, 21],
            "University": ["BHU", "", "DU", ""],
        }

        common_pre_processing = CommonPreProcessing(
            pd.DataFrame(details_initial), drop_columns=["Age", "University"]
        )

        details_expected = {
            "Name": ["Ankit", "", "Shaurya", "Shivangi"],
        }

        assert common_pre_processing.drop_columns().equals(
            pd.DataFrame(details_expected)
        )

    def test_handle_nan_columns(self):
        details_initial = {
            "Name": ["Ankit", "", "Shaurya", "Shivangi"],
            "Age": [23, None, 22, 21],
            "University": ["BHU", "", "DU", ""],
        }

        common_pre_processing = CommonPreProcessing(pd.DataFrame(details_initial))

        details_expected = {
            "Name": ["Ankit", None, "Shaurya", "Shivangi"],
            "Age": [23, None, 22, 21],
            "University": ["BHU", None, "DU", None],
        }

        assert common_pre_processing.handle_nan_columns().equals(
            pd.DataFrame(details_expected)
        )

    def test_convert_list_to_tuple(self):
        details_initial = {
            "Name": [["Ankit"], ["Aishwarya"], ["Shaurya"], ["Shivangi"]],
        }

        common_pre_processing = CommonPreProcessing(
            pd.DataFrame(details_initial), list_columns=["Name"]
        )

        details_expected = {
            "Name": [("Ankit",), ("Aishwarya",), ("Shaurya",), ("Shivangi",)],
        }

        common_pre_processing.df = pd.DataFrame(details_initial)
        assert common_pre_processing.convert_list_to_tuple().equals(
            pd.DataFrame(details_expected)
        )

    def test_remove_duplicated_rows(self):
        details_initial = {
            "Name": ["Ankit", "DUMMY", "Shaurya", "DUMMY"],
            "Age": [23, None, 22, None],
            "University": ["BHU", "AU", "DU", "AU"],
        }

        common_pre_processing = CommonPreProcessing(pd.DataFrame(details_initial))

        details_expected = {
            "Name": ["Ankit", "DUMMY", "Shaurya"],
            "Age": [23, None, 22],
            "University": ["BHU", "AU", "DU"],
        }

        common_pre_processing.df = pd.DataFrame(details_initial)
        assert common_pre_processing.remove_duplicated_rows().equals(
            pd.DataFrame(details_expected)
        )


class TestPricePreProcessing:
    @classmethod
    def setup_class(cls):
        cls.price_pre_processing = PricePreProcessing(pd.DataFrame())

    def test_remove_dollar_from_price(self):
        assert self.price_pre_processing.remove_dollar_from_price("$300.4") == "300.4"
        assert self.price_pre_processing.remove_dollar_from_price("$1.40") == "1.40"
        assert self.price_pre_processing.remove_dollar_from_price(None) is None

    def test_apply_dollar_removal(self):
        assert self.price_pre_processing.apply_dollar_removal(0) is None
        assert self.price_pre_processing.apply_dollar_removal(0.0) is None

    def test_convert_price_to_float(self):
        assert self.price_pre_processing.convert_price_to_float("200") == 200.0
        assert self.price_pre_processing.convert_price_to_float("200.5") == 200.5

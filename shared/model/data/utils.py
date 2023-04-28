import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.append("../")

from shared.model.data.features.embedding import (
    DescriptionEmbedding,
    FeatureEmbedding,
    ImageGroupEmbedding,
    TitleEmbedding,
)
from shared.model.data.features.engineering import (
    AlsoBuyRecommendation,
    AlsoViewRecommendation,
    Brand,
    BrandMapCategoryProbabilities,
    BrandMapPriceMedian,
    Price,
)
from shared.utils.constants import SHARED_DATA_FOLDER
from shared.utils.loaders.df import df_into_csv_gzip, json_gzip_into_df
from shared.utils.pre_processing import CommonPreProcessing, PricePreProcessing


def get_shared_data_folder_children(shared_data_folder: str = SHARED_DATA_FOLDER):
    SHARED_DATA_FOLDER_IMAGES = shared_data_folder + "/images"
    SHARED_DATA_FOLDER_BRAND = shared_data_folder + "/brand_map"
    SHARED_DATA_FOLDER_PRICE = shared_data_folder + "/price_map"

    os.makedirs(SHARED_DATA_FOLDER_IMAGES, exist_ok=True)
    os.makedirs(SHARED_DATA_FOLDER_BRAND, exist_ok=True)
    os.makedirs(SHARED_DATA_FOLDER_PRICE, exist_ok=True)

    return SHARED_DATA_FOLDER_IMAGES, SHARED_DATA_FOLDER_BRAND, SHARED_DATA_FOLDER_PRICE


def pre_process_initial_file(
    shared_data_folder: str = SHARED_DATA_FOLDER,
    complete_file_name: str = "amz_products_small.jsonl.gz",
    output_file_name_append: str = "_pre_processed",
    output_file_extension: str = ".csv.gz",
):

    # Need to check if folder already exists
    os.makedirs(shared_data_folder, exist_ok=True)

    # Check if file already pre-processed
    file_name = complete_file_name.split(".")[0]

    complete_file_path_output: str = (
        shared_data_folder
        + "/"
        + file_name
        + output_file_name_append
        + output_file_extension
    )
    if os.path.isfile(complete_file_path_output):
        df = pd.read_csv(complete_file_path_output, compression="gzip")
        return df

    print("Pre-Processing Initial Data")
    df = json_gzip_into_df(shared_data_folder + "/" + complete_file_name)
    df = CommonPreProcessing(df).pre_process()
    df = PricePreProcessing(df).pre_process()

    df_into_csv_gzip(df, SHARED_DATA_FOLDER + "/" + file_name + output_file_name_append)

    return df


def get_map_engineering_features(
    shared_data_folder_price: str,
    shared_data_folder_brand: str,
    df: pd.DataFrame = None,
    create_new_price_map: bool = True,
    create_new_brand_map: bool = True,
):
    """
    Initializing the maps for the engineering features:
    - brand
    - price

    :param df: dataframe with information of all products
    :param create_new_map: if we want to create new maps or use the last ones created
    """

    # Check the path exists
    os.makedirs(shared_data_folder_price, exist_ok=True)
    os.makedirs(shared_data_folder_brand, exist_ok=True)

    shared_data_folder_price_values = sorted(os.listdir(shared_data_folder_price))
    shared_data_folder_brand_values = sorted(os.listdir(shared_data_folder_brand))

    # Get the price_df
    if create_new_price_map or shared_data_folder_price_values == 0:
        if df.empty or isinstance(df, type(None)):
            print("We do not have the data")

        print("Creating Brand Map Price Median")
        brand_map_price_median = BrandMapPriceMedian(df, shared_data_folder_price)
        price_df = brand_map_price_median.get_price_df()
    else:
        # Read last created price_df
        price_df = pd.read_csv(
            shared_data_folder_price + "/" + shared_data_folder_price_values[-1],
            index_col=[0],
        )

    # Get the brand_df
    if create_new_brand_map or shared_data_folder_brand_values == 0:
        if df.empty or isinstance(df, type(None)):
            print("We do not have the data")

        print("Creating Brand Map Category Probabilities")
        brand_map_category_probabilities = BrandMapCategoryProbabilities(
            df, shared_data_folder_brand
        )
        brand_df = brand_map_category_probabilities.get_brand_df()
        brand_map_category_probabilities.save_brand_df()
    else:
        # Read last created brand_df
        brand_df = pd.read_csv(
            shared_data_folder_brand + "/" + shared_data_folder_brand_values[-1],
            index_col=[0],
        )

    return price_df, brand_df


class EngineeringFeatures:
    def __init__(self, price_df: pd.DataFrame, brand_df: pd.DataFrame):
        self._price_df = price_df
        self._brand_df = brand_df

        # Initiailizing the features
        self._price_feature = Price(self._price_df)
        self._brand_feature = Brand(self._brand_df)
        self._also_buy_feature = AlsoBuyRecommendation()
        self._also_view_feature = AlsoViewRecommendation()

    def get_features(self, price, brand_name, also_buy, also_view):
        price = self._price_feature.get_feature(price, brand_name)
        brand = self._brand_feature.get_feature(brand_name)
        also_buy = self._also_buy_feature.get_feature(also_buy)
        also_view = self._also_view_feature.get_feature(also_view)

        return price, brand, also_buy, also_view


class EmbeddingFeatures:
    def __init__(self, shared_data_folder_images: str):
        self._image_embedding = ImageGroupEmbedding(shared_data_folder_images)
        self._description_embedding = DescriptionEmbedding()
        self._feature_embedding = FeatureEmbedding()
        self._title_embedding = TitleEmbedding()

    async def get_features(self, image, description, feature, title):
        image = await self._image_embedding.get_image_group_embedding(image)
        image = self.handle_non_tensors(image, 768)

        description = self._description_embedding.get_text_group_embedding(description)
        description = self.handle_non_tensors(description, 768)

        # If we only have one value, should pass it like a tuple string
        try:
            feature = self._feature_embedding.get_text_group_embedding(feature)
        except SyntaxError:
            feature = self._feature_embedding.get_text_group_embedding(
                f"""('{feature}',)"""
            )
        feature = self.handle_non_tensors(feature, 384)

        title = self._title_embedding.get_text_group_embedding(f"""('{title}',)""")
        title = self.handle_non_tensors(title, 384)

        return image, description, feature, title

    @staticmethod
    def handle_non_tensors(value, length: int):
        """If we do not have a certain value, we will return an array of 0 for the expected length for that feature"""
        if not torch.is_tensor(value):
            value = np.zeros(length)

        return value

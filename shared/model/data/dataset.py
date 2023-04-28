import numpy as np
import pandas as pd
import torch
from asgiref.sync import async_to_sync
from torch.utils.data import Dataset

from shared.model.data.utils import (
    EmbeddingFeatures,
    EngineeringFeatures,
    get_map_engineering_features,
    get_shared_data_folder_children,
    pre_process_initial_file,
)


class ProductsDataset(Dataset):
    def __init__(
        self,
        shared_data_folder: str,
        complete_file_name: str,
        create_new_price_map: bool = False,
        create_new_brand_map: bool = False,
    ):
        self._complete_file_name = complete_file_name
        self._shared_data_folder = shared_data_folder

        (
            self._shared_data_folder_images,
            self._shared_data_folder_brand,
            self._shared_data_folder_price,
        ) = get_shared_data_folder_children(self._shared_data_folder)

        # Pre-Processing dataframe
        self._df = pre_process_initial_file(shared_data_folder, complete_file_name)
        self._df_main_cat_dummies = pd.get_dummies(self._df["main_cat"]).astype(int)
        self._main_cat_values = self._df_main_cat_dummies.columns.values

        # Creating maps
        self._price_df, self._brand_df = get_map_engineering_features(
            self._shared_data_folder_price,
            self._shared_data_folder_brand,
            self._df,
            create_new_price_map,
            create_new_brand_map,
        )

        # Initializing the feature creators
        self._engineering_features = EngineeringFeatures(self._price_df, self._brand_df)
        self._embedding_features = EmbeddingFeatures(self._shared_data_folder_images)

        self._get_engineering_features = self._engineering_features.get_features
        self._get_embedding_features_sync = async_to_sync(
            self._embedding_features.get_features
        )  # NOTE: Getitem cannot run async functions

    def __len__(self):
        return len(self._df)

    def __getitem__(self, idx):
        row = self._df.iloc[idx, :]

        price, brand, also_buy, also_view = self._get_engineering_features(
            row["price"], row["brand"], row["also_buy"], row["also_view"]
        )

        image, description, feature, title = self._get_embedding_features_sync(
            row["image"],
            row["description"],
            row["feature"],
            row["title"],
        )

        target = np.array(self._df_main_cat_dummies.iloc[idx, :])
        input, target = torch.from_numpy(
            np.array(
                [
                    price,
                    also_buy,
                    also_view,
                    *brand,
                    *image,
                    *description,
                    *feature,
                    *title,
                ]
            )
        ), torch.from_numpy(target)

        input, target = input.to(torch.float32), target.to(torch.float32)
        return input, target

    @property
    def main_cat_values(self):
        return self._main_cat_values


if __name__ == "__main__":
    products_dataset = ProductsDataset("../data", "amz_products_small.jsonl.gz")
    for [
        price,
        brand,
        also_buy,
        also_view,
        image,
        description,
        feature,
        title,
    ], target in products_dataset:
        print(
            [
                price,
                brand,
                also_buy,
                also_view,
                image,
                description,
                feature,
                title,
            ]
        )
        print(target)
        break

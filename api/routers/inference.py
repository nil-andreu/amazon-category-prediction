import typing

import numpy as np
import torch
from fastapi import APIRouter

from api.utils.model import load_mlp_model
from shared.model.data.utils import (
    EmbeddingFeatures,
    EngineeringFeatures,
    get_map_engineering_features,
)
from shared.utils.constants import MAIN_CAT, SHARED_DATA_FOLDER, SHARED_PATH

inference_router = APIRouter(prefix="/inference")


@inference_router.post("/", tags=["main_cat_inference"])
async def get_inference_main_cat(
    brand: str,
    description: str,
    price: typing.Optional[float] = None,
    also_buy: typing.Optional[str] = None,
    also_view: typing.Optional[str] = None,
    title: typing.Optional[str] = None,
    image: typing.Optional[str] = None,
    feature: typing.Optional[str] = None,
):
    mlp_model = load_mlp_model(SHARED_PATH)
    price_df, brand_df = get_map_engineering_features(
        SHARED_DATA_FOLDER + "/price_map",
        SHARED_DATA_FOLDER + "/brand_map",
        None,
        False,
        False,
    )
    engineering_features = EngineeringFeatures(price_df, brand_df)
    embedding_features = EmbeddingFeatures(SHARED_DATA_FOLDER + "/images")

    print("Engineering Initial Features")
    price, brand, also_buy, also_view = engineering_features.get_features(
        price, brand, also_buy, also_view
    )

    image, description, feature, title = await embedding_features.get_features(
        image, description, feature, title
    )

    # WOULD NEED TODO
    # If probability for a certain category given a brand is >90% for example, directly predict this category
    # Would make the code a lot faster

    inputs = torch.from_numpy(
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
    )

    _, output_probs = mlp_model(torch.tensor(inputs, dtype=torch.float32))

    return {
        MAIN_CAT[idx]: round(output_prob, 2)
        for idx, output_prob in enumerate(output_probs.tolist())
    }

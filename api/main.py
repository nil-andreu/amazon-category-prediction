import os

import pandas as pd
import uvicorn
from fastapi import FastAPI

from api.routers.inference import inference_router
from shared.model.data.utils import (
    get_map_engineering_features,
    pre_process_initial_file,
)
from shared.utils.constants import SHARED_DATA_FOLDER

app = FastAPI()
app.include_router(inference_router)


@app.on_event("startup")
async def initial_data():
    # On app startup, check files for price_df and brand_df exist. If not, should create them
    complete_file_path_output = (
        SHARED_DATA_FOLDER + "/" + "amz_products_small_pre_processed.csv.gz"
    )
    if not os.path.isfile(complete_file_path_output):
        df = pre_process_initial_file(SHARED_DATA_FOLDER)
    else:
        df = pd.read_csv(complete_file_path_output, compression="gzip")

    create_price_map = not os.path.isdir(SHARED_DATA_FOLDER + "/price_map") or not len(
        os.listdir(SHARED_DATA_FOLDER + "/price_map")
    )
    create_brand_map = not os.path.isdir(SHARED_DATA_FOLDER + "/brand_map") or not len(
        os.listdir(SHARED_DATA_FOLDER + "/brand_map")
    )
    if create_price_map or create_brand_map:
        _ = get_map_engineering_features(
            SHARED_DATA_FOLDER + "/price_map",
            SHARED_DATA_FOLDER + "/brand_map",
            df,
            create_price_map,
            create_brand_map,
        )

    # For storing images
    os.makedirs(SHARED_DATA_FOLDER + "/images", exist_ok=True)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)

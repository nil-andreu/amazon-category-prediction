import asyncio
import typing

import pandas as pd
from tqdm import tqdm

from shared.utils.images.extractor import ImageExtractor


class ImageExtractorDf(ImageExtractor):
    def __init__(
        self,
        df: pd.DataFrame,
    ):
        super().__init__()
        self._df = df

    async def extract_images_df(
        self, start: typing.Optional[int] = 0, size: typing.Optional[int] = None
    ):
        amount_extracted: int = 0
        with tqdm(total=len(self._df[start:])) as pbar:
            for _, row in self._df[start:].iterrows():
                image_urls: str = row["image"]

                await self.extraction_and_save_image_urls(image_urls)

                pbar.update(1)
                amount_extracted += 1
                if size and amount_extracted > size:
                    return


if __name__ == "__main__":
    df = pd.read_csv(
        "../shared/data/amz_products_small_pre_processed.csv.gz", compression="gzip"
    )

    image_extractor = ImageExtractorDf(df)

    loop = asyncio.get_event_loop()
    loop.run_until_complete(image_extractor.extract_images_df(start=35159))

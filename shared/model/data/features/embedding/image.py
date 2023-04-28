import typing

import torch
from transformers import AutoFeatureExtractor, AutoModel

from shared.utils.processing.embedding.image import ImageGroupProcessing


class ImageEmbedding:
    def __init__(self, model_ckpt: str = "nateraw/vit-base-beans"):
        self._model = AutoModel.from_pretrained(model_ckpt)
        self._extractor = AutoFeatureExtractor.from_pretrained(model_ckpt)

    def get_embedding(self, images: typing.List[typing.Any]):
        new_batch = {"pixel_values": images}
        with torch.no_grad():
            embeddings = self._model(**new_batch).last_hidden_state[:, 0].cpu()

        return embeddings


class ImageGroupEmbedding(ImageEmbedding):
    def __init__(self, source_folder: str):
        super().__init__()
        self._source_folder = source_folder

    async def get_image_group_embedding(self, image_group: str):
        image_group_processing = ImageGroupProcessing(
            image_group, source_folder=self._source_folder
        )
        image_batch_transformed = (
            await image_group_processing.get_image_group_processed()
        )

        if not image_batch_transformed:
            return None

        image_embeddings = self.get_embedding(torch.stack(image_batch_transformed))

        return self.mean_embeddings(image_embeddings)

    @staticmethod
    def mean_embeddings(embeddings):
        return torch.mean(embeddings, dim=0)


if __name__ == "__main__":
    import asyncio

    image_group_text_example = """('https://images-na.ssl-images-amazon.com/images/I/412r3BU7ChL._SX38_SY50_CR,0,0,38,50_.jpg', 'https://images-na.ssl-images-amazon.com/images/I/41MBe%2BaOI0L._SX38_SY50_CR,0,0,38,50_.jpg', 'https://images-na.ssl-images-amazon.com/images/I/51DzpLO1LxL._SX38_SY50_CR,0,0,38,50_.jpg', 'https://images-na.ssl-images-amazon.com/images/I/416akjHkaIL._SX38_SY50_CR,0,0,38,50_.jpg', 'https://images-na.ssl-images-amazon.com/images/I/41Mc5WJAWkL._SX38_SY50_CR,0,0,38,50_.jpg', 'https://images-na.ssl-images-amazon.com/images/I/41eS-1oMa3L._SX38_SY50_CR,0,0,38,50_.jpg', 'https://images-na.ssl-images-amazon.com/images/I/41wjlGNGDoL._SX38_SY50_CR,0,0,38,50_.jpg', 'https://images-na.ssl-images-amazon.com/images/I/41b-YeD-rEL._SX38_SY50_CR,0,0,38,50_.jpg', 'https://images-na.ssl-images-amazon.com/images/I/41foQn1vo2L._SX38_SY50_CR,0,0,38,50_.jpg', 'https://images-na.ssl-images-amazon.com/images/I/41wjlc1%2BRXL._SX38_SY50_CR,0,0,38,50_.jpg')"""
    image_group_embedding = ImageGroupEmbedding(source_folder="../../../data/images")

    image_embedding = asyncio.run(
        image_group_embedding.get_image_group_embedding(image_group_text_example)
    )
    print(image_embedding)

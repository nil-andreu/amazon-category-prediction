import typing
from os import path

import torchvision.transforms as T
from PIL import Image

from shared.utils.constants import IMAGE_MEAN, IMAGE_STD
from shared.utils.images.extractor import ImageExtractor


class ImageProcessing:
    def __init__(
        self,
        img,
        img_size: int = 224,
        img_mean: typing.List[float] = IMAGE_MEAN,
        img_std: typing.List[float] = IMAGE_STD,
    ):
        self._img = img
        self._img_size = img_size
        self._img_mean = img_mean
        self._transformer = T.Compose(
            [
                # We first resize the input image to 256x256 and then we take center crop.
                T.Resize(img_size),
                T.CenterCrop(img_size),
                T.ToTensor(),
                T.Normalize(mean=img_mean, std=img_std),
            ]
        )
        self._img_transformed = img

    def process(self):
        self._img_transformed = self._transformer(self._img)
        return self._img_transformed


class ImageGroupProcessing:
    def __init__(self, image_group: str, source_folder: str):
        self._image_group = image_group
        self._source_folder = source_folder

    async def get_image_group_processed(self):
        file_paths = await self.get_missing_files()

        image_batch_transformed = []
        for file_path in file_paths:
            try:
                img = ImageProcessing(Image.open(file_path)).process()
                image_batch_transformed.append(img)

            except (FileNotFoundError, RuntimeError):
                continue
        return image_batch_transformed

    async def get_missing_files(self):
        image_extractor = ImageExtractor(output_folder=self._source_folder)

        image_urls: typing.List[str] = image_extractor.get_list_image_urls(
            self._image_group
        )
        file_paths: typing.List[str] = image_extractor.get_file_paths_from_image_urls(
            image_urls
        )
        files_missing: typing.List[bool] = self.check_images_file_system(file_paths)

        image_urls_missing = [
            image_url for idx, image_url in enumerate(image_urls) if files_missing[idx]
        ]
        file_paths_missing = [
            file_path for idx, file_path in enumerate(file_paths) if files_missing[idx]
        ]

        if image_urls_missing:
            contents = await image_extractor.gather_request_img(image_urls_missing)
            await image_extractor.gather_save_img(contents, file_paths_missing)

        return file_paths

    @staticmethod
    def check_images_file_system(file_paths) -> typing.List[bool]:
        file_paths_missing: typing.List[bool] = [
            not path.exists(file_path) for file_path in file_paths
        ]
        return file_paths_missing

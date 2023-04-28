import asyncio
import typing

import aiofiles
import aiohttp
import numpy as np
from aiohttp.client_exceptions import InvalidURL

from shared.utils.common import get_list_from_text_tuple


class ImageExtractor:
    def __init__(self, output_folder: str = "./data/images"):
        self._output_folder = output_folder

    async def extraction_and_save_image_urls(self, image_urls: str):
        if not image_urls or image_urls is np.nan:
            return None

        image_urls: typing.List[str] = self.get_list_image_urls(image_urls)
        if not image_urls:
            return []

        contents = await self.gather_request_img(image_urls)
        file_paths = self.get_file_paths_from_image_urls(image_urls)
        await self.gather_save_img(contents, file_paths)

        return file_paths

    def get_list_image_urls(self, image_urls: str) -> typing.List[str]:
        image_urls = get_list_from_text_tuple(image_urls)
        return self.get_stripped_list_img_urls(image_urls)

    async def gather_request_img(
        self,
        image_urls: typing.List[str],
    ):
        async with aiohttp.ClientSession() as session:
            contents = await asyncio.gather(
                *[self.request_img(image_url, session) for image_url in image_urls]
            )

        return contents

    async def gather_save_img(
        self, contents: typing.List[bytes], file_paths: typing.List[str]
    ):

        await asyncio.gather(
            *[
                self.save_img(content, file_path)
                for content, file_path in zip(contents, file_paths)
                if content
            ]
        )

    @staticmethod
    def filter_empty_image_url(image_urls: str):
        return filter(None, image_urls.split("', '"))

    @staticmethod
    def get_stripped_list_img_urls(image_urls: typing.List[str]):
        return [image_url.strip() for image_url in image_urls]

    @staticmethod
    async def request_img(image_url, session):
        try:
            async with session.get(image_url) as response:
                if response.status == 200:
                    content = await response.content.read()
                    return content

        except InvalidURL:
            return None

    def get_file_paths_from_image_urls(
        self, image_urls: typing.List[str]
    ) -> typing.List[str]:
        return [
            f"{self._output_folder}/{hash(image_url)}.jpg" for image_url in image_urls
        ]

    @staticmethod
    async def save_img(content, file_path: str):
        async with aiofiles.open(file_path, "wb") as f:
            await f.write(content)

    @property
    def output_folder(self):
        return self._output_folder

    @output_folder.setter
    def output_folder(self, new_output_folder: str):
        self._output_folder = new_output_folder

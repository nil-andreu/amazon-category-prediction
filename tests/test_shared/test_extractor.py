from shared.utils.images.extractor import ImageExtractor


class TestImageExtractor:
    @classmethod
    def setup_class(cls):
        cls.image_extractor = ImageExtractor("")

    """Test properties"""

    def test_output_folder(self):
        assert self.image_extractor.output_folder == ""

    def test_set_output_folder(self):
        new_path: str = "./data"
        self.image_extractor.output_folder = new_path
        assert self.image_extractor.output_folder == new_path

        # Clean the changes
        self.image_extractor.output_folder = ""

    """"""

import numpy as np

from shared.utils.common import (
    get_list_from_text_tuple,
    join_multiple_text_tuple,
    min_max_scale,
)


class TestCommon:
    def test_join_multiple_text_tuple(self):
        assert join_multiple_text_tuple("), (") == ", "
        assert (
            join_multiple_text_tuple("('This is one tuple'), ('This is second tuple')")
            == "('This is one tuple', 'This is second tuple')"
        )

    def test_get_list_from_text_tuple(self):
        assert not get_list_from_text_tuple(np.nan)
        assert not get_list_from_text_tuple(None)

    def test_description_get_list_from_text_tuple(self):
        example_description = """(\'Bumpersticker: A day without sunshine is like, well, night. 11"x2.5" sticker. Art by NSI\',)"""
        assert get_list_from_text_tuple(example_description) == [
            'Bumpersticker: A day without sunshine is like, well, night. 11"x2.5" sticker. Art by NSI'
        ]

        example_description = """('Brand new in original packaging. Exactly the same as shown in the picture! Come with both tail lights. High quality red and smoke tail lights with latest 3D style led light bar. These lights are made by an OE approved and ISO certified manufacturer with the quality meet or exceed all OE standards. Do not come with installation instructions. Professional installation highly recommended. Fitment : 92-98 BMW E36 3-SERIES 2dr models (318I 325I 325IS 328I 328IS M3',)"""
        assert get_list_from_text_tuple(example_description) == [
            "Brand new in original packaging. Exactly the same as shown in the picture! Come with both tail lights. High quality red and smoke tail lights with latest 3D style led light bar. These lights are made by an OE approved and ISO certified manufacturer with the quality meet or exceed all OE standards. Do not come with installation instructions. Professional installation highly recommended. Fitment : 92-98 BMW E36 3-SERIES 2dr models (318I 325I 325IS 328I 328IS M3"
        ]

        example_description_multiple = """(\'\', \'<b>Pair of Tailgate Liftgate Cables</b>\', \'<b>Fitment</b>\', "<b>(NOTE)</b> Fits both the Driver\'s and Passenger\'s sides", \'<b>Quality</b><br>Unless noted otherwise, these are new aftermarket parts. They align with Original Equipment (OE) specifications and act as a direct replacement for the factory part. They will fit and function as the original factory part did.\', \'See Seller details for available warranty, return policy and more.\')"""
        assert sorted(get_list_from_text_tuple(example_description_multiple)) == sorted(
            [
                "<b>Pair of Tailgate Liftgate Cables</b>",
                "See Seller details for available warranty, return policy and more.",
                "<b>(NOTE)</b> Fits both the Driver's and Passenger's sides",
                "<b>Fitment</b>",
                "<b>Quality</b><br>Unless noted otherwise, these are new aftermarket parts. They align with Original Equipment (OE) specifications and act as a direct replacement for the factory part. They will fit and function as the original factory part did.",
            ]
        )

    def test_image_get_list_from_text_group(self):
        example_image = "('https://images-na.ssl-images-amazon.com/images/I/21ZKs%2B0pzmL._SS40_.jpg',)"
        assert get_list_from_text_tuple(example_image) == [
            "https://images-na.ssl-images-amazon.com/images/I/21ZKs%2B0pzmL._SS40_.jpg"
        ]

        example_image_duplicated = """('https://images-na.ssl-images-amazon.com/images/I/412XCLbr6HL._SS40_.jpg', 'https://images-na.ssl-images-amazon.com/images/I/412XCLbr6HL._SS40_.jpg')"""
        assert get_list_from_text_tuple(example_image_duplicated) == [
            "https://images-na.ssl-images-amazon.com/images/I/412XCLbr6HL._SS40_.jpg"
        ]

        example_image_multiple = """('https://images-na.ssl-images-amazon.com/images/I/51qft01nzeL._SS40_.jpg', 'https://images-na.ssl-images-amazon.com/images/I/51G4G1RAfiL._SS40_.jpg'), ('https://images-na.ssl-images-amazon.com/images/I/412XCLbr6HL._SS40_.jpg', 'https://images-na.ssl-images-amazon.com/images/I/412XCLbr6HL._SS40_.jpg')"""
        assert sorted(get_list_from_text_tuple(example_image_multiple)) == [
            "https://images-na.ssl-images-amazon.com/images/I/412XCLbr6HL._SS40_.jpg",
            "https://images-na.ssl-images-amazon.com/images/I/51G4G1RAfiL._SS40_.jpg",
            "https://images-na.ssl-images-amazon.com/images/I/51qft01nzeL._SS40_.jpg",
        ]

    def test_min_max_scale(self):
        assert min_max_scale(10, 100, 20) == -0.5
        assert min_max_scale(110, 100, 20) == 0.5
        assert min_max_scale(50, 100, 0) == 0
        assert round(min_max_scale(50, 100, 20), 3) == 0.375 - 0.5

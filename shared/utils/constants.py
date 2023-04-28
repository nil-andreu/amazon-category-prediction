INITIAL_COLUMNS_LIST_TYPE = ["also_buy", "also_view", "description", "feature", "image"]
INITIAL_COLUMNS_DROP = ["category"]  # this features is not going to be used

SHARED_PATH = "../shared"
SHARED_DATA_FOLDER = SHARED_PATH + "/data"

MAIN_CAT = [
    "All Electronics",
    "Amazon Fashion",
    "Amazon Home",
    "Arts, Crafts & Sewing",
    "Automotive",
    "Books",
    "Camera & Photo",
    "Cell Phones & Accessories",
    "Computers",
    "Digital Music",
    "Grocery",
    "Health & Personal Care",
    "Home Audio & Theater",
    "Industrial & Scientific",
    "Movies & TV",
    "Musical Instruments",
    "Office Products",
    "Pet Supplies",
    "Sports & Outdoors",
    "Tools & Home Improvement",
    "Toys & Games",
    "Video Games",
]

# For min-max scaling
PRICE_MINIMUM_VALUE = 0.0
PRICE_MAXIMUM_VALUE = 5.4
MIN_BUY_MINIMUM_VALUE = 0.0
MAX_BUY_MAXIMUM_VALUE = 59
MIN_VIEW_MINIMUM_VALUE = 0.0
MAX_VIEW_MAXIMUM_VALUE = 53

# Image transformation
IMAGE_MEAN = [0.5, 0.5, 0.5]
IMAGE_STD = [0.5, 0.5, 0.5]

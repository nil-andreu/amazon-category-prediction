import typing
from ast import literal_eval as make_tuple
from numbers import Number

import numpy as np


def join_multiple_text_tuple(text_tuple: str):
    """In case we have multiple tuples in the text, we join them"""
    text_tuple_splitted = text_tuple.split("), (")
    return ", ".join(text_tuple_splitted)


def get_list_from_text_tuple(text_tuple: str) -> typing.List[str]:
    """Convert the text that contains a tuple into a list of the different strings it contains that tuple"""

    # In the case is nan
    if text_tuple is np.nan or not text_tuple or isinstance(text_tuple, float):
        return []

    text_tuple = join_multiple_text_tuple(str(text_tuple))

    try:
        text_group = [text for text in set(make_tuple(text_tuple)) if text]
        return text_group

    # some texts: ('Nintendo Girls' I Love Yoshi T-Shirt Light Pink',). This is invalid syntax, ' and can't do anything
    except SyntaxError:
        return []


def min_max_scale(value, max_value: Number, min_value: Number = 0) -> Number:
    """Min Max Scaler with pre-defined max and minimum (cropping the value)"""
    min_max_value = (value - min_value) / (max_value - min_value)
    min_max_value = min(min_max_value, 1)  # maximum value possible is 1
    min_max_value = max(min_max_value, 0)  # minimum value possible is 0

    return min_max_value - 0.5  # so the values are (-.5, .5)


def rotate_labels(chart):
    chart.set_xticklabels(chart.get_xticklabels(), rotation=75, size=5)

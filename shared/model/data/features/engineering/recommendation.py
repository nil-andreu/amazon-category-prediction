import typing

from shared.utils.common import get_list_from_text_tuple, min_max_scale
from shared.utils.constants import (
    MAX_BUY_MAXIMUM_VALUE,
    MAX_VIEW_MAXIMUM_VALUE,
    MIN_BUY_MINIMUM_VALUE,
    MIN_VIEW_MINIMUM_VALUE,
)


class Recommendation:
    def __init__(self, max_value: int, min_value: int = 0):
        self._max_value = max_value
        self._min_value = min_value
        self._amount_recommended = None

    def get_feature(self, recommendation_text_tuple: str):
        recommendation_text_group: typing.List[str] = get_list_from_text_tuple(
            recommendation_text_tuple
        )
        self._amount_recommended = len(recommendation_text_group)
        self._amount_recommended = min_max_scale(
            self._amount_recommended,
            max_value=self._max_value,
            min_value=self._min_value,
        )

        return self._amount_recommended


class AlsoBuyRecommendation(Recommendation):
    def __init__(
        self,
        max_value: int = MAX_BUY_MAXIMUM_VALUE,
        min_value: int = MIN_BUY_MINIMUM_VALUE,
    ):
        super().__init__(max_value=max_value, min_value=min_value)


class AlsoViewRecommendation(Recommendation):
    def __init__(
        self,
        max_value: int = MAX_VIEW_MAXIMUM_VALUE,
        min_value: int = MIN_VIEW_MINIMUM_VALUE,
    ):
        super().__init__(max_value=max_value, min_value=min_value)

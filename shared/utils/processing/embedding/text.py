import re
import string
import typing

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from shared.utils.common import get_list_from_text_tuple

nltk.download("stopwords")
nltk.download("punkt")


STOP_WORDS = set(stopwords.words("english"))


class TextProcessing:
    def __init__(
        self,
        text: str,
        minimum_word_length: int = 3,
        stop_words: typing.Set[str] = STOP_WORDS,
    ):
        self._text = text
        self._tokens = []
        self._stop_words = stop_words
        self._minimum_word_length = minimum_word_length
        self._string_punctuations_table = str.maketrans("", "", string.punctuation)

    def process(self) -> str:
        self.remove_html_tags()
        self.remove_spacings()
        self.remove_apostrophes()
        # BERT Tokenizer handles it
        # self.tokenize_words()
        # self.tokens_lower_case()
        # self.tokens_remove_punctutions()
        # self.tokens_stripped()
        # self.tokens_remove_stop_words()
        # self.tokens_filter_length()
        # return " ".join(self.tokens)
        return self._text

    def remove_html_tags(self):
        self._text = re.sub("<[^<]+?>", "", self._text)
        return self._text

    def remove_spacings(self):
        self._text = self._text.replace("\\n", "")
        self._text = self._text.replace("\\t", "")
        return self._text

    def remove_apostrophes(self):
        self._text = self._text.replace("'", "")
        return self._text

    def tokenize_words(self):
        self._tokens = word_tokenize(self._text)
        return self._tokens

    def tokens_lower_case(self):
        self._tokens = [token.lower() for token in self._tokens]
        return self._tokens

    def tokens_filter_length(self):
        self._tokens = [
            token for token in self._tokens if len(token) >= self._minimum_word_length
        ]
        return self._tokens

    def tokens_remove_punctutions(self):
        self._tokens = [
            token.translate(self._string_punctuations_table) for token in self._tokens
        ]
        return self._tokens

    def tokens_stripped(self):
        self._tokens = [token for token in self._tokens if token.isalpha()]
        return self._tokens

    def tokens_remove_stop_words(self):
        self._tokens = [
            token for token in self._tokens if token not in self._stop_words
        ]
        return self._tokens

    @property
    def tokens(self):
        return self._tokens

    @property
    def text(self):
        return self._text

    @property
    def stop_words(self):
        return self._stop_words


class TextGroupProcessing:
    def __init__(self, text_tuple: str):
        """
        In a text group, we have a string that contains a tuple with multiple descriptions or features.
        We are going to process this text and get the words tokens for each of those descriptions/features.
        """
        self._text_tuple = text_tuple
        self._text_group: typing.List[str] = []
        self._text_group_processed: typing.List[str] = []

    def get_text_group_processed(self) -> typing.List[str]:
        """
        For a certain observation, we might have one or more descriptions.
        The aim is to get the word tokens for each of those descriptions.
        """

        self._text_group = get_list_from_text_tuple(self._text_tuple)

        self._text_group_processed = [
            TextProcessing(text).process() for text in self._text_group
        ]

        return [text for text in self._text_group_processed if text]

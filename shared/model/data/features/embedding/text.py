import typing

import torch
from transformers import AutoModel, AutoTokenizer

from shared.utils.processing.embedding.text import TextGroupProcessing


class TextEmbedding:
    def __init__(self, model, tokenizer):
        self._model = model
        self._tokenizer = tokenizer

    def get_embedding(self, texts):
        # Tokenize sentences: takes already care of lemmatizing, ...
        encoded_input = self._tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt"
        )

        # Compute token embeddings
        with torch.no_grad():
            model_output = self._model(**encoded_input, return_dict=True)

        # Perform pooling
        embedding = self.cls_pooling(model_output)

        return embedding

    @staticmethod
    def cls_pooling(model_output):
        return model_output.last_hidden_state[:, 0]


class MultipleTextEmbedding(TextEmbedding):
    def __init__(self, model, tokenizer):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
        )

    def get_text_group_embedding(self, text_group: str):
        text_group_processing = TextGroupProcessing(text_group)
        text_group_tokens: typing.List[
            str
        ] = text_group_processing.get_text_group_processed()

        if not text_group_tokens:
            return None

        # Get embedding in batch
        description_embeddings = self.get_embedding(text_group_tokens)

        return self.mean_embeddings(description_embeddings)

    @staticmethod
    def mean_embeddings(embeddings):
        return torch.mean(embeddings, dim=0)


class DescriptionEmbedding(MultipleTextEmbedding):
    def __init__(self):
        super().__init__(
            model=AutoModel.from_pretrained(
                "sentence-transformers/msmarco-distilbert-base-tas-b"
            ),
            tokenizer=AutoTokenizer.from_pretrained(
                "sentence-transformers/msmarco-distilbert-base-tas-b"
            ),
        )


class FeatureEmbedding(MultipleTextEmbedding):
    def __init__(self):
        super().__init__(
            model=AutoModel.from_pretrained(
                "sentence-transformers/paraphrase-MiniLM-L6-v2"
            ),
            tokenizer=AutoTokenizer.from_pretrained(
                "sentence-transformers/paraphrase-MiniLM-L6-v2"
            ),
        )


class TitleEmbedding(MultipleTextEmbedding):
    def __init__(self):
        super().__init__(
            model=AutoModel.from_pretrained(
                "sentence-transformers/paraphrase-MiniLM-L6-v2"
            ),
            tokenizer=AutoTokenizer.from_pretrained(
                "sentence-transformers/paraphrase-MiniLM-L6-v2"
            ),
        )

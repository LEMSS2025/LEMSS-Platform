from typing import List, Tuple

import torch
from transformers import AutoTokenizer, AutoModel
from torch.nn.functional import cosine_similarity

from rankers.embedding_ranker import EmbeddingRanker
from utils.logger import setup_logger
from constants.constants import COMPETITION_LOG_FILE, COMPETITION_LOG_NAME


class Contriever(EmbeddingRanker):
    """
        Contriever Ranker class that ranks documents based on their similarity to a given query using huggingface's Contriever model.
    """

    def __init__(self, model_name: str):
        """
        Initialize the Contriever ranker with the given model name.

        :param model_name: The name of the model used for ranking.
        """
        super().__init__(model_name)
        self.__logger = setup_logger(COMPETITION_LOG_NAME, COMPETITION_LOG_FILE)

        self.__logger.info(f"Loading model {model_name} for Contriever ranker.")
        try:
            self.__tokenizer = AutoTokenizer.from_pretrained(model_name, device=self.device)
            self.__model = AutoModel.from_pretrained(model_name)
            self.__model.to(self.device)
            self.__logger.info(f"Model {model_name} loaded successfully.")
        except Exception as e:
            self.__logger.error(f"Error loading model {model_name}: {e}")
            raise

    def __mean_pooling(self, token_embeddings, mask):
        mask = mask.bool()  # Convert mask to boolean
        token_embeddings = token_embeddings.masked_fill(~mask[..., None], 0.)
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings

    def rank(self, query: str, documents: List[str]) -> Tuple[List[int], List[float]]:
        """
        Rank documents based on their similarity to the query.

        :param query: A single query.
        :param documents: List of documents.
        :return: List of scores representing the similarity between the query and documents.
        """
        try:
            self.__logger.info(f"Ranking {len(documents)} documents for query: {query}")

            # Apply tokenization and encoding to the input documents and query
            docs_tokens = self.__tokenizer(list(documents), padding=True, truncation=True, return_tensors='pt').to(self.device)
            query_token = self.__tokenizer(query, padding=True, truncation=True, return_tensors='pt').to(self.device)

            # Mean pooling to get sentence embeddings
            with torch.no_grad():
                docs_outputs, query_outputs = self.__model(**docs_tokens), self.__model(**query_token)

            docs_embeddings = self.__mean_pooling(docs_outputs.last_hidden_state, docs_tokens["attention_mask"])
            query_embedding = self.__mean_pooling(query_outputs.last_hidden_state, query_token["attention_mask"])

            # Compute similarity scores between the query and each document
            scores = cosine_similarity(docs_embeddings, query_embedding).cpu().numpy()

            # Use the base class's tie breaker to rank documents
            ranked_scores, scores = super().tie_breaker(scores.flatten().tolist())

            return ranked_scores, scores
        except Exception as e:
            self.__logger.error(f"Error in ranking documents: {e}")
            raise

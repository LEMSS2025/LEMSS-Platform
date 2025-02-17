from typing import List, Tuple

from sentence_transformers import SentenceTransformer

from rankers.embedding_ranker import EmbeddingRanker
from utils.logger import setup_logger
from constants.constants import E5_LOG_FILE, E5_LOG_NAME


class E5(EmbeddingRanker):
    """
        E5 Ranker class that ranks documents based on their similarity to a given query using a SentenceTransformer model.
    """

    def __init__(self, model_name: str):
        """
        Initialize the E5 ranker with the given model name.

        :param model_name: The name of the model used for ranking.
        """
        super().__init__(model_name)
        self.__logger = setup_logger(E5_LOG_NAME, E5_LOG_FILE)

        self.__logger.info(f"Loading model {model_name} for E5 ranker.")
        try:
            self.__model = SentenceTransformer(model_name, device=self.device)
            self.__logger.info(f"Model {model_name} loaded successfully.")
        except Exception as e:
            self.__logger.error(f"Error loading model {model_name}: {e}")
            raise

    def rank(self, query: str, documents: List[str]) -> Tuple[List[int], List[float]]:
        """
        Rank documents based on their similarity to the query.

        :param query: A single query.
        :param documents: List of documents.
        :return: List of scores representing the similarity between the query and documents.
        """
        try:
            self.__logger.info(f"Ranking {len(documents)} documents for query: {query}")

            # Prepare input texts for encoding
            input_texts = [f"query: {query}"] + [f"passage: {doc}" for doc in documents]

            # Generate embeddings for the query and documents
            embeddings = self.__model.encode(input_texts, normalize_embeddings=True)

            # Compute similarity scores between the query and each document
            scores = embeddings[:1] @ embeddings[1:].T

            # Use the base class's tie breaker to rank documents
            ranked_scores, scores = super().tie_breaker(scores.flatten().tolist())

            return ranked_scores, scores
        except Exception as e:
            self.__logger.error(f"Error in ranking documents: {e}")
            raise

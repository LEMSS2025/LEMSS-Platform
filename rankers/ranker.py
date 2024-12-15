from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np
import torch

from utils.logger import setup_logger
from constants.constants import RANKER_LOG_FILE, RANKER_LOG_NAME


class Ranker(ABC):
    """
        Abstract base class for document ranking models.
        Provides the structure for initializing a model, ranking documents, and breaking ties in scores.
    """

    def __init__(self, model_name: str):
        """
        Initialize the Ranker with the given model name.

        :param model_name: The name of the model used for ranking.
        """
        self.__model_name = model_name
        self.__logger = setup_logger(RANKER_LOG_NAME, RANKER_LOG_FILE)
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.__logger.info(f"Ranker initialized with model: {model_name}")

    @abstractmethod
    def rank(self, query: str, documents: List[str]) -> List[float]:
        """
        Abstract method to rank documents based on their similarity to the query.

        :param query: A single query string.
        :param documents: List of documents to be ranked.
        :return: List of scores representing the similarity between the query and each document.
        """
        pass

    def tie_breaker(self, scores: List[float]) -> Tuple[List[int], List[float]]:
        """
        Break ties in scores by adding a small random value to each tied score.

        :param scores: List of similarity scores.
        :return: List of ranks after breaking ties, with the highest score receiving rank 1.
        """
        try:
            scores = np.array(scores, dtype=np.float32)
            unique, counts = np.unique(scores, return_counts=True)
            ties = unique[counts > 1]

            # Adding small random value to each tied score
            for tie in ties:
                indices = np.where(scores == tie)[0]
                if np.any(np.diff(np.sort(scores)) > 0):
                    epsilon = np.min(np.diff(np.sort(scores))[np.diff(np.sort(scores)) > 0]) / 2
                else:
                    epsilon = 1
                scores[indices] += np.random.uniform(0, epsilon, size=len(indices))

            # Rank the scores, with the highest score getting rank 1
            ranks = np.argsort(np.argsort(-scores)) + 1  # argsort ranks in ascending order, so reverse it
            return ranks.tolist(), scores.tolist()
        except Exception as e:
            self.__logger.error(f"Error in tie breaking: {e}")
            raise

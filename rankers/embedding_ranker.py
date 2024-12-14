from rankers.ranker import Ranker
from utils.logger import setup_logger

from abc import ABC
from constants.constants import EMBEDDING_RANKER_LOG_FILE, EMBEDDING_RANKER_LOG_NAME

class EmbeddingRanker(Ranker, ABC):
    """
    Abstract base class for document ranking models based on embeddings.
    """

    def __init__(self, model_name: str):
        """
        Initialize the EmbeddingRanker with the given model name.
        """
        super().__init__(model_name)
        self.__logger = setup_logger(EMBEDDING_RANKER_LOG_NAME, EMBEDDING_RANKER_LOG_FILE)
        self.device = self.device
        self.__logger.info(f"EmbeddingRanker initialized with model: {model_name}")
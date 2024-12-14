from abc import ABC, abstractmethod
import os

import pandas as pd

from rankers.ranker import Ranker
from utils.logger import setup_logger
from constants.constants import INDEX_RANKER_LOG_FILE, INDEX_RANKER_LOG_NAME, PROJECT_DIR


class IndexRanker(Ranker, ABC):
    """
    Abstract base class for document ranking models based on index search.
    """

    def __init__(self, model_name: str, index_name: str, corpus_path: str = None):
        """
        Initialize the EmbeddingRanker with the given model name.

        :param model_name: Name of the model to be used.
        :param index_name: Name of the index.
        :param corpus_path: Path to the corpus (optional).
        """
        super().__init__(model_name)
        self.__logger = setup_logger(INDEX_RANKER_LOG_NAME, INDEX_RANKER_LOG_FILE)
        self.index_name = os.path.join(PROJECT_DIR, index_name)
        self.corpus_path = corpus_path
        self.__logger.info(f"IndexRanker initialized with model: {model_name} and index: {index_name}")

    def get_index_path(self) -> str:
        """
        Get the path to the index.

        :return: Path to the index.
        """
        pass

    def set_index_path(self, index_name: str, output_hash_folder: str):
        """
        Set the path to the index.

        :param index_name: Name of the index.
        :param output_hash_folder: Path to the output hash folder.
        """
        pass

    @abstractmethod
    def initialize_index(self):
        """
        Initialize the index with the required dataset.
        """
        pass

    @abstractmethod
    def add_document(self, documents_df: pd.DataFrame):
        """
        Add the new generated documents to the index.

        :param documents_df: DataFrame containing the documents to be indexed.
        """
        pass

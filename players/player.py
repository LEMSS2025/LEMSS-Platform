from abc import ABC, abstractmethod

import pandas as pd

from utils.logger import setup_logger
from LLMs import LLM
from constants.constants import PLAYER_LOG_FILE, PLAYER_LOG_NAME, PLAYER_HISTORY_COLUMNS


class Player(ABC):
    """
        Abstract class representing a player in the game, responsible for
        generating documents and managing feedback across rounds.
    """

    def __init__(self, name: str, character: str, prompt_format: str, query: str, feedback_func: callable):
        """
        Initialize a Player instance.

        :param name: Name of the player.
        :param character: Character/persona the player will adopt.
        :param prompt_format: Format string for prompts.
        :param query: The query the player will be working with.
        :param feedback_func: Function to generate feedback for the player.
        """
        self.name = name
        self.character = character
        self.prompt_format = prompt_format
        self.query = query
        self.round = 1
        self.document = None
        self.history = pd.DataFrame(columns=PLAYER_HISTORY_COLUMNS)
        self.rank = None
        self.feedback_func = feedback_func
        self.__logger = setup_logger(PLAYER_LOG_NAME, PLAYER_LOG_FILE)

    def get_name(self) -> str:
        """
        Get the name of the player.

        :return: Name of the player.
        """
        return self.name

    def set_rank(self, rank: int) -> None:
        """
        Set the rank of the document.

        :param rank: Rank of the document.
        """
        self.rank = rank

    @abstractmethod
    def generate_document(self, llm: LLM, max_tokens: int, init_doc: str = None,
                          force_max_tokens: bool = False) -> tuple:
        """
        Generate a document based on the query and character using the specified LLM.

        :param llm: The LLM instance used for generating the document.
        :param max_tokens: Maximum number of tokens for the generated document.
        :param init_doc: Initial document text, used in the first round.
        :param force_max_tokens: Whether to manually restrict the number of tokens in the generated document.

        :return: The generated document.
        """
        pass

    @abstractmethod
    def generate_feedback(self, feedback: pd.DataFrame) -> None:
        """
        Generate feedback for the next round based on the provided feedback.

        :param feedback: Feedback for the current round.
        """
        pass
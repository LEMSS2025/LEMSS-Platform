import pandas as pd

from players.player import Player
from utils.logger import setup_logger
from constants.constants import HISTORY_DOCUMENT_COLUMN, STATIC_PLAYER_LOG_FILE, STATIC_PLAYER_LOG_NAME


class StaticPlayer(Player):
    def __init__(self, name: str, query: str, query_id: int, init_document: str,
                 feedback_func: callable):
        super().__init__(name, character="static", prompt_format="-", query=query, query_id=query_id,
                         init_document=init_document, feedback_func=feedback_func)
        self.__logger = setup_logger(
            STATIC_PLAYER_LOG_NAME, STATIC_PLAYER_LOG_FILE)

    def generate_document(self, *args, **kwargs) -> tuple:
        try:

            if self.round == 1:
                return self.init_document, self.init_document, "", ""

            own_document = self.__own_feedback[HISTORY_DOCUMENT_COLUMN].values[0]

            return own_document, own_document, "", ""
        except Exception as e:
            self.__logger.error(f"Error generating document: {e}")
            raise

    def generate_feedback(self, feedback: pd.DataFrame) -> None:
        """
        Generate feedback for the next round based on the provided feedback.

        :param feedback: Feedback for the current round.
        """
        self.round += 1
        self.__own_feedback = self.feedback_func(
            feedback, self.name, self.round)

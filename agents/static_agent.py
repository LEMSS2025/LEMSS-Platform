import pandas as pd
import torch

from agents.agent import Agent
from competition.warm_start import WarmStart
from players.static_player import StaticPlayer
from utils.logger import setup_logger
from constants.constants import (HISTORY_PLAYER_COLUMN, HISTORY_ROUND_COLUMN, STATIC_AGENT_LOG_FILE,
                                 STATIC_AGENT_LOG_NAME)


class StaticAgent(Agent):
    """
        An agent that manages players and feedback in a competition, specifically using Large Language Models (LLMs).
    """

    def __init__(self, name: str, queries_df: pd.DataFrame, warm_start: WarmStart):

        super().__init__(name=name, character="static", prompt_format="", queries_df=queries_df, warm_start=warm_start)
        self.__logger = setup_logger(STATIC_AGENT_LOG_NAME, STATIC_AGENT_LOG_FILE)
        self.device = torch.device("cpu")

        self.build_players()

    def build_players(self) -> None:
        """
        Build the players for each query the agent is responsible for.
        """
        try:
            self.__players = {query_id: StaticPlayer(name=self.name, query=query, query_id=query_id,
                                                     init_document=init_doc, feedback_func=self.generate_feedback)
                              for query_id, init_doc, query in self.queries}
            if self.warm_start:
                for player in self.__players.values():
                    self.set_player(player, player.get_name(), player.get_query_id())
            self.__logger.info(f"Players built successfully for agent: {self.name}")
        except Exception as e:
            self.__logger.error(
                f"Error building players for agent {self.name}: {e}")
            raise

    def get_player(self, query_id: int) -> StaticPlayer:
        """
        Retrieve the player associated with the given query id.

        :param query_id: Query id for the player to retrieve.
        :return: Player instance for the provided query.
        """
        try:
            return self.__players[query_id]
        except KeyError:
            self.__logger.error(f"Player for query '{query_id}' not found.")
            raise

    def generate_feedback(self, feedback: pd.DataFrame, player_name: str, round: int):
        """
        Generate feedback for the next round based on the competition history.

        :param feedback: DataFrame containing the feedback from previous rounds.
        :param player_name: Name of the player requesting feedback.
        :param round: Current round number.
        :return: Tuple containing pairwise feedback and all feedback.
        """

        try:

            # Filter feedback for relevant rounds
            last_round_feedback = feedback[feedback[HISTORY_ROUND_COLUMN] == round - 1]

            # Filter feedback for the player in the current round
            own_feedback = last_round_feedback[last_round_feedback[HISTORY_PLAYER_COLUMN] == player_name]

            return own_feedback

        except Exception as e:
            self.__logger.error(
                f"Error generating feedback for player {player_name} in round {round}: {e}")
            raise

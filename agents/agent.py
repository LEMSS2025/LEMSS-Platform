from abc import ABC, abstractmethod

import pandas as pd

from players.player import Player
from competition.warm_start import WarmStart
from utils.logger import setup_logger
from constants.constants import AGENT_LOG_FILE, AGENT_LOG_NAME


class Agent(ABC):
    """
        Abstract base class representing an agent that manages players
        and handles the feedback mechanism for a competition.
    """

    def __init__(self, name: str, character: str, prompt_format: str, queries_df: pd.DataFrame,
                 warm_start: WarmStart = None):
        """
        Initialize the Agent with the provided configuration.

        :param name: Name of the agent.
        :param character: Character the agent will assume during the competition.
        :param prompt_format: Format string for generating prompts.
        :param queries_df: DataFrame containing the queries.
        :param warm_start: WarmStart instance for the competition.
        """
        self.name = name
        self.character = character
        self.prompt_format = prompt_format
        self.queries = queries_df.values
        self.warm_start = warm_start
        self.history = []
        self.__logger = setup_logger(AGENT_LOG_NAME, AGENT_LOG_FILE)

    def set_history(self, history: pd.DataFrame) -> None:
        """
        Set the history of the agent for the competition.

        :param history: DataFrame containing the agent's history.
        """
        self.history = history

    @abstractmethod
    def build_players(self) -> None:
        """
        Build the players for each query the agent is responsible for.
        """
        pass

    @abstractmethod
    def get_player(self, query: str) -> Player:
        """
        Retrieve the player associated with the given query.

        :param query: Query for which the player is needed.
        :return: Player instance for the provided query.
        """

    @abstractmethod
    def generate_feedback(self, feedback: pd.DataFrame, player_name: str, round: int):
        """
        Generate feedback for the next round based on the competition history.

        :param feedback: DataFrame containing the feedback from previous rounds.
        :param player_name: Name of the player requesting feedback.
        :param round: Current round number.
        :return: Tuple containing pairwise feedback and all feedback.
        """
        pass

    def set_player(self, player: Player, player_name: str, query_id: int):
        """
        Set the player.

        :param player: Player instance.
        :param player_name: Name of the player.
        :param query_id: ID of the query.
        """
        round, document, init_document, rank, game_history = self.warm_start.set_player(player_name, query_id)
        if game_history is None:
            return
        player.set_rank(rank), player.set_round(round)
        player.set_document(document), player.set_init_document(init_document)
        player.generate_feedback(game_history)

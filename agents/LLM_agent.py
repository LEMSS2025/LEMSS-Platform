import pandas as pd


from agents.agent import Agent
from players.llm_player import LLMPlayer
from utils.logger import setup_logger
from constants.constants import (LLM_AGENT_LOG_FILE, LLM_AGENT_LOG_NAME, DEFAULT_LLM_AGENT_DEPTH,
                                 HISTORY_ROUND_COLUMN, HISTORY_RANK_COLUMN, HISTORY_PLAYER_COLUMN)


class LLMAgent(Agent):
    """
        An agent that manages players and feedback in a competition, specifically using Large Language Models (LLMs).
    """

    def __init__(self, name: str, character: str, prompt_format: str, queries_df: pd.DataFrame,
                 pairwise: bool = False, depth: int = DEFAULT_LLM_AGENT_DEPTH):
        """
        Initialize the LLMAgent with the provided configuration.

        :param name: Name of the agent.
        :param character: Character the agent will assume during the competition.
        :param prompt_format: Format string for generating prompts.
        :param queries_df: DataFrame containing the queries.
        :param pairwise: Whether to use pairwise feedback.
        :param depth: The depth of rounds to consider for feedback.
        """
        super().__init__(name, character, prompt_format, queries_df)
        self.__pairwise = pairwise
        self.__depth = depth
        self.__logger = setup_logger(LLM_AGENT_LOG_NAME, LLM_AGENT_LOG_FILE)

        self.build_players()

    def build_players(self) -> None:
        """
        Build the players for each query the agent is responsible for.
        """
        try:
            self.__players = {query: LLMPlayer(name=self.name, character=self.character,
                                               prompt_format=self.prompt_format, query=query,
                                               feedback_func=self.generate_feedback)
                              for query in self.queries}
            self.__logger.info(f"Players built successfully for agent: {self.name}")
        except Exception as e:
            self.__logger.error(f"Error building players for agent {self.name}: {e}")
            raise

    def get_player(self, query: str) -> LLMPlayer:
        """
        Retrieve the player associated with the given query.

        :param query: Query for which the player is needed.
        :return: Player instance for the provided query.
        """
        try:
            return self.__players[query]
        except KeyError:
            self.__logger.error(f"Player for query '{query}' not found.")
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
            pairwise_feedback, all_feedback = None, None

            # Filter feedback for relevant rounds
            feedback = feedback[(feedback[HISTORY_ROUND_COLUMN] >= round - self.__depth) & (feedback[HISTORY_ROUND_COLUMN] != 0)]

            if self.__pairwise:
                # For each round, select 2 random players for pairwise feedback
                pairwise_feedback = pd.concat(
                    [feedback[feedback[HISTORY_ROUND_COLUMN] == int(round)].sample(2) for round in feedback[HISTORY_ROUND_COLUMN].unique()],
                    ignore_index=True
                )
                pairwise_feedback = pairwise_feedback.sort_values([HISTORY_ROUND_COLUMN, HISTORY_RANK_COLUMN], ascending=[False, True])
            else:
                # Filter feedback for the player in the current round
                feedback = feedback[~((feedback[HISTORY_PLAYER_COLUMN] == player_name) & (feedback[HISTORY_ROUND_COLUMN] == round - 1))]

                # Collect all feedback for each round
                all_feedback = pd.concat(
                    [feedback[feedback[HISTORY_ROUND_COLUMN] == int(round)] for round in feedback['round'].unique()],
                    ignore_index=True
                )
                all_feedback = all_feedback.sort_values([HISTORY_ROUND_COLUMN, HISTORY_RANK_COLUMN], ascending=[False, True])

            return pairwise_feedback, all_feedback
        except Exception as e:
            self.__logger.error(f"Error generating feedback for player {player_name} in round {round}: {e}")
            raise
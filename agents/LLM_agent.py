import pandas as pd

from agents.agent import Agent
from players.llm_player import LLMPlayer
from competition.warm_start import WarmStart
from LLMs.hugging_face_llm import HuggingFaceLLM
from LLMs.mlx_llm import MLXLLM
from utils.logger import setup_logger
from constants.constants import (LLM_AGENT_LOG_FILE, LLM_AGENT_LOG_NAME, DEFAULT_LLM_AGENT_DEPTH,
                                 HISTORY_ROUND_COLUMN, HISTORY_RANK_COLUMN, HISTORY_PLAYER_COLUMN,
                                 MLX_IDENTIFIER, CONFIG_LLM_MODEL_NAME_HEADER)


class LLMAgent(Agent):
    """
        An agent that manages players and feedback in a competition, specifically using Large Language Models (LLMs).
    """

    def __init__(self, name: str, character: str, llm: dict, prompt_format: str, queries_df: pd.DataFrame,
                 warm_start: WarmStart, pairwise: bool = False, depth: int = DEFAULT_LLM_AGENT_DEPTH):
        """
        Initialize the LLMAgent with the provided configuration.

        :param name: Name of the agent.
        :param character: Character the agent will assume during the competition.
        :param llm: Configuration for the Large Language Model.
        :param prompt_format: Format string for generating prompts.
        :param queries_df: DataFrame containing the queries.
        :param pairwise: Whether to use pairwise feedback.
        :param depth: The depth of rounds to consider for feedback.
        """
        super().__init__(name, character, prompt_format, queries_df, warm_start)
        self.__pairwise = pairwise
        self.__depth = depth
        self.__logger = setup_logger(LLM_AGENT_LOG_NAME, LLM_AGENT_LOG_FILE)

        self.__setup_llm(llm)
        self.build_players()

    def build_players(self) -> None:
        """
        Build the players for each query the agent is responsible for.
        """
        try:
            self.__players = {query_id: LLMPlayer(name=self.name, character=self.character, llm=self.llm,
                                                  prompt_format=self.prompt_format, query=query, query_id=query_id,
                                                  init_document=init_doc, feedback_func=self.generate_feedback)
                              for query_id, init_doc, query in self.queries}
            if self.warm_start:
                for player in self.__players.values():
                    self.set_player(player, player.get_name(), player.get_query_id())
            self.__logger.info(f"Players built successfully for agent: {self.name}")
        except Exception as e:
            self.__logger.error(f"Error building players for agent {self.name}: {e}")
            raise

    def __setup_llm(self, llm_config):
        """
        Set up the LLM (Large Language Model) based on the configuration.
        """
        try:
            if MLX_IDENTIFIER in llm_config[CONFIG_LLM_MODEL_NAME_HEADER]:
                self.llm = MLXLLM(**llm_config)
            else:
                self.llm = HuggingFaceLLM(**llm_config)

            self.__logger.info("LLM initialized successfully")
        except KeyError as e:
            self.__logger.error(f"Missing LLM configuration key: {e}")
            raise

    def get_player(self, query_id: int) -> LLMPlayer:
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
            pairwise_feedback, all_feedback = None, None

            # Filter feedback for relevant rounds
            feedback = feedback[(feedback[HISTORY_ROUND_COLUMN] >= round - self.__depth) &
                                (feedback[HISTORY_ROUND_COLUMN] != 0)]

            if self.__pairwise:
                # For each round, select 2 random players for pairwise feedback
                pairwise_feedback = pd.concat(
                    [feedback[feedback[HISTORY_ROUND_COLUMN] == int(round)].sample(2)
                     for round in feedback[HISTORY_ROUND_COLUMN].unique()],
                    ignore_index=True
                )
                pairwise_feedback = pairwise_feedback.sort_values([HISTORY_ROUND_COLUMN, HISTORY_RANK_COLUMN],
                                                                  ascending=[False, True])
            else:
                # Filter feedback for the player in the current round
                feedback = feedback[~((feedback[HISTORY_PLAYER_COLUMN] == player_name) &
                                      (feedback[HISTORY_ROUND_COLUMN] == round - 1))]

                # Collect all feedback for each round
                all_feedback = pd.concat(
                    [feedback[feedback[HISTORY_ROUND_COLUMN] == int(round)] for round in feedback['round'].unique()],
                    ignore_index=True
                )
                all_feedback = all_feedback.sort_values([HISTORY_ROUND_COLUMN, HISTORY_RANK_COLUMN],
                                                        ascending=[False, True])

            return pairwise_feedback, all_feedback
        except Exception as e:
            self.__logger.error(f"Error generating feedback for player {player_name} in round {round}: {e}")
            raise

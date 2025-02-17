import pandas as pd

from rankers import ranker
from utils.logger import setup_logger
from competition.warm_start import WarmStart
from constants.constants import (GAME_LOG_FILE, GAME_LOG_NAME,
                                 QUERY_DF_DOCUMENT_COLUMN, QUERY_DF_QUERY_COLUMN, QUERY_DF_QUERY_ID_COLUMN,
                                 GAME_HISTORY_COLUMNS, HISTORY_QUERY_ID_COLUMN, HISTORY_GAME_ID_COLUMN)


class Game:
    """
        Class responsible for managing the entire game process, including generating documents, ranking them,
        providing feedback to players, and tracking the game history.
    """

    def __init__(self, query_info: dict, agents: list, ranker: ranker, max_tokens: int, rounds: int,
                 force_max_tokens: bool = False, warm_start: WarmStart = None):
        """
        Initialize the Game instance.

        :param query_info: Dictionary containing query information (query text, query ID, initial document).
        :param agents: List of agent instances participating in the game.
        :param ranker: Ranker instance used for ranking documents.
        :param llm: LLM instance used for generating documents.
        :param max_tokens: Maximum number of tokens for the generated documents.
        :param rounds: Number of rounds in the game.
        :param force_max_tokens: Whether to manually restrict the number of tokens in the generated documents.
        :param warm_start: WarmStart instance used for initializing player history.
        """
        self.__query = query_info[QUERY_DF_QUERY_COLUMN]
        self.__query_id = query_info[QUERY_DF_QUERY_ID_COLUMN]
        self.__init_doc = query_info[QUERY_DF_DOCUMENT_COLUMN]
        self.__players = [agent.get_player(query_info[QUERY_DF_QUERY_ID_COLUMN]) for agent in agents]
        self.__ranker = ranker
        self.__rounds = rounds
        self.__round = 1
        self.__max_tokens = max_tokens
        self.__force_max_tokens = force_max_tokens
        self.__game_history = pd.DataFrame(columns=GAME_HISTORY_COLUMNS)
        self.__warm_start = warm_start
        self.__logger = setup_logger(GAME_LOG_NAME, GAME_LOG_FILE)

        if self.__warm_start:
            game_history, round = self.__warm_start.set_game(int(self.__query_id))
            self.__game_history = game_history
            self.__round = round + 1
        else:
            # Initialize game history with the initial document for each player
            for player in self.__players:
                self.__game_history.loc[len(self.__game_history)] = [0, player.get_name(), self.__init_doc, None, None,
                                                                     None, None, None]

    def get_query_id(self):
        """
        Get the query ID for the current game.

        :return: Query ID.
        """
        return self.__query_id

    def increase_round(self):
        """
        Increase the round number.
        """
        self.__round += 1

    def generate_documents(self) -> list:
        """
        Generate documents for the current round.

        :return: List of generated documents and prompts.
        """
        try:
            self.__logger.info(f"Generating documents for round {self.__round} for query: {self.__query}")
            documents_prompts = [player.generate_document(self.__max_tokens, force_max_tokens=self.__force_max_tokens)
                                 for player in self.__players]
            return documents_prompts
        except Exception as e:
            self.__logger.error(f"Error generating documents: {e}")
            raise

    def rank_documents(self, documents_prompts: list, docnos: list = None) -> list:
        """
        Rank the provided documents.

        :param documents_prompts: List of documents to rank along with their prompts.
        :param docnos: List of document IDs.
        :return: List of tuples containing the player, document, rank, user prompt, and system prompt.
        """
        try:
            self.__logger.info(f"Ranking documents for round {self.__round} for query: {self.__query}")
            documents, non_cleaned_documents, user_prompts, system_prompts = zip(*documents_prompts)
            if docnos:
                ranks, scores = self.__ranker.rank(self.__query, docnos)
            else:
                ranks, scores = self.__ranker.rank(self.__query, documents)

            return sorted(zip(self.__players, documents, ranks, scores, non_cleaned_documents, user_prompts,
                              system_prompts), key=lambda x: x[2], reverse=True)
        except Exception as e:
            self.__logger.error(f"Error ranking documents: {e}")
            raise

    def create_round_history(self, ranked_players: list) -> pd.DataFrame:
        """
        Create history for the current round and update the rank of each player.

        :param ranked_players: List of ranked players along with their documents and prompts.
        :return: DataFrame containing the feedback for the round.
        """
        try:
            self.__logger.info(f"Creating feedback for round {self.__round} for query: {self.__query}")
            round_df = pd.DataFrame(columns=GAME_HISTORY_COLUMNS)
            for player, doc, rank, score, not_clean_doc, user_prompt, system_prompt in ranked_players:
                round_df.loc[len(round_df)] = [self.__round, player.get_name(), doc, not_clean_doc, rank, score,
                                               user_prompt, system_prompt]
                player.set_rank(rank)

            return round_df
        except Exception as e:
            self.__logger.error(f"Error creating feedback: {e}")
            raise

    def update_game_history(self, round_df: pd.DataFrame) -> None:
        """
        Update the game history with the results of the current round.

        :param round_df: DataFrame containing the results of the current round.
        """
        try:
            self.__logger.info(f"Updating game history for round {self.__round - 1} for query: {self.__query}")
            self.__game_history = pd.concat([self.__game_history, round_df])
            [player.generate_feedback(self.__game_history) for player in self.__players]

        except Exception as e:
            self.__logger.error(f"Error updating history: {e}")
            raise

    def get_game_history(self) -> pd.DataFrame:
        """
        Get the complete game history.

        :return: DataFrame containing the game history.
        """
        self.__game_history.loc[:, HISTORY_QUERY_ID_COLUMN] = self.__query_id
        self.__game_history.loc[:, HISTORY_GAME_ID_COLUMN] = self.__query_id

        self.__logger.info("Game history retrieved.")
        return self.__game_history

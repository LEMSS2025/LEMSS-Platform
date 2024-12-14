import pandas as pd
import os
import shutil

from utils.logger import setup_logger
from parsers.query_parser import QueryParser
from parsers.trec_parser import TrecParser
from rankers.e5 import E5
from rankers.contriever import Contriever
from rankers.okapi import Okapi
from LLMs.hugging_face_llm import HuggingFaceLLM
from LLMs.mlx_llm import MLXLLM
from agents.LLM_agent import LLMAgent
from competition.game import Game
from constants.constants import (COMPETITION_LOG_FILE, COMPETITION_LOG_NAME,
                                 CONFIG_COMPETITION_HEADER, CONFIG_AGENTS_HEADER, CONFIG_GAME_HEADER, CONFIG_INIT_DOCS_PATH_HEADER,
                                 CONFIG_RANKERS_HEADER, CONFIG_ROUND_BY_ROUND_HEADER, CONFIG_LLM_HEADER, CONFIG_LLM_MODEL_NAME_HEADER,
                                 CONFIG_GAME_ROUNDS_HEADER,
                                 HISTORY_ROUND_COLUMN, HISTORY_PLAYER_COLUMN, HISTORY_DOCNO_COLUMN, HISTORY_QUERY_ID_COLUMN, HISTORY_DOCUMENT_COLUMN,
                                 MLX_IDENTIFIER, COMPETITION_HISTORY_FILE_NAME, TRECTEXT_FILE_NAME)


class Competition:
    """
        Class responsible for managing the entire competition process, including setup, execution,
        and generating results in TREC format.
    """

    def __init__(self, config: dict):
        """
        Initialize the Competition class with competition configuration and agents configuration.

        :param config: Dictionary containing competition and agents configuration.
        """
        self.__competition_config = config[CONFIG_COMPETITION_HEADER]
        self.__agents_config = config[CONFIG_AGENTS_HEADER]
        self.__game_config = config[CONFIG_GAME_HEADER]
        self.__games_history = []
        self.__index_based_ranker = False
        self.__logger = setup_logger(COMPETITION_LOG_NAME, COMPETITION_LOG_FILE)

    def __setup_competition(self):
        """
        Set up the competition by loading queries, initializing rankers, LLMs, agents, and games.
        """
        try:
            self.__setup_queries()
            self.__setup_ranker()
            self.__setup_llm()
            self.__setup_agents()
            self.__setup_games()
        except KeyError as e:
            self.__logger.error(f"Missing competition configuration key: {e}")
            raise

    def __setup_queries(self):
        """
        Load the queries from the specified path.
        """
        try:
            self.__queries_df = QueryParser(**self.__competition_config[CONFIG_INIT_DOCS_PATH_HEADER]).query_loader()
            self.__logger.info("Queries loaded successfully.")
        except KeyError as e:
            self.__logger.error(f"Missing competition configuration key: {e}")
            raise

    def __setup_ranker(self):
        """
        Initialize the ranker model if specified in the configuration.
        """
        try:
            if 'e5' in self.__competition_config[CONFIG_RANKERS_HEADER]:
                self.ranker = E5(**self.__competition_config[CONFIG_RANKERS_HEADER]['e5'])
            elif 'contriever' in self.__competition_config[CONFIG_RANKERS_HEADER]:
                self.ranker = Contriever(**self.__competition_config[CONFIG_RANKERS_HEADER]['contriever'])
            elif 'okapi' in self.__competition_config[CONFIG_RANKERS_HEADER]:
                self.__index_based_ranker = True
                if self.__competition_config[CONFIG_ROUND_BY_ROUND_HEADER]:
                    self.ranker = Okapi(**self.__competition_config[CONFIG_RANKERS_HEADER]['okapi'],
                                        output_hash_folder=self.output_folder)
                else:
                    self.ranker = Okapi(**self.__competition_config[CONFIG_RANKERS_HEADER]['okapi'], init_index=False,
                                        output_hash_folder=self.output_folder)
            self.__logger.info("Ranker initialized successfully.")
        except KeyError as e:
            self.__logger.error(f"Missing ranker configuration key: {e}")
            raise

    def __setup_llm(self):
        """
        Set up the LLM (Large Language Model) based on the configuration.
        """
        try:
            if MLX_IDENTIFIER in self.__competition_config[CONFIG_LLM_HEADER][CONFIG_LLM_MODEL_NAME_HEADER]:
                self.llm = MLXLLM(**self.__competition_config[CONFIG_LLM_HEADER])
            else:
                self.llm = HuggingFaceLLM(**self.__competition_config[CONFIG_LLM_HEADER])

            self.__logger.info("LLM initialized successfully")
        except KeyError as e:
            self.__logger.error(f"Missing LLM configuration key: {e}")
            raise

    def __setup_agents(self):
        """
        Initialize the agents for the competition.
        """
        try:
            self.__agents = [LLMAgent(name=agent, **self.__agents_config[agent], queries_df=self.__queries_df)
                             for agent in self.__agents_config]
            self.__logger.info(f"{len(self.__agents)} agents initialized successfully.")
        except KeyError as e:
            self.__logger.error(f"Error initializing agents: {e}")
            raise

    def __setup_games(self):
        """
        Initialize the games for the competition.
        """
        try:
            self.__rounds = self.__game_config[CONFIG_GAME_ROUNDS_HEADER]
            self.__games = {query_id: Game(query_info=query_info, agents=self.__agents, ranker=self.ranker,
                                           llm=self.llm, **self.__game_config)
                            for query_id, query_info in self.__queries_df.iterrows()}
            self.__logger.info(f"{len(self.__games)} games initialized successfully.")
        except KeyError as e:
            self.__logger.error(f"Error initializing games: {e}")
            raise

    def __create_trec_text(self, combined_history: pd.DataFrame, output_folder: str):
        """
        Create a TREC text file from the combined game history.

        :param combined_history: DataFrame containing the combined game history.
        :param output_folder: Path to the output folder.
        """
        try:
            trec_parser = TrecParser([combined_history])

            agents_mapping = trec_parser.create_agent_mapping()

            combined_history[HISTORY_DOCNO_COLUMN] = combined_history.apply(lambda row: f"ROUND-{row[HISTORY_ROUND_COLUMN]:02d}-{row[HISTORY_QUERY_ID_COLUMN]}-{agents_mapping[row[HISTORY_PLAYER_COLUMN]]:02d}", axis=1)
            combined_history.to_csv(os.path.join(output_folder, COMPETITION_HISTORY_FILE_NAME), index=False)
            self.__logger.info("Competition history saved successfully.")


            trec_parser.create_trectext(os.path.join(output_folder, TRECTEXT_FILE_NAME))
            self.__logger.info("TREC text file created successfully.")
        except Exception as e:
            self.__logger.error(f"Error creating TREC text file: {e}")
            raise

    def round_by_round_competition(self):
        """
        Run the competition in a round-by-round manner.
        """
        try:
            # Iterate through each round
            for round_number in range(1, self.__rounds + 1):
                # Initialize storage for the documents and round data
                round_dfs = []
                documents_prompts = []

                # Generate documents for each game and store them in documents_prompts
                for game in self.__games:
                    documents_prompts.append(self.__games[game].generate_documents())

                # If index-based ranker is used, add new documents to the index
                if self.__index_based_ranker:
                    documents_df = pd.DataFrame(columns=[HISTORY_DOCNO_COLUMN, HISTORY_DOCUMENT_COLUMN])
                    for game in self.__games:
                        for player_id, player in enumerate(self.__agents):
                            docno = f"{game}-{round_number}-{player_id}"
                            document = documents_prompts[game][player_id][0]
                            documents_df = pd.concat(
                                [documents_df, pd.DataFrame({HISTORY_DOCNO_COLUMN: docno, HISTORY_DOCUMENT_COLUMN: document}, index=[0])],
                                ignore_index=True
                            )

                    # Add the documents to the index
                    self.ranker.add_document(documents_df)

                # Process each game's documents, rank players, and update histories
                for idx, game in enumerate(self.__games):
                    # Rank players based on the documents generated
                    if self.__index_based_ranker:
                        # Find the document IDs for the current game
                        docnos = documents_df[documents_df[HISTORY_DOCNO_COLUMN].apply(lambda x: x.split("-")[0]) == str(game)].docno.tolist()
                        ranked_players = self.__games[game].rank_documents(documents_prompts[idx], docnos)
                    else:
                        ranked_players = self.__games[game].rank_documents(documents_prompts[idx])

                    # Store round history and update game histories
                    round_dfs.append(self.__games[game].create_round_history(ranked_players))
                    self.__games[game].increase_round()

                    # Update the game's history for subsequent rounds
                    if round_number > 1:
                        self.__games_history[idx] = self.__games[game].get_game_history()
                    else:
                        self.__games_history.append(self.__games[game].get_game_history())

                # Set updated history for each agent and update game histories with round data
                for idx, game in enumerate(self.__games):
                    for agent in self.__agents:
                        agent.set_history(self.__games_history)

                    self.__games[game].update_game_history(round_dfs[idx])
                    if round_number == self.__rounds:
                        self.__games_history[idx] = self.__games[game].get_game_history()

        except Exception as e:
            self.__logger.error(f"Error running round-by-round competition: {e}")
            raise

    def game_by_game_competition(self):
        """
        Run the competition game by game.
        """
        # Iterate through each game
        for game in self.__games:
            if self.__index_based_ranker:
                # Reset the index for each game
                index_path = self.ranker.get_index_path()
                if os.path.exists(index_path):
                    shutil.rmtree(index_path)
                    self.ranker.initialize_index()

            # Iterate through each round
            for round_number in range(1, self.__rounds + 1):
                # Generate documents for the current round
                documents_prompts = self.__games[game].generate_documents()

                # If index-based ranker is used, add new documents to the index
                if self.__index_based_ranker:
                    documents_df = pd.DataFrame(columns=[HISTORY_DOCNO_COLUMN, HISTORY_DOCUMENT_COLUMN])
                    for player_id, player in enumerate(self.__agents):
                        docno = f"{round_number}-{player_id}"
                        document = documents_prompts[player_id][0]
                        documents_df = pd.concat(
                            [documents_df, pd.DataFrame({HISTORY_DOCNO_COLUMN: docno, HISTORY_DOCUMENT_COLUMN: document}, index=[0])],
                            ignore_index=True
                        )

                    # Add the documents to the index
                    self.ranker.add_document(documents_df)

                # Rank players based on the documents generated
                if self.__index_based_ranker:
                    # Extract the document IDs
                    docnos = documents_df.docno.tolist()
                    ranked_players = self.__games[game].rank_documents(documents_prompts, docnos)
                else:
                    ranked_players = self.__games[game].rank_documents(documents_prompts)

                # Create and store round history
                rounds_df = self.__games[game].create_round_history(ranked_players)
                self.__games[game].increase_round()

                # Update the game's history with the new round data
                self.__games[game].update_game_history(rounds_df)

            # Update the game's history for subsequent rounds
            self.__games_history.append(self.__games[game].get_game_history())

            # Set updated history to each agent
            for agent in self.__agents:
                agent.set_history(self.__games_history)

    def run_competition(self, output_folder: str):
        """
        Run the competition for the specified number of rounds and save the game histories.

        :param output_folder: Path to the output folder.
        :return: List of DataFrames containing the game histories.
        """
        try:
            self.output_folder = output_folder
            self.__setup_competition()
            self.__logger.info("Starting competition...")

            if self.__competition_config[CONFIG_ROUND_BY_ROUND_HEADER]:
                self.round_by_round_competition()
            else:
                self.game_by_game_competition()

            competition_history = pd.concat(self.__games_history, ignore_index=True)
            self.__create_trec_text(competition_history, output_folder)
        except Exception as e:
            self.__logger.error(f"Error running competition: {e}")
            raise
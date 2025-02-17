import pandas as pd

from utils.logger import setup_logger
from constants.constants import (TREC_PARSER_LOG_FILE, TREC_PARSER_LOG_NAME,
                                 HISTORY_PLAYER_COLUMN, HISTORY_ROUND_COLUMN, HISTORY_QUERY_ID_COLUMN,
                                 HISTORY_DOCUMENT_COLUMN)


class TrecParser:
    """
        Class responsible for creating TREC text files from game history DataFrames.
    """

    def __init__(self, history_dfs: list[pd.DataFrame]):
        """
        Initialize the TrecParser with a list of history DataFrames.

        :param history_dfs: List of DataFrames, each containing the game history.
        """
        self.__logger = setup_logger(TREC_PARSER_LOG_NAME, TREC_PARSER_LOG_FILE)

        self.__logger.info("Initializing TrecParser and combining history DataFrames.")
        try:
            self.__history_df = pd.concat(history_dfs, ignore_index=True)
            self.__agent_to_id = self.create_agent_mapping()
        except Exception as e:
            self.__logger.error(f"Error initializing TrecParser: {e}")
            raise

    def create_agent_mapping(self) -> dict:
        """
        Create a mapping from agent names to unique numeric author IDs.

        :return: Dictionary mapping agent names to author IDs.
        """
        self.__logger.info("Creating agent-to-ID mapping.")
        agents = self.__history_df[HISTORY_PLAYER_COLUMN].unique()
        agent_mapping = {agent: idx for idx, agent in enumerate(agents)}

        return agent_mapping

    def create_trectext(self, output_file: str):
        """
        Create a TREC text file from the combined history DataFrame.

        :param output_file: Path to the output TREC text file.
        """
        try:
            with open(output_file, 'w') as file:
                for index, row in self.__history_df.iterrows():
                    author_id = self.__agent_to_id[row[HISTORY_PLAYER_COLUMN]]
                    docno = f"ROUND-{row[HISTORY_ROUND_COLUMN]:02d}-{row[HISTORY_QUERY_ID_COLUMN]}-{author_id:02d}"
                    text = row[HISTORY_DOCUMENT_COLUMN].strip()

                    file.write("<DOC>\n")
                    file.write(f"<DOCNO>{docno}</DOCNO>\n")
                    file.write("<TEXT>\n")
                    file.write(f"{text}\n")
                    file.write("</TEXT>\n")
                    file.write("</DOC>\n")
        except Exception as e:
            self.__logger.error(f"Error creating TREC text file: {e}")
            raise

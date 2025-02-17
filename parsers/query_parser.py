import os
import re
import xml.etree.ElementTree as ET

import pandas as pd

from utils.logger import setup_logger
from constants.constants import (QUERY_PARSER_LOG_FILE, QUERY_PARSER_LOG_NAME, PROJECT_DIR,
                                 QUERY_DF_QUERY_COLUMN, QUERY_DF_QUERY_ID_COLUMN, QUERY_DF_DOCUMENT_COLUMN,
                                 XML_TOPIC_HEADER, XML_NUMBER_HEADER, XML_QUERY_HEADER)

class QueryParser:
    """
        Class responsible for parsing queries from XML files and documents from a TREC text file.
    """

    def __init__(self, queries_folder_path: str, docs_file_path: str):
        """
        Initialize the QueryParser with paths to the queries folder and the documents file.

        :param queries_folder_path: Path to the folder containing query XML files.
        :param docs_file_path: Path to the TREC text file containing documents.
        """
        # Load XML files from the queries folder
        queries_folder_path = os.path.join(PROJECT_DIR, queries_folder_path)
        self.__files = [os.path.join(PROJECT_DIR, queries_folder_path, file) for file in os.listdir(queries_folder_path)
                        if file.endswith('.xml')]

        # Raise an error if no XML files are found
        if not self.__files:
            raise FileNotFoundError(f"No XML files found in {queries_folder_path}")

        self.__docs_file_path = os.path.join(PROJECT_DIR, docs_file_path)
        self.__logger = setup_logger(QUERY_PARSER_LOG_NAME, QUERY_PARSER_LOG_FILE)

    def __parse_queries(self) -> dict:
        """
        Parse queries from XML files in the specified folder.

        :return: Dictionary of queries with query IDs as keys.
        """
        queries = {}

        for file_path in self.__files:
            try:
                tree = ET.parse(file_path)
                root = tree.getroot()
                for topic in root.findall(XML_TOPIC_HEADER):
                    qid = topic.get(XML_NUMBER_HEADER)
                    query_text = topic.find(XML_QUERY_HEADER).text.strip()
                    queries[qid] = {QUERY_DF_QUERY_COLUMN: query_text}
            except ET.ParseError as e:
                self.__logger.error(f"Error parsing XML file {file_path}: {e}")
            except Exception as e:
                self.__logger.error(f"Unexpected error processing file {file_path}: {e}")

        max_query = max(map(int, queries.keys()))
        queries = {qid.zfill(len(str(max_query))): queries[qid] for qid in queries}
        return queries

    def __parse_trectext(self) -> pd.DataFrame:
        """
        Parse documents from a TREC text file.

        :return: DataFrame containing parsed documents.
        """
        try:
            with open(self.__docs_file_path, 'r', encoding="utf8") as file:
                content = file.read()
        except IOError as e:
            self.__logger.error(f"Error reading TREC text file from {self.__docs_file_path}: {e}")
            raise

        docs = re.findall(r'<DOC>(.*?)</DOC>', content, re.DOTALL)
        parsed_docs = []

        if not docs:
            self.__logger.error("No documents found in TREC text file.")
            raise ValueError("No documents found in TREC text file.")

        for doc in docs:
            try:
                query_id = re.search(r'<DOCNO>(.*?)</DOCNO>', doc).group(1).split('-')[2]
                text = re.search(r'<TEXT>(.*?)</TEXT>', doc, re.DOTALL).group(1).strip()
                parsed_docs.append((query_id, text))
            except AttributeError as e:
                self.__logger.error(f"Error parsing document: {e}")

        return pd.DataFrame(parsed_docs, columns=[QUERY_DF_QUERY_ID_COLUMN, QUERY_DF_DOCUMENT_COLUMN])

    def query_loader(self) -> pd.DataFrame:
        """
        Load queries and documents, and merge them into a single DataFrame.

        :return: DataFrame containing queries and their corresponding documents.
        """
        queries = self.__parse_queries()
        docs = self.__parse_trectext()

        docs[QUERY_DF_QUERY_COLUMN] = docs[QUERY_DF_QUERY_ID_COLUMN].apply(lambda x: queries.get(
            x, {}).get(QUERY_DF_QUERY_COLUMN, ''))
        if docs[QUERY_DF_QUERY_COLUMN].isnull().any():
            self.__logger.warning("Some documents do not have corresponding queries.")

        return docs

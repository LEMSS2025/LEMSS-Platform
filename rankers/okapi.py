import os
from typing import List, Tuple

import pandas as pd
import ir_datasets

from rankers.index_ranker import IndexRanker
from utils.logger import setup_logger
from constants.constants import OKAPI_RANKER_LOG_FILE, OKAPI_RANKER_LOG_NAME


class Okapi(IndexRanker):
    def __init__(self, index_name: str, init_index: bool = True, output_hash_folder: str = None):
        """
        Initialize the Okapi ranker model.
        """
        super().__init__("Okapi", index_name)
        self.__logger = setup_logger(OKAPI_RANKER_LOG_NAME, OKAPI_RANKER_LOG_FILE)
        self.set_index_path(index_name, output_hash_folder)
        self.__indexer_args = ["-index", self.index_path, "-storeDocvectors", "-storeContents", "-stemmer", "krovetz",
                               "-keepStopwords"]

        if init_index:
            self.initialize_index()

    def set_index_path(self, index_name: str, output_hash_folder: str):
        """
        Set the path to the index.

        :param index_name: Name of the index.
        :param output_hash_folder: Path to the output hash folder.
        """
        self.index_path = os.path.join(output_hash_folder, index_name)

    def get_index_path(self):
        """
        Get the path to the index.
        """
        return self.index_path

    def initialize_index(self):
        from pyserini.index.lucene import LuceneIndexer
        """
        Initialize the index with the Wiki-IR dataset.
        """
        self.__logger.info("Loading Wiki-IR dataset...")
        dataset = ir_datasets.load("wikir/en59k")
        self.__logger.info("Wiki-IR dataset loaded successfully.")

        def process_doc(i, doc):
            return {"id": f"en59k-{i}", "contents": doc.text}

        # Convert the dataset iterator to a list once to avoid repeated access
        docs_iter = list(dataset.docs_iter())
        docs = [process_doc(i, doc) for i, doc in enumerate(docs_iter)]

        # Initialize and update the index
        self.__logger.info(f"Initializing index at {self.index_path}")
        indexer = LuceneIndexer(append=True, args=self.__indexer_args)

        # Add the batch of documents to the index
        indexer.add_batch_dict(docs)
        indexer.close()

        self.__logger.info("Index initialized successfully.")

    def add_document(self, documents_df: pd.DataFrame):
        from pyserini.index.lucene import LuceneIndexer
        """
        Add the new generated documents to the index.

        :param documents_df: DataFrame containing the documents to be indexed.
        """
        # Initialize the LuceneIndexer with append mode to add documents to the existing index
        indexer = LuceneIndexer(append=True, args=self.__indexer_args)

        # Prepare the documents in a format suitable for the Lucene indexer
        docs = [{"id": row.docno, "contents": row.document} for row in documents_df.itertuples(index=False)]

        # Add the batch of documents to the index
        indexer.add_batch_dict(docs)
        indexer.close()

        self.__logger.info("Documents added to the index successfully.")

    def rank(self, query: str, docnos: List[str]) -> Tuple[List[int], List[float]]:
        from pyserini.index.lucene import IndexReader
        from pyserini.analysis import get_lucene_analyzer
        """
        Rank documents based on Okapi BM25 similarity to the query.

        :param query: A single query string.
        :param docnos: List of document IDs.
        :return: List of scores representing the similarity between the query and each document.
        """
        try:
            self.__logger.info(f"Ranking {len(docnos)} documents for query: {query}")

            # Initialize the IndexReader to read the index
            index_reader = IndexReader(self.index_path)

            scores = []
            for docno in docnos:
                # Compute BM25 score for each document
                bm25_score = index_reader.compute_bm25_term_weight(docno, query,
                                                                   analyzer=get_lucene_analyzer(stemmer='krovetz'))
                scores.append(bm25_score)

            # Use the base class's tie breaker to rank documents
            ranked_scores, scores = super().tie_breaker(scores)

            return ranked_scores, scores
        except Exception as e:
            self.__logger.error(f"Error in ranking documents: {e}")
            raise

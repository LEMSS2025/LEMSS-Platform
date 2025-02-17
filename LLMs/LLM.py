from abc import ABC, abstractmethod
import re

import torch

from utils.logger import setup_logger
from constants.constants import LLM_LOG_FILE, LLM_LOG_NAME, CLEANING_PROMPT


class LLM(ABC):
    """
        Abstract base class for a Large Language Model (LLM) that provides the interface for generating and cleaning documents.
    """

    def __init__(self, model_name: str, temperature: float, token: str):
        """
        Initialize the LLM with the given model name and temperature.

        :param model_name: The name of the model used by the LLM.
        :param temperature: The temperature parameter for controlling randomness in text generation.
        :param token: The token used to separate the user and system prompts.
        """
        self.model_name = model_name
        self.temperature = temperature
        self.token = token
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.__logger = setup_logger(LLM_LOG_NAME, LLM_LOG_FILE)
        self.__logger.info(f"LLM initialized with model: {model_name} on device: {self.device}")

    @abstractmethod
    def generate_prompt(self, user: str, system: str, max_tokens: int, clean: bool = True,
                        force_max_tokens: bool = False) -> str:
        """
        Abstract method to generate a document based on user and system prompts.

        :param user: The user prompt.
        :param system: The system prompt.
        :param max_tokens: Maximum number of tokens for the generated document.
        :param clean: Whether to clean the document of extraneous text or not.
        :param force_max_tokens: Whether to manually restrict the number of tokens in the generated document.

        :return: The generated document as a string.
        """
        pass

    def clean_document(self, doc: str, max_tokens: int, model=None) -> str:
        """
        Clean the generated document to remove unnecessary prompts and tags.

        :param doc: The raw generated document.
        :param max_tokens: Maximum number of tokens allowed for the cleaned document.
        :param model: Instance of the LLM model used for generating the cleaned document.
        :return: Cleaned document text.
        """
        if model is None:
            self.__logger.error("Model not provided for document cleaning.")
            raise ValueError("Model must be provided to clean the document.")
        try:
            # Generate the cleaned document using the LLM
            cleaned_document = model.generate_prompt(doc, CLEANING_PROMPT, max_tokens, clean=False)

            # Regular expression to match and remove specific tags and their content (up to 20 characters)
            pattern = re.compile(r'<(ROUND|RANK|PLAYER)>.{0,20}?</\1>', re.DOTALL)
            cleaned_document = re.sub(pattern, '', cleaned_document)

            # Remove specific tags and strip extra whitespace
            cleaned_document = (
                cleaned_document.replace("<DOC>", "")
                .replace("<TEXT>", "")
                .replace("</DOC>", "")
                .replace("</TEXT>", "")
                .strip()
            )

            return cleaned_document
        except Exception as e:
            self.__logger.error(f"Error cleaning document: {e}")
            raise

import pandas as pd

from constants.constants import NUM_TO_STR, HISTORY_ROUND_COLUMN, HISTORY_DOCUMENT_COLUMN, HISTORY_RANK_COLUMN


# Rename it to PromptAgent
class PromptManager:
    """
        Class responsible for building system and user prompts based on the provided format string and feedback data.
    """

    def __init__(self, format_string: str):
        """
        Initialize the PromptManager with the provided format string.

        :param format_string: The format string template for the prompt.
        """
        self.__format_string = format_string

    def build_system_prompt(self, query: str, current_doc: str, character: str) -> str:
        """
        Build the system prompt using the format string, query, document, and character.

        :param query: The candidate query.
        :param current_doc: The candidate document.
        :param character: The character/persona to be used in the prompt.
        :return: A formatted system prompt as a string.
        """
        prompt = self.__format_string
        prompt += (
            f"Input:\n\n - Candidate Query: {query}\n\n - Candidate Document: {current_doc}\n\n - Character: {character}"
        )

        return prompt

    def build_user_prompt(self, pairwise_feedback: pd.DataFrame, all_feedback: pd.DataFrame, query: str) -> str:
        """
        Build the user prompt based on pairwise or general feedback, using the query as context.

        :param pairwise_feedback: DataFrame containing pairwise feedback for the query.
        :param all_feedback: DataFrame containing all feedback for the query.
        :param query: The query to which the feedback relates.
        :return: A formatted user prompt as a string.
        """
        if pairwise_feedback is not None:
            return self.__build_pairwise_prompt(pairwise_feedback, query)
        else:
            return self.__build_general_prompt(all_feedback, query)

    def __build_pairwise_prompt(self, pairwise_feedback: pd.DataFrame, query: str) -> str:
        """
        Build the user prompt based on pairwise feedback.

        :param pairwise_feedback: DataFrame containing pairwise feedback for the query.
        :param query: The query to which the feedback relates.
        :return: A formatted user prompt as a string.
        """
        max_round = int(pairwise_feedback[HISTORY_ROUND_COLUMN].max())
        min_round = int(pairwise_feedback[HISTORY_ROUND_COLUMN].min())
        prompt = ""

        for _round in pairwise_feedback[HISTORY_ROUND_COLUMN].unique():
            _round = int(_round)
            round_pairwise = pairwise_feedback[pairwise_feedback[HISTORY_ROUND_COLUMN] == _round]
            round_str = NUM_TO_STR[max_round - _round + 1]

            prompt += f"\n\nquery: {query}\n\n"
            if _round == max_round:
                prompt += f"* document: {round_pairwise.iloc[0][HISTORY_DOCUMENT_COLUMN]}\n\n{round_str} ranking: {round_pairwise.iloc[0][HISTORY_RANK_COLUMN]}\n\n\n"
                prompt += f"* document: {round_pairwise.iloc[1][HISTORY_DOCUMENT_COLUMN]}\n\n{round_str} ranking: {round_pairwise.iloc[1][HISTORY_RANK_COLUMN]}\n\n\n"
            else:
                prompt += f"* document: {round_pairwise.iloc[0][HISTORY_DOCUMENT_COLUMN]}\n\n{round_str} to latest ranking: {round_pairwise.iloc[0][HISTORY_RANK_COLUMN]}\n\n\n"
                if _round == min_round:
                    prompt += f"* document: {round_pairwise.iloc[1][HISTORY_DOCUMENT_COLUMN]}\n\n{round_str} to latest ranking: {round_pairwise.iloc[1][HISTORY_RANK_COLUMN]}"
                else:
                    prompt += f"* document: {round_pairwise.iloc[1][HISTORY_DOCUMENT_COLUMN]}\n\n{round_str} to latest ranking: {round_pairwise.iloc[1][HISTORY_RANK_COLUMN]}\n\n\n"

        return prompt

    def __build_general_prompt(self, all_feedback: pd.DataFrame, query: str) -> str:
        """
        Build the user prompt based on general feedback.

        :param all_feedback: DataFrame containing all feedback for the query.
        :param query: The query to which the feedback relates.
        :return: A formatted user prompt as a string.
        """
        max_round = int(all_feedback[HISTORY_ROUND_COLUMN].max())
        prompt = f"\n\nquery: {query}\n\n"

        for _round in all_feedback[HISTORY_ROUND_COLUMN].unique():
            _round = int(_round)
            round_all = all_feedback[all_feedback[HISTORY_ROUND_COLUMN] == _round]
            round_str = NUM_TO_STR[max_round - _round + 1]

            if _round == max_round:
                prompt += f"* documents ordered by {round_str} ranking from highest to lowest in relation to the query:\n\n\n"

                for _, row in round_all.iterrows():
                    prompt += f"* {row[HISTORY_DOCUMENT_COLUMN]}\n\n\n"
            else:
                prompt += f"* documents ranked by {round_str} to latest ranking from highest to lowest in relation to the query:\n"
                for _, row in round_all.iterrows():
                    prompt += f"{row[HISTORY_RANK_COLUMN]}. {row[HISTORY_DOCUMENT_COLUMN]}\n"

            prompt += "\n\n\n"

        return prompt

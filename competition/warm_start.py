import pandas as pd


class WarmStart:
    def __init__(self, competition_history_path: str):
        """
        Initialize the WarmStart class with the competition history.

        :param competition_history_path: Path to the competition history.
        """
        self.competition_history_df = pd.read_csv(competition_history_path)
        self.competition_history_df['query_id'] = self.competition_history_df['query_id'].astype(int)
        self.competition_history_df['game_id'] = self.competition_history_df['game_id'].astype(int)

    def set_player(self, player_name: str, query_id: int):
        """
        Set the player's history based on the competition history.

        :param player_name: Name of the player.
        :param query_id: ID of the query.
        """
        player_history = self.competition_history_df[
            (self.competition_history_df['player'] == player_name)
            & (self.competition_history_df['query_id'] == query_id)
        ]

        if player_history.empty:
            return 1, None, None, None, None

        round = player_history['round'].max()
        document = player_history[player_history['round'] == round]['document'].values[0]
        init_document = player_history[player_history['round'] == 0]['document'].values[0]
        rank = player_history[player_history['round'] == round]['rank'].values[0]

        game_history = self.competition_history_df[self.competition_history_df["query_id"] == query_id]

        return round, document, init_document, rank, game_history

    def set_game(self, query_id: int):
        """
        Set the game history based on the competition history.

        :param query_id: ID of the query.
        """
        game_history = self.competition_history_df[self.competition_history_df["query_id"] == query_id]
        round = game_history['round'].max()

        return game_history, round

    def get_last_run(self):
        last_query_id = self.competition_history_df.iloc[-1]['query_id']
        last_round = self.competition_history_df.iloc[-1]['round']

        return last_query_id, last_round

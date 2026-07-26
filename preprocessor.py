"""
Preprocessing module for champion encoding, feature extraction, and matrix generation.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from data_fetcher import EsportsDataFetcher


class ChampionEncoder:
    """Handles mapping between champion names, zero-indexed IDs, and feature representations."""

    NAME_ALIASES = {
        "Nunu": "Nunu & Willump"
    }

    def __init__(self, champion_names: Optional[List[str]] = None):
        self.champions = champion_names or []
        self._champ_to_id: Dict[str, int] = {
            name: idx for idx, name in enumerate(self.champions)
        }

    def update_champions(self, champion_names: List[str]) -> None:
        """Updates the internal lookup table."""
        self.champions = champion_names
        self._champ_to_id = {name: idx for idx, name in enumerate(self.champions)}

    @property
    def num_champions(self) -> int:
        """Returns the total number of unique champions registered."""
        return len(self.champions)

    def encode_id(self, name: str) -> int:
        """Maps a champion name to its zero-indexed ID (0 to N-1). Returns 0 if unknown."""
        normalized_name = self.NAME_ALIASES.get(name, name)
        return self._champ_to_id.get(normalized_name, 0)

    def encode_team_ids(self, team_picks: List[str], target_length: int = 5) -> List[int]:
        """Converts a team draft list into a fixed-length list of champion IDs."""
        ids = [self.encode_id(name) for name in team_picks[:target_length]]
        while len(ids) < target_length:
            ids.append(0)  # Padding for missing slots
        return ids

    def encode_sparse_difference(self, team1_picks: List[str], team2_picks: List[str]) -> List[float]:
        """
        Creates a sparse difference vector for XGBoost/tree models.
        +1.0 for Team 1 picks, -1.0 for Team 2 picks across champion vocabulary.
        """
        vec = [0.0] * max(self.num_champions, 1)
        for name in team1_picks:
            idx = self.encode_id(name)
            if 0 <= idx < len(vec):
                vec[idx] += 1.0

        for name in team2_picks:
            idx = self.encode_id(name)
            if 0 <= idx < len(vec):
                vec[idx] -= 1.0

        return vec


class MatchPreprocessor:
    """Processes raw match data into structured neural network and tree model features."""

    def __init__(self, fetcher: EsportsDataFetcher):
        self.fetcher = fetcher
        champion_names = self.fetcher.fetch_champion_names()
        self.encoder = ChampionEncoder(champion_names)

    def process_match_records(
        self,
        raw_games: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Transforms raw Cargo match records into structured feature matrices:
        - team1_ids: (N, 5) array of champion IDs
        - team2_ids: (N, 5) array of champion IDs
        - sparse_diffs: (N, num_champions) difference vectors for XGBoost
        - y: (N,) array of win indicators (0 or 1)
        """
        team1_ids_list: List[List[int]] = []
        team2_ids_list: List[List[int]] = []
        sparse_diffs_list: List[List[float]] = []
        labels: List[int] = []

        for record in raw_games:
            parsed = self.fetcher.parse_match_record(record)
            t1_picks = parsed["team1_picks"]
            t2_picks = parsed["team2_picks"]
            winner = parsed["winner"]

            if not t1_picks or not t2_picks:
                continue

            t1_ids = self.encoder.encode_team_ids(t1_picks)
            t2_ids = self.encoder.encode_team_ids(t2_picks)
            sparse_diff = self.encoder.encode_sparse_difference(t1_picks, t2_picks)

            team1_ids_list.append(t1_ids)
            team2_ids_list.append(t2_ids)
            sparse_diffs_list.append(sparse_diff)
            labels.append(winner)

        return {
            "team1_ids": np.array(team1_ids_list, dtype=np.int32) if team1_ids_list else np.empty((0, 5), dtype=np.int32),
            "team2_ids": np.array(team2_ids_list, dtype=np.int32) if team2_ids_list else np.empty((0, 5), dtype=np.int32),
            "sparse_diffs": np.array(sparse_diffs_list, dtype=np.float32) if sparse_diffs_list else np.empty((0, self.encoder.num_champions), dtype=np.float32),
            "y": np.array(labels, dtype=np.float32) if labels else np.empty((0,), dtype=np.float32)
        }

    def preprocess_patch(
        self,
        patch: str,
        offset: int = 0,
        limit: int = 500
    ) -> Dict[str, Any]:
        """Extracts and converts patch match records into feature matrices."""
        raw_games = self.fetcher.fetch_scoreboard_games(patch=patch, limit=limit, offset=offset)
        return self.process_match_records(raw_games)

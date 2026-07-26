"""
Legacy compatibility wrapper for dataset generation.
"""

from dataset_manager import DatasetManager
from data_fetcher import EsportsDataFetcher
from preprocessor import MatchPreprocessor


def write_db(championRole: str = "withRole"):
    """Builds and serializes match dataset via DatasetManager."""
    manager = DatasetManager(MatchPreprocessor(EsportsDataFetcher()))
    X, y = manager.build_dataset(champion_type=championRole)
    manager.save_dataset(X, y, f"{championRole}.json")
    return X, y


if __name__ == "__main__":
    write_db()

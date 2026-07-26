"""
Legacy compatibility wrapper for retrieving match datasets.
"""

from dataset_manager import DatasetManager
from data_fetcher import EsportsDataFetcher
from preprocessor import MatchPreprocessor

_manager = DatasetManager(MatchPreprocessor(EsportsDataFetcher()))


def get_db(championRole: str = "withRole"):
    """Loads feature matrix X and target y using DatasetManager."""
    return _manager.load_dataset(championRole)

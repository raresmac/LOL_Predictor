"""
Script to fetch and build an expanded esports match dataset from Leaguepedia Cargo API.
"""

import sys
from data_fetcher import EsportsDataFetcher
from preprocessor import MatchPreprocessor
from dataset_manager import DatasetManager


def main():
    max_records = int(sys.argv[1]) if len(sys.argv) > 1 else 20000

    print(f"=== LOL Predictor Dataset Builder ===")
    print(f"Target record count: {max_records}")

    fetcher = EsportsDataFetcher()
    preprocessor = MatchPreprocessor(fetcher)
    manager = DatasetManager(preprocessor)

    # Fetch records and save dataset
    dataset = manager.fetch_entire_esports_dataset(max_records=max_records)
    saved_path = manager.save_dataset(dataset, filename="expanded_dataset.json")

    print(f"\nDataset successfully built and saved to: {saved_path}")
    print(f"To train on this new dataset, run:\npython main.py {saved_path}")


if __name__ == "__main__":
    main()

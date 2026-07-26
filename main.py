"""
Main entry point for training and benchmarking League of Legends match outcome predictor models.
"""

import sys
import os
import numpy as np
from data_fetcher import EsportsDataFetcher
from preprocessor import MatchPreprocessor
from dataset_manager import DatasetManager
from model import SiameseEmbeddingPredictor, XGBoostPredictorModel


def run_pipeline(dataset_file: str = "bot.txt") -> None:
    """Runs data loading, preprocessing, model training, and side-by-side model comparison."""
    print("=== League of Legends Match Predictor ===")

    fetcher = EsportsDataFetcher()
    preprocessor = MatchPreprocessor(fetcher)
    manager = DatasetManager(preprocessor)

    # Load dataset
    if not os.path.exists(dataset_file):
        print(f"Dataset file '{dataset_file}' not found. Extracting sample dataset...")
        dataset = preprocessor.preprocess_patch(patch="14.1", limit=200)
    else:
        print(f"Loading dataset from '{dataset_file}'...")
        dataset = manager.load_dataset(dataset_file)

    team1_ids = dataset["team1_ids"]
    team2_ids = dataset["team2_ids"]
    sparse_diffs = dataset["sparse_diffs"]
    labels = dataset["y"]

    print(f"Loaded {len(labels)} match records.")
    if len(labels) == 0:
        print("Error: The specified dataset contains 0 records. Please run 'python build_dataset.py 5000' first to generate a dataset.")
        return
    num_champs = preprocessor.encoder.num_champions

    # 1. Siamese Champion Embedding Neural Network
    print("\n--- Training Siamese Embedding Neural Network ---")
    nn_predictor = SiameseEmbeddingPredictor(
        num_champions=num_champs,
        embedding_dim=32,
        hidden_units=(128, 64),
        dropout_rate=0.2,
        learning_rate=0.001
    )
    nn_loss, nn_acc = nn_predictor.train(
        team1_ids=team1_ids,
        team2_ids=team2_ids,
        labels=labels,
        test_size=0.2,
        batch_size=64,
        epochs=25
    )

    # 2. XGBoost Benchmark Classifier
    print("\n--- Training XGBoost Benchmark Model ---")
    xgb_predictor = XGBoostPredictorModel(n_estimators=200, max_depth=4, learning_rate=0.05)
    _, xgb_acc = xgb_predictor.train(
        sparse_diffs=sparse_diffs,
        labels=labels,
        test_size=0.2
    )

    print("\n=== Model Benchmark Results ===")
    print(f"Siamese Embedding NN Test Loss: {nn_loss:.4f}")
    print(f"Siamese Embedding NN Accuracy:  {nn_acc * 100:.2f}%")
    if xgb_acc > 0:
        print(f"XGBoost Benchmark Accuracy:     {xgb_acc * 100:.2f}%")


if __name__ == "__main__":
    target_dataset = sys.argv[1] if len(sys.argv) > 1 else "bot.txt"
    run_pipeline(target_dataset)

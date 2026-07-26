"""
Dataset manager module for building, saving, and loading preprocessed match dataset files.
"""

import json
import time
import os
from typing import List, Tuple, Dict, Any
import numpy as np
from preprocessor import MatchPreprocessor


class DatasetManager:
    """Handles serialization, storage, and batch collection of match datasets."""

    def __init__(self, preprocessor: MatchPreprocessor, data_dir: str = "data"):
        self.preprocessor = preprocessor
        self.data_dir = data_dir
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir, exist_ok=True)

    def fetch_entire_esports_dataset(
        self,
        max_records: int = 50000,
        page_size: int = 100
    ) -> Dict[str, Any]:
        """
        Fetches match records directly from Leaguepedia Cargo API using offset pagination.
        Handles Fandom API rate limits with automatic retries.
        """
        print(f"Fetching up to {max_records} games from Leaguepedia Cargo API...")
        
        all_raw_games: List[Dict[str, Any]] = []
        offset = 0

        while offset < max_records:
            print(f"Fetching records {offset} to {offset + page_size}...")
            params = {
                "action": "cargoquery",
                "tables": "ScoreboardGames",
                "fields": "Team1, Team2, Team1Picks, Team2Picks, Winner",
                "limit": str(page_size),
                "offset": str(offset),
                "format": "json"
            }
            
            success = False
            for retry in range(5):
                try:
                    res = self.preprocessor.fetcher.session.get(
                        self.preprocessor.fetcher.API_URL,
                        params=params,
                        timeout=self.preprocessor.fetcher.timeout
                    )
                    res.raise_for_status()
                    data = res.json()
                    
                    if "error" in data:
                        err_code = data["error"].get("code", "")
                        if err_code == "ratelimited":
                            wait_time = 3 * (retry + 1)
                            print(f"Rate limited by API. Waiting {wait_time} seconds (attempt {retry + 1}/5)...")
                            time.sleep(wait_time)
                            continue
                        else:
                            print(f"API Error Response: {data['error']}")
                            break

                    cargo_results = data.get("cargoquery", [])
                    if not cargo_results:
                        print("No further match records returned by API.")
                        success = True
                        break

                    page_games = [item["title"] for item in cargo_results if "title" in item]
                    all_raw_games.extend(page_games)
                    offset += len(page_games)
                    success = True

                    if len(page_games) < page_size:
                        break

                    time.sleep(0.5)  # Gentle delay between successful requests
                    break
                except Exception as e:
                    print(f"API request failed at offset {offset} (attempt {retry + 1}/5): {e}")
                    time.sleep(3)

            if not success or (offset > 0 and len(all_raw_games) % page_size != 0):
                if not success:
                    break

        print(f"Successfully fetched {len(all_raw_games)} total match records from Cargo API.")
        return self.preprocessor.process_match_records(all_raw_games)

    def save_dataset(self, dataset: Dict[str, np.ndarray], filename: str = "expanded_dataset.json") -> str:
        """Saves a processed dataset dictionary to a JSON file."""
        file_path = os.path.join(self.data_dir, filename)
        
        t1_ids = dataset["team1_ids"].tolist()
        t2_ids = dataset["team2_ids"].tolist()
        y_vals = dataset["y"].tolist()

        serialized = [
            {
                "X": t1_ids[i] + t2_ids[i],
                "y": float(y_vals[i])
            }
            for i in range(len(y_vals))
        ]

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(serialized, f)

        print(f"Saved dataset with {len(serialized)} records to '{file_path}'.")
        return file_path

    def load_dataset(self, filename: str) -> Dict[str, np.ndarray]:
        """
        Loads feature matrices from JSON file or legacy format:
        Returns structured dictionary containing 'team1_ids', 'team2_ids', 'sparse_diffs', and 'y'.
        """
        file_path = filename if os.path.isabs(filename) else os.path.join(self.data_dir, filename)
        if not os.path.exists(file_path) and not file_path.endswith(".txt") and not file_path.endswith(".json"):
            file_path = os.path.join(self.data_dir, f"{filename}.txt")

        if not os.path.exists(file_path):
            file_path = filename

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file '{filename}' was not found.")

        with open(file_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        team1_ids_list = []
        team2_ids_list = []
        sparse_diffs_list = []
        y_list = []

        encoder = self.preprocessor.encoder

        for item in raw_data:
            if "X" in item and "y" in item:
                raw_x = item["X"]
                y_val = item["y"]
                
                if len(raw_x) >= 10:
                    t1_ids = [val % encoder.num_champions for val in raw_x[:5]]
                    t2_ids = [val % encoder.num_champions for val in raw_x[5:10]]
                else:
                    t1_ids = [(val % encoder.num_champions) for val in raw_x[:len(raw_x)//2]]
                    t2_ids = [(val % encoder.num_champions) for val in raw_x[len(raw_x)//2:]]
                    while len(t1_ids) < 5: t1_ids.append(0)
                    while len(t2_ids) < 5: t2_ids.append(0)

                sparse_vec = [0.0] * max(encoder.num_champions, 1)
                for idx in t1_ids:
                    if idx < len(sparse_vec): sparse_vec[idx] += 1.0
                for idx in t2_ids:
                    if idx < len(sparse_vec): sparse_vec[idx] -= 1.0

                team1_ids_list.append(t1_ids)
                team2_ids_list.append(t2_ids)
                sparse_diffs_list.append(sparse_vec)
                y_list.append(y_val)

        return {
            "team1_ids": np.array(team1_ids_list, dtype=np.int32),
            "team2_ids": np.array(team2_ids_list, dtype=np.int32),
            "sparse_diffs": np.array(sparse_diffs_list, dtype=np.float32),
            "y": np.array(y_list, dtype=np.float32)
        }

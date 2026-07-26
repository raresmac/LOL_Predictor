# League of Legends Match Predictor

A machine learning pipeline and neural network architecture built with TensorFlow/Keras, Scikit-Learn, XGBoost, and mwrogue that predicts League of Legends competitive match outcomes based on champion team compositions.

---

## Highlights and Features

- Object-Oriented Architecture: Modular design separating API querying, feature encoding, dataset storage, and neural network modeling.
- Champion Embeddings: Learns 32-dimensional latent feature representations for each champion in the game, replacing ordinal integer multipliers.
- Siamese Network Design: Utilizes a dual-input symmetric architecture (Team 1 vs Team 2) with shared embedding weights and composition difference layers.
- XGBoost Benchmarking: Includes a Gradient Boosted Decision Tree classifier trained on sparse composition difference vectors for performance benchmarking.
- Cargo API Data Ingestion: Live esports match data extraction via mwrogue (Leaguepedia Cargo API) with offline fallback support.

---

## Project Architecture

```
LOL_Predictor/
├── data_fetcher.py       # EsportsDataFetcher: Interfaces with Leaguepedia Cargo API
├── preprocessor.py       # ChampionEncoder & MatchPreprocessor: Maps picks to ID sequences and difference vectors
├── dataset_manager.py    # DatasetManager: Handles dataset building, saving & loading
├── model.py              # SiameseEmbeddingPredictor & XGBoostPredictorModel architectures
├── main.py               # Main CLI pipeline entry point
└── requirements.txt      # Dependency specifications
```

---

## Getting Started

### Setup & Installation

1. Create a Python virtual environment:

```bash
python -m venv .venv
```

2. Activate the virtual environment:

- On Windows (PowerShell):
```powershell
.\.venv\Scripts\Activate.ps1
```

- On Windows (CMD):
```cmd
.\.venv\Scripts\activate.bat
```

3. Install project dependencies:

```bash
pip install -r requirements.txt
```

### Running the Predictor & Benchmark

To train and evaluate both the Siamese Embedding Neural Network and the XGBoost Benchmark model on an existing dataset (e.g. `bot.txt`):

```bash
python main.py bot.txt
```

### Building an Expanded Live Dataset

To download new match records directly from the live Leaguepedia Cargo API:

```bash
python build_dataset.py 20000
```

This will fetch the latest 20,000 competitive games and save them to `data/expanded_dataset.json`.

Then train on the new expanded dataset:

```bash
python main.py data/expanded_dataset.json
```

---

## Model Architecture Details

### 1. Siamese Embedding Neural Network
- Champion Embeddings: Embedding(num_champions, 32)
- Team Representation: GlobalAveragePooling1D across 5 champion slots per team
- Symmetry Layer: Subtract([Vector_Team1, Vector_Team2])
- Classification Head: Dense(128, ReLU) -> Dropout(0.2) -> Dense(64, ReLU) -> Dropout(0.2) -> Dense(1, Sigmoid)
- Optimizer: Adam (learning rate = 0.001)

### 2. XGBoost Benchmark Classifier
- Feature Matrix: Sparse difference vector (+1 for Team 1 picks, -1 for Team 2 picks)
- Estimators: 200 trees, max depth = 4, learning rate = 0.05

---

## License

This project is open-source and available under the MIT License.

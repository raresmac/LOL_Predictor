"""
Machine learning models for League of Legends match outcome prediction,
supporting Keras/TensorFlow, Scikit-Learn MLPClassifier, and XGBoost benchmarking.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

try:
    import tensorflow as tf  # type: ignore
    HAS_TENSORFLOW = True
except ImportError:
    tf = None
    HAS_TENSORFLOW = False

try:
    import xgboost as xgb  # type: ignore
    HAS_XGBOOST = True
except ImportError:
    xgb = None
    HAS_XGBOOST = False


class SiameseEmbeddingPredictor:
    """
    Symmetric Neural Network using learned Champion Embeddings or Scikit-Learn MLP fallback.
    Uses shared weights for Team 1 and Team 2 composition encoding when TensorFlow is available.
    """

    def __init__(
        self,
        num_champions: int = 170,
        embedding_dim: int = 32,
        hidden_units: Tuple[int, int] = (128, 64),
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001
    ):
        self.num_champions = max(num_champions, 1)
        self.embedding_dim = embedding_dim
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.tf_model: Optional[Any] = None
        self.mlp_model: Optional[MLPClassifier] = None

    def build_tf_model(self) -> Any:
        """Constructs a dual-input Siamese Keras architecture with shared champion embeddings."""
        if not HAS_TENSORFLOW or tf is None:
            return None

        team1_input = tf.keras.layers.Input(shape=(5,), name="team1_ids")
        team2_input = tf.keras.layers.Input(shape=(5,), name="team2_ids")

        embedding_layer = tf.keras.layers.Embedding(
            input_dim=self.num_champions + 1,
            output_dim=self.embedding_dim,
            name="champion_embeddings"
        )

        t1_embedded = embedding_layer(team1_input)
        t2_embedded = embedding_layer(team2_input)

        t1_vector = tf.keras.layers.GlobalAveragePooling1D()(t1_embedded)
        t2_vector = tf.keras.layers.GlobalAveragePooling1D()(t2_embedded)

        diff_vector = tf.keras.layers.Subtract()([t1_vector, t2_vector])

        x = tf.keras.layers.Dense(self.hidden_units[0], activation="relu")(diff_vector)
        x = tf.keras.layers.Dropout(self.dropout_rate)(x)
        x = tf.keras.layers.Dense(self.hidden_units[1], activation="relu")(x)
        x = tf.keras.layers.Dropout(self.dropout_rate)(x)
        output = tf.keras.layers.Dense(1, activation="sigmoid", name="win_probability")(x)

        model = tf.keras.Model(inputs=[team1_input, team2_input], outputs=output)
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

        self.tf_model = model
        return model

    def train(
        self,
        team1_ids: np.ndarray,
        team2_ids: np.ndarray,
        labels: np.ndarray,
        test_size: float = 0.2,
        batch_size: int = 64,
        epochs: int = 30,
        random_state: int = 42
    ) -> Tuple[float, float]:
        """Trains the embedding neural network (or MLP fallback) and returns loss and accuracy metrics."""
        if HAS_TENSORFLOW and tf is not None:
            if self.tf_model is None:
                self.build_tf_model()

            indices = np.arange(len(labels))
            train_idx, test_idx = train_test_split(
                indices, test_size=test_size, random_state=random_state
            )

            t1_train, t1_test = team1_ids[train_idx], team1_ids[test_idx]
            t2_train, t2_test = team2_ids[train_idx], team2_ids[test_idx]
            y_train, y_test = labels[train_idx], labels[test_idx]

            self.tf_model.fit(
                [t1_train, t2_train],
                y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=([t1_test, t2_test], y_test),
                verbose=1
            )

            loss, accuracy = self.tf_model.evaluate([t1_test, t2_test], y_test, verbose=0)
            return float(loss), float(accuracy)
        else:
            # Fallback to Scikit-Learn MLPClassifier when TensorFlow wheel is unavailable (e.g. Python 3.14)
            print("TensorFlow not installed. Using Scikit-Learn Neural Network (MLPClassifier)...")
            
            # Construct one-hot difference vectors for MLP input
            N = len(labels)
            X_mlp = np.zeros((N, self.num_champions), dtype=np.float32)
            for i in range(N):
                for id_val in team1_ids[i]:
                    if 0 <= id_val < self.num_champions:
                        X_mlp[i, id_val] += 1.0
                for id_val in team2_ids[i]:
                    if 0 <= id_val < self.num_champions:
                        X_mlp[i, id_val] -= 1.0

            x_train, x_test, y_train, y_test = train_test_split(
                X_mlp, labels, test_size=test_size, random_state=random_state
            )

            self.mlp_model = MLPClassifier(
                hidden_layer_sizes=self.hidden_units,
                max_iter=epochs * 10,
                random_state=random_state
            )
            self.mlp_model.fit(x_train, y_train)
            acc = float(self.mlp_model.score(x_test, y_test))
            return 0.0, acc


class XGBoostPredictorModel:
    """Gradient Boosted Decision Tree classifier for champion composition difference vectors."""

    def __init__(self, n_estimators: int = 200, max_depth: int = 4, learning_rate: float = 0.05):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.model = None

    def train(
        self,
        sparse_diffs: np.ndarray,
        labels: np.ndarray,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[float, float]:
        """Trains XGBoost classifier and evaluates test accuracy."""
        if not HAS_XGBOOST:
            print("XGBoost package is not installed. Skipping XGBoost model training.")
            return 0.0, 0.0

        x_train, x_test, y_train, y_test = train_test_split(
            sparse_diffs, labels, test_size=test_size, random_state=random_state
        )

        self.model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            eval_metric="logloss",
            random_state=random_state
        )

        self.model.fit(x_train, y_train)
        accuracy = float(self.model.score(x_test, y_test))
        return 0.0, accuracy

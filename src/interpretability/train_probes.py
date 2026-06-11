"""
train_probes.py
----------------
Trains linear probes on the per-layer activations to measure how much each
chess concept is linearly encoded at each layer of the 270M transformer.

For each of the 8 chess concepts × 17 layers = 136 probes:
  - Binary concepts  (is_in_check, has_castling_rights, is_endgame):
      Uses LogisticRegression. Metric = accuracy + AUC-ROC.
  - Continuous concepts (material_balance, num_legal_moves, etc.):
      Uses Ridge regression. Metric = R² + Spearman correlation.

Output
------
Saves results to:
  src/interpretability/data/probing_results.npz

with keys:
  concept_names    : list of 8 concept names
  layer_indices    : [0, 1, ..., 16]
  accuracy         : np.ndarray shape (8, 17) -- accuracy or R² per concept per layer
  spearman         : np.ndarray shape (8, 17) -- AUC or Spearman corr.
  baseline         : np.ndarray shape (8,)    -- random baseline per concept

Usage
-----
  cd /Users/shanesarosh/searchless_chess/src
  conda activate searchless
  python -m interpretability.train_probes
"""

import os
import sys

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Ensure src is on sys.path.
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from interpretability.chess_concepts import BINARY_CONCEPTS, CONCEPT_NAMES  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
ACTIVATIONS_PATH = os.path.join(DATA_DIR, 'activations.npz')
DATASET_PATH = os.path.join(DATA_DIR, 'probing_dataset.npz')
OUTPUT_PATH = os.path.join(DATA_DIR, 'probing_results.npz')

NUM_LAYERS_TOTAL = 17  # embedding (0) + 16 transformer blocks
TEST_SIZE = 0.2
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Probing functions
# ---------------------------------------------------------------------------

def _train_binary_probe(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> tuple[float, float]:
    """Train a logistic regression probe and return (accuracy, AUC-ROC)."""
    probe = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED, C=1.0)
    probe.fit(X_train, y_train)
    accuracy = probe.score(X_test, y_test)
    try:
        y_proba = probe.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
    except ValueError:
        auc = float('nan')  # Only one class in test set
    return accuracy, auc


def _train_regression_probe(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> tuple[float, float]:
    """Train a Ridge regression probe and return (R², Spearman correlation)."""
    probe = Ridge(alpha=1.0)
    probe.fit(X_train, y_train)
    r2 = probe.score(X_test, y_test)
    y_pred = probe.predict(X_test)
    spearman_corr, _ = stats.spearmanr(y_test, y_pred)
    return r2, spearman_corr


def _compute_baseline(y: np.ndarray, is_binary: bool) -> float:
    """Compute a simple random/majority baseline for a concept.
    
    Binary: majority class accuracy.
    Continuous: R² of predicting the mean (always 0.0 by definition).
    """
    if is_binary:
        majority_class_count = max(np.sum(y == 0), np.sum(y == 1))
        return majority_class_count / len(y)
    else:
        return 0.0  # R² baseline is always 0 (predicting the mean)


# ---------------------------------------------------------------------------
# Main probing loop
# ---------------------------------------------------------------------------

def run_probing(
    activations: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Train all 136 probes and return a results dict.
    
    Returns:
        Dict with keys:
          'accuracy'  : shape (8, 17) -- primary metric per concept per layer
          'secondary' : shape (8, 17) -- AUC (binary) or Spearman (continuous)
          'baseline'  : shape (8,)    -- baseline per concept
    """
    n_concepts = len(CONCEPT_NAMES)
    accuracy_matrix = np.zeros((n_concepts, NUM_LAYERS_TOTAL), dtype=np.float32)
    secondary_matrix = np.zeros((n_concepts, NUM_LAYERS_TOTAL), dtype=np.float32)
    baseline_array = np.zeros(n_concepts, dtype=np.float32)

    for c_idx, concept in enumerate(CONCEPT_NAMES):
        y = labels[concept]
        is_binary = concept in BINARY_CONCEPTS

        # Compute baseline once per concept.
        baseline_array[c_idx] = _compute_baseline(y, is_binary)

        print(f"\n[{c_idx+1}/{n_concepts}] Concept: '{concept}' "
              f"({'binary' if is_binary else 'regression'}) | "
              f"Baseline: {baseline_array[c_idx]:.3f}")

        for layer_idx in tqdm(range(NUM_LAYERS_TOTAL), desc=f"  Layers", leave=False):
            X = activations[f'layer_{layer_idx}']

            # Train/test split FIRST, then standardize.
            # Fitting the scaler on all data before splitting is data leakage.
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
            )

            # Fit scaler on train only, transform both.
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            if is_binary:
                acc, secondary = _train_binary_probe(X_train, X_test, y_train, y_test)
            else:
                acc, secondary = _train_regression_probe(X_train, X_test, y_train, y_test)

            accuracy_matrix[c_idx, layer_idx] = acc
            secondary_matrix[c_idx, layer_idx] = secondary

        # Print the layer-by-layer accuracy for this concept.
        accs = accuracy_matrix[c_idx]
        best_layer = int(np.argmax(accs))
        print(f"  Best layer: {best_layer} | Best {'Acc' if is_binary else 'R²'}: {accs[best_layer]:.3f}")

    return {
        'accuracy': accuracy_matrix,
        'secondary': secondary_matrix,
        'baseline': baseline_array,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # ---- Step 1: Load activations ----
    if not os.path.exists(ACTIVATIONS_PATH):
        raise FileNotFoundError(
            f"Activations not found at {ACTIVATIONS_PATH}.\n"
            "Run activation_extractor.py first:\n"
            "  python -m interpretability.activation_extractor"
        )
    print(f"Loading activations from {ACTIVATIONS_PATH}...")
    act_data = np.load(ACTIVATIONS_PATH)
    activations = {key: act_data[key] for key in act_data.files}
    print(f"  Loaded layers: {sorted(activations.keys())}")
    n_positions = activations['layer_0'].shape[0]
    print(f"  Positions: {n_positions} | Embedding dim: {activations['layer_0'].shape[1]}")

    # ---- Step 2: Load labels ----
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Probing dataset not found at {DATASET_PATH}.")
    print(f"Loading labels from {DATASET_PATH}...")
    dataset = np.load(DATASET_PATH, allow_pickle=True)
    labels = {name: dataset[f'label_{name}'] for name in CONCEPT_NAMES}

    # ---- Step 3: Train probes ----
    print(f"\nTraining {len(CONCEPT_NAMES)} concepts × {NUM_LAYERS_TOTAL} layers "
          f"= {len(CONCEPT_NAMES) * NUM_LAYERS_TOTAL} probes total...")
    results = run_probing(activations, labels)

    # ---- Step 4: Save ----
    np.savez_compressed(
        OUTPUT_PATH,
        accuracy=results['accuracy'],
        secondary=results['secondary'],
        baseline=results['baseline'],
        concept_names=np.array(CONCEPT_NAMES, dtype=object),
        layer_indices=np.arange(NUM_LAYERS_TOTAL),
    )
    print(f"\nProbing results saved to: {OUTPUT_PATH}")

    # ---- Step 5: Pretty summary table ----
    print("\n" + "="*70)
    print("SUMMARY: Best probe performance per concept")
    print("="*70)
    print(f"  {'Concept':<24} {'Metric':<10} {'Baseline':>10} {'Best':>8} {'Best Layer':>11}")
    print(f"  {'-'*65}")
    for c_idx, concept in enumerate(CONCEPT_NAMES):
        if concept in BINARY_CONCEPTS:
            best_val = results['secondary'][c_idx].max()
            best_layer = results['secondary'][c_idx].argmax()
            baseline = 0.5
            metric_str = 'AUC-ROC'
        else:
            best_val = results['accuracy'][c_idx].max()
            best_layer = results['accuracy'][c_idx].argmax()
            baseline = results['baseline'][c_idx]
            metric_str = 'R²'
            
        print(f"  {concept:<24} {metric_str:<10} "
              f"{baseline:>10.3f} {best_val:>8.3f} {best_layer:>11}")


if __name__ == '__main__':
    main()

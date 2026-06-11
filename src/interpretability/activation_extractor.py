"""
activation_extractor.py
------------------------
Loads the 270M Searchless Chess transformer and extracts per-layer hidden
activations for every position in the probing dataset.

For each position (FEN):
  1. Tokenize the FEN (77 tokens).
  2. Append a dummy action token + dummy return token to match the model's
     expected input shape for action_value inference (79 tokens total).
  3. Run through transformer_decoder_with_activations().
  4. Mean-pool across the token dimension: (1, 79, 1024) -> (1024,).
  5. Store the 1024-dim vector for each of the 17 layers (embedding + 16 blocks).

Output
------
Saves a compressed numpy archive to:
  src/interpretability/data/activations.npz

with keys:
  layer_0  : np.ndarray of shape (N, 1024)  -- embedding output
  layer_1  : np.ndarray of shape (N, 1024)  -- after transformer block 1
  ...
  layer_16 : np.ndarray of shape (N, 1024)  -- after transformer block 16

Usage
-----
  cd /Users/shanesarosh/searchless_chess/src
  conda activate searchless
  python -m interpretability.activation_extractor
"""

import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
from jax import random as jrandom
from tqdm import tqdm

# Resolve import paths.
# _SRC_DIR  = .../searchless_chess/src/
# _PROJECT_PARENT = .../  (the directory containing searchless_chess/)
# We need _PROJECT_PARENT on sys.path for `from searchless_chess.src import ...` to work.
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROJECT_ROOT = os.path.dirname(_SRC_DIR)     # .../searchless_chess/
_PROJECT_PARENT = os.path.dirname(_PROJECT_ROOT)  # .../
if _PROJECT_PARENT not in sys.path:
    sys.path.insert(0, _PROJECT_PARENT)

from searchless_chess.src import tokenizer as tokenizer_lib   # noqa: E402
from searchless_chess.src import training_utils               # noqa: E402
from searchless_chess.src import transformer as transformer_lib  # noqa: E402
from searchless_chess.src import utils                        # noqa: E402

# ---------------------------------------------------------------------------
# Configuration — matches the 270M model spec in engines/constants.py
# ---------------------------------------------------------------------------
MODEL_NAME = '270M'
NUM_LAYERS = 16
EMBEDDING_DIM = 1024
NUM_HEADS = 8
NUM_RETURN_BUCKETS = 128
CHECKPOINT_STEP = 6_400_000
BATCH_SIZE = 32
FEN_TOKEN_LENGTH = 77  # Only pool FEN tokens, not dummy action/return tokens

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
DATASET_PATH = os.path.join(DATA_DIR, 'probing_dataset.npz')
OUTPUT_PATH = os.path.join(DATA_DIR, 'activations.npz')

# Checkpoints are stored relative to the src/ working directory.
CHECKPOINT_DIR = os.path.join(_SRC_DIR, '..', 'checkpoints', MODEL_NAME)


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------

def _build_predictor():
    """Build the 270M activation-extracting predictor and load its weights."""
    predictor_config = transformer_lib.TransformerConfig(
        vocab_size=utils.NUM_ACTIONS,
        output_size=NUM_RETURN_BUCKETS,
        pos_encodings=transformer_lib.PositionalEncodings.LEARNED,
        max_sequence_length=tokenizer_lib.SEQUENCE_LENGTH + 2,  # 77 + action + return
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        embedding_dim=EMBEDDING_DIM,
        apply_post_ln=True,
        apply_qk_layernorm=False,
        use_causal_mask=False,
    )

    # Use the new activation-aware predictor builder.
    predictor = transformer_lib.build_activation_extractor(predictor_config)

    # Load the pre-trained checkpoint weights.
    print(f"Loading {MODEL_NAME} checkpoint from: {CHECKPOINT_DIR}")
    dummy_targets = np.ones((1, 1), dtype=np.uint32)
    initial_params = predictor.initial_params(
        rng=jrandom.PRNGKey(1),
        targets=dummy_targets,
    )
    params = training_utils.load_parameters(
        checkpoint_dir=CHECKPOINT_DIR,
        params=initial_params,
        step=CHECKPOINT_STEP,
    )
    print("Checkpoint loaded successfully.")
    return predictor, params


# ---------------------------------------------------------------------------
# Tokenization helpers
# ---------------------------------------------------------------------------

def _tokenize_fen_for_action_value(fen: str) -> np.ndarray:
    """Tokenize a FEN into the 79-token sequence the action_value model expects.
    
    Format: [77 FEN tokens] + [1 dummy action token] + [1 dummy return token]
    The dummy tokens are zeros; we only care about the 77 FEN token activations.
    """
    fen_tokens = tokenizer_lib.tokenize(fen).astype(np.int32)      # (77,)
    dummy_action = np.zeros(1, dtype=np.int32)                     # (1,)
    dummy_return = np.zeros(1, dtype=np.int32)                     # (1,)
    return np.concatenate([fen_tokens, dummy_action, dummy_return]) # (79,)


# ---------------------------------------------------------------------------
# Activation extraction
# ---------------------------------------------------------------------------

def extract_activations(
    predictor,
    params,
    fens: list[str],
    batch_size: int = BATCH_SIZE,
) -> dict[str, np.ndarray]:
    """Extract mean-pooled activations from all 17 layers for each FEN.

    Processes FENs in batches for efficiency. Only mean-pools over the first
    77 tokens (the FEN representation), excluding the 2 dummy tokens (action
    + return) that are appended solely to match the model's expected input shape.
    
    Args:
        predictor: The activation-extracting predictor.
        params: Pre-trained model weights.
        fens: List of FEN strings.
        batch_size: Number of FENs to process per forward pass.
        
    Returns:
        Dict mapping 'layer_i' -> np.ndarray of shape (N, EMBEDDING_DIM).
    """
    # JIT-compile the predict function for speed.
    jitted_predict = jax.jit(predictor.predict)

    n = len(fens)
    num_layers_total = NUM_LAYERS + 1  # embedding + 16 transformer blocks

    # Pre-allocate output arrays: one (N, D) array per layer.
    layer_activations = {
        f'layer_{i}': np.zeros((n, EMBEDDING_DIM), dtype=np.float32)
        for i in range(num_layers_total)
    }

    # Tokenize all FENs upfront.
    print(f"Tokenizing {n} positions...")
    all_tokens = np.stack(
        [_tokenize_fen_for_action_value(fen) for fen in fens]
    )  # (N, 79)

    # Process in batches.
    num_batches = (n + batch_size - 1) // batch_size
    print(f"Extracting activations: {n} positions, batch_size={batch_size}, "
          f"{num_batches} batches, {num_layers_total} layers...")

    for batch_idx in tqdm(range(num_batches), desc="Batches"):
        start = batch_idx * batch_size
        end = min(start + batch_size, n)
        batch_tokens = all_tokens[start:end]  # (B, 79)

        # Pad the last batch to full batch_size if needed (JAX JIT expects
        # a fixed shape after the first trace).
        actual_batch_len = end - start
        if actual_batch_len < batch_size:
            padding = np.zeros(
                (batch_size - actual_batch_len, batch_tokens.shape[1]),
                dtype=batch_tokens.dtype,
            )
            batch_tokens = np.concatenate([batch_tokens, padding], axis=0)

        # Forward pass — returns (log_probs, [layer_0_h, ..., layer_16_h])
        _log_probs, activations_per_layer = jitted_predict(
            params=params,
            targets=batch_tokens,
            rng=None,
        )

        # Mean-pool only the FEN tokens (first 77), excluding dummy tokens.
        for layer_i, h in enumerate(activations_per_layer):
            # h shape: (B, T, D). Pool only [:, :77, :] (FEN tokens).
            fen_activations = h[:actual_batch_len, :FEN_TOKEN_LENGTH, :]  # (B', 77, D)
            mean_pooled = jnp.mean(fen_activations, axis=1)  # (B', D)
            layer_activations[f'layer_{layer_i}'][start:end] = np.array(mean_pooled)

    return layer_activations


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # ---- Step 1: Load probing dataset ----
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Probing dataset not found at {DATASET_PATH}.\n"
            "Run generate_probing_dataset.py first:\n"
            "  python -m interpretability.generate_probing_dataset"
        )
    print(f"Loading probing dataset from {DATASET_PATH}...")
    data = np.load(DATASET_PATH, allow_pickle=True)
    fens = data['fens'].tolist()
    print(f"Loaded {len(fens)} positions.")

    # ---- Step 2: Build model ----
    predictor, params = _build_predictor()

    # ---- Step 3: Extract activations ----
    layer_activations = extract_activations(predictor, params, fens)

    # ---- Step 4: Save ----
    os.makedirs(DATA_DIR, exist_ok=True)
    np.savez_compressed(OUTPUT_PATH, **layer_activations)
    print(f"\nActivations saved to: {OUTPUT_PATH}")
    size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
    print(f"File size: {size_mb:.1f} MB")

    # Quick shape check.
    print("\nActivation shapes (sample):")
    for key in ['layer_0', 'layer_8', 'layer_16']:
        print(f"  {key}: {layer_activations[key].shape}")


if __name__ == '__main__':
    main()

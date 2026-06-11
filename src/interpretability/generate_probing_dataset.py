"""
generate_probing_dataset.py
----------------------------
Generates a diverse dataset of chess positions for the probing study.

Strategy: Play N random games from the starting position, sampling board
positions at every move with a fixed probability. This naturally produces
positions spanning openings, middlegames, and endgames.

Output
------
Saves a compressed numpy archive to:
  src/interpretability/data/probing_dataset.npz

with keys:
  fens    : list of N FEN strings (stored as object array)
  labels  : dict[concept_name -> np.ndarray of shape (N,)]
  
Usage
-----
  cd /Users/shanesarosh/searchless_chess/src
  conda activate searchless
  python -m interpretability.generate_probing_dataset
"""

import os
import random
import sys

import chess
import numpy as np
from tqdm import tqdm

# Ensure the src directory is on sys.path so imports resolve.
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from interpretability.chess_concepts import CONCEPT_NAMES, compute_labels_array  # noqa: E402


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
NUM_GAMES = 400          # Number of random games to play
MAX_MOVES_PER_GAME = 200 # Maximum half-moves per game (200 allows endgames)
SAMPLE_PROB = 0.25       # Probability of saving a position at each half-move
TARGET_POSITIONS = 5_000 # We stop once we have this many positions
NUM_ENDGAME_POSITIONS = 500  # Extra synthetic endgame positions to guarantee variance
RANDOM_SEED = 42

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'probing_dataset.npz')


# ---------------------------------------------------------------------------
# Position generation
# ---------------------------------------------------------------------------

def _play_random_game(
    max_moves: int,
    sample_prob: float,
    rng: random.Random,
) -> list[str]:
    """Play one random game and return a list of sampled FEN strings."""
    board = chess.Board()
    fens = []
    for _ in range(max_moves):
        if board.is_game_over():
            break
        legal_moves = list(board.legal_moves)
        move = rng.choice(legal_moves)
        board.push(move)
        if rng.random() < sample_prob:
            fens.append(board.fen())
    return fens


def _generate_endgame_positions(
    count: int,
    rng: random.Random,
) -> list[str]:
    """Generate synthetic endgame positions by randomly removing pieces.

    Creates positions with few pieces (endgame-like) to ensure the
    is_endgame concept has positive examples in the probing dataset.
    """
    endgame_fens: list[str] = []
    attempts = 0
    while len(endgame_fens) < count and attempts < count * 10:
        attempts += 1
        board = chess.Board()
        # Remove random pieces until we have 4-10 non-king pieces.
        non_king_squares = [
            sq for sq in chess.SQUARES
            if board.piece_at(sq) is not None
            and board.piece_at(sq).piece_type != chess.KING
        ]
        rng.shuffle(non_king_squares)
        # Decide how many pieces to keep (3-10 non-king pieces).
        pieces_to_keep = rng.randint(3, 10)
        pieces_to_remove = len(non_king_squares) - pieces_to_keep
        for sq in non_king_squares[:pieces_to_remove]:
            board.remove_piece_at(sq)
        # Make some random moves to reach a legal, diverse position.
        for _ in range(rng.randint(0, 10)):
            legal = list(board.legal_moves)
            if not legal or board.is_game_over():
                break
            board.push(rng.choice(legal))
        # Only keep if it's a valid, non-game-over position.
        if not board.is_game_over() and board.is_valid():
            endgame_fens.append(board.fen())
    return endgame_fens


def generate_positions(
    num_games: int = NUM_GAMES,
    max_moves: int = MAX_MOVES_PER_GAME,
    sample_prob: float = SAMPLE_PROB,
    target: int = TARGET_POSITIONS,
    seed: int = RANDOM_SEED,
) -> list[str]:
    """Generate a list of diverse FEN strings via random self-play + synthetic endgames.
    
    Returns:
        List of FEN strings (up to `target` positions).
    """
    rng = random.Random(seed)
    all_fens: list[str] = []

    # 1) Random self-play positions (openings + middlegames).
    print(f"Playing {num_games} random games to generate positions...")
    for game_idx in tqdm(range(num_games), desc="Games"):
        game_fens = _play_random_game(max_moves, sample_prob, rng)
        all_fens.extend(game_fens)
        if len(all_fens) >= target:
            break

    # 2) Synthetic endgame positions to ensure is_endgame has variance.
    print(f"Generating {NUM_ENDGAME_POSITIONS} synthetic endgame positions...")
    endgame_fens = _generate_endgame_positions(NUM_ENDGAME_POSITIONS, rng)
    all_fens.extend(endgame_fens)
    print(f"  Added {len(endgame_fens)} endgame positions.")

    # Deduplicate (some positions may repeat across games).
    unique_fens = list(dict.fromkeys(all_fens))  # preserves order
    print(f"Generated {len(unique_fens)} unique positions (from {len(all_fens)} total).")
    return unique_fens[:target + NUM_ENDGAME_POSITIONS]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---- Step 1: Generate diverse positions ----
    fens = generate_positions()
    print(f"\nUsing {len(fens)} positions for probing dataset.")

    # ---- Step 2: Compute ground-truth labels ----
    print("Computing chess concept labels...")
    labels = compute_labels_array(fens)

    # Quick sanity check: print stats for each concept.
    print("\nLabel statistics:")
    print(f"  {'Concept':<22} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print(f"  {'-'*56}")
    for name in CONCEPT_NAMES:
        arr = labels[name]
        print(f"  {name:<22} {arr.mean():>8.3f} {arr.std():>8.3f} {arr.min():>8.3f} {arr.max():>8.3f}")

    # ---- Step 3: Save to disk ----
    save_dict = {'fens': np.array(fens, dtype=object)}
    for name in CONCEPT_NAMES:
        save_dict[f'label_{name}'] = labels[name]

    np.savez_compressed(OUTPUT_PATH, **save_dict)
    print(f"\nDataset saved to: {OUTPUT_PATH}")
    print(f"File size: {os.path.getsize(OUTPUT_PATH) / 1024:.1f} KB")

    # ---- Sanity check: warn about zero-variance concepts ----
    print("\nSanity checks:")
    for name in CONCEPT_NAMES:
        arr = labels[name]
        if arr.std() == 0:
            print(f"  ⚠ WARNING: '{name}' has zero variance — probe will be useless!")
        elif name in ('is_in_check', 'has_castling_rights', 'is_endgame'):
            pct = arr.mean() * 100
            if pct < 5 or pct > 95:
                print(f"  ⚠ WARNING: '{name}' is very imbalanced ({pct:.1f}% positive)")
            else:
                print(f"  ✓ '{name}': {pct:.1f}% positive — good balance")
        else:
            print(f"  ✓ '{name}': std={arr.std():.3f} — has variance")


if __name__ == '__main__':
    main()

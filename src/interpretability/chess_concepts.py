"""
chess_concepts.py
-----------------
Computes ground-truth labels for 8 chess concepts directly from a FEN string.
All labels are derived purely from python-chess — no ML, no Stockfish needed.

Concepts
--------
  material_balance   (float)  : White piece value minus Black piece value.
                                 Positive = White ahead.
  is_in_check        (float)  : 1.0 if the side to move is in check, else 0.0.
  has_castling_rights(float)  : 1.0 if either side still has castling rights.
  num_legal_moves    (float)  : Number of legal moves available (mobility proxy).
  is_endgame         (float)  : 1.0 if total non-king pieces on board <= 10.
  center_control     (float)  : Count of pieces/pawns occupying the 4 center
                                 squares (d4, d5, e4, e5).
  king_safety        (float)  : Friendly piece count within Chebyshev dist 2
                                 of the side-to-move's king.
  piece_development  (float)  : Fraction of non-pawn, non-king pieces that have
                                 moved off their starting squares (0.0 – 1.0).
"""

import chess
import numpy as np

# ---------------------------------------------------------------------------
# Piece values (standard material counting)
# ---------------------------------------------------------------------------
_PIECE_VALUES: dict[chess.PieceType, int] = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,  # King has no material value
}

# Center squares used for center-control concept.
_CENTER_SQUARES = {chess.D4, chess.D5, chess.E4, chess.E5}

# Starting squares for white and black minor/major pieces (non-pawn, non-king).
# Used to compute piece development.
_WHITE_START_SQUARES = {
    chess.A1, chess.B1, chess.C1, chess.D1,  # Rook, Knight, Bishop, Queen
    chess.F1, chess.G1, chess.H1,             # Bishop, Knight, Rook
}
_BLACK_START_SQUARES = {
    chess.A8, chess.B8, chess.C8, chess.D8,  # Rook, Knight, Bishop, Queen
    chess.F8, chess.G8, chess.H8,             # Bishop, Knight, Rook
}


# ---------------------------------------------------------------------------
# Individual concept functions
# ---------------------------------------------------------------------------

def _material_balance(board: chess.Board) -> float:
    """White total piece value minus Black total piece value."""
    white_val = sum(
        _PIECE_VALUES[pt]
        for pt in chess.PIECE_TYPES
        for _ in board.pieces(pt, chess.WHITE)
    )
    black_val = sum(
        _PIECE_VALUES[pt]
        for pt in chess.PIECE_TYPES
        for _ in board.pieces(pt, chess.BLACK)
    )
    return float(white_val - black_val)


def _is_in_check(board: chess.Board) -> float:
    """1.0 if the side to move is currently in check."""
    return float(board.is_check())


def _has_castling_rights(board: chess.Board) -> float:
    """1.0 if at least one side still has any castling rights remaining."""
    return float(
        board.has_castling_rights(chess.WHITE)
        or board.has_castling_rights(chess.BLACK)
    )


def _num_legal_moves(board: chess.Board) -> float:
    """Count of legal moves for the side to move (mobility proxy)."""
    return float(len(list(board.legal_moves)))


def _is_endgame(board: chess.Board) -> float:
    """1.0 if total non-king pieces on the board <= 10 (endgame threshold)."""
    non_king_count = sum(
        len(board.pieces(pt, color))
        for pt in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]
        for color in [chess.WHITE, chess.BLACK]
    )
    return float(non_king_count <= 10)


def _center_control(board: chess.Board) -> float:
    """Count of pieces occupying the 4 central squares (d4, d5, e4, e5)."""
    return float(sum(1 for sq in _CENTER_SQUARES if board.piece_at(sq) is not None))


def _king_safety(board: chess.Board) -> float:
    """Count of friendly pieces within Chebyshev distance 2 of the side-to-move king.
    
    Higher = king is better protected.
    """
    side = board.turn  # chess.WHITE or chess.BLACK
    king_sq = board.king(side)
    if king_sq is None:
        return 0.0

    king_file = chess.square_file(king_sq)
    king_rank = chess.square_rank(king_sq)

    count = 0
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece is None or piece.color != side or piece.piece_type == chess.KING:
            continue
        sq_file = chess.square_file(sq)
        sq_rank = chess.square_rank(sq)
        # Chebyshev distance = max of file diff and rank diff
        if max(abs(sq_file - king_file), abs(sq_rank - king_rank)) <= 2:
            count += 1

    return float(count)


def _piece_development(board: chess.Board) -> float:
    """Fraction of non-pawn, non-king pieces off their starting squares.
    
    Measures how much a side has 'developed' from the opening position.
    We measure for the side to move only.
    Returns a value in [0.0, 1.0].
    """
    side = board.turn
    start_squares = _WHITE_START_SQUARES if side == chess.WHITE else _BLACK_START_SQUARES
    developed = 0
    total = 0
    for sq in start_squares:
        # Was a piece expected to start here? Check if any non-pawn, non-king piece
        # from our side is still sitting on the starting square.
        total += 1
        piece = board.piece_at(sq)
        if piece is None or piece.color != side:
            # Piece has moved off -> developed!
            developed += 1
    if total == 0:
        return 0.0
    return float(developed) / float(total)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# All concept names, in a fixed order for consistency.
CONCEPT_NAMES = [
    'material_balance',
    'is_in_check',
    'has_castling_rights',
    'num_legal_moves',
    'is_endgame',
    'center_control',
    'king_safety',
    'piece_development',
]

# Which concepts are binary classification (vs regression).
BINARY_CONCEPTS = {'is_in_check', 'has_castling_rights', 'is_endgame'}


def compute_all_concepts(fen: str) -> dict[str, float]:
    """Compute all 8 chess concept labels from a FEN string.
    
    Args:
        fen: A valid FEN string representing a chess position.
        
    Returns:
        A dict mapping concept name -> float label.
    """
    board = chess.Board(fen)
    return {
        'material_balance': _material_balance(board),
        'is_in_check': _is_in_check(board),
        'has_castling_rights': _has_castling_rights(board),
        'num_legal_moves': _num_legal_moves(board),
        'is_endgame': _is_endgame(board),
        'center_control': _center_control(board),
        'king_safety': _king_safety(board),
        'piece_development': _piece_development(board),
    }


def compute_labels_array(fens: list[str]) -> dict[str, np.ndarray]:
    """Compute labels for all concepts across a list of FEN strings.
    
    Args:
        fens: List of FEN strings.
        
    Returns:
        Dict mapping concept_name -> np.ndarray of shape (N,).
    """
    all_labels: dict[str, list[float]] = {name: [] for name in CONCEPT_NAMES}
    for fen in fens:
        concepts = compute_all_concepts(fen)
        for name in CONCEPT_NAMES:
            all_labels[name].append(concepts[name])
    return {name: np.array(vals, dtype=np.float32) for name, vals in all_labels.items()}

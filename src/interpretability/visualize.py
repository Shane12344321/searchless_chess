"""
visualize.py
------------
Generates 4 publication-quality visualizations from the probing results.

Plots produced (saved to src/interpretability/plots/):
  1. layer_accuracy_curves.png  -- Line chart of probe accuracy vs. layer per concept
  2. concept_heatmap.png        -- Heatmap of all concepts × all layers
  3. tsne_layer8.png            -- t-SNE projection of Layer 8 activations
                                   coloured by material balance
  4. per_concept_bars.png       -- Bar charts for each concept across layers

Usage
-----
  cd /Users/shanesarosh/searchless_chess/src
  conda activate searchless
  python -m interpretability.visualize
"""

import os
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# Ensure src is on sys.path.
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from interpretability.chess_concepts import BINARY_CONCEPTS, CONCEPT_NAMES  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots')
RESULTS_PATH = os.path.join(DATA_DIR, 'probing_results.npz')
ACTIVATIONS_PATH = os.path.join(DATA_DIR, 'activations.npz')
DATASET_PATH = os.path.join(DATA_DIR, 'probing_dataset.npz')

NUM_LAYERS_TOTAL = 17

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
PALETTE = sns.color_palette('tab10', n_colors=len(CONCEPT_NAMES))
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'figure.dpi': 150,
})

# Human-readable concept labels for plots.
CONCEPT_LABELS = {
    'material_balance':   'Material Balance',
    'is_in_check':        'Is In Check',
    'has_castling_rights':'Has Castling Rights',
    'num_legal_moves':    'No. Legal Moves',
    'is_endgame':         'Is Endgame',
    'center_control':     'Center Control',
    'king_safety':        'King Safety',
    'piece_development':  'Piece Development',
}

# Y-axis label per concept type.
def _y_label(concept: str) -> str:
    return 'AUC-ROC' if concept in BINARY_CONCEPTS else 'R²'


def _best_metric(accuracy: np.ndarray, secondary: np.ndarray) -> np.ndarray:
    """Build a (n_concepts, n_layers) matrix using the best metric per concept.
    
    Binary concepts use AUC-ROC (from secondary); continuous use R² (from accuracy).
    """
    best = np.copy(accuracy)
    for c_idx, concept in enumerate(CONCEPT_NAMES):
        if concept in BINARY_CONCEPTS:
            best[c_idx] = secondary[c_idx]
    return best


# ---------------------------------------------------------------------------
# Plot 1: Layer accuracy curves
# ---------------------------------------------------------------------------

def plot_layer_accuracy_curves(accuracy: np.ndarray, secondary: np.ndarray,
                               baseline: np.ndarray):
    """Line chart showing probe performance vs. transformer layer per concept.
    
    Uses AUC-ROC for binary concepts and R² for continuous concepts.
    """
    display = _best_metric(accuracy, secondary)
    fig, ax = plt.subplots(figsize=(11, 6))
    layers = list(range(NUM_LAYERS_TOTAL))

    for c_idx, concept in enumerate(CONCEPT_NAMES):
        label = CONCEPT_LABELS[concept]
        metric = 'AUC' if concept in BINARY_CONCEPTS else 'R²'
        ax.plot(layers, display[c_idx], marker='o', markersize=4,
                label=f'{label} ({metric})', color=PALETTE[c_idx], linewidth=2)

    # Shade the embedding region.
    ax.axvspan(-0.5, 0.5, alpha=0.07, color='grey', label='Embedding layer')
    ax.set_xlabel('Transformer Layer (0 = embedding, 1-16 = transformer blocks)')
    ax.set_ylabel('Probe Performance')
    ax.set_title('Chess Concept Emergence Across Transformer Depth\n(270M Searchless Chess Model)',
                 fontweight='bold')
    ax.set_xticks(layers)
    ax.set_xlim(-0.5, 16.5)
    ax.set_ylim(bottom=0)
    ax.legend(loc='lower right', ncol=2, framealpha=0.85)
    ax.grid(axis='y', alpha=0.3)
    ax.grid(axis='x', alpha=0.15)

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, 'layer_accuracy_curves.png')
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 2: Heatmap
# ---------------------------------------------------------------------------

def plot_concept_heatmap(accuracy: np.ndarray, secondary: np.ndarray):
    """Heatmap of probe performance (AUC for binary, R² for continuous)."""
    display = _best_metric(accuracy, secondary)
    fig, ax = plt.subplots(figsize=(14, 5))

    concept_labels = [CONCEPT_LABELS[c] for c in CONCEPT_NAMES]
    layer_labels = [str(i) for i in range(NUM_LAYERS_TOTAL)]

    sns.heatmap(
        display,
        ax=ax,
        xticklabels=layer_labels,
        yticklabels=concept_labels,
        cmap='YlOrRd',
        annot=True,
        fmt='.2f',
        linewidths=0.5,
        linecolor='white',
        cbar_kws={'label': 'Probe Performance (AUC-ROC / R²)'},
        vmin=0.0,
        vmax=1.0,
    )
    ax.set_xlabel('Transformer Layer')
    ax.set_ylabel('Chess Concept')
    ax.set_title('Probing Performance Heatmap: What Does Each Layer Encode?\n'
                 '(270M Searchless Chess Transformer)',
                 fontweight='bold')

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, 'concept_heatmap.png')
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 3: t-SNE of Layer 8 activations
# ---------------------------------------------------------------------------

def plot_tsne(activations: dict[str, np.ndarray], labels: dict[str, np.ndarray]):
    """2D t-SNE projection of Layer 8 activations coloured by material balance."""
    LAYER = 8
    X = activations[f'layer_{LAYER}']
    y = labels['material_balance']

    # Standardize first — activations in deeper layers have very large magnitudes
    # (std > 1000) which cause overflow in PCA's matrix multiplications.
    print("  Standardizing activations...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Reduce to 50 dims with PCA first (standard practice before t-SNE).
    print("  Running PCA (50 dims)...")
    pca = PCA(n_components=50, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    print("  Running t-SNE (this may take ~1 min)...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    X_2d = tsne.fit_transform(X_pca)

    fig, ax = plt.subplots(figsize=(9, 7))
    # Clip material balance to [-10, 10] for better colour scaling.
    y_clipped = np.clip(y, -10, 10)
    scatter = ax.scatter(X_2d[:, 0], X_2d[:, 1],
                         c=y_clipped, cmap='RdYlGn',
                         s=8, alpha=0.7, linewidths=0)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Material Balance (White − Black, clipped to ±10)')
    ax.set_title(f'Layer {LAYER} Activations (t-SNE) — Coloured by Material Balance\n'
                 f'(270M Searchless Chess Transformer)',
                 fontweight='bold')
    ax.set_xlabel('t-SNE Component 1')
    ax.set_ylabel('t-SNE Component 2')
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, f'tsne_layer{LAYER}.png')
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Plot 4: Per-concept bar charts
# ---------------------------------------------------------------------------

def plot_per_concept_bars(accuracy: np.ndarray, secondary: np.ndarray, baseline: np.ndarray):
    """One bar chart per chess concept showing probe performance per layer."""
    n_concepts = len(CONCEPT_NAMES)
    ncols = 4
    nrows = (n_concepts + ncols - 1) // ncols  # = 2

    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 7), sharey=False)
    axes_flat = axes.flatten()
    layers = list(range(NUM_LAYERS_TOTAL))
    layer_colors = plt.cm.viridis(np.linspace(0.2, 0.9, NUM_LAYERS_TOTAL))
    display = _best_metric(accuracy, secondary)

    for c_idx, concept in enumerate(CONCEPT_NAMES):
        ax = axes_flat[c_idx]
        vals = display[c_idx]
        bars = ax.bar(layers, vals, color=layer_colors, width=0.7, zorder=2)
        
        # Baseline horizontal line.
        if concept in BINARY_CONCEPTS:
            concept_baseline = 0.5
        else:
            concept_baseline = baseline[c_idx]
            
        ax.axhline(concept_baseline, color='red', linestyle='--',
                   linewidth=1.5, label=f'Baseline ({concept_baseline:.2f})')
        ax.set_title(CONCEPT_LABELS[concept], fontsize=11, fontweight='bold')
        ax.set_xlabel('Layer', fontsize=9)
        ax.set_ylabel(_y_label(concept), fontsize=9)
        ax.set_xticks(range(0, NUM_LAYERS_TOTAL, 4))
        ax.grid(axis='y', alpha=0.3, zorder=1)
        ax.legend(fontsize=8)
        # Annotate best layer.
        best_layer = int(np.argmax(vals))
        best_val = vals[best_layer]
        ax.annotate(f'Best: L{best_layer}\n({best_val:.2f})',
                    xy=(best_layer, best_val),
                    xytext=(best_layer + 0.5, best_val * 0.85),
                    fontsize=7, color='darkblue',
                    arrowprops=dict(arrowstyle='->', color='darkblue', lw=1))

    # Hide unused subplots if any.
    for i in range(n_concepts, len(axes_flat)):
        axes_flat[i].set_visible(False)

    fig.suptitle('Per-Concept Probe Performance Across Transformer Layers\n'
                 '(270M Searchless Chess Transformer)',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, 'per_concept_bars.png')
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # ---- Load results ----
    if not os.path.exists(RESULTS_PATH):
        raise FileNotFoundError(
            f"Probing results not found at {RESULTS_PATH}.\n"
            "Run train_probes.py first."
        )
    print(f"Loading probing results from {RESULTS_PATH}...")
    results = np.load(RESULTS_PATH, allow_pickle=True)
    accuracy = results['accuracy']     # (8, 17)
    secondary = results['secondary']   # (8, 17) -- AUC for binary, Spearman for continuous
    baseline = results['baseline']     # (8,)

    print(f"Loading activations for t-SNE from {ACTIVATIONS_PATH}...")
    act_data = np.load(ACTIVATIONS_PATH)
    activations = {key: act_data[key] for key in act_data.files}

    print(f"Loading labels for t-SNE from {DATASET_PATH}...")
    dataset = np.load(DATASET_PATH, allow_pickle=True)
    labels = {name: dataset[f'label_{name}'] for name in CONCEPT_NAMES}

    # ---- Generate plots ----
    print("\nGenerating visualizations...")

    print("  [1/4] Layer accuracy curves...")
    plot_layer_accuracy_curves(accuracy, secondary, baseline)

    print("  [2/4] Concept heatmap...")
    plot_concept_heatmap(accuracy, secondary)

    print("  [3/4] t-SNE plot (Layer 8)...")
    plot_tsne(activations, labels)

    print("  [4/4] Per-concept bar charts...")
    plot_per_concept_bars(accuracy, secondary, baseline)

    print(f"\nAll plots saved to: {PLOTS_DIR}/")


if __name__ == '__main__':
    main()

"""
run_full_pipeline.py
---------------------
One-click script that runs the entire mechanistic interpretability pipeline
from scratch:

  Step 1: Generate probing dataset  (chess positions + concept labels)
  Step 2: Extract per-layer activations  (load 270M model)
  Step 3: Train linear probes  (136 probes: 8 concepts × 17 layers)
  Step 4: Generate visualizations  (4 plots saved to interpretability/plots/)

Usage
-----
  cd /Users/shanesarosh/searchless_chess/src
  conda activate searchless
  python -m interpretability.run_full_pipeline

To skip already-completed steps (e.g. if activations are already extracted):
  python -m interpretability.run_full_pipeline --skip-dataset --skip-activations
"""

import argparse
import os
import sys
import time

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


def _header(title: str):
    print('\n' + '='*60)
    print(f'  {title}')
    print('='*60)


def main():
    parser = argparse.ArgumentParser(description='Run full interpretability pipeline.')
    parser.add_argument('--skip-dataset', action='store_true',
                        help='Skip dataset generation if already done.')
    parser.add_argument('--skip-activations', action='store_true',
                        help='Skip activation extraction if already done.')
    parser.add_argument('--skip-probes', action='store_true',
                        help='Skip probe training if already done.')
    args = parser.parse_args()

    total_start = time.time()

    # ---- Step 1: Generate dataset ----
    dataset_path = os.path.join(DATA_DIR, 'probing_dataset.npz')
    if args.skip_dataset and os.path.exists(dataset_path):
        print(f'\n[Step 1] SKIPPED — dataset already exists at {dataset_path}')
    else:
        _header('Step 1/4: Generating Probing Dataset')
        from interpretability.generate_probing_dataset import main as gen_main
        t0 = time.time()
        gen_main()
        print(f'  Done in {time.time() - t0:.1f}s')

    # ---- Step 2: Extract activations ----
    activations_path = os.path.join(DATA_DIR, 'activations.npz')
    if args.skip_activations and os.path.exists(activations_path):
        print(f'\n[Step 2] SKIPPED — activations already exist at {activations_path}')
    else:
        _header('Step 2/4: Extracting Per-Layer Activations (loads 270M model)')
        from interpretability.activation_extractor import main as extract_main
        t0 = time.time()
        extract_main()
        print(f'  Done in {time.time() - t0:.1f}s')

    # ---- Step 3: Train probes ----
    results_path = os.path.join(DATA_DIR, 'probing_results.npz')
    if args.skip_probes and os.path.exists(results_path):
        print(f'\n[Step 3] SKIPPED — probing results already exist at {results_path}')
    else:
        _header('Step 3/4: Training Linear Probes (136 probes)')
        from interpretability.train_probes import main as probe_main
        t0 = time.time()
        probe_main()
        print(f'  Done in {time.time() - t0:.1f}s')

    # ---- Step 4: Visualize ----
    _header('Step 4/4: Generating Visualizations')
    from interpretability.visualize import main as viz_main
    t0 = time.time()
    viz_main()
    print(f'  Done in {time.time() - t0:.1f}s')

    # ---- Done ----
    elapsed = time.time() - total_start
    print(f'\n{"="*60}')
    print(f'  Pipeline complete! Total time: {elapsed/60:.1f} minutes')
    print(f'  Plots: src/interpretability/plots/')
    print(f'  Data:  src/interpretability/data/')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()

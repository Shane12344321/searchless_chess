# Chessbridge & Searchless Chess ♟️

This is the backend for the Chessbridge Chrome extension. It runs a local 270M Transformer model to instantly compute chess moves using a WebSocket pipeline.

## 🔬 Research: Mechanistic Interpretability

> **Read the full study:** [`src/interpretability/README.md`](src/interpretability/README.md)

I conducted a mechanistic interpretability study on the 270M-parameter Searchless Chess transformer to understand how it internally represents chess knowledge without search. Using **136 linear probing classifiers** across 8 chess concepts and 17 layers, the study reveals:

- **Concept Emergence:** The model develops a hierarchical understanding, with simple concepts (material balance) emerging at Layer 1 and complex spatial reasoning (king safety) emerging at Layer 6.
- **Concept Degradation:** Human-interpretable concepts peak in layers 1-3 and decline in deeper layers, suggesting the model transforms conceptual features into task-specific action-value representations.

Read the [full findings and view the layer-wise visualizations here](src/interpretability/README.md).

---

## How to Run This from Scratch

### 1. Start the Server
The backend engine runs inside a Jupyter Notebook and hosts a WebSocket server on port 8000.
1. Open your Mac Terminal.
2. Activate your conda environment and start Jupyter Notebook:
   ```bash
   conda activate searchless
   cd /Users/shanesarosh/searchless_chess/src
   jupyter notebook
   ```
3. A new tab will open in Google Chrome showing the Jupyter directory.
4. Click on **`searchless_chess.ipynb`**.
5. Click **Kernel -> Restart & Run All** in the top menu bar.
6. Scroll to the bottom of the notebook—it takes a few seconds to load the 270M neural network weights into memory. You should eventually see `🚀 Stream Engine running on ws://127.0.0.1:8000`. Leave this tab open in the background.

### 2. Load the Extension & Play
1. Load your `chessbridge` folder in `chrome://extensions/`.
2. Go to [Chess.com/play/computer](https://www.chess.com/play/computer).
3. The extension will automatically connect to your Jupyter WebSocket server, speculate on your opponent's turn, and visually highlight the best moves directly on the board when it's your turn.

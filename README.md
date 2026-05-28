# Chessbridge & Searchless Chess ♟️

This is the backend for the Chessbridge Chrome extension. It runs a local 270M Transformer model to instantly compute chess moves using a WebSocket pipeline.

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

import json

patch_code = """
# === 1. LOAD THE OPENING BOOK INTO MEMORY ===
import chess.polyglot
book_dict = {}
try:
    with chess.polyglot.open_reader("opening_book.bin") as reader:
        for entry in reader:
            if entry.key not in book_dict or entry.weight > book_dict[entry.key]['weight']:
                book_dict[entry.key] = {'move': entry.move.uci(), 'weight': entry.weight}
    print(f"📚 gm2001 Opening Book loaded into RAM! ({len(book_dict)} unique positions)")
except FileNotFoundError:
    print("⚠️ No gm2001.bin found. Ensure the file is in the same directory.")
"""

patch_lookup = """
                    # === 3. OPENING BOOK CHECK (Zero Compute) ===
                    if len(book_dict) > 0:
                        z_hash = chess.polyglot.zobrist_hash(board)
                        if z_hash in book_dict:
                            best_move = book_dict[z_hash]['move']
                            await websocket.send(json.dumps({"best_move": best_move}))
                            t_book = time.perf_counter()
                            print(f"📚 BOOK HIT! Streaming GM theory instantly.")
                            print(f"⏱️ LATENCY: Book Hit in {(t_book - t0) * 1000:.2f} ms\\n")
                            continue
                        else:
                            print(f"⚠️ BOOK MISS! FEN received: {board.fen()}")
                            pass # Out of book, fall through to live calculation
"""

f=open('/Users/shanesarosh/searchless_chess/src/searchless_chess.ipynb')
d=json.load(f)
f.close()

for cell in d['cells']:
    if cell['cell_type'] == 'code':
        src = cell['source']
        src_str = "".join(src)
        if "# === 1. LOAD THE OPENING BOOK ===" in src_str:
            cell['source'] = [line + "\n" for line in patch_code.strip().split("\n")]
        elif "# === 3. OPENING BOOK CHECK" in src_str:
            # Replace the book check block
            new_src = []
            skip = False
            for line in src:
                if "# === 3. OPENING BOOK CHECK" in line:
                    skip = True
                    new_src.extend([l + "\n" for l in patch_lookup.strip("\n").split("\n")])
                elif skip and "t1 = time.perf_counter()" in line:
                    skip = False
                    new_src.append("                    \n                    # Stop the book check timer\n")
                    new_src.append(line)
                elif not skip:
                    new_src.append(line)
            cell['source'] = new_src

with open('/Users/shanesarosh/searchless_chess/src/searchless_chess.ipynb', 'w') as f:
    json.dump(d, f, indent=1)


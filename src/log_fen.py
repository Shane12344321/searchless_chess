import json

def patch():
    f=open('/Users/shanesarosh/searchless_chess/src/searchless_chess.ipynb')
    d=json.load(f)
    f.close()
    for cell in d['cells']:
        if cell['cell_type'] == 'code':
            src = cell['source']
            for i, line in enumerate(src):
                if 'except IndexError:' in line:
                    if 'with open' not in src[i+2]:
                        src.insert(i+2, '                            with open("debug_log.txt", "a") as df: df.write(f"MISS FEN: {incoming_fen} -> {board.fen()}\\n")\n')
                    break
    with open('/Users/shanesarosh/searchless_chess/src/searchless_chess.ipynb', 'w') as f:
        json.dump(d, f, indent=1)

patch()

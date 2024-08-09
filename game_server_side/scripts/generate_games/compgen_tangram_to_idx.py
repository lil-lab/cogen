import os
import json

ROOT_PATH = "/Users/mustafaomergul/Desktop/Cornell/Research/tangrams-compgen/tangrams-ref-dev-omer"
EMPTY_DATA_PATH =  "/Users/mustafaomergul/Desktop/Cornell/Research/kilogram/dataset/tangrams-svg"
JSON_PATH = ROOT_PATH + "/refgame/public/games"

if __name__ == "__main__":
    paths = os.listdir(EMPTY_DATA_PATH)
    paths = [path for path in paths if ".DS_Store" not in path]

    # Construct the mapping
    t2idx = {}
    idx2t = {}
    for i, path in enumerate(paths):
        t2idx[path] = i
        idx2t[i] = path

    # Save the mappings
    with open(os.path.join(JSON_PATH, 'tangram_to_idx.json'), 'w') as f:
        json.dump(t2idx, f)
    with open(os.path.join(JSON_PATH, 'idx_to_tangram.json'), 'w') as f:
        json.dump(idx2t, f)

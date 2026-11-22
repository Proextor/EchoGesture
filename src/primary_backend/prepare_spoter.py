"""Convert the public SPOTER WLASL100 landmark CSVs into this repo's landmark format.

        (WLASL100_{train,val,test}_25fps.csv - 2D body + hand landmarks for the WLASL100 split)
Labels 0..99 are the first 100 glosses of WLASL_v0.3.json, in order.

    python prepare_spoter.py --csv-dir <folder with the 3 CSVs> --wlasl-json WLASL_v0.3.json --out-dir ../data
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import N_POINTS
from .features import normalize_sequence

POSE = ["nose", "leftShoulder", "rightShoulder", "leftElbow", "rightElbow", "leftWrist", "rightWrist"]
HAND = ["wrist"] + [f"{f}{j}" for f, js in [("thumb", ["CMC", "MP", "IP", "Tip"])] +
                    [(f, ["MCP", "PIP", "DIP", "Tip"]) for f in ["index", "middle", "ring", "little"]] for j in js]
NAMES = POSE + [f"{h}_left" for h in HAND] + [f"{h}_right" for h in HAND]
assert len(NAMES) == N_POINTS


def convert(df):
    seqs = []
    for _, r in df.iterrows():
        asp = float(r["video_size_width"]) / float(r["video_size_height"])
        x = np.array([json.loads(r[n + "_X"]) for n in NAMES], np.float32).T      # [frames, 49]
        y = np.array([json.loads(r[n + "_Y"]) for n in NAMES], np.float32).T
        mask = (x != 0) | (y != 0)                 # missing landmarks are stored as (0, 0)
        for sl in (slice(7, 28), slice(28, 49)):   # a hand is present only if all of it is
            mask[:, sl] = mask[:, sl].all(1, keepdims=True)
        pts = np.stack([x * asp, 1.0 - y], -1)     # source has y pointing up; MediaPipe has y down
        seqs.append(normalize_sequence(pts, mask))
    L = np.array([len(s) for s in seqs])
    X = np.zeros((len(seqs), L.max(), N_POINTS, 3), np.float32)
    for i, s in enumerate(seqs):
        X[i, :len(s)] = s
    return X, L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-dir", required=True)
    ap.add_argument("--wlasl-json", required=True)
    ap.add_argument("--out-dir", default="../data")
    a = ap.parse_args()
    glosses = [g["gloss"].upper() for g in json.load(open(a.wlasl_json))[:100]]
    for split in ["train", "val", "test"]:
        df = pd.read_csv(Path(a.csv_dir) / f"WLASL100_{split}_25fps.csv")
        X, L = convert(df)
        y = np.array([glosses[i] for i in df["labels"]])
        np.savez_compressed(Path(a.out_dir) / f"{split}.npz", X=X, L=L, y=y)
        print(split, X.shape, "clips,", len(set(y)), "classes")


if __name__ == "__main__":
    main()

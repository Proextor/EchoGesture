"""Turn your own labelled videos into a landmark dataset with MediaPipe.

    python extract_landmarks.py --csv ../data/train.csv --root ../data --out ../data/train.npz
CSV columns: path,label   (label = gloss/word; stored uppercase). Raw video is read once and never copied.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .config import N_POINTS, T
from .features import extract_raw, normalize_sequence


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", default=".", help="folder that CSV paths are relative to")
    ap.add_argument("--frames", type=int, default=T * 2, help="frames sampled per clip")
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.csv)))
    seqs, y = [], []
    for r in tqdm(rows):
        pts, mask = extract_raw(Path(a.root) / r["path"], a.frames)
        if len(pts) == 0:
            print("skip (unreadable):", r["path"])
            continue
        seqs.append(normalize_sequence(pts, mask))
        y.append(r["label"].strip().upper())
    L = np.array([len(s) for s in seqs])
    X = np.zeros((len(seqs), L.max(), N_POINTS, 3), np.float32)
    for i, s in enumerate(seqs):
        X[i, :len(s)] = s
    np.savez_compressed(a.out, X=X, L=L, y=np.array(y))
    print(f"saved {len(seqs)}/{len(rows)} clips -> {a.out}")


if __name__ == "__main__":
    main()

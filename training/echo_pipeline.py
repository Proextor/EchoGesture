
#!/usr/bin/env python3
"""EchoGesture: rebuild WLASL-100 with the OFFICIAL test split, train 5 seeds, evaluate.

Run from the EchoGesture repo root (the folder that has train.py, build_dataset.py, src/).
Does NOT modify main.py, Dockerfiles, train.py or build_dataset.py. It only calls them.

  python echo_pipeline.py build --meta <WLASL_v0.3.json> --videos <videos dir>
  python echo_pipeline.py train --epochs 160
  python echo_pipeline.py eval

Outputs: out/seed_1.onnx ... out/seed_5.onnx (FP32; your teammate quantizes) and out/test_results.json
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path.cwd()
sys.path.insert(0, str(REPO))
NUM_SEEDS = 5


# ----------------------------------------------------------------------------- build
def _work(task):
    video, out, aug_out, seed = task
    try:
        from src.preprocess import extract_keypoints
        from build_dataset import _augment
        out, aug_out = Path(out), Path(aug_out) if aug_out else None
        if out.exists() and (aug_out is None or aug_out.exists()):
            return video, None
        arr = np.asarray(extract_keypoints(video), dtype=np.float32)
        if arr.shape != (64, 162) or not np.isfinite(arr).all():
            raise ValueError(f"bad array {arr.shape}")
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, arr)
        if aug_out is not None:
            np.save(aug_out, _augment(arr, np.random.default_rng(seed)))
        return video, None
    except Exception as exc:  # noqa: BLE001
        return video, f"{type(exc).__name__}: {exc}"


def cmd_build(a):
    meta = json.loads(Path(a.meta).read_text(encoding="utf-8"))
    selected = meta[:100]  # same rule as build_dataset.py: first 100 glosses = WLASL-100
    words = [str(g["gloss"]).strip().upper() for g in selected]
    vocab = sorted(set(words))
    assert len(vocab) == 100, f"expected 100 classes, got {len(vocab)}"

    ref = REPO / "data" / "vocab.json"
    if ref.is_file():
        old = json.loads(ref.read_text())
        assert old == vocab, "vocab differs from the existing data/vocab.json. STOP: class order must match"
    out_root = Path(a.output)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "vocab.json").write_text(json.dumps(vocab, indent=2) + "\n")

    vids = Path(a.videos)
    tasks, found, missing = [], {"train": 0, "val": 0, "test": 0}, {"train": 0, "val": 0, "test": 0}
    for g, word in zip(selected, words):
        for inst in g.get("instances", []):
            split = {"train": "train", "val": "val", "validation": "val", "test": "test"}.get(inst.get("split"))
            vid = str(inst.get("video_id", "")).strip()
            if not split or not vid:
                continue
            path = vids / f"{vid}.mp4"
            if not path.is_file():
                missing[split] += 1
                continue
            found[split] += 1
            base = out_root / split / word / vid
            aug = base.with_name(vid + "_aug.npy") if split == "train" else None
            tasks.append((str(path), str(base.with_suffix(".npy")), str(aug) if aug else None, 2026 + len(tasks)))
    print("videos found  :", found)
    print("videos missing:", missing, "(original bundle had ~748 train / 165 val / 100 test)")
    if not tasks:
        sys.exit("No videos found. Check --videos (files must be <video_id>.mp4)")

    fails = []
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for i, (src, err) in enumerate(pool.map(_work, tasks), 1):
            if err:
                fails.append((src, err))
            if i % 100 == 0 or i == len(tasks):
                print(f"processed {i}/{len(tasks)}", flush=True)
    print(f"done. failures: {len(fails)}")
    for f in fails[:20]:
        print("  FAILED", f)


# ----------------------------------------------------------------------------- train
def cmd_train(a):
    Path("out").mkdir(exist_ok=True)
    for s in range(1, NUM_SEEDS + 1):
        run = Path("runs") / f"seed_{s}"
        target = Path("out") / f"seed_{s}.onnx"
        if (run / "model.onnx").is_file() and not a.force:
            print(f"seed {s}: already trained, skipping")
        else:
            run.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, "train.py", "--data", a.data, "--vocab", str(Path(a.data) / "vocab.json"),
                   "--output", str(run), "--seed", str(s), "--epochs", str(a.epochs),
                   "--workers", str(a.workers)]
            print("RUN", " ".join(cmd), flush=True)
            with open(run / "train.log", "w") as log:
                rc = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT)
            tail = (run / "train.log").read_text().strip().splitlines()[-2:]
            print("\n".join(tail))
            if rc != 0:
                sys.exit(f"seed {s} failed, see {run/'train.log'}")
        shutil.copy(run / "model.onnx", target)
    print("exported:", sorted(str(p) for p in Path("out").glob("seed_*.onnx")))


# ----------------------------------------------------------------------------- eval
def _corrupt(kp, rng):
    """Same corruption as quantize_and_eval.py: drop 10% of landmarks + 0.01 noise."""
    pts = np.asarray(kp, dtype=np.float32).reshape(64, 54, 3).copy()
    pts[:, rng.random(54) < 0.10, :] = 0.0
    seen = np.any(pts != 0.0, axis=2)
    pts += rng.normal(0.0, 0.01, size=pts.shape).astype(np.float32) * seen[:, :, None]
    return pts.reshape(64, 162)


def _load(folder: Path, vocab):
    xs, ys = [], []
    for w, i in vocab.items():
        for p in sorted((folder / w).glob("*.npy")):
            if not p.stem.endswith("_aug"):
                xs.append(np.load(p).astype(np.float32))
                ys.append(i)
    return np.stack(xs), np.array(ys)


def cmd_eval(a):
    import onnxruntime as ort
    from src.preprocess import engineer_model_features

    vocab = {w: i for i, w in enumerate(json.loads((Path(a.data) / "vocab.json").read_text()))}
    models = sorted(Path("out").glob("seed_*.onnx"))
    assert models, "no out/seed_*.onnx found. Run train first"
    sess = [ort.InferenceSession(str(m), providers=["CPUExecutionProvider"]) for m in models]

    def logits(kp):
        feats = np.stack([engineer_model_features(k) for k in kp]).astype(np.float32)
        return [s.run(None, {s.get_inputs()[0].name: feats})[0] for s in sess]

    res = {"models": [m.name for m in models]}
    rng = np.random.default_rng(2026)
    for split in ("val", "test"):
        kp, y = _load(Path(a.data) / split, vocab)
        per = logits(kp)
        ens = np.mean(per, axis=0)
        acc = lambda z: float((z.argmax(1) == y).mean() * 100)  # noqa: E731
        noisy = logits(np.stack([_corrupt(k, rng) for k in kp]))
        ens_noisy = np.mean(noisy, axis=0)
        res[split] = {
            "n_clips": int(len(y)),
            "per_seed_top1": {m.name: round(acc(z), 2) for m, z in zip(models, per)},
            "ensemble_top1": round(acc(ens), 2),
            "ensemble_top1_corrupted": round(acc(ens_noisy), 2),
            "robustness_drop_pt": round(acc(ens) - acc(ens_noisy), 2),
        }
    res["note"] = ("val was used to pick each seed's checkpoint, so val is optimistic. "
                   "Quote the TEST numbers. Run eval once and do not tune against test.")
    Path("out").mkdir(exist_ok=True)
    Path("out/test_results.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--meta", required=True)
    b.add_argument("--videos", required=True)
    b.add_argument("--output", default="data")
    b.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    b.set_defaults(fn=cmd_build)
    t = sub.add_parser("train")
    t.add_argument("--data", default="data")
    t.add_argument("--epochs", type=int, default=160)
    t.add_argument("--workers", type=int, default=2)
    t.add_argument("--force", action="store_true")
    t.set_defaults(fn=cmd_train)
    e = sub.add_parser("eval")
    e.add_argument("--data", default="data")
    e.set_defaults(fn=cmd_eval)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()

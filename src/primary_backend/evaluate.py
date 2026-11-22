"""Measure the scored metrics on a held-out landmark set and write results/results.csv.

    python evaluate.py --test ../data/test.npz --device "<device>" --runtime "ONNX Runtime INT8 CPU"
avg_power needs a real phone (e.g. Android battery stats / Snapdragon Profiler): pass --power-mw.
"""
import argparse
import csv
import json
import os
import time

import numpy as np

from .augment import degrade
from .config import MODEL_PATH, CLASSES_PATH
from .features import load_sequences, resample, to_model_input
from .main import predict_from_features


def accuracy(seqs, labels, classes, harsh=False, model_path=None):
    idx = {c: i for i, c in enumerate(classes)}
    top1 = top5 = n = 0
    lat = []
    for i, lab in enumerate(labels):
        if lab not in idx:
            continue
        s = resample(seqs[i])
        if harsh:
            s = degrade(s, seed=i)
        p, ms = predict_from_features(to_model_input(s), model_path)
        order = np.argsort(-p)
        top1 += order[0] == idx[lab]
        top5 += idx[lab] in order[:5]
        lat.append(ms)
        n += 1
    return 100 * top1 / n, 100 * top5 / n, lat, n


def thermal_drift(x, minutes):
    """% change in median model latency between the first and last 10 % of a sustained run."""
    total = minutes * 60
    t0, samples = time.time(), []
    while time.time() - t0 < total:
        _, ms = predict_from_features(x)
        samples.append((time.time() - t0, ms))
    first = [m for t, m in samples if t < 0.1 * total]
    last = [m for t, m in samples if t > 0.9 * total]
    return 100 * (np.median(last) - np.median(first)) / np.median(first)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", default="../results/results.csv")
    ap.add_argument("--device", default="")
    ap.add_argument("--runtime", default="ONNX Runtime INT8 CPU")
    ap.add_argument("--power-mw", type=float, default=None)
    ap.add_argument("--thermal-minutes", type=float, default=10)
    a = ap.parse_args()

    seqs, labels = load_sequences(a.test)
    classes = json.loads(CLASSES_PATH.read_text())
    t1, t5, lat, n = accuracy(seqs, labels, classes)
    t1d, t5d, _, _ = accuracy(seqs, labels, classes, harsh=True)
    fp32 = MODEL_PATH.with_name("echogesture_fp32.onnx")
    if fp32.exists():
        f1, _, _, _ = accuracy(seqs, labels, classes, model_path=fp32)
        print(f"fp32 top1 {f1:.2f}%  (INT8 costs {f1 - t1:+.2f} pt)")
    size = os.path.getsize(MODEL_PATH) / 1e6
    x0 = to_model_input(resample(seqs[0]))
    drift = thermal_drift(x0, a.thermal_minutes) if a.thermal_minutes > 0 else None
    p95 = float(np.percentile(lat, 95))

    def row(metric, target, val, unit, ok):
        return [metric, target, "" if val is None else f"{val:.2f}", unit, a.device, a.runtime,
                "" if val is None else ("PASS" if ok(val) else "FAIL")]

    rows = [
        row("top1_accuracy", ">=70", t1, "%", lambda v: v >= 70),
        row("latency_p95", "<=50", p95, "ms", lambda v: v <= 50),
        row("model_size", "<=25", size, "MB", lambda v: v <= 25),
        row("accuracy_drop_robustness", "<=5", t1 - t1d, "pt", lambda v: v <= 5),
        row("avg_power", "<=500", a.power_mw, "mW", lambda v: v <= 500),
        row("thermal_drift_10min", "<=20", drift, "%", lambda v: v <= 20),
    ]
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "target", "measured", "unit", "device", "runtime", "pass"])
        w.writerows(rows)
    print(f"{n} clips | top1 {t1:.2f}%  top5 {t5:.2f}% | degraded top1 {t1d:.2f}% top5 {t5d:.2f}% | "
          f"p95 {p95:.2f} ms (model only) | size {size:.2f} MB" + ("" if drift is None else f" | drift {drift:+.1f}%"))
    print("wrote", a.out)


if __name__ == "__main__":
    main()

"""Export a Python reference for the Android instrumentation tests."""

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import main as inference
from src.preprocess import engineer_model_features, extract_keypoints


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--name", required=True, choices=("saved", "replay"))
    args = parser.parse_args()
    if not args.source.is_file():
        parser.error(f"Missing fixture: {args.source.resolve()}")
    os.environ["ECHOGESTURE_TTA"] = "0"
    # The phone uses the repository's equal-weight, logit-average default.
    for key in ("ECHOGESTURE_MODEL", "ECHOGESTURE_MODEL_SEED2", "ECHOGESTURE_MODEL_SEED3",
                "ECHOGESTURE_VOCAB", "ECHOGESTURE_AVG", "ECHOGESTURE_WEIGHTS"):
        if os.environ.get(key):
            parser.error(f"Unset {key} to compare the repository's default ensemble")

    raw = (np.load(args.source, allow_pickle=False) if args.source.suffix.lower() == ".npy"
           else extract_keypoints(args.source))
    engineered = engineer_model_features(raw)[None, ...]
    result = inference.predict_keypoints(raw, args.source.stem)
    sessions = inference._get_sessions()
    logits = np.stack([s.run(None, {s.get_inputs()[0].name: engineered})[0][0] for s in sessions])
    combined = np.zeros(logits.shape[1], dtype=np.float64)
    for scores in logits:
        combined += scores.astype(np.float64) * (1 / len(sessions))
    summary = {
        "source": str(args.source.resolve()),
        "model_names": [f"seed_{i}.onnx" for i in range(1, 6)] if inference.FIVE_SEED_DIR else [path.name for path in (inference.MODEL_PATH, inference.SEED2_MODEL_PATH,
                                               inference.SEED3_MODEL_PATH) if path.is_file()],
        "result": result,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output.parent / f"{args.name}-python.npz",
                        raw=raw, engineered=engineered, logits=logits, combined=combined)
    fixture = {**summary, "raw": raw.ravel().tolist(), "engineered": engineered.ravel().tolist(),
               "logits": logits.tolist(), "combined": combined.tolist()}
    (args.output / f"{args.name}-python.json").write_text(json.dumps(fixture), encoding="utf-8")
    (args.output.parent / f"{args.name}-summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

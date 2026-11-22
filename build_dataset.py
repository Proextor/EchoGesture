#!/usr/bin/env python3
"""Import WLASL metadata and optionally build keypoint arrays.

Expected input layout (as in common WLASL processed mirrors):
  --metadata data/wlasl/WLASL_v0.3.json
  --videos data/wlasl/videos/  (files named <video_id>.mp4)

Use --import-only to write wlasl_manifest.json without extracting landmarks.
The dataset build also writes train/<WORD>/ and val/<WORD>/ keypoints and
an uppercase vocab.json in the manifest's class order.
"""
from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING

from src.wlasl import SPLIT_DIRECTORIES, import_wlasl, write_json

if TYPE_CHECKING:
    import numpy as np


def _augment(array: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply clip-consistent in-plane rotation/scale and landmark dropout."""
    import numpy as np

    result = np.array(array, dtype=np.float32, copy=True).reshape(64, 54, 3)
    angle = np.deg2rad(rng.uniform(-12.0, 12.0))
    scale = rng.uniform(0.9, 1.1)
    c, s = np.cos(angle), np.sin(angle)
    xy = result[:, :, :2]
    rotation = np.array([[c, -s], [s, c]], dtype=np.float32)
    result[:, :, :2] = (xy @ rotation.T) * scale

    # Drop the same anatomical landmark across the full clip, preserving all
    # three coordinates and yielding coherent partial occlusion.
    dropped = rng.random(54) < 0.10
    result[:, dropped, :] = 0.0
    return result.reshape(64, 162).astype(np.float32, copy=False)


def _process(task: tuple[str, str, str, bool, int]) -> tuple[str, str | None]:
    video_path, output_path, augmented_path, make_aug, seed = task
    try:
        import numpy as np

        from src.preprocess import extract_keypoints

        array = np.asarray(extract_keypoints(video_path), dtype=np.float32)
        if array.shape != (64, 162):
            raise ValueError(f"extract_keypoints returned {array.shape}, expected (64, 162)")
        if not np.isfinite(array).all():
            raise ValueError("extract_keypoints returned non-finite values")
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, array)
        if make_aug:
            np.save(augmented_path, _augment(array, np.random.default_rng(seed)))
        return video_path, None
    except Exception as exc:  # report bad source clips and continue the batch
        return video_path, f"{type(exc).__name__}: {exc}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/wlasl"), help="local WLASL dataset directory")
    parser.add_argument("--metadata", type=Path, help="override the annotation JSON path")
    parser.add_argument("--videos", type=Path, help="override the directory of numbered MP4 clips")
    parser.add_argument("--output", type=Path, default=Path("data"))
    parser.add_argument("--classes", type=int, default=100, help="top-K WLASL glosses (default: 100)")
    parser.add_argument("--vocab", type=Path, help="use the evaluator's glosses and class order instead of --classes")
    parser.add_argument("--import-only", action="store_true", help="write the manifest and report without decoding video")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    if args.classes < 1 or args.workers < 1:
        parser.error("--classes and --workers must be positive")
    metadata = (args.metadata or args.root / "WLASL_v0.3.json").resolve()
    videos = args.videos or args.root / "videos"
    manifest_path = args.output / "wlasl_manifest.json"
    vocab_path = args.output / "vocab.json"
    evaluator_vocab = args.vocab.resolve() if args.vocab else None
    if manifest_path.resolve() in (metadata, evaluator_vocab):
        parser.error("The manifest output path must be distinct from the metadata and evaluator vocabulary")
    if not args.import_only and vocab_path.resolve() == metadata:
        parser.error("The generated vocabulary path must be distinct from the metadata")

    try:
        manifest = import_wlasl(metadata, videos, args.classes, evaluator_vocab)
        write_json(manifest_path, manifest)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))

    report = manifest["report"]
    total_classes = len(manifest["labels"])
    print(f"Imported {len(manifest['clips'])} annotated clips across {total_classes} classes.")
    print(f"Missing videos: {len(report['missing_video_ids'])}. See {manifest_path} for their IDs.")
    for split, counts in report["splits"].items():
        print(
            f"{split}: {counts['available_clips']}/{counts['annotated_clips']} clips available, "
            f"{counts['available_classes']}/{total_classes} classes available "
            f"({counts['annotated_classes']} annotated)"
        )
    if args.import_only:
        return 0

    entries = [clip for clip in manifest["clips"] if clip["available"] and clip["split"] != "test"]
    if not entries:
        raise SystemExit(f"No matching train/val MP4s found under {videos}; expected <video_id>.mp4 files")

    words = [gloss.strip().upper() for gloss in manifest["labels"]]
    if vocab_path.resolve() != evaluator_vocab:
        write_json(vocab_path, words)

    jobs = []
    for index, entry in enumerate(entries):
        split = SPLIT_DIRECTORIES[entry["split"]]
        base = args.output / split / words[entry["label"]] / entry["video_id"]
        jobs.append((entry["path"], str(base.with_suffix(".npy")),
                     str(base.with_name(base.name + "_aug").with_suffix(".npy")),
                     split == "train", args.seed + index))

    failures = []
    completed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_process, job) for job in jobs]
        for future in as_completed(futures):
            source, error = future.result()
            if error:
                failures.append((source, error))
                print(f"FAILED {source}: {error}", flush=True)
            completed += 1
            if completed % 25 == 0 or completed == len(jobs):
                print(f"Processed {completed}/{len(jobs)}", flush=True)

    print(f"Saved vocab for {len(words)} classes; processed {len(jobs) - len(failures)}/{len(jobs)} clips.")
    if failures:
        print(f"{len(failures)} clips failed; see messages above.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

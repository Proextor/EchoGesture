"""Extract versioned MediaPipe features from the prepared WLASL video directory."""

import argparse
from collections import Counter
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import random
from tempfile import TemporaryDirectory
from zipfile import BadZipFile

os.environ.setdefault("GLOG_minloglevel", "3")

import cv2
import mediapipe as mp
import numpy as np

from src.landmark_features import FEATURE_SHAPE, FEATURE_VERSION, FRAME_COUNT, POSE_IDS, make_features
from src.wlasl import write_json


# These two Kaggle files match their original, untrimmed single-sign videos.
# Match the contents as well as the ID so a pretrimmed replacement is not cropped.
UNCUT_SOURCES = {
    "02269": "c9862f2377139ee5f9b44032e4375c5f6e8d79412a72bca81c79300a770096d0",
    "55771": "bb7d0c04db3a956c362c12ee5504b3f3702782a8093f605ec562259fdb837781",
}


def decode(path, clip):
    """Read a clip sequentially and sample against its decoded length."""
    digest = UNCUT_SOURCES.get(clip["video_id"])
    original = digest is not None and hashlib.sha256(path.read_bytes()).hexdigest() == digest
    capture = cv2.VideoCapture(str(path))
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        fps = fps if np.isfinite(fps) and fps > 0 else 25.0
        start = round((clip["frame_start"] - 1) * fps / clip["fps"]) if original else 0
        frames = []
        decoded = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            decoded += 1
            if decoded <= start:
                continue
            height, width = frame.shape[:2]
            scale = 512 / max(height, width)
            if scale < 1:
                frame = cv2.resize(frame, (round(width * scale), round(height * scale)),
                                   interpolation=cv2.INTER_AREA)
            frames.append(frame)
    finally:
        capture.release()
    if not frames:
        raise ValueError("No frames decoded")
    # Container frame counts can differ from decoded counts. Sampling uses
    # the frames that were actually returned by the decoder.
    indices = np.linspace(0, len(frames) - 1, FRAME_COUNT).round().astype(np.int32)
    return [frames[i] for i in indices], indices + start, fps


def detect(frames, indices, fps, hand_options, pose_options):
    hands = np.zeros((FRAME_COUNT, 2, 21, 3), dtype=np.float32)
    pose = np.zeros((FRAME_COUNT, len(POSE_IDS), 3), dtype=np.float32)
    hand_present = np.zeros((FRAME_COUNT, 2), dtype=bool)
    pose_present = np.zeros((FRAME_COUNT, len(POSE_IDS)), dtype=bool)
    vision = mp.tasks.vision
    with ExitStack() as stack:
        hand_model = stack.enter_context(vision.HandLandmarker.create_from_options(hand_options))
        pose_model = stack.enter_context(vision.PoseLandmarker.create_from_options(pose_options))
        timestamp = -1
        for row, (frame, index) in enumerate(zip(frames, indices)):
            if row and index == indices[row - 1]:
                hands[row], pose[row] = hands[row - 1], pose[row - 1]
                hand_present[row], pose_present[row] = hand_present[row - 1], pose_present[row - 1]
                continue
            timestamp = max(timestamp + 1, round(int(index) * 1000 / fps))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            hand_result = hand_model.detect_for_video(image, timestamp)
            pose_result = pose_model.detect_for_video(image, timestamp)
            scores = [-1.0, -1.0]
            for categories, landmarks in zip(hand_result.handedness, hand_result.hand_landmarks):
                category = categories[0]
                side = {"Left": 0, "Right": 1}.get(category.category_name)
                xyz = np.array([(p.x, p.y, p.z) for p in landmarks], dtype=np.float32)
                if side is not None and category.score > scores[side] and np.isfinite(xyz).all():
                    hands[row, side] = xyz
                    hand_present[row, side] = True
                    scores[side] = category.score
            if pose_result.pose_landmarks:
                for column, landmark_id in enumerate(POSE_IDS):
                    p = pose_result.pose_landmarks[0][landmark_id]
                    xyz = (p.x, p.y, p.z)
                    if p.visibility >= 0.5 and np.isfinite(xyz).all():
                        pose[row, column] = xyz
                        pose_present[row, column] = True
    return hands, pose, hand_present, pose_present


def valid_features(features):
    return features.shape == FEATURE_SHAPE and features.dtype == np.float32 and np.isfinite(features).all()


def already_extracted(path, metadata):
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as saved:
            return saved["metadata"].item() == metadata and valid_features(saved["features"])
    except (OSError, ValueError, KeyError, EOFError, BadZipFile):
        return False


def save_clip(path, **fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as directory:
        temp = Path(directory) / path.name
        np.savez_compressed(temp, **fields)
        temp.replace(path)


def log_event(output, status, clip, detail):
    record = {key: clip[key] for key in ("video_id", "gloss", "label", "split")}
    record["detail"] = detail
    with (output / f"{status}.jsonl").open("a", encoding="utf-8") as log:
        log.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--videos", type=Path, required=True, help="directory containing the prepared MP4 clips")
    parser.add_argument("--models", type=Path, default=Path("data/landmark-models"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", type=int, help="extract this many available training clips from distinct glosses")
    parser.add_argument("--progress-every", type=int, default=50)
    args = parser.parse_args()
    if args.progress_every < 1 or (args.smoke is not None and args.smoke < 1):
        parser.error("Progress interval and smoke size must be positive")
    if not args.videos.is_dir():
        parser.error(f"Video directory does not exist: {args.videos}")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    labels = manifest["labels"]
    clips = manifest["clips"]
    present_ids = {path.stem for path in args.videos.glob("*.mp4") if path.is_file()}
    training = {}
    for clip in clips:
        if clip["split"] == "train" and clip["video_id"] in present_ids:
            training.setdefault(clip["gloss"], []).append(clip)
    print("Zero available training videos:", ", ".join(word for word in labels if word not in training), flush=True)
    print(f"{sum(clip['video_id'] in present_ids for clip in clips)}/{len(clips)} annotated clips available",
          flush=True)
    if args.smoke:
        if len(training) < args.smoke:
            parser.error("Too few distinct available training glosses for smoke test")
        rng = random.Random(2026)
        exceptions = [clip for clip in clips if clip["video_id"] in UNCUT_SOURCES
                      and clip["split"] == "train" and clip["video_id"] in present_ids][:args.smoke]
        chosen = {clip["gloss"] for clip in exceptions}
        others = [word for word in training if word not in chosen]
        clips = exceptions + [rng.choice(training[word]) for word in rng.sample(others, args.smoke - len(exceptions))]

    hand_path = args.models / "hand_landmarker.task"
    pose_path = args.models / "pose_landmarker_lite.task"
    if not hand_path.is_file() or not pose_path.is_file():
        parser.error(f"Hand Landmarker and Pose Lite task files are required under {args.models}")
    vision = mp.tasks.vision
    hand_options = vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(hand_path)),
        running_mode=vision.RunningMode.VIDEO, num_hands=2,
    )
    pose_options = vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(pose_path)),
        running_mode=vision.RunningMode.VIDEO,
    )
    cv2.setNumThreads(1)
    output = args.output / FEATURE_VERSION
    write_json(output / "labels.json", labels)
    counts = Counter()
    for current, clip in enumerate(clips, 1):
        video = args.videos / f"{clip['video_id']}.mp4"
        artifact = output / clip["split"] / f"{clip['video_id']}.npz"
        if not video.is_file():
            status, detail = "missing_files", str(video)
        else:
            stat = video.stat()
            metadata = json.dumps({
                **{key: clip[key] for key in ("video_id", "gloss", "label", "split", "frame_start", "frame_end", "fps")},
                "feature_version": FEATURE_VERSION, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            }, sort_keys=True)
            if already_extracted(artifact, metadata):
                status, detail = "resumed", ""
            else:
                try:
                    frames, indices, fps = decode(video, clip)
                except (OSError, ValueError, cv2.error) as error:
                    status, detail = "decode_errors", str(error)
                else:
                    counts["decoded"] += 1
                    height, width = frames[0].shape[:2]
                    aspect = height / width
                    try:
                        raw = detect(frames, indices, fps, hand_options, pose_options)
                        if not raw[2].any():
                            status, detail = "no_hands", "No hand detected in the 48 sampled frames"
                        else:
                            counts["usable_hands"] += 1
                            features = make_features(*raw, aspect)
                            if not valid_features(features):
                                raise ValueError("Expected finite float32 features shaped [48,352]")
                            save_clip(artifact, features=features, hands=raw[0], pose=raw[1],
                                      hand_present=raw[2], pose_present=raw[3], frame_indices=indices,
                                      fps=fps, aspect=aspect, metadata=metadata)
                            status, detail = "written", ""
                    except (RuntimeError, ValueError, cv2.error) as error:
                        status, detail = "feature_errors", str(error)
        counts[status] += 1
        if detail:
            log_event(output, status, clip, detail)
        if args.smoke:
            print(f"{clip['video_id']} {clip['gloss']}: {status}"
                  + (f" {FEATURE_SHAPE}" if status in ("written", "resumed") else ""), flush=True)
        if current % args.progress_every == 0 or current == len(clips):
            print(f"{current}/{len(clips)} clips checked | {dict(counts)}", flush=True)
    if counts["feature_errors"]:
        return 1
    return int(bool(args.smoke and (counts["decoded"] + counts["resumed"] != args.smoke
                                   or not (counts["written"] + counts["resumed"]))))


if __name__ == "__main__":
    raise SystemExit(main())

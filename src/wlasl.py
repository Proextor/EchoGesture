"""Index WLASL clips and measure local coverage of the official splits."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory


# Older metadata mirrors use "validation". Keep the annotation spelling in
# the manifest and use the same validation directory as the dataset builder.
SPLIT_DIRECTORIES = {"train": "train", "val": "val", "validation": "val", "test": "test"}


def _read_vocabulary(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8-sig")
    if path.suffix.lower() == ".json":
        vocabulary = json.loads(text)
        if isinstance(vocabulary, dict):
            indices = list(vocabulary.values())
            expected = list(range(len(indices)))
            if any(type(index) is not int for index in indices) or sorted(indices) != expected:
                raise ValueError("Vocabulary indices must be consecutive integers starting at zero")
            vocabulary = sorted(vocabulary, key=vocabulary.__getitem__)
    else:
        vocabulary = [line for line in text.splitlines() if line.strip()]

    if not isinstance(vocabulary, list) or not vocabulary:
        raise ValueError("Vocabulary must be a nonempty list of glosses or a gloss-to-index mapping")
    if any(not isinstance(gloss, str) or not gloss.strip() for gloss in vocabulary):
        raise ValueError("Vocabulary entries must be nonempty gloss strings")
    return [gloss.strip().casefold() for gloss in vocabulary]


def _select_groups(data: list[dict], limit: int, vocabulary_path: Path | None) -> tuple[list[dict], list[str]]:
    if vocabulary_path is None:
        selected = data[:limit]
        # Keep the established class IDs used by the project's trained models.
        return selected, sorted((group["gloss"] for group in selected), key=lambda gloss: gloss.strip().upper())

    by_gloss = {group["gloss"].strip().casefold(): group for group in data}
    vocabulary = _read_vocabulary(vocabulary_path)
    if len(set(vocabulary)) != len(vocabulary):
        raise ValueError("Vocabulary contains duplicate glosses")
    unknown = [gloss for gloss in vocabulary if gloss not in by_gloss]
    if unknown:
        raise ValueError(f"Glosses absent from WLASL metadata: {', '.join(unknown)}")
    selected = [by_gloss[gloss] for gloss in vocabulary]
    return selected, [group["gloss"] for group in selected]


def import_wlasl(metadata: Path, videos: Path, limit: int, vocabulary: Path | None = None) -> dict:
    """Return all selected clips, including missing files, with coverage counts."""
    videos = videos.resolve()
    if not videos.is_dir():
        raise ValueError(f"Videos directory does not exist: {videos}")
    data = json.loads(metadata.read_text(encoding="utf-8-sig"))
    if not isinstance(data, list):
        raise ValueError("Expected WLASL metadata JSON to contain a list of gloss entries")
    groups, labels = _select_groups(data, limit, vocabulary)
    class_ids = {gloss: index for index, gloss in enumerate(labels)}
    if len({gloss.strip().upper() for gloss in labels}) != len(labels):
        raise ValueError("WLASL metadata contains duplicate selected glosses")

    counts = {
        split: {gloss: {"annotated": 0, "available": 0} for gloss in labels}
        for split in ("train", "val", "test")
    }
    clips = []
    missing_video_ids = []
    seen_ids = set()
    for group in groups:
        gloss = group["gloss"]
        for instance in group.get("instances", []):
            split = instance["split"]
            if split not in SPLIT_DIRECTORIES:
                raise ValueError(f"Unknown WLASL split {split!r} for {gloss!r}")
            video_id = str(instance["video_id"]).strip()
            if not video_id.isascii() or not video_id.isdecimal():
                raise ValueError(f"Invalid WLASL video ID {video_id!r} for {gloss!r}")
            if video_id in seen_ids:
                raise ValueError(f"Duplicate WLASL video ID: {video_id}")
            seen_ids.add(video_id)

            path = videos / f"{video_id}.mp4"
            available = path.is_file()
            clips.append({
                "video_id": video_id,
                "path": str(path),
                "gloss": gloss,
                "label": class_ids[gloss],
                "split": split,
                "frame_start": instance.get("frame_start"),
                "frame_end": instance.get("frame_end"),
                "fps": instance.get("fps"),
                "available": available,
            })
            coverage = counts[SPLIT_DIRECTORIES[split]][gloss]
            coverage["annotated"] += 1
            coverage["available"] += int(available)
            if not available:
                missing_video_ids.append(video_id)

    splits = {}
    for split, by_gloss in counts.items():
        annotated = sum(count["annotated"] for count in by_gloss.values())
        available = sum(count["available"] for count in by_gloss.values())
        splits[split] = {
            "annotated_clips": annotated,
            "available_clips": available,
            "missing_clips": annotated - available,
            "annotated_classes": sum(count["annotated"] > 0 for count in by_gloss.values()),
            "available_classes": sum(count["available"] > 0 for count in by_gloss.values()),
            "per_class": by_gloss,
        }

    return {
        "metadata": str(metadata.resolve()),
        "vocabulary": str(vocabulary.resolve()) if vocabulary else None,
        "class_limit": limit if vocabulary is None else None,
        "labels": labels,
        "clips": clips,
        "report": {"missing_video_ids": missing_video_ids, "splits": splits},
    }


def write_json(path: Path, data: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as directory:
        temporary_path = Path(directory) / path.name
        with temporary_path.open("w", encoding="utf-8") as output:
            json.dump(data, output, ensure_ascii=False, indent=2)
            output.write("\n")
        temporary_path.replace(path)

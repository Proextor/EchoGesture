"""Versioned landmark feature layout shared by offline and future live extraction."""

import numpy as np

FEATURE_VERSION = "hands-pose-lite-48x352-v1"
FRAME_COUNT = 48
FEATURE_SHAPE = (FRAME_COUNT, 352)
# Nose, eyes, ears, shoulders, elbows, wrists, in MediaPipe pose order.
POSE_IDS = (0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16)


def _unit(vectors):
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms > 1e-6)


def make_features(hands, pose, hand_present, pose_present, aspect):
    """Convert 48 frames of image-space XYZ and detection masks to float32."""
    # Convert y to units of image width so geometry respects the aspect ratio.
    axes = np.array([1, aspect, 1], dtype=np.float32)
    hands = hands * axes
    pose = pose * axes
    points = np.concatenate((hands.reshape(FRAME_COUNT, 42, 3), pose), axis=1)
    visible = np.concatenate((np.repeat(hand_present, 21, axis=1), pose_present), axis=1)

    shoulders = pose[:, 5:7, :2]
    centers = shoulders.mean(axis=1)
    spans = np.linalg.norm(shoulders[:, 0] - shoulders[:, 1], axis=1)
    anchored = pose_present[:, 5:7].all(axis=1) & (spans > 1e-6)
    if anchored.any():
        centers[~anchored] = np.median(centers[anchored], axis=0)
        spans[~anchored] = np.median(spans[anchored])
    else:
        centers[:] = (0.5, aspect / 2)
        spans[:] = 1
    xy = (points[..., :2] - centers[:, None]) / spans[:, None, None]
    xy[~visible] = 0

    relative = hands - hands[:, :, :1]
    palm = np.linalg.norm(relative[:, :, [5, 9, 13, 17]], axis=-1).mean(axis=-1)
    shape = relative / np.maximum(palm[..., None, None], 1e-6)
    forward = _unit(relative[:, :, 9])
    normal = _unit(np.cross(forward, relative[:, :, 5] - relative[:, :, 17]))
    orientation = np.concatenate((forward, normal), axis=-1)
    shape[~hand_present] = 0
    orientation[~hand_present] = 0

    velocity = np.zeros_like(xy)
    velocity[1:] = xy[1:] - xy[:-1]
    velocity[1:][~(visible[1:] & visible[:-1])] = 0
    return np.concatenate((
        xy.reshape(FRAME_COUNT, 106),
        shape.reshape(FRAME_COUNT, 126),
        orientation.reshape(FRAME_COUNT, 12),
        hand_present.astype(np.float32),
        velocity.reshape(FRAME_COUNT, 106),
    ), axis=1)

"""Video -> pose-invariant landmark sequence. Raw pixels are never stored or returned.

Landmark sequence format used everywhere: array [N, 49, 3] = (x, y, present) per point,
points ordered: 7 pose joints, 21 left-hand, 21 right-hand (the signer's own left/right).
"""
import numpy as np

from .config import (T, POSE_IDX, N_POSE, N_HAND, N_POINTS, L_HAND, R_HAND, POSE_VIS_MIN,
                    VEL_GAIN, MODEL_IN_DIM)

_holistic = None


def _get_holistic():
    global _holistic
    if _holistic is None:
        import mediapipe as mp
        _holistic = mp.solutions.holistic.Holistic(
            static_image_mode=False, model_complexity=1,
            min_detection_confidence=0.4, min_tracking_confidence=0.4)
    return _holistic


def extract_raw(video_path, n_frames=T):
    """Run MediaPipe on up to n_frames uniformly sampled frames (None = every frame).

    Returns pts [N, 49, 2] in aspect-corrected image units (x * W/H, y; y points down) and mask [N, 49].
    """
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    keep = None
    if n_frames and total > n_frames:
        keep = set(np.linspace(0, total - 1, n_frames).round().astype(int).tolist())
    hol = _get_holistic()
    P, M, i = [], [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if keep is None or i in keep:
            h, w = frame.shape[:2]
            asp = w / h
            res = hol.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            pts = np.zeros((N_POINTS, 2), np.float32)
            mask = np.zeros(N_POINTS, bool)
            if res.pose_landmarks:
                for j, k in enumerate(POSE_IDX):
                    lm = res.pose_landmarks.landmark[k]
                    if lm.visibility >= POSE_VIS_MIN:
                        pts[j] = (lm.x * asp, lm.y)
                        mask[j] = True
            for sl, hand in ((L_HAND, res.left_hand_landmarks), (R_HAND, res.right_hand_landmarks)):
                if hand:
                    pts[sl] = [(lm.x * asp, lm.y) for lm in hand.landmark]
                    mask[sl] = True
            P.append(pts)
            M.append(mask)
        i += 1
    cap.release()
    if not P:
        return np.zeros((0, N_POINTS, 2), np.float32), np.zeros((0, N_POINTS), bool)
    return np.stack(P), np.stack(M)


def normalize_sequence(pts, mask):
    """Body-frame normalisation: origin = mid-shoulder (per frame), scale = shoulder width and
    rotation = shoulder-line angle (clip medians). Returns [N, 49, 3] with missing points zeroed."""
    n = len(pts)
    out = np.zeros((n, N_POINTS, 3), np.float32)
    if n == 0:
        return out
    ok = mask[:, 1] & mask[:, 2]
    d = pts[:, 1] - pts[:, 2]                      # right shoulder -> left shoulder (+x in an unmirrored video)
    width = np.linalg.norm(d, axis=1)
    ok &= width > 1e-4
    if ok.any():
        scale = float(np.median(width[ok]))
        ang = float(np.arctan2(np.median(d[ok, 1]), np.median(d[ok, 0])))
        origin = (pts[:, 1] + pts[:, 2]) / 2
        idx = np.where(ok)[0]
        nearest = idx[np.abs(np.arange(n)[:, None] - idx[None]).argmin(1)]   # fill gaps from nearest valid frame
        origin = origin[nearest]
    else:                                           # no shoulders at all: centre on whatever is visible
        vis = pts[mask]
        origin = np.tile(vis.mean(0) if len(vis) else np.zeros(2), (n, 1))
        scale, ang = 0.35, 0.0
    c, s = np.cos(-ang), np.sin(-ang)
    R = np.array([[c, -s], [s, c]], np.float32)
    xy = ((pts - origin[:, None]) / scale) @ R.T
    out[..., :2] = xy * mask[..., None]
    out[..., 2] = mask
    return out


def resample(seq, t=T):
    """[N, 49, 3] -> [t, 49, 3] by uniform frame picking."""
    n = len(seq)
    if n == 0:
        return np.zeros((t, N_POINTS, 3), np.float32)
    return seq[np.linspace(0, n - 1, t).round().astype(int)]


def to_model_input(seq):
    """[T, 49, 3] -> [T, MODEL_IN_DIM] float32."""
    t = len(seq)
    xy, m = seq[..., :2], seq[..., 2]
    parts = [(xy * m[..., None]).reshape(t, -1)]
    for sl in (L_HAND, R_HAND):                    # hand shape: wrist-centred, scaled by palm length
        h, hm = xy[:, sl], m[:, sl]
        size = np.maximum(np.linalg.norm(h[:, 9] - h[:, 0], axis=-1), 1e-3)
        parts.append((((h - h[:, :1]) / size[:, None, None]) * hm[..., None]).reshape(t, -1))
    parts.append(np.concatenate([m[:, :N_POSE], m[:, N_POSE:N_POSE + 1], m[:, N_POSE + N_HAND:N_POSE + N_HAND + 1]], 1))
    vel = np.zeros_like(xy)
    vel[1:] = (xy[1:] - xy[:-1]) * (m[1:] * m[:-1])[..., None]
    parts.append(vel.reshape(t, -1) * VEL_GAIN)
    x = np.concatenate(parts, axis=1).astype(np.float32)
    assert x.shape[1] == MODEL_IN_DIM
    return np.clip(x, -8, 8)


def video_to_model_input(video_path):
    pts, mask = extract_raw(video_path)
    return to_model_input(resample(normalize_sequence(pts, mask)))


def load_sequences(path):
    """Read a landmark npz once -> (list of [n_i, 49, 3] sequences, labels array)."""
    d = np.load(path)
    X, L, y = d["X"], d["L"], d["y"]          # each access decompresses the whole array: do it once
    return [X[i, :L[i]].copy() for i in range(len(X))], y

"""Landmark-space augmentation and the fixed degradation used for the robustness metric.

All functions take/return landmark sequences [N, 49, 3] = (x, y, present).
"""
import numpy as np

from .config import N_POSE, N_HAND, N_POINTS, L_HAND, R_HAND, T


def rotate(s, deg):
    a = np.deg2rad(deg)
    c, si = np.cos(a), np.sin(a)
    out = s.copy()
    out[..., :2] = s[..., :2] @ np.array([[c, -si], [si, c]], np.float32).T
    return out


def scale(s, fx, fy=None):
    out = s.copy()
    out[..., 0] *= fx
    out[..., 1] *= fx if fy is None else fy
    return out


def noise(s, sigma):
    out = s.copy()
    out[..., :2] += np.random.randn(*s[..., :2].shape).astype(np.float32) * sigma * s[..., 2:]
    return out


def drop_points(s, sl, start=0, length=None):
    """Simulate occlusion: mark points in slice `sl` missing over a time window."""
    out = s.copy()
    end = len(s) if length is None else min(len(s), start + length)
    out[start:end, sl] = 0
    return out


def drop_frames(s, p):
    """Simulate dropped frames: repeat the previous frame with probability p."""
    out = s.copy()
    for i in range(1, len(s)):
        if np.random.rand() < p:
            out[i] = out[i - 1]
    return out


def mirror(s):
    """Left/right-handed signers: flip x, swap hands and paired pose joints."""
    out = s.copy()
    out[..., 0] *= -1
    for a, b in [(1, 2), (3, 4), (5, 6)]:
        out[:, [a, b]] = out[:, [b, a]]
    left, right = out[:, L_HAND].copy(), out[:, R_HAND].copy()
    out[:, L_HAND], out[:, R_HAND] = right, left
    return out


def random_resample(s, t=T):
    """Temporal augmentation: random start/end trim, speed change and frame jitter."""
    n = len(s)
    a = np.random.uniform(0, 0.12) * (n - 1)
    b = (1 - np.random.uniform(0, 0.12)) * (n - 1)
    idx = np.linspace(a, b, t) + np.random.uniform(-0.5, 0.5, t) * (b - a) / t
    return s[np.clip(idx.round().astype(int), 0, n - 1)]


def random_augment(s):
    r = np.random.rand
    if r() < 0.6:
        s = rotate(s, np.random.uniform(-15, 15))
    if r() < 0.6:
        s = scale(s, np.random.uniform(0.85, 1.15), np.random.uniform(0.85, 1.15))
    if r() < 0.6:
        s = noise(s, np.random.uniform(0.005, 0.03))
    n = len(s)
    if r() < 0.25:
        s = drop_points(s, L_HAND if r() < 0.5 else R_HAND, np.random.randint(0, n), np.random.randint(2, n // 3))
    if r() < 0.25:
        j = np.random.choice([3, 4, 5, 6])
        s = drop_points(s, slice(j, j + 1), np.random.randint(0, n), np.random.randint(2, n // 2))
    if r() < 0.25:
        s = drop_frames(s, np.random.uniform(0.05, 0.2))
    if r() < 0.3:
        s = mirror(s)
    return s


def degrade(s, seed=0):
    """Fixed harsh-condition transform for the robustness metric:
    tilted + angled signer, landmark jitter, a partial hand occlusion and dropped frames."""
    state = np.random.get_state()
    np.random.seed(seed)
    n = len(s)
    s = rotate(s, 12)
    s = scale(s, 0.9, 1.0)
    s = noise(s, 0.015)
    s = drop_points(s, L_HAND if np.random.rand() < 0.5 else R_HAND, n // 3, n // 5)
    s = drop_frames(s, 0.15)
    np.random.set_state(state)
    return s

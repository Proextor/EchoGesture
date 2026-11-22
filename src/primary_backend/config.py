"""Shared constants. Keep feature layout identical for training and inference."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "echogesture_int8.onnx"
CLASSES_PATH = MODEL_DIR / "classes.json"
VOCAB_PATH = ROOT / "vocab.txt"

T = 32                                   # frames per clip after temporal resampling
POSE_IDX = [0, 11, 12, 13, 14, 15, 16]   # nose, L/R shoulder, L/R elbow, L/R wrist (MediaPipe pose ids)
N_POSE = len(POSE_IDX)
N_HAND = 21
N_POINTS = N_POSE + 2 * N_HAND           # 49 landmarks, 2D
L_HAND = slice(N_POSE, N_POSE + N_HAND)
R_HAND = slice(N_POSE + N_HAND, N_POINTS)
POSE_VIS_MIN = 0.5                       # pose joints below this visibility count as missing
VEL_GAIN = 4.0

# model input per frame: body-frame xy (98) + hand-local xy (84) + presence flags (9) + velocity (98)
MODEL_IN_DIM = N_POINTS * 2 + 2 * N_HAND * 2 + (N_POSE + 2) + N_POINTS * 2

SEED = 42


def load_vocab():
    """Optional fixed vocabulary (uppercase, one word per line). Empty list = use labels found in data."""
    if not VOCAB_PATH.exists():
        return []
    return [w.strip().upper() for w in VOCAB_PATH.read_text().splitlines() if w.strip()]

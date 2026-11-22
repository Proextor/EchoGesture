"""Export trained EchoNet to ONNX and quantize to INT8 (static, QDQ).

    python convert.py --calib ../data/train.npz                       # single model
    python convert.py --calib ../data/train.npz --ckpt "echonet_s*.pt"  # ensemble of seeds
Writes models/echogesture_fp32.onnx and models/echogesture_int8.onnx
"""
import argparse
import json
import os

import numpy as np
import torch
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static

from .config import MODEL_DIR, MODEL_PATH, CLASSES_PATH
from .features import load_sequences, resample, to_model_input
from .model import EchoNet, Ensemble, example_input


class Reader(CalibrationDataReader):
    def __init__(self, seqs, n=300):
        idx = np.random.RandomState(0).permutation(len(seqs))[:n]
        self.it = iter([{"input": to_model_input(resample(seqs[i]))[None]} for i in idx])

    def get_next(self):
        return next(self.it, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", required=True, help="landmark npz used to calibrate activation ranges")
    ap.add_argument("--ckpt", default="echonet.pt", help="checkpoint glob inside models/; several files = ensemble")
    ap.add_argument("--max-mb", type=float, default=20.0)
    a = ap.parse_args()

    classes = json.loads(CLASSES_PATH.read_text())
    ckpts = sorted(MODEL_DIR.glob(a.ckpt))
    assert ckpts, f"no checkpoints match models/{a.ckpt}"
    nets = []
    for c in ckpts:
        n = EchoNet(len(classes))
        n.load_state_dict(torch.load(c, map_location="cpu"))
        nets.append(n)
    net = (Ensemble(nets) if len(nets) > 1 else nets[0]).eval()
    print("exporting", [c.name for c in ckpts])

    fp32 = MODEL_DIR / "echogesture_fp32.onnx"
    torch.onnx.export(net, example_input(), str(fp32), input_names=["input"], output_names=["logits"],
                      opset_version=17, dynamo=False)
    quantize_static(str(fp32), str(MODEL_PATH), Reader(load_sequences(a.calib)[0]), quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8, per_channel=True)
    mb = os.path.getsize(MODEL_PATH) / 1e6
    print(f"fp32 {os.path.getsize(fp32) / 1e6:.2f} MB -> int8 {mb:.2f} MB")
    assert mb < a.max_mb, f"model is {mb:.1f} MB, over the {a.max_mb} MB budget"


if __name__ == "__main__":
    main()

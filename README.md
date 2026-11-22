# EchoGesture — Samsung XR Hackathon 2026, Track 02

**Team:** VITChennai_Sinedge · Vellore Institute of Technology, Chennai
**GitHub:** https://github.com/Proextor/EchoGesture

Offline isolated American Sign Language (ASL) recognition system (`WLASL-100` vocabulary) built on skeletal landmark extraction and lightweight ONNX CPU inference (`predict(video_path) -> dict`).

## Architecture & Feature Pipeline
Raw video clips are sampled to **64 frames** with **54 landmarks x 3 coordinates** (`(64, 162)`), normalized relative to the shoulder midpoint and inter-shoulder distance, and converted into **`(64, 486)`** features by concatenating:
1. **Normalized 3D positions** (`162` dims)
2. **First-order temporal velocity** (`162` dims)
3. **Parent-relative bone vectors** (`162` dims)

## Evaluation & Benchmarks
### 1. Five-Seed Ensemble (`ECHO_ENGINE=final`, 100-Clip Test Evaluation)
Checkpoints at `artifacts/final_5seed/seed_1.onnx` through `seed_5.onnx` (`[batch, 64, 486]` -> `[batch, 100]`, combined size `8.0976 MB`, full report in `results/test_results.json` and `results/results.csv`):
- **Test Top-1 Accuracy:** `71.00%` (`>= 70%` target)
- **Corrupted Test Top-1 Accuracy:** `67.00%` (`4.00 pt` robustness drop, `<= 5.0 pt` PASS)
- **Classifier-Only P95 Latency:** `~2.5 ms` on ONNX Runtime CPU (excludes video decode and landmark extraction)
- **165-Clip Validation Top-1:** `66.06%`

### 2. Three-Seed Dynamic-INT8 Ensemble (`artifacts/int8_3seed/`, 165-Clip Validation & Thermal Benchmark)
Dynamic-INT8 (`MatMul`/`Gemm`) checkpoints (`model_int8.onnx`, `model_seed2_int8.onnx`, `model_seed3_int8.onnx`, `1.59 MB` each) evaluated locally on Apple Silicon (`arm64`, ONNX Runtime `1.20.1` `CPUExecutionProvider`) in `results/int8_3seed_val_results.csv`:
- **165-Clip Validation Top-1:** `67.27%` (`66.06%` perturbed)
- **Robustness Accuracy Drop:** `1.21 pt` (`<= 5.0 pt` PASS)
- **Combined P95 Latency:** `9.61 ms` (`<= 50 ms` PASS)
- **Combined Model Size:** `4.78 MB` (`<= 20 MB` PASS)
- **10-Min Sustained Thermal Drift:** `19.63%` (`<= 20%` PASS; Minute 1 P95 `12.75 ms` -> Minute 10 P95 `15.25 ms`)

## Reproducible CPU Setup (Python 3.11)
```bash
python3.11 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/verify_package.py
python scripts/cpu_smoke.py
python predict.py /path/to/actual_clip.mp4
```

### API Output Schema (`predict(video_path)`)
Returns a JSON-serializable dictionary with keys:
- `clip_id`: input file stem
- `prediction`: uppercase top-1 word
- `top5`: top-5 uppercase words in descending probability order
- `confidence`: top-1 softmax probability rounded to four decimal places
- `latency_ms`: inference latency rounded to two decimal places

### Engine Modes & Environment Overrides
- `ECHO_ENGINE=final` *(default)*: 64x486 five-seed ensemble (`artifacts/final_5seed/seed_1.onnx`–`seed_5.onnx`), averaging seed logits before softmax.
- `ECHO_ENGINE=legacy`: 3-seed 64x486 dynamic-INT8 ensemble (`artifacts/int8_3seed/`).
- `ECHO_ENGINE=primary`: 32x289 QDQ-quantized model (`src/primary_backend/models/`).
- `ECHOGESTURE_VOCAB`: Override `vocab.json` path (default checks root `vocab.json`, then `data/vocab.json`).
- `ECHOGESTURE_TTA=1`: Enable 4-view deterministic test-time augmentation on keypoint sequences.

### Batch Manifest Prediction & Docker
```bash
python -m src.main --clips test_manifest.csv --out predictions.jsonl
docker build -t echogesture-final .
docker run --rm -v /absolute/videos:/videos:ro echogesture-final /videos/test.mp4
```

## Submission Resources
- **Presentation:** `submission/VITChennai_Sinedge_Proextor_GitHub_FINAL.pptx`
- **Metrics:** `results/results.csv`, `results/test_results.json`, and `results/int8_3seed_val_results.csv`
- **Training Pipeline:** `notebooks/training_and_evaluation.ipynb` and `training/echo_pipeline.py`
- **Live Webcam HUD Demo:** `scripts/hud_demo.py`

# EchoGesture · Track 02

EchoGesture recognizes 100 isolated ASL glosses on Android. This submission uses five FP32 ONNX seeds with equal-weight mean logits, followed by softmax, with test-time augmentation off. The five graphs contain no INT8 operators. Android performs capture, Holistic landmark extraction, and inference locally.

## Install the debug APK

Download the [five-seed debug APK](https://github.com/Proextor/EchoGesture/releases/download/v0.2-five-seed/app-five-seed-debug.apk) from the [release](https://github.com/Proextor/EchoGesture/releases/tag/v0.2-five-seed). Recognition requires Android 9 (API 28) or later. With USB debugging enabled, run from the download directory in PowerShell:

```powershell
adb -d install -r .\app-five-seed-debug.apk
```


## Build

Install Android SDK 35 and JDK 17 or newer. From the repository root:

```powershell
curl.exe --create-dirs -fLsS 'https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/1/holistic_landmarker.task' -o .\data\android-parity\holistic_landmarker.task
Set-Location .\android
.\gradlew.bat :app:assembleDebug :app:testDebugUnitTest
```

The build verifies the Holistic bundle SHA-256 and packages the checked-in seeds and handoff vocabulary byte for byte. MediaPipe and ONNX Runtime are linked as Android dependencies. The APK above is the exact artifact used for the five-seed emulator fixture check.

## Python inference

From the repository root with Python 3.11:

```powershell
python -m pip install -r .\src\requirements.txt
$env:ECHOGESTURE_FIVE_SEED_DIR = (Resolve-Path .\data\five-seed).Path
python -c "import json,sys; from src.main import predict; print(json.dumps(predict(sys.argv[1])))" 'C:/path/to/clip.mp4'
```

The loader requires all five seeds and reads their matching vocabulary. The public entry point is `src.main.predict(video_path)`. ONNX timing in its output covers the inference loop after preprocessing and session loading. Python MediaPipe may need to fetch the legacy Pose Lite asset on first use.

## Results and scope

- [Five-seed fixture parity](results/five-seed-fixture-parity.json) records the supplied fixture result. Python and Android matched all engineered features, all per-seed logits, and the combined logits. Both returned ACCIDENT, JACKET, DARK, COOL, CHANGE at 0.8533 confidence rounded to four decimals. The raw fixture is excluded.
- The teammate reports 71% top-1 on 100 test clips. The evaluation settings are pending confirmation.
- [Historical validation outputs](results/legacy-three-seed-validation/) belong to the previous three-seed configuration. All 165 IDs were checked against the locally available official WLASL-100 validation clips. The 66.06% top-1 and 14.354 ms P95 describe that run. The timing covers ONNX inference after keypoint preprocessing. Camera-to-caption latency, power, thermal drift, and a five-seed organizer prediction set remain open.
- `quantize_and_eval.py` is the historical conversion and evaluation utility. The five FP32 files above are the published build inputs.

[Android implementation and parity instructions](android/README.md) · [WLASL source](https://github.com/dxli94/WLASL) · [MediaPipe Holistic documentation](https://developers.google.com/edge/mediapipe/solutions/vision/holistic_landmarker)

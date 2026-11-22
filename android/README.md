# EchoGesture for Android

Open this directory in Android Studio with Android SDK 35 and JDK 17 or newer. Fetch the pinned [MediaPipe Holistic task bundle](https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/1/holistic_landmarker.task) from this directory in PowerShell. Gradle verifies its SHA-256 and stages it alongside the five handoff ONNX files and vocabulary.

```powershell
curl.exe --create-dirs -fLsS 'https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/1/holistic_landmarker.task' -o ..\data\android-parity\holistic_landmarker.task
.\gradlew.bat :app:assembleDebug :app:testDebugUnitTest :app:lintDebug
.\gradlew.bat :app:connectedDebugAndroidTest
```

The connected tests require an emulator or device. The camera requests permission when first opened, shows the preview, and records one silent clip between Start and Finish. Empty captures and captures under 350 ms are discarded with a retry message. Captures without usable hands also prompt a retry. The clip lives in app-private cache and is deleted after handling. Use the editable sentence field manually or add a recognized word. Save to history stores up to 50 sentences on the device. Speak uses an installed offline English voice.

## Current classifier contract

The five-seed build requires `seed_1.onnx` through `seed_5.onnx` and the handoff `vocab.json` in `data/five-seed/`. The ONNX files accept float32 `[N,64,486]` and return float32 `[N,100]` logits. The graphs use float weights. Android uses equal-weight mean logits followed by softmax with TTA off. The previously tested three-seed APK is preserved at `submission/app-debug.apk`.

[`src/preprocess.py`](../src/preprocess.py) uses Python OpenCV and legacy MediaPipe Holistic. It processes frames in decode order and produces 64 sampled rows in left hand, right hand, then pose order. Android applies the same sampling, pose indices, shoulder normalization, zero filling, and feature engineering to unmirrored recordings using the MediaPipe Holistic Tasks graph. Its pose detector, pose Lite landmarker, and hand landmarker weights match the legacy files byte for byte. The graph and video decoder can still produce different raw landmarks from Python. Recognition requires Android 9 (API 28) for indexed decoding.

The production `SignClassifier` uses ONNX Runtime CPU with the repository's equal-weight logit ensemble and TTA disabled. The labeled UI fixture is compiled into the instrumented test APK. [ONNX Runtime Android documentation](https://onnxruntime.ai/docs/tutorials/mobile/) · [MediaPipe Holistic Tasks documentation](https://developers.google.com/edge/mediapipe/solutions/vision/holistic_landmarker)

## Compare Python and Android

From the repository root, put the handoff `00626.npy` in `data/five-seed/` and use a Python environment with legacy MediaPipe and ONNX Runtime.

```powershell
$env:ECHOGESTURE_FIVE_SEED_DIR = (Resolve-Path .\data\five-seed).Path
python .\android\export_parity.py .\data\five-seed\00626.npy .\data\five-seed\fixtures --name saved
```

The Python loader requires all five files in this mode. The Android saved-fixture test writes its parity report to `files/android-saved-keypoint-comparison.json` in app-private storage. The handoff reports 71% on 100 test clips. Its evaluation settings are pending confirmation.

# Evidence upgrade

Business problem: distracting backgrounds reduce the usefulness of video communication.
Goal: preserve people while reducing processing cost. This remains an offline CPU
prototype, not a shipped real-time or depth-based camera feature.

## Implemented
- OpenCV thread budget (default 2) alongside ONNX threads. Earlier runs left OpenCV
  threads uncontrolled. Do not attribute cross-version gains solely to C++.
- Separate background_blur and blend_and_quantize timers; composite retains the
  combined historical scope.
- Opt-in --blur-scale .25: lower-resolution normalized masked convolution, followed
  by upsampling. Approximate output; evaluate hair and motion. Default 1 preserves
  full-resolution blur. Intermediate output stays float32.
- Paired blend benchmark: identical float32 inputs, contiguous/strided cases,
  alternating execution order, raw samples, numeric parity assertion.
- Cached-frame effect benchmark isolates blur and blend from inference and I/O.
- Supervised residual refiner training on frozen RVM predictions and ground truth.
  This is NOT fine-tuning RVM weights or temporal video training.
- Windows/Linux test CI and macOS Swift/Metal source checks. Inspect CI results;
  source checks do not establish iPhone runtime correctness.

## Windows next run
In Command Prompt, from the project folder, using the existing project venv:

```cmd
".venv-portrait312\Scripts\python.exe" -m portraitfx process --input examples/my_video.mp4 --model models/rvm_mobilenetv3_fp32.onnx --output results/my_blur_fast.mp4 --width 480 --ratio 0.5 --backend cpp --blur-scale .25 --opencv-threads 2 --measure-memory --report results/my_benchmark_fast.json
```

For a controlled baseline use --blur-scale 1 and different output/report names.
The saved playback FPS is not processing speed. No Windows speed target is promised.
Keep your footage and previous reports when updating source.

## Isolate performance
Use your project venv's python for these commands:

```
python benchmark_native.py
python -m experiments.benchmark_effects --input examples/my_video.mp4 --model models/rvm_mobilenetv3_fp32.onnx --frames 30 --repeats 3
```

The second command caches 30 frames (memory cost), runs each effect on the same
RVM outputs and alternates order. RGB MAE against full-scale rendering measures
approximation error, NOT ground-truth quality.
The synthetic blend benchmark requires numpy.random; local Smart App Control
currently blocks that official dependency. Do not skip tests or alter generators
to evade policy. Seek Microsoft/device administrator review of the blocked binary.
CI validates a separate environment; it does not fix local policy.

## Actual training and quality evidence
Obtain licensed RGB portrait images AND corresponding ground-truth grayscale alpha
mattes. Phone footage without labels cannot establish alpha accuracy. Never use
RVM predictions as ground truth. Preserve separate identities/sequences across
train/val/test. Unknown overlap with RVM pretraining must be disclosed.
Use the manifest format in RUN_EXPERIMENTS.md (image, alpha, group, license, split).
Paths are relative to the manifest. Nonempty train/val/test required; duplicate
images and groups crossing splits are rejected.

```
python -m pip install -r requirements-experiments.txt
python -m experiments.train_refiner --manifest data/portraits/manifest.json --model models/rvm_mobilenetv3_fp32.onnx --output results/refiner_real --epochs 20 --size 128
```

The refiner starts at identity, learns alpha residuals using supervised L1 loss,
selects checkpoints by validation group-macro MAE, and evaluates test once at the
end against frozen RVM. Reports include MAE/MSE/boundary MAE, per-image results,
model/manifest hashes and split counts. It streams images but recomputes RVM each
epoch. Square resizing changes aspect ratio. RVM is frame-independent here.
The refiner.pt artifact takes RGB+baseline alpha as NCHW float32 at training size.
It is experimental and not integrated into video/iOS inference. Improvement is
not guaranteed. The synthetic smoke report is only a mechanics check, not evidence
of real-person quality. No real-person training result has been completed here.

For video quality run experiments.evaluate_video with a licensed paired held-out
manifest, both recurrent and independent paths. See RUN_EXPERIMENTS.md for CLI.
Cover multiple sequences: hair, fingers, fast motion, low light, clutter, empty
scene and cuts. Retain failed cases and per-sequence metrics. Do not tune on test.

## Apple platform
CI type-checks Swift and compiles Metal with Xcode tools on macOS. It does not
sign/install an app or measure camera permissions, rendering, device thermal state,
power or preview latency. Follow ios/README.md using a Mac and iPhone 15 Plus.
Windows cannot execute the iOS camera frameworks. This gap remains open until
actual device testing happens. CI itself must pass before claiming source checks.

## Verified in this environment
32 tests passed on Linux. Isolated blend at 480x854 contiguous float32: NumPy p50
3.87ms, C++ 1.34ms, maximum absolute error 0 in this fixture. Strided arrays have
smaller gains; see evidence/blend_paired_linux.json for all samples.
On 10 astronaut smoke frames at width480, cached C++ effects p50 was 22.79ms at
full blur versus 2.60ms at quarter blur; RGB MAE versus full-scale reference 0.00124.
This is approximation deviation, not portrait quality. A separate 320x320 smoke
pipeline had p50 23.93ms. None of these measurements is a Windows/iPhone claim.

Sources:
https://github.com/PeterL1n/RobustVideoMatting/blob/master/documentation/training.md
https://learn.microsoft.com/windows/apps/develop/smart-app-control/overview

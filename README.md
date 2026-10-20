# Portrait Video Effects

A working CPU video-matting pipeline for background blur and replacement, using pretrained Robust Video Matting (RVM). Built as a starting point for camera/video ML engineering experiments.

## Run the new experiments

See [RUN_EXPERIMENTS.md](RUN_EXPERIMENTS.md) for the Windows runner, training, labeled video evaluation, calibrated two-view reconstruction and the iOS source prototype. Synthetic checks have run; real-data quality and Apple device validation are still pending.

## Status
Implemented and tested: recurrent ONNX inference, scene-cut reset heuristic, independent-frame ablation, foreground-aware background blur, image replacement, lossless 16-bit alpha PNG export, per-stage p50/p95 timing, model checksum, and alpha evaluation. The expanded suite has 27 passing tests after building the native module. Real pretrained inference exported and decoded 20 frames successfully.

Not completed: model fine-tuning, real-video labeled quality evaluation, depth estimation, motion-compensated temporal metrics, validated Swift/Core ML device application and Metal performance, device energy/thermal testing. Do not describe this as trained, production-ready, or deployed on an iPhone. Real iPhone footage has been processed on a host CPU.

## Setup (Python 3.10+)
Run commands from this directory. Create and activate a Python virtual environment first.

```sh
pip install -r requirements.txt
python download_model.py
python build_native.py
python -m pytest -q
python -m portraitfx process --input examples/smoke_input.mp4 --model models/rvm_mobilenetv3_fp32.onnx --output results/blur.mp4 --width 320 --ratio .5 --report results/benchmark.json
```

Replace the input with your own video. For a photo background:

```sh
python -m portraitfx process --input your_video.mp4 --background your_background.jpg --model models/rvm_mobilenetv3_fp32.onnx --output results/replaced.mp4 --alpha-dir results/alpha --width 640 --report results/replacement.json
```

Outputs are silent MP4 files; audio preservation is not implemented. Existing output files are overwritten, except the input video cannot be used as output. Use a fresh alpha directory for each run.

## Recurrent ablation
Run the same input and settings with `--independent` to reset model state every frame. Save to separate output/report/alpha directories. This comparison isolates recurrence; do not change resolution or downsample ratio simultaneously. A scene cut is detected by mean absolute change of 32x32 RGB thumbnails, default threshold .35; this is a heuristic and can miss cuts or reset during motion.

## Evaluation
Export predictions to alpha PNGs and supply manually verified ground-truth alpha PNGs with exactly matching filenames, dimensions, and frame alignment:

```sh
python -m portraitfx evaluate --pred results/alpha --truth data/test_alpha --output results/quality.json
```

Reports per-frame and macro-average MAE, MSE, and MAE in a 5x5 morphological band around the thresholded ground-truth boundary. Boundary-band MAE is a custom diagnostic, not an established benchmark score. Empty boundaries are reported as null. Input alpha must be grayscale uint8/uint16. Do not evaluate a prediction against itself and report it as model quality.

Use subject/video-disjoint training, validation and test sets. Select configurations using validation only. Reserve the test set for final evaluation. Include real motion, hair, glasses, occlusion, lighting changes and empty scenes; identify data licensing and consent. Aggregate per video before comparing conditions, and report sample sizes. Raw frame-to-frame differences confound movement with flicker; implement ground-truth temporal error or motion-aware evaluation before claiming stability improvements.

## Measured smoke test
`results/smoke_benchmark.json` records actual execution with the released model: 20 frames at 320x320, CPU, two ONNX threads. Input is an artificially translated still NASA photograph, NOT a natural video or quality benchmark. This small test cannot establish sustained real-time capability. RSS is null because process memory inspection is unavailable in the runner.

Timing includes host preprocessing and synchronous inference; model loading/warmup, first-frame decode and encoder final flush are excluded. Alpha PNG writing is included if requested. CPU inference timing includes RGB conversion and recurrent-state transfer. Memory sampling is opt-in with `--measure-memory` on a host with reliable process IDs. Memory, when available, is sampled process RSS, not exact peak or GPU memory. `requirements-tested.txt` records versions used.

## Engineering decisions
- Use a pretrained recurrent model as a reproducible baseline; no claim of inventing or training it.
- Preserve the four recurrent states in sequence; reset at resolution changes and detected cuts.
- Blur the estimated background using normalized masked convolution to reduce person-color bleed. This is mask-based blur, not physically accurate bokeh or depth-aware rendering.
- Avoid extra smoothing until tested: it may create trails and hide tracking errors.
- CPU backend is explicit. GPU and Neural Engine use are not implied.

## Roadmap to a strong Apple portfolio project
1. Collect licensed natural-video evaluation clips and aligned alpha labels; document splits and difficult conditions.
2. Fine-tune a lightweight model using the original training implementation, preserve original/pretrained comparison, save checkpoints and logs.
3. Compare recurrence and model downsample ratios on validation data, measure quality and stage latency; run multiple seeds for training.
4. Extend the measured C++ compositing implementation; evaluate blur optimization separately.
5. Integrate the model in Swift/Core ML on actual Apple hardware. Measure sustained camera-to-display latency, dropped frames and thermal behavior.
6. Add depth-based blur only after the baseline is evaluated.

## Attribution
RVM: Shanchuan Lin et al., *Robust High-Resolution Video Matting with Temporal Guidance*, WACV 2022. https://github.com/PeterL1n/RobustVideoMatting . Original repository is GPL-3.0; preserve applicable notices and review upstream model terms. Model is downloaded separately, not bundled. Source release: v1.0.0, SHA-256 `88d4531297118f595bf2fd60f6f566aec2e559393802d1f436c380f0cbbd2828`.

Example image: NASA portrait of astronaut Eileen Collins, distributed with scikit-image (public domain). https://github.com/scikit-image/scikit-image/blob/v0.24.0/skimage/data/astronaut.png . Motion in the bundled clip is synthetic translation. It is provided only for a reproducible smoke test.


## C++ compositing backend

`native/composite.cpp` implements a fused, three-channel alpha blend in C++17.
Python calls its C ABI through ctypes. The wrapper validates shapes, converts
strided inputs to contiguous float32, owns the output allocation, and keeps
buffers alive during the native call. The native function neither allocates nor
retains pointers. Alpha and output are clamped to [0,1]; NaNs remain NaN.
The public Python wrapper is the safe entry point; calling the raw C ABI with
invalid pointers or lengths is unsupported.

This is a C++ kernel integrated with an OpenCV/Python application. It does not
use the OpenCV C++ SDK. Inference and Gaussian blur remain unchanged.
No GPU, Metal, Core ML or on-device acceleration is implied.

### Windows build and run

Install Microsoft's Visual Studio Build Tools with the **Desktop development
with C++** workload if no C++ compiler is installed. Open **Developer PowerShell
for VS** (x64) and run from the extracted project folder:

```powershell
& "C:\Users\amarg\.venv\Scripts\python.exe" -m pip install -r requirements.txt
& "C:\Users\amarg\.venv\Scripts\python.exe" build_native.py
& "C:\Users\amarg\.venv\Scripts\python.exe" -m pytest -q
& "C:\Users\amarg\.venv\Scripts\python.exe" download_model.py
& "C:\Users\amarg\.venv\Scripts\python.exe" -m portraitfx process --input examples/smoke_input.mp4 --model models/rvm_mobilenetv3_fp32.onnx --output results/cpp_blur.mp4 --width 320 --ratio 0.5 --backend cpp --report results/cpp_windows.json
Invoke-Item ".\results\cpp_blur.mp4"
```

Linux: install g++, then run `python build_native.py`. macOS: use clang++.
The Linux build is tested here; Windows/MSVC and macOS build paths are supplied
but have not been executed in this environment. Build for the same architecture
as Python. No precompiled native binaries are distributed.

### Reproduce measurements

Run `python benchmark_native.py`. This uses seed 42, five warmups and 60 timed
calls per backend, alternates backend order, and includes Python boundary,
conversion and allocation costs. It excludes inference, Gaussian blur and video
IO. Inputs mimic negative-stride BGR foreground and float64 background from the
existing pipeline; C++ converts to float32, so improvements include reduced
precision/temporary storage as well as fused execution. This is not an isolated
language comparison. See `results/native_benchmark.json` for raw measurements.

At 480x854 on the recorded Linux host, p50 compositing was 28.59 ms NumPy versus
11.15 ms native, with maximum absolute error 8.50e-8. These synthetic results
are not Windows or iPhone timings and do not establish full-pipeline speedup.
Tests cover float16/32/64 inputs, noncontiguous layouts, clipping, alpha endpoints,
invalid shapes and NaN behavior. `results/native_smoke.json` records the full
20-frame pipeline using the native backend.

### Resume wording supported by this version

Implemented a C++17 alpha-compositing kernel integrated with a Python/ONNX video
matting pipeline through ctypes; validated numerical parity and benchmarked
compositing latency across three resolutions.

Understand the buffer layout, pointer lifetime, C ABI and measurement scope
before presenting this as your C++ experience. Real labeled portrait training is documented in REAL_DATA.md. Labeled-video
quality evaluation remains future work.

## Evidence upgrade

See [UPGRADE_GUIDE.md](UPGRADE_GUIDE.md) for controlled timings, approximate blur,
supervised refinement, CI and the outstanding real-data/device requirements.

## Real labeled portrait experiment

See [REAL_DATA.md](REAL_DATA.md) for the pinned P3M-10K subset, supervised residual-refiner protocol, reproducible commands, and measured limitations.

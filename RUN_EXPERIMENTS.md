# Run the gap-closing experiments

## What is implemented, and what remains to validate

| Component | Delivered | Evidence still needed |
|---|---|---|
| C++ | Fused blend, ctypes boundary, strength control, numerical tests | Windows build and your ability to explain/modify it |
| Labeled evaluation | Video/subject grouping, split leakage checks, recurrence ablation, per-sequence MAE/MSE/boundary error | Licensed natural videos and aligned ground-truth alpha masks |
| Training | Small PyTorch RGB-to-alpha CNN, reproducible seed, validation checkpoint selection, held-out comparison, TorchScript export | Real-data training; this is not RVM fine-tuning |
| 3D | SIFT matching, RANSAC essential matrix, relative pose, triangulation, parallax/depth filtering, PLY export; chessboard calibration utility | Calibrated real scene; no bundle adjustment, SLAM or Gaussian splatting |
| iOS | Swift camera source, Vision person segmentation, Metal background replacement, optional Core ML model path | Mac compilation, signing, orientation checks and iPhone measurements |

## First run on Windows
Download this repository ZIP from GitHub and extract it. Install Visual Studio
Build Tools with Desktop development with C++ if needed. Open Developer PowerShell
for VS (x64), change to the extracted folder, then:

```powershell
.\run_windows.ps1 -TrainDemo
```

If PowerShell blocks script execution, do not change machine policy. Run these
commands individually from that same folder:

```powershell
py -m venv .venv
& '.\.venv\Scripts\python.exe' -m pip install -r requirements-experiments.txt
& '.\.venv\Scripts\python.exe' build_native.py
& '.\.venv\Scripts\python.exe' -m pytest -q
& '.\.venv\Scripts\python.exe' -m experiments.make_demo
& '.\.venv\Scripts\python.exe' -m experiments.train --manifest data/synthetic/manifest.json --output results/my_first_training --epochs 20
& '.\.venv\Scripts\python.exe' -m experiments.reconstruct --demo
```

Choose a fresh training output folder for every run. Training defaults to CPU;
GPU is optional with `--device cuda` and a compatible PyTorch installation.
The tiny baseline loads resized samples into RAM. Use it for small datasets;
large-scale training needs a streaming Dataset. It resizes images to a square,
which distorts aspect ratio. This is a documented baseline limitation.

## C++ learning exercise
Rebuild native code after upgrading: the v2 exported symbol rejects an old build.
`--strength 1` gives normal alpha compositing. `--strength 0` gives the model's
estimated foreground RGB, NOT the original camera frame. Intermediate strengths
change effective alpha to `1-strength*(1-alpha)`. This parameter is intentionally
a compositing exercise, not a polished blur-strength user control.

Explain: why HxWx3 arrays must be contiguous, why NumPy must keep buffers alive,
why the C ABI is exported, why ctypes declares argument types, and why no native
buffer is kept after the call. Modify the parameter and add a boundary-case test.

## Real labeled data contract
You must supply data you are allowed to use. Do not use predictions as labels.
Binary segmentation masks are not soft alpha ground truth and cannot validate
fine hair matting. All frames of the same person/sequence belong in one split;
use a shared `group` for that identity. If multiple people recur across sequences,
assign groups so none cross train/val/test. Filename order alone is insufficient.

Create a JSON manifest; paths are relative to that manifest:

```json
{
  "synthetic": false,
  "samples": [
    {"image":"rgb/clip01/000000.png", "alpha":"alpha/clip01/000000.png",
     "split":"test", "group":"subject01", "sequence":"clip01", "frame":0,
     "license":"Record the actual dataset license/source here"}
  ]
}
```

Include train, val and test groups for training. Alpha must be 8/16-bit grayscale,
matching the RGB dimensions and alignment. `frame` must increase in temporal order
within each sequence. Set `sequence` to the clip identifier; `group` controls
subject split membership. The evaluator resets recurrence between sequences.

```powershell
& '.\.venv\Scripts\python.exe' download_model.py
& '.\.venv\Scripts\python.exe' -m experiments.evaluate_video --manifest data/real/manifest.json --model models/rvm_mobilenetv3_fp32.onnx --split val --output results/val_quality.json
& '.\.venv\Scripts\python.exe' -m experiments.train --manifest data/real/manifest.json --output results/real_training --epochs 20 --size 128
```

Tune only with validation data. The training script reports an untrained and
train-mean constant baseline, selects its checkpoint using validation MAE, and
reports the selected model on test data. It does not tune on test results.
Repeatedly choosing experiments based on test scores would invalidate that test.
Video evaluation compares RVM recurrent vs independent frames, weighting each
sequence equally. It does not compute an established temporal consistency metric.
The synthetic fixture has single-frame groups, so its recurrence comparison only
checks execution, not recurrence benefit. Report failures and empty-frame behavior.

## Real 3D capture
Use the same camera lens, zoom and photo resolution throughout. Photograph a
printed 9x6 INNER-corner checkerboard in at least 10 varied positions/angles.
Then take two overlapping views of a static textured nonplanar scene with a
sideways camera translation (not just rotation). Avoid glass, blank walls and
moving people. Transfer full-resolution originals, not compressed chat copies.

```powershell
& '.\.venv\Scripts\python.exe' -m experiments.calibrate --images 'data/boards/*.jpg' --cols 9 --rows 6 --output data/calibration.json --undistort data/left.jpg data/right.jpg
& '.\.venv\Scripts\python.exe' -m experiments.reconstruct --left data/left_undistorted.png --right data/right_undistorted.png --calibration data/calibration.json --output results/real_reconstruction
```

View `points.ply` in a point-cloud viewer. Camera poses and metrics are in
`report.json`; feature matches are in `matches.jpg`. Two-view scale is arbitrary.
Low reprojection error on fitting points does not prove real-world 3D accuracy.
Planar scenes/pure rotation can produce ambiguous poses even with plausible residuals.

## Apple validation
Follow `ios/README.md` on a Mac. Windows cannot validate this iOS app. The default
Vision path needs no trained model. The optional Core ML path is separate from
RVM and must not be described as RVM conversion. Do not put the circle-trained
demo model in a person-segmentation demo except to test model loading.

## Evidence to share after running
Send the test summary, training report JSON, reconstruction report JSON and a
screenshot of the point cloud. Keep raw private footage out of GitHub. For iOS,
share build errors or device performance logs; we can then fix actual failures.

References:
https://docs.opencv.org/4.10.0/d9/d0c/group__calib3d.html
https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial
https://apple.github.io/coremltools/docs-guides/source/image-inputs.html

# Real portrait training experiment

Problem: portrait background effects depend on accurate person alpha mattes, especially at hair and clothing boundaries. This experiment asks whether a small supervised residual CNN improves a frozen RVM baseline on real labeled portraits.

## Data and provenance

- Original: [P3M-10K, Privacy-Preserving Portrait Matting](https://github.com/JizhiziLi/P3M), Li, Jizhizi, Sihan Ma, Jing Zhang, and Dacheng Tao. ACM Multimedia 2021.
- [Dataset release agreement](https://jizhizili.github.io/files/p3m_dataset_agreement/P3M-10k_Dataset_Release_Agreement.pdf): MIT license and additional release conditions apply; read these before downloading/using the data. Original image copyrights remain with their owners.
- Download source: unofficial [nobg/P3M-10K Hugging Face mirror](https://huggingface.co/datasets/nobg/P3M-10K), pinned revision `a239de9297bd56e372cf3c9c82f7085d59e71219`.
- Original embedded PNG alpha masks, not JPEG dataset-viewer previews. RGB mask channels are verified equal before lossless grayscale conversion.
- Two parquet files, approximately 784 MB combined; SHA-256 verified by the preparation script. No raw dataset images are committed to this repository.
- Fixed seed 42 selects 128 training and 32 validation images from the first training shard, and 64 held-out images from the official P3M-500-P test split. This is a bounded subset, not the complete benchmark or a representative random draw from all training shards.
- Groups identify images, not people. Exact image-file duplicates across splits are rejected; identities and near-duplicates are not audited. Pretraining overlap with RVM cannot be ruled out.

## Protocol

Freeze RVM; train only a 4-channel RGB+alpha residual CNN with 16 hidden channels. Resize every pair to 128x128, use independent-frame RVM with ratio 0.5 and 2 CPU threads. Adam, learning rate 0.001, batch 4, seed 42, 20 epochs, deterministic PyTorch operations. Cache frozen predictions in CPU memory. Validation mean absolute error selects the checkpoint, including the untrained identity checkpoint at epoch 0. Evaluate the held-out test exactly once after selection.

Report image-macro alpha MAE, MSE and a 5x5 morphological boundary-band MAE on alpha in [0,1]. Smaller is better. These are low-resolution still-image metrics, not full-resolution benchmark numbers or evidence of temporal stability. RVM weights are not fine-tuned; the learned refiner is a separate experiment and is not integrated into video or the iOS source.

## Reproduce in Windows Developer Command Prompt

Run from the project directory using your existing project environment:

```bat
".venv-portrait312\Scripts\python.exe" -m pip install -r requirements-real-data.txt
".venv-portrait312\Scripts\python.exe" -m experiments.prepare_p3m
".venv-portrait312\Scripts\python.exe" -m experiments.train_refiner --manifest data/p3m-real/manifest.json --model models/rvm_mobilenetv3_fp32.onnx --output results/refiner_real --epochs 20 --size 128 --batch 4 --cache-samples
```

Preparation refuses to overwrite an existing manifest; training requires an empty output directory. Dataset download and extraction require several GB free disk. The existing Windows Application Control restriction on NumPy is not changed or bypassed by this experiment. The measured run in this repository is from Linux CPU, not your Windows PC or iPhone.

### Offline execution used here

The execution environment rejected optional Microsoft runtime telemetry. The final run uses the Linux-only `experiments.run_offline` wrapper: libseccomp denies socket/connect/send network system calls for the process and its descendants, and verifies that opening a network socket fails before importing the training module. This adds network restrictions; it does not change Windows security policy. All data and weights must already be downloaded.

```bash
python -m experiments.run_offline experiments.train_refiner --manifest data/p3m-real/manifest.json --model models/rvm_mobilenetv3_fp32.onnx --output results/refiner_real --epochs 20 --size 128 --batch 4 --cache-samples
```

## Measured result: completed Linux CPU run

20 epochs, checkpoint selected at epoch 20. Evaluation uses the fixed 64-image test subset at 128x128:

| Metric (lower is better) | Frozen RVM | Trained refiner |
| --- | ---: | ---: |
| Alpha MAE | 0.141153 | 0.120036 |
| Alpha MSE | 0.113635 | 0.094904 |
| Boundary-band MAE | 0.295214 | 0.282302 |

**14.96% relative reduction in mean alpha MAE on this subset.** 50 images improved and 14 worsened. A paired image-level bootstrap (10,000 resamples, seed 42) gives a 95% percentile interval of [0.01441, 0.02856] for the absolute mean MAE reduction. This interval does not account for shared identities, subset-selection bias, or training-seed variation.

Evidence:
- [Full training history and per-image test metrics](evidence/refiner_p3m_real.json)
- [Sample IDs, source rows and image/mask SHA-256 values](evidence/p3m_subset_manifest.json) (paths are relative to the prepared dataset, not the evidence folder)
- [Worst cases, regressions, and bootstrap summary](evidence/refiner_p3m_analysis.json)
- [Exact package versions used](evidence/real_data_environment.txt)
- [Selected small checkpoint](evidence/checkpoints/refiner_p3m.pt), SHA-256 `51e4adf51b0755fbab3d3f7ad0ba8ab1680cd35acff84889c593f6e4a164b838`

All 224 selected decoded images were checked; no exact decoded-image duplicates crossed splits. Near-duplicates and identities were not audited. Ground-truth soft alpha occupies between 0.31% and 7.98% of pixels in individual full-size masks, confirming that these are soft mattes rather than binary-only labels.

Execution history: an initial image-copy integrity failure was repaired before training. Two runtime-telemetry-blocked attempts did not produce a completed report. An offline attempt completed training but failed while recording the OpenCV distribution metadata; the identical protocol was rerun after fixing metadata lookup. No test outcomes were inspected to change hyperparameters, samples, or checkpoint selection.

**Interpretation:** this demonstrates real supervised training and held-out evaluation, not production-ready matting. The absolute errors remain substantial; the refiner is capped at an alpha correction of 0.25 and cannot recover large RVM misses. Square resizing distorts aspect ratios and removes hair detail. The 14 regressions require follow-up on new validation data before deployment. No claim of real-time video, temporal robustness, Apple-platform execution, or improvements on the user's own clip follows from this experiment.

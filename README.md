> Language / 语言 / 言語: [English](./README.md) | [中文](./docs/README.zh.md) | [日本語](./docs/README.jp.md)

# JPEG QF Prediction (Regression)

An inference tool for a **regression model** that predicts the JPEG Quality Factor (QF).  

---

## Table of Contents <!-- omit from toc -->

- [Features](#features)
- [Performance](#performance)
- [Inference Speed (Measured)](#inference-speed-measured)
- [Limitations](#limitations)
- [Project Structure](#project-structure)
- [Environment Setup](#environment-setup)
- [Quick Start (Single Image Mode)](#quick-start-single-image-mode)
- [Full Usage Workflow](#full-usage-workflow)
- [Command-line Argument Help](#command-line-argument-help)
- [Output Description (CSV / DB Fields)](#output-description-csv--db-fields)
- [Inference Workflow (Patch Selection Logic)](#inference-workflow-patch-selection-logic)
- [Skip and Error Handling Mechanism](#skip-and-error-handling-mechanism)
- [License](#license)

---

## Features

- ✅ Single JPEG image inference
- ✅ Batch inference for folders
- ✅ Recursive subdirectory scanning
- ✅ Output to `CSV` / `SQLite` / `CSV+SQLite`
- ✅ Separate storage for:
  - Successful predictions (`predictions`)
  - Skipped/error samples (`issues`)
- ✅ TTA (test-time augmentation via flips)
- ✅ Built-in checkpoint calibrator (toggleable)
- ✅ Automatic `cuda/cpu` selection
- ✅ Pixel-limit skip option (avoid excessive resource usage on huge images)
- ✅ Custom file suffix filtering (e.g. `.jpg .jpeg .jfif`)
- ✅ Automatically appends timestamp if output file already exists (prevents overwrite)

---

## Performance

Test set results:

- **Number of test samples: 2763**
- **MAE (after calibration): 0.1541**
- **Acc@1 (after calibration): 99.78%**
- **Acc@2 (after calibration): 99.86%**
- **Acc@5 (after calibration): 99.93%**

---

## Inference Speed (Measured)

- **Average about 3 seconds / image** (measured on the test set)
- Hardware platform (**laptop**):
  - **CPU**: Intel i7-12700H
  - **Memory**: 16GB
  - **GPU**: RTX 3060 Laptop GPU

> Actual speed may vary depending on image size, whether TTA is enabled, disk I/O speed, system load, and other factors.

---
## Limitations

- Only applicable to JPEG or similarly compressed images
- QF is related to encoder implementation (PIL/libjpeg/OpenCV etc. may differ)
- After recompression, multiple saves, cropping/filter processing, the meaning of QF prediction becomes weaker

---
## Project Structure

Recommended repository structure (example):

```text
.
├── infer.py
├── model.py
├── requirements.txt
├── checkpoints/
│   └── model.pth
├── demo/
│   └── test.jpg
├── outputs/
└── README.md
```

---

## Environment Setup

### 1) Python Environment
Python 3.9+ is recommended.

### 2) Install dependencies (using `requirements.txt`)

```bash
pip install -r requirements.txt
```

> If your `requirements.txt` does not include a PyTorch build matching your local CUDA version, install PyTorch first using the official instructions, then run the command above.

### 3) Prepare model weights
- The inference script requires a `.pth` weights file specified via `--ckpt`.
- Model weights are included in the repository (under `checkpoints/`) and managed with Git LFS.
```bash
  git lfs install
  git clone https://github.com/sawakage/jpeg_qf_predictor.git
  cd jpeg_qf_predictor
  git lfs pull
```
---

## Quick Start (Single Image Mode)

Useful for checking whether the script runs successfully.

```bash
python infer.py --ckpt ./checkpoints/model.pth --input ./demo/test.jpg
```

If successful, you will see output similar to the following (single-image mode):

```text
===== 结果 / Result =====
图片 / Image: ./demo/test.jpg
最终QF / Final QF: 92.0000
[+] 成功预测CSV / Predictions CSV: ./outputs/predictions.csv
[+] 问题样本CSV / Issues CSV: ./outputs/issues.csv
[*] 设备 / Device: cuda
...
```

---

## Full Usage Workflow

The sections below explain practical usage scenarios in detail.

> Linux/macOS: use `\` for line continuation  
> Windows CMD: use `^` for line continuation  
> Or write the command on a single line (no line continuation needed)

### 1) Single-image inference

Single-image usage is the same as in Quick Start.

Use cases:
- Debug device / path / permission issues
- Small-scale manual verification

---

### 2) Batch inference for a folder

```bash
python infer.py \
  --ckpt ./checkpoints/model.pth \
  --input ./data/images
```

Default behavior:
- Scans JPEG files in the directory
- **Does not recurse** into subdirectories

---

> The following are additional arguments. Append them after the command above as needed.

### 3) Recursively scan subdirectories

```bash
  --recursive
```

Use cases:
- Dataset directory has deep nesting
- Images are organized by class/source folders

---

### 4) Output results as CSV / SQLite / both

The script supports three output modes:

- `csv` (default)
- `db`
- `both`

#### 4.1 CSV only (default)

```bash
  --output_mode csv
```

Output files (default):
- `./outputs/predictions.csv`
- `./outputs/issues.csv`

#### 4.2 SQLite database only

```bash
  --output_mode db
```

Output file (default):
- `./outputs/results.db`

#### 4.3 Output CSV + SQLite simultaneously

```bash
  --output_mode both \
  --pred_csv ./outputs/predictions.csv \
  --issue_csv ./outputs/issues.csv \
  --output_db ./outputs/results.db
```
**Output path arguments are optional**
> If a target output file already exists, the script automatically appends a timestamp to the filename to avoid overwriting.

---

### 5) Control TTA / calibrator / device

#### 5.1 Disable TTA (for speed)
TTA (flip augmentation aggregation) is enabled by default. Disable it if you want faster inference:

```bash
  --disable_tta
```

#### 5.2 Disable calibrator (if checkpoint contains calibrator)

```bash
  --no_calibrator
```

#### 5.3 Specify device

- Auto select (default): `--device auto`
- Force CPU: `--device cpu`
- Force CUDA: `--device cuda`

If `cuda` is specified but unavailable in the current environment, the script will exit with an error (expected behavior).

---

### 6) Limit pixel count for very large images

The script supports a “pixel-limit skip” mechanism for huge images to avoid excessive resource usage.

#### Enable pixel limit

```bash
  --enable_pixel_limit
```
Default pixel limit:
- 178,956,970 pixels (Pillow default `MAX_IMAGE_PIXELS`; may differ by version)
#### Customize pixel limit

```bash
  --enable_pixel_limit \
  --max_image_pixels 100000000
```

Notes:
- This is a **skip mechanism**, not automatic resizing
- Skipped images are written to `issues.csv` or the `issues` table for later processing

---

### 7) Custom suffix filtering

By default, only the following suffixes are processed:

- `.jpg`
- `.jpeg`
- `.jpe`
- `.jfif`

If you want to expand or restrict the processing scope:

```bash
  --exts .jpg .jpeg
```

Notes:
- If input is a single file and its suffix is not in the allowed list, the script raises an error directly
- If input is a directory, only files with allowed suffixes are scanned and processed

---

### 8) Quiet mode (reduce log output)

Use `--quiet` when you only care about output files and don’t want too much terminal output.

In quiet mode:
- Terminal logs are reduced
- Progress bars are usually hidden in multi-image mode
- CSV / DB results are still written normally

---

## Command-line Argument Help

Run the following command to view the script help message directly:

```bash
python infer.py --help
```
---

## Output Description (CSV / DB Fields)

The script stores “successful predictions” and “problematic samples” separately for easier downstream analysis, cleaning, and reruns.

### 1) Successful prediction CSV (`predictions.csv`)

Fields:
- `path`: full image path
- `filename`: file name
- `pred_qf_final`: final predicted QF

---

### 2) Problem sample CSV (`issues.csv`)

Fields:
- `path`
- `filename`
- `issue_type`: `skipped` or `error`
- `issue_message`: skip reason or exception message

---

### 3) SQLite database (`results.db`)

The database contains two tables:

#### `predictions`
- `id`
- `path`
- `filename`
- `pred_qf_final`

#### `issues`
- `id`
- `path`
- `filename`
- `issue_type`
- `issue_message`

---

## Inference Workflow (Patch Selection Logic)

This section does not go into model details. It focuses on what the script does during real inference so you can understand why some images are skipped.

### Patch filtering strategy

Fixed inference parameters currently used in the script:

- Patch size: `128×128`
- Stride: `128`
- Maximum patches used per image: `24`
- Variance threshold: `15.0`
- Relaxed threshold: `15.0 / 1.5`

Patch selection logic (priority order):
1. If high-threshold patch count ≥ 24: take the first 24 high-threshold patches
2. If high-threshold patch count ≥ 5: use all high-threshold patches
3. If relaxed-threshold patch count ≥ 5: take the first 5 relaxed-threshold patches
4. If relaxed-threshold patch count > 0: use all relaxed-threshold patches
5. Otherwise: image is considered to have too little information; skip and write to `issues`

### TTA (test-time augmentation)

Enabled by default. The script applies flip augmentation and averages:
- Original image
- Horizontal flip
- Vertical flip
- Horizontal + vertical flip

### Final aggregation for a single image

- The model outputs predictions for multiple patches
- The script uses the **median** as the final whole-image prediction (more robust to outlier patches)

---

## Skip and Error Handling Mechanism

The script does not interrupt the entire batch process because a single image fails. Instead, it records the issue to `issues` and continues with later images.

### Common skip reasons (`issue_type=skipped`)

- Image pixel count exceeds the limit (when pixel limit is enabled)
- Image size is smaller than patch size (`128×128`)
- Too little image content/information (no valid patches can be selected)

### Common errors (`issue_type=error`)

- Corrupted / truncated file / decode failure
- Invalid or incomplete JPEG
- Weight file and model architecture mismatch
- `--device cuda` specified but CUDA is unavailable in current environment
- For single-file input, the suffix is not in the allowed list (raises error directly, not part of batch processing)

## License
MIT License

> Language / 语言 / 言語: [English](./README.md) | [中文](./docs/README.zh.md) | [日本語](./docs/README.ja.md)

# JPEG QF Prediction (Regression)

An **inference tool based on a regression model** for predicting the JPEG Quality Factor (QF) of **anime-style / 2D images**.

---

## Table of Contents <!-- omit from toc -->

- [Features](#features)
- [Performance](#performance)
- [Inference Speed (Measured)](#inference-speed-measured)
- [Limitations](#limitations)
- [Project Structure](#project-structure)
- [Environment Setup](#environment-setup)
- [Complete Usage Workflow](#complete-usage-workflow)
- [Output Description (CSV / DB Fields)](#output-description-csv--db-fields)
- [Inference Workflow (Patch Selection Logic)](#inference-workflow-patch-selection-logic)
- [License](#license)

---

## Features

- ✅ Run inference on a single JPEG image or a folder in batch mode
- ✅ Support recursive scanning of subdirectories
- ✅ Output to `CSV` / `SQLite` / `CSV+SQLite`
- ✅ Save results separately:
  - Successful predictions (`predictions`)
  - Skipped/error samples (`issues`)
- ✅ Support TTA (test-time augmentation via flips)
- ✅ Support checkpoint calibrator (toggleable)
- ✅ Support skipping images above a pixel limit (to avoid excessive resource usage on extremely large images)
- ✅ Support custom file extension filters (such as `.jpg .jpeg .jfif`)
- ✅ Automatically append a timestamp if an output file already exists (to avoid overwriting)
- ✅ Compatible with images encoded with various JPEG chroma subsampling/compression settings

---

## Performance

Results on the test set:

- **Number of test samples: 2759**
- **MAE (after calibration): 0.2322**
- **Acc@1 (after calibration): 99.60%**
- **Acc@2 (after calibration): 99.82%**
- **Acc@5 (after calibration): 100.00%**

---

## Inference Speed (Measured)

- **Average: about 6.7 seconds per image** (measured on the test set)
- Hardware platform:
  - **CPU**: AMD Ryzen 7 9700X 8-Core Processor
  - **Memory**: 32GB DDR5 6000MT/s C30
  - **GPU**: NVIDIA Tesla P40

> Actual speed may vary depending on image size, whether TTA is enabled, disk I/O speed, system load, and other factors.

---

## Limitations

- Only applicable to JPEG or similarly compressed images
- QF depends on the encoder implementation (for example, PIL/libjpeg/OpenCV may produce differences)
- After heavy recompression, multiple re-saves, cropping, or filter processing, the meaning of the predicted QF becomes weaker

---

## Project Structure

Recommended repository layout (example):

```text
.
├── infer.py
├── requirements.txt
├── checkpoints/
│   └── model.onnx
│   └── model.meta.json
├── demo/
│   └── test.jpg
├── outputs/
└── README.md
```

---

## Environment Setup

This project can run on CPU, NVIDIA GPU, and AMD GPU environments. To avoid dependency conflicts, please follow the installation steps below carefully.

### 1) Python Environment

Python 3.9+ is recommended.

### 2) Clone the Project and Install Base Dependencies

First, clone the repository and install the general Python dependencies that are independent of hardware:

```bash
git clone https://github.com/sawakage/jpeg-qf-predictor.git
cd jpeg-qf-predictor
pip install -r requirements.txt
```

### 3) Install the Core Deep Learning Libraries (choose 1 of the following 3 according to your hardware)

- Option A: Use an NVIDIA GPU

```bash
# 1. Install PyTorch (CUDA 11.8 is used here as an example; adjust as needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 2. Install the GPU version of ONNX Runtime
pip install onnxruntime-gpu
```

- Option B: Use CPU only

```bash
# 1. Install the CPU version of PyTorch
pip install torch torchvision

# 2. Install the standard version of ONNX Runtime
pip install onnxruntime
```

- Option C: Use an AMD GPU

```bash
# For Windows users (DirectML acceleration):
pip install torch torchvision
pip install onnxruntime-directml

# For Linux users (ROCm architecture, ROCm 5.6 used here as an example):
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm5.6
pip install onnxruntime-rocm
```

### 4) Prepare Model Weights

- The inference script requires both the `.onnx` weight file and the `.meta.json` metadata file, specified via `--ckpt`.

> Make sure the ONNX files are placed in the `checkpoints/` directory under the project root (create it manually if it does not exist).

---

## Complete Usage Workflow

The following sections explain the actual usage flow by scenario.

> Linux/macOS: use `\` for line continuation  
> Windows CMD: use `^` for line continuation  
> Or write the command on a single line directly (no line continuation needed)

### 1) Run Inference on a Single Image

```bash
python infer.py \
  --ckpt ./checkpoints/model.onnx \
  --input ./demo/test.jpg
```

---

### 2) Batch Inference on a Folder

```bash
python infer.py \
  --ckpt ./checkpoints/model.onnx \
  --input ./data/images
```

Default behavior:

- Scan JPEG files under this directory
- **Do not** recurse into subdirectories

---

> The following are additional arguments. Append them after the commands above when needed.

### 3) Recursively Scan Subdirectories

```bash
  --recursive
```

Use case:

- The dataset directory has deep nested folder levels

---

### 4) Output Results to CSV / SQLite / Both

The script supports three output modes:

- `csv` (default)
- `db`
- `both`

#### 4.1 Output CSV Only (Default)

```bash
  --output_mode csv
```

Output files (default):

- `./outputs/predictions.csv`
- `./outputs/issues.csv`

#### 4.2 Output SQLite Database Only

```bash
  --output_mode db
```

Output file (default):

- `./outputs/results.db`

#### 4.3 Output Both CSV + SQLite

```bash
  --output_mode both \
  --pred_csv ./outputs/predictions.csv \
  --issue_csv ./outputs/issues.csv \
  --output_db ./outputs/results.db
```

> If a target output file already exists, the script automatically appends a timestamp to the filename to avoid overwriting the original file.

---

### 5) Control TTA / Calibrator / Device

#### 5.1 Disable TTA (for Speed)

TTA (flip-based aggregation) is enabled by default. If you care more about speed, you can disable it:

```bash
  --disable_tta
```

#### 5.2 Disable the Calibrator

```bash
  --no_calibrator
```

#### 5.3 Specify the Device

- Auto-select (default): `--device auto`
- Force CPU: `--device cpu`
- Force CUDA (NVIDIA GPU): `--device cuda`
- Force DirectML (Windows AMD GPU): `--device dml`
- Force ROCm (Linux AMD GPU): `--device rocm`

If you explicitly specify `cuda`, `dml`, or `rocm` but the corresponding acceleration library is not installed (or the hardware is unavailable), the script will fail strictly and exit. This prevents silently falling back to CPU, which could make inference unexpectedly slow.

---

### 6) Limit the Pixel Count of Extremely Large Images

The script supports a “skip above pixel limit” mechanism for very large images to avoid excessive resource consumption.

#### Enable the Pixel Limit

```bash
  --enable_pixel_limit
```

Default pixel limit:

- 178,956,970 pixels (Pillow's default `MAX_IMAGE_PIXELS`; this may vary by version)

#### Customize the Pixel Limit

```bash
  --enable_pixel_limit \
  --max_image_pixels 100000000
```

Notes:

- This is a **skip mechanism**, not automatic resizing
- Skipped images will be written to `issues.csv` or the `issues` table for later processing

---

### 7) Custom Extension Filtering

By default, only the following extensions are processed:

- `.jpg`
- `.jpeg`
- `.jpe`
- `.jfif`

If you want to expand or restrict the scope:

```bash
  --exts .jpg .jpeg
```

Notes:

- When the input is a single file, the script will raise an error immediately if its extension is not in the allowed list
- When the input is a directory, only files with allowed extensions will be scanned and processed

---

### 8) Quiet Mode (Reduce Log Output)

If you only care about the result files and do not want too much terminal output, use `--quiet`:

```bash
  --quiet
```

---

## Output Description (CSV / DB Fields)

The script saves “successful predictions” and “problematic samples” separately to make subsequent analysis, cleaning, and reruns easier.

### 1) Successful Prediction CSV (`predictions.csv`)

Fields:

- `path`: full path to the image
- `filename`: file name
- `pred_qf_final`: final predicted QF

---

### 2) Problem Sample CSV (`issues.csv`)

Fields:

- `path`
- `filename`
- `issue_type`: `skipped` or `error`
- `issue_message`: skip reason or exception message

---

### 3) SQLite Database (`results.db`)

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

This section does not go into model internals. Instead, it focuses on what the script actually does during inference, which helps explain why some images may be skipped.

### Patch Filtering Strategy

Fixed inference parameters used in the current script:

- Patch size: `64×64`
- Stride: `64`
- Maximum number of patches used per image: `24`
- Variance threshold: `15.0`
- Relaxed threshold: `15.0 / 1.5`

Patch selection logic (in priority order):

1. If the number of high-threshold patches is ≥ 24: take the first 24 high-threshold patches
2. If the number of high-threshold patches is ≥ 5: use all high-threshold patches
3. If the number of relaxed-threshold patches is ≥ 5: take the first 5 relaxed-threshold patches
4. If the number of relaxed-threshold patches is > 0: use all relaxed-threshold patches
5. Otherwise: treat the image as having too little information, skip it, and write it to `issues`

### TTA (Test-Time Augmentation)

Enabled by default. The script applies flip augmentation and averages the results from:

- Original image
- Horizontal flip
- Vertical flip
- Horizontal + vertical flip

### Final Aggregation for a Single Image

- The model outputs predictions for multiple patches
- The script uses the **median** as the final prediction for the whole image (more robust to outlier patches)

## License

Released under the MIT License.

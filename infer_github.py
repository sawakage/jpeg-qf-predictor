import csv
import time
import argparse
import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import torch
from PIL import Image, ImageOps
from tqdm import tqdm

from model import JPEGQFRegNet

# 固定推理参数（与评估脚本一致） / Fixed inference params (same as eval script)
BLOCK_SIZE = 128
STEP = 128
BLOCKS_PER_IMAGE = 24
VAR_THRESHOLD = 15.0
RELAXED_VAR_THRESHOLD = VAR_THRESHOLD / 1.5

# 默认不限制Pillow像素数，由脚本逻辑决定是否跳过 / Disable Pillow pixel limit by default, skip is handled by script logic
Image.MAX_IMAGE_PIXELS = None


class PiecewiseLinearCalibrator:
    def __init__(self, num_bins=12, min_count=30, min_qf=1.0, max_qf=99.0, monotonic=True):
        self.num_bins = num_bins
        self.min_count = min_count
        self.min_qf = float(min_qf)
        self.max_qf = float(max_qf)
        self.monotonic = monotonic
        self.x = np.array([self.min_qf, self.max_qf], dtype=np.float32)
        self.y = np.array([self.min_qf, self.max_qf], dtype=np.float32)

    def load_state_dict(self, sd):
        self.num_bins = int(sd["num_bins"])
        self.min_count = int(sd["min_count"])
        self.min_qf = float(sd["min_qf"])
        self.max_qf = float(sd["max_qf"])
        self.monotonic = bool(sd["monotonic"])
        self.x = np.asarray(sd["x"], dtype=np.float32)
        self.y = np.asarray(sd["y"], dtype=np.float32)
        return self

    def transform(self, pred):
        pred = np.asarray(pred, dtype=np.float32)
        pred = np.clip(pred, self.min_qf, self.max_qf)
        out = np.interp(pred, self.x, self.y)
        return np.clip(out, self.min_qf, self.max_qf).astype(np.float32)


def rgb_variance_score(block):
    r = np.var(block[:, :, 0])
    g = np.var(block[:, :, 1])
    b = np.var(block[:, :, 2])
    gray_var = np.var(np.dot(block[..., :3], [0.2989, 0.5870, 0.1140]))
    return (r + g + b) / 3.0 + gray_var


@torch.no_grad()
def _forward_tta(model, x):
    p0 = model(x)["pred_qf"]
    p1 = model(torch.flip(x, dims=[-1]))["pred_qf"]
    p2 = model(torch.flip(x, dims=[-2]))["pred_qf"]
    p3 = model(torch.flip(x, dims=[-1, -2]))["pred_qf"]
    return (p0 + p1 + p2 + p3) / 4.0


def extract_y_tensor(img_crop):
    """提取Y通道并归一化 / Extract Y channel and normalize"""
    y_img = img_crop.convert("YCbCr").split()[0]
    y_arr = np.array(y_img, dtype=np.float32)
    return torch.from_numpy(y_arr).unsqueeze(0) / 255.0 - 0.5


def process_jpeg_full(
    img_path: str,
    device: torch.device,
    enable_pixel_limit: bool = False,
    max_image_pixels: Optional[int] = None,
):
    img = Image.open(img_path)
    w0, h0 = img.size
    pixel_count = int(w0) * int(h0)

    if enable_pixel_limit and max_image_pixels is not None and pixel_count > int(max_image_pixels):
        return {
            "skip": True,
            "skip_reason": "像素数超过上限 / image pixels exceed limit",
            "strategy": "skip_pixel_limit",
            "width": int(w0),
            "height": int(h0),
            "pixel_count": int(pixel_count),
            "n_high": 0,
            "n_relaxed": 0,
        }

    img = ImageOps.exif_transpose(img)
    img_rgb = img.convert("RGB")
    w, h = img_rgb.size

    if w < BLOCK_SIZE or h < BLOCK_SIZE:
        return {
            "skip": True,
            "skip_reason": "图片过小 / image too small",
            "strategy": "skip_too_small",
            "width": int(w),
            "height": int(h),
            "pixel_count": int(pixel_count),
            "n_high": 0,
            "n_relaxed": 0,
        }

    img_arr = np.array(img_rgb)
    high_candidates = []
    relaxed_candidates = []

    for y in range(0, h - BLOCK_SIZE + 1, STEP):
        for x in range(0, w - BLOCK_SIZE + 1, STEP):
            block = img_arr[y:y + BLOCK_SIZE, x:x + BLOCK_SIZE]
            score = rgb_variance_score(block)
            if score > RELAXED_VAR_THRESHOLD:
                relaxed_candidates.append((score, x, y))
            if score > VAR_THRESHOLD:
                high_candidates.append((score, x, y))

    high_candidates.sort(key=lambda t: t[0], reverse=True)
    relaxed_candidates.sort(key=lambda t: t[0], reverse=True)

    n_high = len(high_candidates)
    n_relaxed = len(relaxed_candidates)

    if n_high >= BLOCKS_PER_IMAGE:
        selected = high_candidates[:BLOCKS_PER_IMAGE]
        strategy = f"high_threshold_top_{BLOCKS_PER_IMAGE}"
    elif n_high >= 5:
        selected = high_candidates
        strategy = "high_threshold_all"
    elif n_relaxed >= 5:
        selected = relaxed_candidates[:5]
        strategy = "relaxed_threshold_top_5"
    elif n_relaxed > 0:
        selected = relaxed_candidates
        strategy = "relaxed_threshold_all_lt_5"
    else:
        return {
            "skip": True,
            "skip_reason": "图片信息过少 / insufficient image information",
            "strategy": "skip_no_relaxed_blocks",
            "width": int(w),
            "height": int(h),
            "pixel_count": int(pixel_count),
            "n_high": int(n_high),
            "n_relaxed": int(n_relaxed),
        }

    tensors = []
    for _, x, y in selected:
        # 不做缩放/重采样/重压缩，仅裁块 / No resize/resample/recompress, crop only
        crop = img_rgb.crop((x, y, x + BLOCK_SIZE, y + BLOCK_SIZE))
        tensors.append(extract_y_tensor(crop))

    return {
        "skip": False,
        "batch_tensors": torch.stack(tensors).to(device),
        "blocks_used": len(selected),
        "width": int(w),
        "height": int(h),
        "pixel_count": int(pixel_count),
        "n_high": int(n_high),
        "n_relaxed": int(n_relaxed),
        "strategy": strategy,
    }


def _autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.amp.autocast("cuda", enabled=True)
    return torch.autocast("cpu", enabled=False)


def load_model_and_calibrator(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_cfg = ckpt.get("cfg", {})

    model = JPEGQFRegNet(
        in_ch=model_cfg.get("in_ch", 4),
        add_hp=model_cfg.get("add_hp", True),
        add_lap=model_cfg.get("add_lap", True),
        add_grid=model_cfg.get("add_grid", True),
    ).to(device)

    if "ema" in ckpt and ckpt["ema"] is not None:
        model.load_state_dict(ckpt["ema"])
        weight_type = "ema"
    else:
        model.load_state_dict(ckpt["model"])
        weight_type = "model"
    model.eval()

    calibrator = None
    if "calibrator" in ckpt and ckpt["calibrator"] is not None:
        calibrator = PiecewiseLinearCalibrator().load_state_dict(ckpt["calibrator"])

    return model, calibrator, model_cfg, weight_type


def predict_one_image(
    img_path: str,
    model,
    device: torch.device,
    disable_tta: bool = False,
    calibrator: Optional[PiecewiseLinearCalibrator] = None,
    use_calibrator: bool = True,
    enable_pixel_limit: bool = False,
    max_image_pixels: Optional[int] = None,
) -> Dict[str, Any]:
    prep = process_jpeg_full(
        img_path=img_path,
        device=device,
        enable_pixel_limit=enable_pixel_limit,
        max_image_pixels=max_image_pixels,
    )

    if prep["skip"]:
        return {
            "path": str(img_path),
            "filename": Path(img_path).name,
            "status": "skipped",
            "skip_reason": prep["skip_reason"],
            "pred_qf_final": None,
        }

    batch_tensors = prep["batch_tensors"]
    with torch.no_grad():
        with _autocast_context(device):
            if not disable_tta:
                pred_batch = _forward_tta(model, batch_tensors)
            else:
                pred_batch = model(batch_tensors)["pred_qf"]
        pred_raw = float(torch.median(pred_batch).item())  # 按图中位数聚合 / Per-image median aggregation

    if use_calibrator and calibrator is not None:
        pred_final = float(calibrator.transform([pred_raw])[0])
    else:
        pred_final = pred_raw

    pred_final = round(pred_final)
    return {
        "path": str(img_path),
        "filename": Path(img_path).name,
        "status": "ok",
        "skip_reason": "",
        "pred_qf_final": float(pred_final),
    }


def collect_images(input_path: str, recursive: bool, exts: List[str]) -> List[str]:
    p = Path(input_path)
    exts_norm = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in exts}

    if p.is_file():
        if p.suffix.lower() not in exts_norm:
            raise ValueError(
                f"输入文件后缀不在允许列表中 / Input file suffix not allowed: {p.suffix}, allowed={sorted(exts_norm)}"
            )
        return [str(p)]

    if not p.is_dir():
        raise FileNotFoundError(f"输入路径不存在或不是文件/文件夹 / Input path is not a file or directory: {input_path}")

    iterator = p.rglob("*") if recursive else p.glob("*")
    files = [str(x) for x in iterator if x.is_file() and x.suffix.lower() in exts_norm]
    files.sort()
    return files


def ensure_parent(path: str):
    parent = Path(path).parent
    if str(parent):
        parent.mkdir(parents=True, exist_ok=True)

def add_timestamp_if_exists(path: str, ts: Optional[str] = None) -> str:
    p = Path(path)
    if not p.exists():
        return path

    ts = ts or time.strftime("%Y%m%d_%H%M%S")
    if p.suffix:
        new_name = f"{p.stem}_{ts}{p.suffix}"
    else:
        new_name = f"{p.name}_{ts}"
    return str(p.with_name(new_name))

def write_pred_csv(rows: List[Dict[str, Any]], csv_path: str):
    ensure_parent(csv_path)
    fieldnames = ["path", "filename", "pred_qf_final"]
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, None) for k in fieldnames})


def write_issue_csv(rows: List[Dict[str, Any]], csv_path: str):
    ensure_parent(csv_path)
    fieldnames = ["path", "filename", "issue_type", "issue_message"]
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, None) for k in fieldnames})


def write_outputs_db(pred_rows: List[Dict[str, Any]], issue_rows: List[Dict[str, Any]], db_path: str):
    """写入SQLite数据库（predictions/issues两表） / Write SQLite DB (predictions/issues tables)"""
    ensure_parent(db_path)
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute("DROP TABLE IF EXISTS predictions")
        cur.execute("DROP TABLE IF EXISTS issues")

        cur.execute("""
            CREATE TABLE predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT,
                filename TEXT,
                pred_qf_final REAL
            )
        """)

        cur.execute("""
            CREATE TABLE issues (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT,
                filename TEXT,
                issue_type TEXT,
                issue_message TEXT
            )
        """)

        if pred_rows:
            cur.executemany(
                "INSERT INTO predictions (path, filename, pred_qf_final) VALUES (?, ?, ?)",
                [(r.get("path"), r.get("filename"), r.get("pred_qf_final")) for r in pred_rows],
            )

        if issue_rows:
            cur.executemany(
                "INSERT INTO issues (path, filename, issue_type, issue_message) VALUES (?, ?, ?, ?)",
                [(r.get("path"), r.get("filename"), r.get("issue_type"), r.get("issue_message")) for r in issue_rows],
            )

        conn.commit()
    finally:
        conn.close()


def build_argparser():
    parser = argparse.ArgumentParser("JPEG QF 实用推理脚本 / JPEG QF Practical Inference")
    parser.add_argument("--ckpt", type=str, required=True, help="模型权重路径(.pth) / Model checkpoint path (.pth)")
    parser.add_argument("--input", type=str, required=True, help="输入图片或文件夹路径 / Input image path or folder path")

    parser.add_argument(
        "--output_mode",
        type=str,
        default="csv",
        choices=["csv", "db", "both"],
        help="输出格式 / Output format: csv | db | both (default: csv)",
    )
    parser.add_argument(
        "--pred_csv",
        type=str,
        default="./outputs/predictions.csv",
        help="成功预测CSV路径 / Success predictions CSV path",
    )
    parser.add_argument(
        "--issue_csv",
        type=str,
        default="./outputs/issues.csv",
        help="问题样本CSV路径（跳过+报错） / Issue samples CSV path (skipped + error)",
    )
    parser.add_argument(
        "--output_db",
        type=str,
        default="./outputs/results.db",
        help="SQLite数据库路径 / SQLite DB path",
    )

    parser.add_argument("--no_calibrator", action="store_true", help="关闭校准器 / Disable calibrator")
    parser.add_argument("--disable_tta", action="store_true", help="关闭TTA / Disable TTA")
    parser.add_argument("--recursive", action="store_true", help="递归搜索子目录 / Recursively scan subfolders")

    parser.add_argument("--enable_pixel_limit", action="store_true", help="启用像素上限跳过 / Enable skip by pixel limit")
    parser.add_argument(
        "--max_image_pixels",
        type=int,
        default=178_956_970,
        help="像素上限（与 --enable_pixel_limit 搭配） / Pixel limit (used with --enable_pixel_limit)",
    )

    parser.add_argument(
        "--exts",
        type=str,
        nargs="+",
        default=[".jpg", ".jpeg", ".jpe", ".jfif"],
        help="允许的后缀名列表 / Allowed file suffixes",
    )
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="推理设备 / Inference device")
    parser.add_argument("--quiet", action="store_true", help="减少终端输出 / Reduce terminal logs")
    return parser


def main():
    args = build_argparser().parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    if args.output_mode in ("csv", "both"):
        args.pred_csv = add_timestamp_if_exists(args.pred_csv, ts)
        args.issue_csv = add_timestamp_if_exists(args.issue_csv, ts)
    if args.output_mode in ("db", "both"):
        args.output_db = add_timestamp_if_exists(args.output_db, ts)

    if args.enable_pixel_limit and args.max_image_pixels <= 0:
        raise ValueError("像素上限必须为正整数 / max_image_pixels must be a positive integer")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("指定了 CUDA 但当前不可用 / CUDA is requested but not available")

    model, calibrator, model_cfg, weight_type = load_model_and_calibrator(args.ckpt, device)
    use_calibrator = not args.no_calibrator
    has_calibrator = calibrator is not None

    image_paths = collect_images(args.input, recursive=args.recursive, exts=args.exts)
    if not image_paths:
        raise RuntimeError("未找到可推理图片 / No valid images found for inference")

    pred_rows: List[Dict[str, Any]] = []
    issue_rows: List[Dict[str, Any]] = []
    ok_count = 0
    skipped_count = 0
    fail_count = 0
    t0_all = time.time()

    iterator = image_paths if (args.quiet or len(image_paths) == 1) else tqdm(image_paths, desc="推理 / Infer")
    for img_path in iterator:
        try:
            row = predict_one_image(
                img_path=img_path,
                model=model,
                device=device,
                disable_tta=args.disable_tta,
                calibrator=calibrator,
                use_calibrator=use_calibrator,
                enable_pixel_limit=args.enable_pixel_limit,
                max_image_pixels=args.max_image_pixels if args.enable_pixel_limit else None,
            )

            if row["status"] == "ok":
                ok_count += 1
                pred_rows.append({
                    "path": row["path"],
                    "filename": row["filename"],
                    "pred_qf_final": row["pred_qf_final"],
                })
            else:
                skipped_count += 1
                issue_rows.append({
                    "path": row["path"],
                    "filename": row["filename"],
                    "issue_type": "skipped",
                    "issue_message": row.get("skip_reason", ""),
                })

        except Exception as e:
            fail_count += 1
            issue_rows.append({
                "path": str(img_path),
                "filename": Path(img_path).name,
                "issue_type": "error",
                "issue_message": repr(e),
            })

    total_sec = time.time() - t0_all

    if args.output_mode in ("csv", "both"):
        write_pred_csv(pred_rows, args.pred_csv)
        write_issue_csv(issue_rows, args.issue_csv)

    if args.output_mode in ("db", "both"):
        write_outputs_db(pred_rows, issue_rows, args.output_db)

    if Path(args.input).is_file():
        print("\n===== 结果 / Result =====")
        if pred_rows:
            print(f"图片 / Image: {pred_rows[0]['path']}")
            print(f"最终QF / Final QF: {pred_rows[0]['pred_qf_final']:.4f}")
        elif issue_rows:
            print(f"图片 / Image: {issue_rows[0]['path']}")
            print(f"问题类型 / Issue type: {issue_rows[0]['issue_type']}")
            print(f"信息 / Message: {issue_rows[0]['issue_message']}")
    else:
        print("\n===== 汇总 / Summary =====")
        print(f"总数 / Found: {len(image_paths)} | 成功 / OK: {ok_count} | 跳过 / Skipped: {skipped_count} | 错误 / Error: {fail_count}")
        print(f"总耗时 / Total time: {total_sec:.3f}s")

    if args.output_mode in ("csv", "both"):
        print(f"[+] 成功预测CSV / Predictions CSV: {args.pred_csv}")
        print(f"[+] 问题样本CSV / Issues CSV: {args.issue_csv}")
    if args.output_mode in ("db", "both"):
        print(f"[+] 数据库文件 / DB file: {args.output_db}")

    if not args.quiet:
        print(f"[*] 设备 / Device: {device}")
        print(f"[*] 权重类型 / Weight type: {weight_type}")
        print(f"[*] 校准器 / Calibrator: {'enabled' if (use_calibrator and has_calibrator) else 'disabled/fallback'}")
        print(f"[*] 像素上限 / Pixel limit: {'enabled' if args.enable_pixel_limit else 'disabled'}"
              f"{f', max={args.max_image_pixels}' if args.enable_pixel_limit else ''}")



if __name__ == "__main__":
    main()
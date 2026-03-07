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

import json

try:
    import onnxruntime as ort
except ImportError as e:
    raise ImportError(
        "onnxruntime is required for ONNX inference. Install via: pip install onnxruntime (CPU) or onnxruntime-gpu (CUDA)"
    ) from e

# 固定推理参数（与评估脚本一致） / Fixed inference params (same as eval script)
BLOCK_SIZE = 64
STEP = 64
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


class OnnxModel:
    def __init__(self, onnx_path: str, device: torch.device, req_dev: str = "auto"):    
        self.onnx_path = str(onnx_path)
        self.device = device
        
        available = set(ort.get_available_providers())
        providers = ["CPUExecutionProvider"]
        
        if req_dev == "cuda":
            if "CUDAExecutionProvider" not in available:
                raise RuntimeError("未检测到 CUDA 支持！请检查是否安装了 onnxruntime-gpu / CUDA support not detected! Please check if onnxruntime-gpu is installed")
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif req_dev == "rocm":
            if "ROCMExecutionProvider" not in available:
                raise RuntimeError("未检测到 ROCm 支持！请检查是否安装了 onnxruntime-rocm / ROCm support not detected! Please check if onnxruntime-rocm is installed")
            providers = ["ROCMExecutionProvider", "CPUExecutionProvider"]
        elif req_dev == "dml":
            if "DmlExecutionProvider" not in available:
                raise RuntimeError("未检测到 DirectML 支持！请检查是否安装了 onnxruntime-directml / DirectML support not detected! Please check if onnxruntime-directml is installed")
            providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
        elif req_dev == "cpu":
            providers = ["CPUExecutionProvider"]
        else:
            if "CUDAExecutionProvider" in available:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            elif "ROCMExecutionProvider" in available:
                providers = ["ROCMExecutionProvider", "CPUExecutionProvider"]
            elif "DmlExecutionProvider" in available:
                providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
            
        self.sess = ort.InferenceSession(self.onnx_path, providers=providers)
        self.input_name = self.sess.get_inputs()[0].name
        self.output_names = [o.name for o in self.sess.get_outputs()]
    
        active_eps = self.sess.get_providers()
        self.is_cuda = "CUDAExecutionProvider" in active_eps
        self.is_rocm = "ROCMExecutionProvider" in active_eps

    def __call__(self, x):
        if not (self.is_cuda or self.is_rocm) or not isinstance(x, torch.Tensor) or not x.is_cuda:
            return self._forward_cpu_fallback(x)
            
        x = x.contiguous()
        if x.dtype != torch.float32:
            x = x.to(torch.float32)

        io_binding = self.sess.io_binding()
        device_id = x.device.index if x.device.index is not None else 0

        ort_device_type = 'rocm' if self.is_rocm else 'cuda'

        io_binding.bind_input(
            name=self.input_name,
            device_type=ort_device_type,
            device_id=device_id,
            element_type=np.float32,
            shape=tuple(x.shape),
            buffer_ptr=x.data_ptr()
        )

        out = {}
        B = x.shape[0]
        
        for output_meta in self.sess.get_outputs():
            name = output_meta.name
            
            shape = []
            if output_meta.shape is not None:
                for dim in output_meta.shape:
                    if isinstance(dim, str) or dim is None:
                        shape.append(B)
                    else:
                        shape.append(dim)
            else:
                shape = [B]

            out_tensor = torch.empty(tuple(shape), dtype=torch.float32, device=x.device)
            out[name] = out_tensor

            io_binding.bind_output(
                name=name,
                device_type=ort_device_type,
                device_id=device_id,
                element_type=np.float32,
                shape=tuple(shape),
                buffer_ptr=out_tensor.data_ptr()
            )

        self.sess.run_with_iobinding(io_binding)
        out_names = [o.name for o in self.sess.get_outputs()]
        if "pred_qf" not in out and len(out_names) >= 1:
            out["pred_qf"] = out[out_names[0]]
        if "pred_sigma" not in out and len(out_names) >= 2:
            out["pred_sigma"] = out[out_names[1]]

        return out

    def _forward_cpu_fallback(self, x):
        if isinstance(x, torch.Tensor):
            x_np = x.detach().cpu().numpy()
        else:
            x_np = np.asarray(x)

        if x_np.dtype != np.float32:
            x_np = x_np.astype(np.float32)

        outs = self.sess.run(None, {self.input_name: x_np})

        out = {}
        for name, arr in zip(self.output_names, outs):
            out[name] = torch.from_numpy(arr).to(self.device)

        if "pred_qf" not in out and len(outs) >= 1:
            out["pred_qf"] = torch.from_numpy(outs[0]).to(self.device)
        if "pred_sigma" not in out and len(outs) >= 2:
            out["pred_sigma"] = torch.from_numpy(outs[1]).to(self.device)

        return out

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


def load_model_and_calibrator(ckpt_path: str, device: torch.device, no_calibrator: bool = False, req_dev: str = "auto"):
    onnx_path = Path(ckpt_path)
    if onnx_path.suffix.lower() != ".onnx":
        raise ValueError(f"仅支持输入ONNX文件 / Only .onnx is supported: {onnx_path}")

    meta_path = onnx_path.with_suffix(".meta.json")

    model = OnnxModel(str(onnx_path), device=device, req_dev=req_dev)

    calibrator = None
    model_cfg = {}
    weight_type = "onnx"

    if not meta_path.exists():
        if not no_calibrator:
            raise FileNotFoundError(
                f"缺少meta文件 / meta file not found: {meta_path}\n"
                "提示：如果不需要校准器，请在命令中添加 --no_calibrator 参数跳过此检查。\n"
                "Tip: If you do not need a calibrator, add the --no_calibrator parameter to the command to skip this check."
            )
    else:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        model_cfg = meta.get("cfg", {})
        
        if not no_calibrator and "calibrator" in meta and meta["calibrator"] is not None:
            calibrator = PiecewiseLinearCalibrator().load_state_dict(meta["calibrator"])
        weight_type = meta.get("weight_type", "onnx")

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
    parser.add_argument("--ckpt", type=str, required=True, help="模型权重路径(.onnx) / Model checkpoint path (.onnx)")
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
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "dml", "rocm"], help="推理设备 / Inference device")
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
    elif args.device in ["dml", "cpu"]:
        device = torch.device("cpu")
    elif args.device in ["cuda", "rocm"]:
        device = torch.device("cuda")
        if not torch.cuda.is_available():
            raise RuntimeError(f"指定了 {args.device}，但 PyTorch 未检测到对应的 GPU / {args.device} is requested but not available in PyTorch")

    model, calibrator, _, weight_type = load_model_and_calibrator(
        args.ckpt, 
        device, 
        no_calibrator=args.no_calibrator,
        req_dev=args.device
    )
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
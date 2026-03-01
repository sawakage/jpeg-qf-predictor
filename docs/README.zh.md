> Language / 语言 / 言語: [English](../README.md) | [中文](./README.zh.md) | [日本語](./README.jp.md)

# JPEG QF 预测 (回归)

一个用于预测 JPEG 质量因子（QF, Quality Factor）的 **回归模型**推理工具。  

---

## 目录 <!-- omit from toc -->

- [功能特性](#功能特性)
- [性能](#性能)
- [推理速度（实测）](#推理速度实测)
- [限制说明](#限制说明)
- [项目结构](#项目结构)
- [环境准备](#环境准备)
- [快速开始（单图模式）](#快速开始单图模式)
- [完整使用流程](#完整使用流程)
- [命令行参数帮助信息](#命令行参数帮助信息)
- [输出结果说明（CSV / DB 字段）](#输出结果说明csv--db-字段)
- [推理流程说明（选块逻辑）](#推理流程说明选块逻辑)
- [跳过与错误处理机制](#跳过与错误处理机制)
- [许可证](#许可证)


---

## 功能特性

- ✅ 单张 JPEG 图片推理
- ✅ 文件夹批量推理
- ✅ 支持递归扫描子目录
- ✅ 输出 `CSV` / `SQLite` / `CSV+SQLite`
- ✅ 分离保存：
  - 成功预测结果（`predictions`）
  - 跳过/错误样本（`issues`）
- ✅ 支持 TTA（翻转测试增强）
- ✅ 支持 checkpoint 内置校准器（可开关）
- ✅ 支持自动选择 `cuda/cpu`
- ✅ 支持像素上限跳过（避免超大图导致资源占用过高）
- ✅ 支持自定义文件后缀过滤（如 `.jpg .jpeg .jfif`）
- ✅ 输出文件已存在时自动追加时间戳（避免覆盖）
- ✅ 适配jpeg各种色度采样压缩下的图片

---

## 性能

测试集结果：

- **测试集样本数：2763**
- **MAE（校准后）：0.1541**
- **Acc@1（校准后）：99.78%**
- **Acc@2（校准后）：99.86%**
- **Acc@5（校准后）：99.93%**

---

## 推理速度（实测）

- **平均约 3 秒 / 张**（测试集推理实测）
- 硬件平台（**笔记本平台**）：
  - **CPU**: Intel i7-12700H
  - **内存**: 16GB
  - **GPU**: RTX 3060 Laptop GPU

> 实际速度会受图片尺寸、是否开启 TTA、磁盘读写速度、系统负载等因素影响。

---
## 限制说明

- 仅适用于 JPEG 或同类型压缩图像
- QF 与编码器实现相关（PIL/libjpeg/OpenCV 等可能存在差异）
- 重压缩、多次保存、裁剪/滤镜处理后，QF 预测含义会变弱

---
## 项目结构

建议仓库结构（示例）：

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

## 环境准备

### 1) Python 环境
建议使用 Python 3.9+。

### 2) 安装依赖（使用 `requirements.txt`）

```bash
pip install -r requirements.txt
```

> 如果你的 `requirements.txt` 未包含与你本机 CUDA 匹配的 PyTorch，请先按官方方式安装 PyTorch，再执行上面的命令。

### 3) 准备模型权重
- 推理脚本需要通过 `--ckpt` 指定 `.pth` 权重文件。
- 模型权重通过 GitHub Releases 分发，不再使用 Git LFS 管理。

#### 自动下载（推荐）
运行项目提供的下载脚本，它会自动检测当前 Git 标签（版本）并下载对应的模型文件到 `checkpoints/` 目录：
```bash
# 安装依赖（如果尚未安装 requests）
pip install requests

python scripts/download_model.py
```

脚本执行后，模型文件将保存为 `checkpoints/model.pth`，推理时可直接使用默认路径或通过 `--ckpt` 指定。

#### 手动下载

你也可以从 `Releases` 页面手动下载对应版本的 `model.pth` 文件，并将其放置于项目根目录下的 `checkpoints/` 文件夹中（如不存在请自行创建）。

---

## 快速开始（单图模式）

可用于检测脚本是否能成功运行

```bash
python infer.py --ckpt ./checkpoints/model.pth --input ./demo/test.jpg
```

如果成功，你会看到类似输出（单图模式）：

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

## 完整使用流程

下面按实际使用场景详细说明。

> Linux/macOS：使用 \ 换行 \
> Windows CMD：使用 ^ 换行 \
> 或直接写成单行命令（不需要换行符）

### 1) 单张图片推理

单图用法同快速开始

适用场景：
- 调试设备 / 路径 / 权限问题
- 做小规模人工核对

---

### 2) 文件夹批量推理

```bash
python infer.py \
  --ckpt ./checkpoints/model.pth \
  --input ./data/images
```

默认行为：
- 扫描该目录下的 JPEG 文件
- **不递归**子目录

---

> 以下为追加参数，使用时在上述命令之后追加

### 3) 递归扫描子目录

```bash
  --recursive
```

适用场景：
- 数据集目录层级较深
- 图片按类别/来源分文件夹管理

---

### 4) 结果输出为 CSV / SQLite / 两者同时

脚本支持三种输出模式：

- `csv`（默认）
- `db`
- `both`

#### 4.1 仅输出 CSV（默认）

```bash
  --output_mode csv
```

输出文件（默认）：
- `./outputs/predictions.csv`
- `./outputs/issues.csv`

#### 4.2 仅输出 SQLite 数据库

```bash
  --output_mode db
```

输出文件（默认）：
- `./outputs/results.db`

#### 4.3 同时输出 CSV + SQLite

```bash
  --output_mode both \
  --pred_csv ./outputs/predictions.csv \
  --issue_csv ./outputs/issues.csv \
  --output_db ./outputs/results.db
```
**输出路径参数为可选项**
> 如果目标输出文件已存在，脚本会自动在文件名后追加时间戳，避免覆盖原文件。

---

### 5) 控制 TTA / 校准器 / 设备

#### 5.1 关闭 TTA（加速）
默认启用 TTA（翻转增强聚合）。如果更关注速度，可关闭：

```bash
  --disable_tta
```

#### 5.2 关闭校准器（若 checkpoint 含校准器）

```bash
  --no_calibrator
```

#### 5.3 指定设备

- 自动选择（默认）：`--device auto`
- 强制 CPU：`--device cpu`
- 强制 CUDA：`--device cuda`

如果指定 `cuda` 但当前环境不可用，脚本会报错并退出（预期行为）。

---

### 6) 限制超大图像像素数

脚本支持对超大图做“像素上限跳过”，避免极端大图占用过多资源。

#### 启用像素上限

```bash
  --enable_pixel_limit
```
默认像素上限：
- 178,956,970 像素（Pillow 默认 MAX_IMAGE_PIXELS，不同版本可能存在差异）
#### 自定义像素上限

```bash
  --enable_pixel_limit \
  --max_image_pixels 100000000
```

说明：
- 这是 **跳过机制**，不是自动缩放
- 被跳过的图片会写入 `issues.csv` 或 `issues` 表中，便于后续处理

---

### 7) 自定义后缀过滤

默认只处理以下后缀：

- `.jpg`
- `.jpeg`
- `.jpe`
- `.jfif`

如果你想扩展或限制处理范围：


```bash
  --exts .jpg .jpeg
```

说明：
- 输入为单文件时，如果后缀不在允许列表，会直接报错
- 输入为目录时，只会扫描并处理允许后缀的文件

---

### 8) 静默模式（减少日志输出）

当你只关心结果文件，不希望终端输出过多信息时可使用 `--quiet`：

静默模式下：
- 会减少终端日志输出
- 多图模式通常不显示进度条
- 仍会正常写出 CSV / DB 结果

---

## 命令行参数帮助信息

可以直接运行以下命令查看脚本帮助信息：

```bash
python infer.py --help
```
---

## 输出结果说明（CSV / DB 字段）

脚本会把“成功预测”和“问题样本”分开保存，便于后续分析、清洗和重跑。

### 1) 成功预测 CSV（`predictions.csv`）

字段：
- `path`：图片完整路径
- `filename`：文件名
- `pred_qf_final`：最终预测 QF

---

### 2) 问题样本 CSV（`issues.csv`）

字段：
- `path`
- `filename`
- `issue_type`：`skipped` 或 `error`
- `issue_message`：跳过原因或异常信息

---

### 3) SQLite 数据库（`results.db`）

数据库包含两张表：

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

## 推理流程说明（选块逻辑）

这部分不展开模型细节，重点说明脚本在实际推理时做了什么，方便理解为什么有些图片会被跳过。

### 图块筛选策略

当前脚本中的固定推理参数：

- 图块大小：`128×128`
- 步长：`128`
- 每图最多使用图块数：`24`
- 方差阈值：`15.0`
- 宽松阈值：`15.0 / 1.5`

图块选择逻辑（按优先级）：
1. 高阈值图块数 ≥ 24：取高阈值图块前 24 个
2. 高阈值图块数 ≥ 5：使用全部高阈值图块
3. 宽松阈值图块数 ≥ 5：取宽松阈值图块前 5 个
4. 宽松阈值图块数 > 0：使用全部宽松阈值图块
5. 否则：判定图片信息过少，跳过并写入 `issues`

### TTA（测试时增强）

默认启用，脚本使用翻转增强后取平均：
- 原图
- 水平翻转
- 垂直翻转
- 水平 + 垂直翻转

### 单图最终结果聚合

- 模型对多个图块输出预测值
- 脚本以**中位数**作为整图预测结果（更抗异常块）

---

## 跳过与错误处理机制

脚本不会因为单张图片失败就中断整个批处理流程，而是将问题记录到 `issues`，然后继续处理后续图片。

### 常见跳过原因（`issue_type=skipped`）

- 图片像素数超过上限（启用像素限制时）
- 图片尺寸小于图块尺寸（`128×128`）
- 图片内容信息量过少（选不到有效图块）

### 常见错误（`issue_type=error`）

- 文件损坏 / 截断 / 解码失败
- 非法或不完整 JPEG
- 权重与模型结构不匹配
- 指定 `--device cuda` 但当前环境 CUDA 不可用
- 输入单文件时后缀不在允许列表（此时会直接报错，不进入批处理）

## 许可证
使用MIT许可证
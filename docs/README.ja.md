> Language / 语言 / 言語: [English](../README.md) | [中文](./README.zh.md) | [日本語](./README.ja.md)

# JPEG QF 推定（回帰）

**アニメ調 / 二次元画像**の JPEG Quality Factor（QF）を推定するための、**回帰モデルベースの推論ツール**です。

---

## 目次 <!-- omit from toc -->

- [主な機能](#主な機能)
- [性能](#性能)
- [推論速度（実測）](#推論速度実測)
- [制限事項](#制限事項)
- [プロジェクト構成](#プロジェクト構成)
- [環境構築](#環境構築)
- [完全な使用手順](#完全な使用手順)
- [出力結果の説明（CSV / DB フィールド）](#出力結果の説明csv--db-フィールド)
- [推論フローの説明（パッチ選択ロジック）](#推論フローの説明パッチ選択ロジック)
- [ライセンス](#ライセンス)

---

## 主な機能

- ✅ 単一 JPEG 画像およびフォルダ単位の一括推論に対応
- ✅ サブディレクトリの再帰走査に対応
- ✅ `CSV` / `SQLite` / `CSV+SQLite` で出力可能
- ✅ 結果を分離して保存:
  - 正常に予測できた結果（`predictions`）
  - スキップ / エラーになったサンプル（`issues`）
- ✅ TTA（テスト時拡張、反転ベース）に対応
- ✅ チェックポイント用キャリブレータに対応（ON/OFF 可能）
- ✅ ピクセル数上限によるスキップに対応（極端に大きい画像による過剰なリソース消費を回避）
- ✅ カスタム拡張子フィルタに対応（例: `.jpg .jpeg .jfif`）
- ✅ 出力ファイルが既に存在する場合は自動でタイムスタンプを付与（上書き防止）
- ✅ さまざまな JPEG 色差サブサンプリング / 圧縮設定の画像に対応

---

## 性能

テストセットでの結果:

- **テストサンプル数: 2759**
- **MAE（キャリブレーション後）: 0.2322**
- **Acc@1（キャリブレーション後）: 99.60%**
- **Acc@2（キャリブレーション後）: 99.82%**
- **Acc@5（キャリブレーション後）: 100.00%**

---

## 推論速度（実測）

- **平均約 6.7 秒 / 枚**（テストセットでの実測）
- ハードウェア構成:
  - **CPU**: AMD Ryzen 7 9700X 8-Core Processor
  - **メモリ**: 32GB DDR5 6000MT/s C30
  - **GPU**: NVIDIA Tesla P40

> 実際の速度は、画像サイズ、TTA の有無、ディスク I/O 速度、システム負荷などの要因によって変動します。

---

## 制限事項

- JPEG または同種の圧縮画像にのみ適用可能です
- QF はエンコーダ実装に依存します（例: PIL / libjpeg / OpenCV などで差異が生じる場合があります）
- 強い再圧縮、複数回保存、トリミング、フィルタ処理などが行われた後は、推定された QF の意味合いが弱くなります

---

## プロジェクト構成

推奨リポジトリ構成（例）:

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

## 環境構築

本プロジェクトは CPU、NVIDIA GPU、AMD GPU 環境で実行できます。依存関係の衝突を避けるため、以下の手順に従ってインストールしてください。

### 1) Python 環境

Python 3.9 以上を推奨します。

### 2) プロジェクトをクローンし、基本依存関係をインストール

まず、リポジトリをクローンし、ハードウェアに依存しない共通の Python ライブラリをインストールします。

```bash
git clone https://github.com/sawakage/jpeg-qf-predictor.git
cd jpeg-qf-predictor
pip install -r requirements.txt
```

### 3) 深層学習のコアライブラリをインストール（環境に応じて 3 つから 1 つ選択）

- オプション A: NVIDIA GPU を使用する場合

```bash
# 1. PyTorch をインストール（ここでは CUDA 11.8 を例にしています。環境に応じて調整してください）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 2. ONNX Runtime GPU 版をインストール
pip install onnxruntime-gpu
```

- オプション B: CPU のみを使用する場合

```bash
# 1. PyTorch CPU 版をインストール
pip install torch torchvision

# 2. ONNX Runtime 標準版をインストール
pip install onnxruntime
```

- オプション C: AMD GPU を使用する場合

```bash
# Windows ユーザー（DirectML アクセラレーション）:
pip install torch torchvision
pip install onnxruntime-directml

# Linux ユーザー（ROCm アーキテクチャ。ここでは ROCm 5.6 を例にしています）:
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm5.6
pip install onnxruntime-rocm
```

### 4) モデル重みを準備

- 推論スクリプトでは、`.onnx` 重みファイルと `.meta.json` メタデータファイルの両方を `--ckpt` で指定する必要があります。

> ONNX ファイルは、プロジェクトルート配下の `checkpoints/` ディレクトリに配置してください（存在しない場合は手動で作成してください）。

---

## 完全な使用手順

以下では、実際の利用シナリオごとに使用手順を説明します。

> Linux/macOS: 改行継続には `\` を使用  
> Windows CMD: 改行継続には `^` を使用  
> または、改行せず 1 行のコマンドとしてそのまま記述しても構いません

### 1) 単一画像を推論する

```bash
python infer.py \
  --ckpt ./checkpoints/model.onnx \
  --input ./demo/test.jpg
```

---

### 2) フォルダを一括推論する

```bash
python infer.py \
  --ckpt ./checkpoints/model.onnx \
  --input ./data/images
```

デフォルト動作:

- このディレクトリ配下の JPEG ファイルを走査します
- サブディレクトリは**再帰的に走査しません**

---

> 以下は追加オプションです。必要に応じて、上記コマンドの後ろに追記してください。

### 3) サブディレクトリを再帰的に走査する

```bash
  --recursive
```

利用シーン:

- データセットのディレクトリ階層が深い場合

---

### 4) 結果を CSV / SQLite / 両方で出力する

スクリプトは次の 3 つの出力モードをサポートします。

- `csv`（デフォルト）
- `db`
- `both`

#### 4.1 CSV のみ出力（デフォルト）

```bash
  --output_mode csv
```

出力ファイル（デフォルト）:

- `./outputs/predictions.csv`
- `./outputs/issues.csv`

#### 4.2 SQLite データベースのみ出力

```bash
  --output_mode db
```

出力ファイル（デフォルト）:

- `./outputs/results.db`

#### 4.3 CSV + SQLite を同時に出力

```bash
  --output_mode both \
  --pred_csv ./outputs/predictions.csv \
  --issue_csv ./outputs/issues.csv \
  --output_db ./outputs/results.db
```

> 出力先ファイルが既に存在する場合、元ファイルを上書きしないように、スクリプトが自動でファイル名にタイムスタンプを付与します。

---

### 5) TTA / キャリブレータ / デバイスを制御する

#### 5.1 TTA を無効化する（高速化）

デフォルトでは TTA（反転拡張の集約）が有効です。速度を優先したい場合は無効化できます。

```bash
  --disable_tta
```

#### 5.2 キャリブレータを無効化する

```bash
  --no_calibrator
```

#### 5.3 デバイスを指定する

- 自動選択（デフォルト）: `--device auto`
- CPU を強制: `--device cpu`
- CUDA を強制（NVIDIA GPU）: `--device cuda`
- DirectML を強制（Windows AMD GPU）: `--device dml`
- ROCm を強制（Linux AMD GPU）: `--device rocm`

`cuda`、`dml`、`rocm` を明示的に指定したにもかかわらず、対応するアクセラレーションライブラリがインストールされていない、またはハードウェアが利用できない場合、スクリプトは厳密にエラーで終了します。これにより、環境不足時に CPU へ暗黙フォールバックして推論が極端に遅くなることを防げます。

---

### 6) 極端に大きい画像のピクセル数を制限する

スクリプトは、極端に大きな画像による過剰なリソース消費を避けるために、「ピクセル数上限を超えたらスキップする」仕組みに対応しています。

#### ピクセル上限を有効にする

```bash
  --enable_pixel_limit
```

デフォルトのピクセル上限:

- 178,956,970 ピクセル（Pillow のデフォルト `MAX_IMAGE_PIXELS`。バージョンによって異なる場合があります）

#### ピクセル上限をカスタマイズする

```bash
  --enable_pixel_limit \
  --max_image_pixels 100000000
```

補足:

- これは**スキップ機構**であり、自動リサイズではありません
- スキップされた画像は、後で処理できるように `issues.csv` または `issues` テーブルへ記録されます

---

### 7) 拡張子フィルタをカスタマイズする

デフォルトでは、次の拡張子のみを処理します。

- `.jpg`
- `.jpeg`
- `.jpe`
- `.jfif`

処理対象を拡張または制限したい場合:

```bash
  --exts .jpg .jpeg
```

補足:

- 入力が単一ファイルの場合、拡張子が許可リストに含まれていなければ即座にエラーになります
- 入力がディレクトリの場合、許可された拡張子のファイルのみを走査して処理します

---

### 8) Quiet モード（ログ出力を減らす）

結果ファイルだけが必要で、端末への出力をできるだけ減らしたい場合は `--quiet` を使用します。

```bash
  --quiet
```

---

## 出力結果の説明（CSV / DB フィールド）

スクリプトは「正常な予測結果」と「問題のあるサンプル」を分けて保存するため、その後の分析、クリーニング、再実行がしやすくなっています。

### 1) 正常予測 CSV（`predictions.csv`）

フィールド:

- `path`: 画像のフルパス
- `filename`: ファイル名
- `pred_qf_final`: 最終予測 QF

---

### 2) 問題サンプル CSV（`issues.csv`）

フィールド:

- `path`
- `filename`
- `issue_type`: `skipped` または `error`
- `issue_message`: スキップ理由または例外メッセージ

---

### 3) SQLite データベース（`results.db`）

データベースには 2 つのテーブルが含まれます。

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

## 推論フローの説明（パッチ選択ロジック）

このセクションではモデル内部の詳細には踏み込みません。代わりに、推論時にスクリプトが実際に何をしているのかに焦点を当て、なぜ一部の画像がスキップされるのかを理解しやすくしています。

### パッチ選別戦略

現在のスクリプトで固定されている推論パラメータ:

- パッチサイズ: `64×64`
- ストライド: `64`
- 1 画像あたりに使用する最大パッチ数: `24`
- 分散しきい値: `15.0`
- 緩和しきい値: `15.0 / 1.5`

パッチ選択ロジック（優先順）:

1. 高しきい値パッチ数が 24 以上: 高しきい値パッチの先頭 24 個を使用
2. 高しきい値パッチ数が 5 以上: 高しきい値パッチをすべて使用
3. 緩和しきい値パッチ数が 5 以上: 緩和しきい値パッチの先頭 5 個を使用
4. 緩和しきい値パッチ数が 0 より大きい: 緩和しきい値パッチをすべて使用
5. それ以外: 画像情報が少なすぎると判定し、スキップして `issues` に記録

### TTA（テスト時拡張）

デフォルトで有効です。スクリプトは反転拡張を適用し、次の結果を平均化します。

- 元画像
- 水平反転
- 垂直反転
- 水平 + 垂直反転

### 単一画像の最終結果集約

- モデルは複数パッチに対して予測値を出力します
- スクリプトは画像全体の最終予測として**中央値**を採用します（外れ値パッチに対してより頑健）

## ライセンス

MIT ライセンスで提供されています。

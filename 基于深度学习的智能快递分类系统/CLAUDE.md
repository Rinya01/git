# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**基于深度学习的智能快递分类系统** — A PyQt5 desktop application for intelligent package classification and waybill (面单) information extraction using YOLOv5 + PaddleOCR.

### Architecture

The system uses a **two-stage detection pipeline**:

1. **Stage 1 — Package Classification** (`PACKAGE_WEIGHTS`): YOLOv5s model trained on 3 classes (`kuaididai`/快递袋, `zhixiang`/纸箱, `miandan`/面单)
2. **Stage 2 — Waybill Detection** (`WAYBILL_WEIGHTS`): YOLOv5s model trained on 1 class (`waybill`) to locate shipping labels
3. **OCR Extraction**: PaddleOCR extracts structured fields (name, phone, address, tracking number) from cropped waybill images

### Key Custom Modules (on top of YOLOv5 base)

| File | Purpose |
|------|---------|
| `config.py` | Centralized configuration — model weights paths, PaddleOCR model paths, DB path, UI resources, inference thresholds, OCR keywords |
| `window.py` | Main PyQt5 application — login dialog, main window with tabs (image detection, video detection, history, admin panels) |
| `ocr_worker.py` | PaddleOCR wrapper — supports both one-shot CLI mode (`python ocr_worker.py <image_path>`) and persistent server mode (`--serve`) for process reuse |
| `db.py` | SQLite database — user auth (with werkzeug password hashing), recognition logs, login logs, admin user management |
| `ui_style.py` | UI theme — QSS stylesheet, color constants, widget factories (chips, stat tiles, nav icons, message boxes) |
| `augment_data_fixed.py` | Offline data augmentation script using albumentations (standalone utility) |

### YOLOv5 Base Structure (from Ultralytics)

- `models/` — YOLOv5 model definitions (`yolo.py`, `common.py`, `experimental.py`, `tf.py`)
- `utils/` — Utilities: datasets, loss, metrics, plots, augmentations, torch utils, loggers
- `train.py` — Model training script
- `val.py` — Model validation script
- `detect.py` — Standalone inference script (CLI)
- `export.py` — Model export (TorchScript, ONNX, etc.)
- `data/` — Dataset configs (`.yaml`), hyperparameters (`data/hyps/`)
- `models/hub/` — Various YOLOv5 architecture configs

## Running the Application

```bash
cd 基于深度学习的智能快递分类系统
python window.py
```

Default admin account: `admin` / `admin123` (created automatically on first launch, should be changed after login)

## Training Models

### Package Classification Model
```bash
python train.py --data data/package_data.yaml --cfg models/yolov5s.yaml --weights pretrained/yolov5s.pt --batch-size 4 --epochs 100
```

### Waybill Detection Model
```bash
python train.py --data data/data.yaml --cfg models/yolov5s.yaml --weights yolov5s.pt --batch-size 8 --epochs 30
```

## Dependencies

From `requirements.txt`:
- **Core**: `torch`, `torchvision`, `numpy`, `opencv-python`, `Pillow`, `PyYAML`
- **UI**: `PyQt5` (not in requirements.txt but required)
- **OCR**: `paddleocr`, `paddlepaddle` (not in requirements.txt but required)
- **DB**: `werkzeug` (for password hashing)
- **Other**: `matplotlib`, `tqdm`, `tensorboard`, `pandas`, `seaborn`, `thop`, `scipy`, `requests`

### External Dependencies

- **PaddleOCR models** must be placed at `E:\paddle_ocr_models\` (configured in `config.py`):
  - `ch_PP-OCRv4_det_infer/`
  - `ch_PP-OCRv4_rec_infer/`
  - `ch_ppocr_mobile_v2.0_cls_infer/`
- **Temp directory**: `E:\ocr_temp\` (for intermediate images — avoids Chinese path issues with cv2/PaddleOCR on Windows)

## Trained Weights

| Model | Path | Training Config |
|-------|------|-----------------|
| Package classification | `runs/train/exp3/weights/best.pt` | `exp3/opt.yaml` — 100 epochs, batch 4, from `pretrained/yolov5s.pt` |
| Waybill detection | `runs/train/waybill_detect4/weights/best.pt` | `waybill_detect4/opt.yaml` — 30 epochs, batch 8, freeze=10 |

Pre-trained COCO weights available in `pretrained/`: `yolov5s.pt`, `yolov5m.pt`, `yolov5l.pt`

## Dataset Structure

Datasets should be placed in `datasets/` with YOLO format:

```
datasets/
  package/
    images/train/  images/val/
    labels/train/  labels/val/
  waybill/
    images/train/  images/val/
    labels/train/  labels/val/
```

Dataset configs: `data/package_data.yaml` (3 classes), `data/data.yaml` (1 class: waybill)

## Configuration Reference

All tunable parameters are centralized in `config.py`:

- **Model paths**: `PACKAGE_WEIGHTS`, `WAYBILL_WEIGHTS`
- **Inference params**: `IMG_SIZE`, `MAX_DET`, confidence/IoU thresholds for both models
- **OCR params**: `OCR_CONF_THRES`, `OCR_MIN_SIDE`, `ADDRESS_MERGE_Y_THRESHOLD`, keyword lists for name/phone/tracking/address
- **UI resources**: `UI_DIR`, icon/placeholder image paths

## Admin Features

Admin users (`role='admin'`) get two extra tabs:
- **User Management**: View all users, edit username/password, delete users
- **Login Logs**: View all login attempts (success/failure) with filtering

## Database Schema (SQLite at `database/app_data.db`)

- `user` — id, username, password (werkzeug hash), role, reg_time, last_login
- `recognition_log` — id, user_id, rec_type, package_type, name, phone, address, tracking_no, rec_time
- `login_log` — id, user_id, username, success, login_time

## Important Notes

- **Windows Chinese path issues**: OpenCV and PaddleOCR cannot handle paths with Chinese characters on Windows. Temp files and PaddleOCR models are placed outside the project at `E:\ocr_temp\` and `E:\paddle_ocr_models\` respectively.
- **OCR persistent service**: `window.py` spawns `ocr_worker.py --serve` as a subprocess. PaddleOCR models load once and the process is reused via stdin/stdout protocol (`line-based path → JSON result`). Crash auto-restart is built in.
- **PyTorch + PaddleOCR incompatibility**: They cannot coexist in the same process, hence the subprocess architecture for OCR.
- The `models/yolov5s.yaml` is customized with `nc: 3` (not the default 80 COCO classes).

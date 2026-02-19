# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FRC Scoring Analyzer — 機器人進球偵測桌面應用，支援 HSV + AI 雙模式球偵測 + YOLO+ByteTrack 機器人多目標追蹤（MOT）+ VitTrack/CSRT 備用追蹤（SOT）+ 區域進入進球判定 + 出手偵測。

> **分離自:** `D:\FRC\frc-video-analyzer` (2026-02-14)
> **參考:** `D:\FRC\高度分析`（球飛行時間分析器，同樣結構的 CustomTkinter 應用）

## Tech Stack

- Python 3.11+
- CustomTkinter 5 (暗色主題桌面 GUI)
- OpenCV 4.9+ (電腦視覺，含 VitTrack)
- NumPy, Pillow, SciPy
- onnxruntime 1.17+ (YOLO ONNX 本地離線推理)
- supervision 0.21+ (ByteTrack 多目標追蹤)

## Run Commands

```bash
# 安裝依賴
pip install -r requirements.txt

# 啟動應用
python main.py

# 啟動並載入影片
python main.py path/to/video.mp4

# 訓練機器人偵測模型（需額外安裝 roboflow, ultralytics）
python train_robot_model.py --api-key YOUR_ROBOFLOW_KEY
```

## Architecture

```
scoring-analyzer/
├── config.py              # 所有常數與設定（HSV、AI 偵測、ByteTrack、UI 配色）
├── detection.py           # 球偵測（HSV + AI 雙模式，AI = YOLOv11 ONNX 本地推理）
├── tracking.py            # 球 CentroidTracker + 軌跡縫合
├── robot_detection.py     # 機器人偵測（YOLO ONNX，支援 NMS-Free 和傳統格式）
├── robot_tracker.py       # 機器人追蹤（MOT: YOLO+ByteTrack / SOT: VitTrack/CSRT）
├── scoring.py             # 進球判定 + 出手偵測（多邊形區域 + 射手歸因 + 命中率）
├── geometry.py            # 幾何工具（線段交叉、點在矩形內、點在多邊形內）
├── utils.py               # 字型載入、格式化工具
├── app.py                 # CustomTkinter 主介面（ScoringAnalyzer 類別）
├── main.py                # 入口點
├── train_robot_model.py   # 機器人偵測模型訓練腳本（獨立工具）
├── models/                # ONNX 模型目錄（VitTrack + fuel + robot）
├── requirements.txt       # 依賴套件
└── README.md              # 使用說明書
```

## Core Algorithms

1. **球偵測 (雙模式)**:
   - **HSV 模式** (預設): HSV 黃色過濾 (H=20-35, S=100-255, V=100-255) → morphologyEx → findContours
   - **AI 模式** (選配): YOLOv11n ONNX 本地推理 → letterbox + NMS → `(cx, cy, area, radius)`
2. **球追蹤**: CentroidTracker (Hungarian 匹配 + EMA 速度預測) + 軌跡縫合
3. **機器人追蹤 (雙模式)**:
   - **MOT 模式** (主要): YOLO 偵測 → ByteTrack 多目標關聯 → Label 映射 → 遮擋插值
   - **SOT 模式** (備用): VitTrack/CSRT 單目標追蹤 + 模板匹配恢復
4. **進球判定**: 球進入 Hub 多邊形區域（ray casting）→ 停留 N 幀確認 → 回溯軌跡找最近機器人歸因
5. **出手偵測**: 速度突增 + 機器人鄰近度 → 追蹤進球/未進 → 命中率統計
6. **播放系統**: 牆鐘時間基準 → `time.monotonic()` + 落後追趕 → 1x/0.5x 速度

## Code Style

- Python PEP 8，使用 type hints
- UI 語言：繁體中文
- 配色定義在 `config.py` 的 `COLORS` dict

## Error Tracking

開發過程中遇到錯誤，請記錄到 `errors.md`。

---
*Last updated: 2026-02-19*

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FRC Scoring Analyzer — 機器人進球偵測桌面應用，支援 HSV + AI 雙模式球偵測 + HSV Bumper / YOLO 雙模式機器人偵測 + 距離匹配+顏色直方圖 Re-ID 多目標追蹤（MOT）+ VitTrack/CSRT 備用追蹤（SOT）+ Temporal Median 背景模型自動前景遮罩 + 2 層射手歸因（HP>Proximity）+ 區域進入進球判定 + 出手偵測 + 4-Panel Debug View。

> **分離自:** `D:\FRC\frc-video-analyzer` (2026-02-14)
> **參考:** `D:\FRC\高度分析`（球飛行時間分析器，同樣結構的 CustomTkinter 應用）

## Tech Stack

- Python 3.11+
- CustomTkinter 5 (暗色主題桌面 GUI)
- OpenCV 4.9+ (電腦視覺，含 VitTrack)
- NumPy, Pillow, SciPy
- onnxruntime 1.17+ (YOLO ONNX 本地離線推理，支援 CUDA/DirectML/CPU 自動選擇)
- supervision 0.21+ (ByteTrack 多目標追蹤)

## Run Commands

```bash
# 安裝依賴
pip install -r requirements.txt

# 啟動應用
python main.py

# 啟動並載入影片
python main.py path/to/video.mp4
```

## Architecture

```
scoring-analyzer/
│
│   ── 核心程式（應用本體，請勿搬動，會破壞 import）──────────────
├── main.py                # 入口點（11 行）
├── app.py                 # CustomTkinter 主 GUI（2839 行）
│                          #   - Pipeline 並行分析（producer-consumer Queue）
│                          #   - 4-Panel Debug View（F3 toggle）
│                          #   - Bumper 取色、遮擋區域 UI、LOST 機器人虛線框
├── config.py              # 所有常數與預設值（HSV/MOT/HP/UI 配色）
├── runtime_config.py      # 動態參數容器 + Preset JSON 存檔系統
├── detection.py           # 球偵測（HSV + AI 雙模式，OpenCL UMat GPU 加速）
├── tracking.py            # 球 CentroidTracker（Hungarian + EMA 速度 + 軌跡縫合）
├── robot_detection.py     # 機器人偵測（HSV BumperDetector + YOLO ONNX，CUDA/DML/CPU）
├── robot_tracker.py       # MOT 多目標追蹤（1507 行）
│                          #   - Track State Machine（ACTIVE/LOST/REMOVED）
│                          #   - Round 1 ACTIVE Hungarian 全域最優匹配
│                          #   - Round 2 LOST 復活 Re-ID + 鄰近守衛
│                          #   - 色彩直方圖 Re-ID（HSV H+S 16x16 EMA update）
│                          #   - 遮擋區域感知（hub 內 450 幀耐心）
│                          #   - 幽靈偵測守衛、每幀 IoU dedup
│                          #   - Pipeline API: detect_raw() / track_update() 拆分
├── scoring.py             # 進球判定 + 2 層射手歸因（2428 行）
│                          #   - HP 線段交叉判定（conf=0.95）
│                          #   - Proximity 80 幀回溯找最近機器人
│                          #   - 出手偵測 + 命中率統計
├── background.py          # Temporal Median 背景模型（分離式前景遮罩）
├── calibration.py         # HSV 自動校正（K-Means 多點取色 + Bumper 模板建立）
├── settings_window.py     # 嵌入式設定面板（slider + HSV 即時預覽）
├── geometry.py            # 幾何工具（線段交叉、點在多邊形、點到線距離）
├── utils.py               # 字型載入、格式化
│
│   ── 測試工具（import 核心，暫留 root）────────────────────────
├── test_analysis.py       # 互動式分析測試（zone drawing + ID 穩定性評估）
├── diagnose.py            # HSV 參數診斷工具
│
│   ── 文件 ───────────────────────────────────────────────────
├── README.md              # 使用說明
├── CLAUDE.md              # 本檔（專案指南）
├── requirements.txt       # 依賴套件
├── PROGRESS.md            # 開發進度（每 session 紀錄）
├── FINDINGS.md            # 技術研究與決策
├── errors.md              # 錯誤追蹤
│
│   ── 文件目錄 ───────────────────────────────────────────────
├── docs/
│   ├── notebook/          # FRC 工程筆記（md/html/txt/EN 4 個版本）
│   ├── plans/             # 設計文件與實作計劃（舊）
│   └── superpowers/       # Superpowers 設計規格與實作計劃
│       ├── specs/
│       └── plans/
│
│   ── 資源目錄 ───────────────────────────────────────────────
├── models/                # ONNX 模型（runtime 用，訓練在 D:\FRC\frc-train-review）
│   ├── frc_robot.onnx             # YOLOv26n 機器人偵測（9.8 MB NMS-Free）
│   └── object_tracking_vittrack_*.onnx  # VitTrack SOT
├── presets/               # Preset JSON（場地設定檔）
│
│   ── 忽略目錄 ───────────────────────────────────────────────
├── videos/                # 測試用影片（gitignored）
└── scratch/               # 除錯圖片、截圖、臨時檔（gitignored）
```

## Core Algorithms

1. **球偵測 (雙模式)**:
   - **HSV 模式** (預設): GaussianBlur(5,5) → HSV 黃色過濾 → Close(7,7)+Open(3,3) 形態學 → findContours（OpenCL UMat GPU 加速）
   - **AI 模式** (選配): YOLOv26n ONNX 本地推理 → letterbox + NMS → `(cx, cy, area, radius)`（支援 CUDA/DML/CPU Provider）
2. **球追蹤**: CentroidTracker (Hungarian 匹配 + EMA 速度預測) + 軌跡縫合
3. **機器人偵測 (雙模式)**:
   - **HSV Bumper 模式**: `BumperDetectorHSV` — HSV 紅藍 bumper 過濾 + 形態學 + NMS + 用戶取色模板直方圖匹配過濾
   - **YOLO 模式** (目前啟用): `RobotDetectorONNX` — YOLOv26n ONNX 推理，需 `models/frc_robot.onnx`
4. **機器人追蹤 (MOT)**: 偵測端中心距離去重 → Track State Machine（ACTIVE/LOST/REMOVED）→ 兩輪匹配（Round 1 ACTIVE Hungarian 全域最優 + Round 2 LOST 復活 Re-ID）→ 幽靈偵測守衛 → 每幀 IoU dedup → Grace Period → 遮擋區域感知（hub 內 450 幀耐心）→ 色彩直方圖 Re-ID → auto_mode 自動分配 Red-N/Blue-N → 後處理 `merge_fragmented_labels()`
5. **進球判定 + 2 層歸因（簡化版）**: 球進入 Hub 多邊形（ray casting）→ 停留 N 幀確認 → HP 判定（軌跡與 HP 線段交叉，conf=0.95）> Proximity 歸因（`goal_frame - 80` 附近找最近機器人）→ `reattribute_shooters()` 後處理用完整機器人位置重新歸因
6. **出手偵測**: 速度突增 + 球往上飛（dy<0）+ 機器人鄰近度 → 追蹤進球/未進 → 命中率統計
7. **背景模型前景遮罩（分離式）**: Temporal Median 背景提取 + 閾值化前景遮罩 → **分離式應用**：球偵測用 `frame_masked`，機器人偵測用原始 `frame`（避免靜止機器人被誤判為背景）
8. **播放系統**: 固定步進 + 30fps 顯示率 + grab() 跳幀 + seek 大跳躍 + 1-5x 速度
9. **視覺化增強**:
   - **追蹤信心度**: 實線粗框=真實偵測、細線灰框=插值推測
   - **LOST 機器人**: 虛線框 + 灰色標籤（區別於 ACTIVE 的實線框）
   - **4-Panel Debug View**: F3 toggle — Foreground Mask / Ball Ownership / Robot Detection / Full Overlay
10. **Pipeline 並行分析架構**: Producer Thread 提前 8 幀偵測（無狀態 `detect_raw()`）→ `Queue(maxsize=8)` → Consumer 追蹤匹配（有狀態 `track_update()`）+ 進球判定 → 吞吐從 75ms/幀降至 ~18ms/幀（4.2x 加速）

## Code Style

- Python PEP 8，使用 type hints
- UI 語言：繁體中文
- 配色定義在 `config.py` 的 `COLORS` dict

## Error Tracking

開發過程中遇到錯誤，請記錄到 `errors.md`。

---
*Last updated: 2026-04-15 (session 32) — 分離 dataset pipeline 至 D:\FRC\frc-train-review*

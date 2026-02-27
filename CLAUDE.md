# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FRC Scoring Analyzer — 機器人進球偵測桌面應用，支援 HSV + AI 雙模式球偵測 + HSV Bumper / YOLO 雙模式機器人偵測 + 距離匹配+顏色直方圖 Re-ID+速度預測 多目標追蹤（MOT）+ VitTrack/CSRT 備用追蹤（SOT）+ Temporal Median 背景模型自動前景遮罩 + 球所有權追蹤 + 3 層射手歸因 + 區域進入進球判定 + 出手偵測 + 4-Panel Debug View。

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

# 訓練機器人偵測模型（需額外安裝 roboflow, ultralytics）
python train_robot_model.py --api-key YOUR_ROBOFLOW_KEY
```

## Architecture

```
scoring-analyzer/
├── config.py              # 所有常數與預設值（HSV、AI 偵測、ROBOT_DETECTION_MODE、BUMPER_* HSV 參數、ByteTrack、球所有權、直方圖Re-ID、背景模型BG_*、UI 配色）
├── runtime_config.py      # RuntimeConfig 動態參數容器 + Preset 系統
├── calibration.py         # HSV 自動校正（K-Means 多點取色 + 單點取色 + 預覽生成）
├── settings_window.py     # 設定面板（SettingsPanel 嵌入式，分頁 slider + HSV 即時預覽）
├── background.py          # 背景模型（BackgroundModel，Temporal Median 背景提取 + 前景遮罩，取代手動場地邊界）
├── detection.py           # 球偵測（HSV + AI 雙模式，OpenCL UMat GPU 加速，CUDA/DML/CPU Provider）
├── tracking.py            # 球 CentroidTracker + 軌跡縫合
├── robot_detection.py     # 機器人偵測（HSV Bumper 偵測器 BumperDetectorHSV + YOLO ONNX RobotDetectorONNX，config ROBOT_DETECTION_MODE 切換，CUDA/DML/CPU Provider）
├── robot_tracker.py       # 機器人追蹤（MOT: YOLO+距離匹配+顏色直方圖Re-ID+速度預測+auto_mode+merge_fragmented_labels(迭代式+允許小重疊)+detected_frames信心度 / SOT: VitTrack/CSRT）
├── scoring.py             # 進球判定 + 球所有權追蹤(compute_ball_ownership) + 3層歸因(HP>Ownership>Proximity) + 出手偵測 + 射手重新歸因 + HP 歸因（多邊形區域 + reattribute_shooters + HP 線段 + 命中率）
├── geometry.py            # 幾何工具（線段交叉、點在矩形內、點在多邊形內、點到線段距離）
├── utils.py               # 字型載入、格式化工具
├── app.py                 # CustomTkinter 主介面（ScoringAnalyzer 類別，含背景模型自動前景遮罩+追蹤信心度視覺化+per-robot計數overlay+F3 4-Panel Debug View）
├── main.py                # 入口點
├── train_robot_model.py   # 機器人偵測模型訓練腳本（獨立工具）
├── extract_frames.py      # 影片取幀工具（提取訓練用圖片）
├── train_colab.ipynb      # Google Colab 訓練 notebook（免費 T4 GPU，不提交 git）
├── models/                # ONNX 模型目錄（VitTrack SOT 追蹤 + frc_robot 機器人偵測）
├── presets/               # Preset JSON 目錄（場地設定檔）
│   └── 預設值.json         # 預設參數值
├── test_analysis.py       # 分析測試工具（互動式 zone drawing + 出手偵測 + ID 穩定性評估）
├── TRAIN_README.txt       # GPU 訓練步驟指南
├── requirements.txt       # 依賴套件
└── README.md              # 使用說明書
```

## Core Algorithms

1. **球偵測 (雙模式)**:
   - **HSV 模式** (預設): GaussianBlur(5,5) → HSV 黃色過濾 → Close(7,7)+Open(3,3) 形態學 → findContours（OpenCL UMat GPU 加速）
   - **AI 模式** (選配): YOLOv11n ONNX 本地推理 → letterbox + NMS → `(cx, cy, area, radius)`（支援 CUDA/DML/CPU Provider）
2. **球追蹤**: CentroidTracker (Hungarian 匹配 + EMA 速度預測) + 軌跡縫合
3. **機器人偵測 (雙模式)**:
   - **HSV Bumper 模式** (預設, `ROBOT_DETECTION_MODE="HSV"`): `BumperDetectorHSV` — GaussianBlur → HSV 紅藍 bumper 過濾（紅色 hue 環繞處理）→ 矩形 morphology 核 Close(9,5)+Open(5,3) → 面積+長寬比過濾 → NMS 去重 → 同介面輸出（無需 ONNX 模型/訓練資料）
   - **YOLO 模式** (備選, `ROBOT_DETECTION_MODE="YOLO"`): `RobotDetectorONNX` — YOLO ONNX 推理 → NMS-Free/傳統格式自動偵測 → 需 `models/frc_robot.onnx`
4. **機器人追蹤 (雙模式)**:
   - **MOT 模式** (主要): 偵測器（HSV/YOLO）混合偵測（全幀+tiled）→ 距離式直接匹配 `_match_direct()`（繞過 ByteTrack IoU）→ **顏色直方圖 Re-ID**（HSV H+S 16x16 bins, EMA 70/30 update, `effective_dist = spatial_dist * (1 + 0.4 * (1 - similarity))`）→ **速度預測**（`_last_known` 含 vx/vy，匹配用預測位置）→ 動態距離閾值 → Label 映射 → 遮擋插值 → auto_mode 自動分配 Red-N/Blue-N label → `_detected_frames` 追蹤真實偵測幀 → 後處理 `merge_fragmented_labels()`（迭代式，max_overlap=15, search_window=±180幀）
   - **SOT 模式** (備用): VitTrack/CSRT 單目標追蹤 + 模板匹配恢復
5. **球所有權追蹤**: `compute_ball_ownership()` — 每幀找球最近機器人標記 owner → velocity-gated transfer（球速低才允許轉移）→ 飛行中保持 owner 不變
6. **進球判定 + 3 層歸因**: 球進入 Hub 多邊形區域（ray casting）→ 停留 N 幀確認 → **3 層歸因**：HP 線段歸因（球軌跡離 HP 線段 300px 內）> Ownership 歸因（出手幀球 owner）> Proximity 歸因（回溯找最近機器人 500px 內）→ `reattribute_shooters()` 後處理用完整（合併+插值後）機器人位置重新歸因
7. **出手偵測**: 速度突增 + 球往上飛（dy<0）+ 機器人鄰近度 → 追蹤進球/未進 → 命中率統計
8. **背景模型前景遮罩**: `BackgroundModel`（`background.py`）— 均勻取樣影片幀 → `np.median` 像素中位數 → 靜態背景圖 → `cv2.absdiff` + 閾值化 + 膨脹 → 前景遮罩 → `cv2.bitwise_and` 在偵測前將靜態背景像素設為黑色（取代舊版手動 `_field_boundary` 多邊形場地遮罩），自動排除觀眾席/記分板等靜態區域
9. **播放系統**: 牆鐘時間基準 → 固定 30fps 顯示率 → grab() 跳幀 + seek 大跳躍 → 1-5x 速度
   - `_show_frame`: seek + read（暫停/拖曳用）
   - `_render_frame`: LANCZOS4 高品質渲染（暫停時）
   - `_render_frame_playback`: INTER_LINEAR 快速渲染 + cv2.putText（播放時）
   - `_play_loop`: 30fps 顯示率 + grab() 跳幀 + 大間距 seek
10. **視覺化增強**:
   - **追蹤信心度**: 實線粗框=真實偵測（`_detected_frames`）、細線灰框=插值推測
   - **Per-robot 計數 overlay**: 底部即時顯示每台機器人累計進球數（`_cumulative_goals`）
   - **4-Panel Debug View**: F3 toggle — Foreground Mask / Ball Ownership / Robot Detection / Full Overlay

## Code Style

- Python PEP 8，使用 type hints
- UI 語言：繁體中文
- 配色定義在 `config.py` 的 `COLORS` dict

## Error Tracking

開發過程中遇到錯誤，請記錄到 `errors.md`。

---
*Last updated: 2026-02-27*

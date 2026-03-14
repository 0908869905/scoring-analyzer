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
- google-genai (Gemini Vision API，自動標註用，`auto_annotate.py` 專用)

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

# Label Editor 互動標註工具
python label_editor.py datasets/2024mslr
python label_editor.py datasets/2024mslr --start 0 --end 1421    # 前半（分工用）
python label_editor.py datasets/2024mslr --start 1421             # 後半（分工用）
python label_editor.py datasets/2024mslr --labels labels_raw      # 自訂標註目錄
python label_editor.py datasets/2026cosp --images images --labels labels  # 自訂圖片+標註目錄
# Space 鍵標記已審核，狀態存 review_state.json，支援斷點恢復
```

## Architecture

```
scoring-analyzer/
├── config.py              # 所有常數與預設值（HSV、AI 偵測、ROBOT_DETECTION_MODE、BUMPER_* HSV 參數、BUMPER_TEMPLATE_SIMILARITY、ByteTrack、球所有權、直方圖Re-ID、背景模型BG_*、MOT遮擋常數MOT_MAX_LOST_FRAMES/MOT_OCCLUSION_PATIENCE/MOT_LOST_GRACE_FRAMES/MOT_LOST_REID_DIST_SCALE/MOT_LOST_MIN_HIST_SIM/MOT_OCCLUSION_MARGIN、UI 配色）
├── runtime_config.py      # RuntimeConfig 動態參數容器 + Preset 系統
├── calibration.py         # HSV 自動校正（K-Means 多點取色 + 單點取色 + 預覽生成）+ Bumper 取色模板建立（build_bumper_template, HSV H+S 16x16 直方圖 + 聯盟自動判斷）
├── settings_window.py     # 設定面板（SettingsPanel 嵌入式，分頁 slider + HSV 即時預覽 + 進球/出手 Tab 雙區架構：slider + 即時預覽 canvas + 重新計算歸因按鈕）
├── background.py          # 背景模型（BackgroundModel，Temporal Median 背景提取 + 前景遮罩，取代手動場地邊界）
├── detection.py           # 球偵測（HSV + AI 雙模式，OpenCL UMat GPU 加速，CUDA/DML/CPU Provider）
├── tracking.py            # 球 CentroidTracker + 軌跡縫合
├── robot_detection.py     # 機器人偵測（HSV Bumper 偵測器 BumperDetectorHSV + YOLO ONNX RobotDetectorONNX，config ROBOT_DETECTION_MODE 切換，CUDA/DML/CPU Provider，Bumper 模板匹配過濾 set_templates/_match_template，threading.local() 預分配 buffer 加速 _preprocess）
├── robot_tracker.py       # 機器人追蹤（MOT: YOLO+Track State Machine(ACTIVE/LOST/REMOVED三態)+兩輪匹配(_match_direct Round1 ACTIVE貪心+Round2 LOST復活Re-ID門檻)+遮擋區域感知(set_occlusion_zones+hub內15秒耐心)+距離匹配+顏色直方圖Re-ID+速度預測+auto_mode+merge_fragmented_labels(迭代式+允許小重疊)+detected_frames信心度+_consume_pending_marker距離式標記匹配+filter_static_labels標記式(_static_labels set)+Pipeline API: detect_raw()/track_update() 拆分 / SOT: VitTrack/CSRT）
├── scoring.py             # 進球判定 + 球所有權追蹤(compute_ball_ownership) + 3層歸因(HP>Ownership>Proximity) + 出手偵測 + 射手重新歸因 + HP 歸因（多邊形區域 + reattribute_shooters + HP 線段交叉判定 segments_intersect + 命中率）
├── geometry.py            # 幾何工具（線段交叉、點在矩形內、點在多邊形內、點到線段距離、點到多邊形邊最短距離 min_distance_to_polygon_edge）
├── utils.py               # 字型載入、格式化工具
├── app.py                 # CustomTkinter 主介面（ScoringAnalyzer 類別，含Pipeline並行分析(Queue+producer Thread偵測提前8幀/consumer依序追蹤)+分離式背景遮罩(球偵測用masked/機器人用原始frame)+MOT永遠距離匹配+追蹤信心度視覺化+per-robot計數overlay+F3 4-Panel Debug View+Bumper取色互動模式+OcclusionZone遮擋區域(dataclass+多邊形UI繪製+半透明填充)+LOST機器人渲染(虛線框_draw_dashed_rect+灰色標籤)）
├── main.py                # 入口點
├── train_robot_model.py   # 機器人偵測模型訓練腳本（獨立工具）
├── extract_frames.py      # 影片取幀工具（提取訓練用圖片）
├── auto_annotate.py       # Gemini Vision 自動標註 pipeline（gemini-3.1-flash-lite-preview + native box_2d 1000-based 座標 + unstructured response regex 解析 + HttpOptions timeout 30s + bumper-only prompt 0-6 bumpers + _EXCLUSION_ZONES 場地固定物件過濾 + YOLO 格式轉換 + 資料集匯出 train/val split）
├── crop_events.py         # 影片裁切工具（TBA 賽事影片批次裁切 + crop.json 記錄 base_w/base_h + 多解析度自動等比縮放 + 統一 resize 輸出）
├── label_editor.py        # 互動標註工具（CustomTkinter + Canvas，YOLO bbox 編輯：選取/移動/resize/繪製/zoom/pan/undo，--start/--end 分工標註，--labels 自訂目錄，--images 自訂圖片子目錄，Space 鍵審核標記+review_state.json 斷點恢復+✓ 已審核 overlay+審核進度計數）
├── merge_datasets.py      # 多賽事資料集合併工具（images+labels → merged/ + data.yaml + 80/20 train/val split）
├── detect_bad_frames.py   # 壞幀偵測工具（掃描 merged dataset 找異常幀：過曝/紙花/計分板/假標註過多）
├── download_matches.py    # TBA API 比賽查詢 + YouTube 影片下載工具
├── train_colab.ipynb      # Google Colab 訓練 notebook（免費 T4 GPU，不提交 git）
├── models/                # ONNX 模型目錄（VitTrack SOT 追蹤 + frc_robot YOLOv26n 機器人偵測 9.8MB NMS-Free + frc_robot_old 備份）
├── datasets/              # 訓練資料目錄（影片幀 + 裁切設定 + 標註）
│   ├── 2023mslr/          # 2023 Magnolia Regional（crop.json + images/ + labels/）
│   ├── 2024mslr/          # 2024 Magnolia Regional（主要訓練資料集）
│   │   ├── crop.json       # 裁切範圍 (1, 105) 1278x534
│   │   ├── images/         # 2842 張裁切後圖片（已移除 36 場廣角比賽幀）
│   │   ├── images_raw/     # 2842 張原始圖片（已移除 36 場廣角比賽幀）
│   │   ├── labels_raw/     # Gemini 原始標註（box_2d 轉 YOLO 格式，含小 bbox）
│   │   ├── labels/         # 過濾後 YOLO labels（移除過小 bbox）
│   │   ├── videos/         # 87 部 720p 官方比賽影片
│   │   └── yolo_dataset/   # YOLO train/val split + data.yaml（訓練入口）
│   ├── 2026mndu/          # 2026 Manade District（214 張 Gemini 標註，審核完成）
│   ├── 2026cosp/          # 2026 Colorado Springs（224 張 Gemini 標註，審核完成）
│   ├── 2026okok/          # 2026 Oklahoma（215 張 Gemini 標註，審核完成）
│   ├── 2026bcvi/          # 2026 Bayou City（222 張 Gemini 標註，審核完成）
│   ├── 2026tuis/          # 2026 Tulsa（218 張 Gemini 標註，審核完成）
│   ├── merged/            # 合併資料集（2024mslr 778張 + 5×2026 1048張 = 1826張，train=1461/val=365，已清除壞幀+同步審核標註+手動逐張修正）
│   └── reviewed/          # 最終審核資料集（4人分工審核收回+最終確認，images/+labels/+data.yaml，本地GPU訓練用）
│       ├── part1~4/       # 分工切割（457/457/457/455張，各含 images/+labels/，用於分發審核）
│       └── data.yaml      # YOLO 訓練設定（相對路徑）
├── docs/plans/            # 設計文件與實作計劃（舊）
├── docs/superpowers/      # Superpowers 設計規格與實作計劃
│   ├── specs/             # 設計規格文件（spec review 通過後提交）
│   └── plans/             # 實作計劃文件（plan review 通過後提交）
├── presets/               # Preset JSON 目錄（場地設定檔）
│   └── 預設值.json         # 預設參數值
├── test_analysis.py       # 分析測試工具（互動式 zone drawing + 出手偵測 + ID 穩定性評估）
├── TRAIN_README.txt       # GPU 訓練步驟指南
├── ENGINEERING_NOTEBOOK.md    # FRC 工程筆記（中文 Markdown 完整版）
├── ENGINEERING_NOTEBOOK.html  # FRC 工程筆記（精簡 HTML 版，帶 CSS 樣式）
├── ENGINEERING_NOTEBOOK.txt   # FRC 工程筆記（中文純文字版）
├── ENGINEERING_NOTEBOOK_EN.txt # FRC 工程筆記（英文翻譯版）
├── requirements.txt       # 依賴套件
└── README.md              # 使用說明書
```

## Core Algorithms

1. **球偵測 (雙模式)**:
   - **HSV 模式** (預設): GaussianBlur(5,5) → HSV 黃色過濾 → Close(7,7)+Open(3,3) 形態學 → findContours（OpenCL UMat GPU 加速）
   - **AI 模式** (選配): YOLOv26n ONNX 本地推理 → letterbox + NMS → `(cx, cy, area, radius)`（支援 CUDA/DML/CPU Provider）
2. **球追蹤**: CentroidTracker (Hungarian 匹配 + EMA 速度預測) + 軌跡縫合
3. **機器人偵測 (雙模式)**:
   - **HSV Bumper 模式** (預設, `ROBOT_DETECTION_MODE="HSV"`): `BumperDetectorHSV` — GaussianBlur → HSV 紅藍 bumper 過濾（紅色 hue 環繞處理）→ 矩形 morphology 核 Close(9,5)+Open(5,3) → 面積+長寬比過濾 → NMS 去重 → **Bumper 模板匹配過濾**（用戶取色建立 HSV H+S 16x16 直方圖模板，`_match_template()` 比對候選，只保留匹配的機器人）→ 同介面輸出（無需 ONNX 模型/訓練資料）
   - **YOLO 模式** (目前啟用, `ROBOT_DETECTION_MODE="YOLO"`): `RobotDetectorONNX` — YOLOv26n ONNX 推理 → NMS-Free/傳統格式自動偵測 → 需 `models/frc_robot.onnx` (9.8 MB, 640x640, NMS-Free [1,300,6])
4. **機器人追蹤 (雙模式)**:
   - **MOT 模式** (主要): 偵測器（HSV/YOLO）混合偵測（全幀+tiled）→ **Track State Machine**（ACTIVE/LOST/REMOVED 三態）→ **兩輪匹配** `_match_direct()`：Round 1（ACTIVE 貪心匹配，保護正常追蹤）+ Round 2（LOST 復活，凍結位置+嚴格 Re-ID 門檻 `MOT_LOST_MIN_HIST_SIM`+距離縮放 `MOT_LOST_REID_DIST_SCALE`）→ **Grace Period**（`MOT_LOST_GRACE_FRAMES=3` 幀容錯防閃爍）→ **遮擋區域感知**（用戶標記 hub 多邊形 `set_occlusion_zones()`，LOST 在遮擋區域內有 `MOT_OCCLUSION_PATIENCE=450` 幀耐心 vs 預設 `MOT_MAX_LOST_FRAMES=90`）→ **顏色直方圖 Re-ID**（HSV H+S 16x16 bins, EMA 70/30 update）→ **速度預測**（`_last_known` 含 vx/vy）→ Label 映射 → `_consume_pending_marker()` 距離式標記匹配 → 遮擋插值 → auto_mode 自動分配 Red-N/Blue-N label → `_detected_frames` 追蹤真實偵測幀 → `filter_static_labels()` 標記式（`_static_labels` set，不刪除只標記 `is_static`）→ 後處理 `merge_fragmented_labels()`（迭代式，max_overlap=15, search_window=±180幀）
   - **SOT 模式** (備用): VitTrack/CSRT 單目標追蹤 + 模板匹配恢復
5. **球所有權追蹤**: `compute_ball_ownership()` — 每幀找球最近機器人標記 owner → velocity-gated transfer（球速低才允許轉移）→ 飛行中保持 owner 不變
6. **進球判定 + 3 層歸因**: 球進入 Hub 多邊形區域（ray casting）→ 停留 N 幀確認 → **3 層歸因**：HP 線段歸因（球軌跡與 HP 線段交叉 `segments_intersect()`）> Ownership 歸因（出手幀球 owner）> Proximity 歸因（回溯找最近機器人 500px 內）→ `reattribute_shooters()` 後處理用完整（合併+插值後）機器人位置重新歸因
7. **出手偵測**: 速度突增 + 球往上飛（dy<0）+ 機器人鄰近度 → 追蹤進球/未進 → 命中率統計
8. **背景模型前景遮罩（分離式）**: `BackgroundModel`（`background.py`）— 均勻取樣影片幀 → `np.median` 像素中位數 → 靜態背景圖 → `cv2.absdiff` + 閾值化 + 膨脹 → 前景遮罩 → **分離式應用**：球偵測用 `frame_masked`（`cv2.bitwise_and` 過濾靜態背景），機器人偵測用原始 `frame`（避免 Temporal Median 將靜止機器人誤判為背景導致 bumper 被抹黑）
9. **播放系統**: 固定步進（每 tick 前進 `fps/30 × speed` 幀，渲染慢自動降速）→ 固定 30fps 顯示率 → grab() 跳幀 + seek 大跳躍 → 1-5x 速度
   - `_show_frame`: seek + read（暫停/拖曳用）
   - `_render_frame`: LANCZOS4 高品質渲染（暫停時）
   - `_render_frame_playback`: INTER_LINEAR 快速渲染 + cv2.putText（播放時）
   - `_play_loop`: 30fps 顯示率 + grab() 跳幀 + 大間距 seek
10. **視覺化增強**:
   - **追蹤信心度**: 實線粗框=真實偵測（`_detected_frames`）、細線灰框=插值推測
   - **LOST 機器人渲染**: 虛線框（`_draw_dashed_rect()`）+ 灰色標籤，區別於 ACTIVE 的實線框
   - **遮擋區域渲染**: 用戶標記的多邊形以半透明灰色填充顯示，繪製時灰色頂點標記
   - **Per-robot 計數 overlay**: 底部即時顯示每台機器人累計進球數（`_cumulative_goals`）
   - **4-Panel Debug View**: F3 toggle — Foreground Mask / Ball Ownership / Robot Detection / Full Overlay
11. **Pipeline 並行分析架構**: `_run_analysis()` 拆為 producer-consumer 架構 → **Producer Thread** 提前 8 幀讀幀 + 球偵測 + 機器人偵測（`detect_raw()`，無狀態）→ `Queue(maxsize=8)` → **Consumer**（分析線程）依序執行追蹤匹配（`track_update()`，有狀態）+ 進球判定 → 偵測與追蹤重疊執行，吞吐從 75ms/幀降至 ~18ms/幀（4.2x 加速）

## Code Style

- Python PEP 8，使用 type hints
- UI 語言：繁體中文
- 配色定義在 `config.py` 的 `COLORS` dict

## Error Tracking

開發過程中遇到錯誤，請記錄到 `errors.md`。

---
*Last updated: 2026-03-14 (session 15)*

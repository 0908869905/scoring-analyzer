# FRC Scoring Analyzer — Progress

## Session: 2026-02-19 (4)

### 完成項目
- [x] 無用檔案清理 — 刪除 `models/fuel_yolov11.onnx`、`models/fuel_yolov11.pt`（球偵測 AI 模型，訓練資料是近距離但需求是廣角遠距，用戶決定棄用 AI 改用 HSV）、`fuel_yolov11/`（ultralytics 暫存）、`datasets/frc_robot/`（WorBots 4145 舊資料集）、`runs/`（訓練輸出）、`yolo11n.pt`（預訓練基底）
- [x] 專案瘦身 — 從 ~370MB 降到 ~1.5MB，僅保留 `models/object_tracking_vittrack_2023sep.onnx` (698KB, SOT 追蹤用)
- [x] 打包給對方訓練 — 建立 `scoring-analyzer-train.tar` (880KB)，排除 .git、__pycache__、.playwright-mcp、CLAUDE.md、PROGRESS.md、FINDINGS.md、errors.md
- [x] Git commit — `chore: 清理無用模型與資料集，更新文件` (e60e08d)
- [x] Push 嘗試 — 失敗（未設定 GitHub remote），用戶選擇直接複製資料夾給對方

### 修改檔案
- `models/fuel_yolov11.onnx` — **刪除**（球偵測 AI 模型，棄用）
- `models/fuel_yolov11.pt` — **刪除**（YOLO 權重，棄用）
- `fuel_yolov11/` — **刪除**（ultralytics 暫存目錄）
- `datasets/frc_robot/` — **刪除**（WorBots 4145 舊資料集）
- `runs/` — **刪除**（訓練輸出目錄）
- `yolo11n.pt` — **刪除**（預訓練基底模型）

### 5-Question Reboot Check
1. **做什麼？** 清理無用模型與資料集，打包專案給 GPU 電腦訓練
2. **進度？** 清理完成，打包檔已建立，等待對方在 GPU 電腦上訓練
3. **下一步？** 把打包檔傳到有 NVIDIA GPU 的電腦 → 對方照 `TRAIN_README.txt` 執行訓練 → 訓練完帶回 `models/frc_robot.onnx` → 回到本機測試 MOT 模式端到端流程
4. **阻礙？** 本機無 GPU；`models/frc_robot.onnx` 尚未產出，MOT 模式無法使用；GitHub remote 未設定
5. **檔案？** `TRAIN_README.txt`（GPU 訓練指南）、`train_robot_model.py`（訓練腳本）、`robot_detection.py`（偵測器）、`models/object_tracking_vittrack_2023sep.onnx`（唯一保留的模型）

---

## Session: 2026-02-19 (3)

### 完成項目
- [x] 資料集評估與切換 — 評估多個 Roboflow FRC 機器人偵測資料集，最終選定 **Main Robot Detection** (1,172 張, Red/Blue 底盤框選) 取代 WorBots 4145（類別為遊戲元素混雜，不夠乾淨）
- [x] `train_robot_model.py` 更新 — 預設資料集從 WorBots 4145 (`worbots-4145/2024-frc/v8`) 改為 Main Robot Detection (`main-wcgiu/robot-detection-xru6m/v16`)
- [x] `robot_detection.py` 類別過濾修復 — `is_robot_class()` 和 `infer_alliance()` 支援 "Red"/"Blue" 類別名（不只 "red_robot"），修復切換資料集後所有偵測結果被過濾掉的 bug
- [x] `extract_frames.py` 新增 — 影片取幀工具，從比賽影片提取訓練用圖片，支援多影片、時間範圍、FPS 設定
- [x] `TRAIN_README.txt` 新增 — GPU 訓練指南（給對方照著跑）
- [x] `.gitignore` 更新 — 排除 datasets/、runs/、*.pt、.playwright-mcp/
- [x] Git commit 完成 — push 失敗（尚未設定 remote repo）

### 修改檔案
- `train_robot_model.py` — 預設資料集改為 Main Robot Detection (main-wcgiu/robot-detection-xru6m/v16)
- `robot_detection.py` — `is_robot_class()` 和 `infer_alliance()` 支援 "Red"/"Blue" 短類別名
- `extract_frames.py` — **新建** — 影片取幀工具（多影片、時間範圍、FPS）
- `TRAIN_README.txt` — **新建** — GPU 訓練步驟指南
- `.gitignore` — 新增 datasets/、runs/、*.pt、.playwright-mcp/ 排除規則

### 5-Question Reboot Check
1. **做什麼？** 切換機器人偵測訓練資料集 + 修復類別過濾 + 準備 GPU 訓練環境
2. **進度？** 訓練腳本、偵測器、取幀工具、訓練指南皆就緒，等待在 GPU 電腦上訓練
3. **下一步？** 把專案複製到有 NVIDIA GPU 的電腦 → 按 `TRAIN_README.txt` 執行訓練 → 產出 `models/frc_robot.onnx` → 回到本機測試 MOT 模式端到端流程
4. **阻礙？** 本機無 GPU（CPU 訓練 ~19 小時不實際）；`models/frc_robot.onnx` 尚未產出，MOT 模式無法使用；remote repo 尚未設定（push 失敗）
5. **檔案？** `train_robot_model.py`（訓練腳本）、`robot_detection.py`（`is_robot_class` 類別過濾）、`TRAIN_README.txt`（GPU 訓練指南）、`extract_frames.py`（取幀工具）

---

## Session: 2026-02-19 (2)

### 完成項目
- [x] 環境驗證 — supervision 0.27.0、onnxruntime 1.24.1、VitTrack 模型、球偵測 ONNX 皆正常，確認缺少 `models/frc_robot.onnx`
- [x] Roboflow 資料集調研 — 比較多個 FRC 機器人資料集，選定 **WorBots 4145 v8**（3,291 張，含 red_robot/blue_robot/black_robot + 遊戲元素）
- [x] `train_robot_model.py` 修正 — 預設資料集從 RF100-VL FSOD（100 張，version 1 不存在）改為 WorBots 4145 v8
- [x] `robot_detection.py` 增強 — 新增 `is_robot_class()` 過濾非機器人類別（note/speaker/display 等）；改進 `infer_alliance()` 避免 `red_display` 等誤判；`detect()` 新增 `robot_only=True` 參數
- [x] `datasets/frc_robot/data.yaml` 路徑修正 — Roboflow 下載的相對路徑 `../datasets/roboflow` 改為正確絕對路徑
- [x] CPU 訓練測試 — 確認可啟動但預估 ~19 小時，用戶決定借 GPU 電腦訓練

### 修改檔案
- `train_robot_model.py` — 預設資料集改為 WorBots 4145 v8（workspace/project/version 全部更新）
- `robot_detection.py` — 新增 `is_robot_class()`、改進 `infer_alliance()`、`detect()` 加 `robot_only` 參數
- `datasets/frc_robot/data.yaml` — 修正訓練/驗證/測試集路徑為絕對路徑

### 5-Question Reboot Check
1. **做什麼？** 選定最佳 FRC 機器人訓練資料集 + 修正偵測器過濾邏輯 + 準備模型訓練
2. **進度？** 訓練腳本和偵測器已就緒，資料集已下載，等待 GPU 訓練
3. **下一步？** 在 GPU 電腦上執行 `python train_robot_model.py --local-dataset datasets/frc_robot` 訓練模型 → 產出 `models/frc_robot.onnx` → 測試 MOT 模式端到端流程
4. **阻礙？** 缺少 GPU 電腦（CPU 訓練需 ~19 小時不實際）；`models/frc_robot.onnx` 尚未產出，MOT 模式無法使用
5. **檔案？** `train_robot_model.py`（訓練腳本）、`robot_detection.py`（`is_robot_class` + `robot_only` 過濾）、`datasets/frc_robot/data.yaml`（資料集路徑）

---

## Session: 2026-02-19

### 完成項目
- [x] **M1: robot_detection.py (新建)** — `RobotDetectorONNX` 類別，支援 NMS-Free (YOLO26) 和傳統 YOLO 兩種 ONNX 輸出格式，自動從模型 metadata 讀取類別名稱並推斷聯盟 (Red/Blue)
- [x] **M1.5: train_robot_model.py (新建)** — 完整 CLI 訓練腳本：Roboflow 下載 → YOLO 訓練 → 驗證 → ONNX 匯出 → 視覺化，支援 `--api-key` 和 `--local-dataset` 兩種資料來源
- [x] **M2: robot_tracker.py (重寫)** — MOT 模式 (`_MOTTracker`: YOLO 偵測 → ByteTrack → Label 映射 → 遮擋線性插值) + SOT 模式 (`_SOTTracker`: VitTrack/CSRT) + `RobotTrackerManager` 統一介面
- [x] **M3: scoring.py (增強)** — 新增 `ShotEvent` dataclass（出手事件）、`RobotScore` 增強（auto_goals/teleop_goals/auto_misses/teleop_misses/accuracy）、`ScoringEngine.detect_shots()` 出手後處理、新增 `SHOT_MIN_VELOCITY`/`SHOT_ROBOT_PROXIMITY` 參數
- [x] **M4: app.py (整合)** — MOT/SOT 雙模式分析流程、得分統計表新增出手/未進/命中率欄位、混合時間軸（進球 + 未進球）、Overlay 顯示機器人 bbox（MOT 模式）、CSV 匯出三區段（摘要/進球/出手）
- [x] **config.py 更新** — `DETECTION_MODE` 預設改為 `"HSV"`；新增 `ROBOT_DETECTION_*` 和 `BYTETRACK_*` 參數群；新增 `SHOT_MIN_VELOCITY`/`SHOT_ROBOT_PROXIMITY`
- [x] **requirements.txt 更新** — 新增 `supervision>=0.21.0`

### 修改檔案
- `robot_detection.py` — **新建** — RobotDetectorONNX 類別（NMS-Free + 傳統 YOLO 雙格式支援）
- `train_robot_model.py` — **新建** — 機器人偵測模型訓練 CLI 腳本
- `config.py` — 新增 ROBOT_DETECTION_*、BYTETRACK_*、SHOT_* 參數；DETECTION_MODE 改為 "HSV"
- `robot_tracker.py` — **重寫** — MOT (YOLO+ByteTrack) + SOT (VitTrack/CSRT) 雙模式架構
- `scoring.py` — 新增 ShotEvent dataclass、RobotScore 增強（出手/未進/命中率）、出手偵測邏輯
- `app.py` — 整合新追蹤系統、UI 表格增強、overlay 更新、CSV 匯出增強
- `requirements.txt` — 新增 supervision>=0.21.0
- `CLAUDE.md` — 架構文件全面更新

### 5-Question Reboot Check
1. **做什麼？** 實作機器人多目標追蹤 (MOT) 系統 + 出手偵測 + 命中率統計，全面升級追蹤和得分引擎
2. **進度？** M1-M4 全部完成。MOT/SOT 雙模式、出手偵測、UI 整合、CSV 匯出皆已實作
3. **下一步？** 用戶需訓練機器人偵測模型（`python train_robot_model.py`）才能使用 MOT 模式 → 推薦 RF100-VL FSOD 2024-FRC 資料集 → 在實際影片上調整 ByteTrack 參數和出手偵測閾值
4. **阻礙？** MOT 模式需要 `models/frc_robot.onnx` 模型檔（用戶尚未訓練）；supervision 套件需額外安裝；出手偵測閾值未經實際影片驗證
5. **檔案？** `robot_detection.py`（機器人偵測）、`robot_tracker.py`（MOT/SOT 管理器）、`scoring.py`（出手偵測）、`train_robot_model.py`（訓練腳本）、`config.py`（BYTETRACK_*/SHOT_* 參數）

---

## Session: 2026-02-15

### 完成項目
- [x] YOLOv11 ONNX 本地離線推理 — 將 AI 球偵測從 Roboflow HTTP API 改為 onnxruntime 本地推理，完全離線
- [x] 權重轉換 — 用戶手動下載 `weights (1).pt`（YOLOv11n, 5.5MB），用 ultralytics 轉換為 `fuel_yolov11.onnx`（10.1MB）
- [x] `detection.py` 完全重寫 — 移除 `RoboflowModel` HTTP API 類別，新增 `FuelDetectorONNX` 類別（letterbox 預處理 + NMS 後處理）
- [x] `config.py` AI 常數更新 — `AI_CONFIDENCE_THRESHOLD` 改 0.25；`AI_MAX_AREA` 改 200000（支援 4K）；移除 `AI_MODEL_ID`
- [x] 依賴清理 — 移除 `requests`、`python-dotenv`；新增 `onnxruntime>=1.17.0`
- [x] UI 狀態文字更新 — "連線 Roboflow AI 模型中..." → "載入 AI 模型中（本地 ONNX）..."
- [x] 驗證通過 — 每幀 ~52ms（CPU），信心值 0.67，完全離線運作

### 修改檔案
- `detection.py` — 完全重寫（Roboflow HTTP API → FuelDetectorONNX 本地 ONNX 推理）
- `config.py` — 更新 `AI_CONFIDENCE_THRESHOLD`=0.25、`AI_MAX_AREA`=200000、移除 `AI_MODEL_ID`
- `requirements.txt` — 移除 requests/python-dotenv，新增 onnxruntime>=1.17.0
- `app.py` — 更新 AI 模式載入狀態文字
- `models/fuel_yolov11.pt` — 用戶下載的 YOLOv11n 權重（新增）
- `models/fuel_yolov11.onnx` — 從 .pt 轉換的 ONNX 模型（新增）

### 5-Question Reboot Check
1. **做什麼？** 將 AI 球偵測從 Roboflow HTTP API 改為 YOLOv11 ONNX 本地離線推理
2. **進度？** 100% 完成，離線推理可用，每幀 ~52ms（CPU）
3. **下一步？** GPU 加速（onnxruntime-directml 或 CUDA）→ 用更多 FRC 影片驗證偵測精度 → 調整 confidence threshold → 端到端進球判定測試
4. **阻礙？** 目前僅 CPU 推理（52ms/幀），若需即時處理可能需 GPU 加速
5. **檔案？** `detection.py`（FuelDetectorONNX 類別）、`config.py`（AI 參數）、`models/fuel_yolov11.onnx`（模型檔）

---

## Session: 2026-02-14 (4)

### 完成項目
- [x] 播放時間修正 — 修復播放時間漂移問題，改用牆鐘時間基準（`_play_start_time` + `_play_start_frame`），確保 1 真實秒 = fps 幀
- [x] 播放追趕機制 — 落後時自動追趕（delay=1ms），正常時精確等待下一幀時間點
- [x] 切速時間基準重設 — `_toggle_speed()` 切換速度時重設 `_play_start_time` 和 `_play_start_frame`，避免跳幀
- [x] README.md 說明書 — 建立完整使用說明文件（安裝、啟動、使用流程、播放控制、快捷鍵、檔案結構）

### 修改檔案
- `app.py` — 修改 `_toggle_play()`、`_toggle_speed()`、`_play_loop()` 三個方法，用牆鐘時間基準取代逐幀延遲補償
- `README.md` — 新建，專案完整說明書

### 5-Question Reboot Check
1. **做什麼？** 修復播放時間漂移 + 建立 README 說明書
2. **進度？** 100% 完成，播放系統穩定，README 已建立
3. **下一步？** 用實際 FRC 影片端到端測試 → 驗證進球判定精度 → 調整 CSRT/CamShift 追蹤參數 → 完善聯盟系統（紅藍方統計分開）
4. **阻礙？** 無
5. **檔案？** `app.py`（播放系統 `_play_loop`）、`scoring.py`（進球引擎）、`robot_tracker.py`（機器人追蹤）

---

## Session: 2026-02-14 (3)

### 完成項目
- [x] 5 大功能實作 — 視覺調整、Tab 分頁面板、CSRT 追蹤器、聯盟系統（Hub 分紅藍方）、機器人分紅藍方
- [x] Downloads 版本合併 — 模板匹配恢復 + ROI 裁切功能
- [x] 播放系統 — 逐幀播放 + 1x/0.5x 速度切換

### 修改檔案
- `app.py` — 大幅重構，加入 Tab 面板、CSRT 追蹤、聯盟系統、播放控制
- `scoring.py` — 聯盟系統支援（紅藍方 Hub 分離）
- `geometry.py` — 幾何工具更新

### 5-Question Reboot Check
1. **做什麼？** 實作 5 大功能 + 合併 Downloads 版本 + 播放系統
2. **進度？** 功能實作完成，但播放時間有漂移問題待修
3. **下一步？** 修復播放時間漂移 → 建立 README
4. **阻礙？** 播放時間漂移（逐幀 delay 累積誤差）
5. **檔案？** `app.py`（播放相關方法）

---

## Session: 2026-02-14 (2)

### 完成項目
- [x] Hub 多邊形標記 — ScoringZone 從矩形改為多邊形支援
- [x] 移除 Upper/Lower Hub 區分 — RobotScore 簡化為 `auto` + `teleop`
- [x] `geometry.py` 新增 `point_in_polygon()` ray casting 演算法
- [x] `scoring.py` — ScoringZone 改用多邊形頂點、RobotScore 移除 upper/lower 欄位、ScoreEvent.zone_name 固定為 "Hub"
- [x] `app.py` — 合併 Upper/Lower 按鈕為單一「標記 Hub」、多邊形點擊互動（左鍵放頂點、右鍵/雙擊完成、ESC 取消）、`cv2.polylines` 繪製、得分統計表和事件時間軸移除 Upper/Lower 欄、CSV 匯出簡化
- [x] 所有模組 import 驗證通過
- [x] `point_in_polygon` 單元測試通過
- [x] `ScoringZone` 多邊形 + `RobotScore` 簡化測試通過

### 修改檔案
- `geometry.py` — 新增 `point_in_polygon(px, py, polygon)` ray casting 演算法
- `scoring.py` — ScoringZone 從矩形改為多邊形、RobotScore 移除 Upper/Lower、ScoreEvent.zone_name 固定 "Hub"
- `app.py` — 合併標記按鈕、多邊形互動模式、繪製更新、統計表/時間軸/CSV 簡化

### 5-Question Reboot Check
1. **做什麼？** 將 Hub 得分區域從矩形改為多邊形標記，並移除 Upper/Lower Hub 區分
2. **進度？** 100% 完成，所有測試通過
3. **下一步？** 用實際 FRC 影片測試多邊形標記流程 → 驗證 ray casting 判定精度 → 調整 CamShift 機器人追蹤參數
4. **阻礙？** 尚未用真實影片驗證多邊形標記的使用體驗
5. **檔案？** `app.py` (GUI 多邊形互動), `scoring.py` (進球引擎), `geometry.py` (多邊形判定)

---

## Session: 2026-02-14

### 完成項目
- [x] 建立專案結構（11 個檔案，2161 行）
- [x] 複用高度分析的球偵測 + 追蹤模組 (detection.py, tracking.py)
- [x] 擴展幾何工具 (geometry.py) — 加入 point_in_rect, distance, clamp_rect
- [x] 實作機器人追蹤 (robot_tracker.py) — HSV 直方圖反投影 + CamShift
- [x] 實作進球判定引擎 (scoring.py) — 區域進入 + 射手歸因 + Auto/Teleop 分離
- [x] 建立 CustomTkinter GUI (app.py) — 暗色主題、70/30 佈局、拖曳標記
- [x] Git 初始化 + 首次 commit
- [x] 所有模組匯入測試通過

### 修改檔案
- 全新專案，所有檔案皆為新建

### 5-Question Reboot Check
1. **做什麼？** 建立 FRC Scoring Analyzer 桌面應用（偵測機器人進球並歸因）
2. **進度？** Phase 1-7 基本完成，所有模組已建立，匯入測試通過
3. **下一步？** 用實際 FRC 影片測試 → 調整偵測參數 → 優化機器人追蹤精度 → Phase 8 優化
4. **阻礙？** 尚未用真實影片驗證，CamShift 追蹤可能需要參數調整
5. **檔案？** `app.py` (GUI 主程式), `scoring.py` (進球引擎), `robot_tracker.py` (機器人追蹤)

---
*Last updated: 2026-02-19 (4)*

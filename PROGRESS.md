# FRC Scoring Analyzer — Progress

## Session: 2026-02-27

### 完成項目
- [x] **背景模型取代手動場地遮罩** — 新建 `background.py`（`BackgroundModel` 類別，Temporal Median 背景提取）
  - 均勻取樣影片幀 → 像素中位數 → 靜態背景圖 → absdiff → 閾值化 → 前景遮罩
  - 自動排除觀眾席/記分板等非場地靜態區域，無需手動繪製場地邊界
  - `config.py` 新增 `BG_SAMPLE_COUNT=50`, `BG_FG_THRESHOLD=30`, `BG_DILATE_KERNEL=5`
  - `runtime_config.py` 新增對應 dataclass 欄位
  - `app.py` 移除所有 `_field_boundary` 相關程式碼（按鈕、互動、繪製、遮罩建立），新增 `_bg_model` + 自動背景模型建立 + 每幀前景遮罩
  - `robot_tracker.py` 移除 `field_boundary` 參數和 `point_in_polygon` 過濾
  - Debug 4-panel 左上改顯示前景遮罩（取代原場地遮罩面板）
- [x] **Bumper 取色偵測設計計劃** — `docs/plans/2026-02-27-bumper-color-pick-design.md`
  - 設計：用戶在影片上點擊機器人 bumper 多次 → 建立 HSV 直方圖模板 → 分析時只追蹤匹配模板的候選
  - 取代現有的框選機器人標記方式
  - 修改範圍：`calibration.py`, `robot_detection.py`, `app.py`, `config.py`, `runtime_config.py`
  - 尚未實作，僅計劃完成

### 修改檔案
- `background.py` — **新建** — `BackgroundModel` 類別（Temporal Median 背景提取 + 前景遮罩生成）
- `config.py` — 新增 `BG_SAMPLE_COUNT`, `BG_FG_THRESHOLD`, `BG_DILATE_KERNEL` 背景模型參數
- `runtime_config.py` — 新增背景模型對應的 dataclass 欄位
- `app.py` — 移除 `_field_boundary` 全部相關程式碼，新增 `_bg_model` 背景模型整合 + 自動建立 + 每幀前景遮罩
- `robot_tracker.py` — 移除 `field_boundary` 參數和 `point_in_polygon` 過濾邏輯
- `docs/plans/2026-02-27-bumper-color-pick-design.md` — **新建** — Bumper 取色偵測實作計劃

### 5-Question Reboot Check
1. **做什麼？** 用 Temporal Median 背景模型自動取代手動場地邊界繪製，並設計 Bumper 取色偵測計劃
2. **進度？** 背景模型已完整實作並整合。Bumper 取色偵測計劃已寫好，尚未開始實作
3. **下一步？** 按照 `docs/plans/2026-02-27-bumper-color-pick-design.md` 實作 Bumper 取色偵測（Task 1-5），取代現有的框選機器人標記方式
4. **阻礙？** 需用實際影片驗證背景模型效果（是否正確分離前景/背景、是否比手動場地遮罩效果好）
5. **檔案？** `background.py`（背景模型）、`app.py`（整合）、`docs/plans/2026-02-27-bumper-color-pick-design.md`（下一步計劃）、`calibration.py` + `robot_detection.py`（Bumper 取色待修改）

---

## Session: 2026-02-26

### 完成項目
- [x] **深度研究 FRC 機器人偵測資料集** — 搜尋 Roboflow/GitHub/Chief Delphi/Kaggle，確認無 2017 Steamworks 或 2026 ReBuilt 機器人偵測資料集存在
- [x] **HSV Bumper 偵測器設計** — 設計文件 `docs/plans/2026-02-26-hsv-bumper-detection-design.md`
- [x] **BumperDetectorHSV 實作** — 完整 HSV 色彩過濾偵測紅藍 bumper（`robot_detection.py`）
  - 紅色 hue 環繞處理（兩段 inRange OR 合併）
  - 矩形 morphology 核（9x5 Close + 5x3 Open）適合水平 bumper
  - 面積 + 長寬比過濾 + NMS 去重
  - 與 RobotDetectorONNX 完全同介面（drop-in replacement）
- [x] **偵測模式切換** — `config.py` 新增 `ROBOT_DETECTION_MODE = "HSV"` 預設值，`app.py` 支援 HSV/YOLO 切換
- [x] **程式碼審查修復** — 修復 3 個問題：
  - tuple 參數 falsy-check 改為 `is not None`
  - `class_names` 從 mutable list 改為 immutable tuple
  - `detect_tiled()` 回空避免 4K 重複偵測

### 修改檔案
- `config.py` — 新增 `ROBOT_DETECTION_MODE`、`BUMPER_RED_*`、`BUMPER_BLUE_*`、`BUMPER_MIN/MAX_AREA/ASPECT`、`BUMPER_NMS_IOU`
- `robot_detection.py` — 新增 `BumperDetectorHSV` 類別（line 368-533）
- `app.py` — 偵測器初始化邏輯改為 config-based 模式切換
- `docs/plans/2026-02-26-hsv-bumper-detection-design.md` — 設計文件
- `FINDINGS.md` — 更新研究結果（資料集搜尋、準確度分析）

### 5-Question Reboot Check
1. **做什麼？** 用 HSV 色彩過濾偵測紅藍 bumper 取代 YOLO 機器人偵測（不需訓練資料，跨年份通用）
2. **進度？** 程式碼完成 + 合成影像測試通過 + 程式碼審查修復完成
3. **下一步？** 用實際 FRC 比賽影片測試 HSV Bumper 偵測效果，調整 HSV 參數（可能需要降低飽和度閾值、調整面積範圍）；之後考慮加入設定面板的偵測模式切換
4. **阻礙？** 需要用實際影片驗證 — 場地燈光、攝影機曝光、bumper 磨損等都可能影響偵測率
5. **檔案？** `robot_detection.py`（BumperDetectorHSV）、`config.py`（BUMPER_* 參數）、`app.py`（偵測器初始化）

---

## Session: 2026-02-25

### 完成項目
- [x] **應用啟動測試** — 多次啟動 `python main.py` 載入影片驗證應用正常運作
- [x] **升級模型 YOLOv11n → YOLOv26n** — 更新訓練腳本和 Colab notebook，將機器人偵測模型從 YOLOv11n 升級到 YOLOv26n
  - `train_robot_model.py` 預設模型 `--model` 從 `yolo11n` 改為 `yolo26n`（docstring 範例也更新）
  - `train_colab.ipynb` 標題改為 "YOLOv26n"、訓練 cell 的 `YOLO("yolo11n.pt")` 改為 `YOLO("yolo26n.pt")`
  - `robot_detection.py` 已支援 YOLOv26 的 NMS-Free 輸出格式 `(1, N, 6)`，無需修改

### 修改檔案
- `train_robot_model.py` — `--model` 預設值 `yolo11n` → `yolo26n`；docstring 範例更新
- `train_colab.ipynb` — 標題和 Cell 10 模型名稱 `yolo11n` → `yolo26n`

### 5-Question Reboot Check
1. **做什麼？** 將機器人偵測模型從 YOLOv11n 升級到 YOLOv26n，以獲得更好的偵測精度
2. **進度？** 訓練腳本和 Colab notebook 已更新完成。模型尚未重新訓練
3. **下一步？** 到 Google Colab 用 T4 GPU 執行 `train_colab.ipynb` 重新訓練 YOLOv26n 模型 → 下載新的 `frc_robot.onnx` → 用實際影片驗證偵測精度是否提升
4. **阻礙？** 需要到 Colab 執行訓練（本機無 GPU）；YOLOv26 是否在 Colab T4 上順利訓練尚未驗證
5. **檔案？** `train_colab.ipynb`（Colab 訓練 notebook）、`train_robot_model.py`（訓練腳本）、`robot_detection.py`（偵測器，已支援 NMS-Free 格式）、`models/frc_robot.onnx`（待重新訓練產出）

---

## Session: 2026-02-24 (2)

### 完成項目
- [x] **場地遮罩座標雙重偏移 Bug 修復** — ROI 裁切後 `_field_boundary` 已儲存 ROI 相對座標 (0..rw, 0..rh)，但 `_run_analysis()` 建立遮罩時又減去 ROI 偏移 `(rx, ry)`，導致多邊形座標變成負數 → 遮罩全為零 → 整幀變黑 → 球偵測和機器人偵測全部為 0

### 修改檔案
- `app.py` — 第 1922-1928 行，移除 ROI 分支中不必要的 `(rx, ry)` 座標偏移，直接使用 `_field_boundary` 座標建立遮罩

### 5-Question Reboot Check
1. **做什麼？** 修復場地遮罩座標雙重偏移導致球偵測和機器人偵測全部為 0 的嚴重 Bug
2. **進度？** 已修復。移除 ROI 分支中多餘的座標偏移，遮罩正確覆蓋場地區域
3. **下一步？** 用實際影片端到端驗證：(1) ROI 裁切 + 場地遮罩是否正常運作 (2) 球偵測數量恢復正常 (3) 機器人偵測數量恢復正常 (4) 進球判定和射手歸因流程完整通過
4. **阻礙？** 無明確阻礙；需實際影片驗證修復效果
5. **檔案？** `app.py`（場地遮罩建立邏輯，`_run_analysis()` 方法中約第 1922 行附近）

---

## Session: 2026-02-24

### 完成項目
- [x] **P0-2: Color Histogram Re-ID** — `robot_tracker.py`: `_extract_histogram()` HSV H+S 16x16 bins, EMA 70/30 update, `_compare_histograms()`, `effective_dist = spatial_dist * (1 + 0.4 * (1 - similarity))` 在 `_match_direct()` 中加權距離匹配
- [x] **P1-3: 前置場地遮罩 (Field Pre-masking)** — `app.py`: 分析前用 `cv2.fillPoly` + `cv2.bitwise_and` 將場地外像素設為黑色，偵測器只處理場地內區域，減少假陽性
- [x] **P1-4: 追蹤信心度視覺化** — `robot_tracker.py` 新增 `_detected_frames` dict 追蹤每個 label 的真實偵測幀；`app.py` overlay 區分：實線粗框=真實偵測、細線灰框=插值推測
- [x] **P1-5: Per-robot 計數 overlay** — `app.py`: `_cumulative_goals` 預計算每幀每機器人累計進球數，播放時底部顯示即時進球計數列（類似 5951 的 "Robot N: X" 計數欄）
- [x] **P2-6: Velocity Prediction** — `robot_tracker.py`: `_last_known` 擴展為 `(f, cx, cy, cls, vx, vy)`，`_match_direct()` 使用預測位置（而非最後已知位置）計算匹配距離
- [x] **P2-7: 4-Panel Debug View** — `app.py`: F3 快捷鍵切換 4 面板佈局：Field Mask / Ball Ownership / Robot Detection / Full Overlay

### 修改檔案
- `config.py` — 新增 `BALL_OWNERSHIP_DIST=200`（球所有權判定距離）、`MOT_HISTOGRAM_WEIGHT=0.4`（直方圖加權係數）
- `scoring.py` — 新增 `compute_ball_ownership()`（球所有權生命週期追蹤）、`_get_ball_owner_at_frame()`；修改 `detect_shots()` 和 `reattribute_shooters()` 使用 3 層歸因（HP > Ownership > Proximity）
- `robot_tracker.py` — 新增 `_extract_histogram()`（HSV H+S 16x16 bins）、`_compare_histograms()`（cv2.compareHist CORREL）、`_detected_frames` dict；`_last_known` 擴展含速度 `(vx, vy)`；`_match_direct()` 加入直方圖加權距離 + 速度預測位置
- `app.py` — 前置場地遮罩（`cv2.fillPoly` + `cv2.bitwise_and`）、追蹤信心度視覺化（實線/細線灰框）、per-robot 計數 overlay（`_cumulative_goals`）、4-panel debug view（F3 toggle）、`_robot_detected_frames` 傳遞

### 5-Question Reboot Check
1. **做什麼？** 實作競品分析後的 P0-P2 共 6 項改進（P0 Ball Ownership 在上一 session 完成），涵蓋 Re-ID、場地遮罩、視覺化、速度預測、debug 面板
2. **進度？** 全部完成並通過單元測試和匯入驗證。7 項功能（含上 session 的 Ball Ownership）全部實作完畢
3. **下一步？** 用實際影片端到端驗證：(1) Color Histogram Re-ID 是否降低 ID swap 和碎片化 (2) Ball Ownership 3 層歸因是否提升射手歸因率至 80%+ (3) 場地遮罩是否減少場外誤框 (4) 速度預測在高速移動場景的匹配改善 (5) 4-panel debug view 的調參效率
4. **阻礙？** 無明確阻礙；所有功能已實作，需實際影片驗證效果和調參
5. **檔案？** `robot_tracker.py`（histogram Re-ID + velocity prediction + detected_frames）、`scoring.py`（ball ownership + 3-tier attribution）、`app.py`（field mask + confidence viz + per-robot count + debug view）、`config.py`（BALL_OWNERSHIP_DIST + MOT_HISTOGRAM_WEIGHT）

---

## Session: 2026-02-23 (4)

### 完成項目
- [x] 射手歸因距離放大 — `SCORE_MAX_SHOOTER_DIST` 300→500（4K 下球射出到 Hub 距離常超 300px）
- [x] 合併策略放寬 — 新增 `MOT_MERGE_MAX_OVERLAP`=15（原 5 幀太短）、`MOT_MERGE_SEARCH_WINDOW`=180（原 ±60 幀放寬到 ±180 幀 ≈ ±3 秒）、`MOT_MERGE_BOUNDARY_DIST`=800（可配置化）
- [x] app.py 整合 `reattribute_shooters()` — 在 `detect_shots()` 後呼叫，與 `test_analysis.py` 流程對齊
- [x] Blue 追蹤品質 — 確認根因是訓練資料不足（confidence ~0.11），修改 1 和修改 2 的放寬可間接提升

### 修改檔案
- `config.py` — `SCORE_MAX_SHOOTER_DIST` 300→500；新增 `MOT_MERGE_MAX_OVERLAP`=15、`MOT_MERGE_BOUNDARY_DIST`=800、`MOT_MERGE_SEARCH_WINDOW`=180
- `robot_tracker.py` — import 新增 3 個 `MOT_MERGE_*` 常數；`merge_fragmented_labels()` 改用 `MOT_MERGE_MAX_OVERLAP`；`_check_boundary_distance()` 改用 `MOT_MERGE_SEARCH_WINDOW` 和 `MOT_MERGE_BOUNDARY_DIST`
- `app.py` — `_run_analysis()` 在 `detect_shots()` 後新增 `reattribute_shooters()` 呼叫

### 5-Question Reboot Check
1. **做什麼？** 4 項待處理修復：射手歸因距離、合併策略放寬、Blue 追蹤品質、app.py 整合 reattribute
2. **進度？** 全部完成。參數已調整，app.py 已整合 reattribute_shooters()，需實際影片驗證效果
3. **下一步？** 用實際影片驗證：(1) 射手歸因率是否從 47% 提升 (2) Red labels 是否從 7→3 (3) Blue 合併效果 (4) 長期：補充 Blue 訓練資料重新訓練模型
4. **阻礙？** Blue 追蹤品質根因是模型訓練資料不足，無法僅靠參數調整根本解決
5. **檔案？** `config.py`（距離+合併參數）、`robot_tracker.py`（合併邏輯）、`app.py`（reattribute 整合）

---

## Session: 2026-02-23 (3)

### 完成項目
- [x] 進球判定測試 — 用 `test_analysis.py` 對影片前 20 秒進行互動式 Hub 區域框選 + 進球判定測試，揭示射手歸因率極低和 ID 碎片化兩個問題
- [x] 射手歸因修復 — `scoring.py` 新增 `reattribute_shooters()` 後處理方法，用合併+插值後的完整 `robot_positions_by_frame` 重新歸因所有進球事件的射手
- [x] ID 碎片化修復 — `robot_tracker.py` 重寫 `merge_fragmented_labels()` 為迭代式合併 + 允許最多 5 幀重疊 + 拆分為子方法（`_find_merge_candidates()`、`_check_boundary_distance()`、`_execute_merges()`）

### 修改檔案
- `scoring.py` — 新增 `reattribute_shooters()` 方法（後處理射手歸因，用完整的合併+插值後機器人位置資料重新匹配）
- `robot_tracker.py` — 重寫 `merge_fragmented_labels()` 為迭代式 + 允許小重疊（max 5 幀）+ 拆分為 `_find_merge_candidates()`、`_check_boundary_distance()`、`_execute_merges()` 三個子方法
- `test_analysis.py` — 在後處理階段加入 `engine.reattribute_shooters()` 呼叫

### 測試結果
測試影片：`e:\FRC模擬賽第二天\A001_02131614_C025.mov`，3840x2160, 60fps, 前20秒

| 指標 | 修改前 | 修改後 |
|------|--------|--------|
| 射手歸因 | 3/42 (7%) | 20/43 (47%) |
| ID Stability | FAIR (11 stable) | GOOD (8 stable) |
| Unique Labels | 14 (Red:9, Blue:5) | 11 (Red:7, Blue:4) |

### 5-Question Reboot Check
1. **做什麼？** 修復射手歸因失敗（`_find_shooter()` 無法在偵測空洞幀找到機器人）+ 修復 ID 碎片化（`merge_fragmented_labels()` 合併策略過嚴）
2. **進度？** 兩個問題均已修復。射手歸因率 7%→47%，ID 穩定性 FAIR→GOOD。但 47% 歸因率仍有提升空間
3. **下一步？** (1) 調大 `SCORE_MAX_SHOOTER_DIST`（當前 300px 在 4K 偏小，未知事件多在 300-400px 間）(2) Red 仍有 7 個 label（期望 3），需調大重疊容忍或改進合併距離判斷 (3) Blue 追蹤品質差（模型 confidence 低），需更多訓練資料 (4) 主 GUI `app.py` 整合 `reattribute_shooters()` 呼叫
4. **阻礙？** 射手歸因率受限於 `SCORE_MAX_SHOOTER_DIST`（需 4K 影片實測確定最佳值）；Blue 偵測品質受限於訓練資料不足
5. **檔案？** `scoring.py`（`reattribute_shooters()`）、`robot_tracker.py`（`merge_fragmented_labels()` + 子方法）、`test_analysis.py`（驗證工具）、`config.py`（`SCORE_MAX_SHOOTER_DIST`）

---

## Session: 2026-02-23 (2)

### 完成項目
- [x] ByteTrack 繞過 — 完全重寫 MOT 追蹤邏輯，繞過 ByteTrack（IoU 匹配在 4K@60fps 完全失效），改用距離式直接匹配 `_match_direct()`
- [x] 全域最短距離匹配 — 距離矩陣 + 貪心最短優先（取代順序依賴的逐一匹配）
- [x] 動態距離閾值 — 依幀間隔縮放 ReID 距離（`base * (1 + sqrt(gap/fps))`）
- [x] 混合偵測 — 全幀偵測每幀 + tiled 偵測每 N 幀，NMS 去重
- [x] 後處理 label 合併 — `merge_fragmented_labels()` 合併碎片化的追蹤 label
- [x] Blue 機器人偵測 — 降低閾值 ROBOT_DETECTION_CONFIDENCE 0.25→0.10，對齊 BYTETRACK_TRACK_THRESH
- [x] test_analysis.py 增強 — 互動式 Hub 區域框選、出手偵測、合併統計、ID 穩定性評估

### 修改檔案
- `config.py` — ROBOT_DETECTION_CONFIDENCE 0.25→0.10, BYTETRACK_TRACK_THRESH 0.25→0.10, BYTETRACK_MATCH_THRESH 0.8→0.3, BYTETRACK_MIN_CONSECUTIVE 3→1, 新增 MOT_DETECT_INTERVAL=10, MOT_REID_MAX_DIST=400(動態縮放), MOT_MIN_TRACK_FRAMES=3
- `robot_tracker.py` — 完全重寫 _MOTTracker：_match_direct(), _match_bytetrack(), _allocate_label(), merge_fragmented_labels(), 混合偵測
- `robot_detection.py` — tile overlap 0.2→0.15
- `test_analysis.py` — 互動式 zone drawing, --no-zones flag, detect_shots() 呼叫, 後處理統計
- `app.py` — 新增 merge_fragmented_labels() 呼叫

### 測試結果
測試影片：3840x2160, 60fps, 前20秒

| 指標 | 修復前 (ByteTrack) | 修復後 |
|------|-------------------|--------|
| Ball 偵測 | 26044 (100%) | 26044 (100%) |
| Robot 原始偵測 | 11 | 1000 |
| Robot 插值後 | 11 | 4521 |
| Robot 幀覆蓋率 | 5.4% | 100% |
| Blue labels | 0 | 5 (2 stable) |
| 出手偵測 | 3 | 59 |
| ID 穩定性 | POOR (18) | FAIR (11 stable) |
| 速度 | 1 fps | 5.4 fps |

### 5-Question Reboot Check
1. **做什麼？** 修復 4K@60fps 機器人追蹤失效（ByteTrack IoU 失效 + Blue 機器人閾值過高）
2. **進度？** 核心問題已修復。Robot 幀覆蓋率 5.4%→100%，Blue 機器人從 0 到 5 labels。ID 穩定性從 POOR 提升到 FAIR
3. **下一步？** (1) ID 穩定性優化 — 11 stable labels 仍多於期望的 ≤6，需更好的 ReID 或更進階的 label 合併策略 (2) 進球偵測 — 用互動式 zone drawing 測試完整進球判定流程 (3) 主 GUI 整合測試 — app.py 尚未完整測試新追蹤邏輯 (4) Blue 機器人模型品質 — confidence ~0.11 偏低，可能需補充訓練資料
4. **阻礙？** ID 穩定性需更好的模型或 ReID 特徵；Blue 偵測覆蓋率受限於模型品質
5. **檔案？** `robot_tracker.py`（_match_direct + merge_fragmented_labels）、`config.py`（MOT_* 參數）、`test_analysis.py`（驗證工具）、`app.py`（merge 呼叫整合）

---

## Session: 2026-02-23

### 完成項目
- [x] 球偵測失效修復 — MAX_BLOB_AREA 10000->50000（4K 影片球面積超過上限被過濾）、GaussianBlur(5,5) 預處理改善運動模糊偵測、形態學從 Open->Close 改為 Close->Open（先補洞再去噪），核大小 Close(7,7)+Open(3,3)
- [x] GPU 加速 — OpenCL UMat GPU 加速球偵測（5 個影像處理操作全在 GPU 執行）、ONNX Provider 加入 DmlExecutionProvider（CUDA > DML > CPU 自動選擇）
- [x] ONNX Provider 診斷 — robot_detection.py 和 detection.py 的 __init__ 印出實際使用的 Provider + 類別名稱
- [x] 分析管線診斷日誌 — _run_analysis() 每 100 幀輸出球/機器人偵測統計，分析結束印出總計
- [x] MOT 自動偵測模式 — _MOTTracker 加入 auto_mode，未標記的 tracker_id 自動分配 Red-N/Blue-N label；RobotTrackerManager 加入 enable_auto_mode() 和 robot_info property；app.py 未標記機器人但 MOT 可用時自動啟用，分析後將自動偵測的機器人加入 _robot_markers
- [x] 用戶 GPU 環境確認 — NVIDIA GeForce RTX 3050 Ti Laptop GPU，安裝 onnxruntime-directml，確認 DmlExecutionProvider 啟用成功

### 修改檔案
- `config.py` — MAX_BLOB_AREA 10000->50000（4K 支援）
- `detection.py` — OpenCL UMat GPU 加速 + GaussianBlur 預處理 + 形態學 Close->Open 改善 + DmlExecutionProvider + Provider 診斷 print
- `robot_detection.py` — DmlExecutionProvider 加入 provider 列表 + __init__ 診斷 print
- `robot_tracker.py` — MOT 自動偵測模式（auto_mode、enable_auto_mode()、robot_info property）
- `app.py` — 分析管線診斷日誌（每 100 幀統計）+ MOT 自動模式啟用 + auto_robots 傳遞至 _robot_markers
- `requirements.txt` — 加入 onnxruntime-gpu / onnxruntime-directml 安裝說明
- `README.md` — 架構更新 + 模型訓練指南

### 5-Question Reboot Check
1. **做什麼？** 修復 4K 影片球偵測失效 + GPU 加速 + MOT 自動偵測模式
2. **進度？** 全部完成。球偵測已修復（MAX_BLOB_AREA + 形態學改善），DML GPU 加速已啟用，MOT auto_mode 可自動分配機器人 label
3. **下一步？** 用實際比賽影片端到端驗證：球偵測精度（4K 下 HSV + GaussianBlur）→ MOT 自動模式的 Red-N/Blue-N 分配是否正確 → 進球判定 + 出手偵測 + 命中率統計 → 調整 ByteTrack 參數
4. **阻礙？** 無明確阻礙；需實際影片驗證各模組整合效果
5. **檔案？** `detection.py`（球偵測 + OpenCL UMat）、`robot_tracker.py`（MOT auto_mode）、`app.py`（診斷日誌 + auto_mode 整合）、`config.py`（MAX_BLOB_AREA）

---

## Session: 2026-02-21

### 完成項目
- [x] GCP 抵免額用途研究 — 評估 Google Cloud Free Trial (7000 TWD) 和 GenAI App Builder (30000 TWD) 對本專案的價值
  - GCP Free Trial：可用於 GPU 訓練（T4 ~$0.54/hr），但 Colab 免費 T4 已足夠
  - GenAI App Builder (30000 TWD)：僅涵蓋 Vertex AI Search & Conversation，對本專案無用
  - Gemini Vision API / Video Intelligence API：比現有 HSV+YOLO 方案差，不值得用
  - 結論：只有 GPU 訓練值得做，且 Colab 免費 T4 已能完成
- [x] 建立 Colab 訓練 Notebook (`train_colab.ipynb`) — 在 Google Colab 免費 T4 GPU 上訓練 YOLOv11n 機器人偵測模型
  - 資料集：Main Robot Detection (1,172 張, Red/Blue 兩類)
  - 已加入 `.gitignore`（含 Roboflow API key）
- [x] 設定 gcloud CLI — 安裝、登入 (redacted@example.com)、設定專案 frc-project-484514、啟用 Compute Engine API、確認 GPU quota=1
- [x] **模型訓練完成** — 用戶在 Colab T4 GPU 完成訓練，產出 `frc_robot.onnx` (10.1 MB)
  - 類別：['Blue', 'Red']，輸入 640x640
  - 複製到 `models/frc_robot.onnx`，推理驗證通過
  - **解決了專案 #1 blocker：MOT 模式（YOLO + ByteTrack）現在可用**

### 修改檔案
- `train_colab.ipynb` — **新建** — Google Colab 訓練 notebook（Roboflow 下載 → YOLOv11n 訓練 → ONNX 匯出 → 下載）
- `.gitignore` — 新增 `train_colab.ipynb` 排除規則（含 API key 不可提交）
- `models/frc_robot.onnx` — **新增** — 訓練好的機器人偵測模型 (10.1 MB, Blue/Red, 640x640)

### 5-Question Reboot Check
1. **做什麼？** 研究 GCP 用途 + 建立 Colab 訓練流程 + 取得機器人偵測模型
2. **進度？** 模型訓練完成，MOT 模式的 #1 blocker 已解除。GCP 研究結論：Colab 免費 T4 足夠，不需要花 GCP 額度
3. **下一步？** 用實際比賽影片端到端測試 MOT 模式（YOLO+ByteTrack）→ 驗證 Red/Blue 偵測精度 → 調整 ByteTrack 參數 → 測試出手偵測和命中率統計
4. **阻礙？** 無明確阻礙；MOT 模式已有模型可用，需實際影片驗證效果
5. **檔案？** `models/frc_robot.onnx`（機器人偵測模型）、`robot_detection.py`（偵測器）、`robot_tracker.py`（MOT/SOT 管理器）、`scoring.py`（出手偵測）、`train_colab.ipynb`（Colab 訓練流程）

---

## Session: 2026-02-20

### 完成項目
- [x] 診斷播放瓶頸 — 影片為 4K (3840x2160) HEVC 60fps，OpenCV 解碼只有 26fps（連 1x 都跑不到），LANCZOS4 縮放每幀吃 14ms
- [x] 重構 `_show_frame` → `_show_frame` + `_render_frame` — 將 seek+read 與渲染分離，播放時可用順序讀取而非每幀 seek
- [x] 新增 `_render_frame_playback` 快速渲染路徑 — 播放專用：INTER_LINEAR（比 LANCZOS4 快 12 倍）、cv2.putText 取代 PIL Draw、精簡 overlay
- [x] 重寫 `_play_loop` — 小間距用 grab() 跳過中間幀、大間距用 seek、固定 30fps 顯示率
- [x] 之前 session 未提交的修改 — 全頁面分頁重構、HSV slider 原地更新修復、播放速度 1-5x

### 修改檔案
- `app.py` — 播放系統重構（_show_frame 拆分為 _show_frame + _render_frame、新增 _render_frame_playback 快速路徑、_play_loop 優化：grab() 跳幀 + 固定 30fps 顯示率）
- `settings_window.py` — SettingsPanel 嵌入式重構（上個 session 未提交，本次包含）

### 5-Question Reboot Check
1. **做什麼？** 修復 4K HEVC 60fps 影片的播放流暢度問題（解碼瓶頸 + 縮放瓶頸）
2. **進度？** 播放系統重構完成，LANCZOS4→LINEAR、PIL→cv2.putText、grab() 跳幀、30fps 顯示率
3. **下一步？** 用實際影片驗證播放流暢度 → 測試 1-5x 各速度 → 驗證 seek vs grab 的邊界條件 → 測試暫停/拖曳/逐幀仍正常
4. **阻礙？** 無明確阻礙；需實際影片驗證各種播放情境
5. **檔案？** `app.py`（`_show_frame`、`_render_frame`、`_render_frame_playback`、`_play_loop`）、`settings_window.py`（SettingsPanel）

---

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
*Last updated: 2026-02-27*

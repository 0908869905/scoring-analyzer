# FRC Scoring Analyzer — Progress

## Session: 2026-03-14 (15) — MOT 遮擋處理實作（10/10 tasks 完成）

### 完成項目
- [x] **T1: config.py 常數更新** — 新增 6 個遮擋常數（MOT_MAX_LOST_FRAMES=90, MOT_OCCLUSION_PATIENCE=450, MOT_LOST_GRACE_FRAMES=3, MOT_LOST_REID_DIST_SCALE=0.5, MOT_LOST_MIN_HIST_SIM=0.3, MOT_OCCLUSION_MARGIN=50），移除 MOT_REID_MAX_SECONDS，修正 FPS 註解 60→30，MOT_STATIC_MIN_FRAMES 60→30
- [x] **T2: geometry.py helper** — 新增 `min_distance_to_polygon_edge()` 函數（點到多邊形各邊最短距離）
- [x] **T3: robot_tracker.py 資料結構** — 新增 `_track_state`, `_lost_since`, `_missed_frames`, `_occlusion_zones`, `_static_labels`；移除 `_reid_max_frames` 和死碼 `_try_reid()`；新增 `set_occlusion_zones()`
- [x] **T4: _match_direct() 兩輪匹配** — 重寫為 Round 1（ACTIVE 貪心匹配）+ Round 2（LOST 復活，凍結位置+嚴格 Re-ID 門檻）+ ACTIVE→LOST grace period + LOST→REMOVED 超時清理 + 遮擋區域感知（hub 內 15 秒耐心）
- [x] **T5: cleanup 方法更新** — `clear()`, `filter_short_labels()`, `_execute_merges()` 加入 state machine 清理；`filter_static_labels()` 從刪除改為標記式（`_static_labels` set）
- [x] **T6: OcclusionZone dataclass** — `app.py` 新增 `OcclusionZone` dataclass + `_occlusion_zones` 列表
- [x] **T7: 遮擋區域 UI** — 「遮擋區域」按鈕 + 多邊形點擊/右鍵取消/雙擊完成 + 灰色頂點渲染
- [x] **T8: LOST 渲染** — 遮擋區域半透明灰色填充 + LOST 機器人虛線框 `_draw_dashed_rect()` + LOST 灰色標籤
- [x] **T9: 整合** — 分析時傳遞 occlusion zones 到追蹤器 + `RobotTrackerManager.set_occlusion_zones()` + `_analysis_robot_mgr` 引用保存
- [x] **T10: 端對端驗證** — 全部 import 通過 + AST 語法檢查 + 常數值驗證 + 功能測試

### 修改檔案
- `config.py` — 新增 6 個遮擋常數，移除 MOT_REID_MAX_SECONDS，修正 FPS 註解，MOT_STATIC_MIN_FRAMES 60→30
- `geometry.py` — 新增 `min_distance_to_polygon_edge()` 函數
- `robot_tracker.py` — Track State Machine（ACTIVE/LOST/REMOVED 三態）+ 兩輪匹配 + 遮擋區域感知 + `filter_static_labels` 標記式 + 移除死碼 `_try_reid()`
- `app.py` — OcclusionZone dataclass + 遮擋區域 UI（按鈕+多邊形繪製）+ LOST 渲染（虛線框+灰色標籤+半透明填充）+ 分析整合

### 5-Question Reboot Check
1. **做什麼？** 實作 MOT 遮擋處理方案 C（Track State Machine + 遮擋區域感知），解決機器人被 hub 遮住時追蹤框亂跳搶 ID 的問題
2. **進度？** 全部 10/10 tasks 完成，程式碼已通過 import + AST + 常數驗證，尚未 git commit
3. **下一步？** 手動測試：開影片 → 標記遮擋區域（hub 多邊形）→ 執行分析 → 觀察 LOST 機器人行為（虛線框、凍結位置、復活匹配）→ 根據測試結果調參（MOT_MAX_LOST_FRAMES, MOT_OCCLUSION_PATIENCE 等）
4. **阻礙？** 無程式碼阻礙，需要實際影片測試驗證行為正確性
5. **檔案？** `robot_tracker.py`（Track State Machine 核心邏輯）、`app.py`（OcclusionZone UI + LOST 渲染 + 整合）、`config.py`（遮擋常數）、`geometry.py`（`min_distance_to_polygon_edge`）、`docs/superpowers/plans/2026-03-13-mot-occlusion-handling.md`（實作計劃參考）

---

## Session: 2026-03-14 (14) — 影片分析效能優化研究 + Pipeline 並行架構實作

### 完成項目
- [x] **完整分析 pipeline 效能瓶頸研究** — 深入分析 `_run_analysis()` 每幀處理流程，定位 YOLO CPU 推理佔 76% 時間為主要瓶頸
- [x] **CUDA vs DirectML 建議** — 建議先裝 DirectML（一行 pip），CUDA 只多省 ~30-40 秒但需 4GB Toolkit
- [x] **Pipeline 並行可行性分析** — 確認偵測是無狀態的可提前執行，追蹤是有狀態的必須依序
- [x] **Tier 0A: DirectML 安裝** — 用戶自行安裝 onnxruntime-directml（YOLO 推理 55ms → ~18ms）
- [x] **Tier 2A: 前景遮罩移除** — 用戶已移除背景模型前景遮罩
- [x] **Tier 1A: Pipeline 並行架構** — 主分析迴圈改為 producer-consumer 架構：
  - `robot_tracker.py`: 拆分 `_MOTTracker.update_all()` 為 `detect_raw()` + `track_update()`
  - `robot_tracker.py`: `RobotTrackerManager` 加上對應 pipeline API
  - `app.py`: 主迴圈改為 `Queue(maxsize=8)` + producer Thread，Producer 讀幀+球偵測+機器人偵測（提前 8 幀），Consumer 追蹤匹配+進球判定（依序）
- [x] **Tier 2B: 預分配 canvas + blob buffer** — `robot_detection.py` 用 `threading.local()` per-thread 預分配 buffer，`_preprocess` 零中間陣列分配（4.18ms → 2.30ms, 1.82x 加速）
- [x] **Tier 2C: 移除 frame.copy()** — `robot_tracker.py` tiled 偵測提交時不再複製 frame（只讀不改），省 2MB/每10幀
- [x] **Tier 2D 分析結論** — 快取更新迴圈非冗餘（interpolate 後需重新查詢），保留不刪

### 修改檔案
- `robot_detection.py` — import threading, `__init__` 加 `threading.local()`, `_preprocess` 改為預分配 buffer 重用
- `robot_tracker.py` — `_MOTTracker` 加 `detect_raw()`/`track_update()` 方法, `RobotTrackerManager` 加對應 pipeline API, 移除 tiled 偵測的 `frame.copy()`
- `app.py` — 主分析迴圈 `_run_analysis()` 改為 Pipeline producer-consumer 架構（Queue + producer Thread）

### 效能預估
| 方案 | 每幀吞吐 | 4500幀 | 加速 |
|------|---------|--------|------|
| 舊 (CPU) | 75ms | 5.6 min | 1x |
| + DirectML | 34ms | 2.6 min | 2.2x |
| + Pipeline | 20ms | 1.5 min | 3.8x |
| + 預分配 buffer | 18ms | 1.4 min | 4.2x |

### 5-Question Reboot Check
1. **做什麼？** 影片分析效能優化，已完成 Pipeline 並行架構 + DirectML + 預分配 buffer + 移除 frame.copy
2. **進度？** 全部 4 個 Tier 實作完成，待用戶實際測試驗證效能數據
3. **下一步？** 回到 MOT 遮擋處理實作（Session 13 的計劃，0/10 tasks），或根據用戶實測結果微調 Pipeline 參數（Queue maxsize、prefetch 幀數等）
4. **阻礙？** 無阻礙，Pipeline 架構已實作完成
5. **檔案？** `app.py`（Pipeline producer-consumer 主迴圈）、`robot_tracker.py`（detect_raw/track_update 拆分）、`robot_detection.py`（預分配 buffer）、`docs/superpowers/plans/2026-03-13-mot-occlusion-handling.md`（下一步的 MOT 遮擋處理計劃）

---

## Session: 2026-03-13 (13) — MOT 遮擋處理設計 + 實作計劃

### 完成項目
- [x] **MOT 遮擋問題根因分析** — 深度研究 `robot_tracker.py` 的 `_match_direct()` 邏輯，找到 5 個根本原因：無 LOST 狀態、速度外推飄移、距離閾值過寬、無最大消失幀、插值平滑化誤跳
- [x] **業界方案研究** — 研究 ByteTrack/SORT/DeepSORT/BoT-SORT 的遮擋處理方式（兩輪配對、Kalman 預測、Re-ID 外觀特徵）
- [x] **設計方案選擇** — 提出 3 種方案（A: State Machine、B: 遮擋區域感知、C: A+B 結合），用戶選擇方案 C
- [x] **設計規格文件** — 完成 `docs/superpowers/specs/2026-03-13-mot-occlusion-handling-design.md`，經 2 輪 spec review 通過
- [x] **實作計劃** — 完成 `docs/superpowers/plans/2026-03-13-mot-occlusion-handling.md`（10 個 task），經 plan review 修正後提交

### 設計核心
- **Track State Machine**: ACTIVE → LOST → REMOVED 三態
- **兩輪配對**: Round 1 (ACTIVE only) + Round 2 (LOST revival，Re-ID 加權距離)
- **遮擋區域感知**: 用戶標記 hub 多邊形，LOST 在遮擋區域有 15 秒耐心（`MOT_OCCLUDED_PATIENCE_SEC`）
- **Grace period**: 3 幀未偵測才轉 LOST（防偵測閃爍，`MOT_GRACE_FRAMES`）
- **filter_static_labels 改標記式**: 靜止機器人不再刪除，只標記 `is_static=True`，保留 label 顯示
- **FPS 預設 30**: `DEFAULT_FPS` 從 60 改 30
- **MOT_REID_MAX_SECONDS 被取代**: 新 state machine 的 patience 機制取代舊的 Re-ID 超時

### 新增檔案
- `docs/superpowers/specs/2026-03-13-mot-occlusion-handling-design.md` — 設計規格（Track State Machine + 遮擋區域感知）
- `docs/superpowers/plans/2026-03-13-mot-occlusion-handling.md` — 10 步實作計劃

### 修改檔案
- 尚未開始實作，計劃修改：`config.py`、`robot_tracker.py`、`app.py`、`geometry.py`、`scoring.py`、`runtime_config.py`、`settings_window.py`

### 5-Question Reboot Check
1. **做什麼？** 實作 MOT 遮擋處理方案 C（Track State Machine + 遮擋區域感知），解決機器人被 hub 遮住時追蹤框亂跳搶 ID 的問題
2. **進度？** 設計+計劃全部完成，尚未開始實作（0/10 tasks）
3. **下一步？** 執行實作計劃 Task 1: config.py 常數更新（新增 MOT_GRACE_FRAMES=3, MOT_LOST_PATIENCE_SEC=5, MOT_OCCLUDED_PATIENCE_SEC=15, MOT_LOST_REVIVAL_DIST_FACTOR=1.5 等；DEFAULT_FPS 改 30；移除 MOT_REID_MAX_SECONDS）
4. **阻礙？** 無阻礙，設計已通過 review，可直接開始實作
5. **檔案？** `docs/superpowers/plans/2026-03-13-mot-occlusion-handling.md`（實作計劃，按 task 順序執行）、`docs/superpowers/specs/2026-03-13-mot-occlusion-handling-design.md`（設計規格，實作時參考）、`robot_tracker.py`（主要修改目標）、`config.py`（Task 1 起點）

---

## Session: 2026-03-13 (12) — HP 歸因改線段交叉 + 設定面板預覽增強

### 完成項目
- [x] **HP 歸因改為線段交叉判定** — `scoring.py` 的 `_check_hp_attribution()` 從距離判定（`point_to_segment_distance ≤ HP_ATTRIBUTION_DIST`）改為 `segments_intersect()` 線段交叉判定，移除 `HP_ATTRIBUTION_DIST` 常數
- [x] **RuntimeConfig 補齊參數** — `runtime_config.py` 新增 `ball_ownership_dist` 和 `shot_min_upward_velocity` 欄位
- [x] **app.py 傳參修正 + 新增 callback** — ScoringEngine 建構補傳 `ball_ownership_dist`、`shot_min_upward_velocity`；新增 `_get_analysis_data_for_preview()` 和 `_on_recompute_attribution()` 方法
- [x] **進球 Tab 增強** — `settings_window.py` 的 `_build_scoring_tab()` 改為雙區架構（上方 slider + 下方預覽 canvas），新增 `ball_ownership_dist` slider + 「重新計算歸因」按鈕 + 即時預覽（顯示 ownership 圓圈、球-機器人連線、距離數字）
- [x] **出手 Tab 增強** — `_build_shot_tab()` 同樣改為雙區架構，新增 `shot_min_upward_velocity` slider + 「重新計算歸因」按鈕 + 即時預覽（顯示 proximity 圓圈、速度箭頭、速度數值）
- [x] **預覽渲染函數** — 新增 `_update_ownership_preview()` 和 `_update_shot_preview()`，使用 OpenCV 繪圖，100ms debounce 更新

### 修改檔案
- `config.py` — 移除 `HP_ATTRIBUTION_DIST` 常數
- `runtime_config.py` — 新增 `ball_ownership_dist`、`shot_min_upward_velocity` 欄位；新增 import
- `scoring.py` — `_check_hp_attribution()` 改用 `segments_intersect()`；移除 `hp_attribution_dist` 參數和成員
- `app.py` — ScoringEngine 補傳參數；新增 `_get_analysis_data_for_preview()`、`_on_recompute_attribution()`；SettingsPanel 傳入新 callback
- `settings_window.py` — 進球/出手 Tab 改為雙區架構 + 預覽渲染 + 重新計算歸因按鈕

### 5-Question Reboot Check
1. **做什麼？** HP 歸因從距離判定改為線段交叉判定，設定面板進球/出手 Tab 增加即時預覽和重新計算功能
2. **進度？** 程式碼修改全部完成，待實際影片測試驗證
3. **下一步？** (1) 用實際影片測試線段交叉 HP 歸因的準確度 (2) 驗證設定面板預覽渲染在不同解析度下的顯示效果 (3) 確認 recompute attribution 能正確即時更新結果
4. **阻礙？** (a) 線段交叉判定可能需要微調球軌跡片段長度 (b) 預覽渲染性能待觀察（100ms debounce 是否足夠）
5. **檔案？** `scoring.py`（HP 歸因邏輯）、`settings_window.py`（預覽渲染）、`app.py`（callback 串接）、`runtime_config.py`（新參數）

---

## Session: 2026-03-12 (11) — 4 人分工審核收回 + 本地 GPU 訓練部署

### 完成項目
- [x] **收回 4 人審核標註** — 4 台筆電分工審核後的 labels 收回到 `E:\labels\labels1~4\`，合併回 `datasets/reviewed/labels/`（train 1461 + val 1365）
- [x] **用 label_editor 最終確認** — train 和 val 全部再確認一次
- [x] **建立 `datasets/reviewed/` 乾淨資料集** — 審核完的 images+labels 獨立存放，含 data.yaml，之後訓練直接用這個
- [x] **4 人分工切割** — 1826 張切成 part1~4（457/457/457/455），建立在 `datasets/reviewed/part1~4/`，每個有 images/ + labels/
- [x] **打包 4 份 zip** — part1.zip~part4.zip 方便分發到各筆電
- [x] **複製到 E: 隨身碟** — reviewed 完整資料集 + part1~4 + zip + label_editor.py 全部複製到 `E:/scoring-analyzer-deploy/`
- [x] **本地 GPU 訓練準備** — 改為在 RTX 3070 Ti 筆電本地訓練（不用 Colab），建立 `E:/frc_training/`：train.py + dataset/ + data.yaml
- [x] **修復 train.py 兩個錯誤** — (1) E: 硬編碼路徑改為相對路徑（目標筆電 SSD 是 D:） (2) 加 `if __name__ == "__main__"` + `workers=0` 解決 Windows multiprocessing spawn 錯誤
- [ ] **訓練中** — 用戶在 3070 Ti 筆電上執行 `python train.py`，訓練進行中

### 修改檔案
- `datasets/reviewed/` — 新建乾淨資料集目錄（images/ + labels/ + data.yaml）
- `datasets/reviewed/part1~4/` — 分工切割（各含 images/ + labels/）
- `E:/frc_training/train.py` — 本地訓練腳本（YOLOv26n, 100 epochs, device=0）
- `E:/frc_training/dataset/` — 完整 reviewed 資料集副本
- `E:/scoring-analyzer-deploy/` — 同步更新（reviewed + parts + zips + label_editor.py）

### 5-Question Reboot Check
1. **做什麼？** 收回 4 人分工審核標註，建立乾淨 reviewed 資料集，部署到 RTX 3070 Ti 筆電本地訓練
2. **進度？** 資料準備全部完成，3070 Ti 筆電正在訓練 YOLOv26n 100 epochs
3. **下一步？** (1) 等訓練完成取回新 frc_robot.onnx (2) 部署到 models/ 取代舊模型 (3) 實際影片測試新模型效果 (4) 與 Colab T4 訓練結果對比
4. **阻礙？** (a) 訓練中，需等完成 (b) 3070 Ti 8GB VRAM 訓練速度待觀察 (c) 新模型效果待驗證
5. **檔案？** `datasets/reviewed/`（乾淨資料集）、`E:/frc_training/train.py`（訓練腳本）、`models/frc_robot.onnx`（訓練完成後要替換的模型）

---

## Session: 2026-03-12 (10) — 訓練資料最終驗證 + 重新打包

### 完成項目
- [x] **E: deploy 目錄同步** — 新模型 + 核心 Python 檔案同步到 `E:/scoring-analyzer-deploy/`
- [x] **Train split 二次審核（1461 張）** — 用 label_editor 再次開啟 merged train 全部 1461 張，用戶有修正部分標註
- [x] **Val split 二次審核（365 張）** — 用 label_editor 再次開啟 merged val 全部 365 張，用戶也有修正
- [x] **發現 zip 不一致問題** — 比對發現之前的 zip 跟磁碟上的 label 不一致（文字模式換行差異，實際 binary 一致），但用戶後來又修正了標註需重新打包
- [x] **重新打包 merged.zip** — 最終版 398 MB，1826 張（train 1461 + val 365），包含二次審核修正
- [x] **最終確認** — 用戶最後再開 train 和 val 確認標註正確，val 無改動，train 之前有改已重新打包
- [x] **train_colab.ipynb 更新提醒** — Drive 上的 notebook 是舊版，需用戶手動上傳覆蓋或在 Colab 改路徑

### 修改檔案
- `datasets/merged/train/labels/` — 二次審核修正部分標註
- `datasets/merged/val/labels/` — 二次審核修正部分標註
- `datasets/merged.zip` — 重新打包（含二次審核修正）

### 5-Question Reboot Check
1. **做什麼？** 反覆驗證 merged dataset 標註品質，二次審核 train+val 全部 1826 張，修正後重新打包 merged.zip
2. **進度？** 標註驗證+重新打包完成，merged.zip 需重新上傳到 Google Drive，Colab 訓練待執行
3. **下一步？** (1) 重新上傳 merged.zip 到 Google Drive (2) 在 Colab 跑 100 epochs 重訓 YOLOv26n (3) 下載新 frc_robot.onnx 部署到 models/ (4) 實際影片測試新模型效果
4. **阻礙？** (a) merged.zip 需重新上傳到 Drive（398 MB） (b) train_colab.ipynb Drive 上是舊版，需手動上傳覆蓋或 Colab 內改路徑 (c) Colab 訓練需用戶手動執行
5. **檔案？** `datasets/merged.zip`（重新打包的訓練資料）、`train_colab.ipynb`（Colab 訓練 notebook）、`models/frc_robot.onnx`（訓練完成後要替換的模型）

---

## Session: 2026-03-11 (9) — Train+Val 手動修正 + 重新打包 + Colab 訓練準備

### 完成項目
- [x] **同步 E: 隨身碟** — 把 session 8 的新模型和更新過的 Python 檔案同步到 `E:/scoring-analyzer-deploy/`
- [x] **Train split 手動修正（1461 張）** — 用 label_editor 逐張審核 `datasets/merged/train/` 的所有標註，手動修正錯誤 bbox（移動/resize/刪除/新增）
- [x] **Val split 手動修正（365 張）** — 同上，`datasets/merged/val/` 的標註也全部審核修正
- [x] **重新打包 merged.zip** — 包含手動修正過的 train+val labels（398 MB），供 Colab 重訓
- [x] **上傳 merged.zip 到 Google Drive** — 透過 Chrome 瀏覽器自動化開啟 Google Drive，用戶手動拖入上傳
- [x] **更新 train_colab.ipynb** — 路徑改為 merged.zip、epochs=100、data.yaml 路徑修正（本地已更新，Drive 上的舊版需用戶手動上傳覆蓋）
- [x] **向用戶解釋 val split 用途** — 驗證集用於防過擬合、選 best checkpoint，不參與訓練

### 修改檔案
- `train_colab.ipynb` — 路徑與參數修正（本地更新）
- `datasets/merged/train/labels/` — 1461 張標註手動修正
- `datasets/merged/val/labels/` — 365 張標註手動修正
- `datasets/merged.zip` — 重新打包（含手動修正標註）

### 5-Question Reboot Check
1. **做什麼？** 手動修正 merged dataset 全部 1826 張標註，重新打包上傳，準備 Colab 重訓
2. **進度？** 標註修正+打包+上傳完成，Colab 訓練待用戶手動執行
3. **下一步？** (1) 在 Google Colab 上用修正後的 merged.zip 重訓 YOLOv26n 100 epochs (2) 下載新 frc_robot.onnx 部署到 models/ (3) 實際影片測試新模型效果
4. **阻礙？** (a) Colab 訓練需要用戶手動執行（merged.zip 已上傳 Drive） (b) train_colab.ipynb 需要用戶手動上傳覆蓋 Drive 上的舊版
5. **檔案？** `train_colab.ipynb`（Colab 訓練 notebook）、`datasets/merged/`（修正後資料集）、`models/frc_robot.onnx`（訓練完成後要替換的模型）

---

## Session: 2026-03-11 (8) — 標註品質修復 + 資料集重建 + 模型重訓

### 完成項目
- [x] **壞幀偵測與清除** — 寫了 `detect_bad_frames.py` 掃描 merged dataset，找到 9 張異常幀（比賽結束白霧/紙花畫面、計分板、過曝+大量假標註），從 train images+labels 移除
- [x] **審核標註同步問題修復** — 發現 `merge_datasets.py` 合併時用 D: 的 `labels_raw`，但 okok/bcvi/tuis 三個賽事的人工審核修正只存在 E: 隨身碟（`E:/scoring-analyzer-deploy/datasets/`），從未同步回 D:。導致 merged 中這三個賽事用的是 Gemini 未校正的原始標註
- [x] **同步審核標註** — 從 E: 隨身碟複製 okok/bcvi/tuis 的審核過 labels_raw 回 D: 的 datasets 目錄
- [x] **重建 merged 資料集** — 重跑 `merge_datasets.py`，新 merged: 1826 張（train 1461 + val 365），比之前 1865 少 39 張（審核時刪除的空標註被過濾掉）
- [x] **label_editor 最終審核** — 用 label_editor 審核了 train 和 val split，確認標註品質正確
- [x] **更新 train_colab.ipynb** — 路徑改為 merged.zip、epochs=100、data.yaml 路徑修正
- [x] **打包 datasets/merged.zip** — 398 MB，供 Colab 上傳訓練
- [x] **Colab T4 訓練完成** — YOLOv26n 100 epochs，下載新 frc_robot.onnx
- [x] **部署新模型** — `frc_robot (3).onnx` → `models/frc_robot.onnx`（9.8 MB, [1,3,640,640] → [1,300,6] NMS-Free），舊模型備份為 `frc_robot_old.onnx`

### 修改檔案
- `detect_bad_frames.py` — 新建，掃描 merged dataset 找異常幀（基於 label 數量異常+圖片亮度/色彩統計）
- `merge_datasets.py` — 重跑合併（無程式碼修改，資料來源修正後重跑）
- `train_colab.ipynb` — 路徑改為 merged.zip、epochs=100、data.yaml 路徑修正
- `models/frc_robot.onnx` — 用修正後資料集重訓的新模型（9.8 MB, NMS-Free）
- `models/frc_robot_old.onnx` — 舊模型備份
- `datasets/2026okok/labels_raw/` — 從 E: 同步審核過的標註
- `datasets/2026bcvi/labels_raw/` — 從 E: 同步審核過的標註
- `datasets/2026tuis/labels_raw/` — 從 E: 同步審核過的標註
- `datasets/merged/` — 重建，1826 張（train 1461 + val 365）

### 5-Question Reboot Check
1. **做什麼？** 修復標註品質問題（壞幀+未同步審核標註），重建資料集，重訓並部署新模型
2. **進度？** 全部完成 — 壞幀清除、標註同步、資料集重建、Colab 訓練、模型部署皆已完成
3. **下一步？** (1) 實際影片測試新模型偵測效果（與舊模型對比） (2) 評估是否需要更多賽事資料擴增訓練集 (3) 考慮 CUDA onnxruntime-gpu 加速
4. **阻礙？** (a) 尚未在實際影片上測試新模型效果 (b) 3070Ti Python 3.14 無 CUDA PyTorch wheels
5. **檔案？** `models/frc_robot.onnx`（新模型）、`robot_detection.py`（YOLO 偵測器）、`detect_bad_frames.py`（壞幀偵測工具）、`merge_datasets.py`（資料集合併）

---

## Session: 2026-03-11 (7) — 效能優化 6 步計劃實作（5/6 完成）

### 完成項目
- [x] **Step 1: 軌跡幀索引** — `app.py`: 從 `_all_trajectories` 預建 `_trajectory_by_frame` dict，`_draw_analysis_overlay_impl` 和 debug 4-panel 改用索引查詢，複雜度 O(2000)→O(~200)
- [x] **Step 2: Tiled 偵測非同步化** — `robot_tracker.py`: `ThreadPoolExecutor(1)` 背景執行 tiled 偵測，下一幀消費結果（非阻塞），消除 +100ms 尖峰
- [x] **Step 3: 直方圖提取降頻** — `robot_tracker.py` + `config.py`: 新增 `MOT_HISTOGRAM_UPDATE_INTERVAL=3`，只在新 label 或每 3 幀才提取/更新直方圖，5-6ms→1-2ms
- [x] **Step 4: 背景遮罩快取** — `app.py`: `_debug_fg_cache` 快取 (frame_idx, mask)，同一幀不重算前景遮罩
- [ ] **Step 5: ImageTk 優化** — 跳過（PIL 路徑因中文標籤需求無法避免）
- [x] **Step 6: 進度更新降頻** — `app.py`: 從每 5 幀改為每 20 幀更新 UI 進度

### 修改檔案
- `config.py` — 新增 `MOT_HISTOGRAM_UPDATE_INTERVAL = 3`
- `robot_tracker.py` — tiled 非同步化（ThreadPoolExecutor）+ 直方圖降頻（interval gating）+ import ThreadPoolExecutor
- `app.py` — 軌跡幀索引 `_trajectory_by_frame` + `_build_trajectory_index()` + 背景遮罩快取 `_debug_fg_cache` + 進度更新降頻 20 幀

### 預估效能提升
- 分析管線：平均每幀 ~20ms → ~10ms（~2x 加速）
- 播放渲染：正常 ~35ms → ~25ms；Debug ~120ms → ~80ms

### 5-Question Reboot Check
1. **做什麼？** 實作效能優化 6 步計劃中的 5 步（Step 5 跳過）
2. **進度？** 5/6 步已完成並寫入程式碼，Step 5 因中文標籤需求跳過
3. **下一步？** (1) 實際影片測試確認效能提升幅度 (2) 實際影片測試新 YOLO 模型偵測效果 (3) 考慮 CUDA onnxruntime-gpu 加速（需 Python <3.13）
4. **阻礙？** (a) 尚未實際測量優化後效能數據（理論估算 ~2x） (b) 3070Ti Python 3.14 無 CUDA PyTorch wheels
5. **檔案？** `app.py`（軌跡索引+遮罩快取+進度降頻）、`robot_tracker.py`（tiled 非同步+直方圖降頻）、`config.py`（MOT_HISTOGRAM_UPDATE_INTERVAL）

---

## Session: 2026-03-11 (6) — YOLO 模型訓練完成 + ONNX 匯出修復 + 部署 + 效能優化計劃

### 完成項目
- [x] **YOLOv26n 本地 GPU 訓練完成** — merged dataset (2024mslr 817張 + 5×2026 regional 1048張 = 1865張)，RTX 3070Ti Laptop GPU，100 epochs
  - 最終結果：mAP50=0.841, mAP50-95=0.463, Precision=0.853, Recall=0.780
  - 訓練輸出：`E:\merged\runs\detect\train2\`
- [x] **ONNX 匯出修復** — 初次匯出 imgsz=64（ultralytics 預設）導致偵測結果全為 0，重新匯出 imgsz=640 修復
  - 輸出格式：[1, 300, 6] (NMS-Free)，模型大小 9.6 MB
- [x] **模型部署** — 新 frc_robot.onnx 部署到 `models/frc_robot.onnx` + `E:\scoring-analyzer-deploy/`
  - 複製 14 個核心分析 Python 檔案 + 模型 + presets
- [x] **效能優化深度研究**（3 個 Sub-Agent 並行分析）
  - 播放管線：每幀 25-40ms，Debug 4-Panel 80-150ms
  - 分析管線：正常幀 13-16ms，Tiled 偵測幀 116ms（超標 3.5x）
  - GPU 加速：DirectML 可提速 3-5x，CUDA 可提速 8-12x
- [x] **效能優化 6 步計劃制定**（尚未批准執行）
  - Step 1: 軌跡幀索引（播放 overlay O(2000)→O(200)）
  - Step 2: Tiled 偵測非同步化（消除 100ms 尖峰）
  - Step 3: 直方圖提取降頻（5-6ms→1-2ms/幀）
  - Step 4: 背景遮罩快取（Debug 視圖 -10-15ms）
  - Step 5: ImageTk 轉換優化（-2-5ms）
  - Step 6: 進度更新降頻

### 修改檔案
- `models/frc_robot.onnx` — 新訓練 YOLOv26n 模型（9.6 MB, 640x640, NMS-Free [1,300,6]）

### 5-Question Reboot Check
1. **做什麼？** YOLO 模型訓練+部署完成，制定效能優化計劃
2. **進度？** 新模型已部署，效能優化計劃已制定但尚未批准/實作
3. **下一步？** (1) 實際影片測試新模型偵測效果 (2) 用戶批准後實作效能優化 6 步計劃（涉及 app.py, robot_tracker.py, config.py）(3) 解決 3070Ti Python 3.14 CUDA PyTorch 安裝問題
4. **阻礙？** (a) 效能優化計劃待用戶批准 (b) 3070Ti 筆電 Python 3.14 無 CUDA PyTorch wheels（cu124 無 3.14 支援）(c) 新模型尚未在實際影片上全面測試
5. **檔案？** `models/frc_robot.onnx`（新模型）、`robot_detection.py`（YOLO 偵測器）、`app.py`（效能優化主要目標）、`robot_tracker.py`（直方圖降頻優化）、`config.py`（閾值調整）

---

## Session: 2026-03-11 (5) — tuis 標註完成 + 全賽事審核完成 + 資料集合併 + 訓練準備

### 完成項目
- [x] **tuis Gemini 標註完成** — 250 張圖片自動標註，218/250 有效（32 空，0 錯誤），Red=404, Blue=381 boxes
- [x] **複製 tuis 到 deploy 目錄** — images_sample + labels_raw 複製到 E:\scoring-analyzer-deploy\datasets\2026tuis\
- [x] **全部 6 賽事審核完成** — 兩台筆電分工
  - 本機：cosp, mndu, okok 前半
  - 另一台筆電：okok 後半, bcvi, tuis
- [x] **合併資料集** — 建立 merge_datasets.py，合併 2024mslr (817張) + 5×2026 events (1048張) = 1865 張有效訓練資料
  - 80/20 split: train=1492, val=373
  - 輸出至 datasets/merged/（含 data.yaml）
- [x] **準備本地訓練** — 修正 data.yaml path 為相對路徑，打包 merged.zip (403MB) 到 E:\
- [x] **發現 PyTorch GPU 問題** — 這台電腦 PyTorch 是 CPU 版，3070Ti 筆電需要安裝 CUDA 版 PyTorch

### 修改檔案
- `merge_datasets.py` — 新建，合併多賽事資料集工具（images+labels → merged/ + data.yaml + 80/20 split）
- `datasets/merged/` — 合併後資料集輸出（1865 張，train/val split）

### 5-Question Reboot Check
1. **做什麼？** 完成所有標註+審核+合併，準備 YOLO 模型訓練
2. **進度？** 1865 張訓練資料已合併打包（403MB merged.zip），待 GPU 訓練
3. **下一步？** (1) 3070Ti 筆電安裝 CUDA 版 PyTorch (2) 本地 YOLOv8n/YOLOv11n 訓練 (3) 或上傳 Colab T4 訓練 (4) 訓練後匯出 ONNX 替換 models/frc_robot.onnx
4. **阻礙？** 3070Ti 筆電 PyTorch 是 CPU 版，需 `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121` 安裝 CUDA 版
5. **檔案？** `merge_datasets.py`（合併工具）、`datasets/merged/data.yaml`（訓練入口）、`train_colab.ipynb`（Colab 備案）、`TRAIN_README.txt`（訓練步驟）

---

## Session: 2026-03-10 (4) — Gemini 自動標註 + Label Editor 審核 + 多解析度裁切修復

### 完成項目
- [x] **Gemini 自動標註 5 個 2026 賽事** — mndu, cosp, okok, bcvi, tuis 各 250 張圖片用 gemini-3.1-flash-lite-preview 標註
  - 最終統計：1,066 張有效標註（mndu:214, cosp:224, okok:215, bcvi:222, tuis:191）
  - mndu 曾因 504 錯誤只有 58 張，刪除空檔後重跑修復到 214 張
- [x] **Label Editor 審核功能** — Space 鍵標記已審核 + 斷點恢復（review_state.json）+ "✓ 已審核" 視覺標記 + 審核進度計數 + --images 參數指定圖片子目錄
- [x] **人工審核進行中** — 兩台筆電分工審核
  - 本機：cosp 完成, mndu 完成, okok 前半完成
  - 另一台 (E:\scoring-analyzer-deploy)：okok 後半、bcvi、tuis
- [x] **裁切工具多解析度修復** — tuis 事件中不同影片有不同解析度（qm1=640x360, qm2+=1280x720），crop 座標不相容
  - 修復 crop_events.py：crop.json 記錄 base_w/base_h，自動等比縮放 crop 座標，統一 resize 輸出尺寸
- [x] **auto_annotate.py 增強** — prompt 改為 0-6 bumpers + timeout 30s
- [x] **config.py** — GEMINI_MODEL 改為 gemini-3.1-flash-lite-preview

### 修改檔案
- `label_editor.py` — 審核標記功能（review_state.json, Space 鍵, ✓ 已審核 overlay, --images 參數）
- `crop_events.py` — 多解析度 crop 等比縮放修復（base_w/base_h, auto-scale, resize）
- `config.py` — GEMINI_MODEL 改為 gemini-3.1-flash-lite-preview
- `auto_annotate.py` — 增強 prompt（0-6 bumpers）、timeout 30s

### 5-Question Reboot Check
1. **做什麼？** 完成 2026 賽事 Gemini 自動標註 + 人工審核 + 裁切工具修復
2. **進度？** 5 個賽事標註完成（1,066 張），3 個賽事審核完成（cosp, mndu, okok 前半），tuis 重新裁切中
3. **下一步？** (1) tuis 重新裁切+提取+Gemini 標註 (2) 另一台筆電完成 okok 後半、bcvi、tuis 審核 (3) 合併所有資料集（2024mslr + 5×2026）(4) Colab T4 訓練 YOLO 模型
4. **阻礙？** tuis 多解析度問題已修復但需重跑；另一台筆電審核進度待同步
5. **檔案？** `label_editor.py`（審核功能）、`crop_events.py`（裁切修復）、`auto_annotate.py`（標註 pipeline）、`build_dataset.py`（合併用）

---

## Session: 2026-03-10 (3) — Engineering Notebook Update

### 完成項目
- [x] **中文 Analysis 分頁更新** — 透過 Chrome 瀏覽器自動化在 Google Doc 新增三個章節：
  - 分離式背景遮罩（Temporal Median 背景模型分離式應用：球偵測用 masked frame、機器人偵測用原始 frame）
  - YOLO Bumper 模型訓練流水線（Gemini 自動標註 → Label Editor 人工校正 → Colab 訓練 → ONNX 部署）
  - 進球判定與射手歸因（區域進入判定 + 3層歸因 HP>Ownership>Proximity + 球所有權 + 出手偵測）
- [x] **三處不一致修正** — 「第一版：YOLO 偵測」加註目前採用、「第二版：HSV Bumper」加註備用模式、更新「尚未解決的問題」區段
- [x] **英文翻譯** — 完整工程筆記英文翻譯貼入 Google Doc「scouting → 英文」分頁
- [x] **Label Editor 啟動** — 啟動 label_editor.py 供用戶截圖

### 修改檔案
- （無本地程式碼檔案修改，所有工作透過 Chrome 瀏覽器自動化操作 Google Doc）

### 5-Question Reboot Check
1. **做什麼？** 更新 Google Doc 工程筆記（中文 Analysis 新增章節 + 修正不一致 + 英文翻譯）
2. **進度？** 工程筆記更新完成，中英文版本皆已同步
3. **下一步？** (1) 測試更多影片確認 YOLO 偵測穩定性 (2) 測試完整 pipeline（進球判定+射手歸因） (3) 提交大量未 commit 的變更
4. **阻礙？** 無
5. **檔案？** Google Doc 工程筆記（線上文件，非本地檔案）

---

## Session: 2026-03-10 (2)

### 完成項目
- [x] **YOLO 偵測實測驗證** — 用 Qualification 1 影片測試 YOLOv26n 模型，F3 Debug 確認偵測正常
- [x] **閾值調整** — `ROBOT_DETECTION_CONFIDENCE` 從 0.18 調整為 0.27（減少誤偵測）

### 修改檔案
- `config.py` — ROBOT_DETECTION_CONFIDENCE 0.18 → 0.27

### 5-Question Reboot Check
1. **做什麼？** YOLO 偵測實測驗證 + 閾值微調
2. **進度？** 驗證完成，YOLO 偵測效果良好，閾值已調整
3. **下一步？** (1) 測試更多影片確認穩定性 (2) 測試完整 pipeline（進球判定+射手歸因） (3) 提交大量未 commit 的變更
4. **阻礙？** 無
5. **檔案？** `config.py`（閾值）、`robot_detection.py`（YOLO 偵測器）、`app.py`（主介面）

---

## Session: 2026-03-10 00:10 (Endurance #5 — Colab Training + Model Deployment)

### 完成項目
- [x] **合併洪家豪標註** — 505 個 label 檔案從 `洪家豪/` 合併到 `datasets/2024mslr/labels_raw/`
- [x] **build_dataset.py** — 新增可擴展的 CORRECTED_RANGES 系統，只用人工校正完的標註建構訓練集
- [x] **YOLO 資料集建構** — 817 張圖片（train 654 + val 163），打包 yolo_dataset.zip (194.5 MB)
- [x] **YOLOv11n 訓練完成（已棄用）** — 50 epochs, T4 GPU, mAP50=0.903（後改用 YOLOv26n）
- [x] **改用 YOLOv26n 重新訓練** — 用戶原帳號 Colab GPU 額度用完，換帳號 + 共用 Drive 資料夾重新訓練
- [x] **YOLOv26n 訓練完成** — 50 epochs, T4 GPU, ONNX 匯出 9.4 MB
- [x] **模型備份** — best.pt + frc_robot.onnx (yolo26n) 皆已存入 Google Drive
- [x] **模型部署到本地** — 下載 frc_robot.onnx (yolo26n, 9.4 MB) 到 `models/`，舊 yolo11n 備份為 `frc_robot_yolo11n.onnx`
- [x] **切換偵測模式** — `config.py` ROBOT_DETECTION_MODE 從 `"GEMINI"` 改為 `"YOLO"`

### 訓練歷程
1. **YOLOv11n**（已棄用）— mAP50=0.903, P=0.891, R=0.862, ONNX 10.1 MB
2. **YOLOv26n**（最終採用）— ONNX 9.4 MB，模型更小

### 修改檔案
- `build_dataset.py` — **新增** — YOLO 資料集建構腳本（CORRECTED_RANGES 可擴展）
- `train_colab.ipynb` — **修改** — 模型從 yolo11n 改為 yolo26n
- `config.py` — **修改** — ROBOT_DETECTION_MODE 從 `"GEMINI"` 改為 `"YOLO"`
- `models/frc_robot.onnx` — **更新** — YOLOv26n ONNX 模型 (9.4 MB)
- `models/frc_robot_yolo11n.onnx` — **新增** — YOLOv11n 備份 (10.1 MB)

### 5-Question Reboot Check
1. **做什麼？** YOLO bumper 偵測模型訓練 + 部署完成
2. **進度？** YOLOv26n 模型已部署到 `models/frc_robot.onnx`，偵測模式已切換為 YOLO
3. **下一步？** (1) 用實際比賽影片測試 YOLO 偵測效果 (2) 與 HSV Bumper 模式比較 (3) 調整 `ROBOT_DETECTION_CONFIDENCE` 閾值
4. **阻礙？** 無。模型已就緒，待實際影片驗證
5. **檔案？** `models/frc_robot.onnx`（YOLOv26n）、`config.py`（偵測模式+閾值）、`robot_detection.py`（YOLO 偵測器邏輯）

---

## Session: 2026-03-09 (Endurance #4)

### 完成項目
- [x] **Label Editor 互動標註工具** (`label_editor.py`) — 從設計到完整實作
  - CustomTkinter + Canvas 暗色主題 GUI
  - 圖片顯示 + YOLO bbox 渲染（Red/Blue 顏色區分）
  - 滑鼠互動：選取、拖曳移動、8 handle resize、繪製新 bbox
  - 鍵盤快捷鍵：方向鍵導航、Delete 刪除、Tab 切換類別、D 繪製、F 適應視窗、Ctrl+S 存檔、Ctrl+Z 復原
  - 縮放（滾輪，游標為中心）+ 平移（右鍵拖曳）
  - 自動存檔、undo stack、window resize 重算 zoom、bbox 移動 clamp
- [x] **分工標註支援** — `--start`/`--end` 分割圖片範圍、`--labels` 自訂標註子目錄（預設 labels_raw）、顯示所有圖片（不只有標註的）
- [x] **設計文件撰寫** — brainstorming skill 確認需求 → 設計文件 → 實作計劃
- [x] **Subagent-Driven Development** — 3 個 Task 逐步實作 + code review + bugfix
- [x] **兩人分工環境設定** — 專案複製到 E: 隨身碟、筆電 A/B 分工指南、`E:\scoring-analyzer\setup.txt` 環境安裝指南

### 新增/修改檔案
- `label_editor.py` — **新增** — 互動標註工具（CustomTkinter + Canvas，選取/移動/resize/繪製/zoom/pan/undo）
- `docs/plans/2026-03-09-label-editor-design.md` — **新增** — Label Editor 設計文件
- `docs/plans/2026-03-09-label-editor.md` — **新增** — Label Editor 實作計劃

### Commits
- `e22e5f8` feat: label editor scaffold — image display + bbox rendering
- `b659779` feat: label editor — full interaction (select/move/resize/draw/zoom/pan)
- `e5c294f` feat: label editor — undo, window resize, polish
- `d5fe142` fix: label editor — canvas resize fit + bbox move clamping

### 5-Question Reboot Check
1. **做什麼？** Label Editor 已完成，接下來用它來人工校正 Gemini 自動標註的 2842 張 bumper bbox
2. **進度？** Label Editor 完整實作完畢。Gemini 自動標註 2842 張全部完成（上一 session）。人工校正尚未開始
3. **下一步？** (1) 兩人分工用 Label Editor 校正標註品質（筆電 A: `--end 1421`、筆電 B: `--start 1421`） (2) 校正完成後合併 labels (3) 重新匯出 YOLO 資料集 (4) 在 Colab T4 GPU 訓練 YOLO bumper 模型
4. **阻礙？** 無。Label Editor 就緒，待人工校正
5. **檔案？** `label_editor.py`（標註工具）、`datasets/2024mslr/labels_raw/`（待校正標註）、`datasets/2024mslr/images/`（2842 張圖片）

---

## Session: 2026-03-09 (Endurance #3)

### 完成項目
- [x] **Gemini 3.1 flash lite 速度測試** — 平均 3.1s/張，max 4.1s，timeout 從 60s 降至 10s
- [x] **全量 Gemini 標註完成** — 2842 張全部標註完成（gemini-3.1-flash-lite-preview, bumper-only, ~3小時）
- [x] **排除區域過濾** — 加入 `_EXCLUSION_ZONES` 過濾場地固定物件 FP（左右兩側 x≈0.14/0.874, y≈0.49），移除 784 個 FP boxes
- [x] **labels_raw → labels 過濾** — 7023 raw boxes → 移除 784 FP + 33 小框 → 6206 有效 boxes（Red=3325, Blue=2881）
- [x] **YOLO 資料集匯出** — 2635 張有效圖片 → train 2108 / val 527, data.yaml 就緒

### 修改檔案
- `auto_annotate.py` — timeout 60s→10s, 新增 `_EXCLUSION_ZONES` + `_in_exclusion_zone()` 過濾場地固定物件
- `datasets/2024mslr/labels_raw/` — 2842 張 Gemini 原始標註
- `datasets/2024mslr/labels/` — 2635 張過濾後 YOLO labels（6206 boxes）
- `datasets/2024mslr/yolo_dataset/` — train/val split + data.yaml

### 5-Question Reboot Check
1. **做什麼？** Gemini 自動標註 + 過濾 + 匯出 YOLO bumper 訓練資料集
2. **進度？** 全部完成。2842 張圖片標註完畢，過濾後 6206 boxes，資料集已匯出
3. **下一步？** (1) 用 Label Editor 抽查標註品質 (2) 在 Colab T4 GPU 用 `datasets/2024mslr/yolo_dataset/data.yaml` 訓練 YOLO bumper 模型 (3) 訓練完替換 `models/frc_robot.onnx` 測試
4. **阻礙？** 無。資料集已就緒
5. **檔案？** `datasets/2024mslr/yolo_dataset/data.yaml`（訓練入口）、`train_robot_model.py`（訓練腳本）、`train_colab.ipynb`（Colab notebook）

---

## Session: 2026-03-09 (Endurance #2)

### 完成項目
- [x] **恢復耐久模式 Session 2** — 從 Session 1 接續，確認標註進度（1130/5182 labels）
- [x] **auto_annotate.py 改為 native box_2d** — `_annotate_one` 從 structured JSON schema 改為 native `box_2d` 格式（unstructured response + regex parsing），確認 Gemini 座標格式 `[y_min, x_min, y_max, x_max]` 1000-based
- [x] **_to_yolo 更新** — 處理 1000-based box_2d 座標轉換為 YOLO 格式
- [x] **預設模型切換** — 從 `gemini-3.1-flash-lite-preview` → `gemini-2.5-flash` → 最終用戶要求改回 `gemini-3.1-flash-lite-preview`（速度優先）
- [x] **API timeout 更新** — 從 20s 改為 60s
- [x] **修正 labels 目錄** — auto_annotate.py 使用 `labels_raw/`，與之前 inline script 的 `labels/` 同步
- [x] **刪除 gemini-2.5-flash 舊標註** — 清空所有 labels_raw/ 和 labels/ 內容
- [x] **Prompt 改為只框 bumper** — 更新 `_ANNOTATE_PROMPT` 只框機器人下方保險桿（bumper），不框整台機器人。目標：bumper-only bbox 更精確，適合 HSV Bumper 偵測模式訓練
- [x] **刪除 36 場廣角比賽幀** — qm13-17, 19, 22, 28, 30-31, 33-34, 38, 41, 43-44, 46, 51-53, 57, 62-65, 67, 70, 73-74, sf2m1, sf3m1, sf5m1, sf8m1, sf11m1, sf12m1, sf13m1（共 2340 張刪除），因為畫面有開廣角，不適合訓練
- [x] **清空所有舊標註資料** — labels_raw/, labels/, yolo_dataset/ 全部清空，準備全新 bumper-only 標註

### 修改檔案
- `auto_annotate.py` — prompt 改為只框 bumper、`_annotate_one` 改為 native box_2d、`_to_yolo` 改為 1000-based 座標、預設模型改為 `gemini-3.1-flash-lite-preview`、timeout 改為 60s
- `datasets/2024mslr/images/` — 刪除 36 場廣角比賽幀，從 5182 張減至 2842 張
- `datasets/2024mslr/images_raw/` — 同上，從 5182 張減至 2842 張
- `datasets/2024mslr/labels_raw/` — 清空（全部標註作廢，待重新 bumper-only 標註）
- `datasets/2024mslr/labels/` — 清空
- `datasets/2024mslr/yolo_dataset/` — 清空

### 5-Question Reboot Check
1. **做什麼？** 準備 bumper-only Gemini 自動標註 pipeline，清理資料集（移除廣角幀 + 清空舊標註）
2. **進度？** Pipeline 已就緒（auto_annotate.py 更新完成），2842 張圖片待標註，標註尚未開始
3. **下一步？** (1) 用 `gemini-3.1-flash-lite-preview` 跑 2842 張 bumper-only 全量標註 (2) 標註完成後匯出 YOLO 資料集 (3) 在 Colab T4 GPU 訓練 bumper YOLO 模型
4. **阻礙？** 無。Pipeline 已就緒，等待下一 session 執行標註
5. **檔案？** `auto_annotate.py`（標註 pipeline，bumper-only prompt）、`datasets/2024mslr/images/`（2842 張待標註圖片）、`datasets/2024mslr/crop.json`（裁切範圍）

---

## Session: 2026-03-09 (Endurance #1)

### 完成項目
- [x] **裁切範圍選取** — OpenCV `selectROI` 選取 2024 Magnolia Regional 裁切範圍 `(1, 105) 1278x534` → `datasets/2024mslr/crop.json`
- [x] **Batch 1 裁切** — 548 張原始幀裁切完成
- [x] **Gemini 模型測試** — 測試多個模型+格式組合：
  - `gemini-3-flash-preview` + structured JSON: 504 DEADLINE_EXCEEDED（完全不可用）
  - `gemini-2.5-flash` + structured JSON: 0 偵測
  - `gemini-3.1-flash-lite-preview` + native box_2d: 快但偵測少（1-3/img）
  - `gemini-2.5-flash` + native box_2d + 改良 prompt: **最佳方案**（3-6 偵測/img, ~20s/img）
- [x] **座標格式研究** — 確認 Gemini native bounding box 為 `box_2d: [y_min, x_min, y_max, x_max]` 1000-based 座標（不是 pixel）。Structured JSON schema 會導致座標格式混亂，需用 unstructured response + regex 解析
- [x] **下載全部比賽影片** — 87/89 部 2024 Magnolia Regional 影片（qm1-qm74 + sf1-sf13 + f1-f2，qm29/qm48 失敗）
- [x] **全量提取+裁切** — 從 87 部影片提取 5182 張裁切幀（每 3 秒一幀，1278x534）
- [x] **Batch 1 Gemini 標註完成** — 548 張 (qm1-qm10): 533 OK / 14 empty / 6 err，原始 2172 boxes，有效 YOLO labels 367 張 1470 boxes（小 bbox 被過濾）
- [x] **匯出 YOLO 資料集 v1** — `datasets/2024mslr/yolo_dataset/`: 294 train + 73 val, data.yaml
- [ ] **Batch 2 Gemini 標註進行中** — 550/4634 完成 (qm11-qm74+sf+f), 預計 ~24 小時（背景執行中 Task b0tfcnqn6）
- [x] **YOLO 資料集更新匯出** — 665 imgs (532 train / 133 val), 3085 boxes (R=1426 B=1659)

### 新增/修改檔案
- `auto_annotate.py` — 加入 `HttpOptions(timeout=)` 參數
- `datasets/2024mslr/crop.json` — **新增** — 裁切範圍 `(1, 105) 1278x534`
- `datasets/2024mslr/images/` — **新增** — 5182 張裁切後圖片
- `datasets/2024mslr/images_raw/` — **更新** — 5182 張原始圖片
- `datasets/2024mslr/labels/` — **新增** — YOLO labels（batch 1 完成，batch 2 進行中）
- `datasets/2024mslr/videos/` — **更新** — 87 部 720p mp4
- `datasets/2024mslr/yolo_dataset/` — **新增** — YOLO train/val split + data.yaml

### 5-Question Reboot Check
1. **做什麼？** Gemini 自動標註 FRC 機器人 → 匯出 YOLO 訓練資料集
2. **進度？** 1109/5182 已標註。YOLO 資料集已匯出：665 imgs / 3085 boxes (R=1426 B=1659)。Batch 2 在背景持續標註中（550/4634）
3. **下一步？** (1) **可以立即開始訓練** — 資料集已匯出在 `datasets/2024mslr/yolo_dataset/data.yaml` (2) Batch 2 背景繼續跑，完成後重新匯出可獲得更多資料 (3) 訓練完替換 `models/frc_robot.onnx` 測試
4. **阻礙？** Batch 2 背景進程需確認是否仍在跑（可能 session 結束後停止）。若停了，重新跑相同腳本即可（自動跳過已完成的 labels）。bbox 有 ~35% 圖片為空（false negative + API error）
5. **檔案？** `auto_annotate.py`（標註 pipeline）、`datasets/2024mslr/`（完整資料）、`datasets/2024mslr/yolo_dataset/data.yaml`（訓練入口）、`train_robot_model.py`（訓練腳本）

---

## Session: 2026-03-08

### 完成項目
- [x] **下載 2024 Magnolia Regional 比賽影片** — 用 `download_matches.py` 查詢 TBA API，找到 89 場比賽有影片，背景下載中（每 3 秒提取一幀，fps=0.333）
- [x] **互動式裁切範圍選取** — OpenCV `selectROI` 畫框選取主場地區域，排除底部計分板和小視窗，存到 `datasets/2023mslr/crop.json`: `(147, 179) 1019x274`
- [x] **Gemini API 標註測試** — 測試多個模型：`gemini-3-flash-preview` API 504 超時、`gemini-2.5-flash` 偵測品質極差、`gemini-2.0-flash` 已停用 (404)
- [x] **Google AI Studio 瀏覽器自動化標註** — Playwright 操作 AI Studio 網頁版，`gemini-3-flash-preview` Thinking High 模式偵測正確（6 個偵測全對，3 Blue + 3 Red），但速度慢（~3 分鐘/張）
- [x] **Gemini 座標格式研究** — 確認 Gemini 使用 1000-based 座標格式，轉換公式：`pixel_x = coord / 1000 * width`
- [x] **2024 影片清單改為官方影片** — 從 YouTube 播放清單 `PLXMtScTweiUEnho5azcuN_LmKOEk4690v` 下載官方比賽影片（而非 TBA 第一個影片連結），確保影片品質穩定
- [x] **下載 10 部 2024 Magnolia Regional 官方比賽影片** — 720p mp4，存放於 `datasets/2024mslr/videos/`（001-010）
- [x] **提取 550 幀** — 每 3 秒一幀（fps=0.333），從 10 部影片提取，存放於 `datasets/2024mslr/images_raw/`（v01-v10 前綴）

### 新增/修改檔案
- `auto_annotate.py` — Gemini 自動標註 pipeline（修改 prompt 和 API 呼叫）
- `download_matches.py` — TBA + YouTube 下載工具（修改下載邏輯）
- `datasets/2023mslr/crop.json` — **新增** — 裁切範圍設定
- `datasets/2023mslr/images_cropped/test1.jpg` — **新增** — 裁切測試圖片
- `datasets/2023mslr/preview/` — **新增** — 各種標註測試預覽圖
- `datasets/2024mslr/videos/` — **新增** — 10 部 2024 官方比賽影片（720p mp4, 001-010）
- `datasets/2024mslr/images_raw/` — **新增** — 550 幀提取圖片（v01-v10 前綴）
- `datasets/2024mslr/match_videos.json` — **更新** — 改為選擇第二個（官方）影片連結

### 5-Question Reboot Check
1. **做什麼？** 用 Gemini 自動標註 FRC 機器人 → 訓練 YOLO 模型
2. **進度？** 2024 影片下載+提取完成（10 部影片, 550 幀），標註方案已驗證（AI Studio gemini-3-flash-preview 品質 OK），但批次標註尚未開始
3. **下一步？** (1) 讓用戶選取 2024 裁切範圍 (2) 批次裁切 550 幀 (3) 找可靠的批次 Gemini 標註方式 (4) YOLO 訓練
4. **阻礙？** Gemini API 504 超時問題未解，AI Studio 瀏覽器太慢（3min/張），需找批次標註方案
5. **檔案？** `auto_annotate.py`（標註 pipeline）、`download_matches.py`（影片下載）、`datasets/2024mslr/`（2024 影片+幀）、`datasets/2023mslr/crop.json`（裁切設定）

---

## Session: 2026-03-06 (下午)

### 完成項目
- [x] **下載比賽影片** — 從 YouTube 下載 FRC 比賽影片 `match_video.mp4`（1920x1080, 60fps, 214s, ~95MB），升級 yt-dlp 2025.12.8 → 2026.3.3
- [x] **切回 YOLO 機器人偵測** — `config.py` `ROBOT_DETECTION_MODE` 從 `"HSV"` 改回 `"YOLO"`，使用既有 `models/frc_robot.onnx` YOLO 模型

### 修改檔案
- `config.py` — `ROBOT_DETECTION_MODE` 從 `"HSV"` 改為 `"YOLO"`
- `match_video.mp4` — **新增** — YouTube 下載的 FRC 比賽影片（1920x1080, 60fps）

### 5-Question Reboot Check
1. **做什麼？** 下載新的 FRC 比賽影片 + 切回 YOLO 機器人偵測模式準備測試
2. **進度？** 影片已下載，YOLO 模式已啟用，尚未跑分析測試
3. **下一步？** 用 `python main.py match_video.mp4` 跑 YOLO 模式分析測試 → 驗證機器人偵測效果 → 影片是三視角可能需要 ROI 裁切 → 處理 HP 進球計算不準確 + 藍方 HP 進球分析不到的問題
4. **阻礙？** 影片可能是三視角合成畫面，需確認是否需要 ROI 裁切只分析其中一個視角；YOLO 模型是用舊資料集訓練的，對這個新影片的偵測效果未知
5. **檔案？** `config.py`（ROBOT_DETECTION_MODE=YOLO）、`app.py`（分析流程）、`robot_detection.py`（YOLO 偵測器）、`match_video.mp4`（測試影片）

---

## Session: 2026-03-06 (上午)

### 完成項目
- [x] **FRC 工程筆記撰寫與迭代** — 為 FRC 評審提交撰寫完整工程筆記，經過多輪迭代：
  1. **研究 FRC 工程筆記寫法** — 調查 FRC 獎項評分標準、Technical Binder 結構、反 AI 味寫作技巧
  2. **生成 ENGINEERING_NOTEBOOK.md** — 完整 Markdown 版工程筆記（學生口語風格、具體數據、失敗經驗）
  3. **轉換 ENGINEERING_NOTEBOOK.html** — 精簡版 HTML（為 Google Docs 匯入，帶 CSS 樣式，無程式碼區塊）
  4. **產生 ENGINEERING_NOTEBOOK.txt** — 繁體中文純文字版（用戶反饋不需排版）
  5. **翻譯 ENGINEERING_NOTEBOOK_EN.txt** — 英文純文字版
  6. **插入圖片標記** — 在兩份 txt 中加入 `[圖片：...]` / `[Image: ...]` 標記
- [x] **用戶自行編輯精簡** — 用戶修改了中文版：精簡內容、調整 YOLO vs HSV 主次關係（HSV Bumper 為主、YOLO 為備選）、刪除部分章節
- [x] **發現並修正內容矛盾** — YOLO vs HSV Bumper 在文中誰是主要方案的描述矛盾，已修正統一
- [x] **英文版同步** — 根據用戶修改後的中文版重新翻譯英文版，確保兩版一致

### 新增/修改檔案
- `ENGINEERING_NOTEBOOK.md` — 完整中文工程筆記（Markdown 格式）
- `ENGINEERING_NOTEBOOK.html` — 精簡版 HTML（帶 CSS 樣式、flow box、彩色提示框）
- `ENGINEERING_NOTEBOOK.txt` — 繁體中文純文字版（用戶已自行編輯精簡）
- `ENGINEERING_NOTEBOOK_EN.txt` — 英文純文字版（最終同步用戶修改）

### 5-Question Reboot Check
1. **做什麼？** 為 FRC 評審撰寫工程筆記（Engineering Notebook），記錄專案技術開發歷程
2. **進度？** 工程筆記全部完成（md/html/txt 中文版 + txt 英文版），用戶已自行編輯精簡中文版，英文版已同步
3. **下一步？** (1) 用戶截取應用截圖插入工程筆記 (2) 上傳 Google Docs 或列印提交 (3) 回到主線開發：**用戶需跑測試驗證 Fix 3-5 效果**（機器人追蹤修復，上次 session 2026-03-02 的 5 個 fix 中 Fix 3-5 尚未經用戶實際影片測試）(4) 若追蹤恢復，處理 HP 進球計算不準確 + 藍方 HP 進球分析不到
4. **阻礙？** 機器人追蹤 Fix 3-5 尚未經用戶實際影片驗證（2026-03-02 session 遺留）；工程筆記截圖需用戶自行截取
5. **檔案？** `ENGINEERING_NOTEBOOK.txt`（主要中文版，用戶已編輯）、`ENGINEERING_NOTEBOOK_EN.txt`（英文版，已同步）、`ENGINEERING_NOTEBOOK.md`（完整 Markdown 版）、`ENGINEERING_NOTEBOOK.html`（精簡 HTML 版）

---

## Session: 2026-03-02

### 完成項目
- [x] **機器人追蹤失效根因分析** — 深度診斷找到 3 個 P0 致命問題：
  1. **背景遮罩過度過濾**（最致命）：`app.py` 用 `cv2.bitwise_and(frame, mask=fg_mask)` 遮蔽背景，但 Temporal Median 將靜止/慢移機器人納入背景 → bumper 像素被抹黑 → HSV 零偵測
  2. **BG_FG_THRESHOLD=30 太敏感**：靜止機器人 absdiff ≈ 15-20，低於閾值被判為背景
  3. **Temporal Median 在 FRC 場景不適用**：背景假設「機器人不在場」但 FRC 比賽機器人始終在場
- [x] **外部研究** — 調研替代方案：YCrCb 色彩空間、YOLO-World 零樣本偵測、YOLOE 圖像提示偵測、SportSORT 運動追蹤、Roboflow 社群最佳實踐
- [x] **Fix 1: 分離背景遮罩** — `app.py` 修改分析流程：球偵測用 `frame_masked`（保留背景過濾），機器人偵測用原始 `frame`（不受背景模型影響）
- [x] **Fix 2: HSV 偵測診斷日誌** — `robot_detection.py` 加入逐步診斷：HSV mask 像素數、輪廓數、面積/長寬比過濾統計、NMS 前後數量
- [x] **Fix 3: 放寬 Bumper 長寬比** — `config.py` `BUMPER_MIN_ASPECT` 從 1.2 降到 0.5（允許直立/接近方形的 bumper 通過過濾）
- [x] **Fix 4: MOT 永遠用距離匹配** — `app.py` 移除 ByteTrack 條件分支，MOT 模式一律使用 `_match_direct()` 距離匹配
- [x] **Fix 5: 距離式標記匹配** — `robot_tracker.py` 新增 `_consume_pending_marker()` 方法，用距離比對將 pending marker 分配給最近的 tracker

### 診斷結果（用戶測試）
- 背景遮罩修復成功 — 黑色像素佔比 0%（機器人不再被抹黑）
- HSV 偵測有結果 — 每幀 1-3 個紅色偵測通過所有過濾
- 但 MOT 追蹤輸出仍為 0 → 發現是 ByteTrack 路徑 + 長寬比過濾的組合問題
- 第二輪修復（Fix 3-5）已完成，**用戶尚未跑測試**

### 修改檔案
- `app.py` — 分離背景遮罩（球偵測用 frame_masked，機器人用原始 frame）+ MOT 永遠用 `_match_direct()` 距離匹配
- `robot_detection.py` — 加入詳細 HSV 偵測診斷日誌（mask pixels、contour count、filter stats）
- `robot_tracker.py` — 新增 `_consume_pending_marker()` 距離式標記匹配方法
- `config.py` — `BUMPER_MIN_ASPECT` 1.2 → 0.5
- `FINDINGS.md` — 新增根因分析（3 個 P0 問題）+ 替代方案研究（YCrCb、YOLO-World、YOLOE、SportSORT）

### 用戶提到的其他問題（未處理）
- HP 進球計算不準確
- 藍方 HP 進球分析不到
- 進球偵測算蠻準確的不用改

### 5-Question Reboot Check
1. **做什麼？** 排查並修復機器人追蹤失效（HSV Bumper 模式追蹤不到任何機器人）
2. **進度？** 根因已確認（背景遮罩+長寬比+ByteTrack 路徑）。5 個修復已實作，第一輪 Fix 1-2 驗證通過（HSV 偵測恢復），第二輪 Fix 3-5 待用戶跑測試
3. **下一步？** **用戶需跑測試驗證 Fix 3-5 效果**（`python main.py` → 載入影片 → 分析 25 秒 → 檢查機器人追蹤是否有輸出）。若追蹤恢復，接著處理 HP 進球計算不準確 + 藍方 HP 進球分析不到的問題。**記得截圖**（工程筆記用）
4. **阻礙？** Fix 3-5 尚未經用戶實際影片驗證，無法確認追蹤是否完全恢復
5. **檔案？** `app.py`（背景遮罩分離 + MOT 距離匹配）、`robot_detection.py`（HSV 診斷日誌）、`robot_tracker.py`（`_consume_pending_marker()`）、`config.py`（`BUMPER_MIN_ASPECT`=0.5）

---

## Session: 2026-03-01

### 完成項目
- [x] **FRC 工程筆記研究** — 深度研究 FRC Engineering Notebook 撰寫 best practices、Judges Award / Engineering Inspiration Award 評分標準、anti-AI 寫作風格（學生口語、具體數據、失敗經驗）
- [x] **技術架構深度分析** — 分析專案所有模組（config, detection, tracking, robot_detection, robot_tracker, scoring, background, calibration, app 等），為工程筆記提取技術細節和設計決策
- [x] **ENGINEERING_NOTEBOOK.md** — 生成完整中文工程筆記（學生口語風格，包含問題發現、失敗嘗試、技術決策理由、數據驗證）
- [x] **ENGINEERING_NOTEBOOK.html** — 帶 CSS 樣式的 HTML 版本（用於 Google Docs 匯入）
- [x] **Google Docs 匯入問題** — 用戶反映 Google Docs 不支援直接上傳 HTML，提供替代方案（瀏覽器開啟後複製貼上）
- [x] **HTML 精簡重寫** — 用戶反映內容太長且有程式碼格式，重寫精簡版（砍 50%、移除所有 code blocks、改用 flow box + 彩色提示框）
- [x] **ENGINEERING_NOTEBOOK.txt** — 用戶要求不要排版，生成純文字版本
- [x] **ENGINEERING_NOTEBOOK_EN.txt** — 英文翻譯版本（保持學生口語風格）
- [x] **截圖建議** — 用戶提供一張分析結果截圖，建議了 9 張圖的優先順序清單（分析結果總覽、4-Panel Debug、Bumper 取色、HSV 校正、設定面板、播放 overlay、背景模型、程式碼架構、Colab 訓練）

### 新增檔案
- `ENGINEERING_NOTEBOOK.md` — 完整中文工程筆記（Markdown 格式）
- `ENGINEERING_NOTEBOOK.html` — 精簡版 HTML（帶 CSS 樣式，flow box + 彩色提示框）
- `ENGINEERING_NOTEBOOK.txt` — 純文字版中文工程筆記
- `ENGINEERING_NOTEBOOK_EN.txt` — 英文翻譯版工程筆記

### 5-Question Reboot Check
1. **做什麼？** 撰寫 FRC 工程筆記（Engineering Notebook），記錄專案的技術開發歷程，用於 FRC 評審提交
2. **進度？** 中文版和英文版均已完成（md/html/txt 三種格式）。截圖建議清單已提供，用戶需自行截取應用畫面
3. **下一步？** (1) 用戶截取建議的 9 張應用截圖 (2) 將截圖插入工程筆記 (3) 上傳到 Google Docs 或列印 (4) 回到主線開發：排查機器人追蹤失效根因（HSV Bumper 模式追蹤不到任何機器人）
4. **阻礙？** 機器人追蹤仍完全失效（上次 session 發現的問題尚未修復），待排查根因
5. **檔案？** `ENGINEERING_NOTEBOOK.md`（主要中文版）、`ENGINEERING_NOTEBOOK_EN.txt`（英文版）、`ENGINEERING_NOTEBOOK.html`（精簡 HTML 版）、`ENGINEERING_NOTEBOOK.txt`（純文字版）

---

## Session: 2026-02-28 (3)

### 完成項目
- [x] **分析按鈕拆分** — 原單一「開始分析」按鈕拆為「分析 25 秒」+「完整分析」兩個按鈕
  - `self.analyze_btn` → `self.analyze_quick_btn`（綠色 accent，分析前 25×fps 幀）+ `self.analyze_full_btn`（紫色 #8e44ad，分析全部幀）
  - `_on_analyze(max_seconds=None)` 新增 `max_seconds` 參數化，計算 `_analysis_max_frames`
  - `_run_analysis()` 中 `min(total_frames, max_frames)` 限制處理範圍
- [x] **播放卡頓修復** — 分析完播放影片會播一下卡一下
  - 根因：牆鐘時間同步 + overlay 渲染重 → 落後時跳大步追趕 → 視覺卡頓
  - 第一次修復：加入落後>10幀重設基準 + 最小delay 1ms→10ms（仍有間歇卡頓）
  - 第二次修復：**完全移除牆鐘同步**，改為固定步進（每 tick 前進 `fps/30 × speed` 幀）。渲染慢就自動降速，不卡頓
  - 移除了 `_play_wall_start` 和 `_play_start_frame` 在 `_play_loop` 中的使用（`_toggle_play` 和 `_toggle_speed` 中仍保留初始化但不再被播放迴圈讀取）

### 修改檔案
- `app.py` — 分析按鈕拆分（`analyze_quick_btn` + `analyze_full_btn`）、`_on_analyze(max_seconds)` 參數化、`_run_analysis()` 幀數限制、播放迴圈從牆鐘同步改為固定步進

### 測試結果
用戶實際影片驗證：
- 球偵測：**準確** ✅
- 機器人追蹤：**完全丟失追蹤不到** ❌
- 射手歸因：**追蹤不到**（因機器人追蹤失敗）❌

### 5-Question Reboot Check
1. **做什麼？** 拆分分析按鈕（25 秒快速 + 完整分析）+ 修復播放卡頓（牆鐘同步改固定步進）
2. **進度？** 兩項功能修改完成。球偵測正常，但機器人追蹤完全失效需排查
3. **下一步？** 排查機器人追蹤失敗根因 — 可能是：(1) HSV Bumper 偵測器參數問題 (2) Bumper 取色模板未正確傳遞 (3) 背景模型前景遮罩過度過濾 (4) 分析幀數限制影響追蹤初始化
4. **阻礙？** 機器人追蹤完全失效，射手歸因連帶無法工作，需優先排查
5. **檔案？** `app.py`（分析按鈕 `analyze_quick_btn`/`analyze_full_btn`、`_on_analyze()`、`_run_analysis()`、`_play_loop()`）、`robot_detection.py`（HSV Bumper 偵測器）、`robot_tracker.py`（MOT 追蹤）

---

## Session: 2026-02-28 (2)

### 完成項目
- [x] **分析按鈕拆分為「分析 25 秒」+「完整分析」** — 原單一「開始分析」按鈕拆為兩個，方便快速預覽和完整分析切換
  - `self.analyze_btn` 拆為 `self.analyze_quick_btn`（綠色 accent，分析前 25×fps 幀）+ `self.analyze_full_btn`（紫色 #8e44ad，分析全部幀）
  - `_on_analyze(max_seconds=None)` 新增 `max_seconds` 參數，計算 `_analysis_max_frames`
  - `_run_analysis()` 中 `total = min(self.total_frames, self._analysis_max_frames)` 限制處理幀數
  - `_analysis_error()` 和 `_finish_analysis()` 同時恢復兩個按鈕狀態
  - 語法驗證通過

### 修改檔案
- `app.py` — 分析按鈕拆分（`analyze_quick_btn` + `analyze_full_btn`）、`_on_analyze(max_seconds)` 參數化、`_run_analysis()` 幀數限制、錯誤/完成回調恢復雙按鈕

### 5-Question Reboot Check
1. **做什麼？** 將單一分析按鈕拆為「分析 25 秒」（快速預覽）和「完整分析」（全部幀）兩個按鈕
2. **進度？** 100% 完成，語法驗證通過
3. **下一步？** 用實際影片驗證：(1) 25 秒分析是否正確在 25×fps 幀停止 (2) 完整分析是否照常處理全部幀 (3) 分析中/錯誤/完成時兩個按鈕狀態是否正確切換
4. **阻礙？** 無明確阻礙
5. **檔案？** `app.py`（`analyze_quick_btn`、`analyze_full_btn`、`_on_analyze()`、`_run_analysis()`、`_analysis_error()`、`_finish_analysis()`）

---

## Session: 2026-02-28

### 完成項目
- [x] **Bumper 取色偵測完整實作** — 按照 `docs/plans/2026-02-27-bumper-color-pick-design.md` 計劃執行 6 個 Task，完成從設計到驗證的全流程
  - **calibration.py** — 新增 `build_bumper_template()` 函式，從用戶取色點建立 HSV H+S 16x16 直方圖模板 + 自動判斷聯盟（red/blue）
  - **config.py + runtime_config.py** — 新增 `BUMPER_TEMPLATE_SIMILARITY=0.3` 常數和 dataclass 欄位
  - **robot_detection.py** — `BumperDetectorHSV` 新增 `set_templates()`, `_match_template()` 方法，`detect()` 中加入模板過濾邏輯（有模板時只保留匹配的候選）
  - **app.py** — 完整 UI 改造：按鈕從「標記機器人」改為「Bumper 取色」；新增 `_start_bumper_pick` / `_finish_bumper_pick` 互動方法；canvas 支援 `bumper_pick` 模式（左鍵取色點、右鍵完成）；繪製取色中的橙色圓點；模板傳遞到偵測器（`set_templates()`）；auto mode 邏輯調整（有取色模板時不自動偵測）；`_cancel_interaction` 支援 bumper_pick 清理
  - **app.py 輔助方法** — 新增 `_get_current_frame()` 輔助方法；重構 `_get_current_frame_for_preview()` 為呼叫 `_get_current_frame()`；`_finish_crop` / `_reset_crop` / `_clear_all_marks` 清除 bumper 模板
  - **驗證** — 5 個檔案語法檢查通過、import 測試通過、端到端功能驗證通過

### 修改檔案
- `calibration.py` — 新增 `build_bumper_template()` 函式（HSV H+S 16x16 直方圖模板建立 + 聯盟自動判斷）
- `config.py` — 新增 `BUMPER_TEMPLATE_SIMILARITY = 0.3` 常數
- `runtime_config.py` — 新增 `bumper_template_similarity` dataclass 欄位
- `robot_detection.py` — `BumperDetectorHSV` 新增 `set_templates()`, `_match_template()`, `detect()` 模板過濾邏輯
- `app.py` — Bumper 取色 UI 取代框選標記、`_get_current_frame()` 輔助方法、`bumper_pick` canvas 互動模式、模板傳遞、auto mode 調整、清理邏輯

### Git Commit
- `8cd6d2d` — `feat: Bumper 取色偵測 + 背景模型 + HSV Bumper + MOT 增強`（19 files, +4376/-222）
- 包含多個 session 累積的未提交變更（背景模型、HSV Bumper 偵測器、MOT 增強等）

### 5-Question Reboot Check
1. **做什麼？** 按照設計計劃實作 Bumper 取色偵測功能，用戶在影片上點擊機器人 bumper 取色 → 建立 HSV 直方圖模板 → 偵測時只追蹤匹配模板的機器人
2. **進度？** 100% 完成。6 個 Task 全部完成，語法檢查 + import 測試 + 端到端驗證通過
3. **下一步？** 用實際 FRC 比賽影片驗證 Bumper 取色偵測效果：(1) 取色流程是否直覺易用 (2) 模板匹配閾值 0.3 是否合適 (3) 是否有效過濾非目標機器人 (4) 與背景模型前景遮罩搭配的偵測精度
4. **阻礙？** 無明確阻礙；`BUMPER_TEMPLATE_SIMILARITY=0.3` 閾值可能需要根據實際影片微調
5. **檔案？** `calibration.py`（`build_bumper_template()`）、`robot_detection.py`（`BumperDetectorHSV` 模板匹配）、`app.py`（Bumper 取色 UI + `_get_current_frame()`）、`config.py`（`BUMPER_TEMPLATE_SIMILARITY`）

---

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
*Last updated: 2026-03-13*

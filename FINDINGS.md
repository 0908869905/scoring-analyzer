# FRC Scoring Analyzer — Technical Findings

*記錄開發過程中的技術發現和決策*

## 2026-02-14: Hub 得分區域改為多邊形

### 問題
原本 Hub 得分區域用矩形 (x, y, w, h) 定義，並區分 Upper Hub / Lower Hub 兩個區域。但 FRC 場地的 Hub 形狀不規則，矩形無法精確覆蓋，且 Upper/Lower 區分增加了標記和統計的複雜度。

### 解決方案
1. **ScoringZone 改為多邊形** — 用頂點列表 `List[Tuple[int, int]]` 取代 `(x, y, w, h)` 矩形
2. **新增 `point_in_polygon()` ray casting** — 經典射線法判斷點是否在多邊形內，O(n) 複雜度，n 為頂點數
3. **移除 Upper/Lower 區分** — 合併為單一 "Hub" 區域，`RobotScore` 簡化為 `auto` + `teleop` 兩個欄位

### 選擇理由
- **Ray casting vs 其他演算法**: Ray casting 實作簡單、數值穩定、無需預處理，適合即時逐幀判定
- **移除 Upper/Lower**: 在影片分析中難以可靠區分球進入 Hub 的高度位置，簡化後減少誤判並讓 UI 更直覺
- **多邊形互動模式**: 左鍵放置頂點、右鍵/雙擊完成多邊形、ESC 取消，符合常見標註工具的操作慣例

---

## 2026-02-14: 播放時間漂移修復 — 牆鐘時間基準

### 問題
逐幀播放時使用 `after(delay)` 延遲補償，但每幀的處理時間不固定（影像解碼、偵測、繪製等），導致 delay 計算的累積誤差使播放速度越來越偏離實際時間。例如 30fps 影片在 10 秒後可能只播了 280 幀而非 300 幀。

### 原因
原本做法：每幀測量處理耗時 `elapsed`，計算 `delay = frame_interval - elapsed`。問題在於 `after()` 本身的排程精度有限（Tk event loop 不保證精確觸發），加上每幀的微小誤差逐幀累積。

### 解決方案
改用**牆鐘時間基準**（wall-clock time base）：
1. 播放開始時記錄 `_play_start_time = time.monotonic()` 和 `_play_start_frame = current_frame`
2. 每次 `_play_loop()` 計算：`target_frame = start_frame + elapsed * fps * speed`
3. 若當前幀落後於 target_frame，立即推進（delay=1ms）；否則計算精確等待時間
4. 切換速度時重設時間基準，避免瞬間跳幀

### 選擇理由
- **牆鐘時間 vs 逐幀延遲**: 牆鐘時間基準是業界標準做法（影片播放器、遊戲引擎皆採用），可自動修正累積誤差
- **`time.monotonic()` vs `time.time()`**: monotonic 不受系統時間調整影響，更適合測量時間間隔
- **追趕機制**: 落後時設 delay=1ms 而非 0ms，確保 Tk event loop 有機會處理 UI 事件，避免界面凍結

---

## 2026-02-15: AI 球偵測從 Roboflow HTTP API 改為 YOLOv11 ONNX 本地推理

### 問題
原本 AI 球偵測使用 Roboflow HTTP API（`fueldetection_grayscale/3`），需要網路連線。用戶明確需要離線運作。嘗試過 `inference-gpu` 和 `inference` SDK 但都不支援 Python 3.14（需 <3.13）。

### 原因
- Roboflow `inference` SDK 依賴鏈深（PyTorch、supervision 等），且不支援 Python 3.14
- HTTP API 可用但需要網路，不符合離線需求
- 用戶已有訓練好的 YOLOv11n 權重檔（5.5MB）

### 解決方案
1. **用 ultralytics 將 .pt 轉 .onnx** — `model.export(format="onnx", imgsz=640, opset=12, simplify=True)`，產出 10.1MB ONNX 檔
2. **FuelDetectorONNX 類別** — 使用 onnxruntime 載入 ONNX，實作完整推理管線：
   - **letterbox 預處理**: 等比例縮放到 640x640 + padding，避免形變
   - **NMS 後處理**: 轉置 YOLOv11 輸出格式 `(1, 5, 8400)` → `(8400, 5)`，confidence 過濾 + cv2.dnn.NMSBoxes
3. **移除所有網路依賴** — 不再需要 requests、python-dotenv、.env 檔

### 選擇理由
- **ONNX vs PyTorch**: ONNX 輕量（10MB vs PyTorch 數百MB），啟動快，無需 CUDA toolkit，支援 DirectML GPU 加速
- **onnxruntime vs ultralytics runtime**: onnxruntime 依賴少、跨平台、支援多種硬體加速後端
- **letterbox vs 直接 resize**: letterbox 保持長寬比，避免因形變降低偵測精度
- **opset 12**: 相容性最佳，支援所有 onnxruntime 版本

### 關鍵發現
- 模型名含 "grayscale" 但 ONNX 模型實際使用 **BGR 3 通道輸入**（灰階輸入反而效果差）
- 4K 影片中球的 bounding box area 約 70000，原始 `AI_MAX_AREA=50000` 太小，改為 200000
- `AI_CONFIDENCE_THRESHOLD` 從 0.5 降至 0.25，本地推理信心值分布與 HTTP API 不同
- `inference-gpu` 和 `inference` SDK 都不支援 Python 3.14（需要 <3.13）

---

## 2026-02-15: YOLOv11 輸出格式注意事項

### 問題
YOLOv11 ONNX 輸出 shape 為 `(1, 5, 8400)` 而非 YOLOv5 的 `(1, 8400, 5)`，若不轉置會導致 NMS 完全失效。

### 解決方案
偵測輸出 shape，若第二維 < 第三維則轉置：`predictions.transpose()` 得到 `(8400, 5)` 格式（cx, cy, w, h, confidence）。

### 選擇理由
YOLOv11 改變了輸出 tensor layout，這是 v11 與 v5/v8 的已知差異，轉置是標準處理方式。

---

## 2026-02-19: 機器人追蹤從純 SOT 升級為 MOT+SOT 雙模式架構

### 問題
原有的機器人追蹤僅支援 SOT（單目標追蹤），需要用戶手動框選每個機器人。在 FRC 比賽中有 6 台機器人同時在場，手動框選效率低且容易因為遮擋而丟失追蹤。

### 解決方案
建立 MOT/SOT 雙模式架構：
1. **MOT 模式**（主要）：`RobotDetectorONNX` 每幀偵測 → ByteTrack 多目標關聯 → Label 映射（偵測類別 → 機器人 ID）→ 遮擋時線性插值補全位置
2. **SOT 模式**（備用）：保留原有 VitTrack/CSRT 單目標追蹤，作為無偵測模型時的 fallback
3. **`RobotTrackerManager`** 統一介面：自動偵測是否有模型和 supervision 套件，決定使用哪種模式

### 選擇理由
- **ByteTrack vs SORT/DeepSORT**: ByteTrack 不需要 Re-ID 特徵提取（額外模型），僅用 IoU + 運動模型關聯，速度快、依賴少
- **`lost_track_buffer=120`**（4 秒 @30fps）：FRC 比賽中機器人可能被其他機器人完全遮擋數秒，較長的 buffer 可避免頻繁 ID 切換
- **遮擋線性插值**: ByteTrack 丟失期間用最後已知位置和速度線性外推，比直接跳過更適合射手歸因（需要連續位置資訊）
- **Graceful degradation**：模型不存在 → SOT；supervision 未安裝 → SOT。確保應用永遠可用

---

## 2026-02-19: RobotDetectorONNX 支援 NMS-Free 和傳統 YOLO 雙格式

### 問題
不同版本的 YOLO 匯出 ONNX 時輸出格式不同：傳統 YOLO (v5/v8/v11) 輸出需要 NMS 後處理，而新版 YOLO (v10/v26, end-to-end) 輸出已經過 NMS，格式為 `(1, N, 6)` 且含 padding rows。

### 解決方案
在 `RobotDetectorONNX.__init__()` 中自動偵測輸出格式：
- 若最後一維 == 6 且第二維 >= 100 → NMS-Free 模式（過濾 padding rows by score > 0）
- 否則 → 傳統模式（手動 confidence 過濾 + `cv2.dnn.NMSBoxes`）
- 類別名稱優先從 ONNX metadata 的 `names` 欄位讀取（ultralytics 格式），fallback 到 `class_names`

### 選擇理由
- **自動偵測 vs 手動設定**: 減少用戶配置負擔，支援更多模型來源
- **metadata 讀取類別名稱**: ultralytics 匯出的 ONNX 會在 metadata 中嵌入類別名稱，利用這個特性可省去額外的 class map 設定檔

---

## 2026-02-19: 出手偵測（Shot Detection）設計

### 問題
原有系統只記錄「進球」事件，無法統計「出手未進」，因此無法計算命中率。在 FRC 分析中，命中率是評估機器人投射能力的關鍵指標。

### 解決方案
在 `ScoringEngine` 新增 `detect_shots()` 後處理方法：
1. 掃描每條球軌跡，找出速度突增點（`velocity > SHOT_MIN_VELOCITY`）作為出手候選
2. 檢查出手點附近是否有機器人（`distance < SHOT_ROBOT_PROXIMITY`），歸因射手
3. 追蹤球軌跡後續是否進入 Hub 區域 → 進球或未進
4. `RobotScore` 擴展：`auto_goals`/`teleop_goals`/`auto_misses`/`teleop_misses`/`accuracy`
5. 保持 `auto` 和 `teleop` 屬性向後相容（返回 goals 值）

### 選擇理由
- **速度突增偵測 vs 位置啟發式**: 球被射出時速度明顯增加，比「球離開機器人 bbox」更魯棒（不依賴精確的機器人邊界框）
- **後處理 vs 即時偵測**: 分析完所有幀後再統一處理出手事件，可利用完整軌跡資訊（知道球最終是否進入 Hub），避免即時判定的不確定性
- **`SHOT_MIN_VELOCITY` 和 `SHOT_ROBOT_PROXIMITY` 可調**: 不同比賽場地和攝影機角度下閾值可能需要調整

---

## 2026-02-19: Roboflow FRC 機器人資料集選擇

### 問題
需要訓練 FRC 機器人偵測模型，原預設資料集 RF100-VL FSOD（`rf100-vl-fsod`）僅 100 張圖片且 version 1 不存在（API 回傳錯誤），無法使用。

### 調研比較
| 資料集 | 圖片數 | 類別 | 評估 |
|--------|--------|------|------|
| **WorBots 4145 v8** | 3,291 | red_robot, blue_robot, black_robot, note, speaker, display | 最佳 — 資料量大、紅藍黑機器人分類、多季賽場地 |
| RF100-VL FSOD | 100 | Robot | 太小、version 1 不存在 |
| AutoNav / a-xvsqd | 112 | Robot | 太小、僅單一類別 |

### 解決方案
將 `train_robot_model.py` 預設資料集改為 WorBots 4145 v8：
- workspace: `worbots-4145`
- project: `worbots-4145-b7rgm`
- version: `8`
- format: `yolov11`

### 選擇理由
- **資料量**: 3,291 張遠優於 100-112 張，YOLO 訓練需要足夠資料量才能收斂
- **類別設計**: red_robot/blue_robot/black_robot 天然對應 FRC 聯盟（紅方/藍方），無需額外標註
- **多元場景**: 包含多個賽季和場地的照片，泛化能力更好
- **附帶遊戲元素**: note/speaker/display 等雖非機器人，但可作為場景理解的輔助（需在偵測時過濾）

---

## 2026-02-19: RobotDetectorONNX 類別過濾 — 區分機器人與遊戲元素

### 問題
WorBots 4145 資料集包含非機器人類別（note、speaker、display 等），若不過濾會將場地元素誤認為機器人。此外 `infer_alliance()` 原本只檢查類別名是否含 "red"/"blue"，導致 `red_display` 等非機器人物件被歸類為紅方機器人。

### 解決方案
1. **`is_robot_class(class_name)`** — 白名單判斷：類別名含 "robot" 或 "bot"，或完全等於 "red"/"blue"/"black"，才算機器人
2. **`infer_alliance()` 改進** — 先檢查是否含 "robot"，再判斷 "red"/"blue"，避免 `red_display` → Red 的誤判
3. **`detect(robot_only=True)`** — 預設過濾非機器人偵測結果，呼叫端無需額外處理

### 選擇理由
- **白名單 vs 黑名單**: 白名單更安全 — 只允許已知的機器人類別名稱模式通過，新增的非機器人類別自動被排除
- **預設過濾 vs 手動過濾**: `robot_only=True` 預設值確保不小心忘記過濾時仍然安全，需要全部偵測結果時可設為 `False`

---

## 2026-02-19: 資料集切換 — WorBots 4145 → Main Robot Detection

### 問題
WorBots 4145 資料集 (3,291 張) 雖然資料量大，但類別混雜（red_robot/blue_robot/black_robot + note/speaker/display 等遊戲元素），需要額外過濾邏輯。且 "black_robot" 類別在實際 FRC 比賽中難以歸屬聯盟（紅方或藍方）。

### 調研比較（更新）
| 資料集 | 圖片數 | 類別 | 評估 |
|--------|--------|------|------|
| WorBots 4145 v8 | 3,291 | red_robot, blue_robot, black_robot, note, speaker, display | 類別混雜，black_robot 難歸屬聯盟 |
| **Main Robot Detection v16** | 1,172 | Red, Blue | **最佳** — 純機器人底盤框選，類別乾淨直接對應聯盟 |
| RF100-VL FSOD | 100 | Robot | 太小、version 1 不存在 |

### 解決方案
1. 將 `train_robot_model.py` 預設資料集改為 Main Robot Detection (`main-wcgiu/robot-detection-xru6m/v16`)
2. `robot_detection.py` 的 `is_robot_class()` 和 `infer_alliance()` 更新，支援 "Red"/"Blue" 短類別名

### 選擇理由
- **類別乾淨**: 只有 Red/Blue 兩類，無需過濾遊戲元素，類別名直接對應聯盟
- **底盤框選**: 標註聚焦在機器人底盤（最穩定可見的部分），比整機框選更一致
- **資料量足夠**: 1,172 張對 2 類別的 YOLO 訓練已足夠收斂

---

## 2026-02-19: 類別過濾 Bug — 切換資料集導致所有偵測被過濾掉

### 問題
從 WorBots 4145 切換到 Main Robot Detection 後，`is_robot_class()` 無法辨識 "Red"/"Blue" 類別名（只認 "red_robot"/"blue_robot"），導致所有偵測結果被 `robot_only=True` 過濾掉，MOT 模式完全無輸出。

### 原因
`is_robot_class()` 原本只檢查類別名是否含 "robot" 或 "bot"，Main Robot Detection 的類別名是 "Red"/"Blue"（不含 "robot"），全部被判定為非機器人。

### 解決方案
更新 `is_robot_class()` 白名單邏輯：類別名含 "robot" 或 "bot"，**或完全等於** "red"/"blue"/"black"（不分大小寫），才算機器人。同步更新 `infer_alliance()` 支援 "Red"/"Blue" 短類別名直接推斷聯盟。

### 預防
切換訓練資料集時，必須同步檢查偵測器的類別過濾邏輯是否相容新資料集的類別名稱。

---

## 2026-02-19: 棄用 AI 球偵測模型 — 訓練資料與實際需求不匹配

### 問題
球偵測 AI 模型（`fuel_yolov11.onnx`, 10.1MB）在實際 FRC 比賽影片上效果不佳。

### 原因
訓練資料（Roboflow `fueldetection_grayscale`）是**近距離**拍攝的球照片，但實際使用場景是**廣角遠距**的比賽全場鏡頭。模型在近距離表現良好（信心值 0.67），但遠距離時球體像素過小、背景完全不同，導致偵測率極低。

### 解決方案
棄用 AI 球偵測模式，改為只使用 **HSV 模式**（黃色過濾）進行球偵測。刪除 `models/fuel_yolov11.onnx` 和 `models/fuel_yolov11.pt`。

### 選擇理由
- **HSV 模式在廣角遠距場景更穩定**: FRC 比賽球（黃色 Cargo/Note）顏色鮮明且穩定，HSV 過濾在不同距離和角度下都能偵測
- **AI 模型需重新訓練**: 要用廣角比賽影片重新標註和訓練才可能有效，但球偵測用 HSV 已夠用，不值得投入
- **瘦身效果顯著**: 刪除模型後專案從 ~370MB 降到 ~1.5MB

---
*Created: 2026-02-14*

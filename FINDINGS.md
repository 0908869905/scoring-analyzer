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

## 2026-02-20: 4K HEVC 60fps 播放瓶頸分析與優化

### 問題
影片為 4K (3840x2160) HEVC 60fps，播放時卡頓嚴重，連 1x 速度都無法流暢播放。

### 原因（兩個瓶頸）
1. **解碼瓶頸**: OpenCV VideoCapture 對 HEVC 4K 影片的 `cap.read()` 只有 ~26fps，而影片是 60fps，連 1x 播放都不夠快
2. **縮放瓶頸**: `cv2.resize` 使用 LANCZOS4 插值將 4K 縮放到顯示尺寸，每幀耗時 ~14ms

### 解決方案

#### 1. `_show_frame` 拆分為 `_show_frame` + `_render_frame`
- `_show_frame` 負責 seek + read（取得幀資料）
- `_render_frame` 負責渲染（縮放 + overlay + 顯示）
- 播放時使用順序 `cap.read()` 而非每幀 `cap.set(CAP_PROP_POS_FRAMES)` seek

#### 2. `_render_frame_playback` 快速渲染路徑
播放專用的精簡渲染，三個關鍵優化：
- **INTER_LINEAR 取代 LANCZOS4**: 從 14ms 降到 0.04ms（快 ~350 倍），視覺差異在播放中不可察覺
- **cv2.putText 取代 PIL ImageDraw**: 省去 NumPy→PIL→NumPy 來回轉換的開銷
- **精簡 overlay**: 播放時只繪製必要資訊，省去暫停時才需要的詳細標註

#### 3. `_play_loop` 重寫
- **固定 30fps 顯示率**: 不管影片原始幀率（60fps），顯示端最多 30fps，每次跳 2 幀（speed=1x 時）
- **grab() 跳過中間幀**: 小間距（<10 幀）用 `cap.grab()` 逐幀推進（~35fps，比 read 的 26fps 快），只在目標幀才 `cap.retrieve()`
- **大間距用 seek**: 需跳過 >10 幀時才用 `cap.set(CAP_PROP_POS_FRAMES)` seek

### 關鍵發現
| 指標 | 優化前 | 優化後 | 改善 |
|------|--------|--------|------|
| resize 插值 | LANCZOS4 (14ms) | INTER_LINEAR (0.04ms) | ~350x |
| 文字繪製 | PIL ImageDraw | cv2.putText | 省去格式轉換 |
| 跳幀方式 | 每幀 seek | grab() 順序推進 | ~35fps vs seek 延遲 |
| 顯示幀率 | 跟原始 fps (60) | 固定 30fps | 渲染壓力減半 |
| `cap.grab()` vs `cap.read()` | — | 35fps vs 26fps | grab 快 35% |
| `cap.set(CAP_PROP_POS_FRAMES)` | 用於每幀 | 僅大跳躍 | 壓縮影片 seek 很慢 |

### 選擇理由
- **INTER_LINEAR vs LANCZOS4**: LANCZOS4 適合靜態圖片放大（銳利邊緣），但播放時每幀只顯示 33ms，肉眼無法分辨差異，LINEAR 的速度優勢遠大於畫質損失
- **30fps 顯示率**: 人眼對 30fps 以上的流暢度感知差異很小，固定 30fps 可將解碼壓力減半且確保穩定
- **grab() vs seek**: 對壓縮影片（HEVC/H.264），seek 需要找到最近的 I-frame 再解碼到目標幀，非常慢；順序 grab() 只做解封裝（demux）不做解碼，適合跳過少量幀
- **暫停時仍用 LANCZOS4**: 暫停時用戶會仔細觀察畫面，此時不需要速度，用高品質渲染

---

## 2026-02-21: GCP 抵免額對 FRC Scoring Analyzer 的價值評估

### 問題
用戶有兩筆 Google Cloud 抵免額：Free Trial (7000 TWD) 和 GenAI App Builder (30000 TWD)，需要評估對本專案是否有用。

### 調研比較
| GCP 服務 | 費用 | 對本專案價值 | 評估 |
|----------|------|-------------|------|
| Compute Engine GPU (T4) | ~$0.54/hr (~17 TWD/hr) | 訓練機器人偵測模型 | 可用，但 Colab 免費 T4 已足夠 |
| Gemini Vision API | ~$0.002/張 | 球/機器人偵測 | 比 HSV+YOLO 本地方案慢且貴，需網路 |
| Video Intelligence API | 按分鐘計費 | 物件追蹤 | 泛用模型不認 FRC 物件，精度不如專用 YOLO |
| GenAI App Builder (30000 TWD) | — | 無 | 僅涵蓋 Vertex AI Search & Conversation，與視覺分析無關 |

### 結論
- **唯一值得做的事：GPU 訓練模型**，但 Google Colab 免費 T4 已能完成（實測約 30 分鐘訓完）
- **GenAI App Builder 的 30000 TWD 完全無用** — 只涵蓋搜尋和對話 AI，不涵蓋任何視覺 API
- **GCP 視覺 API 全部不如現有方案** — 本專案需要離線、即時、專用模型，雲端泛用 API 在每個維度都輸

### 選擇理由
- **Colab vs GCP Compute Engine**: Colab 免費提供 T4 GPU，設定簡單，對一次性訓練任務完全足夠，不需要花 GCP 額度
- **本地 ONNX vs 雲端 API**: 本地推理 ~50ms/幀（CPU），離線可用，無 API 成本；雲端 API 有延遲、需網路、按量計費

---

## 2026-02-21: 使用 Google Colab 免費 T4 GPU 訓練 YOLO 模型

### 問題
本機無 NVIDIA GPU，CPU 訓練 YOLOv11n 需要 ~19 小時，不實際。之前計畫借 GPU 電腦或用 GCP，但都有門檻。

### 解決方案
建立 `train_colab.ipynb` 在 Google Colab 免費 T4 GPU 上訓練：
1. 安裝 ultralytics + roboflow
2. 從 Roboflow 下載 Main Robot Detection v16 資料集 (1,172 張, Red/Blue)
3. YOLOv11n 訓練 100 epochs（實測 T4 約 30 分鐘完成）
4. 匯出 ONNX (opset=12, simplify=True, imgsz=640)
5. 下載 `frc_robot.onnx` 到本機 `models/` 目錄

### 訓練結果
- 模型大小：10.1 MB
- 類別：['Blue', 'Red']
- 輸入：640x640
- 推理驗證：本機 onnxruntime CPU 推理正常

### 選擇理由
- **Colab vs GCP VM**: Colab 零設定、免費、自帶 T4 GPU；GCP 需設定 VM、安裝 driver、可能遇 quota 問題
- **notebook 加入 .gitignore**: 含 Roboflow API key，不可提交到 git
- **opset=12 + simplify**: 確保與 onnxruntime 1.17+ 的最大相容性

---

## 2026-02-23: 4K 影片球偵測失效 — MAX_BLOB_AREA 過小 + 形態學順序錯誤

### 問題
4K (3840x2160) 影片中球偵測完全失效，分析後零球偵測。

### 原因（三個疊加因素）
1. **MAX_BLOB_AREA=10000 太小**: 4K 影片中球的 contour area 可達 30000-50000 像素，超過上限被全部過濾掉
2. **形態學順序 Open→Close 不佳**: 先 Open（去噪）再 Close（補洞），但球的 HSV mask 本身有空洞，先去噪會把碎片刪除，Close 也救不回來
3. **缺少模糊預處理**: 運動中的球有運動模糊，HSV 邊界不連續，直接 inRange 產生破碎 mask

### 解決方案
1. **MAX_BLOB_AREA 10000→50000** — 支援 4K 影片中較大的球面積
2. **形態學 Open→Close 改為 Close→Open** — 先 Close(7,7) 補洞（合併破碎區域），再 Open(3,3) 去噪（移除小雜點）
3. **新增 GaussianBlur(5,5)** — 在 HSV 轉換前做模糊預處理，平滑運動模糊邊界，讓 inRange 產生更連續的 mask

### 選擇理由
- **Close→Open vs Open→Close**: 球偵測的主要問題是 mask 破碎（空洞），不是噪點過多。先 Close 補洞再 Open 去噪的順序更適合這個場景
- **核大小 Close(7,7) + Open(3,3)**: Close 用大核確保能橋接較大空洞，Open 用小核只去除真正的小噪點
- **GaussianBlur(5,5)**: 適度模糊（5x5 不會讓球輪廓模糊到消失），足以平滑運動模糊造成的色彩不連續

---

## 2026-02-23: OpenCL UMat GPU 加速球偵測

### 問題
球偵測管線有 5 個連續的影像處理操作（GaussianBlur → cvtColor → inRange → morphologyEx × 2），全在 CPU 執行，是逐幀分析的瓶頸之一。

### 解決方案
使用 OpenCV 的 UMat（Unified Matrix）將影像資料送到 OpenCL GPU 執行：
1. `frame_gpu = cv2.UMat(frame)` — 上傳到 GPU
2. 在 GPU 上執行全部 5 個操作（GaussianBlur、cvtColor、inRange、morphologyEx close、morphologyEx open）
3. `mask = mask_gpu.get()` — 下載回 CPU（僅 mask，遠小於原始影像）
4. findContours 在 CPU 執行（OpenCV 的 findContours 不支援 UMat）

### 選擇理由
- **UMat vs CUDA API**: UMat 是 OpenCV 的跨平台 GPU 抽象，自動使用 OpenCL（Intel/AMD/NVIDIA 皆支援），無需安裝 CUDA toolkit
- **批量操作**: 5 個操作連續在 GPU 執行，避免反覆 CPU↔GPU 傳輸
- **降級安全**: 若 OpenCL 不可用，UMat 自動 fallback 到 CPU 執行，無需 try/except

---

## 2026-02-23: ONNX DirectML Provider — 不需 CUDA Toolkit 的 GPU 推理

### 問題
ONNX 推理預設用 CPU，速度受限。用戶有 NVIDIA GeForce RTX 3050 Ti，但安裝 CUDA Toolkit 門檻較高。

### 解決方案
ONNX Provider 優先順序改為：`CUDAExecutionProvider` > `DmlExecutionProvider` > `CPUExecutionProvider`
- **DmlExecutionProvider** 來自 `onnxruntime-directml` 套件（pip install 即可）
- 不需要安裝 CUDA Toolkit、cuDNN 等
- 支援所有 DirectX 12 GPU（NVIDIA、AMD、Intel）

### 選擇理由
- **DML vs CUDA**: DML 安裝簡單（pip install onnxruntime-directml），CUDA 需要裝 CUDA Toolkit + cuDNN（數 GB + 版本配對複雜）
- **三層 fallback**: CUDA（最快）> DML（方便）> CPU（通用），確保任何環境都能執行
- **診斷 print**: __init__ 印出實際使用的 Provider，方便用戶確認 GPU 是否啟用

---

## 2026-02-23: MOT 自動偵測模式 — 無需手動標記機器人

### 問題
MOT 模式（YOLO+ByteTrack）能自動偵測和追蹤機器人，但原本需要用戶先手動在影片上標記機器人（框選 + 指定 label），才能進行進球歸因。這抵消了 MOT 自動偵測的優勢。

### 解決方案
新增 `auto_mode` 機制：
1. **_MOTTracker.auto_mode**: 啟用後，ByteTrack 分配的未標記 tracker_id 自動根據偵測類別分配 label（Red-1, Red-2, Blue-1 等）
2. **RobotTrackerManager.enable_auto_mode()**: 啟用自動模式的公開 API
3. **RobotTrackerManager.robot_info property**: 回傳所有追蹤中的機器人資訊（label + 最後已知位置）
4. **app.py 整合**: 若用戶未手動標記任何機器人但 MOT 可用，自動啟用 auto_mode；分析結束後將自動偵測到的機器人加入 `_robot_markers` 供後續 UI 顯示

### 選擇理由
- **自動 vs 手動**: 對於只想快速分析的用戶，無需任何前置作業即可得到機器人歸因結果
- **Red-N/Blue-N 命名**: 按偵測類別（Red/Blue）+ 出現順序編號，清楚標示聯盟歸屬
- **回寫 _robot_markers**: 分析後將自動偵測的機器人回寫到 UI 的標記列表，讓用戶可在 UI 上看到偵測結果並手動修正

---

## 2026-02-23: ByteTrack IoU 匹配在 4K@60fps 完全失效

### 問題
MOT 模式（YOLO+ByteTrack）在 4K@60fps 影片上機器人追蹤幾乎完全失敗 — 1200 幀中只有 11 次偵測，幀覆蓋率僅 5.4%，Blue 機器人完全偵測不到。

### 原因（三個疊加因素）
1. **ByteTrack IoU=0**: 4K@60fps 下機器人偵測框僅 ~25px，但每幀移動 30-50px。IoU 需要 bbox 有重疊，但位移大於框尺寸時 IoU=0，ByteTrack 無法關聯前後幀的同一機器人
2. **閾值不匹配**: `ROBOT_DETECTION_CONFIDENCE=0.25` 但 `BYTETRACK_TRACK_THRESH=0.25`，兩者應對齊，否則低信心偵測被偵測器保留但被 ByteTrack 忽略
3. **Blue 機器人低信心**: Blue 機器人 confidence ~0.11（模型訓練資料中 Blue 標本不足），原閾值 0.25 直接過濾全部

### 解決方案
1. **繞過 ByteTrack** — 新增 `_match_direct()` 距離式直接匹配：建立距離矩陣 + 貪心最短優先分配，不依賴 IoU
2. **動態距離閾值** — `base_dist * (1 + sqrt(frame_gap / fps))`，幀間隔越大允許越遠的匹配
3. **閾值對齊** — ROBOT_DETECTION_CONFIDENCE 和 BYTETRACK_TRACK_THRESH 統一降到 0.10
4. **混合偵測** — 全幀偵測（每幀）+ tiled 偵測（每 N 幀），NMS 去重合併
5. **後處理 label 合併** — `merge_fragmented_labels()` 將碎片化的追蹤 ID 合併為連續軌跡

### 選擇理由
- **距離匹配 vs IoU 匹配**: 距離匹配對小框高速移動的場景遠比 IoU 魯棒。IoU 需要 bbox 重疊，距離只需要位置接近
- **貪心最短 vs Hungarian**: 貪心最短實作簡單且對 6 台機器人的場景足夠，Hungarian 在少量目標時優勢不大
- **動態閾值 vs 固定閾值**: 幀間隔不固定（跳幀、偵測間隔），固定距離閾值無法適應，動態縮放依幀間隔自動調整
- **保留 ByteTrack 作為 fallback**: `_match_bytetrack()` 仍可用，但 `_match_direct()` 為預設路徑

### 關鍵數據
| 指標 | ByteTrack (IoU) | 距離匹配 (Direct) |
|------|-----------------|-------------------|
| Robot 原始偵測 | 11 | 1000 |
| Robot 插值後 | 11 | 4521 |
| 幀覆蓋率 | 5.4% | 100% |
| Blue labels | 0 | 5 (2 stable) |
| ID 穩定性 | POOR (18) | FAIR (11 stable) |

---

## 2026-02-23: 全幀 vs Tiled 偵測互補策略

### 問題
單純全幀偵測（將 4K 縮放到 640x640）解析度損失太大，小機器人容易漏偵測；單純 tiled 偵測（裁切多個 tile 分別偵測）缺少全局上下文，邊界處容易漏偵測或重複偵測。

### 解決方案
混合偵測策略：
1. **全幀偵測**（每幀）：將整幀縮放到模型輸入尺寸，保留全局上下文
2. **Tiled 偵測**（每 MOT_DETECT_INTERVAL 幀）：將幀裁切為多個有 overlap 的 tile，每個 tile 獨立偵測
3. **NMS 去重**：合併兩種偵測結果，用 NMS 移除重複框

### 選擇理由
- **互補性**: 全幀有上下文優勢（不會在 tile 邊界漏偵測），tiled 有解析度優勢（小物件偵測率更高）
- **頻率差異化**: 全幀偵測成本低（單次推理），每幀都做；tiled 偵測成本高（多次推理），降頻到每 N 幀
- **tile overlap 0.15**: 確保邊界物件至少出現在一個 tile 的非邊界區域

---

## 2026-02-23: 後處理 label 合併 — merge_fragmented_labels()

### 問題
距離匹配在機器人被遮擋或暫時消失後會分配新的 label（例如 Red-1 消失後重新出現被分配為 Red-3），導致同一台機器人有多個 label，ID 穩定性差。

### 解決方案
分析結束後的後處理合併：
1. 按聯盟分組所有 label
2. 對同聯盟內的 label 對，檢查是否存在時間重疊
3. 無重疊且最後位置接近的 label 合併為同一個
4. 更新所有 frame_robots 中的 label 引用

### 選擇理由
- **後處理 vs 即時合併**: 後處理可以看到完整軌跡資訊（知道哪些 label 永遠不會同時出現），比即時判斷更準確
- **無重疊條件**: 同時出現的兩個 label 不可能是同一台機器人，這是最基本的約束

---

## 2026-02-23: 射手歸因失敗 — `_find_shooter()` 只用當前幀的機器人位置

### 問題
42 個進球事件中只有 3 個（7%）能歸因到具體機器人。絕大多數進球的 shooter 欄位為空。

### 原因
`_find_shooter()` 方法在進球判定時只查詢**當前幀**的 `robot_positions`，但 MOT 偵測器不是每幀都能偵測到所有機器人（偵測間隔、遮擋、低信心等），導致很多幀的 `robot_positions` 為空或不完整。進球發生時若該幀恰好沒偵測到附近的機器人，歸因就失敗。

### 解決方案
新增 `reattribute_shooters()` 後處理方法，在 `merge_fragmented_labels()` + `interpolate()` 完成之後執行：
1. 用**完整的**（合併 + 插值後的）`robot_positions_by_frame` 作為查詢資料來源
2. 對每個進球事件，回溯球軌跡的 lookback 視窗（默認 10 幀）
3. 每個軌跡點查詢該幀的所有機器人位置，找距離最近的
4. 取所有候選中距離最短的作為射手

### 選擇理由
- **後處理 vs 修改即時判定**：後處理可以利用合併 + 插值後的完整機器人位置資料（覆蓋率 100%），而即時判定時只有原始偵測資料（覆蓋率可能很低）
- **多幀回溯 vs 單幀查詢**：球從射出到進入 Hub 需要數幀時間，回溯多幀可以找到射出瞬間機器人最近的位置
- **結果**：射手歸因率從 7%（3/42）提升到 47%（20/43）

---

## 2026-02-23: ID 碎片化 — `merge_fragmented_labels()` 合併策略過嚴

### 問題
同一台機器人被分配多個 label（如 Red-1, Red-5, Red-9 是同一台），ID 穩定性僅 FAIR（11 個 stable labels，期望 ≤6）。

### 原因（兩個因素）
1. **零重疊要求太嚴格**：原策略要求兩個 label 的幀範圍完全不重疊才合併，但追蹤過渡期間同一台機器人可能有兩個 label 短暫共存（新 label 已出現但舊 label 尚未消失）
2. **單輪合併**：只執行一輪合併，無法處理連鎖合併（A→B 合併後，B 的幀範圍變化可能使 B→C 合併成為可能）

### 解決方案
重寫 `merge_fragmented_labels()`：
1. **允許最多 5 幀重疊**（`max_overlap=5`）：過渡期兩個 label 短暫共存是正常現象，小量重疊不代表是不同機器人
2. **迭代式合併**：重複執行合併直到無法再合併，解決 A→B 合併後啟用 B→C 合併的連鎖效應
3. **重疊幀資料保留**：重疊幀中保留 long_label（幀數更多的 label）的位置資料，避免跳動
4. **拆分為子方法**：`_find_merge_candidates()` 搜尋候選對、`_check_boundary_distance()` 檢查邊界距離、`_execute_merges()` 執行合併

### 選擇理由
- **允許小重疊 vs 嚴格零重疊**：4K@60fps 下追蹤器轉換期間幾幀的重疊很常見，嚴格零重疊會錯過大量合理的合併機會
- **迭代 vs 單輪**：合併會改變幀範圍，可能使之前不滿足條件的合併變得可行，單輪無法捕捉這種連鎖效應
- **結果**：ID Stability FAIR→GOOD（11→8 stable labels），成功合併 3 對：Red-9→Red-1, Red-5→Red-6, Blue-2→Blue-1

---

## 2026-02-23: 射手歸因距離 300px 在 4K 下不足

### 問題
`reattribute_shooters()` 實作後歸因率只有 47%（20/43），剩餘 53% 的進球仍無法歸因射手。

### 原因
`SCORE_MAX_SHOOTER_DIST=300`（像素）是在 1080p 假設下設定的。在 4K (3840x2160) 影片中，機器人射出球到進入 Hub 的距離常超過 300px（球飛行軌跡在 4K 畫面上拉得更長），導致多數射出點附近的機器人因距離超出閾值而被排除。

### 解決方案
`SCORE_MAX_SHOOTER_DIST` 從 300 改為 500（像素）。

### 選擇理由
- **500 vs 更大值**：從 test_analysis.py 的輸出觀察，未歸因事件中最近機器人距離多在 300-450px 範圍，500px 足以涵蓋且不會引入太多誤歸因
- **像素閾值 vs 相對閾值**：相對閾值（如佔畫面寬度百分比）更通用，但當前只需支援 4K，直接加大像素值最簡單有效
- **長期考量**：若需支援多種解析度，應考慮根據影片解析度動態計算閾值

---

## 2026-02-23: merge_fragmented_labels() 硬編碼參數提取為可配置常數

### 問題
`merge_fragmented_labels()` 和 `_check_boundary_distance()` 中有三個硬編碼數值：最大允許重疊 5 幀、搜尋視窗 ±60 幀、邊界距離閾值 800px。這些值在 4K@60fps 的實際測試中被證明過於保守：
- 5 幀重疊太短：追蹤器過渡期間同一機器人的兩個 label 可能共存超過 5 幀
- ±60 幀搜尋視窗太窄：±1 秒@60fps，機器人消失後重新出現可能超過 1 秒
- 800px 邊界距離在 4K 下合理，但需要可調整

### 解決方案
在 `config.py` 新增三個可配置常數：
- `MOT_MERGE_MAX_OVERLAP = 15`：允許最多 15 幀重疊（原 5），覆蓋追蹤過渡期間的共存情況
- `MOT_MERGE_SEARCH_WINDOW = 180`：搜尋視窗 ±180 幀（原 ±60，即 ±3 秒@60fps），允許更長的消失-重現間隔
- `MOT_MERGE_BOUNDARY_DIST = 800`：邊界距離閾值 800px（原值不變，但從硬編碼改為可配置）

`robot_tracker.py` 的 `merge_fragmented_labels()` 和 `_check_boundary_distance()` 改為引用這些常數。

### 選擇理由
- **可配置 vs 硬編碼**：不同影片條件（解析度、幀率、攝影機角度）下最佳值不同，提取為常數讓調參更容易
- **15 幀重疊**：觀察到實際追蹤過渡期間兩個 label 共存可達 10-12 幀，15 幀留有餘裕且不會誤合併真正不同的機器人（真正不同的機器人重疊幀數遠超 15）
- **±180 幀搜尋**：3 秒@60fps 足以涵蓋大部分遮擋場景（機器人被完全遮擋後重新出現通常在 1-3 秒內），超過 3 秒的消失更可能是真正不同的機器人
- **集中在 config.py**：所有可調參數統一放在 config.py，維護一致性

---

## 2026-02-24: Color Histogram Re-ID — 外觀特徵加權距離匹配

### 問題
純距離匹配在機器人交錯穿越時容易產生 ID swap（A、B 交叉通過時距離相近，可能互換 label），且遮擋後重新出現時無法區分外觀不同的機器人。

### 解決方案
在 `_match_direct()` 的距離匹配中加入顏色直方圖相似度：
1. **`_extract_histogram(frame, bbox)`** — 從機器人 bbox 提取 HSV H+S 雙通道直方圖（16x16 bins），歸一化後作為外觀特徵
2. **EMA 更新** — 已知機器人的直方圖用指數移動平均更新（70% 舊 + 30% 新），適應光照和角度變化
3. **加權距離** — `effective_dist = spatial_dist * (1 + MOT_HISTOGRAM_WEIGHT * (1 - similarity))`，外觀越不相似距離懲罰越大

### 選擇理由
- **HSV H+S vs RGB**: HSV 的色調（H）和飽和度（S）對光照變化較魯棒，明度（V）受光照影響大故不使用
- **16x16 bins vs 更多**: 16 bins 足以區分紅/藍聯盟和不同機體配色，過多 bins 會使稀疏直方圖的比對不穩定
- **EMA 70/30 vs 固定直方圖**: 機器人旋轉、傾斜時外觀會變，EMA 讓特徵緩慢適應而不會因單幀噪音突變
- **乘法加權 vs 加法加權**: 乘法使懲罰與距離成比例（近距離時外觀不同的懲罰較小，遠距離時較大），比固定加法更合理
- **`MOT_HISTOGRAM_WEIGHT=0.4`**: 適度加權，不會讓直方圖完全主導匹配（距離仍是主要因素），但足以在距離相近時區分不同機器人

---

## 2026-02-24: Ball Ownership Tracking — 3 層歸因取代單純回溯

### 問題
原有的射手歸因僅在進球時刻回溯 N 幀找最近機器人，射手歸因率約 50%。球飛行時間長或射手已移動時容易歸因失敗。

### 解決方案
新增 `compute_ball_ownership()` 球所有權生命週期追蹤 + 3 層歸因：
1. **Ball Ownership** — 每幀掃描所有球位置，距離最近且在 `BALL_OWNERSHIP_DIST`（200px）內的機器人為 owner，owner 在球飛行中保持不變（velocity-gated transfer：球速低於閾值且有新的最近機器人才轉移）
2. **3 層歸因**（`detect_shots()` 和 `reattribute_shooters()` 統一使用）：
   - **HP 歸因**（最高優先）：球軌跡距 HP 線段 300px 內
   - **Ownership 歸因**（次高）：出手幀的球 owner
   - **Proximity 歸因**（最低）：回溯找最近機器人

### 選擇理由
- **Ownership vs 回溯**: Ownership 在球被拾取時就確定射手，不受射出後的飛行距離和射手移動影響
- **Velocity-gated transfer**: 防止球在空中飛行時因經過其他機器人附近而誤轉移 ownership
- **3 層 fallback**: HP 歸因最精確（基於球軌跡幾何關係），Ownership 次之，Proximity 作為最後手段，確保歸因率最大化
- **`BALL_OWNERSHIP_DIST=200`**: 200px 約為機器人寬度的 2-3 倍（4K 下），足以覆蓋球被持有的合理範圍

---

## 2026-02-24: Velocity Prediction — 預測位置取代最後已知位置

### 問題
`_match_direct()` 使用最後已知位置計算匹配距離，但機器人在偵測空洞期間可能已移動相當距離，導致距離計算不準確，增加 ID swap 風險。

### 解決方案
擴展 `_last_known` 為 `(f, cx, cy, cls, vx, vy)`，包含速度估計：
1. 每次匹配成功時計算速度 `(vx, vy) = (new_cx - old_cx, new_cy - old_cy) / frame_gap`
2. 匹配時使用預測位置 `(cx + vx * gap, cy + vy * gap)` 計算距離，而非直接用 `(cx, cy)`

### 選擇理由
- **簡單線性外推 vs Kalman filter**: 機器人在短時間內（幾幀到幾十幀）運動近似線性，Kalman filter 增加的複雜度對準確度提升不大
- **速度更新策略**: 每次匹配成功才更新速度，避免用插值位置計算出錯誤的速度
- **與直方圖互補**: 速度預測改善位置估計，直方圖改善身份辨識，兩者從不同維度降低 ID swap

---

## 2026-02-24: 4-Panel Debug View — 加速開發和調參

### 問題
調參和驗證時需要同時觀察多種資訊（場地遮罩效果、球所有權、機器人偵測品質），但單一 overlay 畫面資訊過於密集。

### 解決方案
F3 快捷鍵切換 4 面板 debug 視圖：
1. **Field Mask** — 場地遮罩效果（場地外區域顯示為黑色/灰色）
2. **Ball Ownership** — 球軌跡按 owner 著色（不同機器人不同顏色）
3. **Robot Detection** — 機器人偵測框 + 信心度 + label（區分真實偵測/插值）
4. **Full Overlay** — 完整標註疊加

### 選擇理由
- **F3 toggle vs 固定面板**: 正常使用時不需要 debug 面板佔用空間，toggle 方式兼顧日常使用和開發調參
- **4 面板 vs 更多/更少**: 4 面板對應 4 個核心子系統（遮罩、球、機器人、完整），覆蓋主要調參需求且不會因面板太多而看不清

---

## 2026-02-24: 競品分析 — Team 5951 (Makers Assemble) CV Scouting 系統

### 資料來源
1. **Chief Delphi 論壇貼文**: [Computer Vision Scouting](https://www.chiefdelphi.com/t/computer-vision-scouting/511642) — Team 5951 (Makers Assemble, Tel-Aviv, Israel) 的詳細技術分享
2. **YouTube Demo 影片**: [Computer Vision Scouting Demo 1](https://www.youtube.com/watch?v=Syyl-cbjPiM) — 12 秒的 4 面板即時展示（已逐秒截圖至 `research/` 目錄）

### Team 5951 系統概覽

#### 技術架構
| 元件 | 5951 方案 | 我們的方案 |
|------|-----------|-----------|
| **機器人偵測** | YOLOv8 | YOLOv11n ONNX |
| **機器人追蹤** | ByteTrack（IoU 匹配有效） | ByteTrack + 距離式直接匹配（IoU 在 4K 失效） |
| **球偵測** | HSV 閾值（嘗試過 ML，放棄） | HSV 閾值（同樣嘗試過 AI，放棄） |
| **重新辨識 (Re-ID)** | 速度預測 + **顏色直方圖** | 僅距離匹配 |
| **進球歸因** | **球 ownership 生命週期追蹤** | 進球時回溯最近機器人 |
| **效能** | >100 FPS（RTX 3060/3070） | 離線後處理 |
| **部署** | 即時分析 | 後處理分析 |

#### 影片分析（4 面板佈局）

從 Demo 影片截圖可見系統分為 4 個即時面板：

1. **左上 — "Field Mask & Crop"**
   - 灰階場地影像 + 黃色 ROI 邊界框
   - 彩色圓點標記追蹤中的機器人位置
   - 紅色邊界線劃定場地區域
   - **關鍵特徵**: 場地遮罩裁切，排除觀眾、裁判等場外干擾

2. **右上 — "Fuel → Ownership"**
   - 黑色背景上的彩色軌跡線
   - 每條軌跡按**擁有者（機器人）著色**，而非按球 ID
   - 軌跡終點集中在 Goal 區域，清楚顯示哪台機器人的球進了哪個 Goal
   - **關鍵特徵**: 球的**所有權生命週期**追蹤 — 球從被拾取到進球，全程標記是誰的球

3. **左下 — "Robots (Green=Conf, Yellow=Pend)"**
   - 原始比賽影片 + 機器人偵測 overlay
   - **綠色圓圈 + 編號** = 確認追蹤（Confirmed）
   - **黃色圓圈 + 編號** = 待確認追蹤（Pending）
   - 每台機器人有唯一編號（1-6）
   - **關鍵特徵**: 追蹤信心度視覺化，區分確定和不確定的追蹤

4. **右下 — 完整 Overlay**
   - 比賽影片 + 所有標註疊加（機器人、球、區域）

5. **底部控制欄**
   - 按鈕: `FIELD`, `GOAL`, `CLOSE`, `SAVE` — 互動式區域定義
   - 即時計數: `Robot 1: 2  Robot 2: 3  Robot 3: 3  Robot 4: 1  Robot 5: 2  Robot 6: 0`
   - 分數隨影片播放即時更新（觀察到 Robot 4: 1→2→3→4, Robot 5: 2→3→4）

### 核心差異分析

#### 1. 球所有權追蹤（Fuel Ownership） — 最大差異

**5951 的做法**: 球的生命週期中持續追蹤「這顆球屬於誰」：
- 球被機器人拾取 → 標記 ownership
- 球在空中飛行 → 維持 ownership
- 球進入 Goal → 歸因給 owner

**我們的做法**: 只在進球時刻回溯查找最近機器人：
- 球進入 Hub 區域 → 回溯 N 幀找最近機器人 → 歸因

**影響**: 5951 的方法更準確，因為球射出後飛行期間可能已遠離射手，但 ownership 已在拾取時確定。我們的回溯方法在球飛行時間長或射手移動快時容易歸因錯誤。

#### 2. 顏色直方圖 Re-ID

**5951 的做法**: 機器人被遮擋後重新出現時，用**顏色直方圖**比對外觀特徵來重新辨識：
- 提取機器人 bbox 內的顏色分佈
- 與已知機器人的歷史直方圖比對
- 結合速度預測（預測下一幀位置）來匹配

**我們的做法**: 純距離匹配 + 後處理合併（merge_fragmented_labels）：
- 僅用位置距離關聯前後幀
- 遮擋後用邊界距離 + 時間窗口嘗試合併

**影響**: 顏色直方圖能區分外觀不同的機器人（紅方和藍方天然不同色，同聯盟機器人也通常有不同的保險桿/機體配色），大幅降低 ID swap 和碎片化。

#### 3. 場地遮罩（Field Masking）

**5951 的做法**: 預先定義場地邊界多邊形，將場地外區域完全遮罩（灰階/忽略），只在場地內做偵測。

**我們的做法**: 有場地邊界過濾（偵測後移除 bbox 中心在場外的偵測），但不是前置遮罩。

**影響**: 前置遮罩效率更高（減少偵測器處理量），且更徹底（不依賴 bbox 中心判斷）。

#### 4. 追蹤信心度視覺化

**5951 的做法**: 用顏色區分追蹤狀態（綠色=確認, 黃色=待確認），讓使用者一眼看出哪些追蹤可靠。

**我們的做法**: 所有追蹤統一用聯盟顏色（紅/藍），無法區分追蹤品質。

### 我們已做對的事

| 面向 | 說明 |
|------|------|
| **HSV 球偵測** | 兩個專案都從 ML 轉回 HSV — 驗證了我們的技術決策正確 |
| **距離式匹配** | 我們的 `_match_direct()` 解決了 4K 下 IoU=0 的問題，5951 沒遇到（他們可能用較低解析度） |
| **後處理合併** | `merge_fragmented_labels()` 是 5951 沒提到的進階功能 |
| **出手偵測** | 我們有出手/命中率統計，5951 未提及此功能 |
| **多模式追蹤** | MOT/SOT 雙模式 + graceful degradation，5951 只有 MOT |

### 改進建議（按優先順序）

#### P0 — 高影響、合理工作量

1. **球所有權追蹤 (Ball Ownership Tracking)**
   - 球軌跡每幀找最近機器人，標記 owner
   - owner 在球飛行中保持不變（除非被另一機器人接住）
   - 進球時直接用 owner 歸因，取代回溯查找
   - **預期效果**: 射手歸因率從 ~50% 提升至 80%+

2. **顏色直方圖 Re-ID**
   - 機器人 bbox 提取 HSV 直方圖作為外觀特徵
   - 新偵測匹配時結合距離 + 直方圖相似度
   - **預期效果**: 大幅減少 ID swap 和 label 碎片化

#### P1 — 中等影響、低工作量

3. **前置場地遮罩**
   - 偵測前將場地外像素設為黑色或灰色
   - 減少偵測器工作量和假陽性
   - **預期效果**: 減少場外人員誤框，提升偵測速度

4. **追蹤信心度視覺化**
   - overlay 區分確認追蹤（實線/綠色）和不確定追蹤（虛線/黃色）
   - **預期效果**: 改善使用者體驗，快速判斷分析品質

5. **即時 per-robot 計數 overlay**
   - 影片上方/下方顯示每台機器人的即時進球數
   - 類似 5951 底部的 "Robot N: X" 計數欄
   - **預期效果**: 分析播放時無需切換到統計面板

#### P2 — 長期優化

6. **速度預測 (Velocity Prediction)**
   - 用歷史軌跡預測下一幀位置（Kalman filter 或簡單線性外推）
   - 結合到距離匹配中（與預測位置的距離而非當前位置）
   - **預期效果**: 改善高速移動時的匹配準確度

7. **4 面板 Debug 視圖**
   - 開發/驗證時顯示多面板（場地遮罩、球軌跡、機器人偵測、完整 overlay）
   - **預期效果**: 大幅加速開發和調參

### 關鍵學習

> **"他們在 2017 FIRST Steamworks 做 Fuel（球）追蹤，跟我們做的事情幾乎一模一樣，但他們的系統更成熟。最大的架構差異是 Ball Ownership Tracking — 他們追蹤球的整個生命週期（拾取→持有→射出→進球），而我們只在進球瞬間才回溯找射手。這個根本性的差異解釋了為什麼我們的射手歸因率偏低。"**

### 截圖索引
| 檔案 | 時間 | 主要觀察 |
|------|------|----------|
| `research/frame_00s.png` | 0s | 4 面板完整佈局、Robot 計數初始值 |
| `research/frame_01s.png` | 1s | Robot 5: 2→3 |
| `research/frame_02s.png` | 2s | Robot 5: 4，球軌跡增加 |
| `research/frame_03s.png` | 3s | 軌跡累積 |
| `research/frame_04s.png` | 4s | Robot 位置變化 |
| `research/frame_05s.png` | 5s | 場地遮罩細節可見 |
| `research/frame_06s.png` | 6s | Robot 4: 2 |
| `research/frame_07s.png` | 7s | Robot 4: 3，軌跡密集 |
| `research/frame_08s.png` | 8s | Robot 4: 3，追蹤穩定 |
| `research/frame_09s.png` | 9s | Robot 4: 4，最終統計 |

---

## 2026-02-24: 場地遮罩座標雙重偏移 — ROI 裁切後遮罩全黑

### 問題
用戶設定 ROI 裁切 + 場地邊界後，執行分析時球偵測和機器人偵測全部為 0。分析日誌顯示每幀零偵測，但 HSV 偵測器和 YOLO 偵測器本身沒有報錯。

### 原因
座標系統混淆導致雙重偏移：
1. 用戶在 ROI 裁切後的畫面上繪製場地邊界 → `_field_boundary` 儲存的是 **ROI 相對座標** `(0..rw, 0..rh)`
2. `_run_analysis()` 建立場地遮罩時（app.py 約第 1922 行），ROI 分支又對每個頂點減去 ROI 偏移 `(rx, ry)`
3. 結果：多邊形頂點座標 = `(原始值 - rx, 原始值 - ry)` → 大部分變成**負數**
4. `cv2.fillPoly` 用負數座標繪製 → 遮罩完全不包含可見畫面 → 遮罩全為零
5. `cv2.bitwise_and(frame, frame, mask=mask)` → 整幀變黑
6. 球偵測和機器人偵測在全黑幀上執行 → 全部為 0

### 解決方案
移除 ROI 分支中不必要的座標偏移，直接使用 `_field_boundary` 座標：

```python
# 修復前（錯誤）：ROI 分支中又減去偏移
pts = np.array([(int(x - rx), int(y - ry)) for x, y in self._field_boundary], dtype=np.int32)

# 修復後（正確）：直接使用 ROI 相對座標
pts = np.array([(int(x), int(y)) for x, y in self._field_boundary], dtype=np.int32)
```

### 預防
- **座標系統規則**：`_field_boundary` 在設定時就已經轉換為當前顯示座標系（ROI 裁切後 = ROI 相對座標、無 ROI = 原始座標），後續使用時**不應再次轉換**
- **場地遮罩驗證**：在 `_run_analysis()` 建立遮罩後，可加入 `assert mask.sum() > 0` 斷言，確保遮罩不是全零

---

## 2026-02-25: 機器人偵測模型從 YOLOv11n 升級到 YOLOv26n

### 問題
現有機器人偵測模型使用 YOLOv11n 訓練，YOLOv26 已發布且在偵測精度上有所提升，考慮升級以改善偵測品質（特別是 Blue 機器人低信心度問題）。

### 解決方案
更新訓練基礎設施：
1. `train_robot_model.py` 預設模型 `--model` 從 `yolo11n` 改為 `yolo26n`
2. `train_colab.ipynb` 訓練 cell 的 `YOLO("yolo11n.pt")` 改為 `YOLO("yolo26n.pt")`
3. `robot_detection.py` **無需修改** — 在 2026-02-19 建立時已支援 NMS-Free 輸出格式 `(1, N, 6)`，這正是 YOLOv26 end-to-end 模型的輸出格式

### 選擇理由
- **YOLOv26n vs YOLOv11n**: YOLOv26 架構改進（end-to-end NMS-free 設計），在同等模型大小下精度更好
- **n (nano) 變體**: 保持使用 nano 變體，平衡推理速度和精度（模型約 10MB，CPU 推理 ~50ms/幀）
- **前向相容**: `robot_detection.py` 的自動格式偵測（NMS-Free vs 傳統）確保新舊模型都能使用，無需修改推理代碼
- **待驗證**: 模型尚未重新訓練，需到 Colab T4 GPU 執行訓練後才能比較實際精度提升

---

## 2026-02-26: FRC 開源訓練資料集深度調查

### 問題
目前使用的 Main Robot Detection 資料集（1,172 張）Blue 機器人偵測品質極差（confidence ~0.11），準確度非常低。2024 年的機器人外觀跟 2026 ReBuilt（2017 Steamworks 重製）差太多，需要找 2017 或 2026 場景的訓練資料。

### 核心結論

**2017 Steamworks 機器人偵測資料集：不存在。** 2017 年 Roboflow/YOLO 在 FRC 社群尚未普及，沒有任何團隊留下機器人偵測標註資料。2017 年的 FRC CV 全部用於車載視覺定位（反光膠帶偵測），不是從廣播影片偵測機器人。

**2026 ReBuilt 機器人偵測資料集：也不存在。** 所有 2026 資料集都只有 Fuel（球）偵測，沒有機器人偵測。

**社群共識：偵測 bumper 而非整台機器人。** Roboflow 官方和多個 CD 帖子都建議標註紅藍 bumper，因為 bumper 規格（高度、形狀、尺寸、顏色）每年都一樣，不會像機器人本體那樣每年完全不同。

### 搜尋空白確認

| 搜尋方向 | 結果 |
|----------|------|
| Roboflow: 2017 Steamworks 機器人偵測 | **無** |
| Roboflow: 2026 ReBuilt 機器人偵測 | **無**（只有 Fuel 球偵測） |
| GitHub: 2017 FRC robot detection | **無** |
| GitHub: 2026 FRC robot detection | **無** |
| Kaggle: FRC bumper detection | **無** |
| Reddit r/FRC: 2026 robot detection | **無** |
| Team 254 (2017 Steamworks): 機器人偵測資料 | **無**（只有車載反光膠帶偵測） |
| Team 5951 Asaf CV Scouting: 訓練資料 | **未公開**（承諾開源但至今未發布） |
| Dataset Colab (Team 4169): 30K+ 圖片 | **repo 幾乎空的，網站已停擺** |

### Roboflow 上的所有 FRC Red/Blue 機器人資料集

| 資料集 | 圖片數 | 類別 | 年份 | 評估 |
|--------|--------|------|------|------|
| RF 100 VL — 2024 FRC | 1,402 | Blue-Robot, Red-Robot 等 | 2024 | 品質好但 2024 外觀 |
| FRC v2025 Reefscape | ~1,700 | Blue Robot, Red Robot | 2025 | 有 Red/Blue 但 2025 外觀 |
| FRC v2025 (Team 611) | 1,713 | Blue Robot, Red Robot | 2025 | 同上 |
| Main Robot Detection（目前用）| 1,172 | Red, Blue | 混合 | Blue 品質差 |
| FRC Scouting Application (4739) | 139 | blue, red | 混合 | 太少 |
| FRC robots (frc-08aim) | 122 | Blue, Red | 混合 | 太少 |
| frc robot pov (Greg Zetko) | 71 | blue bumper, red bumper | 2023 | Bumper 標註但太少 |
| FRC Test (Frc scouting data) | 6,470 | 不明 | 不明 | **最大但類別未確認** |

### 其他大型通用資料集（無 Red/Blue 分類）

| 資料集 | 圖片數 | 評估 |
|--------|--------|------|
| FRC 2024 (FRCDRHS/Michael Jansen) | 3,260 | 只有 "robot" 一類 |
| WorBots 4145 v8 | 3,291 | 混雜 game pieces |
| Robot Detection Part Two (Charlie) | 2,059 | CD 提及，需確認 |

### 2026 ReBuilt Fuel（球）偵測資料集

| 資料集 | 圖片數 | 說明 |
|--------|--------|------|
| 2026 WiredCat Fuel Detection | 2,784 | 最大 |
| FRC 2026 Fuel (frcroboraiders) | 706 | — |
| FRC 2026 Fuel (-wrw23) | 464 | — |
| FRC 2026 Fuel (J. Pankratz) | 363 | — |
| Fuel FRC (adasdsd) | ~589 | 可能是 2017 原始資料（灰階） |

### CD 論壇重要發現

1. **Collaborative Image Labeling (2026)**
   連結: https://www.chiefdelphi.com/t/collaborative-image-labeling-for-object-detection/510324
   自架 CVAT 伺服器 cvat.samfreund.dev，2026-01 發起，計畫標註完放 Kaggle。狀態未知。

2. **FRC Bot Detection Feasibility**
   連結: https://www.chiefdelphi.com/t/frc-bot-detection-feasibility/457409
   專家建議：「HSV filter 10 分鐘就能偵測紅藍 bumper，比 ML 更快」

3. **Robot Detection (PhotonVision)**
   連結: https://www.chiefdelphi.com/t/robot-detection/474254
   提到 Dataset Colab 有 4,805 張機器人圖片（但網站已停擺）

### 推薦策略

**唯一可靠的路徑：自建 2026 ReBuilt 資料集**

1. 從 YouTube 下載 2026 ReBuilt 比賽影片（Week 1+ 開始有大量影片）
2. 用 `extract_frames.py` 或 Roboflow YouTube downloader 提取幀
3. 在 Roboflow 或 CVAT 標註 **bumper**（Red / Blue），不是整台機器人
4. 目標：500-1000 張，Red 和 Blue 均衡，多場比賽多角度
5. 可先用現有資料集做 pretrain 再用 2026 資料 fine-tune

**備選：HSV Bumper 偵測（無需訓練資料）**

CD 上有人建議：直接用 HSV 色彩過濾偵測紅藍 bumper，10 分鐘就能完成。我們已有 HSV 基礎設施，可以快速嘗試。

---

## 2026-02-26: FRC 自動化視覺 Scouting 生態調查（修正版）

### 系統現狀的誠實評估

> **之前說「功能完整度已領先大部分專案」是錯誤的。** 功能多但準確度極低等於沒用。目前的核心問題是：
> - Blue 機器人偵測 confidence ~0.11（幾乎偵測不到）
> - 射手歸因率僅 47%（一半進球無法歸因）
> - ID 穩定性僅 GOOD（仍有碎片化問題）
> - 訓練資料品質差（2024 外觀不適用 2026 ReBuilt）
>
> Asaf 的系統用不到一場比賽的標註資料就能 >100 FPS 即時分析，而我們的離線分析連基本準確度都不夠。**差距不在功能而在準確度。**

### 從廣播影片做自動化 scouting 是 2024-2026 年的新興趨勢

2017 年的 FRC CV 全部是車載視覺定位（反光膠帶偵測），沒有人從廣播影片分析比賽。推動因素：YOLO 性能突飛猛進 + Roboflow 降低門檻 + 2026 ReBuilt 高投射量比賽手動計數極困難。

### 已知 FRC CV Scouting 專案

| # | 專案/團隊 | 來源 | 技術 | 開源 | 狀態 | 訓練資料 |
|---|-----------|------|------|------|------|----------|
| 1 | **Asaf (CD)** | CD 論壇 | YOLOv8+ByteTrack+HSV+直方圖Re-ID+球ownership | 承諾中 | 最成熟(>100FPS) | **未公開** |
| 2 | **NimbleValley/auto-scout** | GitHub | YOLOv8+場地分割+Homography+距離匹配 | **是** | 活躍 | 未含在 repo |
| 3 | **BBE Heat Seeker** | GitHub | YOLOv8+XGBoost+SHAP+React/Go | **是**(GPL-3.0) | CV 完成度不明 | 無 |
| 4 | **frc-robot-tracking** | GitHub | Roboflow+ByteTrack+Heatmap | **是**(MIT) | WIP | 依賴 Roboflow |
| 5 | **Team1710CIO** | CD 論壇 | YOLO 廣播畫面分離+SSIM | 無 | 開發中 | 無 |
| 6 | **frc-livescore** | GitHub | ORB+Tesseract OCR 計分板讀取 | **是** | 支援到 2017 | N/A |
| 7 | **FieldAC (Team 4904)** | GitHub | Darknet YOLO+OCR bumper 隊號 | **是** | 未知 | 無 |

### 可參考的非 FRC 運動視覺分析

| 專案 | 可借鏡重點 |
|------|-----------|
| Tony-Luna/soccer-video-analytics | bbox 重疊球持有比純距離更精確；Homography 鳥瞰圖 |
| roboflow/sports | 自動場地校正（keypoint detection） |
| computer-vision-football-analysis | 攝影機運動補償 pan/zoom |

### 我們最需要改進的（按優先順序）

1. **訓練資料品質** — 根本問題，沒有好的資料什麼演算法都沒用
2. **Blue 機器人偵測** — confidence ~0.11 基本上等於偵測不到
3. **射手歸因準確度** — 47% 離實用還很遠
4. **ID 穩定性** — 碎片化導致歸因錯誤

---

## 2026-02-26: HSV Bumper 偵測器取代 YOLO 機器人偵測

### 問題
YOLO 機器人偵測模型（`frc_robot.onnx`）依賴訓練資料，而 2017 Steamworks 和 2026 ReBuilt 均無可用資料集。現有 2024 年資料集訓練的模型 Blue 機器人 confidence 僅 ~0.11，基本偵測不到。每年機器人外觀完全不同，模型無法跨年使用。

### 解決方案
實作 `BumperDetectorHSV` 類別，用 HSV 色彩過濾偵測紅藍 bumper：
1. GaussianBlur(5,5) 預處理 → HSV 轉換
2. 紅色 hue 環繞處理（H=0~10 OR H=170~180）+ 藍色單段 inRange
3. 矩形 morphology 核（Close 9x5 + Open 5x3）適合水平 bumper 形狀
4. 面積 + 長寬比過濾 + NMS 去重
5. 與 `RobotDetectorONNX` 完全相同介面（drop-in replacement）

新增 `ROBOT_DETECTION_MODE` 設定（`"HSV"` 預設 / `"YOLO"` 備選），`app.py` 根據 config 自動選擇偵測器。

### 選擇理由
- **HSV vs YOLO**: FRC bumper 顏色（紅/藍）和形狀（水平矩形條）高度標準化，每年規格不變，HSV 過濾天然跨年份通用，無需任何訓練資料
- **CD 社群共識**: 多個 Chief Delphi 帖子和 Roboflow 官方建議偵測 bumper 而非整台機器人
- **矩形 morphology 核 vs 方形核**: bumper 是水平長條形，矩形核（9x5 close, 5x3 open）更能保留水平結構並消除垂直噪點
- **紅色 hue 環繞**: OpenCV HSV 中紅色 hue 跨越 0/180 邊界，需兩段 inRange OR 合併
- **偵測模式切換**: 保留 YOLO 作為備選，未來有好的訓練資料時可隨時切回

### 程式碼審查修復（同次 session）
1. **tuple 參數 falsy-check**: `if param:` 改為 `if param is not None:`（空 tuple `()` 是 falsy 但合法參數）
2. **mutable class_names**: `class_names: list = [...]` 改為 `class_names: tuple = (...)`（避免所有實例共享同一 list）
3. **detect_tiled 重複偵測**: HSV 模式下 `detect_tiled()` 直接回空，避免與全幀偵測重複

---

## 2026-02-27: 背景模型取代手動場地邊界 — Temporal Median 前景分離

### 問題
手動繪製場地邊界（`_field_boundary`）有多個問題：
1. 每次換影片/換攝影機角度都要重新繪製
2. 座標系統容易出錯（2026-02-24 的雙重偏移 Bug 就是這個原因）
3. 多邊形邊界是固定的，無法適應鏡頭微小移動或不同照明條件
4. 使用者體驗差 — 需要多次點擊才能完成場地邊界設定

### 解決方案
新建 `background.py`，實作 `BackgroundModel` 類別：
1. **背景建立**: 均勻取樣影片中 `BG_SAMPLE_COUNT`（50）幀 → `np.median` 計算每個像素的中位數 → 產生靜態背景圖
2. **前景遮罩**: 每幀 `cv2.absdiff(frame, background)` → 灰階 → 閾值化（`BG_FG_THRESHOLD=30`）→ `cv2.dilate`（`BG_DILATE_KERNEL=5`）填補空洞 → 前景二值遮罩
3. **整合**: `app.py` 分析開始時自動建立背景模型，每幀用前景遮罩取代舊的固定場地遮罩

同時移除所有 `_field_boundary` 相關程式碼：
- `app.py`: 移除場地邊界按鈕、點擊互動、多邊形繪製、遮罩建立邏輯
- `robot_tracker.py`: 移除 `field_boundary` 參數和 `point_in_polygon` 過濾

### 選擇理由
- **Temporal Median vs GMM (MOG2/KNN)**: Temporal Median 適合固定攝影機場景（FRC 比賽攝影機位置不變），計算簡單且結果穩定；GMM 適合需要持續適應背景變化的場景，但 FRC 比賽中背景是靜態的，GMM 的適應性反而可能把靜止的機器人學進背景
- **Median vs Mean**: Median 對離群值（偶爾出現的機器人/球）更魯棒 — 只要物體在超過 50% 的取樣幀中不在同一位置，就不會進入背景模型
- **自動 vs 手動**: 完全自動化，無需使用者操作；消除座標系統錯誤的可能性
- **每幀動態遮罩 vs 固定多邊形遮罩**: 固定遮罩只能排除場地外區域，背景模型還能排除場地內的靜態物體（記分板、護欄等）

### 已知限制
- 攝影機必須固定（若攝影機平移/縮放，背景模型失效）
- 靜止超過半數取樣幀的機器人可能被視為背景（但 FRC 比賽中機器人很少靜止不動）
- 建立背景模型需要讀取影片多次（~50 幀），分析開始前有額外延遲

---
*Created: 2026-02-14*

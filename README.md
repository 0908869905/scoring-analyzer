# FRC Scoring Analyzer

FRC 機器人進球偵測桌面應用 — HSV + AI 雙模式球偵測 + YOLO+ByteTrack 機器人多目標追蹤（MOT）+ VitTrack/CSRT 備用追蹤（SOT）+ 區域進入進球判定 + 出手偵測。

## 環境需求

- Python 3.11+
- Windows / macOS / Linux

## 安裝

```bash
pip install -r requirements.txt
```

主要套件：`opencv-contrib-python`、`customtkinter`、`numpy`、`Pillow`、`scipy`、`onnxruntime`、`supervision`

## 模型檔案

`models/` 目錄下需要以下 ONNX 模型（不包含在 git 中）：

| 檔案 | 用途 | 取得方式 |
|------|------|----------|
| `frc_robot.onnx` | 機器人偵測（MOT 模式） | Colab 訓練或 `train_robot_model.py` |
| `object_tracking_vittrack_2023sep.onnx` | VitTrack 追蹤（SOT 備用） | [OpenCV Zoo](https://github.com/opencv/opencv_zoo) |

沒有模型也能執行，但功能受限：
- 沒有 `frc_robot.onnx` → 回退到 SOT 追蹤模式（需手動框選機器人）
- 沒有 VitTrack 模型 → SOT 模式回退到 CSRT 追蹤器

### 訓練機器人偵測模型

**方式 A：Google Colab（推薦，免費 T4 GPU）**

1. 上傳 `train_colab.ipynb` 到 [Google Colab](https://colab.research.google.com/)
2. Runtime → Change runtime type → T4 GPU
3. 依序執行所有 cell
4. 下載 `frc_robot.onnx` 放到 `models/`

**方式 B：本地 GPU**

```bash
pip install roboflow ultralytics
python train_robot_model.py --api-key YOUR_ROBOFLOW_API_KEY --device cuda:0
```

詳見 `TRAIN_README.txt`。

## 啟動

```bash
# 直接啟動（從 UI 開啟影片）
python main.py

# 啟動並載入影片
python main.py "影片路徑.mov"
```

## 使用流程

1. **開啟影片** — 點「開啟影片」或啟動時帶入路徑
2. **裁切畫面**（可選）— 點「裁切畫面」框選分析區域
3. **標記 Hub** — 點「標記紅方 Hub」或「標記藍方 Hub」，點擊多邊形頂點，右鍵完成
4. **標記機器人** — 點「標記機器人」，框選機器人，輸入編號並選擇聯盟（紅/藍）
5. **開始分析** — 逐幀追蹤球與機器人，偵測進球與出手事件
6. **查看結果** — 右側面板「得分統計」和「進球事件」分頁
7. **匯出 CSV** — 儲存分析結果

## 播放控制

| 操作 | 方式 |
|------|------|
| 播放 / 暫停 | 「▶ 播放」按鈕 或 `空白鍵` |
| 拖曳進度 | 底部進度條 |
| 切換倍速 | 1x ~ 5x 倍速按鈕 |

## 檔案結構

```
scoring-analyzer/
├── main.py                # 入口點
├── app.py                 # CustomTkinter 主介面
├── config.py              # 所有常數與預設值
├── runtime_config.py      # RuntimeConfig 動態參數 + Preset 系統
├── calibration.py         # HSV 自動校正
├── settings_window.py     # 設定面板（嵌入式）
├── detection.py           # 球偵測（HSV + AI 雙模式）
├── tracking.py            # 球追蹤（CentroidTracker）
├── robot_detection.py     # 機器人偵測（YOLO ONNX）
├── robot_tracker.py       # 機器人追蹤（MOT + SOT）
├── scoring.py             # 進球判定 + 出手偵測
├── geometry.py            # 幾何工具
├── utils.py               # 字型載入、格式化
├── models/                # ONNX 模型（不含在 git 中）
├── presets/               # Preset JSON 設定檔
├── train_robot_model.py   # 模型訓練腳本
├── train_colab.ipynb      # Colab 訓練 notebook（不含在 git 中）
└── requirements.txt       # 依賴套件
```

## 給其他人使用

1. 複製整個專案資料夾（**包含 `models/` 目錄**）
2. `pip install -r requirements.txt`
3. `python main.py`

> `models/` 被 `.gitignore` 排除，git clone 不會包含模型檔案，需要另外傳。

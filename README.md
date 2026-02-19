# FRC Scoring Analyzer

FRC 機器人進球偵測桌面應用 — 使用 HSV 球偵測 + CSRT 機器人追蹤 + 區域進入進球判定。

## 環境需求

- Python 3.11+
- Windows / macOS / Linux

## 安裝

```bash
pip install -r requirements.txt
```

主要套件：`opencv-contrib-python`、`customtkinter`、`numpy`、`Pillow`、`scipy`

## 啟動

```bash
# 直接啟動（從 UI 開啟影片）
python main.py

# 啟動並載入影片
python main.py "影片路徑.mov"
```

範例：

```bash
python main.py "E:\FRC模擬賽第二天\A001_02131614_C025.mov"
```

## 使用流程

1. **開啟影片** — 啟動後點「開啟影片」或啟動時帶入路徑
2. **裁切畫面**（可選）— 點「裁切畫面」框選分析區域，聚焦場地
3. **標記 Hub** — 點「標記紅方 Hub」或「標記藍方 Hub」，在畫面上點擊多邊形頂點，右鍵完成
4. **標記機器人** — 點「標記機器人」，在畫面上框選機器人，輸入編號並選擇聯盟（紅/藍）
5. **開始分析** — 點「開始分析」，程式會逐幀追蹤球與機器人，偵測進球事件
6. **查看結果** — 右側面板「得分統計」和「進球事件」分頁檢視結果
7. **匯出 CSV** — 點「匯出 CSV」儲存分析結果

## 播放控制

| 操作 | 方式 |
|------|------|
| 播放 / 暫停 | 點擊「▶ 播放」按鈕 或 按 `空白鍵` |
| 拖曳進度 | 拖動底部進度條 |
| 切換倍速 | 點擊 `1x` / `0.5x` 按鈕 |

## 快捷鍵

| 按鍵 | 功能 |
|------|------|
| `空白鍵` | 播放 / 暫停 |
| `右鍵點擊` | 完成多邊形標記 |

## 檔案結構

```
scoring-analyzer/
├── main.py            # 入口點
├── app.py             # CustomTkinter 主介面
├── config.py          # 所有常數與設定
├── detection.py       # 黃球 HSV 偵測
├── tracking.py        # 球追蹤（CentroidTracker）
├── robot_tracker.py   # 機器人追蹤（CSRT + 模板匹配恢復）
├── scoring.py         # 進球判定引擎
├── geometry.py        # 幾何工具
├── utils.py           # 字型載入、格式化
└── requirements.txt   # 依賴套件
```

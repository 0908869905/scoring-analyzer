# Settings Window Design — HSV 校正 + 全參數調整

**日期**: 2026-02-20
**狀態**: Approved

## 需求

- 所有 `config.py` 參數可在 GUI 中即時調整
- HSV 球偵測參數調整時有即時預覽（mask + 偵測結果）
- 點擊取色：點畫面上的球 → 自動取樣 HSV 範圍
- 自動校正：多點 K-Means 演算法，離線、零依賴、< 5ms
- Preset 儲存/載入：JSON 檔，按場地命名

## 架構選擇

**方案 A：獨立設定視窗 (CTkToplevel)** — 選定

理由：參數多（6 大類 20+ 參數），需要足夠空間放 slider 和即時預覽，同時保持主視窗影片可見。

## UI 佈局

```
┌───────────────────────────────────────────────────────────────┐
│  設定    [Preset ▼ 預設值]  [儲存] [載入] [重置為預設]         │
├───────────────────────────────────────────────────────────────┤
│ [球偵測 HSV] [球追蹤] [進球判定] [出手偵測] [機器人追蹤] [比賽] │
├─────────────────────────┬─────────────────────────────────────┤
│  Slider 參數區           │  即時預覽區（僅 HSV Tab）            │
│                         │  - 原始影像 + 偵測標記                │
│                         │  - HSV Mask                          │
│                         │  - 偵測數量                           │
│                         │                                      │
│  [點擊取色] [自動校正]    │                                      │
└─────────────────────────┴─────────────────────────────────────┘
```

## 六個 Tab

| Tab | 參數 |
|-----|------|
| 球偵測 HSV | H/S/V 上下限 (6) + 面積上下限 (2) + 即時預覽 |
| 球追蹤 | MAX_MATCH_DIST, MAX_MISSED, VELOCITY_SMOOTH, MIN_TRAJ_LEN 等 |
| 進球判定 | 回溯幀數, 射手最大距離, 停留幀數, 冷卻幀數 |
| 出手偵測 | SHOT_MIN_VELOCITY, SHOT_ROBOT_PROXIMITY |
| 機器人追蹤 | 追蹤器類型, 丟失幀數, ByteTrack 參數, 偵測信心度 |
| 比賽 | AUTO_DURATION_SEC, TELEOP_START_SEC |

## 自動校正演算法

取代 Gemini（比賽現場無網路），純 OpenCV + NumPy：

1. 用戶暫停影片，按「自動校正」
2. 在畫面上點擊 3-5 顆球
3. 每個點取 30×30 像素區域 → 轉 HSV
4. K-Means (k=2) 分離球色 vs 背景
5. 取飽和度最高的 cluster（= 球）
6. 用 5th/95th 百分位 + margin 算出最佳 HSV 範圍
7. 即時預覽 mask 結果

## Preset 系統

- 目錄：`presets/`
- 格式：JSON，包含所有 6 個 Tab 的參數
- 內建：「預設值」= config.py 原始常數
- 儲存：輸入名稱 → `presets/<name>.json`
- 載入：下拉選單列出所有 preset

## 新增檔案

| 檔案 | 用途 |
|------|------|
| `settings_window.py` | 設定視窗 (CTkToplevel) |
| `runtime_config.py` | RuntimeConfig 動態參數容器 |
| `calibration.py` | K-Means 自動校正演算法 |
| `presets/default.json` | 預設值 preset |

## 修改檔案

| 檔案 | 修改 |
|------|------|
| `app.py` | 加「設定」按鈕，傳遞 RuntimeConfig |
| `scoring.py` | 加參數覆蓋支援 |
| `tracking.py` | 加參數覆蓋支援 |

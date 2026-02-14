# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FRC Scoring Analyzer — 機器人進球偵測桌面應用，使用 HSV 球偵測 + CentroidTracker + CamShift 機器人追蹤 + 區域進入進球判定。

> **分離自:** `D:\FRC\frc-video-analyzer` (2026-02-14)
> **參考:** `D:\FRC\高度分析`（球飛行時間分析器，同樣結構的 CustomTkinter 應用）

## Tech Stack

- Python 3.11+
- CustomTkinter 5 (暗色主題桌面 GUI)
- OpenCV 4.8+ (電腦視覺)
- NumPy, Pillow, SciPy
- 無需 PyTorch — 純 OpenCV 即可

## Run Commands

```bash
# 安裝依賴
pip install -r requirements.txt

# 啟動應用
python main.py

# 啟動並載入影片
python main.py path/to/video.mp4
```

## Architecture

```
scoring-analyzer/
├── config.py          # 所有常數與設定（HSV、追蹤、UI 配色）
├── detection.py       # 黃球 HSV 偵測
├── tracking.py        # 球 CentroidTracker + 軌跡縫合
├── robot_tracker.py   # 機器人追蹤（HSV 直方圖反投影 + CamShift）
├── scoring.py         # 進球判定引擎（區域進入 + 射手歸因）
├── geometry.py        # 幾何工具（線段交叉、點在矩形內）
├── utils.py           # 字型載入、格式化工具
├── app.py             # CustomTkinter 主介面（ScoringAnalyzer 類別）
├── main.py            # 入口點
└── requirements.txt   # 依賴套件
```

## Core Algorithms

1. **球偵測**: HSV 黃色過濾 (H=20-35, S=100-255, V=100-255) → morphologyEx → findContours
2. **球追蹤**: CentroidTracker (Hungarian 匹配 + EMA 速度預測) + 軌跡縫合
3. **機器人追蹤**: 使用者框選初始 bbox → HSV 直方圖 → calcBackProject → CamShift
4. **進球判定**: 球進入得分區域 → 停留 N 幀確認 → 回溯軌跡找最近機器人歸因

## Code Style

- Python PEP 8，使用 type hints
- UI 語言：繁體中文
- 配色定義在 `config.py` 的 `COLORS` dict

## Error Tracking

開發過程中遇到錯誤，請記錄到 `errors.md`。

---
*Created: 2026-02-14*

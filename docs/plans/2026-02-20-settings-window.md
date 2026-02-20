# Settings Window Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a full settings window with real-time HSV preview, auto-calibration, and preset save/load, so users can adapt detection parameters to different venue lighting conditions.

**Architecture:** New `RuntimeConfig` dataclass holds all adjustable parameters (replaces hardcoded config imports at call sites). `SettingsWindow` (CTkToplevel) provides slider UI + HSV preview. `calibration.py` implements K-Means auto-calibration. Presets stored as JSON in `presets/`.

**Tech Stack:** Python 3.11+, CustomTkinter 5, OpenCV 4.9+, NumPy, JSON

---

## Task 1: RuntimeConfig — Dynamic Parameter Container

**Files:**
- Create: `runtime_config.py`

**Step 1: Create RuntimeConfig dataclass**

```python
"""FRC Scoring Analyzer — 動態參數容器"""

import json
import os
from dataclasses import dataclass, field, asdict

from config import (
    YELLOW_LOW, YELLOW_HIGH, MIN_BLOB_AREA, MAX_BLOB_AREA,
    MAX_MATCH_DIST, MAX_MISSED, VELOCITY_SMOOTH, MIN_TRAJ_LEN,
    AREA_WEIGHT, AREA_SCALE, MAX_AREA_RATIO, STITCH_AMBIGUITY_RATIO,
    SCORE_PROXIMITY_FRAMES, SCORE_MAX_SHOOTER_DIST,
    SCORE_ZONE_DWELL_FRAMES, SCORE_COOLDOWN_FRAMES,
    SHOT_MIN_VELOCITY, SHOT_ROBOT_PROXIMITY,
    ROBOT_TRACKER_TYPE, ROBOT_MAX_LOST_FRAMES,
    ROBOT_DETECTION_CONFIDENCE, ROBOT_DETECTION_NMS_IOU,
    BYTETRACK_TRACK_THRESH, BYTETRACK_LOST_BUFFER,
    BYTETRACK_MATCH_THRESH, BYTETRACK_MIN_CONSECUTIVE,
    AUTO_DURATION_SEC, TELEOP_START_SEC,
    VITTRACK_SCORE_THRESHOLD,
    AI_CONFIDENCE_THRESHOLD, AI_MIN_AREA, AI_MAX_AREA,
    DETECTION_MODE,
)

PRESETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "presets")


@dataclass
class RuntimeConfig:
    """所有可調參數的動態容器，初始值來自 config.py。"""

    # ── 球偵測 HSV ──
    hsv_h_low: int = YELLOW_LOW[0]
    hsv_s_low: int = YELLOW_LOW[1]
    hsv_v_low: int = YELLOW_LOW[2]
    hsv_h_high: int = YELLOW_HIGH[0]
    hsv_s_high: int = YELLOW_HIGH[1]
    hsv_v_high: int = YELLOW_HIGH[2]
    min_blob_area: int = MIN_BLOB_AREA
    max_blob_area: int = MAX_BLOB_AREA

    # ── AI 球偵測 ──
    detection_mode: str = DETECTION_MODE
    ai_confidence: float = AI_CONFIDENCE_THRESHOLD
    ai_min_area: int = AI_MIN_AREA
    ai_max_area: int = AI_MAX_AREA

    # ── 球追蹤 ──
    max_match_dist: int = MAX_MATCH_DIST
    max_missed: int = MAX_MISSED
    velocity_smooth: float = VELOCITY_SMOOTH
    min_traj_len: int = MIN_TRAJ_LEN
    area_weight: float = AREA_WEIGHT
    area_scale: int = AREA_SCALE
    max_area_ratio: float = MAX_AREA_RATIO
    stitch_ambiguity_ratio: float = STITCH_AMBIGUITY_RATIO

    # ── 進球判定 ──
    score_proximity_frames: int = SCORE_PROXIMITY_FRAMES
    score_max_shooter_dist: int = SCORE_MAX_SHOOTER_DIST
    score_zone_dwell_frames: int = SCORE_ZONE_DWELL_FRAMES
    score_cooldown_frames: int = SCORE_COOLDOWN_FRAMES

    # ── 出手偵測 ──
    shot_min_velocity: int = SHOT_MIN_VELOCITY
    shot_robot_proximity: int = SHOT_ROBOT_PROXIMITY

    # ── 機器人追蹤 ──
    robot_tracker_type: str = ROBOT_TRACKER_TYPE
    robot_max_lost_frames: int = ROBOT_MAX_LOST_FRAMES
    vittrack_score_threshold: float = VITTRACK_SCORE_THRESHOLD

    # ── 機器人偵測 (YOLO) ──
    robot_detection_confidence: float = ROBOT_DETECTION_CONFIDENCE
    robot_detection_nms_iou: float = ROBOT_DETECTION_NMS_IOU

    # ── ByteTrack ──
    bytetrack_track_thresh: float = BYTETRACK_TRACK_THRESH
    bytetrack_lost_buffer: int = BYTETRACK_LOST_BUFFER
    bytetrack_match_thresh: float = BYTETRACK_MATCH_THRESH
    bytetrack_min_consecutive: int = BYTETRACK_MIN_CONSECUTIVE

    # ── 比賽 ──
    auto_duration_sec: int = AUTO_DURATION_SEC
    teleop_start_sec: int = TELEOP_START_SEC

    @property
    def hsv_low(self) -> tuple[int, int, int]:
        return (self.hsv_h_low, self.hsv_s_low, self.hsv_v_low)

    @property
    def hsv_high(self) -> tuple[int, int, int]:
        return (self.hsv_h_high, self.hsv_s_high, self.hsv_v_high)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "RuntimeConfig":
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)

    def save_preset(self, name: str) -> str:
        os.makedirs(PRESETS_DIR, exist_ok=True)
        path = os.path.join(PRESETS_DIR, f"{name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        return path

    @classmethod
    def load_preset(cls, name: str) -> "RuntimeConfig":
        path = os.path.join(PRESETS_DIR, f"{name}.json")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def list_presets(cls) -> list[str]:
        if not os.path.isdir(PRESETS_DIR):
            return []
        return [
            f[:-5] for f in sorted(os.listdir(PRESETS_DIR))
            if f.endswith(".json")
        ]
```

**Step 2: Create default preset**

Create `presets/default.json` by saving the default RuntimeConfig:
```python
# One-time generation (or just create manually)
RuntimeConfig().save_preset("預設值")
```

**Step 3: Verify**

Run: `python -c "from runtime_config import RuntimeConfig; c = RuntimeConfig(); print(c.hsv_low, c.hsv_high); c.save_preset('test'); c2 = RuntimeConfig.load_preset('test'); print(c2.hsv_low == c.hsv_low)"`

Expected: `(20, 100, 100) (35, 255, 255)` and `True`

**Step 4: Commit**

```bash
git add runtime_config.py presets/
git commit -m "feat: add RuntimeConfig dynamic parameter container + preset system"
```

---

## Task 2: Calibration Module — K-Means Auto HSV Calibration

**Files:**
- Create: `calibration.py`

**Step 1: Create calibration module**

```python
"""FRC Scoring Analyzer — HSV 自動校正（K-Means 取色）"""

import cv2
import numpy as np


def calibrate_hsv_from_points(
    frame: np.ndarray,
    points: list[tuple[int, int]],
    patch_radius: int = 20,
    margin_h: int = 8,
    margin_s: int = 40,
    margin_v: int = 40,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """
    根據用戶點擊的球位置，自動計算最佳 HSV 範圍。

    演算法：
    1. 取每個點擊位置周圍的像素
    2. K-Means (k=2) 分離球色 vs 背景
    3. 選飽和度最高的 cluster（= 球）
    4. 用 5th/95th 百分位 + margin 算範圍

    Args:
        frame: BGR 影像
        points: [(x, y), ...] 用戶點擊的球位置
        patch_radius: 取樣半徑（像素）
        margin_h/s/v: HSV 各通道擴展 margin

    Returns:
        (hsv_low, hsv_high) — 可直接用於 cv2.inRange
    """
    if not points:
        raise ValueError("至少需要一個取樣點")

    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    all_pixels = []
    for px, py in points:
        x1 = max(0, px - patch_radius)
        y1 = max(0, py - patch_radius)
        x2 = min(w, px + patch_radius)
        y2 = min(h, py + patch_radius)
        patch = hsv[y1:y2, x1:x2].reshape(-1, 3)
        all_pixels.append(patch)

    all_pixels = np.vstack(all_pixels).astype(np.float32)

    # K-Means 分離球色 vs 背景
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(
        all_pixels, 2, None, criteria, 5, cv2.KMEANS_PP_CENTERS
    )

    # 選飽和度最高的 cluster（球是高飽和度物體）
    best_cluster = int(np.argmax(centers[:, 1]))
    ball_pixels = all_pixels[labels.flatten() == best_cluster]

    if len(ball_pixels) == 0:
        raise ValueError("無法從取樣點中分離出球的顏色")

    # 百分位數邊界（抗 outlier）
    low = np.percentile(ball_pixels, 5, axis=0).astype(int)
    high = np.percentile(ball_pixels, 95, axis=0).astype(int)

    hsv_low = (
        max(0, int(low[0]) - margin_h),
        max(0, int(low[1]) - margin_s),
        max(0, int(low[2]) - margin_v),
    )
    hsv_high = (
        min(179, int(high[0]) + margin_h),
        min(255, int(high[1]) + margin_s),
        min(255, int(high[2]) + margin_v),
    )

    return hsv_low, hsv_high


def sample_hsv_at_point(
    frame: np.ndarray,
    px: int, py: int,
    patch_radius: int = 8,
    margin_h: int = 10,
    margin_s: int = 40,
    margin_v: int = 40,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """
    單點取色：取指定位置周圍像素的 HSV 中位數 ± margin。

    比 K-Means 更快，適合單點即時預覽。

    Args:
        frame: BGR 影像
        px, py: 取樣中心點
        patch_radius: 取樣半徑

    Returns:
        (hsv_low, hsv_high)
    """
    h, w = frame.shape[:2]
    x1 = max(0, px - patch_radius)
    y1 = max(0, py - patch_radius)
    x2 = min(w, px + patch_radius)
    y2 = min(h, py + patch_radius)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    patch = hsv[y1:y2, x1:x2].reshape(-1, 3)

    median = np.median(patch, axis=0).astype(int)

    hsv_low = (
        max(0, int(median[0]) - margin_h),
        max(0, int(median[1]) - margin_s),
        max(0, int(median[2]) - margin_v),
    )
    hsv_high = (
        min(179, int(median[0]) + margin_h),
        min(255, int(median[1]) + margin_s),
        min(255, int(median[2]) + margin_v),
    )
    return hsv_low, hsv_high


def generate_preview(
    frame: np.ndarray,
    hsv_low: tuple[int, int, int],
    hsv_high: tuple[int, int, int],
    min_area: int = 150,
    max_area: int = 10000,
) -> tuple[np.ndarray, int]:
    """
    生成 HSV 遮罩預覽 + 計算偵測球數。

    Args:
        frame: BGR 影像
        hsv_low, hsv_high: HSV 範圍
        min_area, max_area: 面積過濾

    Returns:
        (mask_colored, ball_count) — mask_colored 是 BGR 彩色遮罩
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(hsv_low), np.array(hsv_high))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
    ball_count = sum(1 for c in contours if min_area <= cv2.contourArea(c) <= max_area)

    # 彩色遮罩：偵測區域高亮黃色，其餘暗色
    mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    # 將白色區域染成黃色
    mask_colored[mask > 0] = [0, 200, 255]  # BGR 黃色

    return mask_colored, ball_count
```

**Step 2: Verify**

Run: `python -c "from calibration import calibrate_hsv_from_points, generate_preview; print('calibration module OK')"`

Expected: `calibration module OK`

**Step 3: Commit**

```bash
git add calibration.py
git commit -m "feat: add HSV auto-calibration module (K-Means + point sampling)"
```

---

## Task 3: Settings Window — Core UI

**Files:**
- Create: `settings_window.py`

**Step 1: Create SettingsWindow class**

This is the largest task. The window has:
- Top bar: Preset dropdown + Save/Load/Reset buttons
- 6 tabs with sliders
- HSV tab has real-time preview (right side)

```python
"""FRC Scoring Analyzer — 設定視窗"""

import tkinter as tk
from tkinter import simpledialog, messagebox

import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image, ImageTk

from config import COLORS
from runtime_config import RuntimeConfig
from calibration import (
    calibrate_hsv_from_points,
    sample_hsv_at_point,
    generate_preview,
)


class SettingsWindow(ctk.CTkToplevel):
    """獨立設定視窗 — 全參數調整 + HSV 即時預覽 + 自動校正。"""

    def __init__(self, master, config: RuntimeConfig,
                 get_current_frame=None,
                 on_config_changed=None):
        """
        Args:
            master: 父視窗
            config: RuntimeConfig 實例（直接修改）
            get_current_frame: callback，回傳當前影片幀 (BGR np.ndarray) 或 None
            on_config_changed: callback，設定變更時呼叫
        """
        super().__init__(master)
        self.title("設定")
        self.geometry("900x620")
        self.minsize(750, 500)
        self.transient(master)

        self.config = config
        self._get_frame = get_current_frame
        self._on_changed = on_config_changed

        # 即時預覽用
        self._preview_photo = None
        self._mask_photo = None
        self._preview_after_id = None

        # 自動校正取色點
        self._calibration_points: list[tuple[int, int]] = []
        self._calibrating = False

        self._build_ui()
        self._load_values_to_sliders()
        self._schedule_preview()

    # ══════════════════════════════════════════════════
    # UI 建構
    # ══════════════════════════════════════════════════

    def _build_ui(self):
        # ── 頂部：Preset 控制列 ──
        top_bar = ctk.CTkFrame(self, fg_color=COLORS["bg_card"],
                                corner_radius=10)
        top_bar.pack(fill=tk.X, padx=8, pady=(8, 4))

        ctk.CTkLabel(top_bar, text="Preset:",
                      text_color=COLORS["text"],
                      font=ctk.CTkFont(size=13)
                      ).pack(side=tk.LEFT, padx=(12, 4))

        self._preset_var = ctk.StringVar(value="（未選擇）")
        self._preset_menu = ctk.CTkOptionMenu(
            top_bar, width=160, height=30,
            variable=self._preset_var,
            values=self._get_preset_list(),
            fg_color=COLORS["bg_secondary"],
            button_color=COLORS["border"],
            button_hover_color=COLORS["border_hover"],
            dropdown_fg_color=COLORS["bg_card"],
            dropdown_hover_color=COLORS["border"],
            text_color=COLORS["text"],
            font=ctk.CTkFont(size=12),
            command=self._on_preset_selected)
        self._preset_menu.pack(side=tk.LEFT, padx=4)

        ctk.CTkButton(
            top_bar, text="儲存", width=60, height=30,
            fg_color=COLORS["success"], hover_color=COLORS["success_hover"],
            text_color="white", corner_radius=6,
            font=ctk.CTkFont(size=12),
            command=self._save_preset
        ).pack(side=tk.LEFT, padx=4)

        ctk.CTkButton(
            top_bar, text="重置為預設", width=90, height=30,
            fg_color=COLORS["border"], hover_color=COLORS["border_hover"],
            text_color=COLORS["text"], corner_radius=6,
            font=ctk.CTkFont(size=12),
            command=self._reset_to_defaults
        ).pack(side=tk.LEFT, padx=4)

        # ── 主體：Tabs ──
        self._tabview = ctk.CTkTabview(
            self, fg_color=COLORS["bg_card"], corner_radius=12,
            segmented_button_fg_color=COLORS["border"],
            segmented_button_selected_color=COLORS["accent"],
            segmented_button_unselected_color=COLORS["bg_secondary"])
        self._tabview.pack(fill=tk.BOTH, expand=True, padx=8, pady=(4, 8))

        self._build_hsv_tab()
        self._build_tracking_tab()
        self._build_scoring_tab()
        self._build_shot_tab()
        self._build_robot_tab()
        self._build_match_tab()

    def _build_hsv_tab(self):
        """球偵測 HSV Tab — 左側 slider + 右側即時預覽。"""
        tab = self._tabview.add("球偵測 HSV")
        tab.columnconfigure(0, weight=2)
        tab.columnconfigure(1, weight=3)
        tab.rowconfigure(0, weight=1)

        # 左側：Sliders
        left = ctk.CTkScrollableFrame(tab, fg_color="transparent")
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 4))

        self._hsv_sliders = {}
        slider_defs = [
            ("hsv_h_low", "H 下限", 0, 179),
            ("hsv_h_high", "H 上限", 0, 179),
            ("hsv_s_low", "S 下限", 0, 255),
            ("hsv_s_high", "S 上限", 0, 255),
            ("hsv_v_low", "V 下限", 0, 255),
            ("hsv_v_high", "V 上限", 0, 255),
            ("min_blob_area", "面積下限", 0, 50000),
            ("max_blob_area", "面積上限", 0, 50000),
        ]
        for key, label, lo, hi in slider_defs:
            self._hsv_sliders[key] = self._add_slider(
                left, key, label, lo, hi, self._on_hsv_changed)

        # 按鈕區
        btn_frame = ctk.CTkFrame(left, fg_color="transparent")
        btn_frame.pack(fill=tk.X, pady=(12, 4))

        ctk.CTkButton(
            btn_frame, text="點擊取色", height=32, corner_radius=8,
            fg_color=COLORS["info"], hover_color="#3b82f6",
            text_color="white",
            font=ctk.CTkFont(family="Microsoft JhengHei UI", size=12),
            command=self._start_point_sample
        ).pack(side=tk.LEFT, padx=(0, 6))

        ctk.CTkButton(
            btn_frame, text="自動校正 (多點)", height=32, corner_radius=8,
            fg_color=COLORS["purple"], hover_color="#9333ea",
            text_color="white",
            font=ctk.CTkFont(family="Microsoft JhengHei UI", size=12),
            command=self._start_auto_calibrate
        ).pack(side=tk.LEFT, padx=6)

        # 校正狀態
        self._calibration_label = ctk.CTkLabel(
            left, text="",
            text_color=COLORS["text_secondary"],
            font=ctk.CTkFont(size=11))
        self._calibration_label.pack(anchor=tk.W, pady=(4, 0))

        # 右側：預覽
        right = ctk.CTkFrame(tab, fg_color="transparent")
        right.grid(row=0, column=1, sticky="nsew", padx=(4, 0))
        right.rowconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)

        # 原始影像預覽
        self._preview_canvas = tk.Canvas(
            right, bg="black", highlightthickness=0, bd=0)
        self._preview_canvas.grid(row=0, column=0, sticky="nsew", pady=(0, 2))

        # Mask 預覽
        self._mask_canvas = tk.Canvas(
            right, bg="black", highlightthickness=0, bd=0)
        self._mask_canvas.grid(row=1, column=0, sticky="nsew", pady=(2, 0))

        # 偵測數量
        self._detect_count_label = ctk.CTkLabel(
            right, text="偵測到: -- 個球",
            text_color=COLORS["accent"],
            font=ctk.CTkFont(size=12, weight="bold"))
        self._detect_count_label.grid(row=2, column=0, pady=(4, 0))

    def _build_tracking_tab(self):
        """球追蹤 Tab。"""
        tab = self._tabview.add("球追蹤")
        frame = ctk.CTkScrollableFrame(tab, fg_color="transparent")
        frame.pack(fill=tk.BOTH, expand=True)

        self._add_slider(frame, "max_match_dist", "最大匹配距離", 10, 500)
        self._add_slider(frame, "max_missed", "最大丟失幀數", 1, 60)
        self._add_slider(frame, "velocity_smooth", "速度平滑因子", 0, 100,
                          is_float=True, float_scale=100)
        self._add_slider(frame, "min_traj_len", "最少軌跡點數", 1, 30)
        self._add_slider(frame, "area_weight", "面積權重", 0, 100,
                          is_float=True, float_scale=100)
        self._add_slider(frame, "area_scale", "面積縮放", 1, 1000)
        self._add_slider(frame, "max_area_ratio", "縫合最大面積比", 100, 500,
                          is_float=True, float_scale=100)
        self._add_slider(frame, "stitch_ambiguity_ratio", "縫合歧義比率", 100, 500,
                          is_float=True, float_scale=100)

    def _build_scoring_tab(self):
        """進球判定 Tab。"""
        tab = self._tabview.add("進球判定")
        frame = ctk.CTkScrollableFrame(tab, fg_color="transparent")
        frame.pack(fill=tk.BOTH, expand=True)

        self._add_slider(frame, "score_proximity_frames", "射手回溯幀數", 1, 60)
        self._add_slider(frame, "score_max_shooter_dist", "射手最大距離 (px)", 50, 1000)
        self._add_slider(frame, "score_zone_dwell_frames", "區域停留幀數", 1, 30)
        self._add_slider(frame, "score_cooldown_frames", "進球冷卻幀數", 1, 60)

    def _build_shot_tab(self):
        """出手偵測 Tab。"""
        tab = self._tabview.add("出手偵測")
        frame = ctk.CTkScrollableFrame(tab, fg_color="transparent")
        frame.pack(fill=tk.BOTH, expand=True)

        self._add_slider(frame, "shot_min_velocity", "出手最低速度 (px/幀)", 1, 100)
        self._add_slider(frame, "shot_robot_proximity", "出手機器人距離 (px)", 50, 500)

    def _build_robot_tab(self):
        """機器人追蹤 Tab。"""
        tab = self._tabview.add("機器人追蹤")
        frame = ctk.CTkScrollableFrame(tab, fg_color="transparent")
        frame.pack(fill=tk.BOTH, expand=True)

        # 追蹤器類型（下拉）
        type_frame = ctk.CTkFrame(frame, fg_color="transparent")
        type_frame.pack(fill=tk.X, pady=(4, 0))
        ctk.CTkLabel(type_frame, text="SOT 追蹤器類型:",
                      text_color=COLORS["text"],
                      font=ctk.CTkFont(size=12)
                      ).pack(side=tk.LEFT)
        self._tracker_type_var = ctk.StringVar(
            value=self.config.robot_tracker_type)
        ctk.CTkOptionMenu(
            type_frame, width=100, height=28,
            values=["VIT", "CSRT", "KCF", "MIL"],
            variable=self._tracker_type_var,
            fg_color=COLORS["bg_secondary"],
            button_color=COLORS["border"],
            text_color=COLORS["text"],
            font=ctk.CTkFont(size=12),
            command=lambda v: self._set_config("robot_tracker_type", v)
        ).pack(side=tk.LEFT, padx=(6, 0))

        self._add_slider(frame, "robot_max_lost_frames", "SOT 最大丟失幀數", 1, 120)
        self._add_slider(frame, "vittrack_score_threshold",
                          "VitTrack 信心閾值", 0, 100,
                          is_float=True, float_scale=100)
        self._add_slider(frame, "robot_detection_confidence",
                          "YOLO 偵測信心度", 0, 100,
                          is_float=True, float_scale=100)
        self._add_slider(frame, "robot_detection_nms_iou",
                          "NMS IoU 閾值", 0, 100,
                          is_float=True, float_scale=100)

        # ByteTrack 參數
        ctk.CTkLabel(frame, text="── ByteTrack (MOT) ──",
                      text_color=COLORS["accent"],
                      font=ctk.CTkFont(size=12, weight="bold")
                      ).pack(anchor=tk.W, pady=(12, 4))

        self._add_slider(frame, "bytetrack_track_thresh",
                          "追蹤啟動閾值", 0, 100,
                          is_float=True, float_scale=100)
        self._add_slider(frame, "bytetrack_lost_buffer",
                          "丟失緩衝 (幀)", 1, 300)
        self._add_slider(frame, "bytetrack_match_thresh",
                          "匹配閾值", 0, 100,
                          is_float=True, float_scale=100)
        self._add_slider(frame, "bytetrack_min_consecutive",
                          "最少連續幀", 1, 10)

    def _build_match_tab(self):
        """比賽 Tab。"""
        tab = self._tabview.add("比賽")
        frame = ctk.CTkScrollableFrame(tab, fg_color="transparent")
        frame.pack(fill=tk.BOTH, expand=True)

        self._add_slider(frame, "auto_duration_sec", "Auto 時長 (秒)", 0, 60)
        self._add_slider(frame, "teleop_start_sec", "Teleop 開始 (秒)", 0, 60)

        # 偵測模式
        det_frame = ctk.CTkFrame(frame, fg_color="transparent")
        det_frame.pack(fill=tk.X, pady=(12, 0))
        ctk.CTkLabel(det_frame, text="球偵測模式:",
                      text_color=COLORS["text"],
                      font=ctk.CTkFont(size=12)
                      ).pack(side=tk.LEFT)
        self._det_mode_var = ctk.StringVar(value=self.config.detection_mode)
        ctk.CTkOptionMenu(
            det_frame, width=80, height=28,
            values=["HSV", "AI"],
            variable=self._det_mode_var,
            fg_color=COLORS["bg_secondary"],
            button_color=COLORS["border"],
            text_color=COLORS["text"],
            font=ctk.CTkFont(size=12),
            command=lambda v: self._set_config("detection_mode", v)
        ).pack(side=tk.LEFT, padx=(6, 0))

    # ══════════════════════════════════════════════════
    # Slider 工具
    # ══════════════════════════════════════════════════

    def _add_slider(self, parent, key: str, label: str,
                     lo: int, hi: int, callback=None,
                     is_float=False, float_scale=1):
        """新增一個 slider + 數值顯示。回傳 (slider, value_label) tuple。"""
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill=tk.X, pady=(4, 0))

        value = getattr(self.config, key)
        if is_float:
            display_val = value
            slider_val = int(value * float_scale)
        else:
            display_val = value
            slider_val = value

        name_label = ctk.CTkLabel(
            row, text=f"{label}:", width=160, anchor=tk.W,
            text_color=COLORS["text"],
            font=ctk.CTkFont(size=12))
        name_label.pack(side=tk.LEFT)

        val_label = ctk.CTkLabel(
            row, text=str(display_val), width=60, anchor=tk.E,
            text_color=COLORS["accent"],
            font=ctk.CTkFont(size=12, weight="bold"))
        val_label.pack(side=tk.RIGHT, padx=(4, 0))

        def on_slide(v):
            int_v = int(float(v))
            if is_float:
                real_v = int_v / float_scale
                val_label.configure(text=f"{real_v:.2f}")
                self._set_config(key, real_v)
            else:
                val_label.configure(text=str(int_v))
                self._set_config(key, int_v)
            if callback:
                callback()

        slider = ctk.CTkSlider(
            row, from_=lo, to=hi,
            number_of_steps=hi - lo,
            button_color=COLORS["accent"],
            button_hover_color=COLORS["accent_hover"],
            progress_color=COLORS["accent"],
            fg_color=COLORS["border"],
            command=on_slide)
        slider.set(slider_val)
        slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 4))

        return (slider, val_label, is_float, float_scale)

    # ══════════════════════════════════════════════════
    # Config 操作
    # ══════════════════════════════════════════════════

    def _set_config(self, key: str, value):
        """設定 RuntimeConfig 的一個欄位。"""
        setattr(self.config, key, value)
        if self._on_changed:
            self._on_changed()

    def _load_values_to_sliders(self):
        """從 config 載入所有值到 slider。"""
        for key, (slider, val_label, is_float, float_scale) in \
                self._hsv_sliders.items():
            value = getattr(self.config, key)
            if is_float:
                slider.set(int(value * float_scale))
                val_label.configure(text=f"{value:.2f}")
            else:
                slider.set(value)
                val_label.configure(text=str(value))

    def _reload_all_sliders(self):
        """重新載入所有 Tab 的 slider 值（用於 preset 載入/重置）。"""
        # 銷毀並重建所有 tab 內容比較簡單，但更輕量的做法是
        # 遍歷所有 slider widget 更新值。
        # 這裡用重建方式，因為 slider 實例散布在各 tab 中不易追蹤。
        self._tabview.destroy()
        self._tabview = ctk.CTkTabview(
            self, fg_color=COLORS["bg_card"], corner_radius=12,
            segmented_button_fg_color=COLORS["border"],
            segmented_button_selected_color=COLORS["accent"],
            segmented_button_unselected_color=COLORS["bg_secondary"])
        self._tabview.pack(fill=tk.BOTH, expand=True, padx=8, pady=(4, 8))
        self._hsv_sliders = {}
        self._build_hsv_tab()
        self._build_tracking_tab()
        self._build_scoring_tab()
        self._build_shot_tab()
        self._build_robot_tab()
        self._build_match_tab()
        self._load_values_to_sliders()
        self._schedule_preview()

    # ══════════════════════════════════════════════════
    # Preset 操作
    # ══════════════════════════════════════════════════

    def _get_preset_list(self) -> list[str]:
        presets = RuntimeConfig.list_presets()
        if not presets:
            return ["（無 preset）"]
        return presets

    def _refresh_preset_menu(self):
        presets = self._get_preset_list()
        self._preset_menu.configure(values=presets)

    def _on_preset_selected(self, name: str):
        if name == "（無 preset）":
            return
        try:
            loaded = RuntimeConfig.load_preset(name)
            # 複製所有欄位
            for f in self.config.__dataclass_fields__:
                setattr(self.config, f, getattr(loaded, f))
            self._reload_all_sliders()
            if self._on_changed:
                self._on_changed()
        except Exception as e:
            messagebox.showerror("載入失敗", f"無法載入 preset '{name}':\n{e}",
                                 parent=self)

    def _save_preset(self):
        name = simpledialog.askstring(
            "儲存 Preset",
            "請輸入 Preset 名稱（如「區域賽-台中」）:",
            parent=self)
        if not name:
            return
        name = name.strip()
        try:
            path = self.config.save_preset(name)
            self._refresh_preset_menu()
            self._preset_var.set(name)
            self._calibration_label.configure(
                text=f"已儲存: {name}",
                text_color=COLORS["success"])
        except Exception as e:
            messagebox.showerror("儲存失敗", str(e), parent=self)

    def _reset_to_defaults(self):
        default = RuntimeConfig()
        for f in self.config.__dataclass_fields__:
            setattr(self.config, f, getattr(default, f))
        self._reload_all_sliders()
        self._preset_var.set("（未選擇）")
        self._calibration_label.configure(
            text="已重置為預設值",
            text_color=COLORS["info"])
        if self._on_changed:
            self._on_changed()

    # ══════════════════════════════════════════════════
    # HSV 即時預覽
    # ══════════════════════════════════════════════════

    def _on_hsv_changed(self):
        """HSV slider 變更時，排程更新預覽。"""
        self._schedule_preview()

    def _schedule_preview(self):
        """延遲 100ms 更新預覽（避免 slider 拖曳時過度渲染）。"""
        if self._preview_after_id is not None:
            self.after_cancel(self._preview_after_id)
        self._preview_after_id = self.after(100, self._update_preview)

    def _update_preview(self):
        """更新 HSV 預覽畫面。"""
        self._preview_after_id = None

        if not self._get_frame:
            return
        frame = self._get_frame()
        if frame is None:
            return

        hsv_low = self.config.hsv_low
        hsv_high = self.config.hsv_high
        min_a = self.config.min_blob_area
        max_a = self.config.max_blob_area

        # 生成預覽
        mask_colored, ball_count = generate_preview(
            frame, hsv_low, hsv_high, min_a, max_a)

        # 在原始影像上標記偵測結果
        preview_frame = frame.copy()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(hsv_low), np.array(hsv_high))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_a <= area <= max_a:
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    r = max(4, int(np.sqrt(area / np.pi)))
                    cv2.circle(preview_frame, (cx, cy), r,
                               (0, 255, 255), 2, cv2.LINE_AA)

        # 顯示到 canvas
        self._show_on_canvas(self._preview_canvas, preview_frame)
        self._show_on_canvas(self._mask_canvas, mask_colored)
        self._detect_count_label.configure(text=f"偵測到: {ball_count} 個球")

    def _show_on_canvas(self, canvas, bgr_image):
        """將 BGR 影像縮放並顯示到 Canvas。"""
        cw = canvas.winfo_width()
        ch = canvas.winfo_height()
        if cw < 10 or ch < 10:
            cw, ch = 400, 250

        h, w = bgr_image.shape[:2]
        scale = min(cw / w, ch / h)
        new_w, new_h = int(w * scale), int(h * scale)

        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(bgr_image, (new_w, new_h), interpolation=interp)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        pil_img = Image.fromarray(rgb)
        photo = ImageTk.PhotoImage(pil_img)

        canvas.delete("all")
        offset_x = (cw - new_w) // 2
        offset_y = (ch - new_h) // 2
        canvas.create_image(offset_x, offset_y, anchor=tk.NW, image=photo)

        # 保持引用防止 GC
        if canvas is self._preview_canvas:
            self._preview_photo = photo
        else:
            self._mask_photo = photo

    # ══════════════════════════════════════════════════
    # 點擊取色
    # ══════════════════════════════════════════════════

    def _start_point_sample(self):
        """進入點擊取色模式。"""
        if not self._get_frame or self._get_frame() is None:
            self._calibration_label.configure(
                text="請先開啟影片",
                text_color=COLORS["error"])
            return
        self._calibrating = False
        self._calibration_label.configure(
            text="請在主視窗影片上點擊球的位置...",
            text_color=COLORS["info"])
        # 通知主視窗進入取色模式
        if hasattr(self.master, '_start_color_pick'):
            self.master._start_color_pick(mode="single", callback=self._on_point_sampled)

    def _on_point_sampled(self, px: int, py: int):
        """單點取色回調。"""
        frame = self._get_frame()
        if frame is None:
            return
        hsv_low, hsv_high = sample_hsv_at_point(frame, px, py)
        self.config.hsv_h_low = hsv_low[0]
        self.config.hsv_s_low = hsv_low[1]
        self.config.hsv_v_low = hsv_low[2]
        self.config.hsv_h_high = hsv_high[0]
        self.config.hsv_s_high = hsv_high[1]
        self.config.hsv_v_high = hsv_high[2]
        self._reload_all_sliders()
        self._calibration_label.configure(
            text=f"取色完成: H({hsv_low[0]}-{hsv_high[0]}) "
                 f"S({hsv_low[1]}-{hsv_high[1]}) V({hsv_low[2]}-{hsv_high[2]})",
            text_color=COLORS["success"])
        if self._on_changed:
            self._on_changed()

    # ══════════════════════════════════════════════════
    # 自動校正（多點 K-Means）
    # ══════════════════════════════════════════════════

    def _start_auto_calibrate(self):
        """進入自動校正模式（多點取色）。"""
        if not self._get_frame or self._get_frame() is None:
            self._calibration_label.configure(
                text="請先開啟影片",
                text_color=COLORS["error"])
            return
        self._calibrating = True
        self._calibration_points = []
        self._calibration_label.configure(
            text="請在主視窗點擊 3-5 顆球，右鍵完成校正",
            text_color=COLORS["purple"])
        if hasattr(self.master, '_start_color_pick'):
            self.master._start_color_pick(
                mode="multi",
                callback=self._on_calibration_point,
                finish_callback=self._finish_auto_calibrate)

    def _on_calibration_point(self, px: int, py: int):
        """收集校正點。"""
        self._calibration_points.append((px, py))
        n = len(self._calibration_points)
        self._calibration_label.configure(
            text=f"已取 {n} 個點 — 右鍵完成校正（建議 3-5 點）",
            text_color=COLORS["purple"])

    def _finish_auto_calibrate(self):
        """完成自動校正。"""
        if len(self._calibration_points) < 1:
            self._calibration_label.configure(
                text="至少需要 1 個取樣點",
                text_color=COLORS["error"])
            return

        frame = self._get_frame()
        if frame is None:
            return

        try:
            hsv_low, hsv_high = calibrate_hsv_from_points(
                frame, self._calibration_points)

            self.config.hsv_h_low = hsv_low[0]
            self.config.hsv_s_low = hsv_low[1]
            self.config.hsv_v_low = hsv_low[2]
            self.config.hsv_h_high = hsv_high[0]
            self.config.hsv_s_high = hsv_high[1]
            self.config.hsv_v_high = hsv_high[2]

            self._reload_all_sliders()
            n = len(self._calibration_points)
            self._calibration_label.configure(
                text=f"校正完成 ({n} 點): H({hsv_low[0]}-{hsv_high[0]}) "
                     f"S({hsv_low[1]}-{hsv_high[1]}) V({hsv_low[2]}-{hsv_high[2]})",
                text_color=COLORS["success"])
            if self._on_changed:
                self._on_changed()
        except Exception as e:
            self._calibration_label.configure(
                text=f"校正失敗: {e}",
                text_color=COLORS["error"])

        self._calibrating = False
        self._calibration_points = []

    # ══════════════════════════════════════════════════
    # 清理
    # ══════════════════════════════════════════════════

    def destroy(self):
        if self._preview_after_id is not None:
            self.after_cancel(self._preview_after_id)
        super().destroy()
```

**Step 2: Verify import**

Run: `python -c "from settings_window import SettingsWindow; print('settings_window module OK')"`

Expected: `settings_window module OK`

**Step 3: Commit**

```bash
git add settings_window.py
git commit -m "feat: add SettingsWindow with sliders, HSV preview, auto-calibration, presets"
```

---

## Task 4: Integrate Settings into App — Button + Color Pick Mode

**Files:**
- Modify: `app.py`

**Step 1: Add RuntimeConfig + Settings button to app.py**

In `app.py`, make these changes:

**4a. Add imports and RuntimeConfig initialization**

At the top of `app.py`, add import:
```python
from runtime_config import RuntimeConfig
from settings_window import SettingsWindow
```

In `ScoringAnalyzer.__init__()`, add after `self._detection_mode`:
```python
# 動態設定
self._runtime_config = RuntimeConfig()
self._settings_window = None  # SettingsWindow instance
```

**4b. Add "設定" button to toolbar**

In `_build_ui()`, after the `analyze_btn` block, add:
```python
self.settings_btn = ctk.CTkButton(
    toolbar, text="設定", height=32, corner_radius=8,
    fg_color=COLORS["purple"] if "purple" in COLORS else "#a855f7",
    hover_color="#9333ea",
    text_color="white",
    font=ctk.CTkFont(family="Microsoft JhengHei UI", size=12),
    command=self._open_settings)
self.settings_btn.pack(side=tk.RIGHT, padx=(4, 4), pady=6)
```

**4c. Add settings window management methods**

```python
def _open_settings(self):
    """開啟設定視窗。"""
    if self._settings_window is not None and self._settings_window.winfo_exists():
        self._settings_window.focus()
        return
    self._settings_window = SettingsWindow(
        self,
        config=self._runtime_config,
        get_current_frame=self._get_current_frame_for_preview,
        on_config_changed=self._on_settings_changed)

def _get_current_frame_for_preview(self):
    """提供當前幀給設定視窗預覽。"""
    if not self.cap:
        return None
    if self._first_frame is not None:
        return self._first_frame.copy()
    # fallback: 讀取當前幀
    self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
    ret, frame = self.cap.read()
    if ret:
        if self._roi:
            rx, ry, rw, rh = self._roi
            frame = frame[ry:ry+rh, rx:rx+rw]
        return frame
    return None

def _on_settings_changed(self):
    """設定變更回調。"""
    cfg = self._runtime_config
    self._detection_mode = cfg.detection_mode
    self._det_mode_var.set(cfg.detection_mode)
    self.auto_duration = cfg.auto_duration_sec
    self.auto_entry.delete(0, tk.END)
    self.auto_entry.insert(0, str(cfg.auto_duration_sec))
```

**4d. Add color pick mode for settings window**

```python
def _start_color_pick(self, mode="single", callback=None, finish_callback=None):
    """進入取色模式（供 SettingsWindow 呼叫）。"""
    self._color_pick_mode = mode  # "single" or "multi"
    self._color_pick_callback = callback
    self._color_pick_finish = finish_callback
    self.interaction_mode = "color_pick"
    self.canvas.config(cursor="crosshair")
    self._set_status("在影片上點擊取色位置", COLORS["info"])
```

**4e. Handle color pick clicks in existing canvas handlers**

In `_on_canvas_press()`, add a handler for color_pick mode before the existing polygon check:
```python
if self.interaction_mode == "color_pick":
    vx_int, vy_int = int(vx), int(vy)
    if self._color_pick_mode == "single":
        if self._color_pick_callback:
            self._color_pick_callback(vx_int, vy_int)
        self.interaction_mode = None
        self.canvas.config(cursor="")
        self._set_status("取色完成", COLORS["success"])
    else:  # multi
        if self._color_pick_callback:
            self._color_pick_callback(vx_int, vy_int)
        # 畫個小圓點標記
        self._show_frame(self.current_frame)
    return
```

In `_on_canvas_right_click()`, add multi-mode finish:
```python
if self.interaction_mode == "color_pick" and self._color_pick_mode == "multi":
    if self._color_pick_finish:
        self._color_pick_finish()
    self.interaction_mode = None
    self.canvas.config(cursor="")
    self._set_status("校正完成", COLORS["success"])
    return
```

**4f. Pass RuntimeConfig to analysis pipeline**

In `_run_analysis()`, replace hardcoded config values with RuntimeConfig:

Replace the `ball_tracker` initialization:
```python
cfg = self._runtime_config
ball_tracker = CentroidTracker(
    max_distance=cfg.max_match_dist,
    max_missed=cfg.max_missed)
```

Replace the ball detection function:
```python
def detect_balls(frame):
    if use_ai and ai_model is not None:
        return detect_fuel_ai(frame, ai_model,
                              confidence=cfg.ai_confidence,
                              min_area=cfg.ai_min_area,
                              max_area=cfg.ai_max_area)
    return detect_yellow_balls(frame,
                               hsv_low=cfg.hsv_low,
                               hsv_high=cfg.hsv_high,
                               min_area=cfg.min_blob_area,
                               max_area=cfg.max_blob_area)
```

Replace the `ScoringEngine` initialization:
```python
engine = ScoringEngine(
    fps=self.fps,
    auto_sec=cfg.auto_duration_sec,
    teleop_start_sec=cfg.teleop_start_sec,
    proximity_frames=cfg.score_proximity_frames,
    max_shooter_dist=cfg.score_max_shooter_dist,
    zone_dwell_frames=cfg.score_zone_dwell_frames,
    cooldown_frames=cfg.score_cooldown_frames,
    shot_min_velocity=cfg.shot_min_velocity,
    shot_robot_proximity=cfg.shot_robot_proximity,
)
```

Also update `self._detection_mode` check:
```python
use_ai = cfg.detection_mode == "AI"
```

And update the `auto_duration` reading (remove the `auto_entry` parsing, use config directly):
```python
# 在 _on_analyze 開頭讀取 auto 設定時：
try:
    auto_val = float(self.auto_entry.get())
    self._runtime_config.auto_duration_sec = int(auto_val)
    self._runtime_config.teleop_start_sec = int(auto_val)
except ValueError:
    pass
```

**Step 2: Verify**

Run: `python -c "from app import ScoringAnalyzer; print('app integration OK')"`

Expected: `app integration OK`

**Step 3: Commit**

```bash
git add app.py
git commit -m "feat: integrate SettingsWindow into app (button, color pick, config passing)"
```

---

## Task 5: Add Parameter Override Support to scoring.py

**Files:**
- Modify: `scoring.py`

**Step 1: Add constructor parameters for configurable constants**

Modify `ScoringEngine.__init__()` to accept all configurable parameters:

```python
def __init__(self, fps=30.0, auto_sec=AUTO_DURATION_SEC,
             teleop_start_sec=TELEOP_START_SEC,
             proximity_frames=SCORE_PROXIMITY_FRAMES,
             max_shooter_dist=SCORE_MAX_SHOOTER_DIST,
             zone_dwell_frames=SCORE_ZONE_DWELL_FRAMES,
             cooldown_frames=SCORE_COOLDOWN_FRAMES,
             shot_min_velocity=SHOT_MIN_VELOCITY,
             shot_robot_proximity=SHOT_ROBOT_PROXIMITY):
    self.fps = fps
    self.auto_end_frame = int(auto_sec * fps)
    self.teleop_start_frame = int(teleop_start_sec * fps)
    self._proximity_frames = proximity_frames
    self._max_shooter_dist = max_shooter_dist
    self._zone_dwell_frames = zone_dwell_frames
    self._cooldown_frames = cooldown_frames
    self._shot_min_velocity = shot_min_velocity
    self._shot_robot_proximity = shot_robot_proximity

    self.zones = []
    self.events = []
    self.shot_events = []
    self.robot_scores = {}
    self._ball_zone_frames = {}
    self._ball_in_zone = {}
    self._ball_cooldown = {}
```

**Step 2: Replace hardcoded constants with instance variables**

In `process_frame()`:
- Replace `SCORE_ZONE_DWELL_FRAMES` → `self._zone_dwell_frames`
- Replace `SCORE_COOLDOWN_FRAMES` → `self._cooldown_frames`

In `_find_shooter()`:
- Replace `SCORE_PROXIMITY_FRAMES` → `self._proximity_frames`
- Replace `SCORE_MAX_SHOOTER_DIST` → `self._max_shooter_dist`

In `_nearest_robot_now()`:
- Replace `SCORE_MAX_SHOOTER_DIST` → `self._max_shooter_dist`

In `detect_shots()`:
- Replace `SHOT_MIN_VELOCITY` → `self._shot_min_velocity`
- Replace `SHOT_ROBOT_PROXIMITY` → `self._shot_robot_proximity`

**Step 3: Verify**

Run: `python -c "from scoring import ScoringEngine; e = ScoringEngine(proximity_frames=20); print(e._proximity_frames)"`

Expected: `20`

**Step 4: Commit**

```bash
git add scoring.py
git commit -m "refactor: make ScoringEngine configurable via constructor parameters"
```

---

## Task 6: Sync Right-Panel Settings with RuntimeConfig

**Files:**
- Modify: `app.py`

**Step 1: Sync existing right-panel settings with RuntimeConfig**

The right panel currently has `auto_entry` and `det_mode_menu` widgets that duplicate config. Ensure they stay in sync:

- When the user changes `auto_entry`, update `self._runtime_config.auto_duration_sec`
- When `det_mode_menu` changes, update `self._runtime_config.detection_mode`
- When settings window changes, update `auto_entry` and `det_mode_menu`

In `_on_detection_mode_change()`:
```python
def _on_detection_mode_change(self, value):
    self._detection_mode = value
    self._runtime_config.detection_mode = value
    self._ai_model = None
    mode_name = "AI 模型" if value == "AI" else "HSV 色彩過濾"
    self._set_status(f"偵測模式: {mode_name}", COLORS["info"])
```

**Step 2: Commit**

```bash
git add app.py
git commit -m "fix: sync right-panel settings with RuntimeConfig"
```

---

## Task 7: Create Default Preset + Final Verification

**Files:**
- Create: `presets/預設值.json`
- Modify: `CLAUDE.md` (architecture section)

**Step 1: Generate default preset**

Run: `python -c "from runtime_config import RuntimeConfig; RuntimeConfig().save_preset('預設值'); print('default preset created')"`

**Step 2: End-to-end verification**

Run: `python -c "
from runtime_config import RuntimeConfig
from calibration import calibrate_hsv_from_points, generate_preview, sample_hsv_at_point
from settings_window import SettingsWindow

# Test RuntimeConfig
c = RuntimeConfig()
assert c.hsv_low == (20, 100, 100)
d = c.to_dict()
c2 = RuntimeConfig.from_dict(d)
assert c2.hsv_high == c.hsv_high

# Test preset
c.save_preset('test_verify')
c3 = RuntimeConfig.load_preset('test_verify')
assert c3.max_match_dist == c.max_match_dist

# Test preset list
presets = RuntimeConfig.list_presets()
assert '預設值' in presets
assert 'test_verify' in presets

print('All verifications passed!')
"`

Clean up: `python -c "import os; os.remove('presets/test_verify.json')"`

**Step 3: Update CLAUDE.md architecture**

Add new files to the architecture section:
```
├── runtime_config.py      # RuntimeConfig 動態參數容器 + Preset 系統
├── calibration.py         # HSV 自動校正（K-Means + 單點取色）
├── settings_window.py     # 設定視窗（CTkToplevel，全參數 slider + 即時預覽）
├── presets/               # Preset JSON 目錄
│   └── 預設值.json         # 預設參數值
```

**Step 4: Final commit**

```bash
git add presets/ CLAUDE.md
git commit -m "feat: complete settings window with presets, calibration, and full parameter control"
```

---

## Summary

| Task | File(s) | Description |
|------|---------|-------------|
| 1 | `runtime_config.py`, `presets/` | RuntimeConfig dataclass + preset save/load |
| 2 | `calibration.py` | K-Means auto-calibration + point sampling + preview |
| 3 | `settings_window.py` | CTkToplevel with 6 tabs, sliders, HSV preview |
| 4 | `app.py` | Settings button, color pick mode, config passing |
| 5 | `scoring.py` | Parameterize ScoringEngine constructor |
| 6 | `app.py` | Sync right-panel with RuntimeConfig |
| 7 | `presets/`, `CLAUDE.md` | Default preset + final verification |

**Dependencies:** Task 1 → Task 3 → Task 4. Task 2 → Task 3. Task 5 can be done in parallel with Task 3.

"""
FRC Scoring Analyzer — GUI 主類別
"""

import csv
import math
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, simpledialog
from pathlib import Path

import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageTk

from config import (
    COLORS, ALLIANCE_COLORS, MAX_MATCH_DIST, MAX_MISSED, MAX_ROBOTS,
    ROBOT_COLORS, ROBOT_COLORS_HEX, AUTO_DURATION_SEC, DETECTION_MODE,
    ROBOT_DETECTION_MODE,
)
from detection import (detect_yellow_balls, detect_fuel_ai, load_ai_model,
                       reset_diagnostics)
from tracking import CentroidTracker, stitch_trajectories
from robot_detection import load_robot_model, BumperDetectorHSV
from robot_tracker import RobotTrackerManager
from scoring import ScoringEngine, ScoringZone
from runtime_config import RuntimeConfig
from settings_window import SettingsPanel
from utils import load_font, format_time
from dataclasses import dataclass


@dataclass
class OcclusionZone:
    """遮擋區域定義（多邊形）。"""
    name: str
    polygon: list[tuple[int, int]]


class ScoringAnalyzer(ctk.CTk):
    def __init__(self, video_path=None):
        super().__init__()
        self.title("FRC Scoring Analyzer")
        self.geometry("1500x900")
        self.minsize(1100, 700)

        # Treeview 暗色主題
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("Dark.Treeview",
                         background=COLORS["bg_card"],
                         foreground=COLORS["text"],
                         fieldbackground=COLORS["bg_card"],
                         rowheight=28, borderwidth=0)
        style.configure("Dark.Treeview.Heading",
                         background=COLORS["border"],
                         foreground=COLORS["accent"],
                         borderwidth=0,
                         font=("Segoe UI", 10, "bold"))
        style.map("Dark.Treeview",
                   background=[("selected", COLORS["accent"])],
                   foreground=[("selected", COLORS["bg_primary"])])

        # PIL 字型
        self._overlay_font = load_font(18, bold=True)
        self._label_font = load_font(14, bold=True)
        self._small_font = load_font(12)

        # 影片狀態
        self.cap = None
        self.video_path = None
        self.total_frames = 0
        self.fps = 30.0
        self.current_frame = 0
        self.is_playing = False
        self._playback_speed = 1.0     # 播放倍速
        self._play_wall_start = 0.0    # 播放起始牆鐘時間
        self._play_start_frame = 0     # 播放起始幀
        self.photo_image = None
        self.video_width = 0
        self.video_height = 0
        self._first_frame = None  # 快取第一幀（用於標記）

        # ROI 裁切
        self._roi = None              # (x, y, w, h) 原始影片座標，None = 全畫面
        self._original_video_width = 0
        self._original_video_height = 0
        self._original_first_frame = None

        # 顯示用的偏移/縮放
        self._display_scale = 1.0
        self._display_offset_x = 0
        self._display_offset_y = 0

        # 互動模式
        self.interaction_mode = None  # None, "bumper_pick", "mark_zone_polygon", "mark_hp_line", "mark_occlusion_zone"
        self._drag_start = None      # 拖曳起始點（影片座標）
        self._drag_current = None    # 拖曳當前點
        self._polygon_points = []    # 多邊形標記中的頂點列表
        self._current_polygon_alliance = ""  # 目前正在標記的多邊形聯盟

        # HP 線段標記
        self._hp_lines = []          # [{"name": str, "alliance": str, "p1": (x,y), "p2": (x,y)}]
        self._hp_line_first_point = None  # 標記中的第一個點
        self._hp_line_alliance = ""       # 標記中的聯盟

        # 背景模型（Temporal Median，分析時自動建立）
        self._bg_model = None  # BackgroundModel 實例或 None

        # Debug 4 面板視圖
        self._debug_view = False

        # 縮放/平移
        self._zoom_level = 1.0
        self._pan_x = 0.0   # 平移偏移（影片像素）
        self._pan_y = 0.0
        self._crop_x = 0    # 目前裁切原點（影片像素）
        self._crop_y = 0
        self._pan_drag_start = None   # (canvas_x, canvas_y)
        self._pan_drag_origin = None  # (pan_x, pan_y)

        # 機器人追蹤
        self.robot_manager = RobotTrackerManager()
        self._analysis_robot_mgr = None  # 分析後的 RobotTrackerManager 引用
        self._robot_markers = []  # [(label, alliance, x, y, w, h, frame_idx)]

        # Bumper 取色模板
        self._bumper_templates = []  # [(label, alliance, histogram), ...]
        self._bumper_pick_points = []  # 取色中的點擊座標

        # 得分區域
        self._scoring_zones = []  # [ScoringZone, ...]
        self._occlusion_zones = []  # [OcclusionZone, ...]

        # 偵測模式
        self._detection_mode = DETECTION_MODE  # "AI" or "HSV"
        self._ai_model = None  # 延遲載入

        # 動態設定
        self._runtime_config = RuntimeConfig()
        self._settings_panel = None  # 在 _build_ui 中建立
        self._color_pick_mode = None
        self._color_pick_callback = None
        self._color_pick_finish = None

        # 分析引擎
        self.scoring_engine = ScoringEngine(fps=self.fps)
        self._analyzing = False
        self._analysis_cancel = False  # 分析中斷旗標
        self._analysis_done = False

        # 分析結果快取
        self._all_trajectories = {}
        self._trajectory_by_frame: dict[int, list[tuple]] = {}  # 軌跡幀索引
        self._frame_detections = {}
        self._robot_positions_cache = {}  # frame_idx -> {label: (cx, cy)}
        self._robot_bboxes_cache = {}     # frame_idx -> {label: (x1, y1, x2, y2)}
        self._robot_detected_frames = {}  # label -> set(frame_idx) 實際偵測幀
        self._cumulative_goals = {}       # frame_idx -> {label: count} 即時累計進球

        # Auto/Teleop 分界
        self.auto_duration = AUTO_DURATION_SEC

        self._build_ui()
        self._bind_keys()

        if video_path:
            self.after(100, lambda: self._open_video(video_path))

    # ══════════════════════════════════════════════════════
    # UI 建構
    # ══════════════════════════════════════════════════════

    def _build_ui(self):
        # ── 選單列 ──
        menubar = tk.Menu(self)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="開啟影片 (Ctrl+O)",
                              command=self._on_open_video)
        file_menu.add_separator()
        file_menu.add_command(label="匯出 CSV", command=self._export_csv)
        file_menu.add_separator()
        file_menu.add_command(label="結束", command=self.destroy)
        menubar.add_cascade(label="檔案", menu=file_menu)
        self.configure(menu=menubar)

        # ── 頂部狀態列（永遠可見）──
        status_bar = ctk.CTkFrame(self, fg_color=COLORS["bg_card"],
                                   corner_radius=8, height=36)
        status_bar.pack(fill=tk.X, padx=8, pady=(8, 0))
        status_bar.pack_propagate(False)

        self.status_label = ctk.CTkLabel(
            status_bar, text="就緒 — 請開啟影片",
            text_color=COLORS["text_secondary"],
            font=ctk.CTkFont(family="Microsoft JhengHei UI", size=12))
        self.status_label.pack(side=tk.LEFT, padx=10)

        ctk.CTkLabel(status_bar, text="機器人:",
                      text_color=COLORS["text_secondary"],
                      font=ctk.CTkFont(size=10)
                      ).pack(side=tk.LEFT, padx=(16, 2))
        self.robot_list_label = ctk.CTkLabel(
            status_bar, text="（無）",
            text_color=COLORS["text"],
            font=ctk.CTkFont(size=10))
        self.robot_list_label.pack(side=tk.LEFT, padx=(0, 8))

        ctk.CTkLabel(status_bar, text="區域:",
                      text_color=COLORS["text_secondary"],
                      font=ctk.CTkFont(size=10)
                      ).pack(side=tk.LEFT, padx=(0, 2))
        self.zone_list_label = ctk.CTkLabel(
            status_bar, text="（無）",
            text_color=COLORS["text"],
            font=ctk.CTkFont(size=10))
        self.zone_list_label.pack(side=tk.LEFT)

        self.progress_bar = ctk.CTkProgressBar(
            status_bar, progress_color=COLORS["accent"],
            fg_color=COLORS["border"], corner_radius=4, height=6,
            width=200)
        self.progress_bar.set(0)
        self.progress_bar.pack(side=tk.RIGHT, padx=10)
        self.progress_bar.pack_forget()

        # ── 頂層分頁 ──
        self.tabview = ctk.CTkTabview(
            self, fg_color=COLORS["bg_card"], corner_radius=12,
            segmented_button_fg_color=COLORS["border"],
            segmented_button_selected_color=COLORS["accent"],
            segmented_button_unselected_color=COLORS["bg_secondary"])
        self.tabview.pack(fill=tk.BOTH, expand=True, padx=8, pady=(4, 8))

        # ════════════════════════════════════════════
        # Tab 1: 影片分析
        # ════════════════════════════════════════════
        tab_video = self.tabview.add("影片分析")
        tab_video.rowconfigure(0, weight=1)
        tab_video.columnconfigure(0, weight=1)

        video_card = ctk.CTkFrame(tab_video, fg_color=COLORS["bg_primary"],
                                   corner_radius=8)
        video_card.grid(row=0, column=0, sticky="nsew")
        video_card.rowconfigure(0, weight=1)
        video_card.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(video_card, bg="black",
                                 highlightthickness=0, bd=0)
        self.canvas.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        self.canvas.bind("<ButtonPress-1>", self._on_canvas_press)
        self.canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_canvas_release)
        self.canvas.bind("<Double-Button-1>", self._on_canvas_double_click)
        self.canvas.bind("<ButtonPress-3>", self._on_canvas_right_click)
        self.canvas.bind("<Control-MouseWheel>", self._on_zoom)
        self.canvas.bind("<B3-Motion>", self._on_pan_drag)
        self.canvas.bind("<ButtonRelease-3>", self._on_pan_end)

        # 播放控制列
        playback = ctk.CTkFrame(tab_video, fg_color=COLORS["bg_secondary"],
                                 corner_radius=8)
        playback.grid(row=1, column=0, sticky="ew", pady=(4, 0))
        playback.columnconfigure(1, weight=1)

        self.play_btn = ctk.CTkButton(
            playback, text="▶ 播放", width=90, height=32,
            fg_color=COLORS["border"], hover_color=COLORS["border_hover"],
            text_color=COLORS["text"], corner_radius=8,
            font=ctk.CTkFont(family="Segoe UI", size=13),
            command=self._toggle_play)
        self.play_btn.grid(row=0, column=0, padx=(8, 6), pady=8)

        self.slider = ctk.CTkSlider(
            playback, from_=0, to=1, number_of_steps=1,
            button_color=COLORS["accent"],
            button_hover_color=COLORS["accent_hover"],
            progress_color=COLORS["accent"],
            fg_color=COLORS["border"],
            command=self._on_slider)
        self.slider.grid(row=0, column=1, sticky="ew", padx=4, pady=8)
        self.slider.set(0)

        self.frame_label = ctk.CTkLabel(
            playback, text="幀: 0 / 0  |  0.000s",
            text_color=COLORS["text_secondary"],
            font=ctk.CTkFont(family="Segoe UI", size=12))
        self.frame_label.grid(row=0, column=2, padx=6)

        self.speed_btn = ctk.CTkButton(
            playback, text="1x", width=50, height=28,
            fg_color=COLORS["border"], hover_color=COLORS["border_hover"],
            text_color=COLORS["accent"], corner_radius=6,
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
            command=self._toggle_speed)
        self.speed_btn.grid(row=0, column=3, padx=4)

        self.fps_label = ctk.CTkLabel(
            playback, text="FPS: --",
            text_color=COLORS["text_secondary"],
            font=ctk.CTkFont(family="Segoe UI", size=12))
        self.fps_label.grid(row=0, column=4, padx=(0, 8))

        # 工具列容器
        toolbar_wrap = ctk.CTkFrame(tab_video, fg_color="transparent")
        toolbar_wrap.grid(row=2, column=0, sticky="ew", pady=(4, 0))

        # 第一行：標記按鈕
        toolbar_row1 = ctk.CTkFrame(toolbar_wrap,
                                     fg_color=COLORS["bg_secondary"],
                                     corner_radius=8)
        toolbar_row1.pack(fill=tk.X)

        self.mark_robot_btn = ctk.CTkButton(
            toolbar_row1, text="Bumper 取色", height=30, corner_radius=8,
            fg_color=COLORS["success"], hover_color=COLORS["success_hover"],
            text_color="white",
            font=ctk.CTkFont(family="Microsoft JhengHei UI", size=11),
            command=self._start_bumper_pick)
        self.mark_robot_btn.pack(side=tk.LEFT, padx=(8, 3), pady=4)

        self.mark_red_hub_btn = ctk.CTkButton(
            toolbar_row1, text="紅方 Hub", height=30, corner_radius=8,
            fg_color="#dc2626", hover_color="#b91c1c",
            text_color="white",
            font=ctk.CTkFont(family="Microsoft JhengHei UI", size=11),
            command=lambda: self._start_mark_zone_polygon("red"))
        self.mark_red_hub_btn.pack(side=tk.LEFT, padx=3, pady=4)

        self.mark_blue_hub_btn = ctk.CTkButton(
            toolbar_row1, text="藍方 Hub", height=30, corner_radius=8,
            fg_color="#2563eb", hover_color="#1d4ed8",
            text_color="white",
            font=ctk.CTkFont(family="Microsoft JhengHei UI", size=11),
            command=lambda: self._start_mark_zone_polygon("blue"))
        self.mark_blue_hub_btn.pack(side=tk.LEFT, padx=3, pady=4)

        # 分隔
        ctk.CTkLabel(toolbar_row1, text="|",
                      text_color=COLORS["border"],
                      font=ctk.CTkFont(size=14)).pack(
            side=tk.LEFT, padx=4)

        self.mark_red_hp_btn = ctk.CTkButton(
            toolbar_row1, text="紅方 HP", height=30, corner_radius=8,
            fg_color="#dc2626", hover_color="#b91c1c",
            text_color="white",
            font=ctk.CTkFont(family="Microsoft JhengHei UI", size=11),
            command=lambda: self._start_mark_hp_line("red"))
        self.mark_red_hp_btn.pack(side=tk.LEFT, padx=3, pady=4)

        self.mark_blue_hp_btn = ctk.CTkButton(
            toolbar_row1, text="藍方 HP", height=30, corner_radius=8,
            fg_color="#2563eb", hover_color="#1d4ed8",
            text_color="white",
            font=ctk.CTkFont(family="Microsoft JhengHei UI", size=11),
            command=lambda: self._start_mark_hp_line("blue"))
        self.mark_blue_hp_btn.pack(side=tk.LEFT, padx=3, pady=4)

        # 分隔
        ctk.CTkLabel(toolbar_row1, text="|",
                      text_color=COLORS["border"],
                      font=ctk.CTkFont(size=14)).pack(
            side=tk.LEFT, padx=4)

        self.mark_occlusion_btn = ctk.CTkButton(
            toolbar_row1, text="遮擋區域", height=30, corner_radius=8,
            fg_color="#4b5563", hover_color="#6b7280",
            text_color="white",
            font=ctk.CTkFont(family="Microsoft JhengHei UI", size=11),
            command=self._start_mark_occlusion_zone)
        self.mark_occlusion_btn.pack(side=tk.LEFT, padx=3, pady=4)

        # 分隔
        ctk.CTkLabel(toolbar_row1, text="|",
                      text_color=COLORS["border"],
                      font=ctk.CTkFont(size=14)).pack(
            side=tk.LEFT, padx=4)

        self.clear_marks_btn = ctk.CTkButton(
            toolbar_row1, text="清除標記", height=30, corner_radius=8,
            fg_color=COLORS["border"], hover_color=COLORS["border_hover"],
            text_color=COLORS["text"],
            font=ctk.CTkFont(family="Microsoft JhengHei UI", size=11),
            command=self._clear_all_marks)
        self.clear_marks_btn.pack(side=tk.RIGHT, padx=(3, 8), pady=4)

        # 第二行：操作按鈕
        toolbar_row2 = ctk.CTkFrame(toolbar_wrap,
                                     fg_color=COLORS["bg_secondary"],
                                     corner_radius=8)
        toolbar_row2.pack(fill=tk.X, pady=(2, 0))

        self.analyze_quick_btn = ctk.CTkButton(
            toolbar_row2, text="分析 25 秒", height=32, corner_radius=8,
            fg_color=COLORS["accent"], hover_color=COLORS["accent_hover"],
            text_color=COLORS["bg_primary"],
            font=ctk.CTkFont(family="Microsoft JhengHei UI",
                              size=13, weight="bold"),
            command=lambda: self._on_analyze(max_seconds=25))
        self.analyze_quick_btn.pack(side=tk.LEFT, padx=(8, 2), pady=4)

        self.analyze_full_btn = ctk.CTkButton(
            toolbar_row2, text="完整分析", height=32, corner_radius=8,
            fg_color="#8e44ad", hover_color="#7d3c98",
            text_color="white",
            font=ctk.CTkFont(family="Microsoft JhengHei UI",
                              size=13, weight="bold"),
            command=lambda: self._on_analyze(max_seconds=None))
        self.analyze_full_btn.pack(side=tk.LEFT, padx=(2, 4), pady=4)

        self.analyze_cancel_btn = ctk.CTkButton(
            toolbar_row2, text="⏹ 中斷", height=32, corner_radius=8,
            fg_color=COLORS["error"], hover_color=COLORS["error_hover"],
            text_color="white",
            font=ctk.CTkFont(family="Microsoft JhengHei UI",
                              size=13, weight="bold"),
            command=self._cancel_analysis)
        # 預設隱藏，分析時才顯示

        self.crop_btn = ctk.CTkButton(
            toolbar_row2, text="裁切畫面", height=30, corner_radius=8,
            fg_color="#e67e22", hover_color="#d35400",
            text_color="white",
            font=ctk.CTkFont(family="Microsoft JhengHei UI", size=11),
            command=self._start_crop)
        self.crop_btn.pack(side=tk.LEFT, padx=3, pady=4)

        self.reset_crop_btn = ctk.CTkButton(
            toolbar_row2, text="重置裁切", height=30, corner_radius=8,
            fg_color=COLORS["border"], hover_color=COLORS["border_hover"],
            text_color=COLORS["text"],
            font=ctk.CTkFont(family="Microsoft JhengHei UI", size=11),
            command=self._reset_crop)
        self.reset_crop_btn.pack(side=tk.LEFT, padx=3, pady=4)

        self.export_btn = ctk.CTkButton(
            toolbar_row2, text="匯出 CSV", height=30, corner_radius=8,
            fg_color=COLORS["border"], hover_color=COLORS["border_hover"],
            text_color=COLORS["text"],
            font=ctk.CTkFont(family="Microsoft JhengHei UI", size=11),
            command=self._export_csv)
        self.export_btn.pack(side=tk.RIGHT, padx=(3, 8), pady=4)

        # ════════════════════════════════════════════
        # Tab 2: 設定
        # ════════════════════════════════════════════
        tab_settings = self.tabview.add("設定")
        tab_settings.rowconfigure(0, weight=1)
        tab_settings.columnconfigure(0, weight=1)

        self._settings_panel = SettingsPanel(
            tab_settings,
            config=self._runtime_config,
            get_current_frame=self._get_current_frame_for_preview,
            on_config_changed=self._on_settings_changed,
            start_color_pick=self._start_color_pick,
            get_analysis_data=self._get_analysis_data_for_preview,
            on_recompute_attribution=self._on_recompute_attribution)
        self._settings_panel.grid(row=0, column=0, sticky="nsew")

        # ════════════════════════════════════════════
        # Tab 3: 分析結果
        # ════════════════════════════════════════════
        tab_results = self.tabview.add("分析結果")
        tab_results.rowconfigure(0, weight=1)
        tab_results.columnconfigure(0, weight=1)
        tab_results.columnconfigure(1, weight=1)

        # 左半：得分統計
        score_tree_frame = ctk.CTkFrame(tab_results, fg_color="transparent")
        score_tree_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        score_tree_frame.rowconfigure(1, weight=1)
        score_tree_frame.columnconfigure(0, weight=1)

        ctk.CTkLabel(score_tree_frame, text="得分統計",
                      text_color=COLORS["accent"],
                      font=ctk.CTkFont(size=14, weight="bold")
                      ).grid(row=0, column=0, sticky="w", padx=4, pady=(4, 2))

        score_inner = ctk.CTkFrame(score_tree_frame, fg_color="transparent")
        score_inner.grid(row=1, column=0, sticky="nsew")
        score_inner.rowconfigure(0, weight=1)
        score_inner.columnconfigure(0, weight=1)

        score_cols = ("robot", "alliance", "auto", "teleop", "total",
                      "shots", "miss", "acc")
        self.score_tree = ttk.Treeview(
            score_inner, columns=score_cols, show="headings",
            style="Dark.Treeview")
        self.score_tree.heading("robot", text="機器人")
        self.score_tree.heading("alliance", text="聯盟")
        self.score_tree.heading("auto", text="Auto")
        self.score_tree.heading("teleop", text="Teleop")
        self.score_tree.heading("total", text="進球")
        self.score_tree.heading("shots", text="出手")
        self.score_tree.heading("miss", text="未進")
        self.score_tree.heading("acc", text="命中率")
        self.score_tree.column("robot", width=80, anchor=tk.CENTER)
        self.score_tree.column("alliance", width=50, anchor=tk.CENTER)
        self.score_tree.column("auto", width=50, anchor=tk.CENTER)
        self.score_tree.column("teleop", width=60, anchor=tk.CENTER)
        self.score_tree.column("total", width=50, anchor=tk.CENTER)
        self.score_tree.column("shots", width=50, anchor=tk.CENTER)
        self.score_tree.column("miss", width=50, anchor=tk.CENTER)
        self.score_tree.column("acc", width=60, anchor=tk.CENTER)

        score_scroll = ttk.Scrollbar(score_inner, orient=tk.VERTICAL,
                                      command=self.score_tree.yview)
        self.score_tree.configure(yscrollcommand=score_scroll.set)
        self.score_tree.grid(row=0, column=0, sticky="nsew")
        score_scroll.grid(row=0, column=1, sticky="ns")

        # 右半：進球事件
        event_tree_frame = ctk.CTkFrame(tab_results, fg_color="transparent")
        event_tree_frame.grid(row=0, column=1, sticky="nsew", padx=(4, 0))
        event_tree_frame.rowconfigure(1, weight=1)
        event_tree_frame.columnconfigure(0, weight=1)

        ctk.CTkLabel(event_tree_frame, text="進球事件",
                      text_color=COLORS["accent"],
                      font=ctk.CTkFont(size=14, weight="bold")
                      ).grid(row=0, column=0, sticky="w", padx=4, pady=(4, 2))

        event_inner = ctk.CTkFrame(event_tree_frame, fg_color="transparent")
        event_inner.grid(row=1, column=0, sticky="nsew")
        event_inner.rowconfigure(0, weight=1)
        event_inner.columnconfigure(0, weight=1)

        ecols = ("idx", "type", "time", "period", "alliance", "shooter")
        self.event_tree = ttk.Treeview(
            event_inner, columns=ecols, show="headings",
            style="Dark.Treeview")
        self.event_tree.heading("idx", text="#")
        self.event_tree.heading("type", text="類型")
        self.event_tree.heading("time", text="時間")
        self.event_tree.heading("period", text="期間")
        self.event_tree.heading("alliance", text="聯盟")
        self.event_tree.heading("shooter", text="射手")
        self.event_tree.column("idx", width=40, anchor=tk.CENTER)
        self.event_tree.column("type", width=60, anchor=tk.CENTER)
        self.event_tree.column("time", width=90, anchor=tk.CENTER)
        self.event_tree.column("period", width=60, anchor=tk.CENTER)
        self.event_tree.column("alliance", width=50, anchor=tk.CENTER)
        self.event_tree.column("shooter", width=80, anchor=tk.CENTER)

        event_scroll = ttk.Scrollbar(event_inner, orient=tk.VERTICAL,
                                      command=self.event_tree.yview)
        self.event_tree.configure(yscrollcommand=event_scroll.set)
        self.event_tree.grid(row=0, column=0, sticky="nsew")
        event_scroll.grid(row=0, column=1, sticky="ns")

        self.event_tree.bind("<ButtonRelease-1>", self._on_event_click)

    def _bind_keys(self):
        self.bind("<Right>", lambda e: self._step(1))
        self.bind("<Left>", lambda e: self._step(-1))
        self.bind("<d>", lambda e: self._step(1))
        self.bind("<a>", lambda e: self._step(-1))
        self.bind("<D>", lambda e: self._step(1))
        self.bind("<A>", lambda e: self._step(-1))
        self.bind("<Shift-Right>", lambda e: self._step(5))
        self.bind("<Shift-Left>", lambda e: self._step(-5))
        self.bind("<space>", lambda e: self._toggle_play())
        self.bind("<Control-o>", lambda e: self._on_open_video())
        self.bind("<Escape>", lambda e: self._cancel_interaction())
        self.bind("<F3>", lambda e: self._toggle_debug_view())
        self.bind("<Home>", lambda e: self._reset_zoom())

    # ══════════════════════════════════════════════════════
    # 座標轉換
    # ══════════════════════════════════════════════════════

    def _canvas_to_video(self, cx, cy):
        vx = (cx - self._display_offset_x) / self._display_scale + self._crop_x
        vy = (cy - self._display_offset_y) / self._display_scale + self._crop_y
        return (vx, vy)

    def _video_to_canvas(self, vx, vy):
        cx = (vx - self._crop_x) * self._display_scale + self._display_offset_x
        cy = (vy - self._crop_y) * self._display_scale + self._display_offset_y
        return (cx, cy)

    def _video_to_resized(self, point, scale):
        return (int((point[0] - self._crop_x) * scale),
                int((point[1] - self._crop_y) * scale))

    def _draw_dashed_rect(self, img, p1, p2, color, thickness=1,
                          dash_len=8, gap_len=5):
        """繪製虛線矩形。"""
        x1, y1 = p1
        x2, y2 = p2
        edges = [
            ((x1, y1), (x2, y1)),
            ((x2, y1), (x2, y2)),
            ((x2, y2), (x1, y2)),
            ((x1, y2), (x1, y1)),
        ]
        for (ex1, ey1), (ex2, ey2) in edges:
            length = math.hypot(ex2 - ex1, ey2 - ey1)
            if length == 0:
                continue
            dx = (ex2 - ex1) / length
            dy = (ey2 - ey1) / length
            pos = 0.0
            while pos < length:
                start = (int(ex1 + dx * pos), int(ey1 + dy * pos))
                end_pos = min(pos + dash_len, length)
                end = (int(ex1 + dx * end_pos), int(ey1 + dy * end_pos))
                cv2.line(img, start, end, color, thickness, cv2.LINE_AA)
                pos += dash_len + gap_len

    # ══════════════════════════════════════════════════════
    # 縮放/平移
    # ══════════════════════════════════════════════════════

    def _apply_zoom_crop(self, frame, cw, ch):
        """依據縮放/平移裁切 frame 到可見區域。設定 _crop_x/_crop_y。"""
        self._crop_x = 0
        self._crop_y = 0
        if self._zoom_level <= 1.0:
            return frame

        fh, fw = frame.shape[:2]
        base_scale = min(cw / fw, ch / fh)
        eff_scale = base_scale * self._zoom_level

        # 可見區域（影片像素）
        view_w = cw / eff_scale
        view_h = ch / eff_scale

        # 視點中心
        cx = fw / 2 + self._pan_x
        cy = fh / 2 + self._pan_y

        x1 = max(0, int(cx - view_w / 2))
        y1 = max(0, int(cy - view_h / 2))
        x2 = min(fw, int(cx + view_w / 2 + 0.5))
        y2 = min(fh, int(cy + view_h / 2 + 0.5))

        if x2 - x1 < 10 or y2 - y1 < 10:
            return frame

        self._crop_x = x1
        self._crop_y = y1
        return frame[y1:y2, x1:x2]

    def _on_zoom(self, event):
        """Ctrl+滾輪 縮放（向游標位置縮放）。"""
        if not self.cap:
            return

        # 縮放前：游標下的影片座標
        old_vx, old_vy = self._canvas_to_video(event.x, event.y)

        if event.delta > 0:
            new_zoom = min(self._zoom_level * 1.25, 8.0)
        else:
            new_zoom = max(self._zoom_level / 1.25, 1.0)

        if new_zoom <= 1.0:
            self._zoom_level = 1.0
            self._pan_x = 0.0
            self._pan_y = 0.0
        else:
            # 計算新 pan 使游標下的影片座標不變
            fw = self.video_width
            fh = self.video_height
            if self._roi:
                _, _, fw, fh = self._roi
            cw = self.canvas.winfo_width()
            ch = self.canvas.winfo_height()
            if cw < 10 or ch < 10:
                cw, ch = 900, 600
            base_scale = min(cw / fw, ch / fh)
            new_eff = base_scale * new_zoom

            self._zoom_level = new_zoom
            self._pan_x = old_vx - fw / 2 - (event.x - cw / 2) / new_eff
            self._pan_y = old_vy - fh / 2 - (event.y - ch / 2) / new_eff
            self._clamp_pan()

        self._show_frame(self.current_frame)

    def _on_pan_drag(self, event):
        """Shift+右鍵拖曳平移。"""
        if not self._pan_drag_start:
            return
        dx = event.x - self._pan_drag_start[0]
        dy = event.y - self._pan_drag_start[1]

        fw = self.video_width
        fh = self.video_height
        if self._roi:
            _, _, fw, fh = self._roi
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 10 or ch < 10:
            cw, ch = 900, 600
        base_scale = min(cw / fw, ch / fh)
        eff_scale = base_scale * self._zoom_level

        self._pan_x = self._pan_drag_origin[0] - dx / eff_scale
        self._pan_y = self._pan_drag_origin[1] - dy / eff_scale
        self._clamp_pan()
        self._show_frame(self.current_frame)

    def _on_pan_end(self, event):
        """右鍵放開結束平移。"""
        if self._pan_drag_start:
            self._pan_drag_start = None
            self._pan_drag_origin = None
            self.canvas.config(cursor="")

    def _clamp_pan(self):
        """限制平移範圍，避免超出影片邊界。"""
        if self._zoom_level <= 1.0:
            self._pan_x = 0.0
            self._pan_y = 0.0
            return
        fw = self.video_width
        fh = self.video_height
        if self._roi:
            _, _, fw, fh = self._roi
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 10 or ch < 10:
            cw, ch = 900, 600
        base_scale = min(cw / fw, ch / fh)
        eff_scale = base_scale * self._zoom_level

        view_w = cw / eff_scale
        view_h = ch / eff_scale
        max_px = max(0, (fw - view_w) / 2)
        max_py = max(0, (fh - view_h) / 2)
        self._pan_x = max(-max_px, min(max_px, self._pan_x))
        self._pan_y = max(-max_py, min(max_py, self._pan_y))

    def _reset_zoom(self):
        """重設縮放（Home 鍵）。"""
        if self._zoom_level != 1.0:
            self._zoom_level = 1.0
            self._pan_x = 0.0
            self._pan_y = 0.0
            if self.cap:
                self._show_frame(self.current_frame)

    # ══════════════════════════════════════════════════════
    # 影片操作
    # ══════════════════════════════════════════════════════

    def _on_open_video(self):
        path = filedialog.askopenfilename(
            title="選擇影片檔案",
            filetypes=[("影片檔案", "*.mp4 *.avi *.mov *.mkv"),
                       ("所有檔案", "*.*")]
        )
        if path:
            self._open_video(path)

    def _open_video(self, path):
        if self.cap:
            self.cap.release()
            self.is_playing = False

        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            messagebox.showerror("錯誤", f"無法開啟影片:\n{path}")
            self.cap = None
            return

        self.video_path = Path(path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.current_frame = 0

        # 備份原始尺寸（供裁切功能使用）
        self._original_video_width = self.video_width
        self._original_video_height = self.video_height
        self._roi = None
        self._zoom_level = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0

        # 快取第一幀
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = self.cap.read()
        if ret:
            self._first_frame = frame.copy()
            self._original_first_frame = frame.copy()

        self.slider.configure(to=max(1, self.total_frames - 1),
                              number_of_steps=max(1, self.total_frames - 1))
        self.fps_label.configure(text=f"FPS: {self.fps:.2f}")
        self.title(f"FRC Scoring Analyzer — {self.video_path.name}")

        # 更新引擎 FPS
        self.scoring_engine.fps = self.fps

        self._clear_all_marks()
        self._clear_analysis()
        self._set_status("影片已載入 — 請取色 Bumper 和標記得分區域", COLORS["info"])
        self._show_frame(0)

    def _show_frame(self, frame_idx):
        """跳轉到指定幀並渲染（用於 slider、手動跳轉）。"""
        if not self.cap:
            return
        frame_idx = max(0, min(frame_idx, self.total_frames - 1))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if not ret:
            return
        self.current_frame = frame_idx
        self._render_frame(frame, frame_idx)

    def _render_frame(self, frame, frame_idx):
        """將已解碼的影格渲染到畫布（不含 seek）。"""
        # ROI 裁切
        if self._roi:
            rx, ry, rw, rh = self._roi
            frame = frame[ry:ry+rh, rx:rx+rw]

        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 10 or ch < 10:
            cw, ch = 900, 600

        # Debug 4 面板視圖
        if self._debug_view and self._analysis_done:
            debug_canvas = self._render_debug_4panel(frame, frame_idx, cw, ch)
            if debug_canvas is not None:
                time_sec = frame_idx / self.fps
                period = "Auto" if time_sec < self.auto_duration else "Teleop"
                cv2.putText(debug_canvas,
                            f"DEBUG | F{frame_idx} | {time_sec:.1f}s | {period}",
                            (cw // 2 - 120, 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (245, 158, 11), 1, cv2.LINE_AA)
                rgb = cv2.cvtColor(debug_canvas, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
                self.photo_image = ImageTk.PhotoImage(pil_img)
                self.canvas.delete("all")
                self.canvas.create_image(0, 0, anchor=tk.NW,
                                         image=self.photo_image)
                self.frame_label.configure(
                    text=f"幀: {frame_idx}/{self.total_frames - 1}  |  "
                         f"{time_sec:.3f}s  [DEBUG]")
                self.slider.set(frame_idx)
                return

        # 縮放裁切
        frame = self._apply_zoom_crop(frame, cw, ch)

        fh, fw = frame.shape[:2]
        scale = min(cw / fw, ch / fh)
        new_w, new_h = int(fw * scale), int(fh * scale)
        self._display_scale = scale
        self._display_offset_x = (cw - new_w) // 2
        self._display_offset_y = (ch - new_h) // 2

        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LANCZOS4
        resized = cv2.resize(frame, (new_w, new_h), interpolation=interp)

        # 繪製得分區域（多邊形，依聯盟顏色）
        for zone in self._scoring_zones:
            pts = [self._video_to_resized(p, scale) for p in zone.points]
            pts_np = np.array(pts, dtype=np.int32)
            if zone.alliance in ALLIANCE_COLORS:
                zone_color = ALLIANCE_COLORS[zone.alliance]["bgr"]
            else:
                zone_color = (200, 100, 255)
            cv2.polylines(resized, [pts_np], isClosed=True,
                          color=zone_color, thickness=2,
                          lineType=cv2.LINE_AA)

        # 繪製 HP 線段
        for hp in self._hp_lines:
            p1 = self._video_to_resized(hp["p1"], scale)
            p2 = self._video_to_resized(hp["p2"], scale)
            if hp["alliance"] in ALLIANCE_COLORS:
                hp_color = ALLIANCE_COLORS[hp["alliance"]]["bgr"]
            else:
                hp_color = (200, 200, 0)
            cv2.line(resized, p1, p2, hp_color, 3, cv2.LINE_AA)
            # 繪製端點
            cv2.circle(resized, p1, 5, hp_color, -1, cv2.LINE_AA)
            cv2.circle(resized, p2, 5, hp_color, -1, cv2.LINE_AA)

        # 繪製遮擋區域（半透明灰色）
        for oz in self._occlusion_zones:
            pts = [self._video_to_resized(p, scale) for p in oz.polygon]
            pts_np = np.array(pts, dtype=np.int32)
            overlay = resized.copy()
            cv2.fillPoly(overlay, [pts_np], (80, 80, 80))
            cv2.addWeighted(overlay, 0.3, resized, 0.7, 0, resized)
            cv2.polylines(resized, [pts_np], isClosed=True,
                          color=(128, 128, 128), thickness=1,
                          lineType=cv2.LINE_AA)

        # 繪製 HP 標記進行中的第一個點
        if self.interaction_mode == "mark_hp_line" and self._hp_line_first_point:
            pt = self._video_to_resized(self._hp_line_first_point, scale)
            if self._hp_line_alliance in ALLIANCE_COLORS:
                c = ALLIANCE_COLORS[self._hp_line_alliance]["bgr"]
            else:
                c = (200, 200, 0)
            cv2.circle(resized, pt, 6, c, -1, cv2.LINE_AA)

        # 繪製機器人初始標記（在標記幀或未分析時）
        if not self._analysis_done:
            for i, (label, alliance, x, y, w, h, mark_f) in enumerate(self._robot_markers):
                if frame_idx != mark_f:
                    continue
                color = self._get_robot_color(label)
                p1 = self._video_to_resized((x, y), scale)
                p2 = self._video_to_resized((x + w, y + h), scale)
                cv2.rectangle(resized, p1, p2, color["bgr"], 2, cv2.LINE_AA)

        # 繪製取色中的 bumper 點擊點（橙色圓點）
        if self.interaction_mode == "bumper_pick" and self._bumper_pick_points:
            for pt in self._bumper_pick_points:
                rpt = self._video_to_resized(pt, scale)
                cv2.circle(resized, rpt, 6, (0, 165, 255), -1, cv2.LINE_AA)

        # 繪製拖曳中的矩形（裁切區域）
        if self._drag_start and self._drag_current and \
                self.interaction_mode in ("crop_region",):
            sx, sy = self._drag_start
            ex, ey = self._drag_current
            p1 = self._video_to_resized((min(sx, ex), min(sy, ey)), scale)
            p2 = self._video_to_resized((max(sx, ex), max(sy, ey)), scale)
            color = (0, 165, 255) if self.interaction_mode == "crop_region" \
                else (0, 255, 0)
            cv2.rectangle(resized, p1, p2, color, 2, cv2.LINE_AA)

        # 繪製多邊形標記中的頂點和連線（依聯盟顏色）
        if self.interaction_mode == "mark_zone_polygon" and self._polygon_points:
            if self._current_polygon_alliance in ALLIANCE_COLORS:
                poly_color = ALLIANCE_COLORS[self._current_polygon_alliance]["bgr"]
            else:
                poly_color = (200, 100, 255)
            pts = [self._video_to_resized(p, scale)
                   for p in self._polygon_points]
            for pt in pts:
                cv2.circle(resized, pt, 5, poly_color, -1, cv2.LINE_AA)
            if len(pts) >= 2:
                pts_np = np.array(pts, dtype=np.int32)
                cv2.polylines(resized, [pts_np], isClosed=False,
                              color=poly_color, thickness=2,
                              lineType=cv2.LINE_AA)

        # 繪製遮擋區域標記中的頂點和連線（灰色）
        if self.interaction_mode == "mark_occlusion_zone" and self._polygon_points:
            poly_color = (128, 128, 128)
            pts = [self._video_to_resized(p, scale)
                   for p in self._polygon_points]
            for pt in pts:
                cv2.circle(resized, pt, 5, poly_color, -1, cv2.LINE_AA)
            if len(pts) >= 2:
                pts_np = np.array(pts, dtype=np.int32)
                cv2.polylines(resized, [pts_np], isClosed=False,
                              color=poly_color, thickness=2,
                              lineType=cv2.LINE_AA)

        # 分析完成後的 overlay
        if self._analysis_done:
            self._draw_analysis_overlay(resized, frame_idx, scale)

        # BGR → RGB → PIL overlay
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        draw = ImageDraw.Draw(pil_img)

        # 幀號/時間 overlay
        time_sec = frame_idx / self.fps
        overlay_text = f"Frame {frame_idx}  |  {time_sec:.3f}s"
        period = "Auto" if time_sec < self.auto_duration else "Teleop"
        overlay_text += f"  |  {period}"
        draw.text((10, 8), overlay_text, fill=(245, 158, 11),
                  font=self._overlay_font)

        # 機器人標籤
        if not self._analysis_done:
            for i, (label, alliance, x, y, w, h, mark_f) in enumerate(self._robot_markers):
                if frame_idx != mark_f:
                    continue
                color = self._get_robot_color(label)
                pt = self._video_to_resized((x, y - 4), scale)
                draw.text((pt[0], max(0, pt[1] - 16)), label,
                          fill=color["rgb"], font=self._label_font)

        # 得分區域標籤（依聯盟顏色）
        for zone in self._scoring_zones:
            top_pt = min(zone.points, key=lambda p: p[1])
            pt = self._video_to_resized((top_pt[0], top_pt[1] - 4), scale)
            if zone.alliance in ALLIANCE_COLORS:
                label_color = ALLIANCE_COLORS[zone.alliance]["rgb"]
            else:
                label_color = (200, 100, 255)
            draw.text((pt[0], max(0, pt[1] - 16)), zone.name,
                      fill=label_color, font=self._small_font)

        # HP 線段標籤
        for hp in self._hp_lines:
            mid_x = (hp["p1"][0] + hp["p2"][0]) // 2
            mid_y = (hp["p1"][1] + hp["p2"][1]) // 2
            pt = self._video_to_resized((mid_x, mid_y - 4), scale)
            if hp["alliance"] in ALLIANCE_COLORS:
                hp_label_color = ALLIANCE_COLORS[hp["alliance"]]["rgb"]
            else:
                hp_label_color = (200, 200, 0)
            draw.text((pt[0], max(0, pt[1] - 16)), hp["name"],
                      fill=hp_label_color, font=self._small_font)

        # 分析後的文字標籤
        if self._analysis_done:
            self._draw_analysis_labels(draw, frame_idx, scale)

        self.photo_image = ImageTk.PhotoImage(pil_img)
        self.canvas.delete("all")
        self.canvas.create_image(self._display_offset_x,
                                 self._display_offset_y,
                                 anchor=tk.NW, image=self.photo_image)

        self.frame_label.configure(
            text=f"幀: {frame_idx}/{self.total_frames - 1}  |  "
                 f"{time_sec:.3f}s")
        self.slider.set(frame_idx)

    def _render_frame_playback(self, frame, frame_idx):
        """播放專用快速渲染：INTER_LINEAR + 精簡 overlay。"""
        if self._roi:
            rx, ry, rw, rh = self._roi
            frame = frame[ry:ry+rh, rx:rx+rw]

        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 10 or ch < 10:
            cw, ch = 900, 600

        # Debug 4 面板（播放中也支援）
        if self._debug_view and self._analysis_done:
            debug_canvas = self._render_debug_4panel(frame, frame_idx, cw, ch)
            if debug_canvas is not None:
                time_sec = frame_idx / self.fps
                cv2.putText(debug_canvas,
                            f"DEBUG | F{frame_idx} | {time_sec:.1f}s",
                            (cw // 2 - 80, 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (245, 158, 11), 1, cv2.LINE_AA)
                rgb = cv2.cvtColor(debug_canvas, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
                self.photo_image = ImageTk.PhotoImage(pil_img)
                self.canvas.delete("all")
                self.canvas.create_image(0, 0, anchor=tk.NW,
                                         image=self.photo_image)
                self.frame_label.configure(
                    text=f"幀: {frame_idx}/{self.total_frames - 1}  |  "
                         f"{time_sec:.3f}s  [DEBUG]")
                self.slider.set(frame_idx)
                return

        # 縮放裁切
        frame = self._apply_zoom_crop(frame, cw, ch)

        fh, fw = frame.shape[:2]
        scale = min(cw / fw, ch / fh)
        new_w, new_h = int(fw * scale), int(fh * scale)
        self._display_scale = scale
        self._display_offset_x = (cw - new_w) // 2
        self._display_offset_y = (ch - new_h) // 2

        # 播放時一律用 LINEAR（比 LANCZOS4 快 12 倍）
        resized = cv2.resize(frame, (new_w, new_h),
                             interpolation=cv2.INTER_LINEAR)

        # 得分區域多邊形
        for zone in self._scoring_zones:
            pts = [self._video_to_resized(p, scale) for p in zone.points]
            pts_np = np.array(pts, dtype=np.int32)
            if zone.alliance in ALLIANCE_COLORS:
                zone_color = ALLIANCE_COLORS[zone.alliance]["bgr"]
            else:
                zone_color = (200, 100, 255)
            cv2.polylines(resized, [pts_np], isClosed=True,
                          color=zone_color, thickness=2,
                          lineType=cv2.LINE_AA)

        # HP 線段
        for hp in self._hp_lines:
            p1 = self._video_to_resized(hp["p1"], scale)
            p2 = self._video_to_resized(hp["p2"], scale)
            if hp["alliance"] in ALLIANCE_COLORS:
                hp_color = ALLIANCE_COLORS[hp["alliance"]]["bgr"]
            else:
                hp_color = (200, 200, 0)
            cv2.line(resized, p1, p2, hp_color, 3, cv2.LINE_AA)

        # 遮擋區域（半透明灰色）
        for oz in self._occlusion_zones:
            pts = [self._video_to_resized(p, scale) for p in oz.polygon]
            pts_np = np.array(pts, dtype=np.int32)
            overlay = resized.copy()
            cv2.fillPoly(overlay, [pts_np], (80, 80, 80))
            cv2.addWeighted(overlay, 0.3, resized, 0.7, 0, resized)
            cv2.polylines(resized, [pts_np], isClosed=True,
                          color=(128, 128, 128), thickness=1,
                          lineType=cv2.LINE_AA)

        # 分析 overlay（球偵測、軌跡、機器人框）
        if self._analysis_done:
            self._draw_analysis_overlay(resized, frame_idx, scale)

        # BGR → RGB → Tk（跳過 PIL Draw，直接用 cv2.putText）
        time_sec = frame_idx / self.fps
        period = "Auto" if time_sec < self.auto_duration else "Teleop"
        overlay_text = f"Frame {frame_idx} | {time_sec:.1f}s | {period}"
        cv2.putText(resized, overlay_text, (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (245, 158, 11), 1,
                    cv2.LINE_AA)

        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        # 分析後的文字標籤（機器人名稱等，需要中文字型）
        if self._analysis_done:
            draw = ImageDraw.Draw(pil_img)
            self._draw_analysis_labels(draw, frame_idx, scale)

        self.photo_image = ImageTk.PhotoImage(pil_img)
        self.canvas.delete("all")
        self.canvas.create_image(self._display_offset_x,
                                 self._display_offset_y,
                                 anchor=tk.NW, image=self.photo_image)

        self.frame_label.configure(
            text=f"幀: {frame_idx}/{self.total_frames - 1}  |  "
                 f"{time_sec:.3f}s")
        self.slider.set(frame_idx)

    def _draw_analysis_overlay(self, resized, frame_idx, scale):
        """繪製分析結果的幾何圖形 overlay。"""
        try:
            self._draw_analysis_overlay_impl(resized, frame_idx, scale)
        except Exception as e:
            # 防止 overlay 渲染錯誤導致整個畫面消失
            print(f"[WARN] overlay 渲染錯誤 (frame {frame_idx}): {e}")

    def _draw_analysis_overlay_impl(self, resized, frame_idx, scale):
        """overlay 渲染實作。"""
        # 球偵測圈
        if frame_idx in self._frame_detections:
            for det in self._frame_detections[frame_idx]:
                cx, cy, area = det[0], det[1], det[2]
                pt = self._video_to_resized((cx, cy), scale)
                radius = max(4, int(math.sqrt(area / math.pi) * scale))
                cv2.circle(resized, pt, radius, (36, 191, 251), 2,
                           cv2.LINE_AA)

        # 球軌跡（使用幀索引加速查詢）
        traj_points: dict[int, list[tuple]] = {}  # tid -> [(f, cx, cy), ...]
        for f in range(frame_idx - 20, frame_idx + 21):
            entries = self._trajectory_by_frame.get(f)
            if entries:
                for tid, cx, cy in entries:
                    if tid in traj_points:
                        traj_points[tid].append((f, cx, cy))
                    else:
                        traj_points[tid] = [(f, cx, cy)]
        color = (0, 200, 200)
        for tid, points in traj_points.items():
            if len(points) < 2:
                continue
            points.sort(key=lambda p: p[0])
            for i in range(len(points) - 1):
                p1 = self._video_to_resized((points[i][1], points[i][2]),
                                            scale)
                p2 = self._video_to_resized((points[i+1][1], points[i+1][2]),
                                            scale)
                cv2.line(resized, p1, p2, color, 1, cv2.LINE_AA)

        # 機器人追蹤框（偵測=實線, 插值=虛線）
        robot_bboxes = self._robot_bboxes_cache.get(frame_idx, {})
        if frame_idx in self._robot_positions_cache:
            for label, pos in self._robot_positions_cache[frame_idx].items():
                cx, cy = pos[0], pos[1]
                color = self._get_robot_color(label)
                det_frames = self._robot_detected_frames.get(label)
                is_detected = (det_frames is not None
                               and frame_idx in det_frames)
                # 偵測幀：實線+粗邊框；插值幀：薄邊框+灰色
                thickness = 2 if is_detected else 1
                border_color = (255, 255, 255) if is_detected \
                    else (128, 128, 128)
                if label in robot_bboxes:
                    x1, y1, x2, y2 = robot_bboxes[label]
                    p1 = self._video_to_resized((x1, y1), scale)
                    p2 = self._video_to_resized((x2, y2), scale)
                    bw = abs(p2[0] - p1[0])
                    bh = abs(p2[1] - p1[1])
                    if bw < 10 or bh < 10:
                        cpt = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
                        cv2.circle(resized, cpt, 12 if is_detected else 8,
                                   color["bgr"], -1, cv2.LINE_AA)
                        cv2.circle(resized, cpt, 14 if is_detected else 10,
                                   border_color, thickness, cv2.LINE_AA)
                    else:
                        cv2.rectangle(resized, p1, p2, color["bgr"],
                                      thickness, cv2.LINE_AA)
                        if is_detected:
                            # 偵測幀額外加白色外框
                            cv2.rectangle(
                                resized,
                                (p1[0] - 1, p1[1] - 1),
                                (p2[0] + 1, p2[1] + 1),
                                border_color, 1, cv2.LINE_AA)
                else:
                    pt = self._video_to_resized((cx, cy), scale)
                    cv2.circle(resized, pt, 12 if is_detected else 8,
                               color["bgr"], -1, cv2.LINE_AA)
                    cv2.circle(resized, pt, 14 if is_detected else 10,
                               border_color, thickness, cv2.LINE_AA)

        # LOST 機器人：虛線框在最後已知位置
        if hasattr(self, '_analysis_robot_mgr') and self._analysis_robot_mgr:
            tracker = self._analysis_robot_mgr
            if hasattr(tracker, '_impl') and hasattr(tracker._impl, '_track_state'):
                impl = tracker._impl
                for label, state in impl._track_state.items():
                    if state != "lost":
                        continue
                    cur_robots = self._robot_positions_cache.get(frame_idx) or {}
                    if label in cur_robots:
                        continue
                    lk = impl._last_known.get(label)
                    if not lk:
                        continue
                    color = self._get_robot_color(label)
                    bbox_list = impl._bboxes.get(label, [])
                    if bbox_list:
                        x1, y1, x2, y2 = (bbox_list[-1][1], bbox_list[-1][2],
                                           bbox_list[-1][3], bbox_list[-1][4])
                        p1 = self._video_to_resized((x1, y1), scale)
                        p2 = self._video_to_resized((x2, y2), scale)
                        self._draw_dashed_rect(
                            resized, p1, p2, color["bgr"], thickness=1)
                    else:
                        last_cx, last_cy = lk[1], lk[2]
                        pt = self._video_to_resized(
                            (last_cx, last_cy), scale)
                        cv2.circle(resized, pt, 8,
                                   (128, 128, 128), 1, cv2.LINE_AA)

        # 進球事件標記
        for event in self.scoring_engine.events:
            if abs(event.frame_idx - frame_idx) <= 5:
                alpha = 1.0 - abs(event.frame_idx - frame_idx) / 6.0
                for zone in self._scoring_zones:
                    if zone.name == event.zone_name:
                        zc = zone.center
                        pt = self._video_to_resized(zc, scale)
                        r = int(15 * scale * alpha)
                        cv2.circle(resized, pt, r, (0, 255, 255), 1,
                                   cv2.LINE_AA)

        # 即時 per-robot 計數 overlay（底部橫排）
        goals = self._cumulative_goals.get(frame_idx)
        if goals:
            h_img = resized.shape[0]
            x_offset = 10
            y_pos = h_img - 12
            for label, count in sorted(goals.items()):
                color = self._get_robot_color(label)
                text = f"{label}: {count}"
                cv2.putText(resized, text, (x_offset, y_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            color["bgr"], 1, cv2.LINE_AA)
                x_offset += len(text) * 10 + 20

    def _draw_analysis_labels(self, draw, frame_idx, scale):
        """繪製分析後的文字標籤。"""
        try:
            self._draw_analysis_labels_impl(draw, frame_idx, scale)
        except Exception as e:
            print(f"[WARN] label 渲染錯誤 (frame {frame_idx}): {e}")

    def _draw_analysis_labels_impl(self, draw, frame_idx, scale):
        """label 渲染實作。"""
        # 機器人名稱標籤
        robot_bboxes = self._robot_bboxes_cache.get(frame_idx, {})
        if frame_idx in self._robot_positions_cache:
            for label, pos in self._robot_positions_cache[frame_idx].items():
                cx, cy = pos[0], pos[1]
                color = self._get_robot_color(label)
                if label in robot_bboxes:
                    x1, y1, x2, y2 = robot_bboxes[label]
                    pt = self._video_to_resized((x1, y1 - 4), scale)
                else:
                    pt = self._video_to_resized((cx, cy - 15), scale)
                draw.text((pt[0], max(0, pt[1] - 16)), label,
                          fill=color["rgb"], font=self._label_font)

        # LOST 機器人標籤（灰色）
        if hasattr(self, '_analysis_robot_mgr') and self._analysis_robot_mgr:
            tracker = self._analysis_robot_mgr
            if hasattr(tracker, '_impl') and hasattr(tracker._impl, '_track_state'):
                impl = tracker._impl
                for label, state in impl._track_state.items():
                    if state != "lost":
                        continue
                    cur_robots = self._robot_positions_cache.get(frame_idx) or {}
                    if label in cur_robots:
                        continue
                    lk = impl._last_known.get(label)
                    if not lk:
                        continue
                    last_cx, last_cy = lk[1], lk[2]
                    pt = self._video_to_resized((last_cx, last_cy - 15), scale)
                    draw.text((pt[0], max(0, pt[1] - 16)),
                              f"{label} [LOST]",
                              fill=(128, 128, 128), font=self._label_font)

        # 進球事件文字
        for event in self.scoring_engine.events:
            if event.frame_idx == frame_idx:
                for zone in self._scoring_zones:
                    if zone.name == event.zone_name:
                        zc = zone.center
                        pt = self._video_to_resized(zc, scale)
                        text = f"SCORED! {event.shooter_label}"
                        draw.text((pt[0] - 30, pt[1] - 30), text,
                                  fill=(255, 255, 0), font=self._small_font)

    def _toggle_debug_view(self):
        """切換 4 面板 Debug 視圖（F3）。"""
        if not self._analysis_done:
            self._set_status("Debug 視圖需要先完成分析", COLORS["error"])
            return
        self._debug_view = not self._debug_view
        mode = "開啟" if self._debug_view else "關閉"
        self._set_status(f"Debug 4 面板視圖已{mode}（F3 切換）",
                         COLORS["info"])
        self._show_frame(self.current_frame)

    def _render_debug_4panel(self, frame, frame_idx, target_w, target_h):
        """渲染 4 面板 Debug 視圖。

        左上: 場地遮罩（灰階 + 偵測標記）
        右上: 球軌跡 + ownership 著色
        左下: 機器人偵測（原始影像 + bbox）
        右下: 完整 overlay
        """
        pw, ph = target_w // 2, target_h // 2
        if pw < 50 or ph < 50:
            return None

        fh, fw = frame.shape[:2]
        scale = min(pw / fw, ph / fh)
        sw, sh = int(fw * scale), int(fh * scale)

        # 共用縮放
        small = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_LINEAR)

        # ── 左上：灰階 + 機器人位置圓點 ──
        panel_tl = cv2.cvtColor(
            cv2.cvtColor(small, cv2.COLOR_BGR2GRAY),
            cv2.COLOR_GRAY2BGR)
        if frame_idx in self._robot_positions_cache:
            for label, pos in self._robot_positions_cache[frame_idx].items():
                pt = (int(pos[0] * scale), int(pos[1] * scale))
                color = self._get_robot_color(label)
                cv2.circle(panel_tl, pt, 6, color["bgr"], -1, cv2.LINE_AA)
        cv2.putText(panel_tl, "Grayscale", (4, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        # ── 右上：球軌跡 + ownership 著色 ──
        panel_tr = np.zeros((sh, sw, 3), dtype=np.uint8)
        # 預先計算每個 tid 的 owner 顏色
        tid_colors: dict[int, tuple] = {}
        from collections import Counter
        for tid in self._all_trajectories:
            ownership = self.scoring_engine._ball_ownership.get(tid, {})
            if ownership:
                most_common = Counter(ownership.values()).most_common(1)
                if most_common:
                    c = self._get_robot_color(most_common[0][0])
                    tid_colors[tid] = c["bgr"]
        # 收集可見軌跡點
        debug_traj: dict[int, list[tuple]] = {}
        for f in range(frame_idx - 60, frame_idx + 61):
            entries = self._trajectory_by_frame.get(f)
            if entries:
                for tid, cx, cy in entries:
                    if tid in debug_traj:
                        debug_traj[tid].append((f, cx, cy))
                    else:
                        debug_traj[tid] = [(f, cx, cy)]
        for tid, pts_list in debug_traj.items():
            if len(pts_list) < 2:
                continue
            pts_list.sort(key=lambda p: p[0])
            owner_color = tid_colors.get(tid, (0, 200, 200))
            for i in range(len(pts_list) - 1):
                p1 = (int(pts_list[i][1] * scale),
                      int(pts_list[i][2] * scale))
                p2 = (int(pts_list[i+1][1] * scale),
                      int(pts_list[i+1][2] * scale))
                cv2.line(panel_tr, p1, p2, owner_color, 1, cv2.LINE_AA)
        # 得分區域
        for zone in self._scoring_zones:
            pts = [self._video_to_resized(p, scale) for p in zone.points]
            cv2.polylines(panel_tr, [np.array(pts, np.int32)],
                          True, (100, 100, 100), 1, cv2.LINE_AA)
        cv2.putText(panel_tr, "Ball Ownership", (4, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        # ── 左下：機器人偵測 overlay ──
        panel_bl = small.copy()
        robot_bboxes = self._robot_bboxes_cache.get(frame_idx, {})
        if frame_idx in self._robot_positions_cache:
            for label, pos in self._robot_positions_cache[frame_idx].items():
                color = self._get_robot_color(label)
                det_frames = self._robot_detected_frames.get(label)
                is_det = det_frames and frame_idx in det_frames
                if label in robot_bboxes:
                    x1, y1, x2, y2 = robot_bboxes[label]
                    p1 = (int(x1 * scale), int(y1 * scale))
                    p2 = (int(x2 * scale), int(y2 * scale))
                    c = (0, 255, 0) if is_det else (0, 200, 200)
                    cv2.rectangle(panel_bl, p1, p2, c, 1, cv2.LINE_AA)
                pt = (int(pos[0] * scale), int(pos[1] * scale))
                # 編號標記
                num = label.split("-")[-1] if "-" in label else label
                cv2.putText(panel_bl, num, (pt[0] - 4, pt[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                            (0, 255, 0) if is_det else (0, 200, 200),
                            1, cv2.LINE_AA)
        cv2.putText(panel_bl, "Robots (Green=Det, Cyan=Interp)", (4, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

        # ── 右下：完整 overlay ──
        panel_br = small.copy()
        # 球偵測
        if frame_idx in self._frame_detections:
            for det in self._frame_detections[frame_idx]:
                pt = (int(det[0] * scale), int(det[1] * scale))
                cv2.circle(panel_br, pt, 4, (36, 191, 251), 1, cv2.LINE_AA)
        # 機器人
        if frame_idx in self._robot_positions_cache:
            for label, pos in self._robot_positions_cache[frame_idx].items():
                color = self._get_robot_color(label)
                pt = (int(pos[0] * scale), int(pos[1] * scale))
                cv2.circle(panel_br, pt, 5, color["bgr"], -1, cv2.LINE_AA)
        # 得分區域
        for zone in self._scoring_zones:
            pts = [self._video_to_resized(p, scale) for p in zone.points]
            zc = ALLIANCE_COLORS.get(zone.alliance, {}).get("bgr", (200, 100, 255))
            cv2.polylines(panel_br, [np.array(pts, np.int32)],
                          True, zc, 1, cv2.LINE_AA)
        cv2.putText(panel_br, "Full Overlay", (4, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        # 組合 4 面板
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        canvas[0:sh, 0:sw] = panel_tl
        canvas[0:sh, pw:pw+sw] = panel_tr
        canvas[ph:ph+sh, 0:sw] = panel_bl
        canvas[ph:ph+sh, pw:pw+sw] = panel_br

        # 分隔線
        cv2.line(canvas, (pw, 0), (pw, target_h), (60, 60, 60), 1)
        cv2.line(canvas, (0, ph), (target_w, ph), (60, 60, 60), 1)

        return canvas

    def _get_robot_color_idx(self, label):
        """取得機器人對應的顏色索引。"""
        labels = [m[0] for m in self._robot_markers]
        if label in labels:
            return labels.index(label) % len(ROBOT_COLORS)
        return 0

    def _get_robot_alliance(self, label):
        """取得機器人的聯盟。"""
        for lbl, alliance, *_ in self._robot_markers:
            if lbl == label:
                return alliance
        # HP labels
        for hp in self._hp_lines:
            if hp["name"] == label:
                return hp["alliance"]
        return ""

    def _get_robot_color(self, label):
        """根據聯盟回傳顏色字典 {bgr, rgb, hex}。"""
        alliance = self._get_robot_alliance(label)
        if alliance in ALLIANCE_COLORS:
            return ALLIANCE_COLORS[alliance]
        idx = self._get_robot_color_idx(label)
        return {
            "bgr": (ROBOT_COLORS[idx][2], ROBOT_COLORS[idx][1],
                     ROBOT_COLORS[idx][0]),
            "rgb": ROBOT_COLORS[idx],
            "hex": ROBOT_COLORS_HEX[idx],
        }

    # ══════════════════════════════════════════════════════
    # 播放控制
    # ══════════════════════════════════════════════════════

    def _step(self, delta):
        if not self.cap:
            return
        self.is_playing = False
        self.play_btn.configure(text="▶ 播放")
        self._show_frame(self.current_frame + delta)

    def _on_slider(self, value):
        if not self.cap:
            return
        frame_idx = int(float(value))
        if frame_idx != self.current_frame:
            self._show_frame(frame_idx)

    def _toggle_play(self):
        if not self.cap:
            return
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.play_btn.configure(text="⏸ 暫停")
            self._play_wall_start = time.monotonic()
            self._play_start_frame = self.current_frame
            self._play_loop()
        else:
            self.play_btn.configure(text="▶ 播放")

    def _toggle_speed(self):
        """切換播放倍速 1x → 2x → 3x → 4x → 5x → 1x。"""
        speeds = [1.0, 2.0, 3.0, 4.0, 5.0]
        try:
            idx = speeds.index(self._playback_speed)
            self._playback_speed = speeds[(idx + 1) % len(speeds)]
        except ValueError:
            self._playback_speed = 1.0
        # 重置時間基準，避免切速後瞬間跳幀
        self._play_wall_start = time.monotonic()
        self._play_start_frame = self.current_frame
        label = f"{int(self._playback_speed)}x"
        self.speed_btn.configure(text=label)

    def _play_loop(self):
        if not self.is_playing or not self.cap:
            return
        if self.current_frame >= self.total_frames - 1:
            self.is_playing = False
            self.play_btn.configure(text="▶ 播放")
            return

        t_start = time.monotonic()

        # 固定步進：每 tick 前進 (fps/30 * speed) 幀
        # 渲染慢時自動降速，不會卡頓
        step = max(1, round(self.fps / 30.0 * self._playback_speed))
        target = min(self.current_frame + step, self.total_frames - 1)
        gap = target - self.current_frame

        if gap == 1:
            ret, frame = self.cap.read()
        elif gap <= 5:
            for _ in range(gap - 1):
                self.cap.grab()
            ret, frame = self.cap.read()
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            ret, frame = self.cap.read()

        if ret and frame is not None:
            self.current_frame = target
            self._render_frame_playback(frame, target)

        # 扣除渲染耗時，保持穩定節奏
        render_ms = (time.monotonic() - t_start) * 1000
        delay = max(10, int(33 - render_ms))
        self.after(delay, self._play_loop)

    # ══════════════════════════════════════════════════════
    # 標記互動（機器人 + 得分區域）
    # ══════════════════════════════════════════════════════

    def _start_bumper_pick(self):
        """進入 bumper 取色模式。"""
        if not self.cap:
            self._set_status("請先開啟影片", COLORS["error"])
            return
        if len(self._bumper_templates) >= MAX_ROBOTS:
            self._set_status(f"最多取色 {MAX_ROBOTS} 台機器人", COLORS["error"])
            return
        self.interaction_mode = "bumper_pick"
        self._bumper_pick_points = []
        self._set_status(
            "在影片上點擊機器人的 bumper（多次），右鍵完成",
            COLORS["accent"])
        self.canvas.config(cursor="crosshair")

    def _start_mark_zone_polygon(self, alliance=""):
        if not self.cap:
            self._set_status("請先開啟影片", COLORS["error"])
            return
        self.interaction_mode = "mark_zone_polygon"
        self._polygon_points = []
        self._current_polygon_alliance = alliance
        alliance_name = "紅方" if alliance == "red" else "藍方" if alliance == "blue" else ""
        self._set_status(
            f"標記{alliance_name} Hub — 左鍵點擊放置頂點，"
            "右鍵或雙擊完成多邊形（至少 3 點），ESC 取消",
            COLORS["accent"])
        self.canvas.config(cursor="crosshair")

    def _start_mark_hp_line(self, alliance):
        """進入 HP 線段標記模式。"""
        if not self.cap:
            self._set_status("請先開啟影片", COLORS["error"])
            return
        self.interaction_mode = "mark_hp_line"
        self._hp_line_first_point = None
        self._hp_line_alliance = alliance
        alliance_name = "紅方" if alliance == "red" else "藍方"
        self._set_status(
            f"標記{alliance_name} HP — 左鍵點擊線段兩端點，ESC 取消",
            COLORS["accent"])
        self.canvas.config(cursor="crosshair")

    def _start_mark_occlusion_zone(self):
        """開始標記遮擋區域多邊形。"""
        if not self.cap:
            self._set_status("請先開啟影片", COLORS["error"])
            return
        self.interaction_mode = "mark_occlusion_zone"
        self._polygon_points = []
        self.canvas.config(cursor="crosshair")
        self._set_status(
            "標記遮擋區域 — 左鍵點擊放置頂點，"
            "右鍵或雙擊完成多邊形（至少 3 點），ESC 取消",
            COLORS["info"])

    def _cancel_interaction(self):
        if self.interaction_mode:
            if self.interaction_mode == "color_pick":
                self._color_pick_callback = None
                self._color_pick_finish = None
                self._color_pick_mode = None
            if self.interaction_mode == "bumper_pick":
                self._bumper_pick_points = []
            self.interaction_mode = None
            self._drag_start = None
            self._drag_current = None
            self._polygon_points = []
            self._current_polygon_alliance = ""
            self._hp_line_first_point = None
            self._hp_line_alliance = ""
            self.canvas.config(cursor="")
            self._set_status("已取消", COLORS["text_secondary"])
            self._show_frame(self.current_frame)

    def _start_crop(self):
        """進入裁切模式。"""
        if not self.cap:
            self._set_status("請先開啟影片", COLORS["error"])
            return
        # 若已有 ROI，先還原到全畫面以便重新裁切
        if self._roi:
            self._roi = None
            self.video_width = self._original_video_width
            self.video_height = self._original_video_height
            self._first_frame = self._original_first_frame.copy()
            self._clear_all_marks()
        self.interaction_mode = "crop_region"
        self._set_status("在影片上拖曳選取裁切區域", COLORS["accent"])
        self.canvas.config(cursor="crosshair")
        self._show_frame(self.current_frame)

    def _finish_crop(self, x, y, w, h):
        """裁切完成，更新影片空間。"""
        # 如果已有 ROI，座標是相對於原始全畫面的（因為 _start_crop 已重置）
        self._roi = (x, y, w, h)
        self.video_width = w
        self.video_height = h

        # 裁切第一幀快取
        if self._original_first_frame is not None:
            self._first_frame = self._original_first_frame[y:y+h, x:x+w].copy()

        # 清除所有標記（座標系已改變）
        self._robot_markers.clear()
        self._bumper_templates.clear()
        self._scoring_zones.clear()
        self._hp_lines.clear()
        self._update_robot_list()
        self._update_zone_list()
        self._clear_analysis()

        self.interaction_mode = None
        self.canvas.config(cursor="")
        self._set_status(f"已裁切畫面至 {w}x{h}，請重新取色 Bumper 和標記得分區域",
                         COLORS["success"])
        self._show_frame(self.current_frame)

    def _reset_crop(self):
        """重置裁切，回到全畫面。"""
        if not self.cap:
            return
        if not self._roi:
            self._set_status("目前沒有裁切設定", COLORS["text_secondary"])
            return

        self._roi = None
        self.video_width = self._original_video_width
        self.video_height = self._original_video_height

        if self._original_first_frame is not None:
            self._first_frame = self._original_first_frame.copy()

        # 清除所有標記
        self._robot_markers.clear()
        self._bumper_templates.clear()
        self._scoring_zones.clear()
        self._hp_lines.clear()
        self._update_robot_list()
        self._update_zone_list()
        self._clear_analysis()

        self.interaction_mode = None
        self.canvas.config(cursor="")
        self._set_status("已重置裁切，回到全畫面", COLORS["success"])
        self._show_frame(self.current_frame)

    def _on_canvas_press(self, event):
        if not self.interaction_mode or not self.cap:
            return
        vx, vy = self._canvas_to_video(event.x, event.y)

        # 裁切模式下使用原始尺寸（在全畫面上操作）
        if self.interaction_mode == "crop_region":
            bw = self._original_video_width if self._roi else self.video_width
            bh = self._original_video_height if self._roi else self.video_height
        else:
            bw, bh = self.video_width, self.video_height
        if not (0 <= vx < bw and 0 <= vy < bh):
            return

        if self.interaction_mode == "color_pick":
            vx_int, vy_int = int(vx), int(vy)
            if self._color_pick_mode == "single":
                if self._color_pick_callback:
                    self._color_pick_callback(vx_int, vy_int)
                self.interaction_mode = None
                self.canvas.config(cursor="")
                self._set_status("取色完成", COLORS["success"])
            else:
                if self._color_pick_callback:
                    self._color_pick_callback(vx_int, vy_int)
                self._show_frame(self.current_frame)
            return

        if self.interaction_mode == "bumper_pick":
            self._bumper_pick_points.append((int(vx), int(vy)))
            n = len(self._bumper_pick_points)
            self._set_status(
                f"已取色 {n} 個點 — 繼續點擊或右鍵完成（至少 2 點）",
                COLORS["accent"])
            self._show_frame(self.current_frame)
            return

        if self.interaction_mode == "mark_zone_polygon":
            self._polygon_points.append((int(vx), int(vy)))
            n = len(self._polygon_points)
            self._set_status(
                f"已放置 {n} 個頂點 — 右鍵或雙擊完成（至少 3 點），ESC 取消",
                COLORS["accent"])
            self._show_frame(self.current_frame)
            return

        if self.interaction_mode == "mark_hp_line":
            pt = (int(vx), int(vy))
            if self._hp_line_first_point is None:
                self._hp_line_first_point = pt
                self._set_status("已放置第 1 點 — 請點擊第 2 點完成 HP 線段",
                                 COLORS["accent"])
                self._show_frame(self.current_frame)
            else:
                self._finish_mark_hp_line(pt)
            return

        if self.interaction_mode == "mark_occlusion_zone":
            self._polygon_points.append((int(vx), int(vy)))
            n = len(self._polygon_points)
            self._set_status(
                f"已放置 {n} 個頂點 — 右鍵或雙擊完成（至少 3 點），ESC 取消",
                COLORS["info"])
            self._show_frame(self.current_frame)
            return

        # 機器人框選拖曳
        self._drag_start = (vx, vy)
        self._drag_current = (vx, vy)

    def _on_canvas_drag(self, event):
        if not self._drag_start:
            return
        vx, vy = self._canvas_to_video(event.x, event.y)
        vx = max(0, min(vx, self.video_width))
        vy = max(0, min(vy, self.video_height))
        self._drag_current = (vx, vy)
        self._show_frame(self.current_frame)

    def _on_canvas_release(self, event):
        if self.interaction_mode in ("bumper_pick", "mark_zone_polygon", "mark_hp_line", "mark_occlusion_zone"):
            return  # 這些模式由點擊/右鍵/雙擊完成

        if not self._drag_start or not self._drag_current:
            return

        sx, sy = self._drag_start
        ex, ey = self._drag_current
        x = int(min(sx, ex))
        y = int(min(sy, ey))
        w = int(abs(ex - sx))
        h = int(abs(ey - sy))

        # 最小尺寸檢查
        if w < 10 or h < 10:
            self._drag_start = None
            self._drag_current = None
            self._set_status("框選太小，請重試", COLORS["error"])
            return

        if self.interaction_mode == "crop_region":
            self._finish_crop(x, y, w, h)

        self._drag_start = None
        self._drag_current = None

    def _on_canvas_double_click(self, event):
        """雙擊完成多邊形標記。"""
        if self.interaction_mode == "mark_zone_polygon":
            self._finish_mark_polygon()
        if self.interaction_mode == "mark_occlusion_zone":
            self._finish_mark_occlusion_zone()

    def _on_canvas_right_click(self, event):
        """右鍵完成多邊形標記 / bumper 取色完成 / HSV 取色完成。"""
        # Shift+右鍵：啟動拖曳平移
        if event.state & 0x1 and self._zoom_level > 1.0:
            self._pan_drag_start = (event.x, event.y)
            self._pan_drag_origin = (self._pan_x, self._pan_y)
            self.canvas.config(cursor="fleur")
            return

        if self.interaction_mode == "bumper_pick":
            self._finish_bumper_pick()
            return

        if self.interaction_mode == "color_pick" and self._color_pick_mode == "multi":
            if self._color_pick_finish:
                self._color_pick_finish()
            self.interaction_mode = None
            self.canvas.config(cursor="")
            self._set_status("校正完成", COLORS["success"])
            return

        if self.interaction_mode == "mark_zone_polygon":
            self._finish_mark_polygon()
            return

        if self.interaction_mode == "mark_occlusion_zone":
            self._finish_mark_occlusion_zone()
            return

    def _finish_mark_polygon(self):
        """完成 Hub 多邊形標記。"""
        if len(self._polygon_points) < 3:
            self._set_status("至少需要 3 個頂點才能完成多邊形", COLORS["error"])
            return

        alliance = self._current_polygon_alliance
        if alliance == "red":
            zone_name = "紅方 Hub"
        elif alliance == "blue":
            zone_name = "藍方 Hub"
        else:
            zone_name = "Hub"

        # 移除同名的舊區域
        self._scoring_zones = [
            z for z in self._scoring_zones if z.name != zone_name
        ]

        zone = ScoringZone(zone_name, list(self._polygon_points), alliance)
        self._scoring_zones.append(zone)
        self._update_zone_list()

        self.interaction_mode = None
        self._polygon_points = []
        self._current_polygon_alliance = ""
        self.canvas.config(cursor="")
        self._set_status(f"已設定 {zone_name} 區域", COLORS["success"])
        self._show_frame(self.current_frame)

    def _finish_mark_occlusion_zone(self):
        """完成遮擋區域多邊形標記。"""
        if len(self._polygon_points) < 3:
            self._set_status("至少需要 3 個頂點才能完成多邊形", COLORS["error"])
            return

        zone_name = f"遮擋區域 {len(self._occlusion_zones) + 1}"
        zone = OcclusionZone(zone_name, list(self._polygon_points))
        self._occlusion_zones.append(zone)
        self._update_zone_list()

        self.interaction_mode = None
        self._polygon_points = []
        self.canvas.config(cursor="")
        self._set_status(f"已設定 {zone_name}", COLORS["success"])
        self._show_frame(self.current_frame)

    def _finish_mark_hp_line(self, second_point):
        """完成 HP 線段標記。"""
        alliance = self._hp_line_alliance
        if alliance == "red":
            hp_name = "紅方 HP"
        elif alliance == "blue":
            hp_name = "藍方 HP"
        else:
            hp_name = "HP"

        # 移除同名的舊 HP 線段
        self._hp_lines = [h for h in self._hp_lines if h["name"] != hp_name]

        self._hp_lines.append({
            "name": hp_name,
            "alliance": alliance,
            "p1": self._hp_line_first_point,
            "p2": second_point,
        })

        self.interaction_mode = None
        self._hp_line_first_point = None
        self._hp_line_alliance = ""
        self.canvas.config(cursor="")
        self._set_status(f"已設定 {hp_name} 線段", COLORS["success"])
        self._update_zone_list()
        self._show_frame(self.current_frame)

    def _finish_bumper_pick(self):
        """完成 bumper 取色，建立模板。"""
        if len(self._bumper_pick_points) < 2:
            self._set_status("至少需要 2 個取色點", COLORS["error"])
            return

        from calibration import build_bumper_template

        # 取得當前幀
        frame = self._get_current_frame()
        if frame is None:
            self._set_status("無法取得當前幀", COLORS["error"])
            return

        histogram, auto_alliance = build_bumper_template(
            frame, self._bumper_pick_points)

        # 彈出輸入框取得機器人編號
        label = simpledialog.askstring(
            "機器人編號",
            "請輸入機器人編號（例如 6998）:",
            parent=self)
        if not label:
            self._set_status("已取消取色", COLORS["text_secondary"])
            self._bumper_pick_points = []
            self.interaction_mode = None
            self.canvas.config(cursor="")
            return

        label = label.strip()

        # 檢查是否重複
        for existing_label, *_ in self._bumper_templates:
            if existing_label == label:
                self._set_status(f"機器人 {label} 已存在", COLORS["error"])
                self._bumper_pick_points = []
                return

        # 確認聯盟（自動判斷，用戶可覆蓋）
        if auto_alliance:
            alliance = auto_alliance
        else:
            alliance = self._ask_alliance()
            if alliance is None:
                self._set_status("已取消取色", COLORS["text_secondary"])
                self._bumper_pick_points = []
                self.interaction_mode = None
                self.canvas.config(cursor="")
                return

        self._bumper_templates.append((label, alliance, histogram))
        # 同步到 _robot_markers（供顯示用）
        self._robot_markers.append(
            (label, alliance, 0, 0, 0, 0, self.current_frame))
        self._update_robot_list()

        self._bumper_pick_points = []
        self.interaction_mode = None
        self.canvas.config(cursor="")
        alliance_name = "紅方" if alliance == "red" else "藍方"
        self._set_status(
            f"已取色{alliance_name} {label}（{len(self._bumper_templates)} 台已註冊）",
            COLORS["success"])
        self._show_frame(self.current_frame)

    def _ask_alliance(self):
        """彈窗詢問聯盟（紅方/藍方）。回傳 'red'/'blue' 或 None（取消）。"""
        result = {"alliance": None}

        dialog = ctk.CTkToplevel(self)
        dialog.title("選擇聯盟")
        dialog.geometry("260x130")
        dialog.resizable(False, False)
        dialog.transient(self)
        dialog.grab_set()

        ctk.CTkLabel(dialog, text="請選擇此機器人的聯盟:",
                      font=ctk.CTkFont(family="Microsoft JhengHei UI",
                                        size=14)).pack(pady=(20, 16))

        btn_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        btn_frame.pack()

        def choose(alliance):
            result["alliance"] = alliance
            dialog.destroy()

        ctk.CTkButton(btn_frame, text="紅方", width=100, height=36,
                       fg_color="#dc2626", hover_color="#b91c1c",
                       text_color="white",
                       font=ctk.CTkFont(family="Microsoft JhengHei UI",
                                         size=13),
                       command=lambda: choose("red")
                       ).pack(side=tk.LEFT, padx=8)
        ctk.CTkButton(btn_frame, text="藍方", width=100, height=36,
                       fg_color="#2563eb", hover_color="#1d4ed8",
                       text_color="white",
                       font=ctk.CTkFont(family="Microsoft JhengHei UI",
                                         size=13),
                       command=lambda: choose("blue")
                       ).pack(side=tk.LEFT, padx=8)

        dialog.wait_window()
        return result["alliance"]

    def _clear_all_marks(self):
        """清除所有標記。"""
        self._robot_markers.clear()
        self._bumper_templates.clear()
        self._bumper_pick_points = []
        self._scoring_zones.clear()
        self._hp_lines.clear()
        self._occlusion_zones.clear()
        self._hp_line_first_point = None
        self._hp_line_alliance = ""
        self._drag_start = None
        self._drag_current = None
        self._polygon_points = []
        self._current_polygon_alliance = ""
        self.interaction_mode = None
        self.canvas.config(cursor="")
        self._update_robot_list()
        self._update_zone_list()
        self._clear_analysis()
        if self.cap:
            self._show_frame(self.current_frame)

    def _update_robot_list(self):
        """更新右側面板的機器人列表顯示。"""
        if not self._robot_markers:
            self.robot_list_label.configure(text="（無）")
        else:
            lines = []
            for i, (label, alliance, x, y, w, h, mark_f) in enumerate(self._robot_markers):
                prefix = "[紅]" if alliance == "red" else "[藍]" if alliance == "blue" else ""
                lines.append(f"  {prefix} {label} ({w}x{h}) @F{mark_f}")
            self.robot_list_label.configure(text="\n".join(lines))

    def _update_zone_list(self):
        """更新右側面板的得分區域列表顯示。"""
        lines = []
        for zone in self._scoring_zones:
            lines.append(f"  {zone.name} ({len(zone.points)} 頂點)")
        for hp in self._hp_lines:
            lines.append(f"  {hp['name']} (線段)")
        for oz in self._occlusion_zones:
            lines.append(f"  {oz.name} ({len(oz.polygon)} 頂點) [遮擋]")
        if not lines:
            self.zone_list_label.configure(text="（無）")
        else:
            self.zone_list_label.configure(text="\n".join(lines))

    # ══════════════════════════════════════════════════════
    # 分析引擎
    # ══════════════════════════════════════════════════════

    def _on_detection_mode_change(self, value):
        """偵測模式切換回調。"""
        self._detection_mode = value
        self._runtime_config.detection_mode = value
        self._ai_model = None  # 重置，分析時再載入

    def _on_analyze(self, max_seconds=None):
        if not self.cap:
            self._set_status("請先開啟影片", COLORS["error"])
            return
        if not self._scoring_zones:
            self._set_status("請先標記至少一個得分區域", COLORS["error"])
            return
        if self._analyzing:
            self._set_status("分析進行中，請點擊「中斷」按鈕停止", COLORS["accent"])
            return

        # 計算最大幀數
        if max_seconds is not None:
            self._analysis_max_frames = int(max_seconds * self.fps)
        else:
            self._analysis_max_frames = None

        # 從 RuntimeConfig 讀取 auto 時間設定
        self.auto_duration = self._runtime_config.auto_duration_sec

        self._clear_analysis()
        self._analyzing = True
        self._analysis_cancel = False
        self.analyze_quick_btn.configure(state="disabled")
        self.analyze_full_btn.configure(state="disabled")
        self.analyze_cancel_btn.pack(side=tk.LEFT, padx=(2, 4), pady=4)
        self.progress_bar.pack(fill=tk.X, padx=12, pady=(0, 6))
        self.progress_bar.set(0)
        label = f"分析中（前 {max_seconds} 秒）..." if max_seconds else "分析中（完整影片）..."
        self._set_status(label, COLORS["info"])

        thread = threading.Thread(target=self._run_analysis, daemon=True)
        thread.start()

    def _run_analysis(self):
        """背景線程：逐幀球偵測+追蹤+機器人追蹤+進球判定。"""
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            self.after(0, lambda: self._analysis_error("無法開啟影片"))
            return

        # AI 球偵測模型載入（若選擇 AI 模式）
        cfg = self._runtime_config
        use_ai = cfg.detection_mode == "AI"
        ai_model = None
        if use_ai:
            try:
                self.after(0, lambda: self._set_status(
                    "載入球偵測 AI 模型中（本地 ONNX）...", COLORS["info"]))
                ai_model = load_ai_model()
            except Exception as e:
                err_msg = str(e)
                print(f"[WARN] AI 球偵測模型載入失敗，回退到 HSV: {err_msg}")
                use_ai = False
                self.after(0, lambda em=err_msg: self._set_status(
                    f"球偵測 AI 載入失敗: {em[:60]}，改用 HSV", COLORS["error"]))

        # 機器人偵測器載入（HSV / YOLO / GEMINI）
        robot_detector = None
        if ROBOT_DETECTION_MODE == "GEMINI":
            try:
                from robot_detection import RobotDetectorGemini
                robot_detector = RobotDetectorGemini()
                self.after(0, lambda: self._set_status(
                    "機器人偵測: Gemini API（零樣本）", COLORS["info"]))
            except Exception as e:
                print(f"[WARN] Gemini 偵測器初始化失敗，回退 HSV Bumper: {e}")
                robot_detector = BumperDetectorHSV()
                self.after(0, lambda em=str(e)[:40]: self._set_status(
                    f"機器人偵測: HSV Bumper（Gemini 失敗: {em}）",
                    COLORS["error"]))
        elif ROBOT_DETECTION_MODE == "YOLO":
            try:
                robot_detector = load_robot_model()
                self.after(0, lambda: self._set_status(
                    "機器人偵測: YOLO 模型（MOT 模式）", COLORS["info"]))
            except (FileNotFoundError, Exception) as e:
                print(f"[WARN] YOLO 模型載入失敗，回退 HSV Bumper: {e}")
                robot_detector = BumperDetectorHSV()
                self.after(0, lambda: self._set_status(
                    "機器人偵測: HSV Bumper（YOLO 不可用）", COLORS["info"]))
        else:
            robot_detector = BumperDetectorHSV()
            self.after(0, lambda: self._set_status(
                "機器人偵測: HSV Bumper（不需模型）", COLORS["info"]))

        # 傳遞 bumper 取色模板給偵測器
        if self._bumper_templates and hasattr(robot_detector, 'set_templates'):
            robot_detector.set_templates(
                self._bumper_templates,
                similarity=cfg.bumper_template_similarity)

        # 初始化球追蹤器
        ball_tracker = CentroidTracker(max_distance=cfg.max_match_dist,
                                       max_missed=cfg.max_missed)

        # 初始化機器人追蹤器（MOT 或 SOT）
        robot_mgr = RobotTrackerManager(
            detector=robot_detector, fps=self.fps)
        tracking_mode = "MOT" if robot_mgr.use_mot else "SOT"
        print(f"[INFO] 追蹤模式: {tracking_mode}")

        # MOT 永遠使用距離匹配（ByteTrack IoU 在 FRC 場景失效）
        if robot_mgr.use_mot:
            robot_mgr.enable_auto_mode()
            if self._occlusion_zones:
                robot_mgr.set_occlusion_zones(self._occlusion_zones)
                print(f"[INFO] MOT 已設定 {len(self._occlusion_zones)} 個遮擋區域")
            print("[INFO] MOT 距離匹配模式（自動偵測所有機器人）")

        # 初始化進球引擎
        engine = ScoringEngine(
            fps=self.fps,
            auto_sec=cfg.auto_duration_sec,
            teleop_start_sec=cfg.teleop_start_sec,
            proximity_frames=cfg.score_proximity_frames,
            max_shooter_dist=cfg.score_max_shooter_dist,
            zone_dwell_frames=cfg.score_zone_dwell_frames,
            cooldown_frames=cfg.score_cooldown_frames,
            shot_min_velocity=cfg.shot_min_velocity,
            shot_min_upward_velocity=cfg.shot_min_upward_velocity,
            shot_robot_proximity=cfg.shot_robot_proximity,
            ball_ownership_dist=cfg.ball_ownership_dist,
        )
        engine.set_zones(self._scoring_zones)
        engine.hp_lines = list(self._hp_lines)

        total = self.total_frames
        if self._analysis_max_frames is not None:
            total = min(total, self._analysis_max_frames)
        frame_detections = {}
        robot_positions_cache = {}
        robot_bboxes_cache = {}

        # ROI 裁切
        roi = self._roi

        # 建立「哪一幀要初始化哪些機器人」的對照表
        markers_by_frame = {}
        for label, alliance, x, y, w, h, mark_f in self._robot_markers:
            markers_by_frame.setdefault(mark_f, []).append(
                (label, alliance, x, y, w, h))

        # MOT 模式：預先註冊所有待匹配的標記（不需要影像）
        if robot_mgr.use_mot:
            for mark_f, markers in markers_by_frame.items():
                for label, alliance, x, y, w, h in markers:
                    robot_mgr.add_robot(label, (x, y, w, h), None,
                                        mark_f, alliance)

        # 重設球偵測診斷計數器
        reset_diagnostics()

        # 偵測函式選擇
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

        ball_mode = "AI" if use_ai else "HSV"
        mode_label = f"{ball_mode}+{tracking_mode}"

        # 診斷計數器
        total_ball_dets = 0
        total_robot_dets = 0

        self._bg_model = None

        # ═══ Pipeline 分析：偵測提前執行，追蹤依序消費 ═══
        # Producer 線程：讀幀 + 球偵測 + 機器人偵測（提前 N 幀）
        # Consumer（本線程）：追蹤匹配 + 進球判定（必須依序）
        import queue as _queue
        det_queue = _queue.Queue(maxsize=8)

        def _detection_producer():
            """背景線程：讀幀 + 所有偵測，結果排入佇列。"""
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            for fidx in range(total):
                if self._analysis_cancel:
                    break
                ret, frame = cap.read()
                if not ret:
                    break
                if roi:
                    rx, ry, rw, rh = roi
                    frame = frame[ry:ry+rh, rx:rx+rw]
                ball_dets = detect_balls(frame)
                robot_raw = robot_mgr.detect_raw(frame, fidx)
                det_queue.put((fidx, frame, ball_dets, robot_raw))
            det_queue.put(None)  # sentinel

        producer = threading.Thread(
            target=_detection_producer, daemon=True)
        producer.start()

        # Consumer：依序追蹤 + 進球判定
        while True:
            if self._analysis_cancel:
                break
            item = det_queue.get()
            if item is None:
                break
            frame_idx, frame, dets, robot_raw = item

            # SOT 模式：在標記幀初始化追蹤器（需要影像）
            if not robot_mgr.use_mot and frame_idx in markers_by_frame:
                for label, alliance, x, y, w, h in \
                        markers_by_frame[frame_idx]:
                    robot_mgr.add_robot(
                        label, (x, y, w, h), frame, frame_idx,
                        alliance)

            # 追蹤更新（必須依序執行）
            robot_mgr.track_update(robot_raw, frame_idx, frame)

            frame_detections[frame_idx] = dets
            total_ball_dets += len(dets)
            ball_positions = ball_tracker.update(dets, frame_idx)

            robot_pos = robot_mgr.get_all_positions(frame_idx)
            total_robot_dets += len(robot_pos)
            robot_positions_cache[frame_idx] = \
                robot_mgr.get_all_display_positions(frame_idx)
            robot_bboxes_cache[frame_idx] = \
                robot_mgr.get_all_bboxes(frame_idx)

            # 進球判定
            engine.process_frame(frame_idx, ball_positions, robot_pos,
                                 ball_tracker.trajectories)

            # 診斷日誌（每 100 幀）
            if frame_idx > 0 and frame_idx % 100 == 0:
                print(f"[INFO] Frame 0-{frame_idx}: "
                      f"球偵測 {total_ball_dets} 個, "
                      f"機器人偵測 {total_robot_dets} 個")

            # 更新進度（每 20 幀）
            if frame_idx % 20 == 0:
                pct = (frame_idx + 1) / total * 100
                self.after(0, lambda p=pct, f=frame_idx, m=mode_label:
                           self._update_progress(p, f, m))

        producer.join()
        cap.release()

        if self._analysis_cancel:
            self._analysis_cancel = False
            self.after(0, lambda: self._analysis_cancelled())
            return

        print(f"[INFO] 分析完成: 共 {total} 幀, "
              f"球偵測總計 {total_ball_dets}, "
              f"機器人偵測總計 {total_robot_dets}")

        # 後處理：合併碎片 label + 過濾短期 label + 位置插值（MOT 模式）
        if robot_mgr.use_mot:
            # 合併前診斷
            pre_merge = list(robot_mgr.robot_info.keys())
            print(f"[INFO] MOT 後處理: 合併前 {len(pre_merge)} 個 label")

        robot_mgr.merge_fragmented_labels()
        robot_mgr.filter_short_labels()
        robot_mgr.filter_static_labels()
        robot_mgr.interpolate_positions()
        # 完整重建快取（清除後處理前的舊資料，避免已移除/合併 label 殘留）
        if robot_mgr.use_mot:
            robot_positions_cache.clear()
            robot_bboxes_cache.clear()
            for frame_idx in range(total):
                pos = robot_mgr.get_all_display_positions(frame_idx)
                if pos:
                    robot_positions_cache[frame_idx] = pos
                bbox = robot_mgr.get_all_bboxes(frame_idx)
                if bbox:
                    robot_bboxes_cache[frame_idx] = bbox

        # 軌跡縫合
        all_trajectories = stitch_trajectories(dict(ball_tracker.trajectories))

        # 球所有權追蹤（用完整位置資料計算每顆球的 owner）
        engine.compute_ball_ownership(all_trajectories, robot_positions_cache)

        # 出手偵測後處理
        engine.detect_shots(all_trajectories, robot_positions_cache)

        # 射手重新歸因（優先用 ownership，fallback 到距離搜尋）
        engine.reattribute_shooters(all_trajectories, robot_positions_cache)

        # 收集自動偵測的機器人資訊（補充未手動標記的 label）
        auto_robots = []
        if robot_mgr.use_mot:
            existing_labels = {m[0] for m in self._robot_markers}
            for label, info in robot_mgr.robot_info.items():
                if label not in existing_labels:
                    auto_robots.append((label, info.get("alliance", "")))
            if auto_robots:
                print(f"[INFO] MOT 自動偵測: 補充 "
                      f"{len(auto_robots)} 台未標記機器人")
                for lbl, alliance in auto_robots:
                    print(f"  - {lbl} ({alliance})")

        # 診斷：positions cache 覆蓋率
        frames_with_robots = sum(
            1 for f in range(total)
            if robot_positions_cache.get(f))
        print(f"[INFO] 機器人 overlay 覆蓋率: "
              f"{frames_with_robots}/{total} 幀 "
              f"({frames_with_robots/max(total,1)*100:.0f}%)")

        # 收集偵測信心度資訊
        detected_frames = {}
        if hasattr(robot_mgr, '_impl') and hasattr(robot_mgr._impl, '_detected_frames'):
            detected_frames = dict(robot_mgr._impl._detected_frames)

        # 回到主線程
        self.after(0, lambda: self._finish_analysis(
            all_trajectories, frame_detections,
            robot_positions_cache, robot_bboxes_cache, engine,
            auto_robots, detected_frames, robot_mgr
        ))

    def _update_progress(self, pct, frame_idx, mode=""):
        self.progress_bar.set(pct / 100.0)
        mode_str = f" [{mode}]" if mode else ""
        self._set_status(
            f"分析中{mode_str}... 幀 {frame_idx}/{self.total_frames - 1} ({pct:.0f}%)",
            COLORS["info"])

    def _cancel_analysis(self):
        """用戶點擊中斷按鈕。"""
        self._analysis_cancel = True
        self.analyze_cancel_btn.configure(state="disabled")
        self._set_status("正在中斷分析...", COLORS["error"])

    def _restore_analyze_buttons(self):
        """恢復分析按鈕到初始狀態。"""
        self.analyze_quick_btn.configure(state="normal")
        self.analyze_full_btn.configure(state="normal")
        self.analyze_cancel_btn.pack_forget()
        self.analyze_cancel_btn.configure(state="normal")

    def _analysis_cancelled(self):
        """分析被用戶中斷。"""
        self._analyzing = False
        self._restore_analyze_buttons()
        self.progress_bar.pack_forget()
        self._set_status("分析已中斷", COLORS["error"])
        print("[INFO] 分析已被用戶中斷")

    def _analysis_error(self, msg):
        self._analyzing = False
        self._restore_analyze_buttons()
        self.progress_bar.pack_forget()
        self._set_status(f"分析錯誤: {msg}", COLORS["error"])

    def _finish_analysis(self, trajectories, frame_dets,
                         robot_cache, robot_bbox_cache, engine,
                         auto_robots=None, detected_frames=None,
                         robot_mgr=None):
        """分析完成，更新 UI。"""
        self._analysis_robot_mgr = robot_mgr
        self._all_trajectories = trajectories
        self._build_trajectory_index(trajectories)
        self._frame_detections = frame_dets
        self._robot_positions_cache = robot_cache
        self._robot_bboxes_cache = robot_bbox_cache
        self._robot_detected_frames = detected_frames or {}
        self.scoring_engine = engine
        self._analysis_done = True
        self._analyzing = False

        self._restore_analyze_buttons()
        self.progress_bar.pack_forget()

        # 添加自動偵測的機器人到標記列表（用於顯示和統計）
        if auto_robots:
            for label, alliance in auto_robots:
                self._robot_markers.append(
                    (label, alliance, 0, 0, 0, 0, -1))
            self._update_robot_list()

        # 預計算 per-robot 累計進球數（用於即時 overlay）
        self._cumulative_goals = {}  # {frame_idx: {label: count}}
        running = {}  # {label: count}
        if engine.events:
            events_sorted = sorted(engine.events, key=lambda e: e.frame_idx)
            ev_idx = 0
            for f in range(self.total_frames):
                while (ev_idx < len(events_sorted)
                       and events_sorted[ev_idx].frame_idx <= f):
                    lbl = events_sorted[ev_idx].shooter_label or "未知"
                    running[lbl] = running.get(lbl, 0) + 1
                    ev_idx += 1
                if running:
                    self._cumulative_goals[f] = dict(running)

        # 更新得分統計表
        self._update_score_table()

        # 更新事件時間軸
        self._update_event_timeline()

        n_goals = len(engine.events)
        n_shots = len(engine.shot_events)
        n_robots = len(self._robot_markers)
        self._set_status(
            f"分析完成！{n_goals} 進球, {n_shots} 出手, {n_robots} 台機器人",
            COLORS["success"])

        self._show_frame(self.current_frame)

    def _build_trajectory_index(self, trajectories: dict):
        """從軌跡資料預建幀索引，加速播放渲染時的軌跡查詢。"""
        index: dict[int, list[tuple]] = {}
        for tid, traj in trajectories.items():
            for point in traj:
                f = point[0]
                entry = (tid, point[1], point[2])  # (tid, cx, cy)
                if f in index:
                    index[f].append(entry)
                else:
                    index[f] = [entry]
        self._trajectory_by_frame = index

    def _clear_analysis(self):
        """清除分析結果。"""
        self._analysis_done = False
        self._analysis_robot_mgr = None
        self._all_trajectories.clear()
        self._trajectory_by_frame.clear()
        self._frame_detections.clear()
        self._robot_positions_cache.clear()
        self._robot_bboxes_cache.clear()
        self._robot_detected_frames.clear()
        self._cumulative_goals = {}
        self.scoring_engine.reset()

        for item in self.score_tree.get_children():
            self.score_tree.delete(item)
        for item in self.event_tree.get_children():
            self.event_tree.delete(item)

    def _update_score_table(self):
        """更新得分統計表（依聯盟分組）。"""
        for item in self.score_tree.get_children():
            self.score_tree.delete(item)

        summary = self.scoring_engine.get_summary()

        # 確保所有已標記的機器人和 HP 都在表格中
        all_labels = set(m[0] for m in self._robot_markers)
        for hp in self._hp_lines:
            all_labels.add(hp["name"])
        all_labels.add("未知")

        # 依聯盟分組：紅方 → 藍方 → 未指定
        for alliance_display, alliance_key in [("紅", "red"), ("藍", "blue"), ("—", "")]:
            for label in sorted(all_labels):
                if self._get_robot_alliance(label) != alliance_key:
                    continue
                if label in summary:
                    s = summary[label]
                    acc_str = f"{s.accuracy:.0%}" if s.total_shots > 0 else "—"
                    self.score_tree.insert("", tk.END, values=(
                        label, alliance_display,
                        s.auto, s.teleop, s.total,
                        s.total_shots, s.total_misses, acc_str
                    ))
                else:
                    self.score_tree.insert("", tk.END, values=(
                        label, alliance_display,
                        0, 0, 0, 0, 0, "—"
                    ))

    def _update_event_timeline(self):
        """更新事件時間軸（進球 + 出手事件混合）。"""
        for item in self.event_tree.get_children():
            self.event_tree.delete(item)

        all_events = self.scoring_engine.get_all_events_timeline()
        for i, evt in enumerate(all_events):
            time_sec = evt["frame_idx"] / self.fps
            time_str = format_time(time_sec)
            alliance_display = ("紅" if evt["alliance"] == "red" else
                                "藍" if evt["alliance"] == "blue" else "—")
            type_display = "進球" if evt["type"] == "goal" else "未進"
            self.event_tree.insert("", tk.END, iid=str(i), values=(
                i + 1,
                type_display,
                f"{time_str} ({time_sec:.1f}s)",
                evt["period"],
                alliance_display,
                evt["shooter"],
            ))

        # 儲存事件列表供點擊跳轉使用
        self._all_events_list = all_events

    def _on_event_click(self, event):
        """點擊事件時跳到對應幀。"""
        sel = self.event_tree.selection()
        if not sel:
            return
        idx = int(sel[0])
        events_list = getattr(self, '_all_events_list', [])
        if 0 <= idx < len(events_list):
            self._show_frame(events_list[idx]["frame_idx"])

    # ══════════════════════════════════════════════════════
    # CSV 匯出
    # ══════════════════════════════════════════════════════

    def _export_csv(self):
        """匯出得分結果為 CSV。"""
        if not self.scoring_engine.events and not self.scoring_engine.shot_events:
            messagebox.showinfo("提示", "沒有分析結果可匯出")
            return

        default_name = ""
        if self.video_path:
            default_name = f"{self.video_path.stem}_scoring.csv"

        path = filedialog.asksaveasfilename(
            title="匯出 CSV",
            defaultextension=".csv",
            initialfile=default_name,
            filetypes=[("CSV 檔案", "*.csv")]
        )
        if not path:
            return

        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)

            # 得分摘要
            writer.writerow(["=== 得分摘要 ==="])
            writer.writerow(["機器人", "聯盟", "Auto進球", "Teleop進球",
                             "總進球", "總出手", "未進球", "命中率"])
            summary = self.scoring_engine.get_summary()
            for label in sorted(summary.keys()):
                s = summary[label]
                alliance = self._get_robot_alliance(label)
                alliance_display = ("紅方" if alliance == "red" else
                                    "藍方" if alliance == "blue" else "—")
                acc_str = f"{s.accuracy:.1%}" if s.total_shots > 0 else "—"
                writer.writerow([label, alliance_display,
                                 s.auto_goals, s.teleop_goals, s.total,
                                 s.total_shots, s.total_misses, acc_str])
            writer.writerow([])

            # 進球事件
            writer.writerow(["=== 進球事件 ==="])
            writer.writerow(["編號", "幀", "時間(秒)", "期間", "聯盟",
                             "射手", "射手距離", "區域"])
            timeline = self.scoring_engine.get_timeline()
            for i, event in enumerate(timeline):
                time_sec = event.frame_idx / self.fps
                alliance_display = ("紅方" if event.alliance == "red" else
                                    "藍方" if event.alliance == "blue" else "—")
                writer.writerow([
                    i + 1, event.frame_idx, f"{time_sec:.3f}",
                    event.period, alliance_display,
                    event.shooter_label, f"{event.shooter_dist:.1f}",
                    event.zone_name
                ])
            writer.writerow([])

            # 出手事件（含未進球）
            writer.writerow(["=== 出手事件 ==="])
            writer.writerow(["編號", "幀", "時間(秒)", "期間", "聯盟",
                             "射手", "結果", "進球幀", "區域"])
            shot_timeline = self.scoring_engine.get_shot_timeline()
            for i, shot in enumerate(shot_timeline):
                time_sec = shot.frame_idx / self.fps
                alliance_display = ("紅方" if shot.alliance == "red" else
                                    "藍方" if shot.alliance == "blue" else "—")
                result_display = "進球" if shot.result == "goal" else "未進"
                goal_f = str(shot.goal_frame) if shot.goal_frame >= 0 else "—"
                writer.writerow([
                    i + 1, shot.frame_idx, f"{time_sec:.3f}",
                    shot.period, alliance_display,
                    shot.shooter_label, result_display,
                    goal_f, shot.zone_name
                ])

        self._set_status(f"已匯出至 {Path(path).name}", COLORS["success"])

    # ══════════════════════════════════════════════════════
    # 共用
    # ══════════════════════════════════════════════════════

    def _set_status(self, text, color=None):
        self.status_label.configure(
            text=text,
            text_color=color or COLORS["text_secondary"])

    # ══════════════════════════════════════════════════════
    # 設定視窗
    # ══════════════════════════════════════════════════════

    def _get_current_frame(self) -> "np.ndarray | None":
        """取得當前影片幀（BGR，含 ROI 裁切）。"""
        if not self.cap:
            return None
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = self.cap.read()
        if not ret:
            return None
        if self._roi:
            rx, ry, rw, rh = self._roi
            frame = frame[ry:ry+rh, rx:rx+rw]
        return frame

    def _get_current_frame_for_preview(self):
        """提供當前播放幀給設定視窗預覽。"""
        return self._get_current_frame()

    def _get_analysis_data_for_preview(self):
        """提供當前幀的分析資料給設定面板預覽。"""
        if not self._analysis_done:
            return None
        f = self.current_frame
        return {
            "frame_idx": f,
            "ball_detections": self._frame_detections.get(f, []),
            "robot_positions": self._robot_positions_cache.get(f, {}),
            "trajectories": self._all_trajectories,
            "ball_ownership": self.scoring_engine._ball_ownership,
        }

    def _on_recompute_attribution(self):
        """用當前 RuntimeConfig 參數重新計算歸因。"""
        if not self._analysis_done:
            return
        cfg = self._runtime_config
        engine = self.scoring_engine
        # 更新引擎參數
        engine._ownership_dist = cfg.ball_ownership_dist
        engine._shot_min_velocity = cfg.shot_min_velocity
        engine._shot_min_upward_velocity = cfg.shot_min_upward_velocity
        engine._shot_robot_proximity = cfg.shot_robot_proximity
        engine._proximity_frames = cfg.score_proximity_frames
        engine._max_shooter_dist = cfg.score_max_shooter_dist
        # 重新計算
        engine.reset()
        # 從軌跡重建每幀的 ball_positions {track_id: (cx, cy, area)}
        ball_by_frame: dict[int, dict] = {}
        for tid, traj in self._all_trajectories.items():
            for point in traj:
                f = point[0]
                if f not in ball_by_frame:
                    ball_by_frame[f] = {}
                ball_by_frame[f][tid] = (point[1], point[2],
                                         point[3] if len(point) > 3 else 0)
        # 重新跑進球判定
        for f in sorted(ball_by_frame.keys()):
            robot_pos = self._robot_positions_cache.get(f, {})
            engine.process_frame(f, ball_by_frame[f], robot_pos,
                                 self._all_trajectories)
        engine.compute_ball_ownership(
            self._all_trajectories, self._robot_positions_cache)
        engine.detect_shots(
            self._all_trajectories, self._robot_positions_cache)
        engine.reattribute_shooters(
            self._all_trajectories, self._robot_positions_cache)
        # 更新 UI
        self._cumulative_goals = {}
        running = {}
        if engine.events:
            events_sorted = sorted(engine.events, key=lambda e: e.frame_idx)
            ev_idx = 0
            for f in range(self.total_frames):
                while (ev_idx < len(events_sorted)
                       and events_sorted[ev_idx].frame_idx <= f):
                    lbl = events_sorted[ev_idx].shooter_label or "未知"
                    running[lbl] = running.get(lbl, 0) + 1
                    ev_idx += 1
                if running:
                    self._cumulative_goals[f] = dict(running)
        self._update_score_table()
        self._update_event_timeline()
        n_goals = len(engine.events)
        n_shots = len(engine.shot_events)
        self._set_status(
            f"重新計算完成！{n_goals} 進球, {n_shots} 出手",
            COLORS["success"])
        self._show_frame(self.current_frame)

    def _on_settings_changed(self):
        cfg = self._runtime_config
        self._detection_mode = cfg.detection_mode
        self._ai_model = None  # 偵測模式可能已變更，重置
        self.auto_duration = cfg.auto_duration_sec

    # ══════════════════════════════════════════════════════
    # 取色模式（供 SettingsWindow 呼叫）
    # ══════════════════════════════════════════════════════

    def _start_color_pick(self, mode="single", callback=None, finish_callback=None):
        self._color_pick_mode = mode
        self._color_pick_callback = callback
        self._color_pick_finish = finish_callback
        self.interaction_mode = "color_pick"
        self.canvas.config(cursor="crosshair")
        self._set_status("在影片上點擊取色位置", COLORS["info"])

    def destroy(self):
        self.is_playing = False
        if self.cap:
            self.cap.release()
        super().destroy()

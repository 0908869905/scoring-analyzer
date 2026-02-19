"""
FRC Scoring Analyzer — GUI 主類別
"""

import csv
import math
import time
import threading
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
)
from detection import detect_yellow_balls, detect_fuel_ai, load_ai_model
from tracking import CentroidTracker, stitch_trajectories
from robot_detection import load_robot_model
from robot_tracker import RobotTrackerManager
from scoring import ScoringEngine, ScoringZone
from utils import load_font, format_time


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
        self._playback_speed = 1.0     # 播放倍速（1.0 / 0.5）
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
        self.interaction_mode = None  # None, "mark_robot", "mark_zone_polygon"
        self._drag_start = None      # 拖曳起始點（影片座標）
        self._drag_current = None    # 拖曳當前點
        self._polygon_points = []    # 多邊形標記中的頂點列表
        self._current_polygon_alliance = ""  # 目前正在標記的多邊形聯盟

        # 機器人追蹤
        self.robot_manager = RobotTrackerManager()
        self._robot_markers = []  # [(label, alliance, x, y, w, h, frame_idx)]

        # 得分區域
        self._scoring_zones = []  # [ScoringZone, ...]

        # 偵測模式
        self._detection_mode = DETECTION_MODE  # "AI" or "HSV"
        self._ai_model = None  # 延遲載入

        # 分析引擎
        self.scoring_engine = ScoringEngine(fps=self.fps)
        self._analyzing = False
        self._analysis_done = False

        # 分析結果快取
        self._all_trajectories = {}
        self._frame_detections = {}
        self._robot_positions_cache = {}  # frame_idx -> {label: (cx, cy)}
        self._robot_bboxes_cache = {}     # frame_idx -> {label: (x1, y1, x2, y2)}

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

        # ── 主容器 ──
        main = ctk.CTkFrame(self, fg_color="transparent")
        main.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        main.columnconfigure(0, weight=7)
        main.columnconfigure(1, weight=3)
        main.rowconfigure(0, weight=1)

        # ── 左側：影片 ──
        left = ctk.CTkFrame(main, fg_color="transparent")
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)

        video_card = ctk.CTkFrame(left, fg_color=COLORS["bg_card"],
                                   corner_radius=12)
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

        # 播放控制列
        playback = ctk.CTkFrame(left, fg_color=COLORS["bg_card"],
                                 corner_radius=10)
        playback.grid(row=1, column=0, sticky="ew", pady=(6, 0))
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

        # 工具列
        toolbar = ctk.CTkFrame(left, fg_color=COLORS["bg_card"],
                                corner_radius=10)
        toolbar.grid(row=2, column=0, sticky="ew", pady=(4, 0))

        self.mark_robot_btn = ctk.CTkButton(
            toolbar, text="標記機器人", height=32, corner_radius=8,
            fg_color=COLORS["success"], hover_color=COLORS["success_hover"],
            text_color="white",
            font=ctk.CTkFont(family="Microsoft JhengHei UI", size=12),
            command=self._start_mark_robot)
        self.mark_robot_btn.pack(side=tk.LEFT, padx=(8, 4), pady=6)

        self.mark_red_hub_btn = ctk.CTkButton(
            toolbar, text="標記紅方 Hub", height=32, corner_radius=8,
            fg_color="#dc2626", hover_color="#b91c1c",
            text_color="white",
            font=ctk.CTkFont(family="Microsoft JhengHei UI", size=12),
            command=lambda: self._start_mark_zone_polygon("red"))
        self.mark_red_hub_btn.pack(side=tk.LEFT, padx=4, pady=6)

        self.mark_blue_hub_btn = ctk.CTkButton(
            toolbar, text="標記藍方 Hub", height=32, corner_radius=8,
            fg_color="#2563eb", hover_color="#1d4ed8",
            text_color="white",
            font=ctk.CTkFont(family="Microsoft JhengHei UI", size=12),
            command=lambda: self._start_mark_zone_polygon("blue"))
        self.mark_blue_hub_btn.pack(side=tk.LEFT, padx=4, pady=6)

        self.crop_btn = ctk.CTkButton(
            toolbar, text="裁切畫面", height=32, corner_radius=8,
            fg_color="#e67e22", hover_color="#d35400",
            text_color="white",
            font=ctk.CTkFont(family="Microsoft JhengHei UI", size=12),
            command=self._start_crop)
        self.crop_btn.pack(side=tk.LEFT, padx=4, pady=6)

        self.reset_crop_btn = ctk.CTkButton(
            toolbar, text="重置裁切", height=32, corner_radius=8,
            fg_color=COLORS["border"], hover_color=COLORS["border_hover"],
            text_color=COLORS["text"],
            font=ctk.CTkFont(family="Microsoft JhengHei UI", size=12),
            command=self._reset_crop)
        self.reset_crop_btn.pack(side=tk.LEFT, padx=4, pady=6)

        self.clear_marks_btn = ctk.CTkButton(
            toolbar, text="清除標記", height=32, corner_radius=8,
            fg_color=COLORS["border"], hover_color=COLORS["border_hover"],
            text_color=COLORS["text"],
            font=ctk.CTkFont(family="Microsoft JhengHei UI", size=12),
            command=self._clear_all_marks)
        self.clear_marks_btn.pack(side=tk.LEFT, padx=4, pady=6)

        self.analyze_btn = ctk.CTkButton(
            toolbar, text="開始分析", height=32, corner_radius=8,
            fg_color=COLORS["accent"], hover_color=COLORS["accent_hover"],
            text_color=COLORS["bg_primary"],
            font=ctk.CTkFont(family="Microsoft JhengHei UI",
                              size=13, weight="bold"),
            command=self._on_analyze)
        self.analyze_btn.pack(side=tk.LEFT, padx=(12, 4), pady=6)

        self.export_btn = ctk.CTkButton(
            toolbar, text="匯出 CSV", height=32, corner_radius=8,
            fg_color=COLORS["border"], hover_color=COLORS["border_hover"],
            text_color=COLORS["text"],
            font=ctk.CTkFont(family="Microsoft JhengHei UI", size=12),
            command=self._export_csv)
        self.export_btn.pack(side=tk.RIGHT, padx=(4, 8), pady=6)

        # ── 右側面板 ──
        right = ctk.CTkFrame(main, fg_color="transparent")
        right.grid(row=0, column=1, sticky="nsew", padx=(4, 0))
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)

        # 設定面板
        settings_card = ctk.CTkFrame(right, fg_color=COLORS["bg_card"],
                                      corner_radius=12)
        settings_card.grid(row=0, column=0, sticky="ew", pady=(0, 6))

        ctk.CTkLabel(settings_card, text="設定",
                      text_color=COLORS["accent"],
                      font=ctk.CTkFont(family="Microsoft JhengHei UI",
                                        size=14, weight="bold")
                      ).pack(anchor=tk.W, padx=12, pady=(12, 6))

        # Auto 時間設定
        auto_frame = ctk.CTkFrame(settings_card, fg_color="transparent")
        auto_frame.pack(fill=tk.X, padx=12, pady=(0, 4))

        ctk.CTkLabel(auto_frame, text="Auto 時長 (秒):",
                      text_color=COLORS["text"],
                      font=ctk.CTkFont(size=12)
                      ).pack(side=tk.LEFT)

        self.auto_entry = ctk.CTkEntry(
            auto_frame, width=60, height=28,
            fg_color=COLORS["bg_secondary"],
            text_color=COLORS["text"],
            border_color=COLORS["border"])
        self.auto_entry.pack(side=tk.LEFT, padx=(6, 0))
        self.auto_entry.insert(0, str(AUTO_DURATION_SEC))

        # 偵測模式設定
        det_frame = ctk.CTkFrame(settings_card, fg_color="transparent")
        det_frame.pack(fill=tk.X, padx=12, pady=(0, 4))

        ctk.CTkLabel(det_frame, text="Ball Detection:",
                      text_color=COLORS["text"],
                      font=ctk.CTkFont(size=12)
                      ).pack(side=tk.LEFT)

        self._det_mode_var = ctk.StringVar(value=self._detection_mode)
        self.det_mode_menu = ctk.CTkOptionMenu(
            det_frame, width=80, height=28,
            values=["AI", "HSV"],
            variable=self._det_mode_var,
            fg_color=COLORS["bg_secondary"],
            button_color=COLORS["border"],
            button_hover_color=COLORS["border_hover"],
            dropdown_fg_color=COLORS["bg_card"],
            dropdown_hover_color=COLORS["border"],
            text_color=COLORS["text"],
            font=ctk.CTkFont(size=12),
            command=self._on_detection_mode_change)
        self.det_mode_menu.pack(side=tk.LEFT, padx=(6, 0))

        # 狀態
        self.status_label = ctk.CTkLabel(
            settings_card, text="就緒 — 請開啟影片",
            text_color=COLORS["text_secondary"],
            font=ctk.CTkFont(family="Microsoft JhengHei UI", size=12))
        self.status_label.pack(anchor=tk.W, padx=12, pady=(4, 4))

        self.progress_bar = ctk.CTkProgressBar(
            settings_card, progress_color=COLORS["accent"],
            fg_color=COLORS["border"], corner_radius=4, height=6)
        self.progress_bar.set(0)
        self.progress_bar.pack(fill=tk.X, padx=12, pady=(0, 6))
        self.progress_bar.pack_forget()

        # 機器人列表
        robot_list_card = ctk.CTkFrame(settings_card, fg_color="transparent")
        robot_list_card.pack(fill=tk.X, padx=12, pady=(0, 12))

        ctk.CTkLabel(robot_list_card, text="已標記機器人:",
                      text_color=COLORS["text_secondary"],
                      font=ctk.CTkFont(size=11)
                      ).pack(anchor=tk.W)

        self.robot_list_label = ctk.CTkLabel(
            robot_list_card, text="（無）",
            text_color=COLORS["text"],
            font=ctk.CTkFont(size=11), justify=tk.LEFT)
        self.robot_list_label.pack(anchor=tk.W, padx=(8, 0))

        # 得分區域列表
        ctk.CTkLabel(robot_list_card, text="得分區域:",
                      text_color=COLORS["text_secondary"],
                      font=ctk.CTkFont(size=11)
                      ).pack(anchor=tk.W, pady=(6, 0))

        self.zone_list_label = ctk.CTkLabel(
            robot_list_card, text="（無）",
            text_color=COLORS["text"],
            font=ctk.CTkFont(size=11), justify=tk.LEFT)
        self.zone_list_label.pack(anchor=tk.W, padx=(8, 0))

        # ── 分頁面板 ──
        self.tabview = ctk.CTkTabview(
            right, fg_color=COLORS["bg_card"], corner_radius=12,
            segmented_button_fg_color=COLORS["border"],
            segmented_button_selected_color=COLORS["accent"],
            segmented_button_unselected_color=COLORS["bg_secondary"])
        self.tabview.grid(row=1, column=0, sticky="nsew")

        # Tab 1: 得分統計
        tab_score = self.tabview.add("得分統計")
        tab_score.rowconfigure(0, weight=1)
        tab_score.columnconfigure(0, weight=1)

        score_tree_frame = ctk.CTkFrame(tab_score, fg_color="transparent")
        score_tree_frame.grid(row=0, column=0, sticky="nsew")
        score_tree_frame.rowconfigure(0, weight=1)
        score_tree_frame.columnconfigure(0, weight=1)

        score_cols = ("robot", "alliance", "auto", "teleop", "total",
                      "shots", "miss", "acc")
        self.score_tree = ttk.Treeview(
            score_tree_frame, columns=score_cols, show="headings",
            style="Dark.Treeview")
        self.score_tree.heading("robot", text="機器人")
        self.score_tree.heading("alliance", text="聯盟")
        self.score_tree.heading("auto", text="Auto")
        self.score_tree.heading("teleop", text="Teleop")
        self.score_tree.heading("total", text="進球")
        self.score_tree.heading("shots", text="出手")
        self.score_tree.heading("miss", text="未進")
        self.score_tree.heading("acc", text="命中率")
        self.score_tree.column("robot", width=60, anchor=tk.CENTER)
        self.score_tree.column("alliance", width=35, anchor=tk.CENTER)
        self.score_tree.column("auto", width=40, anchor=tk.CENTER)
        self.score_tree.column("teleop", width=48, anchor=tk.CENTER)
        self.score_tree.column("total", width=40, anchor=tk.CENTER)
        self.score_tree.column("shots", width=40, anchor=tk.CENTER)
        self.score_tree.column("miss", width=40, anchor=tk.CENTER)
        self.score_tree.column("acc", width=48, anchor=tk.CENTER)

        score_scroll = ttk.Scrollbar(score_tree_frame, orient=tk.VERTICAL,
                                      command=self.score_tree.yview)
        self.score_tree.configure(yscrollcommand=score_scroll.set)
        self.score_tree.grid(row=0, column=0, sticky="nsew")
        score_scroll.grid(row=0, column=1, sticky="ns")

        # Tab 2: 進球事件
        tab_events = self.tabview.add("進球事件")
        tab_events.rowconfigure(0, weight=1)
        tab_events.columnconfigure(0, weight=1)

        event_tree_frame = ctk.CTkFrame(tab_events, fg_color="transparent")
        event_tree_frame.grid(row=0, column=0, sticky="nsew")
        event_tree_frame.rowconfigure(0, weight=1)
        event_tree_frame.columnconfigure(0, weight=1)

        ecols = ("idx", "type", "time", "period", "alliance", "shooter")
        self.event_tree = ttk.Treeview(
            event_tree_frame, columns=ecols, show="headings",
            style="Dark.Treeview")
        self.event_tree.heading("idx", text="#")
        self.event_tree.heading("type", text="類型")
        self.event_tree.heading("time", text="時間")
        self.event_tree.heading("period", text="期間")
        self.event_tree.heading("alliance", text="聯盟")
        self.event_tree.heading("shooter", text="射手")
        self.event_tree.column("idx", width=28, anchor=tk.CENTER)
        self.event_tree.column("type", width=40, anchor=tk.CENTER)
        self.event_tree.column("time", width=75, anchor=tk.CENTER)
        self.event_tree.column("period", width=50, anchor=tk.CENTER)
        self.event_tree.column("alliance", width=35, anchor=tk.CENTER)
        self.event_tree.column("shooter", width=55, anchor=tk.CENTER)

        event_scroll = ttk.Scrollbar(event_tree_frame, orient=tk.VERTICAL,
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

    # ══════════════════════════════════════════════════════
    # 座標轉換
    # ══════════════════════════════════════════════════════

    def _canvas_to_video(self, cx, cy):
        vx = (cx - self._display_offset_x) / self._display_scale
        vy = (cy - self._display_offset_y) / self._display_scale
        return (vx, vy)

    def _video_to_canvas(self, vx, vy):
        cx = vx * self._display_scale + self._display_offset_x
        cy = vy * self._display_scale + self._display_offset_y
        return (cx, cy)

    def _video_to_resized(self, point, scale):
        return (int(point[0] * scale), int(point[1] * scale))

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
        self._set_status("影片已載入 — 請標記機器人和得分區域", COLORS["info"])
        self._show_frame(0)

    def _show_frame(self, frame_idx):
        if not self.cap:
            return
        frame_idx = max(0, min(frame_idx, self.total_frames - 1))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if not ret:
            return

        self.current_frame = frame_idx

        # ROI 裁切
        if self._roi:
            rx, ry, rw, rh = self._roi
            frame = frame[ry:ry+rh, rx:rx+rw]

        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 10 or ch < 10:
            cw, ch = 900, 600

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

        # 繪製機器人初始標記（在標記幀或未分析時）
        if not self._analysis_done:
            for i, (label, alliance, x, y, w, h, mark_f) in enumerate(self._robot_markers):
                if frame_idx != mark_f:
                    continue
                color = self._get_robot_color(label)
                p1 = self._video_to_resized((x, y), scale)
                p2 = self._video_to_resized((x + w, y + h), scale)
                cv2.rectangle(resized, p1, p2, color["bgr"], 2, cv2.LINE_AA)

        # 繪製拖曳中的矩形（機器人框選 / 裁切區域）
        if self._drag_start and self._drag_current and \
                self.interaction_mode in ("mark_robot", "crop_region"):
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

    def _draw_analysis_overlay(self, resized, frame_idx, scale):
        """繪製分析結果的幾何圖形 overlay。"""
        # 球偵測圈
        if frame_idx in self._frame_detections:
            for det in self._frame_detections[frame_idx]:
                cx, cy, area = det[0], det[1], det[2]
                pt = self._video_to_resized((cx, cy), scale)
                radius = max(4, int(math.sqrt(area / math.pi) * scale))
                cv2.circle(resized, pt, radius, (36, 191, 251), 2,
                           cv2.LINE_AA)

        # 球軌跡
        for tid, traj in self._all_trajectories.items():
            points = [p for p in traj if abs(p[0] - frame_idx) <= 20]
            if len(points) < 2:
                continue
            color = (0, 200, 200)
            for i in range(len(points) - 1):
                p1 = self._video_to_resized((points[i][1], points[i][2]),
                                            scale)
                p2 = self._video_to_resized((points[i+1][1], points[i+1][2]),
                                            scale)
                cv2.line(resized, p1, p2, color, 1, cv2.LINE_AA)

        # 機器人追蹤框（優先 bbox，無 bbox 則畫圓點）
        robot_bboxes = self._robot_bboxes_cache.get(frame_idx, {})
        if frame_idx in self._robot_positions_cache:
            for label, (cx, cy) in self._robot_positions_cache[frame_idx].items():
                color = self._get_robot_color(label)
                if label in robot_bboxes:
                    x1, y1, x2, y2 = robot_bboxes[label]
                    p1 = self._video_to_resized((x1, y1), scale)
                    p2 = self._video_to_resized((x2, y2), scale)
                    cv2.rectangle(resized, p1, p2, color["bgr"], 2,
                                  cv2.LINE_AA)
                else:
                    pt = self._video_to_resized((cx, cy), scale)
                    cv2.circle(resized, pt, 8, color["bgr"], -1, cv2.LINE_AA)
                    cv2.circle(resized, pt, 10, color["bgr"], 2,
                               cv2.LINE_AA)

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

    def _draw_analysis_labels(self, draw, frame_idx, scale):
        """繪製分析後的文字標籤。"""
        # 機器人名稱標籤
        robot_bboxes = self._robot_bboxes_cache.get(frame_idx, {})
        if frame_idx in self._robot_positions_cache:
            for label, (cx, cy) in self._robot_positions_cache[frame_idx].items():
                color = self._get_robot_color(label)
                if label in robot_bboxes:
                    x1, y1, x2, y2 = robot_bboxes[label]
                    pt = self._video_to_resized((x1, y1 - 4), scale)
                else:
                    pt = self._video_to_resized((cx, cy - 15), scale)
                draw.text((pt[0], max(0, pt[1] - 16)), label,
                          fill=color["rgb"], font=self._label_font)

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
            self._play_loop()
        else:
            self.play_btn.configure(text="▶ 播放")

    def _toggle_speed(self):
        """切換播放倍速 1x ↔ 0.5x。"""
        if self._playback_speed == 1.0:
            self._playback_speed = 0.5
            self.speed_btn.configure(text="0.5x")
        else:
            self._playback_speed = 1.0
            self.speed_btn.configure(text="1x")

    def _play_loop(self):
        if not self.is_playing or not self.cap:
            return
        if self.current_frame >= self.total_frames - 1:
            self.is_playing = False
            self.play_btn.configure(text="▶ 播放")
            return

        t0 = time.monotonic()
        self._show_frame(self.current_frame + 1)
        render_ms = (time.monotonic() - t0) * 1000

        # 每幀目標間隔（ms）— 扣除渲染時間
        frame_ms = 1000.0 / (self.fps * self._playback_speed)
        delay = max(1, int(frame_ms - render_ms))
        self.after(delay, self._play_loop)

    # ══════════════════════════════════════════════════════
    # 標記互動（機器人 + 得分區域）
    # ══════════════════════════════════════════════════════

    def _start_mark_robot(self):
        if not self.cap:
            self._set_status("請先開啟影片", COLORS["error"])
            return
        if len(self._robot_markers) >= MAX_ROBOTS:
            self._set_status(f"最多標記 {MAX_ROBOTS} 台機器人", COLORS["error"])
            return
        self.interaction_mode = "mark_robot"
        self._set_status("在影片上拖曳框選機器人", COLORS["accent"])
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

    def _cancel_interaction(self):
        if self.interaction_mode:
            self.interaction_mode = None
            self._drag_start = None
            self._drag_current = None
            self._polygon_points = []
            self._current_polygon_alliance = ""
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
        self._scoring_zones.clear()
        self._update_robot_list()
        self._update_zone_list()
        self._clear_analysis()

        self.interaction_mode = None
        self.canvas.config(cursor="")
        self._set_status(f"已裁切畫面至 {w}x{h}，請重新標記機器人和得分區域",
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
        self._scoring_zones.clear()
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

        if self.interaction_mode == "mark_zone_polygon":
            self._polygon_points.append((int(vx), int(vy)))
            n = len(self._polygon_points)
            self._set_status(
                f"已放置 {n} 個頂點 — 右鍵或雙擊完成（至少 3 點），ESC 取消",
                COLORS["accent"])
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
        if self.interaction_mode == "mark_zone_polygon":
            return  # 多邊形模式由右鍵/雙擊完成

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

        if self.interaction_mode == "mark_robot":
            self._finish_mark_robot(x, y, w, h)
        elif self.interaction_mode == "crop_region":
            self._finish_crop(x, y, w, h)

        self._drag_start = None
        self._drag_current = None

    def _on_canvas_double_click(self, event):
        """雙擊完成多邊形標記。"""
        if self.interaction_mode == "mark_zone_polygon":
            self._finish_mark_polygon()

    def _on_canvas_right_click(self, event):
        """右鍵完成多邊形標記。"""
        if self.interaction_mode == "mark_zone_polygon":
            self._finish_mark_polygon()

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

    def _finish_mark_robot(self, x, y, w, h):
        """完成機器人標記。"""
        # 彈出輸入框取得機器人編號
        label = simpledialog.askstring(
            "機器人編號",
            "請輸入機器人編號（例如 6998）:",
            parent=self
        )
        if not label:
            self._set_status("已取消標記", COLORS["text_secondary"])
            self.interaction_mode = None
            self.canvas.config(cursor="")
            self._show_frame(self.current_frame)
            return

        label = label.strip()

        # 檢查是否重複
        for existing_label, *_ in self._robot_markers:
            if existing_label == label:
                self._set_status(f"機器人 {label} 已存在", COLORS["error"])
                return

        # 彈窗詢問聯盟
        alliance = self._ask_alliance()
        if alliance is None:
            self._set_status("已取消標記", COLORS["text_secondary"])
            self.interaction_mode = None
            self.canvas.config(cursor="")
            self._show_frame(self.current_frame)
            return

        self._robot_markers.append((label, alliance, x, y, w, h, self.current_frame))
        self._update_robot_list()

        alliance_name = "紅方" if alliance == "red" else "藍方"
        self.interaction_mode = None
        self.canvas.config(cursor="")
        self._set_status(
            f"已標記{alliance_name}機器人 {label}（幀 {self.current_frame}）",
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
        self._scoring_zones.clear()
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
        if not self._scoring_zones:
            self.zone_list_label.configure(text="（無）")
        else:
            lines = []
            for zone in self._scoring_zones:
                lines.append(f"  {zone.name} ({len(zone.points)} 頂點)")
            self.zone_list_label.configure(text="\n".join(lines))

    # ══════════════════════════════════════════════════════
    # 分析引擎
    # ══════════════════════════════════════════════════════

    def _on_detection_mode_change(self, value):
        """偵測模式切換回調。"""
        self._detection_mode = value
        self._ai_model = None  # 重置，分析時再載入
        mode_name = "AI 模型" if value == "AI" else "HSV 色彩過濾"
        self._set_status(f"偵測模式: {mode_name}", COLORS["info"])

    def _on_analyze(self):
        if not self.cap:
            self._set_status("請先開啟影片", COLORS["error"])
            return
        if not self._scoring_zones:
            self._set_status("請先標記至少一個得分區域", COLORS["error"])
            return
        if self._analyzing:
            self._set_status("分析進行中...", COLORS["accent"])
            return

        # 讀取 auto 時間設定
        try:
            self.auto_duration = float(self.auto_entry.get())
        except ValueError:
            self.auto_duration = AUTO_DURATION_SEC

        self._clear_analysis()
        self._analyzing = True
        self.analyze_btn.configure(state="disabled")
        self.progress_bar.pack(fill=tk.X, padx=12, pady=(0, 6))
        self.progress_bar.set(0)
        self._set_status("分析中...", COLORS["info"])

        thread = threading.Thread(target=self._run_analysis, daemon=True)
        thread.start()

    def _run_analysis(self):
        """背景線程：逐幀球偵測+追蹤+機器人追蹤+進球判定。"""
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            self.after(0, lambda: self._analysis_error("無法開啟影片"))
            return

        # AI 球偵測模型載入（若選擇 AI 模式）
        use_ai = self._detection_mode == "AI"
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

        # 機器人偵測模型載入（MOT 模式）
        robot_detector = None
        try:
            self.after(0, lambda: self._set_status(
                "載入機器人偵測模型中...", COLORS["info"]))
            robot_detector = load_robot_model()
            self.after(0, lambda: self._set_status(
                "機器人偵測模型已載入（MOT 模式）", COLORS["info"]))
        except FileNotFoundError as e:
            print(f"[INFO] 機器人偵測模型不可用，使用 SOT 追蹤: {e}")
            self.after(0, lambda: self._set_status(
                "機器人偵測模型不可用，使用 SOT 追蹤模式", COLORS["text_secondary"]))
        except Exception as e:
            print(f"[WARN] 機器人偵測模型載入失敗: {e}")
            self.after(0, lambda: self._set_status(
                "機器人偵測模型載入失敗，使用 SOT 追蹤", COLORS["error"]))

        # 初始化球追蹤器
        ball_tracker = CentroidTracker(max_distance=MAX_MATCH_DIST,
                                       max_missed=MAX_MISSED)

        # 初始化機器人追蹤器（MOT 或 SOT）
        robot_mgr = RobotTrackerManager(
            detector=robot_detector, fps=self.fps)
        tracking_mode = "MOT" if robot_mgr.use_mot else "SOT"

        # 初始化進球引擎
        engine = ScoringEngine(
            fps=self.fps,
            auto_sec=self.auto_duration,
            teleop_start_sec=self.auto_duration
        )
        engine.set_zones(self._scoring_zones)

        total = self.total_frames
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

        # 偵測函式選擇
        def detect_balls(frame):
            if use_ai and ai_model is not None:
                return detect_fuel_ai(frame, ai_model)
            return detect_yellow_balls(frame)

        ball_mode = "AI" if use_ai else "HSV"
        mode_label = f"{ball_mode}+{tracking_mode}"

        # 逐幀處理
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for frame_idx in range(total):
            ret, frame = cap.read()
            if not ret:
                break

            # ROI 裁切
            if roi:
                rx, ry, rw, rh = roi
                frame = frame[ry:ry+rh, rx:rx+rw]

            # SOT 模式：在標記幀初始化追蹤器（需要影像）
            if not robot_mgr.use_mot and frame_idx in markers_by_frame:
                for label, alliance, x, y, w, h in markers_by_frame[frame_idx]:
                    robot_mgr.add_robot(
                        label, (x, y, w, h), frame, frame_idx, alliance)

            # 球偵測+追蹤
            dets = detect_balls(frame)
            frame_detections[frame_idx] = dets
            ball_positions = ball_tracker.update(dets, frame_idx)

            # 機器人追蹤
            robot_mgr.update_all(frame, frame_idx)
            robot_pos = robot_mgr.get_all_positions(frame_idx)
            robot_positions_cache[frame_idx] = \
                robot_mgr.get_all_display_positions(frame_idx)
            robot_bboxes_cache[frame_idx] = \
                robot_mgr.get_all_bboxes(frame_idx)

            # 進球判定
            engine.process_frame(frame_idx, ball_positions, robot_pos,
                                 ball_tracker.trajectories)

            # 更新進度
            if frame_idx % 5 == 0:
                pct = (frame_idx + 1) / total * 100
                self.after(0, lambda p=pct, f=frame_idx, m=mode_label:
                           self._update_progress(p, f, m))

        cap.release()

        # 後處理：位置插值（MOT 模式）
        robot_mgr.interpolate_positions()
        # 更新插值後的位置快取
        if robot_mgr.use_mot:
            for frame_idx in range(total):
                pos = robot_mgr.get_all_display_positions(frame_idx)
                if pos:
                    robot_positions_cache[frame_idx] = pos
                bbox = robot_mgr.get_all_bboxes(frame_idx)
                if bbox:
                    robot_bboxes_cache[frame_idx] = bbox

        # 軌跡縫合
        all_trajectories = stitch_trajectories(dict(ball_tracker.trajectories))

        # 出手偵測後處理
        engine.detect_shots(all_trajectories, robot_positions_cache)

        # 回到主線程
        self.after(0, lambda: self._finish_analysis(
            all_trajectories, frame_detections,
            robot_positions_cache, robot_bboxes_cache, engine
        ))

    def _update_progress(self, pct, frame_idx, mode=""):
        self.progress_bar.set(pct / 100.0)
        mode_str = f" [{mode}]" if mode else ""
        self._set_status(
            f"分析中{mode_str}... 幀 {frame_idx}/{self.total_frames - 1} ({pct:.0f}%)",
            COLORS["info"])

    def _analysis_error(self, msg):
        self._analyzing = False
        self.analyze_btn.configure(state="normal")
        self.progress_bar.pack_forget()
        self._set_status(f"分析錯誤: {msg}", COLORS["error"])

    def _finish_analysis(self, trajectories, frame_dets,
                         robot_cache, robot_bbox_cache, engine):
        """分析完成，更新 UI。"""
        self._all_trajectories = trajectories
        self._frame_detections = frame_dets
        self._robot_positions_cache = robot_cache
        self._robot_bboxes_cache = robot_bbox_cache
        self.scoring_engine = engine
        self._analysis_done = True
        self._analyzing = False

        self.analyze_btn.configure(state="normal")
        self.progress_bar.pack_forget()

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

    def _clear_analysis(self):
        """清除分析結果。"""
        self._analysis_done = False
        self._all_trajectories.clear()
        self._frame_detections.clear()
        self._robot_positions_cache.clear()
        self._robot_bboxes_cache.clear()
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

        # 確保所有已標記的機器人都在表格中
        all_labels = set(m[0] for m in self._robot_markers)
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

    def destroy(self):
        self.is_playing = False
        if self.cap:
            self.cap.release()
        super().destroy()

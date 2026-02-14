"""
FRC Scoring Analyzer — GUI 主類別
"""

import csv
import math
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, simpledialog
from pathlib import Path

import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageTk

from config import (
    COLORS, MAX_MATCH_DIST, MAX_MISSED, MAX_ROBOTS,
    ROBOT_COLORS, ROBOT_COLORS_HEX, AUTO_DURATION_SEC,
)
from detection import detect_yellow_balls
from tracking import CentroidTracker, stitch_trajectories
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
        self.photo_image = None
        self.video_width = 0
        self.video_height = 0
        self._first_frame = None  # 快取第一幀（用於標記）

        # 顯示用的偏移/縮放
        self._display_scale = 1.0
        self._display_offset_x = 0
        self._display_offset_y = 0

        # 互動模式
        self.interaction_mode = None  # None, "mark_robot", "mark_zone_upper", "mark_zone_lower"
        self._drag_start = None      # 拖曳起始點（影片座標）
        self._drag_current = None    # 拖曳當前點

        # 機器人追蹤
        self.robot_manager = RobotTrackerManager()
        self._robot_markers = []  # [(label, x, y, w, h)] 使用者標記的初始 bbox

        # 得分區域
        self._scoring_zones = []  # [ScoringZone, ...]

        # 分析引擎
        self.scoring_engine = ScoringEngine(fps=self.fps)
        self._analyzing = False
        self._analysis_done = False

        # 分析結果快取
        self._all_trajectories = {}
        self._frame_detections = {}
        self._robot_positions_cache = {}  # frame_idx -> {label: (cx, cy)}

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

        self.fps_label = ctk.CTkLabel(
            playback, text="FPS: --",
            text_color=COLORS["text_secondary"],
            font=ctk.CTkFont(family="Segoe UI", size=12))
        self.fps_label.grid(row=0, column=3, padx=(0, 8))

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

        self.mark_upper_btn = ctk.CTkButton(
            toolbar, text="標記 Upper HUB", height=32, corner_radius=8,
            fg_color="#a855f7", hover_color="#9333ea",
            text_color="white",
            font=ctk.CTkFont(family="Microsoft JhengHei UI", size=12),
            command=lambda: self._start_mark_zone("Upper HUB"))
        self.mark_upper_btn.pack(side=tk.LEFT, padx=4, pady=6)

        self.mark_lower_btn = ctk.CTkButton(
            toolbar, text="標記 Lower HUB", height=32, corner_radius=8,
            fg_color=COLORS["info"], hover_color="#3b82f6",
            text_color="white",
            font=ctk.CTkFont(family="Microsoft JhengHei UI", size=12),
            command=lambda: self._start_mark_zone("Lower HUB"))
        self.mark_lower_btn.pack(side=tk.LEFT, padx=4, pady=6)

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
        right.rowconfigure(2, weight=1)
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

        # 得分統計表格
        score_card = ctk.CTkFrame(right, fg_color=COLORS["bg_card"],
                                   corner_radius=12)
        score_card.grid(row=1, column=0, sticky="ew", pady=(0, 6))

        ctk.CTkLabel(score_card, text="得分統計",
                      text_color=COLORS["accent"],
                      font=ctk.CTkFont(family="Microsoft JhengHei UI",
                                        size=14, weight="bold")
                      ).pack(anchor=tk.W, padx=12, pady=(12, 6))

        score_tree_frame = ctk.CTkFrame(score_card, fg_color="transparent")
        score_tree_frame.pack(fill=tk.X, padx=8, pady=(0, 12))

        cols = ("robot", "auto", "teleop", "upper", "lower", "total")
        self.score_tree = ttk.Treeview(
            score_tree_frame, columns=cols, show="headings",
            height=6, style="Dark.Treeview")
        self.score_tree.heading("robot", text="機器人")
        self.score_tree.heading("auto", text="Auto")
        self.score_tree.heading("teleop", text="Teleop")
        self.score_tree.heading("upper", text="Upper")
        self.score_tree.heading("lower", text="Lower")
        self.score_tree.heading("total", text="總計")
        self.score_tree.column("robot", width=70, anchor=tk.CENTER)
        self.score_tree.column("auto", width=50, anchor=tk.CENTER)
        self.score_tree.column("teleop", width=55, anchor=tk.CENTER)
        self.score_tree.column("upper", width=50, anchor=tk.CENTER)
        self.score_tree.column("lower", width=50, anchor=tk.CENTER)
        self.score_tree.column("total", width=50, anchor=tk.CENTER)
        self.score_tree.pack(fill=tk.X)

        # 時間軸事件列表
        events_card = ctk.CTkFrame(right, fg_color=COLORS["bg_card"],
                                    corner_radius=12)
        events_card.grid(row=2, column=0, sticky="nsew")
        events_card.rowconfigure(1, weight=1)
        events_card.columnconfigure(0, weight=1)

        ctk.CTkLabel(events_card, text="進球事件時間軸",
                      text_color=COLORS["accent"],
                      font=ctk.CTkFont(family="Microsoft JhengHei UI",
                                        size=14, weight="bold")
                      ).grid(row=0, column=0, sticky="w", padx=12, pady=(12, 6))

        event_tree_frame = ctk.CTkFrame(events_card, fg_color="transparent")
        event_tree_frame.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))
        event_tree_frame.rowconfigure(0, weight=1)
        event_tree_frame.columnconfigure(0, weight=1)

        ecols = ("idx", "time", "period", "zone", "shooter")
        self.event_tree = ttk.Treeview(
            event_tree_frame, columns=ecols, show="headings",
            height=12, style="Dark.Treeview")
        self.event_tree.heading("idx", text="#")
        self.event_tree.heading("time", text="時間")
        self.event_tree.heading("period", text="期間")
        self.event_tree.heading("zone", text="區域")
        self.event_tree.heading("shooter", text="射手")
        self.event_tree.column("idx", width=30, anchor=tk.CENTER)
        self.event_tree.column("time", width=65, anchor=tk.CENTER)
        self.event_tree.column("period", width=55, anchor=tk.CENTER)
        self.event_tree.column("zone", width=55, anchor=tk.CENTER)
        self.event_tree.column("shooter", width=65, anchor=tk.CENTER)

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

        # 快取第一幀
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = self.cap.read()
        if ret:
            self._first_frame = frame.copy()

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

        # 繪製得分區域
        for zone in self._scoring_zones:
            color = (200, 100, 255) if "upper" in zone.name.lower() \
                else (255, 200, 100)
            p1 = self._video_to_resized((zone.x, zone.y), scale)
            p2 = self._video_to_resized((zone.x + zone.w, zone.y + zone.h), scale)
            cv2.rectangle(resized, p1, p2, color, 2, cv2.LINE_AA)

        # 繪製機器人初始標記（在第一幀或未分析時）
        if not self._analysis_done:
            for i, (label, x, y, w, h) in enumerate(self._robot_markers):
                ci = i % len(ROBOT_COLORS)
                color_bgr = (ROBOT_COLORS[ci][2], ROBOT_COLORS[ci][1],
                             ROBOT_COLORS[ci][0])
                p1 = self._video_to_resized((x, y), scale)
                p2 = self._video_to_resized((x + w, y + h), scale)
                cv2.rectangle(resized, p1, p2, color_bgr, 2, cv2.LINE_AA)

        # 繪製拖曳中的矩形
        if self._drag_start and self._drag_current:
            sx, sy = self._drag_start
            ex, ey = self._drag_current
            p1 = self._video_to_resized((min(sx, ex), min(sy, ey)), scale)
            p2 = self._video_to_resized((max(sx, ex), max(sy, ey)), scale)
            if self.interaction_mode == "mark_robot":
                cv2.rectangle(resized, p1, p2, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                color = (200, 100, 255) if "upper" in (self.interaction_mode or "") \
                    else (255, 200, 100)
                cv2.rectangle(resized, p1, p2, color, 2, cv2.LINE_AA)

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
            for i, (label, x, y, w, h) in enumerate(self._robot_markers):
                ci = i % len(ROBOT_COLORS)
                color_rgb = ROBOT_COLORS[ci]
                pt = self._video_to_resized((x, y - 4), scale)
                draw.text((pt[0], max(0, pt[1] - 16)), label,
                          fill=color_rgb, font=self._label_font)

        # 得分區域標籤
        for zone in self._scoring_zones:
            color = (200, 100, 255) if "upper" in zone.name.lower() \
                else (255, 200, 100)
            pt = self._video_to_resized((zone.x, zone.y - 4), scale)
            draw.text((pt[0], max(0, pt[1] - 16)), zone.name,
                      fill=color, font=self._small_font)

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
                cv2.line(resized, p1, p2, color, 2, cv2.LINE_AA)

        # 機器人追蹤框
        if frame_idx in self._robot_positions_cache:
            for label, (cx, cy) in self._robot_positions_cache[frame_idx].items():
                idx = self._get_robot_color_idx(label)
                color_bgr = (ROBOT_COLORS[idx][2], ROBOT_COLORS[idx][1],
                             ROBOT_COLORS[idx][0])
                pt = self._video_to_resized((cx, cy), scale)
                cv2.circle(resized, pt, 8, color_bgr, -1, cv2.LINE_AA)
                cv2.circle(resized, pt, 10, color_bgr, 2, cv2.LINE_AA)

        # 進球事件標記
        for event in self.scoring_engine.events:
            if abs(event.frame_idx - frame_idx) <= 5:
                # 在進球事件附近顯示閃爍標記
                alpha = 1.0 - abs(event.frame_idx - frame_idx) / 6.0
                # 用亮色圓圈標記得分區域
                for zone in self._scoring_zones:
                    if ("upper" in zone.name.lower() and
                            event.zone_name == "Upper") or \
                       ("lower" in zone.name.lower() and
                            event.zone_name == "Lower"):
                        zc = zone.center
                        pt = self._video_to_resized(zc, scale)
                        r = int(30 * scale * alpha)
                        cv2.circle(resized, pt, r, (0, 255, 255), 3,
                                   cv2.LINE_AA)

    def _draw_analysis_labels(self, draw, frame_idx, scale):
        """繪製分析後的文字標籤。"""
        # 機器人名稱標籤
        if frame_idx in self._robot_positions_cache:
            for label, (cx, cy) in self._robot_positions_cache[frame_idx].items():
                idx = self._get_robot_color_idx(label)
                color_rgb = ROBOT_COLORS[idx]
                pt = self._video_to_resized((cx, cy - 15), scale)
                draw.text((pt[0] - 10, pt[1] - 16), label,
                          fill=color_rgb, font=self._label_font)

        # 進球事件文字
        for event in self.scoring_engine.events:
            if event.frame_idx == frame_idx:
                for zone in self._scoring_zones:
                    if ("upper" in zone.name.lower() and
                            event.zone_name == "Upper") or \
                       ("lower" in zone.name.lower() and
                            event.zone_name == "Lower"):
                        zc = zone.center
                        pt = self._video_to_resized(zc, scale)
                        text = f"SCORED! {event.shooter_label}"
                        draw.text((pt[0] - 30, pt[1] - 30), text,
                                  fill=(255, 255, 0), font=self._label_font)

    def _get_robot_color_idx(self, label):
        """取得機器人對應的顏色索引。"""
        labels = [m[0] for m in self._robot_markers]
        if label in labels:
            return labels.index(label) % len(ROBOT_COLORS)
        return 0

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

    def _play_loop(self):
        if not self.is_playing or not self.cap:
            return
        if self.current_frame >= self.total_frames - 1:
            self.is_playing = False
            self.play_btn.configure(text="▶ 播放")
            return
        self._show_frame(self.current_frame + 1)
        delay = max(1, int(1000 / self.fps))
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

    def _start_mark_zone(self, zone_name):
        if not self.cap:
            self._set_status("請先開啟影片", COLORS["error"])
            return
        mode = "mark_zone_upper" if "upper" in zone_name.lower() \
            else "mark_zone_lower"
        self.interaction_mode = mode
        self._set_status(f"在影片上拖曳框選 {zone_name}", COLORS["accent"])
        self.canvas.config(cursor="crosshair")

    def _cancel_interaction(self):
        if self.interaction_mode:
            self.interaction_mode = None
            self._drag_start = None
            self._drag_current = None
            self.canvas.config(cursor="")
            self._set_status("已取消", COLORS["text_secondary"])
            self._show_frame(self.current_frame)

    def _on_canvas_press(self, event):
        if not self.interaction_mode or not self.cap:
            return
        vx, vy = self._canvas_to_video(event.x, event.y)
        if 0 <= vx < self.video_width and 0 <= vy < self.video_height:
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
        elif self.interaction_mode in ("mark_zone_upper", "mark_zone_lower"):
            self._finish_mark_zone(x, y, w, h)

        self._drag_start = None
        self._drag_current = None

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

        self._robot_markers.append((label, x, y, w, h))
        self._update_robot_list()

        self.interaction_mode = None
        self.canvas.config(cursor="")
        self._set_status(f"已標記機器人 {label}", COLORS["success"])
        self._show_frame(self.current_frame)

    def _finish_mark_zone(self, x, y, w, h):
        """完成得分區域標記。"""
        if self.interaction_mode == "mark_zone_upper":
            zone_name = "Upper HUB"
        else:
            zone_name = "Lower HUB"

        # 移除同名的舊區域
        self._scoring_zones = [
            z for z in self._scoring_zones if z.name != zone_name
        ]

        zone = ScoringZone(zone_name, x, y, w, h)
        self._scoring_zones.append(zone)
        self._update_zone_list()

        self.interaction_mode = None
        self.canvas.config(cursor="")
        self._set_status(f"已設定 {zone_name}", COLORS["success"])
        self._show_frame(self.current_frame)

    def _clear_all_marks(self):
        """清除所有標記。"""
        self._robot_markers.clear()
        self._scoring_zones.clear()
        self._drag_start = None
        self._drag_current = None
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
            for i, (label, x, y, w, h) in enumerate(self._robot_markers):
                ci = i % len(ROBOT_COLORS_HEX)
                lines.append(f"  {label} ({w}x{h})")
            self.robot_list_label.configure(text="\n".join(lines))

    def _update_zone_list(self):
        """更新右側面板的得分區域列表顯示。"""
        if not self._scoring_zones:
            self.zone_list_label.configure(text="（無）")
        else:
            lines = []
            for zone in self._scoring_zones:
                lines.append(f"  {zone.name} ({zone.w}x{zone.h})")
            self.zone_list_label.configure(text="\n".join(lines))

    # ══════════════════════════════════════════════════════
    # 分析引擎
    # ══════════════════════════════════════════════════════

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

        # 初始化球追蹤器
        ball_tracker = CentroidTracker(max_distance=MAX_MATCH_DIST,
                                       max_missed=MAX_MISSED)

        # 初始化機器人追蹤器
        robot_mgr = RobotTrackerManager()

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

        # 讀取第一幀並初始化機器人追蹤
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, first_frame = cap.read()
        if not ret:
            self.after(0, lambda: self._analysis_error("無法讀取第一幀"))
            return

        for label, x, y, w, h in self._robot_markers:
            robot_mgr.add_robot(label, (x, y, w, h), first_frame, 0)

        # 處理第一幀
        dets = detect_yellow_balls(first_frame)
        frame_detections[0] = dets
        ball_positions = ball_tracker.update(dets, 0)
        robot_results = robot_mgr.update_all(first_frame, 0)
        robot_pos = robot_mgr.get_all_positions(0)
        robot_positions_cache[0] = robot_pos
        engine.process_frame(0, ball_positions, robot_pos,
                             ball_tracker.trajectories)

        # 逐幀處理
        for frame_idx in range(1, total):
            ret, frame = cap.read()
            if not ret:
                break

            # 球偵測+追蹤
            dets = detect_yellow_balls(frame)
            frame_detections[frame_idx] = dets
            ball_positions = ball_tracker.update(dets, frame_idx)

            # 機器人追蹤
            robot_mgr.update_all(frame, frame_idx)
            robot_pos = robot_mgr.get_all_positions(frame_idx)
            robot_positions_cache[frame_idx] = robot_pos

            # 進球判定
            engine.process_frame(frame_idx, ball_positions, robot_pos,
                                 ball_tracker.trajectories)

            # 更新進度
            if frame_idx % 5 == 0:
                pct = (frame_idx + 1) / total * 100
                self.after(0, lambda p=pct, f=frame_idx:
                           self._update_progress(p, f))

        cap.release()

        # 軌跡縫合
        all_trajectories = stitch_trajectories(dict(ball_tracker.trajectories))

        # 回到主線程
        self.after(0, lambda: self._finish_analysis(
            all_trajectories, frame_detections, robot_positions_cache, engine
        ))

    def _update_progress(self, pct, frame_idx):
        self.progress_bar.set(pct / 100.0)
        self._set_status(
            f"分析中... 幀 {frame_idx}/{self.total_frames - 1} ({pct:.0f}%)",
            COLORS["info"])

    def _analysis_error(self, msg):
        self._analyzing = False
        self.analyze_btn.configure(state="normal")
        self.progress_bar.pack_forget()
        self._set_status(f"分析錯誤: {msg}", COLORS["error"])

    def _finish_analysis(self, trajectories, frame_dets, robot_cache, engine):
        """分析完成，更新 UI。"""
        self._all_trajectories = trajectories
        self._frame_detections = frame_dets
        self._robot_positions_cache = robot_cache
        self.scoring_engine = engine
        self._analysis_done = True
        self._analyzing = False

        self.analyze_btn.configure(state="normal")
        self.progress_bar.pack_forget()

        # 更新得分統計表
        self._update_score_table()

        # 更新事件時間軸
        self._update_event_timeline()

        n_events = len(engine.events)
        n_robots = len(self._robot_markers)
        self._set_status(
            f"分析完成！{n_events} 個進球事件，{n_robots} 台機器人",
            COLORS["success"])

        self._show_frame(self.current_frame)

    def _clear_analysis(self):
        """清除分析結果。"""
        self._analysis_done = False
        self._all_trajectories.clear()
        self._frame_detections.clear()
        self._robot_positions_cache.clear()
        self.scoring_engine.reset()

        for item in self.score_tree.get_children():
            self.score_tree.delete(item)
        for item in self.event_tree.get_children():
            self.event_tree.delete(item)

    def _update_score_table(self):
        """更新得分統計表。"""
        for item in self.score_tree.get_children():
            self.score_tree.delete(item)

        summary = self.scoring_engine.get_summary()

        # 確保所有已標記的機器人都在表格中
        all_labels = set(m[0] for m in self._robot_markers)
        all_labels.add("未知")

        for label in sorted(all_labels):
            if label in summary:
                s = summary[label]
                self.score_tree.insert("", tk.END, values=(
                    label, s.auto_total, s.teleop_total,
                    s.upper_total, s.lower_total, s.total
                ))
            else:
                self.score_tree.insert("", tk.END, values=(
                    label, 0, 0, 0, 0, 0
                ))

    def _update_event_timeline(self):
        """更新事件時間軸。"""
        for item in self.event_tree.get_children():
            self.event_tree.delete(item)

        timeline = self.scoring_engine.get_timeline()
        for i, event in enumerate(timeline):
            time_sec = event.frame_idx / self.fps
            time_str = format_time(time_sec)
            self.event_tree.insert("", tk.END, iid=str(i), values=(
                i + 1,
                f"{time_str} ({time_sec:.1f}s)",
                event.period,
                event.zone_name,
                event.shooter_label,
            ))

    def _on_event_click(self, event):
        """點擊事件時跳到對應幀。"""
        sel = self.event_tree.selection()
        if not sel:
            return
        idx = int(sel[0])
        timeline = self.scoring_engine.get_timeline()
        if 0 <= idx < len(timeline):
            self._show_frame(timeline[idx].frame_idx)

    # ══════════════════════════════════════════════════════
    # CSV 匯出
    # ══════════════════════════════════════════════════════

    def _export_csv(self):
        """匯出得分結果為 CSV。"""
        if not self.scoring_engine.events:
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
            writer.writerow(["機器人", "Auto得分", "Teleop得分",
                             "Upper", "Lower", "總計"])
            summary = self.scoring_engine.get_summary()
            for label in sorted(summary.keys()):
                s = summary[label]
                writer.writerow([label, s.auto_total, s.teleop_total,
                                 s.upper_total, s.lower_total, s.total])
            writer.writerow([])

            # 事件詳細
            writer.writerow(["=== 進球事件 ==="])
            writer.writerow(["編號", "幀", "時間(秒)", "期間",
                             "區域", "射手", "射手距離"])
            timeline = self.scoring_engine.get_timeline()
            for i, event in enumerate(timeline):
                time_sec = event.frame_idx / self.fps
                writer.writerow([
                    i + 1, event.frame_idx, f"{time_sec:.3f}",
                    event.period, event.zone_name,
                    event.shooter_label, f"{event.shooter_dist:.1f}"
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

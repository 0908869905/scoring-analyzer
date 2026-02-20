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
            number_of_steps=max(1, hi - lo),
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
        if name in ("（無 preset）", "（未選擇）"):
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

        # 生成預覽（含偵測球位置）
        mask_colored, ball_count, balls = generate_preview(
            frame, hsv_low, hsv_high, min_a, max_a)

        # 在原始影像上標記偵測結果
        preview_frame = frame.copy()
        for cx, cy, r in balls:
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

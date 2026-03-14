"""FRC Scoring Analyzer — 設定面板（嵌入右側分頁）"""

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


class SettingsPanel(ctk.CTkFrame):
    """設定面板 — 嵌入右側 TabView 中，全參數調整 + HSV 即時預覽 + 自動校正。"""

    def __init__(self, master, config: RuntimeConfig,
                 get_current_frame=None,
                 on_config_changed=None,
                 start_color_pick=None,
                 get_analysis_data=None,
                 on_recompute_attribution=None):
        """
        Args:
            master: 父 widget（tabview 的 tab frame）
            config: RuntimeConfig 實例（直接修改）
            get_current_frame: callback，回傳當前影片幀 (BGR np.ndarray) 或 None
            on_config_changed: callback，設定變更時呼叫
            start_color_pick: callback(mode, callback, finish_callback)，啟動取色模式
            get_analysis_data: callback，回傳當前幀分析資料 dict 或 None
            on_recompute_attribution: callback，重新計算歸因
        """
        super().__init__(master, fg_color="transparent")
        self.cfg = config
        self._get_frame = get_current_frame
        self._on_changed = on_config_changed
        self._start_color_pick_fn = start_color_pick
        self._get_analysis_data = get_analysis_data
        self._on_recompute = on_recompute_attribution

        # 即時預覽用
        self._preview_photo = None
        self._mask_photo = None
        self._preview_after_id = None
        self._ownership_preview_after_id = None
        self._shot_preview_after_id = None
        self._ownership_photo = None
        self._shot_photo = None

        # 所有 slider 追蹤：key -> (slider, val_label, is_float, float_scale)
        self._all_sliders: dict = {}

        # 自動校正取色點
        self._calibration_points: list[tuple[int, int]] = []
        self._calibrating = False

        self._build_ui()
        self._schedule_preview()

    # ══════════════════════════════════════════════════
    # UI 建構
    # ══════════════════════════════════════════════════

    def _build_ui(self):
        # ── 頂部：Preset 控制列 ──
        top_bar = ctk.CTkFrame(self, fg_color=COLORS["bg_secondary"],
                                corner_radius=8)
        top_bar.pack(fill=tk.X, padx=4, pady=(4, 2))

        ctk.CTkLabel(top_bar, text="Preset:",
                      text_color=COLORS["text"],
                      font=ctk.CTkFont(size=11)
                      ).pack(side=tk.LEFT, padx=(8, 2))

        self._preset_var = ctk.StringVar(value="（未選擇）")
        self._preset_menu = ctk.CTkOptionMenu(
            top_bar, width=120, height=26,
            variable=self._preset_var,
            values=self._get_preset_list(),
            fg_color=COLORS["bg_primary"],
            button_color=COLORS["border"],
            button_hover_color=COLORS["border_hover"],
            dropdown_fg_color=COLORS["bg_card"],
            dropdown_hover_color=COLORS["border"],
            text_color=COLORS["text"],
            font=ctk.CTkFont(size=11),
            command=self._on_preset_selected)
        self._preset_menu.pack(side=tk.LEFT, padx=2)

        ctk.CTkButton(
            top_bar, text="儲存", width=45, height=26,
            fg_color=COLORS["success"], hover_color=COLORS["success_hover"],
            text_color="white", corner_radius=6,
            font=ctk.CTkFont(size=11),
            command=self._save_preset
        ).pack(side=tk.LEFT, padx=2)

        ctk.CTkButton(
            top_bar, text="重置", width=45, height=26,
            fg_color=COLORS["border"], hover_color=COLORS["border_hover"],
            text_color=COLORS["text"], corner_radius=6,
            font=ctk.CTkFont(size=11),
            command=self._reset_to_defaults
        ).pack(side=tk.LEFT, padx=2)

        # ── 主體：子分頁 ──
        self._tabview = ctk.CTkTabview(
            self, fg_color=COLORS["bg_secondary"], corner_radius=8,
            segmented_button_fg_color=COLORS["border"],
            segmented_button_selected_color=COLORS["accent"],
            segmented_button_unselected_color=COLORS["bg_primary"],
            height=28)
        self._tabview.pack(fill=tk.BOTH, expand=True, padx=4, pady=(2, 4))

        self._build_hsv_tab()
        self._build_tracking_tab()
        self._build_scoring_tab()
        self._build_shot_tab()
        self._build_robot_tab()
        self._build_match_tab()

    def _build_hsv_tab(self):
        """球偵測 HSV Tab — 上方 slider + 下方即時預覽。"""
        tab = self._tabview.add("HSV")
        tab.rowconfigure(0, weight=1)
        tab.rowconfigure(1, weight=0)
        tab.columnconfigure(0, weight=1)

        # 上方：Sliders（可滾動）
        slider_frame = ctk.CTkScrollableFrame(tab, fg_color="transparent")
        slider_frame.grid(row=0, column=0, sticky="nsew")

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
            self._add_slider(
                slider_frame, key, label, lo, hi, self._on_hsv_changed)

        # 按鈕區
        btn_frame = ctk.CTkFrame(slider_frame, fg_color="transparent")
        btn_frame.pack(fill=tk.X, pady=(8, 2))

        ctk.CTkButton(
            btn_frame, text="點擊取色", height=28, corner_radius=6,
            fg_color=COLORS["info"], hover_color="#3b82f6",
            text_color="white",
            font=ctk.CTkFont(size=11),
            command=self._start_point_sample
        ).pack(side=tk.LEFT, padx=(0, 4))

        ctk.CTkButton(
            btn_frame, text="多點校正", height=28, corner_radius=6,
            fg_color=COLORS["purple"], hover_color="#9333ea",
            text_color="white",
            font=ctk.CTkFont(size=11),
            command=self._start_auto_calibrate
        ).pack(side=tk.LEFT, padx=4)

        # 校正狀態
        self._calibration_label = ctk.CTkLabel(
            slider_frame, text="",
            text_color=COLORS["text_secondary"],
            font=ctk.CTkFont(size=10))
        self._calibration_label.pack(anchor=tk.W, pady=(2, 0))

        # 下方：預覽區域（固定高度）
        preview_frame = ctk.CTkFrame(tab, fg_color="transparent", height=160)
        preview_frame.grid(row=1, column=0, sticky="ew", pady=(2, 0))
        preview_frame.grid_propagate(False)
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.columnconfigure(1, weight=1)
        preview_frame.rowconfigure(0, weight=1)

        # 原始影像 + Mask 並排
        self._preview_canvas = tk.Canvas(
            preview_frame, bg="black", highlightthickness=0, bd=0)
        self._preview_canvas.grid(row=0, column=0, sticky="nsew", padx=(0, 1))

        self._mask_canvas = tk.Canvas(
            preview_frame, bg="black", highlightthickness=0, bd=0)
        self._mask_canvas.grid(row=0, column=1, sticky="nsew", padx=(1, 0))

        # 偵測數量
        self._detect_count_label = ctk.CTkLabel(
            preview_frame, text="偵測: --",
            text_color=COLORS["accent"],
            font=ctk.CTkFont(size=11, weight="bold"))
        self._detect_count_label.grid(row=1, column=0, columnspan=2, pady=(2, 0))

    def _build_tracking_tab(self):
        """球追蹤 Tab。"""
        tab = self._tabview.add("追蹤")
        frame = ctk.CTkScrollableFrame(tab, fg_color="transparent")
        frame.pack(fill=tk.BOTH, expand=True)

        self._add_slider(frame, "max_match_dist", "最大匹配距離", 10, 500)
        self._add_slider(frame, "max_missed", "最大丟失幀數", 1, 60)
        self._add_slider(frame, "velocity_smooth", "速度平滑", 0, 100,
                          is_float=True, float_scale=100)
        self._add_slider(frame, "min_traj_len", "最少軌跡點", 1, 30)
        self._add_slider(frame, "area_weight", "面積權重", 0, 100,
                          is_float=True, float_scale=100)
        self._add_slider(frame, "area_scale", "面積縮放", 1, 1000)
        self._add_slider(frame, "max_area_ratio", "縫合面積比", 100, 500,
                          is_float=True, float_scale=100)
        self._add_slider(frame, "stitch_ambiguity_ratio", "縫合歧義比", 100, 500,
                          is_float=True, float_scale=100)

    def _build_scoring_tab(self):
        """進球判定 Tab — 上方 slider + 下方即時預覽。"""
        tab = self._tabview.add("進球")
        tab.rowconfigure(0, weight=1)
        tab.rowconfigure(1, weight=0)
        tab.columnconfigure(0, weight=1)

        # 上方：Sliders（可滾動）
        slider_frame = ctk.CTkScrollableFrame(tab, fg_color="transparent")
        slider_frame.grid(row=0, column=0, sticky="nsew")

        self._add_slider(slider_frame, "score_proximity_frames",
                         "射手回溯幀數", 1, 60, self._on_scoring_changed)
        self._add_slider(slider_frame, "score_max_shooter_dist",
                         "射手最大距離", 50, 1000, self._on_scoring_changed)
        self._add_slider(slider_frame, "score_zone_dwell_frames",
                         "區域停留幀數", 1, 30, self._on_scoring_changed)
        self._add_slider(slider_frame, "score_cooldown_frames",
                         "進球冷卻幀數", 1, 60, self._on_scoring_changed)
        self._add_slider(slider_frame, "ball_ownership_dist",
                         "球所有權距離", 50, 500, self._on_scoring_changed)

        # 重新計算按鈕
        ctk.CTkButton(
            slider_frame, text="重新計算歸因", height=28, corner_radius=6,
            fg_color=COLORS["accent"], hover_color=COLORS["accent_hover"],
            text_color="white",
            font=ctk.CTkFont(size=11),
            command=self._recompute_attribution
        ).pack(fill=tk.X, pady=(8, 2))

        # 下方：預覽區域（固定高度）
        preview_frame = ctk.CTkFrame(tab, fg_color="transparent", height=160)
        preview_frame.grid(row=1, column=0, sticky="ew", pady=(2, 0))
        preview_frame.grid_propagate(False)
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(0, weight=1)

        self._ownership_canvas = tk.Canvas(
            preview_frame, bg="black", highlightthickness=0, bd=0)
        self._ownership_canvas.grid(row=0, column=0, sticky="nsew")

        self._ownership_label = ctk.CTkLabel(
            preview_frame, text="所有權: --",
            text_color=COLORS["accent"],
            font=ctk.CTkFont(size=11, weight="bold"))
        self._ownership_label.grid(row=1, column=0, pady=(2, 0))

    def _build_shot_tab(self):
        """出手偵測 Tab — 上方 slider + 下方即時預覽。"""
        tab = self._tabview.add("出手")
        tab.rowconfigure(0, weight=1)
        tab.rowconfigure(1, weight=0)
        tab.columnconfigure(0, weight=1)

        # 上方：Sliders（可滾動）
        slider_frame = ctk.CTkScrollableFrame(tab, fg_color="transparent")
        slider_frame.grid(row=0, column=0, sticky="nsew")

        self._add_slider(slider_frame, "shot_min_velocity",
                         "最低速度 (px/幀)", 1, 100, self._on_shot_changed)
        self._add_slider(slider_frame, "shot_robot_proximity",
                         "機器人距離 (px)", 50, 500, self._on_shot_changed)
        self._add_slider(slider_frame, "shot_min_upward_velocity",
                         "最低上升速度", 1, 30, self._on_shot_changed)

        # 重新計算按鈕
        ctk.CTkButton(
            slider_frame, text="重新計算歸因", height=28, corner_radius=6,
            fg_color=COLORS["accent"], hover_color=COLORS["accent_hover"],
            text_color="white",
            font=ctk.CTkFont(size=11),
            command=self._recompute_attribution
        ).pack(fill=tk.X, pady=(8, 2))

        # 下方：預覽區域（固定高度）
        preview_frame = ctk.CTkFrame(tab, fg_color="transparent", height=160)
        preview_frame.grid(row=1, column=0, sticky="ew", pady=(2, 0))
        preview_frame.grid_propagate(False)
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(0, weight=1)

        self._shot_canvas = tk.Canvas(
            preview_frame, bg="black", highlightthickness=0, bd=0)
        self._shot_canvas.grid(row=0, column=0, sticky="nsew")

        self._shot_label = ctk.CTkLabel(
            preview_frame, text="速度: --",
            text_color=COLORS["accent"],
            font=ctk.CTkFont(size=11, weight="bold"))
        self._shot_label.grid(row=1, column=0, pady=(2, 0))

    def _build_robot_tab(self):
        """機器人追蹤 Tab。"""
        tab = self._tabview.add("機器人")
        frame = ctk.CTkScrollableFrame(tab, fg_color="transparent")
        frame.pack(fill=tk.BOTH, expand=True)

        # 追蹤器類型（下拉）
        type_frame = ctk.CTkFrame(frame, fg_color="transparent")
        type_frame.pack(fill=tk.X, pady=(4, 0))
        ctk.CTkLabel(type_frame, text="SOT 類型:",
                      text_color=COLORS["text"],
                      font=ctk.CTkFont(size=11)
                      ).pack(side=tk.LEFT)
        self._tracker_type_var = ctk.StringVar(
            value=self.cfg.robot_tracker_type)
        ctk.CTkOptionMenu(
            type_frame, width=80, height=26,
            values=["VIT", "CSRT", "KCF", "MIL"],
            variable=self._tracker_type_var,
            fg_color=COLORS["bg_primary"],
            button_color=COLORS["border"],
            text_color=COLORS["text"],
            font=ctk.CTkFont(size=11),
            command=lambda v: self._set_config("robot_tracker_type", v)
        ).pack(side=tk.LEFT, padx=(4, 0))

        self._add_slider(frame, "robot_max_lost_frames", "SOT 丟失幀數", 1, 120)
        self._add_slider(frame, "vittrack_score_threshold",
                          "VitTrack 閾值", 0, 100,
                          is_float=True, float_scale=100)
        self._add_slider(frame, "robot_detection_confidence",
                          "YOLO 信心度", 0, 100,
                          is_float=True, float_scale=100)
        self._add_slider(frame, "robot_detection_nms_iou",
                          "NMS IoU", 0, 100,
                          is_float=True, float_scale=100)

        # ByteTrack 參數
        ctk.CTkLabel(frame, text="── ByteTrack ──",
                      text_color=COLORS["accent"],
                      font=ctk.CTkFont(size=11, weight="bold")
                      ).pack(anchor=tk.W, pady=(8, 2))

        self._add_slider(frame, "bytetrack_track_thresh",
                          "追蹤閾值", 0, 100,
                          is_float=True, float_scale=100)
        self._add_slider(frame, "bytetrack_lost_buffer",
                          "丟失緩衝", 1, 300)
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
        det_frame.pack(fill=tk.X, pady=(8, 0))
        ctk.CTkLabel(det_frame, text="球偵測模式:",
                      text_color=COLORS["text"],
                      font=ctk.CTkFont(size=11)
                      ).pack(side=tk.LEFT)
        self._det_mode_var = ctk.StringVar(value=self.cfg.detection_mode)
        ctk.CTkOptionMenu(
            det_frame, width=70, height=26,
            values=["HSV", "AI"],
            variable=self._det_mode_var,
            fg_color=COLORS["bg_primary"],
            button_color=COLORS["border"],
            text_color=COLORS["text"],
            font=ctk.CTkFont(size=11),
            command=lambda v: self._set_config("detection_mode", v)
        ).pack(side=tk.LEFT, padx=(4, 0))

    # ══════════════════════════════════════════════════
    # Slider 工具
    # ══════════════════════════════════════════════════

    def _add_slider(self, parent, key: str, label: str,
                     lo: int, hi: int, callback=None,
                     is_float=False, float_scale=1):
        """新增一個 slider + 數值顯示。"""
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill=tk.X, pady=(2, 0))

        value = getattr(self.cfg, key)
        if is_float:
            display_val = value
            slider_val = int(value * float_scale)
        else:
            display_val = value
            slider_val = value

        name_label = ctk.CTkLabel(
            row, text=f"{label}:", width=120, anchor=tk.W,
            text_color=COLORS["text"],
            font=ctk.CTkFont(size=11))
        name_label.pack(side=tk.LEFT)

        val_label = ctk.CTkLabel(
            row, text=str(display_val), width=50, anchor=tk.E,
            text_color=COLORS["accent"],
            font=ctk.CTkFont(size=11, weight="bold"))
        val_label.pack(side=tk.RIGHT, padx=(2, 0))

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
        slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 2))

        self._all_sliders[key] = (slider, val_label, is_float, float_scale)
        return (slider, val_label, is_float, float_scale)

    # ══════════════════════════════════════════════════
    # Config 操作
    # ══════════════════════════════════════════════════

    def _set_config(self, key: str, value):
        """設定 RuntimeConfig 的一個欄位。"""
        setattr(self.cfg, key, value)
        if self._on_changed:
            self._on_changed()

    def _reload_all_sliders(self):
        """從 config 原地更新所有 slider 值（不銷毀 UI）。"""
        for key, (slider, val_label, is_float, float_scale) in \
                self._all_sliders.items():
            value = getattr(self.cfg, key)
            if is_float:
                slider.set(int(value * float_scale))
                val_label.configure(text=f"{value:.2f}")
            else:
                slider.set(value)
                val_label.configure(text=str(value))
        # 同步下拉選單
        if hasattr(self, '_tracker_type_var'):
            self._tracker_type_var.set(self.cfg.robot_tracker_type)
        if hasattr(self, '_det_mode_var'):
            self._det_mode_var.set(self.cfg.detection_mode)
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
            for f in self.cfg.__dataclass_fields__:
                setattr(self.cfg, f, getattr(loaded, f))
            self._reload_all_sliders()
            if self._on_changed:
                self._on_changed()
        except Exception as e:
            messagebox.showerror("載入失敗", f"無法載入 preset '{name}':\n{e}",
                                 parent=self.winfo_toplevel())

    def _save_preset(self):
        name = simpledialog.askstring(
            "儲存 Preset",
            "請輸入 Preset 名稱（如「區域賽-台中」）:",
            parent=self.winfo_toplevel())
        if not name:
            return
        name = name.strip()
        try:
            self.cfg.save_preset(name)
            self._refresh_preset_menu()
            self._preset_var.set(name)
            self._calibration_label.configure(
                text=f"已儲存: {name}",
                text_color=COLORS["success"])
        except Exception as e:
            messagebox.showerror("儲存失敗", str(e),
                                 parent=self.winfo_toplevel())

    def _reset_to_defaults(self):
        default = RuntimeConfig()
        for f in self.cfg.__dataclass_fields__:
            setattr(self.cfg, f, getattr(default, f))
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
        self._schedule_preview()

    def _schedule_preview(self):
        if self._preview_after_id is not None:
            self.after_cancel(self._preview_after_id)
        self._preview_after_id = self.after(100, self._update_preview)
        # 幀切換時也更新進球/出手預覽
        self._schedule_ownership_preview()
        self._schedule_shot_preview()

    def _update_preview(self):
        self._preview_after_id = None
        if not self._get_frame:
            return
        frame = self._get_frame()
        if frame is None:
            return

        hsv_low = self.cfg.hsv_low
        hsv_high = self.cfg.hsv_high
        min_a = self.cfg.min_blob_area
        max_a = self.cfg.max_blob_area

        mask_colored, ball_count, balls = generate_preview(
            frame, hsv_low, hsv_high, min_a, max_a)

        preview_frame = frame.copy()
        for cx, cy, r in balls:
            cv2.circle(preview_frame, (cx, cy), r,
                       (0, 255, 255), 2, cv2.LINE_AA)

        self._show_on_canvas(self._preview_canvas, preview_frame)
        self._show_on_canvas(self._mask_canvas, mask_colored)
        self._detect_count_label.configure(text=f"偵測: {ball_count} 個球")

    def _show_on_canvas(self, canvas, bgr_image, store_attr=None):
        cw = canvas.winfo_width()
        ch = canvas.winfo_height()
        if cw < 10 or ch < 10:
            cw, ch = 200, 120

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

        if store_attr:
            setattr(self, store_attr, photo)
        elif canvas is self._preview_canvas:
            self._preview_photo = photo
        else:
            self._mask_photo = photo

    # ══════════════════════════════════════════════════
    # 點擊取色
    # ══════════════════════════════════════════════════

    def _start_point_sample(self):
        if not self._get_frame or self._get_frame() is None:
            self._calibration_label.configure(
                text="請先開啟影片",
                text_color=COLORS["error"])
            return
        self._calibrating = False
        self._calibration_label.configure(
            text="請在影片上點擊球的位置...",
            text_color=COLORS["info"])
        if self._start_color_pick_fn:
            self._start_color_pick_fn(
                mode="single", callback=self._on_point_sampled)

    def _on_point_sampled(self, px: int, py: int):
        frame = self._get_frame()
        if frame is None:
            return
        hsv_low, hsv_high = sample_hsv_at_point(frame, px, py)
        self.cfg.hsv_h_low = hsv_low[0]
        self.cfg.hsv_s_low = hsv_low[1]
        self.cfg.hsv_v_low = hsv_low[2]
        self.cfg.hsv_h_high = hsv_high[0]
        self.cfg.hsv_s_high = hsv_high[1]
        self.cfg.hsv_v_high = hsv_high[2]
        self._reload_all_sliders()
        self._calibration_label.configure(
            text=f"H({hsv_low[0]}-{hsv_high[0]}) "
                 f"S({hsv_low[1]}-{hsv_high[1]}) "
                 f"V({hsv_low[2]}-{hsv_high[2]})",
            text_color=COLORS["success"])
        if self._on_changed:
            self._on_changed()

    # ══════════════════════════════════════════════════
    # 自動校正（多點 K-Means）
    # ══════════════════════════════════════════════════

    def _start_auto_calibrate(self):
        if not self._get_frame or self._get_frame() is None:
            self._calibration_label.configure(
                text="請先開啟影片",
                text_color=COLORS["error"])
            return
        self._calibrating = True
        self._calibration_points = []
        self._calibration_label.configure(
            text="點擊 3-5 顆球，右鍵完成",
            text_color=COLORS["purple"])
        if self._start_color_pick_fn:
            self._start_color_pick_fn(
                mode="multi",
                callback=self._on_calibration_point,
                finish_callback=self._finish_auto_calibrate)

    def _on_calibration_point(self, px: int, py: int):
        self._calibration_points.append((px, py))
        n = len(self._calibration_points)
        self._calibration_label.configure(
            text=f"已取 {n} 點 — 右鍵完成",
            text_color=COLORS["purple"])

    def _finish_auto_calibrate(self):
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

            self.cfg.hsv_h_low = hsv_low[0]
            self.cfg.hsv_s_low = hsv_low[1]
            self.cfg.hsv_v_low = hsv_low[2]
            self.cfg.hsv_h_high = hsv_high[0]
            self.cfg.hsv_s_high = hsv_high[1]
            self.cfg.hsv_v_high = hsv_high[2]

            self._reload_all_sliders()
            n = len(self._calibration_points)
            self._calibration_label.configure(
                text=f"校正完成 ({n}點) "
                     f"H({hsv_low[0]}-{hsv_high[0]}) "
                     f"S({hsv_low[1]}-{hsv_high[1]})",
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
    # 進球 / 出手 預覽
    # ══════════════════════════════════════════════════

    def _on_scoring_changed(self):
        self._schedule_ownership_preview()

    def _on_shot_changed(self):
        self._schedule_shot_preview()

    def _schedule_ownership_preview(self):
        if self._ownership_preview_after_id is not None:
            self.after_cancel(self._ownership_preview_after_id)
        self._ownership_preview_after_id = self.after(
            100, self._update_ownership_preview)

    def _schedule_shot_preview(self):
        if self._shot_preview_after_id is not None:
            self.after_cancel(self._shot_preview_after_id)
        self._shot_preview_after_id = self.after(
            100, self._update_shot_preview)

    def _update_ownership_preview(self):
        """球所有權即時預覽 — 顯示 ownership_dist 圓圈和球-機器人連線。"""
        self._ownership_preview_after_id = None
        if not self._get_frame:
            return
        frame = self._get_frame()
        if frame is None:
            return

        preview = frame.copy()
        h, w = preview.shape[:2]
        ownership_dist = self.cfg.ball_ownership_dist

        data = self._get_analysis_data() if self._get_analysis_data else None
        info_text = "請先完成分析"

        if data:
            robot_pos = data.get("robot_positions", {})
            ball_dets = data.get("ball_detections", [])
            ball_ownership = data.get("ball_ownership", {})
            frame_idx = data.get("frame_idx", 0)

            # 畫機器人 ownership 圓圈
            for label, pos in robot_pos.items():
                rx, ry = int(pos[0]), int(pos[1])
                color = (0, 200, 200)  # 青色
                cv2.circle(preview, (rx, ry), int(ownership_dist),
                           color, 1, cv2.LINE_AA)
                cv2.circle(preview, (rx, ry), 5, color, -1, cv2.LINE_AA)
                cv2.putText(preview, label, (rx + 8, ry - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            color, 1, cv2.LINE_AA)

            # 畫球到最近機器人的連線
            closest_info = None
            for det in ball_dets:
                bx, by = int(det[0]), int(det[1])
                cv2.circle(preview, (bx, by), 6, (0, 255, 255), 2,
                           cv2.LINE_AA)

                best_label = None
                best_dist = float('inf')
                best_pos = None
                for label, pos in robot_pos.items():
                    d = ((bx - pos[0])**2 + (by - pos[1])**2) ** 0.5
                    if d < best_dist:
                        best_dist = d
                        best_label = label
                        best_pos = (int(pos[0]), int(pos[1]))

                if best_pos:
                    in_range = best_dist <= ownership_dist
                    line_color = (0, 255, 0) if in_range else (0, 0, 255)
                    cv2.line(preview, (bx, by), best_pos,
                             line_color, 2, cv2.LINE_AA)
                    mid_x = (bx + best_pos[0]) // 2
                    mid_y = (by + best_pos[1]) // 2
                    cv2.putText(preview, f"{best_dist:.0f}px",
                                (mid_x, mid_y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                line_color, 1, cv2.LINE_AA)
                    if closest_info is None or best_dist < closest_info[1]:
                        closest_info = (best_label, best_dist, in_range)

            if closest_info:
                lbl, d, ok = closest_info
                status = "範圍內" if ok else "超出"
                info_text = f"所有權: {lbl} (距離: {d:.0f}px, {status})"
            elif robot_pos:
                info_text = f"閾值: {ownership_dist}px（無球偵測）"
            else:
                info_text = f"閾值: {ownership_dist}px（無機器人）"
        else:
            # 無分析資料 — 只畫參數說明
            cv2.putText(preview, f"ownership_dist = {ownership_dist}px",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 200, 200), 1, cv2.LINE_AA)
            cv2.circle(preview, (w // 2, h // 2), ownership_dist,
                       (0, 200, 200), 1, cv2.LINE_AA)

        self._show_on_canvas(self._ownership_canvas, preview,
                             store_attr="_ownership_photo")
        self._ownership_label.configure(text=info_text)

    def _update_shot_preview(self):
        """出手偵測即時預覽 — 顯示 proximity 圓圈和速度箭頭。"""
        self._shot_preview_after_id = None
        if not self._get_frame:
            return
        frame = self._get_frame()
        if frame is None:
            return

        preview = frame.copy()
        h, w = preview.shape[:2]
        proximity = self.cfg.shot_robot_proximity
        min_vel = self.cfg.shot_min_velocity
        min_upward = self.cfg.shot_min_upward_velocity

        data = self._get_analysis_data() if self._get_analysis_data else None
        info_text = "請先完成分析"

        if data:
            robot_pos = data.get("robot_positions", {})
            trajectories = data.get("trajectories", {})
            frame_idx = data.get("frame_idx", 0)

            # 畫機器人 proximity 圓圈
            for label, pos in robot_pos.items():
                rx, ry = int(pos[0]), int(pos[1])
                color = (200, 100, 0)  # 橘藍
                cv2.circle(preview, (rx, ry), int(proximity),
                           color, 1, cv2.LINE_AA)
                cv2.circle(preview, (rx, ry), 5, color, -1, cv2.LINE_AA)
                cv2.putText(preview, label, (rx + 8, ry - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            color, 1, cv2.LINE_AA)

            # 畫球速度箭頭
            max_vel_shown = 0.0
            for tid, traj in trajectories.items():
                traj_sorted = sorted(traj, key=lambda p: p[0])
                for i in range(1, len(traj_sorted)):
                    f0, x0, y0, *_ = traj_sorted[i - 1]
                    f1, x1, y1, *_ = traj_sorted[i]
                    if not (f0 <= frame_idx <= f1):
                        continue
                    dt = f1 - f0
                    if dt <= 0:
                        continue
                    vx = (x1 - x0) / dt
                    vy = (y1 - y0) / dt
                    vel = (vx**2 + vy**2) ** 0.5
                    max_vel_shown = max(max_vel_shown, vel)

                    bx, by = int(x0), int(y0)
                    # 箭頭長度 = 速度 × 3（視覺放大）
                    arrow_scale = 3.0
                    ex = int(bx + vx * arrow_scale)
                    ey = int(by + vy * arrow_scale)
                    exceeds = vel >= min_vel and vy <= -min_upward
                    arrow_color = (0, 0, 255) if exceeds else (128, 128, 128)
                    cv2.arrowedLine(preview, (bx, by), (ex, ey),
                                    arrow_color, 2, cv2.LINE_AA,
                                    tipLength=0.3)
                    cv2.putText(preview, f"{vel:.1f}",
                                (bx + 10, by - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                                arrow_color, 1, cv2.LINE_AA)

            info_text = (f"速度: {max_vel_shown:.1f} px/幀 "
                         f"(閾值: {min_vel}, 上升: {min_upward})")
        else:
            cv2.putText(preview, f"min_vel={min_vel}, proximity={proximity}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (200, 100, 0), 1, cv2.LINE_AA)

        self._show_on_canvas(self._shot_canvas, preview,
                             store_attr="_shot_photo")
        self._shot_label.configure(text=info_text)

    def _recompute_attribution(self):
        """重新計算歸因（呼叫 app.py callback）。"""
        if self._on_recompute:
            self._on_recompute()
            # 重新整理預覽
            self._schedule_ownership_preview()
            self._schedule_shot_preview()

    # ══════════════════════════════════════════════════
    # 清理
    # ══════════════════════════════════════════════════

    def destroy(self):
        if self._preview_after_id is not None:
            self.after_cancel(self._preview_after_id)
        if self._ownership_preview_after_id is not None:
            self.after_cancel(self._ownership_preview_after_id)
        if self._shot_preview_after_id is not None:
            self.after_cancel(self._shot_preview_after_id)
        super().destroy()

"""
FRC Scoring Analyzer — 常數與設定
"""

import customtkinter as ctk

# ── CustomTkinter 全域設定 ────────────────────────────
ctk.set_appearance_mode("dark")

try:
    from scipy.optimize import linear_sum_assignment
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ── UI 配色 ───────────────────────────────────────────
COLORS = {
    "bg_primary": "#1a1a2e",
    "bg_secondary": "#16213e",
    "bg_card": "#1f2937",
    "accent": "#f59e0b",
    "accent_hover": "#d97706",
    "text": "#f3f4f6",
    "text_secondary": "#9ca3af",
    "success": "#22c55e",
    "success_hover": "#16a34a",
    "error": "#ef4444",
    "error_hover": "#dc2626",
    "border": "#374151",
    "border_hover": "#4b5563",
    "info": "#60a5fa",
    "purple": "#a855f7",
}

# ── 球偵測參數 ────────────────────────────────────────
YELLOW_LOW = (20, 100, 100)
YELLOW_HIGH = (35, 255, 255)
MIN_BLOB_AREA = 150
MAX_BLOB_AREA = 50000

# ── 球追蹤參數 ────────────────────────────────────────
MAX_MATCH_DIST = 200
MAX_MISSED = 10
VELOCITY_SMOOTH = 0.7        # 速度 EMA 平滑因子
AREA_WEIGHT = 0.3            # 面積差異權重
AREA_SCALE = 100             # 面積差異縮放
MAX_AREA_RATIO = 1.5         # 縫合最大面積比
VELOCITY_POINTS = 4          # 縫合速度計算點數
STITCH_AMBIGUITY_RATIO = 2.0 # 縫合歧義比率
MIN_TRAJ_LEN = 5             # 最少軌跡點數

# ── 機器人追蹤參數 ────────────────────────────────────
ROBOT_TRACKER_TYPE = "VIT"   # 追蹤器類型（VIT → CSRT → KCF → MIL fallback）
ROBOT_MAX_LOST_FRAMES = 30   # 最大丟失幀數
MAX_ROBOTS = 6               # 最多追蹤機器人數

# ── AI 偵測模式 ─────────────────────────────────────
DETECTION_MODE = "HSV"              # "HSV" or "AI"（預設 HSV，AI 模式需訓練球模型）
AI_CONFIDENCE_THRESHOLD = 0.25     # ONNX 本地推理：小數（0.25=25%）
AI_MIN_AREA = 50
AI_MAX_AREA = 200000

# ── VitTrack 機器人追蹤（SOT fallback）─────────────
VITTRACK_MODEL_PATH = "models/object_tracking_vittrack_2023sep.onnx"
VITTRACK_SCORE_THRESHOLD = 0.3

# ── 機器人 YOLO 偵測（MOT 主要模式）──────────────────
ROBOT_DETECTION_MODEL_PATH = "models/frc_robot.onnx"
ROBOT_DETECTION_CONFIDENCE = 0.4
ROBOT_DETECTION_NMS_IOU = 0.45

# ── ByteTrack 多目標追蹤參數 ──────────────────────────
BYTETRACK_TRACK_THRESH = 0.3       # 追蹤啟動閾值
BYTETRACK_LOST_BUFFER = 120        # 丟失緩衝（幀數，4秒@30fps）
BYTETRACK_MATCH_THRESH = 0.8       # 最小匹配閾值
BYTETRACK_MIN_CONSECUTIVE = 3      # 最少連續幀確認新追蹤

# ── 進球判定參數 ──────────────────────────────────────
SCORE_PROXIMITY_FRAMES = 15  # 回溯射手歸因的幀數
SCORE_MAX_SHOOTER_DIST = 300 # 射手歸因最大距離（像素）
SCORE_ZONE_DWELL_FRAMES = 3  # 球在區域內停留幀數才算進球
SCORE_COOLDOWN_FRAMES = 10   # 同一球連續進球冷卻幀數

# ── 出手偵測參數 ──────────────────────────────────────
SHOT_MIN_VELOCITY = 15       # 出手最低速度（像素/幀）
SHOT_ROBOT_PROXIMITY = 150   # 出手時球距機器人最大距離（像素）

# ── 比賽時間設定（預設值）────────────────────────────
AUTO_DURATION_SEC = 15       # 自主期秒數
TELEOP_START_SEC = 15        # 操控期開始秒數（= AUTO 結束後）

# ── 軌跡顏色（最多 6 台機器人）────────────────────────
ROBOT_COLORS = [
    (255, 0, 0),     # 紅
    (0, 128, 255),   # 藍
    (0, 200, 0),     # 綠
    (255, 165, 0),   # 橙
    (200, 0, 200),   # 紫
    (0, 200, 200),   # 青
]

ROBOT_COLORS_HEX = [
    "#ff0000",
    "#0080ff",
    "#00c800",
    "#ffa500",
    "#c800c8",
    "#00c8c8",
]

# ── 聯盟配色 ─────────────────────────────────────────
ALLIANCE_COLORS = {
    "red": {
        "bgr": (0, 0, 220),
        "rgb": (220, 0, 0),
        "hex": "#dc0000",
    },
    "blue": {
        "bgr": (220, 100, 0),
        "rgb": (0, 100, 220),
        "hex": "#0064dc",
    },
}

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
MIN_BLOB_AREA = 50
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

# ── 機器人偵測模式 ──────────────────────────────────
ROBOT_DETECTION_MODE = "YOLO"       # "HSV" / "YOLO" / "GEMINI"（Gemini API 零樣本偵測）

# ── Gemini API 機器人偵測 ─────────────────────────────
GEMINI_MODEL = "gemini-3.1-flash-lite-preview"  # 速度優先（~3s/張）
GEMINI_DETECT_STRATEGY = "adaptive"  # every_2 | every_n | adaptive | first_only
GEMINI_DETECT_INTERVAL = 10         # every_n 模式的間隔幀數
GEMINI_MAX_IMAGE_SIZE = 1280        # 送 API 的最大影像邊長（縮小省流量）

# ── 機器人 YOLO 偵測（備選模式）──────────────────────
ROBOT_DETECTION_MODEL_PATH = "models/frc_robot.onnx"
ROBOT_DETECTION_CONFIDENCE = 0.27   # 實測調整（0.18 仍有少量誤偵測）
ROBOT_DETECTION_NMS_IOU = 0.6       # 提高以保留靠近機器人的獨立偵測（0.45→0.6）

# ── HSV Bumper 偵測（無需訓練資料的機器人偵測）──────────
# 紅色 bumper（色相環繞：0~10 和 170~180 兩段）
BUMPER_RED_LOW_1 = (0, 120, 80)
BUMPER_RED_HIGH_1 = (10, 255, 255)
BUMPER_RED_LOW_2 = (170, 120, 80)
BUMPER_RED_HIGH_2 = (180, 255, 255)
# 藍色 bumper
BUMPER_BLUE_LOW = (100, 120, 60)
BUMPER_BLUE_HIGH = (130, 255, 255)
# 面積過濾（像素²）
BUMPER_MIN_AREA = 300       # 太小 = 噪點（4K 下 bumper 最小約 500px²）
BUMPER_MAX_AREA = 80000     # 太大 = 誤偵測（4K 下 bumper 最大約 50000px²）
# 長寬比過濾（bumper 是扁長方形）
BUMPER_MIN_ASPECT = 0.5     # 最小長寬比（寬/高），允許直立視角的 bumper
BUMPER_MAX_ASPECT = 12.0    # 最大長寬比，避免偵測到長條噪點
# NMS IoU（合併重疊偵測）
BUMPER_NMS_IOU = 0.4
# Bumper 模板比對相似度閾值（0~1，越高越嚴格）
BUMPER_TEMPLATE_SIMILARITY = 0.3

# ── ByteTrack 多目標追蹤參數 ──────────────────────────
BYTETRACK_TRACK_THRESH = 0.10      # 追蹤啟動閾值（必須 <= ROBOT_DETECTION_CONFIDENCE）
BYTETRACK_LOST_BUFFER = 120        # 丟失緩衝（幀數，4秒@30fps）
BYTETRACK_MATCH_THRESH = 0.3       # IoU 匹配閾值（0.8→0.3: FRC 機器人移動快，需要寬鬆匹配）
BYTETRACK_MIN_CONSECUTIVE = 1      # 最少連續幀（3→1: 加速新追蹤建立）

# ── MOT 進階參數 ────────────────────────────────────
MOT_DETECT_INTERVAL = 10           # 每 N 幀做 tiled 偵測（全幀偵測每幀都做）
MOT_REID_MAX_DIST = 400            # 重新辨識基準距離（像素，會依幀間隔動態縮放）
MOT_HISTOGRAM_WEIGHT = 0.4         # 顏色直方圖 Re-ID 權重（0=純距離, 1=全靠外觀）
MOT_HISTOGRAM_UPDATE_INTERVAL = 3  # 直方圖更新頻率（每 N 幀更新一次，降低計算開銷）
MOT_MIN_TRACK_FRAMES = 15          # 最少偵測幀數（少於此值的 label 在後處理中被移除，30fps 下 ≈ 0.5 秒）
MOT_STATIC_MAX_VARIANCE = 100      # 靜止判定：位置變異數閾值（px²，標準差 <10px 視為靜止）
MOT_STATIC_MIN_FRAMES = 30         # 靜止判定：至少追蹤 N 幀才判定（30fps 下 1 秒，避免誤判短暫停留）
MOT_MERGE_MAX_OVERLAP = 15         # 合併最大重疊幀數（30fps 下 ≈ 500ms）
MOT_MERGE_BOUNDARY_DIST = 800      # 合併邊界距離閾值（像素）
MOT_MERGE_SEARCH_WINDOW = 180      # 合併邊界搜尋窗口（幀數，30fps 下 ≈ 6 秒）
# ── MOT 遮擋處理 ────────────────────────────────────
MOT_MAX_LOST_FRAMES = 90           # LOST 最大容忍幀數（30fps 下 3 秒）
MOT_OCCLUSION_PATIENCE = 450       # 遮擋區域 LOST 耐心（30fps 下 15 秒）
MOT_LOST_GRACE_FRAMES = 3          # 連續未偵測幀數閾值，超過後 ACTIVE → LOST
MOT_LOST_REID_DIST_SCALE = 0.5     # LOST 復活距離閾值倍率（更嚴格）
MOT_LOST_MIN_HIST_SIM = 0.3        # LOST 復活最低直方圖相似度
MOT_OCCLUSION_MARGIN = 50          # 遮擋區域邊緣緩衝（像素）
MOT_DEDUP_IOU = 0.5                # 重複追蹤清除：同幀兩軌跡 bbox IoU 超過此值視為重複
MOT_ACTIVE_MAX_DIST = 200          # ACTIVE 匹配最大距離（像素，防止遠距離搶 ID）

# ── 進球判定參數 ──────────────────────────────────────
SCORE_PROXIMITY_FRAMES = 80  # 回溯射手歸因的幀數（80幀 ≈ 2.7秒 @30fps）
SCORE_MAX_SHOOTER_DIST = 500 # 射手歸因最大距離（像素，4K 下需 500+）
SCORE_HP_PROXIMITY = 150     # HP 歸因距離容忍（像素，球軌跡任一點距 HP 線段的最大距離）
SCORE_HP_ORIGIN_DIST = 400   # HP 歸因起源距離（像素，球軌跡最早點距 HP 線段的最大距離，更寬鬆）
SCORE_EXTRAPOLATE_FRAMES = 20 # 軌跡外推幀數（球首次偵測已在飛行時，用初始速度外推估計發射位置）
SCORE_DIRECTION_WEIGHT = 0.3  # 方向對齊權重（0~1，球飛離射手方向的距離加權修正）
SCORE_MIN_VELOCITY_FOR_DIR = 5 # 方向評分最低球速（像素/幀，低速球方向不穩不計算方向修正）
SCORE_ALLIANCE_BIAS = 0.85    # 聯盟一致性偏好（0~1，同聯盟射手分數乘此值，越低偏好越強）
SCORE_TRACE_MAX_DT = 3        # 軌跡追溯最大可信幀間隔（超過此值的速度估算不可靠，可能是球被遮擋）
SCORE_ZONE_DWELL_FRAMES = 1  # 球進入區域即算進球
SCORE_COOLDOWN_FRAMES = 10   # 同一球連續進球冷卻幀數
BALL_OWNERSHIP_DIST = 200    # 球所有權：球距機器人多近才算「持有」（像素，4K 下約機器人 2-3 倍半徑）

# ── 出手偵測參數 ──────────────────────────────────────
SHOT_MIN_VELOCITY = 15       # 出手最低速度（像素/幀）
SHOT_MIN_UPWARD_VELOCITY = 3 # 球垂直上升最低速度（dy<0 時的絕對值，像素/幀）
SHOT_ROBOT_PROXIMITY = 150   # 出手時球距機器人最大距離（像素）

# ── 背景模型參數 ──────────────────────────────────────
BG_SAMPLE_COUNT = 50         # 取樣幀數（均勻取樣用於建立背景中位數）
BG_FG_THRESHOLD = 30         # 前景閾值 (0-255)，absdiff 大於此值視為前景
BG_DILATE_KERNEL = 5         # 膨脹核大小，填補前景遮罩中的小空洞

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

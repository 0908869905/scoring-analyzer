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
    SHOT_MIN_VELOCITY, SHOT_MIN_UPWARD_VELOCITY, SHOT_ROBOT_PROXIMITY,
    BALL_OWNERSHIP_DIST,
    ROBOT_TRACKER_TYPE, ROBOT_MAX_LOST_FRAMES,
    ROBOT_DETECTION_CONFIDENCE, ROBOT_DETECTION_NMS_IOU,
    BYTETRACK_TRACK_THRESH, BYTETRACK_LOST_BUFFER,
    BYTETRACK_MATCH_THRESH, BYTETRACK_MIN_CONSECUTIVE,
    AUTO_DURATION_SEC, TELEOP_START_SEC,
    VITTRACK_SCORE_THRESHOLD,
    AI_CONFIDENCE_THRESHOLD, AI_MIN_AREA, AI_MAX_AREA,
    DETECTION_MODE,
    BG_SAMPLE_COUNT, BG_FG_THRESHOLD, BG_DILATE_KERNEL,
    BUMPER_TEMPLATE_SIMILARITY,
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

    # ── 進球判定（進階）──
    ball_ownership_dist: int = BALL_OWNERSHIP_DIST

    # ── 出手偵測 ──
    shot_min_velocity: int = SHOT_MIN_VELOCITY
    shot_min_upward_velocity: int = SHOT_MIN_UPWARD_VELOCITY
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

    # ── Bumper 模板 ──
    bumper_template_similarity: float = BUMPER_TEMPLATE_SIMILARITY

    # ── 背景模型 ──
    bg_sample_count: int = BG_SAMPLE_COUNT
    bg_fg_threshold: int = BG_FG_THRESHOLD
    bg_dilate_kernel: int = BG_DILATE_KERNEL

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
        safe_name = "".join(
            c for c in name
            if c.isalnum() or c in ('-', '_', ' ', '\u4e00-\u9fff')
            or '\u4e00' <= c <= '\u9fff'
            or '\u3400' <= c <= '\u4dbf'
        ).strip()
        if not safe_name:
            raise ValueError("Preset 名稱必須包含有效字元")
        os.makedirs(PRESETS_DIR, exist_ok=True)
        path = os.path.join(PRESETS_DIR, f"{safe_name}.json")
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

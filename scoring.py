"""
FRC Scoring Analyzer — 進球判定引擎（區域進入 + 射手歸因）
"""

from dataclasses import dataclass, field

from config import (
    SCORE_PROXIMITY_FRAMES, SCORE_MAX_SHOOTER_DIST,
    SCORE_ZONE_DWELL_FRAMES, SCORE_COOLDOWN_FRAMES,
    AUTO_DURATION_SEC, TELEOP_START_SEC,
)
from geometry import point_in_rect, distance


@dataclass
class ScoringZone:
    """得分區域定義。"""
    name: str       # "Upper HUB" 或 "Lower HUB"
    x: int
    y: int
    w: int
    h: int

    def contains(self, px, py):
        return point_in_rect(px, py, self.x, self.y, self.w, self.h)

    @property
    def center(self):
        return (self.x + self.w / 2, self.y + self.h / 2)

    def as_tuple(self):
        return (self.x, self.y, self.w, self.h)


@dataclass
class ScoreEvent:
    """進球事件。"""
    frame_idx: int
    ball_track_id: int
    zone_name: str       # "Upper" 或 "Lower"
    shooter_label: str   # 機器人標籤，或 "未知"
    shooter_dist: float  # 射手距離
    period: str          # "Auto" 或 "Teleop"


@dataclass
class RobotScore:
    """單一機器人的得分統計。"""
    label: str
    auto_upper: int = 0
    auto_lower: int = 0
    teleop_upper: int = 0
    teleop_lower: int = 0

    @property
    def auto_total(self):
        return self.auto_upper + self.auto_lower

    @property
    def teleop_total(self):
        return self.teleop_upper + self.teleop_lower

    @property
    def upper_total(self):
        return self.auto_upper + self.teleop_upper

    @property
    def lower_total(self):
        return self.auto_lower + self.teleop_lower

    @property
    def total(self):
        return self.auto_upper + self.auto_lower + self.teleop_upper + self.teleop_lower


class ScoringEngine:
    """進球判定引擎。"""

    def __init__(self, fps=30.0, auto_sec=AUTO_DURATION_SEC,
                 teleop_start_sec=TELEOP_START_SEC):
        self.fps = fps
        self.auto_end_frame = int(auto_sec * fps)
        self.teleop_start_frame = int(teleop_start_sec * fps)

        self.zones = []          # List[ScoringZone]
        self.events = []         # List[ScoreEvent]
        self.robot_scores = {}   # label -> RobotScore

        # 內部狀態：追蹤每顆球的區域停留
        self._ball_zone_frames = {}  # (track_id, zone_name) -> consecutive_frames
        self._ball_in_zone = {}      # track_id -> set of zone_names currently in
        self._ball_cooldown = {}     # track_id -> cooldown_remaining

    def add_zone(self, name, x, y, w, h):
        """新增得分區域。"""
        self.zones.append(ScoringZone(name, x, y, w, h))

    def clear_zones(self):
        """清除所有得分區域。"""
        self.zones.clear()

    def set_zones(self, zones):
        """設定得分區域列表。"""
        self.zones = list(zones)

    def reset(self):
        """重置所有分析結果。"""
        self.events.clear()
        self.robot_scores.clear()
        self._ball_zone_frames.clear()
        self._ball_in_zone.clear()
        self._ball_cooldown.clear()

    def get_period(self, frame_idx):
        """判斷幀所在的比賽期間。"""
        if frame_idx < self.auto_end_frame:
            return "Auto"
        return "Teleop"

    def process_frame(self, frame_idx, ball_positions, robot_positions,
                      ball_trajectories=None):
        """
        處理一幀的進球判定。

        Args:
            frame_idx: 幀索引
            ball_positions: {track_id: (cx, cy, area)} 當前幀的球位置
            robot_positions: {label: (cx, cy)} 當前幀的機器人位置
            ball_trajectories: {track_id: [(f, cx, cy, area), ...]} 完整軌跡（用於回溯）
        """
        if not self.zones:
            return

        # 更新冷卻計時
        to_remove = []
        for tid in self._ball_cooldown:
            self._ball_cooldown[tid] -= 1
            if self._ball_cooldown[tid] <= 0:
                to_remove.append(tid)
        for tid in to_remove:
            del self._ball_cooldown[tid]

        for tid, (cx, cy, *_rest) in ball_positions.items():
            # 跳過冷卻中的球
            if tid in self._ball_cooldown:
                continue

            prev_zones = self._ball_in_zone.get(tid, set())
            curr_zones = set()

            for zone in self.zones:
                if zone.contains(cx, cy):
                    curr_zones.add(zone.name)

                    key = (tid, zone.name)
                    self._ball_zone_frames[key] = \
                        self._ball_zone_frames.get(key, 0) + 1

                    # 球剛進入得分區域（之前不在此區域）
                    if zone.name not in prev_zones:
                        self._ball_zone_frames[key] = 1

                    # 停留夠久 → 判定進球
                    if self._ball_zone_frames[key] == SCORE_ZONE_DWELL_FRAMES:
                        shooter, dist = self._find_shooter(
                            tid, frame_idx, robot_positions,
                            ball_trajectories
                        )
                        period = self.get_period(frame_idx)
                        zone_type = "Upper" if "upper" in zone.name.lower() \
                            else "Lower"

                        event = ScoreEvent(
                            frame_idx=frame_idx,
                            ball_track_id=tid,
                            zone_name=zone_type,
                            shooter_label=shooter,
                            shooter_dist=dist,
                            period=period,
                        )
                        self.events.append(event)
                        self._update_robot_score(shooter, zone_type, period)

                        # 設定冷卻
                        self._ball_cooldown[tid] = SCORE_COOLDOWN_FRAMES
                else:
                    # 離開區域 → 重置停留計數
                    key = (tid, zone.name)
                    if key in self._ball_zone_frames:
                        del self._ball_zone_frames[key]

            self._ball_in_zone[tid] = curr_zones

    def _find_shooter(self, ball_tid, frame_idx, robot_positions,
                      ball_trajectories):
        """
        回溯球軌跡找最近的機器人（射手歸因）。

        Returns:
            (shooter_label, distance) 或 ("未知", 0)
        """
        if ball_trajectories is None or ball_tid not in ball_trajectories:
            # 無軌跡資料時用當前幀最近的機器人
            return self._nearest_robot_now(robot_positions, ball_trajectories,
                                           ball_tid, frame_idx)

        traj = ball_trajectories[ball_tid]

        # 回溯 N 幀找球離開機器人附近的位置
        lookback_start = max(0, frame_idx - SCORE_PROXIMITY_FRAMES)
        relevant_points = [
            (f, cx, cy) for f, cx, cy, *_ in traj
            if lookback_start <= f <= frame_idx
        ]

        if not relevant_points:
            return self._nearest_robot_now(robot_positions, ball_trajectories,
                                           ball_tid, frame_idx)

        best_label = "未知"
        best_dist = float('inf')

        for label, (rx, ry) in robot_positions.items():
            for f, bx, by in relevant_points:
                d = distance((bx, by), (rx, ry))
                if d < best_dist:
                    best_dist = d
                    best_label = label

        if best_dist > SCORE_MAX_SHOOTER_DIST:
            return ("未知", best_dist)

        return (best_label, best_dist)

    def _nearest_robot_now(self, robot_positions, ball_trajectories,
                           ball_tid, frame_idx):
        """用當前幀位置找最近的機器人。"""
        if not robot_positions:
            return ("未知", 0)

        # 取得球的當前位置
        ball_pos = None
        if ball_trajectories and ball_tid in ball_trajectories:
            traj = ball_trajectories[ball_tid]
            for f, cx, cy, *_ in reversed(traj):
                if f <= frame_idx:
                    ball_pos = (cx, cy)
                    break

        if ball_pos is None:
            return ("未知", 0)

        best_label = "未知"
        best_dist = float('inf')
        for label, (rx, ry) in robot_positions.items():
            d = distance(ball_pos, (rx, ry))
            if d < best_dist:
                best_dist = d
                best_label = label

        if best_dist > SCORE_MAX_SHOOTER_DIST:
            return ("未知", best_dist)

        return (best_label, best_dist)

    def _update_robot_score(self, label, zone_type, period):
        """更新機器人得分統計。"""
        if label not in self.robot_scores:
            self.robot_scores[label] = RobotScore(label=label)
        score = self.robot_scores[label]

        if period == "Auto":
            if zone_type == "Upper":
                score.auto_upper += 1
            else:
                score.auto_lower += 1
        else:
            if zone_type == "Upper":
                score.teleop_upper += 1
            else:
                score.teleop_lower += 1

    def get_summary(self):
        """取得所有機器人的得分摘要。"""
        return dict(self.robot_scores)

    def get_events_at_frame(self, frame_idx):
        """取得指定幀的進球事件。"""
        return [e for e in self.events if e.frame_idx == frame_idx]

    def get_timeline(self):
        """取得按時間排序的事件時間軸。"""
        return sorted(self.events, key=lambda e: e.frame_idx)

"""
FRC Scoring Analyzer — 進球判定引擎（區域進入 + 射手歸因 + 出手偵測）
"""

import math
from dataclasses import dataclass, field

from config import (
    SCORE_PROXIMITY_FRAMES, SCORE_MAX_SHOOTER_DIST,
    SCORE_ZONE_DWELL_FRAMES, SCORE_COOLDOWN_FRAMES,
    AUTO_DURATION_SEC, TELEOP_START_SEC,
    SHOT_MIN_VELOCITY, SHOT_ROBOT_PROXIMITY,
)
from geometry import point_in_polygon, distance


@dataclass
class ScoringZone:
    """得分區域定義（多邊形）。"""
    name: str                          # "紅方 Hub" / "藍方 Hub"
    points: list[tuple[int, int]]      # 多邊形頂點 [(x1,y1), ...]
    alliance: str = ""                 # "red" / "blue"

    def contains(self, px, py):
        return point_in_polygon(px, py, self.points)

    @property
    def center(self):
        n = len(self.points)
        cx = sum(p[0] for p in self.points) / n
        cy = sum(p[1] for p in self.points) / n
        return (cx, cy)


@dataclass
class ScoreEvent:
    """進球事件。"""
    frame_idx: int
    ball_track_id: int
    zone_name: str       # "紅方 Hub" / "藍方 Hub"
    shooter_label: str   # 機器人標籤，或 "未知"
    shooter_dist: float  # 射手距離
    period: str          # "Auto" 或 "Teleop"
    alliance: str = ""   # "red" / "blue"


@dataclass
class ShotEvent:
    """出手事件（包含進球和未進球）。"""
    frame_idx: int           # 出手幀
    ball_track_id: int
    shooter_label: str
    shooter_dist: float
    period: str              # "Auto" / "Teleop"
    alliance: str = ""       # 射手聯盟
    result: str = "miss"     # "goal" / "miss"
    goal_frame: int = -1     # 進球幀（miss 則為 -1）
    zone_name: str = ""      # 進球區域（miss 則為空）


@dataclass
class RobotScore:
    """單一機器人的得分統計（含出手數據）。"""
    label: str
    alliance: str = ""
    auto_goals: int = 0
    teleop_goals: int = 0
    auto_misses: int = 0
    teleop_misses: int = 0

    @property
    def auto(self) -> int:
        """Auto 期間進球數（向後相容）。"""
        return self.auto_goals

    @property
    def teleop(self) -> int:
        """Teleop 期間進球數（向後相容）。"""
        return self.teleop_goals

    @property
    def total(self) -> int:
        return self.auto_goals + self.teleop_goals

    @property
    def total_shots(self) -> int:
        return (self.auto_goals + self.teleop_goals +
                self.auto_misses + self.teleop_misses)

    @property
    def total_misses(self) -> int:
        return self.auto_misses + self.teleop_misses

    @property
    def accuracy(self) -> float:
        """命中率 (0.0 ~ 1.0)，無出手則回傳 0.0。"""
        total = self.total_shots
        return self.total / total if total > 0 else 0.0


class ScoringEngine:
    """進球判定引擎。"""

    def __init__(self, fps=30.0, auto_sec=AUTO_DURATION_SEC,
                 teleop_start_sec=TELEOP_START_SEC):
        self.fps = fps
        self.auto_end_frame = int(auto_sec * fps)
        self.teleop_start_frame = int(teleop_start_sec * fps)

        self.zones = []          # List[ScoringZone]
        self.events = []         # List[ScoreEvent]
        self.shot_events = []    # List[ShotEvent]
        self.robot_scores = {}   # label -> RobotScore

        # 內部狀態：追蹤每顆球的區域停留
        self._ball_zone_frames = {}  # (track_id, zone_name) -> consecutive_frames
        self._ball_in_zone = {}      # track_id -> set of zone_names currently in
        self._ball_cooldown = {}     # track_id -> cooldown_remaining

    def add_zone(self, name, points, alliance=""):
        self.zones.append(ScoringZone(name, points, alliance))

    def clear_zones(self):
        self.zones.clear()

    def set_zones(self, zones):
        self.zones = list(zones)

    def reset(self):
        self.events.clear()
        self.shot_events.clear()
        self.robot_scores.clear()
        self._ball_zone_frames.clear()
        self._ball_in_zone.clear()
        self._ball_cooldown.clear()

    def get_period(self, frame_idx):
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
            ball_trajectories: {track_id: [(f, cx, cy, area), ...]} 完整軌跡
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

                    if zone.name not in prev_zones:
                        self._ball_zone_frames[key] = 1

                    if self._ball_zone_frames[key] == SCORE_ZONE_DWELL_FRAMES:
                        shooter, dist = self._find_shooter(
                            tid, frame_idx, robot_positions,
                            ball_trajectories
                        )
                        period = self.get_period(frame_idx)

                        event = ScoreEvent(
                            frame_idx=frame_idx,
                            ball_track_id=tid,
                            zone_name=zone.name,
                            shooter_label=shooter,
                            shooter_dist=dist,
                            period=period,
                            alliance=zone.alliance,
                        )
                        self.events.append(event)
                        self._update_robot_goal(shooter, period, zone.alliance)
                        self._ball_cooldown[tid] = SCORE_COOLDOWN_FRAMES
                else:
                    key = (tid, zone.name)
                    if key in self._ball_zone_frames:
                        del self._ball_zone_frames[key]

            self._ball_in_zone[tid] = curr_zones

    def detect_shots(self, ball_trajectories: dict, robot_positions_by_frame: dict):
        """
        後處理：偵測出手事件（進球 + 未進球）。

        Args:
            ball_trajectories: {track_id: [(f, cx, cy, area), ...]}
            robot_positions_by_frame: {frame_idx: {label: (cx, cy)}}
        """
        self.shot_events.clear()

        # 建立進球事件的快速查找 {ball_track_id: ScoreEvent}
        goal_by_ball = {}
        for event in self.events:
            goal_by_ball.setdefault(event.ball_track_id, []).append(event)

        for tid, traj in ball_trajectories.items():
            if len(traj) < 3:
                continue

            traj_sorted = sorted(traj, key=lambda p: p[0])

            # 計算逐幀速度
            velocities = []
            for i in range(1, len(traj_sorted)):
                f0, x0, y0, *_ = traj_sorted[i - 1]
                f1, x1, y1, *_ = traj_sorted[i]
                dt = f1 - f0
                if dt <= 0:
                    velocities.append(0.0)
                    continue
                v = math.hypot(x1 - x0, y1 - y0) / dt
                velocities.append(v)

            # 找出手點：速度突增 + 球在機器人附近
            shot_detected = False
            for i, vel in enumerate(velocities):
                if vel < SHOT_MIN_VELOCITY:
                    continue
                if shot_detected:
                    continue  # 同一軌跡只取第一次出手

                f_shot = traj_sorted[i + 1][0]  # 速度高的那幀
                bx = traj_sorted[i + 1][1]
                by = traj_sorted[i + 1][2]

                # 找出手時最近的機器人
                robot_pos = robot_positions_by_frame.get(f_shot, {})
                if not robot_pos:
                    # 嘗試附近幀
                    for delta in range(-3, 4):
                        robot_pos = robot_positions_by_frame.get(
                            f_shot + delta, {})
                        if robot_pos:
                            break

                best_label = "未知"
                best_dist = float('inf')
                best_alliance = ""

                for label, pos in robot_pos.items():
                    rx, ry = pos[0], pos[1]
                    d = math.hypot(bx - rx, by - ry)
                    if d < best_dist:
                        best_dist = d
                        best_label = label

                if best_dist > SHOT_ROBOT_PROXIMITY:
                    continue  # 太遠，不算出手

                # 取得射手聯盟
                if best_label in self.robot_scores:
                    best_alliance = self.robot_scores[best_label].alliance

                period = self.get_period(f_shot)

                # 判斷結果：此球是否有進球事件
                result = "miss"
                goal_frame = -1
                zone_name = ""
                if tid in goal_by_ball:
                    for ge in goal_by_ball[tid]:
                        if ge.frame_idx >= f_shot:
                            result = "goal"
                            goal_frame = ge.frame_idx
                            zone_name = ge.zone_name
                            break

                shot = ShotEvent(
                    frame_idx=f_shot,
                    ball_track_id=tid,
                    shooter_label=best_label,
                    shooter_dist=best_dist,
                    period=period,
                    alliance=best_alliance,
                    result=result,
                    goal_frame=goal_frame,
                    zone_name=zone_name,
                )
                self.shot_events.append(shot)
                shot_detected = True

                # 更新 miss 統計
                if result == "miss":
                    self._update_robot_miss(best_label, period, best_alliance)

        self.shot_events.sort(key=lambda s: s.frame_idx)

    def _find_shooter(self, ball_tid, frame_idx, robot_positions,
                      ball_trajectories):
        if ball_trajectories is None or ball_tid not in ball_trajectories:
            return self._nearest_robot_now(robot_positions, ball_trajectories,
                                           ball_tid, frame_idx)

        traj = ball_trajectories[ball_tid]
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
        if not robot_positions:
            return ("未知", 0)

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

    def _update_robot_goal(self, label, period, alliance=""):
        """更新機器人進球統計。"""
        if label not in self.robot_scores:
            self.robot_scores[label] = RobotScore(
                label=label, alliance=alliance)
        score = self.robot_scores[label]
        if period == "Auto":
            score.auto_goals += 1
        else:
            score.teleop_goals += 1

    def _update_robot_miss(self, label, period, alliance=""):
        """更新機器人未進球統計。"""
        if label not in self.robot_scores:
            self.robot_scores[label] = RobotScore(
                label=label, alliance=alliance)
        score = self.robot_scores[label]
        if period == "Auto":
            score.auto_misses += 1
        else:
            score.teleop_misses += 1

    def get_summary(self):
        return dict(self.robot_scores)

    def get_events_at_frame(self, frame_idx):
        return [e for e in self.events if e.frame_idx == frame_idx]

    def get_timeline(self):
        return sorted(self.events, key=lambda e: e.frame_idx)

    def get_shot_timeline(self):
        """取得所有出手事件（按時間排序）。"""
        return sorted(self.shot_events, key=lambda s: s.frame_idx)

    def get_all_events_timeline(self):
        """取得所有事件（進球 + 出手）混合時間軸。"""
        all_events = []
        for e in self.events:
            all_events.append({
                "type": "goal",
                "frame_idx": e.frame_idx,
                "shooter": e.shooter_label,
                "alliance": e.alliance,
                "period": e.period,
                "zone": e.zone_name,
            })
        for s in self.shot_events:
            if s.result == "miss":
                all_events.append({
                    "type": "miss",
                    "frame_idx": s.frame_idx,
                    "shooter": s.shooter_label,
                    "alliance": s.alliance,
                    "period": s.period,
                    "zone": "",
                })
        all_events.sort(key=lambda e: e["frame_idx"])
        return all_events

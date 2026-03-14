"""
FRC Scoring Analyzer — 進球判定引擎（區域進入 + 射手歸因 + 出手偵測）
"""

import math
from dataclasses import dataclass, field

from config import (
    SCORE_PROXIMITY_FRAMES, SCORE_MAX_SHOOTER_DIST,
    SCORE_ZONE_DWELL_FRAMES, SCORE_COOLDOWN_FRAMES,
    AUTO_DURATION_SEC, TELEOP_START_SEC,
    SHOT_MIN_VELOCITY, SHOT_MIN_UPWARD_VELOCITY, SHOT_ROBOT_PROXIMITY,
    BALL_OWNERSHIP_DIST,
)
from geometry import point_in_polygon, segments_intersect, distance


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
                 teleop_start_sec=TELEOP_START_SEC,
                 proximity_frames=SCORE_PROXIMITY_FRAMES,
                 max_shooter_dist=SCORE_MAX_SHOOTER_DIST,
                 zone_dwell_frames=SCORE_ZONE_DWELL_FRAMES,
                 cooldown_frames=SCORE_COOLDOWN_FRAMES,
                 shot_min_velocity=SHOT_MIN_VELOCITY,
                 shot_min_upward_velocity=SHOT_MIN_UPWARD_VELOCITY,
                 shot_robot_proximity=SHOT_ROBOT_PROXIMITY,
                 ball_ownership_dist=BALL_OWNERSHIP_DIST):
        self.fps = fps
        self.auto_end_frame = int(auto_sec * fps)
        self.teleop_start_frame = int(teleop_start_sec * fps)
        self._proximity_frames = proximity_frames
        self._max_shooter_dist = max_shooter_dist
        self._zone_dwell_frames = zone_dwell_frames
        self._cooldown_frames = cooldown_frames
        self._shot_min_velocity = shot_min_velocity
        self._shot_min_upward_velocity = shot_min_upward_velocity
        self._shot_robot_proximity = shot_robot_proximity
        self._ownership_dist = ball_ownership_dist

        self.zones = []          # List[ScoringZone]
        self.hp_lines = []       # [{"name": str, "alliance": str, "p1": (x,y), "p2": (x,y)}]
        self.events = []         # List[ScoreEvent]
        self.shot_events = []    # List[ShotEvent]
        self.robot_scores = {}   # label -> RobotScore

        # 內部狀態：追蹤每顆球的區域停留
        self._ball_zone_frames = {}  # (track_id, zone_name) -> consecutive_frames
        self._ball_in_zone = {}      # track_id -> set of zone_names currently in
        self._ball_cooldown = {}     # track_id -> cooldown_remaining

        # 球所有權追蹤（後處理計算）
        self._ball_ownership = {}    # {ball_track_id: {frame_idx: owner_label}}

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
        self._ball_ownership.clear()
        # hp_lines 不清除（由 UI 管理）

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

                    if self._ball_zone_frames[key] == self._zone_dwell_frames:
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
                        self._ball_cooldown[tid] = self._cooldown_frames
                else:
                    key = (tid, zone.name)
                    if key in self._ball_zone_frames:
                        del self._ball_zone_frames[key]

            self._ball_in_zone[tid] = curr_zones

    def compute_ball_ownership(self, ball_trajectories: dict,
                               robot_positions_by_frame: dict):
        """
        後處理：計算每顆球在每幀的所有者（最近機器人）。

        球接近機器人（距離 <= ownership_dist）且球速低於出手速度時轉移所有權；
        球在飛行中（高速度或無機器人在附近）時保持上一個所有者。

        Args:
            ball_trajectories: {track_id: [(f, cx, cy, area), ...]}
            robot_positions_by_frame: {frame_idx: {label: (cx, cy)}}
        """
        self._ball_ownership.clear()
        dist_threshold = self._ownership_dist
        # 球速低於出手速度一半才算「被持有」，避免滾動中的球被誤歸
        speed_threshold = self._shot_min_velocity * 0.5
        total_owned = 0
        total_points = 0

        for tid, traj in ball_trajectories.items():
            if len(traj) < 2:
                continue

            traj_sorted = sorted(traj, key=lambda p: p[0])
            ownership = {}
            current_owner = None
            prev_x, prev_y, prev_f = None, None, None

            for point in traj_sorted:
                f, bx, by = point[0], point[1], point[2]
                total_points += 1

                # 計算球速度（像素/幀）
                ball_speed = 0.0
                if prev_x is not None and f > prev_f:
                    dt = f - prev_f
                    ball_speed = math.hypot(bx - prev_x, by - prev_y) / dt
                prev_x, prev_y, prev_f = bx, by, f

                # 球在飛行中（高速度）→ 保持 current_owner，不允許轉移
                if ball_speed >= speed_threshold:
                    if current_owner:
                        ownership[f] = current_owner
                        total_owned += 1
                    continue

                robot_pos = robot_positions_by_frame.get(f, {})
                if not robot_pos:
                    if current_owner:
                        ownership[f] = current_owner
                        total_owned += 1
                    continue

                best_label = None
                best_dist = float('inf')
                for label, pos in robot_pos.items():
                    rx, ry = pos[0], pos[1]
                    d = distance((bx, by), (rx, ry))
                    if d < best_dist:
                        best_dist = d
                        best_label = label

                if best_dist <= dist_threshold:
                    current_owner = best_label

                if current_owner:
                    ownership[f] = current_owner
                    total_owned += 1

            if ownership:
                self._ball_ownership[tid] = ownership

        n_balls = len(self._ball_ownership)
        coverage = total_owned / max(total_points, 1) * 100
        print(f"[INFO] Ball Ownership: {n_balls} 條軌跡有 owner, "
              f"覆蓋率 {coverage:.0f}% ({total_owned}/{total_points} 點)")

    def _get_ball_owner_at_frame(self, ball_track_id: int, frame_idx: int,
                                  lookback: int = 0) -> str | None:
        """查找球在指定幀（或往前 lookback 幀內）的所有者。"""
        ownership = self._ball_ownership.get(ball_track_id)
        if not ownership:
            return None
        for f in range(frame_idx, frame_idx - lookback - 1, -1):
            if f in ownership:
                return ownership[f]
        return None

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

            # 計算逐幀速度和垂直速度
            velocities = []
            dy_values = []
            for i in range(1, len(traj_sorted)):
                f0, x0, y0, *_ = traj_sorted[i - 1]
                f1, x1, y1, *_ = traj_sorted[i]
                dt = f1 - f0
                if dt <= 0:
                    velocities.append(0.0)
                    dy_values.append(0.0)
                    continue
                v = math.hypot(x1 - x0, y1 - y0) / dt
                velocities.append(v)
                dy_values.append((y1 - y0) / dt)

            # 找出手點：速度突增 + 球往上飛 + 球在機器人附近
            shot_detected = False
            for i, vel in enumerate(velocities):
                if vel < self._shot_min_velocity:
                    continue
                # 球必須往畫面上方飛（dy < 0，攝影機斜下拍）
                if dy_values[i] >= -self._shot_min_upward_velocity:
                    continue
                if shot_detected:
                    continue  # 同一軌跡只取第一次出手

                f_shot = traj_sorted[i + 1][0]  # 速度高的那幀
                bx = traj_sorted[i + 1][1]
                by = traj_sorted[i + 1][2]

                # 優先用 ownership 歸因
                owner = self._get_ball_owner_at_frame(
                    tid, f_shot, lookback=self._proximity_frames)

                best_label = "未知"
                best_dist = float('inf')
                best_alliance = ""

                if owner:
                    best_label = owner
                    best_dist = 0.0  # ownership-based
                else:
                    # Fallback: 找出手時最近的機器人
                    robot_pos = robot_positions_by_frame.get(f_shot, {})
                    if not robot_pos:
                        for delta in range(-3, 4):
                            robot_pos = robot_positions_by_frame.get(
                                f_shot + delta, {})
                            if robot_pos:
                                break

                    for label, pos in robot_pos.items():
                        rx, ry = pos[0], pos[1]
                        d = math.hypot(bx - rx, by - ry)
                        if d < best_dist:
                            best_dist = d
                            best_label = label

                    if best_dist > self._shot_robot_proximity:
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
        lookback_start = max(0, frame_idx - self._proximity_frames)
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

        if best_dist > self._max_shooter_dist:
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

        if best_dist > self._max_shooter_dist:
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

    def reattribute_shooters(self, ball_trajectories: dict,
                              robot_positions_by_frame: dict):
        """後處理：用完整機器人位置資料重新歸因射手。

        在 merge_fragmented_labels() + interpolate_positions() 之後呼叫，
        利用完整的（合併 + 插值後的）位置資料重新找出每個進球事件的射手。
        若設定了 HP 線段且球軌跡經過 HP 附近，優先歸因給 HP。

        Args:
            ball_trajectories: {track_id: [(f, cx, cy, area), ...]}
            robot_positions_by_frame: {frame_idx: {label: (cx, cy)}}
        """
        # 清空進球統計（保留 miss 統計），稍後重新累計
        for score in self.robot_scores.values():
            score.auto_goals = 0
            score.teleop_goals = 0

        ownership_used = 0
        proximity_used = 0
        hp_used = 0

        for event in self.events:
            traj = ball_trajectories.get(event.ball_track_id, [])
            if not traj:
                self._update_robot_goal(
                    event.shooter_label, event.period, event.alliance)
                continue

            lookback_start = max(0,
                                 event.frame_idx - self._proximity_frames)
            relevant_points = [
                (f, cx, cy) for f, cx, cy, *_ in traj
                if lookback_start <= f <= event.frame_idx
            ]

            if not relevant_points:
                self._update_robot_goal(
                    event.shooter_label, event.period, event.alliance)
                continue

            # 優先順序 1: HP 歸因
            hp_label = self._check_hp_attribution(
                relevant_points, event.alliance)
            if hp_label:
                event.shooter_label = hp_label
                event.shooter_dist = 0.0
                self._update_robot_goal(
                    hp_label, event.period, event.alliance)
                hp_used += 1
                continue

            # 優先順序 2: Ball ownership 查表
            owner = self._get_ball_owner_at_frame(
                event.ball_track_id, event.frame_idx,
                lookback=self._proximity_frames)
            if owner:
                event.shooter_label = owner
                event.shooter_dist = 0.0
                self._update_robot_goal(
                    owner, event.period, event.alliance)
                ownership_used += 1
                continue

            # 優先順序 3: Fallback — 回溯距離搜尋（原邏輯）
            best_label = "未知"
            best_dist = float('inf')

            for f, bx, by in relevant_points:
                robot_pos = robot_positions_by_frame.get(f, {})
                for label, pos in robot_pos.items():
                    rx, ry = pos[0], pos[1]
                    d = distance((bx, by), (rx, ry))
                    if d < best_dist:
                        best_dist = d
                        best_label = label

            if best_dist <= self._max_shooter_dist:
                event.shooter_label = best_label
                event.shooter_dist = best_dist
            else:
                event.shooter_label = "未知"
                event.shooter_dist = best_dist

            proximity_used += 1
            self._update_robot_goal(
                event.shooter_label, event.period, event.alliance)

        total_events = len(self.events)
        print(f"[INFO] 射手歸因: {total_events} 進球 — "
              f"HP:{hp_used}, Ownership:{ownership_used}, "
              f"Proximity:{proximity_used}")

    def _check_hp_attribution(self, relevant_points, zone_alliance):
        """檢查球軌跡是否穿過 HP 線段（線段交叉判定），回傳 HP label 或 None。"""
        if not self.hp_lines:
            return None
        for hp in self.hp_lines:
            # HP 的聯盟要跟進球區域的聯盟一致
            if hp["alliance"] and zone_alliance and \
                    hp["alliance"] != zone_alliance:
                continue
            p3, p4 = hp["p1"], hp["p2"]
            for i in range(1, len(relevant_points)):
                p1 = (relevant_points[i - 1][1], relevant_points[i - 1][2])
                p2 = (relevant_points[i][1], relevant_points[i][2])
                if segments_intersect(p1, p2, p3, p4):
                    return hp["name"]
        return None

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

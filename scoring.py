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
    BALL_OWNERSHIP_DIST, SCORE_HP_PROXIMITY, SCORE_HP_ORIGIN_DIST,
    SCORE_EXTRAPOLATE_FRAMES, SCORE_DIRECTION_WEIGHT,
    SCORE_MIN_VELOCITY_FOR_DIR, SCORE_ALLIANCE_BIAS,
    SCORE_TRACE_MAX_DT,
)
from geometry import (point_in_polygon, segments_intersect, distance,
                      point_to_segment_distance)


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
    confidence: float = 0.0  # 歸因置信度 (0.0~1.0)
    attribution_method: str = ""  # "hp"/"trace"/"extrapolation"/"proximity"/"unknown"
    certainty_level: str = ""  # "certain"/"likely"/"uncertain"/"unknown"


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
    confidence: float = 0.0  # 歸因置信度 (0.0~1.0)
    attribution_method: str = ""  # "hp"/"trace"/"extrapolation"/"proximity"/"unknown"


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

                f_shot = traj_sorted[i + 1][0]  # 速度高的那幀（出手時間標記）

                # 提前查找進球事件的 zone alliance（用於聯盟偏好）
                shot_zone_alliance = ""
                if tid in goal_by_ball:
                    for ge in goal_by_ball[tid]:
                        if ge.frame_idx >= f_shot:
                            shot_zone_alliance = ge.alliance
                            break

                # 用軌跡歸因找射手（與 reattribute_shooters 一致）
                best_label, best_dist, method, shot_conf = \
                    self._find_shooter_by_trajectory(
                        traj_sorted, f_shot, robot_positions_by_frame,
                        zone_alliance=shot_zone_alliance)

                if best_dist > self._shot_robot_proximity:
                    continue  # 太遠，不算出手

                best_alliance = ""

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

                # Shot Speed Profile：驗證出手速度合理性
                # FRC 典型射球速度 ≈ 8~25 px/f
                if vel > self._shot_min_velocity * 5:
                    shot_conf = max(0.0, shot_conf - 0.08)
                    # 異常高速，可能是偵測跳幀
                elif vel < self._shot_min_velocity * 1.2:
                    shot_conf = max(0.0, shot_conf - 0.04)
                    # 勉強超過門檻

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
                    confidence=shot_conf,
                    attribution_method=method,
                )
                self.shot_events.append(shot)
                shot_detected = True

                # 更新 miss 統計
                if result == "miss":
                    self._update_robot_miss(best_label, period, best_alliance)

        self.shot_events.sort(key=lambda s: s.frame_idx)

        # 診斷摘要
        n_goals = sum(1 for s in self.shot_events if s.result == "goal")
        n_miss = sum(1 for s in self.shot_events if s.result == "miss")
        print(f"[INFO] 出手偵測: {len(self.shot_events)} 出手 "
              f"(進球={n_goals}, 未進={n_miss})")
        # Per-robot 出手統計
        shot_by_robot = {}
        for s in self.shot_events:
            sr = shot_by_robot.setdefault(s.shooter_label, [0, 0])
            if s.result == "goal":
                sr[0] += 1
            else:
                sr[1] += 1
        for lbl in sorted(shot_by_robot.keys()):
            g, m = shot_by_robot[lbl]
            acc = g / (g + m) * 100 if (g + m) > 0 else 0
            print(f"[INFO]   {lbl}: {g + m}射 "
                  f"(進={g} 未進={m} 命中率={acc:.0f}%)")

    def _find_shooter(self, ball_tid, frame_idx, robot_positions,
                      ball_trajectories):
        """即時歸因（僅有當前幀機器人位置，結果會被 reattribute_shooters 覆蓋）。"""
        if ball_trajectories is None or ball_tid not in ball_trajectories:
            return self._nearest_robot_now(robot_positions, ball_trajectories,
                                           ball_tid, frame_idx)

        traj = ball_trajectories[ball_tid]
        # 找回溯目標幀（proximity_frames 幀前）的球位置
        target_frame = max(0, frame_idx - self._proximity_frames)
        ball_pos = None
        # 找最接近 target_frame 的軌跡點
        best_gap = float('inf')
        for f, cx, cy, *_ in traj:
            gap = abs(f - target_frame)
            if gap < best_gap:
                best_gap = gap
                ball_pos = (cx, cy)

        if ball_pos is None:
            return self._nearest_robot_now(robot_positions, ball_trajectories,
                                           ball_tid, frame_idx)

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

    def _proximity_fallback(self, traj, goal_frame,
                            robot_positions_by_frame,
                            zone_alliance=""):
        """Proximity fallback: 用 proximity_frames 前的球位置找最近機器人。

        Trace + Extrapolation 都失敗時的最後防線。
        使用目標幀附近 ±5 幀的球位置平均，減少單幀位置誤差。
        同聯盟機器人有軟偏好（分數 ×SCORE_ALLIANCE_BIAS）。

        Returns:
            (label, dist, confidence)
        """
        # 時間窗口自適應：球速快 → 更早的幀，球速慢 → 更近的幀
        prox_lookback = self._proximity_frames
        traj_sorted = sorted(
            [(f, cx, cy) for f, cx, cy, *_ in traj if f <= goal_frame],
            key=lambda p: p[0])
        if len(traj_sorted) >= 3:
            # 估算球最後幾段的平均速度
            last_n = min(5, len(traj_sorted) - 1)
            spds = []
            for si in range(len(traj_sorted) - last_n, len(traj_sorted)):
                if si < 1:
                    continue
                sdt = traj_sorted[si][0] - traj_sorted[si - 1][0]
                if sdt > 0:
                    s = math.hypot(
                        traj_sorted[si][1] - traj_sorted[si - 1][1],
                        traj_sorted[si][2] - traj_sorted[si - 1][2]) / sdt
                    spds.append(s)
            if spds:
                avg_spd = sum(spds) / len(spds)
                if avg_spd >= self._shot_min_velocity:
                    prox_lookback = int(prox_lookback * 1.3)  # 快球看更遠
                elif avg_spd < SCORE_MIN_VELOCITY_FOR_DIR:
                    prox_lookback = int(prox_lookback * 0.7)  # 慢球看更近
        target_frame = max(0, goal_frame - prox_lookback)
        # 收集目標幀附近 ±5 幀的球位置，取平均
        _WINDOW = 5
        nearby_points = []
        for f, cx, cy, *_ in traj:
            if abs(f - target_frame) <= _WINDOW:
                nearby_points.append((cx, cy))
        if not nearby_points:
            # Fallback: 找最接近的單點
            best_gap = float('inf')
            ball_pos = None
            for f, cx, cy, *_ in traj:
                gap = abs(f - target_frame)
                if gap < best_gap:
                    best_gap = gap
                    ball_pos = (cx, cy)
            if ball_pos is None:
                return ("未知", float('inf'), 0.0)
            nearby_points = [ball_pos]

        avg_x = sum(p[0] for p in nearby_points) / len(nearby_points)
        avg_y = sum(p[1] for p in nearby_points) / len(nearby_points)

        # 多幀機器人距離加權平均（±5 幀，時間衰減）
        robot_agg = {}  # label → (weighted_dist_sum, weight_sum)
        for delta in range(-_WINDOW, _WINDOW + 1):
            f_check = target_frame + delta
            rp = robot_positions_by_frame.get(f_check)
            if not rp:
                continue
            w = 1.0 / (1.0 + abs(delta) / 3.0)  # 時間衰減
            for label, pos in rp.items():
                d = math.hypot(avg_x - pos[0], avg_y - pos[1])
                if label not in robot_agg:
                    robot_agg[label] = (0.0, 0.0)
                wd, ws = robot_agg[label]
                robot_agg[label] = (wd + d * w, ws + w)
        if not robot_agg:
            return ("未知", float('inf'), 0.0)

        # 找進球 zone 中心（用於守門員排除）
        zone_center = None
        for z in self.zones:
            if z.alliance == zone_alliance:
                zone_center = z.center
                break

        # 球飛行方向（用於角度評分）
        prox_ball_dir = None
        if len(traj_sorted) >= 2:
            # 用目標幀附近的球速度方向（反向 = 球來自的方向）
            for pi in range(len(traj_sorted) - 1, 0, -1):
                pdt = traj_sorted[pi][0] - traj_sorted[pi - 1][0]
                if pdt > 0:
                    pdx = traj_sorted[pi][1] - traj_sorted[pi - 1][1]
                    pdy = traj_sorted[pi][2] - traj_sorted[pi - 1][2]
                    plen = math.hypot(pdx, pdy)
                    if plen > 3:
                        # 反方向 = 球來自的方向
                        prox_ball_dir = (-pdx / plen, -pdy / plen)
                        break

        best_label = "未知"
        best_score = float('inf')
        best_dist = float('inf')
        for label, (wd, ws) in robot_agg.items():
            d = wd / ws  # 加權平均距離
            score = d
            # 角度評分：機器人在球來源方向上 → 更可能是射手
            if prox_ball_dir and d > 5:
                rp_target = robot_positions_by_frame.get(target_frame, {})
                if label in rp_target:
                    r2b_x = avg_x - rp_target[label][0]
                    r2b_y = avg_y - rp_target[label][1]
                    r2b_len = math.hypot(r2b_x, r2b_y)
                    if r2b_len > 1:
                        # 機器人→球方向 vs 球來源方向 cosine
                        dir_sim = ((prox_ball_dir[0] * r2b_x
                                    + prox_ball_dir[1] * r2b_y)
                                   / r2b_len)
                        if dir_sim > 0.3:
                            score *= 0.85  # 方向一致
                        elif dir_sim < -0.3:
                            score *= 1.15  # 方向相反
            # Occupancy History：機器人在球附近停留越久分數越低（越可能是射手）
            occupancy_frames = 0
            for occ_delta in range(-20, 1):
                occ_f = target_frame + occ_delta
                occ_rp = robot_positions_by_frame.get(occ_f, {})
                if label in occ_rp:
                    occ_d = math.hypot(avg_x - occ_rp[label][0],
                                       avg_y - occ_rp[label][1])
                    if occ_d < 300:
                        occupancy_frames += 1
            if occupancy_frames >= 15:
                score *= 0.85  # 長時間在附近
            elif occupancy_frames <= 3:
                score *= 1.10  # 短暫出現
            if self._alliance_matches(label, zone_alliance):
                score *= SCORE_ALLIANCE_BIAS
            # 對方聯盟守門員排除：離 zone 很近的對方機器人
            # 很可能是防守方守門員，而非射手
            elif zone_alliance and zone_center:
                # 取目標幀的機器人位置檢查離 zone 距離
                rp = robot_positions_by_frame.get(target_frame, {})
                if label in rp:
                    d_to_zone = math.hypot(
                        rp[label][0] - zone_center[0],
                        rp[label][1] - zone_center[1])
                    if d_to_zone < 120:
                        # 對方聯盟且離 zone 很近 = 高度疑似守門員
                        score *= 1.8  # 大幅懲罰
            # Zone-Relative Position：機器人在球和zone之間（前方）
            # 更可能是射手；在zone後方更可能是防守/干擾
            if zone_center and d > 5:
                rp_zr = robot_positions_by_frame.get(target_frame, {})
                if label in rp_zr:
                    # 球→zone 向量
                    b2z_x = zone_center[0] - avg_x
                    b2z_y = zone_center[1] - avg_y
                    b2z_len = math.hypot(b2z_x, b2z_y)
                    # 機器人→zone 向量
                    r2z_x = zone_center[0] - rp_zr[label][0]
                    r2z_y = zone_center[1] - rp_zr[label][1]
                    r2z_len = math.hypot(r2z_x, r2z_y)
                    if b2z_len > 10 and r2z_len > 10:
                        # 機器人離zone比球離zone更遠=在球後方（射手方向）
                        if r2z_len > b2z_len + 50:
                            score *= 0.90  # 偏好（可能是射手）
                        elif r2z_len < b2z_len - 100:
                            score *= 1.10  # 在zone附近（可能是防守）
            if score < best_score:
                best_score = score
                best_dist = d
                best_label = label

        if best_dist > self._max_shooter_dist:
            return ("未知", best_dist, 0.0)
        # Proximity Alliance Second-Best Check：
        # 如果最佳候選是對方聯盟，但有同聯盟候選距離差 <50px → 切換
        if (zone_alliance and best_label != "未知"
                and not self._alliance_matches(best_label, zone_alliance)):
            for alt_lbl, (alt_wd, alt_ws) in robot_agg.items():
                if not self._alliance_matches(alt_lbl, zone_alliance):
                    continue
                alt_d = alt_wd / alt_ws
                if alt_d < best_dist * 1.3 and alt_d < self._max_shooter_dist:
                    best_label = alt_lbl
                    best_dist = alt_d
                    break
        # Proximity 置信度：距離分級
        if best_dist < 80:
            conf = 0.45  # 非常近（幾乎接觸）
        elif best_dist < 150:
            conf = 0.38  # 近距離
        elif best_dist < 300:
            conf = 0.30  # 中距離
        else:
            conf = 0.22  # 遠距離（可靠度最低）
        if len(nearby_points) >= 3:
            conf += 0.05  # 多點平均更可靠
        if self._alliance_matches(best_label, zone_alliance):
            conf += 0.05
        # 候選位置穩定性：機器人在目標幀附近位置穩定 → 更可靠
        if best_label != "未知":
            rp_before = robot_positions_by_frame.get(target_frame - 5, {})
            rp_after = robot_positions_by_frame.get(target_frame + 5, {})
            if best_label in rp_before and best_label in rp_after:
                r_move = math.hypot(
                    rp_after[best_label][0] - rp_before[best_label][0],
                    rp_after[best_label][1] - rp_before[best_label][1])
                if r_move < 30:
                    conf += 0.03  # 靜止/緩慢移動（位置可靠）
                elif r_move > 150:
                    conf -= 0.03  # 快速移動（位置不確定）
        # Proximity 候選數量懲罰：太多候選 = 歧義度高
        n_prox_candidates = len(robot_agg)
        if n_prox_candidates >= 6:
            conf -= 0.06
        elif n_prox_candidates >= 4:
            conf -= 0.03
        return (best_label, best_dist, conf)

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

        簡化邏輯：
        1. HP：球軌跡穿越 HP 線 → HP 進球
        2. 非 HP：沿球軌跡前 80 幀找最近的機器人

        Args:
            ball_trajectories: {track_id: [(f, cx, cy, area), ...]}
            robot_positions_by_frame: {frame_idx: {label: (cx, cy)}}
        """
        # 清空進球統計（保留 miss 統計），稍後重新累計
        for score in self.robot_scores.values():
            score.auto_goals = 0
            score.teleop_goals = 0

        hp_used = 0
        proximity_used = 0
        unknown_used = 0

        for event in self.events:
            traj = ball_trajectories.get(event.ball_track_id, [])
            if not traj:
                self._update_robot_goal(
                    event.shooter_label, event.period, event.alliance)
                event.confidence = 0.0
                event.attribution_method = "unknown"
                unknown_used += 1
                continue

            # ── Step 1: HP 歸因檢查 ──
            # 只要球穿越 HP 線就算 HP 進球
            lookback_start = max(0,
                                 event.frame_idx - self._proximity_frames)
            relevant_points = [
                (f, cx, cy) for f, cx, cy, *_ in traj
                if lookback_start <= f <= event.frame_idx
            ]
            hp_label = None
            hp_conf = 0.0
            if relevant_points:
                hp_label, hp_conf = self._check_hp_attribution(
                    relevant_points, event.alliance)

            if hp_label:
                # HP 進球：不需要軌跡分析
                event.shooter_label = hp_label
                event.shooter_dist = 0.0
                event.confidence = hp_conf
                event.attribution_method = "hp"
                event.certainty_level = "certain"
                self._update_robot_goal(
                    hp_label, event.period, event.alliance)
                hp_used += 1
                print(f"[INFO] HP 歸因: 幀 {event.frame_idx} "
                      f"→ {hp_label} (conf={hp_conf:.2f})")
                continue

            # ── Step 2: 沿軌跡前 80 幀找最近機器人 ──
            label, dist, conf = self._simple_proximity_attribution(
                traj, event.frame_idx, robot_positions_by_frame,
                zone_alliance=event.alliance)

            # 確定性分級
            if label == "未知":
                certainty = "unknown"
            elif conf >= 0.80:
                certainty = "certain"
            elif conf >= 0.50:
                certainty = "likely"
            elif conf >= 0.30:
                certainty = "uncertain"
            else:
                certainty = "unknown"

            old_label = event.shooter_label
            event.shooter_label = label
            event.shooter_dist = dist
            event.confidence = conf
            event.attribution_method = "proximity"
            event.certainty_level = certainty
            self._update_robot_goal(label, event.period, event.alliance)

            if label != "未知":
                proximity_used += 1
            else:
                unknown_used += 1

            if old_label != label and label != "未知":
                print(f"[INFO] 歸因: 幀 {event.frame_idx} "
                      f"{old_label} → {label} (proximity, "
                      f"dist={dist:.0f}px, conf={conf:.2f})")

        # Shot-Goal 同步
        if self.shot_events:
            goal_map = {}
            for event in self.events:
                goal_map[(event.ball_track_id, event.frame_idx)] = event
            synced = 0
            for shot in self.shot_events:
                if shot.result != "goal" or shot.goal_frame < 0:
                    continue
                ge = goal_map.get((shot.ball_track_id, shot.goal_frame))
                if ge and shot.shooter_label != ge.shooter_label:
                    shot.shooter_label = ge.shooter_label
                    shot.shooter_dist = ge.shooter_dist
                    shot.confidence = ge.confidence
                    shot.attribution_method = ge.attribution_method
                    synced += 1
            if synced:
                print(f"[INFO] Shot/Goal 同步: {synced} 個出手事件"
                      f"已更新為與進球事件一致")

        total_events = len(self.events)
        print(f"[INFO] 射手歸因: {total_events} 進球 — "
              f"HP:{hp_used}, proximity:{proximity_used}, "
              f"未知:{unknown_used}")

    def _check_hp_attribution(self, relevant_points, zone_alliance):
        """檢查球軌跡是否穿越 HP 線。

        簡化判定：只要球軌跡線段與 HP 線段相交，就算 HP 進球。

        Returns:
            (hp_label, confidence) 或 (None, 0.0)
        """
        if not self.hp_lines:
            return (None, 0.0)
        for hp in self.hp_lines:
            # HP 的聯盟要跟進球區域的聯盟一致
            if hp["alliance"] and zone_alliance and \
                    hp["alliance"] != zone_alliance:
                continue
            p3, p4 = hp["p1"], hp["p2"]
            # 檢查球軌跡是否穿越 HP 線段
            for i in range(1, len(relevant_points)):
                p1 = (relevant_points[i - 1][1], relevant_points[i - 1][2])
                p2 = (relevant_points[i][1], relevant_points[i][2])
                if segments_intersect(p1, p2, p3, p4):
                    return (hp["name"], 0.95)
        return (None, 0.0)

    def _simple_proximity_attribution(self, traj, goal_frame,
                                       robot_positions_by_frame,
                                       zone_alliance=""):
        """在球軌跡第 80 幀前的那個點，找最近的機器人。

        沿軌跡回溯到第 80 幀前（SCORE_PROXIMITY_FRAMES），
        取該點的球位置，找當時最近的機器人作為射手。

        Returns:
            (label, dist, confidence) 或 ("未知", 0.0, 0.0)
        """
        lookback = self._proximity_frames  # 80 幀
        target_frame = goal_frame - lookback

        # 收集進球前的軌跡點，按時間排序
        traj_points = sorted(
            [(f, cx, cy) for f, cx, cy, *_ in traj
             if f <= goal_frame],
            key=lambda p: p[0])

        if not traj_points:
            return ("未知", 0.0, 0.0)

        # 找軌跡中最接近 target_frame 的點
        best_pt = min(traj_points,
                      key=lambda p: abs(p[0] - target_frame))
        bf, bx, by = best_pt

        # 取該幀的機器人位置
        rpos = robot_positions_by_frame.get(int(bf), {})
        if not rpos:
            # 嘗試附近 ±5 幀
            for delta in range(1, 6):
                rpos = robot_positions_by_frame.get(int(bf) - delta, {})
                if rpos:
                    break
                rpos = robot_positions_by_frame.get(int(bf) + delta, {})
                if rpos:
                    break

        # 找最近的機器人
        best_label = "未知"
        best_dist = float('inf')

        for lbl, (rx, ry) in rpos.items():
            d = math.hypot(bx - rx, by - ry)
            if d < best_dist and d < self._max_shooter_dist:
                best_dist = d
                best_label = lbl

        if best_label == "未知":
            return ("未知", 0.0, 0.0)

        # 置信度根據距離
        if best_dist < 100:
            conf = 0.90
        elif best_dist < 200:
            conf = 0.75
        elif best_dist < 350:
            conf = 0.55
        else:
            conf = 0.35

        # 同聯盟加分
        if zone_alliance and self._alliance_matches(
                best_label, zone_alliance):
            conf = min(1.0, conf + 0.05)

        return (best_label, best_dist, conf)

    def _get_robot_positions_near_frame(self, robot_positions_by_frame,
                                         target_frame, search_range=10):
        """取得目標幀附近的機器人位置（優先查過去幀，再查未來幀）。"""
        pos = robot_positions_by_frame.get(target_frame)
        if pos:
            return pos
        for delta in range(1, search_range + 1):
            # 優先過去幀（機器人在射出前的位置更可靠）
            pos = robot_positions_by_frame.get(target_frame - delta)
            if pos:
                return pos
            pos = robot_positions_by_frame.get(target_frame + delta)
            if pos:
                return pos
        return {}

    @staticmethod
    def _alliance_matches(label, zone_alliance):
        """檢查標籤是否匹配 zone_alliance。"""
        if not zone_alliance:
            return False
        return label.lower().startswith(zone_alliance)

    def _find_shooter_by_trajectory(self, traj, goal_frame,
                                     robot_positions_by_frame,
                                     debug_frame=0,
                                     zone_alliance=""):
        """用球軌跡起源分析找出射手。

        策略：Trace（多點共識）→ Extrapolation → unknown

        Returns:
            (label, dist, method, confidence)
        """
        traj_before = sorted(
            [(f, cx, cy) for f, cx, cy, *_ in traj if f <= goal_frame],
            key=lambda p: p[0])

        if len(traj_before) < 2:
            return ("未知", float('inf'), "unknown", 0.0)

        # 軌跡 EMA 平滑（減少偵測抖動）
        # alpha=0.7 保留 70% 原始值，30% 平滑（輕量去噪）
        if len(traj_before) >= 3:
            _EMA_ALPHA = 0.7
            smoothed = [traj_before[0]]
            for i in range(1, len(traj_before)):
                sf = traj_before[i][0]  # 幀號不平滑
                sx = (_EMA_ALPHA * traj_before[i][1]
                      + (1 - _EMA_ALPHA) * smoothed[-1][1])
                sy = (_EMA_ALPHA * traj_before[i][2]
                      + (1 - _EMA_ALPHA) * smoothed[-1][2])
                smoothed.append((sf, sx, sy))
            traj_before = smoothed

        # 計算逐段速度
        seg_speeds = [0.0]
        seg_vels = [(0.0, 0.0)]
        seg_dts = [0]
        for i in range(1, len(traj_before)):
            f0, x0, y0 = traj_before[i - 1]
            f1, x1, y1 = traj_before[i]
            dt = f1 - f0
            seg_dts.append(dt)
            if dt <= 0:
                seg_speeds.append(0.0)
                seg_vels.append((0.0, 0.0))
                continue
            vx = (x1 - x0) / dt
            vy = (y1 - y0) / dt
            seg_speeds.append(math.hypot(vx, vy))
            seg_vels.append((vx, vy))

        # 方法 1: 軌跡追溯 (Trace)
        trace_result = self._trace_shooter(
            traj_before, seg_speeds, seg_vels, seg_dts,
            goal_frame, robot_positions_by_frame,
            zone_alliance, debug_frame)

        # 方法 2: 速度外推 (Extrapolation)
        extrap_result = self._extrapolate_shooter(
            traj_before, seg_speeds, seg_vels, seg_dts,
            goal_frame, robot_positions_by_frame,
            zone_alliance, debug_frame if not trace_result else 0)

        # 交叉驗證 + Adaptive Ensemble：根據兩方法信心度加權融合
        if trace_result:
            if (extrap_result[0] != "未知"
                    and extrap_result[0] == trace_result[0]):
                # 兩方法一致 → 置信度加權融合（取較高值+bonus）
                ensemble_conf = min(1.0, max(trace_result[3],
                                             extrap_result[3]) + 0.08)
                if debug_frame:
                    print(f"[DEBUG] 幀 {debug_frame} "
                          f"Ensemble一致: {trace_result[0]} "
                          f"(T={trace_result[3]:.2f},"
                          f"E={extrap_result[3]:.2f}"
                          f"→{ensemble_conf:.2f})")
                return (trace_result[0], trace_result[1],
                        trace_result[2], ensemble_conf)
            elif (extrap_result[0] != "未知"
                  and extrap_result[0] != trace_result[0]):
                # 兩方法不一致 → 信心度加權選擇勝者
                t_conf, e_conf = trace_result[3], extrap_result[3]
                if t_conf >= e_conf + 0.15:
                    # Trace 明顯更強 → 用 Trace，小幅減分
                    winner = trace_result
                    final_conf = max(0.0, t_conf - 0.04)
                elif e_conf >= t_conf + 0.15:
                    # Extrap 明顯更強 → 用 Extrap，小幅減分
                    winner = extrap_result
                    final_conf = max(0.0, e_conf - 0.04)
                else:
                    # 勢均力敵 → 默認 Trace 但大幅減分
                    winner = trace_result
                    final_conf = max(0.0, t_conf - 0.10)
                if debug_frame:
                    print(f"[DEBUG] 幀 {debug_frame} "
                          f"Ensemble衝突: T={trace_result[0]}"
                          f"({t_conf:.2f}) vs "
                          f"E={extrap_result[0]}"
                          f"({e_conf:.2f}) → "
                          f"{winner[0]}({final_conf:.2f})")
                return (winner[0], winner[1],
                        winner[2], final_conf)
            return trace_result

        if extrap_result[0] != "未知":
            return extrap_result

        # 方法 3: 短軌跡 — 球首次偵測即接近 zone，用起始位置找最近機器人
        if len(traj_before) <= 5:
            f0, x0, y0 = traj_before[0]
            robot_pos = self._get_robot_positions_near_frame(
                robot_positions_by_frame, int(f0))
            if robot_pos:
                # 短軌跡方向輔助：用球飛行方向反推射手
                short_dir = None
                if len(traj_before) >= 2:
                    sdx = traj_before[1][1] - traj_before[0][1]
                    sdy = traj_before[1][2] - traj_before[0][2]
                    sd_len = math.hypot(sdx, sdy)
                    if sd_len > 3:
                        short_dir = (-sdx / sd_len, -sdy / sd_len)
                best_lbl = "未知"
                best_score = float('inf')
                best_d = float('inf')
                for lbl, pos in robot_pos.items():
                    d = math.hypot(x0 - pos[0], y0 - pos[1])
                    sc = d
                    # 方向輔助：偏好球飛行反方向上的機器人
                    if short_dir and d > 5:
                        r2b_x = (pos[0] - x0) / d
                        r2b_y = (pos[1] - y0) / d
                        dir_sim = (short_dir[0] * r2b_x
                                   + short_dir[1] * r2b_y)
                        if dir_sim > 0.3:
                            sc *= 0.85  # 方向一致
                        elif dir_sim < -0.3:
                            sc *= 1.15  # 方向相反
                    # 短軌跡聯盟強偏好：資訊少，聯盟作為強先驗
                    if self._alliance_matches(lbl, zone_alliance):
                        sc *= 0.70
                    if sc < best_score:
                        best_score = sc
                        best_d = d
                        best_lbl = lbl
                if best_d <= self._max_shooter_dist:
                    conf = 0.25  # 短軌跡可靠度低
                    if self._alliance_matches(best_lbl, zone_alliance):
                        conf += 0.08  # 聯盟匹配加更多（資訊少時先驗重要）
                    # Short Traj Zone Distance：球起點離 zone 的距離
                    # 短軌跡球通常在 zone 附近偵測到
                    for z in self.zones:
                        if z.alliance == zone_alliance:
                            stz_cx, stz_cy = z.center
                            st_d = math.hypot(x0 - stz_cx, y0 - stz_cy)
                            if st_d > 500:
                                conf -= 0.04  # 球起點離 zone 遠
                            elif st_d < 200:
                                conf += 0.02  # 球起點在 zone 附近
                            break
                    if debug_frame:
                        print(f"[DEBUG] 幀 {debug_frame} "
                              f"短軌跡({len(traj_before)}pts): "
                              f"{best_lbl} {best_d:.0f}px "
                              f"conf={conf:.2f}")
                    return (best_lbl, best_d, "short_traj", conf)

        return ("未知", float('inf'), "unknown", 0.0)

    def _trace_shooter(self, traj_before, seg_speeds, seg_vels, seg_dts,
                        goal_frame, robot_positions_by_frame,
                        zone_alliance, debug_frame):
        """軌跡追溯（Trace）— 多點共識版。

        從後往前收集所有「球速 < 出手速度 且 機器人在附近」的軌跡點，
        投票決定射手。

        Returns:
            (label, dist, "trace", conf) 或 None（無命中）
        """
        trace_dist = self._ownership_dist  # 200px
        # 回溯深度自適應：快球（高平均速度）縮短回溯，
        # 慢球（低速/短軌跡）延長回溯以找到持球點
        base_lookback = self._proximity_frames  # 80 幀 ≈ 2.7s @30fps
        if len(seg_speeds) >= 3:
            recent_speeds = [s for s in seg_speeds[-min(10, len(seg_speeds)):]
                             if s > 0]
            if recent_speeds:
                avg_recent = sum(recent_speeds) / len(recent_speeds)
                if avg_recent >= self._shot_min_velocity * 1.5:
                    # 快球：球飛行時間短，回溯 60% 即可
                    base_lookback = int(base_lookback * 0.6)
                elif avg_recent < SCORE_MIN_VELOCITY_FOR_DIR:
                    # 慢球：可能是滾地球或持球，延長回溯 130%
                    base_lookback = int(base_lookback * 1.3)
        trace_min_frame = max(0, goal_frame - base_lookback)
        _CONSENSUS_MAX_HITS = 8
        _CONSENSUS_WINDOW = 30

        def _get_flight_dir_at(idx):
            for j in range(idx + 1, len(traj_before)):
                if (seg_speeds[j] >= SCORE_MIN_VELOCITY_FOR_DIR
                        and seg_dts[j] <= SCORE_TRACE_MAX_DT):
                    return seg_vels[j]
            return None

        # 收集 trace 命中點
        trace_hits = []
        first_hit_frame = None

        for i in range(len(traj_before) - 1, 0, -1):
            if traj_before[i][0] < trace_min_frame:
                break
            if first_hit_frame is not None:
                if (traj_before[i][0] < first_hit_frame - _CONSENSUS_WINDOW
                        or len(trace_hits) >= _CONSENSUS_MAX_HITS):
                    break
            if seg_speeds[i] >= self._shot_min_velocity:
                continue
            if seg_dts[i] > SCORE_TRACE_MAX_DT:
                continue
            # 偏轉守衛
            lookback = min(3, i)
            if lookback >= 2:
                high_before = sum(
                    1 for k in range(1, lookback + 1)
                    if seg_speeds[i - k] >= self._shot_min_velocity
                    and 0 < seg_dts[i - k] <= SCORE_TRACE_MAX_DT)
                if high_before >= 2:
                    prev_speed = seg_speeds[i - 1] if i > 0 else 0
                    if prev_speed >= self._shot_min_velocity * 2:
                        continue

            f = int(traj_before[i][0])
            bx, by = traj_before[i][1], traj_before[i][2]

            robot_pos = self._get_robot_positions_near_frame(
                robot_positions_by_frame, f, search_range=3)

            candidates = []
            for label, pos in robot_pos.items():
                d = math.hypot(bx - pos[0], by - pos[1])
                if d <= trace_dist:
                    candidates.append((label, d, pos[0], pos[1]))

            if not candidates:
                continue

            scored_cands = None
            if len(candidates) == 1:
                hit_label, hit_dist = candidates[0][0], candidates[0][1]
            else:
                flight_dir = _get_flight_dir_at(i)
                hit_label = candidates[0][0]
                hit_dist = candidates[0][1]
                best_score = float('inf')
                scored_cands = []
                for label, d, rx, ry in candidates:
                    score = d
                    dir_cs = 0.0
                    ally = self._alliance_matches(label, zone_alliance)
                    # 機器人移動方向排除：快速遠離球的機器人不太可能是射手
                    rp_prev = robot_positions_by_frame.get(f - 2, {})
                    rp_next = robot_positions_by_frame.get(f + 2, {})
                    if label in rp_prev and label in rp_next:
                        rvx = (rp_next[label][0] - rp_prev[label][0]) / 4
                        rvy = (rp_next[label][1] - rp_prev[label][1]) / 4
                        r_spd = math.hypot(rvx, rvy)
                        if r_spd > 3:  # 機器人在移動
                            # 機器人→球 方向
                            to_ball_x = bx - rx
                            to_ball_y = by - ry
                            tb_len = math.hypot(to_ball_x, to_ball_y)
                            if tb_len > 1:
                                # 機器人移動方向 vs 朝球方向 cosine
                                mov_cs = ((rvx * to_ball_x
                                           + rvy * to_ball_y)
                                          / (r_spd * tb_len))
                                if mov_cs < -0.5:
                                    # 快速遠離球 → 懲罰
                                    score *= 1.3
                    if flight_dir:
                        fd_len = math.hypot(flight_dir[0], flight_dir[1])
                        if fd_len > 0.1:
                            rtb_x = bx - rx
                            rtb_y = by - ry
                            rtb_len = math.hypot(rtb_x, rtb_y)
                            if rtb_len > 1:
                                fd_nx = flight_dir[0] / fd_len
                                fd_ny = flight_dir[1] / fd_len
                                dir_cs = (rtb_x * fd_nx +
                                          rtb_y * fd_ny) / rtb_len
                                # 非線性方向加權
                                if dir_cs > 0:
                                    df = (1 - SCORE_DIRECTION_WEIGHT
                                          * dir_cs * dir_cs)
                                else:
                                    df = (1 + SCORE_DIRECTION_WEIGHT
                                          * abs(dir_cs))
                                score = d * df
                    if ally:
                        score *= SCORE_ALLIANCE_BIAS
                    scored_cands.append(
                        (label, d, score, dir_cs, ally))
                    if score < best_score:
                        best_score = score
                        hit_label = label
                        hit_dist = d

            # 球速分類：靜止(< 3 px/f)=持球，低速(3~shot)=滾地
            is_held = seg_speeds[i] < 3.0
            trace_hits.append((hit_label, hit_dist, i, f, bx, by,
                               candidates, scored_cands, is_held))
            if first_hit_frame is None:
                first_hit_frame = f

        if not trace_hits:
            # 自適應擴展：初始窗口無命中，擴展 50% 再試一次
            expanded_min = max(0, int(trace_min_frame
                                      - base_lookback * 0.5))
            if expanded_min < trace_min_frame:
                for i in range(len(traj_before) - 1, 0, -1):
                    if traj_before[i][0] < expanded_min:
                        break
                    if traj_before[i][0] >= trace_min_frame:
                        continue  # 已搜過
                    if seg_speeds[i] >= self._shot_min_velocity:
                        continue
                    if seg_dts[i] > SCORE_TRACE_MAX_DT:
                        continue
                    f = int(traj_before[i][0])
                    bx, by = traj_before[i][1], traj_before[i][2]
                    robot_pos = self._get_robot_positions_near_frame(
                        robot_positions_by_frame, f, search_range=3)
                    for label, pos in robot_pos.items():
                        d = math.hypot(bx - pos[0], by - pos[1])
                        if d <= trace_dist:
                            is_held = seg_speeds[i] < 3.0
                            trace_hits.append(
                                (label, d, i, f, bx, by,
                                 [(label, d, pos[0], pos[1])],
                                 None, is_held))
                            if first_hit_frame is None:
                                first_hit_frame = f
                            break
                    if trace_hits:
                        break  # 找到一個就夠了
            if not trace_hits:
                if debug_frame:
                    print(f"[DEBUG] 幀 {debug_frame} Trace 無命中: "
                          f"軌跡{len(traj_before)}pts, "
                          f"搜索範圍f{expanded_min}~f{goal_frame}")
                return None

        # ── Trace hit 品質過濾 ──
        # 過濾低品質 hit：距離遠 + 時間間隔大的 hit 可靠度低
        if len(trace_hits) > 2:
            # 計算 hit 距離中位數
            hit_dists = sorted(h[1] for h in trace_hits)
            median_dist = hit_dists[len(hit_dists) // 2]
            # 過濾距離 > 中位數 × 2 的離群 hit（但至少保留 2 個）
            filtered = [h for h in trace_hits
                        if h[1] <= median_dist * 2.0]
            if len(filtered) >= 2:
                trace_hits = filtered

        # ── 持球停留偵測（dwell detection）──
        # 同一機器人連續 N 個 hit 且距離穩定 → 持球證據
        dwell_label = None
        dwell_count = 0
        if len(trace_hits) >= 3:
            # 按幀排序（早→晚），找連續 hit 同一 label
            sorted_hits = sorted(trace_hits, key=lambda h: h[3])
            run_label = sorted_hits[0][0]
            run_len = 1
            best_run_label = run_label
            best_run_len = 1
            for hi in range(1, len(sorted_hits)):
                if sorted_hits[hi][0] == run_label:
                    run_len += 1
                else:
                    if run_len > best_run_len:
                        best_run_len = run_len
                        best_run_label = run_label
                    run_label = sorted_hits[hi][0]
                    run_len = 1
            if run_len > best_run_len:
                best_run_len = run_len
                best_run_label = run_label
            if best_run_len >= 3:
                dwell_label = best_run_label
                dwell_count = best_run_len

        # ── 多點共識（距離+時間+速度轉換銳度加權投票）──
        votes = {}
        vote_weights = {}
        n_held_hits = sum(1 for h in trace_hits if h[8])  # 持球 hit 數
        # 找最小距離 hit（最接近點獲得額外權重）
        min_hit_dist = min(h[1] for h in trace_hits) if trace_hits else 999
        for hit in trace_hits:
            # 距離反比：近距離 hit 權重更高
            w_dist = 1.0 / (1.0 + hit[1] / 100.0)
            # 最接近點額外加權
            if hit[1] <= min_hit_dist * 1.1:  # 在最小距離 10% 內
                w_dist *= 1.3
            # 時間衰減：越近期（離 goal_frame 越近）權重越高
            dt_to_goal = goal_frame - hit[3]  # 幀差
            w_time = 1.0 / (1.0 + dt_to_goal / 30.0)  # 30 幀 ≈ 1 秒
            # 強化最近幀（射出前最後 10 幀最關鍵）
            if dt_to_goal <= 10:
                w_time *= 1.5
            w = w_dist * w_time
            # 速度轉換銳度：hit 後速度突增越大 → 越可能是射出點
            hit_i = hit[2]
            if (hit_i + 1 < len(seg_speeds)
                    and seg_speeds[hit_i] > 0):
                speed_ratio = seg_speeds[hit_i + 1] / max(1, seg_speeds[hit_i])
                if speed_ratio >= 3.0:
                    w *= 1.5  # 強速度轉換
                elif speed_ratio >= 2.0:
                    w *= 1.2  # 中等速度轉換
            # 靜止持球加權：球幾乎不動 + 機器人在旁 = 強持球證據
            if hit[8]:  # is_held
                w *= 1.3
            # 持球停留加權：dwell label 的票權 ×1.5
            if dwell_label and hit[0] == dwell_label:
                w *= 1.5
            votes[hit[0]] = votes.get(hit[0], 0) + 1
            vote_weights[hit[0]] = vote_weights.get(hit[0], 0.0) + w

        # 優先選擇速度突變點（低速→高速的最後一個低速 hit）
        # 這是球離開機器人的精確時刻
        transition_hit = None
        for hi in range(len(trace_hits)):
            hit = trace_hits[hi]
            hit_i = hit[2]  # 軌跡索引
            if (hit_i + 1 < len(seg_speeds)
                    and seg_speeds[hit_i + 1] >= self._shot_min_velocity):
                transition_hit = hi
                break  # 最近的突變點優先

        sel_idx = transition_hit if transition_hit is not None else 0
        sel = trace_hits[sel_idx]
        best_label, best_dist = sel[0], sel[1]
        sel_i, sel_f = sel[2], sel[3]
        sel_bx, sel_by = sel[4], sel[5]
        sel_candidates, sel_scored = sel[6], sel[7]

        if len(trace_hits) >= 3:
            primary_weight = vote_weights[best_label]
            for lbl, wt in sorted(vote_weights.items(),
                                   key=lambda x: -x[1]):
                if lbl != best_label and wt > primary_weight:
                    old_label = best_label
                    for hit in trace_hits:
                        if hit[0] == lbl:
                            best_label, best_dist = hit[0], hit[1]
                            sel_i, sel_f = hit[2], hit[3]
                            sel_bx, sel_by = hit[4], hit[5]
                            sel_candidates = hit[6]
                            sel_scored = hit[7]
                            break
                    if debug_frame:
                        print(f"[DEBUG] 幀 {debug_frame} "
                              f"共識覆蓋: {old_label}"
                              f"(w={primary_weight:.2f}) → "
                              f"{best_label}(w={wt:.2f}), "
                              f"共{len(trace_hits)}點")
                    break

        # ── 距離趨勢分析 ──
        # 多個 hit 中同一 label 的距離是否在增大（球遠離=射出方向正確）
        dist_trend = 0.0  # >0 = 球遠離（好），<0 = 球靠近（可疑）
        conv_div_pattern = False  # 趨近-發散模式
        label_dists = [(hit[3], hit[1]) for hit in trace_hits
                       if hit[0] == best_label]  # (frame, dist)
        dist_stability = 0.0  # 距離穩定性（低 = 穩定）
        if len(label_dists) >= 2:
            # 按時間排序（早→晚），看距離變化
            label_dists.sort(key=lambda x: x[0])
            d_first = label_dists[0][1]
            d_last = label_dists[-1][1]
            dist_trend = d_last - d_first  # 正=球越來越遠
            # 持球距離穩定性：距離方差低 = 球穩定在機器人旁
            if len(label_dists) >= 3:
                dists = [d for _, d in label_dists]
                mean_d = sum(dists) / len(dists)
                var_d = sum((d - mean_d) ** 2 for d in dists) / len(dists)
                dist_stability = var_d ** 0.5  # 標準差
        # 趨近-發散模式：用完整軌跡（非僅 hit 點）檢查
        # 在 sel_i 前球靠近 best_label，sel_i 後球遠離
        if (best_label != "未知" and sel_i >= 2
                and sel_i < len(traj_before) - 2):
            pre_dists = []
            post_dists = []
            for j in range(max(0, sel_i - 4), sel_i):
                jf = int(traj_before[j][0])
                jrp = self._get_robot_positions_near_frame(
                    robot_positions_by_frame, jf, search_range=2)
                if best_label in jrp:
                    jd = math.hypot(traj_before[j][1] - jrp[best_label][0],
                                    traj_before[j][2] - jrp[best_label][1])
                    pre_dists.append(jd)
            for j in range(sel_i + 1,
                           min(sel_i + 5, len(traj_before))):
                jf = int(traj_before[j][0])
                jrp = self._get_robot_positions_near_frame(
                    robot_positions_by_frame, jf, search_range=2)
                if best_label in jrp:
                    jd = math.hypot(traj_before[j][1] - jrp[best_label][0],
                                    traj_before[j][2] - jrp[best_label][1])
                    post_dists.append(jd)
            if (len(pre_dists) >= 2 and len(post_dists) >= 2
                    and pre_dists[-1] < pre_dists[0] - 10
                    and post_dists[-1] > post_dists[0] + 10):
                conv_div_pattern = True

        # ── 鄰近幀穩定性驗證 ──
        verify_dist = trace_dist * 1.5
        neighbor_confirmed = 0
        for di in [-2, -1, 1, 2]:
            ni = sel_i + di
            if ni < 0 or ni >= len(traj_before):
                continue
            nf = int(traj_before[ni][0])
            nbx, nby = traj_before[ni][1], traj_before[ni][2]
            nrp = self._get_robot_positions_near_frame(
                robot_positions_by_frame, nf, search_range=2)
            if best_label in nrp:
                nd = math.hypot(nbx - nrp[best_label][0],
                                nby - nrp[best_label][1])
                if nd <= verify_dist:
                    neighbor_confirmed += 1

        # ── Trace 置信度 ──
        conf = 0.80
        if len(sel_candidates) == 1:
            conf += 0.10
        elif len(sel_candidates) > 1 and sel_scored:
            ss = sorted(sel_scored, key=lambda c: c[2])
            gap = (ss[1][2] / ss[0][2]
                   if ss[0][2] > 0 else 1.0)
            if gap >= 1.5:
                conf += 0.05
            elif gap < 1.15:
                conf -= 0.10
            # 原始距離差異：前兩名距離差 < 20px = 距離上無法區分
            raw_d_gap = abs(ss[1][1] - ss[0][1]) if len(ss) >= 2 else 999
            if raw_d_gap < 20:
                conf -= 0.05  # 距離幾乎相同，純靠方向/聯盟區分
        if best_dist < 100:
            conf += 0.05
        if self._alliance_matches(best_label, zone_alliance):
            conf += 0.05
        # 機器人群聚罰分：候選點附近 150px 內多台機器人 → 難以區分
        if len(sel_candidates) >= 4:
            conf -= 0.08  # 4+ 台機器人密集
        elif len(sel_candidates) >= 3:
            conf -= 0.04  # 3 台機器人
        if neighbor_confirmed >= 2:
            conf += 0.05
        elif neighbor_confirmed == 0:
            conf -= 0.15
        # 機器人偵測一致性：在 trace 範圍內持續被偵測到的機器人更可靠
        if best_label != "未知":
            vis_count = 0
            vis_total = 0
            check_start = max(0, sel_f - 10)
            for vf in range(check_start, sel_f + 10):
                vis_total += 1
                rp = robot_positions_by_frame.get(vf, {})
                if best_label in rp:
                    vis_count += 1
            if vis_total > 0:
                vis_ratio = vis_count / vis_total
                if vis_ratio > 0.8:
                    conf += 0.03  # 高可見度
                elif vis_ratio < 0.3:
                    conf -= 0.05  # 低可見度（可能是插值位置）
        # 持球停留加分：同一機器人連續 3+ 個低速 hit = 強持球證據
        if dwell_label == best_label and dwell_count >= 3:
            conf += 0.10 if dwell_count >= 5 else 0.05
        # 趨近-發散模式：球先靠近再遠離 = 完整接球→射出週期
        if conv_div_pattern:
            conf += 0.08
        # 靜止持球 hit 數：多個持球點 = 機器人確實在控球
        if n_held_hits >= 3:
            conf += 0.05
        # 距離趨勢：球遠離機器人（射出）加分，球靠近（接球）減分
        if dist_trend > 30:
            conf += 0.05  # 球明顯遠離
        elif dist_trend < -30 and len(label_dists) >= 3:
            conf -= 0.10  # 球明顯靠近，可能是接球方
        # 飛行方向一致性：多 hit 的飛行方向（機器人→球）是否指向同一方向
        # 不一致 = 球可能在多次反彈，歸因不可靠
        if len(trace_hits) >= 3:
            dir_angles = []
            for hit in trace_hits:
                if hit[0] != best_label:
                    continue
                hi_i = hit[2]
                fd = _get_flight_dir_at(hi_i)
                if fd and math.hypot(fd[0], fd[1]) > 0.1:
                    angle = math.atan2(fd[1], fd[0])
                    dir_angles.append(angle)
            if len(dir_angles) >= 3:
                # 計算角度的圓形方差（circular variance）
                sin_sum = sum(math.sin(a) for a in dir_angles)
                cos_sum = sum(math.cos(a) for a in dir_angles)
                r_bar = math.hypot(sin_sum, cos_sum) / len(dir_angles)
                # r_bar=1 完全一致，r_bar=0 完全發散
                if r_bar < 0.5:
                    conf -= 0.08  # 方向高度不一致
                elif r_bar > 0.85:
                    conf += 0.03  # 方向高度一致
        # Trace Hit Spatial Clustering：trace hits 空間集中度
        # hits 形成緊密叢集 = 持球位置明確 → 加分
        if len(trace_hits) >= 3:
            hit_xs = [h[4] for h in trace_hits]
            hit_ys = [h[5] for h in trace_hits]
            cx_mean = sum(hit_xs) / len(hit_xs)
            cy_mean = sum(hit_ys) / len(hit_ys)
            cluster_radius = (sum(math.hypot(x - cx_mean, y - cy_mean)
                                  for x, y in zip(hit_xs, hit_ys))
                              / len(hit_xs))
            if cluster_radius < 30:
                conf += 0.04  # 極度集中（球位置幾乎不變）
            elif cluster_radius < 60:
                conf += 0.02  # 較集中
            elif cluster_radius > 200:
                conf -= 0.04  # 分散（球在不同位置命中）
        # Trace Hit Temporal Gap：trace hits 之間時間間隔大 = 不連續觀測
        if len(trace_hits) >= 2:
            sorted_hit_frames = sorted(h[3] for h in trace_hits)
            max_hit_gap = max(sorted_hit_frames[i] - sorted_hit_frames[i-1]
                              for i in range(1, len(sorted_hit_frames)))
            if max_hit_gap > 15:
                conf -= 0.06  # 大間隔（>0.5 秒），觀測不連續
            elif max_hit_gap > 8:
                conf -= 0.03  # 中等間隔
            # Consecutive Frame Hit Bonus：連續幀命中是強持球證據
            consecutive_runs = 1
            max_consecutive = 1
            for ci in range(1, len(sorted_hit_frames)):
                if sorted_hit_frames[ci] - sorted_hit_frames[ci-1] <= 2:
                    consecutive_runs += 1
                    max_consecutive = max(max_consecutive, consecutive_runs)
                else:
                    consecutive_runs = 1
            if max_consecutive >= 4:
                conf += 0.06  # 4+ 連續幀命中 = 持球確認
            elif max_consecutive >= 3:
                conf += 0.03  # 3 連續幀命中
        # 候選多樣性：不同候選 label 太多 → 高歧義 → 降信心
        n_unique_labels = len(votes)
        if n_unique_labels >= 4:
            conf -= 0.08  # 4+ 不同候選
        elif n_unique_labels == 3:
            conf -= 0.04  # 3 不同候選
        total_hits = len(trace_hits)
        same_hits = votes.get(best_label, 0)
        if total_hits >= 2 and same_hits == total_hits:
            conf += 0.10
        elif total_hits >= 2 and same_hits > total_hits // 2:
            conf += 0.05
        # 持球距離穩定性：距離方差低 = 球穩定在機器人旁（強持球證據）
        if dist_stability > 0 and len(label_dists) >= 3:
            if dist_stability < 15:
                conf += 0.05  # 極度穩定（距離標準差 <15px）
            elif dist_stability > 60:
                conf -= 0.03  # 距離變化大（可能是球滾過而非持球）
        # Trace Pre-Shot Deceleration：
        # 球在 trace hit 前應減速（被機器人接住），而非持續加速
        if sel_i >= 3:
            pre_speeds = seg_speeds[max(1, sel_i-3):sel_i+1]
            if len(pre_speeds) >= 2 and all(s > 0 for s in pre_speeds):
                # 速度是否遞減？
                decreasing = sum(1 for k in range(1, len(pre_speeds))
                                 if pre_speeds[k] < pre_speeds[k-1])
                if decreasing == len(pre_speeds) - 1:
                    conf += 0.03  # 球在減速（被接住的跡象）
                elif decreasing == 0 and pre_speeds[-1] > pre_speeds[0] * 1.5:
                    conf -= 0.03  # 球在加速（穿越而非被持有）
        # Robot Stationary Bonus：射手在 trace 點靜止 = 瞄準/射球姿態
        if best_label != "未知":
            rp_shot_b = robot_positions_by_frame.get(sel_f - 3, {})
            rp_shot_a = robot_positions_by_frame.get(sel_f + 3, {})
            if best_label in rp_shot_b and best_label in rp_shot_a:
                r_move_shot = math.hypot(
                    rp_shot_a[best_label][0] - rp_shot_b[best_label][0],
                    rp_shot_a[best_label][1] - rp_shot_b[best_label][1])
                if r_move_shot < 15:
                    conf += 0.04  # 幾乎靜止（瞄準姿態）
                elif r_move_shot < 30:
                    conf += 0.02  # 緩慢移動
        # 速度梯度分析：trace 點之後是否有顯著加速（射出跡象）
        speed_at_hit = seg_speeds[sel_i]
        max_speed_after = 0.0
        max_speed_after_idx = sel_i
        for j in range(sel_i + 1, min(sel_i + 6, len(seg_speeds))):
            if seg_dts[j] <= SCORE_TRACE_MAX_DT:
                if seg_speeds[j] > max_speed_after:
                    max_speed_after = seg_speeds[j]
                    max_speed_after_idx = j
        if max_speed_after > 0 and speed_at_hit > 0:
            gradient = max_speed_after / speed_at_hit
            if gradient >= 3.0:
                conf += 0.10  # 強加速（明確射出）
            elif gradient >= 2.0:
                conf += 0.05  # 中等加速
        elif max_speed_after >= self._shot_min_velocity:
            conf += 0.05  # speed_at_hit=0 但之後有高速 → 射出
        # 射出方向驗證：加速後球飛行方向是否遠離射手？
        if (max_speed_after >= self._shot_min_velocity
                and max_speed_after_idx > sel_i
                and max_speed_after_idx < len(seg_vels)):
            shot_vx, shot_vy = seg_vels[max_speed_after_idx]
            shot_len = math.hypot(shot_vx, shot_vy)
            # 取 hit 點的機器人位置
            hit_rpos = self._get_robot_positions_near_frame(
                robot_positions_by_frame, int(traj_before[sel_i][0]),
                search_range=2)
            if best_label in hit_rpos and shot_len > 1:
                r_pos = hit_rpos[best_label]
                # 機器人→球方向
                rtb_x = sel_bx - r_pos[0]
                rtb_y = sel_by - r_pos[1]
                rtb_len = math.hypot(rtb_x, rtb_y)
                if rtb_len > 1:
                    # 球飛行方向 vs 機器人→球方向的 cosine
                    shot_cs = ((shot_vx * rtb_x + shot_vy * rtb_y)
                               / (shot_len * rtb_len))
                    if shot_cs > 0.5:
                        conf += 0.05  # 球從機器人方向飛出（正常射出）
                    elif shot_cs < -0.3:
                        conf -= 0.05  # 球飛向機器人（不太合理）
        conf = max(0.0, min(1.0, conf))

        # 穩定性硬門檻：孤立 trace 點 + 低信心 → 放棄，讓 Extrapolation 嘗試
        if neighbor_confirmed == 0 and total_hits == 1 and conf < 0.65:
            if debug_frame:
                print(f"[DEBUG] 幀 {debug_frame} Trace 放棄: "
                      f"孤立點 conf={conf:.2f} < 0.65, "
                      f"fallback to Extrapolation")
            return None

        if debug_frame:
            trace_depth = goal_frame - sel_f
            n_cand = len(sel_candidates)
            if n_cand > 1 and sel_scored:
                ss = sorted(sel_scored, key=lambda c: c[2])
                cand_str = ", ".join(
                    f"{c[0]}={c[1]:.0f}px(s={c[2]:.0f}"
                    f",dir={c[3]:.2f}"
                    f"{',ally' if c[4] else ''})"
                    for c in ss[:3])
            else:
                cand_str = (f"{sel_candidates[0][0]}"
                            f"={sel_candidates[0][1]:.0f}px")
            consensus_str = ""
            if total_hits > 1:
                vote_parts = [f"{l}={c}(w={vote_weights.get(l,0):.2f})"
                              for l, c
                              in sorted(votes.items(),
                                        key=lambda x: -x[1])]
                consensus_str = (f", 共識{total_hits}點"
                                 f"({','.join(vote_parts)})")
            if dwell_label:
                consensus_str += (f", dwell={dwell_label}"
                                  f"×{dwell_count}")
            print(f"[DEBUG] 幀 {debug_frame} 軌跡追溯: "
                  f"ball@({sel_bx:.0f},{sel_by:.0f})@f{sel_f}, "
                  f"depth={trace_depth}f, "
                  f"speed={seg_speeds[sel_i]:.1f}, "
                  f"候選{n_cand}: {cand_str}, "
                  f"shooter={best_label} {best_dist:.0f}px "
                  f"conf={conf:.2f}{consensus_str}")
        return (best_label, best_dist, "trace", conf)

    def _extrapolate_shooter(self, traj_before, seg_speeds, seg_vels,
                              seg_dts, goal_frame,
                              robot_positions_by_frame,
                              zone_alliance, debug_frame):
        """速度外推（Extrapolation）歸因。

        用球初始速度外推回發射位置，在該位置找最近機器人。

        Returns:
            (label, dist, "extrapolation", conf)
        """
        reliable_indices = [
            i + 1 for i in range(min(5, len(seg_vels) - 1))
            if 0 < seg_dts[i + 1] <= SCORE_TRACE_MAX_DT]
        if not reliable_indices:
            n_init = min(5, len(seg_vels) - 1)
            if n_init <= 0:
                return ("未知", float('inf'), "unknown", 0.0)
            reliable_indices = list(range(1, n_init + 1))

        # 分段外推：如果軌跡早期方向變化大，只用最前面 2 個點
        # （越早的點越接近真實射出方向）
        if len(reliable_indices) > 2:
            early_angles = []
            for j in range(1, min(3, len(reliable_indices))):
                i_p = reliable_indices[j - 1]
                i_c = reliable_indices[j]
                v0 = seg_vels[i_p]
                v1 = seg_vels[i_c]
                l0 = math.hypot(v0[0], v0[1])
                l1 = math.hypot(v1[0], v1[1])
                if l0 > 1 and l1 > 1:
                    ca = ((v0[0] * v1[0] + v0[1] * v1[1])
                          / (l0 * l1))
                    ca = max(-1.0, min(1.0, ca))
                    early_angles.append(math.acos(ca))
            if early_angles and max(early_angles) > 0.52:  # > 30°
                # 軌跡彎曲，只用最早 2 個可靠點
                reliable_indices = reliable_indices[:2]

        rel_vx = sorted(seg_vels[i][0] for i in reliable_indices)
        rel_vy = sorted(seg_vels[i][1] for i in reliable_indices)
        n_rel = len(rel_vx)
        if n_rel % 2 == 1:
            vx_avg = rel_vx[n_rel // 2]
            vy_avg = rel_vy[n_rel // 2]
        else:
            vx_avg = (rel_vx[n_rel // 2 - 1] + rel_vx[n_rel // 2]) / 2
            vy_avg = (rel_vy[n_rel // 2 - 1] + rel_vy[n_rel // 2]) / 2
        speed_avg = math.hypot(vx_avg, vy_avg)

        # 速度一致性評估：各幀速度的變異係數（CV）
        rel_speeds = [math.hypot(seg_vels[i][0], seg_vels[i][1])
                      for i in reliable_indices]
        speed_cv = 0.0  # coefficient of variation
        if len(rel_speeds) >= 2 and speed_avg > 0:
            mean_s = sum(rel_speeds) / len(rel_speeds)
            var_s = sum((s - mean_s) ** 2 for s in rel_speeds) / len(rel_speeds)
            speed_cv = (var_s ** 0.5) / mean_s if mean_s > 0 else 0

        # 曲率評估：前幾個可靠速度向量的方向變化
        max_angle_change = 0.0
        for j in range(1, len(reliable_indices)):
            i_prev = reliable_indices[j - 1]
            i_curr = reliable_indices[j]
            vx0, vy0 = seg_vels[i_prev]
            vx1, vy1 = seg_vels[i_curr]
            len0 = math.hypot(vx0, vy0)
            len1 = math.hypot(vx1, vy1)
            if len0 > 1 and len1 > 1:
                cos_a = (vx0 * vx1 + vy0 * vy1) / (len0 * len1)
                cos_a = max(-1.0, min(1.0, cos_a))
                angle = math.acos(cos_a)  # radians
                max_angle_change = max(max_angle_change, angle)

        f0, x0, y0 = traj_before[0]
        extra = SCORE_EXTRAPOLATE_FRAMES
        if speed_avg > 0:
            max_extra = self._max_shooter_dist / speed_avg
            extra = min(extra, max_extra)
        # 拋物線擬合：用前幾個可靠軌跡點做二次多項式外推
        # 比線性+重力修正更精確（自動捕捉加速度/曲率）
        _GRAVITY_PX = 0.5  # 像素/幀²，4K 畫面下的經驗重力值
        use_parabolic = False
        parabolic_origin = None
        fit_residual = 0.0
        n_fit = min(6, len(traj_before))
        if n_fit >= 4:
            # 用前 n_fit 個點做最小二乘二次擬合 x(t), y(t)
            fit_pts = traj_before[:n_fit]
            t_vals = [p[0] - fit_pts[0][0] for p in fit_pts]
            x_vals = [p[1] for p in fit_pts]
            y_vals = [p[2] for p in fit_pts]
            # 手動二次擬合（避免 numpy 依賴）：y = a*t² + b*t + c
            # 使用正規方程 (X^T X)^-1 X^T y
            n = len(t_vals)
            s0 = float(n)
            s1 = sum(t_vals)
            s2 = sum(t * t for t in t_vals)
            s3 = sum(t ** 3 for t in t_vals)
            s4 = sum(t ** 4 for t in t_vals)
            # 行列式
            det = (s0 * (s2 * s4 - s3 * s3)
                   - s1 * (s1 * s4 - s2 * s3)
                   + s2 * (s1 * s3 - s2 * s2))
            if abs(det) > 1e-6:
                for vals, coord in [(x_vals, 'x'), (y_vals, 'y')]:
                    r0 = sum(vals)
                    r1 = sum(t * v for t, v in zip(t_vals, vals))
                    r2 = sum(t * t * v for t, v in zip(t_vals, vals))
                    c = ((r0 * (s2 * s4 - s3 * s3)
                          - s1 * (r1 * s4 - r2 * s3)
                          + s2 * (r1 * s3 - r2 * s2)) / det)
                    b = ((s0 * (r1 * s4 - r2 * s3)
                          - r0 * (s1 * s4 - s2 * s3)
                          + s2 * (s1 * r2 - s2 * r1)) / det)
                    a = ((s0 * (s2 * r2 - s3 * r1)
                          - s1 * (s1 * r2 - s2 * r1)
                          + r0 * (s1 * s3 - s2 * s2)) / det)
                    if coord == 'x':
                        ax, bx, cx_coeff = a, b, c
                    else:
                        ay, by, cy_coeff = a, b, c
                # 擬合殘差（fit residual）— 衡量拋物線擬合品質
                fit_residual = 0.0
                for fi_idx in range(n_fit):
                    t = t_vals[fi_idx]
                    pred_x = ax * t * t + bx * t + cx_coeff
                    pred_y = ay * t * t + by * t + cy_coeff
                    fit_residual += math.hypot(
                        x_vals[fi_idx] - pred_x,
                        y_vals[fi_idx] - pred_y)
                fit_residual /= n_fit  # 平均殘差 (px)
                # 外推到 t = -extra
                t_origin = -extra
                px = ax * t_origin ** 2 + bx * t_origin + cx_coeff
                py = ay * t_origin ** 2 + by * t_origin + cy_coeff
                parabolic_origin = (px, py)
                use_parabolic = True
        # 線性+重力+空氣阻力修正（fallback 或用於比較）
        # 空氣阻力：速度越快減速越多（簡化模型 drag_factor）
        _DRAG_FACTOR = 0.005  # 每幀速度衰減比例
        drag_scale = 1.0 / (1.0 + _DRAG_FACTOR * extra)  # 減速比例
        effective_vx = vx_avg / drag_scale  # 反推初始速度（更大）
        effective_vy = vy_avg / drag_scale
        gravity_correction = 0.5 * _GRAVITY_PX * extra * extra
        origin_x = x0 - effective_vx * extra
        origin_y = y0 - effective_vy * extra - gravity_correction
        # 如果拋物線擬合可用，取拋物線和線性的平均（穩健折衷）
        if use_parabolic and parabolic_origin:
            origin_x = (origin_x + parabolic_origin[0]) / 2
            origin_y = (origin_y + parabolic_origin[1]) / 2
        origin_frame = max(0, int(f0 - extra))

        ball_dir = None
        if speed_avg >= SCORE_MIN_VELOCITY_FOR_DIR:
            ball_dir = (vx_avg / speed_avg, vy_avg / speed_avg)

        # ── 多點驗證：沿外推路徑在多個距離檢查機器人 ──
        # 速度估算可能有誤差，在 100%/66%/33% 外推距離處都找機器人
        check_fracs = [1.0, 0.66, 0.33]
        best_per_label = {}  # label → (dist, score, cs, ally, n_checks)

        for frac in check_fracs:
            check_extra = extra * frac
            cx = x0 - vx_avg * check_extra
            cy = (y0 - vy_avg * check_extra
                  - 0.5 * _GRAVITY_PX * check_extra * check_extra)
            cf = max(0, int(f0 - check_extra))

            robot_pos = self._get_robot_positions_near_frame(
                robot_positions_by_frame, cf)
            if not robot_pos:
                continue

            for label, pos in robot_pos.items():
                rx, ry = pos[0], pos[1]
                # 機器人速度補償：用 ±2 幀估算機器人速度，
                # 反推它在球首次偵測幀的位置
                rpos_prev = robot_positions_by_frame.get(cf - 2, {})
                rpos_next = robot_positions_by_frame.get(cf + 2, {})
                if (label in rpos_prev and label in rpos_next):
                    rp_p = rpos_prev[label]
                    rp_n = rpos_next[label]
                    rvx = (rp_n[0] - rp_p[0]) / 4.0
                    rvy = (rp_n[1] - rp_p[1]) / 4.0
                    # 從 cf 回推到 f0 的時間差
                    dt_to_ball = f0 - cf
                    rx_adj = rx + rvx * dt_to_ball
                    ry_adj = ry + rvy * dt_to_ball
                    # 取補償後和原始位置中較近的
                    d_adj = math.hypot(cx - rx_adj, cy - ry_adj)
                    d_orig = math.hypot(cx - rx, cy - ry)
                    if d_adj < d_orig:
                        rx, ry = rx_adj, ry_adj
                dist = math.hypot(cx - rx, cy - ry)
                if dist > self._max_shooter_dist:
                    continue
                score = dist
                cs = 0.0
                if ball_dir:
                    first_x = traj_before[0][1]
                    first_y = traj_before[0][2]
                    rtb_x = first_x - rx
                    rtb_y = first_y - ry
                    rtb_len = math.hypot(rtb_x, rtb_y)
                    if rtb_len > 1:
                        cs = (rtb_x * ball_dir[0] +
                              rtb_y * ball_dir[1]) / rtb_len
                        # 非線性方向加權：正向 cs 用平方放大獎勵，
                        # 負向 cs 用絕對值懲罰
                        if cs > 0:
                            dir_factor = 1 - SCORE_DIRECTION_WEIGHT * cs * cs
                        else:
                            dir_factor = 1 + SCORE_DIRECTION_WEIGHT * abs(cs)
                        score = dist * dir_factor
                ally = self._alliance_matches(label, zone_alliance)
                if ally:
                    score *= SCORE_ALLIANCE_BIAS

                prev = best_per_label.get(label)
                if prev is None:
                    best_per_label[label] = (dist, score, cs, ally, 1)
                else:
                    n = prev[4] + 1
                    if score < prev[1]:
                        best_per_label[label] = (dist, score, cs, ally, n)
                    else:
                        best_per_label[label] = (*prev[:4], n)

        if not best_per_label:
            return ("未知", float('inf'), "extrapolation", 0.0)

        # 轉換為候選列表
        candidates = [(lbl, d, s, c, a)
                       for lbl, (d, s, c, a, _n)
                       in best_per_label.items()]
        best_label = "未知"
        best_score = float('inf')
        best_dist = float('inf')
        for label, dist, score, cs, ally in candidates:
            if score < best_score:
                best_score = score
                best_label = label
                best_dist = dist

        # Extrapolation 置信度
        conf = 0.50
        if best_dist <= 200:
            conf += 0.10
        if candidates:
            best_cs = max(c[3] for c in candidates if c[0] == best_label)
            if best_cs > 0.5:
                conf += 0.10
        if self._alliance_matches(best_label, zone_alliance):
            conf += 0.05
        # 多點確認加分 + 收斂驗證
        n_checks = best_per_label.get(best_label, (0, 0, 0, 0, 0))[4]
        if n_checks >= 3:
            conf += 0.10
            # 收斂驗證：多個候選中同一機器人一直最佳 → 額外加分
            # 其他候選的最佳 check 數
            other_max_checks = max(
                (v[4] for l, v in best_per_label.items()
                 if l != best_label), default=0)
            if n_checks >= other_max_checks + 2:
                conf += 0.03  # 明顯收斂
        elif n_checks >= 2:
            conf += 0.05
        # 速度剖面分類：根據射出類型調整信心
        # lob（拋物線）= 中速 + 明顯垂直分量 → 拋物線擬合更可靠
        # line drive = 高速 + 水平 → 線性外推更可靠
        # roller = 低速 + 水平 → 外推都不太可靠
        if speed_avg > 0:
            vy_ratio = abs(vy_avg) / speed_avg  # 垂直分量比
            if speed_avg >= self._shot_min_velocity * 1.2:
                if vy_ratio < 0.3:
                    # Line drive：高速水平 → 線性外推可靠
                    if not use_parabolic:
                        conf += 0.03
                else:
                    # 高速 lob → 拋物線擬合更好
                    if use_parabolic and fit_residual < 15:
                        conf += 0.03
            elif speed_avg < SCORE_MIN_VELOCITY_FOR_DIR:
                # Roller：低速球 → 外推不可靠
                conf -= 0.05
        # 擬合殘差信心：拋物線擬合品質好 → 外推更可靠
        if use_parabolic and fit_residual < 5:
            conf += 0.05  # 擬合殘差很小 (< 5px)
        elif use_parabolic and fit_residual > 30:
            conf -= 0.05  # 擬合品質差
        # 候選位置歷史驗證：機器人在外推時間段內是否持續在附近
        # 如果機器人在中間時段出現在球附近，更可信
        if best_label != "未知":
            history_near = 0
            history_checks = 0
            for t_check in range(origin_frame, int(f0), max(1, int(extra / 4))):
                rp = robot_positions_by_frame.get(t_check, {})
                if best_label in rp:
                    history_checks += 1
                    # 在外推路徑附近？
                    dt_from_f0 = f0 - t_check
                    path_x = x0 - vx_avg * dt_from_f0
                    path_y = y0 - vy_avg * dt_from_f0
                    pd = math.hypot(rp[best_label][0] - path_x,
                                    rp[best_label][1] - path_y)
                    if pd < self._max_shooter_dist * 0.8:
                        history_near += 1
            if history_checks >= 2 and history_near >= 2:
                conf += 0.05  # 機器人持續在路徑附近
        # Acceleration Profile Validation：射球後球應逐漸減速（空氣阻力）
        # 持續加速的球不太像正常射球（可能是偵測跳幀造成的假速度）
        if len(rel_speeds) >= 3:
            accel_count = sum(1 for k in range(1, len(rel_speeds))
                              if rel_speeds[k] > rel_speeds[k-1] * 1.1)
            decel_count = sum(1 for k in range(1, len(rel_speeds))
                              if rel_speeds[k] < rel_speeds[k-1] * 0.95)
            if accel_count >= len(rel_speeds) - 1 and decel_count == 0:
                conf -= 0.04  # 持續加速 = 異常
            elif decel_count >= len(rel_speeds) - 1:
                conf += 0.03  # 持續減速 = 典型射球
        # Trajectory Length Confidence：更多軌跡點 = 速度估計更準確
        n_traj_pts = len(traj_before)
        if n_traj_pts >= 10:
            conf += 0.04  # 豐富軌跡
        elif n_traj_pts <= 3:
            conf -= 0.06  # 極短軌跡，外推不可靠
        elif n_traj_pts <= 5:
            conf -= 0.03
        # Best-Dist-to-Origin Ratio：候選距離佔外推總距離的比例
        # 候選離原點越遠 → 外推越不精確
        extrap_total_dist = speed_avg * extra if speed_avg > 0 else 1
        if best_dist > 0 and extrap_total_dist > 0:
            dist_ratio = best_dist / extrap_total_dist
            if dist_ratio > 0.5:
                conf -= 0.06  # 候選在外推距離的一半以外
            elif dist_ratio < 0.15:
                conf += 0.03  # 候選非常接近原點
        # Origin Uncertainty：速度不確定性 → 原點位置不確定性半徑
        # 不確定性越大 → 候選匹配越不可靠
        if speed_avg > 0 and speed_cv > 0:
            # 不確定性半徑 ≈ speed_cv × speed × extra_frames
            origin_uncertainty = speed_cv * speed_avg * extra
            if origin_uncertainty > 300:
                conf -= 0.08  # 大不確定性
            elif origin_uncertainty > 150:
                conf -= 0.04
        # 速度一致性懲罰：CV > 0.5 代表速度變化大，外推不可靠
        if speed_cv > 0.8:
            conf -= 0.15
        elif speed_cv > 0.5:
            conf -= 0.08
        # 曲率懲罰：方向急變 > 45° 代表球可能反彈，線性外推不可靠
        if max_angle_change > 1.05:  # > 60°
            conf -= 0.15
        elif max_angle_change > 0.79:  # > 45°
            conf -= 0.08
        # 多候選分散度懲罰：前兩名分數接近 → 難以區分 → 降低置信
        if len(candidates) >= 2:
            sorted_scores = sorted(c[2] for c in candidates)
            if sorted_scores[0] > 0:
                gap_ratio = sorted_scores[1] / sorted_scores[0]
                if gap_ratio < 1.1:
                    conf -= 0.10  # 非常接近
                elif gap_ratio < 1.2:
                    conf -= 0.05  # 較接近
        # Extrap Candidate Spatial Spread：
        # 候選空間分散度高 = 外推原點不精確 → 降信心
        if len(candidates) >= 3:
            cand_positions = []
            for c_lbl, c_d, c_s, c_cs, c_ally in candidates:
                cp_rpos = self._get_robot_positions_near_frame(
                    robot_positions_by_frame, origin_frame)
                if c_lbl in cp_rpos:
                    cand_positions.append(cp_rpos[c_lbl])
            if len(cand_positions) >= 3:
                cp_cx = sum(p[0] for p in cand_positions) / len(cand_positions)
                cp_cy = sum(p[1] for p in cand_positions) / len(cand_positions)
                cp_spread = (sum(math.hypot(p[0] - cp_cx, p[1] - cp_cy)
                                 for p in cand_positions) / len(cand_positions))
                if cp_spread > 400:
                    conf -= 0.04  # 候選非常分散
                elif cp_spread < 100:
                    conf -= 0.02  # 候選密集（難以區分）
        # Extrap Alliance Tie-Breaker：候選分數接近時，偏好同聯盟
        if (len(candidates) >= 2 and zone_alliance
                and not self._alliance_matches(best_label, zone_alliance)):
            sorted_cands_ally = sorted(candidates, key=lambda c: c[2])
            if len(sorted_cands_ally) >= 2:
                top_score = sorted_cands_ally[0][2]
                for sc in sorted_cands_ally[1:]:
                    if (sc[4]  # is ally
                            and sc[2] < top_score * 1.15):
                        # 同聯盟候選分數在 15% 內 → 切換
                        best_label = sc[0]
                        best_dist = sc[1]
                        break
        # 候選分數熵：分數分布越均勻 → 歸因越不確定
        if len(candidates) >= 3:
            scores = [c[2] for c in candidates if c[2] > 0]
            if scores:
                total_s = sum(scores)
                # 反轉分數（低分=好），計算歸一化分布
                inv_scores = [total_s / s for s in scores]
                inv_total = sum(inv_scores)
                probs = [s / inv_total for s in inv_scores]
                entropy = -sum(p * math.log(p + 1e-10) for p in probs)
                max_entropy = math.log(len(probs))
                if max_entropy > 0:
                    norm_entropy = entropy / max_entropy
                    if norm_entropy > 0.9:
                        conf -= 0.06  # 高熵（分數非常均勻）
                    elif norm_entropy > 0.8:
                        conf -= 0.03
        # Extrap Gravity Direction Validation：
        # 拋物線球應有 vy < 0（向上）的初始階段
        # 如果球初始 vy > 0（向下）且用了重力修正 = 修正可能不合理
        if (use_parabolic and gravity_correction > 10
                and vy_avg > 0 and speed_avg > SCORE_MIN_VELOCITY_FOR_DIR):
            # 球初始就在往下飛卻加了向上重力修正 → 不合理
            conf = max(0.0, conf - 0.04)
        # Extrap Direction Consistency：外推方向 vs 球初始速度方向
        # 最佳候選應在球來的反方向上，否則物理上不合理
        if best_label != "未知" and ball_dir:
            cand_rpos = self._get_robot_positions_near_frame(
                robot_positions_by_frame, origin_frame)
            if best_label in cand_rpos:
                c_rx, c_ry = cand_rpos[best_label]
                # 候選→球首點方向
                c2b_x = traj_before[0][1] - c_rx
                c2b_y = traj_before[0][2] - c_ry
                c2b_len = math.hypot(c2b_x, c2b_y)
                if c2b_len > 5:
                    dir_cs = ((c2b_x * ball_dir[0] + c2b_y * ball_dir[1])
                              / c2b_len)
                    if dir_cs > 0.7:
                        conf = min(1.0, conf + 0.04)  # 方向高度一致
                    elif dir_cs < 0:
                        conf = max(0.0, conf - 0.06)  # 方向相反
        # 反向驗證：球前幾個軌跡點是否在遠離候選？
        # 如果球靠近候選（而非遠離），不太可能是射手
        if best_label != "未知" and len(traj_before) >= 3:
            rpos = self._get_robot_positions_near_frame(
                robot_positions_by_frame, int(traj_before[0][0]))
            if best_label in rpos:
                rx, ry = rpos[best_label][0], rpos[best_label][1]
                d0 = math.hypot(traj_before[0][1] - rx,
                                traj_before[0][2] - ry)
                d_last = math.hypot(
                    traj_before[min(3, len(traj_before) - 1)][1] - rx,
                    traj_before[min(3, len(traj_before) - 1)][2] - ry)
                if d_last < d0 - 20:  # 球在靠近候選，不是遠離
                    conf -= 0.10
                elif d_last > d0 + 30:  # 球在遠離候選（正常射出）
                    conf += 0.05
        conf = max(0.0, min(1.0, conf))

        if debug_frame and candidates:
            sorted_cands = sorted(candidates, key=lambda c: c[2])
            top = sorted_cands[:3]
            cands_str = ", ".join(
                f"{c[0]}={c[1]:.0f}px(s={c[2]:.0f}"
                f",dir={c[3]:.2f}"
                f"{',ally' if c[4] else ''})"
                for c in top)
            checks_str = (f", checks={n_checks}"
                          if n_checks > 1 else "")
            quality_parts = []
            if speed_cv > 0.3:
                quality_parts.append(f"cv={speed_cv:.2f}")
            if max_angle_change > 0.3:
                quality_parts.append(
                    f"curve={math.degrees(max_angle_change):.0f}°")
            quality_str = (f", quality=[{','.join(quality_parts)}]"
                           if quality_parts else "")
            grav_str = (f", grav={gravity_correction:.0f}px"
                        if gravity_correction > 1 else "")
            print(f"[DEBUG] 幀 {debug_frame} 軌跡外推: "
                  f"origin=({origin_x:.0f},{origin_y:.0f})"
                  f"@f{origin_frame}, "
                  f"traj={len(traj_before)}pts, "
                  f"候選: {cands_str}, "
                  f"conf={conf:.2f}{checks_str}"
                  f"{quality_str}{grav_str}")

        if best_dist > self._max_shooter_dist:
            return ("未知", best_dist, "extrapolation", 0.0)
        return (best_label, best_dist, "extrapolation", conf)

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

    def get_attribution_report(self):
        """結構化歸因報告（可用於 UI 或離線分析）。

        Returns:
            dict: {
                "total_goals": int,
                "method_counts": {method: count},
                "avg_confidence": float,
                "quality_score": int (0-100),
                "events": [{frame, shooter, method, conf, dist, zone}],
                "per_robot": {label: {goals, methods, avg_conf}},
            }
        """
        method_counts = {}
        per_robot = {}
        for e in self.events:
            m = e.attribution_method or "unknown"
            method_counts[m] = method_counts.get(m, 0) + 1
            r = per_robot.setdefault(e.shooter_label, {
                "goals": 0, "methods": {}, "confs": []})
            r["goals"] += 1
            r["methods"][m] = r["methods"].get(m, 0) + 1
            r["confs"].append(e.confidence)

        total = len(self.events)
        avg_conf = (sum(e.confidence for e in self.events) / total
                    if total else 0.0)
        unknown_count = method_counts.get("unknown", 0)
        known = total - unknown_count
        high_conf = sum(1 for e in self.events if e.confidence >= 0.6)
        strong = (method_counts.get("hp", 0)
                  + method_counts.get("trace", 0))
        ally_match = sum(
            1 for e in self.events
            if (e.alliance and e.shooter_label != "未知"
                and e.shooter_label.lower().startswith(e.alliance)))

        quality = 0
        if total:
            quality = int(
                (known / total) * 25
                + (high_conf / total) * 30
                + (strong / total) * 25
                + ((ally_match / known) if known else 0) * 20)

        robot_summary = {}
        for lbl, data in per_robot.items():
            robot_summary[lbl] = {
                "goals": data["goals"],
                "methods": data["methods"],
                "avg_conf": (sum(data["confs"]) / len(data["confs"])
                             if data["confs"] else 0),
            }

        # 確定性分級統計
        certainty_counts = {}
        for e in self.events:
            cl = e.certainty_level or "unknown"
            certainty_counts[cl] = certainty_counts.get(cl, 0) + 1

        # 聯盟不一致統計
        alliance_mismatches = sum(
            1 for e in self.events
            if (e.alliance and e.shooter_label != "未知"
                and not e.shooter_label.lower().startswith(e.alliance)))

        # Per-method quality breakdown
        method_quality = {}
        for m_name, m_count in method_counts.items():
            m_events = [e for e in self.events
                        if (e.attribution_method or "unknown") == m_name]
            if m_events:
                m_avg = sum(e.confidence for e in m_events) / len(m_events)
                m_high = sum(1 for e in m_events if e.confidence >= 0.6)
                m_ally = sum(
                    1 for e in m_events
                    if (e.alliance and e.shooter_label != "未知"
                        and e.shooter_label.lower().startswith(e.alliance)))
                method_quality[m_name] = {
                    "count": m_count,
                    "avg_confidence": round(m_avg, 3),
                    "high_confidence_ratio": round(
                        m_high / len(m_events), 3),
                    "alliance_match_ratio": round(
                        m_ally / len(m_events), 3)
                        if m_events else 0,
                }

        # Period breakdown：Auto vs Teleop 信心度差異
        period_stats = {}
        for e in self.events:
            p = e.period or "Unknown"
            ps = period_stats.setdefault(p, {"count": 0, "confs": []})
            ps["count"] += 1
            ps["confs"].append(e.confidence)
        period_summary = {}
        for p, ps in period_stats.items():
            period_summary[p] = {
                "count": ps["count"],
                "avg_confidence": round(
                    sum(ps["confs"]) / len(ps["confs"]), 3)
                    if ps["confs"] else 0,
            }

        return {
            "total_goals": total,
            "method_counts": method_counts,
            "method_quality": method_quality,
            "avg_confidence": round(avg_conf, 3),
            "quality_score": quality,
            "certainty_counts": certainty_counts,
            "alliance_mismatches": alliance_mismatches,
            "period_breakdown": period_summary,
            "events": [
                {
                    "frame": e.frame_idx,
                    "shooter": e.shooter_label,
                    "method": e.attribution_method,
                    "confidence": round(e.confidence, 3),
                    "certainty": e.certainty_level,
                    "distance": round(e.shooter_dist, 1),
                    "zone": e.zone_name,
                    "period": e.period,
                    "alliance": e.alliance,
                }
                for e in self.events
            ],
            "per_robot": robot_summary,
        }

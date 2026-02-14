"""
FRC Scoring Analyzer — 質心追蹤器 + 軌跡縫合（複用高度分析邏輯）
"""

import math

import numpy as np

from config import (
    HAS_SCIPY, MAX_MATCH_DIST, MAX_MISSED, VELOCITY_SMOOTH,
    AREA_WEIGHT, AREA_SCALE, VELOCITY_POINTS, MAX_AREA_RATIO,
    STITCH_AMBIGUITY_RATIO,
)

if HAS_SCIPY:
    from scipy.optimize import linear_sum_assignment


class CentroidTracker:
    """速度預測 + Hungarian 最佳匹配 + 複合成本質心追蹤器。"""

    def __init__(self, max_distance=MAX_MATCH_DIST, max_missed=MAX_MISSED):
        self.max_distance = max_distance
        self.max_missed = max_missed
        self.next_id = 0
        self.tracks = {}       # track_id -> {cx, cy, area, missed, vx, vy}
        self.trajectories = {} # track_id -> [(frame_idx, cx, cy, area), ...]

    def update(self, detections, frame_idx):
        """
        detections: [(cx, cy, area), ...] 或 [(cx, cy, area, radius), ...]
        回傳活躍追蹤 {track_id: (cx, cy, area)}
        """
        # 統一為 3-tuple
        dets = [(d[0], d[1], d[2]) for d in detections]

        if not self.tracks:
            for (cx, cy, area) in dets:
                self._register(cx, cy, area, frame_idx)
            return {tid: (t["cx"], t["cy"], t["area"])
                    for tid, t in self.tracks.items()}

        if not dets:
            to_remove = []
            for tid in self.tracks:
                t = self.tracks[tid]
                t["missed"] += 1
                t["cx"] += t["vx"]
                t["cy"] += t["vy"]
                if t["missed"] > self.max_missed:
                    to_remove.append(tid)
            for tid in to_remove:
                del self.tracks[tid]
            return {tid: (t["cx"], t["cy"], t["area"])
                    for tid, t in self.tracks.items()}

        track_ids = list(self.tracks.keys())
        n_tracks = len(track_ids)
        n_dets = len(dets)

        # 用速度預測下一幀位置
        predicted = []
        for tid in track_ids:
            t = self.tracks[tid]
            px = t["cx"] + t["vx"]
            py = t["cy"] + t["vy"]
            predicted.append((px, py, t["area"]))

        # 複合成本矩陣
        cost_matrix = []
        for ti in range(n_tracks):
            row = []
            for di in range(n_dets):
                spatial_dist = math.hypot(predicted[ti][0] - dets[di][0],
                                          predicted[ti][1] - dets[di][1])
                area_diff = abs(predicted[ti][2] - dets[di][2])
                cost = spatial_dist + AREA_WEIGHT * area_diff / AREA_SCALE
                row.append(cost)
            cost_matrix.append(row)

        # Hungarian 或 greedy 配對
        matches = []
        used_tracks = set()
        used_dets = set()

        if HAS_SCIPY and n_tracks > 0 and n_dets > 0:
            cost_arr = np.array(cost_matrix)
            row_ind, col_ind = linear_sum_assignment(cost_arr)
            for ti, di in zip(row_ind, col_ind):
                if cost_matrix[ti][di] <= self.max_distance:
                    matches.append((ti, di))
                    used_tracks.add(ti)
                    used_dets.add(di)
        else:
            pairs = []
            for ti in range(n_tracks):
                for di in range(n_dets):
                    pairs.append((cost_matrix[ti][di], ti, di))
            pairs.sort()
            for cost, ti, di in pairs:
                if ti in used_tracks or di in used_dets:
                    continue
                if cost > self.max_distance:
                    break
                matches.append((ti, di))
                used_tracks.add(ti)
                used_dets.add(di)

        # 更新匹配的追蹤
        for ti, di in matches:
            tid = track_ids[ti]
            cx, cy, area = dets[di]
            old_cx = self.tracks[tid]["cx"]
            old_cy = self.tracks[tid]["cy"]
            inst_vx = cx - old_cx
            inst_vy = cy - old_cy
            alpha = VELOCITY_SMOOTH
            self.tracks[tid]["vx"] = alpha * self.tracks[tid]["vx"] + (1 - alpha) * inst_vx
            self.tracks[tid]["vy"] = alpha * self.tracks[tid]["vy"] + (1 - alpha) * inst_vy
            self.tracks[tid]["cx"] = cx
            self.tracks[tid]["cy"] = cy
            self.tracks[tid]["area"] = area
            self.tracks[tid]["missed"] = 0
            self.trajectories[tid].append((frame_idx, cx, cy, area))

        # 未匹配追蹤
        to_remove = []
        for ti in range(n_tracks):
            if ti not in used_tracks:
                tid = track_ids[ti]
                t = self.tracks[tid]
                t["missed"] += 1
                t["cx"] += t["vx"]
                t["cy"] += t["vy"]
                if t["missed"] > self.max_missed:
                    to_remove.append(tid)
        for tid in to_remove:
            del self.tracks[tid]

        # 新偵測
        for di in range(n_dets):
            if di not in used_dets:
                cx, cy, area = dets[di]
                self._register(cx, cy, area, frame_idx)

        return {tid: (t["cx"], t["cy"], t["area"])
                for tid, t in self.tracks.items()}

    def _register(self, cx, cy, area, frame_idx):
        self.tracks[self.next_id] = {
            "cx": cx, "cy": cy, "area": area,
            "missed": 0, "vx": 0.0, "vy": 0.0
        }
        self.trajectories[self.next_id] = [(frame_idx, cx, cy, area)]
        self.next_id += 1


def stitch_trajectories(trajectories, max_frame_gap=15, max_pred_dist=120):
    """
    合併斷裂的軌跡：同一顆球因追蹤丟失而被分成多段。
    使用末端速度預測 + 面積相似度 + 歧義偵測。
    """
    changed = True
    while changed:
        changed = False

        track_ends = {}
        track_starts = {}

        for tid, traj in trajectories.items():
            if len(traj) < 1:
                continue

            n_start = min(VELOCITY_POINTS, len(traj))
            start_avg_area = sum(p[3] for p in traj[:n_start]) / n_start
            track_starts[tid] = (traj[0][0], traj[0][1], traj[0][2], start_avg_area)

            n_end = min(VELOCITY_POINTS, len(traj))
            tail = traj[-n_end:]
            end_avg_area = sum(p[3] for p in tail) / n_end

            if len(tail) >= 2:
                last_f, last_x, last_y = tail[-1][0], tail[-1][1], tail[-1][2]
                first_f, first_x, first_y = tail[0][0], tail[0][1], tail[0][2]
                dt = last_f - first_f
                if dt > 0:
                    vx = (last_x - first_x) / dt
                    vy = (last_y - first_y) / dt
                else:
                    vx, vy = 0, 0
                track_ends[tid] = (last_f, last_x, last_y, vx, vy, end_avg_area)
            else:
                f, x, y = traj[0][0], traj[0][1], traj[0][2]
                track_ends[tid] = (f, x, y, 0, 0, end_avg_area)

        best_merge = None
        best_score = float('inf')
        second_best_score = float('inf')

        for tid1, (lf, lx, ly, vx, vy, end_area) in track_ends.items():
            for tid2, (sf, sx, sy, start_area) in track_starts.items():
                if tid1 == tid2:
                    continue
                gap = sf - lf
                if gap < 1 or gap > max_frame_gap:
                    continue

                if end_area > 0 and start_area > 0:
                    area_ratio = max(end_area, start_area) / min(end_area, start_area)
                    if area_ratio > MAX_AREA_RATIO:
                        continue

                pred_x = lx + vx * gap
                pred_y = ly + vy * gap
                dist = math.hypot(pred_x - sx, pred_y - sy)

                if dist < max_pred_dist:
                    if dist < best_score:
                        second_best_score = best_score
                        best_score = dist
                        best_merge = (tid1, tid2)
                    elif dist < second_best_score:
                        second_best_score = dist

        if best_merge and second_best_score < best_score * STITCH_AMBIGUITY_RATIO:
            best_merge = None

        if best_merge:
            tid1, tid2 = best_merge
            trajectories[tid1] = trajectories[tid1] + trajectories[tid2]
            del trajectories[tid2]
            changed = True

    return trajectories

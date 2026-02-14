"""
FRC Scoring Analyzer — 機器人追蹤（直方圖反投影 + CamShift）
"""

import cv2
import numpy as np

from config import (
    ROBOT_SEARCH_MARGIN, ROBOT_MIN_CONFIDENCE,
    ROBOT_MAX_LOST_FRAMES, ROBOT_HIST_BINS,
)
from geometry import clamp_rect, rect_center


class RobotTrack:
    """單一機器人的追蹤資料。"""

    def __init__(self, label, bbox, frame, frame_idx):
        """
        Args:
            label: 機器人標籤（如 "6998"）
            bbox: (x, y, w, h) 初始 bounding box
            frame: BGR 影像（用於提取直方圖）
            frame_idx: 起始幀索引
        """
        self.label = label
        self.bbox = bbox  # (x, y, w, h)
        self.lost_frames = 0
        self.is_lost = False
        self.positions = []    # [(frame_idx, cx, cy), ...]
        self.bbox_history = [] # [(frame_idx, x, y, w, h), ...]

        # 提取 HSV 直方圖（色彩模型）
        x, y, w, h = bbox
        roi = frame[y:y+h, x:x+w]
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # 使用 H 和 S 通道建立 2D 直方圖
        self.hist = cv2.calcHist(
            [hsv_roi], [0, 1], None,
            [ROBOT_HIST_BINS, ROBOT_HIST_BINS],
            [0, 180, 0, 256]
        )
        cv2.normalize(self.hist, self.hist, 0, 255, cv2.NORM_MINMAX)

        # 保存初始模板
        self.template = roi.copy()
        self.template_size = (w, h)

        # 記錄初始位置
        cx, cy = rect_center(x, y, w, h)
        self.positions.append((frame_idx, cx, cy))
        self.bbox_history.append((frame_idx, x, y, w, h))

    def update(self, frame, frame_idx):
        """
        追蹤此機器人在新幀中的位置。

        Returns:
            (x, y, w, h) 更新後的 bbox，或 None（丟失）
        """
        if self.is_lost:
            return None

        h_frame, w_frame = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 直方圖反投影
        back_proj = cv2.calcBackProject(
            [hsv], [0, 1], self.hist,
            [0, 180, 0, 256], 1
        )

        # 在前一位置周圍建立搜尋窗口
        x, y, w, h = self.bbox
        margin = ROBOT_SEARCH_MARGIN
        search_x = max(0, x - margin)
        search_y = max(0, y - margin)
        search_w = min(w + 2 * margin, w_frame - search_x)
        search_h = min(h + 2 * margin, h_frame - search_y)

        # CamShift 追蹤（使用反投影圖）
        track_window = (x, y, w, h)
        term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

        try:
            ret, track_window = cv2.CamShift(
                back_proj, track_window, term_crit
            )
            x, y, w, h = track_window

            # 確保 bbox 在影像範圍內
            x, y, w, h = clamp_rect(x, y, max(w, 20), max(h, 20),
                                     w_frame, h_frame)
        except cv2.error:
            self.lost_frames += 1
            if self.lost_frames > ROBOT_MAX_LOST_FRAMES:
                self.is_lost = True
            return None

        # 計算信心分數（搜尋區域內反投影強度比）
        roi_bp = back_proj[y:y+h, x:x+w]
        if roi_bp.size > 0:
            confidence = np.mean(roi_bp) / 255.0
        else:
            confidence = 0

        if confidence < ROBOT_MIN_CONFIDENCE:
            self.lost_frames += 1
            if self.lost_frames > ROBOT_MAX_LOST_FRAMES:
                self.is_lost = True
            # 仍然更新位置（低信心但未完全丟失）
            cx, cy = rect_center(x, y, w, h)
            self.positions.append((frame_idx, cx, cy))
            self.bbox_history.append((frame_idx, x, y, w, h))
            self.bbox = (x, y, w, h)
            return (x, y, w, h)

        # 追蹤成功
        self.lost_frames = 0
        self.bbox = (x, y, w, h)

        cx, cy = rect_center(x, y, w, h)
        self.positions.append((frame_idx, cx, cy))
        self.bbox_history.append((frame_idx, x, y, w, h))

        return (x, y, w, h)

    def get_position_at(self, frame_idx):
        """取得指定幀的機器人位置，若無則回傳 None。"""
        for f, cx, cy in self.positions:
            if f == frame_idx:
                return (cx, cy)
        return None

    def get_nearest_position(self, frame_idx, max_gap=5):
        """取得最接近指定幀的機器人位置。"""
        best = None
        best_gap = float('inf')
        for f, cx, cy in self.positions:
            gap = abs(f - frame_idx)
            if gap < best_gap:
                best_gap = gap
                best = (cx, cy)
        if best_gap <= max_gap:
            return best
        return None


class RobotTrackerManager:
    """管理多台機器人的追蹤。"""

    def __init__(self):
        self.robots = {}  # label -> RobotTrack

    def add_robot(self, label, bbox, frame, frame_idx):
        """
        新增一台機器人追蹤。

        Args:
            label: 機器人標籤（如 "6998"）
            bbox: (x, y, w, h) 初始 bounding box
            frame: BGR 影像
            frame_idx: 起始幀索引
        """
        self.robots[label] = RobotTrack(label, bbox, frame, frame_idx)

    def remove_robot(self, label):
        """移除一台機器人追蹤。"""
        if label in self.robots:
            del self.robots[label]

    def update_all(self, frame, frame_idx):
        """
        更新所有機器人在新幀中的位置。

        Returns:
            {label: (x, y, w, h) or None}
        """
        results = {}
        for label, robot in self.robots.items():
            results[label] = robot.update(frame, frame_idx)
        return results

    def get_all_positions(self, frame_idx):
        """取得所有機器人在指定幀的位置。"""
        positions = {}
        for label, robot in self.robots.items():
            pos = robot.get_position_at(frame_idx)
            if pos:
                positions[label] = pos
        return positions

    def clear(self):
        """清除所有機器人追蹤。"""
        self.robots.clear()

    @property
    def count(self):
        return len(self.robots)

    @property
    def labels(self):
        return list(self.robots.keys())

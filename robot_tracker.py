"""
FRC Scoring Analyzer — 機器人追蹤（VitTrack / CSRT + 模板匹配恢復）
"""

import os

import cv2
import numpy as np

from config import (
    ROBOT_MAX_LOST_FRAMES, ROBOT_TRACKER_TYPE,
    VITTRACK_MODEL_PATH, VITTRACK_SCORE_THRESHOLD,
)
from geometry import rect_center

# 模板匹配恢復參數
_RECOVER_THRESHOLD = 0.35   # 模板匹配最低相關係數
_SEARCH_MARGIN_MULT = 3     # 搜尋區域 = bbox 尺寸 × 此倍數
_TEMPLATE_UPDATE_INTERVAL = 30  # 每 N 幀更新模板


def _resolve_vittrack_path() -> str:
    """解析 VitTrack ONNX 模型的絕對路徑。"""
    if os.path.isabs(VITTRACK_MODEL_PATH):
        return VITTRACK_MODEL_PATH
    # 相對於專案根目錄
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, VITTRACK_MODEL_PATH)


def _create_tracker(tracker_type=ROBOT_TRACKER_TYPE):
    """工廠函式：建立追蹤器（VIT → CSRT → KCF → MIL fallback）。"""

    # VitTrack（需要 OpenCV 4.9+）
    if tracker_type == "VIT":
        model_path = _resolve_vittrack_path()
        if os.path.isfile(model_path):
            try:
                params = cv2.TrackerVit.Params()
                params.net = model_path
                tracker = cv2.TrackerVit.create(params)
                return tracker
            except AttributeError:
                print("[WARN] cv2.TrackerVit 不可用，回退到 CSRT")
        else:
            print(f"[WARN] VitTrack 模型不存在: {model_path}，回退到 CSRT")

    # 傳統追蹤器 fallback chain
    factories = {
        "CSRT": lambda: cv2.TrackerCSRT.create(),
        "KCF": lambda: cv2.TrackerKCF.create(),
        "MIL": lambda: cv2.TrackerMIL.create(),
    }

    # Try requested type (if not VIT)
    if tracker_type in factories:
        try:
            return factories[tracker_type]()
        except AttributeError:
            pass

    # Fallback chain
    for name in ["CSRT", "KCF", "MIL"]:
        try:
            return factories[name]()
        except AttributeError:
            continue

    raise RuntimeError("No suitable tracker available. Install opencv-contrib-python.")


class RobotTrack:
    """單一機器人的追蹤資料（CSRT + 模板匹配恢復）。"""

    def __init__(self, label, bbox, frame, frame_idx, alliance=""):
        """
        Args:
            label: 機器人標籤（如 "6998"）
            bbox: (x, y, w, h) 初始 bounding box
            frame: BGR 影像
            frame_idx: 起始幀索引
            alliance: 聯盟（"red" 或 "blue"）
        """
        self.label = label
        self.alliance = alliance
        self.bbox = bbox  # (x, y, w, h)
        self.lost_frames = 0
        self.is_lost = False
        self.positions = []    # [(frame_idx, cx, cy), ...]
        self.bbox_history = [] # [(frame_idx, x, y, w, h), ...]

        # 模板（用於恢復追蹤）
        x, y, w, h = bbox
        roi = frame[y:y+h, x:x+w]
        self._init_template = roi.copy()       # 初始模板（不變）
        self.template = roi.copy()             # 動態更新的模板
        self.template_size = (w, h)
        self._last_template_update = frame_idx

        # 建立追蹤器並初始化
        self._tracker = _create_tracker()
        self._is_vittrack = hasattr(self._tracker, "getTrackingScore")
        self._tracker.init(frame, bbox)

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
        # 即使已標記丟失，仍嘗試模板匹配恢復
        if self.is_lost:
            return self._try_recover(frame, frame_idx)

        success, box = self._tracker.update(frame)

        # VitTrack 獨有：信心分數（0~1）— 低於閾值主動觸發恢復
        if success and self._is_vittrack:
            score = self._tracker.getTrackingScore()
            if score < VITTRACK_SCORE_THRESHOLD:
                success = False

        if success:
            x, y, w, h = [int(v) for v in box]
            # 邊界合法性檢查
            fh, fw = frame.shape[:2]
            if x < 0 or y < 0 or x + w > fw or y + h > fh or w < 5 or h < 5:
                success = False
            else:
                self.lost_frames = 0
                self.bbox = (x, y, w, h)

                cx, cy = rect_center(x, y, w, h)
                self.positions.append((frame_idx, cx, cy))
                self.bbox_history.append((frame_idx, x, y, w, h))

                # 定期更新模板
                if frame_idx - self._last_template_update >= _TEMPLATE_UPDATE_INTERVAL:
                    self.template = frame[y:y+h, x:x+w].copy()
                    self._last_template_update = frame_idx

                return (x, y, w, h)

        # 追蹤失敗 — 嘗試模板匹配恢復
        recovered = self._try_recover(frame, frame_idx)
        if recovered:
            return recovered

        self.lost_frames += 1
        if self.lost_frames > ROBOT_MAX_LOST_FRAMES:
            self.is_lost = True
        return None

    def _try_recover(self, frame, frame_idx):
        """嘗試使用模板匹配在 bbox 附近重新找到機器人。"""
        fh, fw = frame.shape[:2]
        lx, ly, lw, lh = self.bbox
        margin = max(lw, lh) * _SEARCH_MARGIN_MULT

        # 定義搜尋區域
        sx = max(0, int(lx - margin))
        sy = max(0, int(ly - margin))
        ex = min(fw, int(lx + lw + margin))
        ey = min(fh, int(ly + lh + margin))

        search_region = frame[sy:ey, sx:ex]
        th, tw = self.template.shape[:2]

        if search_region.shape[0] < th or search_region.shape[1] < tw:
            return None

        # 嘗試動態模板
        result = cv2.matchTemplate(search_region, self.template,
                                   cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        # 如果動態模板信心不足，再試初始模板
        if max_val < _RECOVER_THRESHOLD:
            init_th, init_tw = self._init_template.shape[:2]
            if search_region.shape[0] >= init_th and search_region.shape[1] >= init_tw:
                result2 = cv2.matchTemplate(search_region, self._init_template,
                                            cv2.TM_CCOEFF_NORMED)
                _, max_val2, _, max_loc2 = cv2.minMaxLoc(result2)
                if max_val2 > max_val:
                    max_val, max_loc = max_val2, max_loc2
                    tw, th = init_tw, init_th

        if max_val < _RECOVER_THRESHOLD:
            return None

        # 恢復成功
        mx, my = max_loc
        new_x = sx + mx
        new_y = sy + my
        new_bbox = (new_x, new_y, tw, th)

        # 重建追蹤器
        self._tracker = _create_tracker()
        self._is_vittrack = hasattr(self._tracker, "getTrackingScore")
        self._tracker.init(frame, new_bbox)

        self.lost_frames = 0
        self.is_lost = False
        self.bbox = new_bbox

        # 更新模板
        self.template = frame[new_y:new_y+th, new_x:new_x+tw].copy()
        self._last_template_update = frame_idx

        cx, cy = rect_center(new_x, new_y, tw, th)
        self.positions.append((frame_idx, cx, cy))
        self.bbox_history.append((frame_idx, new_x, new_y, tw, th))

        return new_bbox

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

    def add_robot(self, label, bbox, frame, frame_idx, alliance=""):
        """
        新增一台機器人追蹤。

        Args:
            label: 機器人標籤（如 "6998"）
            bbox: (x, y, w, h) 初始 bounding box
            frame: BGR 影像
            frame_idx: 起始幀索引
            alliance: 聯盟（"red" 或 "blue"）
        """
        self.robots[label] = RobotTrack(label, bbox, frame, frame_idx, alliance)

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
        """取得所有機器人在指定幀的位置（精確匹配）。"""
        positions = {}
        for label, robot in self.robots.items():
            pos = robot.get_position_at(frame_idx)
            if pos:
                positions[label] = pos
        return positions

    def get_all_display_positions(self, frame_idx, max_gap=15):
        """取得所有機器人在指定幀的顯示位置（含最近已知位置回退）。"""
        positions = {}
        for label, robot in self.robots.items():
            pos = robot.get_position_at(frame_idx)
            if not pos:
                pos = robot.get_nearest_position(frame_idx, max_gap=max_gap)
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

"""
FRC Scoring Analyzer — 機器人追蹤

主要模式: YOLO 偵測 + ByteTrack 多目標追蹤 (MOT)
備用模式: VitTrack / CSRT 單目標追蹤 (SOT)
"""

import os

import cv2
import numpy as np

from config import (
    ROBOT_MAX_LOST_FRAMES, ROBOT_TRACKER_TYPE,
    VITTRACK_MODEL_PATH, VITTRACK_SCORE_THRESHOLD,
    BYTETRACK_TRACK_THRESH, BYTETRACK_LOST_BUFFER,
    BYTETRACK_MATCH_THRESH, BYTETRACK_MIN_CONSECUTIVE,
)
from geometry import rect_center

try:
    import supervision as sv
    HAS_SUPERVISION = True
except ImportError:
    HAS_SUPERVISION = False

# ═══════════════════════════════════════════════════════
# MOT: YOLO 偵測 + ByteTrack 多目標追蹤
# ═══════════════════════════════════════════════════════


class _MOTTracker:
    """Detection-based multi-object tracking (YOLO + ByteTrack)。"""

    def __init__(self, detector, fps: float = 30.0):
        """
        Args:
            detector: RobotDetectorONNX 實例
            fps: 影片 FPS
        """
        self._detector = detector
        self._fps = fps

        self._tracker = sv.ByteTrack(
            track_activation_threshold=BYTETRACK_TRACK_THRESH,
            lost_track_buffer=BYTETRACK_LOST_BUFFER,
            minimum_matching_threshold=BYTETRACK_MATCH_THRESH,
            frame_rate=int(fps),
            minimum_consecutive_frames=BYTETRACK_MIN_CONSECUTIVE,
        )

        # Label 映射: {bytetrack_tracker_id: (label, alliance)}
        self._label_map: dict[int, tuple[str, str]] = {}

        # 用戶標記等待匹配
        self._pending_markers: list[tuple] = []
        # [(label, alliance, bbox_xywh, mark_frame)]

        # 機器人資訊（標記順序記錄）
        self._robot_info: dict[str, dict] = {}
        # {label: {"alliance": str, "mark_frame": int}}

        # 位置記錄: {label: [(frame_idx, cx, cy), ...]}
        self._positions: dict[str, list[tuple]] = {}

        # Bbox 記錄: {label: [(frame_idx, x1, y1, x2, y2), ...]}
        self._bboxes: dict[str, list[tuple]] = {}

        # 用於快速查詢的幀索引: {frame_idx: {label: (cx, cy)}}
        self._frame_positions: dict[int, dict[str, tuple]] = {}

        # Bbox 幀索引: {frame_idx: {label: (x1, y1, x2, y2)}}
        self._frame_bboxes: dict[int, dict[str, tuple]] = {}

        # 自動偵測模式（不需用戶標記）
        self._auto_mode = False

    def enable_auto_mode(self):
        """啟用自動偵測模式（不需用戶標記）。"""
        self._auto_mode = True

    def add_robot(self, label: str, bbox_xywh: tuple, frame: np.ndarray,
                  frame_idx: int, alliance: str = ""):
        """註冊用戶標記的機器人。"""
        self._pending_markers.append((label, alliance, bbox_xywh, frame_idx))
        self._robot_info[label] = {
            "alliance": alliance,
            "mark_frame": frame_idx,
        }
        self._positions.setdefault(label, [])
        self._bboxes.setdefault(label, [])

    def update_all(self, frame: np.ndarray, frame_idx: int) -> dict:
        """
        偵測 + 追蹤所有機器人。

        Returns:
            {label: (x, y, w, h) or None} — 相容舊 API
        """
        # 1. YOLO 偵測
        raw_dets = self._detector.detect(frame)

        # 2. 建構 sv.Detections
        if raw_dets:
            xyxy = np.array([[d[0], d[1], d[2], d[3]] for d in raw_dets],
                            dtype=np.float32)
            conf = np.array([d[4] for d in raw_dets], dtype=np.float32)
            cls = np.array([d[5] for d in raw_dets], dtype=int)
            detections = sv.Detections(
                xyxy=xyxy, confidence=conf, class_id=cls)
        else:
            detections = sv.Detections.empty()

        # 3. ByteTrack 更新
        tracked = self._tracker.update_with_detections(detections)

        # 4. 匹配待定標記
        self._match_pending_markers(tracked, frame_idx)

        # 4.5. 自動模式：未標記的 tracker_id 自動分配 label
        if self._auto_mode and tracked.tracker_id is not None:
            for i, tid in enumerate(tracked.tracker_id):
                tid = int(tid)
                if tid in self._label_map:
                    continue
                class_id = int(tracked.class_id[i]) if tracked.class_id is not None else -1
                alliance = self._detector.infer_alliance(class_id) if class_id >= 0 else ""
                if alliance == "red":
                    label = f"Red-{tid}"
                elif alliance == "blue":
                    label = f"Blue-{tid}"
                else:
                    label = f"Robot-{tid}"
                self._label_map[tid] = (label, alliance)
                self._robot_info[label] = {"alliance": alliance, "mark_frame": -1}
                self._positions.setdefault(label, [])
                self._bboxes.setdefault(label, [])
                print(f"[INFO] MOT 自動偵測: {label} (tracker_id={tid})")

        # 5. 記錄已標記機器人的位置
        results = {}
        frame_pos = {}
        frame_bbox = {}

        if tracked.tracker_id is not None and len(tracked.tracker_id) > 0:
            for i, tid in enumerate(tracked.tracker_id):
                tid = int(tid)
                if tid not in self._label_map:
                    continue

                label, alliance = self._label_map[tid]
                x1, y1, x2, y2 = tracked.xyxy[i]
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                w = x2 - x1
                h = y2 - y1

                self._positions[label].append((frame_idx, cx, cy))
                self._bboxes[label].append(
                    (frame_idx, float(x1), float(y1), float(x2), float(y2)))

                frame_pos[label] = (cx, cy)
                frame_bbox[label] = (float(x1), float(y1),
                                     float(x2), float(y2))
                results[label] = (int(x1), int(y1), int(w), int(h))

        self._frame_positions[frame_idx] = frame_pos
        self._frame_bboxes[frame_idx] = frame_bbox

        # 未偵測到的機器人回傳 None
        for label in self._robot_info:
            if label not in results:
                results[label] = None

        return results

    def _match_pending_markers(self, tracked, frame_idx: int):
        """將用戶標記的機器人 bbox 與 ByteTrack 追蹤 ID 匹配。"""
        remaining = []

        for label, alliance, bbox_xywh, mark_frame in self._pending_markers:
            if frame_idx < mark_frame:
                remaining.append((label, alliance, bbox_xywh, mark_frame))
                continue

            # 只在標記幀嘗試匹配（及之後幾幀作為容錯）
            if frame_idx > mark_frame + 5:
                # 超時未匹配 — 丟棄
                print(f"[WARN] 機器人 {label} 無法在幀 {mark_frame} "
                      f"附近匹配 YOLO 偵測，請檢查標記位置")
                continue

            x, y, w, h = bbox_xywh
            marker_xyxy = np.array([x, y, x + w, y + h], dtype=np.float32)

            best_iou = 0.0
            best_tid = None

            if tracked.tracker_id is not None and len(tracked.tracker_id) > 0:
                for i, tid in enumerate(tracked.tracker_id):
                    tid = int(tid)
                    if tid in self._label_map:
                        continue  # 已分配
                    det_xyxy = tracked.xyxy[i]
                    iou = self._compute_iou(marker_xyxy, det_xyxy)
                    if iou > best_iou:
                        best_iou = iou
                        best_tid = tid

            if best_tid is not None and best_iou > 0.1:
                self._label_map[best_tid] = (label, alliance)
                print(f"[INFO] 機器人 {label} 匹配 ByteTrack ID {best_tid}"
                      f" (IoU={best_iou:.2f})")
            else:
                remaining.append((label, alliance, bbox_xywh, mark_frame))

        self._pending_markers = remaining

    @staticmethod
    def _compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
        """計算兩個 xyxy bbox 的 IoU。"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter

        return inter / union if union > 0 else 0.0

    def get_all_positions(self, frame_idx: int) -> dict[str, tuple]:
        """取得所有機器人在指定幀的位置（精確匹配）。"""
        return dict(self._frame_positions.get(frame_idx, {}))

    def get_all_display_positions(self, frame_idx: int,
                                  max_gap: int = 15) -> dict[str, tuple]:
        """取得所有機器人的顯示位置（含最近已知回退）。"""
        positions = self.get_all_positions(frame_idx)

        for label in self._robot_info:
            if label in positions:
                continue
            # 向前搜尋最近已知位置
            pos_list = self._positions.get(label, [])
            best_pos = None
            best_gap = float('inf')
            for f, cx, cy in pos_list:
                gap = abs(f - frame_idx)
                if gap < best_gap:
                    best_gap = gap
                    best_pos = (cx, cy)
            if best_pos and best_gap <= max_gap:
                positions[label] = best_pos

        return positions

    def get_all_bboxes(self, frame_idx: int) -> dict[str, tuple]:
        """取得所有機器人在指定幀的 bbox (x1, y1, x2, y2)。"""
        return dict(self._frame_bboxes.get(frame_idx, {}))

    def interpolate_positions(self):
        """對所有機器人的丟失幀做線性位置插值。"""
        for label in list(self._positions.keys()):
            pos_list = self._positions[label]
            if len(pos_list) < 2:
                continue

            pos_list.sort(key=lambda p: p[0])
            interpolated = []

            for i in range(len(pos_list) - 1):
                f1, cx1, cy1 = pos_list[i]
                f2, cx2, cy2 = pos_list[i + 1]

                interpolated.append(pos_list[i])

                gap = f2 - f1
                if gap > 1:
                    for f in range(f1 + 1, f2):
                        t = (f - f1) / gap
                        cx = cx1 + (cx2 - cx1) * t
                        cy = cy1 + (cy2 - cy1) * t
                        interpolated.append((f, cx, cy))

            interpolated.append(pos_list[-1])
            self._positions[label] = interpolated

            # 更新 frame_positions 快取
            for f, cx, cy in interpolated:
                if f not in self._frame_positions:
                    self._frame_positions[f] = {}
                if label not in self._frame_positions[f]:
                    self._frame_positions[f][label] = (cx, cy)

        # 同樣插值 bboxes
        for label in list(self._bboxes.keys()):
            bbox_list = self._bboxes[label]
            if len(bbox_list) < 2:
                continue

            bbox_list.sort(key=lambda p: p[0])
            interpolated_bb = []

            for i in range(len(bbox_list) - 1):
                f1, x1a, y1a, x2a, y2a = bbox_list[i]
                f2, x1b, y1b, x2b, y2b = bbox_list[i + 1]

                interpolated_bb.append(bbox_list[i])

                gap = f2 - f1
                if gap > 1:
                    for f in range(f1 + 1, f2):
                        t = (f - f1) / gap
                        interpolated_bb.append((
                            f,
                            x1a + (x1b - x1a) * t,
                            y1a + (y1b - y1a) * t,
                            x2a + (x2b - x2a) * t,
                            y2a + (y2b - y2a) * t,
                        ))

            interpolated_bb.append(bbox_list[-1])
            self._bboxes[label] = interpolated_bb

            for f, x1, y1, x2, y2 in interpolated_bb:
                if f not in self._frame_bboxes:
                    self._frame_bboxes[f] = {}
                if label not in self._frame_bboxes[f]:
                    self._frame_bboxes[f][label] = (x1, y1, x2, y2)

    def clear(self):
        self._label_map.clear()
        self._pending_markers.clear()
        self._robot_info.clear()
        self._positions.clear()
        self._bboxes.clear()
        self._frame_positions.clear()
        self._frame_bboxes.clear()

    @property
    def count(self) -> int:
        return len(self._robot_info)

    @property
    def labels(self) -> list[str]:
        return list(self._robot_info.keys())


# ═══════════════════════════════════════════════════════
# SOT: VitTrack / CSRT 單目標追蹤（備用模式）
# ═══════════════════════════════════════════════════════

_RECOVER_THRESHOLD = 0.35
_SEARCH_MARGIN_MULT = 3
_TEMPLATE_UPDATE_INTERVAL = 30


def _resolve_vittrack_path() -> str:
    if os.path.isabs(VITTRACK_MODEL_PATH):
        return VITTRACK_MODEL_PATH
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, VITTRACK_MODEL_PATH)


def _create_sot_tracker(tracker_type=ROBOT_TRACKER_TYPE):
    """建立 SOT 追蹤器（VIT → CSRT → KCF → MIL fallback）。"""
    if tracker_type == "VIT":
        model_path = _resolve_vittrack_path()
        if os.path.isfile(model_path):
            try:
                params = cv2.TrackerVit.Params()
                params.net = model_path
                return cv2.TrackerVit.create(params)
            except AttributeError:
                print("[WARN] cv2.TrackerVit 不可用，回退到 CSRT")
        else:
            print(f"[WARN] VitTrack 模型不存在: {model_path}，回退到 CSRT")

    factories = {
        "CSRT": lambda: cv2.TrackerCSRT.create(),
        "KCF": lambda: cv2.TrackerKCF.create(),
        "MIL": lambda: cv2.TrackerMIL.create(),
    }
    if tracker_type in factories:
        try:
            return factories[tracker_type]()
        except AttributeError:
            pass
    for name in ["CSRT", "KCF", "MIL"]:
        try:
            return factories[name]()
        except AttributeError:
            continue
    raise RuntimeError("No suitable tracker. Install opencv-contrib-python.")


class _SOTRobotTrack:
    """單一機器人的 SOT 追蹤。"""

    def __init__(self, label, bbox, frame, frame_idx, alliance=""):
        self.label = label
        self.alliance = alliance
        self.bbox = bbox
        self.lost_frames = 0
        self.is_lost = False
        self.positions = []
        self.bbox_history = []

        x, y, w, h = bbox
        roi = frame[y:y+h, x:x+w]
        self._init_template = roi.copy()
        self.template = roi.copy()
        self.template_size = (w, h)
        self._last_template_update = frame_idx

        self._tracker = _create_sot_tracker()
        self._is_vittrack = hasattr(self._tracker, "getTrackingScore")
        self._tracker.init(frame, bbox)

        cx, cy = rect_center(x, y, w, h)
        self.positions.append((frame_idx, cx, cy))
        self.bbox_history.append((frame_idx, x, y, w, h))

    def update(self, frame, frame_idx):
        if self.is_lost:
            return self._try_recover(frame, frame_idx)

        success, box = self._tracker.update(frame)

        if success and self._is_vittrack:
            score = self._tracker.getTrackingScore()
            if score < VITTRACK_SCORE_THRESHOLD:
                success = False

        if success:
            x, y, w, h = [int(v) for v in box]
            fh, fw = frame.shape[:2]
            if x < 0 or y < 0 or x + w > fw or y + h > fh or w < 5 or h < 5:
                success = False
            else:
                self.lost_frames = 0
                self.bbox = (x, y, w, h)
                cx, cy = rect_center(x, y, w, h)
                self.positions.append((frame_idx, cx, cy))
                self.bbox_history.append((frame_idx, x, y, w, h))
                if frame_idx - self._last_template_update >= _TEMPLATE_UPDATE_INTERVAL:
                    self.template = frame[y:y+h, x:x+w].copy()
                    self._last_template_update = frame_idx
                return (x, y, w, h)

        recovered = self._try_recover(frame, frame_idx)
        if recovered:
            return recovered

        self.lost_frames += 1
        if self.lost_frames > ROBOT_MAX_LOST_FRAMES:
            self.is_lost = True
        return None

    def _try_recover(self, frame, frame_idx):
        fh, fw = frame.shape[:2]
        lx, ly, lw, lh = self.bbox
        margin = max(lw, lh) * _SEARCH_MARGIN_MULT

        sx = max(0, int(lx - margin))
        sy = max(0, int(ly - margin))
        ex = min(fw, int(lx + lw + margin))
        ey = min(fh, int(ly + lh + margin))

        search_region = frame[sy:ey, sx:ex]
        th, tw = self.template.shape[:2]

        if search_region.shape[0] < th or search_region.shape[1] < tw:
            return None

        result = cv2.matchTemplate(search_region, self.template,
                                   cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val < _RECOVER_THRESHOLD:
            init_th, init_tw = self._init_template.shape[:2]
            if (search_region.shape[0] >= init_th and
                    search_region.shape[1] >= init_tw):
                result2 = cv2.matchTemplate(
                    search_region, self._init_template, cv2.TM_CCOEFF_NORMED)
                _, max_val2, _, max_loc2 = cv2.minMaxLoc(result2)
                if max_val2 > max_val:
                    max_val, max_loc = max_val2, max_loc2
                    tw, th = init_tw, init_th

        if max_val < _RECOVER_THRESHOLD:
            return None

        mx, my = max_loc
        new_x = sx + mx
        new_y = sy + my
        new_bbox = (new_x, new_y, tw, th)

        self._tracker = _create_sot_tracker()
        self._is_vittrack = hasattr(self._tracker, "getTrackingScore")
        self._tracker.init(frame, new_bbox)

        self.lost_frames = 0
        self.is_lost = False
        self.bbox = new_bbox
        self.template = frame[new_y:new_y+th, new_x:new_x+tw].copy()
        self._last_template_update = frame_idx

        cx, cy = rect_center(new_x, new_y, tw, th)
        self.positions.append((frame_idx, cx, cy))
        self.bbox_history.append((frame_idx, new_x, new_y, tw, th))

        return new_bbox

    def get_position_at(self, frame_idx):
        for f, cx, cy in self.positions:
            if f == frame_idx:
                return (cx, cy)
        return None

    def get_nearest_position(self, frame_idx, max_gap=5):
        best = None
        best_gap = float('inf')
        for f, cx, cy in self.positions:
            gap = abs(f - frame_idx)
            if gap < best_gap:
                best_gap = gap
                best = (cx, cy)
        return best if best_gap <= max_gap else None


class _SOTTracker:
    """SOT 多機器人追蹤管理（舊版 VitTrack/CSRT 模式）。"""

    def __init__(self):
        self._robots: dict[str, _SOTRobotTrack] = {}

    def add_robot(self, label, bbox, frame, frame_idx, alliance=""):
        self._robots[label] = _SOTRobotTrack(
            label, bbox, frame, frame_idx, alliance)

    def update_all(self, frame, frame_idx):
        results = {}
        for label, robot in self._robots.items():
            results[label] = robot.update(frame, frame_idx)
        return results

    def get_all_positions(self, frame_idx):
        positions = {}
        for label, robot in self._robots.items():
            pos = robot.get_position_at(frame_idx)
            if pos:
                positions[label] = pos
        return positions

    def get_all_display_positions(self, frame_idx, max_gap=15):
        positions = {}
        for label, robot in self._robots.items():
            pos = robot.get_position_at(frame_idx)
            if not pos:
                pos = robot.get_nearest_position(frame_idx, max_gap=max_gap)
            if pos:
                positions[label] = pos
        return positions

    def get_all_bboxes(self, frame_idx):
        """SOT 模式的 bbox 查詢。"""
        bboxes = {}
        for label, robot in self._robots.items():
            for f, x, y, w, h in robot.bbox_history:
                if f == frame_idx:
                    bboxes[label] = (float(x), float(y),
                                     float(x + w), float(y + h))
                    break
        return bboxes

    def interpolate_positions(self):
        """SOT 模式不需要額外插值（追蹤器本身每幀更新）。"""
        pass

    def clear(self):
        self._robots.clear()

    @property
    def count(self):
        return len(self._robots)

    @property
    def labels(self):
        return list(self._robots.keys())


# ═══════════════════════════════════════════════════════
# 統一介面: RobotTrackerManager
# ═══════════════════════════════════════════════════════


class RobotTrackerManager:
    """
    機器人追蹤管理器（統一介面）。

    - 提供 detector → 使用 MOT 模式（YOLO + ByteTrack）
    - 不提供 detector → 使用 SOT 模式（VitTrack/CSRT）
    """

    def __init__(self, detector=None, fps: float = 30.0):
        """
        Args:
            detector: RobotDetectorONNX 實例，None 則使用 SOT fallback
            fps: 影片 FPS（MOT 模式使用）
        """
        self._use_mot = detector is not None and HAS_SUPERVISION

        if self._use_mot:
            self._impl = _MOTTracker(detector, fps)
        else:
            self._impl = _SOTTracker()
            if detector is not None and not HAS_SUPERVISION:
                print("[WARN] supervision 套件未安裝，"
                      "回退到 SOT 追蹤模式。"
                      "安裝方式: pip install supervision>=0.21.0")

    @property
    def use_mot(self) -> bool:
        """是否使用 MOT 模式。"""
        return self._use_mot

    def add_robot(self, label: str, bbox: tuple, frame: np.ndarray,
                  frame_idx: int, alliance: str = ""):
        self._impl.add_robot(label, bbox, frame, frame_idx, alliance)

    def remove_robot(self, label: str):
        """移除機器人（僅 SOT 模式支援）。"""
        if hasattr(self._impl, '_robots') and label in self._impl._robots:
            del self._impl._robots[label]

    def enable_auto_mode(self):
        """啟用 MOT 自動偵測模式（不需用戶標記）。"""
        if self._use_mot:
            self._impl.enable_auto_mode()

    @property
    def robot_info(self) -> dict:
        """取得所有機器人的資訊 {label: {"alliance": str, ...}}。"""
        if hasattr(self._impl, '_robot_info'):
            return dict(self._impl._robot_info)
        return {}

    def update_all(self, frame: np.ndarray, frame_idx: int) -> dict:
        return self._impl.update_all(frame, frame_idx)

    def get_all_positions(self, frame_idx: int) -> dict[str, tuple]:
        return self._impl.get_all_positions(frame_idx)

    def get_all_display_positions(self, frame_idx: int,
                                  max_gap: int = 15) -> dict[str, tuple]:
        return self._impl.get_all_display_positions(frame_idx, max_gap)

    def get_all_bboxes(self, frame_idx: int) -> dict[str, tuple]:
        """取得所有機器人在指定幀的 bbox (x1, y1, x2, y2)。"""
        return self._impl.get_all_bboxes(frame_idx)

    def interpolate_positions(self):
        """對丟失幀做位置插值（MOT 模式專用）。"""
        self._impl.interpolate_positions()

    def clear(self):
        self._impl.clear()

    @property
    def count(self) -> int:
        return self._impl.count

    @property
    def labels(self) -> list[str]:
        return self._impl.labels

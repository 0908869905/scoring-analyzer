"""
FRC Scoring Analyzer — 機器人追蹤

主要模式: YOLO 偵測 + ByteTrack 多目標追蹤 (MOT)
備用模式: VitTrack / CSRT 單目標追蹤 (SOT)
"""

import math
import os

import cv2
import numpy as np

from config import (
    ROBOT_MAX_LOST_FRAMES, ROBOT_TRACKER_TYPE,
    VITTRACK_MODEL_PATH, VITTRACK_SCORE_THRESHOLD,
    BYTETRACK_TRACK_THRESH, BYTETRACK_LOST_BUFFER,
    BYTETRACK_MATCH_THRESH, BYTETRACK_MIN_CONSECUTIVE,
    MOT_DETECT_INTERVAL, MOT_REID_MAX_DIST, MOT_HISTOGRAM_WEIGHT,
    MOT_REID_MAX_SECONDS, MOT_MIN_TRACK_FRAMES,
    MOT_MERGE_MAX_OVERLAP, MOT_MERGE_BOUNDARY_DIST, MOT_MERGE_SEARCH_WINDOW,
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

    def __init__(self, detector, fps: float = 30.0,
                 detect_interval: int = MOT_DETECT_INTERVAL):
        """
        Args:
            detector: RobotDetectorONNX 實例
            fps: 影片 FPS
            detect_interval: 每 N 幀偵測一次（1=每幀）
        """
        self._detector = detector
        self._fps = fps
        self._detect_interval = max(1, detect_interval)

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

        # ── 距離式重新辨識（防止 ByteTrack ID 碎片化）──
        self._next_red_id = 1
        self._next_blue_id = 1
        self._next_robot_id = 1
        self._last_known: dict[str, tuple] = {}
        # {label: (frame_idx, cx, cy, class_id, vx, vy)}
        self._reid_max_dist = MOT_REID_MAX_DIST
        self._reid_max_frames = int(fps * MOT_REID_MAX_SECONDS)

        # ── 顏色直方圖 Re-ID ──
        self._histograms: dict[str, np.ndarray] = {}
        # {label: normalized HSV histogram}
        self._hist_weight = MOT_HISTOGRAM_WEIGHT

        # ── 追蹤信心度（區分偵測 vs 插值）──
        self._detected_frames: dict[str, set[int]] = {}
        # {label: set of frame indices with actual detections}

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
        # 1. 混合偵測：全幀每幀 + tiled 每 N 幀
        #    全幀偵測速度快、上下文完整（但機器人僅 ~25px）
        #    Tiled 偵測解析度高（~75px）但缺乏上下文
        raw_dets = list(self._detector.detect(frame))

        if frame_idx % self._detect_interval == 0:
            tiled_dets = self._detector.detect_tiled(frame)
            raw_dets.extend(tiled_dets)

        # 去重：全幀和 tiled 可能偵測到同一機器人
        if len(raw_dets) > 1:
            boxes = [[float(d[0]), float(d[1]),
                      float(d[2] - d[0]), float(d[3] - d[1])]
                     for d in raw_dets]
            confs = [float(d[4]) for d in raw_dets]
            indices = cv2.dnn.NMSBoxes(boxes, confs, 0.01, 0.5)
            if len(indices) > 0:
                raw_dets = [raw_dets[i] for i in indices.flatten()]
            else:
                raw_dets = []

        # 診斷：前幾幀印出偵測詳情
        if frame_idx < 5:
            det_summary = []
            for d in raw_dets:
                cls_name = self._detector.get_class_name(d[5])
                det_summary.append(f"{cls_name}:{d[4]:.2f}")
            src = "+tiled" if frame_idx % self._detect_interval == 0 else ""
            print(f"[DIAG] MOT f{frame_idx}: "
                  f"{len(raw_dets)} dets{src} [{', '.join(det_summary)}]")

        # 2. 追蹤匹配
        if self._auto_mode:
            # ── 距離式直接匹配（繞過 ByteTrack）──
            # ByteTrack 的 IoU 匹配在 4K@60fps 完全失效
            # （偵測框 ~25px，每幀移動 30-50px → IoU=0）
            results, frame_pos, frame_bbox = \
                self._match_direct(raw_dets, frame_idx, frame)
        else:
            # ── ByteTrack 手動標記模式 ──
            results, frame_pos, frame_bbox = \
                self._match_bytetrack(raw_dets, frame_idx)

        self._frame_positions[frame_idx] = frame_pos
        self._frame_bboxes[frame_idx] = frame_bbox

        for label in self._robot_info:
            if label not in results:
                results[label] = None

        return results

    def _match_direct(self, raw_dets, frame_idx,
                       frame: np.ndarray | None = None):
        """全域最短距離匹配（auto_mode 用，不使用 ByteTrack）。

        使用距離矩陣 + 貪心最短優先匹配，避免順序依賴性。
        距離閾值依幀間隔動態縮放（近幀嚴格、遠幀寬鬆）。
        當 frame 不為 None 且 hist_weight > 0 時，加入顏色直方圖加權。
        """
        results = {}
        frame_pos = {}
        frame_bbox = {}

        if not raw_dets:
            return results, frame_pos, frame_bbox

        use_hist = (frame is not None and self._hist_weight > 0
                    and len(self._histograms) > 0)

        # 解析所有偵測 + 提取直方圖
        det_info = []
        det_hists = []
        for d in raw_dets:
            x1, y1, x2, y2, conf, cls_id = \
                d[0], d[1], d[2], d[3], d[4], int(d[5])
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            det_info.append((cx, cy, x1, y1, x2, y2, conf, int(cls_id)))
            if use_hist:
                det_hists.append(
                    self._extract_histogram(frame, x1, y1, x2, y2))
            else:
                det_hists.append(None)

        # 建構距離矩陣：(有效距離, det_idx, label)
        candidates = list(self._last_known.keys())
        pairs = []
        for di, (cx, cy, *_, cls_id) in enumerate(det_info):
            for label in candidates:
                lk = self._last_known[label]
                last_f, last_cx, last_cy, last_cls = lk[0], lk[1], lk[2], lk[3]
                last_vx = lk[4] if len(lk) > 4 else 0.0
                last_vy = lk[5] if len(lk) > 5 else 0.0
                # 類別不相容（紅不配藍）
                if (last_cls >= 0 and cls_id >= 0
                        and last_cls != cls_id):
                    continue
                # 超過時間限制
                frame_gap = frame_idx - last_f
                if frame_gap > self._reid_max_frames:
                    continue
                # 速度預測：用歷史速度外推到當前幀的預測位置
                pred_cx = last_cx + last_vx * frame_gap
                pred_cy = last_cy + last_vy * frame_gap
                spatial_dist = math.hypot(cx - pred_cx, cy - pred_cy)
                # 動態距離閾值：基準 × (1 + sqrt(幀間隔/fps))
                # 連續幀 → 1.13x；10幀 → 1.41x；60幀 → 2x；180幀 → 2.73x
                max_dist = self._reid_max_dist * (
                    1 + math.sqrt(frame_gap / self._fps))
                if spatial_dist >= max_dist:
                    continue

                # 顏色直方圖加權：距離 × (1 + w * (1 - similarity))
                # 外觀相似 → factor ≈ 1（不影響）
                # 外觀不同 → factor 最大 1 + w（距離放大 40%）
                effective_dist = spatial_dist
                if (use_hist and det_hists[di] is not None
                        and label in self._histograms):
                    sim = self._compare_histograms(
                        det_hists[di], self._histograms[label])
                    sim = max(0.0, sim)  # CORREL 可能為負
                    effective_dist = spatial_dist * (
                        1 + self._hist_weight * (1 - sim))

                pairs.append((effective_dist, di, label))

        # 按有效距離排序，貪心匹配（最短距離優先）
        pairs.sort(key=lambda p: p[0])
        used_dets = set()
        used_labels = set()
        det_label_map = {}  # det_idx → label

        for dist, di, label in pairs:
            if di in used_dets or label in used_labels:
                continue
            det_label_map[di] = label
            used_dets.add(di)
            used_labels.add(label)

        # 處理所有偵測（匹配的 + 新建的）
        for di, (cx, cy, x1, y1, x2, y2, conf, cls_id) in enumerate(det_info):
            if di in det_label_map:
                label = det_label_map[di]
            else:
                # 全新機器人
                alliance = (self._detector.infer_alliance(cls_id)
                            if cls_id >= 0 else "")
                label = self._allocate_label(alliance)
                if frame_idx < 5 or frame_idx % 100 == 0:
                    print(f"[INFO] MOT new: {label} (f{frame_idx})")

            # 記錄位置
            w, h = x2 - x1, y2 - y1
            self._positions[label].append((frame_idx, cx, cy))
            self._bboxes[label].append(
                (frame_idx, float(x1), float(y1), float(x2), float(y2)))
            self._detected_frames.setdefault(label, set()).add(frame_idx)
            frame_pos[label] = (cx, cy)
            frame_bbox[label] = (float(x1), float(y1),
                                 float(x2), float(y2))
            results[label] = (int(x1), int(y1), int(w), int(h))
            # 計算速度（像素/幀）
            vx, vy = 0.0, 0.0
            if label in self._last_known:
                prev = self._last_known[label]
                dt = frame_idx - prev[0]
                if dt > 0:
                    vx = (cx - prev[1]) / dt
                    vy = (cy - prev[2]) / dt
            self._last_known[label] = (frame_idx, cx, cy, cls_id, vx, vy)

            # 更新顏色直方圖（EMA 平滑）
            if frame is not None:
                new_hist = (det_hists[di] if det_hists[di] is not None
                            else self._extract_histogram(
                                frame, x1, y1, x2, y2))
                if new_hist is not None:
                    if label in self._histograms:
                        # EMA: 70% 舊 + 30% 新（漸進更新，抗瞬時遮擋）
                        self._histograms[label] = (
                            0.7 * self._histograms[label] + 0.3 * new_hist)
                    else:
                        self._histograms[label] = new_hist

        # 診斷
        if frame_idx < 5:
            labels = list(results.keys())
            print(f"[DIAG] MOT f{frame_idx}: "
                  f"matched {len(results)} robots {labels}")

        return results, frame_pos, frame_bbox

    def _match_bytetrack(self, raw_dets, frame_idx):
        """ByteTrack 匹配（手動標記模式用）。"""
        results = {}
        frame_pos = {}
        frame_bbox = {}

        # 建構 sv.Detections
        if raw_dets:
            xyxy = np.array([[d[0], d[1], d[2], d[3]] for d in raw_dets],
                            dtype=np.float32)
            conf = np.array([d[4] for d in raw_dets], dtype=np.float32)
            cls = np.array([d[5] for d in raw_dets], dtype=int)
            detections = sv.Detections(
                xyxy=xyxy, confidence=conf, class_id=cls)
        else:
            detections = sv.Detections.empty()

        tracked = self._tracker.update_with_detections(detections)
        self._match_pending_markers(tracked, frame_idx)

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
                self._detected_frames.setdefault(label, set()).add(frame_idx)
                frame_pos[label] = (cx, cy)
                frame_bbox[label] = (float(x1), float(y1),
                                     float(x2), float(y2))
                results[label] = (int(x1), int(y1), int(w), int(h))

        return results, frame_pos, frame_bbox

    def _allocate_label(self, alliance: str) -> str:
        """分配新的機器人 label（順序編號）。"""
        if alliance == "red":
            label = f"Red-{self._next_red_id}"
            self._next_red_id += 1
        elif alliance == "blue":
            label = f"Blue-{self._next_blue_id}"
            self._next_blue_id += 1
        else:
            label = f"Robot-{self._next_robot_id}"
            self._next_robot_id += 1
        self._robot_info[label] = {
            "alliance": alliance, "mark_frame": -1}
        self._positions.setdefault(label, [])
        self._bboxes.setdefault(label, [])
        return label

    def _try_reid(self, cx: float, cy: float, class_id: int,
                  frame_idx: int, lost_labels: set) -> str | None:
        """嘗試將新偵測與丟失的機器人匹配（距離式重新辨識）。

        注意：此方法已不被 _match_direct 使用（改用全域距離矩陣）。
        保留供 _match_bytetrack 手動模式使用。

        Returns:
            匹配的 label 或 None
        """
        best_label = None
        best_dist = float('inf')

        for label in lost_labels:
            if label not in self._last_known:
                continue

            lk = self._last_known[label]
            last_f, last_cx, last_cy, last_cls = lk[0], lk[1], lk[2], lk[3]

            # 超過時間限制不重新辨識
            frame_gap = frame_idx - last_f
            if frame_gap > self._reid_max_frames:
                continue

            # 類別不相容（紅不配藍）
            if (last_cls >= 0 and class_id >= 0
                    and last_cls != class_id):
                continue

            # 動態距離閾值
            max_dist = self._reid_max_dist * (
                1 + math.sqrt(frame_gap / self._fps))

            d = math.hypot(cx - last_cx, cy - last_cy)
            if d < max_dist and d < best_dist:
                best_dist = d
                best_label = label

        return best_label

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

    @staticmethod
    def _extract_histogram(frame: np.ndarray,
                           x1: float, y1: float,
                           x2: float, y2: float) -> np.ndarray | None:
        """從 bbox 區域提取 HSV 顏色直方圖（H+S 雙通道，忽略 V 以抗光照）。

        Returns:
            正規化後的直方圖向量，或 None（bbox 無效時）
        """
        h, w = frame.shape[:2]
        ix1 = max(0, int(x1))
        iy1 = max(0, int(y1))
        ix2 = min(w, int(x2))
        iy2 = min(h, int(y2))
        if ix2 - ix1 < 5 or iy2 - iy1 < 5:
            return None
        roi = frame[iy1:iy2, ix1:ix2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # H: 0-180 (16 bins), S: 0-256 (16 bins) → 256 維特徵
        hist = cv2.calcHist([hsv], [0, 1], None,
                            [16, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        return hist.flatten()

    @staticmethod
    def _compare_histograms(h1: np.ndarray, h2: np.ndarray) -> float:
        """比較兩個直方圖的相似度（0~1，1=完全相同）。"""
        return cv2.compareHist(
            h1.reshape(-1, 1).astype(np.float32),
            h2.reshape(-1, 1).astype(np.float32),
            cv2.HISTCMP_CORREL
        )

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

    def merge_fragmented_labels(self):
        """後處理：合併碎片化的 label（迭代式）。

        合併條件（必須全部滿足）：
        1. 同類別（Red 不合併 Blue）
        2. 重疊幀數 ≤ MAX_OVERLAP（允許過渡期短暫共存）
        3. 邊界幀位置距離合理（800px 內）

        迭代直到無法再合併為止。
        """
        pre_labels = list(self._robot_info.keys())
        # 診斷：合併前各 label 的偵測幀數
        for label in sorted(pre_labels):
            count = len(self._positions.get(label, []))
            print(f"  {label}: {count} 幀偵測")

        iteration = 0
        while True:
            iteration += 1
            merge_into = self._find_merge_candidates(MOT_MERGE_MAX_OVERLAP)
            if not merge_into:
                break

            self._execute_merges(merge_into, MOT_MERGE_MAX_OVERLAP)

            if iteration > 20:
                break  # 安全閥

        post_labels = list(self._robot_info.keys())
        if len(pre_labels) != len(post_labels):
            print(f"[INFO] MOT merge: {len(pre_labels)} → "
                  f"{len(post_labels)} 個 label "
                  f"({len(pre_labels) - len(post_labels)} 個已合併)")

    def _find_merge_candidates(self, max_overlap: int) -> dict:
        """找出可合併的 label 對。"""
        # 建立每個 label 的幀範圍和類別
        label_info = {}
        for label, pos_list in self._positions.items():
            if not pos_list:
                continue
            frames = [p[0] for p in pos_list]
            frame_set = set(frames)
            alliance = self._robot_info.get(label, {}).get("alliance", "")
            label_info[label] = {
                "frames": frame_set,
                "min_f": min(frames),
                "max_f": max(frames),
                "count": len(pos_list),
                "alliance": alliance,
            }

        merge_into = {}  # short_label → long_label
        labels_sorted = sorted(label_info.keys(),
                               key=lambda l: label_info[l]["count"],
                               reverse=True)

        for i, long_label in enumerate(labels_sorted):
            long_info = label_info[long_label]
            if long_label in merge_into:
                continue

            for short_label in labels_sorted[i + 1:]:
                if short_label in merge_into:
                    continue
                short_info = label_info[short_label]

                # 條件 1：同類別
                if long_info["alliance"] != short_info["alliance"]:
                    continue

                # 條件 2：重疊幀數在容忍範圍內
                overlap = long_info["frames"] & short_info["frames"]
                if len(overlap) > max_overlap:
                    continue

                # 條件 3：邊界幀位置距離
                close_enough = self._check_boundary_distance(
                    short_label, long_label, short_info, long_info)

                if close_enough:
                    merge_into[short_label] = long_label

        return merge_into

    def _check_boundary_distance(self, short_label, long_label,
                                  short_info, long_info) -> bool:
        """檢查兩個 label 的邊界幀位置是否足夠接近。"""
        short_pos = {p[0]: (p[1], p[2])
                     for p in self._positions[short_label]}
        long_pos = {p[0]: (p[1], p[2])
                    for p in self._positions[long_label]}

        for boundary_f in [short_info["min_f"], short_info["max_f"]]:
            if boundary_f not in short_pos:
                continue
            sx, sy = short_pos[boundary_f]
            best_dist = float('inf')
            for delta in range(-MOT_MERGE_SEARCH_WINDOW,
                               MOT_MERGE_SEARCH_WINDOW + 1):  # ±3 秒
                f = boundary_f + delta
                if f in long_pos:
                    lx, ly = long_pos[f]
                    d = math.hypot(sx - lx, sy - ly)
                    if d < best_dist:
                        best_dist = d
            if best_dist < MOT_MERGE_BOUNDARY_DIST:
                return True

        return False

    def _execute_merges(self, merge_into: dict, max_overlap: int):
        """執行 label 合併。"""
        for short_label, long_label in merge_into.items():
            print(f"[INFO] MOT merge: {short_label} → {long_label}")

            # 移除重疊幀（保留 long_label 的資料）
            long_frames = {p[0] for p in self._positions[long_label]}
            self._positions[short_label] = [
                p for p in self._positions[short_label]
                if p[0] not in long_frames
            ]
            self._bboxes[short_label] = [
                b for b in self._bboxes.get(short_label, [])
                if b[0] not in long_frames
            ]

            # 合併位置
            self._positions[long_label].extend(
                self._positions.pop(short_label, []))

            # 合併 bbox
            self._bboxes[long_label].extend(
                self._bboxes.pop(short_label, []))

            # 合併 frame_positions
            for f, pos_dict in self._frame_positions.items():
                if short_label in pos_dict:
                    if long_label not in pos_dict:
                        pos_dict[long_label] = pos_dict.pop(short_label)
                    else:
                        del pos_dict[short_label]

            # 合併 frame_bboxes
            for f, bbox_dict in self._frame_bboxes.items():
                if short_label in bbox_dict:
                    if long_label not in bbox_dict:
                        bbox_dict[long_label] = bbox_dict.pop(short_label)
                    else:
                        del bbox_dict[short_label]

            # 移除 robot_info
            self._robot_info.pop(short_label, None)

            # 更新 last_known
            if short_label in self._last_known:
                self._last_known.pop(short_label)

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

    def filter_short_labels(self, min_frames: int = MOT_MIN_TRACK_FRAMES):
        """移除偵測幀數不足的短期 label（噪點過濾）。

        Args:
            min_frames: 最少偵測幀數，低於此值的 label 會被完全移除
        """
        to_remove = []
        for label, pos_list in self._positions.items():
            if len(pos_list) < min_frames:
                to_remove.append((label, len(pos_list)))

        for label, count in to_remove:
            self._positions.pop(label, None)
            self._bboxes.pop(label, None)
            self._robot_info.pop(label, None)
            self._last_known.pop(label, None)
            for f in list(self._frame_positions.keys()):
                self._frame_positions[f].pop(label, None)
            for f in list(self._frame_bboxes.keys()):
                self._frame_bboxes[f].pop(label, None)

        if to_remove:
            removed_names = [f"{l}({c}f)" for l, c in to_remove]
            print(f"[INFO] MOT filter: 移除 {len(to_remove)} 個短期 label "
                  f"(< {min_frames} 幀): {', '.join(removed_names)}")
            remaining = list(self._robot_info.keys())
            print(f"[INFO] MOT filter: 保留 {len(remaining)} 個 label: "
                  f"{', '.join(remaining)}")

    def clear(self):
        self._label_map.clear()
        self._pending_markers.clear()
        self._robot_info.clear()
        self._positions.clear()
        self._bboxes.clear()
        self._frame_positions.clear()
        self._frame_bboxes.clear()
        self._last_known.clear()
        self._histograms.clear()
        self._detected_frames.clear()
        self._next_red_id = 1
        self._next_blue_id = 1
        self._next_robot_id = 1

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

    def merge_fragmented_labels(self):
        """後處理：合併碎片化的 label（MOT 模式專用）。"""
        if hasattr(self._impl, 'merge_fragmented_labels'):
            self._impl.merge_fragmented_labels()

    def filter_short_labels(self, min_frames: int | None = None):
        """移除偵測幀數不足的短期 label（MOT 模式專用）。"""
        if hasattr(self._impl, 'filter_short_labels'):
            self._impl.filter_short_labels(
                min_frames if min_frames is not None
                else MOT_MIN_TRACK_FRAMES)

    def interpolate_positions(self):
        """對丟失幀做位置插值（MOT 模式專用）。"""
        self._impl.interpolate_positions()

    def is_detected(self, label: str, frame_idx: int) -> bool:
        """查詢指定 label 在指定幀是否為實際偵測（非插值）。"""
        if hasattr(self._impl, '_detected_frames'):
            frames = self._impl._detected_frames.get(label)
            return frames is not None and frame_idx in frames
        # SOT 模式：所有追蹤幀都算「偵測」
        return True

    def clear(self):
        self._impl.clear()

    @property
    def count(self) -> int:
        return self._impl.count

    @property
    def labels(self) -> list[str]:
        return self._impl.labels

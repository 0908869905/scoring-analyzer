# MOT Occlusion Handling Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix robot tracking boxes wandering during hub occlusion by adding a Track State Machine (ACTIVE/LOST/REMOVED), occlusion zone awareness, FPS corrections, and static label display fix.

**Architecture:** Introduce 3-state lifecycle in `_MOTTracker._match_direct()` with two-round matching (ACTIVE first, LOST revival second). Occlusion zones reuse existing polygon UI. Static labels change from deletion to marking. All changes are backward-compatible.

**Tech Stack:** Python 3.11+, OpenCV, NumPy, CustomTkinter

**Spec:** `docs/superpowers/specs/2026-03-13-mot-occlusion-handling-design.md`

---

## Chunk 1: config.py Constants + geometry.py Helper

### Task 1: Update config.py constants

**Files:**
- Modify: `config.py:107-118`

- [ ] **Step 1: Add new MOT occlusion constants and remove MOT_REID_MAX_SECONDS**

In `config.py`, after the existing MOT section (line 107-118), replace with:

```python
# ── MOT 進階參數 ────────────────────────────────────
MOT_DETECT_INTERVAL = 10           # 每 N 幀做 tiled 偵測（全幀偵測每幀都做）
MOT_REID_MAX_DIST = 400            # 重新辨識基準距離（像素，會依幀間隔動態縮放）
MOT_HISTOGRAM_WEIGHT = 0.4         # 顏色直方圖 Re-ID 權重（0=純距離, 1=全靠外觀）
MOT_HISTOGRAM_UPDATE_INTERVAL = 3  # 直方圖更新頻率（每 N 幀更新一次，降低計算開銷）
MOT_MIN_TRACK_FRAMES = 15          # 最少偵測幀數（少於此值的 label 在後處理中被移除，30fps 下 ≈ 0.5 秒）
MOT_STATIC_MAX_VARIANCE = 100      # 靜止判定：位置變異數閾值（px²，標準差 <10px 視為靜止）
MOT_STATIC_MIN_FRAMES = 30         # 靜止判定：至少追蹤 N 幀才判定（30fps 下 1 秒，避免誤判短暫停留）
MOT_MERGE_MAX_OVERLAP = 15         # 合併最大重疊幀數（30fps 下 ≈ 500ms）
MOT_MERGE_BOUNDARY_DIST = 800      # 合併邊界距離閾值（像素）
MOT_MERGE_SEARCH_WINDOW = 180      # 合併邊界搜尋窗口（幀數，30fps 下 ≈ 6 秒）
# ── MOT 遮擋處理 ────────────────────────────────────
MOT_MAX_LOST_FRAMES = 90           # LOST 最大容忍幀數（30fps 下 3 秒）
MOT_OCCLUSION_PATIENCE = 450       # 遮擋區域 LOST 耐心（30fps 下 15 秒）
MOT_LOST_GRACE_FRAMES = 3          # 連續未偵測幀數閾值，超過後 ACTIVE → LOST
MOT_LOST_REID_DIST_SCALE = 0.5     # LOST 復活距離閾值倍率（更嚴格）
MOT_LOST_MIN_HIST_SIM = 0.3        # LOST 復活最低直方圖相似度
MOT_OCCLUSION_MARGIN = 50          # 遮擋區域邊緣緩衝（像素）
```

Key changes:
- Remove `MOT_REID_MAX_SECONDS = 5` (superseded by state machine)
- Fix `MOT_STATIC_MIN_FRAMES` from 60 → 30
- Fix all FPS comments from "60fps" → "30fps"
- Add 6 new occlusion constants

- [ ] **Step 2: Verify config.py is syntactically valid**

Run: `python -c "import config; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add config.py
git commit -m "feat: add MOT occlusion constants, fix FPS comments (60→30)"
```

---

### Task 2: Add min_distance_to_polygon_edge() to geometry.py

**Files:**
- Modify: `geometry.py` (add function after `point_to_segment_distance`)

- [ ] **Step 1: Add the helper function**

After `point_to_segment_distance()` (line 82), add:

```python
def min_distance_to_polygon_edge(px, py, polygon):
    """計算點 (px, py) 到多邊形最近邊的距離。

    Args:
        px, py: 點座標
        polygon: [(x1, y1), (x2, y2), ...] 多邊形頂點列表
    """
    n = len(polygon)
    if n < 2:
        return float('inf')
    min_dist = float('inf')
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        d = point_to_segment_distance(px, py, x1, y1, x2, y2)
        if d < min_dist:
            min_dist = d
    return min_dist
```

- [ ] **Step 2: Verify**

Run: `python -c "from geometry import min_distance_to_polygon_edge; print(min_distance_to_polygon_edge(0, 0, [(10, 0), (10, 10), (0, 10)]))"`
Expected: `0.0` (point is on the polygon edge vertex)

- [ ] **Step 3: Commit**

```bash
git add geometry.py
git commit -m "feat: add min_distance_to_polygon_edge() geometry helper"
```

---

## Chunk 2: robot_tracker.py — State Machine Core

### Task 3: Add new data structures and update imports in _MOTTracker

**Files:**
- Modify: `robot_tracker.py:16-26` (imports)
- Modify: `robot_tracker.py:86-112` (init)

- [ ] **Step 1: Update imports**

Replace the import block (lines 16-27):

```python
from config import (
    ROBOT_MAX_LOST_FRAMES, ROBOT_TRACKER_TYPE,
    VITTRACK_MODEL_PATH, VITTRACK_SCORE_THRESHOLD,
    BYTETRACK_TRACK_THRESH, BYTETRACK_LOST_BUFFER,
    BYTETRACK_MATCH_THRESH, BYTETRACK_MIN_CONSECUTIVE,
    MOT_DETECT_INTERVAL, MOT_REID_MAX_DIST, MOT_HISTOGRAM_WEIGHT,
    MOT_HISTOGRAM_UPDATE_INTERVAL,
    MOT_MAX_LOST_FRAMES, MOT_OCCLUSION_PATIENCE,
    MOT_LOST_GRACE_FRAMES, MOT_LOST_REID_DIST_SCALE,
    MOT_LOST_MIN_HIST_SIM, MOT_OCCLUSION_MARGIN,
    MOT_MIN_TRACK_FRAMES,
    MOT_MERGE_MAX_OVERLAP, MOT_MERGE_BOUNDARY_DIST, MOT_MERGE_SEARCH_WINDOW,
    MOT_STATIC_MAX_VARIANCE, MOT_STATIC_MIN_FRAMES,
)
from geometry import rect_center, point_in_polygon, min_distance_to_polygon_edge
```

Key: remove `MOT_REID_MAX_SECONDS`, add 6 new occlusion constants + 2 geometry imports.

- [ ] **Step 2: Add new data structures in __init__**

After `self._hist_update_interval` (line 112), add:

```python
        # ── Track State Machine ──
        self._track_state: dict[str, str] = {}
        # {label: "active" | "lost"}
        self._lost_since: dict[str, int] = {}
        # {label: frame_idx when LOST started}
        self._missed_frames: dict[str, int] = {}
        # {label: consecutive frames without detection}

        # ── 遮擋區域 ──
        self._occlusion_zones: list = []
        # list of objects with .polygon attribute

        # ── 靜止標記 ──
        self._static_labels: set[str] = set()
```

Also remove the line:
```python
        self._reid_max_frames = int(fps * MOT_REID_MAX_SECONDS)
```

- [ ] **Step 3: Remove dead `_try_reid()` method**

Delete the `_try_reid()` method (lines 442-481) entirely. It references the removed `_reid_max_frames` and is dead code (never called by `_match_direct`; `_match_bytetrack` doesn't use it either). Its docstring already notes "此方法已不被 _match_direct 使用".

- [ ] **Step 4: Add set_occlusion_zones method**

After `enable_auto_mode()` (line 116), add:

```python
    def set_occlusion_zones(self, zones):
        """設定遮擋區域列表。"""
        self._occlusion_zones = list(zones)
```

- [ ] **Step 5: Verify**

Run: `python -c "import robot_tracker; print('OK')"`
Expected: `OK`

- [ ] **Step 6: Commit**

```bash
git add robot_tracker.py
git commit -m "feat: add track state machine data structures to _MOTTracker"
```

---

### Task 4: Refactor _match_direct() into two-round matching

**Files:**
- Modify: `robot_tracker.py:200-345` (`_match_direct` method)

This is the core change. The existing single-loop matching becomes Round 1 (ACTIVE only), then Round 2 (LOST revival) is added.

**Behavior change:** The old `if not raw_dets: return` early exit is removed. Now, frames with zero detections will cause ACTIVE tracks to increment their `_missed_frames` counter and eventually transition to LOST. This is intentional — the state machine needs to process every frame to manage track states correctly.

- [ ] **Step 1: Rewrite _match_direct()**

Replace the entire `_match_direct` method (lines 200-345) with the new implementation:

```python
    def _match_direct(self, raw_dets, frame_idx,
                       frame: np.ndarray | None = None):
        """全域最短距離匹配 + Track State Machine。

        Round 1: ACTIVE 軌跡正常貪心匹配
        Round 2: 剩餘偵測嘗試復活 LOST 軌跡（凍結位置 + 嚴格門檻）
        """
        results = {}
        frame_pos = {}
        frame_bbox = {}

        use_hist = (frame is not None and self._hist_weight > 0
                    and len(self._histograms) > 0)
        hist_frame = (frame_idx % self._hist_update_interval == 0)

        # 解析所有偵測 + 條件提取直方圖
        det_info = []
        det_hists = []
        for d in raw_dets:
            x1, y1, x2, y2, conf, cls_id = \
                d[0], d[1], d[2], d[3], d[4], int(d[5])
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            det_info.append((cx, cy, x1, y1, x2, y2, conf, int(cls_id)))
            if use_hist and hist_frame:
                det_hists.append(
                    self._extract_histogram(frame, x1, y1, x2, y2))
            else:
                det_hists.append(None)

        # ═══ Round 1: ACTIVE 軌跡匹配 ═══
        active_labels = [
            label for label in self._last_known
            if self._track_state.get(label) != "lost"
        ]
        pairs = []
        for di, (cx, cy, *_, cls_id) in enumerate(det_info):
            for label in active_labels:
                lk = self._last_known[label]
                last_f, last_cx, last_cy, last_cls = lk[0], lk[1], lk[2], lk[3]
                last_vx = lk[4] if len(lk) > 4 else 0.0
                last_vy = lk[5] if len(lk) > 5 else 0.0
                if (last_cls >= 0 and cls_id >= 0
                        and last_cls != cls_id):
                    continue
                frame_gap = frame_idx - last_f
                # 速度預測
                pred_cx = last_cx + last_vx * frame_gap
                pred_cy = last_cy + last_vy * frame_gap
                spatial_dist = math.hypot(cx - pred_cx, cy - pred_cy)
                max_dist = self._reid_max_dist * (
                    1 + math.sqrt(frame_gap / self._fps))
                if spatial_dist >= max_dist:
                    continue
                effective_dist = spatial_dist
                if (use_hist and det_hists[di] is not None
                        and label in self._histograms):
                    sim = self._compare_histograms(
                        det_hists[di], self._histograms[label])
                    sim = max(0.0, sim)
                    effective_dist = spatial_dist * (
                        1 + self._hist_weight * (1 - sim))
                pairs.append((effective_dist, di, label))

        pairs.sort(key=lambda p: p[0])
        used_dets = set()
        used_labels = set()
        det_label_map = {}

        for dist, di, label in pairs:
            if di in used_dets or label in used_labels:
                continue
            det_label_map[di] = label
            used_dets.add(di)
            used_labels.add(label)

        # ═══ Round 2: LOST 軌跡復活 ═══
        lost_labels = [
            label for label in self._last_known
            if self._track_state.get(label) == "lost"
        ]
        if lost_labels:
            lost_max_dist = self._reid_max_dist * MOT_LOST_REID_DIST_SCALE
            lost_pairs = []
            for di in range(len(det_info)):
                if di in used_dets:
                    continue
                cx, cy, *_, cls_id = det_info[di]
                for label in lost_labels:
                    lk = self._last_known[label]
                    last_f, last_cx, last_cy, last_cls = (
                        lk[0], lk[1], lk[2], lk[3])
                    if (last_cls >= 0 and cls_id >= 0
                            and last_cls != cls_id):
                        continue
                    # LOST: 用凍結位置（不外推速度）
                    spatial_dist = math.hypot(cx - last_cx, cy - last_cy)
                    if spatial_dist >= lost_max_dist:
                        continue
                    # 直方圖門檻
                    if label in self._histograms:
                        det_hist = (det_hists[di] if det_hists[di] is not None
                                    else self._extract_histogram(
                                        frame, det_info[di][2], det_info[di][3],
                                        det_info[di][4], det_info[di][5])
                                    if frame is not None else None)
                        if det_hist is not None:
                            sim = self._compare_histograms(
                                det_hist, self._histograms[label])
                            sim = max(0.0, sim)
                            if sim < MOT_LOST_MIN_HIST_SIM:
                                continue
                            spatial_dist = spatial_dist * (
                                1 + self._hist_weight * (1 - sim))
                    lost_pairs.append((spatial_dist, di, label))

            lost_pairs.sort(key=lambda p: p[0])
            for dist, di, label in lost_pairs:
                if di in used_dets or label in used_labels:
                    continue
                det_label_map[di] = label
                used_dets.add(di)
                used_labels.add(label)
                # 復活: LOST → ACTIVE
                self._track_state[label] = "active"
                self._lost_since.pop(label, None)
                self._missed_frames[label] = 0

        # ═══ 處理所有偵測（匹配的 + 新建的）═══
        for di, (cx, cy, x1, y1, x2, y2, conf, cls_id) in enumerate(det_info):
            if di in det_label_map:
                label = det_label_map[di]
            else:
                label = self._consume_pending_marker(
                    cx, cy, cls_id, frame_idx)
                if not label:
                    alliance = (self._detector.infer_alliance(cls_id)
                                if cls_id >= 0 else "")
                    label = self._allocate_label(alliance)
                if frame_idx < 5 or frame_idx % 100 == 0:
                    print(f"[INFO] MOT new: {label} (f{frame_idx})")

            w, h = x2 - x1, y2 - y1
            self._positions.setdefault(label, []).append((frame_idx, cx, cy))
            self._bboxes.setdefault(label, []).append(
                (frame_idx, float(x1), float(y1), float(x2), float(y2)))
            self._detected_frames.setdefault(label, set()).add(frame_idx)
            frame_pos[label] = (cx, cy)
            frame_bbox[label] = (float(x1), float(y1),
                                 float(x2), float(y2))
            results[label] = (int(x1), int(y1), int(w), int(h))
            vx, vy = 0.0, 0.0
            if label in self._last_known:
                prev = self._last_known[label]
                dt = frame_idx - prev[0]
                if dt > 0:
                    vx = (cx - prev[1]) / dt
                    vy = (cy - prev[2]) / dt
            self._last_known[label] = (frame_idx, cx, cy, cls_id, vx, vy)

            # 狀態：確認 ACTIVE + 重置 missed
            self._track_state[label] = "active"
            self._missed_frames[label] = 0

            is_new_label = label not in self._histograms
            if frame is not None and (is_new_label or hist_frame):
                new_hist = (det_hists[di] if det_hists[di] is not None
                            else self._extract_histogram(
                                frame, x1, y1, x2, y2))
                if new_hist is not None:
                    if not is_new_label:
                        self._histograms[label] = (
                            0.7 * self._histograms[label] + 0.3 * new_hist)
                    else:
                        self._histograms[label] = new_hist

        # ═══ 更新未匹配 ACTIVE 軌跡的狀態 ═══
        for label in active_labels:
            if label in used_labels:
                continue
            # 遞增 missed 計數
            missed = self._missed_frames.get(label, 0) + 1
            self._missed_frames[label] = missed
            if missed >= MOT_LOST_GRACE_FRAMES:
                # Grace period 結束 → 轉為 LOST
                self._track_state[label] = "lost"
                if label not in self._lost_since:
                    self._lost_since[label] = frame_idx

        # ═══ 清除超時 LOST 軌跡 (REMOVED) ═══
        for label in list(self._lost_since.keys()):
            if self._track_state.get(label) != "lost":
                continue
            lost_duration = frame_idx - self._lost_since[label]
            # 判斷是否在遮擋區域
            lk = self._last_known.get(label)
            if lk and self._occlusion_zones:
                last_cx, last_cy = lk[1], lk[2]
                in_occlusion = any(
                    point_in_polygon(last_cx, last_cy, z.polygon)
                    or min_distance_to_polygon_edge(
                        last_cx, last_cy, z.polygon) < MOT_OCCLUSION_MARGIN
                    for z in self._occlusion_zones
                )
                patience = (MOT_OCCLUSION_PATIENCE if in_occlusion
                            else MOT_MAX_LOST_FRAMES)
            else:
                patience = MOT_MAX_LOST_FRAMES
            if lost_duration > patience:
                # REMOVED: 從 runtime dicts 移除，保留歷史資料
                self._last_known.pop(label, None)
                self._track_state.pop(label, None)
                self._lost_since.pop(label, None)
                self._missed_frames.pop(label, None)
                # 不刪 _positions, _bboxes, _detected_frames（後處理需要）

        # 診斷
        if frame_idx < 5:
            labels = list(results.keys())
            lost_count = sum(
                1 for s in self._track_state.values() if s == "lost")
            print(f"[DIAG] MOT f{frame_idx}: "
                  f"matched {len(results)} robots {labels}, "
                  f"lost={lost_count}")

        return results, frame_pos, frame_bbox
```

- [ ] **Step 2: Verify module loads**

Run: `python -c "import robot_tracker; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add robot_tracker.py
git commit -m "feat: refactor _match_direct() into two-round matching with state machine"
```

---

### Task 5: Update cleanup methods and filter_static_labels

**Files:**
- Modify: `robot_tracker.py` — `clear()`, `filter_short_labels()`, `_execute_merges()`, `filter_static_labels()`

- [ ] **Step 1: Update clear()**

Add to the `clear()` method (after line 907 `self._detected_frames.clear()`):

```python
        self._track_state.clear()
        self._lost_since.clear()
        self._missed_frames.clear()
        self._static_labels.clear()
        self._occlusion_zones.clear()
```

- [ ] **Step 2: Update filter_short_labels()**

In `filter_short_labels()`, after the existing cleanup lines (after `self._last_known.pop(label, None)` at line 846), add:

```python
            self._track_state.pop(label, None)
            self._lost_since.pop(label, None)
            self._missed_frames.pop(label, None)
            self._detected_frames.pop(label, None)
            self._histograms.pop(label, None)
            self._static_labels.discard(label)
```

- [ ] **Step 3: Update _execute_merges()**

In `_execute_merges()`, after `self._last_known.pop(short_label)` (line 759), add:

```python
            self._track_state.pop(short_label, None)
            self._lost_since.pop(short_label, None)
            self._missed_frames.pop(short_label, None)
            self._static_labels.discard(short_label)
            # 合併 detected_frames
            if short_label in self._detected_frames:
                self._detected_frames.setdefault(
                    long_label, set()).update(
                    self._detected_frames.pop(short_label))
```

- [ ] **Step 4: Rewrite filter_static_labels() to mark-only**

Replace the entire `filter_static_labels()` method:

```python
    def filter_static_labels(self,
                             max_variance: float = MOT_STATIC_MAX_VARIANCE,
                             min_frames: int = MOT_STATIC_MIN_FRAMES):
        """標記位置幾乎不動的 label（不再刪除，改為標記）。

        靜止判定：追蹤幀數 >= min_frames 且位置變異數 < max_variance。
        標記結果存入 self._static_labels，不影響追蹤資料。
        """
        self._static_labels.clear()
        marked = []
        for label, pos_list in self._positions.items():
            if len(pos_list) < min_frames:
                continue
            xs = [p[1] for p in pos_list]
            ys = [p[2] for p in pos_list]
            var_x = np.var(xs)
            var_y = np.var(ys)
            total_var = var_x + var_y
            if total_var < max_variance:
                self._static_labels.add(label)
                marked.append((label, len(pos_list), total_var))

        if marked:
            marked_names = [f"{l}({c}f, var={v:.1f})" for l, c, v in marked]
            print(f"[INFO] MOT static filter: 標記 {len(marked)} 個靜止 label "
                  f"(var < {max_variance}): {', '.join(marked_names)}")
```

- [ ] **Step 5: Verify**

Run: `python -c "import robot_tracker; print('OK')"`
Expected: `OK`

- [ ] **Step 6: Commit**

```bash
git add robot_tracker.py
git commit -m "feat: update cleanup methods + filter_static_labels mark-only"
```

---

## Chunk 3: app.py — Occlusion Zone UI + LOST Rendering

### Task 6: Add OcclusionZone dataclass and zone storage

**Files:**
- Modify: `app.py` (imports + __init__)

- [ ] **Step 1: Add OcclusionZone import/class**

Near the top of `app.py`, after the existing imports (around line 30), add:

```python
from dataclasses import dataclass

@dataclass
class OcclusionZone:
    """遮擋區域定義（多邊形）。"""
    name: str
    polygon: list[tuple[int, int]]
```

Note: check if `dataclass` is already imported (it may be via `scoring.py`). If so, just add the class.

- [ ] **Step 2: Add _occlusion_zones to __init__**

After `self._scoring_zones = []` (line 126), add:

```python
        self._occlusion_zones = []  # [OcclusionZone, ...]
```

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat: add OcclusionZone dataclass and storage in app.py"
```

---

### Task 7: Add occlusion zone drawing UI

**Files:**
- Modify: `app.py` — zone interaction methods

- [ ] **Step 1: Update interaction_mode to support "mark_occlusion_zone"**

In `app.py`, the `interaction_mode` (line 91) comment should include `"mark_occlusion_zone"`:

```python
        self.interaction_mode = None  # None, "bumper_pick", "mark_zone_polygon", "mark_hp_line", "mark_occlusion_zone"
```

- [ ] **Step 2: Add button to start occlusion zone marking**

Find the existing zone marking buttons (search for "Hub 區域" or "mark_zone_polygon") and add a new button nearby:

```python
        ctk.CTkButton(
            zone_frame,  # or whatever parent frame the zone buttons use
            text="遮擋區域",
            command=self._start_mark_occlusion_zone,
            width=80,
        ).pack(side=tk.LEFT, padx=2)
```

- [ ] **Step 3: Add _start_mark_occlusion_zone method**

```python
    def _start_mark_occlusion_zone(self):
        """開始標記遮擋區域多邊形。"""
        if not self.cap:
            self._set_status("請先開啟影片", COLORS["error"])
            return
        self.interaction_mode = "mark_occlusion_zone"
        self._polygon_points = []
        self.canvas.config(cursor="crosshair")
        self._set_status(
            "點擊標記遮擋區域頂點，右鍵或雙擊完成",
            COLORS["info"])
```

- [ ] **Step 4: Handle clicks for occlusion zone mode**

In `_on_canvas_click()`, add handling for `"mark_occlusion_zone"` — it uses the same polygon point collection as scoring zones. Find where `"mark_zone_polygon"` click is handled and add:

```python
        if self.interaction_mode == "mark_occlusion_zone":
            self._polygon_points.append(pt)
            self._show_frame(self.current_frame)
            return
```

In `_on_canvas_double_click()`, add:
```python
        if self.interaction_mode == "mark_occlusion_zone":
            self._finish_mark_occlusion_zone()
```

In `_on_canvas_right_click()`, add:
```python
        if self.interaction_mode == "mark_occlusion_zone":
            self._finish_mark_occlusion_zone()
            return
```

In `_on_canvas_release()` (line 1657), update the guard to include the new mode:
```python
        if self.interaction_mode in ("bumper_pick", "mark_zone_polygon", "mark_hp_line", "mark_occlusion_zone"):
            return
```

- [ ] **Step 5: Add _finish_mark_occlusion_zone method**

```python
    def _finish_mark_occlusion_zone(self):
        """完成遮擋區域多邊形標記。"""
        if len(self._polygon_points) < 3:
            self._set_status("至少需要 3 個頂點才能完成多邊形", COLORS["error"])
            return

        zone_name = f"遮擋區域 {len(self._occlusion_zones) + 1}"
        zone = OcclusionZone(zone_name, list(self._polygon_points))
        self._occlusion_zones.append(zone)
        self._update_zone_list()

        self.interaction_mode = None
        self._polygon_points = []
        self.canvas.config(cursor="")
        self._set_status(f"已設定 {zone_name}", COLORS["success"])
        self._show_frame(self.current_frame)
```

- [ ] **Step 6: Update _update_zone_list to show occlusion zones**

In `_update_zone_list()` (line 1907), after HP lines, add:

```python
        for oz in self._occlusion_zones:
            lines.append(f"  {oz.name} ({len(oz.polygon)} 頂點) [遮擋]")
```

- [ ] **Step 7: Update _clear_all_marks to clear occlusion zones**

In `_clear_all_marks()` (line 1875), add:
```python
        self._occlusion_zones.clear()
```

- [ ] **Step 8: Commit**

```bash
git add app.py
git commit -m "feat: add occlusion zone drawing UI"
```

---

### Task 8: Render occlusion zones + LOST robot boxes

**Files:**
- Modify: `app.py` — rendering methods

- [ ] **Step 1: Render occlusion zone overlay**

In `_render_frame` (paused rendering, around line 860) and `_render_frame_playback` (around line 1000), after the scoring zone polylines rendering, add occlusion zone rendering:

```python
        # 遮擋區域（半透明灰色）
        for oz in self._occlusion_zones:
            pts = [self._video_to_resized(p, scale) for p in oz.polygon]
            pts_np = np.array(pts, dtype=np.int32)
            overlay = resized.copy()
            cv2.fillPoly(overlay, [pts_np], (80, 80, 80))
            cv2.addWeighted(overlay, 0.3, resized, 0.7, 0, resized)
            cv2.polylines(resized, [pts_np], isClosed=True,
                          color=(128, 128, 128), thickness=1,
                          lineType=cv2.LINE_AA)
```

- [ ] **Step 2: Render LOST robots with dashed semi-transparent box**

In `_draw_analysis_overlay_impl()` (line 1094), the current code renders robot boxes for robots in `_robot_positions_cache`. LOST robots won't be in the cache (they have no detection that frame). We need to add LOST rendering.

After the existing robot bbox rendering block (after line 1134), add:

```python
        # LOST 機器人：虛線框在最後已知位置
        if hasattr(self, '_robot_tracker') and self._analysis_robot_mgr:
            tracker = self._analysis_robot_mgr
            if hasattr(tracker, '_impl') and hasattr(tracker._impl, '_track_state'):
                impl = tracker._impl
                for label, state in impl._track_state.items():
                    if state != "lost":
                        continue
                    if label in (self._robot_positions_cache.get(frame_idx) or {}):
                        continue  # 已經渲染過
                    lk = impl._last_known.get(label)
                    if not lk:
                        continue
                    last_cx, last_cy = lk[1], lk[2]
                    color = self._get_robot_color(label)
                    # 用最後已知的 bbox（從 _bboxes 歷史記錄取最後一筆）
                    bbox_list = impl._bboxes.get(label, [])
                    bbox = (bbox_list[-1][1], bbox_list[-1][2],
                            bbox_list[-1][3], bbox_list[-1][4]) if bbox_list else None
                    if bbox:
                        x1, y1, x2, y2 = bbox
                        p1 = self._video_to_resized((x1, y1), scale)
                        p2 = self._video_to_resized((x2, y2), scale)
                        # 虛線效果：用短線段模擬
                        self._draw_dashed_rect(
                            resized, p1, p2, color["bgr"], thickness=1)
                    else:
                        pt = self._video_to_resized(
                            (last_cx, last_cy), scale)
                        cv2.circle(resized, pt, 8,
                                   (128, 128, 128), 1, cv2.LINE_AA)
```

- [ ] **Step 3: Add _draw_dashed_rect helper**

```python
    def _draw_dashed_rect(self, img, p1, p2, color, thickness=1,
                          dash_len=8, gap_len=5):
        """繪製虛線矩形。"""
        x1, y1 = p1
        x2, y2 = p2
        edges = [
            ((x1, y1), (x2, y1)),  # top
            ((x2, y1), (x2, y2)),  # right
            ((x2, y2), (x1, y2)),  # bottom
            ((x1, y2), (x1, y1)),  # left
        ]
        for (ex1, ey1), (ex2, ey2) in edges:
            length = math.hypot(ex2 - ex1, ey2 - ey1)
            if length == 0:
                continue
            dx = (ex2 - ex1) / length
            dy = (ey2 - ey1) / length
            pos = 0.0
            while pos < length:
                start = (int(ex1 + dx * pos), int(ey1 + dy * pos))
                end_pos = min(pos + dash_len, length)
                end = (int(ex1 + dx * end_pos), int(ey1 + dy * end_pos))
                cv2.line(img, start, end, color, thickness, cv2.LINE_AA)
                pos += dash_len + gap_len
```

- [ ] **Step 4: Render LOST labels in _draw_analysis_labels_impl**

In `_draw_analysis_labels_impl()` (line 1171), after the existing label rendering, add LOST labels:

```python
        # LOST 機器人標籤（半透明）
        if hasattr(self, '_robot_tracker') and self._analysis_robot_mgr:
            tracker = self._analysis_robot_mgr
            if hasattr(tracker, '_impl') and hasattr(tracker._impl, '_track_state'):
                impl = tracker._impl
                for label, state in impl._track_state.items():
                    if state != "lost":
                        continue
                    if label in (self._robot_positions_cache.get(frame_idx) or {}):
                        continue
                    lk = impl._last_known.get(label)
                    if not lk:
                        continue
                    last_cx, last_cy = lk[1], lk[2]
                    color = self._get_robot_color(label)
                    pt = self._video_to_resized((last_cx, last_cy - 15), scale)
                    # 半透明效果：用灰色文字
                    draw.text((pt[0], max(0, pt[1] - 16)),
                              f"{label} [LOST]",
                              fill=(128, 128, 128), font=self._label_font)
```

- [ ] **Step 5: Commit**

```bash
git add app.py
git commit -m "feat: render occlusion zones + LOST robot dashed boxes"
```

---

### Task 9: Pass occlusion zones to tracker + preset save/load

**Files:**
- Modify: `app.py` — analysis setup
- Modify: `runtime_config.py` — preset serialization

- [ ] **Step 1: Pass occlusion zones to tracker during analysis**

In `_on_analyze()` or wherever `RobotTracker` is created (search for `robot_mgr` or `RobotTracker` creation), after tracker creation, add:

```python
        if hasattr(robot_mgr, 'set_occlusion_zones'):
            robot_mgr.set_occlusion_zones(self._occlusion_zones)
```

Also store reference for rendering:
```python
        self._analysis_robot_mgr = robot_mgr
```

- [ ] **Step 2: Add set_occlusion_zones to RobotTracker wrapper**

In the `RobotTracker` wrapper class at the end of `robot_tracker.py` (check if it delegates to `_impl`), add:

```python
    def set_occlusion_zones(self, zones):
        if hasattr(self._impl, 'set_occlusion_zones'):
            self._impl.set_occlusion_zones(zones)
```

- [ ] **Step 3: Note on persistence**

Scoring zones (`_scoring_zones`) and HP lines (`_hp_lines`) are currently NOT saved to preset JSON — they are in-memory only, set per analysis session via UI interaction. Occlusion zones follow the same pattern: stored in `self._occlusion_zones` in-memory, cleared when `_clear_all_marks()` is called. Preset save/load for zones is deferred to a future iteration (same as scoring zones).

- [ ] **Step 4: Verify module loads**

Run: `python -c "import app; print('OK')"`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add app.py robot_tracker.py
git commit -m "feat: wire occlusion zones to tracker + analysis integration"
```

---

## Chunk 4: Integration Verification

### Task 10: End-to-end verification

**Files:** None (verification only)

- [ ] **Step 1: Verify all imports resolve**

Run: `python -c "import config; import geometry; import robot_tracker; import app; print('ALL OK')"`
Expected: `ALL OK`

- [ ] **Step 2: Verify app launches**

Run: `python main.py`
Expected: App window opens without errors

- [ ] **Step 3: Manual test checklist**

1. Open a video file
2. Mark a scoring zone (Hub)
3. Click "遮擋區域" button → mark a polygon → right-click to finish
4. Verify occlusion zone renders as semi-transparent gray
5. Run analysis with auto mode
6. During playback, observe:
   - Robots entering the hub area go LOST (dashed box appears)
   - LOST boxes stay frozen at last known position
   - Robots exiting hub get re-associated (ID preserved)
   - Static robots keep their labels displayed
7. Check console output for `[DIAG] MOT ... lost=N` messages

- [ ] **Step 4: Final commit if any fixes needed**

```bash
git add -u
git commit -m "fix: integration fixes for MOT occlusion handling"
```

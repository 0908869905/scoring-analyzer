# MOT 遮擋處理設計：Track State Machine + 遮擋區域感知

**Date:** 2026-03-13
**Status:** Approved
**Branch:** `feat/gemini-detector`

---

## Problem

When a robot is fully occluded by the hub, the MOT tracker's bounding box "wanders" and latches onto other robots. This causes ID switches and corrupts tracking data downstream (shooter attribution, ball ownership, etc.).

### Root Causes

1. **No "lost" state** — Occluded robots remain in `_last_known` and compete for new detections on equal footing with visible robots.
2. **Velocity extrapolation drifts** — `pred_cx = last_cx + vx * gap` projects the predicted position into other robots' territory during long occlusions.
3. **Overly permissive distance threshold** — `MOT_REID_MAX_DIST=400px` with dynamic scaling allows matches up to ~700px+ at moderate frame gaps.
4. **No maximum consecutive lost frames in MOT** — Unlike SOT's `ROBOT_MAX_LOST_FRAMES=30`, MOT tracks survive 150 frames (5 sec) with no detection.
5. **Interpolation smooths over jumps** — Erroneous ID switches during occlusion get linearly interpolated, making them harder to detect post-hoc.

---

## Design

### 1. Track State Machine

Introduce a 3-state lifecycle for each tracked robot in `_MOTTracker`:

```
ACTIVE ──(consecutive missed detections)──> LOST ──(exceeds max_lost_frames)──> REMOVED
  ^                                           |
  └──────(re-detected + Re-ID match)──────────┘
```

#### State Definitions

| State | Participates in main matching | Position prediction | UI display | Re-association |
|-------|-------------------------------|--------------------:|------------|----------------|
| **ACTIVE** | Yes (Round 1) | Velocity extrapolation | Solid box | N/A |
| **LOST** | No — Round 2 only (leftover detections) | **Frozen** at last known position | Dashed semi-transparent box at last position | Stricter threshold: `max_dist * 0.5` + histogram similarity >= 0.3 |
| **REMOVED** | No | None | Not displayed | Cannot be revived |

#### Two-Round Matching in `_match_direct()`

**Round 1 (existing logic, restricted to ACTIVE tracks):**
1. Build distance matrix: detections × ACTIVE tracks only
2. Greedy match by shortest effective distance (spatial + histogram)
3. Update matched ACTIVE tracks normally (position, velocity, histogram)

**Round 2 (new, LOST track revival):**
1. Take remaining unmatched detections from Round 1
2. For each unmatched detection, compute distance to all LOST tracks using **frozen position** (no velocity extrapolation)
3. Apply stricter thresholds:
   - `max_dist = MOT_REID_MAX_DIST * MOT_LOST_REID_DIST_SCALE` (200px)
   - Histogram similarity >= `MOT_LOST_MIN_HIST_SIM` (0.3)
   - Class compatibility still enforced
4. If matched, revive track: state → ACTIVE, reset `_lost_since`, update position/velocity

**Unmatched ACTIVE tracks** (not detected this frame):
- Transition to LOST state
- Record `_lost_since[label] = frame_idx`
- Freeze position (stop velocity extrapolation)

**LOST tracks exceeding patience:**
- If `frame_idx - _lost_since[label] > max_patience` → REMOVED
- REMOVED tracks are deleted from `_last_known`, `_track_state`, `_lost_since`
- Their historical `_positions` and `_bboxes` data is **preserved** for post-processing

#### New Data Structures (`_MOTTracker`)

```python
self._track_state: dict[str, str] = {}   # {label: "active" | "lost" | "removed"}
self._lost_since: dict[str, int] = {}    # {label: frame_idx when LOST started}
```

#### New Constants (`config.py`)

```python
MOT_MAX_LOST_FRAMES = 90           # Default LOST patience (3s @ 30fps)
MOT_OCCLUSION_PATIENCE = 450       # LOST patience in occlusion zones (15s @ 30fps)
MOT_LOST_REID_DIST_SCALE = 0.5     # Distance threshold multiplier for LOST revival
MOT_LOST_MIN_HIST_SIM = 0.3        # Minimum histogram similarity for LOST revival
```

---

### 2. Occlusion Zone Awareness

Users can mark occlusion zones (e.g., hub) on the field. Robots that go LOST near these zones receive extended patience.

#### Patience Calculation

```python
last_pos = (last_cx, last_cy)
in_occlusion = any(point_in_polygon(last_pos, zone.polygon)
                    for zone in occlusion_zones)
max_patience = MOT_OCCLUSION_PATIENCE if in_occlusion else MOT_MAX_LOST_FRAMES
# 450 frames (15s) vs 90 frames (3s)
```

#### Data Model

```python
# In app.py, alongside _scoring_zones
self._occlusion_zones: list[OcclusionZone] = []

# OcclusionZone (simple dataclass):
@dataclass
class OcclusionZone:
    name: str
    polygon: list[tuple[int, int]]  # vertex coordinates in video space
```

#### UI

- Reuse existing zone-drawing interaction (same as scoring zones)
- New mode/button: "Mark Occlusion Zone"
- Occlusion zones rendered as semi-transparent gray overlay
- Saved/loaded from preset JSON under `"occlusion_zones"` key

#### Integration with Tracker

- `app.py` passes `occlusion_zones` to `RobotTracker` (or to `_match_direct()` via parameter)
- Tracker checks robot's last known position against zones when transitioning to LOST
- No zone marked → system uses default `MOT_MAX_LOST_FRAMES` everywhere

---

### 3. FPS Default Correction (60→30)

#### Constants Update

| Constant | Old value | New value | Reason |
|----------|-----------|-----------|--------|
| `MOT_STATIC_MIN_FRAMES` | 60 | 30 | Was "1 sec @ 60fps", now "1 sec @ 30fps" |
| `MOT_MIN_TRACK_FRAMES` | 15 | 15 (unchanged) | 0.5s @ 30fps is still reasonable |
| Comments throughout | "60fps 下" | "30fps 下" | Accuracy |

`app.py`'s `self.fps = 30.0` is already correct. The actual video FPS is read from the file at load time.

---

### 4. Static Robot Label Display Fix

#### Current Behavior (Problem)

`filter_static_labels()` completely removes robots with position variance < 100 (i.e., nearly stationary). This deletes:
- `_positions[label]`
- `_bboxes[label]`
- `_robot_info[label]`
- `_last_known[label]`
- All entries in `_frame_positions` and `_frame_bboxes`

Real robots that park near the hub (e.g., waiting to receive a ball) get erased.

#### New Behavior

Change `filter_static_labels()` from **deletion** to **marking**:

```python
self._static_labels: set[str] = set()  # Labels flagged as stationary
```

- Static labels are **not removed** from any data structure
- They remain visible in the UI with normal label display
- Their tracking data is preserved for scoring/attribution
- Optional: render with a subtle visual distinction (e.g., thinner box outline)

If a static label later shows movement (variance exceeds threshold), remove it from `_static_labels`.

---

## Files Changed

| File | Changes |
|------|---------|
| `config.py` | Add `MOT_MAX_LOST_FRAMES`, `MOT_OCCLUSION_PATIENCE`, `MOT_LOST_REID_DIST_SCALE`, `MOT_LOST_MIN_HIST_SIM`; update `MOT_STATIC_MIN_FRAMES` 60→30; fix FPS comments |
| `robot_tracker.py` | Add `_track_state`, `_lost_since`, `_static_labels` to `_MOTTracker`; refactor `_match_direct()` into two-round matching; modify `filter_static_labels()` to mark-only; add LOST→REMOVED timeout logic |
| `app.py` | Add `_occlusion_zones` list + zone drawing UI; render LOST robots with dashed semi-transparent box; pass occlusion zones to tracker |
| Preset JSON | Add `"occlusion_zones"` field |

## Files NOT Changed

| File | Reason |
|------|--------|
| `scoring.py` | Attribution logic unchanged; benefits from better tracking input |
| `robot_detection.py` | Detector unchanged |
| `background.py` | Background model unchanged |
| `detection.py` | Ball detection unchanged |
| `settings_window.py` | Occlusion zone settings deferred to future iteration |

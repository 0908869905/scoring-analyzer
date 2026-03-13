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
ACTIVE ──(grace period expired)──> LOST ──(exceeds max_lost_frames)──> REMOVED
  ^                                          |
  └──────(re-detected + Re-ID match)─────────┘
```

#### State Definitions

| State | Participates in main matching | Position prediction | UI display | Re-association |
|-------|-------------------------------|--------------------:|------------|----------------|
| **ACTIVE** | Yes (Round 1) | Velocity extrapolation | Solid box | N/A |
| **LOST** | No — Round 2 only (leftover detections) | **Frozen** at last known position | Dashed semi-transparent box at last position | Stricter threshold: `max_dist * 0.5` + histogram similarity >= 0.3 |
| **REMOVED** | No | None | Not displayed | Cannot be revived |

#### Grace Period (ACTIVE → LOST transition)

A single missed detection does NOT immediately transition to LOST. Instead, tracks have a **grace period** of `MOT_LOST_GRACE_FRAMES` (default: 3 frames). During the grace period:
- The track remains ACTIVE and participates in Round 1 matching
- Velocity extrapolation continues normally
- A counter `_missed_frames[label]` increments each frame without detection
- If detected during grace period, the counter resets to 0

Only when `_missed_frames[label] >= MOT_LOST_GRACE_FRAMES` does the track transition to LOST.

#### Two-Round Matching in `_match_direct()`

**Round 1 (existing logic, restricted to ACTIVE tracks):**
1. Build distance matrix: detections × ACTIVE tracks only (including those in grace period)
2. Greedy match by shortest effective distance (spatial + histogram)
3. Update matched ACTIVE tracks normally (position, velocity, histogram)

**Round 2 (new, LOST track revival):**
1. Take remaining unmatched detections from Round 1
2. For each unmatched detection, compute distance to all LOST tracks using **frozen position** (no velocity extrapolation)
3. Apply stricter thresholds:
   - `max_dist = MOT_REID_MAX_DIST * MOT_LOST_REID_DIST_SCALE` (200px)
   - Histogram similarity >= `MOT_LOST_MIN_HIST_SIM` (0.3)
   - Class compatibility still enforced
4. If matched, revive track: state → ACTIVE, reset `_lost_since` and `_missed_frames`, update position/velocity

**Unmatched ACTIVE tracks** (not detected this frame):
- Increment `_missed_frames[label]`
- If `_missed_frames[label] >= MOT_LOST_GRACE_FRAMES`:
  - Transition to LOST state
  - Record `_lost_since[label] = frame_idx`
  - Freeze position (stop velocity extrapolation)

**LOST tracks exceeding patience:**
- If `frame_idx - _lost_since[label] > max_patience` → REMOVED
- REMOVED tracks are deleted from `_last_known`, `_track_state`, `_lost_since`, `_missed_frames`
- Their historical `_positions` and `_bboxes` data is **preserved** for post-processing

#### Superseding `MOT_REID_MAX_SECONDS`

The new `MOT_MAX_LOST_FRAMES` replaces `MOT_REID_MAX_SECONDS` (which computed `_reid_max_frames = fps * 5 = 150`). The old `_reid_max_frames` guard in `_match_direct()` is removed; track lifetime is now governed entirely by the state machine:
- ACTIVE tracks: always matchable (no time limit for Round 1)
- LOST tracks: governed by `MOT_MAX_LOST_FRAMES` / `MOT_OCCLUSION_PATIENCE`
- REMOVED tracks: gone

`MOT_REID_MAX_SECONDS` is removed from `config.py`.

#### New Data Structures (`_MOTTracker`)

```python
self._track_state: dict[str, str] = {}     # {label: "active" | "lost"}
self._lost_since: dict[str, int] = {}      # {label: frame_idx when LOST started}
self._missed_frames: dict[str, int] = {}   # {label: consecutive frames without detection}
```

Note: REMOVED tracks are deleted from all dicts, not stored with a "removed" state value.

#### Cleanup Requirements

All new data structures (`_track_state`, `_lost_since`, `_missed_frames`) must be cleaned up in:
- `clear()` — clear all three dicts
- `filter_short_labels()` — remove entries for deleted labels
- `merge_fragmented_labels()` / `_execute_merges()` — when merging `short_label` into `long_label`, remove `short_label` entries; keep `long_label` entries as-is

#### New Constants (`config.py`)

```python
MOT_MAX_LOST_FRAMES = 90           # Default LOST patience (3s @ 30fps)
MOT_OCCLUSION_PATIENCE = 450       # LOST patience in occlusion zones (15s @ 30fps)
MOT_LOST_GRACE_FRAMES = 3          # Missed frames before ACTIVE → LOST transition
MOT_LOST_REID_DIST_SCALE = 0.5     # Distance threshold multiplier for LOST revival
MOT_LOST_MIN_HIST_SIM = 0.3        # Minimum histogram similarity for LOST revival
```

#### Removed Constants

```python
# REMOVED: MOT_REID_MAX_SECONDS — superseded by MOT_MAX_LOST_FRAMES state machine
```

---

### 2. Occlusion Zone Awareness

Users can mark occlusion zones (e.g., hub) on the field. Robots that go LOST near these zones receive extended patience.

#### Patience Calculation

```python
last_cx, last_cy = last_known_position
in_occlusion = any(point_in_polygon(last_cx, last_cy, zone.polygon)
                    for zone in occlusion_zones)
max_patience = MOT_OCCLUSION_PATIENCE if in_occlusion else MOT_MAX_LOST_FRAMES
# 450 frames (15s) vs 90 frames (3s)
```

#### Occlusion Zone Margin

A robot approaching the hub may have its last detection slightly outside the zone polygon. To handle this, the occlusion check includes a **margin buffer**: expand each zone polygon outward by `MOT_OCCLUSION_MARGIN` pixels (default: 50px) before testing point containment.

Implementation: instead of modifying the polygon geometry, use a distance-based fallback:
```python
in_occlusion = any(
    point_in_polygon(last_cx, last_cy, zone.polygon)
    or min_distance_to_polygon_edge(last_cx, last_cy, zone.polygon) < MOT_OCCLUSION_MARGIN
    for zone in occlusion_zones
)
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

- `app.py` passes `occlusion_zones` to `RobotTracker` via `set_occlusion_zones(zones)` method
- Tracker stores zones and checks robot's last known position against them when transitioning ACTIVE → LOST
- No zone marked → system uses default `MOT_MAX_LOST_FRAMES` everywhere

#### New Constants

```python
MOT_OCCLUSION_MARGIN = 50    # Buffer around occlusion zones (px) for patience check
```

---

### 3. FPS Default Correction (60→30)

#### Constants Update

| Constant | Old value | Old comment | New value | New comment |
|----------|-----------|-------------|-----------|-------------|
| `MOT_STATIC_MIN_FRAMES` | 60 | "60fps 下 1 秒" | 30 | "30fps 下 1 秒" |
| `MOT_MIN_TRACK_FRAMES` | 15 | "60fps 下 15 幀 ≈ 0.25 秒" | 15 (unchanged) | "30fps 下 ≈ 0.5 秒" |
| `MOT_MERGE_MAX_OVERLAP` | 15 | "60fps 下 ≈ 250ms" | 15 (unchanged) | "30fps 下 ≈ 500ms" |
| `MOT_MERGE_SEARCH_WINDOW` | 180 | "60fps 下 ≈ 3 秒" | 180 (unchanged) | "30fps 下 ≈ 6 秒" |
| `BYTETRACK_LOST_BUFFER` | 120 | "4秒@30fps" | 120 (unchanged) | already correct |

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

#### Static Label Interaction with State Machine

- Static marking is a **post-processing annotation**, applied after `merge_fragmented_labels()` and `filter_short_labels()`
- It does not affect the ACTIVE/LOST state machine during live tracking
- Static labels in ACTIVE state remain ACTIVE and participate in Round 1 matching normally
- A LOST track can be marked as static if it was stationary before going LOST (no special treatment)

#### Un-marking Timing

The static check runs only once during post-processing (same as current `filter_static_labels()`). It does not run incrementally during live tracking. Since this is a display-only annotation, real-time un-marking is unnecessary.

#### Cleanup

`_static_labels` must be cleared in `clear()`.

---

### 5. Interpolation Behavior During LOST Periods

When a track goes LOST at frame N and is revived at frame M, the frames N..M will have no entries in `_positions` or `_bboxes`. The existing `interpolate_positions()` will linearly interpolate through this gap, producing positions inside the occlusion zone.

This is **acceptable and expected** — post-processing interpolation provides a smooth trajectory for visualization and scoring attribution. The interpolated positions during LOST periods are visually distinguished by the existing `_detected_frames` mechanism (thin gray line vs solid line).

---

## Files Changed

| File | Changes |
|------|---------|
| `config.py` | Add `MOT_MAX_LOST_FRAMES`, `MOT_OCCLUSION_PATIENCE`, `MOT_LOST_GRACE_FRAMES`, `MOT_LOST_REID_DIST_SCALE`, `MOT_LOST_MIN_HIST_SIM`, `MOT_OCCLUSION_MARGIN`; remove `MOT_REID_MAX_SECONDS`; update `MOT_STATIC_MIN_FRAMES` 60→30; fix all FPS comments (including `MOT_MERGE_*`, `MOT_MIN_TRACK_FRAMES`) |
| `robot_tracker.py` | Add `_track_state`, `_lost_since`, `_missed_frames`, `_static_labels` to `_MOTTracker`; remove `_reid_max_frames`; refactor `_match_direct()` into two-round matching with grace period; modify `filter_static_labels()` to mark-only; add LOST→REMOVED timeout logic; add `set_occlusion_zones()` method; clean up new dicts in `clear()`, `filter_short_labels()`, `merge_fragmented_labels()` |
| `app.py` | Add `_occlusion_zones` list + zone drawing UI; render LOST robots with dashed semi-transparent box; pass occlusion zones to tracker via `set_occlusion_zones()` |
| `geometry.py` | Add `min_distance_to_polygon_edge()` helper (for occlusion margin check) |
| Preset JSON | Add `"occlusion_zones"` field |

## Files NOT Changed

| File | Reason |
|------|--------|
| `scoring.py` | Attribution logic unchanged; benefits from better tracking input |
| `robot_detection.py` | Detector unchanged |
| `background.py` | Background model unchanged |
| `detection.py` | Ball detection unchanged |
| `settings_window.py` | Occlusion zone settings deferred to future iteration |

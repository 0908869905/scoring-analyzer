# HSV Bumper Detection Design

## Goal
Replace YOLO-based robot detection with HSV color filtering of red/blue bumpers. No training data needed.

## Architecture

`BumperDetectorHSV` class in `robot_detection.py` implements the same `detect()` interface as `RobotDetectorONNX`. Drop-in replacement — tracker code unchanged.

```
app.py
  └→ RobotTrackerManager(detector=BumperDetectorHSV())
       └→ _MOTTracker._detector.detect(frame)  # same interface
```

## HSV Ranges

- **Red bumper** (hue wraps): H=0~10 OR H=170~180, S=120~255, V=80~255
- **Blue bumper**: H=100~130, S=120~255, V=60~255

## Pipeline

1. GaussianBlur(5,5) → cvtColor BGR2HSV
2. inRange for red (two ranges, OR merge) and blue (one range)
3. morphologyEx Close(9,5 rect) + Open(5,3 rect) — 矩形核適合水平 bumper
4. findContours → filter by area → boundingRect
5. Return `(x1, y1, x2, y2, conf, class_id)` — same as YOLO

## Key Decisions

- conf = simulated score based on area and saturation (not ML confidence)
- class_id: 0=Red, 1=Blue (matches existing YOLO model class order)
- class_names: ["Red", "Blue"]
- detect_tiled() not needed (HSV runs on full resolution natively)
- Field mask already filters scene noise (audience, field elements)

## Files Changed

| File | Change |
|------|--------|
| `robot_detection.py` | Add `BumperDetectorHSV` class |
| `config.py` | Add `BUMPER_RED_*` / `BUMPER_BLUE_*` HSV params |
| `app.py` | Switch detector init to `BumperDetectorHSV()` when no YOLO model |
| `robot_tracker.py` | No changes |
| `scoring.py` | No changes |

## Success Criteria

- Red and Blue bumpers detected reliably on both side-view and broadcast footage
- Output format compatible with existing MOT tracker
- No YOLO model dependency required

"""
FRC Scoring Analyzer — 無頭分析測試（前 N 秒）

用法:
    python test_analysis.py <video_path> [--seconds 20]
    python test_analysis.py <video_path> --seconds 20 --no-zones

功能:
- 開始前可互動框選 Hub 區域（Scoring Zone）
- 分析球偵測 + 機器人追蹤 + 進球 + 出手
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

from config import MAX_MATCH_DIST, MAX_MISSED
from detection import detect_yellow_balls, reset_diagnostics
from tracking import CentroidTracker
from robot_tracker import RobotTrackerManager
from scoring import ScoringEngine, ScoringZone
from runtime_config import RuntimeConfig


# ────────────────────────────────────────────────────
# 互動式 Scoring Zone 框選
# ────────────────────────────────────────────────────

def draw_zones_interactive(frame: np.ndarray) -> list[ScoringZone]:
    """在影片第一幀上讓用戶框選 Hub 區域。

    操作:
      左鍵點擊 = 新增頂點
      Z        = 撤銷上一個頂點
      Enter    = 完成當前區域，開始下一個
      Q / Esc  = 完成所有區域，開始分析
    """
    zones: list[ScoringZone] = []
    current_points: list[tuple[int, int]] = []

    zone_presets = [
        ("Red Hub", "red", (0, 0, 255)),
        ("Blue Hub", "blue", (255, 100, 0)),
    ]
    zone_idx = 0

    def on_mouse(event, x, y, flags, param):
        nonlocal current_points
        if event == cv2.EVENT_LBUTTONDOWN:
            current_points.append((x, y))

    win = "Draw Scoring Zones"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    h, w = frame.shape[:2]
    scale = min(1280 / w, 720 / h, 1.0)
    cv2.resizeWindow(win, int(w * scale), int(h * scale))
    cv2.setMouseCallback(win, on_mouse)

    print()
    print("-" * 40)
    print("  Scoring Zone Drawing")
    print("-" * 40)
    print("  Left click = add vertex")
    print("  Z          = undo last vertex")
    print("  Enter      = finish current zone")
    print("  Q / Esc    = done, start analysis")
    print()

    while True:
        disp = frame.copy()

        # 已完成的 zone（半透明填色）
        for z in zones:
            pts = np.array(z.points, np.int32).reshape((-1, 1, 2))
            c = (0, 0, 255) if z.alliance == "red" else (255, 100, 0)
            overlay = disp.copy()
            cv2.fillPoly(overlay, [pts], c)
            cv2.addWeighted(overlay, 0.25, disp, 0.75, 0, disp)
            cv2.polylines(disp, [pts], True, c, 3)
            cx, cy = int(z.center[0]), int(z.center[1])
            cv2.putText(disp, z.name, (cx - 60, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, c, 3)

        # 正在繪製的點和線
        if current_points:
            for pt in current_points:
                cv2.circle(disp, pt, 8, (0, 255, 0), -1)
            if len(current_points) > 1:
                cv2.polylines(
                    disp,
                    [np.array(current_points, np.int32)],
                    False, (0, 255, 0), 2)
            if len(current_points) >= 3:
                cv2.polylines(
                    disp,
                    [np.array(current_points, np.int32)],
                    True, (0, 255, 0), 1)

        # 頂部指示列
        cv2.rectangle(disp, (0, 0), (w, 50), (0, 0, 0), -1)
        if zone_idx < len(zone_presets):
            name = zone_presets[zone_idx][0]
            txt = f"Draw: {name} | click vertices, Enter=finish, Q=done"
        else:
            txt = "Press Q to start analysis, or Enter to add more zones"
        cv2.putText(disp, txt, (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow(win, disp)
        key = cv2.waitKey(30) & 0xFF

        if key == 13:  # Enter — 完成當前 zone
            if len(current_points) >= 3:
                if zone_idx < len(zone_presets):
                    name, alliance, _ = zone_presets[zone_idx]
                else:
                    name, alliance = f"Zone-{zone_idx + 1}", ""
                zones.append(ScoringZone(
                    name=name,
                    points=list(current_points),
                    alliance=alliance))
                print(f"  [+] {name}: {len(current_points)} vertices")
                zone_idx += 1
                current_points = []
            elif current_points:
                print("  (need >= 3 vertices)")

        elif key == ord('q') or key == 27:  # Q/Esc — 結束
            if len(current_points) >= 3:
                if zone_idx < len(zone_presets):
                    name, alliance, _ = zone_presets[zone_idx]
                else:
                    name, alliance = f"Zone-{zone_idx + 1}", ""
                zones.append(ScoringZone(
                    name=name,
                    points=list(current_points),
                    alliance=alliance))
                print(f"  [+] {name}: {len(current_points)} vertices")
            break

        elif key == ord('z'):  # Z — 撤銷
            if current_points:
                current_points.pop()

    cv2.destroyAllWindows()
    cv2.waitKey(1)
    return zones


# ────────────────────────────────────────────────────
# 主流程
# ────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="FRC Scoring Analyzer — analysis test")
    parser.add_argument("video", help="video path")
    parser.add_argument("--seconds", type=int, default=20,
                        help="analyze first N seconds (default 20)")
    parser.add_argument("--no-zones", action="store_true",
                        help="skip interactive zone drawing")
    args = parser.parse_args()

    if not os.path.isfile(args.video):
        print(f"Video not found: {args.video}")
        sys.exit(1)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Cannot open: {args.video}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    max_frame = min(int(fps * args.seconds), total_frames)

    print("=" * 60)
    print("FRC Scoring Analyzer — Test")
    print("=" * 60)
    print(f"Video: {args.video}")
    print(f"Resolution: {width}x{height}, FPS: {fps:.1f}")
    print(f"Range: first {args.seconds}s ({max_frame} frames)")

    cfg = RuntimeConfig()
    engine = ScoringEngine(fps=fps)

    # ── Zone drawing ──────────────────────────────
    if not args.no_zones:
        ret, first_frame = cap.read()
        if ret:
            zones = draw_zones_interactive(first_frame)
            for z in zones:
                engine.add_zone(z.name, z.points, z.alliance)
            if zones:
                print(f"\n[OK] {len(zones)} scoring zone(s) set")
            else:
                print("\n[INFO] No zones drawn — goal detection disabled")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    else:
        print("[INFO] --no-zones: skipping zone drawing")

    # ── Robot model ───────────────────────────────
    robot_detector = None
    try:
        from robot_detection import load_robot_model
        robot_detector = load_robot_model()
        print("[OK] Robot detection model loaded")
    except Exception as e:
        print(f"[WARN] Robot model: {e}")

    # ── Init ──────────────────────────────────────
    reset_diagnostics()
    ball_tracker = CentroidTracker(
        max_distance=cfg.max_match_dist, max_missed=cfg.max_missed)
    robot_mgr = RobotTrackerManager(detector=robot_detector, fps=fps)
    tracking_mode = "MOT" if robot_mgr.use_mot else "SOT"
    print(f"[INFO] Tracking: {tracking_mode}")

    if robot_mgr.use_mot:
        robot_mgr.enable_auto_mode()
        print("[INFO] MOT auto mode enabled")

    # ── Stats ─────────────────────────────────────
    total_ball_dets = 0
    total_robot_dets = 0
    frames_with_balls = 0
    frames_with_robots = 0
    robot_labels_seen: set[str] = set()
    label_frame_counts: dict[str, int] = {}
    robot_positions_by_frame: dict[int, dict] = {}

    print()
    t_start = time.time()

    # ── Analysis loop ─────────────────────────────
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for frame_idx in range(max_frame):
        ret, frame = cap.read()
        if not ret:
            break

        # Ball detection
        dets = detect_yellow_balls(
            frame, hsv_low=cfg.hsv_low, hsv_high=cfg.hsv_high,
            min_area=cfg.min_blob_area, max_area=cfg.max_blob_area)
        total_ball_dets += len(dets)
        if dets:
            frames_with_balls += 1

        ball_positions = ball_tracker.update(dets, frame_idx)

        # Robot tracking
        robot_mgr.update_all(frame, frame_idx)
        robot_pos = robot_mgr.get_all_positions(frame_idx)
        total_robot_dets += len(robot_pos)
        if robot_pos:
            frames_with_robots += 1
            robot_labels_seen.update(robot_pos.keys())
            for lbl in robot_pos:
                label_frame_counts[lbl] = label_frame_counts.get(lbl, 0) + 1
            robot_positions_by_frame[frame_idx] = dict(robot_pos)

        # Goal detection (per frame)
        engine.process_frame(frame_idx, ball_positions, robot_pos,
                             ball_tracker.trajectories)

        # Progress
        if frame_idx > 0 and frame_idx % 100 == 0:
            elapsed = time.time() - t_start
            speed = frame_idx / elapsed
            print(f"  Frame {frame_idx}/{max_frame} "
                  f"({speed:.1f} fps) "
                  f"balls={total_ball_dets} "
                  f"robots={total_robot_dets} "
                  f"({len(robot_labels_seen)} labels) "
                  f"goals={len(engine.events)}")

    cap.release()
    elapsed = time.time() - t_start

    # ── Post-processing ───────────────────────────
    robot_mgr.merge_fragmented_labels()
    robot_mgr.filter_short_labels()
    robot_mgr.filter_static_labels()
    robot_mgr.interpolate_positions()

    # 用合併+插值後的位置重新收集統計和射球資料
    robot_labels_seen.clear()
    label_frame_counts.clear()
    robot_positions_by_frame.clear()
    total_robot_dets = 0
    frames_with_robots = 0

    for f in range(max_frame):
        pos = robot_mgr.get_all_positions(f)
        if pos:
            robot_positions_by_frame[f] = dict(pos)
            frames_with_robots += 1
            total_robot_dets += len(pos)
            robot_labels_seen.update(pos.keys())
            for lbl in pos:
                label_frame_counts[lbl] = label_frame_counts.get(lbl, 0) + 1

    # 射手重新歸因（用合併+插值後的完整位置資料）
    engine.reattribute_shooters(ball_tracker.trajectories,
                                robot_positions_by_frame)

    # 射球偵測（後處理：速度突增 + 機器人鄰近度）
    engine.detect_shots(ball_tracker.trajectories, robot_positions_by_frame)

    # ── Results ───────────────────────────────────
    print()
    print("=" * 60)
    print("Results")
    print("=" * 60)
    print(f"Time: {elapsed:.1f}s ({max_frame / elapsed:.1f} fps)")
    print()

    # Ball
    print("[Ball Detection]")
    print(f"  Total detections: {total_ball_dets}")
    pct = frames_with_balls / max_frame * 100 if max_frame else 0
    print(f"  Frames with balls: {frames_with_balls}/{max_frame} ({pct:.1f}%)")
    n_traj = len(ball_tracker.trajectories)
    long_traj = sum(1 for t in ball_tracker.trajectories.values()
                    if len(t) >= 10)
    print(f"  Trajectories: {n_traj} total, {long_traj} with >=10 points")
    if total_ball_dets == 0:
        print(f"  !! Zero! Try: python diagnose.py {args.video}")
    print()

    # Robot
    print(f"[Robot Tracking] ({tracking_mode})")
    print(f"  Total detections: {total_robot_dets}")
    pct = frames_with_robots / max_frame * 100 if max_frame else 0
    print(f"  Frames with robots: {frames_with_robots}/{max_frame} ({pct:.1f}%)")
    red_n = sum(1 for l in robot_labels_seen if l.startswith("Red"))
    blue_n = sum(1 for l in robot_labels_seen if l.startswith("Blue"))
    print(f"  Unique labels: {len(robot_labels_seen)} "
          f"(Red: {red_n}, Blue: {blue_n})")
    if label_frame_counts:
        print(f"  Per-robot:")
        for lbl in sorted(label_frame_counts.keys()):
            print(f"    {lbl}: {label_frame_counts[lbl]} frames")
    print()

    # ID Stability（過濾短軌跡 label）
    from config import MOT_MIN_TRACK_FRAMES
    stable_labels = {l for l, c in label_frame_counts.items()
                     if c >= MOT_MIN_TRACK_FRAMES}
    short_labels = {l for l, c in label_frame_counts.items()
                    if c < MOT_MIN_TRACK_FRAMES}
    n_stable = len(stable_labels)
    if n_stable <= 8:
        stability = "GOOD"
    elif n_stable <= 15:
        stability = "FAIR"
    else:
        stability = "POOR"
    print(f"[ID Stability] {stability} "
          f"({n_stable} stable labels, {len(short_labels)} short-lived, "
          f"expect <=6)")
    if short_labels:
        print(f"  Short-lived (< {MOT_MIN_TRACK_FRAMES} frames): "
              f"{sorted(short_labels)}")
    print()

    # Goals
    print("[Goals]")
    if not engine.zones:
        print("  (No zones — rerun without --no-zones to draw Hub areas)")
    print(f"  Events: {len(engine.events)}")
    for ev in engine.events:
        sec = ev.frame_idx / fps
        print(f"    @{sec:.1f}s: {ev.shooter_label} -> {ev.zone_name} "
              f"({ev.period}, dist={ev.shooter_dist:.0f}px)")
    print()

    # Shots
    print("[Shots]")
    n_goals = sum(1 for s in engine.shot_events if s.result == "goal")
    n_miss = sum(1 for s in engine.shot_events if s.result == "miss")
    print(f"  Total: {len(engine.shot_events)} "
          f"(goals: {n_goals}, misses: {n_miss})")
    if engine.shot_events:
        acc = n_goals / len(engine.shot_events) * 100
        print(f"  Accuracy: {acc:.0f}%")
    for s in engine.shot_events[:20]:  # 最多顯示 20 個
        sec = s.frame_idx / fps
        print(f"    @{sec:.1f}s: {s.shooter_label} {s.result} "
              f"(dist={s.shooter_dist:.0f}px)")
    if len(engine.shot_events) > 20:
        print(f"    ... and {len(engine.shot_events) - 20} more")


if __name__ == "__main__":
    main()

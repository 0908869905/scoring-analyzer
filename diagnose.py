"""
FRC Scoring Analyzer — 偵測診斷工具

用法:
    python diagnose.py <video_path> [--seconds 20] [--save-images]

功能:
    1. 球偵測診斷：逐步顯示 HSV 管線每一步的結果
    2. 機器人偵測診斷：測試 YOLO 模型在影片上的偵測結果
    3. 產生診斷報告 + 可選儲存診斷影像
"""

import argparse
import os
import sys

import cv2
import numpy as np

from config import YELLOW_LOW, YELLOW_HIGH, MIN_BLOB_AREA, MAX_BLOB_AREA
from runtime_config import RuntimeConfig


def diagnose_ball_detection(frame, frame_idx, cfg, save_dir=None):
    """診斷球偵測管線每一步。"""
    hsv_low = np.array(cfg.hsv_low)
    hsv_high = np.array(cfg.hsv_high)
    min_a = cfg.min_blob_area
    max_a = cfg.max_blob_area

    # Step 1: GaussianBlur
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)

    # Step 2: BGR → HSV
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Step 3: inRange
    mask = cv2.inRange(hsv, hsv_low, hsv_high)
    mask_pixels = cv2.countNonZero(mask)

    # Step 4: Morphology Close
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
    closed_pixels = cv2.countNonZero(mask_closed)

    # Step 5: Morphology Open
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_final = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel_open)
    final_pixels = cv2.countNonZero(mask_final)

    # Step 6: findContours
    contours, _ = cv2.findContours(
        mask_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Step 7: Area filter
    areas = [cv2.contourArea(c) for c in contours]
    passed = [a for a in areas if min_a <= a <= max_a]

    result = {
        "frame_idx": frame_idx,
        "hsv_range": f"H:{cfg.hsv_low[0]}-{cfg.hsv_high[0]} "
                     f"S:{cfg.hsv_low[1]}-{cfg.hsv_high[1]} "
                     f"V:{cfg.hsv_low[2]}-{cfg.hsv_high[2]}",
        "mask_pixels_after_inrange": mask_pixels,
        "mask_pixels_after_close": closed_pixels,
        "mask_pixels_after_open": final_pixels,
        "total_contours": len(contours),
        "contour_areas": sorted(areas, reverse=True)[:10],
        "passed_area_filter": len(passed),
        "area_range": f"{min_a}-{max_a}",
    }

    # 如果 mask 完全為空，取樣一些像素的 HSV 值來幫助診斷
    if mask_pixels == 0:
        h, w = frame.shape[:2]
        sample_points = [
            (w // 4, h // 4), (w // 2, h // 4), (3 * w // 4, h // 4),
            (w // 4, h // 2), (w // 2, h // 2), (3 * w // 4, h // 2),
            (w // 4, 3 * h // 4), (w // 2, 3 * h // 4), (3 * w // 4, 3 * h // 4),
        ]
        hsv_samples = []
        for px, py in sample_points:
            patch = hsv[max(0, py - 5):py + 5, max(0, px - 5):px + 5]
            if patch.size > 0:
                median_hsv = np.median(patch.reshape(-1, 3), axis=0).astype(int)
                hsv_samples.append(
                    f"({px},{py}): H={median_hsv[0]} S={median_hsv[1]} V={median_hsv[2]}")
        result["hsv_samples"] = hsv_samples

    # 儲存診斷影像
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

        # 原始幀
        cv2.imwrite(os.path.join(save_dir, f"frame_{frame_idx:05d}_original.jpg"),
                     frame)

        # HSV mask（inRange 後）
        cv2.imwrite(os.path.join(save_dir, f"frame_{frame_idx:05d}_mask_inrange.jpg"),
                     mask)

        # 最終 mask
        cv2.imwrite(os.path.join(save_dir, f"frame_{frame_idx:05d}_mask_final.jpg"),
                     mask_final)

        # Contour overlay
        overlay = frame.copy()
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                color = (0, 255, 0) if min_a <= area <= max_a else (0, 0, 255)
                cv2.putText(overlay, f"A={int(area)}", (cx, cy - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.imwrite(os.path.join(save_dir, f"frame_{frame_idx:05d}_contours.jpg"),
                     overlay)

    return result


def diagnose_robot_detection(frame, frame_idx, save_dir=None):
    """診斷機器人偵測（包含 tile-based 和普通偵測比較）。"""
    try:
        from robot_detection import load_robot_model
        detector = load_robot_model()
    except Exception as e:
        return {"error": str(e)}

    h, w = frame.shape[:2]
    results = {"resolution": f"{w}x{h}"}

    # 普通偵測 vs tile-based 偵測比較
    for mode_name, detect_fn in [
        ("normal", lambda conf: detector.detect(frame, confidence=conf, robot_only=False)),
        ("tiled", lambda conf: detector.detect_tiled(frame, confidence=conf, robot_only=False)),
    ]:
        mode_results = {}
        for conf in [0.1, 0.15, 0.2, 0.25, 0.3, 0.4]:
            dets = detect_fn(conf)
            robot_dets = [d for d in dets if detector.is_robot_class(d[5])]

            by_class = {}
            for d in robot_dets:
                cls_name = detector.get_class_name(d[5])
                by_class.setdefault(cls_name, []).append(round(d[4], 3))

            mode_results[f"conf_{conf}"] = {
                "total_all": len(dets),
                "total_robots": len(robot_dets),
                "by_class": {k: f"{len(v)} ({min(v):.2f}-{max(v):.2f})"
                             for k, v in by_class.items()} if by_class else {},
            }
        results[mode_name] = mode_results

    # 儲存偵測結果影像（使用 tiled + conf=0.25）
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        dets = detector.detect_tiled(frame, confidence=0.2, robot_only=True)
        overlay = frame.copy()
        for d in dets:
            x1, y1, x2, y2, conf, cls_id = d
            cls_name = detector.get_class_name(cls_id)
            alliance = detector.infer_alliance(cls_id)
            color = (0, 0, 255) if alliance == "red" else \
                    (255, 100, 0) if alliance == "blue" else (0, 255, 0)
            cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)),
                          color, 2)
            cv2.putText(overlay, f"{cls_name} {conf:.2f}",
                        (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.imwrite(
            os.path.join(save_dir, f"frame_{frame_idx:05d}_robots.jpg"),
            overlay)

    return results


def auto_detect_ball_hsv(frames):
    """嘗試自動偵測球的 HSV 範圍（找高飽和度、非灰色的運動區域）。"""
    if len(frames) < 2:
        return None

    # 用幀差法找運動區域
    moving_hsv_pixels = []

    for i in range(1, min(len(frames), 10)):
        prev_gray = cv2.cvtColor(frames[i - 1], cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)

        diff = cv2.absdiff(prev_gray, curr_gray)
        _, motion_mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

        # 只看運動區域中飽和度較高的像素（球通常比背景飽和度高）
        hsv = cv2.cvtColor(frames[i], cv2.COLOR_BGR2HSV)
        sat_mask = hsv[:, :, 1] > 50  # 飽和度 > 50

        combined_mask = (motion_mask > 0) & sat_mask
        if np.any(combined_mask):
            pixels = hsv[combined_mask]
            moving_hsv_pixels.append(pixels)

    if not moving_hsv_pixels:
        return None

    all_pixels = np.vstack(moving_hsv_pixels)
    if len(all_pixels) < 100:
        return None

    # K-Means 分群，找最顯眼的顏色
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    k = min(5, max(2, len(all_pixels) // 100))
    pixels_f32 = all_pixels.astype(np.float32)
    _, labels, centers = cv2.kmeans(
        pixels_f32, k, None, criteria, 5, cv2.KMEANS_PP_CENTERS)

    # 報告各 cluster
    cluster_info = []
    for i in range(k):
        cluster_pixels = all_pixels[labels.flatten() == i]
        count = len(cluster_pixels)
        center = centers[i]
        cluster_info.append({
            "cluster": i,
            "count": count,
            "center_H": int(center[0]),
            "center_S": int(center[1]),
            "center_V": int(center[2]),
            "suggested_low": (
                max(0, int(center[0]) - 10),
                max(0, int(center[1]) - 50),
                max(0, int(center[2]) - 50),
            ),
            "suggested_high": (
                min(179, int(center[0]) + 10),
                min(255, int(center[1]) + 50),
                min(255, int(center[2]) + 50),
            ),
        })

    cluster_info.sort(key=lambda x: x["center_S"], reverse=True)
    return cluster_info


def main():
    parser = argparse.ArgumentParser(
        description="FRC Scoring Analyzer 偵測診斷工具")
    parser.add_argument("video", help="影片路徑")
    parser.add_argument("--seconds", type=int, default=20,
                        help="分析前 N 秒（預設 20）")
    parser.add_argument("--save-images", action="store_true",
                        help="儲存診斷影像到 diagnose_output/")
    parser.add_argument("--sample-interval", type=int, default=30,
                        help="每 N 幀取樣一次（預設 30）")
    args = parser.parse_args()

    if not os.path.isfile(args.video):
        print(f"影片不存在: {args.video}")
        sys.exit(1)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"無法開啟影片: {args.video}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    max_frame = min(int(fps * args.seconds), total_frames)

    print("=" * 60)
    print("FRC Scoring Analyzer — 偵測診斷報告")
    print("=" * 60)
    print(f"影片: {args.video}")
    print(f"解析度: {width}x{height}")
    print(f"FPS: {fps:.1f}")
    print(f"總幀數: {total_frames}")
    print(f"分析範圍: 前 {args.seconds} 秒 ({max_frame} 幀)")
    print()

    cfg = RuntimeConfig()
    save_dir = "diagnose_output" if args.save_images else None

    # 收集取樣幀
    sampled_frames = []
    ball_results = []
    total_ball_detections = 0

    print("-" * 60)
    print("【球偵測診斷】")
    print(f"HSV 範圍: H:{cfg.hsv_low[0]}-{cfg.hsv_high[0]} "
          f"S:{cfg.hsv_low[1]}-{cfg.hsv_high[1]} "
          f"V:{cfg.hsv_low[2]}-{cfg.hsv_high[2]}")
    print(f"面積範圍: {cfg.min_blob_area}-{cfg.max_blob_area}")
    print("-" * 60)

    frame_idx = 0
    while frame_idx < max_frame:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx < 10 or frame_idx % args.sample_interval == 0:
            sampled_frames.append(frame.copy())

        if frame_idx % args.sample_interval == 0:
            result = diagnose_ball_detection(frame, frame_idx, cfg, save_dir)
            ball_results.append(result)
            total_ball_detections += result["passed_area_filter"]

            # 只印關鍵資訊
            mask_px = result["mask_pixels_after_inrange"]
            contours = result["total_contours"]
            passed = result["passed_area_filter"]
            areas = result["contour_areas"][:5]

            status = "OK" if passed > 0 else ("MASK空" if mask_px == 0 else "面積過濾")
            print(f"  幀 {frame_idx:5d}: mask={mask_px:6d}px  "
                  f"contours={contours:3d}  "
                  f"passed={passed}  "
                  f"[{status}]"
                  + (f"  areas={areas}" if areas else ""))

            if mask_px == 0 and "hsv_samples" in result:
                print(f"           HSV 取樣: {result['hsv_samples'][:3]}")

        frame_idx += 1

    print(f"\n球偵測總結: {max_frame} 幀中偵測到 {total_ball_detections} 個球")

    if total_ball_detections == 0:
        print("\n⚠ 球偵測完全失效！可能原因：")
        print("  1. HSV 範圍與影片中的遊戲物件顏色不匹配")
        print("  2. 請使用設定面板的「HSV 校正」功能重新校正")
        print("\n正在嘗試自動偵測可能的球 HSV 範圍...")
        clusters = auto_detect_ball_hsv(sampled_frames)
        if clusters:
            print("\n運動區域中的高飽和度顏色 cluster（按飽和度排序）：")
            for c in clusters[:5]:
                print(f"  Cluster {c['cluster']}: "
                      f"H={c['center_H']} S={c['center_S']} V={c['center_V']} "
                      f"(count={c['count']})")
                print(f"    建議 HSV 範圍: "
                      f"low={c['suggested_low']} high={c['suggested_high']}")

    # 機器人偵測診斷
    print()
    print("-" * 60)
    print("【機器人偵測診斷】")
    print("-" * 60)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    robot_tested = False
    for frame_idx in range(0, max_frame, max(1, max_frame // 5)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        result = diagnose_robot_detection(frame, frame_idx, save_dir)
        if "error" in result:
            print(f"  機器人偵測錯誤: {result['error']}")
            break

        robot_tested = True
        res = result.get("resolution", "?")
        print(f"\n  幀 {frame_idx} ({res}):")

        for mode_name in ["normal", "tiled"]:
            if mode_name not in result:
                continue
            mode_data = result[mode_name]
            print(f"    [{mode_name}]:")
            for conf_key, data in sorted(mode_data.items()):
                conf_val = conf_key.replace("conf_", "")
                n_robots = data["total_robots"]
                classes = data["by_class"]
                if n_robots > 0:
                    print(f"      conf={conf_val}: "
                          f"機器人={n_robots} {classes}")
            # 只印有結果的最佳 conf
            best = None
            for conf_key in sorted(mode_data.keys()):
                if mode_data[conf_key]["total_robots"] > 0:
                    best = conf_key
                    break
            if best is None:
                print(f"      ⚠ 所有 confidence 都偵測不到機器人")

    cap.release()

    if args.save_images and save_dir:
        print(f"\n診斷影像已儲存到: {save_dir}/")

    print()
    print("=" * 60)
    print("診斷完成")
    print("=" * 60)


if __name__ == "__main__":
    main()

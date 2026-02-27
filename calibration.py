"""FRC Scoring Analyzer — HSV 自動校正（K-Means 取色）"""

import cv2
import numpy as np


def calibrate_hsv_from_points(
    frame: np.ndarray,
    points: list[tuple[int, int]],
    patch_radius: int = 20,
    margin_h: int = 8,
    margin_s: int = 40,
    margin_v: int = 40,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """
    根據用戶點擊的球位置，自動計算最佳 HSV 範圍。

    演算法：
    1. 取每個點擊位置周圍的像素
    2. K-Means (k=2) 分離球色 vs 背景
    3. 選飽和度最高的 cluster（= 球）
    4. 用 5th/95th 百分位 + margin 算範圍

    Args:
        frame: BGR 影像
        points: [(x, y), ...] 用戶點擊的球位置
        patch_radius: 取樣半徑（像素）
        margin_h/s/v: HSV 各通道擴展 margin

    Returns:
        (hsv_low, hsv_high) — 可直接用於 cv2.inRange
    """
    if not points:
        raise ValueError("至少需要一個取樣點")

    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    all_pixels = []
    for px, py in points:
        x1 = max(0, px - patch_radius)
        y1 = max(0, py - patch_radius)
        x2 = min(w, px + patch_radius)
        y2 = min(h, py + patch_radius)
        patch = hsv[y1:y2, x1:x2].reshape(-1, 3)
        all_pixels.append(patch)

    all_pixels = np.vstack(all_pixels).astype(np.float32)

    # K-Means 分離球色 vs 背景
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(
        all_pixels, 2, None, criteria, 5, cv2.KMEANS_PP_CENTERS
    )

    # 選飽和度最高的 cluster（球是高飽和度物體）
    best_cluster = int(np.argmax(centers[:, 1]))
    ball_pixels = all_pixels[labels.flatten() == best_cluster]

    if len(ball_pixels) == 0:
        raise ValueError("無法從取樣點中分離出球的顏色")

    # 百分位數邊界（抗 outlier）
    low = np.percentile(ball_pixels, 5, axis=0).astype(int)
    high = np.percentile(ball_pixels, 95, axis=0).astype(int)

    hsv_low = (
        max(0, int(low[0]) - margin_h),
        max(0, int(low[1]) - margin_s),
        max(0, int(low[2]) - margin_v),
    )
    hsv_high = (
        min(179, int(high[0]) + margin_h),
        min(255, int(high[1]) + margin_s),
        min(255, int(high[2]) + margin_v),
    )

    return hsv_low, hsv_high


def sample_hsv_at_point(
    frame: np.ndarray,
    px: int, py: int,
    patch_radius: int = 8,
    margin_h: int = 10,
    margin_s: int = 40,
    margin_v: int = 40,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """
    單點取色：取指定位置周圍像素的 HSV 中位數 ± margin。

    比 K-Means 更快，適合單點即時預覽。

    Args:
        frame: BGR 影像
        px, py: 取樣中心點
        patch_radius: 取樣半徑

    Returns:
        (hsv_low, hsv_high)
    """
    h, w = frame.shape[:2]
    x1 = max(0, px - patch_radius)
    y1 = max(0, py - patch_radius)
    x2 = min(w, px + patch_radius)
    y2 = min(h, py + patch_radius)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    patch = hsv[y1:y2, x1:x2].reshape(-1, 3)

    median = np.median(patch, axis=0).astype(int)

    hsv_low = (
        max(0, int(median[0]) - margin_h),
        max(0, int(median[1]) - margin_s),
        max(0, int(median[2]) - margin_v),
    )
    hsv_high = (
        min(179, int(median[0]) + margin_h),
        min(255, int(median[1]) + margin_s),
        min(255, int(median[2]) + margin_v),
    )
    return hsv_low, hsv_high


def generate_preview(
    frame: np.ndarray,
    hsv_low: tuple[int, int, int],
    hsv_high: tuple[int, int, int],
    min_area: int = 150,
    max_area: int = 10000,
) -> tuple[np.ndarray, int]:
    """
    生成 HSV 遮罩預覽 + 計算偵測球數。

    Args:
        frame: BGR 影像
        hsv_low, hsv_high: HSV 範圍
        min_area, max_area: 面積過濾

    Returns:
        (mask_colored, ball_count) — mask_colored 是 BGR 彩色遮罩
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(hsv_low), np.array(hsv_high))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

    balls = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area <= area <= max_area:
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                r = max(4, int(np.sqrt(area / np.pi)))
                balls.append((cx, cy, r))

    # 彩色遮罩：偵測區域高亮黃色，其餘暗色
    mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    # 將白色區域染成黃色
    mask_colored[mask > 0] = [0, 200, 255]  # BGR 黃色

    return mask_colored, len(balls), balls


def build_bumper_template(
    frame: np.ndarray,
    points: list[tuple[int, int]],
    patch_radius: int = 20,
) -> tuple[np.ndarray, str]:
    """
    從用戶取色點建立 bumper 顏色直方圖模板。

    Args:
        frame: BGR 影像
        points: [(x, y), ...] 用戶點擊的 bumper 位置
        patch_radius: 取樣半徑（像素）

    Returns:
        (histogram, alliance) — histogram 為正規化 HSV H+S 16x16 直方圖，
        alliance 為自動判斷的聯盟 ("red" / "blue" / "")
    """
    if not points:
        raise ValueError("至少需要一個取樣點")

    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    all_patches = []
    hue_values = []
    for px, py in points:
        x1 = max(0, px - patch_radius)
        y1 = max(0, py - patch_radius)
        x2 = min(w, px + patch_radius)
        y2 = min(h, py + patch_radius)
        patch = hsv[y1:y2, x1:x2]
        all_patches.append(patch)
        hue_values.extend(patch[:, :, 0].flatten().tolist())

    # 合併所有 patch 計算 H+S 直方圖
    combined = np.vstack([p.reshape(-1, 3) for p in all_patches])
    combined_hsv = combined.reshape(-1, 1, 3).astype(np.uint8)
    hist = cv2.calcHist([combined_hsv], [0, 1], None,
                        [16, 16], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    histogram = hist.flatten()

    # 自動判斷聯盟：從 hue 中位數推斷
    median_hue = np.median(hue_values)
    if median_hue <= 10 or median_hue >= 170:
        alliance = "red"
    elif 100 <= median_hue <= 130:
        alliance = "blue"
    else:
        alliance = ""

    return histogram, alliance

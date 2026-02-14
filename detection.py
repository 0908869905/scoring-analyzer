"""
FRC Scoring Analyzer — 黃色球偵測（複用高度分析邏輯）
"""

import cv2
import numpy as np

from config import YELLOW_LOW, YELLOW_HIGH, MIN_BLOB_AREA, MAX_BLOB_AREA


def detect_yellow_balls(frame, hsv_low=None, hsv_high=None,
                        min_area=None, max_area=None):
    """
    HSV 過濾 + 形態學清理 + 輪廓偵測。

    Args:
        frame: BGR 影像
        hsv_low: HSV 下限 tuple，None 則使用預設
        hsv_high: HSV 上限 tuple，None 則使用預設
        min_area: 最小面積，None 則使用預設
        max_area: 最大面積，None 則使用預設

    Returns:
        [(cx, cy, area, radius), ...]
    """
    low = np.array(hsv_low or YELLOW_LOW)
    high = np.array(hsv_high or YELLOW_HIGH)
    min_a = min_area if min_area is not None else MIN_BLOB_AREA
    max_a = max_area if max_area is not None else MAX_BLOB_AREA

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, low, high)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    results = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_a <= area <= max_a:
            M = cv2.moments(cnt)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                radius = int(np.sqrt(area / np.pi))
                results.append((cx, cy, area, radius))
    return results

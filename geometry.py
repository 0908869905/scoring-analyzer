"""
FRC Scoring Analyzer — 幾何工具（擴展自高度分析）
"""

import math


def cross(o, a, b):
    """2D cross product of vectors OA and OB."""
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def on_segment(p, q, r):
    """檢查點 r 是否在線段 p-q 上（假設三點共線）。"""
    return (min(p[0], q[0]) <= r[0] <= max(p[0], q[0]) and
            min(p[1], q[1]) <= r[1] <= max(p[1], q[1]))


def segments_intersect(p1, p2, p3, p4):
    """判斷線段 p1-p2 與 p3-p4 是否相交。"""
    d1 = cross(p3, p4, p1)
    d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3)
    d4 = cross(p1, p2, p4)

    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True

    if d1 == 0 and on_segment(p3, p4, p1):
        return True
    if d2 == 0 and on_segment(p3, p4, p2):
        return True
    if d3 == 0 and on_segment(p1, p2, p3):
        return True
    if d4 == 0 and on_segment(p1, p2, p4):
        return True

    return False


def point_in_rect(px, py, rx, ry, rw, rh):
    """判斷點 (px, py) 是否在矩形 (rx, ry, rw, rh) 內。"""
    return rx <= px <= rx + rw and ry <= py <= ry + rh


def point_in_polygon(px, py, polygon):
    """
    Ray casting 演算法判斷點 (px, py) 是否在多邊形內。

    Args:
        px, py: 點座標
        polygon: [(x1, y1), (x2, y2), ...] 多邊形頂點列表
    """
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > py) != (yj > py)) and \
           (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def distance(p1, p2):
    """兩點間的歐幾里得距離。"""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def point_to_segment_distance(px, py, x1, y1, x2, y2):
    """計算點 (px, py) 到線段 (x1,y1)-(x2,y2) 的最短距離。"""
    dx, dy = x2 - x1, y2 - y1
    len_sq = dx * dx + dy * dy
    if len_sq == 0:
        return math.hypot(px - x1, py - y1)
    t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / len_sq))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return math.hypot(px - proj_x, py - proj_y)


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


def rect_center(rx, ry, rw, rh):
    """矩形中心座標。"""
    return (rx + rw / 2, ry + rh / 2)


def clamp_rect(rx, ry, rw, rh, img_w, img_h):
    """將矩形限制在影像範圍內。"""
    rx = max(0, min(rx, img_w - rw))
    ry = max(0, min(ry, img_h - rh))
    rw = min(rw, img_w - rx)
    rh = min(rh, img_h - ry)
    return (int(rx), int(ry), int(rw), int(rh))

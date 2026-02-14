"""
FRC Scoring Analyzer — 工具函式
"""

import os
import sys

from PIL import ImageFont


def load_font(size, bold=False):
    """載入系統字型，優先使用 Windows 中文字型。"""
    font_candidates = []
    if sys.platform == "win32":
        windir = os.environ.get("WINDIR", r"C:\Windows")
        fonts_dir = os.path.join(windir, "Fonts")
        if bold:
            font_candidates = [
                os.path.join(fonts_dir, "msjhbd.ttc"),
                os.path.join(fonts_dir, "msyhbd.ttc"),
                os.path.join(fonts_dir, "arialbd.ttf"),
                os.path.join(fonts_dir, "segoeui.ttf"),
            ]
        else:
            font_candidates = [
                os.path.join(fonts_dir, "msjh.ttc"),
                os.path.join(fonts_dir, "msyh.ttc"),
                os.path.join(fonts_dir, "arial.ttf"),
                os.path.join(fonts_dir, "segoeui.ttf"),
            ]
    for path in font_candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def format_time(seconds):
    """格式化秒數為 mm:ss 格式。"""
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m}:{s:02d}"


def format_frame_time(frame_idx, fps):
    """格式化幀索引為時間字串。"""
    sec = frame_idx / fps if fps > 0 else 0
    return f"{sec:.3f}s"

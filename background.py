"""
FRC Scoring Analyzer — 背景模型（Temporal Median）

固定攝影機 FRC 比賽影片：均勻取樣 → 像素中位數 → 靜態背景圖
每幀 absdiff → 閾值化 → 前景遮罩，自動排除觀眾席/記分板等非場地區域。
"""

import cv2
import numpy as np


class BackgroundModel:
    """Temporal Median 背景模型。"""

    def __init__(self, fg_threshold: int = 30, dilate_kernel: int = 5):
        """
        Args:
            fg_threshold: 前景閾值 (0-255)，absdiff 大於此值視為前景
            dilate_kernel: 膨脹核大小，填補前景區域中的小空洞
        """
        self._fg_threshold = fg_threshold
        self._dilate_kernel = dilate_kernel
        self._background: np.ndarray | None = None

    @property
    def background_image(self) -> np.ndarray | None:
        """取得背景圖（BGR），None 表示尚未建立。"""
        return self._background

    def build(self, cap: cv2.VideoCapture, roi: tuple | None = None,
              sample_count: int = 50,
              progress_callback=None) -> np.ndarray:
        """
        從影片均勻取樣 → 計算每像素中位數 → 背景圖。

        Args:
            cap: 已開啟的 VideoCapture
            roi: (x, y, w, h) 裁切區域，None = 全畫面
            sample_count: 取樣幀數
            progress_callback: fn(current, total) 進度回調

        Returns:
            背景圖 (BGR numpy array)
        """
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 1:
            raise ValueError("影片無幀可取樣")

        sample_count = min(sample_count, total_frames)
        # 均勻取樣的幀索引
        indices = np.linspace(0, total_frames - 1, sample_count, dtype=int)

        samples = []
        for i, fidx in enumerate(indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fidx))
            ret, frame = cap.read()
            if not ret:
                continue
            if roi:
                rx, ry, rw, rh = roi
                frame = frame[ry:ry+rh, rx:rx+rw]
            samples.append(frame)
            if progress_callback:
                progress_callback(i + 1, sample_count)

        if not samples:
            raise ValueError("無法從影片讀取任何幀")

        # 堆疊 → 中位數（沿 axis=0）
        stack = np.stack(samples, axis=0)
        self._background = np.median(stack, axis=0).astype(np.uint8)

        print(f"[INFO] 背景模型已建立（取樣 {len(samples)} 幀，"
              f"尺寸 {self._background.shape[1]}x{self._background.shape[0]}）")
        return self._background

    def get_foreground_mask(self, frame: np.ndarray) -> np.ndarray:
        """
        計算前景遮罩。

        Args:
            frame: 當前幀 (BGR)

        Returns:
            二值遮罩 (uint8)，255=前景，0=背景
        """
        if self._background is None:
            raise RuntimeError("請先呼叫 build() 建立背景模型")

        # absdiff → 取每通道最大差異
        diff = cv2.absdiff(frame, self._background)
        max_diff = np.max(diff, axis=2)

        # 閾值化
        _, mask = cv2.threshold(max_diff, self._fg_threshold, 255,
                                cv2.THRESH_BINARY)

        # 膨脹填補空洞
        if self._dilate_kernel > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (self._dilate_kernel, self._dilate_kernel))
            mask = cv2.dilate(mask, kernel, iterations=2)

        return mask

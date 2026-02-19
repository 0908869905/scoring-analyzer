"""
FRC Scoring Analyzer — 球偵測（HSV + AI 雙模式）
"""

import math
import os

import cv2
import numpy as np

from config import (
    YELLOW_LOW, YELLOW_HIGH, MIN_BLOB_AREA, MAX_BLOB_AREA,
    AI_CONFIDENCE_THRESHOLD, AI_MIN_AREA, AI_MAX_AREA,
)


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


# ═══════════════════════════════════════════════════════
# AI 偵測（YOLOv11 ONNX — 本地離線推理）
# ═══════════════════════════════════════════════════════

_FUEL_ONNX_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "models", "fuel_yolov11.onnx",
)


class FuelDetectorONNX:
    """YOLOv11 ONNX 本地推理 — 完全離線，不需網路。"""

    def __init__(self, model_path: str, nms_iou: float = 0.45):
        import onnxruntime as ort

        available = ort.get_available_providers()
        providers = [p for p in ["CUDAExecutionProvider", "CPUExecutionProvider"]
                     if p in available]
        self._sess = ort.InferenceSession(model_path, providers=providers)
        self._input_name = self._sess.get_inputs()[0].name
        # 輸入形狀: [1, 3, H, W]
        self._input_shape = self._sess.get_inputs()[0].shape
        self._img_h = self._input_shape[2]  # 640
        self._img_w = self._input_shape[3]  # 640
        self._nms_iou = nms_iou

    def _preprocess(self, frame: np.ndarray):
        """BGR 影像 → 正規化 + 等比縮放 + 填充（letterbox）。"""
        h0, w0 = frame.shape[:2]
        # 等比縮放
        scale = min(self._img_w / w0, self._img_h / h0)
        new_w = int(w0 * scale)
        new_h = int(h0 * scale)
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # 灰色填充到 640x640（letterbox）
        canvas = np.full((self._img_h, self._img_w, 3), 114, dtype=np.uint8)
        pad_x = (self._img_w - new_w) // 2
        pad_y = (self._img_h - new_h) // 2
        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

        # BGR → RGB → [0,1] → CHW → batch
        blob = canvas[:, :, ::-1].astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[np.newaxis, ...]
        return blob, scale, pad_x, pad_y

    def detect(self, frame: np.ndarray,
               confidence: float = 0.25,
               min_area: int = 50,
               max_area: int = 50000):
        """
        本地推理，回傳與 HSV 完全一致的格式。

        Returns:
            [(cx, cy, area, radius), ...]
        """
        h0, w0 = frame.shape[:2]
        blob, scale, pad_x, pad_y = self._preprocess(frame)

        # 推理
        outputs = self._sess.run(None, {self._input_name: blob})
        # output shape: (1, 5, 8400) → 轉置為 (8400, 5)
        preds = outputs[0][0].T  # (8400, 5): [x, y, w, h, conf]

        # 信心過濾
        scores = preds[:, 4]
        mask = scores >= confidence
        preds = preds[mask]

        if len(preds) == 0:
            return []

        # 轉換座標：模型座標 → 原始影像座標
        boxes_xywh = preds[:, :4].copy()
        # 去除 letterbox 偏移
        boxes_xywh[:, 0] -= pad_x
        boxes_xywh[:, 1] -= pad_y
        # 還原縮放
        boxes_xywh[:, :4] /= scale

        # NMS（用 OpenCV 的 NMSBoxes）
        # 轉為 (x1, y1, w, h) 格式給 cv2.dnn.NMSBoxes
        boxes_for_nms = []
        for bx in boxes_xywh:
            x1 = bx[0] - bx[2] / 2
            y1 = bx[1] - bx[3] / 2
            boxes_for_nms.append([float(x1), float(y1), float(bx[2]), float(bx[3])])

        confidences = preds[:, 4].tolist()
        indices = cv2.dnn.NMSBoxes(boxes_for_nms, confidences,
                                   confidence, self._nms_iou)

        detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                cx = int(np.clip(boxes_xywh[i, 0], 0, w0 - 1))
                cy = int(np.clip(boxes_xywh[i, 1], 0, h0 - 1))
                w = boxes_xywh[i, 2]
                h = boxes_xywh[i, 3]
                area = float(w * h)
                if min_area <= area <= max_area:
                    radius = int(math.sqrt(area / math.pi))
                    detections.append((cx, cy, area, radius))

        return detections


def load_ai_model(model_path: str | None = None) -> FuelDetectorONNX:
    """
    載入 YOLOv11 ONNX 模型（完全離線）。

    Args:
        model_path: ONNX 模型路徑，None 則使用預設

    Returns:
        FuelDetectorONNX 物件

    Raises:
        FileNotFoundError: 模型檔不存在
    """
    path = model_path or _FUEL_ONNX_PATH
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"AI 模型檔不存在: {path}\n"
            "請將 fuel_yolov11.onnx 放到 models/ 目錄。"
        )
    return FuelDetectorONNX(path)


def detect_fuel_ai(frame, model: FuelDetectorONNX,
                   confidence: float | None = None,
                   min_area: int | None = None,
                   max_area: int | None = None):
    """
    AI Fuel 偵測，回傳格式與 HSV 完全一致。

    Args:
        frame: BGR 影像
        model: FuelDetectorONNX 物件
        confidence: 信心閾值，None 則使用預設
        min_area: 最小面積過濾，None 則使用預設
        max_area: 最大面積過濾，None 則使用預設

    Returns:
        [(cx, cy, area, radius), ...] — 與 detect_yellow_balls() 同格式
    """
    conf = confidence if confidence is not None else AI_CONFIDENCE_THRESHOLD
    min_a = min_area if min_area is not None else AI_MIN_AREA
    max_a = max_area if max_area is not None else AI_MAX_AREA
    return model.detect(frame, confidence=conf, min_area=min_a, max_area=max_a)

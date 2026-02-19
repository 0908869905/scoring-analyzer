"""
FRC Scoring Analyzer — 機器人偵測（YOLO ONNX 本地離線推理）

支援兩種 ONNX 輸出格式:
- 傳統 YOLO: [1, 4+num_classes, num_proposals] — 需手動 NMS
- NMS-Free (YOLO26/end2end): [1, num_detections, 6] — 已去重
"""

import os

import cv2
import numpy as np

from config import (
    ROBOT_DETECTION_MODEL_PATH,
    ROBOT_DETECTION_CONFIDENCE,
    ROBOT_DETECTION_NMS_IOU,
)

_ROBOT_ONNX_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "models", "frc_robot.onnx",
)


class RobotDetectorONNX:
    """YOLO ONNX 機器人偵測器 — 支援多類別（Red Robot / Blue Robot / Robot）。"""

    def __init__(self, model_path: str, nms_iou: float = 0.45):
        import onnxruntime as ort

        available = ort.get_available_providers()
        providers = [p for p in ["CUDAExecutionProvider", "CPUExecutionProvider"]
                     if p in available]
        self._sess = ort.InferenceSession(model_path, providers=providers)
        self._input_name = self._sess.get_inputs()[0].name
        self._input_shape = self._sess.get_inputs()[0].shape
        self._img_h = self._input_shape[2]  # typically 640
        self._img_w = self._input_shape[3]  # typically 640
        self._nms_iou = nms_iou

        # 讀取類別名稱（若模型 metadata 有提供）
        self.class_names = self._read_class_names()

    def _read_class_names(self) -> list[str]:
        """嘗試從 ONNX 模型 metadata 讀取類別名稱。"""
        meta = self._sess.get_modelmeta()
        custom = meta.custom_metadata_map
        if "names" in custom:
            # ultralytics 格式: "{0: 'class0', 1: 'class1', ...}"
            import ast
            try:
                names_dict = ast.literal_eval(custom["names"])
                return [names_dict[i] for i in sorted(names_dict.keys())]
            except Exception:
                pass
        return []

    def _preprocess(self, frame: np.ndarray):
        """BGR 影像 → 正規化 + 等比縮放 + 填充（letterbox）。"""
        h0, w0 = frame.shape[:2]
        scale = min(self._img_w / w0, self._img_h / h0)
        new_w = int(w0 * scale)
        new_h = int(h0 * scale)
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        canvas = np.full((self._img_h, self._img_w, 3), 114, dtype=np.uint8)
        pad_x = (self._img_w - new_w) // 2
        pad_y = (self._img_h - new_h) // 2
        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

        blob = canvas[:, :, ::-1].astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[np.newaxis, ...]
        return blob, scale, pad_x, pad_y

    def detect(self, frame: np.ndarray,
               confidence: float | None = None,
               robot_only: bool = True) -> list[tuple]:
        """
        偵測機器人，回傳 bounding boxes。

        Args:
            frame: BGR 影像
            confidence: 信心閾值，None 使用預設
            robot_only: 只回傳機器人類別（過濾 note/speaker 等）

        Returns:
            [(x1, y1, x2, y2, conf, class_id), ...]
            座標為原始影像座標
        """
        conf_thresh = confidence if confidence is not None else ROBOT_DETECTION_CONFIDENCE
        h0, w0 = frame.shape[:2]
        blob, scale, pad_x, pad_y = self._preprocess(frame)

        outputs = self._sess.run(None, {self._input_name: blob})
        raw = outputs[0]  # [1, A, B]

        # 自動偵測輸出格式
        _, dim1, dim2 = raw.shape

        if dim2 == 6 and dim1 > 6:
            # NMS-Free: [1, N, 6] = [x1, y1, x2, y2, conf, class_id]
            results = self._postprocess_nms_free(
                raw, scale, pad_x, pad_y, w0, h0, conf_thresh)
        elif dim1 <= 20 and dim2 > 100:
            # 傳統 YOLO: [1, 4+C, N] 其中 C 是類別數
            results = self._postprocess_traditional(
                raw, scale, pad_x, pad_y, w0, h0, conf_thresh)
        elif dim2 <= 20 and dim1 > 100:
            # 轉置的傳統格式: [1, N, 4+C]
            raw = raw.transpose(0, 2, 1)
            results = self._postprocess_traditional(
                raw, scale, pad_x, pad_y, w0, h0, conf_thresh)
        else:
            print(f"[WARN] 無法辨識輸出格式 {raw.shape}，嘗試傳統格式")
            results = self._postprocess_traditional(
                raw, scale, pad_x, pad_y, w0, h0, conf_thresh)

        # 過濾非機器人類別
        if robot_only and self.class_names:
            results = [r for r in results if self.is_robot_class(r[5])]

        return results

    def _postprocess_nms_free(self, raw, scale, pad_x, pad_y,
                               w0, h0, confidence):
        """NMS-Free (YOLO26/end2end) 後處理。"""
        preds = raw[0]  # (N, 6): [x1, y1, x2, y2, conf, class_id]

        mask = preds[:, 4] >= confidence
        preds = preds[mask]

        if len(preds) == 0:
            return []

        results = []
        for det in preds:
            x1, y1, x2, y2, conf, cls_id = det
            x1 = (x1 - pad_x) / scale
            y1 = (y1 - pad_y) / scale
            x2 = (x2 - pad_x) / scale
            y2 = (y2 - pad_y) / scale

            x1 = float(max(0, min(x1, w0)))
            y1 = float(max(0, min(y1, h0)))
            x2 = float(max(0, min(x2, w0)))
            y2 = float(max(0, min(y2, h0)))

            if x2 - x1 > 5 and y2 - y1 > 5:
                results.append((x1, y1, x2, y2, float(conf), int(cls_id)))

        return results

    def _postprocess_traditional(self, raw, scale, pad_x, pad_y,
                                  w0, h0, confidence):
        """傳統 YOLO 後處理（含 NMS）。"""
        preds_raw = raw[0]  # (4+C, N)
        num_features = preds_raw.shape[0]
        preds = preds_raw.T  # (N, 4+C)

        if num_features == 5:
            # 單類別: [cx, cy, w, h, conf]
            scores = preds[:, 4]
            class_ids = np.zeros(len(preds), dtype=int)
        else:
            # 多類別: [cx, cy, w, h, cls0_conf, cls1_conf, ...]
            class_confs = preds[:, 4:]
            class_ids = np.argmax(class_confs, axis=1)
            scores = np.max(class_confs, axis=1)

        mask = scores >= confidence
        preds = preds[mask]
        scores = scores[mask]
        class_ids = class_ids[mask]

        if len(preds) == 0:
            return []

        # xywh → 原始座標
        boxes_xywh = preds[:, :4].copy()
        boxes_xywh[:, 0] -= pad_x
        boxes_xywh[:, 1] -= pad_y
        boxes_xywh[:, :4] /= scale

        # xywh → xyxy
        x1 = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
        y1 = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
        x2 = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
        y2 = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2

        # NMS (cv2.dnn.NMSBoxes 使用 x,y,w,h 格式)
        boxes_for_nms = []
        for i in range(len(preds)):
            boxes_for_nms.append([
                float(x1[i]), float(y1[i]),
                float(boxes_xywh[i, 2]), float(boxes_xywh[i, 3])
            ])

        indices = cv2.dnn.NMSBoxes(
            boxes_for_nms, scores.tolist(), confidence, self._nms_iou)

        results = []
        if len(indices) > 0:
            for i in indices.flatten():
                bx1 = float(max(0, min(x1[i], w0)))
                by1 = float(max(0, min(y1[i], h0)))
                bx2 = float(max(0, min(x2[i], w0)))
                by2 = float(max(0, min(y2[i], h0)))
                if bx2 - bx1 > 5 and by2 - by1 > 5:
                    results.append((bx1, by1, bx2, by2,
                                    float(scores[i]), int(class_ids[i])))

        return results

    def get_class_name(self, class_id: int) -> str:
        """取得類別名稱。"""
        if 0 <= class_id < len(self.class_names):
            return self.class_names[class_id]
        return f"class_{class_id}"

    def is_robot_class(self, class_id: int) -> bool:
        """判斷是否為機器人類別（過濾非機器人偵測）。

        支援多種資料集命名：
        - "red_robot" / "blue_robot" / "black_robot" (WorBots 4145)
        - "Red" / "Blue" (Main Robot Detection)
        - "robot" (通用)
        """
        name = self.get_class_name(class_id).lower()
        if "robot" in name or "bot" in name:
            return True
        if name in ("red", "blue", "black"):
            return True
        return False

    def infer_alliance(self, class_id: int) -> str:
        """從類別名稱推斷聯盟（red/blue/""）。"""
        if not self.is_robot_class(class_id):
            return ""
        name = self.get_class_name(class_id).lower()
        if "red" in name:
            return "red"
        if "blue" in name:
            return "blue"
        return ""


def load_robot_model(model_path: str | None = None) -> RobotDetectorONNX:
    """
    載入機器人偵測 ONNX 模型。

    Args:
        model_path: ONNX 模型路徑，None 則使用預設

    Returns:
        RobotDetectorONNX 物件

    Raises:
        FileNotFoundError: 模型檔不存在
    """
    path = model_path or _ROBOT_ONNX_PATH
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"機器人偵測模型不存在: {path}\n"
            "請使用 train_robot_model.py 訓練模型，\n"
            "或手動放置 ONNX 模型到 models/ 目錄。"
        )
    return RobotDetectorONNX(path, nms_iou=ROBOT_DETECTION_NMS_IOU)

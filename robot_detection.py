"""
FRC Scoring Analyzer — 機器人偵測

支援三種偵測模式:
1. HSV Bumper 偵測（預設）: 用色彩過濾偵測紅藍 bumper，不需訓練資料
2. YOLO ONNX 偵測（備選）: 需要 frc_robot.onnx 模型檔
3. Gemini API 偵測（零樣本）: 用 Gemini Vision 語義偵測，不需訓練資料

ONNX 支援兩種輸出格式:
- 傳統 YOLO: [1, 4+num_classes, num_proposals] — 需手動 NMS
- NMS-Free (YOLO26/end2end): [1, num_detections, 6] — 已去重
"""

import os
import threading

import cv2
import numpy as np

from config import (
    ROBOT_DETECTION_MODEL_PATH,
    ROBOT_DETECTION_CONFIDENCE,
    ROBOT_DETECTION_NMS_IOU,
    BUMPER_RED_LOW_1, BUMPER_RED_HIGH_1,
    BUMPER_RED_LOW_2, BUMPER_RED_HIGH_2,
    BUMPER_BLUE_LOW, BUMPER_BLUE_HIGH,
    BUMPER_MIN_AREA, BUMPER_MAX_AREA,
    BUMPER_MIN_ASPECT, BUMPER_MAX_ASPECT,
    BUMPER_NMS_IOU,
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
        providers = [p for p in ["CUDAExecutionProvider", "DmlExecutionProvider",
                                  "CPUExecutionProvider"]
                     if p in available]
        self._sess = ort.InferenceSession(model_path, providers=providers)
        active_provider = self._sess.get_providers()[0] if self._sess.get_providers() else "unknown"
        print(f"[INFO] 機器人偵測 ONNX Provider: {active_provider}")

        self._input_name = self._sess.get_inputs()[0].name
        self._input_shape = self._sess.get_inputs()[0].shape
        self._img_h = self._input_shape[2]  # typically 640
        self._img_w = self._input_shape[3]  # typically 640
        self._nms_iou = nms_iou
        self._local = threading.local()  # per-thread buffer 預分配

        # 讀取類別名稱（若模型 metadata 有提供）
        self.class_names = self._read_class_names()
        if self.class_names:
            print(f"[INFO] 機器人偵測模型: {len(self.class_names)} 類 {self.class_names}")

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
        """BGR 影像 → 正規化 + 等比縮放 + 填充（letterbox）。

        使用 per-thread 預分配 buffer 避免每次推理重新分配記憶體。
        """
        h0, w0 = frame.shape[:2]
        scale = min(self._img_w / w0, self._img_h / h0)
        new_w = int(w0 * scale)
        new_h = int(h0 * scale)
        resized = cv2.resize(frame, (new_w, new_h),
                             interpolation=cv2.INTER_LINEAR)

        # 重用 per-thread buffer（thread-safe，支援並行偵測）
        loc = self._local
        if not hasattr(loc, 'canvas'):
            loc.canvas = np.empty(
                (self._img_h, self._img_w, 3), dtype=np.uint8)
            loc.blob = np.empty(
                (1, 3, self._img_h, self._img_w), dtype=np.float32)
        canvas = loc.canvas
        blob = loc.blob

        canvas[:] = 114
        pad_x = (self._img_w - new_w) // 2
        pad_y = (self._img_h - new_h) // 2
        canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

        # BGR→RGB + uint8→float32 + HWC→CHW — 零中間陣列分配
        for c in range(3):
            np.true_divide(canvas[:, :, 2 - c], 255.0, out=blob[0, c])

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

    def detect_tiled(self, frame: np.ndarray,
                     confidence: float | None = None,
                     robot_only: bool = True,
                     tile_size: int = 1280,
                     overlap: float = 0.15) -> list[tuple]:
        """
        Tile-based 偵測 — 將大影像切割成重疊小塊分別偵測再合併。

        解決 4K 影像縮放到 640x640 導致機器人太小的問題。
        例: 3840x2160 直接縮放 → 機器人僅 25px；
            切成 1280px tiles → 機器人 ~75px，偵測率大幅提升。

        Args:
            frame: BGR 影像
            confidence: 信心閾值
            robot_only: 只回傳機器人類別
            tile_size: tile 邊長（像素）
            overlap: tile 重疊比例 (0-0.5)

        Returns:
            [(x1, y1, x2, y2, conf, class_id), ...]
        """
        h, w = frame.shape[:2]
        conf_thresh = confidence if confidence is not None else ROBOT_DETECTION_CONFIDENCE

        # 小影像直接用普通偵測
        if max(h, w) <= tile_size * 1.2:
            return self.detect(frame, confidence, robot_only)

        step = int(tile_size * (1 - overlap))
        all_dets = []

        # 滑動窗口切割
        for y_start in range(0, h, step):
            for x_start in range(0, w, step):
                # tile 範圍（確保不超出邊界且維持 tile_size）
                x_end = min(x_start + tile_size, w)
                y_end = min(y_start + tile_size, h)
                x_begin = max(0, x_end - tile_size)
                y_begin = max(0, y_end - tile_size)

                tile = frame[y_begin:y_end, x_begin:x_end]

                # 對 tile 偵測（不過濾 robot_only，合併後再過濾）
                dets = self.detect(tile, confidence=conf_thresh, robot_only=False)

                # 將 tile 座標偏移到原始影像座標
                for d in dets:
                    all_dets.append((
                        d[0] + x_begin, d[1] + y_begin,
                        d[2] + x_begin, d[3] + y_begin,
                        d[4], d[5],
                    ))

        if not all_dets:
            return []

        # 全域 NMS 去除重疊區域的重複偵測
        boxes_for_nms = []
        confs = []
        for d in all_dets:
            boxes_for_nms.append([
                float(d[0]), float(d[1]),
                float(d[2] - d[0]), float(d[3] - d[1]),  # x,y,w,h
            ])
            confs.append(float(d[4]))

        indices = cv2.dnn.NMSBoxes(
            boxes_for_nms, confs, conf_thresh, self._nms_iou)

        results = []
        if len(indices) > 0:
            for i in indices.flatten():
                results.append(all_dets[i])

        # 過濾非機器人
        if robot_only and self.class_names:
            results = [r for r in results if self.is_robot_class(r[5])]

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


# ═══════════════════════════════════════════════════════
# HSV Bumper 偵測（不需訓練資料）
# ═══════════════════════════════════════════════════════


class BumperDetectorHSV:
    """HSV 色彩過濾偵測紅藍 bumper — 與 RobotDetectorONNX 同介面。

    FRC bumper 是標準化零件（紅/藍布包覆 pool noodle），顏色鮮明飽和，
    適合用 HSV 過濾偵測。不需要任何訓練資料，跨年份通用。
    """

    # 類別定義（與 YOLO 模型一致，tuple 避免意外修改）
    class_names = ("Red", "Blue")

    def __init__(self,
                 red_low_1=None, red_high_1=None,
                 red_low_2=None, red_high_2=None,
                 blue_low=None, blue_high=None,
                 min_area=None, max_area=None,
                 min_aspect=None, max_aspect=None,
                 nms_iou=None):
        self._red_low_1 = np.array(red_low_1 if red_low_1 is not None else BUMPER_RED_LOW_1, dtype=np.uint8)
        self._red_high_1 = np.array(red_high_1 if red_high_1 is not None else BUMPER_RED_HIGH_1, dtype=np.uint8)
        self._red_low_2 = np.array(red_low_2 if red_low_2 is not None else BUMPER_RED_LOW_2, dtype=np.uint8)
        self._red_high_2 = np.array(red_high_2 if red_high_2 is not None else BUMPER_RED_HIGH_2, dtype=np.uint8)
        self._blue_low = np.array(blue_low if blue_low is not None else BUMPER_BLUE_LOW, dtype=np.uint8)
        self._blue_high = np.array(blue_high if blue_high is not None else BUMPER_BLUE_HIGH, dtype=np.uint8)
        self._min_area = min_area if min_area is not None else BUMPER_MIN_AREA
        self._max_area = max_area if max_area is not None else BUMPER_MAX_AREA
        self._min_aspect = min_aspect if min_aspect is not None else BUMPER_MIN_ASPECT
        self._max_aspect = max_aspect if max_aspect is not None else BUMPER_MAX_ASPECT
        self._nms_iou = nms_iou if nms_iou is not None else BUMPER_NMS_IOU

        self._kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5))
        self._kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))

        self._diag_count = 0

        # Bumper 顏色模板（取色註冊）
        self._templates: list[tuple[str, str, np.ndarray]] = []
        # [(label, alliance, histogram), ...]
        self._template_similarity = 0.3

        print(f"[INFO] HSV Bumper 偵測器啟用 — 不需 ONNX 模型")
        print(f"[INFO]   Red HSV: {tuple(self._red_low_1)}-{tuple(self._red_high_1)} "
              f"| {tuple(self._red_low_2)}-{tuple(self._red_high_2)}")
        print(f"[INFO]   Blue HSV: {tuple(self._blue_low)}-{tuple(self._blue_high)}")
        print(f"[INFO]   Area: {self._min_area}-{self._max_area}  "
              f"Aspect: {self._min_aspect}-{self._max_aspect}")

    def set_templates(self, templates: list[tuple[str, str, np.ndarray]],
                      similarity: float = 0.3):
        """註冊 bumper 顏色模板。

        Args:
            templates: [(label, alliance, histogram), ...] 每台機器人的直方圖
            similarity: 相似度閾值 (0~1)
        """
        self._templates = list(templates)
        self._template_similarity = similarity
        if templates:
            print(f"[INFO] Bumper 模板已註冊: {len(templates)} 台 "
                  f"(閾值={similarity:.2f})")

    def _match_template(self, frame: np.ndarray,
                        x1: float, y1: float,
                        x2: float, y2: float) -> tuple[str, str, float] | None:
        """將候選 bbox 與所有模板比對，回傳最佳匹配。

        Returns:
            (label, alliance, similarity) 或 None（無匹配）
        """
        if not self._templates:
            return None

        h, w = frame.shape[:2]
        ix1, iy1 = max(0, int(x1)), max(0, int(y1))
        ix2, iy2 = min(w, int(x2)), min(h, int(y2))
        if ix2 - ix1 < 5 or iy2 - iy1 < 5:
            return None

        roi = frame[iy1:iy2, ix1:ix2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None,
                            [16, 16], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        det_hist = hist.flatten()

        best_label, best_alliance, best_sim = "", "", -1.0
        for label, alliance, tmpl_hist in self._templates:
            sim = cv2.compareHist(
                det_hist.reshape(-1, 1).astype(np.float32),
                tmpl_hist.reshape(-1, 1).astype(np.float32),
                cv2.HISTCMP_CORREL)
            if sim > best_sim:
                best_label, best_alliance, best_sim = label, alliance, sim

        if best_sim >= self._template_similarity:
            return (best_label, best_alliance, best_sim)
        return None

    def _detect_color(self, hsv: np.ndarray, mask_low, mask_high,
                      mask_low_2=None, mask_high_2=None,
                      class_id: int = 0) -> list[tuple]:
        """偵測單一顏色的 bumper。"""
        mask = cv2.inRange(hsv, mask_low, mask_high)
        if mask_low_2 is not None:
            mask2 = cv2.inRange(hsv, mask_low_2, mask_high_2)
            mask = cv2.bitwise_or(mask, mask2)

        # ── 診斷：HSV 遮罩像素統計 ──
        if self._diag_count < 3:
            color_name = "Red" if class_id == 0 else "Blue"
            mask_pixels = int(cv2.countNonZero(mask))
            total_pixels = mask.shape[0] * mask.shape[1]
            print(f"[DIAG] {color_name} HSV mask: "
                  f"{mask_pixels}/{total_pixels} pixels "
                  f"({mask_pixels/total_pixels*100:.3f}%)")

        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel_close)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel_open)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        # ── 診斷：輪廓過濾統計 ──
        n_total = len(contours)
        n_area_fail = 0
        n_aspect_fail = 0

        results = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self._min_area or area > self._max_area:
                n_area_fail += 1
                # 診斷：印出被過濾的面積
                if self._diag_count < 3 and area > 50:
                    x, y, w, h = cv2.boundingRect(cnt)
                    print(f"[DIAG]   面積過濾: area={area:.0f} "
                          f"(要求 {self._min_area}-{self._max_area}) "
                          f"bbox={w}x{h} @ ({x},{y})")
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            if h == 0:
                continue

            aspect = w / h
            if aspect < self._min_aspect or aspect > self._max_aspect:
                n_aspect_fail += 1
                if self._diag_count < 3:
                    print(f"[DIAG]   長寬比過濾: aspect={aspect:.2f} "
                          f"(要求 {self._min_aspect}-{self._max_aspect}) "
                          f"bbox={w}x{h} area={area:.0f}")
                continue

            # conf 用面積佔比模擬（越大的 bumper 越確定）
            area_ratio = min(area / self._max_area, 1.0)
            conf = 0.5 + 0.5 * area_ratio

            results.append((
                float(x), float(y),
                float(x + w), float(y + h),
                float(conf), class_id,
            ))

        if self._diag_count < 3:
            color_name = "Red" if class_id == 0 else "Blue"
            print(f"[DIAG] {color_name} 輪廓: "
                  f"total={n_total} → area_fail={n_area_fail} "
                  f"aspect_fail={n_aspect_fail} → pass={len(results)}")

        return results

    def detect(self, frame: np.ndarray,
               confidence: float | None = None,
               robot_only: bool = True) -> list[tuple]:
        """偵測紅藍 bumper，回傳格式與 RobotDetectorONNX.detect() 一致。

        Args:
            frame: BGR 影像
            confidence: 信心閾值（用於過濾小偵測）
            robot_only: 忽略（HSV bumper 只偵測機器人）

        Returns:
            [(x1, y1, x2, y2, conf, class_id), ...]
            class_id: 0=Red, 1=Blue
        """
        conf_thresh = confidence if confidence is not None else 0.3

        # ── 診斷：輸入 frame 統計 ──
        if self._diag_count < 3:
            h, w = frame.shape[:2]
            black_pixels = int(np.sum(np.all(frame == 0, axis=2)))
            total_pixels = h * w
            mean_bgr = frame.mean(axis=(0, 1))
            print(f"[DIAG] detect() 輸入: {w}x{h}, "
                  f"黑色像素={black_pixels}/{total_pixels} "
                  f"({black_pixels/total_pixels*100:.1f}%), "
                  f"平均BGR=({mean_bgr[0]:.0f},{mean_bgr[1]:.0f},{mean_bgr[2]:.0f})")

        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # 偵測紅色和藍色 bumper
        red_dets = self._detect_color(
            hsv, self._red_low_1, self._red_high_1,
            self._red_low_2, self._red_high_2, class_id=0)
        blue_dets = self._detect_color(
            hsv, self._blue_low, self._blue_high, class_id=1)

        all_dets = red_dets + blue_dets

        # 過濾低信心
        all_dets = [d for d in all_dets if d[4] >= conf_thresh]

        if not all_dets:
            return []

        # NMS 去重
        boxes = [[float(d[0]), float(d[1]),
                  float(d[2] - d[0]), float(d[3] - d[1])]
                 for d in all_dets]
        confs = [float(d[4]) for d in all_dets]
        indices = cv2.dnn.NMSBoxes(boxes, confs, conf_thresh, self._nms_iou)

        results = []
        if len(indices) > 0:
            for i in indices.flatten():
                results.append(all_dets[i])

        # 模板過濾：只保留匹配已註冊模板的候選
        if self._templates and results:
            filtered = []
            for det in results:
                match = self._match_template(
                    frame, det[0], det[1], det[2], det[3])
                if match:
                    label, alliance, sim = match
                    cls_id = 0 if alliance == "red" else 1
                    filtered.append((
                        det[0], det[1], det[2], det[3],
                        det[4], cls_id,
                    ))
            if self._diag_count <= 3:
                print(f"[DIAG] 模板過濾: {len(results)} → {len(filtered)} 個候選")
            results = filtered

        # 診斷
        self._diag_count += 1
        if self._diag_count <= 3:
            n_red = sum(1 for d in results if d[5] == 0)
            n_blue = sum(1 for d in results if d[5] == 1)
            print(f"[DIAG] Bumper 偵測 幀{self._diag_count}: "
                  f"Red={n_red} Blue={n_blue} total={len(results)}")

        return results

    def detect_tiled(self, frame: np.ndarray,
                     confidence: float | None = None,
                     robot_only: bool = True,
                     tile_size: int = 1280,
                     overlap: float = 0.15) -> list[tuple]:
        """HSV 在全解析度上跑，detect() 已涵蓋 — tiled 回空避免重複偵測。"""
        return []

    def get_class_name(self, class_id: int) -> str:
        if 0 <= class_id < len(self.class_names):
            return self.class_names[class_id]
        return f"class_{class_id}"

    def is_robot_class(self, class_id: int) -> bool:
        return 0 <= class_id < len(self.class_names)

    def infer_alliance(self, class_id: int) -> str:
        if class_id == 0:
            return "red"
        if class_id == 1:
            return "blue"
        return ""


# ═══════════════════════════════════════════════════════
# Gemini API 偵測（零樣本語義偵測）
# ═══════════════════════════════════════════════════════

_GEMINI_PROMPT = """\
You are an FRC (FIRST Robotics Competition) robot detection expert.
Analyze this FRC match frame and locate ALL visible FRC robots.

Rules:
- FRC robots have red or blue bumpers (pool noodle covered in fabric) \
around their chassis
- There are at most 6 robots on the field (3 red + 3 blue), but some \
may be fully occluded
- Only report robots you can actually see — do NOT guess occluded ones
- The bumper color determines the alliance: red bumper = red alliance, \
blue bumper = blue alliance
- Field elements (trench, hub, scoring stations) are NOT robots — \
ignore them even if they have red/blue coloring
- Report bounding boxes as pixel coordinates [x_left, y_top, x_right, y_bottom]

Image dimensions: {width} x {height} pixels
"""


class RobotDetectorGemini:
    """Gemini API 機器人偵測器 — 零樣本語義偵測，與 RobotDetectorONNX 同介面。

    使用 Gemini Vision 分析每幀畫面，找出所有可見的 FRC 機器人。
    不需要任何訓練資料，跨年份/場地/角度通用。
    """

    class_names = ("Red", "Blue")

    def __init__(self, model: str | None = None,
                 strategy: str | None = None,
                 detect_interval: int | None = None,
                 max_image_size: int | None = None):
        from config import (GEMINI_MODEL, GEMINI_DETECT_STRATEGY,
                            GEMINI_DETECT_INTERVAL, GEMINI_MAX_IMAGE_SIZE)

        self._model_name = model or GEMINI_MODEL
        self._strategy = strategy or GEMINI_DETECT_STRATEGY
        self._detect_interval = detect_interval or GEMINI_DETECT_INTERVAL
        self._max_image_size = max_image_size or GEMINI_MAX_IMAGE_SIZE
        self._call_count = 0       # detect() 被呼叫的總次數
        self._api_call_count = 0   # 實際送 API 的次數
        self._client = None
        self._init_client()

        # adaptive 模式狀態
        self._adaptive_dense_calls = 30   # 前 N 次呼叫密集偵測
        self._adaptive_sparse_interval = 15

        print(f"[INFO] Gemini 偵測器: model={self._model_name}, "
              f"strategy={self._strategy}, interval={self._detect_interval}")

    def _init_client(self):
        """初始化 Gemini API client。"""
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        # 若環境變數未設定，嘗試讀取專案目錄的 .env 檔
        if not api_key:
            env_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), ".env")
            if os.path.isfile(env_path):
                with open(env_path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("GEMINI_API_KEY="):
                            api_key = line.split("=", 1)[1].strip()
                        elif not api_key and line.startswith("GOOGLE_API_KEY="):
                            api_key = line.split("=", 1)[1].strip()
        if not api_key:
            raise ValueError(
                "未設定 Gemini API Key。請設定環境變數 GEMINI_API_KEY "
                "或在專案目錄建立 .env 檔（GEMINI_API_KEY=your_key）")
        from google import genai
        self._client = genai.Client(api_key=api_key)

    def _should_detect(self) -> bool:
        """根據策略判斷是否該送 API 偵測。"""
        n = self._call_count
        if self._strategy == "every_2":
            return n % 2 == 0
        elif self._strategy == "every_n":
            return n % self._detect_interval == 0
        elif self._strategy == "adaptive":
            if n < self._adaptive_dense_calls:
                return n % 2 == 0  # 前期密集
            return (n - self._adaptive_dense_calls) \
                % self._adaptive_sparse_interval == 0
        elif self._strategy == "first_only":
            return n < 5  # 只偵測前 5 幀
        return n % self._detect_interval == 0  # fallback

    def _frame_to_pil(self, frame: np.ndarray):
        """將 OpenCV BGR frame 轉為 PIL Image（縮小以省流量）。"""
        from PIL import Image as PILImage
        h, w = frame.shape[:2]
        max_dim = self._max_image_size
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h),
                               interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return PILImage.fromarray(rgb)

    def _call_gemini(self, frame: np.ndarray) -> list[tuple]:
        """呼叫 Gemini API 偵測機器人。"""
        from google.genai import types
        h, w = frame.shape[:2]
        pil_img = self._frame_to_pil(frame)
        img_w, img_h = pil_img.size  # 縮放後的尺寸

        prompt = _GEMINI_PROMPT.format(width=img_w, height=img_h)

        response_schema = {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "bbox": {
                        "type": "ARRAY",
                        "items": {"type": "NUMBER"},
                        "description": "Bounding box [x_left, y_top, x_right, y_bottom] in pixels"
                    },
                    "alliance": {
                        "type": "STRING",
                        "description": "Alliance color: red or blue"
                    },
                    "confidence": {
                        "type": "NUMBER",
                        "description": "Detection confidence 0.0-1.0"
                    }
                },
                "required": ["bbox", "alliance", "confidence"]
            }
        }

        try:
            response = self._client.models.generate_content(
                model=self._model_name,
                contents=[pil_img, prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_json_schema=response_schema,
                    temperature=0.1,
                ),
            )
            self._api_call_count += 1
        except Exception as e:
            print(f"[WARN] Gemini API 呼叫失敗: {e}")
            return []

        # 解析回傳的 JSON
        import json
        try:
            detections = json.loads(response.text)
        except (json.JSONDecodeError, AttributeError) as e:
            print(f"[WARN] Gemini 回傳格式錯誤: {e}")
            return []

        # 轉換為標準格式 (x1, y1, x2, y2, conf, class_id)
        # 如果影像有縮放，座標需要還原到原始尺寸
        scale_x = w / img_w
        scale_y = h / img_h

        results = []
        for det in detections:
            bbox = det.get("bbox", [])
            if len(bbox) != 4:
                continue
            alliance = det.get("alliance", "").lower()
            conf = float(det.get("confidence", 0.5))

            x1 = float(bbox[0]) * scale_x
            y1 = float(bbox[1]) * scale_y
            x2 = float(bbox[2]) * scale_x
            y2 = float(bbox[3]) * scale_y

            # 邊界檢查
            x1 = max(0.0, min(x1, w))
            y1 = max(0.0, min(y1, h))
            x2 = max(0.0, min(x2, w))
            y2 = max(0.0, min(y2, h))

            if x2 - x1 < 5 or y2 - y1 < 5:
                continue

            class_id = 0 if alliance == "red" else 1
            results.append((x1, y1, x2, y2, conf, class_id))

        if self._api_call_count <= 3 or self._api_call_count % 50 == 0:
            n_red = sum(1 for d in results if d[5] == 0)
            n_blue = sum(1 for d in results if d[5] == 1)
            print(f"[DIAG] Gemini 偵測 call#{self._api_call_count}: "
                  f"Red={n_red} Blue={n_blue} total={len(results)}")

        return results

    def detect(self, frame: np.ndarray,
               confidence: float | None = None,
               robot_only: bool = True) -> list[tuple]:
        """偵測機器人，根據策略決定是否呼叫 Gemini API。

        非偵測幀回傳空 list，由 MOT 追蹤器用距離匹配維持追蹤。
        """
        should = self._should_detect()
        self._call_count += 1

        if not should:
            return []

        results = self._call_gemini(frame)

        if confidence is not None:
            results = [r for r in results if r[4] >= confidence]

        return results

    def detect_tiled(self, frame: np.ndarray,
                     confidence: float | None = None,
                     robot_only: bool = True,
                     tile_size: int = 1280,
                     overlap: float = 0.15) -> list[tuple]:
        """Gemini 理解全幀語義，不需要 tiled 偵測。"""
        return []

    def get_class_name(self, class_id: int) -> str:
        if 0 <= class_id < len(self.class_names):
            return self.class_names[class_id]
        return f"class_{class_id}"

    def is_robot_class(self, class_id: int) -> bool:
        return 0 <= class_id < len(self.class_names)

    def infer_alliance(self, class_id: int) -> str:
        if class_id == 0:
            return "red"
        if class_id == 1:
            return "blue"
        return ""

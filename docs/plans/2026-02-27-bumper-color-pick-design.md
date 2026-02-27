# Bumper 取色偵測 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 用 bumper 取色取代框選機器人標記，只追蹤用戶取色過的機器人，消除非機器人誤偵測。

**Architecture:** 用戶在影片上點擊特定機器人的 bumper 多次 → 系統收集像素建立 HSV 顏色直方圖模板。分析時 `BumperDetectorHSV` 找到所有 bumper 候選後，比對已註冊模板的直方圖相似度，只保留匹配的候選，同時標記該候選屬於哪台機器人。

**Tech Stack:** OpenCV HSV histogram（H+S 16x16 bins）、cv2.compareHist CORREL、NumPy

---

## Task 1: `calibration.py` — 新增 `build_bumper_template()`

**Files:**
- Modify: `calibration.py`

**Step 1: 新增 `build_bumper_template` 函式**

在 `calibration.py` 末尾新增：

```python
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
        alliance 為自動判斷的聯盟 ("red" / "blue")
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
```

**Step 2: Commit**

```bash
git add calibration.py
git commit -m "feat: add build_bumper_template for bumper color sampling"
```

---

## Task 2: `config.py` + `runtime_config.py` — 新增 bumper 模板比對閾值

**Files:**
- Modify: `config.py`
- Modify: `runtime_config.py`

**Step 1: `config.py` 新增常數**

在 `# ── HSV Bumper 偵測` 區塊末尾（`BUMPER_NMS_IOU` 之後）新增：

```python
BUMPER_TEMPLATE_SIMILARITY = 0.3  # 模板比對相似度閾值（0~1，越高越嚴格）
```

**Step 2: `runtime_config.py` 新增 dataclass 欄位**

import 加上 `BUMPER_TEMPLATE_SIMILARITY`，dataclass 在 `bytetrack_min_consecutive` 之後新增：

```python
    # ── Bumper 模板 ──
    bumper_template_similarity: float = BUMPER_TEMPLATE_SIMILARITY
```

**Step 3: Commit**

```bash
git add config.py runtime_config.py
git commit -m "feat: add BUMPER_TEMPLATE_SIMILARITY config"
```

---

## Task 3: `robot_detection.py` — `BumperDetectorHSV` 加入模板過濾

**Files:**
- Modify: `robot_detection.py`

**Step 1: 新增 `set_templates()` 方法和模板匹配邏輯**

在 `BumperDetectorHSV.__init__` 末尾新增模板儲存：

```python
        # Bumper 顏色模板（取色註冊）
        self._templates: list[tuple[str, str, np.ndarray]] = []
        # [(label, alliance, histogram), ...]
        self._template_similarity = 0.3
```

新增方法（在 `detect` 之前）：

```python
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
```

**Step 2: 修改 `detect()` 方法，有模板時過濾候選**

在 `detect()` 方法的 NMS 去重之後、return 之前（目前的診斷區塊之前），加入模板過濾：

```python
        # 模板過濾：只保留匹配已註冊模板的候選
        if self._templates and results:
            filtered = []
            for det in results:
                match = self._match_template(
                    frame, det[0], det[1], det[2], det[3])
                if match:
                    # 替換 class_id 為模板的聯盟資訊
                    label, alliance, sim = match
                    cls_id = 0 if alliance == "red" else 1
                    filtered.append((
                        det[0], det[1], det[2], det[3],
                        det[4], cls_id,
                    ))
            if self._diag_count <= 3:
                print(f"[DIAG] 模板過濾: {len(results)} → {len(filtered)} 個候選")
            results = filtered
```

**Step 3: Commit**

```bash
git add robot_detection.py
git commit -m "feat: BumperDetectorHSV template matching filter"
```

---

## Task 4: `app.py` — 取色 UI + 取代框選標記

**Files:**
- Modify: `app.py`

### Step 1: 新增 bumper 取色狀態屬性

在 `__init__` 的 `self._robot_markers` 附近新增：

```python
        # Bumper 取色模板
        self._bumper_templates = []  # [(label, alliance, histogram), ...]
        self._bumper_pick_points = []  # 取色中的點擊座標
```

### Step 2: 工具列按鈕改為「Bumper 取色」

將現有的 `mark_robot_btn`（`text="標記機器人"`）改為 `text="Bumper 取色"`，command 改為 `self._start_bumper_pick`。

### Step 3: 新增取色互動方法

新增 `_start_bumper_pick`（進入取色模式）：
- 設 `self.interaction_mode = "bumper_pick"`
- 清空 `self._bumper_pick_points`
- 顯示狀態「在影片上點擊機器人的 bumper（多次），右鍵完成」

修改 `_on_canvas_press`：新增 `bumper_pick` 分支，將點擊座標加入 `_bumper_pick_points`，更新狀態顯示已取色幾點。

修改 `_on_canvas_right_click`：新增 `bumper_pick` 分支，呼叫 `_finish_bumper_pick()`。

修改 `_on_canvas_release`：在不處理的模式列表中加入 `"bumper_pick"`。

### Step 4: 新增 `_finish_bumper_pick()`

```python
    def _finish_bumper_pick(self):
        """完成 bumper 取色，建立模板。"""
        if len(self._bumper_pick_points) < 2:
            self._set_status("至少需要 2 個取色點", COLORS["error"])
            return

        from calibration import build_bumper_template

        # 取得當前幀
        frame = self._get_current_frame()
        if frame is None:
            self._set_status("無法取得當前幀", COLORS["error"])
            return

        histogram, auto_alliance = build_bumper_template(
            frame, self._bumper_pick_points)

        # 彈出輸入框取得機器人編號
        label = simpledialog.askstring(
            "機器人編號",
            "請輸入機器人編號（例如 6998）:",
            parent=self)
        if not label:
            self._set_status("已取消取色", COLORS["text_secondary"])
            self._bumper_pick_points = []
            self.interaction_mode = None
            self.canvas.config(cursor="")
            return

        label = label.strip()

        # 檢查是否重複
        for existing_label, *_ in self._bumper_templates:
            if existing_label == label:
                self._set_status(f"機器人 {label} 已存在", COLORS["error"])
                self._bumper_pick_points = []
                return

        # 確認聯盟（自動判斷，用戶可覆蓋）
        if auto_alliance:
            alliance = auto_alliance
            alliance_name = "紅方" if alliance == "red" else "藍方"
            self._set_status(f"自動判斷為{alliance_name}", COLORS["info"])
        else:
            alliance = self._ask_alliance()
            if alliance is None:
                self._set_status("已取消取色", COLORS["text_secondary"])
                self._bumper_pick_points = []
                self.interaction_mode = None
                self.canvas.config(cursor="")
                return

        self._bumper_templates.append((label, alliance, histogram))
        # 同步到 _robot_markers（供顯示用）
        self._robot_markers.append(
            (label, alliance, 0, 0, 0, 0, self.current_frame))
        self._update_robot_list()

        self._bumper_pick_points = []
        self.interaction_mode = None
        self.canvas.config(cursor="")
        alliance_name = "紅方" if alliance == "red" else "藍方"
        self._set_status(
            f"已取色{alliance_name} {label}（{len(self._bumper_templates)} 台已註冊）",
            COLORS["success"])
        self._show_frame(self.current_frame)
```

### Step 5: 修改 `_run_analysis` — 傳遞模板給偵測器

在 `BumperDetectorHSV()` 建立之後，傳入模板：

```python
        if self._bumper_templates:
            robot_detector.set_templates(
                self._bumper_templates,
                similarity=cfg.bumper_template_similarity)
```

移除 MOT 自動模式的判斷（有取色模板時不需要 auto mode）：

```python
        # MOT 自動模式：無取色模板且未標記機器人時，自動偵測
        if not self._bumper_templates and not self._robot_markers and robot_mgr.use_mot:
            robot_mgr.enable_auto_mode()
```

### Step 6: 修改 `_render_frame` — 繪製取色標記點

在取色模式中，繪製已點擊的取色點（橙色圓點）：

```python
        if self.interaction_mode == "bumper_pick" and self._bumper_pick_points:
            for pt in self._bumper_pick_points:
                rpt = self._video_to_resized(pt, scale)
                cv2.circle(resized, rpt, 6, (0, 165, 255), -1, cv2.LINE_AA)
```

### Step 7: 修改 `_clear_all_marks` — 清除取色模板

```python
        self._bumper_templates.clear()
        self._bumper_pick_points = []
```

### Step 8: Commit

```bash
git add app.py
git commit -m "feat: bumper color pick replaces robot bbox marking"
```

---

## Task 5: 輔助方法 + 清理

**Files:**
- Modify: `app.py`

### Step 1: 新增 `_get_current_frame` 輔助方法

若尚未存在，新增：

```python
    def _get_current_frame(self) -> np.ndarray | None:
        """取得當前影片幀（BGR）。"""
        if not self.cap:
            return None
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = self.cap.read()
        if not ret:
            return None
        if self._roi:
            rx, ry, rw, rh = self._roi
            frame = frame[ry:ry+rh, rx:rx+rw]
        return frame
```

### Step 2: 修改 `_finish_crop` / `_reset_crop` — 清除取色模板

在兩處清除標記的位置加上：

```python
        self._bumper_templates.clear()
```

### Step 3: 更新 interaction_mode 註解

```python
        self.interaction_mode = None  # None, "bumper_pick", "mark_zone_polygon", "mark_hp_line"
```

### Step 4: Commit

```bash
git add app.py
git commit -m "feat: helper methods and bumper template cleanup"
```

---

## Task 6: 驗證

### Step 1: 語法檢查

```bash
python -c "import ast; ast.parse(open('app.py', encoding='utf-8').read()); print('OK')"
python -c "import calibration; print('OK')"
python -c "from robot_detection import BumperDetectorHSV; d = BumperDetectorHSV(); print('OK')"
python -c "from runtime_config import RuntimeConfig; rc = RuntimeConfig(); print(rc.bumper_template_similarity)"
```

### Step 2: 功能驗證

```bash
python main.py
```

1. 開啟 FRC 影片
2. 點「Bumper 取色」→ 在某台機器人的 bumper 上點擊 3-5 次 → 右鍵完成 → 輸入編號
3. 確認聯盟自動判斷正確
4. 重複取色第二台機器人
5. 點「分析」→ 確認只追蹤取色過的機器人
6. F3 debug view 確認前景遮罩正常

### Step 3: Final commit

```bash
git add -A
git commit -m "feat: bumper color pick detection — replace manual robot marking"
```

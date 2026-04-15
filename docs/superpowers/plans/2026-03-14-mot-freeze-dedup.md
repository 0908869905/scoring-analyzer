# MOT 凍結位置 + 重複清除 Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修復 MOT 追蹤框搶 ID 問題 — 丟失目標的追蹤框不再用速度外推飄移到別的機器人上，改為凍結在最後位置等待復活；加上重複清除防止兩個框追同一台。

**Architecture:** 修改 `_match_direct()` 的 3 處邏輯：(1) grace-period 軌跡用凍結位置配對 (2) LOST 復活時鄰近守衛 (3) 每幀結束後 bbox IoU 重複清除。新增一個常數 `MOT_DEDUP_IOU`。

**Tech Stack:** Python, OpenCV (NMS), NumPy

---

## Chunk 1: 全部修改

### Task 1: 新增常數

**Files:**
- Modify: `config.py:118-124`

- [ ] **Step 1: 新增 MOT_DEDUP_IOU 常數**

在 `config.py` 的 MOT 遮擋處理區段末尾新增：

```python
MOT_DEDUP_IOU = 0.5                 # 重複追蹤清除：同幀兩軌跡 bbox IoU 超過此值視為重複
```

- [ ] **Step 2: 驗證**

Run: `python -c "from config import MOT_DEDUP_IOU; print(MOT_DEDUP_IOU)"`
Expected: `0.5`

---

### Task 2: Round 1 凍結位置

**Files:**
- Modify: `robot_tracker.py:278-284`

- [ ] **Step 1: 修改速度預測邏輯**

將 Round 1 匹配中的速度預測改為條件式 — 只有 `missed_frames == 0` 的健康軌跡才用速度外推，grace-period 軌跡用凍結位置：

```python
                frame_gap = frame_idx - last_f
                # 只有上一幀成功匹配的軌跡才用速度外推
                # grace-period 軌跡凍結在最後位置，避免飄移搶別人偵測
                missed = self._missed_frames.get(label, 0)
                if missed == 0:
                    pred_cx = last_cx + last_vx * frame_gap
                    pred_cy = last_cy + last_vy * frame_gap
                else:
                    pred_cx = last_cx
                    pred_cy = last_cy
                spatial_dist = math.hypot(cx - pred_cx, cy - pred_cy)
                max_dist = self._reid_max_dist * (
                    1 + math.sqrt(frame_gap / self._fps))
```

**原理：** 當追蹤框連續 1-3 幀沒匹配到偵測，速度外推方向不可靠（機器人可能已經停下或轉彎）。凍結在最後已知位置，只有偵測重新出現在附近才會重新配對，避免飄移到別的機器人。

---

### Task 3: Round 2 鄰近守衛

**Files:**
- Modify: `robot_tracker.py:349-359`

- [ ] **Step 1: LOST 復活時檢查鄰近已匹配偵測**

在 Round 2 的 `lost_pairs` 迴圈中，復活前檢查該偵測是否太靠近已被匹配的偵測：

```python
            lost_pairs.sort(key=lambda p: p[0])
            for dist, di, label in lost_pairs:
                if di in used_dets or label in used_labels:
                    continue
                # 鄰近守衛：如果這個偵測太靠近已匹配的偵測，跳過
                # （避免 LOST 復活在重複偵測上，造成兩框追同一台）
                det_cx, det_cy = det_info[di][0], det_info[di][1]
                too_close = False
                for matched_di in used_dets:
                    mcx, mcy = det_info[matched_di][0], det_info[matched_di][1]
                    if math.hypot(det_cx - mcx, det_cy - mcy) < 80:
                        too_close = True
                        break
                if too_close:
                    continue
                det_label_map[di] = label
                used_dets.add(di)
                used_labels.add(label)
                # 復活: LOST → ACTIVE
                self._track_state[label] = "active"
                self._lost_since.pop(label, None)
                self._missed_frames[label] = 0
```

**原理：** 全幀 + tiled 偵測可能對同一個機器人產生兩個 bbox（NMS 未合併）。如果 LOST 軌跡配到重複偵測，就會復活在已有追蹤的機器人上。80px 門檻約為一個 bumper bbox 的寬度。

---

### Task 4: 每幀重複清除

**Files:**
- Modify: `robot_tracker.py` — import 新增 `MOT_DEDUP_IOU`
- Modify: `robot_tracker.py:408-409` — 在「處理所有偵測」之後、「更新未匹配 ACTIVE 軌跡」之前插入

- [ ] **Step 1: 新增 import**

在 `robot_tracker.py` 頂部的 config import 中加入 `MOT_DEDUP_IOU`：

```python
from config import (
    ...
    MOT_OCCLUSION_MARGIN,
    MOT_DEDUP_IOU,
    ...
)
```

- [ ] **Step 2: 插入重複清除邏輯**

在 line 408（直方圖更新結束）之後、line 409（更新未匹配 ACTIVE 軌跡）之前插入：

```python
        # ═══ 重複追蹤清除 ═══
        # 如果兩個軌跡的 bbox 高度重疊（IoU > 閾值），移除偵測幀較少的
        if len(frame_bbox) > 1:
            labels_in_frame = list(frame_bbox.keys())
            to_remove = set()
            for i in range(len(labels_in_frame)):
                la = labels_in_frame[i]
                if la in to_remove:
                    continue
                ba = frame_bbox[la]  # (x1, y1, x2, y2)
                for j in range(i + 1, len(labels_in_frame)):
                    lb = labels_in_frame[j]
                    if lb in to_remove:
                        continue
                    bb = frame_bbox[lb]
                    # 計算 IoU
                    ix1 = max(ba[0], bb[0])
                    iy1 = max(ba[1], bb[1])
                    ix2 = min(ba[2], bb[2])
                    iy2 = min(ba[3], bb[3])
                    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                    area_a = (ba[2] - ba[0]) * (ba[3] - ba[1])
                    area_b = (bb[2] - bb[0]) * (bb[3] - bb[1])
                    union = area_a + area_b - inter
                    iou = inter / union if union > 0 else 0
                    if iou > MOT_DEDUP_IOU:
                        # 保留偵測幀多的，移除少的
                        fa = len(self._detected_frames.get(la, set()))
                        fb = len(self._detected_frames.get(lb, set()))
                        victim = lb if fa >= fb else la
                        to_remove.add(victim)
            for victim in to_remove:
                frame_pos.pop(victim, None)
                frame_bbox.pop(victim, None)
                results.pop(victim, None)
                # 標記為 LOST（讓它自然超時被清除）
                self._track_state[victim] = "lost"
                if victim not in self._lost_since:
                    self._lost_since[victim] = frame_idx
                self._missed_frames[victim] = 0
```

**原理：** 這是最後一道防線。無論前面的守衛怎麼漏掉，只要兩個追蹤框在同一幀疊在同一個機器人上（IoU > 0.5），就移除歷史短的那個。標記為 LOST 而非直接刪除，讓它走正常超時流程。

---

### Task 5: 端對端驗證

- [ ] **Step 1: Import 檢查**

Run: `python -c "import robot_tracker; print('OK')"`
Expected: `OK`

- [ ] **Step 2: AST 語法檢查**

Run: `python -c "import ast; ast.parse(open('robot_tracker.py').read()); print('AST OK')"`
Expected: `AST OK`

- [ ] **Step 3: 常數驗證**

Run: `python -c "from config import MOT_DEDUP_IOU; assert MOT_DEDUP_IOU == 0.5; print('constants OK')"`
Expected: `constants OK`

- [ ] **Step 4: 手動測試**

開啟影片 → 標記遮擋區域 → 執行分析 → 觀察：
1. 機器人被遮擋後，追蹤框是否凍結在原位不亂飄
2. 機器人重新出現時，追蹤框是否正確恢復追蹤
3. 不再出現兩個框追同一台機器人的情況

# FRC Scoring Analyzer — Error Log

*記錄開發過程中遇到的錯誤和解決方案*

## 2026-02-24: 場地遮罩座標雙重偏移 — 偵測全部為 0

### 症狀
設定 ROI 裁切 + 場地邊界後執行分析，球偵測和機器人偵測全部為 0。分析日誌每幀零偵測。

### 原因
`_field_boundary` 儲存 ROI 相對座標 `(0..rw, 0..rh)`，但 `_run_analysis()` 建立遮罩時（app.py 約第 1922 行）又減去 ROI 偏移 `(rx, ry)` → 座標變負數 → `cv2.fillPoly` 繪製在畫面外 → 遮罩全零 → `cv2.bitwise_and` 輸出全黑幀 → 偵測器在黑幀上執行 → 全部為 0。

### 解決
移除 `app.py` ROI 分支中多餘的 `(x - rx, y - ry)` 偏移，直接使用 `_field_boundary` 座標建立遮罩。

### 預防
- **座標系統規則**：`_field_boundary` 在設定時就已經轉換為當前顯示座標系（ROI 裁切後 = ROI 相對座標），後續使用時不應再次轉換
- 可在建立遮罩後加入 `assert mask.sum() > 0` 斷言，提前發現全零遮罩

---
*Created: 2026-02-14*

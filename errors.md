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

## 2026-03-10: tuis 多解析度裁切座標錯位

### 症狀
tuis 賽事 crop_events.py 裁切結果錯位，qm1 正常但 qm2 以後的圖片 crop 範圍偏離。

### 原因
tuis 賽事中 qm1 影片為 640x360，qm2+ 為 1280x720。crop.json 用 qm1 校正，座標是基於 640x360 的絕對像素值，直接套用到 1280x720 影片時 crop 區域完全錯位（只裁到左上角 1/4）。

### 解決
修改 crop_events.py：crop.json 新增 `base_w`/`base_h` 記錄校正時的影片解析度，裁切時讀取每個影片的實際解析度，自動等比縮放 crop 座標（`scale_x = actual_w / base_w`），輸出統一 resize 到固定尺寸。

### 預防
- crop.json 必須記錄 base 解析度，不能假設所有影片同解析度
- FRC 同賽事影片可能混合多種解析度（特別是 qualifier 早期場次）

---

## 2026-03-10: Gemini API 504 產生空 label 檔案

### 症狀
mndu 賽事 Gemini 標註完成後只有 58 張有效標註（應為 ~214），大量 .txt 檔案為 0 bytes。

### 原因
Gemini API 回傳 504 Gateway Timeout 時，auto_annotate.py 的 except 路徑仍會建立空的 .txt 檔案。重跑時因檔案已存在（`os.path.exists()` 檢查）而跳過，導致永遠無法修復。

### 解決
手動刪除所有 0 bytes 的 label 檔案（`find labels -size 0 -delete`），然後重跑 Gemini 標註。mndu 從 58 張修復到 214 張。

### 預防
- auto_annotate.py 應在寫入前檢查 API 回應是否有效，不應在錯誤時建立空檔
- 或在重跑時加入 `--force` 選項覆蓋空檔
- 可加入 0 bytes 檔案自動清理邏輯

---
*Created: 2026-02-14*

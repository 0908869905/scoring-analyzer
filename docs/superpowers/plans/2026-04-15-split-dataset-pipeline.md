# 分離 frc-train-review 新專案 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 scoring-analyzer 的資料集 pipeline（13 支腳本 + 58 GB `datasets/` + 訓練相關檔案 + 模型備份）完整搬出為獨立 git repo `D:\FRC\frc-train-review`，搬完後兩個專案可獨立運作。

**Architecture:** 同磁碟機 rename 操作（秒級完成）。先 commit scoring-analyzer 累積的 session 28-31 工作讓歷史乾淨，切新分支 `chore/split-dataset-pipeline`，建立新專案骨架與設定檔，複製 `frc_robot.onnx`（兩邊各留一份執行期與權威來源），搬動腳本/資料/訓練相關檔案，各自 commit，最後雙邊驗證。

**Tech Stack:** Git、bash（Git Bash on Windows，支援 `mv` 在同磁碟機做 rename）、Python（僅用於驗證 import / `--help`）

**Spec reference:** `docs/superpowers/specs/2026-04-15-split-dataset-pipeline-design.md`

---

## Chunk 1: scoring-analyzer 準備

### Task 1: Commit scoring-analyzer 累積的 session 28-31 工作

**Files:**
- Modify: 多個核心 Python 檔（`app.py`、`config.py`、`presets/1.json`、`robot_tracker.py`、`scoring.py`、`errors.md`、`FINDINGS.md`、`PROGRESS.md`、`CLAUDE.md`、`.gitignore`）
- Delete: `ENGINEERING_NOTEBOOK.*`（4 檔）、`TRAIN_README.txt`、13 支舊位置 Python、`bad_frames_review.jpg`
- New: `docs/TRAIN_README.txt`、`docs/notebook/`、`docs/superpowers/plans/2026-03-14-mot-freeze-dedup.md`、`models/`（含 `backup/`）、`scripts/dataset/`（13 檔）

- [ ] **Step 1: 確認目前分支與狀態**

Run:
```bash
cd /d/FRC/scoring-analyzer
git branch --show-current
git status --short
```
Expected: 分支為 `feat/gemini-detector`，有大量 M/D/?? 檔案。

- [ ] **Step 2: 追蹤所有修改與刪除（tracked files）**

Run:
```bash
cd /d/FRC/scoring-analyzer
git add -u
```
Expected: 無輸出（成功 stage 所有 M/D tracked files）。

- [ ] **Step 3: 加入所有新增的未追蹤檔案（不含 `models/`）**

**重要：** 故意**不**加入 `models/`。原因：`models/*.onnx` 被 `.gitignore` 覆蓋但 `models/backup/frc_robot (4).onnx`（9.4 MB）未被任何規則 ignore，若 `git add models/` 會把它放進 git 歷史，變成永久 bloat。Task 10 會把 `backup/` 整個搬到新專案，搬完後 `?? models/` 會自動消失。

Run:
```bash
cd /d/FRC/scoring-analyzer
git add docs/TRAIN_README.txt docs/notebook/ docs/superpowers/plans/2026-03-14-mot-freeze-dedup.md scripts/
git status --short
```
Expected: `git status --short` 顯示大量 A/M/D 前綴（已 staged），剩餘 `??` 應只有：
- `?? models/`（故意不加，Task 10 後消失）
- `?? docs/superpowers/plans/2026-04-15-split-dataset-pipeline.md`（本 plan，執行 plan 前已在獨立 commit 中；若執行者看到此 `??`，代表 plan doc 尚未 commit，應先 `git add` + commit 再繼續）

如果出現其他未預期的 `??` 項目，停止並排查。

- [ ] **Step 4: Commit**

Run:
```bash
cd /d/FRC/scoring-analyzer
git commit -m "$(cat <<'EOF'
feat: MOT Hungarian 匹配 + 歸因簡化 + 專案結構整理（session 28-31）

- robot_tracker.py: Round 1 改用 Hungarian 全域最優匹配
- scoring.py: 射手歸因從 140 輪迭代簡化為 HP 線段交叉 + 80 幀 proximity
- 專案結構：dataset 腳本移至 scripts/dataset/，文件移至 docs/
- CLAUDE.md、.gitignore 更新反映新結構

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```
Expected: Commit 成功，輸出 commit hash。

- [ ] **Step 5: 驗證 commit 後狀態**

Run:
```bash
cd /d/FRC/scoring-analyzer
git status --short
```
Expected: 只剩 `?? models/`（故意保留，Task 10 後會消失）。若本 plan 尚未獨立 commit，也會看到 `?? docs/superpowers/plans/2026-04-15-split-dataset-pipeline.md`。無其他 M/D 項目。

---

### Task 2: 切出新分支 `chore/split-dataset-pipeline`

**Files:** 無（純 git 操作）

- [ ] **Step 1: 從 `feat/gemini-detector` 切出新分支**

Run:
```bash
cd /d/FRC/scoring-analyzer
git checkout -b chore/split-dataset-pipeline
git branch --show-current
```
Expected: 輸出 `chore/split-dataset-pipeline`。

- [ ] **Step 2: 驗證上游分支正確**

Run:
```bash
cd /d/FRC/scoring-analyzer
git log --oneline -3
```
Expected: 最新 commit 是 Task 1 的 "feat: MOT Hungarian 匹配 ..." commit，前面是 "docs: 分離 frc-train-review 新專案設計文件"（spec commit）與 `27dcd8b chore: 新增資料集工具腳本`。

---

## Chunk 2: 新專案骨架

### Task 3: 建立 `D:\FRC\frc-train-review` + `git init`

**Files:**
- Create: `D:\FRC\frc-train-review\` 目錄（新專案 root）

- [ ] **Step 1: 確認目的地不存在**

Run:
```bash
ls -la /d/FRC/frc-train-review 2>&1 | head -5
```
Expected: `ls: cannot access '/d/FRC/frc-train-review': No such file or directory`。若目錄已存在則停止，排查後再繼續。

- [ ] **Step 2: 建立目錄**

Run:
```bash
mkdir -p /d/FRC/frc-train-review
cd /d/FRC/frc-train-review
pwd
```
Expected: `/d/FRC/frc-train-review`。

- [ ] **Step 3: git init**

Run:
```bash
cd /d/FRC/frc-train-review
git init
```
Expected: `Initialized empty Git repository in D:/FRC/frc-train-review/.git/`。

---

### Task 4: 寫新專案 `.gitignore`

**Files:**
- Create: `D:\FRC\frc-train-review\.gitignore`

- [ ] **Step 1: 建立 `.gitignore`**

Create `D:\FRC\frc-train-review\.gitignore` with exactly this content:
```
# Python
__pycache__/
*.pyc
*.pyo
.venv/
venv/

# 資料與模型（大檔案，不入 git）
datasets/
models/*.onnx
models/backup/
runs/

# IDE
.vscode/
.idea/

# 環境變數（含 GEMINI_API_KEY、TBA_API_KEY）
.env
```

- [ ] **Step 2: 驗證**

Run:
```bash
cat /d/FRC/frc-train-review/.gitignore | head -20
```
Expected: 顯示上述內容。

---

### Task 5: 寫新專案 `requirements.txt`

**Files:**
- Create: `D:\FRC\frc-train-review\requirements.txt`

- [ ] **Step 1: 建立 `requirements.txt`**

Create `D:\FRC\frc-train-review\requirements.txt` with exactly this content:
```
# 核心（多數腳本使用）
opencv-python>=4.9.0
numpy>=1.24.0
Pillow>=10.0.0

# label_editor.py 審核 GUI
customtkinter>=5.2.0

# auto_annotate.py Gemini Vision 自動標註
google-genai>=1.0.0

# 訓練（選配，只有執行 train_robot_model.py 時需要）
# pip install ultralytics roboflow
```

- [ ] **Step 2: 驗證**

Run:
```bash
cat /d/FRC/frc-train-review/requirements.txt
```
Expected: 顯示上述內容。

---

### Task 6: 寫新專案 `README.md`

**Files:**
- Create: `D:\FRC\frc-train-review\README.md`

- [ ] **Step 1: 建立 `README.md`**

Create `D:\FRC\frc-train-review\README.md` with exactly this content:
````markdown
# FRC Train & Review

FRC 機器人偵測模型的資料集 pipeline — 下載、自動標註、審核、訓練。

> **分離自:** `D:\FRC\scoring-analyzer` (2026-04-15)
> **最終模型用於:** scoring-analyzer 的 YOLO 機器人偵測模式

## Pipeline 4 階段

```
[下載]         → [取幀/裁切]    → [自動標註]      → [審核]          → [合併/訓練]
download       extract          auto_annotate    label_editor     merge/train
matches/       frames/          (Gemini Vision)  (CustomTkinter)  (Colab T4)
batch          crop_events
```

## Quick Start

```bash
# 1. 安裝依賴
pip install -r requirements.txt

# 2. 設定環境變數
export GEMINI_API_KEY=your_gemini_key
export TBA_API_KEY=your_tba_key

# 3. 下載比賽影片（TBA API + YouTube）
python download_matches.py --event 2026inmis
python batch_download.py                                 # 批次多賽事

# 4. 從影片取幀
python extract_frames.py datasets/2026inmis/videos/ --fps 2

# 5. 自動標註（Gemini Vision）
python auto_annotate.py --images datasets/2026inmis/images/
python batch_annotate.py --sample 400                    # 三階段 pipeline

# 6. 收集 + 審核
python collect_to_notyet.py --auto
python label_editor.py datasets/notyet                   # Space 標記已審核

# 7. 同步審核完成的資料到 reviewed/
python sync_reviewed.py

# 8. 合併為訓練集
python merge_datasets.py

# 9. 訓練（本地 GPU）
python train_robot_model.py --local-dataset datasets/merged/data.yaml

# 或使用 Colab（推薦，需先 zip + 上傳 merged.zip 到 Google Drive）
# 開啟 train_colab.ipynb
```

## 目錄結構

```
frc-train-review/
├── *.py                      # 13 支 pipeline 腳本（flat layout）
├── train_colab.ipynb         # Colab T4 訓練 notebook（final 模型用）
├── TRAIN_README.txt          # GPU 訓練步驟
│
├── datasets/                 # gitignored（58 GB）
│   ├── merged/               # final 模型訓練集（1826 張）
│   ├── merged.zip            # Colab 上傳包（417 MB）
│   ├── reviewed/             # 審核完成
│   ├── notyet/               # 待審核
│   ├── 2026*/                # 14+ 個 2026 賽事原始素材
│   ├── 2023mslr/, 2024*/     # 舊賽季
│   └── labels_raw_orphoned/  # 孤兒標註
│
└── models/                   # gitignored
    ├── frc_robot.onnx        # 最新訓練輸出（2026-03-13, YOLOv26n）
    ├── frc_robot_old*.onnx   # 歷史版本
    └── backup/               # 其他 onnx 備份
```

## 環境變數

| 變數 | 用途 | 取得方式 |
|---|---|---|
| `GEMINI_API_KEY` | `auto_annotate.py` Gemini Vision API | https://ai.google.dev |
| `TBA_API_KEY` | `download_matches.py` TBA 賽事查詢 | https://www.thebluealliance.com/account |

## 與 scoring-analyzer 的關聯

- 本專案訓練的 `models/frc_robot.onnx` 是 scoring-analyzer 執行期使用的機器人偵測模型
- 訓練新模型後，複製到 `D:\FRC\scoring-analyzer\models\frc_robot.onnx` 即可生效
- 本專案不 import scoring-analyzer 的任何模組
````

- [ ] **Step 2: 驗證檔案建立**

Run:
```bash
ls -la /d/FRC/frc-train-review/README.md
wc -l /d/FRC/frc-train-review/README.md
```
Expected: 檔案存在，行數 > 50。

---

## Chunk 3: 複製 + 搬家

### Task 7: 複製 `frc_robot.onnx` 到新專案（scoring-analyzer 保留一份）

**Files:**
- Copy: `D:\FRC\scoring-analyzer\models\frc_robot.onnx` → `D:\FRC\frc-train-review\models\frc_robot.onnx`

- [ ] **Step 1: 建立 `models/` 目錄**

Run:
```bash
mkdir -p /d/FRC/frc-train-review/models
```
Expected: 無輸出。

- [ ] **Step 2: 複製 final 模型**

Run:
```bash
cp /d/FRC/scoring-analyzer/models/frc_robot.onnx /d/FRC/frc-train-review/models/frc_robot.onnx
```
Expected: 無輸出。

- [ ] **Step 3: 驗證兩邊都有該檔**

Run:
```bash
ls -la /d/FRC/scoring-analyzer/models/frc_robot.onnx /d/FRC/frc-train-review/models/frc_robot.onnx
```
Expected: 兩個檔案都存在，大小相同（9805923 bytes）。

---

### Task 8: 搬動 13 支腳本到新專案 root（flat layout）

**Files:**
- Move: `scoring-analyzer/scripts/dataset/*.py` (13 檔) → `frc-train-review/*.py`

- [ ] **Step 1: 驗證來源目錄有 13 支腳本**

Run:
```bash
ls /d/FRC/scoring-analyzer/scripts/dataset/*.py | wc -l
```
Expected: `13`。

- [ ] **Step 2: 一次搬動 13 支腳本**

Run:
```bash
mv /d/FRC/scoring-analyzer/scripts/dataset/auto_annotate.py \
   /d/FRC/scoring-analyzer/scripts/dataset/batch_annotate.py \
   /d/FRC/scoring-analyzer/scripts/dataset/batch_download.py \
   /d/FRC/scoring-analyzer/scripts/dataset/build_dataset.py \
   /d/FRC/scoring-analyzer/scripts/dataset/collect_to_notyet.py \
   /d/FRC/scoring-analyzer/scripts/dataset/crop_events.py \
   /d/FRC/scoring-analyzer/scripts/dataset/detect_bad_frames.py \
   /d/FRC/scoring-analyzer/scripts/dataset/download_matches.py \
   /d/FRC/scoring-analyzer/scripts/dataset/extract_frames.py \
   /d/FRC/scoring-analyzer/scripts/dataset/label_editor.py \
   /d/FRC/scoring-analyzer/scripts/dataset/merge_datasets.py \
   /d/FRC/scoring-analyzer/scripts/dataset/sync_reviewed.py \
   /d/FRC/scoring-analyzer/scripts/dataset/train_robot_model.py \
   /d/FRC/frc-train-review/
```
Expected: 無輸出。

- [ ] **Step 3: 驗證搬動成功**

Run:
```bash
ls /d/FRC/frc-train-review/*.py | wc -l
ls /d/FRC/scoring-analyzer/scripts/dataset/*.py 2>&1 | head -5
```
Expected: 第一行輸出 `13`；第二行輸出 `ls: cannot access ... No such file or directory`（或空列）。

---

### Task 9: 搬動 `datasets/` 目錄（58 GB）

**Files:**
- Move: `scoring-analyzer/datasets/` → `frc-train-review/datasets/`

- [ ] **Step 1: 目標目錄不能事先存在**

Run:
```bash
ls -la /d/FRC/frc-train-review/datasets 2>&1 | head -3
```
Expected: `ls: cannot access ... No such file or directory`。若存在則停止，檢查是先前執行殘留還是 Task 4 `.gitignore` 被誤解為建立目錄。

- [ ] **Step 2: 搬動 datasets**

同磁碟機 `mv` = `rename()` 系統呼叫，58 GB 秒級完成。

Run:
```bash
mv /d/FRC/scoring-analyzer/datasets /d/FRC/frc-train-review/datasets
```
Expected: 無輸出（秒完成）。

- [ ] **Step 3: 驗證搬動成功**

Run:
```bash
ls -la /d/FRC/frc-train-review/datasets/ | head -10
ls /d/FRC/scoring-analyzer/datasets 2>&1 | head -3
```
Expected: 第一個 ls 列出 `merged/`、`merged.zip`、`reviewed/`、`notyet/`、`2024mslr/`、多個 `2026*/` 等；第二個 ls 輸出 `No such file or directory`。

- [ ] **Step 4: 驗證關鍵檔案存在**

Run:
```bash
ls -la /d/FRC/frc-train-review/datasets/merged/data.yaml
ls -la /d/FRC/frc-train-review/datasets/merged.zip
ls /d/FRC/frc-train-review/datasets/merged/images/train 2>&1 | head -3
```
Expected: `data.yaml` 存在，`merged.zip` 存在（~417 MB），`images/train` 目錄下有圖片。

---

### Task 10: 搬動訓練相關檔案與模型備份

**Files:**
- Move: `scoring-analyzer/train_colab.ipynb` → `frc-train-review/train_colab.ipynb`
- Move: `scoring-analyzer/docs/TRAIN_README.txt` → `frc-train-review/TRAIN_README.txt`
- Move: `scoring-analyzer/models/frc_robot_old.onnx` → `frc-train-review/models/`
- Move: `scoring-analyzer/models/frc_robot_old2.onnx` → `frc-train-review/models/`
- Move: `scoring-analyzer/models/frc_robot_yolo11n.onnx` → `frc-train-review/models/`
- Move: `scoring-analyzer/models/backup/` → `frc-train-review/models/backup/`

- [ ] **Step 1: 搬動 Colab notebook 與 TRAIN_README**

Run:
```bash
mv /d/FRC/scoring-analyzer/train_colab.ipynb /d/FRC/frc-train-review/train_colab.ipynb
mv /d/FRC/scoring-analyzer/docs/TRAIN_README.txt /d/FRC/frc-train-review/TRAIN_README.txt
```
Expected: 無輸出。

- [ ] **Step 2: 搬動舊版模型備份**

Run:
```bash
mv /d/FRC/scoring-analyzer/models/frc_robot_old.onnx /d/FRC/frc-train-review/models/frc_robot_old.onnx
mv /d/FRC/scoring-analyzer/models/frc_robot_old2.onnx /d/FRC/frc-train-review/models/frc_robot_old2.onnx
mv /d/FRC/scoring-analyzer/models/frc_robot_yolo11n.onnx /d/FRC/frc-train-review/models/frc_robot_yolo11n.onnx
```
Expected: 無輸出。

- [ ] **Step 3: 搬動 `models/backup/` 目錄**

Run:
```bash
mv /d/FRC/scoring-analyzer/models/backup /d/FRC/frc-train-review/models/backup
```
Expected: 無輸出。

- [ ] **Step 4: 驗證搬動結果**

Run:
```bash
ls /d/FRC/frc-train-review/ | sort
echo "---"
ls /d/FRC/frc-train-review/models/ | sort
echo "---"
ls /d/FRC/scoring-analyzer/models/ | sort
```
Expected:
- `frc-train-review/` root 包含 13 支 `.py`、`train_colab.ipynb`、`TRAIN_README.txt`、`README.md`、`requirements.txt`、`.gitignore`、`datasets/`、`models/`、`.git/`
- `frc-train-review/models/` 包含 `frc_robot.onnx`、`frc_robot_old.onnx`、`frc_robot_old2.onnx`、`frc_robot_yolo11n.onnx`、`backup/`
- `scoring-analyzer/models/` 只剩 `frc_robot.onnx` 與 `object_tracking_vittrack_2023sep.onnx`

---

## Chunk 4: 新專案 initial commit

### Task 11: 新專案 `git add` + initial commit

**Files:**
- All non-gitignored files in `D:\FRC\frc-train-review`

- [ ] **Step 1: 檢查 git status**

Run:
```bash
cd /d/FRC/frc-train-review
git status --short
```
Expected: 所有 `.py`、`.gitignore`、`requirements.txt`、`README.md`、`TRAIN_README.txt`、`train_colab.ipynb` 顯示為 `??`。`datasets/` 與 `models/*.onnx`、`models/backup/` **不應出現**（被 `.gitignore` 排除）。

- [ ] **Step 2: 驗證 .gitignore 正確排除大檔**

Run:
```bash
cd /d/FRC/frc-train-review
git check-ignore -v datasets/ models/frc_robot.onnx models/backup/
```
Expected: 三行輸出，每行顯示對應的 `.gitignore` 規則（如 `.gitignore:8:datasets/	datasets/`）。

- [ ] **Step 3: 加入所有應追蹤的檔案**

Run:
```bash
cd /d/FRC/frc-train-review
git add .gitignore requirements.txt README.md TRAIN_README.txt train_colab.ipynb *.py
git status --short
```
Expected: 所有檔案顯示 `A` 前綴（已 staged）。

- [ ] **Step 4: Initial commit**

Run:
```bash
cd /d/FRC/frc-train-review
git commit -m "$(cat <<'EOF'
initial: dataset pipeline split from scoring-analyzer

13 支 pipeline 腳本（下載 / 取幀 / 自動標註 / 審核 / 合併 / 訓練）
從 D:\FRC\scoring-analyzer\scripts\dataset\ 搬出為獨立 repo。

- auto_annotate.py, batch_annotate.py, collect_to_notyet.py（標註）
- label_editor.py, sync_reviewed.py, detect_bad_frames.py（審核）
- download_matches.py, batch_download.py（下載）
- extract_frames.py, crop_events.py（取幀/裁切）
- merge_datasets.py, build_dataset.py（合併）
- train_robot_model.py, train_colab.ipynb（訓練）

Final 模型 frc_robot.onnx 由 datasets/merged/ 在 Colab T4 訓練產出。

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```
Expected: Commit 成功，輸出 commit hash 與檔案數統計。

- [ ] **Step 5: 驗證 commit 乾淨**

Run:
```bash
cd /d/FRC/frc-train-review
git log --oneline
git status
```
Expected: `git log` 顯示 1 個 commit（initial）。`git status` 顯示 "nothing to commit, working tree clean"。

---

## Chunk 5: scoring-analyzer 清理

### Task 12: 刪除空 `scripts/` 目錄

**Files:**
- Delete: `scoring-analyzer/scripts/` (整個目錄，空)

- [ ] **Step 1: 驗證 `scripts/dataset/` 已空**

Run:
```bash
ls /d/FRC/scoring-analyzer/scripts/dataset/ 2>&1
```
Expected: 無檔案列出（目錄空），或輸出 `cannot access` 若目錄也被 mv 帶走（bash mv 不會，應仍存在但空）。

- [ ] **Step 2: 刪除空目錄**

Run:
```bash
rmdir /d/FRC/scoring-analyzer/scripts/dataset 2>/dev/null
rmdir /d/FRC/scoring-analyzer/scripts 2>/dev/null
ls /d/FRC/scoring-analyzer/scripts 2>&1 | head -3
```
Expected: 最後的 `ls` 輸出 `ls: cannot access ... No such file or directory`。

---

### Task 13: 更新 scoring-analyzer 的 `CLAUDE.md`

**Files:**
- Modify: `D:\FRC\scoring-analyzer\CLAUDE.md` (多處 edits)

- [ ] **Step 1: 移除 Tech Stack 中的 `google-genai`**

Edit `D:\FRC\scoring-analyzer\CLAUDE.md`:

`old_string`:
```
- onnxruntime 1.17+ (YOLO ONNX 本地離線推理，支援 CUDA/DirectML/CPU 自動選擇)
- supervision 0.21+ (ByteTrack 多目標追蹤)
- google-genai (Gemini Vision API，自動標註用，`scripts/dataset/auto_annotate.py` 專用)
```

`new_string`:
```
- onnxruntime 1.17+ (YOLO ONNX 本地離線推理，支援 CUDA/DirectML/CPU 自動選擇)
- supervision 0.21+ (ByteTrack 多目標追蹤)
```

- [ ] **Step 2: 移除訓練與 dataset pipeline 的 Run Commands**

Edit `D:\FRC\scoring-analyzer\CLAUDE.md`:

`old_string`:
```
# 啟動並載入影片
python main.py path/to/video.mp4

# 訓練機器人偵測模型（需額外安裝 roboflow, ultralytics）
python scripts/dataset/train_robot_model.py --api-key YOUR_ROBOFLOW_KEY

# Label Editor 互動標註工具
python scripts/dataset/label_editor.py datasets/2024mslr
python scripts/dataset/label_editor.py datasets/2024mslr --start 0 --end 1421    # 前半（分工用）
python scripts/dataset/label_editor.py datasets/2024mslr --start 1421             # 後半（分工用）
python scripts/dataset/label_editor.py datasets/2024mslr --labels labels_raw      # 自訂標註目錄
python scripts/dataset/label_editor.py datasets/2026cosp --images images --labels labels
# Space 鍵標記已審核，狀態存 review_state.json，支援斷點恢復

# 批次下載 + 標註 pipeline
python scripts/dataset/batch_download.py                       # 批次下載多賽事影片
python scripts/dataset/batch_annotate.py --sample 400          # 三階段：抽樣→Gemini標註→收集
python scripts/dataset/collect_to_notyet.py                    # 收集多賽事標註到 datasets/notyet/
python scripts/dataset/sync_reviewed.py                        # 審核完成後同步到 datasets/reviewed/
```
```

`new_string`:
```
# 啟動並載入影片
python main.py path/to/video.mp4
```
```

- [ ] **Step 3: 移除 `train_colab.ipynb` 行**

Edit `D:\FRC\scoring-analyzer\CLAUDE.md`:

`old_string`:
```
├── errors.md              # 錯誤追蹤
├── train_colab.ipynb      # Colab 訓練 notebook（gitignored）
│
│   ── 工具腳本 ───────────────────────────────────────────────
```

`new_string`:
```
├── errors.md              # 錯誤追蹤
│
```

- [ ] **Step 4: 移除 `scripts/dataset/` 目錄樹區塊**

Edit `D:\FRC\scoring-analyzer\CLAUDE.md`:

`old_string`:
```
├── scripts/
│   └── dataset/           # 13 個資料集處理工具（不 import 核心）
│       ├── download_matches.py    # TBA API 查詢 + YouTube 下載
│       ├── batch_download.py      # 批次下載多賽事影片
│       ├── extract_frames.py      # 影片取幀
│       ├── crop_events.py         # 影片裁切（crop.json 多解析度等比縮放）
│       ├── auto_annotate.py       # Gemini Vision 自動標註 pipeline
│       ├── batch_annotate.py      # 三階段標註 pipeline
│       ├── label_editor.py        # 互動 YOLO bbox 編輯器（自製）
│       ├── collect_to_notyet.py   # 收集多賽事 Gemini 標註到 notyet/
│       ├── sync_reviewed.py       # 審核完成自動轉移到 reviewed/
│       ├── merge_datasets.py      # 多賽事資料集合併 + train/val split
│       ├── detect_bad_frames.py   # 壞幀偵測（過曝/紙花/假標註）
│       ├── build_dataset.py       # 最終資料集組合
│       └── train_robot_model.py   # 本地 YOLO 訓練腳本
│
│   ── 文件目錄 ───────────────────────────────────────────────
```

`new_string`:
```
│   ── 文件目錄 ───────────────────────────────────────────────
```

- [ ] **Step 5: 移除 docs 下的 `TRAIN_README.txt`**

Edit `D:\FRC\scoring-analyzer\CLAUDE.md`:

`old_string`:
```
├── docs/
│   ├── notebook/          # FRC 工程筆記（md/html/txt/EN 4 個版本）
│   ├── plans/             # 設計文件與實作計劃（舊）
│   ├── superpowers/       # Superpowers 設計規格與實作計劃
│   │   ├── specs/
│   │   └── plans/
│   └── TRAIN_README.txt   # GPU 訓練步驟指南
```

`new_string`:
```
├── docs/
│   ├── notebook/          # FRC 工程筆記（md/html/txt/EN 4 個版本）
│   ├── plans/             # 設計文件與實作計劃（舊）
│   └── superpowers/       # Superpowers 設計規格與實作計劃
│       ├── specs/
│       └── plans/
```

- [ ] **Step 6: 更新 `models/` 目錄樹說明（移除已搬走的備份）**

Edit `D:\FRC\scoring-analyzer\CLAUDE.md`:

`old_string`:
```
├── models/                # ONNX 模型
│   ├── frc_robot.onnx             # YOLOv26n 機器人偵測（9.8 MB NMS-Free）
│   ├── frc_robot_old*.onnx        # 舊版備份
│   ├── object_tracking_vittrack_*.onnx  # VitTrack SOT
│   └── backup/                    # 離根目錄的備份檔
├── datasets/              # 訓練資料（gitignored）
│   ├── 2024mslr/          # 主訓練集（2842 張）
│   │   ├── images/, images_raw/, labels/, labels_raw/
│   │   ├── videos/                # 87 部 720p 比賽影片
│   │   └── yolo_dataset/          # YOLO train/val split + data.yaml
│   ├── 2023mslr/          # 2023 Magnolia Regional
│   ├── 2026*/             # 2026 賽季 14+ 個賽事（分工審核中）
│   ├── notyet/            # 待審核標註收集區
│   ├── merged/            # 合併資料集（1826 張）
│   ├── reviewed/          # 最終審核資料集（part1~4 分工切割）
│   └── labels_raw_orphoned/       # 早期孤立原始標註（2842 張 txt）
├── presets/               # Preset JSON（場地設定檔）
```

`new_string`:
```
├── models/                # ONNX 模型（runtime 用，訓練在 D:\FRC\frc-train-review）
│   ├── frc_robot.onnx             # YOLOv26n 機器人偵測（9.8 MB NMS-Free）
│   └── object_tracking_vittrack_*.onnx  # VitTrack SOT
├── presets/               # Preset JSON（場地設定檔）
```

- [ ] **Step 7: 更新 footer session 標記**

Edit `D:\FRC\scoring-analyzer\CLAUDE.md`:

`old_string`:
```
*Last updated: 2026-04-14 (session 31) — 專案結構整理：腳本移至 scripts/dataset/，文件移至 docs/*
```

`new_string`:
```
*Last updated: 2026-04-15 (session 32) — 分離 dataset pipeline 至 D:\FRC\frc-train-review*
```

- [ ] **Step 8: 驗證 CLAUDE.md 編輯結果**

Run:
```bash
grep -c "scripts/dataset" /d/FRC/scoring-analyzer/CLAUDE.md
grep -c "datasets/" /d/FRC/scoring-analyzer/CLAUDE.md
grep -c "google-genai" /d/FRC/scoring-analyzer/CLAUDE.md
grep "frc-train-review" /d/FRC/scoring-analyzer/CLAUDE.md
```
Expected:
- 第 1 行：`0`（所有 `scripts/dataset` 已移除）
- 第 2 行：`0`（所有 `datasets/` 已移除）
- 第 3 行：`0`（`google-genai` 已移除）
- 第 4 行：顯示 `2026-04-15 (session 32)` 與 `D:\FRC\frc-train-review` 的引用

---

### Task 14: scoring-analyzer Commit 分離操作

**Files:** 無新增，只有 CLAUDE.md 修改 + 檔案搬動的刪除記錄

- [ ] **Step 1: 檢查 git status**

Run:
```bash
cd /d/FRC/scoring-analyzer
git status --short
```
Expected（只列預期可能出現的項目；實際順序可能不同）：
- `M CLAUDE.md`（Task 13 修改）
- `D docs/TRAIN_README.txt`（Task 10 搬走）
- `D scripts/dataset/auto_annotate.py`（及其他 12 支，Task 8 搬走）

**不會出現**：
- `models/` 相關條目（整個 `models/` 未被 Task 1 git add，本來就不在索引中；Task 10 是純檔案層級搬動）
- `datasets/`（本來就 gitignored）
- `train_colab.ipynb`（本來就 gitignored）
- `scripts/`（空目錄，`rmdir` 後不留痕跡）

- [ ] **Step 2: 加入 CLAUDE.md 變更與所有刪除記錄**

Run:
```bash
cd /d/FRC/scoring-analyzer
git add -u
git status --short
```
Expected: 所有 M/D 項目變為已 staged（`M`/`D` 在左邊欄），右邊欄（工作目錄）乾淨。

- [ ] **Step 3: Commit**

Run:
```bash
cd /d/FRC/scoring-analyzer
git commit -m "$(cat <<'EOF'
chore: 分離 dataset pipeline 至 D:\FRC\frc-train-review

搬出內容：
- scripts/dataset/ 13 支腳本 → flat layout
- datasets/ 58 GB 全部（含 merged/ final 訓練集）
- train_colab.ipynb、docs/TRAIN_README.txt
- models/frc_robot_old*.onnx、frc_robot_yolo11n.onnx、backup/

保留於 scoring-analyzer：
- 核心 14 支 Python + test_analysis.py、diagnose.py
- models/frc_robot.onnx（runtime 用，與新專案同步）
- models/object_tracking_vittrack_*.onnx（球追蹤 SOT）
- docs/notebook/ENGINEERING_NOTEBOOK.*（比賽工程日誌）

CLAUDE.md 更新反映新結構。完整設計文件見
docs/superpowers/specs/2026-04-15-split-dataset-pipeline-design.md。

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```
Expected: Commit 成功，輸出 commit hash。

- [ ] **Step 4: 驗證 commit 後狀態乾淨**

Run:
```bash
cd /d/FRC/scoring-analyzer
git status --short
git log --oneline -5
```
Expected: `git status --short` 無輸出。`git log` 顯示最新 commit 為 "chore: 分離 dataset pipeline ..."，前面是 "feat: MOT Hungarian 匹配 ..."（session 28-31 commit）與 "docs: 分離 frc-train-review 新專案設計文件"（spec commit）。

---

## Chunk 6: 驗證

### Task 15: 驗證 scoring-analyzer runtime 完整

**Files:** 無修改，僅驗證

- [ ] **Step 1: 核心模組可 import**

Run:
```bash
cd /d/FRC/scoring-analyzer
python -c "import app, main, detection, tracking, robot_detection, robot_tracker, scoring, background, calibration, config, runtime_config, settings_window, geometry, utils; print('core OK')"
```
Expected: `core OK`。若任一模組 import 失敗，代表搬家誤動核心檔，立即停止並排查。

- [ ] **Step 2: runtime 模型檔存在**

Run:
```bash
ls -la /d/FRC/scoring-analyzer/models/frc_robot.onnx /d/FRC/scoring-analyzer/models/object_tracking_vittrack_*.onnx
```
Expected: `frc_robot.onnx` 存在（9805923 bytes），至少一個 `object_tracking_vittrack_*.onnx` 存在。

- [ ] **Step 3: 資料/腳本確實已搬走**

Run:
```bash
ls /d/FRC/scoring-analyzer/datasets 2>&1 | head -3
ls /d/FRC/scoring-analyzer/scripts 2>&1 | head -3
ls /d/FRC/scoring-analyzer/train_colab.ipynb 2>&1 | head -3
ls /d/FRC/scoring-analyzer/docs/TRAIN_README.txt 2>&1 | head -3
ls /d/FRC/scoring-analyzer/models/frc_robot_old.onnx 2>&1 | head -3
ls /d/FRC/scoring-analyzer/models/backup 2>&1 | head -3
```
Expected: 全部 6 行輸出 `No such file or directory`。

- [ ] **Step 4: `python main.py --help` 能執行**

Run:
```bash
cd /d/FRC/scoring-analyzer
python main.py --help 2>&1 | head -20
```
Expected: 顯示 `main.py` 的使用說明（或至少不因 import 錯誤而崩潰）。若 `main.py` 不支援 `--help`，改跑 `python -c "import main"`。

---

### Task 16: 驗證 frc-train-review pipeline 可用

**Files:** 無修改，僅驗證

- [ ] **Step 1: 13 支腳本 + Colab notebook 存在**

Run:
```bash
ls /d/FRC/frc-train-review/*.py | wc -l
ls /d/FRC/frc-train-review/train_colab.ipynb
ls /d/FRC/frc-train-review/TRAIN_README.txt
```
Expected: `13`，且後兩個檔案存在。

- [ ] **Step 2: 關鍵腳本可跑 `--help`（無 syntax error）**

Run:
```bash
cd /d/FRC/frc-train-review
python auto_annotate.py --help 2>&1 | head -5
python label_editor.py --help 2>&1 | head -5
python train_robot_model.py --help 2>&1 | head -5
python merge_datasets.py --help 2>&1 | head -5
```
Expected: 每個腳本輸出使用說明（argparse help text）。若缺少依賴（如 `google-genai`、`customtkinter`），腳本應印出友善錯誤訊息而非 stack trace（確認腳本內部有 `try/except ImportError` 保護）。**如果出現 Python syntax error 則失敗**；缺 dependency 不算失敗（使用者在新專案 `pip install -r requirements.txt` 才會補齊）。

- [ ] **Step 3: 資料集關鍵檔案存在**

Run:
```bash
ls -la /d/FRC/frc-train-review/datasets/merged/data.yaml
ls -la /d/FRC/frc-train-review/datasets/merged.zip
ls -la /d/FRC/frc-train-review/datasets/merged/images/train 2>&1 | head -3
ls /d/FRC/frc-train-review/datasets/ | wc -l
```
Expected: `data.yaml` 存在；`merged.zip` 約 417 MB；`images/train/` 目錄下有圖片；最後一行 datasets 子項目數 > 15（包含 `merged/`、`merged.zip`、`reviewed/`、`notyet/`、各 `2024*`、各 `2026*` 等）。

- [ ] **Step 4: 模型檔都在**

Run:
```bash
ls -la /d/FRC/frc-train-review/models/
```
Expected: 列出 `frc_robot.onnx`、`frc_robot_old.onnx`、`frc_robot_old2.onnx`、`frc_robot_yolo11n.onnx`、`backup/`。

- [ ] **Step 5: git 狀態乾淨**

Run:
```bash
cd /d/FRC/frc-train-review
git log --oneline
git status
```
Expected: 1 個 commit（initial），working tree clean。

- [ ] **Step 6: scoring-analyzer 分支狀態**

Run:
```bash
cd /d/FRC/scoring-analyzer
git branch --show-current
git log --oneline -4
git status
```
Expected:
- 分支為 `chore/split-dataset-pipeline`
- 最新 4 個 commit 包含：`chore: 分離 dataset pipeline ...`、`feat: MOT Hungarian 匹配 ...`、`docs: 分離 frc-train-review 新專案設計文件`、`27dcd8b chore: 新增資料集工具腳本...`
- Working tree clean

---

## 驗收 Checklist（對照 spec § 8）

全部通過視為分離成功：

1. [ ] `D:\FRC\frc-train-review\` 為獨立 git repo，`git log` 顯示 1 個 initial commit
2. [ ] 13 支腳本在 `frc-train-review/` root，皆可 `--help` 或至少不 syntax error
3. [ ] `frc-train-review/datasets/merged/data.yaml` 與 `merged.zip` 存在
4. [ ] `frc-train-review/models/frc_robot.onnx` 存在（9.8 MB）
5. [ ] `scoring-analyzer/` 中已無 `scripts/`、`datasets/`、`train_colab.ipynb`、`docs/TRAIN_README.txt`、`models/frc_robot_old*.onnx`、`models/frc_robot_yolo11n.onnx`、`models/backup/`
6. [ ] `scoring-analyzer/models/frc_robot.onnx` 仍存在
7. [ ] `scoring-analyzer/models/object_tracking_vittrack_*.onnx` 仍存在
8. [ ] `python -c "import app, main, detection, robot_tracker, scoring, robot_detection"` 在 scoring-analyzer 成功
9. [ ] `python main.py --help` 在 scoring-analyzer 成功
10. [ ] scoring-analyzer `feat/gemini-detector` 分支有 1 個新 commit（session 28-31 累積）
11. [ ] scoring-analyzer `chore/split-dataset-pipeline` 分支有 1 個新 commit（分離操作）

---

## 失敗恢復路徑

如果在過程中發現問題需要 rollback：

**Chunk 1-2（還沒動檔案）：**
- `git reset --hard HEAD~1` 撤銷 Task 1 commit
- `git branch -d chore/split-dataset-pipeline` 刪新分支
- `rm -rf /d/FRC/frc-train-review`

**Chunk 3（搬家中出錯）：**
- 檔案搬動都是 `mv` 同磁碟機 rename，用反向 `mv` 即可復原
- `mv /d/FRC/frc-train-review/*.py /d/FRC/scoring-analyzer/scripts/dataset/`
- `mv /d/FRC/frc-train-review/datasets /d/FRC/scoring-analyzer/datasets`
- 以此類推

**Chunk 4-5（新專案已 commit）：**
- 新專案 repo：`rm -rf /d/FRC/frc-train-review/.git` 後再重新做決定
- scoring-analyzer：`git reset --hard <前一個 commit hash>` 配合手動把檔案搬回來

**重要：** `frc_robot.onnx` 是複製操作（Task 7），scoring-analyzer 永遠有一份，runtime 永遠不會壞。這是整個計畫的安全氣囊。

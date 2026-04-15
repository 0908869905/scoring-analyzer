# 分離 `frc-train-review` 新專案 — 設計文件

**Date:** 2026-04-15
**Author:** Lee chung-yu (via Claude Code)
**Status:** Design — 待實作
**Source branch:** `feat/gemini-detector` → `chore/split-dataset-pipeline`

---

## 1. 目標

把 scoring-analyzer 的資料集 pipeline（下載 → 取幀 → 自動標註 → 審核 → 合併 → 訓練）完整搬出為獨立 repo `D:\FRC\frc-train-review`。搬完後兩個專案可獨立運作、不互相依賴。

**動機：** 使用者要在新專案進行「訓練與審核介面」相關的持續工作，scoring-analyzer 的主軸是即時分析（core Python + UI），兩邊的開發節奏與依賴不同，分離可降低互相干擾、簡化各自的 git 歷史與 `requirements.txt`。

**非目標：**
- 不調整 scoring-analyzer 的核心模組（`main.py`、`app.py`、`detection.py` 等 14 支 Python）
- 不重構腳本內部邏輯（只搬位置、最小必要的路徑修正）
- 不碰比賽工程日誌 `docs/notebook/ENGINEERING_NOTEBOOK.*`
- 不碰球追蹤模型 `models/object_tracking_vittrack_*.onnx`

---

## 2. 現況調查

### 2.1 最終版模型訓練資料來源（已驗證）

- **模型檔：** `models/frc_robot.onnx`（2026-03-13 07:55，9.8 MB，YOLOv26n NMS-Free）
- **訓練資料：** `datasets/merged/`（406 MB，1826 張，train/val split）
- **訓練方式：** Google Colab T4 GPU，100 epochs，YOLOv26n 預訓練權重
- **訓練腳本：** `train_colab.ipynb`（root，gitignored）
- **上傳包：** `datasets/merged.zip`（417 MB，2026-03-12 00:05）
- **資料來源鏈：** `datasets/2026*/images_sample/` + Gemini 標註 → `notyet/` → 審核 → `reviewed/` → `merge_datasets.py` → `merged/`

### 2.2 搬家來源盤點

| 類別 | 路徑 | 大小 | 備註 |
|---|---|---|---|
| 腳本 | `scripts/dataset/*.py` | 13 檔 | 全部不 import 核心模組（已驗證） |
| 資料集 | `datasets/` | 58 GB | 含各賽事 + merged + reviewed + notyet |
| Colab notebook | `train_colab.ipynb`（root） | 5.7 KB | gitignored |
| 訓練文件 | `docs/TRAIN_README.txt` | 916 B | 訓練步驟說明 |
| 模型備份 | `models/frc_robot_old.onnx`、`frc_robot_old2.onnx`、`frc_robot_yolo11n.onnx` | 約 30 MB | 歷史版本備份 |
| 模型備份 | `models/backup/` | 9.4 MB | `frc_robot (4).onnx` 一個孤立備份 |

### 2.3 關鍵依賴分析

- 13 支腳本**完全不 import** scoring-analyzer 核心模組（`config`、`detection`、`scoring`、`robot_tracker` 等）— 已用 grep 驗證
- `batch_annotate.py` 用 `subprocess.run(["python", "auto_annotate.py"])` 呼叫其他腳本，倚賴 cwd = 腳本所在目錄
- `train_robot_model.py:30` 用 `PROJECT_ROOT = Path(__file__).parent.resolve()` 推 `MODELS_DIR`，在 `scripts/dataset/` 下會解析到 `scripts/dataset/models/`（錯誤）— 但使用者沒從該路徑跑過，bug 未觸發
- scoring-analyzer `requirements.txt` 目前沒有 `google-genai`、`roboflow`、`ultralytics` 等訓練/標註相依套件（都是選配）

---

## 3. 搬家清單

### 3.1 移動（scoring-analyzer → frc-train-review）

| Source | Destination | 方式 |
|---|---|---|
| `scripts/dataset/auto_annotate.py` | `auto_annotate.py` | move |
| `scripts/dataset/batch_annotate.py` | `batch_annotate.py` | move |
| `scripts/dataset/batch_download.py` | `batch_download.py` | move |
| `scripts/dataset/build_dataset.py` | `build_dataset.py` | move |
| `scripts/dataset/collect_to_notyet.py` | `collect_to_notyet.py` | move |
| `scripts/dataset/crop_events.py` | `crop_events.py` | move |
| `scripts/dataset/detect_bad_frames.py` | `detect_bad_frames.py` | move |
| `scripts/dataset/download_matches.py` | `download_matches.py` | move |
| `scripts/dataset/extract_frames.py` | `extract_frames.py` | move |
| `scripts/dataset/label_editor.py` | `label_editor.py` | move |
| `scripts/dataset/merge_datasets.py` | `merge_datasets.py` | move |
| `scripts/dataset/sync_reviewed.py` | `sync_reviewed.py` | move |
| `scripts/dataset/train_robot_model.py` | `train_robot_model.py` | move |
| `datasets/` (58 GB) | `datasets/` | move（含 `merged.zip`、`2024mslr/`、所有 `2026*/`、`notyet/`、`reviewed/`、`labels_raw_orphoned/`、`_preview/`、`_test_samples/`、`download_log.txt`） |
| `train_colab.ipynb` | `train_colab.ipynb` | move |
| `docs/TRAIN_README.txt` | `TRAIN_README.txt` | move |
| `models/frc_robot_old.onnx` | `models/frc_robot_old.onnx` | move |
| `models/frc_robot_old2.onnx` | `models/frc_robot_old2.onnx` | move |
| `models/frc_robot_yolo11n.onnx` | `models/frc_robot_yolo11n.onnx` | move |
| `models/backup/` | `models/backup/` | move |

### 3.2 複製（scoring-analyzer → frc-train-review；兩邊各留一份）

| Source | Destination | 理由 |
|---|---|---|
| `models/frc_robot.onnx` | `models/frc_robot.onnx` | 新專案持有權威來源；scoring-analyzer 執行期仍需要一份 |

### 3.3 留在 scoring-analyzer 不動

- 核心 14 支 Python（`main.py`、`app.py`、`detection.py`、`tracking.py`、`robot_detection.py`、`robot_tracker.py`、`scoring.py`、`background.py`、`calibration.py`、`config.py`、`runtime_config.py`、`settings_window.py`、`geometry.py`、`utils.py`）
- `test_analysis.py`、`diagnose.py`（測試工具，import 核心）
- `models/frc_robot.onnx`（runtime 需要）
- `models/object_tracking_vittrack_*.onnx`（球追蹤 SOT 模型）
- `docs/notebook/ENGINEERING_NOTEBOOK.*`（4 個版本的 FRC 工程日誌 — md/html/txt/EN）
- `docs/superpowers/`、`docs/plans/`（設計文件）
- `PROGRESS.md`、`FINDINGS.md`、`CLAUDE.md`、`errors.md`、`README.md`、`requirements.txt`
- `presets/`、`videos/`、`scratch/`

### 3.4 scoring-analyzer 清理（搬走後）

**刪除（空目錄 + 遺留檔）：**
- `scripts/`（整個目錄）
- `datasets/`（已搬走）
- `train_colab.ipynb`
- `docs/TRAIN_README.txt`
- `models/frc_robot_old.onnx`、`frc_robot_old2.onnx`、`frc_robot_yolo11n.onnx`、`models/backup/`

**更新：**
- `CLAUDE.md`：移除 dataset Run Commands 段落、移除 `scripts/dataset/` 目錄樹區塊、移除 `datasets/` 目錄樹區塊、更新 footer session 備註
- `.gitignore`：可選擇移除 `datasets/`、`train_colab.ipynb` 條目（保留也無害，留著當防呆）

---

## 4. 新專案結構

```
D:\FRC\frc-train-review\
├── .git/                       # git init 獨立 repo
├── .gitignore                  # datasets/ models/*.onnx runs/ __pycache__ .env 等
├── README.md                   # 新寫，pipeline 4 階段說明 + 使用範例
├── requirements.txt            # 新寫（見第 5 節）
├── TRAIN_README.txt            # 從 scoring-analyzer/docs/ 搬來
│
├── auto_annotate.py            # 自動標註
├── batch_annotate.py
├── collect_to_notyet.py
├── label_editor.py             # 審核 GUI
├── sync_reviewed.py
├── detect_bad_frames.py
├── download_matches.py         # 下載
├── batch_download.py
├── extract_frames.py
├── crop_events.py
├── merge_datasets.py           # 資料集合併
├── build_dataset.py
├── train_robot_model.py        # 訓練主程式
├── train_colab.ipynb           # Colab 訓練 notebook
│
├── datasets/                   # 58 GB（gitignored）
│   ├── merged/                 # ← final 模型訓練集
│   ├── merged.zip
│   ├── reviewed/               # 審核完成
│   ├── notyet/                 # 待審核
│   ├── 2023mslr/, 2024*/       # 舊賽季素材
│   ├── 2026*/ × 14 賽事
│   ├── labels_raw_orphoned/    # 孤兒標註
│   ├── _preview/, _test_samples/
│   └── download_log.txt
│
└── models/                     # gitignored
    ├── frc_robot.onnx          # 權威來源（與 scoring-analyzer 同步）
    ├── frc_robot_old.onnx
    ├── frc_robot_old2.onnx
    ├── frc_robot_yolo11n.onnx
    └── backup/
```

---

## 5. 新專案依賴與設定檔

### 5.1 `requirements.txt`

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

### 5.2 `.gitignore`

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

# Colab（可選：若想保留 notebook 可移除此行）
# train_colab.ipynb

# IDE
.vscode/
.idea/

# 環境變數（含 GEMINI_API_KEY、TBA_API_KEY）
.env
```

### 5.3 `README.md`

最小內容：
- 專案定位：FRC 資料集 pipeline（下載 → 標註 → 審核 → 訓練）
- 4 階段工作流程 + 對應腳本清單
- Quick Start：`pip install -r requirements.txt` + 各階段範例指令
- 最終模型訓練流程：`merge_datasets.py` → `train_colab.ipynb`（Google Colab）
- 環境變數：`GEMINI_API_KEY`、`TBA_API_KEY`

---

## 6. 執行流程

### Step 0: 在 `feat/gemini-detector` 上 commit 堆積變更

**目的：** 搬家前清空 uncommitted state，讓「搬家」commit 是乾淨的一次操作。

現有 uncommitted 變更混雜了 session 28-30 的程式碼工作（MOT Hungarian 匹配 + 歸因簡化 + 其他 iteration）與 session 31 的檔案結構整理（腳本移到 `scripts/dataset/`、文件移到 `docs/`）。為避免試圖分開檔案歸屬時誤判，採用**單一 commit 策略**：

```bash
cd D:\FRC\scoring-analyzer

# 追蹤所有已修改、已刪除、新增的檔案
git add -A

# 一次 commit 涵蓋 session 28-31 的累積工作
git commit -m "feat: MOT Hungarian 匹配 + 歸因簡化 + 專案結構整理（session 28-31）

- robot_tracker.py: Round 1 改用 Hungarian 全域最優匹配
- scoring.py: 射手歸因從 140 輪迭代簡化為 HP 線段交叉 + 80 幀 proximity
- 專案結構：dataset 腳本移至 scripts/dataset/，文件移至 docs/
- CLAUDE.md、.gitignore 更新反映新結構"
```

**為何不拆 commit：** session 30 的 code 修改（`robot_tracker.py`、`scoring.py`）與早先 session 28/29 的其他修改（`app.py`、`config.py`、`presets/1.json`）難以精確區分，嘗試拆 commit 風險大於收益。一次完整 commit 之後的分離操作仍能清楚呈現在下一個 commit。

### Step 1: 切出新分支

```bash
git checkout -b chore/split-dataset-pipeline
```

### Step 2: 建立新專案骨架

```cmd
mkdir D:\FRC\frc-train-review
cd D:\FRC\frc-train-review
git init
```

接著寫 `.gitignore`、`requirements.txt`、`README.md`（未 commit，等檔案搬完一起進第一個 commit）。

### Step 3: 複製 `frc_robot.onnx`

```cmd
copy D:\FRC\scoring-analyzer\models\frc_robot.onnx D:\FRC\frc-train-review\models\frc_robot.onnx
```

此步驟不動 scoring-analyzer 的檔案，失敗可重試。

### Step 4: 搬家（robocopy /MOVE 同磁碟機 = 秒完成）

**批次 1: 腳本**
```cmd
for %f in (auto_annotate batch_annotate batch_download build_dataset collect_to_notyet crop_events detect_bad_frames download_matches extract_frames label_editor merge_datasets sync_reviewed train_robot_model) do move "D:\FRC\scoring-analyzer\scripts\dataset\%f.py" "D:\FRC\frc-train-review\%f.py"
```

**批次 2: 資料集（58 GB，robocopy /MOVE /E 秒完成）**
```cmd
robocopy "D:\FRC\scoring-analyzer\datasets" "D:\FRC\frc-train-review\datasets" /MOVE /E
```

**批次 3: 其他檔案**
```cmd
move D:\FRC\scoring-analyzer\train_colab.ipynb D:\FRC\frc-train-review\
move D:\FRC\scoring-analyzer\docs\TRAIN_README.txt D:\FRC\frc-train-review\
move D:\FRC\scoring-analyzer\models\frc_robot_old.onnx D:\FRC\frc-train-review\models\
move D:\FRC\scoring-analyzer\models\frc_robot_old2.onnx D:\FRC\frc-train-review\models\
move D:\FRC\scoring-analyzer\models\frc_robot_yolo11n.onnx D:\FRC\frc-train-review\models\
robocopy "D:\FRC\scoring-analyzer\models\backup" "D:\FRC\frc-train-review\models\backup" /MOVE /E
```

### Step 5: 新專案首個 commit

```bash
cd D:\FRC\frc-train-review
git add .gitignore requirements.txt README.md TRAIN_README.txt *.py train_colab.ipynb
# 注意：datasets/ 和 models/*.onnx 已被 .gitignore 排除，不會進 commit
git commit -m "initial: dataset pipeline split from scoring-analyzer"
```

### Step 6: scoring-analyzer 清理

```bash
cd D:\FRC\scoring-analyzer
rmdir /s scripts      # 應已經空
# datasets/ 應已被 robocopy 搬空
# train_colab.ipynb、docs/TRAIN_README.txt 應已被 move 搬走
# models/ 僅剩 frc_robot.onnx + object_tracking_vittrack_*.onnx
```

更新 `CLAUDE.md`：
- 移除 "Run Commands" 中的 dataset 相關段落
- 移除目錄樹中的 `scripts/dataset/`、`datasets/`、`models/` 的備份檔說明
- Footer 改為：`*Last updated: 2026-04-15 (session 32) — 分離 dataset pipeline 至 D:\FRC\frc-train-review*`

```bash
git add CLAUDE.md
git add -u   # 追蹤所有被刪除的檔案
git commit -m "chore: 分離 dataset pipeline 至 frc-train-review"
```

### Step 7: 雙邊驗證

**scoring-analyzer：**
```cmd
cd /d D:\FRC\scoring-analyzer
python -c "import app, main, detection, robot_tracker, scoring, robot_detection; print('core OK')"
dir models\frc_robot.onnx
python main.py --help
```

**frc-train-review：**
```cmd
cd /d D:\FRC\frc-train-review
python auto_annotate.py --help
python label_editor.py --help
python train_robot_model.py --help
dir datasets\merged\data.yaml
dir models\frc_robot.onnx
```

---

## 7. 風險與注意事項

### 7.1 58 GB 搬家不可逆

- `robocopy /MOVE` 完成後 scoring-analyzer 的 `datasets/` 會消失
- 必須確認目的地在 **D: 磁碟機**（同磁碟 = 秒完成的 rename；跨磁碟 = 真實 I/O copy + delete，可能數小時）
- 建議在 Step 4 的 robocopy 前手動 `dir D:\FRC\frc-train-review` 確認目錄已建立

### 7.2 `train_robot_model.py` 的 `PROJECT_ROOT` bug 順便修

- scoring-analyzer 的 `scripts/dataset/train_robot_model.py:30` `PROJECT_ROOT = Path(__file__).parent.resolve()` 解析到 `scripts/dataset/`，`MODELS_DIR` 會指向 `scripts/dataset/models/`（不存在）
- 搬到 flat 後 `Path(__file__).parent` = project root，自動修復
- **不需要額外修改**這個檔案

### 7.3 `batch_annotate.py` 的 subprocess 路徑

- 現有 `subprocess.run(["python", "auto_annotate.py"])` 倚賴 cwd = 腳本所在目錄
- 在新專案 flat layout 下從 project root 跑 `python batch_annotate.py` 時，cwd 與腳本位置一致，路徑能正確解析
- **不需要額外修改**這個檔案

### 7.4 `DATASET_ROOT = Path("datasets")` 相對路徑

- 多個腳本使用 `Path("datasets")`，依賴 cwd = project root
- 新專案 flat layout + 從 project root 執行 = 路徑正確
- 使用者若從其他目錄 `python D:\...\auto_annotate.py` 執行會失敗（和現況相同，不是新問題）

### 7.5 Git 歷史

- 新 repo 獨立 `git init`，**不保留** `scripts/dataset/*.py` 的歷史 commit
- 理由：這些腳本 2026-03 才進 scoring-analyzer，歷史很短（大多藏在 `27dcd8b chore: 新增資料集工具腳本` 一個 commit 裡），保留成本 > 價值
- scoring-analyzer 這邊會有 `git rm` 產生的 "delete" commit，日後 `git log -- scripts/dataset/auto_annotate.py` 仍可回溯到移除前的版本

### 7.6 分支策略

- 工作分支：從 `feat/gemini-detector`（含 session 28-31 所有 commit）切出 `chore/split-dataset-pipeline`
- 搬家相關的 commit 只在 `chore/split-dataset-pipeline`，避免污染 `feat/gemini-detector` 的語意
- 分離完成後，使用者決定：
  - (1) 直接把 `chore/split-dataset-pipeline` 併回 `feat/gemini-detector` 繼續 Gemini detector 工作
  - (2) 或 merge 到 `main`，再從 main 重新開 gemini detector 分支

### 7.7 環境變數

- `auto_annotate.py` 需要 `GEMINI_API_KEY`
- `download_matches.py` 需要 `TBA_API_KEY`
- 這些變數目前可能存在 scoring-analyzer 的 `.env` 或系統環境變數
- 新專案的 `README.md` 需說明這些 key 的設定方式；不搬 `.env` 檔（若有）

---

## 8. 驗收條件

分離完成視為成功，當且僅當以下全部通過：

1. ✅ `D:\FRC\frc-train-review\` 存在且為獨立 git repo（`git log` 顯示 1 個 initial commit）
2. ✅ 13 支腳本在 `frc-train-review/` root，皆可 `--help`
3. ✅ `frc-train-review/datasets/merged/data.yaml` 存在，`merged.zip` 存在
4. ✅ `frc-train-review/models/frc_robot.onnx` 存在（9.8 MB）
5. ✅ `scoring-analyzer/` 目錄中：`scripts/`、`datasets/`、`train_colab.ipynb`、`docs/TRAIN_README.txt`、`models/frc_robot_old*.onnx`、`models/frc_robot_yolo11n.onnx`、`models/backup/` 皆已不存在
6. ✅ `scoring-analyzer/models/frc_robot.onnx` 仍存在（runtime 用）
7. ✅ `scoring-analyzer/models/object_tracking_vittrack_*.onnx` 仍存在
8. ✅ `python -c "import app, main, detection, robot_tracker, scoring, robot_detection"` 在 scoring-analyzer 目錄內成功
9. ✅ `python main.py --help` 在 scoring-analyzer 目錄內成功
10. ✅ scoring-analyzer 的 `feat/gemini-detector` 分支有 1 個新 commit（session 28-31 累積工作）
11. ✅ scoring-analyzer 的 `chore/split-dataset-pipeline` 分支有 1 個新 commit（分離操作）

---

## 9. 後續（不屬本次範圍）

- 新專案的 `README.md` 正式內容（目前只寫最小占位）
- 新專案是否要 push 到 GitHub（使用者自行決定）
- scoring-analyzer 是否把 session 30 的 Hungarian 匹配做影片實測（獨立工作）
- `feat/gemini-detector` 分支的 Gemini detector 實作（搬完後回該分支繼續）

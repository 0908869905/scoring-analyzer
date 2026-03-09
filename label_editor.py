"""
YOLO Label Editor — interactive bbox annotation tool.

Usage:
    python label_editor.py <dataset_dir>
    # dataset_dir must contain images/ and labels/ subdirectories
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import customtkinter as ctk
from PIL import Image, ImageTk

# Constants
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
CLASS_NAMES = {0: "Red", 1: "Blue"}
CLASS_COLORS = {0: "#FF3333", 1: "#3388FF"}
HANDLE_SIZE = 6


class LabelEditor(ctk.CTk):
    def __init__(self, img_dir: Path, lbl_dir: Path):
        super().__init__()
        self.title("YOLO Label Editor")
        self.geometry("1400x800")
        ctk.set_appearance_mode("dark")

        self.img_dir = img_dir
        self.lbl_dir = lbl_dir

        # Image list (only images that have label files)
        all_imgs = sorted(p for p in img_dir.iterdir()
                          if p.suffix.lower() in IMAGE_EXTS)
        self.images = [p for p in all_imgs
                       if (lbl_dir / f"{p.stem}.txt").is_file()]
        if not self.images:
            print("[ERROR] No annotated images found.")
            sys.exit(1)

        self.idx = 0
        self.labels: list[list] = []
        self.selected = -1
        self.mode = "select"
        self.zoom = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self._photo: Optional[ImageTk.PhotoImage] = None
        self._pil_img: Optional[Image.Image] = None
        self._dirty = False

        self._build_toolbar()
        self._build_canvas()
        self._build_statusbar()
        self._load_current()

    def _build_toolbar(self):
        tb = ctk.CTkFrame(self, height=40)
        tb.pack(fill="x", padx=5, pady=(5, 0))

        self.btn_prev = ctk.CTkButton(tb, text="◀ Prev", width=80,
                                       command=self._prev)
        self.btn_prev.pack(side="left", padx=2)

        self.btn_next = ctk.CTkButton(tb, text="Next ▶", width=80,
                                       command=self._next)
        self.btn_next.pack(side="left", padx=2)

        self.btn_save = ctk.CTkButton(tb, text="Save", width=60,
                                       command=self._save_labels)
        self.btn_save.pack(side="left", padx=10)

        self.lbl_progress = ctk.CTkLabel(tb, text="0/0")
        self.lbl_progress.pack(side="left", padx=10)

        self.lbl_mode = ctk.CTkLabel(tb, text="Mode: Select",
                                      text_color="#88FF88")
        self.lbl_mode.pack(side="right", padx=10)

    def _build_canvas(self):
        self.canvas = ctk.CTkCanvas(self, bg="#1a1a1a",
                                     highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=5, pady=5)

    def _build_statusbar(self):
        sb = ctk.CTkFrame(self, height=30)
        sb.pack(fill="x", padx=5, pady=(0, 5))

        self.lbl_zoom = ctk.CTkLabel(sb, text="Zoom: 100%")
        self.lbl_zoom.pack(side="left", padx=10)

        self.lbl_file = ctk.CTkLabel(sb, text="")
        self.lbl_file.pack(side="right", padx=10)

    def _load_labels(self, path: Path) -> list[list]:
        if not path.is_file():
            return []
        out = []
        for line in path.read_text().strip().splitlines():
            p = line.split()
            if len(p) == 5:
                out.append([int(p[0]),
                            float(p[1]), float(p[2]),
                            float(p[3]), float(p[4])])
        return out

    def _save_labels(self):
        path = self.lbl_dir / f"{self.images[self.idx].stem}.txt"
        with open(path, "w") as f:
            for c, cx, cy, bw, bh in self.labels:
                f.write(f"{c} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
        self._dirty = False

    def _load_current(self):
        ip = self.images[self.idx]
        self._pil_img = Image.open(ip)
        self.labels = self._load_labels(self.lbl_dir / f"{ip.stem}.txt")
        self.selected = -1
        self._dirty = False
        self._fit_zoom()
        self._redraw()
        self._update_ui()

    def _fit_zoom(self):
        self.update_idletasks()
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 10 or ch < 10:
            cw, ch = 1380, 700
        iw, ih = self._pil_img.size
        self.zoom = min(cw / iw, ch / ih, 3.0)
        self.pan_x = (cw - iw * self.zoom) / 2
        self.pan_y = (ch - ih * self.zoom) / 2

    def _img_to_canvas(self, ix: float, iy: float) -> tuple[float, float]:
        return ix * self.zoom + self.pan_x, iy * self.zoom + self.pan_y

    def _canvas_to_img(self, cx: float, cy: float) -> tuple[float, float]:
        return (cx - self.pan_x) / self.zoom, (cy - self.pan_y) / self.zoom

    def _yolo_to_pixel(self, cx, cy, bw, bh):
        iw, ih = self._pil_img.size
        x1 = (cx - bw / 2) * iw
        y1 = (cy - bh / 2) * ih
        x2 = (cx + bw / 2) * iw
        y2 = (cy + bh / 2) * ih
        return x1, y1, x2, y2

    def _pixel_to_yolo(self, x1, y1, x2, y2):
        iw, ih = self._pil_img.size
        cx = (x1 + x2) / 2 / iw
        cy = (y1 + y2) / 2 / ih
        bw = abs(x2 - x1) / iw
        bh = abs(y2 - y1) / ih
        return cx, cy, bw, bh

    def _redraw(self):
        self.canvas.delete("all")
        if self._pil_img is None:
            return
        iw, ih = self._pil_img.size
        disp_w = max(1, int(iw * self.zoom))
        disp_h = max(1, int(ih * self.zoom))
        resized = self._pil_img.resize((disp_w, disp_h), Image.LANCZOS)
        self._photo = ImageTk.PhotoImage(resized)
        self.canvas.create_image(self.pan_x, self.pan_y,
                                  anchor="nw", image=self._photo,
                                  tags="image")
        for i, (cls, cx, cy, bw, bh) in enumerate(self.labels):
            self._draw_bbox(i, cls, cx, cy, bw, bh)

    def _draw_bbox(self, idx: int, cls: int, cx, cy, bw, bh):
        x1, y1, x2, y2 = self._yolo_to_pixel(cx, cy, bw, bh)
        cx1, cy1 = self._img_to_canvas(x1, y1)
        cx2, cy2 = self._img_to_canvas(x2, y2)
        color = CLASS_COLORS.get(cls, "#00FF00")
        is_sel = (idx == self.selected)
        width = 3 if is_sel else 2
        self.canvas.create_rectangle(
            cx1, cy1, cx2, cy2,
            outline=color, width=width,
            tags=f"bbox_{idx}")
        label = f"{idx + 1}:{CLASS_NAMES.get(cls, '?')}"
        self.canvas.create_text(
            cx1 + 3, cy1 - 2,
            anchor="sw", text=label, fill=color,
            font=("Consolas", 11, "bold"),
            tags=f"label_{idx}")
        if is_sel:
            hs = HANDLE_SIZE
            handles = [
                (cx1, cy1), ((cx1+cx2)/2, cy1), (cx2, cy1),
                (cx1, (cy1+cy2)/2),              (cx2, (cy1+cy2)/2),
                (cx1, cy2), ((cx1+cx2)/2, cy2), (cx2, cy2),
            ]
            for hx, hy in handles:
                self.canvas.create_rectangle(
                    hx - hs, hy - hs, hx + hs, hy + hs,
                    fill="white", outline=color, width=1,
                    tags="handle")

    def _update_ui(self):
        total = len(self.images)
        self.lbl_progress.configure(text=f"{self.idx + 1}/{total}")
        self.lbl_zoom.configure(text=f"Zoom: {int(self.zoom * 100)}%")
        self.lbl_file.configure(text=self.images[self.idx].name)
        nr = sum(1 for l in self.labels if l[0] == 0)
        nb = sum(1 for l in self.labels if l[0] == 1)
        mode_text = f"Mode: {'Draw' if self.mode == 'draw' else 'Select'}"
        mode_text += f"  |  Red:{nr}  Blue:{nb}"
        self.lbl_mode.configure(
            text=mode_text,
            text_color="#FF8888" if self.mode == "draw" else "#88FF88")

    def _auto_save(self):
        if self._dirty:
            self._save_labels()

    def _prev(self):
        self._auto_save()
        self.idx = (self.idx - 1) % len(self.images)
        self._load_current()

    def _next(self):
        self._auto_save()
        self.idx = (self.idx + 1) % len(self.images)
        self._load_current()


def main():
    parser = argparse.ArgumentParser(description="YOLO Label Editor")
    parser.add_argument("dataset", type=Path,
                        help="Dataset directory (must contain images/ and labels/)")
    args = parser.parse_args()

    img_dir = args.dataset / "images"
    lbl_dir = args.dataset / "labels"

    if not img_dir.is_dir():
        print(f"[ERROR] images/ not found in {args.dataset}")
        sys.exit(1)
    if not lbl_dir.is_dir():
        print(f"[ERROR] labels/ not found in {args.dataset}")
        sys.exit(1)

    app = LabelEditor(img_dir, lbl_dir)
    app.mainloop()


if __name__ == "__main__":
    main()

# Label Editor Design

## Overview
Standalone annotation review/correction tool for YOLO bounding box labels.
- **Tech**: CustomTkinter + Canvas
- **Usage**: `python label_editor.py datasets/2024mslr/`
- **Data**: YOLO format labels (class_id cx cy w h, normalized), 1278x534 images, 2 classes (Red=0, Blue=1)

## UI Layout
```
┌──────────────────────────────────────────┐
│ [Prev] [Next] [Save]  │  15/1130        │  Toolbar
├──────────────────────────────────────────┤
│                                          │
│        Canvas (image + bbox rectangles)  │
│                                          │
├──────────────────────────────────────────┤
│ Zoom: 100% │ Mode: Select │ filename    │  Status bar
└──────────────────────────────────────────┘
```

## Features
1. **Adjust bbox size/position** — click to select, drag to move, drag handles to resize
2. **Delete unwanted boxes** — select + Delete key
3. **Zoom in/out** — mouse wheel zoom, F to fit window
4. **Add new bbox** — D to enter draw mode, drag to create
5. **Toggle class** — Tab to switch Red/Blue on selected bbox
6. **Navigation** — Left/Right arrow keys, progress display (reviewed/total)

## Interaction Modes
| Mode | Activation | Behavior |
|------|-----------|----------|
| Select (default) | Esc | Click bbox to select, drag to move, drag corners to resize |
| Draw | D key | Click+drag to create new bbox (default Red class) |

## Key Bindings
| Key | Action |
|-----|--------|
| Left/Right | Prev/Next image |
| Delete | Delete selected bbox |
| Tab | Toggle class Red<->Blue |
| D | Enter draw mode |
| Esc | Back to select mode |
| Mouse wheel | Zoom in/out |
| F | Fit to window |
| Ctrl+S | Save |

## Bbox Rendering
- Red (class 0): red outline
- Blue (class 1): blue outline
- Selected: thick outline + 8 resize handles
- Auto-save on image navigation

## Data Flow
```
images/*.jpg + labels/*.txt (YOLO normalized)
    -> Load into memory
    -> Canvas display + interactive editing
    -> Save back to labels/*.txt (overwrite)
```

## Technical Notes
- Canvas-native rectangles for bbox (built-in hit testing, drag support)
- PIL for image loading, displayed as Canvas image item
- Zoom via Canvas scale transform + image re-render
- Coordinate mapping: canvas coords <-> image coords <-> YOLO normalized

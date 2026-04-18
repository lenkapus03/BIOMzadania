"""
detection/annotator.py - Play a video with ground-truth annotations overlaid.

Draws:
  - Bounding box  (4 corner points from boundingBox)
  - 68 facial landmarks (from landmarks2D)

Controls during playback:
  SPACE          - pause / resume
  [< PREV] btn   - previous video
  [NEXT >] btn   - next video
  + / -          - increase / decrease display scale
  Q / ESC        - quit
"""

import cv2
import numpy as np

# ----- visual style -----
BOX_COLOR      = (0, 255, 0)      # green
BOX_THICKNESS  = 1
LM_COLOR       = (0, 0, 255)      # red
LM_RADIUS      = 1
TEXT_COLOR     = (255, 255, 255)  # white
BTN_COLOR      = (60, 60, 60)     # dark grey button background
BTN_HL_COLOR   = (100, 100, 100)  # highlighted button
FONT           = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE     = 0.35
FONT_THICKNESS = 1
LINE_HEIGHT    = 14               # px per text line
PAD_V          = 4                # vertical padding inside bars

FPS            = 25
SCALE_STEP     = 0.25
SCALE_MIN      = 0.25
SCALE_MAX      = 4.0
SCALE_DEFAULT  = 1.0

WINDOW_NAME    = "Annotated video"

# shared state for mouse callback
_mouse = {"x": 0, "y": 0, "click": False}


def _mouse_cb(event, x, y, flags, param):
    _mouse["x"] = x
    _mouse["y"] = y
    if event == cv2.EVENT_LBUTTONDOWN:
        _mouse["click"] = True


def _wrap_text(text: str, max_width: int) -> list[str]:
    """Split text into lines that each fit within max_width pixels."""
    words  = text.split()
    lines  = []
    current = ""
    for word in words:
        test = (current + " " + word).strip()
        (w, _), _ = cv2.getTextSize(test, FONT, FONT_SCALE, FONT_THICKNESS)
        if w <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [text]


def _make_text_bar(text: str, width: int) -> np.ndarray:
    """Render a text bar where each \\n becomes a new line."""
    lines  = text.split("\n")
    n      = len(lines)
    bar_h  = PAD_V + n * LINE_HEIGHT + PAD_V
    bar    = np.zeros((bar_h, width, 3), dtype=np.uint8)
    for idx, line in enumerate(lines):
        y = PAD_V + (idx + 1) * LINE_HEIGHT - 2
        cv2.putText(bar, line, (6, y), FONT, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS)
    return bar


def _make_bottom_bar(text: str, width: int) -> tuple[np.ndarray, dict]:
    """
    Render bottom bar: text lines on top, PREV/NEXT buttons on their own line at the bottom.
    Returns the bar image and a dict with button rects: {"prev": (x1,y1,x2,y2), "next": ...}
    """
    BTN_W, BTN_H = 60, 18
    BTN_MARGIN   = 4

    lines  = text.split("\n")
    n      = len(lines)
    bar_h  = PAD_V + n * LINE_HEIGHT + BTN_MARGIN + BTN_H + PAD_V
    bar    = np.zeros((bar_h, width, 3), dtype=np.uint8)

    # text lines
    for idx, line in enumerate(lines):
        y = PAD_V + (idx + 1) * LINE_HEIGHT - 2
        cv2.putText(bar, line, (6, y), FONT, FONT_SCALE, TEXT_COLOR, FONT_THICKNESS)

    # buttons on their own line at the bottom
    btn_y1 = bar_h - PAD_V - BTN_H
    btn_y2 = btn_y1 + BTN_H

    prev_x1 = BTN_MARGIN
    prev_x2 = prev_x1 + BTN_W
    next_x1 = prev_x2 + BTN_MARGIN
    next_x2 = next_x1 + BTN_W

    cv2.rectangle(bar, (prev_x1, btn_y1), (prev_x2, btn_y2), BTN_COLOR, -1)
    cv2.rectangle(bar, (next_x1, btn_y1), (next_x2, btn_y2), BTN_COLOR, -1)

    cv2.putText(bar, "< PREV", (prev_x1 + 4, btn_y2 - 5), FONT, 0.32, TEXT_COLOR, 1)
    cv2.putText(bar, "NEXT >", (next_x1 + 4, btn_y2 - 5), FONT, 0.32, TEXT_COLOR, 1)

    btns = {
        "prev": (prev_x1, btn_y1, prev_x2, btn_y2),
        "next": (next_x1, btn_y1, next_x2, btn_y2),
    }
    return bar, btns


def _in_rect(x, y, rect, y_offset: int) -> bool:
    x1, y1, x2, y2 = rect
    return x1 <= x <= x2 and (y1 + y_offset) <= y <= (y2 + y_offset)


def draw_frame(frame: np.ndarray, box: np.ndarray, landmarks: np.ndarray,
               top_text: str, bottom_text: str, scale: float):
    """
    Annotate frame, resize, attach text bars.
    Returns (composite_image, btn_rects, top_bar_height).
    """
    img = frame.copy()

    # bounding box
    x1, y1 = box[:, 0].min(), box[:, 1].min()
    x2, y2 = box[:, 0].max(), box[:, 1].max()
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), BOX_COLOR, BOX_THICKNESS)

    # landmarks
    for (x, y) in landmarks:
        cv2.circle(img, (int(x), int(y)), LM_RADIUS, LM_COLOR, -1)

    # resize
    if scale != 1.0:
        H, W = img.shape[:2]
        img = cv2.resize(img, (int(W * scale), int(H * scale)),
                         interpolation=cv2.INTER_LINEAR)

    W = img.shape[1]
    top_bar              = _make_text_bar(top_text, W)
    bottom_bar, btn_rects = _make_bottom_bar(bottom_text, W)

    composite = np.vstack([top_bar, img, bottom_bar])
    return composite, btn_rects, top_bar.shape[0]


def play_video(video_data, video_index: int, total: int) -> str:
    """
    Play one video clip with annotations.
    Returns "next", "prev", or "quit".
    """
    frames    = video_data.frames
    boxes     = video_data.boxes
    landmarks = video_data.landmarks
    n         = len(frames)
    delay     = int(1000 / FPS)
    paused    = False
    scale     = SCALE_DEFAULT

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, _mouse_cb)

    print(f"  [{video_index + 1}/{total}] '{video_data.name}'  ({n} frames)")

    i = 0
    while True:
        top_text    = f"[{video_index+1}/{total}] {video_data.name}\nframe {i+1}/{n}\nscale {scale:.2f}x"
        bottom_text = "SPACE=pause\n+/-=scale\nQ=quit"

        composite, btn_rects, top_h = draw_frame(
            frames[i], boxes[i], landmarks[i], top_text, bottom_text, scale
        )

        # bottom bar starts after top_bar + image
        bottom_y0  = top_h + int(frames[i].shape[0] * scale)

        cv2.imshow(WINDOW_NAME, composite)
        h, w = composite.shape[:2]
        cv2.resizeWindow(WINDOW_NAME, w, h)
        key = cv2.waitKey(1 if paused else delay) & 0xFF

        # --- mouse click on buttons ---
        if _mouse["click"]:
            _mouse["click"] = False
            mx, my = _mouse["x"], _mouse["y"]
            if _in_rect(mx, my, btn_rects["prev"], bottom_y0):
                return "prev"
            if _in_rect(mx, my, btn_rects["next"], bottom_y0):
                return "next"

        # --- keyboard ---
        if key in (ord("q"), 27):
            return "quit"
        elif key == ord(" "):
            paused = not paused
        elif key in (ord("+"), ord("=")):
            scale = min(scale + SCALE_STEP, SCALE_MAX)
        elif key == ord("-"):
            scale = max(scale - SCALE_STEP, SCALE_MIN)
        elif not paused:
            i = (i + 1) % n
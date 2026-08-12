"""Crop a screenshot by framing it with your hands in front of the webcam:
pinch (touch thumb tip to index fingertip) on BOTH hands at once to start,
move your hands to size/reposition the box live while staying pinched, then
release either pinch to capture the crop as it was the instant before you
let go. The window stays open after a capture -- release again to grab
another region from the same screenshot -- and closes on Esc, on making a
closed fist with either hand (held briefly to confirm), or after hands are
absent for a while.

Hand tracking is MediaPipe Hands (bundled landmark model, CPU-only, no
network calls). The live view is a plain OpenCV window (cv2.imshow) rather
than a transparent full-screen overlay -- this venv's Python has no Tk
support, so the HUD look (chrome frame, vignette, neon bloom, status bar)
is all drawn directly onto the frame each loop rather than composited as
separate transparent layers.

This is a press-drag-release interaction, like a mouse-drag selection:
double-pinch = mouse down, moving hands while pinched = drag, releasing
either pinch = mouse up / capture. A short RELEASE_CONFIRM_SECONDS debounce
guards against one noisy tracking frame reading as a release mid-drag.

This blocks the calling thread until you release a valid box or cancel
(Esc, or no hands visible for a while) -- it's meant to be called from a
voice command that's expected to wait for you at the keyboard/webcam, not
from a background monitor.
"""

import math
import subprocess
import tempfile
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

_mp_hands = mp.solutions.hands

WINDOW_NAME = "JARVIS VISION - pinch both hands, move to size, release to capture (fist or Esc to exit)"
THUMB_TIP, INDEX_TIP, WRIST, MIDDLE_MCP = 4, 8, 0, 9
# Index is deliberately excluded from the fist check: pinching curls it
# toward the thumb, which reads a lot like "curled into the fist" and caused
# false fist-exits while pinching. Middle/ring/pinky aren't involved in the
# pinch gesture at all, so they're a clean signal on their own.
FINGER_TIPS = (12, 16, 20)  # middle, ring, pinky
FINGER_PIPS = (10, 14, 18)  # their middle knuckles

# Pinch distance is measured as a ratio of hand size (wrist-to-middle-
# knuckle span) rather than raw pixels, so it doesn't depend on how close
# your hand is to the camera. Lower = fingers must touch more precisely.
PINCH_RATIO_THRESHOLD = 0.45
RELEASE_CONFIRM_SECONDS = 0.25  # both-pinch must actually drop this long (debounces tracking noise)
MIN_BOX_PX = 30  # ignore a box if the two pinch points are basically on top of each other
FIST_CONFIRM_SECONDS = 0.35  # either hand held as a fist this long exits the whole session
NO_HANDS_TIMEOUT_SECONDS = 8.0
DISPLAY_MAX_DIM = 1400

BAR_HEIGHT = 42
CHROME_BRACKET_LEN = 26
BLOOM_SIGMA = 7

# Neon HUD palette, BGR (OpenCV's channel order).
CYAN = (255, 255, 0)
DIM_CYAN = (130, 110, 20)
MAGENTA = (255, 0, 255)
LOCK_GREEN = (100, 255, 120)
WHITE = (255, 255, 255)
AMBER = (0, 170, 255)
FIST_RED = (50, 50, 255)


def _capture_screenshot_path() -> Path:
    path = Path(tempfile.mktemp(suffix=".png"))
    subprocess.run(["screencapture", "-x", str(path)], check=True)
    return path


def _copy_png_to_clipboard(path: Path) -> None:
    # `read ... as «class PNGf»` puts raw image bytes on the clipboard (not
    # just a file reference), so a paste elsewhere inserts the actual image.
    script = f'set the clipboard to (read (POSIX file "{path}") as «class PNGf»)'
    subprocess.run(["osascript", "-e", script], check=False)


def _lerp_color(c1: tuple, c2: tuple, t: float) -> tuple:
    t = max(0.0, min(1.0, t))
    return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))


def _hand_info(hand_landmarks, disp_w: int, disp_h: int):
    """Returns (thumb_pt, index_pt, pinch_point, pinch_ratio, wrist_pt) in display px."""
    lm = hand_landmarks.landmark
    thumb = (lm[THUMB_TIP].x * disp_w, lm[THUMB_TIP].y * disp_h)
    index = (lm[INDEX_TIP].x * disp_w, lm[INDEX_TIP].y * disp_h)
    wrist = (lm[WRIST].x * disp_w, lm[WRIST].y * disp_h)
    middle_mcp = (lm[MIDDLE_MCP].x * disp_w, lm[MIDDLE_MCP].y * disp_h)
    hand_scale = math.hypot(wrist[0] - middle_mcp[0], wrist[1] - middle_mcp[1]) or 1.0
    pinch_dist = math.hypot(thumb[0] - index[0], thumb[1] - index[1])
    pinch_ratio = pinch_dist / hand_scale
    pinch_point = ((thumb[0] + index[0]) / 2, (thumb[1] + index[1]) / 2)
    return thumb, index, pinch_point, pinch_ratio, wrist


def _is_fist(hand_landmarks) -> bool:
    """A hand is a fist if ALL of middle/ring/pinky have curled in closer to
    the wrist than their own middle knuckle -- a ratio-free comparison (tip
    vs. pip, both measured from the same wrist), so it works regardless of
    hand size or distance from the camera. Thumb and index are excluded:
    thumb doesn't fold the same way, and index is claimed by the pinch
    gesture (curls toward the thumb even when not making a fist)."""
    lm = hand_landmarks.landmark
    wrist = lm[WRIST]
    for tip_idx, pip_idx in zip(FINGER_TIPS, FINGER_PIPS):
        tip_dist = math.hypot(lm[tip_idx].x - wrist.x, lm[tip_idx].y - wrist.y)
        pip_dist = math.hypot(lm[pip_idx].x - wrist.x, lm[pip_idx].y - wrist.y)
        if tip_dist >= pip_dist:
            return False
    return True


def _make_vignette(w: int, h: int) -> np.ndarray:
    """A soft radial darkening toward the edges, computed once per session
    (fixed size) rather than per frame -- gives the feed a lens/HUD look
    instead of a flat webcam rectangle."""
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot(xx - w / 2, yy - h / 2) / math.hypot(w / 2, h / 2)
    mask = np.clip(1.0 - 0.45 * (r ** 2), 0.55, 1.0).astype(np.float32)
    return mask[..., None]


def _draw_scanlines(canvas: np.ndarray) -> None:
    canvas[::3] = (canvas[::3] * 0.85).astype(np.uint8)


def _draw_chrome(canvas: np.ndarray, w: int, h: int) -> None:
    """Static framing chrome: an outer viewport border and corner brackets,
    independent of anything being tracked -- what makes the window read as
    a HUD rather than a bare camera preview."""
    cv2.rectangle(canvas, (2, 2), (w - 3, h - 3), DIM_CYAN, 1, cv2.LINE_AA)
    for cx, cy, dx, dy in ((0, 0, 1, 1), (w, 0, -1, 1), (0, h, 1, -1), (w, h, -1, -1)):
        cv2.line(canvas, (cx, cy), (cx + dx * CHROME_BRACKET_LEN, cy), CYAN, 2, cv2.LINE_AA)
        cv2.line(canvas, (cx, cy), (cx, cy + dy * CHROME_BRACKET_LEN), CYAN, 2, cv2.LINE_AA)
    _draw_hud_text(canvas, "JARVIS VISION", (14, 22), DIM_CYAN, scale=0.5)


def _draw_status_bar(canvas: np.ndarray, text: str, color: tuple) -> None:
    h, w = canvas.shape[:2]
    region = canvas[h - BAR_HEIGHT:h, 0:w]
    canvas[h - BAR_HEIGHT:h, 0:w] = cv2.addWeighted(region, 0.35, np.zeros_like(region), 0.65, 0)
    cv2.line(canvas, (0, h - BAR_HEIGHT), (w, h - BAR_HEIGHT), color, 1, cv2.LINE_AA)
    _draw_hud_text(canvas, text, (14, h - 15), color, scale=0.58)


def _draw_hand_hud(canvas, glow, thumb, index, wrist, pinch_ratio: float, is_pinching: bool, is_fist: bool) -> None:
    to_i = lambda p: (int(p[0]), int(p[1]))
    if is_fist:
        for layer, thickness in ((canvas, 2), (glow, 5)):
            cv2.circle(layer, to_i(wrist), 30, FIST_RED, thickness, cv2.LINE_AA)
        return
    t = min(pinch_ratio / PINCH_RATIO_THRESHOLD, 1.0)
    color = LOCK_GREEN if is_pinching else _lerp_color(MAGENTA, CYAN, t)
    for layer, thickness, radius in ((canvas, 2, 9), (glow, 4, 11)):
        cv2.line(layer, to_i(thumb), to_i(index), color, thickness, cv2.LINE_AA)
        cv2.circle(layer, to_i(thumb), radius, color, thickness, cv2.LINE_AA)
        cv2.circle(layer, to_i(index), radius, color, thickness, cv2.LINE_AA)
    if is_pinching:
        mid = to_i(((thumb[0] + index[0]) / 2, (thumb[1] + index[1]) / 2))
        for layer, thickness in ((canvas, 2), (glow, 4)):
            cv2.circle(layer, mid, 16, color, thickness, cv2.LINE_AA)
        cv2.circle(canvas, mid, 3, color, -1, cv2.LINE_AA)


def _draw_corner_brackets(canvas, glow, box, color, length: int = 34, thickness: int = 3) -> None:
    x1, y1, x2, y2 = box
    for layer, th in ((canvas, thickness), (glow, thickness + 2)):
        for cx, cy, dx, dy in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
            cv2.line(layer, (cx, cy), (cx + dx * length, cy), color, th, cv2.LINE_AA)
            cv2.line(layer, (cx, cy), (cx, cy + dy * length), color, th, cv2.LINE_AA)


def _draw_dragging_box(canvas, glow, box) -> tuple:
    """Draws the 'still being framed' look (spotlight + grid + green
    brackets) and returns its status-bar text/color. Shared by the active
    drag and by the brief grace window right after a pinch blip, so a single
    noisy tracking frame doesn't visibly flash to a different look."""
    dimmed = (canvas * 0.32).astype(np.uint8)
    dimmed[box[1]:box[3], box[0]:box[2]] = canvas[box[1]:box[3], box[0]:box[2]]
    canvas[:] = dimmed
    _draw_thirds_grid(canvas, box, DIM_CYAN)
    cv2.rectangle(canvas, (box[0], box[1]), (box[2], box[3]), LOCK_GREEN, 1, cv2.LINE_AA)
    _draw_corner_brackets(canvas, glow, box, LOCK_GREEN)
    box_w, box_h = box[2] - box[0], box[3] - box[1]
    return f"RELEASE TO CAPTURE  ·  {box_w}x{box_h}px", LOCK_GREEN


def _draw_thirds_grid(canvas, box, color) -> None:
    x1, y1, x2, y2 = box
    for i in (1, 2):
        x = x1 + (x2 - x1) * i // 3
        y = y1 + (y2 - y1) * i // 3
        cv2.line(canvas, (x, y1), (x, y2), color, 1, cv2.LINE_AA)
        cv2.line(canvas, (x1, y), (x2, y), color, 1, cv2.LINE_AA)


def _draw_hud_text(canvas, text: str, org: tuple, color: tuple, scale: float = 0.65) -> None:
    # Cheap glow: a soft dark halo behind, then the crisp neon line on top.
    cv2.putText(canvas, text, org, cv2.FONT_HERSHEY_DUPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(canvas, text, org, cv2.FONT_HERSHEY_DUPLEX, scale, color, 1, cv2.LINE_AA)


def _apply_bloom(canvas: np.ndarray, glow: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(glow, (0, 0), sigmaX=BLOOM_SIGMA, sigmaY=BLOOM_SIGMA)
    return cv2.add(canvas, blurred)


def crop_screenshot_with_gesture() -> str:
    """Opens a fresh full-screen screenshot and a webcam HUD window. Pinch
    (touch thumb to index) on BOTH hands at once to start framing, move your
    hands to size/reposition the box while staying pinched, then release
    either pinch to capture the crop as it was right before you let go --
    each release copies that crop to the clipboard (overwriting the
    previous one) without closing the window, so you can immediately frame
    and capture another region from the same screenshot. The window and
    webcam close when you press Esc, make a closed fist with either hand, or
    leave both hands out of frame for a while. Returns a summary of what was
    captured."""
    shot_path = _capture_screenshot_path()
    try:
        image = Image.open(shot_path).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        shot_path.unlink(missing_ok=True)
        return f"error: could not read the screenshot: {exc}"
    img_w, img_h = image.size

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        shot_path.unlink(missing_ok=True)
        return "error: could not open the webcam."

    scale = min(1.0, DISPLAY_MAX_DIM / max(img_w, img_h))
    disp_w, disp_h = max(1, int(img_w * scale)), max(1, int(img_h * scale))
    base_display = cv2.resize(cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR), (disp_w, disp_h))
    vignette = _make_vignette(disp_w, disp_h)

    was_dragging = False  # both hands were pinched as of the last frame
    last_box = None  # most recent valid box seen while dragging
    release_since = None  # when the pinch first dropped, mid possible-release
    fist_since = None  # when a fist was first seen, mid possible-exit
    last_hands_seen = time.time()
    crop_count = 0
    last_crop_size = None
    result_text = "Cancelled: no crop confirmed."

    try:
        with _mp_hands.Hands(
            max_num_hands=2, min_detection_confidence=0.6, min_tracking_confidence=0.5
        ) as hands:
            while True:
                ok, frame = cap.read()
                if not ok:
                    result_text = "error: lost the webcam feed."
                    break
                frame = cv2.flip(frame, 1)  # mirror so hand movement feels natural
                found = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                canvas = (base_display.astype(np.float32) * vignette).astype(np.uint8)
                _draw_scanlines(canvas)
                glow = np.zeros_like(canvas)

                hands_info = []
                if found.multi_hand_landmarks:
                    last_hands_seen = time.time()
                    for hand_landmarks in found.multi_hand_landmarks[:2]:
                        thumb, index, pinch_point, pinch_ratio, wrist = _hand_info(hand_landmarks, disp_w, disp_h)
                        # Raw, un-smoothed ratio: react to a slight release
                        # immediately. Debounce (below) -- not smoothing --
                        # is what keeps a single noisy frame from flashing
                        # the UI or capturing prematurely.
                        is_pinching = pinch_ratio < PINCH_RATIO_THRESHOLD
                        # A hand can't be pinching and fisted at once; if it
                        # reads as pinching, trust that over the fist check.
                        is_fist = (not is_pinching) and _is_fist(hand_landmarks)
                        hands_info.append((thumb, index, pinch_point, is_pinching, is_fist))
                        _draw_hand_hud(canvas, glow, thumb, index, wrist, pinch_ratio, is_pinching, is_fist)

                any_fist = any(h[4] for h in hands_info)
                status_text, status_color = None, CYAN
                exit_now = False

                if any_fist:
                    # A closed fist on either hand exits the whole session,
                    # regardless of drag state -- held briefly so a single
                    # noisy tracking frame can't slam the window shut.
                    fist_since = fist_since or time.time()
                    held = time.time() - fist_since
                    status_text, status_color = (
                        f"FIST DETECTED - HOLD TO EXIT ({max(0.0, FIST_CONFIRM_SECONDS - held):.1f}s)",
                        FIST_RED,
                    )
                    exit_now = held >= FIST_CONFIRM_SECONDS
                else:
                    fist_since = None
                    both_pinching = len(hands_info) == 2 and all(h[3] for h in hands_info)
                    box = None
                    if both_pinching:
                        p1, p2 = hands_info[0][2], hands_info[1][2]
                        candidate = (int(min(p1[0], p2[0])), int(min(p1[1], p2[1])),
                                     int(max(p1[0], p2[0])), int(max(p1[1], p2[1])))
                        if candidate[2] - candidate[0] >= MIN_BOX_PX and candidate[3] - candidate[1] >= MIN_BOX_PX:
                            box = candidate

                    if box:
                        # Currently pinched with a valid box: keep dragging, and
                        # remember this as the box to use if released next frame.
                        release_since = None
                        was_dragging = True
                        last_box = box
                        status_text, status_color = _draw_dragging_box(canvas, glow, box)
                    elif was_dragging:
                        # Pinch just dropped mid-drag: debounce briefly in case
                        # it's one noisy tracking frame, not a real release. Keep
                        # showing the normal dragging look during the first half
                        # of that debounce so a single blip doesn't flash the UI.
                        release_since = release_since or time.time()
                        held_release = time.time() - release_since
                        if last_box and held_release < RELEASE_CONFIRM_SECONDS * 0.5:
                            status_text, status_color = _draw_dragging_box(canvas, glow, last_box)
                        elif last_box:
                            _draw_corner_brackets(canvas, glow, last_box, WHITE)
                            status_text, status_color = "RELEASING...", WHITE
                        if held_release >= RELEASE_CONFIRM_SECONDS:
                            crop_box_orig = tuple(int(v / scale) for v in last_box)
                            cropped = image.crop(crop_box_orig)
                            out_path = shot_path.with_name(shot_path.stem + f"_crop{crop_count}.png")
                            cropped.save(out_path)
                            _copy_png_to_clipboard(out_path)
                            out_path.unlink(missing_ok=True)
                            crop_count += 1
                            last_crop_size = cropped.size
                            flash = cv2.add(canvas, np.full_like(canvas, 200))
                            cv2.imshow(WINDOW_NAME, flash)
                            cv2.waitKey(90)
                            # Stay open: reset drag state instead of exiting, so
                            # another region can be framed from the same shot.
                            was_dragging = False
                            last_box = None
                            release_since = None
                    else:
                        if crop_count:
                            status_text = (
                                f"COPIED {last_crop_size[0]}x{last_crop_size[1]}px "
                                f"({crop_count} so far)  ·  pinch again, or Esc to finish"
                            )
                            status_color = LOCK_GREEN
                        elif len(hands_info) == 2:
                            status_text, status_color = "PINCH BOTH HANDS TO START", CYAN
                        elif len(hands_info) == 1:
                            status_text, status_color = "SHOW YOUR OTHER HAND TOO", AMBER
                        else:
                            status_text, status_color = "SHOW BOTH HANDS - THUMB + INDEX", AMBER
                            if time.time() - last_hands_seen > NO_HANDS_TIMEOUT_SECONDS:
                                result_text = "Cancelled: no hands detected in time."
                                break

                canvas = _apply_bloom(canvas, glow)
                _draw_chrome(canvas, disp_w, disp_h)
                if status_text:
                    _draw_status_bar(canvas, status_text, status_color)

                cv2.imshow(WINDOW_NAME, canvas)

                if exit_now:
                    result_text = (
                        f"Closed by fist gesture after {crop_count} crop(s); most recent copied to "
                        f"the clipboard was {last_crop_size[0]}x{last_crop_size[1]}."
                        if crop_count else "Closed by fist gesture: no crop confirmed."
                    )
                    cv2.waitKey(150)
                    break

                if (cv2.waitKey(1) & 0xFF) == 27:  # Esc
                    result_text = (
                        f"Closed after {crop_count} crop(s); most recent copied to the "
                        f"clipboard was {last_crop_size[0]}x{last_crop_size[1]}."
                        if crop_count else "Cancelled by Esc: no crop confirmed."
                    )
                    break
    finally:
        cap.release()
        try:
            cv2.destroyWindow(WINDOW_NAME)
        except cv2.error:
            pass
        shot_path.unlink(missing_ok=True)

    return result_text


ACTIONS = {
    "crop_screenshot_with_gesture": crop_screenshot_with_gesture,
}

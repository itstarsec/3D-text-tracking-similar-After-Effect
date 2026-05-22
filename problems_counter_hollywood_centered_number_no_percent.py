import json
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


# ============================================================
# POV LYRIC SAMPLE STYLE V5
# FIXED TOP-LEFT + SMOOTH / NO JITTER
# ============================================================
#
# Optimized from v4_fixed_top_left:
# - Text is anchored at fixed top-left position for 9:16 videos.
# - Optical-flow shake is disabled by default to remove lyric jitter.
# - Vanishing point is optional and smoothed; final lyric position does not depend on it.
# - Motion settles quickly, then the text stays stable while words reveal.
# - Left aligned, clean white, no black outline.
#
# Input:
#   input.mp4
#   lyrics.json
#
# Output:
#   output_pov_lyric_sample_style_v5_fixed_top_left_smooth.mp4
# ============================================================


INPUT_VIDEO = "input.mp4"
LYRICS_JSON = "lyrics.json"
OUTPUT_VIDEO = "output_pov_lyric_hollywood_centered_number_no_percent.mp4"

FONT_PATH = "fonts/Anton-Regular.ttf"
TEMP_VIDEO_NO_AUDIO = "_temp_pov_lyric_hollywood_centered_number_no_percent.mp4"

MAX_FRAMES = None

# Smooth mode: disable shake to prevent text jitter.
ENABLE_VANISHING_POINT = True
ENABLE_OPTICAL_FLOW_SHAKE = False
ENABLE_DEPTH_BLUR = True
ENABLE_LIGHT_WRAP = True
ENABLE_VECTOR_MOTION_BLUR = True
ENABLE_VIGNETTE = True

# Detect VP only rarely because it only affects the short intro movement.
VANISHING_POINT_EVERY_N_FRAMES = 8

DEBUG_DRAW_VANISHING_POINT = False
FORCE_UPPERCASE = True

# ============================================================
# AUTO PROBLEMS COUNTER MODE - NO lyrics.json REQUIRED
# ============================================================
# Fix flicker by removing segment-based alpha and lyrics.json timing.
# The counter is generated directly from current video timestamp:
#   100% -> 0% in exactly PROBLEMS_COUNTER_DURATION seconds.
#
# PROBLEMS is fixed and fades continuously based on the percentage.
# 0% has a dissolve/tan biến effect.
AUTO_PROBLEMS_COUNTER_MODE = True
PROBLEMS_COUNTER_DURATION = 24.80
PROBLEMS_COUNTER_START_TIME = 0.0

# Larger value = slower / smoother number transition.
# 1.0 means each 1% step takes exactly 25/100 seconds.
PERCENT_TRANSITION_SMOOTHNESS = 1.0

# Giữ 0% thêm một đoạn để người xem nhìn rõ trước khi tan biến.
PROBLEMS_ZERO_HOLD_SECONDS = 0.45

# Final ending accent:
# Sau khi loading chạm 0%, giữ "0% + PROBLEMS" ngắn,
# rồi chỉ giữ lại chữ PROBLEMS 2-3 giây như điểm nhấn cuối,
# cuối cùng tan ra / phân mảnh đẹp mắt.
PROBLEMS_FINAL_SOLO_HOLD_SECONDS = 2.40
PROBLEMS_FINAL_DISSOLVE_SECONDS = 1.35

# Final emphasis / pulse
PROBLEMS_FINAL_PULSE_SCALE = 1.038
PROBLEMS_FINAL_PULSE_ALPHA = 0.06

# Fragmentation / shatter tuning
PROBLEMS_FINAL_FRAGMENT_TILE = 8
PROBLEMS_FINAL_FRAGMENT_SPREAD_X = 390
PROBLEMS_FINAL_FRAGMENT_SPREAD_Y = 130
PROBLEMS_FINAL_FRAGMENT_UPWARD = 105
PROBLEMS_FINAL_FRAGMENT_BLUR = 11
PROBLEMS_FINAL_FRAGMENT_ALPHA_FALLOFF = 1.18

# Title-sequence shard styling
PROBLEMS_FINAL_SHARD_STAGGER = 0.24
PROBLEMS_FINAL_SHARD_ROTATE_DEG = 32.0
PROBLEMS_FINAL_SHARD_STREAK = 13
PROBLEMS_FINAL_SHARD_GLOW = 0.30
PROBLEMS_FINAL_CRACK_JITTER = 18

# Hollywood title sequence final hit
HOLLYWOOD_FINAL_FLASH_SECONDS = 0.18
HOLLYWOOD_FLASH_INTENSITY = 0.72
HOLLYWOOD_LIGHT_RAY_LENGTH = 0.42
HOLLYWOOD_CRACK_LINES = 32
HOLLYWOOD_CRACK_ALPHA = 0.72
HOLLYWOOD_SHOCKWAVE_ALPHA = 0.34
HOLLYWOOD_WARM_SPARKS = True
HOLLYWOOD_SPARK_COUNT = 95

# Smooth handoff tuning:
# Tránh lỗi PROBLEMS mờ dần rồi bật sáng lại đột ngột.
# Sau khi 0%, PROBLEMS sẽ được rebuild opacity từ từ rồi mới vào shard.
PROBLEMS_HANDOFF_REBUILD_SECONDS = 0.85
PROBLEMS_ZERO_HANDOFF_ALPHA = 0.28
PROBLEMS_FINAL_READY_ALPHA = 1.00

# 0% sẽ fade out trong lúc PROBLEMS rebuild, tạo cảm giác chuyển cảnh mượt.
ZERO_PERCENT_HANDOFF_FADE_SECONDS = 0.55

# Light sweep nhẹ trong lúc handoff để che transition, giống title sequence.
PROBLEMS_HANDOFF_LIGHT_SWEEP = True
PROBLEMS_HANDOFF_SWEEP_ALPHA = 0.20
PROBLEMS_HANDOFF_SWEEP_WIDTH = 0.18




# Random loading speed mode:
# % vẫn giảm từ 100 -> 0 đúng trong PROBLEMS_COUNTER_DURATION,
# nhưng thời lượng mỗi bước % sẽ lúc nhanh lúc chậm để tạo cảm giác tò mò.
RANDOM_LOADING_SPEED = True
RANDOM_LOADING_SEED = 20260521

# Hệ số biến thiên tốc độ:
# 0.00 = đều tuyệt đối
# 0.35 = nhẹ
# 0.55 = cinematic, rõ lúc nhanh/lúc chậm
# 0.75 = rất mạnh
RANDOM_LOADING_VARIATION = 0.55

# Bảo vệ nhịp để không có bước quá nhanh/quá chậm.
RANDOM_STEP_MIN_SECONDS = 0.105
RANDOM_STEP_MAX_SECONDS = 0.620


# Keep PROBLEMS stable: no per-segment fade, no blinking.
PROBLEMS_STABLE_ALPHA = True


# SPECIAL MODE:
# Khi lyrics chỉ chứa các số % dạng:
# 100%\n99%\n98%
# thì code tự render chữ PROBLEMS cố định, chỉ phần số % trượt/tịnh tiến xuống.
PROBLEMS_COUNTER_MODE = True

# Khoảng cách giữa số % và chữ PROBLEMS.
# Tăng nhẹ để PROBLEMS thấp hơn, tránh bị số % đè lên.
# Gợi ý:
#   1.05 = gần hơn
#   1.15 = cân bằng
#   1.25 = tách xa hơn
PROBLEMS_LINE_GAP_RATIO = 1.52

# Smooth-roll tuning for percentage number.
# Larger distance = number visibly "vụt/trượt" hơn.
PERCENT_ROLL_DISTANCE_RATIO = 0.070

# Add slight scale motion to the percentage only.
PERCENT_ROLL_SCALE_IN = 0.92
PERCENT_ROLL_SCALE_OUT = 1.12

# Apply directional motion blur to the moving percentage layer.
PERCENT_ROLL_MOTION_BLUR = True
PERCENT_ROLL_BLUR_STRENGTH = 5

# PROBLEMS fade/dissolve tuning.
# Khi % giảm, chữ PROBLEMS cũng mờ dần theo.
PROBLEMS_FADE_BY_PERCENT = True

# 1.00 = linear theo %, 1.25-1.60 = giữ rõ lâu hơn rồi mờ nhanh về cuối.
PROBLEMS_FADE_CURVE = 0.25

# Khi số % <= ngưỡng này thì bắt đầu tan biến mạnh hơn.
PROBLEMS_DISSOLVE_START_PERCENT = 8

# Cường độ tan biến ở giai đoạn 0%.
PROBLEMS_DISSOLVE_STRENGTH = 0.82

# Làm chữ PROBLEMS hơi "vỡ bụi" khi gần 0%.
PROBLEMS_DISSOLVE_PARTICLE_RADIUS = 1.8

# Khi 0%, % cũng tan biến nhẹ, không tắt đột ngột.
ZERO_PERCENT_DISSOLVE = True



@dataclass
class LyricSegment:
    start: float
    end: float
    text: str


@dataclass
class WordBox:
    text: str
    x: int
    y: int
    w: int
    h: int
    index: int
    line_index: int
    font_size: int
    ascent: int
    descent: int


@dataclass
class TextLayout:
    width: int
    height: int
    words: List[WordBox]
    safe_margin: int


def parse_time_to_seconds(t: str) -> float:
    t = t.strip()
    h, m, rest = t.split(":")
    if "." in rest:
        s, ms = rest.split(".")
        ms = ms[:3].ljust(3, "0")
        return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0
    return int(h) * 3600 + int(m) * 60 + int(rest)


def load_lyrics(path: str) -> List[LyricSegment]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Không tìm thấy lyrics file: {path}")

    raw = json.loads(p.read_text(encoding="utf-8"))
    segments: List[LyricSegment] = []

    for item in raw:
        start = parse_time_to_seconds(item["start"])
        end = parse_time_to_seconds(item["end"])
        text = str(item["text"])
        if FORCE_UPPERCASE:
            text = text.upper()
        if end <= start:
            raise ValueError(f"End time phải lớn hơn start time: {item}")
        segments.append(LyricSegment(start=start, end=end, text=text))

    segments.sort(key=lambda x: x.start)
    return segments


def find_active_lyric(segments: List[LyricSegment], t: float) -> Optional[LyricSegment]:
    for seg in segments:
        if seg.start <= t <= seg.end:
            return seg
    return None


def clamp(v: float, min_v: float, max_v: float) -> float:
    return max(min_v, min(max_v, v))


def lerp(a: float, b: float, p: float) -> float:
    return a + (b - a) * p


def smoothstep(edge0: float, edge1: float, x: float) -> float:
    if abs(edge1 - edge0) < 1e-9:
        return 1.0 if x >= edge1 else 0.0
    x = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def ease_out_cubic(x: float) -> float:
    x = clamp(x, 0.0, 1.0)
    return 1.0 - pow(1.0 - x, 3)


class SmoothPoint:
    def __init__(self, alpha: float = 0.025):
        self.alpha = alpha
        self.x = None
        self.y = None

    def update(self, point: Tuple[int, int]) -> Tuple[int, int]:
        px, py = point
        if self.x is None:
            self.x = float(px)
            self.y = float(py)
        else:
            self.x = self.x * (1 - self.alpha) + px * self.alpha
            self.y = self.y * (1 - self.alpha) + py * self.alpha
        return int(self.x), int(self.y)


def detect_vanishing_point(frame_bgr: np.ndarray) -> Tuple[int, int]:
    h, w = frame_bgr.shape[:2]
    fallback = (int(w * 0.52), int(h * 0.20))

    work_w = 640
    scale = work_w / float(w)
    work_h = int(h * scale)

    small = cv2.resize(frame_bgr, (work_w, work_h), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 55, 145)

    mask = np.zeros_like(edges)
    roi = np.array([
        [int(work_w * 0.02), work_h],
        [int(work_w * 0.98), work_h],
        [int(work_w * 0.74), int(work_h * 0.28)],
        [int(work_w * 0.26), int(work_h * 0.28)],
    ], dtype=np.int32)
    cv2.fillPoly(mask, [roi], 255)
    edges = cv2.bitwise_and(edges, mask)

    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180, threshold=42,
        minLineLength=int(work_w * 0.10), maxLineGap=38,
    )
    if lines is None:
        return fallback

    candidates = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = x2 - x1
        dy = y2 - y1
        if abs(dx) < 8:
            continue
        slope = dy / dx
        if abs(slope) < 0.22:
            continue
        candidates.append((x1, y1, x2, y2))

    if len(candidates) < 2:
        return fallback

    intersections = []
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            x1, y1, x2, y2 = candidates[i]
            x3, y3, x4, y4 = candidates[j]
            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if abs(denom) < 1e-6:
                continue
            px = ((x1 * y2 - y1 * x2) * (x3 - x4) -
                  (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
            py = ((x1 * y2 - y1 * x2) * (y3 - y4) -
                  (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
            if 0 <= px <= work_w and int(work_h * 0.04) <= py <= int(work_h * 0.60):
                intersections.append((px, py))

    if not intersections:
        return fallback

    xs = np.array([p[0] for p in intersections], dtype=np.float32)
    ys = np.array([p[1] for p in intersections], dtype=np.float32)
    vp_x = int(clamp(float(np.median(xs)) / scale, 0, w - 1))
    vp_y = int(clamp(float(np.median(ys)) / scale, 0, h - 1))
    return vp_x, vp_y


def load_font(font_path: str, font_size: int):
    if Path(font_path).exists():
        return ImageFont.truetype(font_path, font_size)

    fallback_fonts = [
        "C:/Windows/Fonts/impact.ttf",
        "C:/Windows/Fonts/ariblk.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    for f in fallback_fonts:
        if Path(f).exists():
            return ImageFont.truetype(f, font_size)
    return ImageFont.load_default()


def get_line_scale(line_index: int, total_lines: int, style: dict) -> float:
    if total_lines <= 1:
        return style["line_scale_single"]
    if total_lines == 2:
        return [style["line_scale_small"], style["line_scale_big"]][min(line_index, 1)]
    if total_lines == 3:
        return [style["line_scale_small"], style["line_scale_mid"], style["line_scale_big"]][min(line_index, 2)]
    if line_index == 0:
        return style["line_scale_small"]
    if line_index == total_lines - 1:
        return style["line_scale_big"]
    return style["line_scale_mid"]


def build_text_layout(text: str, font_path: str, style: dict) -> TextLayout:
    lines = [line.strip() for line in text.split("\n")]
    total_lines = len(lines)

    temp = Image.new("RGBA", (10, 10), (0, 0, 0, 0))
    draw = ImageDraw.Draw(temp)
    words: List[WordBox] = []

    safe_margin = style["safe_margin"]
    pad_x = style["pad_x"] + safe_margin
    pad_y = style["pad_y"] + safe_margin

    word_spacing_base = style["word_spacing"]
    line_gap_base = style["line_gap"]

    y = pad_y
    max_w = 0
    global_word_index = 0

    for line_index, line in enumerate(lines):
        parts = [p for p in line.split(" ") if p.strip()]
        scale = get_line_scale(line_index, total_lines, style)
        font_size = max(8, int(style["font_size"] * scale))
        font = load_font(font_path, font_size)
        ascent, descent = font.getmetrics()

        line_advance = ascent + descent
        x = pad_x
        word_spacing = max(2, int(word_spacing_base * scale))

        if not parts:
            y += line_advance + line_gap_base
            continue

        line_max_bottom = y + line_advance
        for part in parts:
            bbox = draw.textbbox((0, 0), part, font=font, stroke_width=0)
            w = bbox[2] - bbox[0]
            h = max(bbox[3] - bbox[1], line_advance)

            words.append(WordBox(
                text=part, x=x, y=y, w=w, h=h,
                index=global_word_index, line_index=line_index,
                font_size=font_size, ascent=ascent, descent=descent,
            ))

            x += w + word_spacing
            global_word_index += 1
            max_w = max(max_w, x - word_spacing)
            line_max_bottom = max(line_max_bottom, y + h)

        y = line_max_bottom + max(1, int(line_gap_base * scale))

    total_w = int(max_w + pad_x + safe_margin)
    total_h = int(y + safe_margin)

    return TextLayout(width=max(1, total_w), height=max(1, total_h), words=words, safe_margin=safe_margin)


def render_progressive_text_rgba(layout: TextLayout, font_path: str, style: dict, progress: float) -> np.ndarray:
    canvas = Image.new("RGBA", (layout.width, layout.height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    words = layout.words
    n = len(words)
    if n == 0:
        return np.array(canvas)

    p = clamp(progress, 0.0, 1.0)
    reveal_portion = style["word_reveal_portion"]
    overlap = style["word_reveal_overlap"]

    base_slot = reveal_portion / max(n, 1)
    fade_len = base_slot * overlap

    for word in words:
        start = word.index * base_slot
        end = start + fade_len
        alpha = 1.0 if p >= reveal_portion else smoothstep(start, end, p)
        if alpha <= 0.001:
            continue

        font = load_font(font_path, word.font_size)

        if style["soft_shadow_alpha"] > 0:
            draw.text(
                (word.x + style["soft_shadow_dx"], word.y + style["soft_shadow_dy"]),
                word.text, font=font,
                fill=(0, 0, 0, int(style["soft_shadow_alpha"] * alpha)),
            )

        draw.text(
            (word.x, word.y), word.text, font=font,
            fill=(style["fill_rgb"][0], style["fill_rgb"][1], style["fill_rgb"][2], int(style["fill_alpha"] * alpha)),
        )

    return np.array(canvas)


def pad_rgba(img: np.ndarray, pad: int) -> np.ndarray:
    if pad <= 0:
        return img
    return cv2.copyMakeBorder(
        img, pad, pad, pad, pad,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0, 0),
    )


def resize_rgba(img: np.ndarray, scale: float) -> np.ndarray:
    h, w = img.shape[:2]
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def apply_depth_blur(rgba: np.ndarray, progress: float, preset: dict) -> np.ndarray:
    if not ENABLE_DEPTH_BLUR:
        return rgba

    p = clamp(progress, 0.0, 1.0)
    # Very light blur only. Strong blur can look like jitter on text.
    far_blur = preset.get("far_blur", 0.10) * (1.0 - p)
    near_blur = preset.get("near_blur", 0.25) * max(0.0, (p - 0.82) / 0.18)
    blur = far_blur + near_blur

    if blur <= 0.20:
        return rgba

    k = int(blur * 2) * 2 + 1
    return cv2.GaussianBlur(rgba, (k, k), blur)


def apply_vector_motion_blur(rgba: np.ndarray, dx: float, dy: float, strength: int) -> np.ndarray:
    if not ENABLE_VECTOR_MOTION_BLUR:
        return rgba

    mag = math.sqrt(dx * dx + dy * dy)
    if mag < 0.5 or strength <= 1:
        return rgba

    dx /= mag
    dy /= mag
    ksize = int(strength)
    if ksize % 2 == 0:
        ksize += 1

    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    center = ksize // 2

    for i in range(ksize):
        offset = i - center
        x = int(center + dx * offset)
        y = int(center + dy * offset)
        if 0 <= x < ksize and 0 <= y < ksize:
            kernel[y, x] = 1.0

    s = kernel.sum()
    if s > 0:
        kernel /= s

    return cv2.filter2D(rgba, -1, kernel)


def build_overlay_for_frame_v5(
    layout: TextLayout,
    frame_w: int,
    frame_h: int,
    progress: float,
    preset: dict,
    style: dict,
    vanishing_point: Tuple[int, int],
) -> np.ndarray:
    p = clamp(progress, 0.0, 1.0)

    # Move/scale settles quickly, then text is fully stable.
    settle_progress = preset["settle_progress"]
    move_p = clamp(p / settle_progress, 0.0, 1.0)
    p_motion = ease_out_cubic(move_p)

    text_rgba = render_progressive_text_rgba(layout, FONT_PATH, style, p)
    text_rgba = pad_rgba(text_rgba, style["transform_overscan"])

    scale = lerp(preset["scale_start"], preset["scale_end"], p_motion)
    transformed = resize_rgba(text_rgba, scale)
    transformed = apply_depth_blur(transformed, p, preset)

    vp_x, vp_y = vanishing_point

    start_x = vp_x + preset["vp_offset_x"]
    start_y = vp_y + preset["vp_offset_y"]

    end_x = frame_w * preset["anchor_left_ratio"]
    end_y = frame_h * preset["anchor_top_ratio"]

    x = int(lerp(start_x, end_x, p_motion))
    y = int(lerp(start_y, end_y, p_motion))

    # Motion blur is only used during the short intro movement.
    if p < settle_progress:
        move_dx = end_x - start_x
        move_dy = end_y - start_y
        vector_blur_strength = int(lerp(preset["vector_blur_start"], preset["vector_blur_end"], p_motion))
        transformed = apply_vector_motion_blur(transformed, move_dx, move_dy, vector_blur_strength)

    alpha_mul = 1.0
    if p >= preset["burst_start"]:
        burst_p = (p - preset["burst_start"]) / max(0.0001, 1.0 - preset["burst_start"])
        burst_p = clamp(burst_p, 0.0, 1.0)
        alpha_mul *= max(0.0, 1.0 - burst_p * burst_p * preset["burst_alpha_power"])

    transformed[:, :, 3] = np.clip(
        transformed[:, :, 3].astype(np.float32) * alpha_mul,
        0, 255
    ).astype(np.uint8)

    full = np.zeros((frame_h, frame_w, 4), dtype=np.uint8)

    th, tw = transformed.shape[:2]
    x1, y1 = x, y
    x2, y2 = x + tw, y + th

    dst_x1 = max(0, x1)
    dst_y1 = max(0, y1)
    dst_x2 = min(frame_w, x2)
    dst_y2 = min(frame_h, y2)

    src_x1 = max(0, -x1)
    src_y1 = max(0, -y1)
    src_x2 = src_x1 + max(0, dst_x2 - dst_x1)
    src_y2 = src_y1 + max(0, dst_y2 - dst_y1)

    if dst_x1 < dst_x2 and dst_y1 < dst_y2:
        full[dst_y1:dst_y2, dst_x1:dst_x2] = transformed[src_y1:src_y2, src_x1:src_x2]

    return full



def extract_problem_percentages(text: str) -> List[int]:
    values = re.findall(r"(\d{1,3})\s*%", text)
    result = []
    for value in values:
        n = int(value)
        if 0 <= n <= 100:
            result.append(n)
    return result


def draw_text_rgba(
    draw: ImageDraw.ImageDraw,
    xy: Tuple[float, float],
    text: str,
    font,
    fill_rgb: Tuple[int, int, int],
    alpha: float,
) -> None:
    alpha_i = int(clamp(alpha, 0.0, 1.0) * 255)
    if alpha_i <= 0:
        return
    x, y = xy
    draw.text((x, y), text, font=font, fill=(fill_rgb[0], fill_rgb[1], fill_rgb[2], alpha_i))



def measure_text_size(text: str, font) -> Tuple[int, int]:
    """
    Return exact visible text width/height for precise horizontal centering.
    """
    temp = Image.new("RGBA", (10, 10), (0, 0, 0, 0))
    draw = ImageDraw.Draw(temp)
    bbox = draw.textbbox((0, 0), text, font=font)
    return max(1, bbox[2] - bbox[0]), max(1, bbox[3] - bbox[1])


def centered_x_for_text(center_x: float, text: str, font, scale: float = 1.0) -> int:
    """
    Calculate x so the visible text is centered around center_x.
    Used to center numbers above the PROBLEMS word.
    """
    text_w, _ = measure_text_size(text, font)
    return int(round(center_x - (text_w * scale) / 2.0))


def render_single_text_layer(
    canvas_w: int,
    canvas_h: int,
    text: str,
    font,
    fill_rgb: Tuple[int, int, int],
    alpha: float,
    scale: float = 1.0,
) -> np.ndarray:
    """
    Render one text element to a transparent RGBA layer.
    The caller will paste this layer at the desired position.
    """
    alpha_i = int(clamp(alpha, 0.0, 1.0) * 255)
    layer = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    if alpha_i <= 0:
        return np.array(layer)

    temp = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(temp)
    draw.text((0, 0), text, font=font, fill=(fill_rgb[0], fill_rgb[1], fill_rgb[2], alpha_i))

    arr = np.array(temp)
    if abs(scale - 1.0) > 1e-4:
        arr = resize_rgba(arr, scale)

    return arr


def paste_layer_rgba(dest: np.ndarray, src: np.ndarray, x: int, y: int) -> None:
    """
    Alpha-compose src over dest at x/y.
    Unlike direct assignment, this preserves partial alpha and allows blur/fade.
    """
    h, w = dest.shape[:2]
    sh, sw = src.shape[:2]

    x1, y1 = x, y
    x2, y2 = x + sw, y + sh

    dx1 = max(0, x1)
    dy1 = max(0, y1)
    dx2 = min(w, x2)
    dy2 = min(h, y2)

    sx1 = max(0, -x1)
    sy1 = max(0, -y1)
    sx2 = sx1 + max(0, dx2 - dx1)
    sy2 = sy1 + max(0, dy2 - dy1)

    if dx1 >= dx2 or dy1 >= dy2:
        return

    src_patch = src[sy1:sy2, sx1:sx2].astype(np.float32)
    dst_patch = dest[dy1:dy2, dx1:dx2].astype(np.float32)

    a = src_patch[:, :, 3:4] / 255.0
    dst_patch[:, :, :3] = src_patch[:, :, :3] * a + dst_patch[:, :, :3] * (1.0 - a)
    dst_patch[:, :, 3:4] = np.maximum(dst_patch[:, :, 3:4], src_patch[:, :, 3:4])

    dest[dy1:dy2, dx1:dx2] = np.clip(dst_patch, 0, 255).astype(np.uint8)


def draw_moving_percent(
    overlay: np.ndarray,
    text: str,
    font,
    fill_rgb: Tuple[int, int, int],
    x: int,
    y: int,
    alpha: float,
    scale: float,
    blur_dy: float,
) -> None:
    """
    Render the percentage number as its own moving layer.
    This allows scale + vector motion blur without affecting fixed PROBLEMS.
    """
    layer = render_single_text_layer(
        canvas_w=max(200, overlay.shape[1] // 2),
        canvas_h=max(120, overlay.shape[0] // 5),
        text=text,
        font=font,
        fill_rgb=fill_rgb,
        alpha=alpha,
        scale=scale,
    )

    if PERCENT_ROLL_MOTION_BLUR and PERCENT_ROLL_BLUR_STRENGTH > 1 and alpha > 0.02:
        layer = apply_vector_motion_blur(
            layer,
            dx=0.0,
            dy=blur_dy,
            strength=PERCENT_ROLL_BLUR_STRENGTH,
        )

    paste_layer_rgba(overlay, layer, x, y)



def percent_to_problem_alpha(percent_value: float) -> float:
    """
    Convert current percentage into PROBLEMS alpha.
    100% => alpha 1.0
    0%   => alpha 0.0
    Curve > 1 keeps it visible longer and fades faster near the end.
    """
    pct = clamp(float(percent_value), 0.0, 100.0) / 100.0
    if not PROBLEMS_FADE_BY_PERCENT:
        return 1.0
    return clamp(pow(pct, PROBLEMS_FADE_CURVE), 0.0, 1.0)


def apply_alpha_to_rgba(rgba: np.ndarray, alpha_mul: float) -> np.ndarray:
    if alpha_mul >= 0.999:
        return rgba
    out = rgba.copy()
    out[:, :, 3] = np.clip(out[:, :, 3].astype(np.float32) * alpha_mul, 0, 255).astype(np.uint8)
    return out


def deterministic_noise_mask(h: int, w: int, seed: int) -> np.ndarray:
    """
    Deterministic noise mask without uint32 overflow warnings.
    Uses uint64 math, then masks to 32-bit explicitly.
    """
    yy, xx = np.indices((h, w), dtype=np.uint64)

    seed_u64 = np.uint64(int(seed) & 0xFFFFFFFF)
    c1 = np.uint64(374761393)
    c2 = np.uint64(668265263)
    c3 = np.uint64(1442695041)
    c4 = np.uint64(1274126177)
    mask32 = np.uint64(0xFFFFFFFF)

    v = (xx * c1 + yy * c2 + seed_u64 * c3) & mask32
    v = ((v ^ (v >> np.uint64(13))) * c4) & mask32
    v = (v ^ (v >> np.uint64(16))) & mask32

    return (v & np.uint64(255)).astype(np.uint8)

def apply_dissolve_to_rgba(rgba: np.ndarray, amount: float, seed: int = 1, particle_radius: float = 1.6) -> np.ndarray:
    """
    Pixel/particle dissolve for final 0% effect.
    amount:
      0.0 = no dissolve
      1.0 = almost fully dissolved
    """
    amount = clamp(amount, 0.0, 1.0)
    if amount <= 0.001:
        return rgba

    out = rgba.copy()
    alpha = out[:, :, 3]
    if alpha.max() == 0:
        return out

    noise = deterministic_noise_mask(alpha.shape[0], alpha.shape[1], seed)
    threshold = int(255 * amount)
    keep = noise > threshold

    # Add slightly granular feathering instead of a hard binary cut.
    soft = np.clip((noise.astype(np.float32) - threshold) / max(1.0, 255 - threshold), 0.0, 1.0)

    new_alpha = alpha.astype(np.float32) * keep.astype(np.float32) * (0.35 + 0.65 * soft)
    out[:, :, 3] = np.clip(new_alpha, 0, 255).astype(np.uint8)

    if particle_radius > 0 and amount > 0.20:
        # Tiny blur on alpha edge gives a dust-like vanish instead of harsh pixel loss.
        k = max(1, int(particle_radius * amount) * 2 + 1)
        out[:, :, 3] = cv2.GaussianBlur(out[:, :, 3], (k, k), 0)

    return out


def render_fixed_text_layer(
    frame_w: int,
    frame_h: int,
    text: str,
    font,
    fill_rgb: Tuple[int, int, int],
    alpha: float,
    x: int,
    y: int,
) -> np.ndarray:
    layer = Image.new("RGBA", (frame_w, frame_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    draw_text_rgba(draw, (x, y), text, font, fill_rgb, alpha)
    return np.array(layer)


def build_problems_counter_overlay(
    frame_w: int,
    frame_h: int,
    lyric_text: str,
    progress: float,
    preset: dict,
    style: dict,
) -> Optional[np.ndarray]:
    """
    Smooth counter mode with fading PROBLEMS:
    - Percentage number rolls/slides smoothly.
    - PROBLEMS is fixed, but alpha follows the current percentage.
    - At 0%, both the number and PROBLEMS dissolve instead of disappearing abruptly.
    """
    percentages = extract_problem_percentages(lyric_text)
    if not percentages:
        return None

    p = clamp(progress, 0.0, 1.0)
    n = len(percentages)

    phase = p * n
    idx = min(n - 1, int(phase))
    local_p = phase - idx
    if idx == n - 1:
        local_p = min(local_p, 1.0)

    roll_p = smoothstep(0.0, 1.0, local_p)

    anchor_x = int(frame_w * preset["anchor_left_ratio"])
    anchor_y = int(frame_h * preset["anchor_top_ratio"])

    percent_font_size = max(16, int(style["font_size"] * 1.35))
    problems_font_size = max(14, int(style["font_size"] * 0.90))

    percent_font = load_font(FONT_PATH, percent_font_size)
    problems_font = load_font(FONT_PATH, problems_font_size)

    overlay = np.zeros((frame_h, frame_w, 4), dtype=np.uint8)
    fill_rgb = style["fill_rgb"]

    group_alpha = smoothstep(0.00, 0.10, p) * (1.0 - smoothstep(0.94, 1.00, p))
    roll_dist = int(frame_h * PERCENT_ROLL_DISTANCE_RATIO)

    current_value = percentages[idx]
    next_value = percentages[idx + 1] if idx + 1 < n else current_value

    # Interpolated value is used only for fading PROBLEMS, not displayed as decimal.
    interp_percent = lerp(float(current_value), float(next_value), roll_p)
    problems_alpha_by_percent = percent_to_problem_alpha(interp_percent)

    # Extra dissolve progress when value approaches 0%.
    low_pct_p = 1.0 - clamp(interp_percent / max(1.0, float(PROBLEMS_DISSOLVE_START_PERCENT)), 0.0, 1.0)
    dissolve_amount = smoothstep(0.0, 1.0, low_pct_p) * PROBLEMS_DISSOLVE_STRENGTH

    # Current % exits downward.
    current_y = anchor_y + int(roll_p * roll_dist)
    current_alpha = group_alpha * (1.0 - smoothstep(0.62, 1.00, local_p))
    current_scale = lerp(1.00, PERCENT_ROLL_SCALE_OUT, roll_p)

    current_layer = render_single_text_layer(
        canvas_w=max(200, frame_w // 2),
        canvas_h=max(120, frame_h // 5),
        text=f"{current_value}%",
        font=percent_font,
        fill_rgb=fill_rgb,
        alpha=current_alpha,
        scale=current_scale,
    )

    if ZERO_PERCENT_DISSOLVE and current_value == 0:
        current_layer = apply_dissolve_to_rgba(
            current_layer,
            amount=max(dissolve_amount, smoothstep(0.35, 1.0, local_p)),
            seed=17,
            particle_radius=PROBLEMS_DISSOLVE_PARTICLE_RADIUS,
        )

    if PERCENT_ROLL_MOTION_BLUR and PERCENT_ROLL_BLUR_STRENGTH > 1 and current_alpha > 0.02:
        current_layer = apply_vector_motion_blur(
            current_layer,
            dx=0.0,
            dy=1.0,
            strength=PERCENT_ROLL_BLUR_STRENGTH,
        )

    paste_layer_rgba(overlay, current_layer, anchor_x, current_y)

    # Next % enters from above.
    if idx + 1 < n:
        next_y = anchor_y - roll_dist + int(roll_p * roll_dist)
        next_alpha = group_alpha * smoothstep(0.18, 0.78, local_p)
        next_scale = lerp(PERCENT_ROLL_SCALE_IN, 1.00, roll_p)

        next_layer = render_single_text_layer(
            canvas_w=max(200, frame_w // 2),
            canvas_h=max(120, frame_h // 5),
            text=f"{next_value}%",
            font=percent_font,
            fill_rgb=fill_rgb,
            alpha=next_alpha,
            scale=next_scale,
        )

        if PERCENT_ROLL_MOTION_BLUR and PERCENT_ROLL_BLUR_STRENGTH > 1 and next_alpha > 0.02:
            next_layer = apply_vector_motion_blur(
                next_layer,
                dx=0.0,
                dy=1.0,
                strength=PERCENT_ROLL_BLUR_STRENGTH,
            )

        paste_layer_rgba(overlay, next_layer, anchor_x, next_y)

    # Fixed PROBLEMS layer: fixed position, but opacity fades with the percentage.
    problems_y = anchor_y + int(percent_font_size * PROBLEMS_LINE_GAP_RATIO)
    problems_alpha = group_alpha * problems_alpha_by_percent

    problems_layer = render_fixed_text_layer(
        frame_w=frame_w,
        frame_h=frame_h,
        text="PROBLEMS",
        font=problems_font,
        fill_rgb=fill_rgb,
        alpha=problems_alpha,
        x=anchor_x,
        y=problems_y,
    )

    if dissolve_amount > 0.001:
        problems_layer = apply_dissolve_to_rgba(
            problems_layer,
            amount=dissolve_amount,
            seed=31,
            particle_radius=PROBLEMS_DISSOLVE_PARTICLE_RADIUS,
        )

    paste_layer_rgba(overlay, problems_layer, 0, 0)

    return overlay


def percent_to_problem_alpha(percent_value: float) -> float:
    pct = clamp(float(percent_value), 0.0, 100.0) / 100.0
    if not PROBLEMS_FADE_BY_PERCENT:
        return 1.0
    return clamp(pow(pct, PROBLEMS_FADE_CURVE), 0.0, 1.0)


def apply_alpha_to_rgba(rgba: np.ndarray, alpha_mul: float) -> np.ndarray:
    if alpha_mul >= 0.999:
        return rgba
    out = rgba.copy()
    out[:, :, 3] = np.clip(out[:, :, 3].astype(np.float32) * alpha_mul, 0, 255).astype(np.uint8)
    return out


def deterministic_noise_mask(h: int, w: int, seed: int) -> np.ndarray:
    """
    Deterministic noise mask without uint32 overflow warnings.
    Uses uint64 math, then masks to 32-bit explicitly.
    """
    yy, xx = np.indices((h, w), dtype=np.uint64)

    seed_u64 = np.uint64(int(seed) & 0xFFFFFFFF)
    c1 = np.uint64(374761393)
    c2 = np.uint64(668265263)
    c3 = np.uint64(1442695041)
    c4 = np.uint64(1274126177)
    mask32 = np.uint64(0xFFFFFFFF)

    v = (xx * c1 + yy * c2 + seed_u64 * c3) & mask32
    v = ((v ^ (v >> np.uint64(13))) * c4) & mask32
    v = (v ^ (v >> np.uint64(16))) & mask32

    return (v & np.uint64(255)).astype(np.uint8)

def apply_dissolve_to_rgba(rgba: np.ndarray, amount: float, seed: int = 1, particle_radius: float = 1.6) -> np.ndarray:
    amount = clamp(amount, 0.0, 1.0)
    if amount <= 0.001:
        return rgba

    out = rgba.copy()
    alpha = out[:, :, 3]
    if alpha.max() == 0:
        return out

    noise = deterministic_noise_mask(alpha.shape[0], alpha.shape[1], seed)
    threshold = int(255 * amount)
    keep = noise > threshold
    soft = np.clip((noise.astype(np.float32) - threshold) / max(1.0, 255 - threshold), 0.0, 1.0)

    new_alpha = alpha.astype(np.float32) * keep.astype(np.float32) * (0.35 + 0.65 * soft)
    out[:, :, 3] = np.clip(new_alpha, 0, 255).astype(np.uint8)

    if particle_radius > 0 and amount > 0.20:
        k = max(1, int(particle_radius * amount) * 2 + 1)
        out[:, :, 3] = cv2.GaussianBlur(out[:, :, 3], (k, k), 0)

    return out


def render_single_text_layer(
    canvas_w: int,
    canvas_h: int,
    text: str,
    font,
    fill_rgb: Tuple[int, int, int],
    alpha: float,
    scale: float = 1.0,
) -> np.ndarray:
    alpha_i = int(clamp(alpha, 0.0, 1.0) * 255)
    layer = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    if alpha_i <= 0:
        return np.array(layer)

    temp = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(temp)
    draw.text((0, 0), text, font=font, fill=(fill_rgb[0], fill_rgb[1], fill_rgb[2], alpha_i))

    arr = np.array(temp)
    if abs(scale - 1.0) > 1e-4:
        arr = resize_rgba(arr, scale)

    return arr


def paste_layer_rgba(dest: np.ndarray, src: np.ndarray, x: int, y: int) -> None:
    h, w = dest.shape[:2]
    sh, sw = src.shape[:2]

    x1, y1 = x, y
    x2, y2 = x + sw, y + sh

    dx1 = max(0, x1)
    dy1 = max(0, y1)
    dx2 = min(w, x2)
    dy2 = min(h, y2)

    sx1 = max(0, -x1)
    sy1 = max(0, -y1)
    sx2 = sx1 + max(0, dx2 - dx1)
    sy2 = sy1 + max(0, dy2 - dy1)

    if dx1 >= dx2 or dy1 >= dy2:
        return

    src_patch = src[sy1:sy2, sx1:sx2].astype(np.float32)
    dst_patch = dest[dy1:dy2, dx1:dx2].astype(np.float32)

    a = src_patch[:, :, 3:4] / 255.0
    dst_patch[:, :, :3] = src_patch[:, :, :3] * a + dst_patch[:, :, :3] * (1.0 - a)
    dst_patch[:, :, 3:4] = np.maximum(dst_patch[:, :, 3:4], src_patch[:, :, 3:4])

    dest[dy1:dy2, dx1:dx2] = np.clip(dst_patch, 0, 255).astype(np.uint8)


def draw_text_rgba(
    draw: ImageDraw.ImageDraw,
    xy: Tuple[float, float],
    text: str,
    font,
    fill_rgb: Tuple[int, int, int],
    alpha: float,
) -> None:
    alpha_i = int(clamp(alpha, 0.0, 1.0) * 255)
    if alpha_i <= 0:
        return
    x, y = xy
    draw.text((x, y), text, font=font, fill=(fill_rgb[0], fill_rgb[1], fill_rgb[2], alpha_i))


def render_fixed_text_layer(
    frame_w: int,
    frame_h: int,
    text: str,
    font,
    fill_rgb: Tuple[int, int, int],
    alpha: float,
    x: int,
    y: int,
) -> np.ndarray:
    layer = Image.new("RGBA", (frame_w, frame_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    draw_text_rgba(draw, (x, y), text, font, fill_rgb, alpha)
    return np.array(layer)



_RANDOM_LOADING_SCHEDULE_CACHE = None


def build_random_loading_schedule(total_duration: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a deterministic random-looking loading schedule.

    Returns:
      cumulative_times: length 101, from 0 to total_duration
      durations:        length 100, each step duration for 100->99 ... 1->0

    The schedule is deterministic by RANDOM_LOADING_SEED, so re-rendering
    gives the same result every time.
    """
    rng = np.random.default_rng(int(RANDOM_LOADING_SEED))

    # Base random weights.
    raw = rng.lognormal(mean=0.0, sigma=float(RANDOM_LOADING_VARIATION), size=100).astype(np.float64)

    # Add wave pulses: some areas intentionally slow down / speed up.
    x = np.linspace(0.0, 1.0, 100)
    wave = (
        1.0
        + 0.22 * np.sin(2.0 * np.pi * (x * 3.0 + 0.10))
        + 0.14 * np.sin(2.0 * np.pi * (x * 8.0 + 0.35))
    )
    raw *= np.clip(wave, 0.45, 1.85)

    # Make the last 10% slightly slower so 10% -> 0% is readable.
    tail_boost = np.ones(100, dtype=np.float64)
    tail_boost[-12:] = np.linspace(1.10, 1.75, 12)
    raw *= tail_boost

    # Normalize, then clamp and renormalize a few times.
    durations = raw / raw.sum() * float(total_duration)

    for _ in range(8):
        durations = np.clip(durations, RANDOM_STEP_MIN_SECONDS, RANDOM_STEP_MAX_SECONDS)
        durations *= float(total_duration) / durations.sum()

    cumulative = np.concatenate([[0.0], np.cumsum(durations)])
    cumulative[-1] = float(total_duration)

    return cumulative, durations


def get_random_counter_state(elapsed: float) -> Optional[Tuple[int, int, float, float]]:
    """
    Return:
      current_value, next_value, local_p, display_elapsed

    Handles:
    - random speed from 100 -> 0 during PROBLEMS_COUNTER_DURATION
    - short 0% hold
    - final PROBLEMS-only hold
    - final PROBLEMS shatter/dissolve
    """
    if elapsed < 0.0:
        return None

    total_visible = (
        PROBLEMS_COUNTER_DURATION
        + PROBLEMS_ZERO_HOLD_SECONDS
        + PROBLEMS_FINAL_SOLO_HOLD_SECONDS
        + PROBLEMS_FINAL_DISSOLVE_SECONDS
    )
    if elapsed > total_visible:
        return None

    if elapsed >= PROBLEMS_COUNTER_DURATION:
        # All post-loading stages still logically stay at 0%.
        return 0, 0, 0.0, elapsed

    if not RANDOM_LOADING_SPEED:
        step_duration = PROBLEMS_COUNTER_DURATION / 100.0
        step_float = elapsed / step_duration
        step_index = int(clamp(math.floor(step_float), 0, 100))
        local_p = clamp(step_float - step_index, 0.0, 1.0)
    else:
        global _RANDOM_LOADING_SCHEDULE_CACHE
        if _RANDOM_LOADING_SCHEDULE_CACHE is None:
            _RANDOM_LOADING_SCHEDULE_CACHE = build_random_loading_schedule(PROBLEMS_COUNTER_DURATION)

        cumulative, durations = _RANDOM_LOADING_SCHEDULE_CACHE

        # segment index 0 means 100 -> 99
        step_index = int(np.searchsorted(cumulative, elapsed, side="right") - 1)
        step_index = int(clamp(step_index, 0, 99))

        segment_start = float(cumulative[step_index])
        segment_duration = float(durations[step_index])
        local_p = clamp((elapsed - segment_start) / max(0.001, segment_duration), 0.0, 1.0)

    current_value = int(clamp(100 - step_index, 0, 100))
    next_value = int(clamp(current_value - 1, 0, 100))

    return current_value, next_value, local_p, elapsed



def make_glow_from_rgba(rgba: np.ndarray, blur_radius: int = 14, alpha_mul: float = 0.22) -> np.ndarray:
    """
    Soft white glow generated from the alpha of an RGBA text layer.
    """
    out = np.zeros_like(rgba)
    alpha = rgba[:, :, 3]
    if alpha.max() == 0:
        return out

    k = max(3, blur_radius | 1)
    glow_alpha = cv2.GaussianBlur(alpha, (k, k), 0)
    out[:, :, :3] = 255
    out[:, :, 3] = np.clip(glow_alpha.astype(np.float32) * alpha_mul, 0, 255).astype(np.uint8)
    return out


def tile_rand_signed(ix: int, iy: int, seed: int) -> float:
    n = (ix * 73856093) ^ (iy * 19349663) ^ (seed * 83492791)
    n = (n << 13) ^ n
    v = 1.0 - (((n * (n * n * 15731 + 789221) + 1376312589) & 0x7FFFFFFF) / 1073741824.0)
    return float(v)  # ~[-1, 1]



def add_hollywood_flash_layer(frame_w: int, frame_h: int, center_x: int, center_y: int, progress: float) -> np.ndarray:
    """
    Warm center flash + horizontal anamorphic light streak for a Hollywood title hit.
    """
    p = clamp(progress, 0.0, 1.0)
    alpha = (1.0 - smoothstep(0.00, 1.00, p)) * HOLLYWOOD_FLASH_INTENSITY
    out = np.zeros((frame_h, frame_w, 4), dtype=np.uint8)
    if alpha <= 0.001:
        return out

    yy, xx = np.indices((frame_h, frame_w), dtype=np.float32)
    dx = (xx - center_x) / max(1.0, frame_w * HOLLYWOOD_LIGHT_RAY_LENGTH)
    dy = (yy - center_y) / max(1.0, frame_h * 0.065)

    streak = np.exp(-(dx * dx * 2.8 + dy * dy * 10.0))
    core = np.exp(-(((xx - center_x) / max(1.0, frame_w * 0.10)) ** 2 + ((yy - center_y) / max(1.0, frame_h * 0.08)) ** 2) * 4.5)
    light = np.clip(streak * 0.9 + core * 1.2, 0.0, 1.0)

    # warm Hollywood gold/white
    out[:, :, 0] = np.clip(255 * light, 0, 255).astype(np.uint8)
    out[:, :, 1] = np.clip(218 * light, 0, 255).astype(np.uint8)
    out[:, :, 2] = np.clip(120 * light, 0, 255).astype(np.uint8)
    out[:, :, 3] = np.clip(255 * light * alpha, 0, 255).astype(np.uint8)
    return out


def add_shockwave_ring(frame_w: int, frame_h: int, center_x: int, center_y: int, progress: float) -> np.ndarray:
    """
    Subtle expanding ring behind the word right before shard explosion.
    """
    p = clamp(progress, 0.0, 1.0)
    out = np.zeros((frame_h, frame_w, 4), dtype=np.uint8)
    if p <= 0.001:
        return out

    yy, xx = np.indices((frame_h, frame_w), dtype=np.float32)
    dist = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)
    radius = lerp(frame_w * 0.04, frame_w * 0.34, p)
    width = lerp(7.0, 18.0, p)

    ring = np.exp(-((dist - radius) ** 2) / (2.0 * width * width))
    alpha = ring * HOLLYWOOD_SHOCKWAVE_ALPHA * (1.0 - smoothstep(0.45, 1.0, p))
    out[:, :, :3] = 255
    out[:, :, 3] = np.clip(alpha * 255, 0, 255).astype(np.uint8)
    return out


def draw_crack_lines(frame_w: int, frame_h: int, min_x: int, max_x: int, min_y: int, max_y: int, progress: float, seed: int = 333) -> np.ndarray:
    """
    Draws thin golden/white crack lines across the word bounding box.
    """
    p = clamp(progress, 0.0, 1.0)
    out = Image.new("RGBA", (frame_w, frame_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(out)
    rng = np.random.default_rng(seed)

    alpha = int(255 * HOLLYWOOD_CRACK_ALPHA * smoothstep(0.05, 0.55, p) * (1.0 - smoothstep(0.72, 1.0, p)))
    if alpha <= 0:
        return np.array(out)

    for _ in range(HOLLYWOOD_CRACK_LINES):
        x1 = int(rng.integers(min_x, max_x + 1))
        y1 = int(rng.integers(min_y, max_y + 1))
        length = int(rng.integers(max(18, (max_x-min_x)//10), max(30, (max_x-min_x)//3)))
        angle = float(rng.normal(0, 0.42))
        if rng.random() < 0.35:
            angle += float(rng.choice([-1, 1]) * rng.uniform(0.7, 1.35))

        x2 = int(x1 + math.cos(angle) * length * p)
        y2 = int(y1 + math.sin(angle) * length * p * 0.55)

        color = (255, int(rng.integers(205, 255)), int(rng.integers(120, 205)), alpha)
        width = 1 if rng.random() < 0.82 else 2
        draw.line((x1, y1, x2, y2), fill=color, width=width)

    return np.array(out)


def add_hollywood_sparks(frame_w: int, frame_h: int, center_x: int, center_y: int, progress: float, seed: int = 777) -> np.ndarray:
    """
    Warm cinematic sparks that fly with the shards.
    """
    p = clamp(progress, 0.0, 1.0)
    out = Image.new("RGBA", (frame_w, frame_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(out)
    rng = np.random.default_rng(seed)

    for _ in range(HOLLYWOOD_SPARK_COUNT):
        angle = rng.uniform(-math.pi * 0.95, math.pi * 0.15)
        speed = rng.uniform(frame_w * 0.06, frame_w * 0.42)
        drift = speed * (p ** 1.25)
        x = center_x + math.cos(angle) * drift + rng.normal(0, 8)
        y = center_y + math.sin(angle) * drift * 0.55 - rng.uniform(0, frame_h * 0.08) * p

        length = rng.uniform(4, 22) * (0.2 + p)
        alpha = int(255 * rng.uniform(0.18, 0.68) * (1.0 - p) * smoothstep(0.05, 0.35, p))
        if alpha <= 0:
            continue

        x2 = x - math.cos(angle) * length
        y2 = y - math.sin(angle) * length * 0.55
        draw.line((x, y, x2, y2), fill=(255, int(rng.integers(160, 230)), int(rng.integers(70, 145)), alpha), width=1)

    return np.array(out)


def apply_fragment_shatter_to_rgba(
    rgba: np.ndarray,
    progress: float,
    seed: int = 101,
) -> np.ndarray:
    """
    Hollywood title sequence shard dissolve.

    This version is more explosive than the cinematic version:
    - flash hit
    - crack lines
    - shockwave ring
    - many small rotating shards
    - warm sparks
    - anamorphic glow trail
    """
    p = clamp(progress, 0.0, 1.0)
    if p <= 0.001:
        return rgba.copy()

    h, w = rgba.shape[:2]
    out = np.zeros_like(rgba)

    alpha = rgba[:, :, 3]
    ys, xs = np.where(alpha > 0)
    if len(xs) == 0:
        return out

    min_x, max_x = int(xs.min()), int(xs.max())
    min_y, max_y = int(ys.min()), int(ys.max())
    bbox_w = max(1, max_x - min_x + 1)
    bbox_h = max(1, max_y - min_y + 1)
    cx = int(0.5 * (min_x + max_x))
    cy = int(0.5 * (min_y + max_y))

    # Flash/cracks are strongest at the beginning.
    flash_p = clamp(p / max(0.001, HOLLYWOOD_FINAL_FLASH_SECONDS / max(0.001, PROBLEMS_FINAL_DISSOLVE_SECONDS)), 0.0, 1.0)
    flash = add_hollywood_flash_layer(w, h, cx, cy, flash_p)
    paste_layer_rgba(out, flash, 0, 0)

    crack = draw_crack_lines(w, h, min_x, max_x, min_y, max_y, progress=p, seed=seed + 101)
    paste_layer_rgba(out, crack, 0, 0)

    shock = add_shockwave_ring(w, h, cx, cy, progress=smoothstep(0.06, 0.95, p))
    paste_layer_rgba(out, shock, 0, 0)

    tile = max(5, int(PROBLEMS_FINAL_FRAGMENT_TILE))
    for y in range(0, h, tile):
        for x in range(0, w, tile):
            piece = rgba[y:y+tile, x:x+tile].copy()
            if piece.shape[0] == 0 or piece.shape[1] == 0:
                continue
            if piece[:, :, 3].max() == 0:
                continue

            pcx = x + piece.shape[1] * 0.5
            pcy = y + piece.shape[0] * 0.5

            rel_x = (pcx - cx) / max(1.0, bbox_w * 0.5)
            rel_y = (pcy - cy) / max(1.0, bbox_h * 0.5)
            radial = min(1.0, math.sqrt(rel_x * rel_x + rel_y * rel_y))

            rand_a = tile_rand_signed(x // tile, y // tile, seed + 1)
            rand_b = tile_rand_signed(x // tile, y // tile, seed + 2)
            rand_c = tile_rand_signed(x // tile, y // tile, seed + 3)

            local_start = 0.02 + radial * (PROBLEMS_FINAL_SHARD_STAGGER * 0.35) + (0.5 + 0.5 * rand_a) * (PROBLEMS_FINAL_SHARD_STAGGER * 0.65)
            local_start = clamp(local_start, 0.0, 0.72)
            local_p = clamp((p - local_start) / max(0.001, 1.0 - local_start), 0.0, 1.0)

            # Pre-fracture jitter
            crack_jitter = smoothstep(0.00, 0.22, p) * (1.0 - local_p)
            jitter_x = rand_a * PROBLEMS_FINAL_CRACK_JITTER * crack_jitter
            jitter_y = rand_b * PROBLEMS_FINAL_CRACK_JITTER * 0.45 * crack_jitter

            vx = pcx - cx
            vy = pcy - cy
            norm = max(1.0, math.sqrt(vx * vx + vy * vy))
            dir_x = vx / norm
            dir_y = vy / norm

            # Hollywood explosion: mostly horizontal, with upward lift and randomness
            accel = local_p ** 1.02
            dx = jitter_x + (
                dir_x * PROBLEMS_FINAL_FRAGMENT_SPREAD_X
                + rand_a * (PROBLEMS_FINAL_FRAGMENT_SPREAD_X * 0.72)
            ) * accel
            dy = jitter_y + (
                dir_y * PROBLEMS_FINAL_FRAGMENT_SPREAD_Y
                - PROBLEMS_FINAL_FRAGMENT_UPWARD
                + rand_b * (PROBLEMS_FINAL_FRAGMENT_SPREAD_Y * 0.55)
            ) * accel

            angle = (rand_a * PROBLEMS_FINAL_SHARD_ROTATE_DEG + rand_c * 11.0) * (local_p ** 0.9)

            # Alpha: hold visible at first, then fade strongly
            alpha_mul = max(0.0, 1.0 - (local_p ** PROBLEMS_FINAL_FRAGMENT_ALPHA_FALLOFF))
            piece[:, :, 3] = np.clip(piece[:, :, 3].astype(np.float32) * alpha_mul, 0, 255).astype(np.uint8)

            ph, pw = piece.shape[:2]
            pad = max(5, int(tile * 0.75))
            padded = np.zeros((ph + pad * 2, pw + pad * 2, 4), dtype=np.uint8)
            padded[pad:pad+ph, pad:pad+pw] = piece

            center = ((padded.shape[1] - 1) * 0.5, (padded.shape[0] - 1) * 0.5)
            M = cv2.getRotationMatrix2D(center, angle, 1.0 + 0.05 * local_p)
            rotated = cv2.warpAffine(
                padded,
                M,
                (padded.shape[1], padded.shape[0]),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0, 0),
            )

            # Directional streak for high-energy Hollywood movement.
            if local_p > 0.035:
                streak_strength = int(max(3, 1 + PROBLEMS_FINAL_SHARD_STREAK * (local_p ** 0.8)))
                rotated = apply_vector_motion_blur(rotated, dx=dx, dy=dy, strength=streak_strength)

            if local_p > 0.18:
                k = max(1, int(1 + PROBLEMS_FINAL_FRAGMENT_BLUR * local_p))
                k = k if k % 2 == 1 else k + 1
                rotated = cv2.GaussianBlur(rotated, (k, k), 0)

            paste_layer_rgba(out, rotated, int(round(x + dx - pad)), int(round(y + dy - pad)))

    # Warm sparks + glow trails.
    if HOLLYWOOD_WARM_SPARKS:
        sparks = add_hollywood_sparks(w, h, cx, cy, progress=p, seed=seed + 303)
        paste_layer_rgba(out, sparks, 0, 0)

    glow = make_glow_from_rgba(
        out,
        blur_radius=31,
        alpha_mul=PROBLEMS_FINAL_SHARD_GLOW * (1.0 - p * 0.26),
    )
    paste_layer_rgba(out, glow, 0, 0)

    return out


def render_scaled_fixed_text_layer(
    frame_w: int,
    frame_h: int,
    text: str,
    font,
    fill_rgb: Tuple[int, int, int],
    alpha: float,
    x: int,
    y: int,
    scale: float = 1.0,
) -> np.ndarray:
    """
    Render a single fixed-position word with optional scale and subtle glow.
    """
    canvas_w = max(320, frame_w // 2)
    canvas_h = max(120, frame_h // 5)

    layer = render_single_text_layer(
        canvas_w=canvas_w,
        canvas_h=canvas_h,
        text=text,
        font=font,
        fill_rgb=fill_rgb,
        alpha=alpha,
        scale=scale,
    )

    out = np.zeros((frame_h, frame_w, 4), dtype=np.uint8)
    paste_layer_rgba(out, layer, x, y)
    return out



def final_problem_handoff_alpha(post_zero_elapsed: float, global_alpha: float) -> float:
    """
    Continuous alpha for the final PROBLEMS word.

    Instead of:
      loading PROBLEMS alpha low -> suddenly alpha 1.0 in final stage

    This does:
      alpha floor -> smooth rebuild -> stable full title -> shatter
    """
    p = smoothstep(0.0, max(0.001, PROBLEMS_HANDOFF_REBUILD_SECONDS), post_zero_elapsed)
    a = lerp(PROBLEMS_ZERO_HANDOFF_ALPHA, PROBLEMS_FINAL_READY_ALPHA, p)
    return global_alpha * clamp(a, 0.0, 1.0)


def make_handoff_light_sweep(
    frame_w: int,
    frame_h: int,
    x: int,
    y: int,
    text_w: int,
    text_h: int,
    progress: float,
) -> np.ndarray:
    """
    Subtle horizontal light sweep over the final PROBLEMS word.
    This hides the alpha handoff and makes the transition feel intentional.
    """
    out = np.zeros((frame_h, frame_w, 4), dtype=np.uint8)
    if not PROBLEMS_HANDOFF_LIGHT_SWEEP:
        return out

    p = clamp(progress, 0.0, 1.0)
    if p <= 0.001 or p >= 0.999:
        return out

    yy, xx = np.indices((frame_h, frame_w), dtype=np.float32)

    sweep_x = lerp(x - text_w * 0.25, x + text_w * 1.15, p)
    width = max(12.0, text_w * PROBLEMS_HANDOFF_SWEEP_WIDTH)

    band = np.exp(-((xx - sweep_x) ** 2) / (2.0 * width * width))
    vertical = np.exp(-((yy - (y + text_h * 0.46)) ** 2) / (2.0 * max(10.0, text_h * 0.35) ** 2))
    alpha = band * vertical * PROBLEMS_HANDOFF_SWEEP_ALPHA * (1.0 - abs(p - 0.5) * 0.45)

    out[:, :, :3] = 255
    out[:, :, 3] = np.clip(alpha * 255, 0, 255).astype(np.uint8)
    return out


def build_auto_problems_counter_overlay(
    frame_w: int,
    frame_h: int,
    current_time: float,
    preset: dict,
    style: dict,
) -> Optional[np.ndarray]:
    """
    Code-driven counter with random loading speed and polished Hollywood ending.

    Optimized in this version:
    - remove the % character: show 100, 99, 98... instead of 100%, 99%, 98%
    - center every number horizontally above the fixed PROBLEMS word
    - keep the final PROBLEMS handoff smooth
    - preserve Hollywood flash/crack/shard dissolve ending
    """
    elapsed = current_time - PROBLEMS_COUNTER_START_TIME
    state = get_random_counter_state(elapsed)
    if state is None:
        return None

    current_value, next_value, local_p, display_elapsed = state
    roll_p = smoothstep(0.0, 1.0, local_p)

    anchor_x = int(frame_w * preset["anchor_left_ratio"])
    anchor_y = int(frame_h * preset["anchor_top_ratio"])

    percent_font_size = max(16, int(style["font_size"] * 1.35))
    problems_font_size = max(14, int(style["font_size"] * 0.90))

    percent_font = load_font(FONT_PATH, percent_font_size)
    problems_font = load_font(FONT_PATH, problems_font_size)

    problems_text_w, problems_text_h = measure_text_size("PROBLEMS", problems_font)
    problems_center_x = anchor_x + problems_text_w / 2.0

    overlay = np.zeros((frame_h, frame_w, 4), dtype=np.uint8)
    fill_rgb = style["fill_rgb"]

    post_zero_elapsed = max(0.0, elapsed - PROBLEMS_COUNTER_DURATION)
    stage_zero_hold_end = PROBLEMS_ZERO_HOLD_SECONDS
    stage_final_hold_end = PROBLEMS_ZERO_HOLD_SECONDS + PROBLEMS_FINAL_SOLO_HOLD_SECONDS
    stage_final_dissolve_end = stage_final_hold_end + PROBLEMS_FINAL_DISSOLVE_SECONDS

    is_loading = elapsed < PROBLEMS_COUNTER_DURATION
    is_zero_hold = (not is_loading) and post_zero_elapsed < stage_zero_hold_end
    is_final_solo = (not is_loading) and (stage_zero_hold_end <= post_zero_elapsed < stage_final_hold_end)
    is_final_dissolve = (not is_loading) and (stage_final_hold_end <= post_zero_elapsed <= stage_final_dissolve_end)

    intro_alpha = smoothstep(0.00, 0.35, elapsed)
    outro_alpha = 1.0
    if is_final_dissolve:
        dissolve_stage_p = clamp(
            (post_zero_elapsed - stage_final_hold_end) / max(0.001, PROBLEMS_FINAL_DISSOLVE_SECONDS),
            0.0, 1.0
        )
        outro_alpha = 1.0 - smoothstep(0.70, 1.00, dissolve_stage_p)

    global_alpha = intro_alpha * outro_alpha
    roll_dist = int(frame_h * PERCENT_ROLL_DISTANCE_RATIO)

    # --------------------------
    # Stage A: loading 100 -> 0
    # --------------------------
    if is_loading:
        interp_percent = lerp(float(current_value), float(next_value), roll_p)
        problems_alpha_by_percent = percent_to_problem_alpha(interp_percent)

        # Current number exits downward.
        current_text = f"{current_value}"
        current_y = anchor_y + int(roll_p * roll_dist)
        current_alpha = global_alpha * (1.0 - smoothstep(0.58, 1.00, local_p))
        current_scale = lerp(1.00, PERCENT_ROLL_SCALE_OUT, roll_p)
        current_x = centered_x_for_text(problems_center_x, current_text, percent_font, current_scale)

        current_layer = render_single_text_layer(
            canvas_w=max(220, frame_w // 2),
            canvas_h=max(140, frame_h // 5),
            text=current_text,
            font=percent_font,
            fill_rgb=fill_rgb,
            alpha=current_alpha,
            scale=current_scale,
        )

        if PERCENT_ROLL_MOTION_BLUR and PERCENT_ROLL_BLUR_STRENGTH > 1 and current_alpha > 0.02 and current_value > 0:
            current_layer = apply_vector_motion_blur(current_layer, dx=0.0, dy=1.0, strength=PERCENT_ROLL_BLUR_STRENGTH)

        paste_layer_rgba(overlay, current_layer, current_x, current_y)

        # Next number enters from above.
        if current_value > 0:
            next_text = f"{next_value}"
            next_y = anchor_y - roll_dist + int(roll_p * roll_dist)
            next_alpha = global_alpha * smoothstep(0.16, 0.76, local_p)
            next_scale = lerp(PERCENT_ROLL_SCALE_IN, 1.00, roll_p)
            next_x = centered_x_for_text(problems_center_x, next_text, percent_font, next_scale)

            next_layer = render_single_text_layer(
                canvas_w=max(220, frame_w // 2),
                canvas_h=max(140, frame_h // 5),
                text=next_text,
                font=percent_font,
                fill_rgb=fill_rgb,
                alpha=next_alpha,
                scale=next_scale,
            )

            if PERCENT_ROLL_MOTION_BLUR and PERCENT_ROLL_BLUR_STRENGTH > 1 and next_alpha > 0.02:
                next_layer = apply_vector_motion_blur(next_layer, dx=0.0, dy=1.0, strength=PERCENT_ROLL_BLUR_STRENGTH)

            paste_layer_rgba(overlay, next_layer, next_x, next_y)

        problems_y = anchor_y + int(percent_font_size * PROBLEMS_LINE_GAP_RATIO)
        problems_alpha = global_alpha * problems_alpha_by_percent
        if current_value <= 1:
            problems_alpha = max(problems_alpha, global_alpha * PROBLEMS_ZERO_HANDOFF_ALPHA)

        problems_layer = render_fixed_text_layer(
            frame_w=frame_w,
            frame_h=frame_h,
            text="PROBLEMS",
            font=problems_font,
            fill_rgb=fill_rgb,
            alpha=problems_alpha,
            x=anchor_x,
            y=problems_y,
        )

        paste_layer_rgba(overlay, problems_layer, 0, 0)
        return overlay

    # --------------------------
    # Stage B: 0 + PROBLEMS
    # --------------------------
    problems_y = anchor_y + int(percent_font_size * PROBLEMS_LINE_GAP_RATIO)

    if is_zero_hold:
        # 0 fades out while PROBLEMS rebuilds its opacity.
        zero_text = "0"
        zero_fade_p = clamp(post_zero_elapsed / max(0.001, ZERO_PERCENT_HANDOFF_FADE_SECONDS), 0.0, 1.0)
        zero_x = centered_x_for_text(problems_center_x, zero_text, percent_font, 1.0)

        percent_layer = render_single_text_layer(
            canvas_w=max(220, frame_w // 2),
            canvas_h=max(140, frame_h // 5),
            text=zero_text,
            font=percent_font,
            fill_rgb=fill_rgb,
            alpha=global_alpha * (1.0 - smoothstep(0.0, 1.0, zero_fade_p)),
            scale=1.0,
        )
        paste_layer_rgba(overlay, percent_layer, zero_x, anchor_y)

        problems_alpha = final_problem_handoff_alpha(post_zero_elapsed, global_alpha)
        problems_layer = render_fixed_text_layer(
            frame_w=frame_w,
            frame_h=frame_h,
            text="PROBLEMS",
            font=problems_font,
            fill_rgb=fill_rgb,
            alpha=problems_alpha,
            x=anchor_x,
            y=problems_y,
        )

        sweep_p = clamp(post_zero_elapsed / max(0.001, PROBLEMS_HANDOFF_REBUILD_SECONDS), 0.0, 1.0)
        sweep = make_handoff_light_sweep(
            frame_w=frame_w,
            frame_h=frame_h,
            x=anchor_x,
            y=problems_y,
            text_w=max(260, int(problems_text_w * 1.18)),
            text_h=max(60, int(problems_text_h * 1.50)),
            progress=sweep_p,
        )
        paste_layer_rgba(overlay, sweep, 0, 0)
        paste_layer_rgba(overlay, problems_layer, 0, 0)
        return overlay

    # --------------------------
    # Stage C: final solo PROBLEMS
    # --------------------------
    if is_final_solo:
        solo_p = clamp(
            (post_zero_elapsed - stage_zero_hold_end) / max(0.001, PROBLEMS_FINAL_SOLO_HOLD_SECONDS),
            0.0, 1.0
        )

        pulse = 1.0 + math.sin(solo_p * math.pi * 2.0) * (PROBLEMS_FINAL_PULSE_SCALE - 1.0)
        pulse_alpha = 1.0 + math.sin(solo_p * math.pi * 2.0) * PROBLEMS_FINAL_PULSE_ALPHA

        continuous_alpha = final_problem_handoff_alpha(post_zero_elapsed, global_alpha)

        # Keep pulse visually centered around the original PROBLEMS center.
        pulse_x = int(round(anchor_x - (problems_text_w * (pulse - 1.0)) / 2.0))

        problems_core = render_scaled_fixed_text_layer(
            frame_w=frame_w,
            frame_h=frame_h,
            text="PROBLEMS",
            font=problems_font,
            fill_rgb=fill_rgb,
            alpha=continuous_alpha * clamp(pulse_alpha, 0.0, 1.15),
            x=pulse_x,
            y=problems_y,
            scale=pulse,
        )

        sweep_p = clamp(post_zero_elapsed / max(0.001, PROBLEMS_HANDOFF_REBUILD_SECONDS), 0.0, 1.0)
        sweep = make_handoff_light_sweep(
            frame_w=frame_w,
            frame_h=frame_h,
            x=anchor_x,
            y=problems_y,
            text_w=max(260, int(problems_text_w * 1.18)),
            text_h=max(60, int(problems_text_h * 1.50)),
            progress=sweep_p,
        )

        glow = make_glow_from_rgba(problems_core, blur_radius=21, alpha_mul=0.18)
        paste_layer_rgba(overlay, glow, 0, 0)
        paste_layer_rgba(overlay, sweep, 0, 0)
        paste_layer_rgba(overlay, problems_core, 0, 0)
        return overlay

    # --------------------------
    # Stage D: final shatter / dissolve
    # --------------------------
    if is_final_dissolve:
        dissolve_p = clamp(
            (post_zero_elapsed - stage_final_hold_end) / max(0.001, PROBLEMS_FINAL_DISSOLVE_SECONDS),
            0.0, 1.0
        )

        base_alpha = final_problem_handoff_alpha(post_zero_elapsed, global_alpha)
        base_problems = render_scaled_fixed_text_layer(
            frame_w=frame_w,
            frame_h=frame_h,
            text="PROBLEMS",
            font=problems_font,
            fill_rgb=fill_rgb,
            alpha=base_alpha,
            x=anchor_x,
            y=problems_y,
            scale=1.0 + 0.02 * smoothstep(0.0, 1.0, dissolve_p),
        )

        shattered = apply_fragment_shatter_to_rgba(base_problems, progress=dissolve_p, seed=177)

        glow = make_glow_from_rgba(shattered, blur_radius=33, alpha_mul=0.28 * (1.0 - dissolve_p * 0.40))
        paste_layer_rgba(overlay, glow, 0, 0)
        paste_layer_rgba(overlay, shattered, 0, 0)
        return overlay

    return overlay


def alpha_overlay_bgr(frame_bgr: np.ndarray, overlay_rgba: np.ndarray) -> np.ndarray:
    overlay_rgb = overlay_rgba[:, :, :3].astype(np.float32)
    alpha = overlay_rgba[:, :, 3:4].astype(np.float32) / 255.0
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    out_rgb = overlay_rgb * alpha + frame_rgb * (1.0 - alpha)
    out_rgb = np.clip(out_rgb, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)


def add_subtle_vignette(frame_bgr: np.ndarray, strength: float = 0.08) -> np.ndarray:
    if not ENABLE_VIGNETTE:
        return frame_bgr

    h, w = frame_bgr.shape[:2]
    x = np.linspace(-1, 1, w)
    y = np.linspace(-1, 1, h)
    xx, yy = np.meshgrid(x, y)
    radius = np.sqrt(xx * xx + yy * yy)

    vignette = 1.0 - np.clip((radius - 0.25) * strength, 0, 0.20)
    vignette = vignette[:, :, None]
    out = frame_bgr.astype(np.float32) * vignette
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_light_wrap_to_frame(frame_bgr: np.ndarray, overlay_rgba: np.ndarray, strength: float = 0.035, radius: int = 8) -> np.ndarray:
    if not ENABLE_LIGHT_WRAP:
        return frame_bgr

    alpha = overlay_rgba[:, :, 3]
    if alpha.max() == 0:
        return frame_bgr

    glow = cv2.GaussianBlur(alpha, (0, 0), sigmaX=radius, sigmaY=radius).astype(np.float32) / 255.0
    glow = glow[:, :, None]

    frame_float = frame_bgr.astype(np.float32)
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gray = gray[:, :, None]

    wrap = 255.0 * glow * strength * (0.55 + gray * 0.30)
    out = frame_float + wrap
    return np.clip(out, 0, 255).astype(np.uint8)


def mux_audio(input_video: str, temp_video: str, output_video: str) -> None:
    cmd = [
        "ffmpeg", "-y",
        "-i", temp_video,
        "-i", input_video,
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        output_video,
    ]
    print("FFmpeg mux audio:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def build_preset_and_style(frame_w: int, frame_h: int):
    is_vertical = frame_h > frame_w

    if is_vertical:
        preset = {
            # Text moves into fixed position only in the first 14% of each lyric.
            "settle_progress": 0.14,

            "scale_start": 0.86,
            "scale_end": 1.00,

            "anchor_left_ratio": 0.055,
            "anchor_top_ratio": 0.065,

            "far_blur": 0.08,
            "near_blur": 0.20,

            "vector_blur_start": 1,
            "vector_blur_end": 2,

            "burst_start": 0.965,
            "burst_alpha_power": 1.02,

            "vp_offset_x": -20,
            "vp_offset_y": 0,
        }

        base_font_size = int(frame_h * 0.055)
        safe_margin = int(frame_h * 0.032)
        overscan = int(frame_h * 0.018)

        style = {
            "font_size": base_font_size,
            "fill_rgb": (245, 245, 245),
            "fill_alpha": 255,

            "line_scale_single": 1.08,
            "line_scale_small": 0.58,
            "line_scale_mid": 0.80,
            "line_scale_big": 1.08,

            "soft_shadow_alpha": 0,
            "soft_shadow_dx": 0,
            "soft_shadow_dy": 0,

            "word_spacing": int(frame_w * 0.010),
            "line_gap": int(frame_h * 0.005),
            "pad_x": int(frame_h * 0.008),
            "pad_y": int(frame_h * 0.006),

            "safe_margin": safe_margin,
            "transform_overscan": overscan,

            "word_reveal_portion": 0.62,
            "word_reveal_overlap": 1.9,
        }
    else:
        preset = {
            "settle_progress": 0.14,

            "scale_start": 0.90,
            "scale_end": 1.00,

            "anchor_left_ratio": 0.075,
            "anchor_top_ratio": 0.065,

            "far_blur": 0.06,
            "near_blur": 0.18,

            "vector_blur_start": 1,
            "vector_blur_end": 2,

            "burst_start": 0.965,
            "burst_alpha_power": 1.02,

            "vp_offset_x": -18,
            "vp_offset_y": 8,
        }

        base_font_size = int(frame_h * 0.094)
        safe_margin = int(frame_h * 0.040)
        overscan = int(frame_h * 0.026)

        style = {
            "font_size": base_font_size,
            "fill_rgb": (245, 245, 245),
            "fill_alpha": 255,

            "line_scale_single": 1.08,
            "line_scale_small": 0.58,
            "line_scale_mid": 0.80,
            "line_scale_big": 1.08,

            "soft_shadow_alpha": 0,
            "soft_shadow_dx": 0,
            "soft_shadow_dy": 0,

            "word_spacing": int(frame_w * 0.006),
            "line_gap": int(frame_h * 0.005),
            "pad_x": int(frame_h * 0.008),
            "pad_y": int(frame_h * 0.006),

            "safe_margin": safe_margin,
            "transform_overscan": overscan,

            "word_reveal_portion": 0.62,
            "word_reveal_overlap": 1.9,
        }

    return preset, style


def render_video() -> None:
    if not Path(INPUT_VIDEO).exists():
        raise FileNotFoundError(f"Không tìm thấy video input: {INPUT_VIDEO}")

    if AUTO_PROBLEMS_COUNTER_MODE:
        lyrics = []
    else:
        lyrics = load_lyrics(LYRICS_JSON)

    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được video: {INPUT_VIDEO}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        raise RuntimeError("Không đọc được FPS từ video.")

    total_to_render = min(total_frames, MAX_FRAMES) if MAX_FRAMES is not None else total_frames
    print(f"Input video: {frame_w}x{frame_h}, FPS={fps:.3f}, frames={total_frames}")
    print(f"Render frames: {total_to_render}")

    preset, style = build_preset_and_style(frame_w, frame_h)

    writer = cv2.VideoWriter(
        TEMP_VIDEO_NO_AUDIO,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_w, frame_h),
    )
    if not writer.isOpened():
        raise RuntimeError("Không tạo được temp video writer.")

    vp_smoother = SmoothPoint(alpha=0.025)
    fallback_vp = (int(frame_w * 0.52), int(frame_h * 0.20))
    current_vp = fallback_vp

    last_text = None
    cached_layout: Optional[TextLayout] = None

    for frame_idx in tqdm(range(total_to_render), desc="Rendering V5 Smooth Fixed Top Left"):
        ret, frame = cap.read()
        if not ret:
            break

        current_time = frame_idx / fps
        active = find_active_lyric(lyrics, current_time)

        if ENABLE_VANISHING_POINT:
            if frame_idx % VANISHING_POINT_EVERY_N_FRAMES == 0:
                raw_vp = detect_vanishing_point(frame)
                current_vp = vp_smoother.update(raw_vp)
            else:
                current_vp = vp_smoother.update(current_vp)
        else:
            current_vp = fallback_vp

        frame = add_subtle_vignette(frame, strength=0.08)

        if AUTO_PROBLEMS_COUNTER_MODE:
            overlay = build_auto_problems_counter_overlay(
                frame_w=frame_w,
                frame_h=frame_h,
                current_time=current_time,
                preset=preset,
                style=style,
            )

            if overlay is not None:
                frame = apply_light_wrap_to_frame(frame, overlay, strength=0.035, radius=8)
                frame = alpha_overlay_bgr(frame, overlay)

        elif active:
            progress = clamp((current_time - active.start) / (active.end - active.start), 0.0, 1.0)

            if active.text != last_text:
                cached_layout = build_text_layout(active.text, FONT_PATH, style)
                last_text = active.text

            overlay = build_overlay_for_frame_v5(
                layout=cached_layout,
                frame_w=frame_w,
                frame_h=frame_h,
                progress=progress,
                preset=preset,
                style=style,
                vanishing_point=current_vp,
            )

            frame = apply_light_wrap_to_frame(frame, overlay, strength=0.035, radius=8)
            frame = alpha_overlay_bgr(frame, overlay)

        if DEBUG_DRAW_VANISHING_POINT:
            cv2.circle(frame, current_vp, 10, (0, 0, 255), -1)
            cv2.putText(frame, f"VP {current_vp}", (current_vp[0] + 12, current_vp[1] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        writer.write(frame)

    cap.release()
    writer.release()

    print("Ghép audio gốc và encode H.264...")
    mux_audio(INPUT_VIDEO, TEMP_VIDEO_NO_AUDIO, OUTPUT_VIDEO)

    try:
        Path(TEMP_VIDEO_NO_AUDIO).unlink()
    except Exception:
        pass

    print(f"Hoàn tất: {OUTPUT_VIDEO}")


if __name__ == "__main__":
    render_video()

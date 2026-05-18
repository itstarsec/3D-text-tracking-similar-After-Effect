import argparse
import json
import math
import subprocess
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# ============================================================
# POV LYRIC SAMPLE STYLE V10.4
# SAMPLE-MATCHED FORWARD FLOW / MULTI-LYRICS OVERLAP / VP LOCKED
# ============================================================
#
# Optimized from v10.1 based on the sample-video behavior:
# - text starts far near the road vanishing point
# - words reveal progressively while the whole phrase is already moving
# - phrase flows forward toward the viewer, grows larger, tilts slightly
# - phrase blasts past the frame, usually toward upper-left
# - multiple lyric segments can overlap, so the previous phrase can still
#   fly out while the next phrase has already spawned
# - vanishing point is detected and locked per lyric segment to reduce jitter
#
# Usage:
# python render_pov_lyrics_sample_style_v10_4_sample_forward_flow.py ^
#   --input input.mp4 ^
#   --lyrics lyrics.json ^
#   --output output_v10_4.mp4 ^
#   --profile sample
# ============================================================

DEFAULT_INPUT_VIDEO = "input.mp4"
DEFAULT_LYRICS_JSON = "lyrics.json"
DEFAULT_OUTPUT_VIDEO = "output_pov_lyric_sample_style_v10_4_sample_forward_flow.mp4"
DEFAULT_FONT_PATH = "fonts/Anton-Regular.ttf"
TEMP_VIDEO_NO_AUDIO = "_temp_v10_4_sample_forward_flow.mp4"

FORCE_UPPERCASE = True
MAX_FRAMES = None

ENABLE_VANISHING_POINT = True
ENABLE_LIGHT_WRAP = True
ENABLE_VIGNETTE = True
ENABLE_CAMERA_PUSH = True
ENABLE_GHOST_TRAIL = True
ENABLE_VECTOR_MOTION_BLUR = True
ENABLE_DEPTH_BLUR = True
DEBUG_DRAW_VP = False

VANISHING_POINT_EVERY_N_FRAMES = 6


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


@dataclass
class TextLayout:
    width: int
    height: int
    words: List[WordBox]
    center_x: float
    center_y: float


class MedianPointSmoother:
    def __init__(self, maxlen: int = 7):
        self.points: Deque[Tuple[int, int]] = deque(maxlen=maxlen)

    def update(self, point: Tuple[int, int]) -> Tuple[int, int]:
        self.points.append(point)
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        return int(np.median(xs)), int(np.median(ys))


def clamp(v: float, min_v: float, max_v: float) -> float:
    return max(min_v, min(max_v, v))


def lerp(a: float, b: float, p: float) -> float:
    return a + (b - a) * p


def smoothstep(edge0: float, edge1: float, x: float) -> float:
    if abs(edge1 - edge0) < 1e-9:
        return 1.0 if x >= edge1 else 0.0
    x = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def ease_in_cubic(x: float) -> float:
    x = clamp(x, 0.0, 1.0)
    return x ** 3


def ease_in_quart(x: float) -> float:
    x = clamp(x, 0.0, 1.0)
    return x ** 4


def ease_out_cubic(x: float) -> float:
    x = clamp(x, 0.0, 1.0)
    return 1.0 - (1.0 - x) ** 3


def ease_in_out_cubic(x: float) -> float:
    x = clamp(x, 0.0, 1.0)
    if x < 0.5:
        return 4.0 * x * x * x
    return 1.0 - ((-2.0 * x + 2.0) ** 3) / 2.0


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


def active_segment_indices(segments: List[LyricSegment], t: float) -> List[int]:
    return [idx for idx, seg in enumerate(segments) if seg.start <= t <= seg.end]


def detect_vanishing_point(frame_bgr: np.ndarray) -> Tuple[int, int]:
    h, w = frame_bgr.shape[:2]
    fallback = (int(w * 0.52), int(h * 0.30))

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
        [int(work_w * 0.75), int(work_h * 0.22)],
        [int(work_w * 0.25), int(work_h * 0.22)],
    ], dtype=np.int32)
    cv2.fillPoly(mask, [roi], 255)
    edges = cv2.bitwise_and(edges, mask)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=42,
        minLineLength=int(work_w * 0.10),
        maxLineGap=38,
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
            if 0 <= px <= work_w and int(work_h * 0.04) <= py <= int(work_h * 0.65):
                intersections.append((px, py))

    if not intersections:
        return fallback

    xs = np.array([p[0] for p in intersections], dtype=np.float32)
    ys = np.array([p[1] for p in intersections], dtype=np.float32)
    vp_x = int(clamp(float(np.median(xs)) / scale, 0, w - 1))
    vp_y = int(clamp(float(np.median(ys)) / scale, 0, h - 1))
    return vp_x, vp_y


def load_font(font_path: str, font_size: int):
    p = Path(font_path)
    if p.exists():
        return ImageFont.truetype(str(p), font_size)
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
    pad_x = style["pad_x"] + style["safe_margin"]
    pad_y = style["pad_y"] + style["safe_margin"]
    word_spacing_base = style["word_spacing"]
    line_gap_base = style["line_gap"]

    y = pad_y
    max_w = 0
    word_index = 0
    min_x = 10**9
    min_y = 10**9
    max_x = 0
    max_y = 0

    for line_index, line in enumerate(lines):
        parts = [p for p in line.split(" ") if p.strip()]
        scale = get_line_scale(line_index, total_lines, style)
        font_size = max(8, int(style["font_size"] * scale))
        font = load_font(font_path, font_size)
        ascent, descent = font.getmetrics()
        line_height = ascent + descent
        word_spacing = max(2, int(word_spacing_base * scale))
        x = pad_x

        if not parts:
            y += line_height + line_gap_base
            continue

        line_bottom = y + line_height
        for token in parts:
            bbox = draw.textbbox((0, 0), token, font=font, stroke_width=0)
            w = bbox[2] - bbox[0]
            h = max(bbox[3] - bbox[1], line_height)
            words.append(WordBox(
                text=token,
                x=x,
                y=y,
                w=w,
                h=h,
                index=word_index,
                line_index=line_index,
                font_size=font_size,
            ))
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x + w)
            max_y = max(max_y, y + h)
            x += w + word_spacing
            word_index += 1
            max_w = max(max_w, x - word_spacing)
            line_bottom = max(line_bottom, y + h)

        y = line_bottom + max(1, int(line_gap_base * scale))

    total_w = int(max_w + pad_x + style["safe_margin"])
    total_h = int(y + style["safe_margin"])
    center_x = (min_x + max_x) / 2 if words else total_w / 2
    center_y = (min_y + max_y) / 2 if words else total_h / 2
    return TextLayout(total_w, total_h, words, center_x, center_y)


def render_progressive_text_rgba(layout: TextLayout, font_path: str, style: dict, progress: float, rush_p: float) -> np.ndarray:
    canvas = Image.new("RGBA", (layout.width, layout.height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    words = layout.words
    n = len(words)
    if n == 0:
        return np.array(canvas)

    reveal_portion = style["word_reveal_end"]
    overlap = style["word_reveal_overlap"]
    tail_fade = style["tail_fade_portion"]
    base_slot = reveal_portion / max(1, n)
    fade_len = base_slot * overlap

    group_tail_alpha = 1.0
    if progress > 1.0 - tail_fade:
        group_tail_alpha = 1.0 - smoothstep(1.0 - tail_fade, 1.0, progress)

    for word in words:
        start = word.index * base_slot
        end = start + fade_len
        alpha = 1.0 if progress >= reveal_portion else smoothstep(start, end, progress)
        alpha *= group_tail_alpha
        if alpha <= 0.001:
            continue

        word_ratio = word.index / max(1, n - 1)
        font_size = max(8, int(word.font_size * (1.0 + rush_p * style["per_word_rush_scale"] * (0.3 + word_ratio * 0.45))))
        font = load_font(font_path, font_size)

        cx = word.x + word.w / 2.0
        cy = word.y + word.h / 2.0
        vx = cx - layout.center_x
        vy = cy - layout.center_y
        mag = math.sqrt(vx * vx + vy * vy)
        if mag < 1e-6:
            vx, vy, mag = 1.0, -0.7, 1.0
        nx, ny = vx / mag, vy / mag
        local_offset = rush_p * style["per_word_rush_offset"] * (0.4 + word_ratio * 0.35)
        tx = word.x + nx * local_offset
        ty = word.y + ny * local_offset * 0.75 - rush_p * style["per_word_rush_lift"]

        if style["shadow_alpha"] > 0:
            draw.text(
                (tx + style["shadow_dx"], ty + style["shadow_dy"]),
                word.text,
                font=font,
                fill=(0, 0, 0, int(style["shadow_alpha"] * alpha)),
            )
        if style["extrude_alpha"] > 0:
            draw.text(
                (tx + style["extrude_dx"], ty + style["extrude_dy"]),
                word.text,
                font=font,
                fill=(226, 226, 226, int(style["extrude_alpha"] * alpha)),
            )
        draw.text(
            (tx, ty),
            word.text,
            font=font,
            fill=(style["fill_rgb"][0], style["fill_rgb"][1], style["fill_rgb"][2], int(style["fill_alpha"] * alpha)),
        )

    return np.array(canvas)


def pad_rgba(img: np.ndarray, pad: int) -> np.ndarray:
    if pad <= 0:
        return img
    return cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))


def resize_rgba(img: np.ndarray, scale: float) -> np.ndarray:
    h, w = img.shape[:2]
    return cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_LINEAR)


def apply_perspective_billboard(rgba: np.ndarray, progress: float, preset: dict) -> np.ndarray:
    h, w = rgba.shape[:2]
    p = clamp(progress, 0.0, 1.0)
    skew = lerp(preset["skew_start"], preset["skew_end"], p)
    side_depth = lerp(preset["side_depth_start"], preset["side_depth_end"], p)
    top_taper = lerp(preset["top_taper_start"], preset["top_taper_end"], p)

    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([
        [side_depth + top_taper, int(h * skew)],
        [w - side_depth - top_taper, 0],
        [w, h],
        [0, int(h * (1.0 - skew))],
    ])
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(rgba, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))


def apply_depth_blur(rgba: np.ndarray, progress: float, preset: dict) -> np.ndarray:
    if not ENABLE_DEPTH_BLUR:
        return rgba
    p = clamp(progress, 0.0, 1.0)
    far_blur = preset["far_blur"] * max(0.0, 1.0 - p)
    near_blur = preset["near_blur"] * max(0.0, (p - preset["near_blur_start"]) / max(1e-6, 1.0 - preset["near_blur_start"]))
    blur = far_blur + near_blur
    if blur <= 0.18:
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


def paste_rgba(dest: np.ndarray, src: np.ndarray, x: int, y: int, alpha_mul: float = 1.0) -> None:
    th, tw = src.shape[:2]
    x1, y1 = x, y
    x2, y2 = x + tw, y + th
    dx1 = max(0, x1)
    dy1 = max(0, y1)
    dx2 = min(dest.shape[1], x2)
    dy2 = min(dest.shape[0], y2)
    sx1 = max(0, -x1)
    sy1 = max(0, -y1)
    sx2 = sx1 + max(0, dx2 - dx1)
    sy2 = sy1 + max(0, dy2 - dy1)
    if dx1 >= dx2 or dy1 >= dy2:
        return
    patch = src[sy1:sy2, sx1:sx2].copy()
    if alpha_mul != 1.0:
        patch[:, :, 3] = np.clip(patch[:, :, 3].astype(np.float32) * alpha_mul, 0, 255).astype(np.uint8)

    existing = dest[dy1:dy2, dx1:dx2].astype(np.float32)
    incoming = patch.astype(np.float32)
    a_in = incoming[:, :, 3:4] / 255.0
    a_dst = existing[:, :, 3:4] / 255.0
    out_rgb = incoming[:, :, :3] * a_in + existing[:, :, :3] * (1.0 - a_in)
    out_a = a_in + a_dst * (1.0 - a_in)
    existing[:, :, :3] = out_rgb
    existing[:, :, 3:4] = out_a * 255.0
    dest[dy1:dy2, dx1:dx2] = np.clip(existing, 0, 255).astype(np.uint8)


def alpha_overlay_bgr(frame_bgr: np.ndarray, overlay_rgba: np.ndarray) -> np.ndarray:
    overlay_rgb = overlay_rgba[:, :, :3].astype(np.float32)
    alpha = overlay_rgba[:, :, 3:4].astype(np.float32) / 255.0
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    out_rgb = overlay_rgb * alpha + frame_rgb * (1.0 - alpha)
    out_rgb = np.clip(out_rgb, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)


def add_subtle_vignette(frame_bgr: np.ndarray, strength: float = 0.10) -> np.ndarray:
    if not ENABLE_VIGNETTE:
        return frame_bgr
    h, w = frame_bgr.shape[:2]
    x = np.linspace(-1, 1, w)
    y = np.linspace(-1, 1, h)
    xx, yy = np.meshgrid(x, y)
    radius = np.sqrt(xx * xx + yy * yy)
    vignette = 1.0 - np.clip((radius - 0.18) * strength, 0, 0.24)
    vignette = vignette[:, :, None]
    return np.clip(frame_bgr.astype(np.float32) * vignette, 0, 255).astype(np.uint8)


def apply_light_wrap_to_frame(frame_bgr: np.ndarray, overlay_rgba: np.ndarray, strength: float = 0.055, radius: int = 10) -> np.ndarray:
    if not ENABLE_LIGHT_WRAP:
        return frame_bgr
    alpha = overlay_rgba[:, :, 3]
    if alpha.max() == 0:
        return frame_bgr
    glow = cv2.GaussianBlur(alpha, (0, 0), sigmaX=radius, sigmaY=radius).astype(np.float32) / 255.0
    glow = glow[:, :, None]
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gray = gray[:, :, None]
    wrap = 255.0 * glow * strength * (0.55 + gray * 0.35)
    return np.clip(frame_bgr.astype(np.float32) + wrap, 0, 255).astype(np.uint8)


def apply_camera_push(frame_bgr: np.ndarray, scale: float, focus: Tuple[int, int]) -> np.ndarray:
    if not ENABLE_CAMERA_PUSH or abs(scale - 1.0) < 1e-4:
        return frame_bgr
    h, w = frame_bgr.shape[:2]
    fx, fy = focus
    M = np.array([[scale, 0, fx - scale * fx], [0, scale, fy - scale * fy]], dtype=np.float32)
    return cv2.warpAffine(frame_bgr, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)


def build_overlay_for_segment(
    layout: TextLayout,
    frame_w: int,
    frame_h: int,
    progress: float,
    preset: dict,
    style: dict,
    locked_vp: Tuple[int, int],
    segment_stack_index: int,
) -> np.ndarray:
    p = clamp(progress, 0.0, 1.0)

    # Sample-like acceleration: starts moving immediately, then rapidly increases scale.
    base_motion = ease_in_out_cubic(p)
    rush_tail = lerp(ease_in_cubic(p), ease_in_quart(p), preset["rush_aggression"])
    p_motion = lerp(base_motion, rush_tail, preset["rush_mix"])

    text_rgba = render_progressive_text_rgba(layout, style["font_path"], style, p, p_motion)
    text_rgba = pad_rgba(text_rgba, style["transform_overscan"])

    scale = lerp(preset["scale_start"], preset["scale_end"], p_motion)
    transformed = resize_rgba(text_rgba, scale)
    transformed = apply_perspective_billboard(transformed, p_motion, preset)
    transformed = apply_depth_blur(transformed, p_motion, preset)

    vp_x, vp_y = locked_vp

    # Stacking offset lets overlapping phrases spawn in slightly different depth lanes.
    stack_y = segment_stack_index * frame_h * preset["stack_y_ratio"]
    stack_x = segment_stack_index * frame_w * preset["stack_x_ratio"]

    start_x = vp_x + preset["vp_offset_x"] + stack_x
    start_y = vp_y + preset["vp_offset_y"] + stack_y

    end_x = frame_w * preset["end_x_ratio"] + stack_x * 0.15
    end_y = frame_h * preset["end_y_ratio"] + stack_y * 0.10

    x = lerp(start_x, end_x, p_motion)
    y = lerp(start_y, end_y, p_motion)

    move_dx = end_x - start_x
    move_dy = end_y - start_y
    blur_strength = int(lerp(preset["vector_blur_start"], preset["vector_blur_end"], p_motion))
    transformed = apply_vector_motion_blur(transformed, move_dx, move_dy, blur_strength)

    alpha_mul = lerp(preset["alpha_start"], 1.0, min(1.0, p_motion * 1.4))
    if p >= preset["fade_start"]:
        fade_p = smoothstep(preset["fade_start"], 1.0, p)
        alpha_mul *= lerp(1.0, preset["alpha_end"], fade_p)

    transformed[:, :, 3] = np.clip(transformed[:, :, 3].astype(np.float32) * alpha_mul, 0, 255).astype(np.uint8)

    full = np.zeros((frame_h, frame_w, 4), dtype=np.uint8)

    if ENABLE_GHOST_TRAIL and p_motion > 0.05:
        mag = math.sqrt(move_dx * move_dx + move_dy * move_dy)
        if mag > 1e-6:
            nx, ny = move_dx / mag, move_dy / mag
        else:
            nx, ny = -1.0, -1.0
        for i in range(preset["trail_count"], 0, -1):
            trail_ratio = i / preset["trail_count"]
            trail_scale = 1.0 - preset["trail_scale_drop"] * trail_ratio
            trail_alpha = preset["trail_alpha"] * (1.0 - trail_ratio * 0.55)
            trail_dx = -nx * preset["trail_step_px"] * i
            trail_dy = -ny * preset["trail_step_px"] * i * 0.82
            trail = resize_rgba(transformed, max(0.2, trail_scale))
            trail[:, :, 3] = np.clip(trail[:, :, 3].astype(np.float32) * trail_alpha, 0, 255).astype(np.uint8)
            paste_rgba(full, trail, int(x + trail_dx), int(y + trail_dy))

    paste_rgba(full, transformed, int(x), int(y))
    return full


def compose_rgba_over_rgba(base: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    out = base.copy().astype(np.float32)
    src = overlay.astype(np.float32)
    a = src[:, :, 3:4] / 255.0
    out[:, :, :3] = src[:, :, :3] * a + out[:, :, :3] * (1.0 - a)
    out[:, :, 3:4] = np.maximum(out[:, :, 3:4], src[:, :, 3:4])
    return np.clip(out, 0, 255).astype(np.uint8)


def build_profile(frame_w: int, frame_h: int, profile_name: str, font_path: str):
    profile = profile_name.lower().strip()
    if profile not in {"sample", "tiktok", "shorts", "reels", "generic"}:
        raise ValueError("Profile không hợp lệ. Chọn: sample, tiktok, shorts, reels, generic")

    is_vertical = frame_h > frame_w
    if is_vertical:
        profile_map = {
            "sample": {"end_x": -0.10, "end_y": -0.04, "scale_end": 3.05, "vp_y": 0.26},
            "tiktok": {"end_x": -0.08, "end_y": -0.03, "scale_end": 2.90, "vp_y": 0.28},
            "shorts": {"end_x": -0.06, "end_y": -0.02, "scale_end": 2.75, "vp_y": 0.28},
            "reels": {"end_x": -0.08, "end_y": -0.03, "scale_end": 2.85, "vp_y": 0.28},
            "generic": {"end_x": -0.06, "end_y": -0.02, "scale_end": 2.75, "vp_y": 0.28},
        }
        pf = profile_map[profile]
        preset = {
            "scale_start": 0.28,
            "scale_end": pf["scale_end"],
            "end_x_ratio": pf["end_x"],
            "end_y_ratio": pf["end_y"],
            "vp_offset_x": -30,
            "vp_offset_y": int(frame_h * pf["vp_y"]),

            "stack_x_ratio": 0.02,
            "stack_y_ratio": 0.12,

            "skew_start": 0.060,
            "skew_end": 0.006,
            "side_depth_start": 18,
            "side_depth_end": 2,
            "top_taper_start": 22,
            "top_taper_end": 0,

            "far_blur": 0.22,
            "near_blur": 1.10,
            "near_blur_start": 0.78,
            "vector_blur_start": 1,
            "vector_blur_end": 7,

            "trail_count": 4,
            "trail_alpha": 0.34,
            "trail_step_px": 20,
            "trail_scale_drop": 0.10,

            "rush_aggression": 0.08,
            "rush_mix": 0.55,
            "alpha_start": 0.42,
            "fade_start": 0.88,
            "alpha_end": 0.02,

            "bg_push_start": 1.000,
            "bg_push_end": 1.055,
        }
        style = {
            "font_path": font_path,
            "font_size": int(frame_h * 0.060),
            "fill_rgb": (248, 248, 248),
            "fill_alpha": 255,
            "line_scale_single": 1.14,
            "line_scale_small": 0.60,
            "line_scale_mid": 0.84,
            "line_scale_big": 1.16,
            "shadow_alpha": 24,
            "shadow_dx": 2,
            "shadow_dy": 3,
            "extrude_alpha": 52,
            "extrude_dx": 2,
            "extrude_dy": 2,
            "word_spacing": int(frame_w * 0.010),
            "line_gap": int(frame_h * 0.004),
            "pad_x": int(frame_h * 0.010),
            "pad_y": int(frame_h * 0.008),
            "safe_margin": int(frame_h * 0.038),
            "transform_overscan": int(frame_h * 0.120),
            "word_reveal_end": 0.50,
            "word_reveal_overlap": 1.9,
            "tail_fade_portion": 0.05,
            "per_word_rush_scale": 0.10,
            "per_word_rush_offset": frame_h * 0.016,
            "per_word_rush_lift": frame_h * 0.008,
        }
    else:
        preset = {
            "scale_start": 0.45,
            "scale_end": 1.80,
            "end_x_ratio": -0.02,
            "end_y_ratio": -0.03,
            "vp_offset_x": -24,
            "vp_offset_y": int(frame_h * 0.20),
            "stack_x_ratio": 0.02,
            "stack_y_ratio": 0.10,
            "skew_start": 0.045,
            "skew_end": 0.006,
            "side_depth_start": 14,
            "side_depth_end": 2,
            "top_taper_start": 16,
            "top_taper_end": 0,
            "far_blur": 0.18,
            "near_blur": 0.75,
            "near_blur_start": 0.78,
            "vector_blur_start": 1,
            "vector_blur_end": 4,
            "trail_count": 3,
            "trail_alpha": 0.28,
            "trail_step_px": 16,
            "trail_scale_drop": 0.09,
            "rush_aggression": 0.10,
            "rush_mix": 0.50,
            "alpha_start": 0.45,
            "fade_start": 0.90,
            "alpha_end": 0.03,
            "bg_push_start": 1.000,
            "bg_push_end": 1.035,
        }
        style = {
            "font_path": font_path,
            "font_size": int(frame_h * 0.098),
            "fill_rgb": (248, 248, 248),
            "fill_alpha": 255,
            "line_scale_single": 1.12,
            "line_scale_small": 0.60,
            "line_scale_mid": 0.84,
            "line_scale_big": 1.14,
            "shadow_alpha": 22,
            "shadow_dx": 2,
            "shadow_dy": 3,
            "extrude_alpha": 45,
            "extrude_dx": 2,
            "extrude_dy": 2,
            "word_spacing": int(frame_w * 0.006),
            "line_gap": int(frame_h * 0.004),
            "pad_x": int(frame_h * 0.010),
            "pad_y": int(frame_h * 0.008),
            "safe_margin": int(frame_h * 0.040),
            "transform_overscan": int(frame_h * 0.090),
            "word_reveal_end": 0.50,
            "word_reveal_overlap": 1.9,
            "tail_fade_portion": 0.05,
            "per_word_rush_scale": 0.08,
            "per_word_rush_offset": frame_h * 0.012,
            "per_word_rush_lift": frame_h * 0.006,
        }
    return preset, style


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


def render_video(input_video: str, lyrics_json: str, output_video: str, font_path: str, profile: str) -> None:
    if not Path(input_video).exists():
        raise FileNotFoundError(f"Không tìm thấy video input: {input_video}")
    lyrics = load_lyrics(lyrics_json)

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được video: {input_video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0:
        raise RuntimeError("Không đọc được FPS từ video.")

    total_to_render = min(total_frames, MAX_FRAMES) if MAX_FRAMES is not None else total_frames
    print(f"Input video: {frame_w}x{frame_h}, FPS={fps:.3f}, frames={total_frames}")
    print(f"Render frames: {total_to_render}")
    print(f"Profile: {profile}")

    preset, style = build_profile(frame_w, frame_h, profile, font_path)

    writer = cv2.VideoWriter(TEMP_VIDEO_NO_AUDIO, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_w, frame_h))
    if not writer.isOpened():
        raise RuntimeError("Không tạo được temp video writer.")

    vp_smoother = MedianPointSmoother(maxlen=7)
    fallback_vp = (int(frame_w * 0.52), int(frame_h * 0.30))
    smoothed_vp = fallback_vp

    layout_cache: Dict[str, TextLayout] = {}
    locked_vps: Dict[int, Tuple[int, int]] = {}

    for frame_idx in tqdm(range(total_to_render), desc="Rendering V10.4 Sample Forward Flow"):
        ret, frame = cap.read()
        if not ret:
            break

        current_time = frame_idx / fps
        active_indices = active_segment_indices(lyrics, current_time)

        if ENABLE_VANISHING_POINT:
            if frame_idx % VANISHING_POINT_EVERY_N_FRAMES == 0:
                smoothed_vp = vp_smoother.update(detect_vanishing_point(frame))
            else:
                smoothed_vp = vp_smoother.update(smoothed_vp)
        else:
            smoothed_vp = fallback_vp

        frame = add_subtle_vignette(frame, strength=0.10)

        if active_indices:
            # Camera push follows the most recently-started active segment.
            newest_idx = active_indices[-1]
            newest = lyrics[newest_idx]
            newest_progress = clamp((current_time - newest.start) / (newest.end - newest.start), 0.0, 1.0)
            p_motion = lerp(ease_in_cubic(newest_progress), ease_in_quart(newest_progress), preset["rush_aggression"])
            if ENABLE_CAMERA_PUSH:
                bg_scale = lerp(preset["bg_push_start"], preset["bg_push_end"], p_motion)
                focus = locked_vps.get(newest_idx, smoothed_vp)
                frame = apply_camera_push(frame, bg_scale, focus)

            combined_overlay = np.zeros((frame_h, frame_w, 4), dtype=np.uint8)
            for stack_i, seg_idx in enumerate(active_indices):
                seg = lyrics[seg_idx]
                if seg_idx not in locked_vps:
                    locked_vps[seg_idx] = smoothed_vp
                if seg.text not in layout_cache:
                    layout_cache[seg.text] = build_text_layout(seg.text, font_path, style)

                progress = clamp((current_time - seg.start) / (seg.end - seg.start), 0.0, 1.0)
                overlay = build_overlay_for_segment(
                    layout=layout_cache[seg.text],
                    frame_w=frame_w,
                    frame_h=frame_h,
                    progress=progress,
                    preset=preset,
                    style=style,
                    locked_vp=locked_vps[seg_idx],
                    segment_stack_index=stack_i,
                )
                combined_overlay = compose_rgba_over_rgba(combined_overlay, overlay)

            frame = apply_light_wrap_to_frame(frame, combined_overlay, strength=0.055, radius=10)
            frame = alpha_overlay_bgr(frame, combined_overlay)

        if DEBUG_DRAW_VP:
            cv2.circle(frame, smoothed_vp, 8, (0, 0, 255), -1)
            cv2.putText(frame, f"VP {smoothed_vp}", (smoothed_vp[0] + 10, smoothed_vp[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        writer.write(frame)

    cap.release()
    writer.release()

    print("Ghép audio gốc và encode H.264...")
    mux_audio(input_video, TEMP_VIDEO_NO_AUDIO, output_video)

    try:
        Path(TEMP_VIDEO_NO_AUDIO).unlink()
    except Exception:
        pass

    print(f"Hoàn tất: {output_video}")


def parse_args():
    parser = argparse.ArgumentParser(description="Render POV lyric animation v10.4 sample forward flow")
    parser.add_argument("--input", default=DEFAULT_INPUT_VIDEO, help="Input video path")
    parser.add_argument("--lyrics", default=DEFAULT_LYRICS_JSON, help="lyrics.json path")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_VIDEO, help="Output video path")
    parser.add_argument("--font", default=DEFAULT_FONT_PATH, help="Font path")
    parser.add_argument("--profile", default="sample", choices=["sample", "tiktok", "shorts", "reels", "generic"], help="Output style profile")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    render_video(
        input_video=args.input,
        lyrics_json=args.lyrics,
        output_video=args.output,
        font_path=args.font,
        profile=args.profile,
    )

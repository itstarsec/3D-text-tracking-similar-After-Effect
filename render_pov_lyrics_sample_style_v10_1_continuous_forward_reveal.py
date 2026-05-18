import argparse
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# ============================================================
# POV LYRIC SAMPLE STYLE V10.1
# CONTINUOUS FORWARD REVEAL / DIAGONAL POV COUNTER-MOTION
# ============================================================
# Requested behavior:
# - Text starts at position A.
# - As soon as the first words appear, the whole text object is already
#   moving forward toward the viewer.
# - The motion is opposite the road POV rush feeling: the camera seems
#   to move forward into the scene while the text grows and comes out
#   toward the viewer on a diagonal trajectory.
# - Words appear progressively one by one, but the visible phrase keeps
#   drifting diagonally and scaling up the whole time.
# - Once the full phrase is visible, it continues accelerating and blasts
#   past the frame.
# ============================================================

DEFAULT_INPUT_VIDEO = "input.mp4"
DEFAULT_LYRICS_JSON = "lyrics.json"
DEFAULT_OUTPUT_VIDEO = "output_pov_lyric_sample_style_v10_1_continuous_forward_reveal.mp4"
DEFAULT_FONT_PATH = "fonts/Anton-Regular.ttf"
TEMP_VIDEO_NO_AUDIO = "_temp_v10_1_continuous_forward_reveal.mp4"

FORCE_UPPERCASE = True
MAX_FRAMES = None

ENABLE_LIGHT_WRAP = True
ENABLE_VIGNETTE = True
ENABLE_CAMERA_PUSH = True
ENABLE_GHOST_TRAIL = True
ENABLE_VECTOR_MOTION_BLUR = True
ENABLE_DEPTH_BLUR = True
DEBUG_ANCHOR = False


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
    return 1.0 - (1.0 - x) ** 3


def ease_in_cubic(x: float) -> float:
    x = clamp(x, 0.0, 1.0)
    return x ** 3


def ease_in_out_cubic(x: float) -> float:
    x = clamp(x, 0.0, 1.0)
    if x < 0.5:
        return 4 * x * x * x
    return 1 - ((-2 * x + 2) ** 3) / 2


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
        segments.append(LyricSegment(start, end, text))
    segments.sort(key=lambda s: s.start)
    return segments


def find_active_lyric_idx(segments: List[LyricSegment], t: float) -> Optional[int]:
    for idx, seg in enumerate(segments):
        if seg.start <= t <= seg.end:
            return idx
    return None


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

        local_bottom = y + line_height
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
            local_bottom = max(local_bottom, y + h)

        y = local_bottom + max(1, int(line_gap_base * scale))

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
        font = load_font(font_path, max(8, int(word.font_size * (1.0 + rush_p * (style["per_word_rush_scale"] * (0.4 + word_ratio * 0.5))))))

        cx = word.x + word.w / 2.0
        cy = word.y + word.h / 2.0
        vx = cx - layout.center_x
        vy = cy - layout.center_y
        mag = math.sqrt(vx * vx + vy * vy)
        if mag < 1e-6:
            vx, vy, mag = 1.0, -0.7, 1.0
        nx, ny = vx / mag, vy / mag

        local_offset = rush_p * style["per_word_rush_offset"] * (0.45 + word_ratio * 0.45)
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
                fill=(225, 225, 225, int(style["extrude_alpha"] * alpha)),
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
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)


def apply_perspective_tilt(rgba: np.ndarray, tilt_strength: float, preset: dict) -> np.ndarray:
    h, w = rgba.shape[:2]
    q = clamp(tilt_strength, 0.0, 1.0)
    skew = lerp(preset["tilt_skew_min"], preset["tilt_skew_max"], q)
    taper = lerp(preset["tilt_taper_min"], preset["tilt_taper_max"], q)
    side = lerp(preset["tilt_side_min"], preset["tilt_side_max"], q)

    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([
        [side + taper, int(h * skew)],
        [w - side - taper, 0],
        [w, h],
        [0, int(h * (1.0 - skew))],
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(rgba, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))


def apply_depth_blur(rgba: np.ndarray, amount: float) -> np.ndarray:
    if not ENABLE_DEPTH_BLUR or amount <= 0.18:
        return rgba
    k = int(amount * 2) * 2 + 1
    return cv2.GaussianBlur(rgba, (k, k), amount)


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
    c = ksize // 2
    for i in range(ksize):
        off = i - c
        x = int(c + dx * off)
        y = int(c + dy * off)
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
    vignette = 1.0 - np.clip((radius - 0.18) * strength, 0, 0.22)
    vignette = vignette[:, :, None]
    out = frame_bgr.astype(np.float32) * vignette
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_light_wrap_to_frame(frame_bgr: np.ndarray, overlay_rgba: np.ndarray, strength: float = 0.05, radius: int = 10) -> np.ndarray:
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
    out = frame_bgr.astype(np.float32) + wrap
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_camera_push(frame_bgr: np.ndarray, scale: float, focus: Tuple[int, int]) -> np.ndarray:
    if not ENABLE_CAMERA_PUSH or abs(scale - 1.0) < 1e-4:
        return frame_bgr
    h, w = frame_bgr.shape[:2]
    fx, fy = focus
    M = np.array([[scale, 0, fx - scale * fx], [0, scale, fy - scale * fy]], dtype=np.float32)
    return cv2.warpAffine(frame_bgr, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT101)


def compute_stage(progress: float, preset: dict) -> Tuple[str, float]:
    p = clamp(progress, 0.0, 1.0)
    if p <= preset["continuous_end"]:
        return "continuous", ease_out_cubic(p / max(1e-6, preset["continuous_end"]))
    if p <= preset["blast_end"]:
        return "blast", ease_in_out_cubic((p - preset["continuous_end"]) / max(1e-6, preset["blast_end"] - preset["continuous_end"]))
    return "fade", smoothstep(preset["blast_end"], 1.0, p)


def build_overlay_for_frame(layout: TextLayout, frame_w: int, frame_h: int, progress: float, preset: dict, style: dict) -> Tuple[np.ndarray, Tuple[int, int]]:
    stage, t = compute_stage(progress, preset)

    anchor_x = int(frame_w * preset["anchor_left_ratio"])
    anchor_y = int(frame_h * preset["anchor_top_ratio"])

    mid_x = anchor_x + int(frame_w * preset["continuous_dx_ratio"])
    mid_y = anchor_y + int(frame_h * preset["continuous_dy_ratio"])
    exit_x = anchor_x + int(frame_w * preset["exit_dx_ratio"])
    exit_y = anchor_y + int(frame_h * preset["exit_dy_ratio"])

    if stage == "continuous":
        # From the very first frame, the phrase is already drifting forward.
        x = lerp(anchor_x, mid_x, t)
        y = lerp(anchor_y, mid_y, t)
        scale = lerp(preset["continuous_scale_start"], preset["continuous_scale_end"], t)
        alpha_mul = lerp(preset["alpha_start"], 1.0, min(1.0, t * 1.3))
        tilt_strength = lerp(0.18, 0.55, t)
        motion_vec = (mid_x - anchor_x, mid_y - anchor_y)
        blur_strength = int(lerp(2, 4, t))
        depth_blur = lerp(preset["continuous_depth_blur_start"], preset["continuous_depth_blur_end"], t)
        rush_p = t
    elif stage == "blast":
        x = lerp(mid_x, exit_x, t)
        y = lerp(mid_y, exit_y, t)
        scale = lerp(preset["continuous_scale_end"], preset["blast_scale"], t)
        alpha_mul = lerp(1.0, preset["blast_alpha_end"], t)
        tilt_strength = lerp(0.55, 1.0, t)
        motion_vec = (exit_x - mid_x, exit_y - mid_y)
        blur_strength = int(lerp(preset["blast_blur_start"], preset["blast_blur_end"], t))
        depth_blur = lerp(preset["continuous_depth_blur_end"], preset["blast_depth_blur_end"], t)
        rush_p = 1.0
    else:
        x = exit_x
        y = exit_y
        scale = preset["blast_scale"]
        alpha_mul = lerp(preset["blast_alpha_end"], 0.0, t)
        tilt_strength = 1.0
        motion_vec = (0.0, 0.0)
        blur_strength = 0
        depth_blur = preset["blast_depth_blur_end"]
        rush_p = 1.0

    text_rgba = render_progressive_text_rgba(layout, style["font_path"], style, progress, rush_p)
    text_rgba = pad_rgba(text_rgba, style["transform_overscan"])
    transformed = resize_rgba(text_rgba, scale)
    transformed = apply_perspective_tilt(transformed, tilt_strength, preset)
    transformed = apply_depth_blur(transformed, depth_blur)

    if blur_strength > 1:
        transformed = apply_vector_motion_blur(transformed, motion_vec[0], motion_vec[1], blur_strength)

    transformed[:, :, 3] = np.clip(transformed[:, :, 3].astype(np.float32) * alpha_mul, 0, 255).astype(np.uint8)

    full = np.zeros((frame_h, frame_w, 4), dtype=np.uint8)

    if ENABLE_GHOST_TRAIL and stage in ("continuous", "blast"):
        mvx, mvy = motion_vec
        mag = math.sqrt(mvx * mvx + mvy * mvy)
        if mag > 1e-6:
            nx, ny = mvx / mag, mvy / mag
        else:
            nx, ny = -1.0, -1.0

        trail_count = preset["trail_count_continuous"] if stage == "continuous" else preset["trail_count_blast"]
        trail_alpha_base = preset["trail_alpha_continuous"] if stage == "continuous" else preset["trail_alpha_blast"]
        trail_step = preset["trail_step_px_continuous"] if stage == "continuous" else preset["trail_step_px_blast"]

        for i in range(trail_count, 0, -1):
            trail_ratio = i / trail_count
            trail_scale = 1.0 - preset["trail_scale_drop"] * trail_ratio
            trail_alpha = trail_alpha_base * (1.0 - trail_ratio * 0.5)
            trail_dx = -nx * trail_step * i
            trail_dy = -ny * trail_step * i * 0.82
            trail = resize_rgba(transformed, max(0.2, trail_scale))
            trail[:, :, 3] = np.clip(trail[:, :, 3].astype(np.float32) * trail_alpha, 0, 255).astype(np.uint8)
            paste_rgba(full, trail, int(x + trail_dx), int(y + trail_dy))

    paste_rgba(full, transformed, int(x), int(y))
    return full, (anchor_x, anchor_y)


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


def build_profile(frame_w: int, frame_h: int, profile_name: str, font_path: str):
    is_vertical = frame_h > frame_w
    profile = profile_name.lower().strip()
    if profile not in {"tiktok", "shorts", "reels", "generic"}:
        raise ValueError("Profile không hợp lệ. Chọn: tiktok, shorts, reels, generic")

    offsets = {
        "tiktok": {"left": 0.050, "top": 0.090},
        "shorts": {"left": 0.054, "top": 0.092},
        "reels": {"left": 0.052, "top": 0.090},
        "generic": {"left": 0.052, "top": 0.090},
    }
    pf = offsets[profile]

    if is_vertical:
        preset = {
            "continuous_end": 0.70,
            "blast_end": 0.95,

            "anchor_left_ratio": pf["left"],
            "anchor_top_ratio": pf["top"],

            # Text already moves diagonally during reveal.
            "continuous_dx_ratio": -0.10,
            "continuous_dy_ratio": -0.08,

            # Then it blasts harder beyond the frame.
            "exit_dx_ratio": -0.28,
            "exit_dy_ratio": -0.22,

            "continuous_scale_start": 0.84,
            "continuous_scale_end": 1.38,
            "blast_scale": 2.90,

            "alpha_start": 0.34,
            "blast_alpha_end": 0.08,

            "continuous_depth_blur_start": 1.00,
            "continuous_depth_blur_end": 0.16,
            "blast_depth_blur_end": 2.10,

            "blast_blur_start": 5,
            "blast_blur_end": 10,

            "tilt_skew_min": 0.02,
            "tilt_skew_max": 0.14,
            "tilt_taper_min": 0,
            "tilt_taper_max": 32,
            "tilt_side_min": 2,
            "tilt_side_max": 24,

            "trail_count_continuous": 2,
            "trail_alpha_continuous": 0.18,
            "trail_step_px_continuous": 10,
            "trail_count_blast": 4,
            "trail_alpha_blast": 0.34,
            "trail_step_px_blast": 24,
            "trail_scale_drop": 0.10,

            "bg_push_continuous": 1.020,
            "bg_push_blast": 1.060,
        }

        style = {
            "font_path": font_path,
            "font_size": int(frame_h * 0.055),
            "fill_rgb": (248, 248, 248),
            "fill_alpha": 255,

            "line_scale_single": 1.10,
            "line_scale_small": 0.62,
            "line_scale_mid": 0.84,
            "line_scale_big": 1.14,

            "shadow_alpha": 26,
            "shadow_dx": 2,
            "shadow_dy": 3,
            "extrude_alpha": 54,
            "extrude_dx": 2,
            "extrude_dy": 2,

            "word_spacing": int(frame_w * 0.010),
            "line_gap": int(frame_h * 0.005),
            "pad_x": int(frame_h * 0.008),
            "pad_y": int(frame_h * 0.006),
            "safe_margin": int(frame_h * 0.040),
            "transform_overscan": int(frame_h * 0.120),

            "word_reveal_end": 0.66,
            "word_reveal_overlap": 2.0,
            "tail_fade_portion": 0.05,

            "per_word_rush_scale": 0.10,
            "per_word_rush_offset": frame_h * 0.018,
            "per_word_rush_lift": frame_h * 0.008,
        }
    else:
        preset = {
            "continuous_end": 0.70,
            "blast_end": 0.95,
            "anchor_left_ratio": 0.075,
            "anchor_top_ratio": 0.070,
            "continuous_dx_ratio": -0.07,
            "continuous_dy_ratio": -0.06,
            "exit_dx_ratio": -0.16,
            "exit_dy_ratio": -0.14,
            "continuous_scale_start": 0.86,
            "continuous_scale_end": 1.22,
            "blast_scale": 2.20,
            "alpha_start": 0.34,
            "blast_alpha_end": 0.08,
            "continuous_depth_blur_start": 0.80,
            "continuous_depth_blur_end": 0.12,
            "blast_depth_blur_end": 1.60,
            "blast_blur_start": 4,
            "blast_blur_end": 8,
            "tilt_skew_min": 0.02,
            "tilt_skew_max": 0.10,
            "tilt_taper_min": 0,
            "tilt_taper_max": 18,
            "tilt_side_min": 2,
            "tilt_side_max": 16,
            "trail_count_continuous": 2,
            "trail_alpha_continuous": 0.16,
            "trail_step_px_continuous": 8,
            "trail_count_blast": 3,
            "trail_alpha_blast": 0.28,
            "trail_step_px_blast": 18,
            "trail_scale_drop": 0.09,
            "bg_push_continuous": 1.015,
            "bg_push_blast": 1.042,
        }

        style = {
            "font_path": font_path,
            "font_size": int(frame_h * 0.092),
            "fill_rgb": (248, 248, 248),
            "fill_alpha": 255,
            "line_scale_single": 1.10,
            "line_scale_small": 0.62,
            "line_scale_mid": 0.84,
            "line_scale_big": 1.14,
            "shadow_alpha": 24,
            "shadow_dx": 2,
            "shadow_dy": 3,
            "extrude_alpha": 48,
            "extrude_dx": 2,
            "extrude_dy": 2,
            "word_spacing": int(frame_w * 0.006),
            "line_gap": int(frame_h * 0.005),
            "pad_x": int(frame_h * 0.008),
            "pad_y": int(frame_h * 0.006),
            "safe_margin": int(frame_h * 0.040),
            "transform_overscan": int(frame_h * 0.100),
            "word_reveal_end": 0.66,
            "word_reveal_overlap": 2.0,
            "tail_fade_portion": 0.05,
            "per_word_rush_scale": 0.08,
            "per_word_rush_offset": frame_h * 0.013,
            "per_word_rush_lift": frame_h * 0.006,
        }

    return preset, style


def camera_push_scale(progress: float, preset: dict) -> float:
    stage, t = compute_stage(progress, preset)
    if stage == "continuous":
        return lerp(1.0, preset["bg_push_continuous"], t)
    if stage == "blast":
        return lerp(preset["bg_push_continuous"], preset["bg_push_blast"], t)
    return preset["bg_push_blast"]


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

    last_text = None
    cached_layout: Optional[TextLayout] = None

    for frame_idx in tqdm(range(total_to_render), desc="Rendering V10.1 Continuous Forward Reveal"):
        ret, frame = cap.read()
        if not ret:
            break

        current_time = frame_idx / fps
        seg_idx = find_active_lyric_idx(lyrics, current_time)

        frame = add_subtle_vignette(frame, strength=0.10)

        if seg_idx is not None:
            active = lyrics[seg_idx]
            progress = clamp((current_time - active.start) / (active.end - active.start), 0.0, 1.0)

            if active.text != last_text:
                cached_layout = build_text_layout(active.text, font_path, style)
                last_text = active.text

            anchor = (int(frame_w * preset["anchor_left_ratio"]), int(frame_h * preset["anchor_top_ratio"]))
            if ENABLE_CAMERA_PUSH:
                frame = apply_camera_push(frame, camera_push_scale(progress, preset), anchor)

            overlay, anchor = build_overlay_for_frame(cached_layout, frame_w, frame_h, progress, preset, style)
            frame = apply_light_wrap_to_frame(frame, overlay, strength=0.05, radius=10)
            frame = alpha_overlay_bgr(frame, overlay)

            if DEBUG_ANCHOR:
                cv2.circle(frame, anchor, 8, (0, 0, 255), -1)

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
    parser = argparse.ArgumentParser(description="Render POV lyric animation v10.1 continuous forward reveal")
    parser.add_argument("--input", default=DEFAULT_INPUT_VIDEO, help="Input video path")
    parser.add_argument("--lyrics", default=DEFAULT_LYRICS_JSON, help="lyrics.json path")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_VIDEO, help="Output video path")
    parser.add_argument("--font", default=DEFAULT_FONT_PATH, help="Font path")
    parser.add_argument("--profile", default="tiktok", choices=["tiktok", "shorts", "reels", "generic"], help="Output style profile")
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

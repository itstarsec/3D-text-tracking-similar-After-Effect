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
# POV LYRIC SAMPLE STYLE V10.2 - HYBRID V3 MOTION + V10.1 SMOOTH
# ============================================================
#
# Main fixes vs V2:
# 1. Prevent bottom cropping / clipped letters
#    - use font ascent+descent for line advance
#    - add safe canvas margins around the whole text block
#    - add transform overscan padding before perspective/blur
#
# 2. Keep sample-like style
#    - white bold uppercase
#    - left aligned stacked layout
#    - smaller top line / bigger emphasis line
#    - word-by-word reveal
#    - forward-rush POV motion
#
# Input:
#   input.mp4
#   lyrics.json
#
# Output:
#   output_pov_lyric_sample_style_v10_2_hybrid_smooth.mp4
# ============================================================


INPUT_VIDEO = "input.mp4"
LYRICS_JSON = "lyrics.json"
OUTPUT_VIDEO = "output_pov_lyric_sample_style_v10_2_hybrid_smooth.mp4"

FONT_PATH = "fonts/Anton-Regular.ttf"
TEMP_VIDEO_NO_AUDIO = "_temp_v10_2_hybrid_smooth.mp4"

MAX_FRAMES = None
# VANISHING_POINT_EVERY_N_FRAMES = 4
VANISHING_POINT_EVERY_N_FRAMES = 12

ENABLE_VANISHING_POINT = True
ENABLE_OPTICAL_FLOW_SHAKE = False  # smoother default; set True for subtle camera-attached shake
ENABLE_DEPTH_BLUR = True
ENABLE_LIGHT_WRAP = True
ENABLE_VECTOR_MOTION_BLUR = True
ENABLE_VIGNETTE = True

DEBUG_DRAW_VANISHING_POINT = False
FORCE_UPPERCASE = True


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


def find_active_lyric_idx(segments: List[LyricSegment], t: float) -> Optional[int]:
    for idx, seg in enumerate(segments):
        if seg.start <= t <= seg.end:
            return idx
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


def ease_in_cubic(x: float) -> float:
    x = clamp(x, 0.0, 1.0)
    return x * x * x


def ease_in_quart(x: float) -> float:
    x = clamp(x, 0.0, 1.0)
    return x ** 4


class SmoothPoint:
    def __init__(self, alpha: float = 0.06):
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


class SmoothVector:
    def __init__(self, alpha: float = 0.16):
        self.alpha = alpha
        self.x = 0.0
        self.y = 0.0
        self.ready = False

    def update(self, vec: Tuple[float, float]) -> Tuple[float, float]:
        vx, vy = vec
        if not self.ready:
            self.x, self.y = vx, vy
            self.ready = True
        else:
            self.x = self.x * (1 - self.alpha) + vx * self.alpha
            self.y = self.y * (1 - self.alpha) + vy * self.alpha
        return self.x, self.y


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
            if 0 <= px <= work_w and int(work_h * 0.04) <= py <= int(work_h * 0.60):
                intersections.append((px, py))

    if not intersections:
        return fallback

    xs = np.array([p[0] for p in intersections], dtype=np.float32)
    ys = np.array([p[1] for p in intersections], dtype=np.float32)

    vp_x = int(clamp(float(np.median(xs)) / scale, 0, w - 1))
    vp_y = int(clamp(float(np.median(ys)) / scale, 0, h - 1))
    return vp_x, vp_y


class CameraShakeTracker:
    def __init__(self):
        self.prev_gray = None

    def update(self, frame_bgr: np.ndarray) -> Tuple[float, float]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (320, 180), interpolation=cv2.INTER_AREA)

        if self.prev_gray is None:
            self.prev_gray = gray
            return 0.0, 0.0

        prev_pts = cv2.goodFeaturesToTrack(
            self.prev_gray,
            maxCorners=140,
            qualityLevel=0.012,
            minDistance=7,
            blockSize=7,
        )
        if prev_pts is None:
            self.prev_gray = gray
            return 0.0, 0.0

        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray,
            gray,
            prev_pts,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
        )
        self.prev_gray = gray

        if next_pts is None or status is None:
            return 0.0, 0.0

        good_prev = prev_pts[status.flatten() == 1]
        good_next = next_pts[status.flatten() == 1]
        if len(good_prev) < 12:
            return 0.0, 0.0

        flow = good_next - good_prev
        dx = float(np.median(flow[:, 0, 0]))
        dy = float(np.median(flow[:, 0, 1]))
        dx = clamp(dx, -7.0, 7.0)
        dy = clamp(dy, -7.0, 7.0)
        return dx * 2.3, dy * 2.3


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
    """
    Uses ascent+descent to avoid clipping bottom glyphs.
    Adds safe margins around the block.
    """
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

        # Use typographic line height instead of bbox height only.
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
            # Use max of bbox and metrics
            h = max(bbox[3] - bbox[1], line_advance)

            words.append(WordBox(
                text=part,
                x=x,
                y=y,
                w=w,
                h=h,
                index=global_word_index,
                line_index=line_index,
                font_size=font_size,
                ascent=ascent,
                descent=descent,
            ))

            x += w + word_spacing
            global_word_index += 1
            max_w = max(max_w, x - word_spacing)
            line_max_bottom = max(line_max_bottom, y + h)

        y = line_max_bottom + max(1, int(line_gap_base * scale))

    total_w = int(max_w + pad_x + safe_margin)
    total_h = int(y + safe_margin)

    return TextLayout(
        width=max(1, total_w),
        height=max(1, total_h),
        words=words,
        safe_margin=safe_margin,
    )


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
                word.text,
                font=font,
                fill=(0, 0, 0, int(style["soft_shadow_alpha"] * alpha)),
            )

        draw.text(
            (word.x, word.y),
            word.text,
            font=font,
            fill=(
                style["fill_rgb"][0],
                style["fill_rgb"][1],
                style["fill_rgb"][2],
                int(style["fill_alpha"] * alpha),
            ),
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


def apply_perspective_billboard(rgba: np.ndarray, progress: float, preset: dict) -> np.ndarray:
    h, w = rgba.shape[:2]
    p = clamp(progress, 0.0, 1.0)
    p_motion = ease_in_cubic(p)

    skew = lerp(preset["skew_start"], preset["skew_end"], p_motion)
    side_depth = lerp(preset["side_depth_start"], preset["side_depth_end"], p_motion)

    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([
        [side_depth, int(h * skew)],
        [w - side_depth, 0],
        [w, h],
        [0, int(h * (1.0 - skew))],
    ])

    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(
        rgba,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )


def apply_depth_blur(rgba: np.ndarray, progress: float, preset: dict) -> np.ndarray:
    if not ENABLE_DEPTH_BLUR:
        return rgba

    p = clamp(progress, 0.0, 1.0)
    far_blur = preset.get("far_blur", 0.16) * (1.0 - p)
    near_blur = preset.get("near_blur", 0.75) * max(0.0, (p - 0.78) / 0.22)
    blur = far_blur + near_blur

    if blur <= 0.25:
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


def build_overlay_for_frame_v3(
    layout: TextLayout,
    frame_w: int,
    frame_h: int,
    progress: float,
    preset: dict,
    style: dict,
    vanishing_point: Tuple[int, int],
    shake: Tuple[float, float],
) -> np.ndarray:
    p = clamp(progress, 0.0, 1.0)
    p_motion = lerp(ease_in_cubic(p), ease_in_quart(p), preset["rush_aggression"])

    text_rgba = render_progressive_text_rgba(layout, FONT_PATH, style, p)

    # Overscan padding before scaling/perspective/blur to avoid edge clipping.
    text_rgba = pad_rgba(text_rgba, style["transform_overscan"])

    scale = lerp(preset["scale_start"], preset["scale_end"], p_motion)
    transformed = resize_rgba(text_rgba, scale)
    transformed = apply_perspective_billboard(transformed, p, preset)
    transformed = apply_depth_blur(transformed, p, preset)

    vp_x, vp_y = vanishing_point
    shake_dx, shake_dy = shake

    start_x = vp_x + preset["vp_offset_x"]
    start_y = vp_y + preset["vp_offset_y"]

    end_x = preset["end_x_ratio"] * frame_w
    end_y = preset["end_y_ratio"] * frame_h

    x = int(lerp(start_x, end_x, p_motion))
    y = int(lerp(start_y, end_y, p_motion))

    if ENABLE_OPTICAL_FLOW_SHAKE:
        x += int(shake_dx * preset["shake_strength"])
        y += int(shake_dy * preset["shake_strength"])

    move_dx = end_x - start_x
    move_dy = end_y - start_y
    vector_blur_strength = int(lerp(
        preset["vector_blur_start"],
        preset["vector_blur_end"],
        p_motion,
    ))
    transformed = apply_vector_motion_blur(transformed, move_dx, move_dy, vector_blur_strength)

    # End burst fade
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

    vignette = 1.0 - np.clip((radius - 0.25) * strength, 0, 0.25)
    vignette = vignette[:, :, None]
    out = frame_bgr.astype(np.float32) * vignette
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_light_wrap_to_frame(frame_bgr: np.ndarray, overlay_rgba: np.ndarray, strength: float = 0.055, radius: int = 10) -> np.ndarray:
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
            "scale_start": 0.34,
            "scale_end": 2.35,
            "end_x_ratio": 0.18,
            "end_y_ratio": 0.17,
            "skew_start": 0.001,
            "skew_end": 0.012,
            "side_depth_start": 1,
            "side_depth_end": 8,
            "far_blur": 0.18,
            "near_blur": 0.90,
            "vector_blur_start": 1,
            "vector_blur_end": 5,
            "burst_start": 0.95,
            "burst_alpha_power": 1.05,
            # "rush_aggression": 0.16,
            "rush_aggression": 0.06,            
            "vp_offset_x": -45,
            "vp_offset_y": 0,
            # "shake_strength": 0.35,
            "shake_strength": 0.12,            
        }
        base_font_size = int(frame_h * 0.061)
        safe_margin = int(frame_h * 0.030)
        overscan = int(frame_h * 0.020)
        style = {
            "font_size": base_font_size,
            "fill_rgb": (245, 245, 245),
            "fill_alpha": 255,

            "line_scale_single": 1.20,
            "line_scale_small": 0.62,
            "line_scale_mid": 0.88,
            "line_scale_big": 1.22,

            "soft_shadow_alpha": 0,
            "soft_shadow_dx": 0,
            "soft_shadow_dy": 0,

            "word_spacing": int(frame_w * 0.010),
            "line_gap": int(frame_h * 0.006),
            "pad_x": int(frame_h * 0.010),
            "pad_y": int(frame_h * 0.008),

            "safe_margin": safe_margin,
            "transform_overscan": overscan,

            "word_reveal_portion": 0.58,
            "word_reveal_overlap": 1.9,
        }
    else:
        preset = {
            # "scale_start": 0.38,
            # "scale_end": 1.95,
            "scale_start": 0.55,
            "scale_end": 1.45,
            "end_x_ratio": 0.20,
            "end_y_ratio": 0.13,
            "skew_start": 0.001,
            "skew_end": 0.010,
            "side_depth_start": 1,
            "side_depth_end": 7,
            "far_blur": 0.16,
            "near_blur": 0.75,
            # "vector_blur_start": 1,
            # "vector_blur_end": 4,
            "vector_blur_start": 1,
            "vector_blur_end": 2,
            "burst_start": 0.96,
            "burst_alpha_power": 1.04,
            "rush_aggression": 0.14,
            "vp_offset_x": -30,
            "vp_offset_y": 12,
            "shake_strength": 0.30,
        }
        base_font_size = int(frame_h * 0.105)
        safe_margin = int(frame_h * 0.040)
        overscan = int(frame_h * 0.028)
        style = {
            "font_size": base_font_size,
            "fill_rgb": (245, 245, 245),
            "fill_alpha": 255,

            "line_scale_single": 1.15,
            "line_scale_small": 0.58,
            "line_scale_mid": 0.84,
            "line_scale_big": 1.18,

            "soft_shadow_alpha": 0,
            "soft_shadow_dx": 0,
            "soft_shadow_dy": 0,

            "word_spacing": int(frame_w * 0.006),
            "line_gap": int(frame_h * 0.006),
            "pad_x": int(frame_h * 0.010),
            "pad_y": int(frame_h * 0.008),

            "safe_margin": safe_margin,
            "transform_overscan": overscan,

            "word_reveal_portion": 0.58,
            "word_reveal_overlap": 1.9,
        }

    return preset, style


def render_video() -> None:
    if not Path(INPUT_VIDEO).exists():
        raise FileNotFoundError(f"Không tìm thấy video input: {INPUT_VIDEO}")

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

    # vp_smoother = SmoothPoint(alpha=0.055)
    vp_smoother = SmoothPoint(alpha=0.025)
    shake_tracker = CameraShakeTracker()
    # shake_smoother = SmoothVector(alpha=0.15)
    shake_smoother = SmoothVector(alpha=0.05)

    fallback_vp = (int(frame_w * 0.52), int(frame_h * 0.20))
    current_vp = fallback_vp

    last_text = None
    cached_layout: Optional[TextLayout] = None

    # V10.2 smoothing: lock vanishing point per lyric segment.
    # This keeps the V3-style movement but avoids per-frame text jitter.
    current_seg_idx = None
    locked_vp_for_segment = fallback_vp

    for frame_idx in tqdm(range(total_to_render), desc="Rendering V10.2 Hybrid Smooth Forward Rush"):
        ret, frame = cap.read()
        if not ret:
            break

        current_time = frame_idx / fps
        seg_idx = find_active_lyric_idx(lyrics, current_time)
        active = lyrics[seg_idx] if seg_idx is not None else None

        if ENABLE_VANISHING_POINT:
            if frame_idx % VANISHING_POINT_EVERY_N_FRAMES == 0:
                raw_vp = detect_vanishing_point(frame)
                current_vp = vp_smoother.update(raw_vp)
            else:
                current_vp = vp_smoother.update(current_vp)
        else:
            current_vp = fallback_vp

        if ENABLE_OPTICAL_FLOW_SHAKE:
            raw_shake = shake_tracker.update(frame)
            shake = shake_smoother.update(raw_shake)
        else:
            shake = (0.0, 0.0)

        frame = add_subtle_vignette(frame, strength=0.10)

        if active:
            if current_seg_idx != seg_idx:
                current_seg_idx = seg_idx
                locked_vp_for_segment = current_vp

            progress = clamp((current_time - active.start) / (active.end - active.start), 0.0, 1.0)

            if active.text != last_text:
                cached_layout = build_text_layout(active.text, FONT_PATH, style)
                last_text = active.text

            overlay = build_overlay_for_frame_v3(
                layout=cached_layout,
                frame_w=frame_w,
                frame_h=frame_h,
                progress=progress,
                preset=preset,
                style=style,
                vanishing_point=locked_vp_for_segment,
                shake=shake,
            )

            frame = apply_light_wrap_to_frame(frame, overlay, strength=0.055, radius=10)
            frame = alpha_overlay_bgr(frame, overlay)
        else:
            current_seg_idx = None

        if DEBUG_DRAW_VANISHING_POINT:
            cv2.circle(frame, locked_vp_for_segment, 10, (0, 0, 255), -1)
            cv2.putText(
                frame,
                f"VP {locked_vp_for_segment}",
                (locked_vp_for_segment[0] + 12, locked_vp_for_segment[1] - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )

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

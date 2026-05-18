# POV Lyric Forward Rush Renderer

Công cụ Python giúp tạo hiệu ứng **lyrics 3D cinematic** cho video POV, đặc biệt phù hợp với video chạy xe, đường phố ban đêm, đường đèo, TikTok, YouTube Shorts, Facebook Reels.

Hiệu ứng chính của tool:

* Chữ xuất hiện gần **vanishing point / tâm đường**.
* Từng từ trong lyrics hiện dần theo nhịp.
* Cả cụm chữ vừa xuất hiện vừa **trôi về phía trước** như camera POV đang lao tới.
* Chữ phóng to dần, nghiêng nhẹ, có motion blur, depth blur, ghost trail.
* Cụm chữ vụt qua khỏi khung hình theo kiểu cinematic / After Effects style.
* Hỗ trợ nhiều đoạn lyrics overlap, giúp cụm trước vẫn đang bay ra ngoài trong khi cụm sau đã bắt đầu xuất hiện.

Dự án này được tối ưu từ bản v10.1 theo hướng sample-matched forward flow: thay vì text chỉ đứng tại một điểm rồi blast, text bắt đầu gần vanishing point và di chuyển ngay từ lúc chữ đầu tiên xuất hiện.

---

## Demo Style

Phong cách hiệu ứng hướng tới các video dạng:

* POV lái xe ban đêm.
* Video đường phố, đường đèo, cinematic road trip.
* Lyrics typography kiểu TikTok / Shorts / Reels.
* 3D text tracking mô phỏng After Effects.
* Text bay từ xa về gần, tràn viền và biến mất.

---

## Features

### Text Animation

* Word-by-word reveal.
* Forward rush movement.
* Scale from small to large.
* Diagonal fly-through motion.
* Phrase-level fade out.
* Multi-phrase overlap.

### Cinematic Effects

* Vanishing point detection.
* Per-segment vanishing point lock to reduce jitter.
* Perspective billboard transform.
* Motion blur based on movement vector.
* Depth blur based on distance.
* Ghost trail / echo text.
* Light wrap around text.
* Subtle camera push.
* Vignette for cinematic mood.

### Output Profiles

Supported profiles:

* `sample` — bám sát style video mẫu nhất.
* `tiktok` — tối ưu video dọc TikTok.
* `shorts` — tối ưu YouTube Shorts.
* `reels` — tối ưu Facebook / Instagram Reels.
* `generic` — profile trung tính.

---

## Project Structure

```text
project/
├── input.mp4
├── lyrics.json
├── render_pov_lyrics_sample_style_v10_4_sample_forward_flow.py
├── README.md
└── fonts/
    └── Anton-Regular.ttf
```

Recommended font:

```text
fonts/Anton-Regular.ttf
```

Anton is recommended because it creates a bold, condensed, cinematic lyric style similar to popular short-form videos.

---

## Requirements

Install Python dependencies:

```bash
pip install opencv-python pillow numpy tqdm
```

Install FFmpeg and make sure it is available in your system PATH.

Check FFmpeg:

```bash
ffmpeg -version
```

---

## Quick Start

### Windows

```bash
python render_pov_lyrics_sample_style_v10_4_sample_forward_flow.py ^
  --input input.mp4 ^
  --lyrics lyrics.json ^
  --output output_v10_4.mp4 ^
  --profile sample
```

### Linux / macOS

```bash
python render_pov_lyrics_sample_style_v10_4_sample_forward_flow.py \
  --input input.mp4 \
  --lyrics lyrics.json \
  --output output_v10_4.mp4 \
  --profile sample
```

---

## Input Files

### 1. Video Input

Default input video:

```text
input.mp4
```

Recommended video format:

* MP4
* 9:16 vertical video for TikTok / Shorts / Reels
* POV road / driving / street footage
* 24 FPS, 30 FPS, or 60 FPS

### 2. Lyrics Input

Default lyrics file:

```text
lyrics.json
```

Example:

```json
[
  {
    "start": "00:00:00.000",
    "end": "00:00:01.500",
    "text": "YOU'RE GONNA\nSAY THAT\nYOU'RE\nSORRY"
  },
  {
    "start": "00:00:01.000",
    "end": "00:00:03.400",
    "text": "AT THE END\nOF THE NIGHT"
  }
]
```

For a style closer to the sample video, allow lyric segments to overlap by around **0.3–0.7 seconds**.

This creates a better cinematic flow because the previous phrase is still flying out while the next phrase starts appearing.

---

## Lyrics JSON Format

Each lyrics object must include:

| Field   | Description                                 |
| ------- | ------------------------------------------- |
| `start` | Start time in `HH:MM:SS.mmm` format         |
| `end`   | End time in `HH:MM:SS.mmm` format           |
| `text`  | Lyrics text. Use `\n` to create line breaks |

Example:

```json
{
  "start": "00:00:05.200",
  "end": "00:00:07.100",
  "text": "I THOUGHT I SAW\nYOUR FACE TODAY"
}
```

---

## Recommended Lyrics Timing

For short cinematic lyrics:

```text
Phrase duration: 1.2s – 2.4s
Overlap:         0.3s – 0.7s
Words per block: 2 – 8 words
Lines per block: 2 – 4 lines
```

Good layout:

```text
YOU'RE GONNA
SAY THAT
YOU'RE
SORRY
```

Another good layout:

```text
AT THE END
OF THE NIGHT
```

Avoid very long single-line lyrics because they may overflow or lose visual impact.

---

## CLI Arguments

```bash
python render_pov_lyrics_sample_style_v10_4_sample_forward_flow.py \
  --input input.mp4 \
  --lyrics lyrics.json \
  --output output.mp4 \
  --font fonts/Anton-Regular.ttf \
  --profile sample
```

| Argument    | Default                                                       | Description        |
| ----------- | ------------------------------------------------------------- | ------------------ |
| `--input`   | `input.mp4`                                                   | Input video path   |
| `--lyrics`  | `lyrics.json`                                                 | Lyrics timing file |
| `--output`  | `output_pov_lyric_sample_style_v10_4_sample_forward_flow.mp4` | Output video path  |
| `--font`    | `fonts/Anton-Regular.ttf`                                     | Font file path     |
| `--profile` | `sample`                                                      | Render profile     |

---

## Profiles

### sample

Best for matching the reference video style.

```bash
--profile sample
```

Behavior:

* starts near the road center / vanishing point
* text grows strongly
* flies out toward upper-left
* stronger depth and motion blur
* more cinematic ghost trail

### tiktok

```bash
--profile tiktok
```

Optimized for TikTok vertical videos.

### shorts

```bash
--profile shorts
```

Optimized for YouTube Shorts.

### reels

```bash
--profile reels
```

Optimized for Facebook / Instagram Reels.

### generic

```bash
--profile generic
```

Balanced profile for general use.

---

## Important Parameters to Customize

Most visual tuning is inside `build_profile()`.

### Make text fly closer / larger

```python
"scale_end": 3.05
```

Increase to:

```python
"scale_end": 3.40
```

### Make text start lower near the road

```python
"vp_y": 0.26
```

Increase to:

```python
"vp_y": 0.32
```

### Make text rush faster

```python
"rush_aggression": 0.08
```

Increase to:

```python
"rush_aggression": 0.14
```

### Make text fly more to upper-left

```python
"end_x_ratio": -0.10,
"end_y_ratio": -0.04
```

More aggressive:

```python
"end_x_ratio": -0.18,
"end_y_ratio": -0.10
```

### Make ghost trail stronger

```python
"trail_count": 4,
"trail_alpha": 0.34
```

More visible:

```python
"trail_count": 5,
"trail_alpha": 0.42
```

---

## How the Effect Works

The renderer processes the video frame by frame:

1. Reads the current timestamp.
2. Finds active lyrics from `lyrics.json`.
3. Detects the road vanishing point.
4. Locks the vanishing point per lyric segment.
5. Renders each lyric phrase as transparent RGBA text.
6. Reveals each word progressively.
7. Moves the whole text phrase from far depth to near camera.
8. Applies scale, perspective, blur, ghost trail, and light wrap.
9. Composites the text over the video.
10. Uses FFmpeg to mux the original audio back into the final video.

---

## Why Overlap Lyrics?

Sample-style videos often do not wait for one phrase to completely disappear before showing the next one.

Instead:

```text
Phrase A appears → Phrase A starts flying out
                    Phrase B appears while Phrase A is still moving
```

This creates a continuous visual rhythm and makes the lyric animation feel closer to real short-form cinematic edits.

---

## Troubleshooting

### FFmpeg not found

Install FFmpeg and add it to PATH.

Check:

```bash
ffmpeg -version
```

### Font not found

Make sure this file exists:

```text
fonts/Anton-Regular.ttf
```

Or use another font:

```bash
python render_pov_lyrics_sample_style_v10_4_sample_forward_flow.py --font path/to/font.ttf
```

### Text is too large

Reduce:

```python
"scale_end"
```

or reduce:

```python
"font_size"
```

### Text is too small

Increase:

```python
"scale_start"
"scale_end"
"font_size"
```

### Text starts too high

Increase:

```python
"vp_y"
```

### Text does not fly out enough

Adjust:

```python
"end_x_ratio"
"end_y_ratio"
"scale_end"
```

### Text movement is too slow

Increase:

```python
"rush_aggression"
"rush_mix"
```

### Text is too blurry

Reduce:

```python
"near_blur"
"vector_blur_end"
```

---

## Recommended Workflow

1. Prepare vertical POV video.
2. Create `lyrics.json` with short lyric blocks.
3. Add overlap between lyric segments.
4. Start with `--profile sample`.
5. Render a short test first.
6. Tune `scale_end`, `vp_y`, and `rush_aggression`.
7. Render final video.

---

## Example Command

```bash
python render_pov_lyrics_sample_style_v10_4_sample_forward_flow.py \
  --input input.mp4 \
  --lyrics lyrics.json \
  --output final_lyrics_video.mp4 \
  --font fonts/Anton-Regular.ttf \
  --profile sample
```

---

## Notes

This project does not require After Effects. It uses Python, OpenCV, Pillow, NumPy, and FFmpeg to generate a similar cinematic lyric motion style programmatically.

The effect is not true 3D tracking, but it simulates a 2.5D / 3D typography feel using:

* vanishing point detection
* perspective transform
* scale animation
* depth blur
* motion blur
* ghost trail
* light wrap
* camera push

---

## License

You can use this project as a personal or commercial video automation tool. Add your preferred license here, for example:

```text
MIT License
```

---

## Credits

Built for cinematic POV lyric videos and short-form social media content.

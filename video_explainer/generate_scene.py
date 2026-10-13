"""One-scene prototype: illustrate + animate + narrate a single Mishnah moment.

Today's scene: Mishnah Shabbat 13:6-7 - a deer wanders into a house on
Shabbat, and the Mishnah works out who becomes liable for "trapping" it
depending on who closes the door (and which part of it).

Usage: python3 generate_scene.py
Requires OPENROUTER_API_KEY in .env (see .env in this folder) and ffmpeg
on PATH (brew install ffmpeg).
"""
import base64
import os
import subprocess
import time

import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ["OPENROUTER_API_KEY"]
BASE = "https://openrouter.ai/api/v1"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUT_DIR, exist_ok=True)

IMAGE_PROMPT = (
    "Flat vector illustration, modern children's-book style, warm earthy palette "
    "(terracotta, cream, sage green), a small startled cartoon deer leaping through an "
    "open wooden doorway into a cozy ancient stone house interior, a surprised hand "
    "reaching for the edge of the door from inside, decorative sparkle and star accents "
    "scattered around, soft warm lighting, vertical 9:16 portrait composition, clean bold "
    "outlines, minimalist uncluttered background, playful whimsical mood, no text or "
    "letters anywhere in the image"
)
VIDEO_PROMPT = (
    "Gentle slow zoom in, the deer's ears twitch and its legs shift as if about to bolt, "
    "the wooden door creaks and begins to swing closed, sparkle particles drift and "
    "twinkle softly, subtle idle breathing motion, smooth cinemagraph-style loopable motion"
)
NARRATION = (
    "A deer wanders into the house on Shabbat. Nobody invited it in, but now there's a "
    "decision to make. Shut that door, and you haven't just closed up for the night, "
    "you've trapped a wild animal. So the Mishnah asks: if two people close the door "
    "together, who's actually responsible? Turns out, it's whoever seals the last gap. "
    "Close most of the door, and you're in the clear. Seal that final crack, and "
    "congratulations, you caught a deer."
)


def generate_image():
    print("Generating illustration (Recraft)...")
    resp = requests.post(
        f"{BASE}/images",
        headers=HEADERS,
        json={
            "model": "recraft/recraft-v4.1",
            "prompt": IMAGE_PROMPT,
            "n": 1,
            "aspect_ratio": "9:16",
            "quality": "high",
            "output_format": "png",
        },
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    b64 = data["data"][0]["b64_json"]
    path = os.path.join(OUT_DIR, "scene.png")
    with open(path, "wb") as f:
        f.write(base64.b64decode(b64))
    print(f"  saved {path} (cost: ${data.get('usage', {}).get('cost', '?')})")
    return path, b64


def generate_video(image_b64):
    print("Animating illustration (image-to-video)...")
    resp = requests.post(
        f"{BASE}/videos",
        headers=HEADERS,
        json={
            "model": "alibaba/wan-2.7",
            "prompt": VIDEO_PROMPT,
            "frame_images": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    "frame_type": "first_frame",
                }
            ],
            "resolution": "720p",
            "aspect_ratio": "9:16",
            "duration": 5,
        },
        timeout=60,
    )
    resp.raise_for_status()
    job = resp.json()
    polling_url = job["polling_url"]
    print(f"  queued (id={job.get('id')}), polling...")

    for _ in range(60):
        time.sleep(5)
        poll = requests.get(polling_url, headers=HEADERS, timeout=30)
        poll.raise_for_status()
        status = poll.json()
        print(f"  status: {status.get('status')}")
        if status.get("status") == "completed":
            video_url = status["unsigned_urls"][0]
            # Despite the name, these URLs still require the bearer token.
            video_bytes = requests.get(video_url, headers=HEADERS, timeout=120).content
            path = os.path.join(OUT_DIR, "scene_raw.mp4")
            with open(path, "wb") as f:
                f.write(video_bytes)
            print(f"  saved {path}")
            return path
        if status.get("status") == "failed":
            raise RuntimeError(f"Video generation failed: {status}")
    raise TimeoutError("Video generation did not complete in time")


def generate_narration():
    print("Generating narration (ElevenLabs TTS)...")
    resp = requests.post(
        f"{BASE}/audio/speech",
        headers=HEADERS,
        json={
            "model": "elevenlabs/eleven-v4",
            "input": NARRATION,
            "voice": "george",
            "response_format": "mp3",
        },
        timeout=60,
    )
    resp.raise_for_status()
    path = os.path.join(OUT_DIR, "narration.mp3")
    with open(path, "wb") as f:
        f.write(resp.content)
    print(f"  saved {path}")
    return path


def assemble(video_path, audio_path):
    print("Assembling final clip (looping video to match narration, adding captions)...")
    final_path = os.path.join(OUT_DIR, "final_scene.mp4")
    caption = NARRATION.replace("'", "’").replace(":", "\\:")
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-stream_loop", "-1", "-i", video_path,
            "-i", audio_path,
            "-filter_complex",
            (
                f"[0:v]drawtext=text='Shabbat 13\\:6-7':fontsize=28:fontcolor=white:"
                f"box=1:boxcolor=black@0.4:boxborderw=10:x=20:y=20,"
                f"drawtext=text='LIABLE?':fontsize=60:fontcolor=white:borderw=3:bordercolor=black:"
                f"x=(w-text_w)/2:y=h-200[v]"
            ),
            "-map", "[v]", "-map", "1:a",
            "-c:v", "libx264", "-c:a", "aac",
            "-shortest",
            final_path,
        ],
        check=True,
    )
    print(f"Done: {final_path}")


if __name__ == "__main__":
    _, img_b64 = generate_image()
    video_path = generate_video(img_b64)
    audio_path = generate_narration()
    assemble(video_path, audio_path)

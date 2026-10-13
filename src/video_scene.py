"""Generates a short AI-illustrated, animated, narrated "explainer scene" video
for a day's Mishnah reading, via OpenRouter (Claude for scene-writing, Recraft
for the illustration, Wan 2.7 for image-to-video, ElevenLabs for narration).

Each day's video is generated once and cached to disk forever at
video_explainer/output/{cycle_id}_{day_num}.mp4, since generation costs real
money (image ~$0.03-0.07, video ~$0.50, TTS small) and a given day's reading
never changes - see cached_video_path/generate_video_for_day.

This module reuses the exact API call patterns (and gotchas) validated in
video_explainer/generate_scene.py, generalized from one hardcoded scene to
any day via an LLM scene-writing step.
"""
import base64
import datetime
import json
import os
import re
import subprocess
import time

import requests
from dotenv import load_dotenv

import sefaria_client as sc

_VIDEO_EXPLAINER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "video_explainer")
load_dotenv(os.path.join(_VIDEO_EXPLAINER_DIR, ".env"))

OUT_DIR = os.path.join(_VIDEO_EXPLAINER_DIR, "output")
os.makedirs(OUT_DIR, exist_ok=True)

# Gates the generate button: this is a public, un-authenticated page, and
# each generation spends real OpenRouter credits - a PIN is just friction
# against a random visitor clicking it, not real security.
VIDEO_GEN_PIN = os.environ.get("VIDEO_GEN_PIN", "")

BASE = "https://openrouter.ai/api/v1"
SCENE_MODEL = "anthropic/claude-sonnet-5.5"
IMAGE_MODEL = "recraft/recraft-v4.1"
VIDEO_MODEL = "alibaba/wan-2.7"
TTS_MODEL = "elevenlabs/eleven-v4"
TTS_VOICE = "george"

SCENE_SYSTEM_PROMPT = """You are a scene-writer for "Daily Mishnah", an app that turns one day's \
Mishnah reading into a ~30-second AI-illustrated, animated, narrated explainer video.

Given the day's Sefaria text, pick ONE compelling, fun, or visually interesting moment (the \
reading may span many mishnayot - choose the single best one, don't try to cover everything). \
Then write all the assets needed to produce that scene.

Respond with ONLY a JSON object with exactly these keys:
- "image_prompt": a single static-illustration prompt for an image generator. MUST follow this \
house style closely: flat vector illustration, modern children's-book style, warm earthy palette \
(terracotta, cream, sage green), bold clean outlines, decorative sparkle and star accents \
scattered around, soft warm lighting, vertical 9:16 portrait composition, minimalist uncluttered \
background, playful whimsical mood, no text or letters anywhere in the image. Pose the character(s) \
mid-action, caught at the most exaggerated, off-balance, comedic instant of the scene (frozen \
mid-leap, mid-flinch, eyes bugging out, arms flailing) - like a freeze-frame from a cartoon, not a \
calm portrait - since this still is the first frame of an animation and needs somewhere wild to go.
- "video_prompt": a motion prompt that makes that illustration genuinely MOVE - this is the single \
most important field. The failure mode to avoid is a slow zoom with nothing else happening. Instead \
describe big, fast, obvious physical action: a character leaping/bolting/spinning/slamming something, \
limbs flying, objects flung across frame, exaggerated squash-and-stretch cartoon motion, a quick \
whip-pan or snap-zoom on the punchline beat, things rapidly shaking/wobbling/bursting. Treat it like \
a 3-second slapstick cartoon gag on loop, not a gentle ambient cinemagraph. Avoid the words "gentle", \
"subtle", and "slow" entirely.
- "narration": an ORIGINAL 40-70 word narration script, in your own words. It must \
paraphrase/summarize the halakhic content - never quote or closely copy any translation. Write it \
like a stand-up comedian's bit, not an explainer: open with an absurd, incredulous angle on the \
situation, escalate with a ridiculous exaggeration or unexpected tangent, and land a punchy, slightly \
unhinged button line. Channel the comedic STYLE of Monty Python - deadpan delivery, wildly over-pedantic logic applied to \
a silly scenario, sudden non-sequiturs, mock-serious escalation, absurd bureaucratic nitpicking. \
This means emulating that general tone and comic technique only - never quote, paraphrase-closely, \
or reference any of their actual specific sketches, characters, or lines; write entirely new jokes \
in that spirit, about this halakha. Be genuinely random and weird, not just "light and whimsical" - \
surprise the listener. It still has to be accurate enough that someone could reconstruct the real \
halakha from it.
- "stamp": a short punchy overlay word or phrase (1-3 words, e.g. "LIABLE?" or "EXEMPT!") \
capturing the twist or key takeaway.
- "citation": a citation string for the specific moment chosen, e.g. "Shabbat 13:6-7".

No prose, no markdown fences, just the raw JSON object."""


def video_path_for(cycle_id, day_num):
    return os.path.join(OUT_DIR, f"{cycle_id}_{day_num}.mp4")


def cached_video_path(cycle_id, day_num):
    path = video_path_for(cycle_id, day_num)
    return path if os.path.exists(path) else None


def _headers():
    return {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"}


def collect_readings_text(day, now_str):
    """Flatten a day's reading(s) into one text blob for the scene-writing prompt."""
    blocks = []
    for reading in day.readings:
        data = sc.fetch_text(reading.sefaria_ref, now_str)
        if not data:
            continue
        paragraphs = sc.flatten_to_paragraphs(data["text"])
        if paragraphs:
            blocks.append(f"{reading.label}:\n" + "\n".join(paragraphs))
    return "\n\n".join(blocks)


def _strip_json_fences(content):
    # Claude via OpenRouter reliably wraps JSON in ```json ... ``` fences even
    # when asked not to, so strip them before parsing rather than relying on
    # response_format (unreliable for non-OpenAI models on OpenRouter).
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip())


def generate_scene_for_day(day, readings_text):
    """Ask the LLM to pick one moment from readings_text and write its scene assets.

    Returns a dict with image_prompt, video_prompt, narration, stamp, citation.
    """
    user_prompt = f"Today's reading:\n\n{readings_text}\n\nPick one moment and write the scene assets."
    resp = requests.post(
        f"{BASE}/chat/completions",
        headers=_headers(),
        json={
            "model": SCENE_MODEL,
            "messages": [
                {"role": "system", "content": SCENE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.9,
        },
        timeout=60,
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    scene = json.loads(_strip_json_fences(content))

    missing = [k for k in ("image_prompt", "video_prompt", "narration", "stamp") if not scene.get(k)]
    if missing:
        raise ValueError(f"Scene response missing required field(s): {missing}")
    if not scene.get("citation"):
        scene["citation"] = ", ".join(r.label for r in day.readings) or day.hebrew_date
    return scene


def _generate_image(prefix, image_prompt):
    resp = requests.post(
        f"{BASE}/images",
        headers=_headers(),
        json={
            "model": IMAGE_MODEL,
            "prompt": image_prompt,
            "n": 1,
            "aspect_ratio": "9:16",
            "quality": "high",
            "output_format": "png",
        },
        timeout=120,
    )
    resp.raise_for_status()
    b64 = resp.json()["data"][0]["b64_json"]
    path = os.path.join(OUT_DIR, f"{prefix}_image.png")
    with open(path, "wb") as f:
        f.write(base64.b64decode(b64))
    return path, b64


def _generate_video(prefix, video_prompt, image_b64):
    resp = requests.post(
        f"{BASE}/videos",
        headers=_headers(),
        json={
            "model": VIDEO_MODEL,
            "prompt": video_prompt,
            "frame_images": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    "frame_type": "first_frame",
                }
            ],
            "resolution": "720p",
            "aspect_ratio": "9:16",
            "duration": 10,  # the model's max; narration (40-70 words, ~15-25s) still
            # loops over this more than once, but far less jarringly than at 5s.
        },
        timeout=60,
    )
    resp.raise_for_status()
    polling_url = resp.json()["polling_url"]

    for _ in range(60):
        time.sleep(5)
        poll = requests.get(polling_url, headers=_headers(), timeout=30)
        poll.raise_for_status()
        status = poll.json()
        if status.get("status") == "completed":
            video_url = status["unsigned_urls"][0]
            # Despite the name, these URLs still require the bearer token.
            video_bytes = requests.get(video_url, headers=_headers(), timeout=120).content
            path = os.path.join(OUT_DIR, f"{prefix}_raw.mp4")
            with open(path, "wb") as f:
                f.write(video_bytes)
            return path
        if status.get("status") == "failed":
            raise RuntimeError(f"Video generation failed: {status}")
    raise TimeoutError("Video generation did not complete in time")


def _generate_narration(prefix, narration_text):
    resp = requests.post(
        f"{BASE}/audio/speech",
        headers=_headers(),
        json={
            "model": TTS_MODEL,
            "input": narration_text,
            "voice": TTS_VOICE,
            "response_format": "mp3",
        },
        timeout=60,
    )
    resp.raise_for_status()
    path = os.path.join(OUT_DIR, f"{prefix}_narration.mp3")
    with open(path, "wb") as f:
        f.write(resp.content)
    return path


def _escape_drawtext(text):
    return text.replace("\\", "\\\\").replace(":", "\\:").replace("'", "’").replace("%", "\\%")


def _assemble(final_path, video_path, audio_path, citation, stamp):
    citation_text = _escape_drawtext(citation)
    stamp_text = _escape_drawtext(stamp)
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-stream_loop", "-1", "-i", video_path,
            "-i", audio_path,
            "-filter_complex",
            (
                f"[0:v]drawtext=text='{citation_text}':fontsize=28:fontcolor=white:"
                f"box=1:boxcolor=black@0.4:boxborderw=10:x=20:y=20,"
                f"drawtext=text='{stamp_text}':fontsize=60:fontcolor=white:borderw=3:bordercolor=black:"
                f"x=(w-text_w)/2:y=h-200[v]"
            ),
            "-map", "[v]", "-map", "1:a",
            "-c:v", "libx264", "-c:a", "aac",
            "-shortest",
            final_path,
        ],
        check=True,
        capture_output=True,
    )


def generate_video_for_day(day, cycle_id):
    """Run the full pipeline (scene-writing, image, video, narration, assembly)
    for one calendar day and cache the result. Returns the final video path.

    Costs real money (~$0.60-1) and takes 1-2 minutes - callers must only
    invoke this on an explicit user action, never automatically.
    """
    now_str = datetime.datetime.now().isoformat()
    readings_text = collect_readings_text(day, now_str)
    if not readings_text:
        raise ValueError("No reading text available for this day.")

    scene = generate_scene_for_day(day, readings_text)
    prefix = f"{cycle_id}_{day.day_num}"
    _, image_b64 = _generate_image(prefix, scene["image_prompt"])
    video_path = _generate_video(prefix, scene["video_prompt"], image_b64)
    audio_path = _generate_narration(prefix, scene["narration"])

    final_path = video_path_for(cycle_id, day.day_num)
    _assemble(final_path, video_path, audio_path, scene["citation"], scene["stamp"])
    return final_path

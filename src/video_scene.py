"""Generates a short AI-illustrated, animated, narrated "explainer scene" video
for a day's Mishnah reading - or for any text a user pastes in (see
generate_video_for_text) - via OpenRouter (Claude for scene-writing, Recraft
for the illustrations, Wan 2.7 for image-to-video, ElevenLabs for narration).

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
import hashlib
import json
import os
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

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

DEFAULT_TARGET_SECONDS = 60
MIN_TARGET_SECONDS = 15
MAX_TARGET_SECONDS = 180
# Wan clips max out at 10s, so a long narration over a single clip loops it
# many times; instead give each ~SECONDS_PER_SHOT of narration its own shot.
SECONDS_PER_SHOT = 20
# ElevenLabs narration runs at roughly 2.5 words per second.
WORDS_PER_SECOND = 2.5

DEFAULT_STYLE = (
    "flat vector illustration, modern children's-book style, warm earthy palette "
    "(terracotta, cream, sage green), bold clean outlines, decorative sparkle and star accents "
    "scattered around, soft warm lighting, minimalist uncluttered background, playful whimsical mood"
)
DEFAULT_TONE = (
    "Write it like a stand-up comedian's bit, not an explainer: open with an absurd, incredulous \
angle on the situation, escalate with a ridiculous exaggeration or unexpected tangent, and land a \
punchy, slightly unhinged button line. Channel the comedic STYLE of Monty Python - deadpan delivery, \
wildly over-pedantic logic applied to a silly scenario, sudden non-sequiturs, mock-serious \
escalation, absurd bureaucratic nitpicking. This means emulating that general tone and comic \
technique only - never quote, paraphrase-closely, or reference any of their actual specific \
sketches, characters, or lines; write entirely new jokes in that spirit. Be genuinely random and \
weird, not just \"light and whimsical\" - surprise the listener."
)

DAILY_SOURCE_INSTRUCTIONS = (
    "Given the day's Sefaria text, pick ONE compelling, fun, or visually interesting moment (the "
    "reading may span many mishnayot - choose the single best one, don't try to cover everything)."
)
CUSTOM_SOURCE_INSTRUCTIONS = (
    "Given a text supplied by the user, dramatize its central idea (if it is long, focus on the "
    "single most compelling, fun, or visually interesting part rather than covering everything)."
)


def num_shots(target_seconds):
    return max(1, round(target_seconds / SECONDS_PER_SHOT))


def build_scene_system_prompt(source_instructions, target_seconds, style=None, tone=None):
    shots = num_shots(target_seconds)
    words = round(target_seconds * WORDS_PER_SECOND)
    # The scene model reliably overshoots the requested range by ~10%, so
    # ask for a range ending at the target rather than centered on it.
    min_words, max_words = round(words * 0.8), words
    style = style or DEFAULT_STYLE
    tone = tone or DEFAULT_TONE
    return f"""You are a scene-writer for an app that turns a text into a ~{target_seconds}-second \
AI-illustrated, animated, narrated explainer video.

{source_instructions} Then write all the assets needed to produce that scene.

Respond with ONLY a JSON object with exactly these keys:
- "shots": a list of exactly {shots} shot(s), in narration order, each covering an equal share of \
the narration (beginning, middle, ... punchline). Each shot is an object with:
  - "image_prompt": a single static-illustration prompt for an image generator. MUST follow this \
visual style closely: {style}, vertical 9:16 portrait composition, no text or letters anywhere in the \
image. Describe recurring characters with the exact same appearance wording in every shot so they \
stay recognizable. Pose the character(s) mid-action, caught at the most exaggerated, off-balance, \
dramatic instant of that beat (frozen mid-leap, mid-flinch, eyes bugging out, arms flailing) - like a \
freeze-frame from a cartoon, not a calm portrait - since this still is the first frame of an \
animation and needs somewhere wild to go.
  - "video_prompt": a motion prompt that makes that illustration genuinely MOVE - this is the most \
important field. The failure mode to avoid is a slow zoom with nothing else happening. Instead \
describe big, fast, obvious physical action: a character leaping/bolting/spinning/slamming something, \
limbs flying, objects flung across frame, exaggerated squash-and-stretch cartoon motion, a quick \
whip-pan or snap-zoom on the punchline beat, things rapidly shaking/wobbling/bursting. Treat it like \
a short slapstick cartoon gag on loop, not a gentle ambient cinemagraph. Avoid the words "gentle", \
"subtle", and "slow" entirely.
- "narration": an ORIGINAL narration script of {min_words}-{max_words} words - a hard limit, since \
the narration's spoken length IS the video's length (about {target_seconds} seconds) - in your own words. It must paraphrase/summarize the content - never quote or \
closely copy any translation. {tone} It still has to be accurate enough that someone could \
reconstruct the real content from it.
- "stamp": a short punchy overlay word or phrase (1-3 words, e.g. "LIABLE?" or "EXEMPT!") \
capturing the twist or key takeaway.
- "citation": a short citation or title for the specific moment chosen, e.g. "Shabbat 13:6-7".

No prose, no markdown fences, just the raw JSON object."""


def video_path_for(cycle_id, day_num):
    return os.path.join(OUT_DIR, f"{cycle_id}_{day_num}.mp4")


def cached_video_path(cycle_id, day_num):
    path = video_path_for(cycle_id, day_num)
    return path if os.path.exists(path) else None


def custom_video_key(text, target_seconds, style, tone):
    """Stable cache key for a custom-text video, so identical requests reuse
    the already-paid-for result instead of generating again."""
    payload = json.dumps([text.strip(), target_seconds, style.strip(), tone.strip()])
    return "custom_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def custom_video_path(key):
    return os.path.join(OUT_DIR, f"{key}.mp4")


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


def generate_scene(system_prompt, user_prompt):
    """Ask the LLM to write the scene assets.

    Returns a dict with shots (each with image_prompt, video_prompt),
    narration, stamp, and citation (possibly empty).
    """
    resp = requests.post(
        f"{BASE}/chat/completions",
        headers=_headers(),
        json={
            "model": SCENE_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.9,
        },
        timeout=120,
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    scene = json.loads(_strip_json_fences(content))

    missing = [k for k in ("shots", "narration", "stamp") if not scene.get(k)]
    missing += [
        f"shots[{i}].{k}"
        for i, shot in enumerate(scene.get("shots") or [])
        for k in ("image_prompt", "video_prompt")
        if not shot.get(k)
    ]
    if missing:
        raise ValueError(f"Scene response missing required field(s): {missing}")
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
            "duration": 10,  # the model's max; each shot covers ~SECONDS_PER_SHOT of
            # narration, so it loops about twice rather than many times.
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


def _audio_duration(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
        check=True, capture_output=True, text=True,
    )
    return float(out.stdout.strip())


def _assemble(final_path, video_paths, audio_path, citation, stamp):
    """Split the narration evenly across the shots, looping each shot's clip
    to fill its share, then overlay the citation and stamp."""
    citation_text = _escape_drawtext(citation)
    stamp_text = _escape_drawtext(stamp)
    segment = _audio_duration(audio_path) / len(video_paths)

    inputs = []
    filters = []
    for i, path in enumerate(video_paths):
        inputs += ["-stream_loop", "-1", "-i", path]
        # Wan's output size varies slightly (e.g. 726x1270), and concat needs
        # identical dimensions/fps across segments.
        filters.append(
            f"[{i}:v]trim=duration={segment:.3f},setpts=PTS-STARTPTS,"
            f"scale=720:1280,setsar=1,fps=30[s{i}]"
        )
    concat_inputs = "".join(f"[s{i}]" for i in range(len(video_paths)))
    filters.append(f"{concat_inputs}concat=n={len(video_paths)}:v=1:a=0[cat]")
    filters.append(
        f"[cat]drawtext=text='{citation_text}':fontsize=28:fontcolor=white:"
        f"box=1:boxcolor=black@0.4:boxborderw=10:x=20:y=20,"
        f"drawtext=text='{stamp_text}':fontsize=60:fontcolor=white:borderw=3:bordercolor=black:"
        f"x=(w-text_w)/2:y=h-200[v]"
    )
    audio_index = len(video_paths)
    subprocess.run(
        [
            "ffmpeg", "-y",
            *inputs,
            "-i", audio_path,
            "-filter_complex", ";".join(filters),
            "-map", "[v]", "-map", f"{audio_index}:a",
            "-c:v", "libx264", "-c:a", "aac",
            "-shortest",
            final_path,
        ],
        check=True,
        capture_output=True,
    )


def _render_scene(scene, prefix, final_path):
    """Generate every shot's image and clip (in parallel - each clip takes a
    minute or more to render) plus the narration, then assemble."""

    def render_shot(i, shot):
        shot_prefix = f"{prefix}_shot{i}"
        _, image_b64 = _generate_image(shot_prefix, shot["image_prompt"])
        return _generate_video(shot_prefix, shot["video_prompt"], image_b64)

    with ThreadPoolExecutor(max_workers=len(scene["shots"]) + 1) as pool:
        shot_futures = [pool.submit(render_shot, i, shot) for i, shot in enumerate(scene["shots"])]
        audio_future = pool.submit(_generate_narration, prefix, scene["narration"])
        video_paths = [f.result() for f in shot_futures]
        audio_path = audio_future.result()

    _assemble(final_path, video_paths, audio_path, scene["citation"], scene["stamp"])
    return final_path


def generate_video_for_day(day, cycle_id, target_seconds=DEFAULT_TARGET_SECONDS):
    """Run the full pipeline (scene-writing, images, videos, narration,
    assembly) for one calendar day and cache the result. Returns the final
    video path.

    Costs real money (~$0.60 per shot, about $2 for a minute) and takes a few
    minutes - callers must only invoke this on an explicit user action, never
    automatically.
    """
    now_str = datetime.datetime.now().isoformat()
    readings_text = collect_readings_text(day, now_str)
    if not readings_text:
        raise ValueError("No reading text available for this day.")

    system_prompt = build_scene_system_prompt(DAILY_SOURCE_INSTRUCTIONS, target_seconds)
    user_prompt = f"Today's reading:\n\n{readings_text}\n\nPick one moment and write the scene assets."
    scene = generate_scene(system_prompt, user_prompt)
    if not scene.get("citation"):
        scene["citation"] = ", ".join(r.label for r in day.readings) or day.hebrew_date

    return _render_scene(scene, f"{cycle_id}_{day.day_num}", video_path_for(cycle_id, day.day_num))


def generate_video_for_text(text, target_seconds=DEFAULT_TARGET_SECONDS, style="", tone=""):
    """Same pipeline as generate_video_for_day, for arbitrary user-supplied
    text with an optional visual style and narration tone (blank = the Daily
    Mishnah house style). Cached by custom_video_key. Same cost caveats.
    """
    if not text.strip():
        raise ValueError("No text to illustrate.")
    target_seconds = max(MIN_TARGET_SECONDS, min(MAX_TARGET_SECONDS, int(target_seconds)))

    key = custom_video_key(text, target_seconds, style, tone)
    system_prompt = build_scene_system_prompt(
        CUSTOM_SOURCE_INSTRUCTIONS, target_seconds, style.strip() or None, tone.strip() or None
    )
    user_prompt = f"The text:\n\n{text.strip()}\n\nWrite the scene assets."
    scene = generate_scene(system_prompt, user_prompt)
    if not scene.get("citation"):
        scene["citation"] = ""

    return _render_scene(scene, key, custom_video_path(key))

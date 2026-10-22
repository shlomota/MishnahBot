"""Generates a short narrated "explainer scene" video for a day's Mishnah
reading, via OpenRouter (Claude for both the narration script AND the actual
canvas animation code, ElevenLabs for narration audio) and a dependency-free
Node/Chrome renderer (video_explainer/jsrender) for the actual video.

Each day's video is generated once and cached to disk forever at
video_explainer/output/{cycle_id}_{day_num}.mp4, since generation costs real
money (two LLM calls - an initial draft plus a critic/revision pass that
looks at rendered probe frames - plus per-segment TTS; total well under
$0.20) and a given day's reading never changes - see
cached_video_path/generate_video_for_day.

Earlier versions tried, in order: an AI video-diffusion model animating an
AI-illustrated still (hallucinated into incoherent chaos - see git history),
a generic pre-built icon library the LLM picked from (read as flat and
repetitive - "that was terrible"). Both failed because they constrained the
model to either unreliable AI motion or a tiny fixed vocabulary, instead of
letting it actually design the scene.

This version follows a proven pattern (a hand-built reference video,
mishnah-shabbat-1, validated this approach before it was adapted here):
Claude writes bespoke canvas drawing code for that day's specific content -
real creative+technical freedom, not slot-filling - calling into a small
shared primitives library (video_explainer/jsrender/primitives.js: simple
shapes, a hand-drawn "boil" wobble, a reusable person/basket/flame, text,
rubber-stamp pop-ins) plus the raw Canvas 2D API for anything bespoke.
Narration is generated per-segment (not as one block) so each segment's
real spoken duration is known exactly, and a key-phrase-to-cue-time trick
(character offset within that segment's own text, proportional to its
duration) fires visuals on the word that describes them without needing
real forced alignment. The renderer (jsrender/render.js) drives headless
Chrome directly over the CDP WebSocket - no Puppeteer, no Remotion, no
React - evaluating a page-defined drawFrame(t) once per output frame and
piping PNGs straight into ffmpeg, muxed against the assembled narration.
"""
import base64
import datetime
import functools
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import time

import requests
from dotenv import load_dotenv

import sefaria_client as sc

_VIDEO_EXPLAINER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "video_explainer")
load_dotenv(os.path.join(_VIDEO_EXPLAINER_DIR, ".env"))

OUT_DIR = os.path.join(_VIDEO_EXPLAINER_DIR, "output")
os.makedirs(OUT_DIR, exist_ok=True)

JSRENDER_DIR = os.path.join(_VIDEO_EXPLAINER_DIR, "jsrender")

# Gates the generate button: this is a public, un-authenticated page, and
# each generation spends real OpenRouter credits - a PIN is just friction
# against a random visitor clicking it, not real security.
VIDEO_GEN_PIN = os.environ.get("VIDEO_GEN_PIN", "")

BASE = "https://openrouter.ai/api/v1"
SCENE_MODEL = "anthropic/claude-opus-5.5"
TTS_MODEL = "elevenlabs/eleven-v4"
TTS_VOICE = "george"

SEGMENTS_MARKER = "===SEGMENTS==="
CODE_MARKER = "===CODE==="

# Kept in sync with video_explainer/jsrender/primitives.js - this is the
# exact API surface available to generated scene code.
PRIMITIVES_REFERENCE = """Available globals in every scene function (canvas is 1280x720, landscape):
- W, H: canvas width/height (1280, 720).
- X: the raw Canvas2D context - use directly for anything bespoke (arcs, bezier paths, fillRect, save/restore/translate/scale/rotate, etc).
- Colors (hex strings): CREAM, INK, RED, GOLD, OLIVE, PLUM, SKY, PAPER, BROWN, GREEN.
- clamp(v,a,b), lerp(a,b,u), ss(a,b,v) [smoothstep], easeOutCubic(u), easeOutBack(u), easeInCubic(u).
- pop(lt,t0,dur=0.42): 0..1+ eased "pop in with overshoot" value, 0 before t0. fade(lt,t0,dur=0.3): 0..1 fade-in value.
- boil(t,seed,amt=2.2): {x,y,r} hand-drawn jitter that re-rolls ~9x/sec (same seed = same wobble pattern).
  withBoil(t,seed,fn,amt): calls fn() with that jitter applied via translate/rotate.
- line(x1,y1,x2,y2,col=INK,w=5)
- circle(x,y,r,fill,stroke=INK,w=5)
- rr(x,y,w,h,r,fill,stroke=INK,lw=5)  [rounded rect]
- text(str,x,y,size,col=INK,opts={align,baseline,heb:bool,bold,outline:px,outlineCol})
  Pass heb:true for Hebrew text (sets RTL + Hebrew font). opts.align/baseline are canvas text-align/baseline values.
- measure(str,size,heb) -> width in px
- sketchRect(x,y,w,h,t,seed,fill,stroke=INK,lw=5)  [hand-wobbled rectangle outline]
- person(x,y,s,o={robe,patches:bool,armL:[x1,y1,x2,y2],armR:[...],beard,longBeard:bool,squint:bool,brows:bool,hat})
  x,y = ground line (feet position), s = scale (1 ~= 200px tall figure)
- basket(x,y,s,t,glow=0..1)
- flame(x,y,s,t,col=GOLD)
- stamp(str,x,y,lt,t0,col,size=54,rot=-0.13,heb)  [rubber-stamp thunks down and pops in at t0]

Do NOT draw the background, the countdown dial, the chapter-tag corner text, or the progress bar -
the fixed harness already draws all of that around every scene automatically."""

DEFAULT_TARGET_SECONDS = 35
DEFAULT_STYLE = (
    "flat vector illustration, children's-book style, warm earthy palette (terracotta, cream, sage "
    "green), bold clean outlines, soft warm lighting, minimalist backgrounds, playful mood"
)
DEFAULT_TONE = (
    'Channel Monty Python\'s comedic STYLE for the narration - deadpan, over-pedantic logic, sudden '
    'non-sequiturs, mock-serious escalation - as a tone/technique only, never quoting or naming their '
    'actual material; be genuinely random and weird, not just "whimsical".'
)


def _scene_system_prompt(content_kind, target_seconds=DEFAULT_TARGET_SECONDS, style=None, tone=None):
    """Build the scene-writing system prompt, parameterized by target length,
    visual style, and narration tone - used for both the fixed-format Daily
    Mishnah videos (defaults) and the custom-text /video page (overrides)."""
    segment_hint = max(2, min(8, round(target_seconds / 11)))
    style_text = style or DEFAULT_STYLE
    tone_text = tone or DEFAULT_TONE
    return f"""You are writing a ~{target_seconds}-second narrated explainer video, in the style of a \
hand-drawn whiteboard explainer: a bit silly, genuinely funny, but accurate. You have full creative \
and technical control - you are not filling in a template, you are designing and coding the whole \
thing yourself, scene by scene, the way a human animator would.

Cover the actual content of {content_kind} (don't just cherry-pick one moment and ignore the rest) - \
break it into about {segment_hint} short segments, however the content naturally divides (an opening \
hook, one segment per key point or case, a punchy close). Narration is ENGLISH (original wording - \
paraphrase, never quote or closely copy the source text verbatim); Hebrew terms/phrases can and \
should appear as on-screen text where relevant, since that reads well even without a Hebrew voice. \
{tone_text} IMPORTANT: do not make every single sentence a joke - that reads as dense and try-hard, \
not funny. Each segment should mostly be a plain, clear sentence or two just setting the scene / \
stating what's actually happening, THEN one punchier comedic line as the payoff. Let the plain setup \
make the joke land, rather than stacking jokes on jokes.

Visual style: {style_text}.

Keep characters visually STABLE. Do not wrap person()/character drawing in withBoil() or apply \
boil() jitter to a character every frame - that reads as distracting shakiness, not charm. Animate \
characters via pop()/fade() entrances and deliberate pose/prop changes between cues, holding a fixed \
pose otherwise; reserve any wobble for minor background accents only, if at all.

{PRIMITIVES_REFERENCE}

Respond in EXACTLY this two-part format, nothing before the first marker, nothing after the code:

{SEGMENTS_MARKER}
<a JSON array, one object per segment, each with:
  "id": short string e.g. "s1",
  "scene": a short camelCase identifier, e.g. "titleCard" - must be unique and must exactly match a
    key you define in window.SCENES in the code part,
  "text": this segment's narration (one to three sentences),
  "marks": an array of exact short substrings COPIED VERBATIM from this segment's own "text" - each
    is a phrase whose on-screen beat should land as that word is spoken. 1-4 per segment is typical.
  "citation": short string like "Shabbat 1:1" - only needed on the segment(s) actually covering that
    specific passage; omit or leave empty otherwise.
>

{CODE_MARKER}
<raw JavaScript, no markdown fence, no explanation - just the code. Define window.SCENES as an object
whose keys are exactly your scene ids, each value a function(lt, d, q, seg):
  lt = seconds elapsed since this segment started (use this to animate - compare against q[i] and d)
  d  = this segment's total duration in seconds
  q  = array of cue times in seconds (relative to segment start), one per "marks" entry, in order
  seg = the raw segment object (seg.text etc, rarely needed)
Write real, specific, content-matched visuals per scene - new characters/props/compositions as the
content calls for, not the same layout repeated. Use the primitives plus raw X canvas calls freely.>
"""


def video_path_for(cycle_id, day_num):
    return os.path.join(OUT_DIR, f"{cycle_id}_{day_num}.mp4")


def cached_video_path(cycle_id, day_num):
    path = video_path_for(cycle_id, day_num)
    return path if os.path.exists(path) else None


def _headers():
    return {"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"}


def _with_retries(attempts=3, retry_on=(requests.exceptions.RequestException, json.JSONDecodeError, ValueError, subprocess.CalledProcessError)):
    """OpenRouter calls here occasionally hit a transient read-timeout, return
    slightly malformed JSON, or (new in this version) generate JS with a syntax
    error - none worth surfacing to a user mid-generation. Retries with a short
    linear backoff before giving up for real.
    """
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(attempts):
                try:
                    return fn(*args, **kwargs)
                except retry_on as exc:
                    last_exc = exc
                    if attempt < attempts - 1:
                        time.sleep(2 * (attempt + 1))
            raise last_exc
        return wrapper
    return decorator


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


def _normalize_sefaria_ref(raw):
    """Turn a casually-typed reference ("Shabbat 2:3", "Pirkei Avot 1") into a
    Sefaria ref string ("Shabbat.2.3"). Refs that already look fully-formed
    (contain a '.') pass through almost as-is."""
    ref = raw.strip().replace(":", ".")
    if not ref:
        return None
    match = re.match(r"^([A-Za-z][A-Za-z.'\s]*?)\s+([\d.\-]+)$", ref)
    if match:
        title, nums = match.groups()
        ref = f"{title}.{nums}"
    return ref.replace(" ", "_")


def fetch_reference_text(raw_ref, now_str=None):
    """Look up a user-typed reference (e.g. "Shabbat 2:3") on Sefaria and
    return its flattened English text, or None if it can't be found/parsed.
    Most Mishnah tractates resolve directly (Sefaria treats a bare
    "Shabbat.2.3" as the Mishnah when there's no daf/amud notation), but a
    few canonical titles require the "Mishnah_" prefix - try both.
    """
    ref = _normalize_sefaria_ref(raw_ref)
    if not ref:
        return None
    now_str = now_str or datetime.datetime.now().isoformat()
    for candidate in (ref, f"Mishnah_{ref}"):
        data = sc.fetch_text(candidate, now_str)
        if data and data.get("text"):
            paragraphs = sc.flatten_to_paragraphs(data["text"])
            if paragraphs:
                return "\n".join(paragraphs)
    return None


def _strip_json_fences(content):
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip())


def _parse_scene_response(content, citation_fallback=""):
    if SEGMENTS_MARKER not in content or CODE_MARKER not in content:
        raise ValueError("Scene response missing required section markers")
    _, rest = content.split(SEGMENTS_MARKER, 1)
    segments_text, code = rest.split(CODE_MARKER, 1)
    segments = json.loads(_strip_json_fences(segments_text))
    code = code.strip()
    if "```" in code:
        code = _strip_json_fences(code)

    if not segments or not code:
        raise ValueError("Scene response had an empty segments list or code block")
    for seg in segments:
        if not seg.get("id") or not seg.get("scene") or not seg.get("text"):
            raise ValueError(f"Malformed segment: {seg}")
        if seg["scene"] not in code:
            raise ValueError(f"Code never defines scene '{seg['scene']}' referenced by segment {seg['id']}")
        for mark in seg.get("marks", []):
            if mark not in seg["text"]:
                raise ValueError(f"Mark {mark!r} is not a substring of its own segment's text")

    citation = next((s["citation"] for s in segments if s.get("citation")), None) or citation_fallback
    return segments, code, citation


@_with_retries()
def _generate_scene(system_prompt, user_prompt, citation_fallback=""):
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
    return _parse_scene_response(content, citation_fallback)


def generate_scene_for_day(day, readings_text):
    """Ask the LLM to write both the narration script (segments) and the
    bespoke canvas animation code for today's reading.

    Returns (segments, code, citation).
    """
    system_prompt = _scene_system_prompt("today's Mishnah reading")
    user_prompt = f"Today's reading:\n\n{readings_text}\n\nWrite the segments and the scene code."
    citation_fallback = ", ".join(r.label for r in day.readings) or day.hebrew_date
    return _generate_scene(system_prompt, user_prompt, citation_fallback)


def generate_scene_for_text(text, target_seconds=DEFAULT_TARGET_SECONDS, style=None, tone=None):
    """Same as generate_scene_for_day, but for arbitrary user-supplied text
    (the /video page) instead of a day's Mishnah reading. Returns (segments,
    code, citation)."""
    system_prompt = _scene_system_prompt("the following text", target_seconds, style, tone)
    user_prompt = f"Text:\n\n{text}\n\nWrite the segments and the scene code."
    return _generate_scene(system_prompt, user_prompt)


@_with_retries()
def _generate_segment_audio(filename, text):
    resp = requests.post(
        f"{BASE}/audio/speech",
        headers=_headers(),
        json={
            "model": TTS_MODEL,
            "input": text,
            "voice": TTS_VOICE,
            "response_format": "mp3",
        },
        timeout=60,
    )
    resp.raise_for_status()
    with open(os.path.join(OUT_DIR, filename), "wb") as f:
        f.write(resp.content)


def _audio_duration_seconds(filename):
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            os.path.join(OUT_DIR, filename),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def _cue_times(seg, dur):
    """Cue times for a segment's "marks" by character offset within its own
    text (speech time is ~proportional to character position) - this fires
    visuals on the word that describes them without needing real forced
    alignment."""
    return [round(dur * (seg["text"].find(m) / len(seg["text"])), 2) for m in seg.get("marks", [])]


def _sequence_segments(segments, durations):
    """Lay out segments back-to-back (with a small gap) given a per-segment
    duration list, computing start/end/cue for each. Returns (timed, total)."""
    t = 0.0
    gap = 0.12
    timed = []
    for seg, dur in zip(segments, durations):
        start = round(t, 3)
        timed.append({**seg, "start": start, "dur": dur, "end": round(start + dur, 3), "cue": _cue_times(seg, dur)})
        t = start + dur + gap
    total_duration = round(t + 0.75, 3)
    return timed, total_duration


def _build_timed_segments(prefix, segments):
    """Generate per-segment narration audio and measure each one, so each
    segment's real spoken duration (and derived cue times) are exact.
    Returns (timed_segments, total_duration)."""
    durations = []
    filenames = []
    for seg in segments:
        filename = f"{prefix}_{seg['id']}.mp3"
        _generate_segment_audio(filename, seg["text"])
        filenames.append(filename)
        durations.append(round(_audio_duration_seconds(filename), 3))
    timed, total_duration = _sequence_segments(segments, durations)
    for seg, filename in zip(timed, filenames):
        seg["audioFile"] = filename
    return timed, total_duration


def _estimate_duration(text):
    # ~2.5 words/sec is a reasonable stand-in for real TTS pace, good enough
    # for probe-rendering purposes (catching visual bugs, not exact timing).
    return max(2.0, round(len(text.split()) / 2.5, 2))


def _combine_narration_audio(prefix, timed_segments, total_duration):
    """Mix the per-segment clips into one track, each delayed to its own start
    time - equivalent to concatenating them with silence gaps, but lets ffmpeg
    do it in one pass from the per-segment files we already have."""
    inputs = []
    filter_parts = []
    for i, seg in enumerate(timed_segments):
        inputs += ["-i", os.path.join(OUT_DIR, seg["audioFile"])]
        delay_ms = int(seg["start"] * 1000)
        filter_parts.append(f"[{i}:a]adelay={delay_ms}|{delay_ms}[a{i}]")
    mix_inputs = "".join(f"[a{i}]" for i in range(len(timed_segments)))
    filter_complex = ";".join(filter_parts) + f";{mix_inputs}amix=inputs={len(timed_segments)}:duration=longest:dropout_transition=0[aout]"
    filename = f"{prefix}_narration.mp3"
    subprocess.run(
        [
            "ffmpeg", "-y", *inputs,
            "-filter_complex", filter_complex, "-map", "[aout]",
            "-t", str(total_duration),
            os.path.join(OUT_DIR, filename),
        ],
        check=True,
        capture_output=True,
    )
    return filename


def _check_js_syntax(path):
    subprocess.run(["node", "--check", path], check=True, capture_output=True, text=True)


def _render_probe_frames(code, segments, max_frames=8):
    """Render a handful of representative frames (no real audio needed yet -
    estimated timing is close enough to catch visual bugs) so a critic pass
    can actually look at the draft before it's finalized, the same way we
    caught the lamp-head bug by eye rather than assuming the code was fine.
    Returns a list of base64-encoded PNGs.
    """
    estimated_durations = [_estimate_duration(seg["text"]) for seg in segments]
    timed, total_duration = _sequence_segments(segments, estimated_durations)

    work_dir = os.path.join(OUT_DIR, "_probe_work")
    shutil.rmtree(work_dir, ignore_errors=True)
    os.makedirs(work_dir)
    try:
        shutil.copy(os.path.join(JSRENDER_DIR, "index.html"), work_dir)
        shutil.copy(os.path.join(JSRENDER_DIR, "primitives.js"), work_dir)
        scenes_path = os.path.join(work_dir, "scenes.js")
        with open(scenes_path, "w") as f:
            f.write(code)
        _check_js_syntax(scenes_path)

        timing = {"duration": total_duration, "narrationFile": "", "titleHeb": "", "segments": timed}
        with open(os.path.join(work_dir, "timing.json"), "w") as f:
            json.dump(timing, f)
        with open(os.path.join(work_dir, "timing.js"), "w") as f:
            f.write("window.TIMING = " + json.dumps(timing) + ";\n")

        probe_times = [round(s["start"] + s["dur"] * 0.7, 2) for s in timed][:max_frames]
        subprocess.run(
            ["node", "render.js", work_dir, "--probe", ",".join(str(t) for t in probe_times)],
            cwd=JSRENDER_DIR,
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        images = []
        for png_path in sorted(glob.glob(os.path.join(work_dir, "probe", "*.png"))):
            with open(png_path, "rb") as f:
                images.append(base64.b64encode(f.read()).decode("ascii"))
        return images
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


CRITIC_SYSTEM_PROMPT = f"""You are reviewing a draft ~30-45s narrated Mishnah explainer video before \
it's finalized. You're given the original narration segments (JSON) and the full canvas drawing code \
written for them, plus several actual rendered probe frames from this draft so you can see what it \
currently looks like. Two things need fixing:

1. VISUAL BUGS: look at each attached frame. Fix anything broken, garbled, badly overlapping, or \
confusing - e.g. a prop awkwardly stacked on a character's head or face, text running off-canvas or \
overlapping other elements, a frame that's mostly empty/sparse, illegible text. If characters look \
shaky/jittery, that's a defect: remove any withBoil()/boil() wrapping on character-drawing code and \
keep characters visually stable (fixed pose, animated only via pop()/fade() entrances and deliberate \
pose/prop changes between cues).

2. NARRATION PACING: the draft likely tries to make every sentence a joke, which reads as dense and \
try-hard rather than funny. Rewrite so each segment mostly reads as a plain, clear sentence or two \
just stating what's actually happening, THEN one punchier comedic payoff line - not constant \
joke-density. Keep whichever jokes already land well; replace the ones that don't.

Keep each segment's "marks" as exact substrings of its (possibly rewritten) "text". Keep each \
segment's "id"/"scene"/"citation" matched to the code unless you deliberately change which scene a \
segment maps to (then update the code to match).

{PRIMITIVES_REFERENCE}

Respond in the exact same two-part format as the original (segments then code) with your revised, \
final version of both - repeat a segment/scene unchanged if it didn't need fixing."""


@_with_retries()
def _critic_review(segments, code, citation_fallback, probe_images_b64):
    draft_text = f"{SEGMENTS_MARKER}\n{json.dumps(segments, ensure_ascii=False)}\n\n{CODE_MARKER}\n{code}"
    user_content = [{"type": "text", "text": draft_text}]
    for b64 in probe_images_b64:
        user_content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
    resp = requests.post(
        f"{BASE}/chat/completions",
        headers=_headers(),
        json={
            "model": SCENE_MODEL,
            "messages": [
                {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.7,
        },
        timeout=120,
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    return _parse_scene_response(content, citation_fallback)


@_with_retries()
def _render(final_path, timed_segments, total_duration, narration_filename, code, title_heb):
    """Assemble a per-generation work directory (shared harness files +
    this day's timing data and generated scene code) and render it via
    jsrender/render.js - headless Chrome driven directly over CDP, no
    Puppeteer/Remotion/React involved. Raises (and lets @_with_retries catch)
    on a JS syntax error in the generated code or a render failure, since
    both are plausibly transient/model-variance issues worth one more try.
    """
    work_dir = os.path.join(OUT_DIR, f"_work_{os.path.basename(final_path)}")
    shutil.rmtree(work_dir, ignore_errors=True)
    os.makedirs(work_dir)
    try:
        shutil.copy(os.path.join(JSRENDER_DIR, "index.html"), work_dir)
        shutil.copy(os.path.join(JSRENDER_DIR, "primitives.js"), work_dir)
        shutil.copy(os.path.join(OUT_DIR, narration_filename), work_dir)

        scenes_path = os.path.join(work_dir, "scenes.js")
        with open(scenes_path, "w") as f:
            f.write(code)
        _check_js_syntax(scenes_path)

        timing = {
            "duration": total_duration,
            "narrationFile": narration_filename,
            "titleHeb": title_heb,
            "segments": timed_segments,
        }
        with open(os.path.join(work_dir, "timing.json"), "w") as f:
            json.dump(timing, f)
        with open(os.path.join(work_dir, "timing.js"), "w") as f:
            f.write("window.TIMING = " + json.dumps(timing) + ";\n")

        subprocess.run(
            ["node", "render.js", work_dir, final_path],
            cwd=JSRENDER_DIR,
            check=True,
            capture_output=True,
            text=True,
            timeout=180,
        )
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def _run_pipeline(segments, code, citation, prefix, final_path):
    """Shared second half of the pipeline (critic pass, narration, render) -
    used by both generate_video_for_day and generate_video_for_text once
    each has produced an initial draft (segments, code, citation)."""
    probe_images = _render_probe_frames(code, segments)
    segments, code, citation = _critic_review(segments, code, citation, probe_images)

    timed_segments, total_duration = _build_timed_segments(prefix, segments)
    narration_filename = _combine_narration_audio(prefix, timed_segments, total_duration)

    _render(final_path, timed_segments, total_duration, narration_filename, code, citation)
    return final_path


def generate_video_for_day(day, cycle_id):
    """Run the full pipeline (scene-writing, critic review, per-segment
    narration, render) for one calendar day and cache the result. Returns
    the final video path.

    Costs real money (well under $0.20) and takes well under two minutes -
    callers must only invoke this on an explicit user action, never
    automatically.
    """
    now_str = datetime.datetime.now().isoformat()
    readings_text = collect_readings_text(day, now_str)
    if not readings_text:
        raise ValueError("No reading text available for this day.")

    segments, code, citation = generate_scene_for_day(day, readings_text)
    prefix = f"{cycle_id}_{day.day_num}"
    final_path = video_path_for(cycle_id, day.day_num)
    return _run_pipeline(segments, code, citation, prefix, final_path)


def custom_video_key(text, length_seconds, style, tone):
    raw = json.dumps([text, length_seconds, style or "", tone or ""])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def custom_video_path(key):
    return os.path.join(OUT_DIR, f"custom_{key}.mp4")


def generate_video_for_text(text, length_seconds=DEFAULT_TARGET_SECONDS, style=None, tone=None):
    """Same pipeline as generate_video_for_day, but for arbitrary user text
    (the /video page) with an optional target length/style/tone. Caches by a
    hash of (text, length, style, tone) - see custom_video_key/path."""
    if not text or not text.strip():
        raise ValueError("No text provided.")

    segments, code, citation = generate_scene_for_text(text, length_seconds, style, tone)
    prefix = f"custom_{custom_video_key(text, length_seconds, style, tone)}"
    final_path = custom_video_path(custom_video_key(text, length_seconds, style, tone))
    return _run_pipeline(segments, code, citation, prefix, final_path)

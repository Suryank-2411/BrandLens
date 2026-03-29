"""
pipeline.py  —  BrandLens LangGraph Pipeline
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Node A  →  Vision     : Gemini analyses frames — logo, object it's on, position, size
Node B  →  Audio      : Whisper transcribes video audio with timestamps
Node C  →  Summary    : For each brand appearance:
                          1. Grabs ±10s audio transcript
                          2. Gemini analyses sentiment (PROMOTING / DEMOTING / NEUTRAL)
                          3. Gemini summarises what's being said
                          4. Gemini generates combined visual+audio insight paragraph

Video sources supported:
  - YouTube URL       (https://youtube.com/watch?v=...)
  - S3 URI            (s3://bucket/key)
  - S3 HTTPS URL      (https://bucket.s3.amazonaws.com/key)
  - Local file path   (/path/to/video.mp4)

To upgrade Gemini model — change GEMINI_MODEL one line below.
"""

import os, json, re, time, tempfile, subprocess
from typing import TypedDict, List, Dict, Any, Optional

import cv2
import yt_dlp
import google.generativeai as genai
from PIL import Image
from langgraph.graph import StateGraph, END

# ══════════════════════════════════════════════════════════════════════════════
#  ↓↓  CHANGE THIS ONE LINE TO UPGRADE MODEL  ↓↓
# ══════════════════════════════════════════════════════════════════════════════
#  Free / demo  :  "gemini-1.5-flash"    (15 req/min, 1500 req/day free)
#  Better       :  "gemini-1.5-pro"
#  Latest best  :  "gemini-2.0-flash"
GEMINI_MODEL = "gemini-2.0-flash"
# ══════════════════════════════════════════════════════════════════════════════

SAMPLE_FPS     = 1      # frames analysed per second
CONFIDENCE_MIN = 0.5    # drop detections below this
GAP_TOLERANCE  = 2.0    # seconds — merge nearby appearances of same brand
MIN_DURATION   = 0.5    # drop appearances shorter than this
REQUEST_DELAY  = 10.0    # seconds between Gemini calls (free tier = 15 req/min)
AUDIO_WINDOW   = 20.0   # seconds either side of detection for audio analysis

TEMP_DIR   = os.path.join(tempfile.gettempdir(), "brandlens")
VIDEO_PATH = os.path.join(TEMP_DIR, "video.mp4")
AUDIO_PATH = os.path.join(TEMP_DIR, "audio.wav")
os.makedirs(TEMP_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  LANGGRAPH STATE
# ══════════════════════════════════════════════════════════════════════════════
class PipelineState(TypedDict):
    # inputs
    video_source:   str
    user_prompt:    str
    gemini_api_key: str
    sample_fps:     int
    confidence_min: float

    # intermediate
    video_path:     str
    frames:         List[Any]   # [(timestamp_float, frame_ndarray), ...]
    raw_detections: List[Dict]  # one dict per logo per frame
    transcript:     List[Dict]  # [{"start":0.0, "end":2.5, "text":"..."}, ...]

    # output
    results:        List[Dict]
    error:          Optional[str]

    # UI callback — injected at runtime, not serialised
    progress_cb:    Optional[Any]


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def _log(state: PipelineState, msg: str):
    cb = state.get("progress_cb")
    if cb:
        cb(msg)
    print(f"[BrandLens] {msg}")


def _parse_gemini_json(text: str) -> Any:
    """Strip markdown fences and parse first JSON structure from Gemini output."""
    text = re.sub(r"```json|```", "", text).strip()
    # Try array first, then object
    for pattern in [r"\[.*\]", r"\{.*\}"]:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                continue
    return [] if text.strip().startswith("[") else {}


# ══════════════════════════════════════════════════════════════════════════════
#  VIDEO DOWNLOAD  (YouTube / S3 / local)
# ══════════════════════════════════════════════════════════════════════════════
def download_video(source: str, progress_cb=None) -> str:
    out = VIDEO_PATH
    if os.path.exists(out):
        os.remove(out)

    # Local file
    if os.path.exists(source):
        if progress_cb: progress_cb(f"Using local file: {source}")
        return source

    # S3
    if source.startswith("s3://") or ".s3." in source or ".s3.amazonaws.com" in source:
        if progress_cb: progress_cb("Downloading from S3...")
        try:
            import boto3
            if source.startswith("s3://"):
                parts  = source[5:].split("/", 1)
                bucket, key = parts[0], parts[1] if len(parts) > 1 else ""
            else:
                from urllib.parse import urlparse
                p      = urlparse(source)
                bucket = p.netloc.split(".")[0]
                key    = p.path.lstrip("/")
            boto3.client("s3").download_file(bucket, key, out)
        except Exception as e:
            raise RuntimeError(f"S3 download failed: {e}")
        if progress_cb: progress_cb("S3 download complete ✅")
        return out

    # YouTube / any yt-dlp supported URL
    if progress_cb: progress_cb("Downloading video from URL...")

    def hook(d):
        if progress_cb and d["status"] == "downloading":
            progress_cb(f"Downloading... {d.get('_percent_str','').strip()}")

    opts = {
        "outtmpl":        out,
        "format":         "best[ext=mp4][height<=720]/best[ext=mp4]/best",
        "progress_hooks": [hook],
        "quiet":          True,
        "no_warnings":    True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([source])

    if progress_cb: progress_cb("Download complete ✅")
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  FRAME EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════
def extract_frames(video_path: str, sample_fps: int = SAMPLE_FPS):
    cap        = cv2.VideoCapture(video_path)
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total      = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval   = max(1, int(native_fps / sample_fps))
    frames, idx = [], 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if idx % interval == 0:
            frames.append((round(idx / native_fps, 2), frame))
        idx += 1

    cap.release()
    duration = round(total / native_fps, 1) if native_fps else 0
    return frames, duration


# ══════════════════════════════════════════════════════════════════════════════
#  NODE A — VISION  (Gemini: logo + object context + position)
# ══════════════════════════════════════════════════════════════════════════════
VISION_PROMPT = """
You are an expert brand analyst reviewing a single video frame.

TASK:
Find every brand logo visible in this image — on clothing, cups, bottles,
billboards, screens, packaging, vehicles, accessories, or anywhere else.

For each logo found, return:
- brand        : exact brand name (e.g. "Nike", "Coca-Cola")
- confidence   : 0.0–1.0, how certain you are this is the correct brand
- object       : what physical object the logo appears on
                 (e.g. "athlete's t-shirt", "coffee cup", "car door",
                  "billboard", "laptop sticker", "shopping bag").
                 If you cannot identify the object, write "unknown object".
- description  : one sentence describing what you see
                 (e.g. "Nike swoosh on white t-shirt worn by the presenter on the left")
- position.x   : horizontal centre of logo, 0.0 (left) to 1.0 (right)
- position.y   : vertical centre of logo, 0.0 (top) to 1.0 (bottom)
- position.quadrant : one of — top-left / top-center / top-right /
                      center-left / center / center-right /
                      bottom-left / bottom-center / bottom-right
- size_pct     : estimated logo area as % of total frame area

{extra}

Return ONLY a valid JSON array. No prose. No markdown. Example:
[
  {{
    "brand": "Nike",
    "confidence": 0.91,
    "object": "athlete's t-shirt",
    "description": "Nike swoosh logo on white t-shirt worn by the presenter",
    "position": {{"x": 0.45, "y": 0.32, "quadrant": "top-center"}},
    "size_pct": 2.1
  }}
]

If NO logos are visible, return exactly: []
"""


def node_vision(state: PipelineState) -> PipelineState:
    """Node A — Send each frame to Gemini, collect logo detections with context."""
    _log(state, "▶ Node A — Vision: starting frame analysis...")

    genai.configure(api_key=state["gemini_api_key"])
    model  = genai.GenerativeModel(GEMINI_MODEL)
    frames = state["frames"]
    conf   = state.get("confidence_min", CONFIDENCE_MIN)

    extra = ""
    if state.get("user_prompt", "").strip():
        extra = f"\nExtra instructions from analyst:\n{state['user_prompt']}"

    prompt     = VISION_PROMPT.format(extra=extra)
    detections = []

    for i, (timestamp, frame) in enumerate(frames):
        if i % 3 == 0:
            pct = int(i / len(frames) * 100)
            _log(state, f"Node A — frame {i+1}/{len(frames)} ({timestamp:.1f}s) [{pct}%]")

        try:
            pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            rsp = model.generate_content([prompt, pil])
            logos = _parse_gemini_json(rsp.text)
            if not isinstance(logos, list):
                logos = []

            for logo in logos:
                if logo.get("confidence", 0) < conf:
                    continue
                p = logo.get("position", {})
                detections.append({
                    "timestamp":   timestamp,
                    "brand":       logo.get("brand", "Unknown"),
                    "confidence":  round(float(logo.get("confidence", 0)), 4),
                    "object":      logo.get("object", "unknown object"),
                    "description": logo.get("description", ""),
                    "size_pct":    round(float(logo.get("size_pct", 0)), 2),
                    "cx":          round(float(p.get("x", 0.5)), 3),
                    "cy":          round(float(p.get("y", 0.5)), 3),
                    "quadrant":    p.get("quadrant", "unknown"),
                })

        except Exception as e:
            _log(state, f"  [WARN] Frame {i} skipped: {e}")

        time.sleep(REQUEST_DELAY)

    _log(state, f"Node A ✅ — {len(detections)} raw detections")
    return {**state, "raw_detections": detections}


# ══════════════════════════════════════════════════════════════════════════════
#  NODE B — AUDIO  (Whisper transcription with timestamps)
# ══════════════════════════════════════════════════════════════════════════════
def node_audio(state: PipelineState) -> PipelineState:
    """Node B — Extract audio from video and transcribe with Whisper."""
    _log(state, "▶ Node B — Audio: extracting and transcribing...")
    segments = []

    try:
        # Extract WAV with ffmpeg
        cmd = [
            "ffmpeg", "-y", "-i", state["video_path"],
            "-ar", "16000", "-ac", "1", "-f", "wav",
            AUDIO_PATH, "-loglevel", "quiet"
        ]
        result = subprocess.run(cmd, capture_output=True)

        if result.returncode != 0:
            _log(state, "  [WARN] ffmpeg audio extraction failed — skipping transcription")
            return {**state, "transcript": []}

        # Whisper — "tiny" for demo speed (free, no API needed)
        # Upgrade to "base" or "small" for better accuracy
        import whisper
        _log(state, "  Loading Whisper tiny model...")
        wmodel = whisper.load_model("tiny")
        _log(state, "  Transcribing audio...")
        out = wmodel.transcribe(AUDIO_PATH, verbose=False)

        for seg in out.get("segments", []):
            segments.append({
                "start": round(seg["start"], 2),
                "end":   round(seg["end"],   2),
                "text":  seg["text"].strip(),
            })

        _log(state, f"Node B ✅ — {len(segments)} transcript segments")

    except Exception as e:
        _log(state, f"  [WARN] Transcription failed: {e} — continuing without audio")

    return {**state, "transcript": segments}


# ══════════════════════════════════════════════════════════════════════════════
#  AGGREGATION  (runs between Node B and Node C)
# ══════════════════════════════════════════════════════════════════════════════
def _get_audio_context(transcript: List[Dict], start: float,
                       end: float, window: float = None) -> str:
    """Return transcript text overlapping [start-window, end+window]."""
    w  = window if window is not None else AUDIO_WINDOW
    lo = start - w
    hi = end   + w
    return " ".join(
        s["text"] for s in transcript
        if s["end"] >= lo and s["start"] <= hi
    ).strip()


def _aggregate(raw: List[Dict]) -> List[Dict]:
    """Merge raw per-frame detections into brand appearance windows."""
    if not raw:
        return []

    import pandas as pd
    df      = pd.DataFrame(raw)
    results = []

    for brand, grp in df.groupby("brand"):
        grp  = grp.sort_values("timestamp").reset_index(drop=True)
        segs = []
        s, e, rows = grp.loc[0,"timestamp"], grp.loc[0,"timestamp"], [grp.loc[0]]

        for i in range(1, len(grp)):
            t = grp.loc[i, "timestamp"]
            if t - e <= GAP_TOLERANCE:
                e = t; rows.append(grp.loc[i])
            else:
                segs.append((s, e, pd.DataFrame(rows)))
                s, e, rows = t, t, [grp.loc[i]]
        segs.append((s, e, pd.DataFrame(rows)))

        appearances = []
        for seg_start, seg_end, seg in segs:
            dur = round(seg_end - seg_start, 2)
            if dur < MIN_DURATION:
                continue
            best = seg.loc[seg["confidence"].idxmax()]
            appearances.append({
                "start_sec":      seg_start,
                "end_sec":        seg_end,
                "duration_sec":   dur,
                "avg_confidence": round(float(seg["confidence"].mean()), 4),
                "best_confidence":round(float(best["confidence"]), 4),
                "avg_size_pct":   round(float(seg["size_pct"].mean()), 2),
                "avg_position": {
                    "x":        round(float(seg["cx"].mean()), 3),
                    "y":        round(float(seg["cy"].mean()), 3),
                    "quadrant": seg["quadrant"].mode()[0],
                },
                "objects":      list(seg["object"].dropna().unique()),
                "descriptions": list(seg["description"].dropna().unique()),
                # filled by Node C
                "audio_context":            "",
                "audio_sentiment":          "",
                "audio_how":                "",
                "audio_transcript_summary": "",
                "summary":                  "",
            })

        if appearances:
            results.append({
                "brand":              brand,
                "total_duration_sec": round(sum(a["duration_sec"] for a in appearances), 2),
                "appearance_count":   len(appearances),
                "appearances":        appearances,
            })

    results.sort(key=lambda x: x["total_duration_sec"], reverse=True)
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  NODE C — SUMMARY  (Gemini: audio sentiment + transcript summary + insight)
# ══════════════════════════════════════════════════════════════════════════════

AUDIO_ANALYSIS_PROMPT = """
You are a brand sentiment analyst.

A brand logo ({brand}) was visually detected in a video between {start}s and {end}s.
Below is the audio transcript from ±10 seconds around that moment.

Transcript:
"{transcript}"

Analyse this transcript and answer THREE things:

1. SENTIMENT — Is the speaker PROMOTING, DEMOTING, or NEUTRAL about {brand}?
   Definitions:
   - PROMOTING     = positive mentions, recommendations, praise, endorsement, excitement
   - DEMOTING      = negative mentions, criticism, warnings, complaints, mockery
   - NEUTRAL       = brand shown/mentioned without clear positive or negative tone
   - NOT MENTIONED = brand is not referred to in speech at all

2. HOW — In 1–2 sentences, explain specifically HOW they are promoting/demoting/neutral.
   What exactly did they say or imply about the brand?
   If NOT MENTIONED, write what the speaker is doing instead.

3. TRANSCRIPT SUMMARY — In 1–2 sentences, summarise what the speaker is generally
   talking about during this time window (not just about the brand).

Return ONLY a valid JSON object. No prose. No markdown:
{{
  "sentiment": "PROMOTING",
  "how": "The speaker directly recommends buying Nike shoes and says they are the best for marathon training.",
  "transcript_summary": "The presenter is reviewing running equipment and giving product recommendations for beginners."
}}
"""

SUMMARY_PROMPT = """
You are a brand placement analyst writing a concise insight paragraph.

You have visual detection data AND audio transcript analysis for one brand appearance.

Brand detected : {brand}
Time window    : {start}s → {end}s  ({dur}s duration)
Logo appears on: {objects}
Visual detail  : {descriptions}
Screen position: {quadrant}  (x={x}, y={y})
Logo size      : {size_pct}% of screen
Confidence     : {confidence}

Audio transcript (±10s):
"{audio}"

Audio sentiment about {brand}: {sentiment}
How they promote/demote/mention: {how}
What's generally being discussed: {transcript_summary}

{extra}

Write ONE paragraph (2–4 sentences) combining ALL the above into a clear,
human-readable brand placement insight. Mention:
• What the logo looked like and what physical object it was on
• What was happening / being said at that moment
• Whether the brand is being promoted, criticised, or just shown passively
• Any noteworthy context about the placement

No bullet points. Plain paragraph only. Under 4 sentences.
"""


def node_summary(state: PipelineState) -> PipelineState:
    """Node C — Audio sentiment analysis + transcript summary + combined insight."""
    _log(state, "▶ Node C — Summary: generating insights...")

    genai.configure(api_key=state["gemini_api_key"])
    model      = genai.GenerativeModel(GEMINI_MODEL)
    transcript = state.get("transcript", [])

    extra = ""
    if state.get("user_prompt", "").strip():
        extra = f"Additional context from analyst: {state['user_prompt']}"

    results = _aggregate(state["raw_detections"])
    total   = sum(b["appearance_count"] for b in results)
    done    = 0

    for brand_data in results:
        for ap in brand_data["appearances"]:
            done += 1
            _log(state, f"Node C — {brand_data['brand']} appearance {done}/{total}")

            # ── Attach ±10s audio context ─────────────────────────────────────
            audio = _get_audio_context(transcript, ap["start_sec"], ap["end_sec"])
            ap["audio_context"] = audio

            # ── Step 1: Audio sentiment & transcript analysis ─────────────────
            audio_sentiment          = "NEUTRAL"
            audio_how                = ""
            audio_transcript_summary = ""

            if audio.strip():
                try:
                    a_prompt = AUDIO_ANALYSIS_PROMPT.format(
                        brand=brand_data["brand"],
                        start=ap["start_sec"],
                        end=ap["end_sec"],
                        transcript=audio,
                    )
                    a_rsp  = model.generate_content(a_prompt)
                    a_json = _parse_gemini_json(a_rsp.text)
                    if isinstance(a_json, dict):
                        audio_sentiment          = a_json.get("sentiment", "NEUTRAL")
                        audio_how                = a_json.get("how", "")
                        audio_transcript_summary = a_json.get("transcript_summary", "")
                    _log(state, f"  Audio sentiment: {audio_sentiment}")
                except Exception as ae:
                    _log(state, f"  [WARN] Audio analysis failed: {ae}")

                time.sleep(REQUEST_DELAY)
            else:
                audio_sentiment          = "NOT MENTIONED"
                audio_how                = "No audio detected at this time window."
                audio_transcript_summary = "No audio available."

            ap["audio_sentiment"]          = audio_sentiment
            ap["audio_how"]                = audio_how
            ap["audio_transcript_summary"] = audio_transcript_summary

            # ── Step 2: Combined visual + audio insight paragraph ─────────────
            try:
                p = ap["avg_position"]
                s_prompt = SUMMARY_PROMPT.format(
                    brand=brand_data["brand"],
                    start=ap["start_sec"],
                    end=ap["end_sec"],
                    dur=ap["duration_sec"],
                    objects=", ".join(ap["objects"]) or "unknown object",
                    descriptions="; ".join(ap["descriptions"]) or "—",
                    quadrant=p["quadrant"],
                    x=p["x"],
                    y=p["y"],
                    size_pct=ap["avg_size_pct"],
                    confidence=ap["best_confidence"],
                    audio=audio or "(no audio detected)",
                    sentiment=audio_sentiment,
                    how=audio_how or "—",
                    transcript_summary=audio_transcript_summary or "—",
                    extra=extra,
                )
                s_rsp      = model.generate_content(s_prompt)
                ap["summary"] = s_rsp.text.strip()

            except Exception as e:
                _log(state, f"  [WARN] Summary generation failed: {e}")
                ap["summary"] = (
                    f"{brand_data['brand']} logo detected on "
                    f"{', '.join(ap['objects']) or 'unknown object'} "
                    f"at {ap['start_sec']}s–{ap['end_sec']}s. "
                    f"Audio sentiment: {audio_sentiment}."
                )

            time.sleep(REQUEST_DELAY)

    _log(state, f"Node C ✅ — {total} appearances processed")
    return {**state, "results": results}


# ══════════════════════════════════════════════════════════════════════════════
#  BUILD & COMPILE LANGGRAPH
# ══════════════════════════════════════════════════════════════════════════════
def _build_graph():
    g = StateGraph(PipelineState)
    g.add_node("vision",  node_vision)
    g.add_node("audio",   node_audio)
    g.add_node("summary", node_summary)
    g.set_entry_point("vision")
    g.add_edge("vision",  "audio")
    g.add_edge("audio",   "summary")
    g.add_edge("summary", END)
    return g.compile()


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC ENTRY POINT  (called by app.py)
# ══════════════════════════════════════════════════════════════════════════════
def run_analysis(
    video_source:   str,
    gemini_api_key: str,
    user_prompt:    str   = "",
    sample_fps:     int   = SAMPLE_FPS,
    confidence_min: float = CONFIDENCE_MIN,
    progress_cb           = None,
) -> List[Dict]:

    if progress_cb: progress_cb("Downloading video...")
    video_path = download_video(video_source, progress_cb=progress_cb)

    if progress_cb: progress_cb("Extracting frames...")
    frames, duration = extract_frames(video_path, sample_fps=sample_fps)
    if progress_cb:
        progress_cb(f"Extracted {len(frames)} frames from {duration}s video ✅")

    graph = _build_graph()

    final = graph.invoke({
        "video_source":   video_source,
        "user_prompt":    user_prompt,
        "gemini_api_key": gemini_api_key,
        "sample_fps":     sample_fps,
        "confidence_min": confidence_min,
        "video_path":     video_path,
        "frames":         frames,
        "raw_detections": [],
        "transcript":     [],
        "results":        [],
        "error":          None,
        "progress_cb":    progress_cb,
    })
    return final["results"]

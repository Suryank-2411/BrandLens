# BrandLens
BrandLens — AI-Powered Brand Intelligence Platform
BrandLens is an AI-powered video analysis platform that automatically detects brand logos in any video, understands the context of each appearance, and generates actionable insights by combining visual and audio intelligence.
Simply provide a YouTube link or S3 video URL, and BrandLens runs a 3-stage AI pipeline — it scans every frame using Gemini Vision to detect brand logos along with what object they appear on (a t-shirt, a cup, a billboard), transcribes the audio using Whisper to capture what's being said at each moment, and then combines both signals to determine whether the brand is being actively promoted, criticised, or passively shown — generating a human-readable insight for every single detection.
The results are presented in an interactive dashboard showing a brand appearance timeline, screen position heatmap, sentiment breakdown, and per-appearance AI summaries, all exportable as CSV or JSON.
Built with: Python · LangGraph · Gemini Vision API · OpenAI Whisper · Streamlit · Plotly
Key capabilities:

Detects any brand logo without a predefined list — powered by vision AI
Identifies what the logo is physically on (clothing, packaging, signage, etc.)
Analyses ±10 seconds of audio per detection to classify brand sentiment
Supports YouTube URLs, S3 links, and local video files

Input (YouTube URL / S3 URL / User Prompt)
              ↓
    ┌─────────────────────┐
    │   Node A — Vision   │  Gemini analyses each frame
    │   Logo + Context    │  "Nike logo on athlete's jersey"
    └──────────┬──────────┘
               │
    ┌──────────▼──────────┐
    │   Node B — Audio    │  Whisper extracts transcript
    │   Transcription     │  + aligns to timestamps
    └──────────┬──────────┘
               │
    ┌──────────▼──────────┐
    │   Node C — Summary  │  Gemini combines vision +
    │   Generation        │  audio → per-frame summary
    └──────────┬──────────┘
               ↓
          Streamlit UI

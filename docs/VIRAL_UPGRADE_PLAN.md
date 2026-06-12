# Viral Upgrade Plan — better videos + new formats

> Drafted 2026-06-12, branch `fix/publish-ready-video-enhancements`.
> Follows REWORK_PLAN.md phases 0–9 (all complete, 269 tests green).
> Goal: production-grade short-form output + new viral video formats.

## Track A — quality of every video

### WP1. Karaoke word-level captions (Opus)

**Implemented.** `subtitle_style = "karaoke"` is the default; word-timing
sidecars (`.words.json`) written next to every SRT from both Whisper and
edge-tts/Azure WordBoundary events; video.py renders per-word highlight
timeline with configurable `subtitle_highlight_color` (default `#FFE600`);
graceful fallback to `"tiktok"` phrase captions when word timings are missing.

The #1 perceived-quality gap vs real TikToks. Whisper already returns
`word_timestamps=True` (subtitle.py) and edge-tts/Azure emit WordBoundary
events (voice.py sub_maker) — but everything is flattened into phrase SRTs.
- New `subtitle_style = "karaoke"` (make it the default): captions of ≤3–4
  words, currently-spoken word highlighted (color + slight scale).
- subtitle.py emits word-timing sidecar (JSON) next to the SRT from both
  timing sources (whisper words / sub_maker word boundaries).
- video.py renders highlight timeline from the sidecar; if no word timings →
  graceful fallback to current phrase captions.
- Files: `app/services/subtitle.py`, `app/services/voice.py` (sidecar only),
  `app/services/video.py`, config keys, tests.

### WP3. SFX + audio asset scaffolding (Sonnet)

**Implemented.** `resource/sfx/` directory with synthesized CC0 sounds
(whoosh, pop, ding, tick, riser) generated via `tools/make_sfx.py`;
`app/services/sfx.py` (`get_sfx_file(name)`); config block in
config.example.toml: `sfx_enabled`, `sfx_volume`, `bgm_ducking_enabled`,
`bgm_duck_volume`.

- `resource/sfx/` with synthesized CC0-free sounds (whoosh, pop, ding, tick,
  riser) generated deterministically via ffmpeg (`tools/make_sfx.py`) — no
  network downloads.
- Helper `app/services/sfx.py` → `get_sfx_file(name) -> str` ('' if missing).
- Config keys: `sfx_enabled`, `sfx_volume`, `bgm_ducking_enabled`,
  `bgm_duck_volume` in config.example.toml (new delimited block).

### WP4. Audio polish: ducking + transition SFX (Opus, after WP1+WP3)

**Implemented.** BGM ducking with ~200 ms ramps (`bgm_ducking_enabled`,
`bgm_duck_volume`); whoosh SFX at scene cuts (`sfx_enabled`, `sfx_volume`);
all non-fatal — missing assets or disabled flags fall back to the original
audio path.

- BGM ducks under voice with ~200 ms ramps; swells in narration gaps and at
  the end. Voice-activity envelope = subtitle/word ranges from WP1 sidecar.
- Whoosh/pop SFX at scene cut points; all behind config flags; missing
  assets / disabled flags → exactly today's behavior (non-fatal).
- Files: `app/services/video.py` (audio mix section), tests.

### WP6. Hook-first scripting + pacing (Opus, wave 2)

**Implemented.** LLM prompts enforce ≤8-word pattern-interrupt opening (EN+RU,
no greetings), plus a script sanitizer; scenes capped at ~4 s via
`scene_max_seconds` (default 4.0, configurable in config.toml under `[app]`);
first-frame visibility fix shipped in studio.py (hook scene skips entrance
animation, fully readable at frame 0).

- llm.py prompts: first sentence = pattern-interrupt hook (≤8 words, no
  greetings/intro filler); tight sentences; optional CTA at the end.
- scenes.py/plan.py: cap scene duration (split long scenes ~≤4 s), first
  visual appears immediately (no slow intro tween).
- Files: `app/services/llm.py`, `app/services/hyperframes/{scenes,plan}.py`.

## Track B — new viral formats (studio modes)

Pattern to follow: `compose_news` in studio.py + mode wiring in
`hyperframes/__init__.py` (`_VALID_MODES`) + task.py orchestration.

### WP2. Quiz + Top-N ranking formats (Opus)

**Implemented.** `video_visual_mode = "quiz"` and `"ranking"` registered;
`llm.generate_quiz` / `llm.generate_ranking` with strict-JSON parsing + retry;
`compose_quiz` / `compose_ranking` in studio.py; `render_quiz_video` /
`render_ranking_video` + task.py dispatch with non-fatal fallback to the
regular pipeline.

- `quiz`: LLM emits structured Q/A JSON → per question: question card →
  animated 3-2-1 countdown (silence/tick in audio between question and
  answer TTS) → answer reveal.
- `ranking`: "Top N" countdown #N→#1, rank-badge cards, suspense beat
  before #1.
- New modes registered as `video_visual_mode = "quiz" | "ranking"`; script
  generation hooks in llm.py with strict-JSON parsing + retry; tests with
  mocked LLM.

### WP5. Fake-chat story format (Opus, wave 2)
- `chat`: messenger-style story (two-person dialogue: hook → escalation →
  twist). Bubbles pop in synced to per-message TTS (durations measured per
  message, concatenated); two voices when available, one-voice fallback.
- Phone-frame canvas in studio.py (`compose_chat`), typing indicator,
  optional pop SFX per bubble (WP3).

**Implemented (single-voice).** Shipped: `llm.generate_chat_story` (strict
JSON: `{title, persons[2], messages[{from,text}]}`, retry, `None` on failure);
`hyperframes.chat_narration` reads the whole exchange as one continuous
`"Name: text"` track (the proportional WP2 timing model is single-track);
`_chat_scene_list` → one weighted scene per message (invisible `⁣<side>⁣`
marker so the composer recovers the bubble side); `studio.compose_chat` renders
ONE persistent phone frame (header avatar+name, accumulating bubble column,
per-incoming typing indicator, deterministic auto-scroll) with per-message GSAP
pop-in tweens; `render_chat_video` + task.py dispatch with the same non-fatal
fallback as quiz/ranking. Captions suppressed (`caption_ranges = []`).

*Future enhancement — multi-voice:* a true two-voice read (a distinct TTS voice
per speaker, each message synthesized separately and the durations measured and
concatenated) would feel far less robotic than the single-track `"Name: text"`
read. It needs per-segment audio assembly the current timing model does not do
yet, plus per-bubble pop SFX (WP3 `get_sfx_file("pop")`). Deferred.

## Execution

Waves (each WP = one subagent; Opus 4.8 routine, Sonnet 4.6 elementary):
1. **Wave 1 (parallel):** WP1, WP2, WP3 — disjoint file ownership.
2. **Wave 2 (parallel):** WP4, WP5, WP6.
3. **Wave 3:** docs/README/bot-help refresh (Sonnet) + full-suite run + commit.

Ground rules for every WP:
- Tests via `.venv\Scripts\python.exe -m pytest` (uv-managed venv, no pip).
- Tests touching `task.start()` must mock `hyperframes.mode` (live-LLM hang).
- Non-fatal fallback chains stay intact: any failure degrades, never raises.
- Full suite (≥269 tests) green before a WP is considered done.

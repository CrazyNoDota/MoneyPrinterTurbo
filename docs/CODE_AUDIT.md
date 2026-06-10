# Code Audit — Phase 0 (REWORK_PLAN.md)

> Date: 2026-06-09 · Branch: `fix/publish-ready-video-enhancements`
> Goal: verify the existing foundation before building the new `avatar/`, `news/`,
> `bot/` modules. Scoring rubric is REWORK_PLAN.md §7 (0–10). A module that scores
> **< 8** gets a verdict (refactor / rewrite / delete) and must be fixed before new
> phases build on it.

## 0. Test baseline

Run with the uv venv (`./.venv/Scripts/python.exe -m unittest ...`, see the
`dev-environment` memory). `pytest` is **not** installed — plain `unittest`.

| | Before audit | After audit |
|---|---|---|
| Full suite | **hangs** (test_task makes a live LLM call) | **137 passed, 4 skipped, ~23s** |

**Root cause of the hang (fixed):** two `TestStartOrchestration`/`TestTaskService`
tests call `task.start()` but did not neutralize the hyperframes branch. Because the
machine's `config.toml` has hyperframes enabled, `start()` entered the *director*
path → `plan.build_plan()` → `llm._generate_response()` over the network → hung.

Fixes applied in this audit (test-only, no product code touched):
- `test_full_pipeline_blends_generated_clips`: added
  `mock.patch.object(tm.hyperframes, "mode", return_value="footage")` so the test
  exercises the footage+AI-clip blend path it was written for, independent of local
  config.
- `test_task_local_materials`: a legacy no-assertion integration smoke test (real
  TTS/LLM/render). Gated behind `@unittest.skipUnless(os.environ.get("RUN_INTEGRATION"))`
  so the default unit suite is clean; still runnable for manual end-to-end checks.

**Lesson for new phases:** any test that calls `task.start()` MUST mock
`hyperframes.mode` (and `voice.tts`, `material.download_*`, `llm.*`). Local
`config.toml` state must never change a unit test's outcome.

## 1. Module registry

Scores are correctness · tests · fallbacks · style · regressions · docs (rubric §7).
"Light pass" = structure + test-pass reviewed, not every line read.

### Core pipeline

| Module | Score | Verdict | Notes |
|---|---|---|---|
| `services/task.py` | 9 | **keep** | Clean orchestration; resume cache, per-stage `stop_at`, non-fatal fallback at every step. Hyperframes splice (solely/mixed) and caption-range plumbing are correct. |
| `services/voice.py` (2490 ln) | 8 | **keep** | Multi-provider TTS dispatch (`tts()` → edge/azure-v2/siliconflow/gemini/silero/qwen). Clean edge_tts 7.x compat shims. RU=Qwen3, EN=edge already wired → directly supports the RU+EN plan (Phase 3). Large but coherent; light pass. |
| `services/material.py` | 9 | **keep** | `list_local_materials` (own-media, Phase 5 reuse) + stock search/download for pexels/pixabay. New `min_dimension` low-res reject is a clean addition. TLS-verify guard present. |
| `services/video.py` (916 ln) | 8 | **keep** | Compositor + `generate_video`. New `subtitle_ranges` (caption only footage scenes) and adaptive caption shrink/split are sound. This is the file Phase 2 extends (presenter layer). Light pass. |
| `services/llm.py` | 8 | **keep** | Provider-agnostic `_generate_response` w/ backoff/retry; `generate_script`/`generate_terms`. The "LLM by cloud API" hook the plan wants (NewsItem→script, Phase 4) lands here. Light pass. |
| `services/vision.py` | 8 | **keep** | Caption/describe-materials w/ on-disk cache; non-fatal when no API key. Tested. |
| `services/subtitle.py` | 8 | **keep** | SRT parse/create/correct (whisper fallback). Tested. Light pass. |
| `services/qwen_worker.py` | 9 | **keep** | Tiny isolated `.venv-qwen` subprocess; float32+eager correct for Pascal (1070 Ti). Matches memory note exactly. |
| `services/videogen/*` | 8 | **keep (dormant)** | Pluggable submit/poll AI-clip layer, default `null` (off). Budget caps + disk cache. Left as the future AI-broll slot; no rework dependency. |

### Hyperframes package (the motion-graphics engine — read in full)

| Module | Score | Verdict | Notes |
|---|---|---|---|
| `hyperframes/__init__.py` | 9 | **keep** | Mode resolver (footage/hyperframes/mixed); director loop groups contiguous MG scenes, degrades MG→footage per block. Clean. |
| `hyperframes/studio.py` | 9 | **keep** | Deterministic template engine (default). stat/statement archetypes; overlap & black-frames impossible by construction. Hero-number extraction is **regex-only** (no LLM enrichment yet — matches `hyperframes-quality-goal` note). |
| `hyperframes/scenes.py` | 9 | **keep** | SRT-timed scenes w/ short-scene merge + total-duration cover; proportional script fallback. |
| `hyperframes/author.py` | 9 | **keep** | studio-first, freeform-LLM fallback, both run through one `_validate` contract (no-black-frame, one-column, deterministic, staged-assets-only). 1 retry w/ feedback. |
| `hyperframes/plan.py` | 9 | **keep** | LLM director w/ numeric heuristic fallback; JSON parse is defensive. |
| `hyperframes/render.py` | 9 | **keep** | Isolated `npx hyperframes` subprocess; bundled ffmpeg/ffprobe prepended to PATH; pinned CLI version; timeout + graceful "". |
| `hyperframes/assemble.py` | 8 | **keep** | Footage segments (subclip/loop/Ken-Burns) normalized + concat in scene order. Deterministic camera motion. moviepy-heavy; light pass on edge cases. |
| `hyperframes/assets.py` | 8 | **keep** | Background photos staged into project `assets/`; footage resolver (material_ref → stock video → stock photo). **Observation:** footage scenes consume *any* unused user material before stock (prefer-own-media); intended but note for Phase 2. |
| `hyperframes/preview.py` | 8 | **keep** | Optional low-fps proxy → per-scene frame → near-empty detector → contact sheet. Off by default, fully non-fatal. |

### Config

| Module | Score | Verdict | Notes |
|---|---|---|---|
| `config/__init__.py` | 9 | **keep** | New: forces UTF-8 on stdout/stderr (Windows cp1252 crashed on Cyrillic/CJK logs). `import sys` present — verified, no startup bug. |
| `config.example.toml` | 9 | **keep** | New keys documented: `hyperframes_engine`, `_burn_subtitles`, `_preview*`, `bg_count`=2. |

**No module scored < 8.** Nothing requires rewrite or deletion before Phase 1.

## 2. Memory notes vs. reality

| Note | Status |
|---|---|
| `dev-environment` | ✅ accurate (uv venv, no pip/pytest, bundled imageio-ffmpeg w/o ffprobe). |
| `qwen-tts-isolated-env` | ✅ accurate (worker float32+eager, voice-name format, config keys). |
| `hyperframes-motion-graphics` | ✅ accurate; 3 modes + studio all present as described. |
| `hyperframes-quality-goal` | ✅ accurate; studio is default, **still heuristic-only** (LLM enrichment NOT built), font not embedded. These remain Phase 7 levers. |
| `videogen-architecture` | ✅ accurate; still default-off `null`. |
| Minor drift | `hyperframes_bg_count` default is now **2** (was 1). Non-blocking. |

No memory file needs correction; one new fact (the test-hang lesson) is worth saving.

## 3. Open items carried into later phases (not blockers)

1. **Cyrillic font** for motion text — studio uses an Arial-Black/Impact system
   stack; non-Windows headless Chrome will differ. Embed Oswald/Rubik/Bebas
   (Phase 7 / talking-head news archetype).
2. **studio LLM enrichment** — hero-number pick + tighter caption + more archetypes
   (list/quote/comparison) — Phase 7.
3. **Working tree is uncommitted** — the 14 "publish-ready enhancement" files are not
   yet committed. Recommend committing this green state before Phase 1 starts.
4. **No live e2e yet** — mixed/solely hyperframes verified by unit tests + a render
   spike, but never on a real network run (stock keys + LLM + TTS together). Worth a
   one-off manual `RUN_INTEGRATION=1` pass.

## 4. Verdict

The foundation is **solid and well-tested** (137 green). Consistent non-fatal
fallback discipline throughout. The one real defect — a hanging test from a live LLM
call — is fixed. **Cleared to proceed to Phase 1 (`avatar/` Wav2Lip).** No module
needs rewrite/delete; the open items above are enhancements owned by later phases.

# resource/sfx — Sound Effect Assets

This directory contains short, CC0-licensed sound effects synthesised
deterministically via ffmpeg (no network downloads, no third-party samples).

## Files

| File        | Duration | Description                                    |
|-------------|----------|------------------------------------------------|
| whoosh.mp3  | ~0.4 s   | Band-passed noise with volume envelope (scene transition swoosh) |
| pop.mp3     | ~0.15 s  | Short sine burst with fast decay (button / bubble pop) |
| ding.mp3    | ~0.6 s   | 880 Hz + harmonic sine with exponential decay (positive chime) |
| tick.mp3    | ~0.1 s   | Filtered click/noise burst (countdown tick) |
| riser.mp3   | ~1.5 s   | Rising sweep 80 Hz → 1600 Hz (suspense riser before reveal) |

## Regenerating

```bash
python tools/make_sfx.py           # skip files that already exist
python tools/make_sfx.py --force   # overwrite everything
```

## Replacing with custom sounds

Drop in any MP3 file with the **same filename** and the pipeline will use it
instead. The synthesised files are committed as a zero-dependency baseline so
the repo works out of the box without any external downloads.

## License

All files in this directory are synthesised purely from ffmpeg built-in signal
generators (anoisesrc, sine, aevalsrc). They contain no sampled audio and are
in the public domain (CC0).

# AGENTS.md — notes for AI coding agents

Working notes for any agent (or human) continuing this project. Read this file
before touching the code; it records the rules, the architecture and the
mistakes that have already been made and fixed.

## What this project is

A desktop tool (PyQt6 + pyqtgraph) that levels the volume of SoundCraft UI16
console recordings. In a service recording the worship band is mixed close to
full scale while the sermon sits around 10 % of that level. The tool measures
the music, uses it as a reference, and brings the quiet stretches up to a target
level expressed **in dB relative to the music** (`0 dB` = the voice ends up
level with the music). Everything is non-destructive: the input file is only
read, and the result is written to a new WAV.

## Hard constraints — do not break these

1. **Never modify the input file.** It is opened read-only through an `mmap`;
   the export always goes to a different path.
2. **Preserve the audio format on export**: same sample rate, same channel
   count, 16-bit PCM (the console format). Never resample.
3. **Never load a whole file into RAM.** Recordings reach 1.7 GB / ~150 min.
   Everything is processed in blocks (`mmap`, min/max pyramid, 4 M-frame chunks).
4. **The source code, comments, docstrings and user-visible strings are written
   in English.** Spanish (and any other language) lives only in the translation
   files under `leveler/i18n/`.
5. **Every user-visible string must go through `tr()`** (or
   `QCoreApplication.translate("context", ...)` outside a `QObject`). Never
   hardcode a string, and never write a translated literal in the source.
6. **Do not add third-party dependencies** (only `PyQt6`, `pyqtgraph`, `numpy`;
   `ffmpeg` is an optional external binary). Ask before adding anything.
7. **Keep `0 dB` meaning "voice level with the music".** Any change to the
   reference measurement must preserve that property and be verified with a real
   file (see "Verify a change").

## Layout

```
leveler/
├── __main__.py        # entry point: from .gui import main
├── gui.py             # Qt interface, worker thread, playback, drawing
├── analysis.py        # levels, statistics, gain curve, segments, limiter, render
├── audiofile.py       # block-wise WAV I/O (AudioFile, WavWriter, parse_riff)
├── translations.py    # language discovery/loading (.ts/.qm) — the i18n entry point
└── i18n/
    ├── leveler_es.ts   # translation source, edited with Qt Linguist
    └── leveler_es.qm   # compiled, loaded at runtime
```

### `audiofile.py`

- `parse_riff()` reads the RIFF headers by hand (no `wave` module for reading)
  to survive files larger than 4 GB and odd chunks.
- `AudioFile` exposes `read(start, count)` / `read_mono(start, count)` returning
  float32 inside `[-1, 1]`, plus `frames`, `duration`, `channels`,
  `sample_rate`, `bits`, `is_float`.
- Native support: PCM 16/32-bit and float32. Anything else is converted once to
  a temporary float32 WAV with `ffmpeg` (raises `AudioError` if it is missing).
- `WavWriter` writes 16/24-bit PCM incrementally; `Cancelled` signals a user
  abort and is caught by the worker.

### `analysis.py`

- `Settings` — every analysis/correction parameter (see the table in the README).
- `WaveLevels` — min/max pyramid with buckets 2048 / 16384 / 131072 frames;
  `envelope(audio, channel, t0, t1, n_px)` reduces it to pixel columns for the
  current view. This is what makes the zoom smooth.
- `compute_frames()` — RMS and peak per 0.4 s frame (0.2 s hop) over the mono mix.
- `compute_stats()` — noise floor (5th percentile), reference `ref_db` (90th
  percentile of the 12 s moving average, over frames with content) and
  `target_db = ref_db + settings.target_db`.
- `compute_gain()` — gain curve in dB, computed from the **region level** and
  faded out below `gate_db`, then smoothed asymmetrically (`_asym_follow`,
  attack 1.0 s / release 0.4 s). Per-segment overrides: `Segment.enabled` and
  `Segment.trim_db`.
- `detect_segments()` — groups frames whose gain > 0.5 dB, merges pauses up to
  3 s and drops stretches shorter than 1 s.
- `limiter_reduction_db()` / `apply_gain_and_limit()` — block-based peak
  limiter (64-sample blocks, 12 ms hold, 25 ms smoothing) with a final
  `np.clip` at the ceiling as a safety net.
- `export_corrected()` — writes the whole file in 4 M-frame chunks with 0.1 s of
  padding between them, to a `<output>.parcial` file that is renamed with
  `os.replace()` only when finished. On any exception the partial file is removed.

### `gui.py`

- `Worker(QThread)` runs a function with `(progress, cancel)` callbacks and
  emits `failed(str)`; the interface stays responsive and never blocks.
- `SeekViewBox(pg.ViewBox)` emits `seeked(float)` on a single left click.
- `TimeAxis(pg.AxisItem)` formats the timeline as `m:ss`, `h:mm:ss` or `m:ss.s`.
- `MainWindow` builds the toolbar, the two channel plots plus the gain strip, the
  side panel, the language menu and the status bar; `_redraw()` repaints only the
  visible range (debounced with a 30 ms one-shot timer).
- All texts live in `_retranslate()`, which is called once at the end of
  `__init__` and again whenever the language changes. **When you add a widget,
  set its text in `_retranslate()`, not at creation time** (use empty strings
  when creating the widget).

## Running

```bash
# from the folder that contains the package
python3 -m leveler
python3 -m leveler "/path/to/service.wav"     # opens the file straight away
```

On this machine the display is `:0.0`, the interpreter is `python3` (there is no
`python`), and the Qt tools live in `/usr/bin` (`pylupdate6`, `lrelease`,
`linguist`).

## Verify a change (there is no test suite)

1. Byte-compile everything:

   ```bash
   python3 -m py_compile leveler/*.py
   ```

2. Headless functional check — builds a synthetic recording (10 s loud music +
   10 s quiet voice), opens it in the real window and exercises the whole flow:

   ```python
   import os, sys, time, wave
   os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
   sys.path.insert(0, "/path/to/repo")          # folder containing "leveler"
   import numpy as np
   from PyQt6 import QtWidgets
   from leveler.gui import MainWindow

   SR, path = 48000, "/tmp/test_niv.wav"
   t = np.arange(20 * SR) / SR
   sig = np.concatenate([(0.7 * np.sin(2 * np.pi * 220 * t))[: 10 * SR],
                         (0.07 * np.sin(2 * np.pi * 180 * t))[10 * SR:]]).astype(np.float32)
   w = wave.open(path, "wb"); w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
   w.writeframes(np.rint(np.stack([sig, sig], 1) * 32767).astype("<i2").tobytes()); w.close()

   app = QtWidgets.QApplication([])
   win = MainWindow()
   win._open_path(path)

   def wait():                      # the analysis runs in a QThread
       while win.worker is not None:
           app.processEvents(); time.sleep(0.02)
       app.processEvents()

   wait()
   print(win.lbl_ref.text(), win.lbl_target.text(), win.lbl_nseg.text())
   win._seek_to(12.0)                        # click-to-play
   win.prepare_render(); wait()              # A/B preview
   print(win._current_source())              # must be the preview WAV after selecting "Corrected"
   win.close()
   ```

   Expected on the synthetic file: reference ≈ −6 dBFS, one detected segment,
   ~20 dB of gain, and `target_db` equal to `ref_db` when the offset is `0.0 dB`.

3. Real run (the interface must open and stay alive):

   ```bash
   DISPLAY=:0.0 timeout 8 python3 -m leveler
   ```

4. If you touched the analysis, confirm that the original file is untouched
   (`sha256sum` before/after) and that the export keeps 48000 Hz, 2 channels,
   16 bits and the exact frame count.

## Changing or adding texts (i18n)

```bash
pylupdate6 leveler/*.py -ts leveler/i18n/leveler_es.ts   # merge new texts
linguist leveler/i18n/leveler_es.ts                        # review/translate
lrelease leveler/i18n/leveler_es.ts                        # compile to .qm
```

- Contexts are the class name for `self.tr()` (`MainWindow`) and the string
  passed to `QCoreApplication.translate()` (`audiofile`, `analysis`).
- `pylupdate6` **does not recognize wrapper functions** (a local `_tr()` helper
  is silently skipped). Always call `tr()` / `translate()` directly.
- `pylupdate6` merges into an existing `.ts` and keeps the translations of
  unchanged messages; a modified source string is marked unfinished again.
- The language menu shows every language whose `.qm` exists
  (`translations.available()`), and the app starts in the system language when a
  translation is available.

## Solved pitfalls — do not reintroduce them

1. **Clicking the waveform did nothing.** `pyqtgraph.MouseClickEvent.double` is a
   *method*; `not ev.double` is always false. Use `ev.double()`.
2. **Playback always restarted from 0:00.** Qt ignores `setPosition()` while the
   media is loading. The jump is postponed to `mediaStatusChanged` with
   `LoadedMedia` / `BufferedMedia` (`_load_source` + `_on_media_status`).
3. **`0 dB` used to raise the music by ~9 dB.** The reference must be measured
   with the same 12 s region level that the correction uses
   (`compute_stats`), otherwise the parity claim is wrong.
4. **Panning seemed broken.** The view needs scrolling headroom; the plots set
   `setLimits(xMin=0.0, xMax=duration)`. Do not remove it.
5. **Dragging must not seek.** `SeekViewBox` only emits on a single click, and a
   real drag never reaches `mouseClickEvent` (pyqtgraph eats early moves).

## Known gaps / where to continue

- **Unused API**: `AudioFile.info_text()`, `analysis.render_slice()`,
  `Segment.trim_db` and `Settings.ref_manual_db` exist but have no caller and no
  interface control. Either wire them up (a manual reference level and a
  per-segment trim would be the obvious features) or delete them.
- **Export is hardcoded to 16 bits** in `gui.export_file()` even though
  `export_corrected()` supports 24. A quality selector would be easy to add.
- **No tests, no packaging, no CI**: no `pyproject.toml`, no PyInstaller/AppImage
  recipe, no GitHub Actions. Dependencies are only listed in the README and
  `pip install PyQt6 pyqtgraph numpy`.
- **Only Spanish is translated.** Adding a language is three commands (see above)
  plus the entry in `translations.NAMES` for its display name.
- **Limiter** is a block-based peak limiter with a final clip; it is not a true
  look-ahead limiter and does not oversample. If you improve it, keep the
  block-wise processing so memory stays bounded.
- **`_redraw()` recomputes the envelopes on every view change.** A small cache
  keyed by `(channel, x0, x1, n_px)` would help on very long files.
- **README screenshot is missing** and there is **no LICENSE file** yet (the
  author has to choose one).

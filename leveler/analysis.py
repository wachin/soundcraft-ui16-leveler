"""Level analysis, suggested correction, limiter and render/export."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from PyQt6.QtCore import QCoreApplication

from .audiofile import AudioFile, AudioError, Cancelled, WavWriter

EPS = 1e-9


# --------------------------------------------------------------------------- settings
@dataclass
class Settings:
    """Analysis and correction parameters (editable from the interface)."""

    frame_s: float = 0.4              # analysis window
    slow_s: float = 12.0              # window for the "region level"
    ref_percentile: float = 90.0      # percentile used as reference (music)
    ref_manual_db: float | None = None
    target_db: float = 0.0            # target (voice) level in dB relative to the music
    global_boost_db: float = 0.0      # constant boost applied to the whole file
    max_gain_db: float = 24.0
    allow_attenuate: bool = False
    max_atten_db: float = 6.0
    content_floor_db: float = 30.0    # do not amplify what is X dB below the reference
    rise_s: float = 1.0
    fall_s: float = 0.4
    limiter: bool = True
    ceiling_db: float = -1.0
    merge_gap_s: float = 3.0
    min_segment_s: float = 1.0

    @property
    def hop_s(self) -> float:
        return self.frame_s / 2.0

    @property
    def ceiling(self) -> float:
        return float(10.0 ** (self.ceiling_db / 20.0))


@dataclass
class Segment:
    """Detected stretch of audio with too little volume."""

    i0: int
    i1: int
    start_s: float
    end_s: float
    level_db: float
    gain_db: float
    trim_db: float = 0.0
    enabled: bool = True

    @property
    def duration(self) -> float:
        return max(0.0, self.end_s - self.start_s)


@dataclass
class AnalysisResult:
    """Result of the analysis of a file."""

    path: str
    sample_rate: int
    duration: float
    frame_s: float
    hop_s: float
    times: np.ndarray
    level_db: np.ndarray
    peak_db: np.ndarray
    active: np.ndarray
    floor_db: float = -90.0
    gate_db: float = -80.0
    ref_db: float = -20.0
    target_db: float = -26.0
    segments: list[Segment] = field(default_factory=list)


# --------------------------------------------------------------------------- levels (min/max pyramid)
class WaveLevels:
    """Min/max pyramid per channel so the waveform can be drawn at any zoom."""

    BUCKETS = (2048, 16384, 131072)

    def __init__(self, audio: AudioFile, progress=None, cancel=None):
        self.buckets = list(self.BUCKETS)
        self.data: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._build(audio, progress, cancel)

    def _build(self, audio: AudioFile, progress=None, cancel=None):
        n = audio.frames
        ch = audio.channels
        finest = min(self.buckets)
        nb = max(1, (n + finest - 1) // finest)
        mins = np.zeros((nb, ch), np.float32)
        maxs = np.zeros((nb, ch), np.float32)
        block = 1 << 22                      # multiple of every bucket
        for start in range(0, n, block):
            if cancel is not None and cancel():
                raise Cancelled()
            end = min(n, start + block)
            data = audio.read(start, end - start)
            m = data.shape[0]
            full = m // finest
            if full:
                d = data[: full * finest].reshape(full, finest, ch)
                b0 = start // finest
                mins[b0:b0 + full] = d.min(axis=1)
                maxs[b0:b0 + full] = d.max(axis=1)
            rest = m - full * finest
            if rest:
                b0 = start // finest + full
                if b0 < nb:
                    d = data[full * finest:]
                    mins[b0] = d.min(axis=0)
                    maxs[b0] = d.max(axis=0)
            if progress is not None:
                progress(min(1.0, end / max(1, n)))
        self.data[finest] = (mins, maxs)
        cur_b, cur = finest, (mins, maxs)
        for b in sorted(self.buckets):
            if b == finest:
                continue
            r = b // cur_b
            self.data[b] = (_reduce_min(cur[0], r), _reduce_max(cur[1], r))
            cur_b, cur = b, self.data[b]

    # ------------------------------------------------------------------ queries
    def envelope(self, audio: AudioFile, channel: int, t0: float, t1: float,
                 n_px: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (times, minima, maxima) for the given range and pixel width."""
        sr = audio.sample_rate
        i0 = max(0, int(t0 * sr))
        i1 = min(audio.frames, int(math.ceil(t1 * sr)))
        n_px = max(1, int(n_px))
        if i1 - i0 < 2:
            z = np.zeros(0, np.float32)
            return z, z, z
        spp = (i1 - i0) / n_px
        bucket = None
        for b in sorted(self.buckets):
            if b <= spp:
                bucket = b
        if bucket is None:
            return self._raw(audio, channel, i0, i1, n_px)
        mn, mx = self.data[bucket]
        b0 = i0 // bucket
        b1 = min(mn.shape[0], int(math.ceil(i1 / bucket)))
        if b1 - b0 < 2:
            return self._raw(audio, channel, i0, i1, n_px)
        return _pixels(mn[b0:b1, channel], mx[b0:b1, channel],
                       n_px, b0 * bucket, bucket, sr)

    def _raw(self, audio: AudioFile, channel: int, i0: int, i1: int,
             n_px: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        data = audio.read(i0, i1 - i0)
        if data.shape[0] == 0:
            z = np.zeros(0, np.float32)
            return z, z, z
        ch = min(channel, data.shape[1] - 1)
        x = np.ascontiguousarray(data[:, ch])
        return _pixels(x, x, n_px, i0, 1, audio.sample_rate)


def _reduce_min(a: np.ndarray, r: int) -> np.ndarray:
    n = a.shape[0]
    pad = (-n) % r
    if pad:
        a = np.concatenate([a, np.full((pad,) + a.shape[1:], np.inf, a.dtype)])
    return a.reshape(-1, r, a.shape[1]).min(axis=1)


def _reduce_max(a: np.ndarray, r: int) -> np.ndarray:
    n = a.shape[0]
    pad = (-n) % r
    if pad:
        a = np.concatenate([a, np.full((pad,) + a.shape[1:], -np.inf, a.dtype)])
    return a.reshape(-1, r, a.shape[1]).max(axis=1)


def _pixel_starts(n: int, n_px: int) -> np.ndarray:
    edges = np.floor(np.linspace(0, n, n_px + 1)).astype(np.int64)
    starts = np.maximum(edges[:-1], np.arange(n_px))
    starts = np.minimum(starts, n - 1)
    return np.maximum.accumulate(starts)


def _pixels(mn: np.ndarray, mx: np.ndarray, n_px: int, first: int, step: int,
            sr: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = mn.shape[0]
    if n <= n_px:
        times = (first + np.arange(n) * step + step / 2.0) / sr
        return times, np.asarray(mn, np.float32), np.asarray(mx, np.float32)
    idx = _pixel_starts(n, n_px)
    mins = np.minimum.reduceat(mn, idx)
    maxs = np.maximum.reduceat(mx, idx)
    times = (first + idx * step + step / 2.0) / sr
    return times, mins.astype(np.float32), maxs.astype(np.float32)


# --------------------------------------------------------------------------- analysis
def compute_frames(audio: AudioFile, frame_s: float, progress=None,
                   cancel=None) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Level (RMS) and peak per frame. Returns (times, level_db, peak_db, frame_s, hop_s)."""
    sr = audio.sample_rate
    n = audio.frames
    fl = max(1, int(round(frame_s * sr)))
    hop = max(1, fl // 2)
    if n < fl:
        fl = max(1, n)
        hop = fl
    n_frames = 1 if n <= fl else 1 + (n - fl) // hop
    sumsq = np.zeros(n_frames, np.float64)
    peaks = np.zeros(n_frames, np.float32)
    frames_por_bloque = max(1, int(2e6 // hop))
    for f0 in range(0, n_frames, frames_por_bloque):
        if cancel is not None and cancel():
            raise Cancelled()
        f1 = min(n_frames, f0 + frames_por_bloque)
        s = f0 * hop
        e = min(n, (f1 - 1) * hop + fl)
        mono = audio.read_mono(s, e - s)
        view = sliding_window_view(mono, fl)[::hop][: f1 - f0]
        sumsq[f0:f1] = (view * view).sum(axis=1, dtype=np.float64)
        peaks[f0:f1] = np.abs(view).max(axis=1)
        if progress is not None:
            progress(min(1.0, e / max(1, n)))
    rms = np.sqrt(sumsq / float(fl))
    level_db = 20.0 * np.log10(np.maximum(rms, 1e-7))
    peak_db = 20.0 * np.log10(np.maximum(peaks, 1e-7))
    times = (np.arange(n_frames) * hop + fl / 2.0) / sr
    return times, level_db, peak_db, fl / sr, hop / sr


def compute_stats(res: AnalysisResult, settings: Settings):
    """Computes the noise floor, the reference (music) and the target (voice) level."""
    level = res.level_db
    if level.size == 0:
        res.active = np.zeros(0, bool)
        res.floor_db, res.gate_db, res.ref_db, res.target_db = -90.0, -80.0, -20.0, -26.0
        return res
    res.floor_db = float(np.percentile(level, 5.0))
    # The reference (music) is measured with the same "region level" used by the
    # correction: that way "0 dB" means the voice ends up level with the music
    # without touching the music, and negative values leave it quieter.
    w = max(1, int(round(settings.slow_s / max(res.hop_s, 1e-6))))
    region = _moving_average(level, w)
    above = region > res.floor_db + 6.0          # stretches with some content
    src = region[above] if above.any() else region
    if settings.ref_manual_db is not None:
        res.ref_db = float(settings.ref_manual_db)
    else:
        res.ref_db = float(np.percentile(src, settings.ref_percentile))
    # Below this level there is no real content: silence or background noise
    res.gate_db = res.ref_db - settings.content_floor_db
    res.active = level > res.gate_db
    # The target is an offset in dB relative to the music: 0 dB leaves the voice
    # at the same level as the music, negative values leave it quieter and
    # positive values raise it above the music.
    res.target_db = res.ref_db + float(settings.target_db)
    return res


def compute_gain(res: AnalysisResult, settings: Settings,
                 segments: list[Segment] | None = None) -> np.ndarray:
    """Gain curve (dB) per frame, based on the level of each REGION.

    The gain is computed over a long-term smoothed level (``slow_s``) instead of
    the instantaneous dynamics: that corrects the whole sermon stretch without
    following it word by word (it avoids the pumping effect) and it does not
    amplify silences or background noise (``content_floor_db``).
    """
    n = res.level_db.size
    lo = -settings.max_atten_db if settings.allow_attenuate else 0.0
    if n == 0:
        return np.zeros(0, np.float32)
    hop = max(res.hop_s, 1e-6)
    w = max(1, int(round(settings.slow_s / hop)))

    region = _moving_average(res.level_db, w)   # region level (slow)
    rpeak = _window_max(res.level_db, w)        # loudest part of the region

    desired = res.target_db - region
    np.clip(desired, lo, settings.max_gain_db, out=desired)

    # Gain is only applied where the region holds real signal (not silence/noise)
    soft = np.clip((rpeak - res.gate_db) / 6.0, 0.0, 1.0)
    desired = desired * soft

    if segments:
        for seg in segments:
            i0, i1 = segment_range(res, seg)
            if i1 <= i0:
                continue
            if not seg.enabled:
                desired[i0:i1] = 0.0
            elif seg.trim_db:
                desired[i0:i1] = desired[i0:i1] + seg.trim_db

    g = _asym_follow(desired, res.hop_s, settings.rise_s, settings.fall_s)
    g = _moving_average(g, 0.5 / hop)
    g = g + settings.global_boost_db
    np.clip(g, lo, settings.max_gain_db, out=g)
    return g.astype(np.float32)


def segment_range(res: AnalysisResult, seg: Segment | None = None,
                  start_s: float | None = None, end_s: float | None = None):
    """Frame indices of the analysis covered by a segment (time -> index)."""
    n = res.level_db.size
    if seg is not None:
        start_s, end_s = seg.start_s, seg.end_s
    i0 = int(round((start_s - res.frame_s / 2.0) / res.hop_s)) if res.hop_s else 0
    i1 = int(round((end_s - res.frame_s / 2.0) / res.hop_s)) if res.hop_s else n
    i0 = max(0, min(n, i0))
    i1 = max(i0, min(n, i1 + 1))
    return i0, i1


def detect_segments(res: AnalysisResult, gain_db: np.ndarray,
                    settings: Settings) -> list[Segment]:
    """Finds the stretches where the volume is missing and merges short pauses."""
    if gain_db.size == 0:
        return []
    mask = (gain_db > 0.5) & (res.level_db > res.floor_db + 1.0)
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    splits = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate([[idx[0]], idx[splits + 1]]).tolist()
    ends = np.concatenate([idx[splits], [idx[-1]]]).tolist()
    bounds = list(zip(starts, ends))

    hop = res.hop_s
    gap = settings.merge_gap_s / max(hop, 1e-9)
    merged: list[list[int]] = []
    for s, e in bounds:
        if merged and s - merged[-1][1] <= gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])

    out: list[Segment] = []
    for s, e in merged:
        start_s = s * hop
        end_s = min(res.duration, e * hop + res.frame_s)
        if end_s - start_s < settings.min_segment_s:
            continue
        seg_level = float(np.mean(res.level_db[s:e + 1]))
        seg_gain = float(np.median(gain_db[s:e + 1]))
        out.append(Segment(i0=s, i1=e + 1, start_s=start_s, end_s=end_s,
                           level_db=seg_level, gain_db=seg_gain))
    return out


def analyze(audio: AudioFile, settings: Settings, progress=None,
            cancel=None) -> tuple[AnalysisResult, np.ndarray]:
    """Analyzes a whole file: levels, reference, gain curve and segments."""
    times, level, peak, frame_s, hop_s = compute_frames(
        audio, settings.frame_s, progress, cancel)
    res = AnalysisResult(
        path=audio.path, sample_rate=audio.sample_rate, duration=audio.duration,
        frame_s=frame_s, hop_s=hop_s, times=times, level_db=level, peak_db=peak,
        active=np.zeros(level.shape, bool),
    )
    compute_stats(res, settings)
    gain = compute_gain(res, settings)
    res.segments = detect_segments(res, gain, settings)
    return res, gain


def predicted_peak_db(res: AnalysisResult, gain_db: np.ndarray,
                      settings: Settings) -> float:
    """Estimated peak (dBFS) after applying gain and limiter."""
    if gain_db.size == 0:
        return -120.0
    peak = float(np.max(res.peak_db + gain_db))
    if settings.limiter:
        peak = min(peak, settings.ceiling_db)
    return peak


# --------------------------------------------------------------------------- numeric helpers
def _moving_average(a: np.ndarray, frames: float) -> np.ndarray:
    w = int(round(frames))
    if w <= 1 or a.size == 0:
        return np.array(a, dtype=np.float64)
    w = min(w, a.size)
    k = np.ones(w, np.float64) / w
    left = w // 2
    right = w - 1 - left
    ext = np.pad(a.astype(np.float64), (left, right), mode="edge")
    return np.convolve(ext, k, mode="valid")[: a.size]


def _asym_follow(g: np.ndarray, hop_s: float, rise_s: float,
                 fall_s: float) -> np.ndarray:
    """Asymmetric smoothing: rises slowly, falls fast (avoids pumping and spikes)."""
    n = g.size
    out = np.empty(n, np.float64)
    if n == 0:
        return out
    a_up = math.exp(-hop_s / max(rise_s, 1e-3))
    a_dn = math.exp(-hop_s / max(fall_s, 1e-3))
    cur = float(g[0])
    for i in range(n):
        want = float(g[i])
        a = a_up if want > cur else a_dn
        cur = want + (cur - want) * a
        out[i] = cur
    return out


def _window_max(a: np.ndarray, w: int) -> np.ndarray:
    if w <= 1 or a.size == 0:
        return a
    w = min(w, a.size)
    ext = np.concatenate([a, np.full(w - 1, a[-1])])
    return sliding_window_view(ext, w).max(axis=1)


def gain_for_range(times: np.ndarray, gain_db: np.ndarray, start: int, n: int,
                   sr: int) -> np.ndarray:
    """Gain (dB) sample by sample for the requested range."""
    t = np.arange(start, start + n, dtype=np.float64) / sr
    return np.interp(t, times, gain_db).astype(np.float32)


# --------------------------------------------------------------------------- limiter and render
def limiter_reduction_db(x: np.ndarray, sr: int, ceiling: float, block: int = 64,
                         hold_ms: float = 12.0, smooth_ms: float = 25.0) -> np.ndarray:
    """Reduction (dB, >= 0) per sample so the ceiling is never exceeded."""
    n = x.shape[0]
    if n == 0:
        return np.zeros(0, np.float32)
    nb = -(-n // block)
    peak = np.abs(x).max(axis=1)
    if nb * block > n:
        peak = np.concatenate([peak, np.full(nb * block - n, peak[-1], np.float32)])
    bp = peak.reshape(nb, block).max(axis=1)
    need = 20.0 * np.log10(np.maximum(bp, 1e-9) / max(ceiling, 1e-9))
    np.clip(need, 0.0, None, out=need)
    if not need.any():
        return np.zeros(n, np.float32)
    w_hold = max(1, int(round(hold_ms / 1000.0 * sr / block)))
    w_smooth = max(1, int(round(smooth_ms / 1000.0 * sr / block)))
    env = _window_max(need, w_hold)
    env = _moving_average(env, w_smooth)
    env = np.maximum(env, need)
    centers = np.arange(nb) * block + block / 2.0
    return np.interp(np.arange(n), centers, env).astype(np.float32)


def apply_gain_and_limit(x: np.ndarray, gain_db: np.ndarray | None, sr: int,
                         settings: Settings) -> np.ndarray:
    """Applies the gain curve and the limiter to a float32 block."""
    if gain_db is not None and gain_db.size == x.shape[0]:
        factor = np.power(10.0, gain_db / 20.0).astype(np.float32)
        x = x * factor[:, None]
    if settings.limiter:
        red = limiter_reduction_db(x, sr, settings.ceiling)
        if red.size:
            x = x * np.power(10.0, -red / 20.0).astype(np.float32)[:, None]
        np.clip(x, -settings.ceiling, settings.ceiling, out=x)
    else:
        np.clip(x, -1.0, 1.0, out=x)
    return x


def _process_block(audio: AudioFile, res: AnalysisResult | None, gain_db: np.ndarray | None,
                   settings: Settings, start: int, n: int, pad: int) -> np.ndarray:
    sr = audio.sample_rate
    s = max(0, start - pad)
    e = min(audio.frames, start + n + pad)
    x = audio.read(s, e - s)
    if gain_db is not None and res is not None:
        g = gain_for_range(res.times, gain_db, s, e - s, sr)
        x = apply_gain_and_limit(x, g, sr, settings)
    elif settings.limiter:
        x = apply_gain_and_limit(x, None, sr, settings)
    return x[start - s: start - s + n]


def export_corrected(audio: AudioFile, res: AnalysisResult | None,
                     gain_db: np.ndarray | None, settings: Settings, out_path: str,
                     bits: int = 16, progress=None, cancel=None) -> str:
    """Writes the whole corrected file (WAV 16/24 bits)."""
    sr = audio.sample_rate
    total = audio.frames
    chunk = 1 << 22
    pad = int(0.1 * sr)
    tmp = out_path + ".parcial"
    w = WavWriter(tmp, sr, audio.channels, bits)
    try:
        for start in range(0, total, chunk):
            if cancel is not None and cancel():
                raise Cancelled()
            n = min(chunk, total - start)
            y = _process_block(audio, res, gain_db, settings, start, n, pad)
            w.write(y)
            if progress is not None:
                progress(min(1.0, (start + n) / total))
        w.close()
        os.replace(tmp, out_path)
    except BaseException:
        w.abort()
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    return out_path


def render_slice(audio: AudioFile, res: AnalysisResult | None, gain_db: np.ndarray | None,
                 settings: Settings | None, t0: float, dur: float, out_path: str,
                 bits: int = 16, progress=None, cancel=None) -> float:
    """Renders a fragment to a WAV (for trial listening). Returns its duration."""
    sr = audio.sample_rate
    start = max(0, int(t0 * sr))
    end = min(audio.frames, start + int(dur * sr))
    if end - start < 2:
        raise AudioError(QCoreApplication.translate(
            "analysis", "The selected fragment is too short."))
    pad = int(0.1 * sr)
    chunk = 1 << 21
    w = WavWriter(out_path, sr, audio.channels, bits)
    try:
        for cs in range(start, end, chunk):
            if cancel is not None and cancel():
                raise Cancelled()
            ce = min(end, cs + chunk)
            if gain_db is not None and res is not None and settings is not None:
                y = _process_block(audio, res, gain_db, settings, cs, ce - cs, pad)
            else:
                y = _process_block(audio, None, None, Settings(limiter=False), cs, ce - cs, pad)
            w.write(y)
            if progress is not None:
                progress((ce - start) / max(1, end - start))
        w.close()
    except BaseException:
        w.abort()
        raise
    return (end - start) / sr

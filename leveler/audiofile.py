"""Block-wise WAV reading and writing.

Recordings from the SoundCraft UI16 console are 16-bit PCM WAV files and can be
several GB long, so they are never loaded entirely into memory: they are mapped
with mmap and read in blocks on demand.

Native support: 16/32-bit PCM and 32-bit float.
Any other format (24-bit, RF64, compressed, etc.) is converted to a temporary
copy with ffmpeg.
"""

from __future__ import annotations

import os
import shutil
import struct
import subprocess
import tempfile
import wave

import numpy as np
from PyQt6.QtCore import QCoreApplication


class Cancelled(Exception):
    """The user cancelled the operation."""


class AudioError(Exception):
    """Failure while opening or processing the audio."""


def parse_riff(path: str) -> dict | None:
    """Reads the WAV (RIFF) headers and returns the format and data offset."""
    try:
        size = os.path.getsize(path)
    except OSError:
        return None
    with open(path, "rb") as f:
        head = f.read(12)
        if len(head) < 12 or head[0:4] != b"RIFF":
            return None
        if head[8:12] != b"WAVE":
            return None
        pos = 12
        info: dict = {"data_offset": None, "data_size": None, "fmt": None}
        while pos + 8 <= size:
            f.seek(pos)
            h = f.read(8)
            if len(h) < 8:
                break
            cid = h[0:4]
            csize = struct.unpack("<I", h[4:8])[0]
            body = pos + 8
            if cid == b"fmt " and csize >= 16:
                f.seek(body)
                d = f.read(min(csize, 40))
                fmt, ch, sr, _br, _ba, bits = struct.unpack("<HHIIHH", d[:16])
                if fmt == 0xFFFE and len(d) >= 26:
                    fmt = struct.unpack("<H", d[24:26])[0]
                info.update(fmt=fmt, channels=ch, sample_rate=sr, bits=bits)
            elif cid == b"data":
                info["data_offset"] = body
                info["data_size"] = min(csize, size - body)
            pos = body + csize + (csize & 1)
            if info["data_offset"] is not None and info["fmt"] is not None:
                break
    if info["data_offset"] is None or info["fmt"] is None:
        return None
    return info


class AudioFile:
    """Random and block-wise access to an audio file (float32 within [-1, 1])."""

    def __init__(self, path: str):
        self.path = os.path.abspath(path)
        if not os.path.isfile(self.path):
            raise AudioError(QCoreApplication.translate(
                "audiofile", "The file does not exist: %s") % self.path)
        self._tmpdir: str | None = None
        self._used_path = self.path

        info = parse_riff(self._used_path)
        if not self._is_supported(info):
            self._used_path = self._convert_with_ffmpeg(self.path)
            info = parse_riff(self._used_path)
            if not self._is_supported(info):
                raise AudioError(QCoreApplication.translate(
                    "audiofile", "The format of the file could not be read."))

        fmt = info["fmt"]
        bits = info["bits"]
        self.channels = int(info["channels"])
        self.sample_rate = int(info["sample_rate"])
        self.bits = int(bits)
        self.is_float = fmt == 3
        if self.is_float:
            dtype = np.dtype("<f4")
        elif bits == 16:
            dtype = np.dtype("<i2")
        else:
            dtype = np.dtype("<i4")
        itemsize = dtype.itemsize
        self.frames = int(info["data_size"] // (self.channels * itemsize))
        self.duration = self.frames / self.sample_rate if self.sample_rate else 0.0
        self._scale = 1.0 if self.is_float else float(1 << (bits - 1))
        self._mem = np.memmap(
            self._used_path, dtype=dtype, mode="r",
            offset=int(info["data_offset"]),
            shape=(self.frames, self.channels),
        )

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _is_supported(info: dict | None) -> bool:
        if not info or info.get("fmt") not in (1, 3):
            return False
        if info.get("fmt") == 1:
            return info.get("bits") in (16, 32)
        return info.get("bits") == 32

    def _convert_with_ffmpeg(self, path: str) -> str:
        if shutil.which("ffmpeg") is None:
            raise AudioError(QCoreApplication.translate(
                "audiofile",
                "The format is not supported directly and 'ffmpeg' is not installed."))
        self._tmpdir = tempfile.mkdtemp(prefix="leveler_")
        out = os.path.join(self._tmpdir, "converted.wav")
        cmd = [
            "ffmpeg", "-v", "error", "-y", "-i", path,
            "-map", "0:a:0", "-c:a", "pcm_f32le", out,
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0 or not os.path.isfile(out):
            raise AudioError(QCoreApplication.translate(
                "audiofile", "ffmpeg could not convert the file:\n%s") % res.stderr.strip())
        return out

    def close(self):
        self._mem = None
        if self._tmpdir and os.path.isdir(self._tmpdir):
            shutil.rmtree(self._tmpdir, ignore_errors=True)
            self._tmpdir = None

    def info_text(self) -> str:
        kind = (QCoreApplication.translate("audiofile", "float") if self.is_float
                else QCoreApplication.translate("audiofile", "PCM"))
        return QCoreApplication.translate("audiofile", "%s — %d channel(s), %d Hz, %d-bit %s") % (
            os.path.basename(self.path), self.channels, self.sample_rate,
            self.bits, kind)

    # ------------------------------------------------------------------ reading
    def read(self, start: int, count: int) -> np.ndarray:
        """Returns (count, channels) float32 within [-1, 1]."""
        start = max(0, int(start))
        end = min(self.frames, start + max(0, int(count)))
        if end <= start:
            return np.zeros((0, self.channels), np.float32)
        data = self._mem[start:end]
        if data.dtype != np.float32:
            data = data.astype(np.float32)
        else:
            data = np.array(data, dtype=np.float32)
        if not self.is_float:
            data /= self._scale
        return data

    def read_mono(self, start: int, count: int) -> np.ndarray:
        """Returns (count,) float32 with the mono mix."""
        data = self.read(start, count)
        if data.shape[1] == 1:
            return np.ascontiguousarray(data[:, 0])
        return data.mean(axis=1, dtype=np.float32)


class WavWriter:
    """Incremental writing of WAV PCM (16 or 24 bits)."""

    def __init__(self, path: str, sample_rate: int, channels: int, bits: int = 16):
        if bits not in (16, 24):
            raise AudioError(QCoreApplication.translate(
                "audiofile", "Exporting is only supported in 16 or 24 bits."))
        self.path = path
        self.bits = bits
        self._w = wave.open(path, "wb")
        self._w.setnchannels(channels)
        self._w.setsampwidth(bits // 8)
        self._w.setframerate(sample_rate)
        self._closed = False

    def write(self, block: np.ndarray):
        """block: float32 (n, channels) within [-1, 1]."""
        if block.size == 0:
            return
        x = np.clip(block, -1.0, 1.0)
        if self.bits == 16:
            data = np.rint(x * 32767.0).astype("<i2")
        else:
            i = np.rint(x * 8388607.0).astype("<i4")
            data = np.ascontiguousarray(i.view(np.uint8).reshape(-1, 4)[:, :3])
        self._w.writeframes(data.tobytes())

    def close(self):
        if not self._closed:
            self._w.close()
            self._closed = True

    def abort(self):
        self.close()
        try:
            os.remove(self.path)
        except OSError:
            pass


def write_wav(path: str, sample_rate: int, channels: int, bits: int,
              blocks, progress=None, cancel=None):
    """Writes a WAV from an iterable of float32 blocks."""
    w = WavWriter(path, sample_rate, channels, bits)
    try:
        for block in blocks:
            if cancel is not None and cancel():
                raise Cancelled()
            w.write(block)
            if progress is not None:
                progress()
        w.close()
    except BaseException:
        w.abort()
        raise

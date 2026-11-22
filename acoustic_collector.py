"""Acoustic hardware capture and synthetic audio generation sources."""
import logging
import queue
from typing import List, Tuple, Optional
import numpy as np

from signal_generators import acoustic_chunk

log = logging.getLogger("aetherhound")


def list_input_devices() -> List[Tuple[int, str, float]]:
    """Lists available audio input devices (index, name, default_sample_rate)."""
    try:
        import sounddevice as sd
        devices = []
        for i, d in enumerate(sd.query_devices()):
            if d.get("max_input_channels", 0) > 0:
                devices.append((i, str(d["name"]), float(d.get("default_samplerate", 44100.0))))
        return devices
    except Exception as e:
        log.warning("Could not query sound devices: %s", e)
        return []


class MicrophoneSource:
    """Acquires live audio from laptop microphone using a non-blocking queue stream."""
    origin = "microphone"
    realtime = True

    def __init__(self, sample_rate: int = 44100, chunk_seconds: float = 1.0, device: Optional[int] = None):
        import sounddevice as sd
        self.fs = int(sample_rate)
        self.chunk_size = int(self.fs * chunk_seconds)
        self._q = queue.Queue(maxsize=500)
        self._buf = np.empty(0, dtype=np.float64)
        self.device = device

        sd.check_input_settings(device=device, channels=1, samplerate=self.fs)
        self._stream = sd.InputStream(
            samplerate=self.fs,
            channels=1,
            dtype="float32",
            device=device,
            blocksize=2048,
            callback=self._audio_callback
        )
        self._stream.start()

    def _audio_callback(self, indata, frames, time_info, status):
        try:
            self._q.put_nowait(indata[:, 0].astype(np.float64).copy())
        except queue.Full:
            pass

    def read_chunk(self) -> np.ndarray:
        """Reads one exact chunk of audio samples."""
        while len(self._buf) < self.chunk_size:
            chunk = self._q.get(timeout=6.0)
            self._buf = np.concatenate([self._buf, chunk])
        result, self._buf = self._buf[:self.chunk_size], self._buf[self.chunk_size:]
        return result

    def set_scenario(self, name: str):
        pass  # Physical microphone captures real ambient acoustics

    def close(self):
        """Releases audio hardware resources."""
        try:
            self._stream.stop()
            self._stream.close()
        except Exception:
            pass


class SyntheticAudioSource:
    """Generates synthetic test audio signals for testing without hardware dependence."""
    origin = "synthetic_audio"
    realtime = False

    def __init__(self, sample_rate: int = 44100, chunk_seconds: float = 1.0,
                 scenario: str = "normal", seed: Optional[int] = None, tone_hz: float = 19500.0):
        self.fs = int(sample_rate)
        self.chunk_size = int(self.fs * chunk_seconds)
        self.scenario = scenario
        self.tone_hz = tone_hz
        self._rng = np.random.default_rng(seed)
        self._t0 = 0.0

    def set_scenario(self, name: str):
        self.scenario = name

    def read_chunk(self) -> np.ndarray:
        x = acoustic_chunk(self.scenario, self.chunk_size, self.fs, self._rng, self._t0, self.tone_hz)
        self._t0 += self.chunk_size / self.fs
        return x

    def close(self):
        pass

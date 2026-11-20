"""Threaded background monitoring engine orchestrating Acoustic and RF detection pipelines."""
from collections import deque
import logging
import threading
import time
from typing import Dict, Any, Optional
import numpy as np

import config
from acoustic_collector import MicrophoneSource, SyntheticAudioSource
from acoustic_detector import AcousticDetector
from alert_manager import AlertManager
from rf_detector import RFDetector
from rf_simulator import SimulatedRFSource
from rf_source import RtlSdrSource

log = logging.getLogger("aetherhound")


def _blank_channel() -> Dict[str, Any]:
    return {
        "status": "stopped",       # "stopped" | "calibrating" | "monitoring" | "error"
        "progress": (0, 0),
        "error": None,
        "warning": None,
        "origin": None,
        "scenario": None,
        "spectrum": None,          # (freqs, power_db)
        "extra": {},
        "history": [],
        "waterfall": [],
        "chunks": 0,
        "last_update": None,
    }


class MonitoringEngine:
    """Thread-safe monitoring coordinator for live cybersecurity telemetry."""

    def __init__(self, db_manager):
        self.db = db_manager
        self.lock = threading.RLock()
        self._stop_event = threading.Event()
        self._threads = []
        self._sources = {}
        self.running = False
        self.current_session_uid = None
        self.settings = config.Settings()
        self.alert_manager = None
        self.recent_alerts = deque(maxlen=50)
        self.state = {
            "acoustic": _blank_channel(),
            "rf": _blank_channel()
        }
        self._hist = {
            "acoustic": deque(maxlen=300),
            "rf": deque(maxlen=300)
        }
        self._waterfall = deque(maxlen=50)

    def start(self, settings: config.Settings, acoustic: Optional[str] = "microphone",
              rf: Optional[str] = "simulated", acoustic_scenario: str = "normal",
              rf_scenario: str = "normal", audio_device: Optional[int] = None):
        """Starts real-time monitoring threads."""
        if self.running:
            return

        self._stop_event.clear()
        self.settings = settings
        self.alert_manager = AlertManager(
            self.db,
            cooldown_s=settings.alert_cooldown_s,
            beep_on_high=settings.beep_on_high,
            console=settings.console_alerts
        )
        self.alert_manager.callbacks.append(lambda a: self.recent_alerts.appendleft(a))
        self._hist["acoustic"].clear()
        self._hist["rf"].clear()
        self._waterfall.clear()
        self.state = {"acoustic": _blank_channel(), "rf": _blank_channel()}
        self._threads = []

        # Record session in database
        self.current_session_uid = self.db.start_session(
            acoustic_source=acoustic or "None",
            rf_source=rf or "None",
            profile_name=settings.profile_name
        )

        if acoustic:
            self._launch_channel("acoustic", self._make_acoustic_source, self._acoustic_loop,
                                 acoustic, acoustic_scenario, audio_device)
        if rf:
            self._launch_channel("rf", self._make_rf_source, self._rf_loop,
                                 rf, rf_scenario, None)

        self.running = True

    def _launch_channel(self, name: str, factory, loop_fn, kind: str, scenario: str, device: Any):
        ch = self.state[name]
        try:
            source = factory(kind, scenario, device)
        except Exception as e:
            ch["status"] = "error"
            ch["error"] = f"{type(e).__name__}: {e}"
            log.exception("Could not start %s source: %s", name, e)
            return

        self._sources[name] = source
        ch.update(status="calibrating", origin=source.origin, scenario=scenario)
        t = threading.Thread(target=loop_fn, args=(source,), daemon=True, name=f"aetherhound-{name}")
        t.start()
        self._threads.append(t)

    def stop(self):
        """Halts all monitoring threads cleanly."""
        self._stop_event.set()
        for t in self._threads:
            t.join(timeout=6.0)
        for s in self._sources.values():
            try:
                s.close()
            except Exception:
                pass
        self._sources.clear()
        self._threads.clear()

        with self.lock:
            for ch in self.state.values():
                if ch["status"] != "error":
                    ch["status"] = "stopped"

        if self.current_session_uid:
            total_alerts = len(self.recent_alerts)
            self.db.end_session(self.current_session_uid, total_alerts)
            self.current_session_uid = None

        self.running = False

    def recalibrate(self):
        """Triggers baseline recalibration while monitoring is active."""
        with self.lock:
            self._recalibrate_flag = True

    def set_scenario(self, channel: str, name: str):
        """Dynamically switches scenario for simulated or synthetic sources."""
        src = self._sources.get(channel)
        if src is not None and self.state[channel]["scenario"] != name:
            src.set_scenario(name)
            with self.lock:
                self.state[channel]["scenario"] = name
            self.db.log_audit("SCENARIO_SWITCH", "MONITORING",
                              f"Switched {channel} scenario to '{name}'")

    def snapshot(self) -> Dict[str, Any]:
        """Returns atomic state snapshot for UI renderers."""
        with self.lock:
            snap = {k: dict(v) for k, v in self.state.items()}
            snap["acoustic"]["history"] = list(self._hist["acoustic"])
            snap["rf"]["history"] = list(self._hist["rf"])
            snap["rf"]["waterfall"] = list(self._waterfall)
            snap["running"] = self.running
            snap["suppressed"] = self.alert_manager.suppressed_count if self.alert_manager else 0
            snap["recent_alerts"] = list(self.recent_alerts)
        return snap

    # =========================================================================
    # SOURCE FACTORIES
    # =========================================================================

    def _make_acoustic_source(self, kind: str, scenario: str, device: Any):
        s = self.settings
        if kind == "microphone":
            return MicrophoneSource(s.sample_rate, s.chunk_seconds, device)
        return SyntheticAudioSource(s.sample_rate, s.chunk_seconds, scenario)

    def _make_rf_source(self, kind: str, scenario: str, device: Any):
        s = self.settings
        if kind == "rtlsdr":
            return RtlSdrSource(s.rf_center_mhz, s.rf_span_mhz * 1e6, s.rf_bins)
        return SimulatedRFSource(s.rf_center_mhz, s.rf_span_mhz, s.rf_bins, scenario)

    # =========================================================================
    # BACKGROUND LOOPS
    # =========================================================================

    def _acoustic_loop(self, source):
        s, fs = self.settings, self.settings.sample_rate
        det = AcousticDetector(s, source.origin)
        ch = self.state["acoustic"]

        try:
            while not self._stop_event.is_set():
                t_start = time.monotonic()
                chunk = source.read_chunk()
                x = np.asarray(chunk, dtype=np.float64)

                # Silence check
                rms = 20.0 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12)
                if rms < -92.0:
                    with self.lock:
                        ch["warning"] = "Microphone input is silent. Verify hardware settings and permissions."
                        ch["last_update"] = time.time()
                    self._pace(source, t_start, s.chunk_seconds)
                    continue

                res = det.process(x, fs)
                for f in res["findings"]:
                    self.alert_manager.submit(f)

                feats = res["features"]
                with self.lock:
                    ch.update(
                        status=res["status"],
                        progress=det.calibration_progress,
                        warning=None,
                        spectrum=(res["freqs"], res["psd_db"]),
                        chunks=ch["chunks"] + 1,
                        last_update=time.time(),
                        extra={
                            "band": res["band"],
                            "features": feats,
                            "known_tones": list(det.known_tones),
                            "baseline": det.baseline
                        }
                    )
                    self._hist["acoustic"].append({
                        "t": time.time(),
                        "prominence_db": feats["prominence_db"],
                        "peak_db": feats["peak_db"],
                        "band_mean_db": feats["mean_db"],
                        "rms_db": feats["rms_db"],
                        "alert": bool(res["findings"])
                    })

                self._pace(source, t_start, s.chunk_seconds)
        except Exception as e:
            log.exception("Acoustic loop failed: %s", e)
            with self.lock:
                ch["status"] = "error"
                ch["error"] = f"{type(e).__name__}: {e}"

    def _rf_loop(self, source):
        s = self.settings
        det = RFDetector(s, source.origin)
        ch = self.state["rf"]

        try:
            while not self._stop_event.is_set():
                t_start = time.monotonic()
                freqs, power = source.read_sweep()
                res = det.process(freqs, power)

                for f in res["findings"]:
                    self.alert_manager.submit(f)

                with self.lock:
                    ch.update(
                        status=res["status"],
                        progress=det.calibration_progress,
                        warning=None,
                        spectrum=(res["freqs"], res["power"]),
                        chunks=ch["chunks"] + 1,
                        last_update=time.time(),
                        extra={
                            "baseline": res["baseline"],
                            "threshold": res["threshold"],
                            "exceed_mask": res["exceed_mask"]
                        }
                    )
                    self._waterfall.append(np.asarray(power, dtype=np.float32))
                    base = res["baseline"]
                    max_excess = float(np.max(power - base)) if base is not None else 0.0
                    self._hist["rf"].append({
                        "t": time.time(),
                        "max_excess_db": max_excess,
                        "exceed_bins": int(res["exceed_mask"].sum()),
                        "alert": bool(res["findings"])
                    })

                self._pace(source, t_start, s.rf_sweep_interval_s)
        except Exception as e:
            log.exception("RF loop failed: %s", e)
            with self.lock:
                ch["status"] = "error"
                ch["error"] = f"{type(e).__name__}: {e}"

    def _pace(self, source, t_start: float, period: float):
        if getattr(source, "realtime", False):
            return
        elapsed = time.monotonic() - t_start
        wait_s = max(0.0, period - elapsed)
        self._stop_event.wait(wait_s)

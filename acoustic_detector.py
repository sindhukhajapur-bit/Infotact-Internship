"""Acoustic covert channel detector: baseline calibration, ultrasonic tone, OOK, and FSK detection."""
from collections import Counter
from typing import Dict, Any, List
import numpy as np

import features as F
from models import Finding


class AcousticDetector:
    """Acoustic covert channel detection engine.

    Workflow:
      1. Baseline Calibration: Records ambient acoustic noise across calibration chunks
         to learn the normal noise floor and variance in the monitored ultrasonic band.
         Persistent narrow tones (e.g., monitor inverter coil whine) are learned as benign.
      2. Feature Extraction: Computes Welch PSD, running-median local noise floor, and peak prominence.
      3. Anomaly Decision: Checks if peak prominence exceeds threshold and rises above baseline.
      4. Persistence Filter: Requires anomaly to persist across consecutive chunks to suppress clicks.
      5. Modulation Classification: Evaluates time-domain spectrogram for OOK and FSK signatures.
    """

    def __init__(self, settings, origin: str = "microphone"):
        self.s = settings
        self.origin = origin
        self._cal = []
        self.baseline = None
        self.known_tones = []          # Whitelist of [(center_hz, half_width_hz)]
        self._tonal_run = 0
        self._broad_run = 0

    @property
    def calibrated(self) -> bool:
        return self.baseline is not None

    @property
    def calibration_progress(self) -> tuple:
        return min(len(self._cal), self.s.acoustic_calibration_chunks), self.s.acoustic_calibration_chunks

    def band(self, fs: int) -> tuple:
        hi = min(self.s.band_high_hz, fs / 2.0 - 300.0)
        lo = self.s.band_low_hz
        if lo >= hi:
            raise ValueError(f"Monitored band {lo}-{hi} Hz is invalid for sample rate {fs} Hz")
        return lo, hi

    def reset(self):
        self._cal = []
        self.baseline = None
        self.known_tones = []
        self._tonal_run = 0
        self._broad_run = 0

    def process(self, chunk: np.ndarray, fs: int) -> Dict[str, Any]:
        """Processes 1 chunk of audio data."""
        x = F.remove_dc(F.to_mono_float(chunk))
        lo, hi = self.band(fs)
        f, psd_db = F.compute_psd(x, fs)
        feats = F.band_features(f, psd_db, lo, hi, self.known_tones)
        feats["rms_db"] = F.rms_db(x)

        result = {
            "freqs": f,
            "psd_db": psd_db,
            "band": (lo, hi),
            "features": feats,
            "findings": [],
            "status": "calibrating" if not self.calibrated else "monitoring"
        }

        # Step 1: Calibration phase
        if not self.calibrated:
            self._calibrate_step(f, psd_db, lo, hi)
            result["status"] = "calibrating" if not self.calibrated else "monitoring"
            return result

        # Step 2: Monitoring phase
        b = self.baseline
        rise = feats["peak_db"] - b["floor_mean"]
        feats["rise_db"] = rise

        tonal = (feats["prominence_db"] >= self.s.tone_prominence_db and
                 rise >= max(self.s.tone_rise_db, 3.0 * b["floor_std"]))

        broad = (not tonal) and (feats["mean_db"] - b["mean_mean"] >=
                                 max(self.s.broadband_rise_db, 4.0 * b["mean_std"]))

        self._tonal_run = self._tonal_run + 1 if tonal else 0
        self._broad_run = self._broad_run + 1 if broad else 0
        feats["tonal"] = bool(tonal)
        feats["broadband_rise"] = bool(broad)

        if tonal and self._tonal_run >= self.s.acoustic_min_consecutive:
            result["findings"].append(self._classify_tonal(x, fs, lo, hi, feats, rise))
        elif broad and self._broad_run >= self.s.acoustic_min_consecutive:
            rise_bb = feats["mean_db"] - b["mean_mean"]
            result["findings"].append(Finding(
                source="ACOUSTIC",
                origin=self.origin,
                kind="ultrasonic_broadband_rise",
                threat_category="Ultrasonic Environmental Noise",
                severity="Low",
                confidence=40,
                summary=(f"Band-wide energy in {lo/1000:.1f}-{hi/1000:.1f} kHz rose by {rise_bb:.1f} dB "
                         f"without a distinct tone. Consistent with ambient noise or air friction."),
                details={"band_hz": [lo, hi], "rise_db": round(rise_bb, 1)},
                freq=(lo + hi) / 2.0,
                freq_tol=(hi - lo)
            ))

        return result

    def _calibrate_step(self, f: np.ndarray, psd_db: np.ndarray, lo: float, hi: float):
        """Accumulates baseline statistics during calibration."""
        raw = F.band_features(f, psd_db, lo, hi)
        self._cal.append(raw)
        if len(self._cal) < self.s.acoustic_calibration_chunks:
            return

        floors = np.array([c["floor_db"] for c in self._cal])
        means = np.array([c["mean_db"] for c in self._cal])
        self.baseline = {
            "floor_mean": float(floors.mean()),
            "floor_std": float(max(floors.std(), 0.5)),
            "mean_mean": float(means.mean()),
            "mean_std": float(max(means.std(), 0.5)),
        }

        # Identify steady benign tones (present in >= 50% calibration chunks with prominence >= 8 dB)
        counts = Counter(int(round(c["peak_freq_hz"] / 50.0) * 50) for c in self._cal
                         if c["prominence_db"] >= 8.0)
        threshold = max(2, len(self._cal) // 2)
        self.known_tones = [(float(fc), 150.0) for fc, n in counts.items() if n >= threshold]

    def _classify_tonal(self, x: np.ndarray, fs: int, lo: float, hi: float,
                        feats: Dict[str, Any], rise: float) -> Finding:
        """Classifies detected tonal anomaly based on temporal modulation."""
        k = F.keying_features(x, fs, feats["peak_freq_hz"], lo, hi, self.known_tones,
                              self.s.keying_min_range_db)
        freq = feats["peak_freq_hz"]
        prom = feats["prominence_db"]

        details = {
            "freq_hz": round(freq, 1),
            "peak_db": round(feats["peak_db"], 1),
            "prominence_db": round(prom, 1),
            "rise_over_baseline_db": round(rise, 1),
            "band_hz": [lo, hi],
            "spectral_flatness": round(feats["flatness"], 4),
            "centroid_hz": round(feats.get("centroid_hz", freq), 1),
            **{kk: (round(v, 3) if isinstance(v, float) else v) for kk, v in k.items()}
        }

        base_conf = 55 + int(min(25, 1.5 * (prom - self.s.tone_prominence_db)))

        if k["fsk"]:
            kind = "ultrasonic_fsk"
            sev = "High"
            conf = min(96, base_conf + 20)
            threat_cat = "Ultrasonic Acoustic Exfiltration"
            summary = (f"Ultrasonic Frequency-Shift Keying (FSK) detected between tones {k['fsk_tones_hz']} Hz "
                       f"({k['fsk_switches']} frequency switches/sec) - indicative of modulated covert channel.")
            dedup_freq = float(np.mean(k["fsk_tones_hz"])) if k["fsk_tones_hz"] else freq
            dedup_tol = 1500.0
        elif k["keyed"]:
            kind = "ultrasonic_keyed_tone"
            sev = "High"
            conf = min(94, base_conf + 15)
            threat_cat = "Ultrasonic Acoustic Exfiltration"
            summary = (f"Pulsed tone detected at {freq:,.0f} Hz with On-Off Keying ({k['transitions']} transitions, "
                       f"{k['level_range_db']:.1f} dB amplitude swing) - resembles binary pulse modulation.")
            dedup_freq = freq
            dedup_tol = 300.0
        else:
            kind = "ultrasonic_tone"
            sev = "Medium"
            conf = min(88, base_conf)
            threat_cat = "Ultrasonic Carrier Tone"
            summary = (f"Persistent narrow ultrasonic carrier tone detected at {freq:,.0f} Hz "
                       f"({prom:.1f} dB prominence above local noise floor). Possible unmodulated beacon.")
            dedup_freq = freq
            dedup_tol = 300.0

        return Finding(
            source="ACOUSTIC",
            origin=self.origin,
            kind=kind,
            threat_category=threat_cat,
            severity=sev,
            confidence=conf,
            summary=summary,
            details=details,
            freq=dedup_freq,
            freq_tol=dedup_tol
        )

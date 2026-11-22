"""Realistic mathematical Radio Frequency spectrum simulation environment.

IMPORTANT DISCLAIMER:
  * This module implements a synthetic mathematical model of RF spectrum emissions.
  * It generates Gaussian thermal noise, ambient benign transmitters, and injected
    narrowband covert emanation patterns.
  * Alerts originating from this source are explicitly labeled SIMULATED throughout AetherHound.
  * It verifies algorithm detection logic under controlled conditions, not real-world RF propagation.
"""
from typing import Dict, Tuple, Optional
import numpy as np

from rf_source import RFSource

RF_SCENARIOS: Dict[str, Tuple[str, bool]] = {
    "normal": ("Normal RF Spectrum with Ambient ISM Transmitters (Benign)", False),
    "interference_spike": ("Transient Broadband Interference Spikes (Benign)", False),
    "covert_carrier": ("New Continuous Narrowband Covert Carrier (Suspicious)", True),
    "covert_keyed": ("New On-Off Keyed (OOK) Narrowband Emitter (Suspicious)", True),
    "covert_weak": ("Weak Sub-Threshold Covert Emitter (Deliberately Missed)", True),
}


class SimulatedRFSource(RFSource):
    """Generates synthetic RF spectrum sweeps for hardware-free simulation."""
    origin = "simulated"
    name = "Simulated RF Generator (Mathematical Model - NOT Real RF)"

    # Fixed legitimate ambient transmitters in ISM band: (offset_mhz, power_db)
    LEGIT_EMITTERS = [
        (-0.75, -62.0),
        (-0.15, -68.0),
        (0.50, -64.0)
    ]

    def __init__(self, center_mhz: float = 433.0, span_mhz: float = 2.4, bins: int = 1024,
                 scenario: str = "normal", seed: Optional[int] = None, covert_offset_mhz: float = 0.32):
        self.rng = np.random.default_rng(seed)
        self.freqs = np.linspace(center_mhz - span_mhz / 2.0, center_mhz + span_mhz / 2.0, bins, endpoint=False)
        self.bin_mhz = span_mhz / bins
        self.center = center_mhz
        self.covert_freq = center_mhz + covert_offset_mhz
        self.scenario = scenario
        self._bit_history = [1, 0]

    def set_scenario(self, name: str):
        if name not in RF_SCENARIOS:
            raise ValueError(f"Unknown RF scenario '{name}'")
        self.scenario = name

    def _emitter_shape(self, fc: float, level_db: float) -> np.ndarray:
        """Models a Gaussian spectral shape for a narrowband emitter."""
        shape = np.exp(-0.5 * ((self.freqs - fc) / (1.2 * self.bin_mhz)) ** 2)
        return (10.0 ** (level_db / 10.0)) * shape

    def _next_mod_bit(self) -> int:
        """Generates alternating or pseudo-random bits for keying."""
        b = int(self.rng.integers(0, 2))
        if len(self._bit_history) >= 2 and self._bit_history[-1] == self._bit_history[-2] == b:
            b = 1 - b
        self._bit_history.append(b)
        self._bit_history = self._bit_history[-4:]
        return b

    def read_sweep(self) -> Tuple[np.ndarray, np.ndarray]:
        # Background thermal noise floor (~ -95 dBm with 1.5 dB variance)
        noise_floor_db = self.rng.normal(-95.0, 1.4, self.freqs.size)
        linear_power = 10.0 ** (noise_floor_db / 10.0)

        # Inject legitimate ambient transmitters
        for offset, level in self.LEGIT_EMITTERS:
            linear_power += self._emitter_shape(self.center + offset, level + self.rng.normal(0, 0.4))

        sc = self.scenario
        # Scenario: Random one-off interference spike (15% chance per sweep)
        if sc == "interference_spike" and self.rng.random() < 0.15:
            rand_freq = self.rng.uniform(self.freqs[15], self.freqs[-15])
            linear_power += self._emitter_shape(rand_freq, -66.0)

        # Scenario: Continuous covert carrier
        elif sc == "covert_carrier":
            linear_power += self._emitter_shape(self.covert_freq, -72.0 + self.rng.normal(0, 0.4))

        # Scenario: On-Off Keyed narrowband emitter (AirHopper / GSMem emanation)
        elif sc == "covert_keyed":
            if self._next_mod_bit():
                linear_power += self._emitter_shape(self.covert_freq, -72.0)

        # Scenario: Sub-threshold weak emitter (near noise floor)
        elif sc == "covert_weak":
            if self._next_mod_bit():
                linear_power += self._emitter_shape(self.covert_freq, -91.5)

        power_db = 10.0 * np.log10(linear_power + 1e-20)
        return self.freqs.copy(), power_db

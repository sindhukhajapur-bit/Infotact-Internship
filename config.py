"""Configuration and detection thresholds for AetherHound.

Includes presets for monitoring sensitivity profiles (Normal, Sensitive, High Security),
path resolutions, and signal processing parameters.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any

# Base directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DOCS_DIR = BASE_DIR / "docs"
REPORTS_DIR = BASE_DIR / "reports"

# Default file paths
DB_PATH = DATA_DIR / "aetherhound.db"
LOG_PATH = DATA_DIR / "aetherhound.log"

DATA_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class Settings:
    """Core runtime settings and detection thresholds."""
    # Profile metadata
    profile_name: str = "Normal (Default)"

    # Acoustic acquisition
    sample_rate: int = 44100
    chunk_seconds: float = 1.0
    audio_device: Any = None
    band_low_hz: float = 17000.0
    band_high_hz: float = 21500.0
    acoustic_calibration_chunks: int = 6

    # Acoustic detection thresholds
    tone_prominence_db: float = 12.0
    tone_rise_db: float = 10.0
    broadband_rise_db: float = 14.0
    acoustic_min_consecutive: int = 2
    keying_min_range_db: float = 10.0

    # RF acquisition & spectrum
    rf_center_mhz: float = 433.0
    rf_span_mhz: float = 2.4
    rf_bins: int = 1024
    rf_calibration_sweeps: int = 15
    rf_sweep_interval_s: float = 0.25

    # RF detection thresholds
    rf_threshold_db: float = 8.0
    rf_sigma: float = 5.0
    rf_window: int = 20
    rf_min_window: int = 8
    rf_max_narrowband_khz: float = 35.0

    # Cross-Channel Correlation
    correlation_window_s: float = 10.0
    multi_channel_boost: int = 25

    # Alert management
    alert_cooldown_s: float = 25.0
    beep_on_high: bool = True
    console_alerts: bool = True


# Predefined Sensitivity Profiles
SENSITIVITY_PROFILES: Dict[str, Dict[str, Any]] = {
    "Normal (Default)": {
        "description": "Balanced sensitivity for standard corporate/office environments.",
        "tone_prominence_db": 12.0,
        "tone_rise_db": 10.0,
        "broadband_rise_db": 14.0,
        "acoustic_min_consecutive": 2,
        "rf_threshold_db": 8.0,
        "rf_sigma": 5.0,
        "rf_min_window": 8,
        "alert_cooldown_s": 25.0,
        "correlation_window_s": 10.0,
    },
    "Sensitive": {
        "description": "Heightened sensitivity for quiet air-gapped server vaults or SCADA environments.",
        "tone_prominence_db": 8.5,
        "tone_rise_db": 6.5,
        "broadband_rise_db": 10.0,
        "acoustic_min_consecutive": 2,
        "rf_threshold_db": 5.5,
        "rf_sigma": 3.5,
        "rf_min_window": 6,
        "alert_cooldown_s": 15.0,
        "correlation_window_s": 15.0,
    },
    "High Security (Paranoid)": {
        "description": "Maximum detection vigilance for critical red-team audits. Faster alerts, lower thresholds.",
        "tone_prominence_db": 5.5,
        "tone_rise_db": 4.0,
        "broadband_rise_db": 7.0,
        "acoustic_min_consecutive": 1,
        "rf_threshold_db": 3.5,
        "rf_sigma": 2.5,
        "rf_min_window": 5,
        "alert_cooldown_s": 10.0,
        "correlation_window_s": 20.0,
    },
}


def apply_profile(settings: Settings, profile_name: str) -> Settings:
    """Applies a preset configuration profile to settings."""
    if profile_name not in SENSITIVITY_PROFILES:
        return settings
    profile = SENSITIVITY_PROFILES[profile_name]
    settings.profile_name = profile_name
    for key, val in profile.items():
        if hasattr(settings, key) and key != "description":
            setattr(settings, key, val)
    return settings

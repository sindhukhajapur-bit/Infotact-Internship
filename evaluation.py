"""Model Evaluation and Benchmark Suite for AetherHound.

Executes offline trials across labelled benign and suspicious synthetic scenarios,
calculates confusion matrices, and determines Precision, Recall, F1-Score,
False Positive Rate (FPR), Accuracy, and Detection Latency.
"""
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

import config
from acoustic_detector import AcousticDetector
from models import SEVERITY_ORDER, SYNTHETIC_EVAL_DISCLAIMER
from rf_detector import RFDetector
from rf_simulator import SimulatedRFSource, RF_SCENARIOS
from signal_generators import acoustic_chunk, ACOUSTIC_SCENARIOS


def _summarize_trial(findings, first_detected_idx):
    max_sev = max((SEVERITY_ORDER.get(f.severity, 0) for f in findings), default=0)
    kinds = sorted({f.kind for f in findings})
    return {
        "max_severity": max_sev,
        "kinds": kinds,
        "first_chunk": first_detected_idx,
    }


def evaluate_acoustic_trial(scenario: str, seed: int, settings: config.Settings, n_eval: int = 5) -> Dict[str, Any]:
    """Runs a single acoustic test trial across calibration + evaluation chunks."""
    fs = settings.sample_rate
    n = int(fs * settings.chunk_seconds)
    rng = np.random.default_rng(seed)
    det = AcousticDetector(settings, origin="synthetic_audio")
    t0 = 0.0

    # Calibration phase with benign background noise
    for _ in range(settings.acoustic_calibration_chunks):
        chunk = acoustic_chunk("normal", n, fs, rng, t0)
        det.process(chunk, fs)
        t0 += settings.chunk_seconds

    # Evaluation phase
    found_findings = []
    first_detected = None
    for i in range(n_eval):
        chunk = acoustic_chunk(scenario, n, fs, rng, t0)
        res = det.process(chunk, fs)
        t0 += settings.chunk_seconds

        active_findings = [f for f in res["findings"] if SEVERITY_ORDER.get(f.severity, 0) >= 2]
        if active_findings and first_detected is None:
            first_detected = i + 1
        found_findings.extend(res["findings"])

    return _summarize_trial(found_findings, first_detected)


def evaluate_rf_trial(scenario: str, seed: int, settings: config.Settings, n_eval: int = 25) -> Dict[str, Any]:
    """Runs a single RF test trial across calibration + evaluation sweeps."""
    src = SimulatedRFSource(settings.rf_center_mhz, settings.rf_span_mhz, settings.rf_bins,
                            scenario="normal", seed=seed)
    det = RFDetector(settings, origin="simulated")

    # Calibration phase
    for _ in range(settings.rf_calibration_sweeps):
        freqs, power = src.read_sweep()
        det.process(freqs, power)

    # Evaluation phase
    src.set_scenario(scenario)
    found_findings = []
    first_detected = None
    for i in range(n_eval):
        freqs, power = src.read_sweep()
        res = det.process(freqs, power)

        active_findings = [f for f in res["findings"] if SEVERITY_ORDER.get(f.severity, 0) >= 2]
        if active_findings and first_detected is None:
            first_detected = i + 1
        found_findings.extend(res["findings"])

    return _summarize_trial(found_findings, first_detected)


def compute_binary_metrics(tp: int, fp: int, tn: int, fn: int) -> Dict[str, Any]:
    """Computes standard machine learning classification metrics."""
    total = tp + fp + tn + fn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    accuracy = (tp + tn) / max(1, total)

    return {
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1": round(f1, 4),
        "Accuracy": round(accuracy, 4),
        "False_Positive_Rate": round(fpr, 4),
        "Total_Trials": total
    }


def run_evaluation(trials: int = 15, settings: Optional[config.Settings] = None, seed_base: int = 1000) -> Dict[str, Any]:
    """Executes full evaluation benchmark across all acoustic and RF scenarios."""
    settings = settings or config.Settings()
    trial_records = []

    # 1. Acoustic Evaluation
    for scn_name, (desc, is_suspicious) in ACOUSTIC_SCENARIOS.items():
        for k in range(trials):
            res = evaluate_acoustic_trial(scn_name, seed_base + k, settings)
            trial_records.append({
                "channel": "ACOUSTIC",
                "scenario": scn_name,
                "description": desc,
                "suspicious": is_suspicious,
                "detected": res["max_severity"] >= 2,
                "low_only": res["max_severity"] == 1,
                "first_chunk": res["first_chunk"],
                "max_severity": res["max_severity"]
            })

    # 2. RF Evaluation
    for scn_name, (desc, is_suspicious) in RF_SCENARIOS.items():
        for k in range(trials):
            res = evaluate_rf_trial(scn_name, seed_base + k, settings)
            trial_records.append({
                "channel": "RF (Simulated)",
                "scenario": scn_name,
                "description": desc,
                "suspicious": is_suspicious,
                "detected": res["max_severity"] >= 2,
                "low_only": res["max_severity"] == 1,
                "first_chunk": res["first_chunk"],
                "max_severity": res["max_severity"]
            })

    df = pd.DataFrame(trial_records)

    # Per-scenario aggregation
    per_scenario = (
        df.groupby(["channel", "scenario", "description", "suspicious"])
        .agg(
            trials=("detected", "size"),
            detected=("detected", "sum"),
            low_only=("low_only", "sum"),
            avg_chunks_to_detect=("first_chunk", "mean")
        ).reset_index()
    )
    per_scenario["detection_rate"] = per_scenario["detected"] / per_scenario["trials"]

    # Overall metrics per channel and combined
    overall_metrics = {}
    for ch_name, subset in [("ACOUSTIC", df[df.channel == "ACOUSTIC"]),
                            ("RF (Simulated)", df[df.channel != "ACOUSTIC"]),
                            ("COMBINED", df)]:
        tp = int(((subset.suspicious) & (subset.detected)).sum())
        fn = int(((subset.suspicious) & (~subset.detected)).sum())
        fp = int(((~subset.suspicious) & (subset.detected)).sum())
        tn = int(((~subset.suspicious) & (~subset.detected)).sum())
        overall_metrics[ch_name] = compute_binary_metrics(tp, fp, tn, fn)

    return {
        "per_scenario": per_scenario,
        "overall": overall_metrics,
        "disclaimer": SYNTHETIC_EVAL_DISCLAIMER
    }

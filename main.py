"""Command-line interface runner for AetherHound."""
import argparse
import sys
import time

import config
from database import DatabaseManager
from monitor import MonitoringEngine
from evaluation import run_evaluation
from report_generator import ReportGenerator


def main():
    parser = argparse.ArgumentParser(description="AetherHound CLI Controller")
    parser.add_argument("--eval", action="store_true", help="Run offline synthetic evaluation benchmark")
    parser.add_argument("--trials", type=int, default=15, help="Number of trials per scenario for evaluation")
    parser.add_argument("--report", action="store_true", help="Generate PDF and CSV reports")
    parser.add_argument("--monitor", action="store_true", help="Start console monitoring loop")
    parser.add_argument("--duration", type=int, default=30, help="Monitoring duration in seconds")
    args = parser.parse_args()

    db = DatabaseManager(config.DB_PATH)

    if args.eval:
        print(f"[*] Running AetherHound evaluation benchmark ({args.trials} trials/scenario)...")
        results = run_evaluation(trials=args.trials)
        print("\n=== OVERALL BENCHMARK METRICS ===")
        for ch, m in results["overall"].items():
            print(f"[{ch}] Precision: {m['Precision']:.3f} | Recall: {m['Recall']:.3f} | F1: {m['F1']:.3f} | Acc: {m['Accuracy']:.3f}")
        return

    if args.report:
        print("[*] Generating official PDF report...")
        rg = ReportGenerator(db)
        out_path = config.REPORTS_DIR / "AetherHound_CLI_Report.pdf"
        rg.generate_pdf_report(out_path)
        print(f"[+] Report generated at: {out_path}")
        return

    if args.monitor:
        print(f"[*] Starting console monitoring for {args.duration} seconds...")
        engine = MonitoringEngine(db)
        engine.start(config.Settings(), acoustic="synthetic", rf="simulated")
        try:
            time.sleep(args.duration)
        finally:
            engine.stop()
            print("[+] Monitoring concluded.")
        return

    print("🐕 AetherHound CLI initialized. Use --help to view commands or run 'streamlit run app.py' to launch the SOC Dashboard.")


if __name__ == "__main__":
    main()

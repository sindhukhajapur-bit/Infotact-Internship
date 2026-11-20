"""SQLite persistence layer for AetherHound.

Manages alerts, security incidents, forensic evidence, audit trails,
monitoring sessions, and evaluation records.
"""
import json
import logging
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd

import config
from models import Alert, Incident, EvidenceRecord, AuditEntry

log = logging.getLogger("aetherhound")


class DatabaseManager:
    """Thread-safe SQLite database manager for enterprise security records."""

    def __init__(self, db_path=config.DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._create_tables()
        self._seed_sample_data_if_empty()

    def _create_tables(self):
        """Initializes database schema."""
        with self._lock:
            cur = self._conn.cursor()
            # 1. Alerts Table
            cur.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                alert_uid TEXT UNIQUE,
                timestamp TEXT,
                source TEXT,
                origin TEXT,
                threat_category TEXT,
                kind TEXT,
                severity TEXT,
                confidence INTEGER,
                frequency REAL,
                summary TEXT,
                details TEXT,
                status TEXT DEFAULT 'New',
                updated_at TEXT
            )""")

            # 2. Incidents Table
            cur.execute("""
            CREATE TABLE IF NOT EXISTS incidents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                incident_uid TEXT UNIQUE,
                title TEXT,
                severity TEXT,
                status TEXT DEFAULT 'Open',
                assigned_to TEXT,
                related_alert_ids TEXT,
                summary TEXT,
                technical_findings TEXT,
                risk_score INTEGER,
                investigator_notes TEXT,
                remediation_plan TEXT,
                created_at TEXT,
                updated_at TEXT
            )""")

            # 3. Evidence Records Table
            cur.execute("""
            CREATE TABLE IF NOT EXISTS evidence_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                evidence_uid TEXT UNIQUE,
                incident_uid TEXT,
                alert_uid TEXT,
                timestamp TEXT,
                evidence_type TEXT,
                description TEXT,
                data_json TEXT
            )""")

            # 4. Monitoring Sessions Table
            cur.execute("""
            CREATE TABLE IF NOT EXISTS monitoring_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_uid TEXT UNIQUE,
                started_at TEXT,
                ended_at TEXT,
                acoustic_source TEXT,
                rf_source TEXT,
                profile_name TEXT,
                alerts_count INTEGER DEFAULT 0,
                status TEXT
            )""")

            # 5. Security Configurations Table
            cur.execute("""
            CREATE TABLE IF NOT EXISTS security_configurations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                config_key TEXT UNIQUE,
                config_value TEXT,
                updated_at TEXT
            )""")

            # 6. Audit Logs Table
            cur.execute("""
            CREATE TABLE IF NOT EXISTS audit_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                action TEXT,
                category TEXT,
                details TEXT,
                user TEXT,
                status TEXT DEFAULT 'SUCCESS'
            )""")

            # 7. Evaluation Benchmark Results Table
            cur.execute("""
            CREATE TABLE IF NOT EXISTS evaluation_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                run_uid TEXT,
                channel TEXT,
                scenario TEXT,
                trials INTEGER,
                detected INTEGER,
                precision REAL,
                recall REAL,
                f1 REAL,
                accuracy REAL,
                fpr REAL,
                avg_latency REAL
            )""")

            self._conn.commit()

    # =========================================================================
    # ALERT MANAGEMENT
    # =========================================================================

    def add_alert(self, alert: Alert) -> int:
        """Stores a new alert in the database."""
        now = datetime.now().isoformat(timespec="seconds")
        with self._lock:
            # Generate next alert UID if not present
            cur = self._conn.cursor()
            cur.execute("SELECT COUNT(*) FROM alerts")
            count = cur.fetchone()[0] + 1
            uid = f"ALT-{count:04d}" if not alert.alert_uid else alert.alert_uid
            alert.alert_uid = uid

            cur.execute("""
                INSERT INTO alerts (
                    alert_uid, timestamp, source, origin, threat_category, kind,
                    severity, confidence, frequency, summary, details, status, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                alert.alert_uid, alert.timestamp, alert.source, alert.origin,
                alert.threat_category, alert.kind, alert.severity, alert.confidence,
                alert.frequency, alert.summary, json.dumps(alert.details, default=str),
                alert.status or "New", now
            ))
            self._conn.commit()
            last_id = cur.lastrowid

        self.log_audit("ALERT_GENERATED", "ALERT", f"Generated alert {uid}: {alert.summary}", status="INFO")
        return last_id

    def update_alert_status(self, alert_id_or_uid: Any, new_status: str, notes: str = ""):
        """Updates the triage status of an alert."""
        now = datetime.now().isoformat(timespec="seconds")
        with self._lock:
            if isinstance(alert_id_or_uid, int):
                self._conn.execute(
                    "UPDATE alerts SET status = ?, updated_at = ? WHERE id = ?",
                    (new_status, now, alert_id_or_uid)
                )
            else:
                self._conn.execute(
                    "UPDATE alerts SET status = ?, updated_at = ? WHERE alert_uid = ?",
                    (new_status, now, str(alert_id_or_uid))
                )
            self._conn.commit()

        self.log_audit("ALERT_STATUS_UPDATE", "ALERT",
                       f"Updated alert {alert_id_or_uid} status to '{new_status}'. Notes: {notes}")

    def fetch_alerts(self, limit: int = 1000) -> pd.DataFrame:
        """Fetches alerts ordered by ID descending as a pandas DataFrame."""
        with self._lock:
            df = pd.read_sql_query(
                "SELECT * FROM alerts ORDER BY id DESC LIMIT ?",
                self._conn, params=(limit,)
            )
        return df

    def get_alert_by_uid(self, uid: str) -> Optional[Dict[str, Any]]:
        """Retrieves a single alert dict by UID."""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("SELECT * FROM alerts WHERE alert_uid = ?", (uid,))
            row = cur.fetchone()
            if row:
                d = dict(row)
                if isinstance(d.get("details"), str):
                    try:
                        d["details"] = json.loads(d["details"])
                    except Exception:
                        pass
                return d
        return None

    def get_alert_counts(self) -> Dict[str, int]:
        """Calculates alert statistics by severity."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT severity, COUNT(*) FROM alerts GROUP BY severity"
            ).fetchall()
            status_rows = self._conn.execute(
                "SELECT status, COUNT(*) FROM alerts GROUP BY status"
            ).fetchall()

        c = {"Low": 0, "Medium": 0, "High": 0, "Critical": 0}
        for sev, count in rows:
            if sev in c:
                c[sev] = count
        c["Total"] = sum(c.values())

        status_counts = {"New": 0, "Acknowledged": 0, "Investigating": 0, "Resolved": 0, "False Positive": 0}
        for st, count in status_rows:
            if st in status_counts:
                status_counts[st] = count
        c.update({f"Status_{k}": v for k, v in status_counts.items()})
        return c

    # =========================================================================
    # INCIDENT MANAGEMENT
    # =========================================================================

    def create_incident(self, incident: Incident) -> str:
        """Creates an incident investigation case."""
        now = datetime.now().isoformat(timespec="seconds")
        with self._lock:
            cur = self._conn.cursor()
            if not incident.incident_uid:
                cur.execute("SELECT COUNT(*) FROM incidents")
                count = cur.fetchone()[0] + 1
                incident.incident_uid = f"INC-{now[:4]}-{count:03d}"

            cur.execute("""
                INSERT INTO incidents (
                    incident_uid, title, severity, status, assigned_to,
                    related_alert_ids, summary, technical_findings, risk_score,
                    investigator_notes, remediation_plan, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                incident.incident_uid, incident.title, incident.severity,
                incident.status or "Open", incident.assigned_to or "Security Admin Intern",
                json.dumps(incident.related_alert_ids), incident.summary,
                incident.technical_findings, incident.risk_score,
                incident.investigator_notes, incident.remediation_plan,
                incident.created_at or now, now
            ))
            self._conn.commit()

        self.log_audit("INCIDENT_CREATED", "INCIDENT",
                       f"Created incident {incident.incident_uid}: {incident.title}")
        return incident.incident_uid

    def update_incident(self, incident_uid: str, **kwargs):
        """Updates fields of an existing incident."""
        now = datetime.now().isoformat(timespec="seconds")
        kwargs["updated_at"] = now
        set_clauses = [f"{k} = ?" for k in kwargs.keys()]
        values = list(kwargs.values()) + [incident_uid]

        with self._lock:
            self._conn.execute(
                f"UPDATE incidents SET {', '.join(set_clauses)} WHERE incident_uid = ?",
                values
            )
            self._conn.commit()

        self.log_audit("INCIDENT_UPDATED", "INCIDENT",
                       f"Updated incident {incident_uid} ({', '.join(kwargs.keys())})")

    def fetch_incidents(self, limit: int = 100) -> pd.DataFrame:
        """Fetches incident records as a DataFrame."""
        with self._lock:
            df = pd.read_sql_query(
                "SELECT * FROM incidents ORDER BY id DESC LIMIT ?",
                self._conn, params=(limit,)
            )
        return df

    def get_incident_by_uid(self, incident_uid: str) -> Optional[Dict[str, Any]]:
        """Retrieves single incident dictionary."""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("SELECT * FROM incidents WHERE incident_uid = ?", (incident_uid,))
            row = cur.fetchone()
            if row:
                d = dict(row)
                if isinstance(d.get("related_alert_ids"), str):
                    try:
                        d["related_alert_ids"] = json.loads(d["related_alert_ids"])
                    except Exception:
                        d["related_alert_ids"] = []
                return d
        return None

    # =========================================================================
    # EVIDENCE RECORDS
    # =========================================================================

    def add_evidence(self, evidence: EvidenceRecord) -> str:
        """Stores a digital forensic evidence artifact."""
        now = datetime.now().isoformat(timespec="seconds")
        with self._lock:
            cur = self._conn.cursor()
            if not evidence.evidence_uid:
                cur.execute("SELECT COUNT(*) FROM evidence_records")
                count = cur.fetchone()[0] + 1
                evidence.evidence_uid = f"EVD-{now[:4]}-{count:04d}"

            cur.execute("""
                INSERT INTO evidence_records (
                    evidence_uid, incident_uid, alert_uid, timestamp,
                    evidence_type, description, data_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                evidence.evidence_uid, evidence.incident_uid, evidence.alert_uid,
                evidence.timestamp or now, evidence.evidence_type,
                evidence.description, evidence.data_json
            ))
            self._conn.commit()

        self.log_audit("EVIDENCE_CAPTURED", "INVESTIGATION",
                       f"Stored evidence {evidence.evidence_uid} for {evidence.incident_uid or evidence.alert_uid}")
        return evidence.evidence_uid

    def fetch_evidence_for_incident(self, incident_uid: str) -> List[Dict[str, Any]]:
        """Retrieves all evidence associated with an incident."""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "SELECT * FROM evidence_records WHERE incident_uid = ? ORDER BY id ASC",
                (incident_uid,)
            )
            return [dict(r) for r in cur.fetchall()]

    # =========================================================================
    # MONITORING SESSIONS
    # =========================================================================

    def start_session(self, acoustic_source: str, rf_source: str, profile_name: str) -> str:
        """Records the beginning of a monitoring session."""
        now = datetime.now().isoformat(timespec="seconds")
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("SELECT COUNT(*) FROM monitoring_sessions")
            count = cur.fetchone()[0] + 1
            session_uid = f"SES-{now[:4]}-{count:03d}"

            cur.execute("""
                INSERT INTO monitoring_sessions (
                    session_uid, started_at, acoustic_source, rf_source, profile_name, status
                ) VALUES (?, ?, ?, ?, ?, 'ACTIVE')
            """, (session_uid, now, acoustic_source, rf_source, profile_name))
            self._conn.commit()

        self.log_audit("MONITORING_STARTED", "MONITORING",
                       f"Session {session_uid} started [Acoustic: {acoustic_source}, RF: {rf_source}, Profile: {profile_name}]")
        return session_uid

    def end_session(self, session_uid: str, alerts_count: int = 0):
        """Concludes a monitoring session."""
        now = datetime.now().isoformat(timespec="seconds")
        with self._lock:
            self._conn.execute("""
                UPDATE monitoring_sessions
                SET ended_at = ?, alerts_count = ?, status = 'COMPLETED'
                WHERE session_uid = ?
            """, (now, alerts_count, session_uid))
            self._conn.commit()

        self.log_audit("MONITORING_STOPPED", "MONITORING",
                       f"Session {session_uid} stopped. Generated {alerts_count} alerts.")

    def fetch_recent_sessions(self, limit: int = 20) -> pd.DataFrame:
        """Fetches monitoring history."""
        with self._lock:
            return pd.read_sql_query(
                "SELECT * FROM monitoring_sessions ORDER BY id DESC LIMIT ?",
                self._conn, params=(limit,)
            )

    # =========================================================================
    # AUDIT TRAIL
    # =========================================================================

    def log_audit(self, action: str, category: str, details: str, user: str = "Security Admin Intern", status: str = "SUCCESS"):
        """Logs an event in the immutable audit trail."""
        now = datetime.now().isoformat(timespec="seconds")
        with self._lock:
            self._conn.execute("""
                INSERT INTO audit_logs (timestamp, action, category, details, user, status)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (now, action, category, details, user, status))
            self._conn.commit()

    def fetch_audit_logs(self, limit: int = 500) -> pd.DataFrame:
        """Fetches recent audit log entries."""
        with self._lock:
            return pd.read_sql_query(
                "SELECT * FROM audit_logs ORDER BY id DESC LIMIT ?",
                self._conn, params=(limit,)
            )

    # =========================================================================
    # MAINTENANCE & DEMO SEEDING
    # =========================================================================

    def clear_alerts(self):
        """Clears alert history for testing."""
        with self._lock:
            self._conn.execute("DELETE FROM alerts")
            self._conn.commit()
        self.log_audit("DATABASE_PURGE", "SYSTEM", "Alerts table cleared by operator", status="WARNING")

    def _seed_sample_data_if_empty(self):
        """Seeds realistic sample demonstration data for internship evaluation."""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("SELECT COUNT(*) FROM alerts")
            if cur.fetchone()[0] > 0:
                return  # Already has data

        now = datetime.now().isoformat(timespec="seconds")

        # 1. Sample Initial Alerts
        sample_alerts = [
            Alert(
                id=1, alert_uid="ALT-0001", timestamp=now, source="ACOUSTIC",
                origin="microphone", threat_category="Ultrasonic Acoustic Exfiltration",
                kind="ultrasonic_fsk", severity="High", confidence=92, frequency=19500.0,
                summary="Ultrasonic FSK data transmission detected (19.0 kHz / 20.0 kHz shift keying).",
                details={
                    "freq_hz": 19500.0, "prominence_db": 18.4, "rise_over_baseline_db": 16.2,
                    "fsk": True, "fsk_switches": 7, "fsk_tones_hz": [19000, 20000],
                    "spectral_flatness": 0.041
                },
                status="Investigating"
            ),
            Alert(
                id=2, alert_uid="ALT-0002", timestamp=now, source="RF",
                origin="simulated", threat_category="Modulated RF Carrier",
                kind="rf_keyed_emitter", severity="High", confidence=88, frequency=433.32,
                summary="New pulsed narrowband RF carrier at 433.320 MHz with periodic keying.",
                details={
                    "freq_mhz": 433.32, "peak_db": -71.8, "excess_over_baseline_db": 22.4,
                    "width_khz": 12.5, "duty_cycle": 0.48, "transitions": 6
                },
                status="Investigating"
            ),
            Alert(
                id=3, alert_uid="ALT-0003", timestamp=now, source="CORRELATED",
                origin="simulated", threat_category="Multi-Channel Coordinated Exfiltration",
                kind="correlated_acoustic_rf_attack", severity="Critical", confidence=96,
                frequency=433.32,
                summary="Coordinated Acoustic & RF anomaly detected within 4.2s window. Possible multi-stage air-gap bridge.",
                details={
                    "acoustic_alert_uid": "ALT-0001", "rf_alert_uid": "ALT-0002",
                    "delta_time_s": 4.2, "combined_risk_score": 94, "mitre_attack": "T1011 / T1048"
                },
                status="Investigating"
            ),
            Alert(
                id=4, alert_uid="ALT-0004", timestamp=now, source="ACOUSTIC",
                origin="microphone", threat_category="Ultrasonic Acoustic Exfiltration",
                kind="ultrasonic_tone", severity="Medium", confidence=74, frequency=18200.0,
                summary="Persistent narrow ultrasonic tone at 18.2 kHz standing 13.5 dB above noise floor.",
                details={"freq_hz": 18200.0, "prominence_db": 13.5, "rise_over_baseline_db": 11.0},
                status="Acknowledged"
            ),
            Alert(
                id=5, alert_uid="ALT-0005", timestamp=now, source="RF",
                origin="simulated", threat_category="Environmental RF Noise",
                kind="rf_wideband_rise", severity="Low", confidence=35, frequency=432.8,
                summary="Wideband transient rise around 432.8 MHz (85 kHz width) consistent with benign switching transient.",
                details={"freq_mhz": 432.8, "width_khz": 85.0, "excess_over_baseline_db": 7.2},
                status="Resolved"
            )
        ]

        for a in sample_alerts:
            self.add_alert(a)

        # 2. Sample Investigation Case
        sample_incident = Incident(
            id=1, incident_uid=f"INC-{now[:4]}-001",
            title="Coordinated Ultrasonic & RF Air-Gap Bridge Anomaly",
            severity="Critical", status="Under Investigation",
            assigned_to="Sindu Shrishail Khajapur (Security Admin L1)",
            related_alert_ids=["ALT-0001", "ALT-0002", "ALT-0003"],
            summary="Multi-sensor detection of coordinated acoustic FSK exfiltration synchronized with 433 MHz RF burst emanation.",
            technical_findings=(
                "Primary acoustic sensor recorded 19.5 kHz dual-tone FSK transmission for 6 seconds. "
                "Simultaneously, RF monitor recorded a correlated 433.32 MHz keyed emitter with 48% duty cycle. "
                "Signal timing correlation delta is 4.2 seconds, indicating an active cross-medium exfiltration attempt."
            ),
            risk_score=94,
            investigator_notes="Initial baseline verified. External audio recorder placed 1.5m from test unit. Initiating RF isolation protocol.",
            remediation_plan=(
                "1. Disconnect peripheral audio inputs and mute hardware sound card.\n"
                "2. Place target workstation inside Faraday RF isolation enclosure.\n"
                "3. Perform forensic memory acquisition on host to check for known air-gap malware (e.g., MOSQUITO, AirHopper)."
            ),
            created_at=now, updated_at=now
        )
        self.create_incident(sample_incident)

        # 3. Sample Evidence
        sample_evidence = EvidenceRecord(
            id=1, evidence_uid=f"EVD-{now[:4]}-0001",
            incident_uid=sample_incident.incident_uid, alert_uid="ALT-0001",
            timestamp=now, evidence_type="SPECTRUM_SNAPSHOT",
            description="Welch PSD spectrum showing dual 19.0 kHz and 20.0 kHz FSK peaks with 18.4 dB SNR.",
            data_json=json.dumps({"peak_freq": 19500, "snr_db": 18.4, "sample_rate": 44100})
        )
        self.add_evidence(sample_evidence)

        # 4. Initial Audit Log
        self.log_audit("SYSTEM_INIT", "SYSTEM", "AetherHound initialized with default security profiles and demo data.")

"""Automated Security Report Generation Engine.

Produces publication-grade executive and technical PDF reports using ReportLab,
as well as standardized CSV data packages for compliance and archival.
"""
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import io
import pandas as pd

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY

import config
from models import DISCLAIMER, SYNTHETIC_EVAL_DISCLAIMER


class ReportGenerator:
    """Generates professional cybersecurity audit and incident reports."""

    def __init__(self, db_manager):
        self.db = db_manager

    def generate_pdf_report(self, filepath: Optional[Path] = None,
                            include_eval: bool = True) -> bytes:
        """Builds a complete, multi-page professional cybersecurity report."""
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            filepath if filepath else buffer,
            pagesize=letter,
            rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40
        )

        styles = getSampleStyleSheet()
        # Custom styles
        title_style = ParagraphStyle(
            "DocTitle",
            parent=styles["Heading1"],
            fontSize=22,
            leading=26,
            textColor=colors.HexColor("#0f172a"),
            alignment=TA_CENTER,
            spaceAfter=6
        )
        subtitle_style = ParagraphStyle(
            "DocSubTitle",
            parent=styles["Normal"],
            fontSize=11,
            leading=15,
            textColor=colors.HexColor("#475569"),
            alignment=TA_CENTER,
            spaceAfter=15
        )
        h1_style = ParagraphStyle(
            "H1",
            parent=styles["Heading2"],
            fontSize=13,
            leading=17,
            textColor=colors.HexColor("#0284c7"),
            spaceBefore=12,
            spaceAfter=6,
            keepWithNext=True
        )
        body_style = ParagraphStyle(
            "Body",
            parent=styles["Normal"],
            fontSize=9.5,
            leading=13.5,
            textColor=colors.HexColor("#1e293b"),
            alignment=TA_JUSTIFY,
            spaceAfter=6
        )
        callout_style = ParagraphStyle(
            "Callout",
            parent=styles["Normal"],
            fontSize=9,
            leading=13,
            textColor=colors.HexColor("#334155"),
            backColor=colors.HexColor("#f8fafc"),
            borderColor=colors.HexColor("#cbd5e1"),
            borderWidth=1,
            borderPadding=8,
            spaceBefore=6,
            spaceAfter=8
        )

        story = []
        now_str = datetime.now().strftime("%d %B %Y, %H:%M:%S")

        # ---------------- 1. HEADER / TITLE BLOCK ----------------
        story.append(Paragraph("INFOTACT SOLUTIONS · CYBER SECURITY DIVISION", ParagraphStyle(
            "OrgHeader", fontName="Helvetica-Bold", fontSize=10, textColor=colors.HexColor("#0284c7"), alignment=TA_CENTER
        )))
        story.append(Spacer(1, 4))
        story.append(Paragraph("AetherHound: Air-Gapped RF & Acoustic Covert Channel Detector", title_style))
        story.append(Paragraph("Automated Threat Intelligence & Security Audit Report", subtitle_style))
        story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0284c7"), spaceAfter=12))

        # Metadata table
        meta_data = [
            [Paragraph("<b>Internship Role:</b> Security Admin Associate L1", body_style),
             Paragraph(f"<b>Report Date:</b> {now_str}", body_style)],
            [Paragraph("<b>Investigator:</b> Sindu Shrishail Khajapur", body_style),
             Paragraph("<b>Project Ref No:</b> ITS/CS/1237", body_style)],
            [Paragraph("<b>Domain:</b> Air-Gap Perimeter Surveillance", body_style),
             Paragraph("<b>Compliance:</b> ISO 9001:2015 / MSME Verified", body_style)],
        ]
        t_meta = Table(meta_data, colWidths=[270, 260])
        t_meta.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f1f5f9")),
            ('PADDING', (0, 0), (-1, -1), 6),
            ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        story.append(t_meta)
        story.append(Spacer(1, 14))

        # ---------------- 2. EXECUTIVE SUMMARY ----------------
        story.append(Paragraph("1. Executive Summary", h1_style))
        counts = self.db.get_alert_counts()
        incidents = self.db.fetch_incidents(10)
        open_incidents = len(incidents[incidents.status != "Closed"]) if not incidents.empty else 0

        posture = "ELEVATED THREAT DETECTED" if counts.get("Critical", 0) > 0 or counts.get("High", 0) > 0 else "NORMAL MONITORING POSTURE"
        posture_color = "#ef4444" if "ELEVATED" in posture else "#10b981"

        exec_text = (
            f"During the surveillance period, AetherHound continuously inspected the physical air-gap "
            f"boundary using dual-modality acoustic sensing and RF spectrum anomaly detection. "
            f"A total of <b>{counts.get('Total', 0)} security alerts</b> were recorded "
            f"({counts.get('Critical', 0)} Critical, {counts.get('High', 0)} High, "
            f"{counts.get('Medium', 0)} Medium, {counts.get('Low', 0)} Low). "
            f"Currently, <b>{open_incidents} active incident cases</b> are under active investigation. "
            f"Current security posture status: <font color='{posture_color}'><b>{posture}</b></font>."
        )
        story.append(Paragraph(exec_text, body_style))
        story.append(Spacer(1, 8))

        # ---------------- 3. SECURITY ALERT BREAKDOWN ----------------
        story.append(Paragraph("2. Threat & Alert Distribution", h1_style))
        alert_summary = [
            ["Severity Tier", "Count", "Percentage", "Operational Response Action"],
            ["Critical", str(counts.get("Critical", 0)),
             f"{(counts.get('Critical', 0)/max(1, counts.get('Total', 1))*100):.1f}%",
             "Immediate RF isolation & host forensic memory dump"],
            ["High", str(counts.get("High", 0)),
             f"{(counts.get('High', 0)/max(1, counts.get('Total', 1))*100):.1f}%",
             "Physical verification of perimeter & peripheral inspection"],
            ["Medium", str(counts.get("Medium", 0)),
             f"{(counts.get('Medium', 0)/max(1, counts.get('Total', 1))*100):.1f}%",
             "Scheduled spectrum review and baseline variance check"],
            ["Low", str(counts.get("Low", 0)),
             f"{(counts.get('Low', 0)/max(1, counts.get('Total', 1))*100):.1f}%",
             "Informational logging (ambient noise / transient spikes)"],
            ["Total Recorded", str(counts.get("Total", 0)), "100.0%", "Complete event audit log maintained"],
        ]
        t_alerts = Table(alert_summary, colWidths=[90, 60, 80, 300])
        t_alerts.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0f172a")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8.5),
            ('PADDING', (0, 0), (-1, -1), 5),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ('ROWBACKGROUNDS', (0, 1), (-1, -2), [colors.white, colors.HexColor("#f8fafc")]),
            ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor("#e2e8f0")),
            ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
        ]))
        story.append(t_alerts)
        story.append(Spacer(1, 12))

        # ---------------- 4. RECENT HIGH-SEVERITY ALERTS ----------------
        story.append(Paragraph("3. Significant Security Incidents & Findings", h1_style))
        alerts_df = self.db.fetch_alerts(limit=5)
        if not alerts_df.empty:
            inc_data = [["Alert ID", "Timestamp", "Channel", "Threat Category", "Severity", "Confidence"]]
            for _, r in alerts_df.head(5).iterrows():
                inc_data.append([
                    str(r.get("alert_uid", "")),
                    str(r.get("timestamp", ""))[:19].replace("T", " "),
                    str(r.get("source", "")),
                    str(r.get("threat_category", "")),
                    str(r.get("severity", "")),
                    f"{r.get('confidence', 0)}/100"
                ])
            t_inc = Table(inc_data, colWidths=[70, 100, 65, 175, 60, 60])
            t_inc.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1e293b")),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('PADDING', (0, 0), (-1, -1), 4.5),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
            ]))
            story.append(t_inc)
        else:
            story.append(Paragraph("No alerts currently recorded in the active database.", body_style))
        story.append(Spacer(1, 12))

        # ---------------- 5. RISK ASSESSMENT & MITRE ATT&CK ----------------
        story.append(Paragraph("4. Security Risk Assessment & Threat Matrix", h1_style))
        risk_text = (
            "AetherHound evaluates potential air-gap covert communication channels following "
            "NIST SP 800-30 and ISO 27005 guidelines. Covert channels are categorized against "
            "the MITRE ATT&CK for Enterprise framework under technique <b>T1011 (Other Network Mediums)</b>, "
            "<b>T1048 (Exfiltration Over Alternative Protocol)</b>, and <b>T1092 (Communication Through "
            "Removable Media / Adjacent Emanation)</b>."
        )
        story.append(Paragraph(risk_text, body_style))
        story.append(Spacer(1, 6))

        matrix_data = [
            ["Threat Vector", "MITRE ID", "Likelihood", "Impact", "Risk Score", "Severity"],
            ["Multi-Channel Acoustic+RF Bridge", "T1011/T1048", "5 / 5", "5 / 5", "25 / 25", "Critical"],
            ["Ultrasonic FSK / OOK Exfiltration", "T1011", "4 / 5", "4 / 5", "16 / 25", "High"],
            ["Modulated Narrowband RF Emitter", "T1011", "4 / 5", "4 / 5", "16 / 25", "High"],
            ["Continuous Ultrasonic Carrier", "T1011", "3 / 5", "3 / 5", "9 / 25", "Medium"],
            ["Ambient Ultrasonic / RF Noise", "N/A", "1 / 5", "1 / 5", "1 / 25", "Low"],
        ]
        t_matrix = Table(matrix_data, colWidths=[170, 75, 65, 65, 75, 80])
        t_matrix.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0f172a")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('PADDING', (0, 0), (-1, -1), 4.5),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ]))
        story.append(t_matrix)
        story.append(Spacer(1, 12))

        # ---------------- 6. ACTIONABLE RECOMMENDATIONS ----------------
        story.append(Paragraph("5. Contextual Security Recommendations & Countermeasures", h1_style))
        recs = [
            "<b>1. Physical Audio Air-Gap Hardening:</b> In high-security enclaves, mechanically disconnect or permanently baffle internal microphone capsules. Enforce OS-level driver disconnections on isolated endpoints.",
            "<b>2. RF Emanation Shielding (TEMPEST):</b> Place air-gapped critical endpoints within certified RF attenuation enclosures or Faraday tents to attenuate parasitic radio emissions from internal bus traces.",
            "<b>3. Ultrasonic Audio Masking:</b> Deploy low-intensity ultrasonic white-noise acoustic emitters (18 kHz - 24 kHz) to desensitize and jam rogue receiver microphones without disturbing human personnel.",
            "<b>4. Peripheral & USB Lockdown:</b> Implement strict port-security policies to prevent unauthorized installation of unauthorized SDR transceivers, Bluetooth bridges, or rogue USB cables.",
            "<b>5. Continuous Baseline Recalibration:</b> Perform periodic recalibration of the RF and acoustic ambient baselines whenever physical room configuration or ambient computing hardware changes."
        ]
        for r in recs:
            story.append(Paragraph(r, body_style))
            story.append(Spacer(1, 3))
        story.append(Spacer(1, 10))

        # ---------------- 7. DISCLAIMERS & LIMITATIONS ----------------
        story.append(Paragraph("6. Evaluation Disclaimers & Technical Limitations", h1_style))
        story.append(Paragraph(f"<b>Operational Disclaimer:</b> {DISCLAIMER}", callout_style))
        story.append(Paragraph(f"<b>Synthetic Benchmark Disclaimer:</b> {SYNTHETIC_EVAL_DISCLAIMER}", callout_style))
        story.append(Spacer(1, 10))

        # ---------------- 8. SIGNATURE BLOCK ----------------
        sig_data = [
            [Paragraph("<b>Prepared By:</b>", body_style), Paragraph("<b>Reviewed & Approved By:</b>", body_style)],
            [Paragraph("Sindu Shrishail Khajapur<br/>Security Admin Associate L1 Intern<br/>Infotact Solutions", body_style),
             Paragraph("Security Operations Lead / Project Mentor<br/>Infotact Solutions Cyber Security Practice<br/>Ref: ITS/CS/1237", body_style)]
        ]
        t_sig = Table(sig_data, colWidths=[265, 265])
        t_sig.setStyle(TableStyle([
            ('PADDING', (0, 0), (-1, -1), 6),
            ('LINEABOVE', (0, 0), (-1, 0), 1, colors.HexColor("#cbd5e1")),
        ]))
        story.append(KeepTogether([t_sig]))

        doc.build(story)
        if filepath:
            with open(filepath, "rb") as f:
                return f.read()
        return buffer.getvalue()

    def export_alerts_csv(self) -> str:
        """Returns CSV string of all alerts."""
        df = self.db.fetch_alerts(10000)
        return df.to_csv(index=False)

    def export_incidents_csv(self) -> str:
        """Returns CSV string of all incidents."""
        df = self.db.fetch_incidents(1000)
        return df.to_csv(index=False)

    def export_audit_csv(self) -> str:
        """Returns CSV string of all audit logs."""
        df = self.db.fetch_audit_logs(10000)
        return df.to_csv(index=False)

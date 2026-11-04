"""
src/ui/dashboard_dialog.py

Self-Learning & Prediction Accuracy Dashboard Dialog.
Displays discovered cards, verified predictions, accuracy statistics,
recent learning events, and the learned card gallery.
"""

from typing import Optional
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QTableWidget, QTableWidgetItem, QHeaderView,
    QTabWidget, QWidget
)

from src.memory.self_learning_db import SelfLearningDatabase


class LearningDashboardDialog(QDialog):
    """Modern dark-themed learning statistics and accuracy dashboard."""

    def __init__(self, db: Optional[SelfLearningDatabase] = None, parent=None):
        super().__init__(parent)
        self.db = db or SelfLearningDatabase()

        self.setWindowTitle("Self-Learning & Accuracy Dashboard")
        self.setMinimumSize(680, 520)
        self.setStyleSheet("""
            QDialog {
                background-color: #0f172a;
                color: #f8fafc;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            }
            QLabel {
                color: #f8fafc;
            }
            QTabWidget::pane {
                border: 1px solid #334155;
                background-color: #1e293b;
                border-radius: 8px;
            }
            QTabBar::tab {
                background: #0f172a;
                color: #94a3b8;
                padding: 8px 16px;
                margin-right: 4px;
                border-top-left-radius: 6px;
                border-top-right-radius: 6px;
                font-weight: bold;
                font-size: 11px;
            }
            QTabBar::tab:selected {
                background: #1e293b;
                color: #38bdf8;
                border-bottom: 2px solid #38bdf8;
            }
            QTableWidget {
                background-color: #0f172a;
                border: 1px solid #334155;
                border-radius: 6px;
                gridline-color: #1e293b;
                color: #f8fafc;
                font-size: 11px;
            }
            QHeaderView::section {
                background-color: #1e293b;
                color: #94a3b8;
                padding: 6px;
                font-weight: bold;
                border: none;
                border-bottom: 1px solid #334155;
            }
            QPushButton {
                background-color: #3b82f6;
                color: white;
                border-radius: 6px;
                padding: 6px 14px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #2563eb;
            }
        """)

        self.init_ui()
        self.refresh_data()

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(18, 18, 18, 18)
        main_layout.setSpacing(14)

        # Header Title
        title_box = QHBoxLayout()
        title_lbl = QLabel("AUTONOMOUS SELF-LEARNING DASHBOARD")
        title_lbl.setStyleSheet("font-size: 15px; font-weight: 800; color: #38bdf8; letter-spacing: 0.8px;")
        title_box.addWidget(title_lbl)
        title_box.addStretch()

        btn_refresh = QPushButton("🔄 Refresh")
        btn_refresh.clicked.connect(self.refresh_data)
        title_box.addWidget(btn_refresh)
        main_layout.addLayout(title_box)

        # Stat Cards Grid (4 boxes)
        stats_layout = QHBoxLayout()
        stats_layout.setSpacing(10)

        self.card_discovered = self._create_stat_card("DISCOVERED CARDS", "0", "#38bdf8")
        self.card_samples = self._create_stat_card("BLUR SAMPLES", "0", "#a78bfa")
        self.card_accuracy = self._create_stat_card("PREDICTION ACCURACY", "0.0%", "#4ade80")
        self.card_record = self._create_stat_card("CORRECT / WRONG", "0 / 0", "#f59e0b")

        stats_layout.addWidget(self.card_discovered)
        stats_layout.addWidget(self.card_samples)
        stats_layout.addWidget(self.card_accuracy)
        stats_layout.addWidget(self.card_record)
        main_layout.addLayout(stats_layout)

        # Tabs: Overview (Recent Events) & Card Gallery
        self.tabs = QTabWidget()

        # Tab 1: Recent Learning Events
        tab1 = QWidget()
        tab1_layout = QVBoxLayout(tab1)
        tab1_layout.setContentsMargins(12, 12, 12, 12)
        tab1_layout.setSpacing(10)

        # Latest Learning Event Banner
        self.banner_frame = QFrame()
        self.banner_frame.setStyleSheet("""
            QFrame {
                background: rgba(30, 41, 59, 0.9);
                border: 1px solid #38bdf8;
                border-radius: 8px;
                padding: 10px;
            }
        """)
        banner_layout = QVBoxLayout(self.banner_frame)
        banner_layout.setContentsMargins(8, 8, 8, 8)
        banner_layout.setSpacing(4)

        self.banner_header = QLabel("LATEST LEARNING EVENT")
        self.banner_header.setStyleSheet("color: #38bdf8; font-size: 10px; font-weight: bold; letter-spacing: 0.8px;")
        self.banner_title = QLabel("No learning events yet")
        self.banner_title.setStyleSheet("color: #f8fafc; font-size: 15px; font-weight: bold;")
        self.banner_details = QLabel("Play Roblox rounds to observe and learn cards automatically.")
        self.banner_details.setStyleSheet("color: #94a3b8; font-size: 11px;")
        self.banner_details.setWordWrap(True)

        banner_layout.addWidget(self.banner_header)
        banner_layout.addWidget(self.banner_title)
        banner_layout.addWidget(self.banner_details)
        tab1_layout.addWidget(self.banner_frame)

        # Table of Recent Events
        self.events_table = QTableWidget()
        self.events_table.setColumnCount(5)
        self.events_table.setHorizontalHeaderLabels(["Session ID", "Card Name", "AI Guess", "Outcome", "Slot Result"])
        self.events_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.events_table.verticalHeader().setVisible(False)
        tab1_layout.addWidget(self.events_table)

        self.tabs.addTab(tab1, "Recent Learning Events")

        # Tab 2: Discovered Cards Gallery
        tab2 = QWidget()
        tab2_layout = QVBoxLayout(tab2)
        tab2_layout.setContentsMargins(12, 12, 12, 12)

        self.gallery_table = QTableWidget()
        self.gallery_table.setColumnCount(4)
        self.gallery_table.setHorizontalHeaderLabels(["Card ID", "Canonical Name", "Observations", "Clear Artwork"])
        self.gallery_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.gallery_table.verticalHeader().setVisible(False)
        tab2_layout.addWidget(self.gallery_table)

        self.tabs.addTab(tab2, "Learned Cards Gallery")

        main_layout.addWidget(self.tabs)

        # Bottom Close Button
        btn_box = QHBoxLayout()
        btn_box.addStretch()
        btn_close = QPushButton("Close")
        btn_close.setStyleSheet("background-color: #475569; padding: 6px 20px;")
        btn_close.clicked.connect(self.accept)
        btn_box.addWidget(btn_close)
        main_layout.addLayout(btn_box)

    def _create_stat_card(self, title: str, value: str, color: str) -> QFrame:
        card = QFrame()
        card.setStyleSheet("""
            QFrame {
                background-color: #1e293b;
                border: 1px solid #334155;
                border-radius: 8px;
                padding: 8px;
            }
        """)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(2)

        lbl_title = QLabel(title)
        lbl_title.setStyleSheet("font-size: 9px; color: #94a3b8; font-weight: bold; letter-spacing: 0.5px;")
        lbl_val = QLabel(value)
        lbl_val.setStyleSheet(f"font-size: 17px; font-weight: 800; color: {color};")
        lbl_val.setObjectName("val")

        layout.addWidget(lbl_title)
        layout.addWidget(lbl_val)
        return card

    def refresh_data(self):
        """Fetches fresh metrics from SelfLearningDatabase and updates views."""
        stats = self.db.get_stats()

        # Update Stat Cards
        self.card_discovered.findChild(QLabel, "val").setText(str(stats["total_discovered_cards"]))
        self.card_samples.findChild(QLabel, "val").setText(str(stats["total_blur_samples"]))
        self.card_accuracy.findChild(QLabel, "val").setText(f"{stats['accuracy_percent']:.1f}%")
        self.card_record.findChild(QLabel, "val").setText(
            f"{stats['total_correct_predictions']} ✓ / {stats['total_incorrect_predictions']} ✕"
        )

        # Update Latest Event Banner
        last_event = stats.get("last_event")
        if last_event:
            cname = last_event["card_name"]
            prev_pred = last_event["previous_prediction"]
            outcome = last_event["outcome"]
            is_corr = (outcome == "CORRECT")

            color = "#4ade80" if is_corr else "#f59e0b"
            status_text = "Correct prediction" if is_corr else "Incorrect prediction"

            self.banner_header.setText(f"CARD LEARNED — {outcome}")
            self.banner_header.setStyleSheet(f"color: {color}; font-size: 10px; font-weight: bold; letter-spacing: 0.8px;")
            self.banner_title.setText(cname.upper())
            self.banner_title.setStyleSheet(f"color: {color}; font-size: 16px; font-weight: 800;")
            self.banner_details.setText(
                f"Previous prediction: {prev_pred}\n"
                f"Outcome: {status_text}\n"
                f"Ground truth: Verified from Roblox green button\n"
                f"Blur sample: Saved  |  Clear artwork: Saved  |  Embedding: Generated  |  Database: Updated"
            )
            self.banner_frame.setStyleSheet(f"""
                QFrame {{
                    background: rgba(30, 41, 59, 0.95);
                    border: 2px solid {color};
                    border-radius: 8px;
                    padding: 8px;
                }}
            """)
        else:
            self.banner_title.setText("Zero Cards Learned (Cold Start)")
            self.banner_details.setText(
                "The database starts completely empty. When you play Roblox Card Clash, the assistant will predict, "
                "observe the game's revealed green button, and permanently learn every round."
            )

        # Update Recent Events Table
        recent = stats.get("recent_learning_events", [])
        self.events_table.setRowCount(len(recent))
        slot_names = ["TL", "TR", "BL", "BR"]
        for row_idx, ev in enumerate(recent):
            self.events_table.setItem(row_idx, 0, QTableWidgetItem(ev["session_id"]))
            self.events_table.setItem(row_idx, 1, QTableWidgetItem(ev["card_name"]))
            self.events_table.setItem(row_idx, 2, QTableWidgetItem(ev["previous_prediction"]))

            out_item = QTableWidgetItem(ev["outcome"])
            if ev["outcome"] == "CORRECT":
                out_item.setForeground(Qt.green)
            else:
                out_item.setForeground(Qt.yellow)
            self.events_table.setItem(row_idx, 3, out_item)

            act_slot = ev.get("actual_slot")
            slot_str = slot_names[act_slot] if act_slot is not None and 0 <= act_slot < 4 else "—"
            self.events_table.setItem(row_idx, 4, QTableWidgetItem(slot_str))

        # Update Gallery Table
        cards = self.db.get_all_cards()
        self.gallery_table.setRowCount(len(cards))
        for row_idx, c in enumerate(cards):
            self.gallery_table.setItem(row_idx, 0, QTableWidgetItem(str(c.card_id)))
            self.gallery_table.setItem(row_idx, 1, QTableWidgetItem(c.canonical_name))
            self.gallery_table.setItem(row_idx, 2, QTableWidgetItem(str(c.observation_count)))
            has_art = "✓ Saved" if c.clear_artwork_path and Path(c.clear_artwork_path).exists() else "—"
            self.gallery_table.setItem(row_idx, 3, QTableWidgetItem(has_art))

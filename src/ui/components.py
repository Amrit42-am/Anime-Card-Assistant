"""
src/ui/components.py

Custom UI widgets and dialogs for the Anime Card Assistant floating overlay.
Includes manual correction search dialog, confidence badges, and candidate score rows.
"""

from typing import List, Optional
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QLineEdit, QListWidget, QListWidgetItem
)


class ConfidenceBadge(QLabel):
    """Color-coded pill badge displaying recognition confidence."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setFixedHeight(22)
        self.set_confidence(0.0, "IDLE")

    def set_confidence(self, score: float, level: str):
        pct = int(score * 100)
        level_upper = level.upper()

        if level_upper == "HIGH":
            bg = "#1b5e20"
            border = "#4caf50"
            text = f"HIGH CONFIDENCE ({pct}%)"
        elif level_upper == "MEDIUM":
            bg = "#0d47a1"
            border = "#2196f3"
            text = f"MATCH ({pct}%)"
        elif level_upper == "UNCERTAIN":
            bg = "#e65100"
            border = "#ff9800"
            text = f"UNCERTAIN ({pct}%)"
        elif level_upper == "ERROR":
            bg = "#b71c1c"
            border = "#f44336"
            text = "ERROR"
        else:
            bg = "#263238"
            border = "#455a64"
            text = "WAITING"

        self.setText(text)
        self.setStyleSheet(f"""
            ConfidenceBadge {{
                background-color: {bg};
                color: #ffffff;
                border: 1px solid {border};
                border-radius: 11px;
                padding: 2px 10px;
                font-size: 11px;
                font-weight: bold;
            }}
        """)


class SourceBadge(QLabel):
    """Pill badge showing whether answer came from Learned Memory or Neural Model."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setFixedHeight(22)
        self.set_source("synthetic_blurs")

    def set_source(self, source: str):
        if source == "learned_memory":
            bg = "#4a148c"
            border = "#ab47bc"
            text = "★ LEARNED MEMORY"
        elif source == "fallback_all_cards":
            bg = "#37474f"
            border = "#78909c"
            text = "FULL SCAN FALLBACK"
        else:
            bg = "#004d40"
            border = "#26a69a"
            text = "DINOv2 MODEL"

        self.setText(text)
        self.setStyleSheet(f"""
            SourceBadge {{
                background-color: {bg};
                color: #ffffff;
                border: 1px solid {border};
                border-radius: 11px;
                padding: 2px 8px;
                font-size: 10px;
                font-weight: bold;
            }}
        """)


class CorrectionDialog(QDialog):
    """Searchable dialog for selecting the correct card name to train memory."""

    def __init__(self, all_card_names: List[str], current_guess: str = "", parent=None):
        super().__init__(parent)
        self.all_card_names = sorted(all_card_names)
        self.selected_name: Optional[str] = None

        self.setWindowTitle("Correct Card Identity")
        self.setFixedSize(380, 480)
        self.setStyleSheet("""
            QDialog { background-color: #1a1c24; color: white; font-family: -apple-system, sans-serif; }
            QLineEdit { background-color: #252936; border: 1px solid #3d4558; border-radius: 5px; padding: 8px; color: white; font-size: 13px; }
            QListWidget { background-color: #14161d; border: 1px solid #2d3342; border-radius: 5px; color: #e0e0e0; font-size: 13px; }
            QListWidget::item { padding: 6px 10px; border-bottom: 1px solid #1f232e; }
            QListWidget::item:selected { background-color: #1976d2; color: white; }
            QPushButton { background-color: #2b313f; border: 1px solid #4a556b; border-radius: 4px; padding: 8px; color: white; font-weight: bold; }
            QPushButton:hover { background-color: #3b4255; }
            QPushButton#confirmBtn { background-color: #2e7d32; border-color: #43a047; }
            QPushButton#confirmBtn:hover { background-color: #388e3c; }
        """)

        self.init_ui(current_guess)

    def init_ui(self, current_guess: str):
        layout = QVBoxLayout(self)

        title = QLabel("Select Correct Card Identity:")
        title.setStyleSheet("font-weight: bold; font-size: 14px; color: #90caf9;")
        layout.addWidget(title)

        if current_guess:
            sub = QLabel(f"Current Prediction: <b style='color:#ffab40;'>{current_guess}</b>")
            layout.addWidget(sub)

        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Type to filter 286 cards...")
        self.search_box.textChanged.connect(self.filter_list)
        layout.addWidget(self.search_box)

        self.list_widget = QListWidget()
        self.list_widget.itemDoubleClicked.connect(self.confirm_selection)
        layout.addWidget(self.list_widget)

        self.populate_list(self.all_card_names)

        btn_box = QHBoxLayout()
        btn_confirm = QPushButton("Confirm Correction")
        btn_confirm.setObjectName("confirmBtn")
        btn_confirm.clicked.connect(self.confirm_selection)
        btn_box.addWidget(btn_confirm)

        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btn_box.addWidget(btn_cancel)

        layout.addLayout(btn_box)

    def populate_list(self, names: List[str]):
        self.list_widget.clear()
        for name in names:
            item = QListWidgetItem(name)
            self.list_widget.addItem(item)
        if self.list_widget.count() > 0:
            self.list_widget.setCurrentRow(0)

    def filter_list(self, query: str):
        q = query.lower().strip()
        filtered = [n for n in self.all_card_names if q in n.lower()]
        self.populate_list(filtered)

    def confirm_selection(self):
        item = self.list_widget.currentItem()
        if item:
            self.selected_name = item.text()
            self.accept()


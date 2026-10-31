import sys
import os
import json
import html
import traceback
import hashlib
import hmac
import secrets
from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QLabel, QPushButton, QLineEdit,
    QTextEdit, QStackedWidget, QFrame, QMessageBox,
    QDialog, QComboBox
)

from ai_engine import LanganAI


APP_NAME = "Langan"
APP_VERSION = "0.4.0"

DATA_DIR = Path.home() / ".langan"
DATA_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_FILE = DATA_DIR / "config.json"
HISTORY_FILE = DATA_DIR / "history.json"

GREEN = "#13B86A"
DARK_GREEN = "#087A45"

DEFAULT_CONFIG = {
    "smol_model": "SmolLM2-360M-Instruct-Q5_0",
    "logic_prompt": (
        "Evaluate the premises carefully using the Logic Engine "
        "rules supplied by the administrator. Do not invent missing "
        "premises. If essential information is missing, respond with "
        "ENGINE_ACTION: REQUEST_PREMISES and explain what is missing. "
        "When sufficient information exists, provide a clear verdict "
        "and supporting reasoning."
    ),
    "planning_model": "Qwen2.5-0.5B-Instruct-Q4_K_M",
    "premise_prompt": (
        "Analyse the user's question and prepare clear, structured "
        "premises for the Logic Engine. Identify assumptions, relevant "
        "facts, possible missing information and questions that need "
        "research. Do not invent facts."
    ),
    "web_engine": "Tavily",
    "tavily_key": "",
    "website_url": "",
    "evidence_prompt": (
        "Explain the final Logic Engine verdict clearly. Present the "
        "reasoning, relevant evidence, assumptions and limitations. "
        "Do not change or reverse the verdict produced by the Logic "
        "Engine."
    )
}


def load_config():
    config = DEFAULT_CONFIG.copy()

    if CONFIG_FILE.exists():
        try:
            saved = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            if isinstance(saved, dict):
                config.update(saved)
        except (json.JSONDecodeError, OSError):
            pass

    return config


def save_config(config):
    temporary_file = CONFIG_FILE.with_suffix(".tmp")
    temporary_file.write_text(
        json.dumps(config, indent=4),
        encoding="utf-8"
    )
    temporary_file.replace(CONFIG_FILE)


def load_history():
    if HISTORY_FILE.exists():
        try:
            data = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data
        except (json.JSONDecodeError, OSError):
            pass
    return []


def save_history(history):
    temporary_file = HISTORY_FILE.with_suffix(".tmp")
    temporary_file.write_text(
        json.dumps(history, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    temporary_file.replace(HISTORY_FILE)


def password_digest(password, salt):
    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        300_000
    ).hex()


class AIWorker(QThread):
    finished_answer = Signal(str)
    failed = Signal(str)
    progress = Signal(str)

    def __init__(self, ai, message, config):
        super().__init__()
        self.ai = ai
        self.message = message
        self.config = config

    def run(self):
        try:
            answer = self.ai.generate_conversation(
                self.message,
                self.config,
                progress_callback=self.progress.emit
            )
            if not answer:
                answer = "The AI engine returned an empty response."
            self.finished_answer.emit(str(answer))
        except Exception as error:
            self.failed.emit(
                f"{error}\n\nTechnical details:\n{traceback.format_exc()}"
            )


class AdminLogin(QDialog):
    """First use creates a password; later visits verify it."""

    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config
        self.password_was_created = False

        self.environment_password = os.environ.get("LANGAN_ADMIN_PASSWORD")
        self.first_setup = (
            not self.environment_password
            and not self.config.get("admin_password_hash")
        )

        self.setWindowTitle("Langan - Admin Access")
        self.setFixedSize(400, 300 if self.first_setup else 245)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 25, 28, 25)
        layout.setSpacing(12)

        title = QLabel(
            "Create Admin Password" if self.first_setup else "Admin Access"
        )
        title.setObjectName("heading")
        layout.addWidget(title)

        description = QLabel(
            "Create a password to protect the Admin Control Center."
            if self.first_setup else
            "Enter your administrator password to continue."
        )
        description.setWordWrap(True)
        description.setObjectName("muted")
        layout.addWidget(description)

        self.password = QLineEdit()
        self.password.setPlaceholderText(
            "Create password" if self.first_setup else "Admin password"
        )
        self.password.setEchoMode(QLineEdit.EchoMode.Password)
        layout.addWidget(self.password)

        self.confirm_password = None
        if self.first_setup:
            self.confirm_password = QLineEdit()
            self.confirm_password.setPlaceholderText("Confirm password")
            self.confirm_password.setEchoMode(QLineEdit.EchoMode.Password)
            layout.addWidget(self.confirm_password)

        unlock = QPushButton(
            "Create Password" if self.first_setup else "Unlock Admin Mode"
        )
        unlock.clicked.connect(self.authenticate)
        layout.addWidget(unlock)
        self.password.returnPressed.connect(self.authenticate)

    def authenticate(self):
        entered = self.password.text()

        if self.first_setup:
            if len(entered) < 8:
                QMessageBox.warning(
                    self, "Password too short",
                    "Choose a password containing at least 8 characters."
                )
                return

            if entered != self.confirm_password.text():
                QMessageBox.warning(
                    self, "Passwords do not match",
                    "Enter the same password in both fields."
                )
                return

            salt = secrets.token_bytes(16)
            self.config["admin_password_salt"] = salt.hex()
            self.config["admin_password_hash"] = password_digest(
                entered, salt
            )
            try:
                save_config(self.config)
            except OSError as error:
                QMessageBox.critical(
                    self, "Could not save password", str(error)
                )
                return

            self.password_was_created = True
            self.accept()
            return

        if self.environment_password:
            valid = hmac.compare_digest(
                entered, self.environment_password
            )
        else:
            try:
                salt = bytes.fromhex(self.config["admin_password_salt"])
                expected = self.config["admin_password_hash"]
                actual = password_digest(entered, salt)
                valid = hmac.compare_digest(actual, expected)
            except (KeyError, ValueError, TypeError):
                valid = False

        if valid:
            self.accept()
        else:
            QMessageBox.warning(
                self, "Access denied",
                "Incorrect administrator password."
            )


class LanganWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle(f"{APP_NAME} - AI Assistant")
        self.resize(1200, 760)
        self.setMinimumSize(900, 600)

        self.config = load_config()
        self.history = load_history()
        self.admin_mode = False
        self.ai = LanganAI()
        self.worker = None

        # One active conversation is updated in place in saved history.
        self.current_conversation = []
        self.active_history_index = None

        self.build_interface()
        self.apply_theme()
        self.refresh_history()

    # MAIN INTERFACE

    def build_interface(self):
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(220)

        side_layout = QVBoxLayout(sidebar)
        side_layout.setContentsMargins(18, 25, 18, 20)
        side_layout.setSpacing(12)

        logo = QLabel("🌿 Langan")
        logo.setObjectName("logo")
        side_layout.addWidget(logo)

        subtitle = QLabel("THINK DEEPER. REASON BETTER.")
        subtitle.setObjectName("muted")
        side_layout.addWidget(subtitle)
        side_layout.addSpacing(30)

        self.chat_button = QPushButton("  Chat")
        self.history_button = QPushButton("  History")
        self.settings_button = QPushButton("  Settings")

        for button in (
            self.chat_button, self.history_button, self.settings_button
        ):
            button.setObjectName("nav")
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            side_layout.addWidget(button)

        self.chat_button.clicked.connect(lambda: self.show_page(0))
        self.history_button.clicked.connect(lambda: self.show_page(1))
        self.settings_button.clicked.connect(self.open_settings)

        side_layout.addStretch()
        version = QLabel(f"LANGAN v{APP_VERSION}")
        version.setObjectName("muted")
        side_layout.addWidget(version)
        main_layout.addWidget(sidebar)

        self.pages = QStackedWidget()
        main_layout.addWidget(self.pages, 1)
        self.pages.addWidget(self.create_chat_page())
        self.pages.addWidget(self.create_history_page())
        self.pages.addWidget(self.create_settings_page())
        self.pages.addWidget(self.create_admin_page())

    def show_page(self, index):
        self.pages.setCurrentIndex(index)

    # CHAT

    def create_chat_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(35, 30, 35, 25)
        layout.setSpacing(15)

        header = QHBoxLayout()
        heading = QLabel("Langan Assistant")
        heading.setObjectName("heading")
        header.addWidget(heading)
        header.addStretch()

        self.new_chat_button = QPushButton("＋ New Conversation")
        self.new_chat_button.setObjectName("secondary")
        self.new_chat_button.clicked.connect(self.start_new_conversation)
        header.addWidget(self.new_chat_button)

        self.status_label = QLabel("● Ready")
        self.status_label.setObjectName("status")
        header.addWidget(self.status_label)
        layout.addLayout(header)

        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setObjectName("chat")
        self.chat_display.setPlaceholderText(
            "Your conversation will appear here."
        )
        layout.addWidget(self.chat_display, 1)

        self.progress_label = QLabel(
            "Qwen plans. Smol evaluates. Langan explains."
        )
        self.progress_label.setObjectName("muted")
        layout.addWidget(self.progress_label)

        input_row = QHBoxLayout()
        self.message_input = QLineEdit()
        self.message_input.setPlaceholderText("Message Langan...")
        self.message_input.returnPressed.connect(self.send_message)
        input_row.addWidget(self.message_input, 1)

        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.send_message)
        input_row.addWidget(self.send_button)
        layout.addLayout(input_row)
        return page

    def append_message(self, speaker, message, color):
        safe_message = html.escape(str(message)).replace("\n", "<br>")
        self.chat_display.append(
            f"<b style='color:{color}'>{speaker}:</b> {safe_message}"
        )

    def build_context_message(self, latest_message):
        """Give the engine recent turns because its API accepts one string."""
        if not self.current_conversation:
            return latest_message

        recent = self.current_conversation[-8:]
        lines = [
            "The following is the recent conversation history. "
            "Use it only as context for the latest user message."
        ]
        for item in recent:
            role = item.get("role", "unknown").upper()
            content = str(item.get("content", ""))[-1200:]
            lines.append(f"{role}: {content}")

        lines.append(f"CURRENT USER MESSAGE: {latest_message}")
        context = "\n\n".join(lines)
        return context[-6500:]

    def send_message(self):
        if self.worker is not None and self.worker.isRunning():
            return

        message = self.message_input.text().strip()
        if not message:
            return

        self.append_message("You", message, DARK_GREEN)
        self.message_input.clear()
        self.current_conversation.append({
            "role": "user", "content": message
        })

        self.set_busy(True)
        self.status_label.setText("● Working")
        self.progress_label.setText("Starting the reasoning process...")

        engine_message = self.build_context_message(message)
        self.worker = AIWorker(
            self.ai, engine_message, self.config.copy()
        )
        self.worker.progress.connect(self.update_progress)
        self.worker.finished_answer.connect(self.on_answer_received)
        self.worker.failed.connect(self.on_ai_error)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.start()

    def set_busy(self, busy):
        self.send_button.setEnabled(not busy)
        self.message_input.setEnabled(not busy)
        self.new_chat_button.setEnabled(not busy)
        self.send_button.setText("Working..." if busy else "Send")

    def update_progress(self, message):
        self.progress_label.setText(str(message))

    def save_active_conversation(self):
        if not self.current_conversation:
            return

        record = {
            "title": self.current_conversation[0]["content"][:60],
            "messages": self.current_conversation.copy()
        }

        if self.active_history_index is None:
            self.history.append(record)
            self.active_history_index = len(self.history) - 1
        else:
            self.history[self.active_history_index] = record

        try:
            save_history(self.history)
        except OSError as error:
            QMessageBox.warning(
                self, "History Save Error",
                f"Your conversation is still visible, but could not be saved:\n{error}"
            )

        self.refresh_history()

    def on_answer_received(self, answer):
        self.append_message("Langan", answer, DARK_GREEN)
        self.current_conversation.append({
            "role": "assistant", "content": answer
        })
        self.save_active_conversation()
        self.status_label.setText("● Ready")
        self.progress_label.setText("Reasoning complete.")

    def on_ai_error(self, error):
        self.append_message(
            "Langan Error",
            "The AI engine could not complete this request. "
            "Check the model files and configuration.",
            "#B3261E"
        )
        QMessageBox.critical(self, "AI Engine Error", str(error))
        self.status_label.setText("● Error")
        self.progress_label.setText("The request could not be completed.")

    def on_worker_finished(self):
        self.set_busy(False)
        if self.status_label.text() != "● Error":
            self.status_label.setText("● Ready")
        self.worker = None

    def start_new_conversation(self):
        if self.worker is not None and self.worker.isRunning():
            return

        if self.current_conversation and self.active_history_index is None:
            # Preserve an unfinished conversation if it contains user input.
            self.save_active_conversation()

        self.current_conversation = []
        self.active_history_index = None
        self.chat_display.clear()
        self.progress_label.setText(
            "New conversation. Qwen plans. Smol evaluates. Langan explains."
        )
        self.status_label.setText("● Ready")
        self.show_page(0)

    # HISTORY

    def create_history_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(35, 30, 35, 25)

        title = QLabel("Conversation History")
        title.setObjectName("heading")
        layout.addWidget(title)

        description = QLabel(
            "Your previous conversations are saved on this computer."
        )
        description.setObjectName("muted")
        layout.addWidget(description)

        self.history_display = QTextEdit()
        self.history_display.setReadOnly(True)
        layout.addWidget(self.history_display, 1)

        clear_button = QPushButton("Clear History")
        clear_button.setObjectName("secondary")
        clear_button.clicked.connect(self.clear_history)
        layout.addWidget(clear_button)
        return page

    def refresh_history(self):
        if not hasattr(self, "history_display"):
            return

        self.history_display.clear()
        if not self.history:
            self.history_display.setPlainText("No saved conversations yet.")
            return

        for number, conversation in enumerate(reversed(self.history), start=1):
            title = conversation.get("title", "Conversation")
            self.history_display.append(f"{number}. {title}")
            for message in conversation.get("messages", []):
                role = message.get("role", "unknown").title()
                content = message.get("content", "")
                self.history_display.append(f"   {role}: {content}")
            self.history_display.append("")

    def clear_history(self):
        answer = QMessageBox.question(
            self, "Clear History",
            "Are you sure you want to delete all saved conversations?"
        )
        if answer != QMessageBox.StandardButton.Yes:
            return

        self.history = []
        self.active_history_index = None
        try:
            save_history(self.history)
        except OSError as error:
            QMessageBox.warning(self, "History Error", str(error))
        self.refresh_history()

    # SETTINGS

    def create_settings_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(40, 35, 40, 35)

        title = QLabel("Settings")
        title.setObjectName("heading")
        layout.addWidget(title)

        description = QLabel("Configure your Langan experience.")
        description.setObjectName("muted")
        layout.addWidget(description)
        layout.addSpacing(25)

        admin_button = QPushButton("🔐  Enter Admin Mode")
        admin_button.setMinimumHeight(55)
        admin_button.clicked.connect(self.open_admin_login)
        layout.addWidget(admin_button)
        layout.addStretch()
        return page

    def open_settings(self):
        self.show_page(2)

    def open_admin_login(self):
        dialog = AdminLogin(self.config, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.admin_mode = True
            self.show_page(3)

    # ADMIN CONTROL CENTER

    def create_admin_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(30, 25, 30, 25)

        heading = QHBoxLayout()
        title = QLabel("Admin Control Center")
        title.setObjectName("heading")
        heading.addWidget(title)
        heading.addStretch()

        badge = QLabel("● ADMIN MODE")
        badge.setObjectName("status")
        heading.addWidget(badge)
        layout.addLayout(heading)

        self.admin_tabs = QComboBox()
        self.admin_tabs.addItems([
            "Dashboard", "Logic", "Planning",
            "Premise Prompt", "Web Engine", "Evidence Prompt"
        ])
        self.admin_tabs.currentIndexChanged.connect(self.change_admin_tab)
        layout.addWidget(self.admin_tabs)

        self.admin_stack = QStackedWidget()
        self.admin_stack.addWidget(self.admin_dashboard())
        self.admin_stack.addWidget(self.logic_page())
        self.admin_stack.addWidget(self.planning_page())
        self.admin_stack.addWidget(
            self.prompt_page("Premise Generator", "premise_prompt")
        )
        self.admin_stack.addWidget(self.web_page())
        self.admin_stack.addWidget(
            self.prompt_page("Evidence Verification", "evidence_prompt")
        )
        layout.addWidget(self.admin_stack, 1)

        buttons = QHBoxLayout()
        buttons.addStretch()

        logout = QPushButton("Exit Admin Mode")
        logout.setObjectName("secondary")
        logout.clicked.connect(self.exit_admin)
        buttons.addWidget(logout)

        save = QPushButton("Save Configuration")
        save.clicked.connect(self.save_settings)
        buttons.addWidget(save)
        layout.addLayout(buttons)
        return page

    def change_admin_tab(self, index):
        self.admin_stack.setCurrentIndex(index)

    def section_title(self, title, description):
        container = QWidget()
        layout = QVBoxLayout(container)
        heading = QLabel(title)
        heading.setObjectName("section")
        layout.addWidget(heading)
        subtitle = QLabel(description)
        subtitle.setObjectName("muted")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)
        return container

    def model_field(self, label, current=""):
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.addWidget(QLabel(label))
        field = QLineEdit(current)
        field.setPlaceholderText("Model name or local model filename")
        layout.addWidget(field)
        return container, field

    def admin_dashboard(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(16)

        title = QLabel("Welcome to Langan Admin")
        title.setObjectName("heading")
        layout.addWidget(title)

        description = QLabel(
            "Manage the intelligence and configuration behind Langan."
        )
        description.setObjectName("muted")
        description.setWordWrap(True)
        layout.addWidget(description)

        sections = [
            ("Logic", "Manage Smol and the logic engine.", 1),
            ("Planning", "Configure the Qwen planning model.", 2),
            ("Premise Prompt", "Control premise generation.", 3),
            ("Web Engine", "Configure information retrieval.", 4),
            ("Evidence Prompt", "Configure final explanations.", 5),
        ]
        for name, description, index in sections:
            button = QPushButton(f"{name}  —  {description}")
            button.setMinimumHeight(55)
            button.clicked.connect(
                lambda checked=False, i=index:
                self.admin_tabs.setCurrentIndex(i)
            )
            layout.addWidget(button)

        layout.addStretch()
        return page

    def logic_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(self.section_title(
            "Logic",
            "Manage Smol and the Logic Engine instructions."
        ))

        model, self.smol_model = self.model_field(
            "Smol Model", self.config["smol_model"]
        )
        layout.addWidget(model)
        layout.addWidget(QLabel("Logic Engine Rules and Instructions"))

        self.logic_prompt = QTextEdit()
        self.logic_prompt.setPlainText(self.config["logic_prompt"])
        self.logic_prompt.setPlaceholderText(
            "Paste your Logic Engine rules and instructions here."
        )
        layout.addWidget(self.logic_prompt, 1)

        note = QLabel(
            "Rules are stored in your local configuration and are not "
            "hardcoded into app.py. Model selection will take effect once "
            "the matching model-path settings are connected in ai_engine.py."
        )
        note.setObjectName("muted")
        note.setWordWrap(True)
        layout.addWidget(note)
        return page

    def planning_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(self.section_title(
            "Planning", "Configure the Qwen planning model."
        ))

        model, self.planning_model = self.model_field(
            "Qwen Planning Model", self.config["planning_model"]
        )
        layout.addWidget(model)

        info = QLabel(
            "Qwen prepares premises and additional information. "
            "The model name is saved here; ai_engine.py must also use "
            "this setting for model selection to change."
        )
        info.setObjectName("muted")
        info.setWordWrap(True)
        layout.addWidget(info)
        layout.addStretch()
        return page

    def prompt_page(self, title, key):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(self.section_title(
            title, "Edit the instructions used by this engine."
        ))

        editor = QTextEdit()
        editor.setPlainText(self.config[key])
        layout.addWidget(editor, 1)

        if key == "premise_prompt":
            self.premise_prompt = editor
        else:
            self.evidence_prompt = editor
        return page

    def web_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(self.section_title(
            "Web Engine", "Configure external information retrieval."
        ))

        self.web_method = QComboBox()
        self.web_method.addItems(["Tavily", "Website URL"])
        index = self.web_method.findText(self.config["web_engine"])
        if index >= 0:
            self.web_method.setCurrentIndex(index)

        layout.addWidget(QLabel("Retrieval method"))
        layout.addWidget(self.web_method)
        layout.addWidget(QLabel("Tavily API Key"))

        self.tavily_key = QLineEdit(self.config["tavily_key"])
        self.tavily_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.tavily_key.setPlaceholderText("Enter your Tavily API key")
        layout.addWidget(self.tavily_key)

        layout.addWidget(QLabel("Website URL"))
        self.website_url = QLineEdit(self.config["website_url"])
        self.website_url.setPlaceholderText("https://example.com")
        layout.addWidget(self.website_url)

        warning = QLabel(
            "API credentials are stored locally. Do not share your "
            "Langan configuration file."
        )
        warning.setObjectName("muted")
        warning.setWordWrap(True)
        layout.addWidget(warning)
        layout.addStretch()
        return page

    def save_settings(self):
        self.config.update({
            "smol_model": self.smol_model.text().strip(),
            "logic_prompt": self.logic_prompt.toPlainText(),
            "planning_model": self.planning_model.text().strip(),
            "premise_prompt": self.premise_prompt.toPlainText(),
            "web_engine": self.web_method.currentText(),
            "tavily_key": self.tavily_key.text().strip(),
            "website_url": self.website_url.text().strip(),
            "evidence_prompt": self.evidence_prompt.toPlainText()
        })

        try:
            save_config(self.config)
            QMessageBox.information(
                self, "Langan", "Configuration saved successfully."
            )
        except OSError as error:
            QMessageBox.critical(self, "Save failed", str(error))

    def exit_admin(self):
        self.admin_mode = False
        self.admin_tabs.setCurrentIndex(0)
        self.show_page(2)

    # THEME

    def apply_theme(self):
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background: #F3F8F4;
                color: #18352A;
                font-family: 'Segoe UI';
                font-size: 14px;
            }
            #sidebar {
                background: #FFFFFF;
                border-right: 1px solid #DCE9DF;
            }
            #logo {
                font-size: 25px;
                font-weight: bold;
                color: #087A45;
            }
            #heading {
                font-size: 25px;
                font-weight: bold;
                color: #087A45;
            }
            #section {
                font-size: 21px;
                font-weight: bold;
                color: #087A45;
            }
            #muted { color: #71847A; }
            #status {
                color: #0BA65A;
                font-weight: bold;
            }
            QPushButton {
                background: #13B86A;
                color: white;
                border: none;
                border-radius: 9px;
                padding: 11px 17px;
                font-weight: bold;
            }
            QPushButton:hover { background: #087A45; }
            QPushButton:disabled {
                background: #A8CDB7;
                color: #F5FFF8;
            }
            QPushButton#nav {
                background: transparent;
                color: #18352A;
                text-align: left;
                padding: 13px;
            }
            QPushButton#nav:hover {
                background: #E4F5E9;
                color: #087A45;
            }
            QPushButton#secondary {
                background: #E1EEE5;
                color: #087A45;
            }
            QLineEdit, QTextEdit, QComboBox {
                background: white;
                border: 1px solid #C9DED0;
                border-radius: 9px;
                padding: 11px;
                selection-background-color: #13B86A;
            }
            QTextEdit#chat {
                border: 1px solid #DCE9DF;
                border-radius: 12px;
                padding: 15px;
            }
            QComboBox { min-height: 30px; }
        """)

    # CLEAN SHUTDOWN

    def closeEvent(self, event):
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.warning(
                self, "Langan is busy",
                "Please wait for the current AI response to finish "
                "before closing Langan."
            )
            event.ignore()
            return

        try:
            if self.current_conversation and self.active_history_index is None:
                self.save_active_conversation()
            self.ai.close()
        except Exception:
            pass

        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setFont(QFont("Segoe UI", 10))

    window = LanganWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

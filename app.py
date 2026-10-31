
import sys
import os
import json
import html
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
APP_VERSION = "0.2.0"

DATA_DIR = Path.home() / ".langan"
DATA_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_FILE = DATA_DIR / "config.json"

GREEN = "#13B86A"
DARK_GREEN = "#087A45"
BACKGROUND = "#F3F8F4"
WHITE = "#FFFFFF"
TEXT = "#18352A"
MUTED = "#71847A"


def load_config():
    defaults = {
        "smol_model": "SmolLM2-360M-Instruct-Q5_0",
        "logic_prompt": (
            "You are Langan's logic engine. "
            "Think carefully, prioritize accuracy, "
            "explain conclusions clearly and do not "
            "invent facts."
        ),
        "planning_model": "Qwen2.5-0.5B-Instruct-Q4_K_M",
        "premise_prompt": (
            "Generate 60 clear, logically structured premises."
        ),
        "web_engine": "Tavily",
        "tavily_key": "",
        "website_url": "",
        "evidence_prompt": (
            "Check whether an analogy is structurally valid. "
            "Identify similarities, differences and evidence."
        )
    }

    if CONFIG_FILE.exists():
        try:
            saved = json.loads(
                CONFIG_FILE.read_text(encoding="utf-8")
            )
            defaults.update(saved)
        except (json.JSONDecodeError, OSError):
            pass

    return defaults


def save_config(config):
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)

    CONFIG_FILE.write_text(
        json.dumps(config, indent=4),
        encoding="utf-8"
    )


class AdminLogin(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Langan - Admin Access")
        self.setFixedSize(380, 230)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 25, 28, 25)
        layout.setSpacing(14)

        title = QLabel("Admin Access")
        title.setObjectName("heading")
        layout.addWidget(title)

        description = QLabel(
            "Enter your administrator password to continue."
        )
        description.setWordWrap(True)
        description.setObjectName("muted")
        layout.addWidget(description)

        self.password = QLineEdit()
        self.password.setPlaceholderText("Admin password")
        self.password.setEchoMode(
            QLineEdit.EchoMode.Password
        )
        layout.addWidget(self.password)

        unlock = QPushButton("Unlock Admin Mode")
        unlock.clicked.connect(self.authenticate)
        layout.addWidget(unlock)

        self.password.returnPressed.connect(self.authenticate)

    def authenticate(self):
        expected = os.environ.get(
            "LANGAN_ADMIN_PASSWORD",
            "langan-dev"
        )

        if self.password.text() == expected:
            self.accept()
        else:
            QMessageBox.warning(
                self,
                "Access denied",
                "Incorrect administrator password."
            )


class ChatWorker(QThread):
    response_ready = Signal(str)
    error_occurred = Signal(str)

    def __init__(self, ai, message, system_prompt):
        super().__init__()

        self.ai = ai
        self.message = message
        self.system_prompt = system_prompt

    def run(self):
        try:
            response = self.ai.generate(
                self.message,
                role="logic",
                system_prompt=self.system_prompt
            )

            self.response_ready.emit(response)

        except Exception as error:
            self.error_occurred.emit(str(error))


class LanganWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle(
            f"{APP_NAME} - AI Assistant"
        )

        self.resize(1200, 760)
        self.setMinimumSize(900, 600)

        self.config = load_config()
        self.admin_mode = False

        self.ai = LanganAI()
        self.chat_worker = None

        self.build_interface()
        self.apply_theme()

    # -----------------------------------------
    # MAIN INTERFACE
    # -----------------------------------------

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
            self.chat_button,
            self.history_button,
            self.settings_button
        ):
            button.setObjectName("nav")
            button.setCursor(
                Qt.CursorShape.PointingHandCursor
            )
            side_layout.addWidget(button)

        self.chat_button.clicked.connect(
            lambda: self.show_page(0)
        )

        self.history_button.clicked.connect(
            lambda: self.show_page(1)
        )

        self.settings_button.clicked.connect(
            self.open_settings
        )

        side_layout.addStretch()

        version = QLabel(
            f"LANGAN v{APP_VERSION}"
        )
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

    # -----------------------------------------
    # CHAT PAGE
    # -----------------------------------------

    def create_chat_page(self):
        page = QWidget()

        layout = QVBoxLayout(page)
        layout.setContentsMargins(45, 35, 45, 25)

        header = QHBoxLayout()

        heading = QLabel("Langan Assistant")
        heading.setObjectName("heading")
        header.addWidget(heading)

        header.addStretch()

        status = QLabel("● AI Ready")
        status.setObjectName("status")
        header.addWidget(status)

        layout.addLayout(header)

        welcome = QLabel("Welcome to Langan")
        welcome.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )
        welcome.setObjectName("welcome")
        layout.addWidget(welcome)

        description = QLabel(
            "Think deeper. Find better answers.\n"
            "Ask questions, explore ideas and examine evidence."
        )

        description.setAlignment(
            Qt.AlignmentFlag.AlignCenter
        )
        description.setObjectName("muted")
        layout.addWidget(description)

        layout.addSpacing(20)

        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setObjectName("chat")
        self.chat_display.setPlaceholderText(
            "Your conversation will appear here."
        )

        layout.addWidget(self.chat_display, 1)

        input_row = QHBoxLayout()

        self.message_input = QLineEdit()
        self.message_input.setPlaceholderText(
            "Message Langan..."
        )

        self.message_input.returnPressed.connect(
            self.send_message
        )

        input_row.addWidget(self.message_input)

        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(
            self.send_message
        )

        input_row.addWidget(self.send_button)

        layout.addLayout(input_row)

        return page

    def send_message(self):
        message = self.message_input.text().strip()

        if not message:
            return

        if self.chat_worker is not None:
            return

        safe_message = html.escape(message)

        self.chat_display.append(
            f"<b style='color:{DARK_GREEN}'>You:</b> "
            f"{safe_message}"
        )

        self.message_input.clear()
        self.message_input.setEnabled(False)
        self.send_button.setEnabled(False)

        self.chat_display.append(
            "<b style='color:#71847A'>Langan:</b> "
            "Thinking..."
        )

        system_prompt = self.config.get(
            "logic_prompt",
            "You are Langan, a helpful AI assistant."
        )

        self.chat_worker = ChatWorker(
            self.ai,
            message,
            system_prompt
        )

        self.chat_worker.response_ready.connect(
            self.handle_ai_response
        )

        self.chat_worker.error_occurred.connect(
            self.handle_ai_error
        )

        self.chat_worker.finished.connect(
            self.chat_worker_finished
        )

        self.chat_worker.start()

    def handle_ai_response(self, response):
        safe_response = html.escape(response)
        safe_response = safe_response.replace(
            "\n", "<br>"
        )

        self.chat_display.append(
            f"<b style='color:{DARK_GREEN}'>Langan:</b> "
            f"{safe_response}"
        )

        self.chat_display.moveCursor(
            self.chat_display.textCursor().MoveOperation.End
        )

    def handle_ai_error(self, error):
        self.chat_display.append(
            "<b style='color:#B00020'>Langan error:</b> "
            f"{html.escape(error)}"
        )

    def chat_worker_finished(self):
        worker = self.chat_worker
        self.chat_worker = None

        self.message_input.setEnabled(True)
        self.send_button.setEnabled(True)
        self.message_input.setFocus()

        if worker is not None:
            worker.deleteLater()

    # -----------------------------------------
    # HISTORY PAGE
    # -----------------------------------------

    def create_history_page(self):
        page = QWidget()

        layout = QVBoxLayout(page)

        title = QLabel("Conversation History")
        title.setObjectName("heading")
        layout.addWidget(title)

        info = QLabel(
            "Your saved conversations will appear here."
        )
        info.setObjectName("muted")
        layout.addWidget(info)

        layout.addStretch()

        return page

    # -----------------------------------------
    # SETTINGS PAGE
    # -----------------------------------------

    def create_settings_page(self):
        page = QWidget()

        layout = QVBoxLayout(page)
        layout.setContentsMargins(40, 35, 40, 35)

        title = QLabel("Settings")
        title.setObjectName("heading")
        layout.addWidget(title)

        layout.addSpacing(20)

        description = QLabel(
            "Configure your Langan experience."
        )
        description.setObjectName("muted")
        layout.addWidget(description)

        layout.addSpacing(25)

        admin_button = QPushButton(
            "🔐  Enter Admin Mode"
        )

        admin_button.setMinimumHeight(55)
        admin_button.clicked.connect(
            self.open_admin_login
        )

        layout.addWidget(admin_button)

        layout.addStretch()

        return page

    def open_settings(self):
        self.show_page(2)

    def open_admin_login(self):
        dialog = AdminLogin(self)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.admin_mode = True
            self.show_page(3)

    # -----------------------------------------
    # ADMIN PAGE
    # -----------------------------------------

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
            "Dashboard",
            "Logic",
            "Planning",
            "Premise Prompt",
            "Web Engine",
            "Evidence Prompt"
        ])

        self.admin_tabs.currentIndexChanged.connect(
            self.change_admin_tab
        )

        layout.addWidget(self.admin_tabs)

        self.admin_stack = QStackedWidget()

        self.admin_stack.addWidget(
            self.admin_dashboard()
        )

        self.admin_stack.addWidget(
            self.logic_page()
        )

        self.admin_stack.addWidget(
            self.planning_page()
        )

        self.admin_stack.addWidget(
            self.prompt_page(
                "Premise Generator",
                "premise_prompt"
            )
        )

        self.admin_stack.addWidget(
            self.web_page()
        )

        self.admin_stack.addWidget(
            self.prompt_page(
                "Evidence Verification",
                "evidence_prompt"
            )
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

        heading = QLabel(label)
        layout.addWidget(heading)

        field = QLineEdit(current)
        field.setPlaceholderText(
            "Enter Hugging Face model name"
        )

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
            "Manage the intelligence and configuration "
            "behind your Langan AI assistant."
        )

        description.setObjectName("muted")
        description.setWordWrap(True)
        layout.addWidget(description)

        sections = [
            ("Logic", "Manage Smol and the logic engine.", 1),
            ("Planning", "Configure the Qwen planning model.", 2),
            ("Premise Prompt", "Control premise generation.", 3),
            ("Web Engine", "Configure web information retrieval.", 4),
            ("Evidence Prompt", "Configure analogy verification.", 5)
        ]

        for name, description, index in sections:
            button = QPushButton(
                f"{name}  —  {description}"
            )

            button.setMinimumHeight(55)

            button.clicked.connect(
                lambda checked=False, i=index:
                    self.admin_tabs.setCurrentIndex(i)
            )

            layout.addWidget(button)

        layout.addStretch()

        return page

    # -----------------------------------------
    # LOGIC SETTINGS
    # -----------------------------------------

    def logic_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        layout.addWidget(
            self.section_title(
                "Logic",
                "Manage the Smol model and its reasoning engine."
            )
        )

        model, self.smol_model = self.model_field(
            "Smol Model",
            self.config.get("smol_model", "")
        )

        layout.addWidget(model)

        layout.addWidget(
            QLabel("Logic Engine Prompt")
        )

        self.logic_prompt = QTextEdit()

        self.logic_prompt.setPlainText(
            self.config.get("logic_prompt", "")
        )

        layout.addWidget(self.logic_prompt)

        note = QLabel(
            "The logic engine instructions guide the Smol model."
        )

        note.setObjectName("muted")
        layout.addWidget(note)

        return page

    # -----------------------------------------
    # PLANNING SETTINGS
    # -----------------------------------------

    def planning_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        layout.addWidget(
            self.section_title(
                "Planning",
                "Configure the Qwen planning model."
            )
        )

        model, self.planning_model = self.model_field(
            "Qwen Planning Model",
            self.config.get("planning_model", "")
        )

        layout.addWidget(model)

        info = QLabel(
            "The planning model is configured here. "
            "Its activation and model switching will be "
            "connected in a later development stage."
        )

        info.setObjectName("muted")
        info.setWordWrap(True)

        layout.addWidget(info)
        layout.addStretch()

        return page

    # -----------------------------------------
    # PROMPT SETTINGS
    # -----------------------------------------

    def prompt_page(self, title, key):
        page = QWidget()
        layout = QVBoxLayout(page)

        layout.addWidget(
            self.section_title(
                title,
                "Edit the instructions used by this engine."
            )
        )

        editor = QTextEdit()

        editor.setPlainText(
            self.config.get(key, "")
        )

        layout.addWidget(editor)

        if key == "premise_prompt":
            self.premise_prompt = editor
        else:
            self.evidence_prompt = editor

        return page

    # -----------------------------------------
    # WEB ENGINE SETTINGS
    # -----------------------------------------

    def web_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        layout.addWidget(
            self.section_title(
                "Web Engine",
                "Configure external information retrieval."
            )
        )

        self.web_method = QComboBox()

        self.web_method.addItems([
            "Tavily",
            "Website URL"
        ])

        current = self.config.get(
            "web_engine",
            "Tavily"
        )

        index = self.web_method.findText(current)

        if index >= 0:
            self.web_method.setCurrentIndex(index)

        layout.addWidget(
            QLabel("Retrieval method")
        )

        layout.addWidget(self.web_method)

        layout.addWidget(
            QLabel("Tavily API Key")
        )

        self.tavily_key = QLineEdit(
            self.config.get("tavily_key", "")
        )

        self.tavily_key.setEchoMode(
            QLineEdit.EchoMode.Password
        )

        self.tavily_key.setPlaceholderText(
            "Enter your Tavily API key"
        )

        layout.addWidget(self.tavily_key)

        layout.addWidget(
            QLabel("Website URL")
        )

        self.website_url = QLineEdit(
            self.config.get("website_url", "")
        )

        self.website_url.setPlaceholderText(
            "https://example.com"
        )

        layout.addWidget(self.website_url)

        warning = QLabel(
            "Web retrieval is not connected yet. "
            "API credentials are currently stored in "
            "local configuration."
        )

        warning.setObjectName("muted")
        warning.setWordWrap(True)

        layout.addWidget(warning)
        layout.addStretch()

        return page

    # -----------------------------------------
    # SAVE SETTINGS
    # -----------------------------------------

    def save_settings(self):
        self.config.update({
            "smol_model": self.smol_model.text().strip(),
            "logic_prompt": self.logic_prompt.toPlainText(),
            "planning_model": self.planning_model.text().strip(),
            "premise_prompt": self.premise_prompt.toPlainText(),
            "web_engine": self.web_method.currentText(),
            "tavily_key": self.tavily_key.text(),
            "website_url": self.website_url.text(),
            "evidence_prompt": self.evidence_prompt.toPlainText()
        })

        try:
            save_config(self.config)

            QMessageBox.information(
                self,
                "Langan",
                "Configuration saved successfully."
            )

        except OSError as error:
            QMessageBox.critical(
                self,
                "Save failed",
                str(error)
            )

    # -----------------------------------------
    # EXIT ADMIN MODE
    # -----------------------------------------

    def exit_admin(self):
        self.admin_mode = False
        self.admin_tabs.setCurrentIndex(0)
        self.show_page(2)

    # -----------------------------------------
    # APPLICATION THEME
    # -----------------------------------------

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

            #welcome {
                font-size: 32px;
                font-weight: bold;
                color: #087A45;
            }

            #section {
                font-size: 21px;
                font-weight: bold;
                color: #087A45;
            }

            #muted {
                color: #71847A;
            }

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

            QPushButton:hover {
                background: #087A45;
            }

            QPushButton:disabled {
                background: #A8CDB6;
                color: #F3F8F4;
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

            QComboBox {
                min-height: 30px;
            }
        """)

    # -----------------------------------------
    # SAFE APPLICATION CLOSING
    # -----------------------------------------

    def closeEvent(self, event):
        if self.chat_worker is not None:
            if self.chat_worker.isRunning():
                self.chat_worker.wait()

        self.ai.unload_model()
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

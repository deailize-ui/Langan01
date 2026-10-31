
import sys
import os
import json
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QLabel, QPushButton, QLineEdit,
    QTextEdit, QStackedWidget, QFrame, QMessageBox,
    QDialog, QFormLayout, QComboBox, QScrollArea
)


APP_NAME = "Langan"
APP_VERSION = "0.1.0"

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
    if CONFIG_FILE.exists():
        try:
            return json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass

    return {
        "smol_model": "",
        "logic_prompt": "Follow logical reasoning. Prioritize accuracy and explain conclusions clearly.",
        "planning_model": "",
        "premise_prompt": "Generate 60 clear, logically structured premises.",
        "web_engine": "Tavily",
        "tavily_key": "",
        "website_url": "",
        "evidence_prompt": "Check whether an analogy is structurally valid. Identify similarities, differences and supporting evidence."
    }


def save_config(config):
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
        self.password.setEchoMode(QLineEdit.EchoMode.Password)
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


class LanganWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle(f"{APP_NAME} - AI Assistant")
        self.resize(1200, 760)
        self.setMinimumSize(900, 600)

        self.config = load_config()
        self.admin_mode = False

        self.build_interface()
        self.apply_theme()

    def build_interface(self):
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Sidebar
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
            button.setCursor(Qt.CursorShape.PointingHandCursor)
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

        version = QLabel(f"LANGAN v{APP_VERSION}")
        version.setObjectName("muted")
        side_layout.addWidget(version)

        main_layout.addWidget(sidebar)

        # Main content
        self.pages = QStackedWidget()
        main_layout.addWidget(self.pages, 1)

        self.pages.addWidget(self.create_chat_page())
        self.pages.addWidget(self.create_history_page())
        self.pages.addWidget(self.create_settings_page())
        self.pages.addWidget(self.create_admin_page())

    def show_page(self, index):
        self.pages.setCurrentIndex(index)

    def create_chat_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(45, 35, 45, 25)

        header = QHBoxLayout()

        heading = QLabel("Langan Assistant")
        heading.setObjectName("heading")
        header.addWidget(heading)

        header.addStretch()

        status = QLabel("● Online")
        status.setObjectName("status")
        header.addWidget(status)

        layout.addLayout(header)

        layout.addStretch()

        welcome = QLabel("Welcome to Langan")
        welcome.setAlignment(Qt.AlignmentFlag.AlignCenter)
        welcome.setObjectName("welcome")
        layout.addWidget(welcome)

        description = QLabel(
            "Think deeper. Find better answers.\n"
            "Ask questions, explore ideas and examine evidence."
        )
        description.setAlignment(Qt.AlignmentFlag.AlignCenter)
        description.setObjectName("muted")
        layout.addWidget(description)

        layout.addSpacing(30)

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

        send = QPushButton("Send")
        send.clicked.connect(self.send_message)
        input_row.addWidget(send)

        layout.addLayout(input_row)

        return page

    def send_message(self):
        message = self.message_input.text().strip()

        if not message:
            return

        self.chat_display.append(
            f"<b style='color:{DARK_GREEN}'>You:</b> "
            f"{message}"
        )

        self.chat_display.append(
            "<b style='color:#71847A'>Langan:</b> "
            "The AI engine will be connected in the next "
            "development stage."
        )

        self.message_input.clear()

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
        admin_button.clicked.connect(self.open_admin_login)
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

        self.admin_stack.addWidget(self.admin_dashboard())
        self.admin_stack.addWidget(self.logic_page())
        self.admin_stack.addWidget(self.planning_page())
        self.admin_stack.addWidget(
            self.prompt_page(
                "Premise Generator",
                "premise_prompt"
            )
        )
        self.admin_stack.addWidget(self.web_page())
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
        layout.addWidget(subtitle)

        return container

    def model_field(self, label, current=""):
        container = QWidget()
        layout = QVBoxLayout(container)

        heading = QLabel(label)
        layout.addWidget(heading)

        row = QHBoxLayout()

        field = QLineEdit(current)
        field.setPlaceholderText(
            "Enter Hugging Face model name"
        )
        row.addWidget(field)

        layout.addLayout(row)
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
            ("Evidence Prompt", "Configure analogy verification.", 5),
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


    def logic_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        layout.addWidget(self.section_title(
            "Logic",
            "Manage the Smol model and its reasoning engine."
        ))

        model, self.smol_model = self.model_field(
            "Smol Model",
            self.config["smol_model"]
        )
        layout.addWidget(model)

        layout.addWidget(QLabel("Logic Engine Prompt"))

        self.logic_prompt = QTextEdit()
        self.logic_prompt.setPlainText(
            self.config["logic_prompt"]
        )
        layout.addWidget(self.logic_prompt)

        note = QLabel(
            "The logic engine instructions will guide "
            "the Smol model during reasoning."
        )
        note.setObjectName("muted")
        layout.addWidget(note)

        return page

    def planning_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        layout.addWidget(self.section_title(
            "Planning",
            "Configure the Qwen planning model."
        ))

        model, self.planning_model = self.model_field(
            "Qwen Planning Model",
            self.config["planning_model"]
        )
        layout.addWidget(model)

        info = QLabel(
            "Hugging Face downloading and model activation "
            "will be implemented in a later stage."
        )
        info.setObjectName("muted")
        layout.addWidget(info)

        layout.addStretch()
        return page

    def prompt_page(self, title, key):
        page = QWidget()
        layout = QVBoxLayout(page)

        layout.addWidget(self.section_title(
            title,
            "Edit the instructions used by this engine."
        ))

        editor = QTextEdit()
        editor.setPlainText(self.config[key])
        layout.addWidget(editor)

        if key == "premise_prompt":
            self.premise_prompt = editor
        else:
            self.evidence_prompt = editor

        return page

    def web_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        layout.addWidget(self.section_title(
            "Web Engine",
            "Configure external information retrieval."
        ))

        self.web_method = QComboBox()
        self.web_method.addItems([
            "Tavily",
            "Website URL"
        ])

        current = self.config["web_engine"]
        index = self.web_method.findText(current)

        if index >= 0:
            self.web_method.setCurrentIndex(index)

        layout.addWidget(QLabel("Retrieval method"))
        layout.addWidget(self.web_method)

        layout.addWidget(QLabel("Tavily API Key"))

        self.tavily_key = QLineEdit(
            self.config["tavily_key"]
        )
        self.tavily_key.setEchoMode(
            QLineEdit.EchoMode.Password
        )
        self.tavily_key.setPlaceholderText(
            "Enter your Tavily API key"
        )
        layout.addWidget(self.tavily_key)

        layout.addWidget(QLabel("Website URL"))

        self.website_url = QLineEdit(
            self.config["website_url"]
        )
        self.website_url.setPlaceholderText(
            "https://example.com"
        )
        layout.addWidget(self.website_url)

        warning = QLabel(
            "Prototype only: API credentials are currently "
            "stored in local configuration. We will move "
            "secrets to secure storage before production."
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

    def exit_admin(self):
        self.admin_mode = False
        self.admin_tabs.setCurrentIndex(0)
        self.show_page(2)

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


def main():
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setFont(QFont("Segoe UI", 10))

    window = LanganWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()

"""
Settings Dialog for Bulk Georeferencer

Configure default settings, API keys, and LLM prompts.
"""

from pathlib import Path

from qgis.PyQt.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QLineEdit,
    QComboBox,
    QCheckBox,
    QGroupBox,
    QTabWidget,
    QWidget,
    QPlainTextEdit,
    QSpinBox,
    QMessageBox
)
from qgis.PyQt.QtCore import QSettings

from magic_georeferencer.core.vision_api import (
    APIProvider,
    VISION_MODELS,
    MODELS_BY_PROVIDER,
    DEFAULT_SYSTEM_PROMPT
)

# Settings keys
SETTINGS_PREFIX = "BulkGeoreferencer"
SETTINGS_PROVIDER = f"{SETTINGS_PREFIX}/provider"
SETTINGS_MODEL = f"{SETTINGS_PREFIX}/model"
SETTINGS_API_KEY_OPENAI = f"{SETTINGS_PREFIX}/api_key_openai"
SETTINGS_API_KEY_ANTHROPIC = f"{SETTINGS_PREFIX}/api_key_anthropic"
SETTINGS_OUTPUT_SUFFIX = f"{SETTINGS_PREFIX}/output_suffix"
SETTINGS_BASEMAP = f"{SETTINGS_PREFIX}/basemap"
SETTINGS_QUALITY = f"{SETTINGS_PREFIX}/quality"
SETTINGS_CUSTOM_PROMPT = f"{SETTINGS_PREFIX}/custom_prompt"
SETTINGS_USE_CUSTOM_PROMPT = f"{SETTINGS_PREFIX}/use_custom_prompt"
SETTINGS_BUFFER_METERS = f"{SETTINGS_PREFIX}/buffer_meters"
SETTINGS_MIN_GCPS = f"{SETTINGS_PREFIX}/min_gcps"


class BulkSettingsDialog(QDialog):
    """Settings dialog for Bulk Georeferencer."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self.settings = QSettings()

        self.setWindowTitle("Bulk Georeferencer Settings")
        self.setMinimumWidth(600)
        self.setMinimumHeight(500)

        self._setup_ui()
        self._load_settings()

    def _setup_ui(self):
        """Setup the settings UI."""
        layout = QVBoxLayout()

        # Tab widget
        tab_widget = QTabWidget()

        # API Settings tab
        api_tab = QWidget()
        api_layout = QVBoxLayout(api_tab)
        api_layout.addWidget(self._create_api_settings_group())
        api_layout.addStretch()
        tab_widget.addTab(api_tab, "API Settings")

        # Processing Settings tab
        proc_tab = QWidget()
        proc_layout = QVBoxLayout(proc_tab)
        proc_layout.addWidget(self._create_processing_settings_group())
        proc_layout.addStretch()
        tab_widget.addTab(proc_tab, "Processing")

        # LLM Prompt tab
        prompt_tab = QWidget()
        prompt_layout = QVBoxLayout(prompt_tab)
        prompt_layout.addWidget(self._create_prompt_settings_group())
        tab_widget.addTab(prompt_tab, "LLM Prompt")

        layout.addWidget(tab_widget)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        reset_btn = QPushButton("Reset to Defaults")
        reset_btn.clicked.connect(self._reset_to_defaults)
        button_layout.addWidget(reset_btn)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)

        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self._save_and_close)
        save_btn.setDefault(True)
        button_layout.addWidget(save_btn)

        layout.addLayout(button_layout)

        self.setLayout(layout)

    def _create_api_settings_group(self) -> QGroupBox:
        """Create API settings group."""
        group = QGroupBox("Vision API Configuration")
        layout = QVBoxLayout()

        # Provider
        provider_layout = QHBoxLayout()
        provider_layout.addWidget(QLabel("Default Provider:"))
        self.provider_combo = QComboBox()
        self.provider_combo.addItem("OpenAI", APIProvider.OPENAI)
        self.provider_combo.addItem("Anthropic", APIProvider.ANTHROPIC)
        self.provider_combo.currentIndexChanged.connect(self._on_provider_changed)
        provider_layout.addWidget(self.provider_combo)
        provider_layout.addStretch()
        layout.addLayout(provider_layout)

        # Model
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Default Model:"))
        self.model_combo = QComboBox()
        model_layout.addWidget(self.model_combo)
        model_layout.addStretch()
        layout.addLayout(model_layout)

        # OpenAI API Key
        layout.addWidget(QLabel(""))
        layout.addWidget(QLabel("OpenAI API Key:"))
        self.openai_key_edit = QLineEdit()
        self.openai_key_edit.setEchoMode(QLineEdit.Password)
        self.openai_key_edit.setPlaceholderText("sk-...")
        layout.addWidget(self.openai_key_edit)

        # Anthropic API Key
        layout.addWidget(QLabel("Anthropic API Key:"))
        self.anthropic_key_edit = QLineEdit()
        self.anthropic_key_edit.setEchoMode(QLineEdit.Password)
        self.anthropic_key_edit.setPlaceholderText("sk-ant-...")
        layout.addWidget(self.anthropic_key_edit)

        # Note about security
        note = QLabel(
            "Note: API keys are stored locally in QGIS settings. "
            "Keep them secure and never share them."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #666; font-size: 10px;")
        layout.addWidget(note)

        group.setLayout(layout)
        return group

    def _create_processing_settings_group(self) -> QGroupBox:
        """Create processing settings group."""
        group = QGroupBox("Default Processing Settings")
        layout = QVBoxLayout()

        # Output suffix
        suffix_layout = QHBoxLayout()
        suffix_layout.addWidget(QLabel("Output filename suffix:"))
        self.suffix_edit = QLineEdit()
        self.suffix_edit.setText("_georef")
        suffix_layout.addWidget(self.suffix_edit)
        layout.addLayout(suffix_layout)

        # Default basemap
        basemap_layout = QHBoxLayout()
        basemap_layout.addWidget(QLabel("Default basemap:"))
        self.basemap_combo = QComboBox()
        self.basemap_combo.addItem("OpenStreetMap Standard", "osm_standard")
        self.basemap_combo.addItem("ESRI World Imagery", "esri_world_imagery")
        self.basemap_combo.addItem("OpenStreetMap Humanitarian", "osm_humanitarian")
        basemap_layout.addWidget(self.basemap_combo)
        basemap_layout.addStretch()
        layout.addLayout(basemap_layout)

        # Default quality
        quality_layout = QHBoxLayout()
        quality_layout.addWidget(QLabel("Default quality:"))
        self.quality_combo = QComboBox()
        self.quality_combo.addItem("Strict (0.85)", "strict")
        self.quality_combo.addItem("Balanced (0.70)", "balanced")
        self.quality_combo.addItem("Permissive (0.55)", "permissive")
        self.quality_combo.addItem("Very Permissive (0.20)", "very_permissive")
        quality_layout.addWidget(self.quality_combo)
        quality_layout.addStretch()
        layout.addLayout(quality_layout)

        # Minimum GCPs
        gcps_layout = QHBoxLayout()
        gcps_layout.addWidget(QLabel("Minimum GCPs required:"))
        self.min_gcps_spin = QSpinBox()
        self.min_gcps_spin.setRange(4, 20)
        self.min_gcps_spin.setValue(6)
        gcps_layout.addWidget(self.min_gcps_spin)
        gcps_layout.addStretch()
        layout.addLayout(gcps_layout)

        # Default buffer for pre-geocoded
        buffer_layout = QHBoxLayout()
        buffer_layout.addWidget(QLabel("Default search radius (pre-geocoded):"))
        self.buffer_spin = QSpinBox()
        self.buffer_spin.setRange(100, 10000)
        self.buffer_spin.setValue(1000)
        self.buffer_spin.setSingleStep(100)
        self.buffer_spin.setSuffix(" m")
        buffer_layout.addWidget(self.buffer_spin)
        buffer_layout.addStretch()
        layout.addLayout(buffer_layout)

        group.setLayout(layout)
        return group

    def _create_prompt_settings_group(self) -> QGroupBox:
        """Create LLM prompt settings group."""
        group = QGroupBox("Location Extraction Prompt")
        layout = QVBoxLayout()

        info = QLabel(
            "Customize the system prompt sent to the Vision AI for extracting "
            "geographic location from images. This prompt instructs the AI on "
            "how to analyze images and return bounding box coordinates."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: #666;")
        layout.addWidget(info)

        self.use_custom_prompt_checkbox = QCheckBox("Use custom prompt")
        self.use_custom_prompt_checkbox.stateChanged.connect(self._on_custom_prompt_changed)
        layout.addWidget(self.use_custom_prompt_checkbox)

        self.prompt_edit = QPlainTextEdit()
        self.prompt_edit.setPlaceholderText("Enter custom prompt...")
        self.prompt_edit.setEnabled(False)
        layout.addWidget(self.prompt_edit)

        # Reset button
        reset_layout = QHBoxLayout()
        reset_prompt_btn = QPushButton("Reset to Default Prompt")
        reset_prompt_btn.clicked.connect(self._reset_prompt)
        reset_layout.addWidget(reset_prompt_btn)
        reset_layout.addStretch()
        layout.addLayout(reset_layout)

        group.setLayout(layout)
        return group

    def _on_provider_changed(self):
        """Handle provider change."""
        self.model_combo.clear()
        provider = self.provider_combo.currentData()
        model_keys = MODELS_BY_PROVIDER.get(provider, [])
        for key in model_keys:
            model = VISION_MODELS[key]
            self.model_combo.addItem(model.display_name, key)

    def _on_custom_prompt_changed(self):
        """Handle custom prompt checkbox change."""
        use_custom = self.use_custom_prompt_checkbox.isChecked()
        self.prompt_edit.setEnabled(use_custom)
        if use_custom and not self.prompt_edit.toPlainText():
            self.prompt_edit.setPlainText(DEFAULT_SYSTEM_PROMPT)

    def _reset_prompt(self):
        """Reset prompt to default."""
        self.prompt_edit.setPlainText(DEFAULT_SYSTEM_PROMPT)

    def _load_settings(self):
        """Load settings from QSettings."""
        # Provider
        provider_str = self.settings.value(SETTINGS_PROVIDER, "openai")
        provider_idx = 0 if provider_str == "openai" else 1
        self.provider_combo.setCurrentIndex(provider_idx)
        self._on_provider_changed()

        # Model
        saved_model = self.settings.value(SETTINGS_MODEL, "")
        if saved_model:
            idx = self.model_combo.findData(saved_model)
            if idx >= 0:
                self.model_combo.setCurrentIndex(idx)

        # API Keys
        self.openai_key_edit.setText(
            self.settings.value(SETTINGS_API_KEY_OPENAI, "")
        )
        self.anthropic_key_edit.setText(
            self.settings.value(SETTINGS_API_KEY_ANTHROPIC, "")
        )

        # Processing settings
        self.suffix_edit.setText(
            self.settings.value(SETTINGS_OUTPUT_SUFFIX, "_georef")
        )

        basemap = self.settings.value(SETTINGS_BASEMAP, "osm_standard")
        idx = self.basemap_combo.findData(basemap)
        if idx >= 0:
            self.basemap_combo.setCurrentIndex(idx)

        quality = self.settings.value(SETTINGS_QUALITY, "balanced")
        idx = self.quality_combo.findData(quality)
        if idx >= 0:
            self.quality_combo.setCurrentIndex(idx)

        self.min_gcps_spin.setValue(
            self.settings.value(SETTINGS_MIN_GCPS, 6, type=int)
        )
        self.buffer_spin.setValue(
            self.settings.value(SETTINGS_BUFFER_METERS, 1000, type=int)
        )

        # Prompt settings
        use_custom = self.settings.value(SETTINGS_USE_CUSTOM_PROMPT, False, type=bool)
        self.use_custom_prompt_checkbox.setChecked(use_custom)
        custom_prompt = self.settings.value(SETTINGS_CUSTOM_PROMPT, "")
        if custom_prompt:
            self.prompt_edit.setPlainText(custom_prompt)
        self._on_custom_prompt_changed()

    def _save_settings(self):
        """Save settings to QSettings."""
        provider = self.provider_combo.currentData()
        self.settings.setValue(SETTINGS_PROVIDER, provider.value)

        model = self.model_combo.currentData()
        if model:
            self.settings.setValue(SETTINGS_MODEL, model)

        # API Keys
        self.settings.setValue(SETTINGS_API_KEY_OPENAI, self.openai_key_edit.text())
        self.settings.setValue(SETTINGS_API_KEY_ANTHROPIC, self.anthropic_key_edit.text())

        # Processing settings
        self.settings.setValue(SETTINGS_OUTPUT_SUFFIX, self.suffix_edit.text())
        self.settings.setValue(SETTINGS_BASEMAP, self.basemap_combo.currentData())
        self.settings.setValue(SETTINGS_QUALITY, self.quality_combo.currentData())
        self.settings.setValue(SETTINGS_MIN_GCPS, self.min_gcps_spin.value())
        self.settings.setValue(SETTINGS_BUFFER_METERS, self.buffer_spin.value())

        # Prompt settings
        self.settings.setValue(
            SETTINGS_USE_CUSTOM_PROMPT,
            self.use_custom_prompt_checkbox.isChecked()
        )
        self.settings.setValue(SETTINGS_CUSTOM_PROMPT, self.prompt_edit.toPlainText())

    def _save_and_close(self):
        """Save settings and close dialog."""
        self._save_settings()
        self.accept()

    def _reset_to_defaults(self):
        """Reset all settings to defaults."""
        result = QMessageBox.question(
            self,
            "Reset Settings",
            "Are you sure you want to reset all settings to defaults?\n\n"
            "This will clear saved API keys.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if result == QMessageBox.Yes:
            # Clear all settings
            self.settings.remove(SETTINGS_PREFIX)

            # Reload defaults
            self._load_settings()

            QMessageBox.information(
                self,
                "Settings Reset",
                "All settings have been reset to defaults."
            )
